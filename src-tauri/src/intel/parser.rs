use crate::db::ColumnMeta;
use crate::intel::library::{self, LoadedLibrary, Technique};
use anyhow::{anyhow, Result};
use rusqlite::{Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashSet};

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GuidedQueryPreview {
    pub intent_token: String,
    pub preview_text: String,
    pub needs_clarification: bool,
    pub clarification_message: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "intent", rename_all = "camelCase")]
pub enum GuidedIntent {
    SuspiciousScan {
        tactic_ids: Vec<String>,
        technique_ids: Vec<String>,
        sort: GuidedSort,
    },
    UserTechniqueTimeline {
        user_value: String,
        user_column: String,
        technique_ids: Vec<String>,
        sort: GuidedSort,
    },
    TechniqueTimeline {
        technique_ids: Vec<String>,
        sort: GuidedSort,
    },
    Unknown {
        message: String,
        suggestions: Vec<String>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GuidedSort {
    ChronologicalAsc,
    RowNumAsc,
}

#[derive(Debug, Clone)]
struct ParserContext {
    confirmed_user_column: Option<ColumnMeta>,
    has_normalized_time: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CandidateKind {
    Tactic,
    Technique,
    Alias,
    Keyword,
}

#[derive(Debug, Clone)]
struct MatchCandidate {
    normalized_phrase: String,
    display: String,
    kind: CandidateKind,
    technique_id: Option<String>,
    tactic_id: Option<String>,
}

#[derive(Debug, Clone, Default)]
struct TechniqueSelection {
    technique_ids: BTreeSet<String>,
    tactic_ids: BTreeSet<String>,
    tactic_names: BTreeSet<String>,
    technique_names: BTreeSet<String>,
    keyword_samples: BTreeSet<String>,
    matched_terms: BTreeSet<String>,
}

#[derive(Debug, Clone)]
struct UserExtraction {
    value: Option<String>,
    unresolved_this_user: bool,
}

pub fn parse_guided_query(
    conn: &Connection,
    columns: &[ColumnMeta],
    query_text: &str,
) -> Result<GuidedQueryPreview> {
    let library = library::load_merged_library()?;
    let context = ParserContext::from_db(conn, columns)?;
    parse_with_context(query_text, &library, &context)
}

pub fn intent_from_token(intent_token: &str) -> Result<GuidedIntent> {
    serde_json::from_str(intent_token)
        .map_err(|err| anyhow!("invalid guided query intent token: {err}"))
}

fn parse_with_context(
    query_text: &str,
    library: &LoadedLibrary,
    context: &ParserContext,
) -> Result<GuidedQueryPreview> {
    let trimmed = query_text.trim();
    if trimmed.is_empty() {
        return clarification(
            "I need a query before I can build a guided search.",
            &["Try: show credential access for alice chronologically"],
        );
    }

    let candidates = build_candidates(library);
    let selection = select_techniques(trimmed, library, &candidates);
    if selection.technique_ids.is_empty() && selection.tactic_ids.is_empty() {
        if let Some(message) = ambiguous_token_message(trimmed, &candidates, library) {
            return clarification(
                &message,
                &[
                    "Use a full tactic such as Credential Access",
                    "Use a technique or keyword such as mimikatz",
                ],
            );
        }
    }

    let user = extract_user(trimmed, &selection);
    if user.unresolved_this_user && user.value.is_none() {
        return clarification(
            "I saw 'this user', but no actual user value was provided.",
            &["Rephrase with the exact identity, for example: show attacks of user alice"],
        );
    }

    let sort = if context.has_normalized_time {
        GuidedSort::ChronologicalAsc
    } else {
        GuidedSort::RowNumAsc
    };

    let has_suspicious_intent = contains_suspicious_intent(trimmed);
    let has_attack_timeline_intent = contains_attack_timeline_intent(trimmed);

    let intent = if let Some(user_value) = user.value {
        let Some(user_column) = context.confirmed_user_column.as_ref() else {
            return clarification(
                "I found a user value, but no user column has been confirmed yet.",
                &[
                    "Confirm the user/account column first",
                    "Then re-run the query with the exact user value",
                ],
            );
        };

        if !selection.technique_ids.is_empty()
            || has_attack_timeline_intent
            || has_suspicious_intent
        {
            GuidedIntent::UserTechniqueTimeline {
                user_value,
                user_column: user_column.sql_name.clone(),
                technique_ids: selection.technique_ids.iter().cloned().collect(),
                sort,
            }
        } else {
            return clarification(
                "I found a user value, but not what suspicious activity or technique to search for.",
                &[
                    "Try: show attacks of user alice",
                    "Try: show credential access for alice chronologically",
                ],
            );
        }
    } else if !selection.technique_ids.is_empty() {
        GuidedIntent::TechniqueTimeline {
            technique_ids: selection.technique_ids.iter().cloned().collect(),
            sort,
        }
    } else if has_suspicious_intent || has_attack_timeline_intent {
        // A bare "attack"/"activity"/"timeline" word is enough to ask for a broad scan - but if
        // the query also names something specific we don't recognize ("shadow credentials
        // attack"), silently broadening to the same generic scan would look like a real targeted
        // result when it isn't one. Ask instead of guessing.
        if let Some(message) = unrecognized_technique_message(trimmed) {
            return clarification(
                &message,
                &[
                    "Try a known technique or keyword, such as mimikatz or credential dumping",
                    "Or add this as a custom category to the library first",
                ],
            );
        }
        GuidedIntent::SuspiciousScan {
            tactic_ids: Vec::new(),
            technique_ids: Vec::new(),
            sort,
        }
    } else {
        return clarification(
            "I could not confidently map that query to a supported guided-search intent.",
            &[
                "Try: find suspicious activity",
                "Try: show credential access for alice chronologically",
                "Try: mimikatz activity for alice in order",
            ],
        );
    };

    let preview_text = preview_text(&intent, &selection, library, context);
    Ok(GuidedQueryPreview {
        intent_token: encode_intent(&intent)?,
        preview_text,
        needs_clarification: false,
        clarification_message: None,
    })
}

fn encode_intent(intent: &GuidedIntent) -> Result<String> {
    serde_json::to_string(intent).map_err(Into::into)
}

fn clarification(message: &str, suggestions: &[&str]) -> Result<GuidedQueryPreview> {
    let intent = GuidedIntent::Unknown {
        message: message.to_string(),
        suggestions: suggestions.iter().map(|s| s.to_string()).collect(),
    };
    Ok(GuidedQueryPreview {
        intent_token: encode_intent(&intent)?,
        preview_text: "No guided query will be run until this is clarified.".to_string(),
        needs_clarification: true,
        clarification_message: Some(format!("{} {}", message, suggestions.join("; "))),
    })
}

impl ParserContext {
    fn from_db(conn: &Connection, columns: &[ColumnMeta]) -> Result<Self> {
        let confirmed_user_column = if table_exists(conn, "_column_roles")? {
            let sql_name: Option<String> = conn
                .query_row(
                    "SELECT sql_name FROM _column_roles
                     WHERE role = 'user' AND status = 'confirmed'
                     LIMIT 1",
                    [],
                    |row| row.get(0),
                )
                .optional()?;
            sql_name.and_then(|name| {
                columns
                    .iter()
                    .find(|column| column.sql_name == name)
                    .cloned()
            })
        } else {
            None
        };

        let has_normalized_time = if table_exists(conn, "_row_time")? {
            conn.query_row(
                "SELECT EXISTS(SELECT 1 FROM _row_time LIMIT 1)",
                [],
                |row| row.get::<_, i64>(0),
            )? != 0
        } else {
            false
        };

        Ok(Self {
            confirmed_user_column,
            has_normalized_time,
        })
    }
}

fn table_exists(conn: &Connection, table: &str) -> rusqlite::Result<bool> {
    conn.query_row(
        "SELECT EXISTS(
            SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?1
         )",
        [table],
        |row| row.get::<_, i64>(0),
    )
    .map(|value| value != 0)
}

fn build_candidates(library: &LoadedLibrary) -> Vec<MatchCandidate> {
    let mut candidates = Vec::new();
    let mut seen = HashSet::new();

    for technique in &library.techniques {
        add_candidate(
            &mut candidates,
            &mut seen,
            &technique.name,
            &technique.name,
            CandidateKind::Technique,
            Some(&technique.technique_id),
            None,
        );
        if let Some((prefix, suffix)) = technique.name.split_once(':') {
            add_candidate(
                &mut candidates,
                &mut seen,
                prefix.trim(),
                prefix.trim(),
                CandidateKind::Technique,
                Some(&technique.technique_id),
                None,
            );
            add_candidate(
                &mut candidates,
                &mut seen,
                suffix.trim(),
                suffix.trim(),
                CandidateKind::Technique,
                Some(&technique.technique_id),
                None,
            );
        }
        add_candidate(
            &mut candidates,
            &mut seen,
            &technique.technique_id,
            &technique.technique_id,
            CandidateKind::Technique,
            Some(&technique.technique_id),
            None,
        );

        for alias in &technique.aliases {
            add_candidate(
                &mut candidates,
                &mut seen,
                alias,
                alias,
                CandidateKind::Alias,
                Some(&technique.technique_id),
                None,
            );
        }
        for keyword in &technique.keywords {
            add_candidate(
                &mut candidates,
                &mut seen,
                &keyword.pattern,
                &keyword.pattern,
                CandidateKind::Keyword,
                Some(&technique.technique_id),
                None,
            );
        }
        for tactic in &technique.tactics {
            add_candidate(
                &mut candidates,
                &mut seen,
                &tactic.name,
                &tactic.name,
                CandidateKind::Tactic,
                None,
                Some(&tactic.id),
            );
        }
    }

    candidates.sort_by(|a, b| {
        b.normalized_phrase
            .len()
            .cmp(&a.normalized_phrase.len())
            .then_with(|| a.display.cmp(&b.display))
    });
    candidates
}

fn add_candidate(
    candidates: &mut Vec<MatchCandidate>,
    seen: &mut HashSet<String>,
    phrase: &str,
    display: &str,
    kind: CandidateKind,
    technique_id: Option<&str>,
    tactic_id: Option<&str>,
) {
    let normalized_phrase = normalize_phrase(phrase);
    if normalized_phrase.is_empty() {
        return;
    }
    let key = format!(
        "{kind:?}|{}|{}|{}",
        normalized_phrase,
        technique_id.unwrap_or_default(),
        tactic_id.unwrap_or_default()
    );
    if seen.insert(key) {
        candidates.push(MatchCandidate {
            normalized_phrase,
            display: display.to_string(),
            kind,
            technique_id: technique_id.map(str::to_string),
            tactic_id: tactic_id.map(str::to_string),
        });
    }
}

fn select_techniques(
    query_text: &str,
    library: &LoadedLibrary,
    candidates: &[MatchCandidate],
) -> TechniqueSelection {
    let query_norm = normalize_phrase(query_text);
    let mut selection = TechniqueSelection::default();
    if query_norm.is_empty() {
        return selection;
    }

    let technique_by_id: BTreeMap<&str, &Technique> = library
        .techniques
        .iter()
        .map(|technique| (technique.technique_id.as_str(), technique))
        .collect();

    for candidate in candidates {
        if !phrase_matches(&query_norm, &candidate.normalized_phrase) {
            continue;
        }
        selection
            .matched_terms
            .insert(candidate.normalized_phrase.clone());
        match candidate.kind {
            CandidateKind::Tactic => {
                let Some(tactic_id) = candidate.tactic_id.as_deref() else {
                    continue;
                };
                selection.tactic_ids.insert(tactic_id.to_string());
                selection.tactic_names.insert(candidate.display.clone());
                for technique in techniques_for_tactic(library, tactic_id) {
                    selection
                        .technique_ids
                        .insert(technique.technique_id.clone());
                    selection.technique_names.insert(technique.name.clone());
                    add_keyword_samples(&mut selection, technique);
                }
            }
            CandidateKind::Technique | CandidateKind::Alias | CandidateKind::Keyword => {
                let Some(technique_id) = candidate.technique_id.as_deref() else {
                    continue;
                };
                if let Some(technique) = technique_by_id.get(technique_id) {
                    selection
                        .technique_ids
                        .insert(technique.technique_id.clone());
                    selection.technique_names.insert(technique.name.clone());
                    if candidate.kind == CandidateKind::Keyword {
                        selection.keyword_samples.insert(candidate.display.clone());
                    } else {
                        add_keyword_samples(&mut selection, technique);
                    }
                }
            }
        }
    }

    selection
}

fn techniques_for_tactic<'a>(
    library: &'a LoadedLibrary,
    tactic_id: &'a str,
) -> impl Iterator<Item = &'a Technique> + 'a {
    library.techniques.iter().filter(move |technique| {
        technique
            .tactics
            .iter()
            .any(|tactic| tactic.id == tactic_id)
    })
}

fn add_keyword_samples(selection: &mut TechniqueSelection, technique: &Technique) {
    for keyword in technique.keywords.iter().take(3) {
        selection.keyword_samples.insert(keyword.pattern.clone());
    }
}

fn ambiguous_token_message(
    query_text: &str,
    candidates: &[MatchCandidate],
    library: &LoadedLibrary,
) -> Option<String> {
    let tokens = raw_tokens(query_text);
    let technique_names: BTreeMap<&str, &str> = library
        .techniques
        .iter()
        .map(|technique| (technique.technique_id.as_str(), technique.name.as_str()))
        .collect();

    for token in tokens {
        let norm = normalize_phrase(&token);
        if norm.len() < 4 || is_noise_word(&norm) || is_intent_word(&norm) {
            continue;
        }
        let mut matches = BTreeSet::new();
        for candidate in candidates {
            let candidate_tokens: HashSet<&str> =
                candidate.normalized_phrase.split_whitespace().collect();
            if !candidate_tokens.contains(norm.as_str()) {
                continue;
            }
            if let Some(technique_id) = candidate.technique_id.as_deref() {
                let display = technique_names
                    .get(technique_id)
                    .copied()
                    .unwrap_or(technique_id);
                matches.insert(display.to_string());
            } else if candidate.tactic_id.is_some() {
                matches.insert(candidate.display.clone());
            }
        }
        if matches.len() > 1 {
            let examples = matches.into_iter().take(4).collect::<Vec<_>>().join(", ");
            return Some(format!(
                "The term '{token}' is ambiguous and could refer to multiple tactics or techniques: {examples}."
            ));
        }
    }

    None
}

/// Called only when `select_techniques` found zero matches at all. Distinguishes a genuinely
/// generic query ("find suspicious activity", "show attack timeline" - no significant content
/// words, intentionally broad) from a query that names something specific we don't recognize
/// ("shadow credentials attack" - "shadow"/"credentials" are real content words with no match in
/// the library). The latter must never silently fall back to a broad suspicious-activity scan:
/// that would look like a real targeted result to the examiner when it's actually the same
/// generic scan they'd get from typing nothing specific at all.
fn unrecognized_technique_message(query_text: &str) -> Option<String> {
    let significant: Vec<String> = raw_tokens(query_text)
        .into_iter()
        .map(|token| normalize_phrase(&token))
        .filter(|norm| norm.len() >= 4 && !is_noise_word(norm) && !is_intent_word(norm))
        .collect();

    if significant.is_empty() {
        return None;
    }

    let mut unique = significant;
    unique.dedup();
    Some(format!(
        "I don't recognize '{}' as a known technique, tactic, category, or keyword in the library.",
        unique.join(" ")
    ))
}

fn extract_user(query_text: &str, selection: &TechniqueSelection) -> UserExtraction {
    let tokens = raw_tokens(query_text);
    let lower_tokens: Vec<String> = tokens
        .iter()
        .map(|token| token.to_ascii_lowercase())
        .collect();
    let unresolved_this_user = lower_tokens
        .windows(2)
        .any(|pair| pair[0] == "this" && pair[1] == "user");

    if let Some(value) = tokens
        .iter()
        .find(|token| is_domain_user(token) || is_upn_like(token) || is_sid_like(token))
    {
        return UserExtraction {
            value: Some(value.clone()),
            unresolved_this_user,
        };
    }

    for (idx, lower) in lower_tokens.iter().enumerate() {
        if matches!(lower.as_str(), "user" | "username" | "account") {
            if let Some(value) = next_user_token(&tokens, idx + 1, selection) {
                return UserExtraction {
                    value: Some(value),
                    unresolved_this_user,
                };
            }
        }
        if matches!(lower.as_str(), "for" | "of") {
            let start = if lower_tokens.get(idx + 1).is_some_and(|next| next == "user") {
                idx + 2
            } else {
                idx + 1
            };
            if let Some(value) = next_user_token(&tokens, start, selection) {
                return UserExtraction {
                    value: Some(value),
                    unresolved_this_user,
                };
            }
        }
    }

    // Deliberately no "guess a leftover word is the username" fallback here. Every documented/
    // supported phrasing ("for alice", "of user alice", "user alice", DOMAIN\user, UPN, SID)
    // anchors on an explicit signal above. Guessing that any single unmatched word must be a
    // username silently misread real queries like "show LSASS memory dumping" (treating
    // "dumping" as a user) and "show powershell downloaders" (treating "downloaders" as a
    // user), both returning a confident-looking but empty/wrong result instead of the correct
    // TechniqueTimeline for the whole matched technique. No user found here is the right,
    // honest answer - the caller falls through to a technique-only or clarification path.
    UserExtraction {
        value: None,
        unresolved_this_user,
    }
}

fn next_user_token(
    tokens: &[String],
    start: usize,
    selection: &TechniqueSelection,
) -> Option<String> {
    let matched_tokens: HashSet<&str> = selection
        .matched_terms
        .iter()
        .flat_map(|term| term.split_whitespace())
        .collect();
    for token in tokens.iter().skip(start).take(3) {
        let norm = normalize_phrase(token);
        if norm == "this" || norm == "that" {
            return None;
        }
        if is_noise_word(&norm) || is_intent_word(&norm) {
            continue;
        }
        if matched_tokens.contains(norm.as_str()) {
            continue;
        }
        if is_user_value(token) {
            return Some(token.clone());
        }
    }
    None
}

fn contains_suspicious_intent(query_text: &str) -> bool {
    let norm = normalize_phrase(query_text);
    phrase_matches(&norm, "suspicious") || phrase_matches(&norm, "dfir suspicious")
}

fn contains_attack_timeline_intent(query_text: &str) -> bool {
    let tokens: HashSet<String> = raw_tokens(query_text)
        .into_iter()
        .map(|token| normalize_phrase(&token))
        .collect();
    tokens.contains("attack")
        || tokens.contains("attacks")
        || tokens.contains("timeline")
        || tokens.contains("activity")
        || tokens.contains("activities")
}

fn preview_text(
    intent: &GuidedIntent,
    selection: &TechniqueSelection,
    library: &LoadedLibrary,
    context: &ParserContext,
) -> String {
    let sort_text = match intent_sort(intent) {
        GuidedSort::ChronologicalAsc => "sorted by normalized UTC timestamp",
        GuidedSort::RowNumAsc => {
            "sorted by source row number because UTC timestamps have not been normalized yet"
        }
    };
    match intent {
        GuidedIntent::SuspiciousScan { .. } => {
            format!("Suspicious activity scan across all MITRE ATT&CK-style matches; {sort_text}.")
        }
        GuidedIntent::TechniqueTimeline { technique_ids, .. } => {
            let scope = describe_technique_scope(technique_ids, selection, library);
            format!("{scope}; {sort_text}.")
        }
        GuidedIntent::UserTechniqueTimeline {
            user_value,
            user_column,
            technique_ids,
            ..
        } => {
            let user_column_display = context
                .confirmed_user_column
                .as_ref()
                .filter(|column| column.sql_name == *user_column)
                .map(|column| column.original_name.as_str())
                .unwrap_or(user_column.as_str());
            let scope = describe_technique_scope(technique_ids, selection, library);
            format!("User column: {user_column_display}; user: {user_value}; {scope}; {sort_text}.")
        }
        GuidedIntent::Unknown { message, .. } => message.clone(),
    }
}

fn intent_sort(intent: &GuidedIntent) -> GuidedSort {
    match intent {
        GuidedIntent::SuspiciousScan { sort, .. }
        | GuidedIntent::UserTechniqueTimeline { sort, .. }
        | GuidedIntent::TechniqueTimeline { sort, .. } => *sort,
        GuidedIntent::Unknown { .. } => GuidedSort::RowNumAsc,
    }
}

fn describe_technique_scope(
    technique_ids: &[String],
    selection: &TechniqueSelection,
    library: &LoadedLibrary,
) -> String {
    if technique_ids.is_empty() {
        return "MITRE matches: all scanned techniques".to_string();
    }
    if selection.tactic_names.len() == 1 {
        let tactic = selection.tactic_names.iter().next().unwrap();
        return format!(
            "category: {tactic}; matched keywords: {}",
            keyword_preview(selection, library, technique_ids)
        );
    }
    if selection.technique_names.len() == 1 {
        let technique = selection.technique_names.iter().next().unwrap();
        return format!(
            "technique: {technique}; matched keywords: {}",
            keyword_preview(selection, library, technique_ids)
        );
    }

    let names = selection
        .technique_names
        .iter()
        .take(4)
        .cloned()
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "techniques: {names}; matched keywords: {}",
        keyword_preview(selection, library, technique_ids)
    )
}

fn keyword_preview(
    selection: &TechniqueSelection,
    library: &LoadedLibrary,
    technique_ids: &[String],
) -> String {
    let mut keywords = selection.keyword_samples.clone();
    if keywords.is_empty() {
        let selected: HashSet<&str> = technique_ids.iter().map(String::as_str).collect();
        for technique in &library.techniques {
            if selected.contains(technique.technique_id.as_str()) {
                for keyword in technique.keywords.iter().take(2) {
                    keywords.insert(keyword.pattern.clone());
                }
            }
        }
    }
    let preview = keywords.into_iter().take(5).collect::<Vec<_>>();
    if preview.is_empty() {
        "none listed".to_string()
    } else {
        preview.join(", ")
    }
}

fn phrase_matches(query_norm: &str, phrase_norm: &str) -> bool {
    if phrase_norm.is_empty() {
        return false;
    }
    let haystack = format!(" {query_norm} ");
    let needle = format!(" {phrase_norm} ");
    haystack.contains(&needle)
}

fn normalize_phrase(value: &str) -> String {
    let mut normalized = String::with_capacity(value.len());
    let mut last_was_space = true;
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() {
            normalized.push(ch.to_ascii_lowercase());
            last_was_space = false;
        } else if !last_was_space {
            normalized.push(' ');
            last_was_space = true;
        }
    }
    normalized.trim().to_string()
}

fn raw_tokens(value: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '\\' | '@' | '.' | '_' | '-' | '$') {
            current.push(ch);
        } else if !current.is_empty() {
            tokens.push(trim_token(&current));
            current.clear();
        }
    }
    if !current.is_empty() {
        tokens.push(trim_token(&current));
    }
    tokens
        .into_iter()
        .filter(|token| !token.is_empty())
        .collect()
}

fn trim_token(value: &str) -> String {
    value
        .trim_matches(|c: char| matches!(c, '.' | ',' | ';' | ':' | '"' | '\''))
        .to_string()
}

fn is_noise_word(value: &str) -> bool {
    matches!(
        value,
        "a" | "an"
            | "and"
            | "anything"
            | "are"
            | "as"
            | "at"
            | "by"
            | "csv"
            | "for"
            | "from"
            | "give"
            | "hello"
            | "hey"
            | "i"
            | "in"
            | "is"
            | "it"
            | "me"
            | "of"
            | "on"
            | "please"
            | "that"
            | "the"
            | "this"
            | "to"
            | "with"
            | "xls"
            | "xlsx"
    )
}

fn is_intent_word(value: &str) -> bool {
    matches!(
        value,
        "activity"
            | "activities"
            | "attack"
            | "attacks"
            | "bad"
            | "chronological"
            | "chronologically"
            | "dfir"
            | "file"
            | "filter"
            | "find"
            | "manner"
            | "order"
            | "rows"
            | "show"
            | "sort"
            | "stuff"
            | "suspicious"
            | "timeline"
            | "user"
            | "username"
    )
}

fn is_user_value(value: &str) -> bool {
    is_domain_user(value) || is_upn_like(value) || is_sid_like(value) || is_simple_user(value)
}

fn is_domain_user(value: &str) -> bool {
    let Some((domain, user)) = value.split_once('\\') else {
        return false;
    };
    !domain.is_empty()
        && !user.is_empty()
        && domain.len() <= 64
        && user.len() <= 128
        && domain
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_'))
        && user
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_' | '$'))
}

fn is_upn_like(value: &str) -> bool {
    let Some((local, domain)) = value.split_once('@') else {
        return false;
    };
    !local.is_empty()
        && domain.contains('.')
        && !domain.ends_with('.')
        && local
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_' | '+'))
        && domain
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-'))
}

fn is_sid_like(value: &str) -> bool {
    value.starts_with("S-1-")
        && value
            .split('-')
            .skip(1)
            .all(|part| !part.is_empty() && part.chars().all(|c| c.is_ascii_digit()))
}

fn is_simple_user(value: &str) -> bool {
    let norm = normalize_phrase(value);
    (2..=64).contains(&value.len())
        && value.chars().any(|c| c.is_ascii_alphabetic())
        && !value.contains(char::is_whitespace)
        && !value.contains('.')
        && !is_noise_word(&norm)
        && !is_intent_word(&norm)
        && value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '$'))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context_with_user() -> ParserContext {
        ParserContext {
            confirmed_user_column: Some(ColumnMeta {
                sql_name: "account".into(),
                original_name: "Account".into(),
                col_index: 0,
                inferred_type: "text".into(),
            }),
            has_normalized_time: true,
        }
    }

    fn parse_intent(query: &str) -> (GuidedQueryPreview, GuidedIntent) {
        let library = library::load_builtin_library().unwrap();
        let preview = parse_with_context(query, &library, &context_with_user()).unwrap();
        let intent = intent_from_token(&preview.intent_token).unwrap();
        (preview, intent)
    }

    #[test]
    fn suspicious_prompts_map_to_suspicious_scan() {
        let (_, intent) = parse_intent("find suspicious activity");
        assert!(matches!(intent, GuidedIntent::SuspiciousScan { .. }));

        let (_, intent) = parse_intent("find anything suspicious in dfir manner");
        assert!(matches!(intent, GuidedIntent::SuspiciousScan { .. }));
    }

    #[test]
    fn credential_access_for_user_maps_to_user_timeline() {
        let (preview, intent) = parse_intent("show credential access for alice chronologically");
        assert!(!preview.needs_clarification);
        match intent {
            GuidedIntent::UserTechniqueTimeline {
                user_value,
                user_column,
                technique_ids,
                sort,
            } => {
                assert_eq!(user_value, "alice");
                assert_eq!(user_column, "account");
                assert_eq!(sort, GuidedSort::ChronologicalAsc);
                assert!(technique_ids.iter().any(|id| id == "T1003.001"));
                assert!(technique_ids.iter().any(|id| id == "T1555"));
            }
            other => panic!("unexpected intent: {other:?}"),
        }
    }

    #[test]
    fn domain_user_and_credential_dumping_are_extracted() {
        let (_, intent) = parse_intent(r"filter user CORP\alice by credential dumping");
        match intent {
            GuidedIntent::UserTechniqueTimeline {
                user_value,
                technique_ids,
                ..
            } => {
                assert_eq!(user_value, r"CORP\alice");
                assert_eq!(technique_ids, vec!["T1003.001"]);
            }
            other => panic!("unexpected intent: {other:?}"),
        }
    }

    #[test]
    fn bare_technique_query_without_a_user_anchor_never_guesses_a_username() {
        // "dumping" and "downloaders" are not usernames - with no "for"/"of"/"user" anchor,
        // these must resolve to a technique-only timeline covering everyone, not a
        // (silently wrong, confidently empty) search for a nonexistent user.
        let (preview, intent) = parse_intent("show LSASS memory dumping");
        assert!(!preview.needs_clarification);
        match intent {
            GuidedIntent::TechniqueTimeline { technique_ids, .. } => {
                assert!(technique_ids.contains(&"T1003.001".to_string()));
            }
            other => panic!("unexpected intent: {other:?}"),
        }

        let (preview, intent) = parse_intent("show powershell downloaders");
        assert!(!preview.needs_clarification);
        match intent {
            GuidedIntent::TechniqueTimeline { technique_ids, .. } => {
                assert!(technique_ids.contains(&"T1059.001".to_string()));
            }
            other => panic!("unexpected intent: {other:?}"),
        }
    }

    #[test]
    fn keyword_for_user_maps_to_specific_technique() {
        let (_, intent) = parse_intent("mimikatz activity for alice in order");
        match intent {
            GuidedIntent::UserTechniqueTimeline {
                user_value,
                technique_ids,
                ..
            } => {
                assert_eq!(user_value, "alice");
                assert_eq!(technique_ids, vec!["T1003.001"]);
            }
            other => panic!("unexpected intent: {other:?}"),
        }
    }

    #[test]
    fn attacks_of_user_without_specific_technique_means_all_matches_for_user() {
        let (_, intent) = parse_intent("show attacks of user alice");
        match intent {
            GuidedIntent::UserTechniqueTimeline {
                user_value,
                technique_ids,
                ..
            } => {
                assert_eq!(user_value, "alice");
                assert!(technique_ids.is_empty());
            }
            other => panic!("unexpected intent: {other:?}"),
        }
    }

    #[test]
    fn vague_or_unresolved_user_prompts_need_clarification() {
        let (preview, intent) = parse_intent("find bad stuff");
        assert!(preview.needs_clarification);
        assert!(matches!(intent, GuidedIntent::Unknown { .. }));

        let (preview, intent) = parse_intent("show attacks of this user");
        assert!(preview.needs_clarification);
        assert!(matches!(intent, GuidedIntent::Unknown { .. }));
    }

    #[test]
    fn unrecognized_specific_term_does_not_silently_fall_back_to_broad_scan() {
        // "attack" alone would normally trigger a broad SuspiciousScan fallback - but "shadow"
        // and "credentials" are real content words, so this must ask for clarification instead
        // of quietly returning the same results as a bare "find suspicious activity" query
        // would. Whether the specific reason is "unrecognized" or "ambiguous" can legitimately
        // shift as the library grows (e.g. "credentials" now matches several real techniques) -
        // the invariant that actually matters is: never SuspiciousScan, always ask.
        let (preview, intent) = parse_intent("shadow credentials attack");
        assert!(preview.needs_clarification);
        assert!(matches!(intent, GuidedIntent::Unknown { .. }));
        let message = preview.clarification_message.unwrap_or_default();
        assert!(!message.is_empty(), "clarification message should not be empty");
    }
}
