use crate::db;
use crate::intel::library::{
    self, BehaviorRule, ConditionOp, LoadedLibrary, MatchKind, Tactic, RULE_CONDITION_ROLES,
};
use aho_corasick::{AhoCorasick, AhoCorasickBuilder};
use anyhow::{anyhow, Result};
use rusqlite::{Connection, OptionalExtension};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

const PROGRESS_INTERVAL_ROWS: i64 = 5000;
const SCAN_BATCH_ROWS: i64 = 1000;
const STAGING_TABLE: &str = "temp._intel_match_staging";
static SCAN_TOKEN_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct IntelScanSummary {
    pub rows_scanned: i64,
    pub match_count: i64,
    pub matched_rows: i64,
    pub library_hash: String,
    pub role_hash: String,
    pub custom_library_error: Option<String>,
    pub tactics: Vec<IntelCountSummary>,
    pub techniques: Vec<IntelCountSummary>,
    pub chains: Vec<crate::intel::chains::IntelChainSummary>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct IntelCountSummary {
    pub id: String,
    pub name: String,
    pub match_count: i64,
    pub row_count: i64,
}

#[derive(Debug, Clone)]
struct CountAccumulator {
    name: String,
    match_count: i64,
    rows: HashSet<i64>,
}

#[derive(Debug, Clone)]
struct PatternMeta {
    tactic_refs: Vec<Tactic>,
    technique_id: String,
    technique_name: String,
    pattern_id: String,
    keyword: String,
    match_kind: MatchKind,
    score: i64,
}

struct CompiledLibrary {
    automaton: AhoCorasick,
    patterns: Vec<PatternMeta>,
    rules: Vec<CompiledRule>,
    library_hash: String,
    custom_library_error: Option<String>,
}

#[derive(Debug, Clone)]
struct CompiledRule {
    rule: BehaviorRule,
    technique_name: String,
    tactics: Vec<Tactic>,
}

/// One behavior rule bound to this dataset's actual columns. `column_indices` index into the
/// combined per-row value vector (evidence columns first, extra rule columns appended).
#[derive(Debug, Clone)]
struct ResolvedCondition {
    column_indices: Vec<usize>,
    op: ConditionOp,
    values_lower: Vec<String>,
}

#[derive(Debug, Clone)]
struct ResolvedRule {
    compiled_idx: usize,
    conditions: Vec<ResolvedCondition>,
}

pub fn scan_connection(
    conn: &mut Connection,
    evidence_columns: &[String],
    mut on_progress: impl FnMut(i64, i64, &str),
) -> Result<IntelScanSummary> {
    let library = library::load_merged_library()?;
    scan_connection_with_library(conn, evidence_columns, library, &mut on_progress)
}

pub fn scan_connection_with_library(
    conn: &mut Connection,
    evidence_columns: &[String],
    library: LoadedLibrary,
    mut on_progress: impl FnMut(i64, i64, &str),
) -> Result<IntelScanSummary> {
    let role_hash = role_hash_for_columns(evidence_columns);
    let compiled = compile_library(library)?;
    scan_with_compiled_library(
        conn,
        evidence_columns,
        &role_hash,
        &compiled,
        &mut on_progress,
    )
}

pub fn role_hash_for_columns(evidence_columns: &[String]) -> String {
    let mut columns = evidence_columns.to_vec();
    columns.sort();
    columns.dedup();

    let mut hasher = Sha256::new();
    hasher.update(b"evidence-columns-v1\0");
    for column in columns {
        hasher.update((column.len() as u64).to_le_bytes());
        hasher.update(column.as_bytes());
    }
    format!("{:x}", hasher.finalize())
}

#[derive(Debug)]
struct PendingMatch {
    row_num: i64,
    pattern_idx: usize,
    tactic_idx: usize,
    column_idx: usize,
}

pub fn confirmed_evidence_columns(conn: &Connection) -> Result<Vec<String>> {
    let roles_exist: i64 = conn.query_row(
        "SELECT EXISTS(
            SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = '_column_roles'
         )",
        [],
        |row| row.get(0),
    )?;
    if roles_exist == 0 {
        return Ok(Vec::new());
    }
    let mut stmt = conn.prepare(
        "SELECT sql_name FROM _column_roles
         WHERE status = 'confirmed'
           AND role IN ('command_line', 'process_name', 'file_name', 'host', 'text_evidence')
         ORDER BY sql_name",
    )?;
    let columns = stmt
        .query_map([], |row| row.get::<_, String>(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(columns)
}

fn compile_library(library: LoadedLibrary) -> Result<CompiledLibrary> {
    let mut pattern_strings = Vec::new();
    let mut patterns = Vec::new();

    let technique_lookup: HashMap<&str, (&str, &[Tactic])> = library
        .techniques
        .iter()
        .map(|technique| {
            (
                technique.technique_id.as_str(),
                (technique.name.as_str(), technique.tactics.as_slice()),
            )
        })
        .collect();
    let mut rules = Vec::with_capacity(library.behavior_rules.len());
    for rule in &library.behavior_rules {
        let Some((technique_name, tactics)) = technique_lookup.get(rule.technique_id.as_str())
        else {
            // Merged custom rules are validated against their own file; a rule referencing a
            // technique this merge does not contain is skipped rather than failing the scan.
            continue;
        };
        rules.push(CompiledRule {
            rule: rule.clone(),
            technique_name: technique_name.to_string(),
            tactics: tactics.to_vec(),
        });
    }

    for technique in library.techniques {
        for keyword in technique.keywords {
            pattern_strings.push(keyword.pattern.clone());
            patterns.push(PatternMeta {
                tactic_refs: technique.tactics.clone(),
                technique_id: technique.technique_id.clone(),
                technique_name: technique.name.clone(),
                pattern_id: keyword.id,
                keyword: keyword.pattern,
                match_kind: keyword.match_kind,
                score: keyword.score,
            });
        }
    }

    if pattern_strings.is_empty() {
        return Err(anyhow!("intel library contains no keyword patterns"));
    }

    let automaton = AhoCorasickBuilder::new()
        .ascii_case_insensitive(true)
        .build(pattern_strings)?;

    Ok(CompiledLibrary {
        automaton,
        patterns,
        rules,
        library_hash: library.library_hash,
        custom_library_error: library.custom_library_error,
    })
}

/// Binds each behavior rule to this dataset. Role conditions use non-rejected data mappings
/// (consistent with the optional-enrichment philosophy); header conditions match the normalized
/// original header or SQL name. Rules with any unresolvable condition are skipped for this
/// dataset — a condition that cannot see its column must not silently pass.
fn resolve_behavior_rules(
    conn: &Connection,
    rules: &[CompiledRule],
    scan_columns: &[String],
) -> Result<(Vec<String>, Vec<ResolvedRule>)> {
    if rules.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    let all_columns = db::load_columns(conn)?;
    let roles_exist: i64 = conn.query_row(
        "SELECT EXISTS(
            SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = '_column_roles'
         )",
        [],
        |row| row.get(0),
    )?;
    let mut role_columns: HashMap<String, Vec<String>> = HashMap::new();
    if roles_exist != 0 {
        let mut stmt = conn.prepare(
            "SELECT role, sql_name FROM _column_roles
             WHERE status IN ('suggested', 'confirmed')
             ORDER BY sql_name",
        )?;
        let mut query = stmt.query([])?;
        while let Some(row) = query.next()? {
            let role: String = row.get(0)?;
            let sql_name: String = row.get(1)?;
            if RULE_CONDITION_ROLES.contains(&role.as_str()) {
                role_columns.entry(role).or_default().push(sql_name);
            }
        }
    }

    let mut combined: Vec<String> = scan_columns.to_vec();
    let mut combined_index: HashMap<String, usize> = combined
        .iter()
        .enumerate()
        .map(|(index, name)| (name.clone(), index))
        .collect();
    let mut resolved_rules = Vec::new();
    for (compiled_idx, compiled) in rules.iter().enumerate() {
        let mut conditions = Vec::with_capacity(compiled.rule.conditions.len());
        let mut resolvable = true;
        for condition in &compiled.rule.conditions {
            let target_columns: Vec<String> = if let Some(role) = &condition.role {
                role_columns.get(role).cloned().unwrap_or_default()
            } else {
                let wanted: HashSet<String> = condition
                    .header_any_of
                    .iter()
                    .map(|candidate| library::normalize_header_token(candidate))
                    .collect();
                all_columns
                    .iter()
                    .filter(|column| {
                        wanted.contains(&library::normalize_header_token(&column.original_name))
                            || wanted.contains(&library::normalize_header_token(&column.sql_name))
                    })
                    .map(|column| column.sql_name.clone())
                    .collect()
            };
            if target_columns.is_empty() {
                resolvable = false;
                break;
            }
            let column_indices = target_columns
                .iter()
                .map(|name| {
                    *combined_index.entry(name.clone()).or_insert_with(|| {
                        combined.push(name.clone());
                        combined.len() - 1
                    })
                })
                .collect();
            conditions.push(ResolvedCondition {
                column_indices,
                op: condition.op,
                values_lower: condition
                    .values
                    .iter()
                    .map(|value| value.trim().to_lowercase())
                    .collect(),
            });
        }
        if resolvable {
            resolved_rules.push(ResolvedRule {
                compiled_idx,
                conditions,
            });
        }
    }
    let extra = combined.split_off(scan_columns.len());
    Ok((extra, resolved_rules))
}

/// Returns the first (column index, cell value) satisfying the condition on this row.
fn condition_match<'row>(
    condition: &ResolvedCondition,
    values: &'row [Option<String>],
) -> Option<(usize, &'row str)> {
    for &column_idx in &condition.column_indices {
        let Some(value) = values.get(column_idx).and_then(|value| value.as_deref()) else {
            continue;
        };
        let cell = value.trim();
        if cell.is_empty() {
            continue;
        }
        let cell_lower = cell.to_lowercase();
        let satisfied = match condition.op {
            ConditionOp::EqualsAny => condition
                .values_lower
                .iter()
                .any(|wanted| cell_lower == *wanted),
            ConditionOp::ContainsAny => condition
                .values_lower
                .iter()
                .any(|wanted| cell_lower.contains(wanted.as_str())),
            ConditionOp::EndsWithAny => condition
                .values_lower
                .iter()
                .any(|wanted| cell_lower.ends_with(wanted.as_str())),
        };
        if satisfied {
            return Some((column_idx, cell));
        }
    }
    None
}

fn bounded_match_value(value: &str) -> String {
    const MAX_MATCH_VALUE_CHARS: usize = 48;
    if value.chars().count() <= MAX_MATCH_VALUE_CHARS {
        value.to_string()
    } else {
        let bounded: String = value.chars().take(MAX_MATCH_VALUE_CHARS).collect();
        format!("{bounded}…")
    }
}

struct PendingRuleMatch {
    row_num: i64,
    resolved_idx: usize,
    tactic_idx: usize,
    column_idx: usize,
    matched_value: String,
}

fn scan_with_compiled_library(
    conn: &mut Connection,
    evidence_columns: &[String],
    role_hash: &str,
    compiled: &CompiledLibrary,
    mut on_progress: impl FnMut(i64, i64, &str),
) -> Result<IntelScanSummary> {
    let scan_columns = validate_evidence_columns(conn, evidence_columns)?;
    let (extra_rule_columns, resolved_rules) =
        resolve_behavior_rules(conn, &compiled.rules, &scan_columns)?;
    let mut combined_columns = scan_columns.clone();
    combined_columns.extend(extra_rule_columns);
    let total_rows = count_rows(conn)?;

    db::create_intel_schema(conn)?;
    create_scan_staging_schema(conn)?;
    let scan_token = begin_scan(conn)?;
    on_progress(0, total_rows, "scanning");

    let select_idents: Vec<String> = combined_columns
        .iter()
        .map(|column| db::quote_ident(column))
        .collect();
    let select_sql = format!(
        "SELECT row_num, {} FROM rows
         WHERE row_num > ?1
         ORDER BY row_num ASC
         LIMIT ?2",
        select_idents.join(", ")
    );

    let mut tactic_counts: HashMap<String, CountAccumulator> = HashMap::new();
    let mut technique_counts: HashMap<String, CountAccumulator> = HashMap::new();
    let mut matched_rows = HashSet::new();
    let mut inserted_match_rows = 0i64;
    let mut rows_scanned = 0i64;
    let mut last_row_num = i64::MIN;
    let mut next_progress_at = PROGRESS_INTERVAL_ROWS;

    let scan_result = (|| -> Result<Vec<crate::intel::chains::IntelChainSummary>> {
        loop {
            // Materialize one keyset page and release its read statement before matching.
            // No main-database lock is held during the CPU-heavy Aho-Corasick pass.
            let batch = {
                let mut select_stmt = conn.prepare(&select_sql)?;
                let mut rows =
                    select_stmt.query(rusqlite::params![last_row_num, SCAN_BATCH_ROWS])?;
                let mut batch = Vec::new();
                while let Some(row) = rows.next()? {
                    let row_num: i64 = row.get(0)?;
                    let mut values = Vec::with_capacity(combined_columns.len());
                    for column_idx in 0..combined_columns.len() {
                        values.push(row.get::<_, Option<String>>(column_idx + 1)?);
                    }
                    batch.push((row_num, values));
                }
                batch
            };

            if batch.is_empty() {
                break;
            }

            let mut pending_matches = Vec::new();
            let mut pending_rule_matches = Vec::new();
            for (row_num, values) in &batch {
                last_row_num = *row_num;
                rows_scanned += 1;

                // Only evidence columns feed the keyword automaton; trailing rule-only
                // columns are fetched solely for behavior-rule conditions.
                for (column_idx, value) in values.iter().take(scan_columns.len()).enumerate() {
                    let Some(value) = value.as_deref().filter(|value| !value.is_empty()) else {
                        continue;
                    };
                    let mut seen_patterns_in_cell = HashSet::new();
                    for mat in compiled.automaton.find_overlapping_iter(value) {
                        let pattern_idx = mat.pattern().as_usize();
                        if !seen_patterns_in_cell.insert(pattern_idx) {
                            continue;
                        }
                        let pattern = &compiled.patterns[pattern_idx];
                        if !passes_boundary_check(value, mat.start(), mat.end(), pattern) {
                            continue;
                        }

                        matched_rows.insert(*row_num);
                        increment_count(
                            &mut technique_counts,
                            &pattern.technique_id,
                            &pattern.technique_name,
                            *row_num,
                        );

                        for tactic_idx in 0..pattern.tactic_refs.len() {
                            pending_matches.push(PendingMatch {
                                row_num: *row_num,
                                pattern_idx,
                                tactic_idx,
                                column_idx,
                            });
                            inserted_match_rows += 1;
                            let tactic = &pattern.tactic_refs[tactic_idx];
                            increment_count(&mut tactic_counts, &tactic.id, &tactic.name, *row_num);
                        }
                    }
                }

                for (resolved_idx, resolved) in resolved_rules.iter().enumerate() {
                    let mut first_hit: Option<(usize, String)> = None;
                    let mut all_hold = true;
                    for condition in &resolved.conditions {
                        match condition_match(condition, values) {
                            Some((column_idx, value)) => {
                                if first_hit.is_none() {
                                    first_hit = Some((column_idx, bounded_match_value(value)));
                                }
                            }
                            None => {
                                all_hold = false;
                                break;
                            }
                        }
                    }
                    if !all_hold {
                        continue;
                    }
                    let Some((column_idx, matched_value)) = first_hit else {
                        continue;
                    };
                    let compiled_rule = &compiled.rules[resolved.compiled_idx];
                    matched_rows.insert(*row_num);
                    increment_count(
                        &mut technique_counts,
                        &compiled_rule.rule.technique_id,
                        &compiled_rule.technique_name,
                        *row_num,
                    );
                    for tactic_idx in 0..compiled_rule.tactics.len() {
                        pending_rule_matches.push(PendingRuleMatch {
                            row_num: *row_num,
                            resolved_idx,
                            tactic_idx,
                            column_idx,
                            matched_value: matched_value.clone(),
                        });
                        inserted_match_rows += 1;
                        let tactic = &compiled_rule.tactics[tactic_idx];
                        increment_count(&mut tactic_counts, &tactic.id, &tactic.name, *row_num);
                    }
                }
            }

            // TEMP staging is private to this connection. Each transaction is bounded to one
            // source-row page and therefore does not block independent main-database writers.
            let tx = conn.transaction()?;
            {
                let mut insert_stmt = tx.prepare(&format!(
                    "INSERT INTO {STAGING_TABLE} (
                        row_num, tactic_id, tactic_name, technique_id, technique_name,
                        pattern_id, keyword, column_name, score
                     ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)"
                ))?;
                for pending in pending_matches {
                    let pattern = &compiled.patterns[pending.pattern_idx];
                    let tactic = &pattern.tactic_refs[pending.tactic_idx];
                    insert_stmt.execute(rusqlite::params![
                        pending.row_num,
                        tactic.id,
                        tactic.name,
                        pattern.technique_id,
                        pattern.technique_name,
                        pattern.pattern_id,
                        pattern.keyword,
                        scan_columns[pending.column_idx],
                        pattern.score
                    ])?;
                }
                for pending in pending_rule_matches {
                    let resolved = &resolved_rules[pending.resolved_idx];
                    let compiled_rule = &compiled.rules[resolved.compiled_idx];
                    let tactic = &compiled_rule.tactics[pending.tactic_idx];
                    insert_stmt.execute(rusqlite::params![
                        pending.row_num,
                        tactic.id,
                        tactic.name,
                        compiled_rule.rule.technique_id,
                        compiled_rule.technique_name,
                        compiled_rule.rule.id,
                        pending.matched_value,
                        combined_columns[pending.column_idx],
                        compiled_rule.rule.score
                    ])?;
                }
            }
            tx.commit()?;

            if rows_scanned >= next_progress_at {
                on_progress(rows_scanned, total_rows, "scanning");
                while next_progress_at <= rows_scanned {
                    next_progress_at += PROGRESS_INTERVAL_ROWS;
                }
            }
        }

        // Chains are computed from this scan's private staging rows so the publication
        // transaction below can install matches and chains as one atomic result.
        let chains = crate::intel::chains::compute_chains(conn, STAGING_TABLE)?;

        // The generation check and replacement share one transaction. Readers see either the
        // previous complete scan or this complete scan, never staged/partial rows.
        let tx = conn.transaction()?;
        let active_token: Option<String> = tx
            .query_row(
                "SELECT token FROM _intel_scan_build WHERE singleton = 1",
                [],
                |row| row.get(0),
            )
            .optional()?;
        if active_token.as_deref() != Some(scan_token.as_str()) {
            return Err(anyhow!(
                "intel scan was superseded by a newer scan before publication"
            ));
        }
        tx.execute("DELETE FROM _intel_match", [])?;
        tx.execute("DELETE FROM _intel_scan_info", [])?;
        tx.execute(
            &format!(
                "INSERT INTO _intel_match (
                    row_num, tactic_id, tactic_name, technique_id, technique_name,
                    pattern_id, keyword, column_name, score
                 )
                 SELECT row_num, tactic_id, tactic_name, technique_id, technique_name,
                        pattern_id, keyword, column_name, score
                 FROM {STAGING_TABLE}"
            ),
            [],
        )?;
        tx.execute(
            "INSERT INTO _intel_scan_info (library_hash, role_hash, completed_at)
             VALUES (?1, ?2, ?3)",
            rusqlite::params![
                compiled.library_hash,
                role_hash,
                chrono::Utc::now().to_rfc3339()
            ],
        )?;
        crate::intel::chains::publish_chains(&tx, &chains)?;
        tx.execute(
            "DELETE FROM _intel_scan_build WHERE singleton = 1 AND token = ?1",
            [&scan_token],
        )?;
        tx.commit()?;
        Ok(chains)
    })();

    let chains = match scan_result {
        Ok(chains) => chains,
        Err(error) => {
            // Conditional cleanup cannot cancel a newer scan. A later upsert also makes restart
            // safe if cleanup itself is interrupted or the process exits here.
            let _ = conn.execute(
                "DELETE FROM _intel_scan_build WHERE singleton = 1 AND token = ?1",
                [&scan_token],
            );
            let _ = conn.execute_batch(&format!("DROP TABLE IF EXISTS {STAGING_TABLE}"));
            return Err(error);
        }
    };

    let _ = conn.execute_batch(&format!("DROP TABLE IF EXISTS {STAGING_TABLE}"));

    on_progress(rows_scanned, total_rows, "complete");

    Ok(IntelScanSummary {
        rows_scanned,
        match_count: inserted_match_rows,
        matched_rows: matched_rows.len() as i64,
        library_hash: compiled.library_hash.clone(),
        role_hash: role_hash.to_string(),
        custom_library_error: compiled.custom_library_error.clone(),
        tactics: finalize_counts(tactic_counts),
        techniques: finalize_counts(technique_counts),
        chains,
    })
}

fn create_scan_staging_schema(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute_batch(&format!(
        "CREATE TABLE IF NOT EXISTS _intel_scan_build (
            singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
            token TEXT NOT NULL,
            started_at TEXT NOT NULL
         );
         DROP TABLE IF EXISTS {STAGING_TABLE};
         CREATE TEMP TABLE _intel_match_staging (
            row_num INTEGER NOT NULL,
            tactic_id TEXT NOT NULL,
            tactic_name TEXT NOT NULL,
            technique_id TEXT NOT NULL,
            technique_name TEXT NOT NULL,
            pattern_id TEXT NOT NULL,
            keyword TEXT NOT NULL,
            column_name TEXT NOT NULL,
            score INTEGER NOT NULL
         );"
    ))
}

fn begin_scan(conn: &mut Connection) -> rusqlite::Result<String> {
    let timestamp_nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let counter = SCAN_TOKEN_COUNTER.fetch_add(1, Ordering::Relaxed);
    let token = format!("{}-{timestamp_nanos}-{counter}", std::process::id());
    let tx = conn.transaction()?;
    tx.execute(
        "INSERT INTO _intel_scan_build (singleton, token, started_at)
         VALUES (1, ?1, ?2)
         ON CONFLICT(singleton) DO UPDATE SET
            token = excluded.token,
            started_at = excluded.started_at",
        rusqlite::params![token, chrono::Utc::now().to_rfc3339()],
    )?;
    tx.commit()?;
    Ok(token)
}

fn validate_evidence_columns(
    conn: &Connection,
    evidence_columns: &[String],
) -> Result<Vec<String>> {
    if evidence_columns.is_empty() {
        return Err(anyhow!("no evidence columns were provided"));
    }

    let known: HashSet<String> = db::load_columns(conn)?
        .into_iter()
        .map(|column| column.sql_name)
        .collect();
    let mut seen = HashSet::new();
    let mut valid = Vec::new();
    for column in evidence_columns {
        if !known.contains(column) {
            return Err(anyhow!("unknown evidence column: {column}"));
        }
        if seen.insert(column.clone()) {
            valid.push(column.clone());
        }
    }

    if valid.is_empty() {
        return Err(anyhow!("no evidence columns were provided"));
    }
    Ok(valid)
}

fn count_rows(conn: &Connection) -> rusqlite::Result<i64> {
    conn.query_row("SELECT COUNT(*) FROM rows", [], |row| row.get(0))
}

fn increment_count(
    counts: &mut HashMap<String, CountAccumulator>,
    id: &str,
    name: &str,
    row_num: i64,
) {
    let entry = counts
        .entry(id.to_string())
        .or_insert_with(|| CountAccumulator {
            name: name.to_string(),
            match_count: 0,
            rows: HashSet::new(),
        });
    entry.match_count += 1;
    entry.rows.insert(row_num);
}

fn finalize_counts(counts: HashMap<String, CountAccumulator>) -> Vec<IntelCountSummary> {
    let mut out: Vec<IntelCountSummary> = counts
        .into_iter()
        .map(|(id, count)| IntelCountSummary {
            id,
            name: count.name,
            match_count: count.match_count,
            row_count: count.rows.len() as i64,
        })
        .collect();
    out.sort_by(|a, b| {
        b.match_count
            .cmp(&a.match_count)
            .then_with(|| a.id.cmp(&b.id))
    });
    out
}

fn passes_boundary_check(haystack: &str, start: usize, end: usize, pattern: &PatternMeta) -> bool {
    let requires_boundary = pattern.match_kind == MatchKind::Word
        || is_short_ascii_alphanumeric_pattern(&pattern.keyword);
    if !requires_boundary {
        return true;
    }

    let before_ok = haystack[..start]
        .chars()
        .next_back()
        .is_none_or(|c| !c.is_alphanumeric());
    let after_ok = haystack[end..]
        .chars()
        .next()
        .is_none_or(|c| !c.is_alphanumeric());
    before_ok && after_ok
}

fn is_short_ascii_alphanumeric_pattern(pattern: &str) -> bool {
    let mut char_count = 0usize;
    for ch in pattern.chars() {
        if !ch.is_ascii_alphanumeric() {
            return false;
        }
        char_count += 1;
    }
    char_count <= 3
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::ColumnMeta;
    use crate::intel::library::{Keyword, LoadedLibrary, MatchKind, RuleCondition, Technique};
    use std::path::{Path, PathBuf};
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    struct TestDbFile(PathBuf);

    impl TestDbFile {
        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TestDbFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
            for suffix in ["-journal", "-wal", "-shm"] {
                let _ = std::fs::remove_file(format!("{}{suffix}", self.0.display()));
            }
        }
    }

    fn setup_db(rows: &[&str]) -> (Connection, Vec<String>) {
        let conn = Connection::open_in_memory().unwrap();
        let columns = vec![ColumnMeta {
            sql_name: "commandline".into(),
            original_name: "CommandLine".into(),
            col_index: 0,
            inferred_type: "text".into(),
        }];
        db::create_schema(&conn, &columns).unwrap();
        for (idx, value) in rows.iter().enumerate() {
            conn.execute(
                "INSERT INTO rows (row_num, commandline) VALUES (?1, ?2)",
                rusqlite::params![(idx as i64) + 1, value],
            )
            .unwrap();
        }
        (conn, vec!["commandline".to_string()])
    }

    fn single_keyword_library(pattern: &str, match_kind: MatchKind) -> LoadedLibrary {
        keyword_library(pattern, match_kind, "testhash")
    }

    fn keyword_library(pattern: &str, match_kind: MatchKind, hash: &str) -> LoadedLibrary {
        LoadedLibrary {
            library_ids: vec!["test".into()],
            behavior_rules: vec![],
            techniques: vec![Technique {
                technique_id: "T9999".into(),
                name: "Boundary Test Technique".into(),
                tactics: vec![Tactic {
                    id: "TA9999".into(),
                    name: "Boundary Test Tactic".into(),
                }],
                aliases: vec![],
                keywords: vec![Keyword {
                    id: "test_pattern".into(),
                    pattern: pattern.into(),
                    match_kind,
                    columns: vec!["command_line".into()],
                    score: 50,
                }],
            }],
            library_hash: hash.into(),
            custom_library_error: None,
        }
    }

    fn setup_file_db(row_count: i64, value: &str) -> TestDbFile {
        let unique = SCAN_TOKEN_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "log-parser-matcher-{}-{}-{unique}.sqlite3",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let mut conn = Connection::open(&path).unwrap();
        let columns = vec![ColumnMeta {
            sql_name: "commandline".into(),
            original_name: "CommandLine".into(),
            col_index: 0,
            inferred_type: "text".into(),
        }];
        db::create_schema(&conn, &columns).unwrap();
        let tx = conn.transaction().unwrap();
        {
            let mut stmt = tx
                .prepare("INSERT INTO rows (row_num, commandline) VALUES (?1, ?2)")
                .unwrap();
            for row_num in 1..=row_count {
                stmt.execute(rusqlite::params![row_num, value]).unwrap();
            }
        }
        tx.commit().unwrap();
        drop(conn);
        TestDbFile(path)
    }

    fn seed_published_scan(conn: &Connection, library_hash: &str) {
        db::create_intel_schema(conn).unwrap();
        conn.execute(
            "INSERT INTO _intel_match (
                row_num, tactic_id, tactic_name, technique_id, technique_name,
                pattern_id, keyword, column_name, score
             ) VALUES (1, 'TA-old', 'Old tactic', 'T-old', 'Old technique',
                       'old-pattern', 'old', 'commandline', 1)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO _intel_scan_info (library_hash, role_hash, completed_at)
             VALUES (?1, 'old-role', '2025-01-01T00:00:00Z')",
            [library_hash],
        )
        .unwrap();
    }

    #[test]
    fn scan_flags_known_powershell_keyword() {
        let (mut conn, evidence_columns) = setup_db(&[
            "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe powershell -enc SQBFAFgA",
        ]);
        let library = library::load_builtin_library().unwrap();
        let summary =
            scan_connection_with_library(&mut conn, &evidence_columns, library, |_, _, _| {})
                .unwrap();

        assert_eq!(summary.rows_scanned, 1);
        assert_eq!(summary.matched_rows, 1);
        assert!(summary
            .techniques
            .iter()
            .any(|t| t.id == "T1059.001" && t.row_count == 1));
        assert!(summary
            .tactics
            .iter()
            .any(|t| t.id == "TA0002" && t.row_count == 1));

        let hit: (String, String, String) = conn
            .query_row(
                "SELECT tactic_id, technique_id, pattern_id FROM _intel_match WHERE row_num = 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(hit.0, "TA0002");
        assert_eq!(hit.1, "T1059.001");
        assert_eq!(hit.2, "t1059_001_powershell_enc");
    }

    #[test]
    fn chains_group_multi_tactic_matches_by_host() {
        let mut conn = Connection::open_in_memory().unwrap();
        let columns = vec![
            ColumnMeta {
                sql_name: "commandline".into(),
                original_name: "CommandLine".into(),
                col_index: 0,
                inferred_type: "text".into(),
            },
            ColumnMeta {
                sql_name: "hostname".into(),
                original_name: "Computer".into(),
                col_index: 1,
                inferred_type: "text".into(),
            },
        ];
        db::create_schema(&conn, &columns).unwrap();
        db::create_column_roles_table(&conn).unwrap();
        conn.execute(
            "INSERT INTO _column_roles (role, sql_name, confidence, status, reasons_json)
             VALUES ('host', 'hostname', 0.9, 'suggested', '[]')",
            [],
        )
        .unwrap();
        for (row_num, command, host) in [
            (1i64, "alphaindicator ran", "SRV-A"),
            (2, "betaindicator ran", "SRV-A"),
            (3, "gammaindicator ran", "SRV-A"),
            (4, "alphaindicator ran", "SRV-B"),
        ] {
            conn.execute(
                "INSERT INTO rows (row_num, commandline, hostname) VALUES (?1, ?2, ?3)",
                rusqlite::params![row_num, command, host],
            )
            .unwrap();
        }

        let technique = |tid: &str, taid: &str, keyword: &str| Technique {
            technique_id: tid.into(),
            name: format!("Tech {tid}"),
            tactics: vec![Tactic {
                id: taid.into(),
                name: format!("Tactic {taid}"),
            }],
            aliases: vec![],
            keywords: vec![Keyword {
                id: format!("kw_{tid}"),
                pattern: keyword.into(),
                match_kind: MatchKind::Substring,
                columns: vec![],
                score: 60,
            }],
        };
        let library = LoadedLibrary {
            library_ids: vec!["test".into()],
            behavior_rules: vec![],
            techniques: vec![
                technique("T0001", "TA0001", "alphaindicator"),
                technique("T0002", "TA0002", "betaindicator"),
                technique("T0003", "TA0003", "gammaindicator"),
            ],
            library_hash: "chainhash".into(),
            custom_library_error: None,
        };

        let evidence_columns = vec!["commandline".to_string()];
        let summary =
            scan_connection_with_library(&mut conn, &evidence_columns, library, |_, _, _| {})
                .unwrap();

        assert_eq!(summary.chains.len(), 1, "only SRV-A spans three tactics");
        let chain = &summary.chains[0];
        assert_eq!(chain.host.as_deref(), Some("SRV-A"));
        assert_eq!(chain.tactic_count, 3);
        assert_eq!(chain.row_count, 3);
        assert_eq!(chain.sample_rows, vec![1, 2, 3]);
        assert_eq!(chain.start_epoch_ms, None);

        let published: i64 = conn
            .query_row("SELECT COUNT(*) FROM _intel_chain", [], |row| row.get(0))
            .unwrap();
        assert_eq!(published, 1);
    }

    #[test]
    fn behavior_rule_matches_only_rows_satisfying_all_conditions() {
        let mut conn = Connection::open_in_memory().unwrap();
        let columns = vec![
            ColumnMeta {
                sql_name: "processname".into(),
                original_name: "ProcessName".into(),
                col_index: 0,
                inferred_type: "text".into(),
            },
            ColumnMeta {
                sql_name: "dstport".into(),
                original_name: "DstPort".into(),
                col_index: 1,
                inferred_type: "text".into(),
            },
        ];
        db::create_schema(&conn, &columns).unwrap();
        for (row_num, process, port) in [
            (1i64, "C:\\Program Files\\Google\\Chrome\\chrome.exe", "445"),
            (2, "C:\\Program Files\\Google\\Chrome\\chrome.exe", "443"),
            (3, "C:\\Windows\\System32\\svchost.exe", "445"),
        ] {
            conn.execute(
                "INSERT INTO rows (row_num, processname, dstport) VALUES (?1, ?2, ?3)",
                rusqlite::params![row_num, process, port],
            )
            .unwrap();
        }

        let mut library = single_keyword_library("zz_no_keyword_hit_zz", MatchKind::Substring);
        library.behavior_rules = vec![BehaviorRule {
            id: "rule_browser_smb".into(),
            name: "Browser process to SMB port".into(),
            technique_id: "T9999".into(),
            score: 80,
            conditions: vec![
                RuleCondition {
                    role: None,
                    header_any_of: vec!["ProcessName".into()],
                    op: ConditionOp::EndsWithAny,
                    values: vec!["chrome.exe".into()],
                },
                RuleCondition {
                    role: None,
                    header_any_of: vec!["DstPort".into()],
                    op: ConditionOp::EqualsAny,
                    values: vec!["445".into()],
                },
            ],
        }];

        // dstport is intentionally not an evidence column: it must be fetched and evaluated
        // as a rule-only extra column without entering the keyword automaton.
        let evidence_columns = vec!["processname".to_string()];
        let summary =
            scan_connection_with_library(&mut conn, &evidence_columns, library, |_, _, _| {})
                .unwrap();

        assert_eq!(summary.rows_scanned, 3);
        assert_eq!(summary.matched_rows, 1);
        assert!(summary
            .techniques
            .iter()
            .any(|t| t.id == "T9999" && t.row_count == 1));

        let hit: (i64, String, String, String, i64) = conn
            .query_row(
                "SELECT row_num, pattern_id, keyword, column_name, score FROM _intel_match",
                [],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                    ))
                },
            )
            .unwrap();
        assert_eq!(hit.0, 1);
        assert_eq!(hit.1, "rule_browser_smb");
        assert_eq!(hit.2, "C:\\Program Files\\Google\\Chrome\\chrome.exe");
        assert_eq!(hit.3, "processname");
        assert_eq!(hit.4, 80);
    }

    #[test]
    fn short_pattern_boundary_rejects_substring_inside_longer_word() {
        let (mut conn, evidence_columns) = setup_db(&["internet explorer", "net user /domain"]);
        let library = single_keyword_library("net", MatchKind::Substring);
        let summary =
            scan_connection_with_library(&mut conn, &evidence_columns, library, |_, _, _| {})
                .unwrap();

        assert_eq!(summary.rows_scanned, 2);
        assert_eq!(summary.matched_rows, 1);
        let matched_row: i64 = conn
            .query_row("SELECT row_num FROM _intel_match", [], |row| row.get(0))
            .unwrap();
        assert_eq!(matched_row, 2);
    }

    #[test]
    fn evidence_role_hash_is_stable_and_uses_only_confirmed_evidence_roles() {
        let (conn, _) = setup_db(&[]);
        db::create_column_roles_table(&conn).unwrap();
        conn.execute(
            "INSERT INTO _column_roles (role, sql_name, confidence, status, reasons_json)
             VALUES ('command_line', 'commandline', 1.0, 'confirmed', '[]')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO _column_roles (role, sql_name, confidence, status, reasons_json)
             VALUES ('user', 'commandline', 1.0, 'confirmed', '[]')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO _column_roles (role, sql_name, confidence, status, reasons_json)
             VALUES ('host', 'ignored', 1.0, 'rejected', '[]')",
            [],
        )
        .unwrap();

        assert_eq!(
            confirmed_evidence_columns(&conn).unwrap(),
            vec!["commandline".to_string()]
        );
        let one = role_hash_for_columns(&["commandline".into()]);
        let reordered_duplicates =
            role_hash_for_columns(&["commandline".into(), "commandline".into()]);
        assert_eq!(one, reordered_duplicates);
        assert_eq!(one.len(), 64);
    }

    #[test]
    fn independent_audit_write_succeeds_while_scan_is_paused_between_batches() {
        let db_file = setup_file_db(PROGRESS_INTERVAL_ROWS + SCAN_BATCH_ROWS, "alpha command");
        let setup_conn = Connection::open(db_file.path()).unwrap();
        setup_conn
            .execute_batch(
                "CREATE TABLE _test_audit (
                    id INTEGER PRIMARY KEY,
                    action TEXT NOT NULL
                 );",
            )
            .unwrap();
        drop(setup_conn);

        let scan_path = db_file.path().to_path_buf();
        let (paused_tx, paused_rx) = mpsc::channel();
        let (resume_tx, resume_rx) = mpsc::channel();
        let scan_thread = thread::spawn(move || {
            let mut conn = Connection::open(scan_path).unwrap();
            let mut paused = false;
            scan_connection_with_library(
                &mut conn,
                &["commandline".to_string()],
                keyword_library("alpha", MatchKind::Word, "concurrent-scan"),
                |rows_done, _, phase| {
                    if phase == "scanning" && rows_done >= PROGRESS_INTERVAL_ROWS && !paused {
                        paused = true;
                        paused_tx.send(()).unwrap();
                        resume_rx.recv().unwrap();
                    }
                },
            )
        });

        paused_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        let writer = Connection::open(db_file.path()).unwrap();
        writer
            .execute("INSERT INTO _test_audit (action) VALUES ('accepted')", [])
            .expect("scan must not retain a main-database write transaction");
        resume_tx.send(()).unwrap();

        scan_thread.join().unwrap().unwrap();
        let audit_count: i64 = writer
            .query_row("SELECT COUNT(*) FROM _test_audit", [], |row| row.get(0))
            .unwrap();
        assert_eq!(audit_count, 1);
    }

    #[test]
    fn failed_scan_keeps_previous_publication_and_discards_staged_matches() {
        let db_file = setup_file_db(PROGRESS_INTERVAL_ROWS + SCAN_BATCH_ROWS, "alpha command");
        let setup_conn = Connection::open(db_file.path()).unwrap();
        seed_published_scan(&setup_conn, "previous-good");
        drop(setup_conn);

        let scan_path = db_file.path().to_path_buf();
        let (paused_tx, paused_rx) = mpsc::channel();
        let (resume_tx, resume_rx) = mpsc::channel();
        let scan_thread = thread::spawn(move || {
            let mut conn = Connection::open(scan_path).unwrap();
            let mut paused = false;
            scan_connection_with_library(
                &mut conn,
                &["commandline".to_string()],
                keyword_library("alpha", MatchKind::Word, "failed-rebuild"),
                |rows_done, _, phase| {
                    if phase == "scanning" && rows_done >= PROGRESS_INTERVAL_ROWS && !paused {
                        paused = true;
                        paused_tx.send(()).unwrap();
                        resume_rx.recv().unwrap();
                    }
                },
            )
        });

        paused_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        let breaker = Connection::open(db_file.path()).unwrap();
        breaker.execute("DROP TABLE rows", []).unwrap();
        resume_tx.send(()).unwrap();

        let error = scan_thread.join().unwrap().unwrap_err();
        assert!(error.to_string().contains("no such table: rows"));
        let published_hash: String = breaker
            .query_row("SELECT library_hash FROM _intel_scan_info", [], |row| {
                row.get(0)
            })
            .unwrap();
        let published_pattern: String = breaker
            .query_row("SELECT pattern_id FROM _intel_match", [], |row| row.get(0))
            .unwrap();
        assert_eq!(published_hash, "previous-good");
        assert_eq!(published_pattern, "old-pattern");
    }

    #[test]
    fn superseded_scan_cannot_overwrite_newer_complete_publication() {
        let db_file = setup_file_db(PROGRESS_INTERVAL_ROWS + SCAN_BATCH_ROWS, "alpha command");
        let setup_conn = Connection::open(db_file.path()).unwrap();
        seed_published_scan(&setup_conn, "previous-good");
        drop(setup_conn);

        let slow_path = db_file.path().to_path_buf();
        let (paused_tx, paused_rx) = mpsc::channel();
        let (resume_tx, resume_rx) = mpsc::channel();
        let slow_thread = thread::spawn(move || {
            let mut conn = Connection::open(slow_path).unwrap();
            let mut paused = false;
            scan_connection_with_library(
                &mut conn,
                &["commandline".to_string()],
                keyword_library("alpha", MatchKind::Word, "superseded"),
                |rows_done, _, phase| {
                    if phase == "scanning" && rows_done >= PROGRESS_INTERVAL_ROWS && !paused {
                        paused = true;
                        paused_tx.send(()).unwrap();
                        resume_rx.recv().unwrap();
                    }
                },
            )
        });

        paused_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        let mut newer_conn = Connection::open(db_file.path()).unwrap();
        let newer = scan_connection_with_library(
            &mut newer_conn,
            &["commandline".to_string()],
            keyword_library("beta", MatchKind::Word, "newer-complete"),
            |_, _, _| {},
        )
        .unwrap();
        assert_eq!(newer.match_count, 0);
        resume_tx.send(()).unwrap();

        let error = slow_thread.join().unwrap().unwrap_err();
        assert!(error.to_string().contains("superseded"));
        let published_hash: String = newer_conn
            .query_row("SELECT library_hash FROM _intel_scan_info", [], |row| {
                row.get(0)
            })
            .unwrap();
        let published_matches: i64 = newer_conn
            .query_row("SELECT COUNT(*) FROM _intel_match", [], |row| row.get(0))
            .unwrap();
        assert_eq!(published_hash, "newer-complete");
        assert_eq!(published_matches, 0);
    }
}
