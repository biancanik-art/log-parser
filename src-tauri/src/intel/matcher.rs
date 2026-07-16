use crate::db;
use crate::intel::library::{self, LoadedLibrary, MatchKind, Tactic};
use aho_corasick::{AhoCorasick, AhoCorasickBuilder};
use anyhow::{anyhow, Result};
use rusqlite::Connection;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};

const PROGRESS_INTERVAL_ROWS: i64 = 5000;

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
    library_hash: String,
    custom_library_error: Option<String>,
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
        library_hash: library.library_hash,
        custom_library_error: library.custom_library_error,
    })
}

fn scan_with_compiled_library(
    conn: &mut Connection,
    evidence_columns: &[String],
    role_hash: &str,
    compiled: &CompiledLibrary,
    mut on_progress: impl FnMut(i64, i64, &str),
) -> Result<IntelScanSummary> {
    let scan_columns = validate_evidence_columns(conn, evidence_columns)?;
    let total_rows = count_rows(conn)?;

    db::create_intel_schema(conn)?;
    on_progress(0, total_rows, "scanning");

    let select_idents: Vec<String> = scan_columns
        .iter()
        .map(|column| db::quote_ident(column))
        .collect();
    let select_sql = format!(
        "SELECT row_num, {} FROM rows ORDER BY row_num ASC",
        select_idents.join(", ")
    );

    let mut tactic_counts: HashMap<String, CountAccumulator> = HashMap::new();
    let mut technique_counts: HashMap<String, CountAccumulator> = HashMap::new();
    let mut matched_rows = HashSet::new();
    let mut inserted_match_rows = 0i64;
    let mut rows_scanned = 0i64;

    let tx = conn.transaction()?;
    tx.execute("DELETE FROM _intel_match", [])?;
    tx.execute("DELETE FROM _intel_scan_info", [])?;

    {
        let mut select_stmt = tx.prepare(&select_sql)?;
        let mut insert_stmt = tx.prepare(
            "INSERT INTO _intel_match (
                row_num,
                tactic_id,
                tactic_name,
                technique_id,
                technique_name,
                pattern_id,
                keyword,
                column_name,
                score
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        )?;
        let mut rows = select_stmt.query([])?;
        while let Some(row) = rows.next()? {
            let row_num: i64 = row.get(0)?;
            rows_scanned += 1;

            for (column_idx, column_name) in scan_columns.iter().enumerate() {
                let value: Option<String> = row.get(column_idx + 1)?;
                let Some(value) = value.filter(|v| !v.is_empty()) else {
                    continue;
                };
                let mut seen_patterns_in_cell = HashSet::new();
                for mat in compiled.automaton.find_overlapping_iter(&value) {
                    let pattern_idx = mat.pattern().as_usize();
                    if !seen_patterns_in_cell.insert(pattern_idx) {
                        continue;
                    }
                    let pattern = &compiled.patterns[pattern_idx];
                    if !passes_boundary_check(&value, mat.start(), mat.end(), pattern) {
                        continue;
                    }

                    matched_rows.insert(row_num);
                    increment_count(
                        &mut technique_counts,
                        &pattern.technique_id,
                        &pattern.technique_name,
                        row_num,
                    );

                    for tactic in &pattern.tactic_refs {
                        insert_stmt.execute(rusqlite::params![
                            row_num,
                            tactic.id,
                            tactic.name,
                            pattern.technique_id,
                            pattern.technique_name,
                            pattern.pattern_id,
                            pattern.keyword,
                            column_name,
                            pattern.score
                        ])?;
                        inserted_match_rows += 1;
                        increment_count(&mut tactic_counts, &tactic.id, &tactic.name, row_num);
                    }
                }
            }

            if rows_scanned % PROGRESS_INTERVAL_ROWS == 0 {
                on_progress(rows_scanned, total_rows, "scanning");
            }
        }
    }

    tx.execute(
        "INSERT INTO _intel_scan_info (library_hash, role_hash, completed_at) VALUES (?1, ?2, ?3)",
        rusqlite::params![
            compiled.library_hash,
            role_hash,
            chrono::Utc::now().to_rfc3339()
        ],
    )?;
    tx.commit()?;

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
    })
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
    use crate::intel::library::{Keyword, LoadedLibrary, MatchKind, Technique};

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
        LoadedLibrary {
            library_ids: vec!["test".into()],
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
            library_hash: "testhash".into(),
            custom_library_error: None,
        }
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
}
