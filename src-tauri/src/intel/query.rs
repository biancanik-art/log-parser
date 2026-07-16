use crate::db::{self, ColumnMeta};
use crate::intel::parser::{self, GuidedIntent, GuidedSort};
use crate::intel::{library, matcher};
use crate::query::{Cursor, QueryPage};
use anyhow::{anyhow, bail, Result};
use rusqlite::{Connection, OptionalExtension};
use std::collections::BTreeSet;

pub fn run_guided_query(
    conn: &Connection,
    columns: &[ColumnMeta],
    intent_token: &str,
    cursor: Option<Cursor>,
    limit: Option<u32>,
) -> Result<QueryPage> {
    let intent = parser::intent_from_token(intent_token)?;
    if matches!(intent, GuidedIntent::Unknown { .. }) {
        bail!("guided query needs clarification before it can be run");
    }
    if !table_exists(conn, "_intel_match")? {
        bail!("scan intel matches before running a guided query");
    }
    validate_intent_against_current_context(conn, &intent)?;

    let requested_sort = intent_sort(&intent);
    let use_time = requested_sort == GuidedSort::ChronologicalAsc && row_time_has_data(conn)?;
    let filter = GuidedFilter::from_intent(&intent, columns)?;
    query_page(conn, columns, &filter, cursor, limit, use_time)
}

fn validate_intent_against_current_context(conn: &Connection, intent: &GuidedIntent) -> Result<()> {
    let loaded = library::load_merged_library()?;
    let scan_context: Option<(String, String)> = conn
        .query_row(
            "SELECT library_hash, role_hash FROM _intel_scan_info ORDER BY rowid DESC LIMIT 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()?;
    let Some((scanned_library_hash, scanned_role_hash)) = scan_context else {
        bail!("scan intel matches before running a guided query");
    };
    if scanned_library_hash != loaded.library_hash {
        bail!("the intelligence library changed after the scan; rescan before running this query");
    }
    let confirmed_evidence = matcher::confirmed_evidence_columns(conn)?;
    let current_role_hash = matcher::role_hash_for_columns(&confirmed_evidence);
    if scanned_role_hash != current_role_hash {
        bail!(
            "confirmed evidence columns changed after the scan; rescan before running this query"
        );
    }
    let known_techniques = loaded
        .techniques
        .iter()
        .map(|technique| technique.technique_id.as_str())
        .collect::<BTreeSet<_>>();
    let known_tactics = loaded
        .techniques
        .iter()
        .flat_map(|technique| technique.tactics.iter().map(|tactic| tactic.id.as_str()))
        .collect::<BTreeSet<_>>();

    match intent {
        GuidedIntent::SuspiciousScan {
            tactic_ids,
            technique_ids,
            ..
        } => {
            validate_ids(technique_ids, &known_techniques, "technique")?;
            validate_ids(tactic_ids, &known_tactics, "tactic")?;
        }
        GuidedIntent::TechniqueTimeline { technique_ids, .. } => {
            if technique_ids.is_empty() {
                bail!("guided technique timeline has no technique IDs");
            }
            validate_ids(technique_ids, &known_techniques, "technique")?;
        }
        GuidedIntent::UserTechniqueTimeline {
            user_value,
            user_column,
            technique_ids,
            ..
        } => {
            if user_value.trim().is_empty() {
                bail!("guided user timeline has an empty user identity");
            }
            validate_ids(technique_ids, &known_techniques, "technique")?;
            if !table_exists(conn, "_column_roles")? {
                bail!("guided user timeline requires a confirmed user column");
            }
            let confirmed: i64 = conn.query_row(
                "SELECT EXISTS(
                    SELECT 1 FROM _column_roles
                    WHERE role = 'user' AND sql_name = ?1 AND status = 'confirmed'
                 )",
                [user_column],
                |row| row.get(0),
            )?;
            if confirmed == 0 {
                bail!(
                    "guided query references a user column that is no longer confirmed: {user_column}"
                );
            }
        }
        GuidedIntent::Unknown { .. } => bail!("guided query needs clarification"),
    }
    Ok(())
}

fn validate_ids(ids: &[String], known: &BTreeSet<&str>, kind: &str) -> Result<()> {
    if let Some(id) = ids.iter().find(|id| !known.contains(id.as_str())) {
        bail!("guided query references unavailable {kind} ID: {id}");
    }
    Ok(())
}

#[derive(Debug, Clone, Default)]
struct GuidedFilter {
    user_column: Option<String>,
    user_value: Option<String>,
    tactic_ids: Vec<String>,
    technique_ids: Vec<String>,
}

impl GuidedFilter {
    fn from_intent(intent: &GuidedIntent, columns: &[ColumnMeta]) -> Result<Self> {
        match intent {
            GuidedIntent::SuspiciousScan {
                tactic_ids,
                technique_ids,
                ..
            } => Ok(Self {
                tactic_ids: dedup(tactic_ids),
                technique_ids: dedup(technique_ids),
                ..Self::default()
            }),
            GuidedIntent::TechniqueTimeline { technique_ids, .. } => Ok(Self {
                technique_ids: dedup(technique_ids),
                ..Self::default()
            }),
            GuidedIntent::UserTechniqueTimeline {
                user_value,
                user_column,
                technique_ids,
                ..
            } => {
                if !columns.iter().any(|column| column.sql_name == *user_column) {
                    bail!("guided query references a user column that no longer exists: {user_column}");
                }
                Ok(Self {
                    user_column: Some(user_column.clone()),
                    user_value: Some(user_value.clone()),
                    technique_ids: dedup(technique_ids),
                    ..Self::default()
                })
            }
            GuidedIntent::Unknown { .. } => bail!("guided query needs clarification"),
        }
    }
}

fn query_page(
    conn: &Connection,
    columns: &[ColumnMeta],
    filter: &GuidedFilter,
    cursor: Option<Cursor>,
    limit: Option<u32>,
    use_time: bool,
) -> Result<QueryPage> {
    let limit = limit.unwrap_or(200).clamp(1, 5000);
    let limit_plus_one = (limit as i64) + 1;

    let select_cols = rows_column_ident_list(columns);
    let mut params: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
    let match_join = build_match_join(filter, &mut params);

    let mut clauses = Vec::new();
    add_user_clause(filter, &mut clauses, &mut params);
    add_cursor_clause(&cursor, use_time, &mut clauses, &mut params)?;

    let where_sql = if clauses.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", clauses.join(" AND "))
    };

    let (time_join, order_sql, sort_select) = if use_time {
        (
            "JOIN (SELECT row_num, epoch_ms AS guided_epoch_ms FROM _row_time) rt ON rt.row_num = rows.row_num",
            "ORDER BY rt.guided_epoch_ms ASC, rows.row_num ASC",
            "rt.guided_epoch_ms, ",
        )
    } else {
        ("", "ORDER BY rows.row_num ASC", "")
    };

    let sql = format!(
        "SELECT rows.row_num, {sort_select}{select_cols}
         FROM rows
         {time_join}
         {match_join}
         {where_sql}
         {order_sql}
         LIMIT ?"
    );

    params.push(Box::new(limit_plus_one));
    let bound_params: Vec<&dyn rusqlite::ToSql> = params.iter().map(|p| p.as_ref()).collect();
    let mut stmt = conn.prepare(&sql)?;
    let mut rows = stmt.query(bound_params.as_slice())?;

    let column_names: Vec<String> = columns
        .iter()
        .map(|column| column.sql_name.clone())
        .collect();
    let mut fetched = Vec::new();
    while let Some(row) = rows.next()? {
        let row_num: i64 = row.get(0)?;
        let (sort_value, value_start_idx) = if use_time {
            let epoch_ms: i64 = row.get(1)?;
            (Some(epoch_ms.to_string()), 2)
        } else {
            (None, 1)
        };
        let mut obj = serde_json::Map::new();
        obj.insert("row_num".to_string(), serde_json::json!(row_num));
        for (idx, name) in column_names.iter().enumerate() {
            let value: Option<String> = row.get(value_start_idx + idx)?;
            obj.insert(name.clone(), serde_json::json!(value.unwrap_or_default()));
        }
        fetched.push(FetchedGuidedRow {
            row_num,
            sort_value,
            value: serde_json::Value::Object(obj),
        });
    }

    let limit_usize = limit as usize;
    let has_more = fetched.len() > limit_usize;
    if has_more {
        fetched.truncate(limit_usize);
    }

    let next_cursor = if has_more {
        fetched.last().map(|row| Cursor {
            sort_value: row.sort_value.clone(),
            row_num: row.row_num,
        })
    } else {
        None
    };
    let rows = fetched.into_iter().map(|row| row.value).collect();

    Ok(QueryPage {
        rows,
        next_cursor,
        has_more,
    })
}

#[derive(Debug)]
struct FetchedGuidedRow {
    row_num: i64,
    sort_value: Option<String>,
    value: serde_json::Value,
}

fn build_match_join(filter: &GuidedFilter, params: &mut Vec<Box<dyn rusqlite::ToSql>>) -> String {
    let mut clauses = Vec::new();
    if !filter.technique_ids.is_empty() {
        clauses.push(in_clause("technique_id", &filter.technique_ids, params));
    }
    if !filter.tactic_ids.is_empty() {
        clauses.push(in_clause("tactic_id", &filter.tactic_ids, params));
    }
    let where_sql = if clauses.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", clauses.join(" OR "))
    };
    format!(
        "JOIN (
            SELECT DISTINCT row_num FROM _intel_match {where_sql}
         ) intel_rows ON intel_rows.row_num = rows.row_num"
    )
}

fn in_clause(
    column: &str,
    values: &[String],
    params: &mut Vec<Box<dyn rusqlite::ToSql>>,
) -> String {
    for value in values {
        params.push(Box::new(value.clone()));
    }
    let placeholders = vec!["?"; values.len()].join(", ");
    format!("{column} IN ({placeholders})")
}

fn add_user_clause(
    filter: &GuidedFilter,
    clauses: &mut Vec<String>,
    params: &mut Vec<Box<dyn rusqlite::ToSql>>,
) {
    let (Some(column), Some(value)) = (&filter.user_column, &filter.user_value) else {
        return;
    };
    let ident = format!("rows.{}", db::quote_ident(column));
    clauses.push(format!("{ident} = ? COLLATE NOCASE"));
    params.push(Box::new(value.clone()));
}

fn add_cursor_clause(
    cursor: &Option<Cursor>,
    use_time: bool,
    clauses: &mut Vec<String>,
    params: &mut Vec<Box<dyn rusqlite::ToSql>>,
) -> Result<()> {
    let Some(cursor) = cursor else {
        return Ok(());
    };
    if use_time {
        let sort_value = cursor
            .sort_value
            .as_deref()
            .ok_or_else(|| anyhow!("chronological guided cursor is missing sort_value"))?
            .parse::<i64>()
            .map_err(|_| anyhow!("chronological guided cursor has invalid sort_value"))?;
        clauses.push("(rt.guided_epoch_ms, rows.row_num) > (?, ?)".to_string());
        params.push(Box::new(sort_value));
        params.push(Box::new(cursor.row_num));
    } else {
        clauses.push("rows.row_num > ?".to_string());
        params.push(Box::new(cursor.row_num));
    }
    Ok(())
}

fn rows_column_ident_list(columns: &[ColumnMeta]) -> String {
    columns
        .iter()
        .map(|column| format!("rows.{}", db::quote_ident(&column.sql_name)))
        .collect::<Vec<_>>()
        .join(", ")
}

fn intent_sort(intent: &GuidedIntent) -> GuidedSort {
    match intent {
        GuidedIntent::SuspiciousScan { sort, .. }
        | GuidedIntent::TechniqueTimeline { sort, .. }
        | GuidedIntent::UserTechniqueTimeline { sort, .. } => *sort,
        GuidedIntent::Unknown { .. } => GuidedSort::RowNumAsc,
    }
}

fn row_time_has_data(conn: &Connection) -> Result<bool> {
    if !table_exists(conn, "_row_time")? {
        return Ok(false);
    }
    let has_data: i64 = conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM _row_time LIMIT 1)",
        [],
        |row| row.get(0),
    )?;
    Ok(has_data != 0)
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

fn dedup(values: &[String]) -> Vec<String> {
    values
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db;
    use crate::intel::parser::{GuidedIntent, GuidedSort};

    fn setup_fixture(include_time: bool) -> (Connection, Vec<ColumnMeta>, String) {
        let conn = Connection::open_in_memory().unwrap();
        let columns = vec![
            ColumnMeta {
                sql_name: "account".into(),
                original_name: "Account".into(),
                col_index: 0,
                inferred_type: "text".into(),
            },
            ColumnMeta {
                sql_name: "event".into(),
                original_name: "Event".into(),
                col_index: 1,
                inferred_type: "text".into(),
            },
        ];
        db::create_schema(&conn, &columns).unwrap();
        db::create_column_roles_table(&conn).unwrap();
        conn.execute(
            "INSERT INTO _column_roles (role, sql_name, confidence, status, reasons_json)
             VALUES ('user', 'account', 1.0, 'confirmed', '[]')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO _column_roles (role, sql_name, confidence, status, reasons_json)
             VALUES ('text_evidence', 'event', 1.0, 'confirmed', '[]')",
            [],
        )
        .unwrap();
        let rows = [
            (1, "alice", "benign login"),
            (2, "alice", "mimikatz observed later in source row order"),
            (3, "bob", "mimikatz for another user"),
            (4, "alice", "mimikatz observed earlier in UTC order"),
        ];
        for (row_num, account, event) in rows {
            conn.execute(
                "INSERT INTO rows (row_num, account, event) VALUES (?1, ?2, ?3)",
                rusqlite::params![row_num, account, event],
            )
            .unwrap();
        }
        db::create_intel_schema(&conn).unwrap();
        let current_library_hash = library::load_merged_library().unwrap().library_hash;
        let current_role_hash = matcher::role_hash_for_columns(&["event".to_string()]);
        conn.execute(
            "INSERT INTO _intel_scan_info (library_hash, role_hash, completed_at)
             VALUES (?1, ?2, '2026-07-16T00:00:00Z')",
            rusqlite::params![current_library_hash, current_role_hash],
        )
        .unwrap();
        for row_num in [2_i64, 3, 4] {
            conn.execute(
                "INSERT INTO _intel_match (
                    row_num, tactic_id, tactic_name, technique_id, technique_name,
                    pattern_id, keyword, column_name, score
                 ) VALUES (
                    ?1, 'TA0006', 'Credential Access', 'T1003.001',
                    'OS Credential Dumping: LSASS Memory', 'mimikatz', 'mimikatz',
                    'event', 95
                 )",
                [row_num],
            )
            .unwrap();
        }
        if include_time {
            db::create_row_time_table(&conn).unwrap();
            let times = [(2_i64, 300_i64), (3, 100), (4, 200)];
            for (row_num, epoch_ms) in times {
                conn.execute(
                    "INSERT INTO _row_time (row_num, epoch_ms, utc_text, source_text, parse_status)
                     VALUES (?1, ?2, '2026-01-01T00:00:00Z', 'source', 'explicit_offset')",
                    rusqlite::params![row_num, epoch_ms],
                )
                .unwrap();
            }
        }
        let intent = GuidedIntent::UserTechniqueTimeline {
            user_value: "alice".into(),
            user_column: "account".into(),
            technique_ids: vec!["T1003.001".into()],
            sort: GuidedSort::ChronologicalAsc,
        };
        let token = serde_json::to_string(&intent).unwrap();
        (conn, columns, token)
    }

    #[test]
    fn guided_query_returns_chronological_pages_without_overlap() {
        let (conn, columns, token) = setup_fixture(true);
        let first = run_guided_query(&conn, &columns, &token, None, Some(1)).unwrap();
        assert!(first.has_more);
        assert_eq!(first.rows.len(), 1);
        assert_eq!(first.rows[0]["row_num"], serde_json::json!(4));

        let second =
            run_guided_query(&conn, &columns, &token, first.next_cursor.clone(), Some(1)).unwrap();
        assert!(!second.has_more);
        assert_eq!(second.rows.len(), 1);
        assert_eq!(second.rows[0]["row_num"], serde_json::json!(2));
    }

    #[test]
    fn guided_user_filter_executes_the_exact_resolved_identity() {
        let (conn, columns, token) = setup_fixture(false);
        conn.execute(
            "INSERT INTO rows (row_num, account, event)
             VALUES (5, 'bad value\\alice', 'mimikatz malformed identity')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO _intel_match (
                row_num, tactic_id, tactic_name, technique_id, technique_name,
                pattern_id, keyword, column_name, score
             ) VALUES (
                5, 'TA0006', 'Credential Access', 'T1003.001',
                'OS Credential Dumping: LSASS Memory', 'mimikatz', 'mimikatz',
                'event', 95
             )",
            [],
        )
        .unwrap();

        let page = run_guided_query(&conn, &columns, &token, None, Some(10)).unwrap();
        let row_nums = page
            .rows
            .iter()
            .map(|row| row["row_num"].as_i64().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(row_nums, vec![2, 4]);
    }

    #[test]
    fn guided_query_falls_back_to_row_order_without_normalized_time() {
        let (conn, columns, token) = setup_fixture(false);
        let page = run_guided_query(&conn, &columns, &token, None, Some(10)).unwrap();
        let row_nums = page
            .rows
            .iter()
            .map(|row| row["row_num"].as_i64().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(row_nums, vec![2, 4]);
        assert!(!page.has_more);
    }

    #[test]
    fn guided_query_rejects_stale_library_or_role_scan_context() {
        let (conn, columns, token) = setup_fixture(false);
        conn.execute("UPDATE _intel_scan_info SET library_hash = 'stale'", [])
            .unwrap();
        let error = run_guided_query(&conn, &columns, &token, None, Some(10)).unwrap_err();
        assert!(error.to_string().contains("library changed"));

        let current_library_hash = library::load_merged_library().unwrap().library_hash;
        conn.execute(
            "UPDATE _intel_scan_info SET library_hash = ?1, role_hash = 'stale'",
            [current_library_hash],
        )
        .unwrap();
        let error = run_guided_query(&conn, &columns, &token, None, Some(10)).unwrap_err();
        assert!(error.to_string().contains("evidence columns changed"));
    }
}
