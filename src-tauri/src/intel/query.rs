use crate::db::{self, ColumnMeta};
use crate::intel::parser::{
    self, GuidedIntent, GuidedSort, RawFilterOp, RawSearchAlternative, RawSortDirection,
};
use crate::intel::{library, matcher, time};
use crate::query::{Cursor, QueryPage, SortDirection};
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
    if matches!(intent, GuidedIntent::RawEvidenceSearch { .. }) {
        let spec = parser::query_spec_from_raw_intent(&intent, cursor, limit)?;
        let normalized = matches!(
            &intent,
            GuidedIntent::RawEvidenceSearch {
                sort: Some(sort),
                ..
            } if sort.normalized_time
        );
        let mut page = if normalized {
            query_raw_normalized_time(conn, columns, &intent, &spec)?
        } else {
            crate::query::query_rows(conn, columns, &spec)?
        };
        annotate_raw_matches(&mut page, &intent);
        return Ok(page);
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

pub fn normalized_raw_sort_direction(
    conn: &Connection,
    columns: &[ColumnMeta],
    intent: &GuidedIntent,
) -> Result<Option<(String, SortDirection)>> {
    let GuidedIntent::RawEvidenceSearch {
        sort: Some(sort), ..
    } = intent
    else {
        return Ok(None);
    };
    if !sort.normalized_time {
        return Ok(None);
    }
    time::require_row_time_binding(conn, columns, &sort.column)?;
    let direction = match sort.direction {
        RawSortDirection::Asc => SortDirection::Asc,
        RawSortDirection::Desc => SortDirection::Desc,
    };
    Ok(Some((sort.column.clone(), direction)))
}

fn query_raw_normalized_time(
    conn: &Connection,
    columns: &[ColumnMeta],
    intent: &GuidedIntent,
    spec: &crate::query::QuerySpec,
) -> Result<QueryPage> {
    let GuidedIntent::RawEvidenceSearch {
        sort: Some(sort), ..
    } = intent
    else {
        bail!("normalized raw timeline is missing its sort plan");
    };
    if !sort.normalized_time {
        bail!("raw timeline did not request normalized time");
    }
    if !columns.iter().any(|column| column.sql_name == sort.column) {
        bail!("raw timeline references an unavailable timestamp column");
    }
    time::require_row_time_binding(conn, columns, &sort.column)?;
    let predicate = crate::query::build_predicate(columns, spec)?;
    let limit = spec.limit.clamp(1, 5000);
    let direction = match sort.direction {
        RawSortDirection::Asc => "ASC",
        RawSortDirection::Desc => "DESC",
    };
    let cursor_operator = match sort.direction {
        RawSortDirection::Asc => ">",
        RawSortDirection::Desc => "<",
    };
    let mut cursor_clause = String::new();
    let mut cursor_params: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
    if let Some(cursor) = &spec.cursor {
        if let Some(sort_value) = cursor.sort_value.as_deref() {
            let epoch_ms = sort_value
                .parse::<i64>()
                .map_err(|_| anyhow!("normalized timeline cursor has an invalid timestamp"))?;
            cursor_clause = format!(
                "WHERE ((rt.epoch_ms IS NOT NULL AND \
                 (rt.epoch_ms {cursor_operator} ? OR \
                  (rt.epoch_ms = ? AND raw.row_num {cursor_operator} ?))) \
                 OR rt.epoch_ms IS NULL)"
            );
            cursor_params.push(Box::new(epoch_ms));
            cursor_params.push(Box::new(epoch_ms));
            cursor_params.push(Box::new(cursor.row_num));
        } else {
            cursor_clause =
                format!("WHERE rt.epoch_ms IS NULL AND raw.row_num {cursor_operator} ?");
            cursor_params.push(Box::new(cursor.row_num));
        }
    }
    let raw_columns = columns
        .iter()
        .map(|column| format!("raw.{}", db::quote_ident(&column.sql_name)))
        .collect::<Vec<_>>()
        .join(", ");
    let limit_plus_one = i64::from(limit) + 1;
    let sql = format!(
        "SELECT raw.row_num, {raw_columns}, rt.epoch_ms
         FROM (SELECT * FROM rows {predicate_where}) raw
         LEFT JOIN _row_time rt ON rt.row_num = raw.row_num
         {cursor_clause}
         ORDER BY CASE WHEN rt.epoch_ms IS NULL THEN 1 ELSE 0 END ASC,
                  rt.epoch_ms {direction}, raw.row_num {direction}
         LIMIT ?",
        predicate_where = predicate.where_sql,
    );
    let mut params = predicate
        .params
        .iter()
        .map(|param| param.as_ref() as &dyn rusqlite::ToSql)
        .collect::<Vec<_>>();
    params.extend(
        cursor_params
            .iter()
            .map(|param| param.as_ref() as &dyn rusqlite::ToSql),
    );
    params.push(&limit_plus_one);

    let mut stmt = conn.prepare(&sql)?;
    let mut rows = stmt.query(params.as_slice())?;
    let mut fetched = Vec::new();
    while let Some(row) = rows.next()? {
        let row_num: i64 = row.get(0)?;
        let mut object = serde_json::Map::new();
        object.insert("row_num".to_string(), serde_json::json!(row_num));
        for (index, column) in columns.iter().enumerate() {
            let value: Option<String> = row.get(index + 1)?;
            object.insert(
                column.sql_name.clone(),
                serde_json::json!(value.unwrap_or_default()),
            );
        }
        let epoch_ms: Option<i64> = row.get(columns.len() + 1)?;
        fetched.push((row_num, epoch_ms, serde_json::Value::Object(object)));
    }
    let has_more = fetched.len() > limit as usize;
    if has_more {
        fetched.truncate(limit as usize);
    }
    let next_cursor = if has_more {
        fetched.last().map(|(row_num, epoch_ms, _)| Cursor {
            sort_value: epoch_ms.map(|value| value.to_string()),
            row_num: *row_num,
        })
    } else {
        None
    };
    Ok(QueryPage {
        rows: fetched.into_iter().map(|(_, _, value)| value).collect(),
        next_cursor,
        has_more,
    })
}

fn annotate_raw_matches(page: &mut QueryPage, intent: &GuidedIntent) {
    let GuidedIntent::RawEvidenceSearch {
        alternatives,
        semantic_row_ids,
        ..
    } = intent
    else {
        return;
    };
    let semantic_ids = semantic_row_ids.iter().copied().collect::<BTreeSet<_>>();
    for row in &mut page.rows {
        let Some(object) = row.as_object_mut() else {
            continue;
        };
        let row_num = object.get("row_num").and_then(serde_json::Value::as_i64);
        let searchable_values = object
            .iter()
            .filter(|(name, _)| name.as_str() != "row_num" && name.as_str() != "__aiMatch")
            .filter_map(|(_, value)| value.as_str())
            .collect::<Vec<_>>();
        let mut reasons = Vec::new();
        for (index, alternative) in alternatives.iter().enumerate() {
            if alternative_matches(object, &searchable_values, alternative) {
                for term in &alternative.terms {
                    reasons.push(format!("alternative {} literal: {term}", index + 1));
                }
                for filter in &alternative.filters {
                    reasons.push(format!(
                        "alternative {} filter: {} {:?} {}",
                        index + 1,
                        filter.column,
                        filter.op,
                        filter.value
                    ));
                }
            }
        }
        if row_num.is_some_and(|row_num| semantic_ids.contains(&row_num)) {
            reasons.push("semantic recall candidate".to_string());
        }
        object.insert("__aiMatch".to_string(), serde_json::json!(reasons));
    }
}

fn alternative_matches(
    object: &serde_json::Map<String, serde_json::Value>,
    searchable_values: &[&str],
    alternative: &RawSearchAlternative,
) -> bool {
    let terms_match = alternative.terms.iter().all(|term| {
        let needle = term.to_ascii_lowercase();
        searchable_values
            .iter()
            .any(|value| value.to_ascii_lowercase().contains(&needle))
    });
    terms_match
        && alternative.filters.iter().all(|filter| {
            object
                .get(&filter.column)
                .and_then(serde_json::Value::as_str)
                .is_some_and(|value| raw_filter_matches(value, filter.op, &filter.value))
        })
}

fn raw_filter_matches(value: &str, op: RawFilterOp, expected: &str) -> bool {
    let value_folded = value.to_ascii_lowercase();
    let expected_folded = expected.to_ascii_lowercase();
    match op {
        RawFilterOp::Equals => value == expected,
        RawFilterOp::NotEquals => value != expected,
        RawFilterOp::Contains => value_folded.contains(&expected_folded),
        RawFilterOp::NotContains => !value_folded.contains(&expected_folded),
        RawFilterOp::StartsWith => value_folded.starts_with(&expected_folded),
        RawFilterOp::EndsWith => value_folded.ends_with(&expected_folded),
        RawFilterOp::IsEmpty => value.is_empty(),
        RawFilterOp::IsNotEmpty => !value.is_empty(),
        RawFilterOp::GreaterThan | RawFilterOp::LessThan => {
            let ordering = match (value.trim().parse::<f64>(), expected.trim().parse::<f64>()) {
                (Ok(left), Ok(right)) => left.partial_cmp(&right),
                _ => Some(value.cmp(expected)),
            };
            match op {
                RawFilterOp::GreaterThan => ordering.is_some_and(|order| order.is_gt()),
                RawFilterOp::LessThan => ordering.is_some_and(|order| order.is_lt()),
                _ => false,
            }
        }
    }
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
    let Some((scanned_library_hash, scanned_evidence_hash)) = scan_context else {
        bail!("scan intel matches before running a guided query");
    };
    if scanned_library_hash != loaded.library_hash {
        bail!("the intelligence library changed after the scan; rescan before running this query");
    }
    let active_evidence = active_evidence_columns(conn)?;
    let current_evidence_hash = matcher::role_hash_for_columns(&active_evidence);
    if scanned_evidence_hash != current_evidence_hash {
        bail!("active evidence columns changed after the scan; rescan before running this query");
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
        GuidedIntent::RawEvidenceSearch { .. } => {}
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

/// Automatic, non-rejected data mappings are sufficient for optional MITRE enrichment. They are
/// deliberately not consulted by the raw AI path.
pub fn active_evidence_columns(conn: &Connection) -> Result<Vec<String>> {
    if !table_exists(conn, "_column_roles")? {
        return Ok(Vec::new());
    }
    let mut stmt = conn.prepare(
        "SELECT sql_name FROM _column_roles
         WHERE status IN ('suggested', 'confirmed')
           AND role IN ('command_line', 'process_name', 'file_name', 'host', 'text_evidence')
         ORDER BY sql_name",
    )?;
    let mut columns = stmt
        .query_map([], |row| row.get::<_, String>(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    columns.sort();
    columns.dedup();
    Ok(columns)
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
            GuidedIntent::RawEvidenceSearch { .. } => {
                bail!("raw evidence search must use the core query engine")
            }
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
        GuidedIntent::RawEvidenceSearch { .. } => GuidedSort::RowNumAsc,
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
    use crate::intel::parser::{
        GuidedIntent, GuidedSort, RawSearchAlternative, RawSearchSort, RawSortDirection,
    };

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
    fn guided_query_rejects_stale_library_or_active_evidence_context() {
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
        assert!(error
            .to_string()
            .contains("active evidence columns changed"));
    }

    #[test]
    fn raw_evidence_search_runs_without_scan_and_unions_semantic_candidates() {
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
        for (row_num, event) in [
            (1_i64, "powershell download"),
            (2, "pwsh encoded command"),
            (3, "script interpreter activity"),
            (4, "ordinary login"),
        ] {
            conn.execute(
                "INSERT INTO rows (row_num, account, event) VALUES (?1, 'alice', ?2)",
                rusqlite::params![row_num, event],
            )
            .unwrap();
        }
        db::populate_fts(&conn, &columns).unwrap();
        let token = serde_json::to_string(&GuidedIntent::RawEvidenceSearch {
            alternatives: vec![
                RawSearchAlternative {
                    terms: vec!["powershell".into()],
                    filters: vec![],
                },
                RawSearchAlternative {
                    terms: vec!["pwsh".into()],
                    filters: vec![],
                },
            ],
            sort: None,
            semantic_row_ids: vec![3],
        })
        .unwrap();
        assert!(!table_exists(&conn, "_intel_match").unwrap());
        let page = run_guided_query(&conn, &columns, &token, None, Some(10)).unwrap();
        let row_nums = page
            .rows
            .iter()
            .map(|row| row["row_num"].as_i64().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(row_nums, vec![1, 2, 3]);
        assert!(page.rows.iter().all(|row| row["__aiMatch"].is_array()));
        assert!(page.rows[2]["__aiMatch"]
            .as_array()
            .unwrap()
            .iter()
            .any(|reason| reason == "semantic recall candidate"));
    }

    #[test]
    fn raw_timeline_keeps_unparsed_rows_last_across_keyset_pages() {
        let mut conn = Connection::open_in_memory().unwrap();
        let columns = vec![
            ColumnMeta {
                sql_name: "event_time".into(),
                original_name: "Event Time".into(),
                col_index: 0,
                inferred_type: "timestamp".into(),
            },
            ColumnMeta {
                sql_name: "event".into(),
                original_name: "Event".into(),
                col_index: 1,
                inferred_type: "text".into(),
            },
        ];
        db::create_schema(&conn, &columns).unwrap();
        conn.execute(
            "INSERT INTO rows (row_num, event_time, event) VALUES
             (1, '2026-01-01T03:00:00+02:00', 'evidence marker'),
             (2, '2025-12-31T23:30:00-01:00', 'evidence marker'),
             (3, 'not-a-time', 'evidence marker invalid'),
             (4, '', 'evidence marker blank'),
             (5, '2026-01-01T00:30:00Z', 'evidence marker tied')",
            [],
        )
        .unwrap();
        db::populate_fts(&conn, &columns).unwrap();
        time::normalize_timestamp_column_with_options(&mut conn, &columns, None, None).unwrap();
        let token = serde_json::to_string(&GuidedIntent::RawEvidenceSearch {
            alternatives: vec![RawSearchAlternative {
                terms: vec!["evidence marker".into()],
                filters: vec![],
            }],
            sort: Some(RawSearchSort {
                column: "event_time".into(),
                direction: RawSortDirection::Asc,
                normalized_time: true,
            }),
            semantic_row_ids: vec![],
        })
        .unwrap();
        let first = run_guided_query(&conn, &columns, &token, None, Some(1)).unwrap();
        assert_eq!(first.rows[0]["row_num"], 2);
        let second = run_guided_query(&conn, &columns, &token, first.next_cursor, Some(1)).unwrap();
        assert_eq!(second.rows[0]["row_num"], 5);
        let third = run_guided_query(&conn, &columns, &token, second.next_cursor, Some(1)).unwrap();
        assert_eq!(third.rows[0]["row_num"], 1);
        let fourth = run_guided_query(&conn, &columns, &token, third.next_cursor, Some(1)).unwrap();
        assert_eq!(fourth.rows[0]["row_num"], 3);
        assert!(fourth.next_cursor.as_ref().unwrap().sort_value.is_none());
        let fifth = run_guided_query(&conn, &columns, &token, fourth.next_cursor, Some(1)).unwrap();
        assert_eq!(fifth.rows[0]["row_num"], 4);
        assert!(!fifth.has_more);

        let descending = serde_json::to_string(&GuidedIntent::RawEvidenceSearch {
            alternatives: vec![RawSearchAlternative {
                terms: vec!["evidence marker".into()],
                filters: vec![],
            }],
            sort: Some(RawSearchSort {
                column: "event_time".into(),
                direction: RawSortDirection::Desc,
                normalized_time: true,
            }),
            semantic_row_ids: vec![],
        })
        .unwrap();
        let mut cursor = None;
        let mut descending_rows = Vec::new();
        loop {
            let page = run_guided_query(&conn, &columns, &descending, cursor, Some(1)).unwrap();
            descending_rows.push(page.rows[0]["row_num"].as_i64().unwrap());
            if !page.has_more {
                break;
            }
            cursor = page.next_cursor;
        }
        assert_eq!(descending_rows, vec![1, 5, 2, 4, 3]);
    }

    #[test]
    fn raw_timeline_rejects_wrong_column_and_stale_binding() {
        let mut conn = Connection::open_in_memory().unwrap();
        let columns = vec![
            ColumnMeta {
                sql_name: "event_time".into(),
                original_name: "Event Time".into(),
                col_index: 0,
                inferred_type: "timestamp".into(),
            },
            ColumnMeta {
                sql_name: "ingest_time".into(),
                original_name: "Ingest Time".into(),
                col_index: 1,
                inferred_type: "timestamp".into(),
            },
            ColumnMeta {
                sql_name: "event".into(),
                original_name: "Event".into(),
                col_index: 2,
                inferred_type: "text".into(),
            },
        ];
        db::create_schema(&conn, &columns).unwrap();
        db::create_column_roles_table(&conn).unwrap();
        conn.execute(
            "INSERT INTO _column_roles (role, sql_name, confidence, status, reasons_json)
             VALUES ('timestamp', 'event_time', 1.0, 'confirmed', '[]')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO rows (row_num, event_time, ingest_time, event)
             VALUES (1, '2026-01-01T00:00:00Z', '2026-01-02T00:00:00Z', 'marker')",
            [],
        )
        .unwrap();
        db::populate_fts(&conn, &columns).unwrap();
        time::normalize_timestamp_column_with_options(&mut conn, &columns, None, None).unwrap();

        let token_for = |column: &str| {
            serde_json::to_string(&GuidedIntent::RawEvidenceSearch {
                alternatives: vec![RawSearchAlternative {
                    terms: vec!["marker".into()],
                    filters: vec![],
                }],
                sort: Some(RawSearchSort {
                    column: column.into(),
                    direction: RawSortDirection::Asc,
                    normalized_time: true,
                }),
                semantic_row_ids: vec![],
            })
            .unwrap()
        };

        let wrong = run_guided_query(&conn, &columns, &token_for("ingest_time"), None, Some(10))
            .unwrap_err();
        assert!(wrong.to_string().contains("bound to column 'event_time'"));

        conn.execute(
            "INSERT INTO rows (row_num, event_time, ingest_time, event)
             VALUES (2, '2026-01-03T00:00:00Z', '2026-01-04T00:00:00Z', 'marker')",
            [],
        )
        .unwrap();
        let stale = run_guided_query(&conn, &columns, &token_for("event_time"), None, Some(10))
            .unwrap_err();
        assert!(stale.to_string().contains("stale"));
    }
}
