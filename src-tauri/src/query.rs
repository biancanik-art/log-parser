use crate::db::{self, ColumnMeta};
use anyhow::{anyhow, Result};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum FilterOp {
    Equals,
    NotEquals,
    Contains,
    NotContains,
    StartsWith,
    EndsWith,
    IsEmpty,
    IsNotEmpty,
    GreaterThan,
    LessThan,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ColumnFilter {
    pub column: String,
    pub op: FilterOp,
    #[serde(default)]
    pub value: String,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SortDirection {
    Asc,
    Desc,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SortSpec {
    pub column: String,
    pub direction: SortDirection,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Cursor {
    pub sort_value: Option<String>,
    pub row_num: i64,
}

fn default_limit() -> u32 {
    200
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QuerySpec {
    #[serde(default)]
    pub search: Option<String>,
    #[serde(default)]
    pub filters: Vec<ColumnFilter>,
    #[serde(default)]
    pub sort: Option<SortSpec>,
    #[serde(default)]
    pub cursor: Option<Cursor>,
    #[serde(default = "default_limit")]
    pub limit: u32,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct QueryPage {
    pub rows: Vec<serde_json::Value>,
    pub next_cursor: Option<Cursor>,
    pub has_more: bool,
}

fn validate_column<'a>(columns: &'a [ColumnMeta], name: &str) -> Result<&'a ColumnMeta> {
    columns
        .iter()
        .find(|c| c.sql_name == name)
        .ok_or_else(|| anyhow!("unknown column: {name}"))
}

fn like_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('%', "\\%").replace('_', "\\_")
}

fn like_wrap(value: &str, leading: bool, trailing: bool) -> String {
    let escaped = like_escape(value);
    format!(
        "{}{escaped}{}",
        if leading { "%" } else { "" },
        if trailing { "%" } else { "" }
    )
}

/// AND-combined predicate shared by `query_rows`, `count_rows`, and `export.rs` — column names
/// are validated against the server-authoritative `columns` list (loaded from `_meta`, never
/// from anything the frontend claims) before being interpolated as quoted identifiers. Every
/// value is bound as a `?` parameter, never string-formatted.
pub(crate) struct Predicate {
    pub where_sql: String,
    pub params: Vec<Box<dyn rusqlite::ToSql>>,
}

pub(crate) fn build_predicate(columns: &[ColumnMeta], spec: &QuerySpec) -> Result<Predicate> {
    let mut clauses: Vec<String> = Vec::new();
    let mut params: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();

    if let Some(search) = spec.search.as_ref().filter(|s| !s.trim().is_empty()) {
        // Wrapped as a literal FTS5 phrase (embedded quotes doubled) so punctuation or
        // boolean-looking words in real log data can't break or be abused as FTS5 query syntax.
        let escaped = search.replace('"', "\"\"");
        let phrase = format!("\"{escaped}\"");
        clauses.push("row_num IN (SELECT rowid FROM rows_fts WHERE rows_fts MATCH ?)".to_string());
        params.push(Box::new(phrase));
    }

    for filter in &spec.filters {
        let col = validate_column(columns, &filter.column)?;
        let ident = db::quote_ident(&col.sql_name);
        match filter.op {
            FilterOp::Equals => {
                clauses.push(format!("{ident} = ?"));
                params.push(Box::new(filter.value.clone()));
            }
            FilterOp::NotEquals => {
                clauses.push(format!("{ident} != ?"));
                params.push(Box::new(filter.value.clone()));
            }
            FilterOp::Contains => {
                clauses.push(format!("{ident} LIKE ? ESCAPE '\\'"));
                params.push(Box::new(like_wrap(&filter.value, true, true)));
            }
            FilterOp::NotContains => {
                clauses.push(format!("{ident} NOT LIKE ? ESCAPE '\\'"));
                params.push(Box::new(like_wrap(&filter.value, true, true)));
            }
            FilterOp::StartsWith => {
                clauses.push(format!("{ident} LIKE ? ESCAPE '\\'"));
                params.push(Box::new(like_wrap(&filter.value, false, true)));
            }
            FilterOp::EndsWith => {
                clauses.push(format!("{ident} LIKE ? ESCAPE '\\'"));
                params.push(Box::new(like_wrap(&filter.value, true, false)));
            }
            FilterOp::IsEmpty => {
                clauses.push(format!("({ident} IS NULL OR {ident} = '')"));
            }
            FilterOp::IsNotEmpty => {
                clauses.push(format!("({ident} IS NOT NULL AND {ident} != '')"));
            }
            // Numeric filter values compare as numbers (CAST ... AS REAL, as before). Anything
            // else - most importantly ISO8601 timestamps like "2026-06-30T09:10:00Z" - compares
            // as plain text instead. SQLite's CAST(text AS REAL) only reads the leading numeric
            // prefix, so every timestamp in the same year previously collapsed to the same
            // value (e.g. 2026.0), making a time-window filter always evaluate false. Plain text
            // comparison is correct for consistently-formatted ISO8601 strings (lexicographic
            // order matches chronological order) and is what an examiner actually needs here.
            FilterOp::GreaterThan => {
                if filter.value.trim().parse::<f64>().is_ok() {
                    clauses.push(format!("CAST({ident} AS REAL) > CAST(? AS REAL)"));
                } else {
                    clauses.push(format!("{ident} > ?"));
                }
                params.push(Box::new(filter.value.clone()));
            }
            FilterOp::LessThan => {
                if filter.value.trim().parse::<f64>().is_ok() {
                    clauses.push(format!("CAST({ident} AS REAL) < CAST(? AS REAL)"));
                } else {
                    clauses.push(format!("{ident} < ?"));
                }
                params.push(Box::new(filter.value.clone()));
            }
        }
    }

    let where_sql = if clauses.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", clauses.join(" AND "))
    };

    Ok(Predicate { where_sql, params })
}

/// `ORDER BY` clause for a given sort spec, defaulting to `row_num ASC`. Shared by `query_rows`
/// (which additionally needs the bare quoted sort identifier for keyset-cursor comparisons, so it
/// doesn't call this directly) and `export.rs` (which previously always exported in `row_num`
/// order regardless of the active on-screen sort — this is what makes exports match the view).
pub(crate) fn build_order_by(columns: &[ColumnMeta], sort: &Option<SortSpec>) -> Result<String> {
    match sort {
        Some(sort) => {
            let col = validate_column(columns, &sort.column)?;
            let ident = db::quote_ident(&col.sql_name);
            let dir = match sort.direction {
                SortDirection::Asc => "ASC",
                SortDirection::Desc => "DESC",
            };
            Ok(format!("ORDER BY {ident} {dir}, row_num {dir}"))
        }
        None => Ok("ORDER BY row_num ASC".to_string()),
    }
}

pub(crate) fn column_ident_list(columns: &[ColumnMeta]) -> String {
    columns
        .iter()
        .map(|c| db::quote_ident(&c.sql_name))
        .collect::<Vec<_>>()
        .join(", ")
}

pub fn count_rows(conn: &Connection, columns: &[ColumnMeta], spec: &QuerySpec) -> Result<i64> {
    let predicate = build_predicate(columns, spec)?;
    let sql = format!("SELECT COUNT(*) FROM rows {}", predicate.where_sql);
    let mut stmt = conn.prepare(&sql)?;
    let params: Vec<&dyn rusqlite::ToSql> = predicate.params.iter().map(|p| p.as_ref()).collect();
    let count: i64 = stmt.query_row(params.as_slice(), |r| r.get(0))?;
    Ok(count)
}

/// Keyset (not OFFSET) pagination: `row_num > cursor` in the default order, or a compound
/// `(sort_col, row_num) > (v, cursor)` keyset when the caller sorts by an arbitrary column, with
/// `row_num` as a tiebreaker for stability across duplicate sort values.
pub fn query_rows(conn: &Connection, columns: &[ColumnMeta], spec: &QuerySpec) -> Result<QueryPage> {
    let predicate = build_predicate(columns, spec)?;
    let limit = spec.limit.clamp(1, 5000);

    let (order_sql, sort_ident) = match &spec.sort {
        Some(sort) => {
            let col = validate_column(columns, &sort.column)?;
            let ident = db::quote_ident(&col.sql_name);
            let dir = match sort.direction {
                SortDirection::Asc => "ASC",
                SortDirection::Desc => "DESC",
            };
            (format!("ORDER BY {ident} {dir}, row_num {dir}"), Some(ident))
        }
        None => ("ORDER BY row_num ASC".to_string(), None),
    };

    let mut clauses: Vec<String> = Vec::new();
    if !predicate.where_sql.is_empty() {
        clauses.push(
            predicate
                .where_sql
                .trim_start_matches("WHERE ")
                .to_string(),
        );
    }

    let mut cursor_params: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
    if let Some(cursor) = &spec.cursor {
        match (&spec.sort, &sort_ident, &cursor.sort_value) {
            (Some(sort), Some(ident), Some(sort_value)) => {
                let op = match sort.direction {
                    SortDirection::Asc => ">",
                    SortDirection::Desc => "<",
                };
                clauses.push(format!("({ident}, row_num) {op} (?, ?)"));
                cursor_params.push(Box::new(sort_value.clone()));
                cursor_params.push(Box::new(cursor.row_num));
            }
            _ => {
                clauses.push("row_num > ?".to_string());
                cursor_params.push(Box::new(cursor.row_num));
            }
        }
    }

    let where_sql = if clauses.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", clauses.join(" AND "))
    };

    // Fetch one extra row to detect has_more without a second COUNT query.
    let limit_plus_one = (limit as i64) + 1;
    let sql = format!(
        "SELECT row_num, {cols} FROM rows {where_sql} {order_sql} LIMIT ?",
        cols = column_ident_list(columns)
    );

    let mut stmt = conn.prepare(&sql)?;
    let mut all_params: Vec<&dyn rusqlite::ToSql> = Vec::new();
    for p in predicate.params.iter() {
        all_params.push(p.as_ref());
    }
    for p in cursor_params.iter() {
        all_params.push(p.as_ref());
    }
    all_params.push(&limit_plus_one);

    let column_names: Vec<String> = columns.iter().map(|c| c.sql_name.clone()).collect();
    let mut rows_out: Vec<serde_json::Value> = Vec::new();
    let mut rows = stmt.query(all_params.as_slice())?;
    while let Some(row) = rows.next()? {
        let row_num: i64 = row.get(0)?;
        let mut obj = serde_json::Map::new();
        obj.insert("row_num".to_string(), serde_json::json!(row_num));
        for (i, name) in column_names.iter().enumerate() {
            let value: String = row.get(i + 1)?;
            obj.insert(name.clone(), serde_json::json!(value));
        }
        rows_out.push(serde_json::Value::Object(obj));
    }

    let limit_usize = limit as usize;
    let has_more = rows_out.len() > limit_usize;
    if has_more {
        rows_out.truncate(limit_usize);
    }

    let next_cursor = if has_more {
        rows_out.last().and_then(|v| {
            let row_num = v.get("row_num")?.as_i64()?;
            let sort_value = spec
                .sort
                .as_ref()
                .and_then(|s| v.get(&s.column))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            Some(Cursor { row_num, sort_value })
        })
    } else {
        None
    };

    Ok(QueryPage {
        rows: rows_out,
        next_cursor,
        has_more,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::ImportInfo;

    fn setup() -> (Connection, Vec<ColumnMeta>) {
        let conn = Connection::open_in_memory().unwrap();
        let columns = vec![
            ColumnMeta {
                sql_name: "account".into(),
                original_name: "Account".into(),
                col_index: 0,
                inferred_type: "text".into(),
            },
            ColumnMeta {
                sql_name: "event_id".into(),
                original_name: "EventID".into(),
                col_index: 1,
                inferred_type: "identifier".into(),
            },
        ];
        db::create_schema(&conn, &columns).unwrap();
        let data = [
            ("alice", "100"),
            ("bob", "200"),
            ("forensic_test_marker_XYZ", "300"),
            ("dave", "400"),
            ("eve", "500"),
        ];
        for (i, (account, event_id)) in data.iter().enumerate() {
            conn.execute(
                "INSERT INTO rows (row_num, account, event_id) VALUES (?1, ?2, ?3)",
                rusqlite::params![(i as i64) + 1, account, event_id],
            )
            .unwrap();
        }
        db::populate_fts(&conn, &columns).unwrap();
        db::record_import_info(
            &conn,
            &ImportInfo {
                source_path: "test.xlsx".into(),
                sheet_name: "Sheet1".into(),
                row_count: data.len() as i64,
                imported_at: "2026-01-01T00:00:00Z".into(),
            },
        )
        .unwrap();
        (conn, columns)
    }

    fn spec() -> QuerySpec {
        QuerySpec {
            search: None,
            filters: vec![],
            sort: None,
            cursor: None,
            limit: 200,
        }
    }

    #[test]
    fn full_text_search_finds_marker() {
        let (conn, columns) = setup();
        let mut s = spec();
        s.search = Some("forensic_test_marker_XYZ".into());
        let page = query_rows(&conn, &columns, &s).unwrap();
        assert_eq!(page.rows.len(), 1);
        assert_eq!(
            page.rows[0]["account"],
            serde_json::json!("forensic_test_marker_XYZ")
        );
    }

    #[test]
    fn column_filter_equals_finds_marker() {
        let (conn, columns) = setup();
        let mut s = spec();
        s.filters.push(ColumnFilter {
            column: "account".into(),
            op: FilterOp::Equals,
            value: "forensic_test_marker_XYZ".into(),
        });
        let page = query_rows(&conn, &columns, &s).unwrap();
        assert_eq!(page.rows.len(), 1);
    }

    #[test]
    fn greater_than_less_than_compare_iso8601_timestamps_as_text_not_numbers() {
        // CAST(text AS REAL) only reads the leading numeric prefix, so every one of these
        // same-year timestamps used to collapse to the same value (2026.0) and a time-window
        // filter always evaluated false. This must compare as text instead.
        let conn = Connection::open_in_memory().unwrap();
        let columns = vec![ColumnMeta {
            sql_name: "ts".into(),
            original_name: "Timestamp".into(),
            col_index: 0,
            inferred_type: "timestamp".into(),
        }];
        db::create_schema(&conn, &columns).unwrap();
        let values = [
            "2026-06-30T07:00:00Z",
            "2026-06-30T09:10:00Z",
            "2026-06-30T11:40:00Z",
            "2026-07-01T08:00:00Z",
        ];
        for (i, value) in values.iter().enumerate() {
            conn.execute(
                "INSERT INTO rows (row_num, ts) VALUES (?1, ?2)",
                rusqlite::params![(i as i64) + 1, value],
            )
            .unwrap();
        }
        db::populate_fts(&conn, &columns).unwrap();
        db::record_import_info(
            &conn,
            &ImportInfo {
                source_path: "test.xlsx".into(),
                sheet_name: "Sheet1".into(),
                row_count: values.len() as i64,
                imported_at: "2026-01-01T00:00:00Z".into(),
            },
        )
        .unwrap();

        let mut s = spec();
        s.filters.push(ColumnFilter {
            column: "ts".into(),
            op: FilterOp::GreaterThan,
            value: "2026-06-30T09:00:00Z".into(),
        });
        s.filters.push(ColumnFilter {
            column: "ts".into(),
            op: FilterOp::LessThan,
            value: "2026-06-30T12:00:00Z".into(),
        });
        let page = query_rows(&conn, &columns, &s).unwrap();
        assert_eq!(
            page.rows.len(),
            2,
            "expected the two 06-30 rows inside the window, got {:?}",
            page.rows
        );
    }

    #[test]
    fn and_combined_filters_narrow_to_zero() {
        let (conn, columns) = setup();
        let mut s = spec();
        s.filters.push(ColumnFilter {
            column: "account".into(),
            op: FilterOp::Equals,
            value: "forensic_test_marker_XYZ".into(),
        });
        s.filters.push(ColumnFilter {
            column: "event_id".into(),
            op: FilterOp::Equals,
            value: "999".into(),
        });
        let page = query_rows(&conn, &columns, &s).unwrap();
        assert_eq!(page.rows.len(), 0);
    }

    #[test]
    fn contains_starts_ends_with() {
        let (conn, columns) = setup();

        let mut s = spec();
        s.filters.push(ColumnFilter {
            column: "account".into(),
            op: FilterOp::Contains,
            value: "marker".into(),
        });
        assert_eq!(query_rows(&conn, &columns, &s).unwrap().rows.len(), 1);

        let mut s = spec();
        s.filters.push(ColumnFilter {
            column: "account".into(),
            op: FilterOp::StartsWith,
            value: "forensic".into(),
        });
        assert_eq!(query_rows(&conn, &columns, &s).unwrap().rows.len(), 1);

        let mut s = spec();
        s.filters.push(ColumnFilter {
            column: "account".into(),
            op: FilterOp::EndsWith,
            value: "XYZ".into(),
        });
        assert_eq!(query_rows(&conn, &columns, &s).unwrap().rows.len(), 1);
    }

    #[test]
    fn numeric_greater_than() {
        let (conn, columns) = setup();
        let mut s = spec();
        s.filters.push(ColumnFilter {
            column: "event_id".into(),
            op: FilterOp::GreaterThan,
            value: "300".into(),
        });
        let page = query_rows(&conn, &columns, &s).unwrap();
        assert_eq!(page.rows.len(), 2); // 400, 500
    }

    #[test]
    fn keyset_pagination_covers_all_rows_without_overlap() {
        let (conn, columns) = setup();
        let mut s = spec();
        s.limit = 2;

        let mut seen = std::collections::HashSet::new();
        let mut cursor = None;
        loop {
            s.cursor = cursor.clone();
            let page = query_rows(&conn, &columns, &s).unwrap();
            for row in &page.rows {
                let row_num = row["row_num"].as_i64().unwrap();
                assert!(seen.insert(row_num), "row {row_num} seen twice");
            }
            if !page.has_more {
                break;
            }
            cursor = page.next_cursor;
        }
        assert_eq!(seen.len(), 5);
    }

    #[test]
    fn unknown_column_is_rejected() {
        let (conn, columns) = setup();
        let mut s = spec();
        s.filters.push(ColumnFilter {
            column: "'; DROP TABLE rows; --".into(),
            op: FilterOp::Equals,
            value: "x".into(),
        });
        assert!(query_rows(&conn, &columns, &s).is_err());
    }

    #[test]
    fn search_with_quotes_does_not_break_fts_syntax() {
        let (conn, columns) = setup();
        let mut s = spec();
        s.search = Some("\"unterminated".into());
        // must not error and must not match anything
        let page = query_rows(&conn, &columns, &s).unwrap();
        assert_eq!(page.rows.len(), 0);
    }
}
