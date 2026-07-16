use crate::db::{self, ColumnMeta};
use anyhow::{anyhow, Result};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
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

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ColumnFilter {
    pub column: String,
    pub op: FilterOp,
    #[serde(default)]
    pub value: String,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum SortDirection {
    Asc,
    Desc,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SortSpec {
    pub column: String,
    pub direction: SortDirection,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct Cursor {
    pub sort_value: Option<String>,
    pub row_num: i64,
}

fn default_limit() -> u32 {
    200
}

/// Maximum complexity accepted for a caller-provided recursive expression. These limits keep an
/// AI-generated plan (or a hostile command payload) from creating pathologically large SQL.
pub const MAX_EXPRESSION_DEPTH: usize = 8;
pub const MAX_EXPRESSION_NODES: usize = 128;
pub const MAX_QUERY_VALUE_LENGTH: usize = 4096;
pub const MAX_ROW_IDS: usize = 1000;
pub const MAX_QUERY_PARAMETERS: usize = 2048;

/// A composable query over the imported raw table. The internally-tagged representation keeps the
/// Tauri command contract explicit, for example:
///
/// `{ "type": "or", "children": [{ "type": "search", "value": "powershell" }] }`
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum QueryExpression {
    And {
        children: Vec<QueryExpression>,
    },
    Or {
        children: Vec<QueryExpression>,
    },
    Not {
        child: Box<QueryExpression>,
    },
    Search {
        value: String,
    },
    Predicate {
        column: String,
        op: FilterOp,
        #[serde(default)]
        value: String,
    },
    /// Trusted backend retrieval can union a bounded semantic candidate set into an expression.
    /// AI-generated JSON must not be permitted to create this variant without backend validation.
    RowIds {
        values: Vec<i64>,
    },
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct QuerySpec {
    #[serde(default)]
    pub search: Option<String>,
    #[serde(default)]
    pub filters: Vec<ColumnFilter>,
    /// Optional recursive raw-table expression. For backwards compatibility it is AND-combined
    /// with the legacy `search` and `filters` fields when either of those are also present.
    #[serde(default)]
    pub expression: Option<QueryExpression>,
    #[serde(default)]
    pub sort: Option<SortSpec>,
    #[serde(default)]
    pub cursor: Option<Cursor>,
    #[serde(default = "default_limit")]
    pub limit: u32,
}

impl Default for QuerySpec {
    fn default() -> Self {
        Self {
            search: None,
            filters: Vec::new(),
            expression: None,
            sort: None,
            cursor: None,
            limit: default_limit(),
        }
    }
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
    s.replace('\\', "\\\\")
        .replace('%', "\\%")
        .replace('_', "\\_")
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

impl std::fmt::Debug for Predicate {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Predicate")
            .field("where_sql", &self.where_sql)
            .field("param_count", &self.params.len())
            .finish()
    }
}

fn validate_value_length(value: &str) -> Result<()> {
    if value.len() > MAX_QUERY_VALUE_LENGTH {
        return Err(anyhow!(
            "query value exceeds maximum length of {MAX_QUERY_VALUE_LENGTH} bytes"
        ));
    }
    Ok(())
}

fn compile_literal_search(
    value: &str,
    params: &mut Vec<Box<dyn rusqlite::ToSql>>,
) -> Result<String> {
    validate_value_length(value)?;
    if value.trim().is_empty() {
        return Err(anyhow!("full-table search value must not be empty"));
    }

    // Wrapped as a literal FTS5 phrase (embedded quotes doubled) so punctuation or
    // boolean-looking words in real log data can't become FTS5 query syntax.
    let escaped = value.replace('"', "\"\"");
    let phrase = format!("\"{escaped}\"");
    params.push(Box::new(phrase));
    Ok("row_num IN (SELECT rowid FROM rows_fts WHERE rows_fts MATCH ?)".to_string())
}

fn compile_column_predicate(
    columns: &[ColumnMeta],
    column: &str,
    op: FilterOp,
    value: &str,
    params: &mut Vec<Box<dyn rusqlite::ToSql>>,
) -> Result<String> {
    validate_value_length(value)?;
    let col = validate_column(columns, column)?;
    let ident = db::quote_ident(&col.sql_name);
    let clause = match op {
        FilterOp::Equals => {
            params.push(Box::new(value.to_string()));
            format!("{ident} = ?")
        }
        FilterOp::NotEquals => {
            params.push(Box::new(value.to_string()));
            format!("{ident} != ?")
        }
        FilterOp::Contains => {
            params.push(Box::new(like_wrap(value, true, true)));
            format!("{ident} LIKE ? ESCAPE '\\'")
        }
        FilterOp::NotContains => {
            params.push(Box::new(like_wrap(value, true, true)));
            format!("{ident} NOT LIKE ? ESCAPE '\\'")
        }
        FilterOp::StartsWith => {
            params.push(Box::new(like_wrap(value, false, true)));
            format!("{ident} LIKE ? ESCAPE '\\'")
        }
        FilterOp::EndsWith => {
            params.push(Box::new(like_wrap(value, true, false)));
            format!("{ident} LIKE ? ESCAPE '\\'")
        }
        FilterOp::IsEmpty => format!("({ident} IS NULL OR {ident} = '')"),
        FilterOp::IsNotEmpty => format!("({ident} IS NOT NULL AND {ident} != '')"),
        // Numeric filter values compare as numbers (CAST ... AS REAL, as before). Anything
        // else - most importantly ISO8601 timestamps - compares as plain text. SQLite's
        // CAST(text AS REAL) only reads the leading numeric prefix.
        FilterOp::GreaterThan => {
            params.push(Box::new(value.to_string()));
            if value.trim().parse::<f64>().is_ok() {
                format!("CAST({ident} AS REAL) > CAST(? AS REAL)")
            } else {
                format!("{ident} > ?")
            }
        }
        FilterOp::LessThan => {
            params.push(Box::new(value.to_string()));
            if value.trim().parse::<f64>().is_ok() {
                format!("CAST({ident} AS REAL) < CAST(? AS REAL)")
            } else {
                format!("{ident} < ?")
            }
        }
    };
    Ok(clause)
}

struct ExpressionCompiler<'a> {
    columns: &'a [ColumnMeta],
    params: &'a mut Vec<Box<dyn rusqlite::ToSql>>,
    node_count: usize,
}

impl ExpressionCompiler<'_> {
    fn compile(&mut self, expression: &QueryExpression, depth: usize) -> Result<String> {
        if depth > MAX_EXPRESSION_DEPTH {
            return Err(anyhow!(
                "query expression exceeds maximum depth of {MAX_EXPRESSION_DEPTH}"
            ));
        }
        self.node_count += 1;
        if self.node_count > MAX_EXPRESSION_NODES {
            return Err(anyhow!(
                "query expression exceeds maximum node count of {MAX_EXPRESSION_NODES}"
            ));
        }

        match expression {
            QueryExpression::And { children } => self.compile_children("AND", children, depth),
            QueryExpression::Or { children } => self.compile_children("OR", children, depth),
            QueryExpression::Not { child } => {
                let clause = self.compile(child, depth + 1)?;
                // Treat SQL NULL as "did not match" before taking the set complement. Without
                // this, NOT(column = value) silently drops rows whose column is NULL.
                Ok(format!("NOT COALESCE(({clause}), 0)"))
            }
            QueryExpression::Search { value } => compile_literal_search(value, self.params),
            QueryExpression::Predicate { column, op, value } => {
                compile_column_predicate(self.columns, column, *op, value, self.params)
            }
            QueryExpression::RowIds { values } => self.compile_row_ids(values),
        }
    }

    fn compile_children(
        &mut self,
        operator: &str,
        children: &[QueryExpression],
        depth: usize,
    ) -> Result<String> {
        if children.is_empty() {
            return Err(anyhow!(
                "{operator} query expression must contain at least one child"
            ));
        }
        let clauses = children
            .iter()
            .map(|child| self.compile(child, depth + 1))
            .collect::<Result<Vec<_>>>()?;
        Ok(format!("({})", clauses.join(&format!(" {operator} "))))
    }

    fn compile_row_ids(&mut self, values: &[i64]) -> Result<String> {
        if values.is_empty() {
            return Err(anyhow!("rowIds query expression must not be empty"));
        }
        if values.len() > MAX_ROW_IDS {
            return Err(anyhow!(
                "rowIds query expression exceeds maximum of {MAX_ROW_IDS} values"
            ));
        }

        let mut unique = std::collections::HashSet::with_capacity(values.len());
        let mut placeholders = Vec::with_capacity(values.len());
        for value in values {
            if *value <= 0 {
                return Err(anyhow!("rowIds query expression values must be positive"));
            }
            if unique.insert(*value) {
                placeholders.push("?");
                self.params.push(Box::new(*value));
            }
        }
        Ok(format!("row_num IN ({})", placeholders.join(", ")))
    }
}

pub(crate) fn build_predicate(columns: &[ColumnMeta], spec: &QuerySpec) -> Result<Predicate> {
    let mut clauses: Vec<String> = Vec::new();
    let mut params: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();

    if let Some(search) = &spec.search {
        validate_value_length(search)?;
        if !search.trim().is_empty() {
            clauses.push(compile_literal_search(search, &mut params)?);
        }
    }

    if spec.filters.len() > MAX_EXPRESSION_NODES {
        return Err(anyhow!(
            "legacy filters exceed maximum count of {MAX_EXPRESSION_NODES}"
        ));
    }

    for filter in &spec.filters {
        clauses.push(compile_column_predicate(
            columns,
            &filter.column,
            filter.op,
            &filter.value,
            &mut params,
        )?);
    }

    if let Some(expression) = &spec.expression {
        let mut compiler = ExpressionCompiler {
            columns,
            params: &mut params,
            node_count: 0,
        };
        clauses.push(compiler.compile(expression, 1)?);
    }

    if params.len() > MAX_QUERY_PARAMETERS {
        return Err(anyhow!(
            "query exceeds maximum parameter count of {MAX_QUERY_PARAMETERS}"
        ));
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
pub fn query_rows(
    conn: &Connection,
    columns: &[ColumnMeta],
    spec: &QuerySpec,
) -> Result<QueryPage> {
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
            (
                format!("ORDER BY {ident} {dir}, row_num {dir}"),
                Some(ident),
            )
        }
        None => ("ORDER BY row_num ASC".to_string(), None),
    };

    let mut clauses: Vec<String> = Vec::new();
    if !predicate.where_sql.is_empty() {
        clauses.push(predicate.where_sql.trim_start_matches("WHERE ").to_string());
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
            Some(Cursor {
                row_num,
                sort_value,
            })
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
            expression: None,
            sort: None,
            cursor: None,
            limit: 200,
        }
    }

    fn expression_spec(expression: QueryExpression) -> QuerySpec {
        QuerySpec {
            expression: Some(expression),
            ..spec()
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

    #[test]
    fn expression_serde_contract_is_internally_tagged_and_camel_case() {
        let expression = QueryExpression::Not {
            child: Box::new(QueryExpression::Predicate {
                column: "account".into(),
                op: FilterOp::NotContains,
                value: "service".into(),
            }),
        };
        let json = serde_json::to_value(&expression).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "type": "not",
                "child": {
                    "type": "predicate",
                    "column": "account",
                    "op": "notContains",
                    "value": "service"
                }
            })
        );
        assert_eq!(
            serde_json::from_value::<QueryExpression>(json).unwrap(),
            expression
        );

        // Older command payloads do not contain the new field and remain valid.
        let legacy: QuerySpec = serde_json::from_value(serde_json::json!({
            "search": "alice",
            "filters": [],
            "limit": 25
        }))
        .unwrap();
        assert_eq!(legacy.expression, None);
    }

    #[test]
    fn synonym_or_full_table_searches_return_each_alternative() {
        let (conn, columns) = setup();
        let s = expression_spec(QueryExpression::Or {
            children: vec![
                QueryExpression::Search {
                    value: "alice".into(),
                },
                QueryExpression::Search {
                    value: "dave".into(),
                },
            ],
        });
        let page = query_rows(&conn, &columns, &s).unwrap();
        let accounts: Vec<_> = page
            .rows
            .iter()
            .map(|row| row["account"].as_str().unwrap())
            .collect();
        assert_eq!(accounts, vec!["alice", "dave"]);
        assert_eq!(count_rows(&conn, &columns, &s).unwrap(), 2);
    }

    #[test]
    fn cross_column_predicate_alternatives_are_or_combined() {
        let (conn, columns) = setup();
        let s = expression_spec(QueryExpression::Or {
            children: vec![
                QueryExpression::Predicate {
                    column: "account".into(),
                    op: FilterOp::Equals,
                    value: "bob".into(),
                },
                QueryExpression::Predicate {
                    column: "event_id".into(),
                    op: FilterOp::GreaterThan,
                    value: "400".into(),
                },
            ],
        });
        let page = query_rows(&conn, &columns, &s).unwrap();
        let row_nums: Vec<_> = page
            .rows
            .iter()
            .map(|row| row["row_num"].as_i64().unwrap())
            .collect();
        assert_eq!(row_nums, vec![2, 5]);
    }

    #[test]
    fn not_expression_excludes_matching_raw_rows() {
        let (conn, columns) = setup();
        let s = expression_spec(QueryExpression::Not {
            child: Box::new(QueryExpression::Or {
                children: vec![
                    QueryExpression::Predicate {
                        column: "account".into(),
                        op: FilterOp::Equals,
                        value: "alice".into(),
                    },
                    QueryExpression::Predicate {
                        column: "event_id".into(),
                        op: FilterOp::Equals,
                        value: "500".into(),
                    },
                ],
            }),
        });
        let page = query_rows(&conn, &columns, &s).unwrap();
        let accounts: Vec<_> = page
            .rows
            .iter()
            .map(|row| row["account"].as_str().unwrap())
            .collect();
        assert_eq!(accounts, vec!["bob", "forensic_test_marker_XYZ", "dave"]);
    }

    #[test]
    fn legacy_fields_are_anded_with_recursive_expression() {
        let (conn, columns) = setup();
        let mut s = expression_spec(QueryExpression::Or {
            children: vec![
                QueryExpression::Search {
                    value: "alice".into(),
                },
                QueryExpression::Search {
                    value: "bob".into(),
                },
            ],
        });
        s.filters.push(ColumnFilter {
            column: "event_id".into(),
            op: FilterOp::GreaterThan,
            value: "150".into(),
        });
        let page = query_rows(&conn, &columns, &s).unwrap();
        assert_eq!(page.rows.len(), 1);
        assert_eq!(page.rows[0]["account"], "bob");
    }

    #[test]
    fn expression_unknown_column_is_rejected_server_side() {
        let (conn, columns) = setup();
        let s = expression_spec(QueryExpression::Predicate {
            column: "account\" OR 1=1 --".into(),
            op: FilterOp::Equals,
            value: "alice".into(),
        });
        let error = query_rows(&conn, &columns, &s).unwrap_err().to_string();
        assert!(error.contains("unknown column"), "{error}");
    }

    #[test]
    fn injection_like_values_are_bound_as_literals() {
        let (conn, columns) = setup();
        let s = expression_spec(QueryExpression::Or {
            children: vec![
                QueryExpression::Predicate {
                    column: "account".into(),
                    op: FilterOp::Equals,
                    value: "alice' OR 1=1 --".into(),
                },
                QueryExpression::Predicate {
                    column: "account".into(),
                    op: FilterOp::Contains,
                    value: "%'; DROP TABLE rows; --_".into(),
                },
            ],
        });
        assert!(query_rows(&conn, &columns, &s).unwrap().rows.is_empty());

        // The table is intact and a normal query still works after the hostile-looking values.
        assert_eq!(count_rows(&conn, &columns, &spec()).unwrap(), 5);
    }

    #[test]
    fn expression_search_treats_fts_operators_as_literal_text() {
        let (conn, columns) = setup();
        let s = expression_spec(QueryExpression::Search {
            value: "alice OR bob account:* \"quoted".into(),
        });
        let page = query_rows(&conn, &columns, &s).unwrap();
        assert!(page.rows.is_empty());
    }

    #[test]
    fn expression_depth_node_and_value_limits_are_enforced() {
        let (_conn, columns) = setup();

        let mut too_deep = QueryExpression::Search {
            value: "alice".into(),
        };
        for _ in 0..MAX_EXPRESSION_DEPTH {
            too_deep = QueryExpression::Not {
                child: Box::new(too_deep),
            };
        }
        let error = build_predicate(&columns, &expression_spec(too_deep))
            .unwrap_err()
            .to_string();
        assert!(error.contains("maximum depth"), "{error}");

        let too_many_nodes = QueryExpression::Or {
            children: (0..MAX_EXPRESSION_NODES)
                .map(|i| QueryExpression::Search {
                    value: format!("term-{i}"),
                })
                .collect(),
        };
        let error = build_predicate(&columns, &expression_spec(too_many_nodes))
            .unwrap_err()
            .to_string();
        assert!(error.contains("maximum node count"), "{error}");

        let too_long = QueryExpression::Predicate {
            column: "account".into(),
            op: FilterOp::Contains,
            value: "x".repeat(MAX_QUERY_VALUE_LENGTH + 1),
        };
        let error = build_predicate(&columns, &expression_spec(too_long))
            .unwrap_err()
            .to_string();
        assert!(error.contains("maximum length"), "{error}");
    }

    #[test]
    fn empty_boolean_groups_and_search_values_are_rejected() {
        let (_conn, columns) = setup();
        for expression in [
            QueryExpression::And { children: vec![] },
            QueryExpression::Or { children: vec![] },
            QueryExpression::Search { value: "  ".into() },
        ] {
            assert!(build_predicate(&columns, &expression_spec(expression)).is_err());
        }
    }

    #[test]
    fn expression_keyset_pagination_covers_filtered_rows_without_overlap() {
        let (conn, columns) = setup();
        let mut s = expression_spec(QueryExpression::Predicate {
            column: "event_id".into(),
            op: FilterOp::GreaterThan,
            value: "100".into(),
        });
        s.limit = 2;

        let mut seen = std::collections::HashSet::new();
        loop {
            let page = query_rows(&conn, &columns, &s).unwrap();
            for row in &page.rows {
                assert!(seen.insert(row["row_num"].as_i64().unwrap()));
            }
            if !page.has_more {
                break;
            }
            s.cursor = page.next_cursor;
        }
        assert_eq!(seen, std::collections::HashSet::from([2, 3, 4, 5]));
    }

    #[test]
    fn row_ids_are_positive_bounded_deduplicated_and_bound() {
        let (conn, columns) = setup();
        let s = expression_spec(QueryExpression::RowIds {
            values: vec![3, 1, 3],
        });
        let predicate = build_predicate(&columns, &s).unwrap();
        assert_eq!(predicate.params.len(), 2, "duplicate row ID must bind once");
        let page = query_rows(&conn, &columns, &s).unwrap();
        let row_nums: Vec<_> = page
            .rows
            .iter()
            .map(|row| row["row_num"].as_i64().unwrap())
            .collect();
        assert_eq!(row_nums, vec![1, 3]);

        for values in [vec![], vec![0], vec![-1]] {
            let invalid = expression_spec(QueryExpression::RowIds { values });
            assert!(build_predicate(&columns, &invalid).is_err());
        }

        let too_many = expression_spec(QueryExpression::RowIds {
            values: (1..=(MAX_ROW_IDS as i64 + 1)).collect(),
        });
        let error = build_predicate(&columns, &too_many)
            .unwrap_err()
            .to_string();
        assert!(error.contains("maximum"), "{error}");

        let too_many_parameters = expression_spec(QueryExpression::Or {
            children: (0..3)
                .map(|_| QueryExpression::RowIds {
                    values: (1..=MAX_ROW_IDS as i64).collect(),
                })
                .collect(),
        });
        let error = build_predicate(&columns, &too_many_parameters)
            .unwrap_err()
            .to_string();
        assert!(error.contains("parameter count"), "{error}");
    }
}
