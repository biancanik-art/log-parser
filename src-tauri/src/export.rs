use crate::db::ColumnMeta;
use crate::intel::time;
use crate::query::{self, QuerySpec};
use anyhow::Result;
use rusqlite::Connection;
use rust_xlsxwriter::Workbook;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

#[derive(Debug)]
pub struct ExportSummary {
    pub row_count: i64,
}

const PROGRESS_EVERY: i64 = 5000;

/// Streams matching rows straight from a `rusqlite` row cursor into the destination file — no
/// intermediate `Vec`/JSON blob of the whole result set is ever materialized. Reuses
/// `query::build_predicate` and `query::build_order_by` so the exported set — filters, search,
/// *and* the active sort — always matches what's on screen, not just the row set.
fn build_export_query(
    columns: &[ColumnMeta],
    spec: &QuerySpec,
) -> Result<(String, query::Predicate)> {
    let predicate = query::build_predicate(columns, spec)?;
    let order_by = query::build_order_by(columns, &spec.sort)?;
    let sql = format!(
        "SELECT {cols} FROM rows {where_sql} {order_by}",
        cols = query::column_ident_list(columns),
        where_sql = predicate.where_sql
    );
    Ok((sql, predicate))
}

fn build_normalized_time_export_query(
    conn: &Connection,
    columns: &[ColumnMeta],
    spec: &QuerySpec,
    source_column: &str,
    direction: query::SortDirection,
) -> Result<(String, query::Predicate)> {
    time::require_row_time_binding(conn, columns, source_column)?;
    let predicate = query::build_predicate(columns, spec)?;
    let raw_columns = columns
        .iter()
        .map(|column| format!("raw.{}", crate::db::quote_ident(&column.sql_name)))
        .collect::<Vec<_>>()
        .join(", ");
    let direction = match direction {
        query::SortDirection::Asc => "ASC",
        query::SortDirection::Desc => "DESC",
    };
    // Keep the predicate inside a rows-only subquery because FTS and trusted rowIds compile to
    // unqualified `row_num`; joining `_row_time` first would make that identifier ambiguous.
    let sql = format!(
        "SELECT {raw_columns}
         FROM (SELECT * FROM rows {where_sql}) raw
         LEFT JOIN _row_time rt ON rt.row_num = raw.row_num
         ORDER BY CASE WHEN rt.epoch_ms IS NULL THEN 1 ELSE 0 END ASC,
                  rt.epoch_ms {direction}, raw.row_num {direction}",
        where_sql = predicate.where_sql,
    );
    Ok((sql, predicate))
}

pub fn export_csv(
    conn: &Connection,
    columns: &[ColumnMeta],
    spec: &QuerySpec,
    dest_path: &Path,
    mut on_progress: impl FnMut(i64),
) -> Result<ExportSummary> {
    let (sql, predicate) = build_export_query(columns, spec)?;

    let file = File::create(dest_path)?;
    let mut writer = csv::Writer::from_writer(BufWriter::new(file));
    let headers: Vec<&str> = columns.iter().map(|c| c.original_name.as_str()).collect();
    writer.write_record(&headers)?;

    let mut stmt = conn.prepare(&sql)?;
    let params: Vec<&dyn rusqlite::ToSql> = predicate.params.iter().map(|p| p.as_ref()).collect();
    let mut rows = stmt.query(params.as_slice())?;

    let mut row_count: i64 = 0;
    let mut record: Vec<String> = vec![String::new(); columns.len()];
    while let Some(row) = rows.next()? {
        for (i, cell) in record.iter_mut().enumerate() {
            *cell = row.get(i)?;
        }
        writer.write_record(&record)?;
        row_count += 1;
        if row_count % PROGRESS_EVERY == 0 {
            on_progress(row_count);
        }
    }
    writer.flush()?;
    on_progress(row_count);

    Ok(ExportSummary { row_count })
}

pub fn export_csv_normalized_time(
    conn: &Connection,
    columns: &[ColumnMeta],
    spec: &QuerySpec,
    source_column: &str,
    direction: query::SortDirection,
    dest_path: &Path,
    mut on_progress: impl FnMut(i64),
) -> Result<ExportSummary> {
    let (sql, predicate) =
        build_normalized_time_export_query(conn, columns, spec, source_column, direction)?;

    let file = File::create(dest_path)?;
    let mut writer = csv::Writer::from_writer(BufWriter::new(file));
    let headers: Vec<&str> = columns
        .iter()
        .map(|column| column.original_name.as_str())
        .collect();
    writer.write_record(&headers)?;

    let mut stmt = conn.prepare(&sql)?;
    let params: Vec<&dyn rusqlite::ToSql> = predicate
        .params
        .iter()
        .map(|param| param.as_ref())
        .collect();
    let mut rows = stmt.query(params.as_slice())?;
    let mut row_count = 0i64;
    let mut record = vec![String::new(); columns.len()];
    while let Some(row) = rows.next()? {
        for (index, cell) in record.iter_mut().enumerate() {
            *cell = row.get(index)?;
        }
        writer.write_record(&record)?;
        row_count += 1;
        if row_count % PROGRESS_EVERY == 0 {
            on_progress(row_count);
        }
    }
    writer.flush()?;
    on_progress(row_count);
    Ok(ExportSummary { row_count })
}

pub fn export_xlsx(
    conn: &Connection,
    columns: &[ColumnMeta],
    spec: &QuerySpec,
    dest_path: &Path,
    mut on_progress: impl FnMut(i64),
) -> Result<ExportSummary> {
    let (sql, predicate) = build_export_query(columns, spec)?;

    let mut workbook = Workbook::new();
    // Flushes each completed row to a temp file instead of buffering the whole sheet in RAM —
    // requires rows to be written in strictly increasing row order, which our
    // `ORDER BY row_num ASC` query already guarantees.
    let worksheet = workbook.add_worksheet_with_constant_memory();

    for (col_idx, col) in columns.iter().enumerate() {
        worksheet.write_string(0, col_idx as u16, col.original_name.as_str())?;
    }

    let mut stmt = conn.prepare(&sql)?;
    let params: Vec<&dyn rusqlite::ToSql> = predicate.params.iter().map(|p| p.as_ref()).collect();
    let mut rows = stmt.query(params.as_slice())?;

    let mut row_count: i64 = 0;
    let mut excel_row: u32 = 1;
    while let Some(row) = rows.next()? {
        for col_idx in 0..columns.len() {
            let value: String = row.get(col_idx)?;
            worksheet.write_string(excel_row, col_idx as u16, value.as_str())?;
        }
        excel_row += 1;
        row_count += 1;
        if row_count % PROGRESS_EVERY == 0 {
            on_progress(row_count);
        }
    }

    workbook.save(dest_path)?;
    on_progress(row_count);

    Ok(ExportSummary { row_count })
}

pub fn export_xlsx_normalized_time(
    conn: &Connection,
    columns: &[ColumnMeta],
    spec: &QuerySpec,
    source_column: &str,
    direction: query::SortDirection,
    dest_path: &Path,
    mut on_progress: impl FnMut(i64),
) -> Result<ExportSummary> {
    let (sql, predicate) =
        build_normalized_time_export_query(conn, columns, spec, source_column, direction)?;
    let mut workbook = Workbook::new();
    let worksheet = workbook.add_worksheet_with_constant_memory();
    for (column_index, column) in columns.iter().enumerate() {
        worksheet.write_string(0, column_index as u16, column.original_name.as_str())?;
    }

    let mut stmt = conn.prepare(&sql)?;
    let params: Vec<&dyn rusqlite::ToSql> = predicate
        .params
        .iter()
        .map(|param| param.as_ref())
        .collect();
    let mut rows = stmt.query(params.as_slice())?;
    let mut row_count = 0i64;
    let mut excel_row = 1u32;
    while let Some(row) = rows.next()? {
        for column_index in 0..columns.len() {
            let value: String = row.get(column_index)?;
            worksheet.write_string(excel_row, column_index as u16, value.as_str())?;
        }
        excel_row += 1;
        row_count += 1;
        if row_count % PROGRESS_EVERY == 0 {
            on_progress(row_count);
        }
    }
    workbook.save(dest_path)?;
    on_progress(row_count);
    Ok(ExportSummary { row_count })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db;
    use calamine::Reader;
    use std::io::Read;

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
        for (i, (account, event_id)) in [("alice", "100"), ("bob", "200"), ("carol", "300")]
            .iter()
            .enumerate()
        {
            conn.execute(
                "INSERT INTO rows (row_num, account, event_id) VALUES (?1, ?2, ?3)",
                rusqlite::params![(i as i64) + 1, account, event_id],
            )
            .unwrap();
        }
        db::populate_fts(&conn, &columns).unwrap();
        (conn, columns)
    }

    fn empty_spec() -> QuerySpec {
        QuerySpec {
            search: None,
            filters: vec![],
            expression: None,
            sort: None,
            cursor: None,
            limit: 200,
        }
    }

    #[test]
    fn csv_export_round_trips() {
        let (conn, columns) = setup();
        let dir = std::env::temp_dir().join(format!("log-parser-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("export.csv");

        let summary = export_csv(&conn, &columns, &empty_spec(), &path, |_| {}).unwrap();
        assert_eq!(summary.row_count, 3);

        let mut contents = String::new();
        File::open(&path)
            .unwrap()
            .read_to_string(&mut contents)
            .unwrap();
        assert!(contents.contains("Account,EventID"));
        assert!(contents.contains("alice,100"));
        assert!(contents.contains("carol,300"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn csv_export_respects_active_sort() {
        let (conn, columns) = setup();
        let dir = std::env::temp_dir().join(format!("log-parser-test-sort-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("export_sorted.csv");

        let mut spec = empty_spec();
        spec.sort = Some(query::SortSpec {
            column: "event_id".to_string(),
            direction: query::SortDirection::Desc,
        });

        export_csv(&conn, &columns, &spec, &path, |_| {}).unwrap();

        let mut contents = String::new();
        File::open(&path)
            .unwrap()
            .read_to_string(&mut contents)
            .unwrap();
        let data_lines: Vec<&str> = contents.lines().skip(1).collect();
        assert_eq!(
            data_lines,
            vec!["carol,300", "bob,200", "alice,100"],
            "export should follow the descending event_id sort, not source row_num order"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn csv_export_respects_recursive_raw_table_expression() {
        let (conn, columns) = setup();
        let dir = std::env::temp_dir().join(format!(
            "log-parser-test-expression-export-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("export_expression.csv");

        let mut spec = empty_spec();
        spec.expression = Some(query::QueryExpression::Or {
            children: vec![
                query::QueryExpression::Predicate {
                    column: "account".to_string(),
                    op: query::FilterOp::Equals,
                    value: "alice".to_string(),
                },
                query::QueryExpression::Predicate {
                    column: "event_id".to_string(),
                    op: query::FilterOp::Equals,
                    value: "300".to_string(),
                },
            ],
        });

        let summary = export_csv(&conn, &columns, &spec, &path, |_| {}).unwrap();
        assert_eq!(summary.row_count, 2);
        let contents = std::fs::read_to_string(&path).unwrap();
        assert!(contents.contains("alice,100"));
        assert!(contents.contains("carol,300"));
        assert!(!contents.contains("bob,200"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn xlsx_export_round_trips() {
        let (conn, columns) = setup();
        let dir = std::env::temp_dir().join(format!("log-parser-test-xlsx-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("export.xlsx");

        let summary = export_xlsx(&conn, &columns, &empty_spec(), &path, |_| {}).unwrap();
        assert_eq!(summary.row_count, 3);

        let mut workbook = calamine::open_workbook_auto(&path).unwrap();
        let sheet_name = workbook.sheet_names()[0].clone();
        let range = workbook.worksheet_range(&sheet_name).unwrap();
        let mut rows = range.rows();
        let header = rows.next().unwrap();
        assert_eq!(header[0].to_string(), "Account");
        let first_data_row = rows.next().unwrap();
        assert_eq!(first_data_row[0].to_string(), "alice");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn guided_csv_and_xlsx_follow_normalized_time_and_omit_ai_annotations() {
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
             (1, '2026-07-17T03:00:00+02:00', 'marker later'),
             (2, '2026-07-17T00:30:00Z', 'marker earlier'),
             (3, 'not-a-time', 'marker invalid'),
             (4, '', 'marker blank')",
            [],
        )
        .unwrap();
        db::populate_fts(&conn, &columns).unwrap();
        time::normalize_timestamp_column_with_options(&mut conn, &columns, None, None).unwrap();
        let mut spec = empty_spec();
        spec.expression = Some(query::QueryExpression::Search {
            value: "marker".into(),
        });
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "log-parser-guided-export-{}-{unique}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let csv_path = dir.join("guided.csv");
        let xlsx_path = dir.join("guided.xlsx");

        let csv = export_csv_normalized_time(
            &conn,
            &columns,
            &spec,
            "event_time",
            query::SortDirection::Asc,
            &csv_path,
            |_| {},
        )
        .unwrap();
        assert_eq!(csv.row_count, 4);
        let csv_text = std::fs::read_to_string(&csv_path).unwrap();
        assert!(!csv_text.contains("__aiMatch"));
        let csv_rows = csv_text.lines().skip(1).collect::<Vec<_>>();
        assert_eq!(
            csv_rows,
            vec![
                "2026-07-17T00:30:00Z,marker earlier",
                "2026-07-17T03:00:00+02:00,marker later",
                "not-a-time,marker invalid",
                ",marker blank"
            ]
        );

        let xlsx = export_xlsx_normalized_time(
            &conn,
            &columns,
            &spec,
            "event_time",
            query::SortDirection::Asc,
            &xlsx_path,
            |_| {},
        )
        .unwrap();
        assert_eq!(xlsx.row_count, 4);
        let mut workbook = calamine::open_workbook_auto(&xlsx_path).unwrap();
        let sheet_name = workbook.sheet_names()[0].clone();
        let range = workbook.worksheet_range(&sheet_name).unwrap();
        let rows = range.rows().collect::<Vec<_>>();
        assert!(rows[0].iter().all(|cell| cell.to_string() != "__aiMatch"));
        assert_eq!(rows[1][1].to_string(), "marker earlier");
        assert_eq!(rows[2][1].to_string(), "marker later");
        assert_eq!(rows[3][1].to_string(), "marker invalid");
        assert_eq!(rows[4][1].to_string(), "marker blank");

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn normalized_export_rejects_stale_binding() {
        let mut conn = Connection::open_in_memory().unwrap();
        let columns = vec![ColumnMeta {
            sql_name: "event_time".into(),
            original_name: "Event Time".into(),
            col_index: 0,
            inferred_type: "timestamp".into(),
        }];
        db::create_schema(&conn, &columns).unwrap();
        conn.execute(
            "INSERT INTO rows (row_num, event_time) VALUES (1, '2026-07-17T00:00:00Z')",
            [],
        )
        .unwrap();
        time::normalize_timestamp_column_with_options(&mut conn, &columns, None, None).unwrap();
        conn.execute(
            "INSERT INTO rows (row_num, event_time) VALUES (2, '2026-07-17T01:00:00Z')",
            [],
        )
        .unwrap();

        let path = std::env::temp_dir().join(format!(
            "log-parser-stale-export-{}.csv",
            std::process::id()
        ));
        let error = export_csv_normalized_time(
            &conn,
            &columns,
            &empty_spec(),
            "event_time",
            query::SortDirection::Asc,
            &path,
            |_| {},
        )
        .expect_err("changed imports must invalidate normalized export ordering");
        assert!(error.to_string().contains("stale"));
        let _ = std::fs::remove_file(path);
    }
}
