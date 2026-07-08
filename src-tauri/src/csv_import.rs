use crate::db::{self, ColumnMeta};
use crate::header_utils::sanitize_headers;
use anyhow::{bail, Context, Result};
use std::fs::File;
use std::path::Path;

pub struct ImportResult {
    pub columns: Vec<ColumnMeta>,
    pub row_count: i64,
}

const BATCH_SIZE: u64 = 5000;

fn csv_reader(path: &Path) -> Result<csv::Reader<File>> {
    csv::ReaderBuilder::new()
        .flexible(true)
        .from_path(path)
        .with_context(|| format!("opening CSV {}", path.display()))
}

fn count_csv_rows(source_path: &Path) -> Result<u64> {
    let mut reader = csv_reader(source_path)?;
    let headers = reader
        .headers()
        .with_context(|| format!("reading CSV header from {}", source_path.display()))?;
    if headers.is_empty() {
        bail!("CSV has no header row");
    }

    let mut count = 0u64;
    for record in reader.records() {
        record.with_context(|| format!("reading CSV row {}", count + 2))?;
        count += 1;
    }
    Ok(count)
}

/// Reads a CSV file and bulk-loads it into a fresh SQLite database at `db_path` using the same
/// cache schema as the Excel importer: all source columns are TEXT, `row_num` is synthetic, and
/// FTS5 is populated after row insertion.
pub fn import_into_db(
    source_path: &Path,
    db_path: &Path,
    mut on_progress: impl FnMut(u64, u64),
) -> Result<ImportResult> {
    let total_rows = count_csv_rows(source_path)?;
    let mut reader = csv_reader(source_path)?;

    let raw_headers: Vec<String> = reader
        .headers()
        .with_context(|| format!("reading CSV header from {}", source_path.display()))?
        .iter()
        .map(|s| s.to_string())
        .collect();
    if raw_headers.is_empty() {
        bail!("CSV has no header row");
    }
    let columns = sanitize_headers(&raw_headers);

    let mut conn = db::open(db_path)?;
    db::set_import_pragmas(&conn)?;
    db::create_schema(&conn, &columns)?;

    let col_idents: Vec<String> = columns
        .iter()
        .map(|c| db::quote_ident(&c.sql_name))
        .collect();
    let placeholders: Vec<String> = (1..=columns.len() + 1).map(|i| format!("?{i}")).collect();
    let insert_sql = format!(
        "INSERT INTO rows (row_num, {}) VALUES ({})",
        col_idents.join(", "),
        placeholders.join(", ")
    );

    let mut records = reader.records().peekable();
    let mut row_count: i64 = 0;

    while records.peek().is_some() {
        let tx = conn.transaction()?;
        {
            let mut stmt = tx.prepare(&insert_sql)?;
            let mut in_batch = 0u64;
            while in_batch < BATCH_SIZE {
                let Some(record) = records.next() else {
                    break;
                };
                let record =
                    record.with_context(|| format!("reading CSV row {}", row_count + 2))?;
                row_count += 1;
                let row_num = row_count;

                let mut params: Vec<Box<dyn rusqlite::ToSql>> =
                    Vec::with_capacity(columns.len() + 1);
                params.push(Box::new(row_num));
                for col_idx in 0..columns.len() {
                    let value = record.get(col_idx).unwrap_or_default().to_string();
                    params.push(Box::new(value));
                }
                let param_refs: Vec<&dyn rusqlite::ToSql> =
                    params.iter().map(|p| p.as_ref()).collect();
                stmt.execute(param_refs.as_slice())?;
                in_batch += 1;
            }
        }
        tx.commit()?;
        on_progress(row_count as u64, total_rows);
    }

    db::populate_fts(&conn, &columns)?;
    db::restore_normal_pragmas(&conn)?;

    Ok(ImportResult { columns, row_count })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!(
            "log-parser-csv-import-{name}-{}-{id}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn csv_basic_import_populates_rows_and_fts() {
        let dir = temp_dir("basic");
        let csv_path = dir.join("basic.csv");
        let db_path = dir.join("basic.sqlite3");
        std::fs::write(
            &csv_path,
            "TimeGenerated,Account\n2026-01-01T00:00:00Z,alice\n2026-01-02T00:00:00Z,bob\n",
        )
        .unwrap();

        let mut progress = Vec::new();
        let result = import_into_db(&csv_path, &db_path, |done, total| {
            progress.push((done, total));
        })
        .unwrap();

        assert_eq!(result.row_count, 2);
        assert_eq!(result.columns[0].sql_name, "timegenerated");
        assert_eq!(result.columns[1].sql_name, "account");
        assert_eq!(progress, vec![(2, 2)]);

        let conn = db::open(&db_path).unwrap();
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM rows", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 2);

        let first_account: String = conn
            .query_row("SELECT account FROM rows WHERE row_num = 1", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(first_account, "alice");

        let fts_hit: i64 = conn
            .query_row(
                "SELECT rowid FROM rows_fts WHERE rows_fts MATCH ?1",
                rusqlite::params!["alice"],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(fts_hit, 1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn csv_import_sanitizes_duplicate_headers_and_stores_values_as_text() {
        let dir = temp_dir("headers");
        let csv_path = dir.join("headers.csv");
        let db_path = dir.join("headers.sqlite3");
        std::fs::write(&csv_path, "Account,Account,EventID\nalice,duplicate,42\n").unwrap();

        let result = import_into_db(&csv_path, &db_path, |_, _| {}).unwrap();
        let names: Vec<&str> = result.columns.iter().map(|c| c.sql_name.as_str()).collect();
        assert_eq!(names, vec!["account", "account_2", "eventid"]);

        let conn = db::open(&db_path).unwrap();
        let duplicate: String = conn
            .query_row("SELECT account_2 FROM rows WHERE row_num = 1", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(duplicate, "duplicate");

        let (event_id, sqlite_type): (String, String) = conn
            .query_row(
                "SELECT eventid, typeof(eventid) FROM rows WHERE row_num = 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(event_id, "42");
        assert_eq!(sqlite_type, "text");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
