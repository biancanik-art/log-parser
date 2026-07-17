use crate::db::{self, ColumnMeta};
pub use crate::header_utils::sanitize_headers;
use anyhow::{Context, Result};
use calamine::{open_workbook_auto, Data, Reader};
use std::path::Path;

pub fn list_sheet_names(path: &Path) -> Result<Vec<String>> {
    let workbook =
        open_workbook_auto(path).with_context(|| format!("opening workbook {}", path.display()))?;
    Ok(workbook.sheet_names().to_owned())
}

pub struct ImportResult {
    pub columns: Vec<ColumnMeta>,
    pub row_count: i64,
}

const BATCH_SIZE: u64 = 5000;

/// Reads `sheet_name` out of `source_path` with calamine and bulk-loads it into a fresh SQLite
/// database at `db_path` (schema + rows + FTS5 index). `on_progress(rows_done, rows_total)` is
/// called after each committed batch so the caller can relay progress to the UI.
///
/// Note: calamine materializes the whole sheet as an in-memory `Range` — there is no lazy/row-
/// streaming reader in this crate version. Peak RAM during import is roughly one full copy of
/// the sheet; acceptable for 100k-row-class files but not unbounded.
pub fn import_into_db(
    source_path: &Path,
    sheet_name: &str,
    db_path: &Path,
    mut on_progress: impl FnMut(u64, u64),
) -> Result<ImportResult> {
    let mut workbook = open_workbook_auto(source_path)
        .with_context(|| format!("opening workbook {}", source_path.display()))?;
    let range = workbook
        .worksheet_range(sheet_name)
        .with_context(|| format!("reading sheet '{sheet_name}'"))?;

    let (height, _width) = range.get_size();
    let total_rows = (height as u64).saturating_sub(1);

    let mut rows_iter = range.rows();
    let header_row = rows_iter.next().context("sheet has no header row")?;
    let raw_headers: Vec<String> = header_row.iter().map(cell_to_string).collect();
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

    let mut rows_iter = rows_iter.peekable();
    let mut row_count: i64 = 0;

    while rows_iter.peek().is_some() {
        let tx = conn.transaction()?;
        {
            let mut stmt = tx.prepare(&insert_sql)?;
            let mut in_batch = 0u64;
            while in_batch < BATCH_SIZE {
                let Some(row) = rows_iter.next() else {
                    break;
                };
                row_count += 1;
                let row_num = row_count;

                let mut params: Vec<Box<dyn rusqlite::ToSql>> =
                    Vec::with_capacity(columns.len() + 1);
                params.push(Box::new(row_num));
                for col_idx in 0..columns.len() {
                    let value = row.get(col_idx).map(cell_to_string).unwrap_or_default();
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

fn cell_to_string(cell: &Data) -> String {
    match cell {
        Data::Empty => String::new(),
        Data::String(s) => s.clone(),
        Data::Int(i) => i.to_string(),
        Data::Float(f) => {
            if f.fract() == 0.0 && f.abs() < 1e15 {
                format!("{}", *f as i64)
            } else {
                f.to_string()
            }
        }
        Data::Bool(b) => b.to_string(),
        Data::DateTime(dt) => dt
            .as_datetime()
            .map(|d| d.format("%Y-%m-%dT%H:%M:%S%.f").to_string())
            .unwrap_or_default(),
        Data::DateTimeIso(s) => s.clone(),
        Data::DurationIso(s) => s.clone(),
        Data::Error(e) => format!("#ERROR:{:?}", e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell_to_string_variants() {
        assert_eq!(cell_to_string(&Data::Empty), "");
        assert_eq!(cell_to_string(&Data::String("hi".into())), "hi");
        assert_eq!(cell_to_string(&Data::Int(42)), "42");
        assert_eq!(cell_to_string(&Data::Float(3.0)), "3");
        assert_eq!(cell_to_string(&Data::Float(3.5)), "3.5");
        assert_eq!(cell_to_string(&Data::Bool(true)), "true");
    }
}
