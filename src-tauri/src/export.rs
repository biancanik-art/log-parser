use crate::db::ColumnMeta;
use crate::intel::time;
use crate::query::{self, QuerySpec};
use anyhow::{Context, Result};
use rusqlite::Connection;
use rust_xlsxwriter::Workbook;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug)]
pub struct ExportSummary {
    pub row_count: i64,
}

const PROGRESS_EVERY: i64 = 5000;
static EXPORT_TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(1);

struct PendingExport {
    path: PathBuf,
    published: bool,
}

impl Drop for PendingExport {
    fn drop(&mut self) {
        if !self.published {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// Writes beside the destination, syncs the complete file, then replaces the destination in one
/// filesystem operation. A query, encoder, disk, or process failure therefore cannot truncate an
/// examiner's existing export.
fn atomic_export(
    dest_path: &Path,
    write: impl FnOnce(&Path) -> Result<ExportSummary>,
) -> Result<ExportSummary> {
    atomic_export_guarded(dest_path, write, publish_completed_export)
}

/// Variant used by dataset-bound exports. The publisher runs only after the complete temporary
/// file has been flushed to stable storage. A caller can hold its dataset-generation lock while
/// invoking `publish_completed_export`, making validation and replacement one guarded operation.
/// If validation fails, the new file is discarded and any existing examiner export is untouched.
fn atomic_export_guarded(
    dest_path: &Path,
    write: impl FnOnce(&Path) -> Result<ExportSummary>,
    publish: impl FnOnce(&Path, &Path) -> Result<()>,
) -> Result<ExportSummary> {
    let parent = dest_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let file_name = dest_path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("log-parser-export");
    let extension = dest_path
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| format!(".{value}"))
        .unwrap_or_default();
    let sequence = EXPORT_TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let temp_path = parent.join(format!(
        ".{file_name}.log-parser-{}-{sequence}.tmp{extension}",
        std::process::id()
    ));

    // Reserve the unique name without following an existing link/file, then let the format
    // writer reopen it. PID + process-local sequence keeps this collision-free in normal use.
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp_path)
        .with_context(|| format!("creating temporary export {}", temp_path.display()))?;
    let mut pending = PendingExport {
        path: temp_path,
        published: false,
    };
    let summary = write(&pending.path)?;
    OpenOptions::new()
        .read(true)
        .write(true)
        .open(&pending.path)?
        .sync_all()?;
    publish(&pending.path, dest_path)?;
    pending.published = true;
    sync_parent_directory(parent)?;
    Ok(summary)
}

pub(crate) fn publish_completed_export(source: &Path, destination: &Path) -> Result<()> {
    atomic_replace(source, destination)
}

#[cfg(windows)]
fn atomic_replace(source: &Path, destination: &Path) -> Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::{ReplaceFileW, REPLACEFILE_WRITE_THROUGH};

    if !destination.exists() {
        return std::fs::rename(source, destination).context("publishing completed export");
    }

    let source = source
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    let destination = destination
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    let succeeded = unsafe {
        ReplaceFileW(
            destination.as_ptr(),
            source.as_ptr(),
            std::ptr::null(),
            REPLACEFILE_WRITE_THROUGH,
            std::ptr::null(),
            std::ptr::null(),
        )
    };
    if succeeded == 0 {
        return Err(std::io::Error::last_os_error()).context("publishing completed export");
    }
    Ok(())
}

#[cfg(not(windows))]
fn atomic_replace(source: &Path, destination: &Path) -> Result<()> {
    std::fs::rename(source, destination).context("publishing completed export")
}

#[cfg(unix)]
fn sync_parent_directory(parent: &Path) -> Result<()> {
    File::open(parent)?.sync_all()?;
    Ok(())
}

#[cfg(not(unix))]
fn sync_parent_directory(_parent: &Path) -> Result<()> {
    Ok(())
}

/// Streams matching rows straight from a `rusqlite` row cursor into the destination file — no
/// intermediate `Vec`/JSON blob of the whole result set is ever materialized. Reuses
/// `query::build_predicate` and `query::build_order_by` so the exported set — filters, search,
/// *and* the active sort — always matches what's on screen, not just the row set.
fn build_export_query(
    conn: &Connection,
    columns: &[ColumnMeta],
    spec: &QuerySpec,
) -> Result<(String, query::Predicate)> {
    let predicate = query::build_predicate_for_connection(conn, columns, spec)?;
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
    let predicate = query::build_predicate_for_connection(conn, columns, spec)?;
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
    on_progress: impl FnMut(i64),
) -> Result<ExportSummary> {
    export_csv_guarded(
        conn,
        columns,
        spec,
        dest_path,
        on_progress,
        publish_completed_export,
    )
}

pub fn export_csv_guarded(
    conn: &Connection,
    columns: &[ColumnMeta],
    spec: &QuerySpec,
    dest_path: &Path,
    mut on_progress: impl FnMut(i64),
    publish: impl FnOnce(&Path, &Path) -> Result<()>,
) -> Result<ExportSummary> {
    let (sql, predicate) = build_export_query(conn, columns, spec)?;
    atomic_export_guarded(
        dest_path,
        |dest_path| {
            let file = File::create(dest_path)?;
            let mut writer = csv::Writer::from_writer(BufWriter::new(file));
            let headers: Vec<&str> = columns.iter().map(|c| c.original_name.as_str()).collect();
            writer.write_record(&headers)?;

            let mut stmt = conn.prepare(&sql)?;
            let params: Vec<&dyn rusqlite::ToSql> =
                predicate.params.iter().map(|p| p.as_ref()).collect();
            let mut rows = stmt.query(params.as_slice())?;

            let mut row_count: i64 = 0;
            let mut record: Vec<String> = vec![String::new(); columns.len()];
            while let Some(row) = rows.next()? {
                for (i, cell) in record.iter_mut().enumerate() {
                    let val: Option<String> = row.get(i)?;
                    *cell = val.unwrap_or_default();
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
        },
        publish,
    )
}

pub fn export_csv_normalized_time(
    conn: &Connection,
    columns: &[ColumnMeta],
    spec: &QuerySpec,
    source_column: &str,
    direction: query::SortDirection,
    dest_path: &Path,
    on_progress: impl FnMut(i64),
) -> Result<ExportSummary> {
    export_csv_normalized_time_guarded(
        conn,
        columns,
        spec,
        source_column,
        direction,
        dest_path,
        on_progress,
        publish_completed_export,
    )
}

pub fn export_csv_normalized_time_guarded(
    conn: &Connection,
    columns: &[ColumnMeta],
    spec: &QuerySpec,
    source_column: &str,
    direction: query::SortDirection,
    dest_path: &Path,
    mut on_progress: impl FnMut(i64),
    publish: impl FnOnce(&Path, &Path) -> Result<()>,
) -> Result<ExportSummary> {
    let (sql, predicate) =
        build_normalized_time_export_query(conn, columns, spec, source_column, direction)?;
    atomic_export_guarded(
        dest_path,
        |dest_path| {
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
                    let val: Option<String> = row.get(index)?;
                    *cell = val.unwrap_or_default();
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
        },
        publish,
    )
}

pub fn export_xlsx(
    conn: &Connection,
    columns: &[ColumnMeta],
    spec: &QuerySpec,
    dest_path: &Path,
    on_progress: impl FnMut(i64),
) -> Result<ExportSummary> {
    export_xlsx_guarded(
        conn,
        columns,
        spec,
        dest_path,
        on_progress,
        publish_completed_export,
    )
}

pub fn export_xlsx_guarded(
    conn: &Connection,
    columns: &[ColumnMeta],
    spec: &QuerySpec,
    dest_path: &Path,
    mut on_progress: impl FnMut(i64),
    publish: impl FnOnce(&Path, &Path) -> Result<()>,
) -> Result<ExportSummary> {
    let (sql, predicate) = build_export_query(conn, columns, spec)?;
    atomic_export_guarded(
        dest_path,
        |dest_path| {
            let mut workbook = Workbook::new();
            // Flushes each completed row to a temp file instead of buffering the whole sheet in RAM —
            // requires rows to be written in strictly increasing row order, which our
            // `ORDER BY row_num ASC` query already guarantees.
            let worksheet = workbook.add_worksheet_with_constant_memory();

            for (col_idx, col) in columns.iter().enumerate() {
                worksheet.write_string(0, col_idx as u16, col.original_name.as_str())?;
            }

            let mut stmt = conn.prepare(&sql)?;
            let params: Vec<&dyn rusqlite::ToSql> =
                predicate.params.iter().map(|p| p.as_ref()).collect();
            let mut rows = stmt.query(params.as_slice())?;

            let mut row_count: i64 = 0;
            let mut excel_row: u32 = 1;
            while let Some(row) = rows.next()? {
                for col_idx in 0..columns.len() {
                    let value: Option<String> = row.get(col_idx)?;
                    worksheet.write_string(excel_row, col_idx as u16, value.as_deref().unwrap_or(""))?;
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
        },
        publish,
    )
}

pub fn export_xlsx_normalized_time(
    conn: &Connection,
    columns: &[ColumnMeta],
    spec: &QuerySpec,
    source_column: &str,
    direction: query::SortDirection,
    dest_path: &Path,
    on_progress: impl FnMut(i64),
) -> Result<ExportSummary> {
    export_xlsx_normalized_time_guarded(
        conn,
        columns,
        spec,
        source_column,
        direction,
        dest_path,
        on_progress,
        publish_completed_export,
    )
}

pub fn export_xlsx_normalized_time_guarded(
    conn: &Connection,
    columns: &[ColumnMeta],
    spec: &QuerySpec,
    source_column: &str,
    direction: query::SortDirection,
    dest_path: &Path,
    mut on_progress: impl FnMut(i64),
    publish: impl FnOnce(&Path, &Path) -> Result<()>,
) -> Result<ExportSummary> {
    let (sql, predicate) =
        build_normalized_time_export_query(conn, columns, spec, source_column, direction)?;
    atomic_export_guarded(
        dest_path,
        |dest_path| {
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
                    let value: Option<String> = row.get(column_index)?;
                    worksheet.write_string(excel_row, column_index as u16, value.as_deref().unwrap_or(""))?;
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
        },
        publish,
    )
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UnifiedExportSummary {
    pub sheets_written: Vec<String>,
    pub total_events: usize,
    pub dest_path: String,
}

pub fn export_unified_multisheet_xlsx(
    targets: &[crate::commands::FileTarget],
    events: &[crate::intel::analyst::CorrelatedTimelineEvent],
    dest_path: &Path,
) -> Result<UnifiedExportSummary> {
    if events.is_empty() {
        anyhow::bail!("no events to export");
    }

    let mut sheets_written = Vec::new();

    atomic_export_guarded(
        dest_path,
        |temp_path| {
            let mut workbook = Workbook::new();
            let mut used_sheet_names = HashSet::new();

            // 1. Write "Unified Timeline" sheet
            let timeline_sheet_name = "Unified Timeline".to_string();
            used_sheet_names.insert(timeline_sheet_name.to_ascii_lowercase());
            sheets_written.push(timeline_sheet_name.clone());

            let timeline_sheet = workbook.add_worksheet();
            timeline_sheet.set_name(&timeline_sheet_name)?;

            let timeline_headers = [
                "#",
                "Source File",
                "File Path",
                "Source Row",
                "Timestamp (UTC)",
                "Epoch (ms)",
                "User / Identity",
                "Host / IP",
                "Operation / Action",
                "MITRE / Threat Tags",
            ];

            for (col_idx, header) in timeline_headers.iter().enumerate() {
                timeline_sheet.write_string(0, col_idx as u16, *header)?;
            }

            for (row_idx, ev) in events.iter().enumerate() {
                let excel_row = (row_idx + 1) as u32;
                timeline_sheet.write_number(excel_row, 0, (row_idx + 1) as f64)?;
                timeline_sheet.write_string(
                    excel_row,
                    1,
                    crate::report::excel_safe_string(&ev.file_name).as_ref(),
                )?;
                timeline_sheet.write_string(
                    excel_row,
                    2,
                    crate::report::excel_safe_string(&ev.path).as_ref(),
                )?;
                timeline_sheet.write_number(excel_row, 3, ev.row_num as f64)?;
                timeline_sheet.write_string(
                    excel_row,
                    4,
                    crate::report::excel_safe_string(ev.utc_text.as_deref().unwrap_or("")).as_ref(),
                )?;
                if let Some(epoch) = ev.epoch_ms {
                    timeline_sheet.write_number(excel_row, 5, epoch as f64)?;
                }
                timeline_sheet.write_string(
                    excel_row,
                    6,
                    crate::report::excel_safe_string(ev.user.as_deref().unwrap_or("")).as_ref(),
                )?;
                timeline_sheet.write_string(
                    excel_row,
                    7,
                    crate::report::excel_safe_string(ev.host.as_deref().unwrap_or("")).as_ref(),
                )?;
                timeline_sheet.write_string(
                    excel_row,
                    8,
                    crate::report::excel_safe_string(ev.action.as_deref().unwrap_or("")).as_ref(),
                )?;
                let tags_str = ev.mitre_tags.join("; ");
                timeline_sheet.write_string(
                    excel_row,
                    9,
                    crate::report::excel_safe_string(&tags_str).as_ref(),
                )?;
            }

            // Set clean column widths for Unified Timeline
            timeline_sheet.set_column_width(0, 6)?;
            timeline_sheet.set_column_width(1, 22)?;
            timeline_sheet.set_column_width(2, 32)?;
            timeline_sheet.set_column_width(3, 12)?;
            timeline_sheet.set_column_width(4, 22)?;
            timeline_sheet.set_column_width(5, 16)?;
            timeline_sheet.set_column_width(6, 25)?;
            timeline_sheet.set_column_width(7, 20)?;
            timeline_sheet.set_column_width(8, 35)?;
            timeline_sheet.set_column_width(9, 35)?;

            // 2. Group events by participating file
            let mut participating_files: Vec<(String, String)> = Vec::new();
            for ev in events {
                if !participating_files.iter().any(|(p, _)| *p == ev.path) {
                    participating_files.push((ev.path.clone(), ev.file_name.clone()));
                }
            }

            for (file_path, file_name) in &participating_files {
                let file_events: Vec<&crate::intel::analyst::CorrelatedTimelineEvent> = events
                    .iter()
                    .filter(|e| e.path == *file_path)
                    .collect();

                if file_events.is_empty() {
                    continue;
                }

                let target = targets.iter().find(|t| t.path == *file_path);
                let sheet_name = target.and_then(|t| t.sheet.clone()).unwrap_or_default();

                let db_path = if let Some(t) = target {
                    if let Some(ref p) = t.cache_db_path {
                        let candidate = PathBuf::from(p);
                        if candidate.exists() {
                            candidate
                        } else {
                            crate::db::cache_db_path(Path::new(file_path), &sheet_name)
                                .unwrap_or_else(|_| PathBuf::from(p))
                        }
                    } else {
                        match crate::db::cache_db_path(Path::new(file_path), &sheet_name) {
                            Ok(p) => p,
                            Err(_) => continue,
                        }
                    }
                } else {
                    match crate::db::cache_db_path(Path::new(file_path), &sheet_name) {
                        Ok(p) => p,
                        Err(_) => continue,
                    }
                };

                if !db_path.exists() {
                    continue;
                }

                let conn = match crate::db::open(&db_path) {
                    Ok(c) => c,
                    Err(_) => continue,
                };

                let columns = match crate::db::load_columns(&conn) {
                    Ok(cols) => cols,
                    Err(_) => continue,
                };

                if columns.is_empty() {
                    continue;
                }

                let raw_title = if !sheet_name.is_empty() {
                    let base_file = Path::new(file_name)
                        .file_stem()
                        .map(|s| s.to_string_lossy().to_string())
                        .unwrap_or_else(|| file_name.clone());
                    format!("{base_file} - {sheet_name}")
                } else {
                    Path::new(file_name)
                        .file_stem()
                        .map(|s| s.to_string_lossy().to_string())
                        .unwrap_or_else(|| file_name.clone())
                };

                let unique_name =
                    crate::report::unique_sheet_name(&raw_title, &mut used_sheet_names);
                sheets_written.push(unique_name.clone());

                let worksheet = workbook.add_worksheet();
                worksheet.set_name(&unique_name)?;

                worksheet.write_string(0, 0, "Row #")?;
                for (col_idx, col) in columns.iter().enumerate() {
                    worksheet.write_string(
                        0,
                        (col_idx + 1) as u16,
                        crate::report::excel_safe_string(&col.original_name).as_ref(),
                    )?;
                }
                let mitre_col_idx = (columns.len() + 1) as u16;
                worksheet.write_string(0, mitre_col_idx, "MITRE / Threat Tags")?;

                let mut tags_by_row: std::collections::HashMap<i64, Vec<String>> =
                    std::collections::HashMap::new();
                for ev in &file_events {
                    if !ev.mitre_tags.is_empty() {
                        tags_by_row
                            .entry(ev.row_num)
                            .or_default()
                            .extend(ev.mitre_tags.clone());
                    }
                }

                let mut ordered_row_nums = Vec::new();
                let mut seen_rows = HashSet::new();
                for ev in &file_events {
                    if seen_rows.insert(ev.row_num) {
                        ordered_row_nums.push(ev.row_num);
                    }
                }

                let col_names = columns
                    .iter()
                    .map(|c| format!("\"{}\"", c.sql_name))
                    .collect::<Vec<_>>()
                    .join(", ");

                let mut excel_row = 1u32;

                for chunk in ordered_row_nums.chunks(400) {
                    let placeholders = vec!["?"; chunk.len()].join(", ");
                    let sql = format!(
                        "SELECT row_num, {col_names} FROM rows WHERE row_num IN ({placeholders})"
                    );

                    let mut stmt = match conn.prepare(&sql) {
                        Ok(s) => s,
                        Err(_) => continue,
                    };

                    let params: Vec<&dyn rusqlite::ToSql> =
                        chunk.iter().map(|r| r as &dyn rusqlite::ToSql).collect();

                    let mut rows = match stmt.query(params.as_slice()) {
                        Ok(r) => r,
                        Err(_) => continue,
                    };

                    let mut chunk_data: std::collections::HashMap<i64, Vec<String>> =
                        std::collections::HashMap::new();
                    while let Ok(Some(row)) = rows.next() {
                        let r_num: i64 = match row.get(0) {
                            Ok(n) => n,
                            Err(_) => continue,
                        };
                        let mut vals = Vec::with_capacity(columns.len());
                        for idx in 0..columns.len() {
                            let val: Option<String> = row.get(idx + 1).unwrap_or(None);
                            vals.push(val.unwrap_or_default());
                        }
                        chunk_data.insert(r_num, vals);
                    }

                    for r_num in chunk {
                        if let Some(vals) = chunk_data.get(r_num) {
                            worksheet.write_number(excel_row, 0, *r_num as f64)?;
                            for (col_idx, val) in vals.iter().enumerate() {
                                worksheet.write_string(
                                    excel_row,
                                    (col_idx + 1) as u16,
                                    crate::report::excel_safe_string(val).as_ref(),
                                )?;
                            }
                            if let Some(tags) = tags_by_row.get(r_num) {
                                let mut unique_tags = tags.clone();
                                unique_tags.sort();
                                unique_tags.dedup();
                                worksheet.write_string(
                                    excel_row,
                                    mitre_col_idx,
                                    crate::report::excel_safe_string(&unique_tags.join("; "))
                                        .as_ref(),
                                )?;
                            }
                            excel_row += 1;
                        }
                    }
                }

                worksheet.set_column_width(0, 10)?;
                for (col_idx, col) in columns.iter().enumerate() {
                    let col_len = col.original_name.chars().count().clamp(12, 45);
                    worksheet.set_column_width((col_idx + 1) as u16, col_len as u16)?;
                }
                worksheet.set_column_width(mitre_col_idx, 30)?;
            }

            workbook.save(temp_path)?;
            Ok(ExportSummary {
                row_count: events.len() as i64,
            })
        },
        publish_completed_export,
    )?;

    Ok(UnifiedExportSummary {
        sheets_written,
        total_events: events.len(),
        dest_path: dest_path.display().to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db;
    use crate::semantic::{self, SemanticEmbedder, SemanticSearchPolicy};
    use calamine::Reader;
    use std::io::Read;

    struct ConstantEmbedder;

    impl SemanticEmbedder for ConstantEmbedder {
        fn embed_batch(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
            Ok(texts
                .iter()
                .map(|_| {
                    let mut vector = vec![0.0; 384];
                    vector[0] = 1.0;
                    vector
                })
                .collect())
        }
    }

    fn semantic_export_fixture(count: usize) -> (Connection, Vec<ColumnMeta>, QuerySpec) {
        let mut conn = Connection::open_in_memory().unwrap();
        let columns = vec![
            ColumnMeta {
                sql_name: "event_id".into(),
                original_name: "Event ID".into(),
                col_index: 0,
                inferred_type: "identifier".into(),
            },
            ColumnMeta {
                sql_name: "message".into(),
                original_name: "Message".into(),
                col_index: 1,
                inferred_type: "text".into(),
            },
        ];
        db::create_schema(&conn, &columns).unwrap();
        let tx = conn.transaction().unwrap();
        {
            let mut insert = tx
                .prepare("INSERT INTO rows(row_num, event_id, message) VALUES (?1, ?2, ?3)")
                .unwrap();
            for index in 0..count {
                insert
                    .execute(rusqlite::params![
                        index as i64 + 1,
                        format!("event-{index}"),
                        "credential dumping process observed"
                    ])
                    .unwrap();
            }
        }
        tx.commit().unwrap();
        let embedder = ConstantEmbedder;
        semantic::ensure_semantic_index_v2(&mut conn, &columns, &embedder, || false, |_| {})
            .unwrap();
        let selection = semantic::create_semantic_selection(
            &mut conn,
            &columns,
            &embedder,
            "credential theft activity",
            SemanticSearchPolicy {
                maximum_documents: 1,
                minimum_score: -1.0,
            },
        )
        .unwrap();
        let spec = QuerySpec {
            expression: Some(query::QueryExpression::SemanticSelection {
                selection_id: selection.selection_id,
            }),
            ..QuerySpec::default()
        };
        (conn, columns, spec)
    }

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
    fn atomic_export_failure_preserves_existing_destination_and_cleans_temp() {
        let dir = std::env::temp_dir().join(format!(
            "log-parser-test-atomic-failure-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("evidence.csv");
        std::fs::write(&path, b"trusted old export").unwrap();

        let error = atomic_export(&path, |temporary| {
            std::fs::write(temporary, b"partial replacement")?;
            Err(anyhow::anyhow!("injected encoder failure"))
        })
        .unwrap_err();
        assert!(error.to_string().contains("injected encoder failure"));
        assert_eq!(std::fs::read(&path).unwrap(), b"trusted old export");
        let leftovers = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(std::result::Result::ok)
            .filter(|entry| entry.file_name().to_string_lossy().contains(".tmp"))
            .collect::<Vec<_>>();
        assert!(leftovers.is_empty(), "leftovers={leftovers:?}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn atomic_export_replaces_existing_destination_after_complete_sync() {
        let dir = std::env::temp_dir().join(format!(
            "log-parser-test-atomic-success-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("evidence.csv");
        std::fs::write(&path, b"old export").unwrap();

        let summary = atomic_export(&path, |temporary| {
            std::fs::write(temporary, b"complete new export")?;
            Ok(ExportSummary { row_count: 7 })
        })
        .unwrap();
        assert_eq!(summary.row_count, 7);
        assert_eq!(std::fs::read(&path).unwrap(), b"complete new export");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn guarded_export_checks_dataset_before_publication() {
        let (conn, columns) = setup();
        let dir = std::env::temp_dir().join(format!(
            "log-parser-test-generation-guard-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("evidence.csv");
        std::fs::write(&path, b"export from still-loaded evidence").unwrap();

        let error = export_csv_guarded(
            &conn,
            &columns,
            &empty_spec(),
            &path,
            |_| {},
            |_, _| anyhow::bail!("the loaded dataset changed"),
        )
        .unwrap_err();
        assert!(error.to_string().contains("loaded dataset changed"));
        assert_eq!(
            std::fs::read(&path).unwrap(),
            b"export from still-loaded evidence"
        );
        let leftovers = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(std::result::Result::ok)
            .filter(|entry| entry.file_name().to_string_lossy().contains(".tmp"))
            .collect::<Vec<_>>();
        assert!(leftovers.is_empty(), "leftovers={leftovers:?}");

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
    fn semantic_selection_has_csv_xlsx_and_query_count_parity_above_legacy_caps() {
        let (conn, columns, spec) = semantic_export_fixture(1_601);
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "log-parser-semantic-export-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let csv_path = dir.join("selection.csv");
        let xlsx_path = dir.join("selection.xlsx");

        let expected = query::count_rows(&conn, &columns, &spec).unwrap();
        let csv = export_csv(&conn, &columns, &spec, &csv_path, |_| {}).unwrap();
        let xlsx = export_xlsx(&conn, &columns, &spec, &xlsx_path, |_| {}).unwrap();
        assert_eq!(expected, 1_601);
        assert_eq!(csv.row_count, expected);
        assert_eq!(xlsx.row_count, expected);
        assert_eq!(
            std::fs::read_to_string(&csv_path).unwrap().lines().count(),
            1_602
        );

        let mut workbook = calamine::open_workbook_auto(&xlsx_path).unwrap();
        let sheet_name = workbook.sheet_names()[0].clone();
        let range = workbook.worksheet_range(&sheet_name).unwrap();
        assert_eq!(range.height(), 1_602);

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

    #[test]
    fn test_export_unified_multisheet_xlsx() {
        use crate::commands::FileTarget;
        use crate::intel::analyst::CorrelatedTimelineEvent;

        let temp_dir = std::env::temp_dir().join(format!("lp_unified_test_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&temp_dir);

        let db1_path = temp_dir.join("db1.sqlite");
        let conn1 = Connection::open(&db1_path).unwrap();
        let cols1 = vec![
            ColumnMeta {
                sql_name: "col_0".into(),
                original_name: "UserPrincipalName".into(),
                col_index: 0,
                inferred_type: "text".into(),
            },
            ColumnMeta {
                sql_name: "col_1".into(),
                original_name: "ActionName".into(),
                col_index: 1,
                inferred_type: "text".into(),
            },
        ];
        db::create_schema(&conn1, &cols1).unwrap();
        conn1.execute("INSERT INTO rows (row_num, col_0, col_1) VALUES (1, 'user1@corp.local', 'UserLoggedIn')", []).unwrap();
        conn1.execute("INSERT INTO rows (row_num, col_0, col_1) VALUES (2, 'user2@corp.local', 'UserLoggedOut')", []).unwrap();
        drop(conn1);

        let db2_path = temp_dir.join("db2.sqlite");
        let conn2 = Connection::open(&db2_path).unwrap();
        let cols2 = vec![
            ColumnMeta {
                sql_name: "col_0".into(),
                original_name: "DestinationIP".into(),
                col_index: 0,
                inferred_type: "text".into(),
            },
            ColumnMeta {
                sql_name: "col_1".into(),
                original_name: "RuleAction".into(),
                col_index: 1,
                inferred_type: "text".into(),
            },
        ];
        db::create_schema(&conn2, &cols2).unwrap();
        conn2.execute("INSERT INTO rows (row_num, col_0, col_1) VALUES (10, '192.168.1.50', 'AllowAll')", []).unwrap();
        conn2.execute("INSERT INTO rows (row_num, col_0, col_1) VALUES (25, '10.0.0.1', 'ForwardRule')", []).unwrap();
        drop(conn2);

        let targets = vec![
            FileTarget {
                path: "audit_auth.xlsx".to_string(),
                sheet: Some("AuthSheet".to_string()),
                cache_db_path: Some(db1_path.to_string_lossy().to_string()),
            },
            FileTarget {
                path: "network_flow.csv".to_string(),
                sheet: None,
                cache_db_path: Some(db2_path.to_string_lossy().to_string()),
            },
        ];

        let events = vec![
            CorrelatedTimelineEvent {
                file_name: "audit_auth.xlsx".to_string(),
                path: "audit_auth.xlsx".to_string(),
                row_num: 1,
                epoch_ms: Some(1773050400000),
                utc_text: Some("2026-03-09 10:00:00 UTC".to_string()),
                user: Some("user1@corp.local".to_string()),
                host: None,
                action: Some("UserLoggedIn".to_string()),
                mitre_tags: vec!["T1078 Valid Accounts".to_string()],
            },
            CorrelatedTimelineEvent {
                file_name: "network_flow.csv".to_string(),
                path: "network_flow.csv".to_string(),
                row_num: 25,
                epoch_ms: Some(1773050700000),
                utc_text: Some("2026-03-09 10:05:00 UTC".to_string()),
                user: None,
                host: Some("10.0.0.1".to_string()),
                action: Some("ForwardRule".to_string()),
                mitre_tags: vec!["T1114.003 Email Forwarding".to_string()],
            },
        ];

        let dest_xlsx = temp_dir.join("test_multisheet_export.xlsx");
        let summary = export_unified_multisheet_xlsx(&targets, &events, &dest_xlsx).expect("export should succeed");

        assert_eq!(summary.total_events, 2);
        assert_eq!(summary.sheets_written.len(), 3);
        assert_eq!(summary.sheets_written[0], "Unified Timeline");
        assert!(summary.sheets_written[1].contains("audit_auth"));
        assert!(summary.sheets_written[2].contains("network_flow"));

        // Verify with Calamine reader
        use calamine::Reader;
        let mut workbook = calamine::open_workbook::<calamine::Xlsx<_>, _>(&dest_xlsx).expect("xlsx should open");
        let sheet_names = workbook.sheet_names().to_vec();
        assert_eq!(sheet_names.len(), 3);

        // Check Unified Timeline sheet
        let timeline_range = workbook.worksheet_range(&sheet_names[0]).unwrap();
        let tl_rows: Vec<_> = timeline_range.rows().collect();
        assert_eq!(tl_rows.len(), 3); // header + 2 events
        assert_eq!(tl_rows[0][1].to_string(), "Source File");
        assert_eq!(tl_rows[1][1].to_string(), "audit_auth.xlsx");
        assert_eq!(tl_rows[2][1].to_string(), "network_flow.csv");

        // Check file 1 sheet (full raw columns)
        let f1_range = workbook.worksheet_range(&sheet_names[1]).unwrap();
        let f1_rows: Vec<_> = f1_range.rows().collect();
        assert_eq!(f1_rows.len(), 2); // header + 1 event
        assert_eq!(f1_rows[0][0].to_string(), "Row #");
        assert_eq!(f1_rows[0][1].to_string(), "UserPrincipalName");
        assert_eq!(f1_rows[0][2].to_string(), "ActionName");
        assert_eq!(f1_rows[1][0].to_string(), "1");
        assert_eq!(f1_rows[1][1].to_string(), "user1@corp.local");
        assert_eq!(f1_rows[1][2].to_string(), "UserLoggedIn");

        // Check file 2 sheet (full raw columns)
        let f2_range = workbook.worksheet_range(&sheet_names[2]).unwrap();
        let f2_rows: Vec<_> = f2_range.rows().collect();
        assert_eq!(f2_rows.len(), 2); // header + 1 event
        assert_eq!(f2_rows[0][0].to_string(), "Row #");
        assert_eq!(f2_rows[0][1].to_string(), "DestinationIP");
        assert_eq!(f2_rows[0][2].to_string(), "RuleAction");
        assert_eq!(f2_rows[1][0].to_string(), "25");
        assert_eq!(f2_rows[1][1].to_string(), "10.0.0.1");
        assert_eq!(f2_rows[1][2].to_string(), "ForwardRule");

        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
