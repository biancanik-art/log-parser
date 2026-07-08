use crate::csv_import;
use crate::db::ColumnMeta;
use crate::excel_import;
use anyhow::{bail, Result};
use std::path::Path;

pub struct ImportResult {
    pub columns: Vec<ColumnMeta>,
    pub row_count: i64,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum SourceFormat {
    Csv,
    Excel,
}

impl From<csv_import::ImportResult> for ImportResult {
    fn from(result: csv_import::ImportResult) -> Self {
        Self {
            columns: result.columns,
            row_count: result.row_count,
        }
    }
}

impl From<excel_import::ImportResult> for ImportResult {
    fn from(result: excel_import::ImportResult) -> Self {
        Self {
            columns: result.columns,
            row_count: result.row_count,
        }
    }
}

fn source_format(path: &Path) -> Result<SourceFormat> {
    match path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase())
        .as_deref()
    {
        Some("csv") => Ok(SourceFormat::Csv),
        Some("xls" | "xlsx" | "xlsm" | "xlsb" | "ods") => Ok(SourceFormat::Excel),
        Some(ext) => bail!("unsupported file extension .{ext}; expected CSV or Excel workbook"),
        None => bail!("unsupported file type with no extension; expected CSV or Excel workbook"),
    }
}

fn csv_sheet_name(path: &Path) -> String {
    path.file_stem()
        .map(|stem| stem.to_string_lossy().trim().to_string())
        .filter(|stem| !stem.is_empty())
        .unwrap_or_else(|| "CSV".to_string())
}

pub fn list_sheet_names(path: &Path) -> Result<Vec<String>> {
    match source_format(path)? {
        SourceFormat::Csv => Ok(vec![csv_sheet_name(path)]),
        SourceFormat::Excel => excel_import::list_sheet_names(path),
    }
}

pub fn import_into_db(
    source_path: &Path,
    sheet_name: &str,
    db_path: &Path,
    on_progress: impl FnMut(u64, u64),
) -> Result<ImportResult> {
    match source_format(source_path)? {
        SourceFormat::Csv => {
            csv_import::import_into_db(source_path, db_path, on_progress).map(Into::into)
        }
        SourceFormat::Excel => {
            excel_import::import_into_db(source_path, sheet_name, db_path, on_progress)
                .map(Into::into)
        }
    }
}
