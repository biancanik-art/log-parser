use anyhow::{Context, Result};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ColumnMeta {
    pub sql_name: String,
    pub original_name: String,
    pub col_index: usize,
    pub inferred_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImportInfo {
    pub source_path: String,
    pub sheet_name: String,
    pub row_count: i64,
    pub imported_at: String,
}

/// Double-quotes a SQL identifier, doubling embedded quotes. Column names coming out of
/// `excel_import::sanitize_headers` are already restricted to `[a-z0-9_]`, but every generated
/// statement quotes identifiers anyway so a future change to that allowlist can't silently
/// reintroduce SQL injection via column names.
pub fn quote_ident(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

fn cache_dir() -> Result<PathBuf> {
    let base = std::env::var_os("LOCALAPPDATA")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let dir = base.join("log-parser").join("cache");
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("creating cache dir {}", dir.display()))?;
    Ok(dir)
}

/// Hashes (canonical path, size, mtime-in-nanoseconds, sheet name) so re-opening the same
/// file+sheet skips re-parsing, but any edit to the source file (which changes size or mtime)
/// invalidates the cache automatically. Nanosecond (not whole-second) mtime precision and
/// including the exact sheet name — not just its human-readable slug, see `slugify_sheet` — are
/// both load-bearing: two differently-named sheets can slugify to the same string (e.g. "A B"
/// and "A_B" both become "a_b"), and without the sheet name in the hash itself they'd collide on
/// the same cache file and silently serve one sheet's data for the other.
fn cache_key(path: &Path, sheet: &str) -> Result<String> {
    let canonical = std::fs::canonicalize(path)
        .with_context(|| format!("canonicalizing {}", path.display()))?;
    let metadata = std::fs::metadata(&canonical)
        .with_context(|| format!("reading metadata for {}", canonical.display()))?;
    let modified_nanos = metadata
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_nanos())
        .unwrap_or(0);

    let mut hasher = DefaultHasher::new();
    canonical.hash(&mut hasher);
    metadata.len().hash(&mut hasher);
    modified_nanos.hash(&mut hasher);
    sheet.hash(&mut hasher);
    Ok(format!("{:016x}", hasher.finish()))
}

fn slugify_sheet(sheet: &str) -> String {
    let mut out = String::new();
    for c in sheet.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        } else if !out.ends_with('_') {
            out.push('_');
        }
    }
    let trimmed = out.trim_matches('_');
    if trimmed.is_empty() {
        "sheet".to_string()
    } else {
        trimmed.to_string()
    }
}

pub fn cache_db_path(source_path: &Path, sheet: &str) -> Result<PathBuf> {
    let key = cache_key(source_path, sheet)?;
    let slug = slugify_sheet(sheet);
    Ok(cache_dir()?.join(format!("{key}_{slug}.sqlite3")))
}

pub fn open(db_path: &Path) -> rusqlite::Result<Connection> {
    Connection::open(db_path)
}

/// Relaxed durability for the initial bulk load only — data is read-only after import, so a
/// crash mid-import just means we re-parse next time (the cache file wouldn't have `_import_info`
/// populated yet), not corrupted state.
pub fn set_import_pragmas(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;
         PRAGMA synchronous = OFF;
         PRAGMA temp_store = MEMORY;",
    )
}

/// Switches back to a single-file rollback journal (merging and dropping any `-wal`/`-shm`
/// sidecar files in the process). Called once import is fully done, so the resulting cache file
/// is self-contained and safe to `rename()` into place — a lingering `-wal` file would otherwise
/// silently ride along (or not) depending on exactly when the rename happens.
pub fn restore_normal_pragmas(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute_batch("PRAGMA synchronous = NORMAL; PRAGMA journal_mode = DELETE;")
}

/// Creates `rows`, `_meta`, `_import_info`, and the external-content `rows_fts` FTS5 table.
/// `columns` must already be sanitized/deduped (see `excel_import::sanitize_headers`) and must
/// not include the reserved name `row_num`.
pub fn create_schema(conn: &Connection, columns: &[ColumnMeta]) -> rusqlite::Result<()> {
    let col_defs: Vec<String> = columns
        .iter()
        .map(|c| format!("{} TEXT", quote_ident(&c.sql_name)))
        .collect();
    let rows_sql = format!(
        "CREATE TABLE rows (row_num INTEGER PRIMARY KEY, {})",
        col_defs.join(", ")
    );
    conn.execute_batch(&format!(
        "{rows_sql};
         CREATE TABLE _meta (
            sql_name TEXT NOT NULL,
            original_name TEXT NOT NULL,
            col_index INTEGER NOT NULL,
            inferred_type TEXT NOT NULL
         );
         CREATE TABLE _import_info (
            source_path TEXT NOT NULL,
            sheet_name TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            imported_at TEXT NOT NULL
         );"
    ))?;

    {
        let mut stmt = conn.prepare(
            "INSERT INTO _meta (sql_name, original_name, col_index, inferred_type) VALUES (?1, ?2, ?3, ?4)",
        )?;
        for c in columns {
            stmt.execute(rusqlite::params![
                c.sql_name,
                c.original_name,
                c.col_index as i64,
                c.inferred_type
            ])?;
        }
    }

    let fts_cols: Vec<String> = columns.iter().map(|c| quote_ident(&c.sql_name)).collect();
    conn.execute_batch(&format!(
        "CREATE VIRTUAL TABLE rows_fts USING fts5({}, content='rows', content_rowid='row_num');",
        fts_cols.join(", ")
    ))?;

    Ok(())
}

pub fn create_column_roles_table(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS _column_roles (
            role TEXT PRIMARY KEY,
            sql_name TEXT NOT NULL,
            confidence REAL NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('suggested', 'confirmed', 'rejected')),
            reasons_json TEXT NOT NULL
         );",
    )
}

pub fn create_row_time_table(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS _row_time (
            row_num INTEGER PRIMARY KEY,
            epoch_ms INTEGER NOT NULL,
            utc_text TEXT NOT NULL,
            source_text TEXT NOT NULL,
            parse_status TEXT NOT NULL
         );
         CREATE INDEX IF NOT EXISTS idx_row_time_epoch ON _row_time(epoch_ms, row_num);",
    )
}

pub fn create_intel_schema(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS _intel_match (
            row_num INTEGER NOT NULL,
            tactic_id TEXT NOT NULL,
            tactic_name TEXT NOT NULL,
            technique_id TEXT NOT NULL,
            technique_name TEXT NOT NULL,
            pattern_id TEXT NOT NULL,
            keyword TEXT NOT NULL,
            column_name TEXT NOT NULL,
            score INTEGER NOT NULL
         );
         CREATE INDEX IF NOT EXISTS idx_intel_tactic_row
            ON _intel_match(tactic_id, row_num);
         CREATE INDEX IF NOT EXISTS idx_intel_tech_row
            ON _intel_match(technique_id, row_num);
         CREATE INDEX IF NOT EXISTS idx_intel_row
            ON _intel_match(row_num);
         CREATE TABLE IF NOT EXISTS _intel_scan_info (
            library_hash TEXT NOT NULL,
            role_hash TEXT NOT NULL,
            completed_at TEXT NOT NULL
         );",
    )
}

/// Bulk-populates the FTS5 index in one pass after all rows are loaded. No triggers are used
/// since `rows` is never updated/deleted after import.
pub fn populate_fts(conn: &Connection, columns: &[ColumnMeta]) -> rusqlite::Result<()> {
    let names: Vec<String> = columns.iter().map(|c| quote_ident(&c.sql_name)).collect();
    let cols_csv = names.join(", ");
    conn.execute_batch(&format!(
        "INSERT INTO rows_fts (rowid, {cols_csv}) SELECT row_num, {cols_csv} FROM rows;"
    ))
}

pub fn record_import_info(conn: &Connection, info: &ImportInfo) -> rusqlite::Result<()> {
    conn.execute(
        "INSERT INTO _import_info (source_path, sheet_name, row_count, imported_at) VALUES (?1, ?2, ?3, ?4)",
        rusqlite::params![info.source_path, info.sheet_name, info.row_count, info.imported_at],
    )?;
    Ok(())
}

pub fn load_columns(conn: &Connection) -> rusqlite::Result<Vec<ColumnMeta>> {
    let mut stmt = conn.prepare(
        "SELECT sql_name, original_name, col_index, inferred_type FROM _meta ORDER BY col_index",
    )?;
    let rows = stmt.query_map([], |row| {
        Ok(ColumnMeta {
            sql_name: row.get(0)?,
            original_name: row.get(1)?,
            col_index: row.get::<_, i64>(2)? as usize,
            inferred_type: row.get(3)?,
        })
    })?;
    rows.collect()
}

pub fn load_import_info(conn: &Connection) -> rusqlite::Result<ImportInfo> {
    conn.query_row(
        "SELECT source_path, sheet_name, row_count, imported_at FROM _import_info LIMIT 1",
        [],
        |row| {
            Ok(ImportInfo {
                source_path: row.get(0)?,
                sheet_name: row.get(1)?,
                row_count: row.get(2)?,
                imported_at: row.get(3)?,
            })
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Confirms FTS5 is actually compiled into rusqlite's bundled SQLite amalgamation. There is
    /// no `fts5` Cargo feature to opt into — this is the empirical check the plan calls for
    /// before anything else is built on top of that assumption.
    #[test]
    fn fts5_is_available_in_bundled_sqlite() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch("CREATE VIRTUAL TABLE t USING fts5(x)")
            .expect("FTS5 must be compiled into the bundled SQLite");
        conn.execute("INSERT INTO t(x) VALUES ('hello world')", [])
            .unwrap();
        let count: i64 = conn
            .query_row("SELECT count(*) FROM t WHERE t MATCH 'hello'", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn create_schema_and_populate_fts_roundtrip() {
        let conn = Connection::open_in_memory().unwrap();
        let columns = vec![
            ColumnMeta {
                sql_name: "time_generated".into(),
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
        ];
        create_schema(&conn, &columns).unwrap();
        conn.execute(
            "INSERT INTO rows (row_num, time_generated, account) VALUES (1, '2026-01-01T00:00:00Z', 'forensic_test_marker_XYZ')",
            [],
        )
        .unwrap();
        populate_fts(&conn, &columns).unwrap();

        let hit: i64 = conn
            .query_row(
                "SELECT rowid FROM rows_fts WHERE rows_fts MATCH ?1",
                rusqlite::params!["forensic_test_marker_XYZ"],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(hit, 1);

        let loaded = load_columns(&conn).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].sql_name, "time_generated");
    }

    #[test]
    fn slugify_sheet_handles_special_chars() {
        assert_eq!(slugify_sheet("Sheet 1!"), "sheet_1");
        assert_eq!(slugify_sheet("###"), "sheet");
    }
}
