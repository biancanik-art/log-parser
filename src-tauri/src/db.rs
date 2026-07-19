use anyhow::{Context, Result};
use rusqlite::{Connection, TransactionBehavior};
use serde::{Deserialize, Serialize};
use std::collections::{hash_map::DefaultHasher, HashSet};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// All production connections to an imported cache use the same bounded wait when another
/// background task is publishing a short SQLite write transaction. Keeping this here (rather
/// than in individual features) prevents semantic indexing, timestamp normalization, and audit
/// writes from failing immediately merely because their batch commits overlap.
pub const CACHE_BUSY_TIMEOUT: Duration = Duration::from_secs(3);
const ROW_TIME_SCAVENGE_TABLE_LIMIT: usize = 4;
const ROW_TIME_SCAVENGE_ROW_LIMIT: i64 = 32_768;
const ROW_TIME_MAX_LIVE_OPERATIONS: usize = 32;
const ROW_TIME_FOREIGN_OPERATION_LEASE: Duration = Duration::from_secs(10 * 60);
const ROW_TIME_RECOVERY_BACKLOG_MESSAGE: &str =
    "bounded timestamp recovery made progress but abandoned staging data remains; retry opening the cache to continue cleanup";
static ROW_TIME_OPERATION_COUNTER: AtomicU64 = AtomicU64::new(1);

#[derive(Debug)]
pub struct RowTimeOperationClaim {
    generation: i64,
    owner_token: String,
    owner_session: String,
    stage_name: String,
}

impl RowTimeOperationClaim {
    pub fn generation(&self) -> i64 {
        self.generation
    }

    pub fn owner_token(&self) -> &str {
        &self.owner_token
    }

    pub fn owner_session(&self) -> &str {
        &self.owner_session
    }

    pub fn stage_name(&self) -> &str {
        &self.stage_name
    }
}

impl Drop for RowTimeOperationClaim {
    fn drop(&mut self) {
        if let Ok(mut live) = live_row_time_operation_tokens().lock() {
            live.remove(&self.owner_token);
        }
    }
}

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

fn row_time_process_session() -> &'static str {
    static SESSION: OnceLock<String> = OnceLock::new();
    SESSION.get_or_init(|| {
        let started = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        format!("{}-{started}", std::process::id())
    })
}

fn live_row_time_operation_tokens() -> &'static Mutex<HashSet<String>> {
    static TOKENS: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    TOKENS.get_or_init(|| Mutex::new(HashSet::new()))
}

fn row_time_recovery_busy(message: impl Into<String>) -> rusqlite::Error {
    rusqlite::Error::SqliteFailure(
        rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_BUSY),
        Some(message.into()),
    )
}

/// Distinguishes the deliberate, progress-making recovery retry from ordinary SQLite busy
/// errors. Cache loading may retry only this exact condition; treating a lock/contention error
/// as recovery progress could otherwise replace or hide an unusable cache.
pub fn is_row_time_recovery_backlog(error: &rusqlite::Error) -> bool {
    matches!(
        error,
        rusqlite::Error::SqliteFailure(code, Some(message))
            if code.code == rusqlite::ErrorCode::DatabaseBusy
                && message == ROW_TIME_RECOVERY_BACKLOG_MESSAGE
    )
}

fn sqlite_table_exists(conn: &Connection, table: &str) -> rusqlite::Result<bool> {
    conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?1)",
        [table],
        |row| row.get::<_, i64>(0),
    )
    .map(|value| value != 0)
}

fn row_time_cleanup_candidates(
    conn: &Connection,
    preserved: &HashSet<String>,
    limit: usize,
) -> rusqlite::Result<Vec<String>> {
    let preserved = preserved.iter().cloned().collect::<Vec<_>>();
    let exclusions = if preserved.is_empty() {
        String::new()
    } else {
        format!(
            " AND name NOT IN ({})",
            (1..=preserved.len())
                .map(|index| format!("?{index}"))
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    let sql = format!(
        "SELECT name FROM sqlite_master
         WHERE type = 'table'
           AND (name GLOB '_row_time_stage_*'
                OR name GLOB '_row_time_previous_*')
           {exclusions}
         ORDER BY name
         LIMIT {limit}"
    );
    let mut statement = conn.prepare(&sql)?;
    let rows = statement.query_map(rusqlite::params_from_iter(preserved.iter()), |row| {
        row.get::<_, String>(0)
    })?;
    let names = rows.collect();
    names
}

/// A foreign process refreshes `updated_at` after every staged batch. A short grace lease avoids
/// mistaking its freshly committed claim for a crashed operation while still making abandoned
/// cross-process objects eligible for bounded cleanup later.
fn row_time_foreign_operation_lease_is_fresh(updated_at: &str) -> bool {
    let Ok(updated_at) = chrono::DateTime::parse_from_rfc3339(updated_at) else {
        return false;
    };
    let age = chrono::Utc::now().signed_duration_since(updated_at.with_timezone(&chrono::Utc));
    if age < chrono::Duration::zero() {
        return true;
    }
    age.to_std()
        .map(|age| age <= ROW_TIME_FOREIGN_OPERATION_LEASE)
        .unwrap_or(false)
}

fn scavenge_abandoned_row_time_objects(conn: &mut Connection) -> rusqlite::Result<()> {
    scavenge_abandoned_row_time_objects_with_hooks(conn, || {}, || {})
}

/// Reclaims timestamp staging objects left by a crashed/panicked operation. The work per open is
/// deliberately capped. If the cap cannot finish recovery, the open fails explicitly after
/// committing bounded progress; retrying continues cleanup instead of silently retaining an
/// unbounded cache. Same-process operations use the live registry; fresh foreign operations use
/// the persisted heartbeat lease.
fn scavenge_abandoned_row_time_objects_with_hooks(
    conn: &mut Connection,
    before_publication_authority: impl FnOnce(),
    after_publication_authority: impl FnOnce(),
) -> rusqlite::Result<()> {
    let has_candidates: bool = conn.query_row(
        "SELECT EXISTS(
            SELECT 1 FROM sqlite_master
            WHERE type = 'table'
              AND (name GLOB '_row_time_stage_*'
                   OR name GLOB '_row_time_previous_*')
         )",
        [],
        |row| row.get::<_, i64>(0),
    )? != 0;
    let has_operation_rows = sqlite_table_exists(conn, "_row_time_operation")?
        && conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM _row_time_operation LIMIT 1)",
            [],
            |row| row.get::<_, i64>(0),
        )? != 0;
    if !has_candidates && !has_operation_rows {
        return Ok(());
    }

    // The test hook models a claim being committed after the cheap read-only preflight. Taking
    // SQLite publication authority before consulting the process registry makes that ordering
    // safe: an operation is either visible in both places, or cannot commit until this cleanup
    // transaction has finished.
    before_publication_authority();
    let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
    after_publication_authority();
    let live_tokens = live_row_time_operation_tokens()
        .lock()
        .map_err(|_| row_time_recovery_busy("timestamp operation registry is unavailable"))?;
    let session = row_time_process_session().to_string();
    let mut preserved = HashSet::new();
    let mut abandoned_tokens = Vec::new();
    if sqlite_table_exists(&tx, "_row_time_operation")? {
        let operations = {
            let mut statement = tx.prepare(
                "SELECT owner_token, owner_session, stage_name, backup_name, updated_at
                 FROM _row_time_operation",
            )?;
            let rows = statement.query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, Option<String>>(3)?,
                    row.get::<_, String>(4)?,
                ))
            })?;
            let operations = rows.collect::<rusqlite::Result<Vec<_>>>()?;
            operations
        };
        for (token, owner_session, stage_name, backup_name, updated_at) in operations {
            let live_in_this_process = owner_session == session && live_tokens.contains(&token);
            let leased_by_another_process =
                owner_session != session && row_time_foreign_operation_lease_is_fresh(&updated_at);
            if live_in_this_process || leased_by_another_process {
                preserved.insert(stage_name);
                if let Some(backup_name) = backup_name {
                    preserved.insert(backup_name);
                }
            } else {
                abandoned_tokens.push(token);
            }
        }
    }

    let candidates =
        row_time_cleanup_candidates(&tx, &preserved, ROW_TIME_SCAVENGE_TABLE_LIMIT + 1)?;
    let mut remaining_rows = ROW_TIME_SCAVENGE_ROW_LIMIT;
    for table_name in candidates.iter().take(ROW_TIME_SCAVENGE_TABLE_LIMIT) {
        let table = quote_ident(table_name);
        let has_row_num: bool = tx.query_row(
            "SELECT EXISTS(SELECT 1 FROM pragma_table_info(?1) WHERE name = 'row_num')",
            [table_name],
            |row| row.get::<_, i64>(0),
        )? != 0;
        if has_row_num && remaining_rows > 0 {
            let deleted = tx.execute(
                &format!(
                    "DELETE FROM {table}
                     WHERE row_num IN (
                        SELECT row_num FROM {table} ORDER BY row_num LIMIT ?1
                     )"
                ),
                [remaining_rows],
            )? as i64;
            remaining_rows = remaining_rows.saturating_sub(deleted);
        }
        let has_rows = if has_row_num {
            tx.query_row(
                &format!("SELECT EXISTS(SELECT 1 FROM {table} LIMIT 1)"),
                [],
                |row| row.get::<_, i64>(0),
            )? != 0
        } else {
            false
        };
        if !has_rows {
            tx.execute_batch(&format!("DROP TABLE IF EXISTS {table}"))?;
        }
    }
    if sqlite_table_exists(&tx, "_row_time_operation")? {
        for token in abandoned_tokens {
            tx.execute(
                "DELETE FROM _row_time_operation WHERE owner_token = ?1",
                [token],
            )?;
        }
    }
    let backlog = !row_time_cleanup_candidates(&tx, &preserved, 1)?.is_empty();
    tx.commit()?;
    if backlog {
        return Err(row_time_recovery_busy(ROW_TIME_RECOVERY_BACKLOG_MESSAGE));
    }
    Ok(())
}

pub fn open(db_path: &Path) -> rusqlite::Result<Connection> {
    let mut conn = Connection::open(db_path)?;
    conn.busy_timeout(CACHE_BUSY_TIMEOUT)?;
    scavenge_abandoned_row_time_objects(&mut conn)?;
    Ok(conn)
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
         CREATE TABLE IF NOT EXISTS _row_time_info (
            binding_version TEXT NOT NULL,
            source_column TEXT NOT NULL,
            schema_sha256 TEXT NOT NULL,
            import_sha256 TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            date_convention TEXT,
            timezone_applied TEXT,
            completed_at TEXT NOT NULL
         );
         CREATE TABLE IF NOT EXISTS _row_time_operation_control (
            singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
            latest_generation INTEGER NOT NULL
         );
         INSERT OR IGNORE INTO _row_time_operation_control(singleton, latest_generation)
            VALUES (1, 0);
         CREATE TABLE IF NOT EXISTS _row_time_operation (
            generation INTEGER PRIMARY KEY,
            owner_token TEXT NOT NULL UNIQUE,
            owner_session TEXT NOT NULL,
            stage_name TEXT NOT NULL UNIQUE,
            backup_name TEXT,
            started_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
         );",
    )?;

    // A staged timestamp build keeps its generation-specific index when its table is atomically
    // renamed to `_row_time`. Do not rebuild the same potentially large index under the legacy
    // canonical name every time normalization runs.
    let mut indexes = conn.prepare("SELECT name FROM pragma_index_list('_row_time')")?;
    let names = indexes
        .query_map([], |row| row.get::<_, String>(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    for name in names {
        let mut columns = conn.prepare("SELECT name FROM pragma_index_info(?1) ORDER BY seqno")?;
        let indexed = columns
            .query_map([name], |row| row.get::<_, String>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        if indexed.starts_with(&["epoch_ms".to_string(), "row_num".to_string()]) {
            return Ok(());
        }
    }

    conn.execute_batch(
        "CREATE INDEX IF NOT EXISTS idx_row_time_epoch ON _row_time(epoch_ms, row_num);",
    )
}

pub fn begin_row_time_operation(
    conn: &mut Connection,
    stage_name: &str,
    index_name: &str,
) -> rusqlite::Result<RowTimeOperationClaim> {
    begin_row_time_operation_with_hook(conn, stage_name, index_name, || {})
}

fn begin_row_time_operation_with_hook(
    conn: &mut Connection,
    stage_name: &str,
    index_name: &str,
    after_publication_authority: impl FnOnce(),
) -> rusqlite::Result<RowTimeOperationClaim> {
    create_row_time_table(conn)?;
    let session = row_time_process_session().to_string();
    let sequence = ROW_TIME_OPERATION_COUNTER.fetch_add(1, Ordering::Relaxed);
    let owner_token = format!("{session}-{sequence}");
    let stage = quote_ident(stage_name);
    let index = quote_ident(index_name);
    let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
    after_publication_authority();

    // Scavenging uses the same SQLite-then-registry lock order. Keep the registry guard through
    // commit so a published operation row and its in-process liveness token become visible as
    // one ordered event to every later scavenger.
    let mut live = live_row_time_operation_tokens()
        .lock()
        .map_err(|_| row_time_recovery_busy("timestamp operation registry is unavailable"))?;
    if live.len() >= ROW_TIME_MAX_LIVE_OPERATIONS {
        return Err(row_time_recovery_busy(
            "too many timestamp normalization operations are active",
        ));
    }
    live.insert(owner_token.clone());

    let result = (|| {
        let advanced = tx.execute(
            "UPDATE _row_time_operation_control
             SET latest_generation = latest_generation + 1
             WHERE singleton = 1 AND latest_generation < ?1",
            [i64::MAX],
        )?;
        if advanced != 1 {
            return Err(row_time_recovery_busy(
                "timestamp operation generation is unavailable",
            ));
        }
        let generation: i64 = tx.query_row(
            "SELECT latest_generation FROM _row_time_operation_control WHERE singleton = 1",
            [],
            |row| row.get(0),
        )?;
        let now = chrono::Utc::now().to_rfc3339();
        tx.execute(
            "INSERT INTO _row_time_operation (
                generation, owner_token, owner_session, stage_name, backup_name,
                started_at, updated_at
             ) VALUES (?1, ?2, ?3, ?4, NULL, ?5, ?5)",
            rusqlite::params![generation, owner_token, session, stage_name, now],
        )?;
        tx.execute_batch(&format!(
            "CREATE TABLE {stage} (
                row_num INTEGER PRIMARY KEY,
                epoch_ms INTEGER NOT NULL,
                utc_text TEXT NOT NULL,
                source_text TEXT NOT NULL,
                parse_status TEXT NOT NULL
             );
             CREATE INDEX {index} ON {stage}(epoch_ms, row_num);"
        ))?;
        tx.commit()?;
        Ok(RowTimeOperationClaim {
            generation,
            owner_token: owner_token.clone(),
            owner_session: session.clone(),
            stage_name: stage_name.to_string(),
        })
    })();
    if result.is_err() {
        live.remove(&owner_token);
    }
    drop(live);
    result
}

pub fn row_time_operation_is_latest(
    conn: &Connection,
    claim: &RowTimeOperationClaim,
) -> rusqlite::Result<bool> {
    // The ownership check doubles as the persisted heartbeat used by other application
    // processes. Scavengers retain a foreign claim while this timestamp remains within the
    // bounded lease window.
    conn.execute(
        "UPDATE _row_time_operation
         SET updated_at = ?5
         WHERE generation = ?1
           AND owner_token = ?2
           AND owner_session = ?3
           AND stage_name = ?4
           AND generation = (
                SELECT latest_generation
                FROM _row_time_operation_control
                WHERE singleton = 1
           )",
        rusqlite::params![
            claim.generation,
            claim.owner_token,
            claim.owner_session,
            claim.stage_name,
            chrono::Utc::now().to_rfc3339()
        ],
    )
    .map(|changed| changed == 1)
}

pub fn set_row_time_operation_backup(
    conn: &Connection,
    claim: &RowTimeOperationClaim,
    backup_name: &str,
) -> rusqlite::Result<bool> {
    conn.execute(
        "UPDATE _row_time_operation
         SET backup_name = ?4, updated_at = ?5
         WHERE generation = ?1 AND owner_token = ?2 AND owner_session = ?3",
        rusqlite::params![
            claim.generation,
            claim.owner_token,
            claim.owner_session,
            backup_name,
            chrono::Utc::now().to_rfc3339()
        ],
    )
    .map(|changed| changed == 1)
}

pub fn retire_row_time_operation(
    conn: &Connection,
    claim: &RowTimeOperationClaim,
) -> rusqlite::Result<bool> {
    conn.execute(
        "DELETE FROM _row_time_operation
         WHERE generation = ?1 AND owner_token = ?2 AND owner_session = ?3",
        rusqlite::params![claim.generation, claim.owner_token, claim.owner_session],
    )
    .map(|changed| changed == 1)
}

pub fn create_anomaly_schema(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS _anomaly (
            row_num INTEGER NOT NULL,
            category TEXT NOT NULL,
            score INTEGER NOT NULL,
            reason TEXT NOT NULL,
            column_name TEXT NOT NULL
         );
         CREATE INDEX IF NOT EXISTS idx_anomaly_row ON _anomaly(row_num);
         CREATE INDEX IF NOT EXISTS idx_anomaly_category ON _anomaly(category, row_num);
         CREATE TABLE IF NOT EXISTS _anomaly_info (
            rows_scanned INTEGER NOT NULL,
            finding_count INTEGER NOT NULL,
            completed_at TEXT NOT NULL
         );",
    )
}

pub fn create_activity_schema(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS _row_activity (
            row_num INTEGER PRIMARY KEY,
            category TEXT NOT NULL,
            detail TEXT NOT NULL,
            source_column TEXT NOT NULL
         );
         CREATE INDEX IF NOT EXISTS idx_row_activity_category
            ON _row_activity(category, row_num);
         CREATE TABLE IF NOT EXISTS _row_activity_info (
            rows_classified INTEGER NOT NULL,
            completed_at TEXT NOT NULL
         );",
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
         );
         CREATE TABLE IF NOT EXISTS _intel_chain (
            chain_id INTEGER NOT NULL,
            host TEXT,
            start_epoch_ms INTEGER,
            end_epoch_ms INTEGER,
            first_row INTEGER NOT NULL,
            last_row INTEGER NOT NULL,
            tactic_count INTEGER NOT NULL,
            event_count INTEGER NOT NULL,
            row_count INTEGER NOT NULL,
            score INTEGER NOT NULL,
            tactic_names TEXT NOT NULL,
            technique_names TEXT NOT NULL,
            sample_rows TEXT NOT NULL
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

    fn timestamp_test_path(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "log-parser-{label}-{}-{}.sqlite3",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ))
    }

    fn create_timestamp_object(conn: &Connection, table_name: &str) {
        conn.execute_batch(&format!(
            "CREATE TABLE {} (
                row_num INTEGER PRIMARY KEY,
                epoch_ms INTEGER NOT NULL,
                utc_text TEXT NOT NULL,
                source_text TEXT NOT NULL,
                parse_status TEXT NOT NULL
             );",
            quote_ident(table_name)
        ))
        .unwrap();
    }

    #[test]
    fn timestamp_recovery_backlog_condition_is_dedicated() {
        let backlog = row_time_recovery_busy(ROW_TIME_RECOVERY_BACKLOG_MESSAGE);
        assert!(is_row_time_recovery_backlog(&backlog));

        let ordinary_busy = row_time_recovery_busy("database is locked by another writer");
        assert!(!is_row_time_recovery_backlog(&ordinary_busy));
        assert!(!is_row_time_recovery_backlog(
            &rusqlite::Error::QueryReturnedNoRows
        ));
    }

    #[test]
    fn empty_timestamp_operation_table_does_not_take_writer_authority_on_open() {
        let path = timestamp_test_path("row-time-empty-operation");
        let setup = Connection::open(&path).unwrap();
        create_row_time_table(&setup).unwrap();
        drop(setup);

        let writer = Connection::open(&path).unwrap();
        writer.execute_batch("BEGIN IMMEDIATE").unwrap();
        let reopened =
            open(&path).expect("zero operation rows must keep ordinary cache opens read-only");
        assert!(sqlite_table_exists(&reopened, "_row_time_operation").unwrap());
        drop(reopened);
        writer.execute_batch("ROLLBACK").unwrap();
        drop(writer);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn scavenger_rechecks_same_process_claim_after_preflight_interleaving() {
        let path = timestamp_test_path("row-time-registry-recheck");
        let setup = Connection::open(&path).unwrap();
        create_timestamp_object(&setup, "_row_time_stage_abandoned");
        drop(setup);

        let mut scavenger = Connection::open(&path).unwrap();
        scavenger.busy_timeout(CACHE_BUSY_TIMEOUT).unwrap();
        let mut claimant = Connection::open(&path).unwrap();
        claimant.busy_timeout(CACHE_BUSY_TIMEOUT).unwrap();
        let mut claim = None;
        scavenge_abandoned_row_time_objects_with_hooks(
            &mut scavenger,
            || {
                claim = Some(
                    begin_row_time_operation(
                        &mut claimant,
                        "_row_time_stage_live_interleaving",
                        "_row_time_stage_live_interleaving_epoch",
                    )
                    .unwrap(),
                );
            },
            || {},
        )
        .unwrap();

        let claim = claim.expect("the interleaved operation must be published");
        assert!(sqlite_table_exists(&scavenger, claim.stage_name()).unwrap());
        assert!(!sqlite_table_exists(&scavenger, "_row_time_stage_abandoned").unwrap());
        assert_eq!(
            claimant
                .query_row(
                    "SELECT COUNT(*) FROM _row_time_operation WHERE owner_token = ?1",
                    [claim.owner_token()],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            1
        );

        retire_row_time_operation(&claimant, &claim).unwrap();
        drop(claim);
        drop(claimant);
        drop(scavenger);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn timestamp_claim_and_scavenger_share_database_then_registry_lock_order() {
        let path = timestamp_test_path("row-time-lock-order");
        let setup = Connection::open(&path).unwrap();
        create_timestamp_object(&setup, "_row_time_stage_abandoned");
        drop(setup);

        let mut scavenger = Connection::open(&path).unwrap();
        scavenger.busy_timeout(CACHE_BUSY_TIMEOUT).unwrap();
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (authority_tx, authority_rx) = std::sync::mpsc::channel();
        let mut claimant = None;
        let claimant_path = path.clone();
        scavenge_abandoned_row_time_objects_with_hooks(
            &mut scavenger,
            || {},
            || {
                claimant = Some(std::thread::spawn(move || {
                    let mut conn = Connection::open(claimant_path).unwrap();
                    conn.busy_timeout(CACHE_BUSY_TIMEOUT).unwrap();
                    started_tx.send(()).unwrap();
                    let claim = begin_row_time_operation_with_hook(
                        &mut conn,
                        "_row_time_stage_waiting_claim",
                        "_row_time_stage_waiting_claim_epoch",
                        || authority_tx.send(()).unwrap(),
                    )
                    .unwrap();
                    (conn, claim)
                }));
                started_rx
                    .recv_timeout(Duration::from_secs(1))
                    .expect("claimant thread must start");

                // The scavenger owns SQLite publication authority, so the claimant must not
                // reach its registry acquisition point yet and the registry itself must remain
                // available. This would fail (or deadlock in production) with registry -> DB.
                assert!(authority_rx
                    .recv_timeout(Duration::from_millis(25))
                    .is_err());
                let registry_deadline = std::time::Instant::now() + Duration::from_secs(1);
                let registry_available = loop {
                    if let Ok(guard) = live_row_time_operation_tokens().try_lock() {
                        drop(guard);
                        break true;
                    }
                    if std::time::Instant::now() >= registry_deadline {
                        break false;
                    }
                    std::thread::sleep(Duration::from_millis(1));
                };
                assert!(registry_available);
            },
        )
        .unwrap();

        let (claimant_conn, claim) = claimant.unwrap().join().unwrap();
        authority_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("claimant must acquire publication authority after scavenging commits");
        assert!(sqlite_table_exists(&claimant_conn, claim.stage_name()).unwrap());
        retire_row_time_operation(&claimant_conn, &claim).unwrap();
        drop(claim);
        drop(claimant_conn);
        drop(scavenger);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn foreign_process_claim_is_leased_then_reclaimed_when_stale() {
        let path = timestamp_test_path("row-time-foreign-lease");
        let setup = Connection::open(&path).unwrap();
        create_row_time_table(&setup).unwrap();
        setup
            .execute(
                "INSERT INTO _row_time (
                    row_num, epoch_ms, utc_text, source_text, parse_status
                 ) VALUES (7, 7, 'canonical', 'canonical', 'test')",
                [],
            )
            .unwrap();
        create_timestamp_object(&setup, "_row_time_stage_foreign");
        create_timestamp_object(&setup, "_row_time_previous_foreign");
        let now = chrono::Utc::now().to_rfc3339();
        setup
            .execute(
                "INSERT INTO _row_time_operation (
                    generation, owner_token, owner_session, stage_name, backup_name,
                    started_at, updated_at
                 ) VALUES (1, 'foreign-token', 'foreign-process-session',
                    '_row_time_stage_foreign', '_row_time_previous_foreign', ?1, ?1)",
                [now],
            )
            .unwrap();
        drop(setup);

        let fresh = open(&path).expect("a fresh foreign-process lease must be retained");
        assert!(sqlite_table_exists(&fresh, "_row_time_stage_foreign").unwrap());
        assert!(sqlite_table_exists(&fresh, "_row_time_previous_foreign").unwrap());
        assert_eq!(
            fresh
                .query_row("SELECT COUNT(*) FROM _row_time_operation", [], |row| {
                    row.get::<_, i64>(0)
                })
                .unwrap(),
            1
        );
        fresh
            .execute(
                "UPDATE _row_time_operation SET updated_at = '2000-01-01T00:00:00Z'",
                [],
            )
            .unwrap();
        drop(fresh);

        let reclaimed = open(&path).expect("an expired foreign-process lease must be reclaimed");
        assert!(!sqlite_table_exists(&reclaimed, "_row_time_stage_foreign").unwrap());
        assert!(!sqlite_table_exists(&reclaimed, "_row_time_previous_foreign").unwrap());
        assert_eq!(
            reclaimed
                .query_row("SELECT COUNT(*) FROM _row_time_operation", [], |row| {
                    row.get::<_, i64>(0)
                })
                .unwrap(),
            0
        );
        assert_eq!(
            reclaimed
                .query_row(
                    "SELECT source_text FROM _row_time WHERE row_num = 7",
                    [],
                    |row| row.get::<_, String>(0),
                )
                .unwrap(),
            "canonical"
        );
        drop(reclaimed);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn cache_connections_share_the_bounded_busy_timeout() {
        let path = std::env::temp_dir().join(format!(
            "log-parser-busy-timeout-{}-{}.sqlite3",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let conn = open(&path).unwrap();
        let timeout_ms: i64 = conn
            .query_row("PRAGMA busy_timeout", [], |row| row.get(0))
            .unwrap();
        assert_eq!(timeout_ms, CACHE_BUSY_TIMEOUT.as_millis() as i64);
        drop(conn);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn timestamp_reopen_scavenging_is_bounded_and_reports_remaining_backlog() {
        let path = std::env::temp_dir().join(format!(
            "log-parser-row-time-scavenge-{}-{}.sqlite3",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let mut setup = Connection::open(&path).unwrap();
        setup
            .execute_batch(
                "CREATE TABLE _row_time_stage_interrupted (
                    row_num INTEGER PRIMARY KEY,
                    epoch_ms INTEGER NOT NULL,
                    utc_text TEXT NOT NULL,
                    source_text TEXT NOT NULL,
                    parse_status TEXT NOT NULL
                 );",
            )
            .unwrap();
        let tx = setup.transaction().unwrap();
        {
            let mut insert = tx
                .prepare(
                    "INSERT INTO _row_time_stage_interrupted (
                        row_num, epoch_ms, utc_text, source_text, parse_status
                     ) VALUES (?1, ?1, 'x', 'x', 'test')",
                )
                .unwrap();
            for row_num in 1..=(ROW_TIME_SCAVENGE_ROW_LIMIT + 1) {
                insert.execute([row_num]).unwrap();
            }
        }
        tx.commit().unwrap();
        drop(setup);

        let first_error = match open(&path) {
            Ok(_) => panic!("one bounded reopen must not silently leave a cleanup backlog"),
            Err(error) => error,
        };
        assert!(first_error
            .to_string()
            .contains("bounded timestamp recovery"));
        let reopened = open(&path).expect("a retry must finish the bounded remainder");
        assert!(!sqlite_table_exists(&reopened, "_row_time_stage_interrupted").unwrap());
        drop(reopened);
        let _ = std::fs::remove_file(path);
    }

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
