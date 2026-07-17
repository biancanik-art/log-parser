use crate::db::{self, ColumnMeta, ImportInfo};
use crate::export;
use crate::intel::matcher::{self, IntelScanSummary};
use crate::intel::{llm_parser, parser as guided_parser, query as guided_query, roles, time};
use crate::query::{self, QueryExpression, QueryPage, QuerySpec};
use crate::report::{self, ReportExportSummary};
use crate::semantic;
use crate::tabular_import;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tauri::{path::BaseDirectory, AppHandle, Emitter, Manager, State};

const IMPORT_CACHE_RECOVERY_MAX_ATTEMPTS: usize = 64;
const IMPORT_CACHE_RECOVERY_MAX_ELAPSED: Duration = Duration::from_secs(15);

#[derive(Debug)]
enum ImportCacheOpenError {
    Reimportable,
    Preserved(String),
}

fn cache_open_error_is_reimportable(error: &rusqlite::Error) -> bool {
    matches!(
        error.sqlite_error_code(),
        Some(rusqlite::ErrorCode::DatabaseCorrupt | rusqlite::ErrorCode::NotADatabase)
    )
}

fn cache_metadata_error_is_reimportable(error: &rusqlite::Error, table: &str) -> bool {
    if cache_open_error_is_reimportable(error) {
        return true;
    }
    match error {
        rusqlite::Error::QueryReturnedNoRows => true,
        rusqlite::Error::SqliteFailure(code, Some(message)) => {
            code.code == rusqlite::ErrorCode::Unknown
                && (message == &format!("no such table: {table}")
                    || message.starts_with("no such column:"))
        }
        // Conversion/index/type failures mean the cache metadata itself does not match the
        // schema this version writes. Other SQLite failures may be transient access, I/O, or
        // contention errors and must preserve the existing database.
        rusqlite::Error::SqliteFailure(_, _) => false,
        rusqlite::Error::FromSqlConversionFailure(_, _, _)
        | rusqlite::Error::IntegralValueOutOfRange(_, _)
        | rusqlite::Error::Utf8Error(_)
        | rusqlite::Error::InvalidColumnIndex(_)
        | rusqlite::Error::InvalidColumnName(_)
        | rusqlite::Error::InvalidColumnType(_, _, _) => true,
        _ => false,
    }
}

fn load_existing_cache_metadata_for_import(
    conn: &rusqlite::Connection,
) -> Result<(Vec<ColumnMeta>, ImportInfo), ImportCacheOpenError> {
    let columns = db::load_columns(conn).map_err(|error| {
        if cache_metadata_error_is_reimportable(&error, "_meta") {
            ImportCacheOpenError::Reimportable
        } else {
            ImportCacheOpenError::Preserved(format!(
                "the existing cache opened, but its column metadata could not be read safely; it was preserved and was not re-imported: {error}"
            ))
        }
    })?;
    let info = db::load_import_info(conn).map_err(|error| {
        if cache_metadata_error_is_reimportable(&error, "_import_info") {
            ImportCacheOpenError::Reimportable
        } else {
            ImportCacheOpenError::Preserved(format!(
                "the existing cache opened, but its import metadata could not be read safely; it was preserved and was not re-imported: {error}"
            ))
        }
    })?;
    Ok((columns, info))
}

fn open_existing_cache_for_import(
    db_path: &Path,
) -> Result<rusqlite::Connection, ImportCacheOpenError> {
    open_existing_cache_for_import_with_limits(
        db_path,
        IMPORT_CACHE_RECOVERY_MAX_ATTEMPTS,
        IMPORT_CACHE_RECOVERY_MAX_ELAPSED,
    )
}

fn open_existing_cache_for_import_with_limits(
    db_path: &Path,
    max_attempts: usize,
    max_elapsed: Duration,
) -> Result<rusqlite::Connection, ImportCacheOpenError> {
    let started = std::time::Instant::now();
    let max_attempts = max_attempts.max(1);
    let mut attempts = 0_usize;
    let mut recovery_started = false;

    loop {
        attempts += 1;
        match db::open(db_path) {
            Ok(conn) => return Ok(conn),
            Err(error) if db::is_row_time_recovery_backlog(&error) => {
                recovery_started = true;
                if attempts >= max_attempts || started.elapsed() >= max_elapsed {
                    return Err(ImportCacheOpenError::Preserved(format!(
                        "timestamp recovery for the existing cache did not finish after {attempts} bounded open attempts; the existing cache was preserved and was not re-imported: {error}"
                    )));
                }
            }
            Err(error) if recovery_started => {
                return Err(ImportCacheOpenError::Preserved(format!(
                    "timestamp recovery for the existing cache could not continue after {attempts} bounded open attempts; the existing cache was preserved and was not re-imported: {error}"
                )));
            }
            Err(error) if cache_open_error_is_reimportable(&error) => {
                return Err(ImportCacheOpenError::Reimportable);
            }
            Err(error) => {
                return Err(ImportCacheOpenError::Preserved(format!(
                    "the existing cache could not be opened safely; it was preserved and was not re-imported. Retry after resolving database access or contention: {error}"
                )));
            }
        }
    }
}

pub struct AppStateInner {
    pub db_path: PathBuf,
    pub columns: Vec<ColumnMeta>,
    pub generation: u64,
}

#[derive(Default)]
pub struct AppState {
    pub loaded: Mutex<Option<AppStateInner>>,
    pub llm: Arc<Mutex<Option<llm_parser::LlmParser>>>,
    pub semantic: Arc<Mutex<Option<Arc<semantic::SemanticModel>>>>,
    semantic_cancel: Mutex<Option<Arc<AtomicBool>>>,
    next_generation: AtomicU64,
    /// Guards against overlapping `import_sheet` calls (e.g. a double-clicked "Load Sheet"
    /// button, or opening a second file while the first is still importing). Without this,
    /// concurrent imports can race on the same cache file and on which result last wins the
    /// `loaded` slot — see AGENT_NOTES.md 2026-07-08 for the QA pass that found this.
    busy: AtomicBool,
    /// Reject duplicate report IPC calls instead of allowing long-running workbook exports to
    /// race at publication.
    report_busy: Arc<AtomicBool>,
}

#[derive(Debug)]
struct ReportExportGuard {
    busy: Arc<AtomicBool>,
}

impl ReportExportGuard {
    fn acquire(busy: &Arc<AtomicBool>) -> Result<Self, String> {
        busy.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .map_err(|_| "another report export is already running".to_string())?;
        Ok(Self {
            busy: Arc::clone(busy),
        })
    }
}

impl Drop for ReportExportGuard {
    fn drop(&mut self) {
        self.busy.store(false, Ordering::SeqCst);
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ImportSummary {
    pub row_count: i64,
    pub columns: Vec<ColumnMeta>,
    pub cache_db_path: String,
    pub elapsed_ms: u128,
    pub from_cache: bool,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct ImportProgressPayload {
    rows_done: u64,
    rows_total: u64,
    phase: String,
}

#[derive(Deserialize, Clone, Copy)]
#[serde(rename_all = "camelCase")]
pub enum ExportFormat {
    Csv,
    Xlsx,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportSummary {
    pub row_count: i64,
    pub dest_path: String,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct ExportProgressPayload {
    rows_done: i64,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct IntelScanProgressPayload {
    rows_done: i64,
    rows_total: i64,
    phase: String,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct SemanticIndexProgressPayload {
    build_id: i64,
    rows_done: i64,
    rows_total: i64,
    documents_embedded: i64,
    mappings_written: i64,
    documents_skipped: i64,
    mappings_skipped: i64,
    cells_truncated: i64,
    columns_omitted: i64,
    chunks_omitted: i64,
    resumed_from_row: i64,
    phase: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticIndexStatus {
    pub ready: bool,
    pub rows_indexed: i64,
    pub documents_skipped: i64,
    pub mappings_skipped: i64,
    pub cells_truncated: i64,
    pub columns_omitted: i64,
    pub chunks_omitted: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SemanticPreviewOutcome {
    used: bool,
    code: &'static str,
    message: String,
    selection_id: Option<String>,
}

enum SemanticPreparation {
    Selection(semantic::SemanticSelectionSummary),
    Fallback(SemanticPreviewOutcome),
}

impl SemanticPreviewOutcome {
    fn fallback(code: &'static str, reason: impl AsRef<str>) -> Self {
        Self {
            used: false,
            code,
            message: format!(
                "Semantic matching was not used: {} Exact and structured search remains available.",
                compact_diagnostic(reason.as_ref())
            ),
            selection_id: None,
        }
    }

    fn fallback_for_selection(
        code: &'static str,
        reason: impl AsRef<str>,
        selection_id: String,
    ) -> Self {
        let mut outcome = Self::fallback(code, reason);
        outcome.selection_id = Some(selection_id);
        outcome
    }

    fn applied(selection: &semantic::SemanticSelectionSummary) -> Self {
        Self {
            used: true,
            code: "applied",
            message: format!(
                "Semantic matching was used: the trusted selection retained {} document(s) and expands to {} raw row(s).",
                selection.documents_retained, selection.rows_matched
            ),
            selection_id: Some(selection.selection_id.clone()),
        }
    }
}

fn compact_diagnostic(value: &str) -> String {
    const MAX_DIAGNOSTIC_CHARS: usize = 1_024;
    let normalized = value.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut characters = normalized.chars();
    let mut compact = characters
        .by_ref()
        .take(MAX_DIAGNOSTIC_CHARS - 3)
        .collect::<String>();
    if characters.next().is_some() {
        compact.push_str("...");
    }
    if !compact
        .chars()
        .last()
        .is_some_and(|character| matches!(character, '.' | '!' | '?'))
    {
        compact.push('.');
    }
    compact
}

fn semantic_selection_id_for_preparation(preparation: &SemanticPreparation) -> Option<&str> {
    match preparation {
        SemanticPreparation::Selection(selection) if selection.documents_retained > 0 => {
            Some(selection.selection_id.as_str())
        }
        SemanticPreparation::Selection(_) | SemanticPreparation::Fallback(_) => None,
    }
}

fn prevalidate_semantic_preparation(
    preparation: SemanticPreparation,
    validate: impl FnOnce(&str) -> Result<(), String>,
) -> SemanticPreparation {
    match preparation {
        SemanticPreparation::Selection(selection) if selection.documents_retained > 0 => {
            match validate(&selection.selection_id) {
                Ok(()) => SemanticPreparation::Selection(selection),
                Err(error) => SemanticPreparation::Fallback(
                    SemanticPreviewOutcome::fallback_for_selection(
                        "selection_application_failed",
                        format!(
                            "the trusted semantic selection could not be attached to the validated preview ({error})"
                        ),
                        selection.selection_id,
                    ),
                ),
            }
        }
        other => other,
    }
}

/// The local model is invoked exactly once. In particular, model, grounding, database, or
/// validation errors must not be reinterpreted by a second potentially different inference.
fn plan_with_prevalidated_semantic_selection<T>(
    selection_id: Option<&str>,
    build: impl FnOnce(Option<&str>) -> Result<T, String>,
) -> Result<T, String> {
    build(selection_id)
}

fn expression_uses_semantic_selection(expression: &QueryExpression, selection_id: &str) -> bool {
    match expression {
        QueryExpression::And { children } | QueryExpression::Or { children } => children
            .iter()
            .any(|child| expression_uses_semantic_selection(child, selection_id)),
        QueryExpression::Not { child } => expression_uses_semantic_selection(child, selection_id),
        QueryExpression::SemanticSelection {
            selection_id: candidate,
        } => candidate == selection_id,
        QueryExpression::Search { .. }
        | QueryExpression::Predicate { .. }
        | QueryExpression::RowIds { .. } => false,
    }
}

fn preview_uses_semantic_selection(
    preview: &guided_parser::GuidedQueryPreview,
    selection_id: &str,
) -> bool {
    preview
        .query_spec
        .as_ref()
        .and_then(|spec| spec.expression.as_ref())
        .is_some_and(|expression| expression_uses_semantic_selection(expression, selection_id))
}

fn prepare_semantic_selection(
    conn: &mut rusqlite::Connection,
    columns: &[ColumnMeta],
    query_text: &str,
    semantic_model: &Arc<Mutex<Option<Arc<semantic::SemanticModel>>>>,
    semantic_paths: &Result<(PathBuf, PathBuf, PathBuf), String>,
) -> SemanticPreparation {
    match semantic::semantic_index_ready(conn, columns) {
        Ok(false) => {
            return SemanticPreparation::Fallback(SemanticPreviewOutcome::fallback(
                "index_not_ready",
                "the semantic index is not ready yet; preparation may still be running. Preview again after semantic matching reports ready",
            ));
        }
        Err(error) => {
            return SemanticPreparation::Fallback(SemanticPreviewOutcome::fallback(
                "index_validation_failed",
                format!(
                    "the semantic index could not be validated because of a database or index-integrity error ({error})"
                ),
            ));
        }
        Ok(true) => {}
    }

    let (model_path, tokenizer_path, config_path) = match semantic_paths {
        Ok(paths) => paths,
        Err(error) => {
            return SemanticPreparation::Fallback(SemanticPreviewOutcome::fallback(
                "resource_unavailable",
                format!("required local semantic resources are unavailable ({error})"),
            ));
        }
    };
    let model = {
        let mut guard = match semantic_model.lock() {
            Ok(guard) => guard,
            Err(_) => {
                return SemanticPreparation::Fallback(SemanticPreviewOutcome::fallback(
                    "model_lock_failed",
                    "the in-memory semantic model lock is unavailable; restart the application before relying on semantic matching",
                ));
            }
        };
        if guard.is_none() {
            match semantic::SemanticModel::load(model_path, tokenizer_path, config_path) {
                Ok(model) => *guard = Some(Arc::new(model)),
                Err(error) => {
                    return SemanticPreparation::Fallback(SemanticPreviewOutcome::fallback(
                        "model_load_failed",
                        format!("the local semantic model could not be loaded ({error})"),
                    ));
                }
            }
        }
        match guard.as_ref().cloned() {
            Some(model) => model,
            None => {
                return SemanticPreparation::Fallback(SemanticPreviewOutcome::fallback(
                    "model_initialization_failed",
                    "the local semantic model did not remain initialized",
                ));
            }
        }
    };

    match semantic::create_semantic_selection(
        conn,
        columns,
        model.as_ref(),
        query_text,
        semantic::SemanticSearchPolicy::default(),
    ) {
        Ok(selection) => SemanticPreparation::Selection(selection),
        Err(error) => SemanticPreparation::Fallback(SemanticPreviewOutcome::fallback(
            "selection_failed",
            format!("semantic candidate ranking or the database-backed selection failed ({error})"),
        )),
    }
}

fn record_semantic_preview_outcome(
    conn: &rusqlite::Connection,
    columns: &[ColumnMeta],
    query_text: &str,
    llm_audit_id: Option<i64>,
    outcome: &SemanticPreviewOutcome,
) -> Result<(), String> {
    let identity = llm_parser::dataset_identity(conn, columns)
        .map_err(|error| format!("binding semantic retrieval audit to the dataset: {error}"))?;
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS _semantic_retrieval_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            llm_audit_id INTEGER,
            input_sha256 TEXT NOT NULL,
            dataset_schema_sha256 TEXT NOT NULL,
            dataset_import_sha256 TEXT NOT NULL,
            semantic_used INTEGER NOT NULL CHECK (semantic_used IN (0, 1)),
            outcome_code TEXT NOT NULL,
            detail TEXT NOT NULL,
            selection_id TEXT,
            created_at TEXT NOT NULL
         );
         CREATE INDEX IF NOT EXISTS _semantic_retrieval_audit_llm
            ON _semantic_retrieval_audit(llm_audit_id);",
    )
    .map_err(|error| format!("creating semantic retrieval audit table: {error}"))?;
    conn.execute(
        "INSERT INTO _semantic_retrieval_audit (
            llm_audit_id, input_sha256, dataset_schema_sha256, dataset_import_sha256,
            semantic_used, outcome_code, detail, selection_id, created_at
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        rusqlite::params![
            llm_audit_id,
            llm_parser::sha256_text(query_text.trim()),
            identity.schema_sha256,
            identity.import_sha256,
            if outcome.used { 1_i64 } else { 0_i64 },
            outcome.code,
            outcome.message,
            outcome.selection_id,
            chrono::Utc::now().to_rfc3339(),
        ],
    )
    .map_err(|error| format!("recording semantic retrieval outcome: {error}"))?;
    Ok(())
}

fn keep_primary_result_after_best_effort<T>(
    primary: Result<T, String>,
    best_effort: impl FnOnce() -> anyhow::Result<()>,
) -> Result<T, String> {
    let value = primary?;
    let _ = best_effort();
    Ok(value)
}

fn accept_and_advance_semantic_archive(
    conn: &mut rusqlite::Connection,
    audit_id: i64,
    intent_token: &str,
) -> Result<(), String> {
    let accepted = guided_parser::accept_llm_audit(conn, audit_id, intent_token)
        .map_err(|error| error.to_string());
    keep_primary_result_after_best_effort(accepted, || {
        semantic::archive_required_semantic_audits_slice(conn).map(|_| ())
    })
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct ReportExportProgressPayload {
    request_id: u64,
    rows_done: i64,
    sheet: String,
}

fn now_marker() -> String {
    chrono::Utc::now().to_rfc3339()
}

fn cancel_semantic_index_build(state: &AppState) {
    if let Ok(mut current) = state.semantic_cancel.lock() {
        if let Some(cancelled) = current.take() {
            cancelled.store(true, Ordering::SeqCst);
        }
    }
}

fn clear_semantic_cancellation_if_current(
    current: &Mutex<Option<Arc<AtomicBool>>>,
    completed: &Arc<AtomicBool>,
) -> Result<(), String> {
    let mut current = current
        .lock()
        .map_err(|_| "semantic cancellation lock poisoned".to_string())?;
    if current
        .as_ref()
        .is_some_and(|active| Arc::ptr_eq(active, completed))
    {
        current.take();
    }
    Ok(())
}

fn finish_semantic_task<T>(
    task_result: Result<T, String>,
    cleanup_result: Result<(), String>,
) -> Result<T, String> {
    match task_result {
        Ok(result) => {
            cleanup_result?;
            Ok(result)
        }
        Err(error) => {
            let _ = cleanup_result;
            Err(error)
        }
    }
}

fn state_snapshot(state: &State<'_, AppState>) -> Result<(PathBuf, Vec<ColumnMeta>, u64), String> {
    let guard = state
        .loaded
        .lock()
        .map_err(|_| "app state lock poisoned".to_string())?;
    let inner = guard
        .as_ref()
        .ok_or_else(|| "no file loaded — call import_sheet first".to_string())?;
    Ok((
        inner.db_path.clone(),
        inner.columns.clone(),
        inner.generation,
    ))
}

fn loaded_generation_is_current(
    state: &AppState,
    expected_db_path: &Path,
    expected_generation: u64,
) -> Result<bool, String> {
    let guard = state
        .loaded
        .lock()
        .map_err(|_| "app state lock poisoned".to_string())?;
    Ok(guard.as_ref().is_some_and(|inner| {
        inner.generation == expected_generation && inner.db_path == expected_db_path
    }))
}

fn publish_export_if_current(
    app: &AppHandle,
    expected_db_path: &Path,
    expected_generation: u64,
    temporary_path: &Path,
    destination_path: &Path,
) -> anyhow::Result<()> {
    let state = app.state::<AppState>();
    publish_export_for_state_if_current(
        &state,
        expected_db_path,
        expected_generation,
        temporary_path,
        destination_path,
    )
}

fn publish_export_for_state_if_current(
    state: &AppState,
    expected_db_path: &Path,
    expected_generation: u64,
    temporary_path: &Path,
    destination_path: &Path,
) -> anyhow::Result<()> {
    let guard = state
        .loaded
        .lock()
        .map_err(|_| anyhow::anyhow!("app state lock poisoned"))?;
    let still_current = guard.as_ref().is_some_and(|inner| {
        inner.generation == expected_generation && inner.db_path == expected_db_path
    });
    if !still_current {
        anyhow::bail!("the loaded file or sheet changed while the export was running");
    }
    // Keep the state lock through the atomic replace. An import cannot publish a new generation
    // between this check and publication of the completed export.
    export::publish_completed_export(temporary_path, destination_path)
}

#[tauri::command]
pub fn list_sheets(path: String) -> Result<Vec<String>, String> {
    tabular_import::list_sheet_names(std::path::Path::new(&path)).map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn import_sheet(
    app: AppHandle,
    state: State<'_, AppState>,
    path: String,
    sheet: String,
) -> Result<ImportSummary, String> {
    if state
        .busy
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return Err(
            "Another file is already being imported — please wait for it to finish.".to_string(),
        );
    }
    cancel_semantic_index_build(&state);
    let generation = state.next_generation.fetch_add(1, Ordering::SeqCst) + 1;
    {
        let mut guard = state
            .loaded
            .lock()
            .map_err(|_| "app state lock poisoned".to_string())?;
        *guard = None;
    }
    let result = import_sheet_locked(&app, &state, path, sheet, generation).await;
    state.busy.store(false, Ordering::SeqCst);
    result
}

/// Does the actual work of `import_sheet`, once the `busy` guard is held. The cache file at
/// `db_path` is only ever written by renaming a fully-built temp file into place — a crash, a
/// disk-full error, or any other failure partway through `tabular_import::import_into_db` leaves
/// only the `.tmp` file behind, never a broken `db_path`. On a cache hit, the recorded
/// `sheet_name` is checked against the requested sheet before trusting it, as defense in depth
/// against the (now hash-prevented, but still worth guarding) case of two different sheets
/// resolving to the same cache path.
async fn import_sheet_locked(
    app: &AppHandle,
    state: &State<'_, AppState>,
    path: String,
    sheet: String,
    generation: u64,
) -> Result<ImportSummary, String> {
    let start = std::time::Instant::now();
    let source_path = PathBuf::from(&path);

    let db_path = db::cache_db_path(&source_path, &sheet).map_err(|e| e.to_string())?;

    let app_for_task = app.clone();
    let path_for_task = source_path.clone();
    let sheet_for_task = sheet.clone();
    let db_path_for_task = db_path.clone();

    let (columns, row_count, from_cache) = tauri::async_runtime::spawn_blocking(
        move || -> Result<(Vec<ColumnMeta>, i64, bool), String> {
            if db_path_for_task.exists() {
                match open_existing_cache_for_import(&db_path_for_task) {
                    Ok(conn) => match load_existing_cache_metadata_for_import(&conn) {
                        Ok((columns, info)) if info.sheet_name == sheet_for_task => {
                            return Ok((columns, info.row_count, true));
                        }
                        Ok(_) | Err(ImportCacheOpenError::Reimportable) => {}
                        Err(ImportCacheOpenError::Preserved(message)) => return Err(message),
                    },
                    Err(ImportCacheOpenError::Preserved(message)) => {
                        return Err(message);
                    }
                    Err(ImportCacheOpenError::Reimportable) => {}
                }
                // Cache file exists but isn't usable (sheet-name mismatch, or corrupt/partial
                // leftovers) — fall through and rebuild it below rather than failing outright.
            }

            let tmp_db_path = PathBuf::from(format!("{}.tmp", db_path_for_task.display()));
            let _ = std::fs::remove_file(&tmp_db_path);

            let import_result = tabular_import::import_into_db(
                &path_for_task,
                &sheet_for_task,
                &tmp_db_path,
                |done, total| {
                    let _ = app_for_task.emit(
                        "import-progress",
                        ImportProgressPayload {
                            rows_done: done,
                            rows_total: total,
                            phase: "reading".to_string(),
                        },
                    );
                },
            );
            let import_result = match import_result {
                Ok(result) => result,
                Err(err) => {
                    let _ = std::fs::remove_file(&tmp_db_path);
                    return Err(err.to_string());
                }
            };

            let _ = app_for_task.emit(
                "import-progress",
                ImportProgressPayload {
                    rows_done: import_result.row_count as u64,
                    rows_total: import_result.row_count as u64,
                    phase: "indexing".to_string(),
                },
            );

            let record_result = db::open(&tmp_db_path).and_then(|conn| {
                db::record_import_info(
                    &conn,
                    &ImportInfo {
                        source_path: path_for_task.display().to_string(),
                        sheet_name: sheet_for_task,
                        row_count: import_result.row_count,
                        imported_at: now_marker(),
                    },
                )
            });
            if let Err(err) = record_result {
                let _ = std::fs::remove_file(&tmp_db_path);
                return Err(err.to_string());
            }

            // Publish atomically: only a fully-imported, fully-recorded DB ever lands at
            // db_path. std::fs::rename fails on Windows if the target exists, so clear any
            // stale leftover first (only reachable via the mismatch/corruption fallback above).
            let _ = std::fs::remove_file(&db_path_for_task);
            std::fs::rename(&tmp_db_path, &db_path_for_task).map_err(|e| e.to_string())?;

            Ok((import_result.columns, import_result.row_count, false))
        },
    )
    .await
    .map_err(|e| format!("import task join error: {e}"))??;

    {
        if state.next_generation.load(Ordering::SeqCst) != generation {
            return Err(
                "the file import was canceled because the loaded-file state changed".into(),
            );
        }
        let mut guard = state
            .loaded
            .lock()
            .map_err(|_| "app state lock poisoned".to_string())?;
        *guard = Some(AppStateInner {
            db_path: db_path.clone(),
            columns: columns.clone(),
            generation,
        });
    }

    Ok(ImportSummary {
        row_count,
        columns,
        cache_db_path: db_path.display().to_string(),
        elapsed_ms: start.elapsed().as_millis(),
        from_cache,
    })
}

#[tauri::command]
pub fn query_rows(state: State<'_, AppState>, spec: QuerySpec) -> Result<QueryPage, String> {
    let (db_path, columns, _) = state_snapshot(&state)?;
    let conn = db::open(&db_path).map_err(|e| e.to_string())?;
    query::query_rows(&conn, &columns, &spec).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn count_rows(state: State<'_, AppState>, spec: QuerySpec) -> Result<i64, String> {
    let (db_path, columns, _) = state_snapshot(&state)?;
    let conn = db::open(&db_path).map_err(|e| e.to_string())?;
    query::count_rows(&conn, &columns, &spec).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn semantic_index_status(state: State<'_, AppState>) -> Result<SemanticIndexStatus, String> {
    let (db_path, columns, _) = state_snapshot(&state)?;
    let conn = db::open(&db_path).map_err(|error| error.to_string())?;
    let ready =
        semantic::semantic_index_ready(&conn, &columns).map_err(|error| error.to_string())?;
    let rows_indexed =
        semantic::semantic_indexed_rows(&conn, &columns).map_err(|error| error.to_string())?;
    let coverage = semantic::semantic_index_coverage(&conn, &columns)
        .map_err(|error| error.to_string())?
        .unwrap_or_default();
    Ok(SemanticIndexStatus {
        ready,
        rows_indexed,
        documents_skipped: coverage.documents_skipped,
        mappings_skipped: coverage.mappings_skipped,
        cells_truncated: coverage.cells_truncated,
        columns_omitted: coverage.columns_omitted,
        chunks_omitted: coverage.chunks_omitted,
    })
}

#[tauri::command]
pub async fn build_semantic_index(
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<semantic::SemanticIndexSummary, String> {
    let (db_path, columns, generation) = state_snapshot(&state)?;
    let indexed_db_path = db_path.clone();
    let model_path = resolve_llm_resource(&app, semantic::MODEL_RESOURCE_PATH)?;
    let tokenizer_path = resolve_llm_resource(&app, semantic::TOKENIZER_RESOURCE_PATH)?;
    let config_path = resolve_llm_resource(&app, semantic::CONFIG_RESOURCE_PATH)?;
    let semantic_model = Arc::clone(&state.semantic);
    let cancellation = Arc::new(AtomicBool::new(false));
    {
        let mut current = state
            .semantic_cancel
            .lock()
            .map_err(|_| "semantic cancellation lock poisoned".to_string())?;
        if let Some(previous) = current.replace(Arc::clone(&cancellation)) {
            previous.store(true, Ordering::SeqCst);
        }
    }
    let app_for_task = app.clone();
    let task_cancellation = Arc::clone(&cancellation);
    let task_result = tauri::async_runtime::spawn_blocking(move || {
        let mut conn = db::open(&db_path).map_err(|error| error.to_string())?;
        let rows_total: i64 = conn
            .query_row("SELECT COUNT(*) FROM rows", [], |row| row.get(0))
            .map_err(|error| error.to_string())?;
        let _ = app_for_task.emit(
            "semantic-index-progress",
            SemanticIndexProgressPayload {
                build_id: 0,
                rows_done: 0,
                rows_total,
                documents_embedded: 0,
                mappings_written: 0,
                documents_skipped: 0,
                mappings_skipped: 0,
                cells_truncated: 0,
                columns_omitted: 0,
                chunks_omitted: 0,
                resumed_from_row: 0,
                phase: "loadingModel".to_string(),
            },
        );
        let model = {
            let mut guard = semantic_model
                .lock()
                .map_err(|_| "semantic model lock poisoned".to_string())?;
            if guard.is_none() {
                *guard = Some(Arc::new(
                    semantic::SemanticModel::load(&model_path, &tokenizer_path, &config_path)
                        .map_err(|error| error.to_string())?,
                ));
            }
            guard
                .as_ref()
                .cloned()
                .ok_or_else(|| "semantic model failed to initialize".to_string())?
        };
        let progress_app = app_for_task.clone();
        let summary = semantic::ensure_semantic_index_v2(
            &mut conn,
            &columns,
            model.as_ref(),
            || task_cancellation.load(Ordering::SeqCst),
            move |progress| {
                let _ = progress_app.emit(
                    "semantic-index-progress",
                    SemanticIndexProgressPayload {
                        build_id: progress.build_id,
                        rows_done: progress.rows_scanned,
                        rows_total: progress.rows_total,
                        documents_embedded: progress.documents_embedded,
                        mappings_written: progress.mappings_written,
                        documents_skipped: progress.documents_skipped,
                        mappings_skipped: progress.mappings_skipped,
                        cells_truncated: progress.cells_truncated,
                        columns_omitted: progress.columns_omitted,
                        chunks_omitted: progress.chunks_omitted,
                        resumed_from_row: progress.resumed_from_row,
                        phase: progress.phase,
                    },
                );
            },
        )
        .map_err(|error| error.to_string())?;
        Ok::<_, String>(summary)
    })
    .await
    .map_err(|error| format!("semantic index task join error: {error}"))
    .and_then(|result| result);

    // Always retire this request's cancellation handle, including worker errors and panics. If a
    // newer request replaced it while this one ran, pointer identity keeps the newer handle live.
    // A cleanup failure is secondary and therefore never hides the worker's primary error.
    let cleanup_result =
        clear_semantic_cancellation_if_current(&state.semantic_cancel, &cancellation);
    let result = finish_semantic_task(task_result, cleanup_result)?;

    let still_current = state
        .loaded
        .lock()
        .map_err(|_| "app state lock poisoned".to_string())?
        .as_ref()
        .is_some_and(|inner| inner.generation == generation && inner.db_path == indexed_db_path);
    if !still_current {
        return Err(
            "the loaded file or sheet changed while the semantic index was building".to_string(),
        );
    }
    Ok(result)
}

#[tauri::command]
pub async fn parse_guided_query(
    app: AppHandle,
    state: State<'_, AppState>,
    query_text: String,
) -> Result<guided_parser::GuidedQueryPreview, String> {
    let (db_path, columns, generation) = state_snapshot(&state)?;
    let parsed_db_path = db_path.clone();
    let model_path = resolve_llm_resource(&app, llm_parser::MODEL_RESOURCE_PATH)?;
    let tokenizer_path = resolve_llm_resource(&app, llm_parser::TOKENIZER_RESOURCE_PATH)?;
    let llm = Arc::clone(&state.llm);
    let semantic_model = Arc::clone(&state.semantic);
    let semantic_paths = (|| {
        Ok::<_, String>((
            resolve_llm_resource(&app, semantic::MODEL_RESOURCE_PATH)?,
            resolve_llm_resource(&app, semantic::TOKENIZER_RESOURCE_PATH)?,
            resolve_llm_resource(&app, semantic::CONFIG_RESOURCE_PATH)?,
        ))
    })();
    let preview = tauri::async_runtime::spawn_blocking(
        move || -> Result<guided_parser::GuidedQueryPreview, String> {
            let mut conn = db::open(&db_path).map_err(|e| e.to_string())?;
            // Semantic retrieval supplements the validated lexical plan. Every non-use path is
            // retained as a concrete preview note and a dataset-bound audit row below.
            let semantic_preparation = prepare_semantic_selection(
                &mut conn,
                &columns,
                &query_text,
                &semantic_model,
                &semantic_paths,
            );
            // Validate immediately before the one permitted model invocation. A stale or
            // otherwise unusable trusted selection degrades to the literal plan up front;
            // arbitrary planner/model/grounding errors are never retried.
            let semantic_preparation =
                prevalidate_semantic_preparation(semantic_preparation, |selection_id| {
                    semantic::validate_semantic_selection(&conn, &columns, selection_id)
                        .map_err(|error| error.to_string())
                });
            let semantic_selection_id =
                semantic_selection_id_for_preparation(&semantic_preparation);
            let mut guard = llm
                .lock()
                .map_err(|_| "local AI model lock poisoned".to_string())?;
            if guard.is_none() {
                *guard = Some(
                    llm_parser::LlmParser::load(&model_path, &tokenizer_path)
                        .map_err(|error| error.to_string())?,
                );
            }
            let model = guard
                .as_mut()
                .ok_or_else(|| "local AI model failed to initialize".to_string())?;
            let mut preview = plan_with_prevalidated_semantic_selection(
                semantic_selection_id,
                |selection_id| {
                    guided_parser::parse_guided_query_with_llm_and_semantic_selection(
                        &conn,
                        &columns,
                        &query_text,
                        model,
                        &[],
                        selection_id,
                    )
                    .map_err(|error| error.to_string())
                },
            )?;
            let outcome = match semantic_preparation {
                SemanticPreparation::Fallback(outcome) => outcome,
                SemanticPreparation::Selection(selection)
                    if selection.documents_retained == 0 =>
                {
                    SemanticPreviewOutcome::fallback_for_selection(
                        "no_candidates",
                        "no semantic document candidates met the bounded ranking policy",
                        selection.selection_id,
                    )
                }
                SemanticPreparation::Selection(selection)
                    if preview_uses_semantic_selection(&preview, &selection.selection_id) =>
                {
                    preview
                        .match_explanation
                        .extend(selection.warnings.iter().cloned());
                    SemanticPreviewOutcome::applied(&selection)
                }
                SemanticPreparation::Selection(selection) => {
                    SemanticPreviewOutcome::fallback_for_selection(
                        "selection_not_applied",
                        "semantic candidates were ranked, but the validated preview contains no trusted semantic selection",
                        selection.selection_id,
                    )
                }
            };
            preview.match_explanation.push(outcome.message.clone());
            record_semantic_preview_outcome(
                &conn,
                &columns,
                &query_text,
                preview.audit_id,
                &outcome,
            )?;
            Ok(preview)
        },
    )
    .await
    .map_err(|error| format!("local AI parse task join error: {error}"))??;

    let still_current = state
        .loaded
        .lock()
        .map_err(|_| "app state lock poisoned".to_string())?
        .as_ref()
        .is_some_and(|inner| inner.generation == generation && inner.db_path == parsed_db_path);
    if !still_current {
        if let Some(audit_id) = preview.audit_id {
            if let Ok(conn) = db::open(&parsed_db_path) {
                let _ = guided_parser::set_llm_audit_decision(
                    &conn,
                    audit_id,
                    &preview.intent_token,
                    guided_parser::ExaminerDecision::Edited,
                );
            }
        }
        return Err(
            "the loaded file or sheet changed while local AI was parsing; the stale preview was discarded"
                .to_string(),
        );
    }
    Ok(preview)
}

#[tauri::command]
pub fn accept_guided_query(
    state: State<'_, AppState>,
    intent_token: String,
    audit_id: i64,
) -> Result<(), String> {
    let (db_path, _, _) = state_snapshot(&state)?;
    let mut conn = db::open(&db_path).map_err(|error| error.to_string())?;
    accept_and_advance_semantic_archive(&mut conn, audit_id, &intent_token)
}

#[tauri::command]
pub fn run_guided_query(
    state: State<'_, AppState>,
    intent_token: String,
    audit_id: i64,
    cursor: Option<query::Cursor>,
    limit: Option<u32>,
) -> Result<QueryPage, String> {
    let (db_path, columns, _) = state_snapshot(&state)?;
    let mut conn = db::open(&db_path).map_err(|e| e.to_string())?;
    accept_and_advance_semantic_archive(&mut conn, audit_id, &intent_token)?;
    guided_query::run_guided_query(&conn, &columns, &intent_token, cursor, limit)
        .map_err(|error| error.to_string())
}

#[tauri::command]
pub fn set_guided_parse_decision(
    state: State<'_, AppState>,
    audit_id: i64,
    intent_token: String,
    decision: guided_parser::ExaminerDecision,
) -> Result<(), String> {
    let (db_path, _, _) = state_snapshot(&state)?;
    let conn = db::open(&db_path).map_err(|error| error.to_string())?;
    guided_parser::set_llm_audit_decision(&conn, audit_id, &intent_token, decision)
        .map_err(|error| error.to_string())
}

#[tauri::command]
pub fn clear_loaded_file(state: State<'_, AppState>) -> Result<(), String> {
    cancel_semantic_index_build(&state);
    let mut guard = state
        .loaded
        .lock()
        .map_err(|_| "app state lock poisoned".to_string())?;
    *guard = None;
    state.next_generation.fetch_add(1, Ordering::SeqCst);
    Ok(())
}

fn resolve_llm_resource(app: &AppHandle, relative_path: &str) -> Result<PathBuf, String> {
    let bundled = app
        .path()
        .resolve(relative_path, BaseDirectory::Resource)
        .map_err(|error| format!("resolving local AI resource: {error}"))?;
    if bundled.is_file() {
        return Ok(bundled);
    }
    let development = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("resources")
        .join(relative_path);
    if development.is_file() {
        return Ok(development);
    }
    Err(format!(
        "local AI resource is missing: {}. Install an AI-enabled build or fetch the pinned model resources before running in development.",
        bundled.display()
    ))
}

#[tauri::command]
pub fn detect_column_roles(
    state: State<'_, AppState>,
) -> Result<Vec<roles::ColumnRoleSuggestion>, String> {
    let (db_path, columns, _) = state_snapshot(&state)?;
    let conn = db::open(&db_path).map_err(|e| e.to_string())?;
    roles::detect_column_roles(&conn, &columns).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn set_column_role_status(
    state: State<'_, AppState>,
    role: String,
    sql_name: String,
    status: roles::RoleDecisionStatus,
) -> Result<roles::ColumnRoleSuggestion, String> {
    let (db_path, columns, _) = state_snapshot(&state)?;
    let conn = db::open(&db_path).map_err(|e| e.to_string())?;
    roles::set_column_role_status(&conn, &columns, &role, &sql_name, status)
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn analyze_timestamp_column(
    state: State<'_, AppState>,
) -> Result<time::TimestampAnalysis, String> {
    let (db_path, columns, _) = state_snapshot(&state)?;
    tauri::async_runtime::spawn_blocking(move || -> Result<time::TimestampAnalysis, String> {
        let conn = db::open(&db_path).map_err(|e| e.to_string())?;
        time::analyze_confirmed_timestamp_column(&conn, &columns).map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("timestamp analysis task join error: {e}"))?
}

#[tauri::command]
pub async fn normalize_timestamp_column(
    app: AppHandle,
    state: State<'_, AppState>,
    naive_timezone: Option<String>,
    date_convention: Option<String>,
) -> Result<time::TimestampNormalizationSummary, String> {
    let (db_path, columns, generation) = state_snapshot(&state)?;
    let normalized_db_path = db_path.clone();
    let task_db_path = normalized_db_path.clone();
    let app_for_task = app.clone();
    let result = tauri::async_runtime::spawn_blocking(
        move || -> Result<time::TimestampNormalizationSummary, String> {
            let mut conn = db::open(&db_path).map_err(|e| e.to_string())?;
            time::normalize_timestamp_column_with_options_guarded(
                &mut conn,
                &columns,
                naive_timezone.as_deref(),
                date_convention.as_deref(),
                || {
                    let current_state = app_for_task.state::<AppState>();
                    loaded_generation_is_current(&current_state, &task_db_path, generation)
                        .map_err(|error| anyhow::anyhow!(error))
                },
            )
            .map_err(|e| e.to_string())
        },
    )
    .await
    .map_err(|e| format!("timestamp normalization task join error: {e}"))??;
    if !loaded_generation_is_current(&state, &normalized_db_path, generation)? {
        return Err(
            "timestamp normalization was superseded because the loaded file or sheet changed"
                .to_string(),
        );
    }
    Ok(result)
}

#[tauri::command]
pub async fn export_data(
    app: AppHandle,
    state: State<'_, AppState>,
    spec: QuerySpec,
    format: ExportFormat,
    dest_path: String,
) -> Result<ExportSummary, String> {
    let (db_path, columns, generation) = state_snapshot(&state)?;
    let exported_db_path = db_path.clone();
    let dest = PathBuf::from(&dest_path);
    let dest_for_task = dest.clone();
    let app_for_progress = app.clone();
    let app_for_publish = app.clone();

    let row_count = tauri::async_runtime::spawn_blocking(move || -> Result<i64, String> {
        let conn = db::open(&db_path).map_err(|e| e.to_string())?;
        let on_progress = |rows_done: i64| {
            let _ = app_for_progress.emit("export-progress", ExportProgressPayload { rows_done });
        };
        let publish = |temporary_path: &Path, destination_path: &Path| {
            publish_export_if_current(
                &app_for_publish,
                &exported_db_path,
                generation,
                temporary_path,
                destination_path,
            )
        };
        let result = match format {
            ExportFormat::Csv => export::export_csv_guarded(
                &conn,
                &columns,
                &spec,
                &dest_for_task,
                on_progress,
                publish,
            ),
            ExportFormat::Xlsx => export::export_xlsx_guarded(
                &conn,
                &columns,
                &spec,
                &dest_for_task,
                on_progress,
                publish,
            ),
        }
        .map_err(|e| e.to_string())?;
        Ok(result.row_count)
    })
    .await
    .map_err(|e| format!("export task join error: {e}"))??;

    Ok(ExportSummary {
        row_count,
        dest_path: dest.display().to_string(),
    })
}

#[tauri::command]
pub async fn export_guided_data(
    app: AppHandle,
    state: State<'_, AppState>,
    intent_token: String,
    audit_id: i64,
    format: ExportFormat,
    dest_path: String,
) -> Result<ExportSummary, String> {
    let (db_path, columns, generation) = state_snapshot(&state)?;
    let exported_db_path = db_path.clone();
    let dest = PathBuf::from(&dest_path);
    let dest_for_task = dest.clone();
    let app_for_progress = app.clone();
    let app_for_publish = app.clone();
    let row_count = tauri::async_runtime::spawn_blocking(move || -> Result<i64, String> {
        let conn = db::open(&db_path).map_err(|error| error.to_string())?;
        guided_parser::accept_llm_audit(&conn, audit_id, &intent_token)
            .map_err(|error| error.to_string())?;
        let intent =
            guided_parser::intent_from_token(&intent_token).map_err(|error| error.to_string())?;
        if !matches!(
            intent,
            guided_parser::GuidedIntent::RawEvidenceSearch { .. }
        ) {
            return Err(
                "AI result export is available only for audited raw evidence searches".to_string(),
            );
        }
        let spec = guided_parser::query_spec_from_raw_intent(&intent, None, None)
            .map_err(|error| error.to_string())?;
        let normalized_sort = guided_query::normalized_raw_sort_direction(&conn, &columns, &intent)
            .map_err(|error| error.to_string())?;
        let on_progress = |rows_done: i64| {
            let _ = app_for_progress.emit("export-progress", ExportProgressPayload { rows_done });
        };
        let publish = |temporary_path: &Path, destination_path: &Path| {
            publish_export_if_current(
                &app_for_publish,
                &exported_db_path,
                generation,
                temporary_path,
                destination_path,
            )
        };
        let summary = match (format, normalized_sort) {
            (ExportFormat::Csv, Some((source_column, direction))) => {
                export::export_csv_normalized_time_guarded(
                    &conn,
                    &columns,
                    &spec,
                    &source_column,
                    direction,
                    &dest_for_task,
                    on_progress,
                    publish,
                )
            }
            (ExportFormat::Xlsx, Some((source_column, direction))) => {
                export::export_xlsx_normalized_time_guarded(
                    &conn,
                    &columns,
                    &spec,
                    &source_column,
                    direction,
                    &dest_for_task,
                    on_progress,
                    publish,
                )
            }
            (ExportFormat::Csv, None) => export::export_csv_guarded(
                &conn,
                &columns,
                &spec,
                &dest_for_task,
                on_progress,
                publish,
            ),
            (ExportFormat::Xlsx, None) => export::export_xlsx_guarded(
                &conn,
                &columns,
                &spec,
                &dest_for_task,
                on_progress,
                publish,
            ),
        }
        .map_err(|error| error.to_string())?;
        Ok(summary.row_count)
    })
    .await
    .map_err(|error| format!("AI result export task join error: {error}"))??;

    Ok(ExportSummary {
        row_count,
        dest_path: dest.display().to_string(),
    })
}

#[tauri::command]
pub async fn scan_intel_matches(
    app: AppHandle,
    state: State<'_, AppState>,
    evidence_columns: Vec<String>,
) -> Result<IntelScanSummary, String> {
    let (db_path, columns, _) = state_snapshot(&state)?;
    if evidence_columns.is_empty() {
        return Err("no evidence columns were provided".to_string());
    }

    for column in &evidence_columns {
        if !columns.iter().any(|meta| meta.sql_name == *column) {
            return Err(format!("unknown evidence column: {column}"));
        }
    }

    let app_for_task = app.clone();
    tauri::async_runtime::spawn_blocking(move || -> Result<IntelScanSummary, String> {
        let mut conn = db::open(&db_path).map_err(|e| e.to_string())?;
        // Suggested or confirmed (but never rejected) automatic mappings are sufficient for
        // optional MITRE enrichment. They no longer sit in front of raw AI retrieval.
        let mut requested_columns = evidence_columns.clone();
        requested_columns.sort();
        requested_columns.dedup();
        let active_columns = guided_query::active_evidence_columns(&conn)
            .map_err(|error| error.to_string())?;
        if requested_columns != active_columns {
            return Err(
                "evidence columns changed or are not active automatic mappings; refresh data mapping before scanning"
                    .to_string(),
            );
        }
        matcher::scan_connection(
            &mut conn,
            &evidence_columns,
            |rows_done, rows_total, phase| {
                let _ = app_for_task.emit(
                    "intel-scan-progress",
                    IntelScanProgressPayload {
                        rows_done,
                        rows_total,
                        phase: phase.to_string(),
                    },
                );
            },
        )
        .map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("intel scan task join error: {e}"))?
}

#[tauri::command]
pub async fn export_report(
    app: AppHandle,
    state: State<'_, AppState>,
    dest_path: String,
    request_id: u64,
) -> Result<ReportExportSummary, String> {
    let report_guard = ReportExportGuard::acquire(&state.report_busy)?;
    let (db_path, columns, generation) = state_snapshot(&state)?;
    let exported_db_path = db_path.clone();
    let dest = PathBuf::from(&dest_path);
    let dest_for_task = dest.clone();
    let app_for_progress = app.clone();
    let app_for_publish = app.clone();

    tauri::async_runtime::spawn_blocking(move || -> Result<ReportExportSummary, String> {
        let _report_guard = report_guard;
        let mut conn = db::open(&db_path).map_err(|e| e.to_string())?;
        let publish = |temporary_path: &Path, destination_path: &Path| {
            publish_export_if_current(
                &app_for_publish,
                &exported_db_path,
                generation,
                temporary_path,
                destination_path,
            )
        };
        report::export_report_guarded(
            &mut conn,
            &columns,
            &dest_for_task,
            |rows_done, sheet| {
                let _ = app_for_progress.emit(
                    "report-export-progress",
                    ReportExportProgressPayload {
                        request_id,
                        rows_done,
                        sheet: sheet.to_string(),
                    },
                );
            },
            publish,
        )
        .map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("report export task join error: {e}"))?
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;

    const SELECTION_ID: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn test_columns() -> Vec<ColumnMeta> {
        vec![ColumnMeta {
            sql_name: "description".to_string(),
            original_name: "Description".to_string(),
            col_index: 0,
            inferred_type: "text".to_string(),
        }]
    }

    fn import_cache_test_path(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "log-parser-{label}-{}-{}.sqlite3",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ))
    }

    fn create_existing_cache_with_abandoned_stage(path: &Path, stage_rows: i64) {
        let columns = test_columns();
        let mut conn = Connection::open(path).unwrap();
        db::create_schema(&conn, &columns).unwrap();
        conn.execute(
            "INSERT INTO rows (row_num, description) VALUES (1, 'preserved evidence')",
            [],
        )
        .unwrap();
        db::record_import_info(
            &conn,
            &ImportInfo {
                source_path: "preserved.xlsx".to_string(),
                sheet_name: "Evidence".to_string(),
                row_count: 1,
                imported_at: "2026-07-17T00:00:00Z".to_string(),
            },
        )
        .unwrap();
        conn.execute_batch(
            "CREATE TABLE _column_roles (marker TEXT NOT NULL);
             INSERT INTO _column_roles(marker) VALUES ('role-marker');
             CREATE TABLE _llm_parse_audit (marker TEXT NOT NULL);
             INSERT INTO _llm_parse_audit(marker) VALUES ('audit-marker');
             CREATE TABLE _semantic_v2_active (marker TEXT NOT NULL);
             INSERT INTO _semantic_v2_active(marker) VALUES ('semantic-marker');
             CREATE TABLE _row_time_stage_interrupted (
                row_num INTEGER PRIMARY KEY,
                epoch_ms INTEGER NOT NULL,
                utc_text TEXT NOT NULL,
                source_text TEXT NOT NULL,
                parse_status TEXT NOT NULL
             );",
        )
        .unwrap();
        let tx = conn.transaction().unwrap();
        {
            let mut insert = tx
                .prepare(
                    "INSERT INTO _row_time_stage_interrupted (
                        row_num, epoch_ms, utc_text, source_text, parse_status
                     ) VALUES (?1, ?1, 'x', 'x', 'test')",
                )
                .unwrap();
            for row_num in 1..=stage_rows {
                insert.execute([row_num]).unwrap();
            }
        }
        tx.commit().unwrap();
    }

    fn marker(conn: &Connection, table: &str) -> String {
        conn.query_row(&format!("SELECT marker FROM {table}"), [], |row| row.get(0))
            .unwrap()
    }

    #[test]
    fn cache_open_retries_timestamp_recovery_without_replacing_saved_state() {
        let path = import_cache_test_path("import-cache-recovery");
        create_existing_cache_with_abandoned_stage(&path, 32_769);

        let conn = open_existing_cache_for_import(&path)
            .expect("normal cache loading must drive bounded timestamp recovery to completion");
        let (columns, info) = load_existing_cache_metadata_for_import(&conn).unwrap();
        assert_eq!(columns.len(), 1);
        assert_eq!(columns[0].sql_name, "description");
        assert_eq!(columns[0].original_name, "Description");
        assert_eq!(info.sheet_name, "Evidence");
        assert_eq!(marker(&conn, "_column_roles"), "role-marker");
        assert_eq!(marker(&conn, "_llm_parse_audit"), "audit-marker");
        assert_eq!(marker(&conn, "_semantic_v2_active"), "semantic-marker");
        assert_eq!(
            conn.query_row(
                "SELECT EXISTS(
                    SELECT 1 FROM sqlite_master
                    WHERE type = 'table' AND name = '_row_time_stage_interrupted'
                 )",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
            0
        );
        drop(conn);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn exhausted_timestamp_recovery_returns_preservation_error_without_reimport() {
        let path = import_cache_test_path("import-cache-recovery-limit");
        create_existing_cache_with_abandoned_stage(&path, 32_769);

        let error =
            match open_existing_cache_for_import_with_limits(&path, 1, Duration::from_secs(60)) {
                Err(ImportCacheOpenError::Preserved(message)) => message,
                Err(other) => panic!("unexpected cache-open classification: {other:?}"),
                Ok(_) => panic!("one bounded pass must leave an explicit recovery backlog"),
            };
        assert!(error.contains("existing cache was preserved"));
        assert!(error.contains("was not re-imported"));

        let raw = Connection::open(&path).unwrap();
        assert_eq!(marker(&raw, "_column_roles"), "role-marker");
        assert_eq!(marker(&raw, "_llm_parse_audit"), "audit-marker");
        assert_eq!(marker(&raw, "_semantic_v2_active"), "semantic-marker");
        assert_eq!(
            raw.query_row(
                "SELECT COUNT(*) FROM _row_time_stage_interrupted",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
            1
        );
        drop(raw);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn cache_error_policy_preserves_contention_and_reimports_only_corruption() {
        let busy = rusqlite::Error::SqliteFailure(
            rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_BUSY),
            Some("database is busy".to_string()),
        );
        assert!(!cache_open_error_is_reimportable(&busy));
        assert!(!cache_metadata_error_is_reimportable(&busy, "_meta"));

        let corrupt = rusqlite::Error::SqliteFailure(
            rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_CORRUPT),
            Some("database disk image is malformed".to_string()),
        );
        assert!(cache_open_error_is_reimportable(&corrupt));
        assert!(cache_metadata_error_is_reimportable(&corrupt, "_meta"));
    }

    #[test]
    fn metadata_read_contention_returns_preservation_error_and_keeps_markers() {
        let path = import_cache_test_path("import-cache-metadata-busy");
        create_existing_cache_with_abandoned_stage(&path, 0);
        let reader = Connection::open(&path).unwrap();
        reader.busy_timeout(Duration::from_millis(1)).unwrap();
        let writer = Connection::open(&path).unwrap();
        writer.execute_batch("BEGIN EXCLUSIVE").unwrap();

        match load_existing_cache_metadata_for_import(&reader) {
            Err(ImportCacheOpenError::Preserved(message)) => {
                assert!(message.contains("preserved"));
                assert!(message.contains("not re-imported"));
            }
            other => panic!("metadata contention must preserve the cache: {other:?}"),
        }
        writer.execute_batch("ROLLBACK").unwrap();
        assert_eq!(marker(&reader, "_column_roles"), "role-marker");
        assert_eq!(marker(&reader, "_llm_parse_audit"), "audit-marker");
        assert_eq!(marker(&reader, "_semantic_v2_active"), "semantic-marker");
        drop(writer);
        drop(reader);
        let _ = std::fs::remove_file(path);
    }

    fn selection(documents_retained: usize) -> semantic::SemanticSelectionSummary {
        semantic::SemanticSelectionSummary {
            selection_id: SELECTION_ID.to_string(),
            documents_above_threshold: documents_retained,
            documents_retained,
            rows_matched: documents_retained as i64,
            documents_truncated: false,
            index_documents_skipped: 0,
            index_mappings_skipped: 0,
            index_cells_truncated: 0,
            index_columns_omitted: 0,
            index_chunks_omitted: 0,
            broad_row_warning: false,
            warnings: Vec::new(),
        }
    }

    #[test]
    fn stale_semantic_selection_degrades_before_the_single_planner_attempt() {
        let mut validations = 0;
        let preparation =
            prevalidate_semantic_preparation(SemanticPreparation::Selection(selection(2)), |_| {
                validations += 1;
                Err("selection belongs to a superseded build".to_string())
            });
        assert_eq!(validations, 1);
        assert_eq!(semantic_selection_id_for_preparation(&preparation), None);
        let SemanticPreparation::Fallback(fallback) = &preparation else {
            panic!("a rejected semantic selection must become an explicit fallback");
        };
        assert!(!fallback.used);
        assert_eq!(fallback.code, "selection_application_failed");
        assert_eq!(fallback.selection_id.as_deref(), Some(SELECTION_ID));
        assert!(fallback
            .message
            .contains("selection belongs to a superseded build"));
        assert!(fallback
            .message
            .contains("Exact and structured search remains available"));

        let mut attempts = 0;
        let preview = plan_with_prevalidated_semantic_selection(
            semantic_selection_id_for_preparation(&preparation),
            |candidate| {
                attempts += 1;
                assert_eq!(candidate, None);
                Ok("literal plan")
            },
        )
        .unwrap();
        assert_eq!(preview, "literal plan");
        assert_eq!(attempts, 1);
    }

    #[test]
    fn non_selection_planner_error_is_attempted_once_and_not_relabelled() {
        let preparation =
            prevalidate_semantic_preparation(SemanticPreparation::Selection(selection(2)), |_| {
                Ok(())
            });
        let mut attempts = 0;
        let error = plan_with_prevalidated_semantic_selection::<()>(
            semantic_selection_id_for_preparation(&preparation),
            |candidate| {
                attempts += 1;
                assert_eq!(candidate, Some(SELECTION_ID));
                Err("grounding rejected the model plan".to_string())
            },
        )
        .unwrap_err();

        assert_eq!(attempts, 1);
        assert_eq!(error, "grounding rejected the model plan");
    }

    #[test]
    fn only_a_retained_and_exact_selection_id_is_treated_as_applied() {
        let preparation = SemanticPreparation::Selection(selection(2));
        assert_eq!(
            semantic_selection_id_for_preparation(&preparation),
            Some(SELECTION_ID)
        );

        let expression = QueryExpression::And {
            children: vec![
                QueryExpression::Search {
                    value: "lsass".to_string(),
                },
                QueryExpression::Or {
                    children: vec![QueryExpression::SemanticSelection {
                        selection_id: SELECTION_ID.to_string(),
                    }],
                },
            ],
        };
        assert!(expression_uses_semantic_selection(
            &expression,
            SELECTION_ID
        ));
        assert!(!expression_uses_semantic_selection(
            &expression,
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        ));

        let empty = SemanticPreparation::Selection(selection(0));
        assert_eq!(semantic_selection_id_for_preparation(&empty), None);
        let fallback = SemanticPreparation::Fallback(SemanticPreviewOutcome::fallback(
            "index_not_ready",
            "the semantic index is still building",
        ));
        assert_eq!(semantic_selection_id_for_preparation(&fallback), None);
    }

    #[test]
    fn semantic_non_use_audit_preserves_specific_reason_and_dataset_binding() {
        let conn = Connection::open_in_memory().unwrap();
        let columns = test_columns();
        db::create_schema(&conn, &columns).unwrap();
        conn.execute(
            "INSERT INTO rows (row_num, description) VALUES (1, 'exact evidence')",
            [],
        )
        .unwrap();
        db::record_import_info(
            &conn,
            &ImportInfo {
                source_path: "audit-test.xlsx".to_string(),
                sheet_name: "Evidence".to_string(),
                row_count: 1,
                imported_at: "2026-07-17T00:00:00Z".to_string(),
            },
        )
        .unwrap();
        let identity = llm_parser::dataset_identity(&conn, &columns).unwrap();
        let query = "  find credential access  ";
        let outcome = SemanticPreviewOutcome::fallback_for_selection(
            "selection_failed",
            "ranking failed because the active build changed",
            SELECTION_ID.to_string(),
        );

        record_semantic_preview_outcome(&conn, &columns, query, Some(42), &outcome).unwrap();

        let stored: (
            i64,
            String,
            String,
            String,
            i64,
            String,
            String,
            Option<String>,
        ) = conn
            .query_row(
                "SELECT llm_audit_id, input_sha256, dataset_schema_sha256,
                        dataset_import_sha256, semantic_used, outcome_code, detail, selection_id
                 FROM _semantic_retrieval_audit",
                [],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                        row.get(5)?,
                        row.get(6)?,
                        row.get(7)?,
                    ))
                },
            )
            .unwrap();
        assert_eq!(stored.0, 42);
        assert_eq!(stored.1, llm_parser::sha256_text(query.trim()));
        assert_eq!(stored.2, identity.schema_sha256);
        assert_eq!(stored.3, identity.import_sha256);
        assert_eq!(stored.4, 0);
        assert_eq!(stored.5, "selection_failed");
        assert_eq!(stored.6, outcome.message);
        assert_eq!(stored.7.as_deref(), Some(SELECTION_ID));
    }

    #[test]
    fn diagnostics_are_whitespace_compacted_and_unicode_safe() {
        let reason = format!("model   error\n{}", "é".repeat(2_000));
        let compact = compact_diagnostic(&reason);
        assert!(compact.starts_with("model error é"));
        assert!(compact.ends_with("..."));
        assert!(compact.chars().count() <= 1_024);
    }

    #[test]
    fn cancellation_cleanup_keeps_newer_request_and_primary_error() {
        let completed = Arc::new(AtomicBool::new(false));
        let newer = Arc::new(AtomicBool::new(false));
        let current = Mutex::new(Some(Arc::clone(&newer)));

        clear_semantic_cancellation_if_current(&current, &completed).unwrap();
        assert!(current
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(|active| Arc::ptr_eq(active, &newer)));

        *current.lock().unwrap() = Some(Arc::clone(&completed));
        clear_semantic_cancellation_if_current(&current, &completed).unwrap();
        assert!(current.lock().unwrap().is_none());

        let result = finish_semantic_task::<()>(
            Err("semantic worker failed".to_string()),
            Err("cleanup failed".to_string()),
        );
        assert_eq!(result.unwrap_err(), "semantic worker failed");
        let cleanup_error =
            finish_semantic_task(Ok(()), Err("cleanup failed".to_string())).unwrap_err();
        assert_eq!(cleanup_error, "cleanup failed");
    }

    #[test]
    fn loaded_generation_check_rejects_replaced_timestamp_context() {
        let state = AppState::default();
        let expected_path = PathBuf::from("expected.sqlite3");
        *state.loaded.lock().unwrap() = Some(AppStateInner {
            db_path: expected_path.clone(),
            columns: Vec::new(),
            generation: 7,
        });
        assert!(loaded_generation_is_current(&state, &expected_path, 7).unwrap());
        assert!(!loaded_generation_is_current(&state, &expected_path, 8).unwrap());
        assert!(
            !loaded_generation_is_current(&state, Path::new("replacement.sqlite3"), 7).unwrap()
        );
        *state.loaded.lock().unwrap() = None;
        assert!(!loaded_generation_is_current(&state, &expected_path, 7).unwrap());
    }

    #[test]
    fn report_export_guard_rejects_overlap_and_releases_on_drop() {
        let busy = Arc::new(AtomicBool::new(false));
        let first = ReportExportGuard::acquire(&busy).unwrap();
        let error = ReportExportGuard::acquire(&busy).unwrap_err();
        assert!(error.contains("already running"));

        drop(first);
        let second = ReportExportGuard::acquire(&busy).unwrap();
        assert!(busy.load(Ordering::SeqCst));
        drop(second);
        assert!(!busy.load(Ordering::SeqCst));

        let busy_during_panic = Arc::clone(&busy);
        let panicked = std::panic::catch_unwind(move || {
            let _guard = ReportExportGuard::acquire(&busy_during_panic).unwrap();
            panic!("simulated report worker panic");
        });
        assert!(panicked.is_err());
        assert!(!busy.load(Ordering::SeqCst));
        drop(ReportExportGuard::acquire(&busy).unwrap());
    }

    #[test]
    fn best_effort_semantic_archival_never_relabels_acceptance() {
        let accepted = keep_primary_result_after_best_effort(Ok::<_, String>("accepted"), || {
            anyhow::bail!("simulated snapshot progress failure")
        })
        .unwrap();
        assert_eq!(accepted, "accepted");

        let archive_called = std::cell::Cell::new(false);
        let rejection = keep_primary_result_after_best_effort::<()>(
            Err("audit token mismatch".to_string()),
            || {
                archive_called.set(true);
                Ok(())
            },
        )
        .unwrap_err();
        assert_eq!(rejection, "audit token mismatch");
        assert!(!archive_called.get());
    }

    #[test]
    fn stale_report_publication_guard_preserves_existing_destination() {
        let state = AppState::default();
        let expected_db = PathBuf::from("expected-report.sqlite3");
        *state.loaded.lock().unwrap() = Some(AppStateInner {
            db_path: expected_db.clone(),
            columns: Vec::new(),
            generation: 12,
        });
        let directory = import_cache_test_path("stale-report-publish").with_extension("dir");
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        let temporary = directory.join("pending.xlsx");
        let destination = directory.join("report.xlsx");
        std::fs::write(&temporary, b"new report").unwrap();
        std::fs::write(&destination, b"existing report").unwrap();

        let error =
            publish_export_for_state_if_current(&state, &expected_db, 11, &temporary, &destination)
                .unwrap_err();
        assert!(error.to_string().contains("loaded file or sheet changed"));
        assert_eq!(std::fs::read(&destination).unwrap(), b"existing report");
        assert_eq!(std::fs::read(&temporary).unwrap(), b"new report");

        let _ = std::fs::remove_dir_all(directory);
    }
}
