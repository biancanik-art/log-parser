use crate::db::{self, ColumnMeta, ImportInfo};
use crate::export;
use crate::intel::matcher::{self, IntelScanSummary};
use crate::intel::{parser as guided_parser, query as guided_query, roles, time};
use crate::query::{self, QueryPage, QuerySpec};
use crate::report::{self, ReportExportSummary};
use crate::tabular_import;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use tauri::{AppHandle, Emitter, State};

pub struct AppStateInner {
    pub db_path: PathBuf,
    pub columns: Vec<ColumnMeta>,
}

#[derive(Default)]
pub struct AppState {
    pub loaded: Mutex<Option<AppStateInner>>,
    /// Guards against overlapping `import_sheet` calls (e.g. a double-clicked "Load Sheet"
    /// button, or opening a second file while the first is still importing). Without this,
    /// concurrent imports can race on the same cache file and on which result last wins the
    /// `loaded` slot — see AGENT_NOTES.md 2026-07-08 for the QA pass that found this.
    busy: AtomicBool,
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
struct ReportExportProgressPayload {
    rows_done: i64,
    sheet: String,
}

fn now_marker() -> String {
    chrono::Utc::now().to_rfc3339()
}

fn state_snapshot(state: &State<'_, AppState>) -> Result<(PathBuf, Vec<ColumnMeta>), String> {
    let guard = state
        .loaded
        .lock()
        .map_err(|_| "app state lock poisoned".to_string())?;
    let inner = guard
        .as_ref()
        .ok_or_else(|| "no file loaded — call import_sheet first".to_string())?;
    Ok((inner.db_path.clone(), inner.columns.clone()))
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
    let result = import_sheet_locked(&app, &state, path, sheet).await;
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
                if let Ok(conn) = db::open(&db_path_for_task) {
                    if let (Ok(columns), Ok(info)) =
                        (db::load_columns(&conn), db::load_import_info(&conn))
                    {
                        if info.sheet_name == sheet_for_task {
                            return Ok((columns, info.row_count, true));
                        }
                    }
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
        let mut guard = state
            .loaded
            .lock()
            .map_err(|_| "app state lock poisoned".to_string())?;
        *guard = Some(AppStateInner {
            db_path: db_path.clone(),
            columns: columns.clone(),
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
    let (db_path, columns) = state_snapshot(&state)?;
    let conn = db::open(&db_path).map_err(|e| e.to_string())?;
    query::query_rows(&conn, &columns, &spec).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn count_rows(state: State<'_, AppState>, spec: QuerySpec) -> Result<i64, String> {
    let (db_path, columns) = state_snapshot(&state)?;
    let conn = db::open(&db_path).map_err(|e| e.to_string())?;
    query::count_rows(&conn, &columns, &spec).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn parse_guided_query(
    state: State<'_, AppState>,
    query_text: String,
) -> Result<guided_parser::GuidedQueryPreview, String> {
    let (db_path, columns) = state_snapshot(&state)?;
    let conn = db::open(&db_path).map_err(|e| e.to_string())?;
    guided_parser::parse_guided_query(&conn, &columns, &query_text).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn run_guided_query(
    state: State<'_, AppState>,
    intent_token: String,
    cursor: Option<query::Cursor>,
    limit: Option<u32>,
) -> Result<QueryPage, String> {
    let (db_path, columns) = state_snapshot(&state)?;
    let conn = db::open(&db_path).map_err(|e| e.to_string())?;
    guided_query::run_guided_query(&conn, &columns, &intent_token, cursor, limit)
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn detect_column_roles(
    state: State<'_, AppState>,
) -> Result<Vec<roles::ColumnRoleSuggestion>, String> {
    let (db_path, columns) = state_snapshot(&state)?;
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
    let (db_path, columns) = state_snapshot(&state)?;
    let conn = db::open(&db_path).map_err(|e| e.to_string())?;
    roles::set_column_role_status(&conn, &columns, &role, &sql_name, status)
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn analyze_timestamp_column(
    state: State<'_, AppState>,
) -> Result<time::TimestampAnalysis, String> {
    let (db_path, columns) = state_snapshot(&state)?;
    tauri::async_runtime::spawn_blocking(move || -> Result<time::TimestampAnalysis, String> {
        let conn = db::open(&db_path).map_err(|e| e.to_string())?;
        time::analyze_confirmed_timestamp_column(&conn, &columns).map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("timestamp analysis task join error: {e}"))?
}

#[tauri::command]
pub async fn normalize_timestamp_column(
    state: State<'_, AppState>,
    naive_timezone: Option<String>,
) -> Result<time::TimestampNormalizationSummary, String> {
    let (db_path, columns) = state_snapshot(&state)?;
    tauri::async_runtime::spawn_blocking(
        move || -> Result<time::TimestampNormalizationSummary, String> {
            let mut conn = db::open(&db_path).map_err(|e| e.to_string())?;
            time::normalize_confirmed_timestamp_column(
                &mut conn,
                &columns,
                naive_timezone.as_deref(),
            )
            .map_err(|e| e.to_string())
        },
    )
    .await
    .map_err(|e| format!("timestamp normalization task join error: {e}"))?
}

#[tauri::command]
pub async fn export_data(
    app: AppHandle,
    state: State<'_, AppState>,
    spec: QuerySpec,
    format: ExportFormat,
    dest_path: String,
) -> Result<ExportSummary, String> {
    let (db_path, columns) = state_snapshot(&state)?;
    let dest = PathBuf::from(&dest_path);
    let dest_for_task = dest.clone();
    let app_for_task = app.clone();

    let row_count = tauri::async_runtime::spawn_blocking(move || -> Result<i64, String> {
        let conn = db::open(&db_path).map_err(|e| e.to_string())?;
        let on_progress = |rows_done: i64| {
            let _ = app_for_task.emit("export-progress", ExportProgressPayload { rows_done });
        };
        let result = match format {
            ExportFormat::Csv => {
                export::export_csv(&conn, &columns, &spec, &dest_for_task, on_progress)
            }
            ExportFormat::Xlsx => {
                export::export_xlsx(&conn, &columns, &spec, &dest_for_task, on_progress)
            }
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
pub async fn scan_intel_matches(
    app: AppHandle,
    state: State<'_, AppState>,
    evidence_columns: Vec<String>,
) -> Result<IntelScanSummary, String> {
    let (db_path, columns) = state_snapshot(&state)?;
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
) -> Result<ReportExportSummary, String> {
    let (db_path, columns) = state_snapshot(&state)?;
    let dest = PathBuf::from(&dest_path);
    let dest_for_task = dest.clone();
    let app_for_task = app.clone();

    tauri::async_runtime::spawn_blocking(move || -> Result<ReportExportSummary, String> {
        let conn = db::open(&db_path).map_err(|e| e.to_string())?;
        report::export_report(&conn, &columns, &dest_for_task, |rows_done, sheet| {
            let _ = app_for_task.emit(
                "report-export-progress",
                ReportExportProgressPayload {
                    rows_done,
                    sheet: sheet.to_string(),
                },
            );
        })
        .map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("report export task join error: {e}"))?
}
