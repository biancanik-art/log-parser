use crate::db::{self, ColumnMeta, ImportInfo};
use crate::export;
use crate::intel::matcher::{self, IntelScanSummary};
use crate::intel::{llm_parser, parser as guided_parser, query as guided_query, roles, time};
use crate::query::{self, QueryPage, QuerySpec};
use crate::report::{self, ReportExportSummary};
use crate::semantic;
use crate::tabular_import;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tauri::{path::BaseDirectory, AppHandle, Emitter, Manager, State};

pub struct AppStateInner {
    pub db_path: PathBuf,
    pub columns: Vec<ColumnMeta>,
    pub generation: u64,
}

#[derive(Default)]
pub struct AppState {
    pub loaded: Mutex<Option<AppStateInner>>,
    pub llm: Arc<Mutex<Option<llm_parser::LlmParser>>>,
    pub semantic: Arc<Mutex<Option<semantic::SemanticModel>>>,
    next_generation: AtomicU64,
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
struct SemanticIndexProgressPayload {
    rows_done: i64,
    rows_total: i64,
    phase: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticIndexStatus {
    pub ready: bool,
    pub rows_indexed: i64,
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
    let rows_indexed = if ready {
        conn.query_row(
            "SELECT rows_indexed FROM _semantic_index_info ORDER BY rowid DESC LIMIT 1",
            [],
            |row| row.get(0),
        )
        .unwrap_or(0)
    } else {
        0
    };
    Ok(SemanticIndexStatus {
        ready,
        rows_indexed,
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
    let app_for_task = app.clone();
    let result = tauri::async_runtime::spawn_blocking(move || {
        let mut conn = db::open(&db_path).map_err(|error| error.to_string())?;
        let rows_total: i64 = conn
            .query_row("SELECT COUNT(*) FROM rows", [], |row| row.get(0))
            .map_err(|error| error.to_string())?;
        let _ = app_for_task.emit(
            "semantic-index-progress",
            SemanticIndexProgressPayload {
                rows_done: 0,
                rows_total,
                phase: "indexing".to_string(),
            },
        );
        let mut guard = semantic_model
            .lock()
            .map_err(|_| "semantic model lock poisoned".to_string())?;
        if guard.is_none() {
            *guard = Some(
                semantic::SemanticModel::load(&model_path, &tokenizer_path, &config_path)
                    .map_err(|error| error.to_string())?,
            );
        }
        let model = guard
            .as_ref()
            .ok_or_else(|| "semantic model failed to initialize".to_string())?;
        let summary = semantic::ensure_semantic_index(&mut conn, &columns, model)
            .map_err(|error| error.to_string())?;
        let _ = app_for_task.emit(
            "semantic-index-progress",
            SemanticIndexProgressPayload {
                rows_done: summary.rows_indexed,
                rows_total,
                phase: "complete".to_string(),
            },
        );
        Ok::<_, String>(summary)
    })
    .await
    .map_err(|error| format!("semantic index task join error: {error}"))??;

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
    let semantic_paths = match (
        resolve_llm_resource(&app, semantic::MODEL_RESOURCE_PATH),
        resolve_llm_resource(&app, semantic::TOKENIZER_RESOURCE_PATH),
        resolve_llm_resource(&app, semantic::CONFIG_RESOURCE_PATH),
    ) {
        (Ok(model), Ok(tokenizer), Ok(config)) => Some((model, tokenizer, config)),
        _ => None,
    };
    let preview = tauri::async_runtime::spawn_blocking(
        move || -> Result<guided_parser::GuidedQueryPreview, String> {
            let conn = db::open(&db_path).map_err(|e| e.to_string())?;
            // Semantic retrieval is optional. A missing/not-yet-built index or model resource
            // never blocks the validated lexical raw-table plan.
            let semantic_row_ids =
                if semantic::semantic_index_ready(&conn, &columns).unwrap_or(false) {
                    semantic_paths
                        .as_ref()
                        .and_then(|(semantic_path, semantic_tokenizer, semantic_config)| {
                            let mut guard = semantic_model.lock().ok()?;
                            if guard.is_none() {
                                *guard = semantic::SemanticModel::load(
                                    semantic_path,
                                    semantic_tokenizer,
                                    semantic_config,
                                )
                                .ok();
                            }
                            let model = guard.as_ref()?;
                            semantic::semantic_search(&conn, model, &query_text, 250, 0.20)
                                .ok()
                                .map(|candidates| {
                                    candidates
                                        .into_iter()
                                        .map(|candidate| candidate.row_num)
                                        .collect::<Vec<_>>()
                                })
                        })
                        .unwrap_or_default()
                } else {
                    Vec::new()
                };
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
            guided_parser::parse_guided_query_with_llm_and_semantic(
                &conn,
                &columns,
                &query_text,
                model,
                &semantic_row_ids,
            )
            .map_err(|error| error.to_string())
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
    let conn = db::open(&db_path).map_err(|error| error.to_string())?;
    guided_parser::accept_llm_audit(&conn, audit_id, &intent_token)
        .map_err(|error| error.to_string())
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
    let conn = db::open(&db_path).map_err(|e| e.to_string())?;
    guided_parser::accept_llm_audit(&conn, audit_id, &intent_token)
        .map_err(|error| error.to_string())?;
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
    state: State<'_, AppState>,
    naive_timezone: Option<String>,
    date_convention: Option<String>,
) -> Result<time::TimestampNormalizationSummary, String> {
    let (db_path, columns, _) = state_snapshot(&state)?;
    tauri::async_runtime::spawn_blocking(
        move || -> Result<time::TimestampNormalizationSummary, String> {
            let mut conn = db::open(&db_path).map_err(|e| e.to_string())?;
            time::normalize_timestamp_column_with_options(
                &mut conn,
                &columns,
                naive_timezone.as_deref(),
                date_convention.as_deref(),
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
    let (db_path, columns, _) = state_snapshot(&state)?;
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
    let app_for_task = app.clone();
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
            let _ = app_for_task.emit("export-progress", ExportProgressPayload { rows_done });
        };
        let summary = match (format, normalized_sort) {
            (ExportFormat::Csv, Some((source_column, direction))) => {
                export::export_csv_normalized_time(
                    &conn,
                    &columns,
                    &spec,
                    &source_column,
                    direction,
                    &dest_for_task,
                    on_progress,
                )
            }
            (ExportFormat::Xlsx, Some((source_column, direction))) => {
                export::export_xlsx_normalized_time(
                    &conn,
                    &columns,
                    &spec,
                    &source_column,
                    direction,
                    &dest_for_task,
                    on_progress,
                )
            }
            (ExportFormat::Csv, None) => {
                export::export_csv(&conn, &columns, &spec, &dest_for_task, on_progress)
            }
            (ExportFormat::Xlsx, None) => {
                export::export_xlsx(&conn, &columns, &spec, &dest_for_task, on_progress)
            }
        }
        .map_err(|error| error.to_string())?;
        Ok(summary.row_count)
    })
    .await
    .map_err(|error| format!("AI result export task join error: {error}"))??;

    let still_current = state
        .loaded
        .lock()
        .map_err(|_| "app state lock poisoned".to_string())?
        .as_ref()
        .is_some_and(|inner| inner.generation == generation && inner.db_path == exported_db_path);
    if !still_current {
        return Err(
            "the loaded file or sheet changed while the AI result export was running".to_string(),
        );
    }
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
) -> Result<ReportExportSummary, String> {
    let (db_path, columns, _) = state_snapshot(&state)?;
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
