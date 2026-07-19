mod commands;
pub mod csv_import;
pub mod db;
pub mod excel_import;
pub mod export;
pub mod header_utils;
pub mod intel;
pub mod query;
pub mod report;
pub mod semantic;
pub mod tabular_import;

use commands::AppState;
use tauri::{path::BaseDirectory, Manager};

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(AppState::default())
        .setup(|app| {
            let library_path = app
                .path()
                .resolve("intel/mitre_core.v1.json", BaseDirectory::Resource)?;
            intel::library::configure_builtin_library_path(library_path)?;
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::list_sheets,
            commands::import_sheet,
            commands::query_rows,
            commands::count_rows,
            commands::semantic_index_status,
            commands::build_semantic_index,
            commands::parse_guided_query,
            commands::accept_guided_query,
            commands::run_guided_query,
            commands::set_guided_parse_decision,
            commands::clear_loaded_file,
            commands::detect_column_roles,
            commands::set_column_role_status,
            commands::analyze_timestamp_column,
            commands::normalize_timestamp_column,
            commands::export_data,
            commands::export_guided_data,
            commands::scan_intel_matches,
            commands::ask_analyst,
            commands::export_report,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
