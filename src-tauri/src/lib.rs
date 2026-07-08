mod commands;
pub mod csv_import;
pub mod db;
pub mod excel_import;
pub mod export;
pub mod header_utils;
pub mod intel;
pub mod query;
pub mod report;
pub mod tabular_import;

use commands::AppState;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            commands::list_sheets,
            commands::import_sheet,
            commands::query_rows,
            commands::count_rows,
            commands::parse_guided_query,
            commands::run_guided_query,
            commands::detect_column_roles,
            commands::set_column_role_status,
            commands::analyze_timestamp_column,
            commands::normalize_timestamp_column,
            commands::export_data,
            commands::scan_intel_matches,
            commands::export_report,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
