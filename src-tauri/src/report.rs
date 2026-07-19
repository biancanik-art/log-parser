use crate::db::{self, ColumnMeta};
use crate::export;
use crate::semantic;
use anyhow::{anyhow, bail, Context, Result};
use rusqlite::{params, Connection, OptionalExtension, TransactionBehavior};
use rust_xlsxwriter::{Workbook, Worksheet};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::net::{IpAddr, Ipv4Addr};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const VPN_RANGES_JSON: &str = include_str!("../resources/intel/vpn_ranges.v1.json");
const PROGRESS_EVERY: i64 = 5000;
const EXCEL_STRING_LIMIT: usize = 32_767;
const REPORT_SNAPSHOT_ATTEMPTS: usize = 4;
static REPORT_TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(1);

struct PendingReportExport {
    path: PathBuf,
    published: bool,
}

impl Drop for PendingReportExport {
    fn drop(&mut self) {
        if !self.published {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ReportExportSummary {
    pub sheets_written: Vec<String>,
    pub row_count: i64,
    pub dest_path: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct VpnRangesFile {
    schema_version: u32,
    ranges: Vec<VpnRangeRecord>,
}

#[derive(Debug, Deserialize)]
struct VpnRangeRecord {
    cidr: String,
    label: String,
}

#[derive(Debug, Clone)]
struct CompiledVpnRange {
    network: u32,
    mask: u32,
    label: String,
}

#[derive(Debug, Clone)]
struct RoleColumn {
    sql_name: String,
    original_name: String,
}

#[derive(Debug, Default)]
struct ConfirmedRoles {
    user: Option<RoleColumn>,
    command_line: Option<RoleColumn>,
    process_name: Option<RoleColumn>,
    file_name: Option<RoleColumn>,
    host: Option<RoleColumn>,
    ip: Option<RoleColumn>,
    text_evidence: Option<RoleColumn>,
}

#[derive(Debug)]
struct TacticSheet {
    tactic_id: String,
    tactic_name: String,
}

#[derive(Debug)]
struct ValueRollup {
    first_row_num: i64,
    count: i64,
}

#[derive(Debug)]
struct Ipv4SubnetRollup {
    first_row_num: i64,
    distinct_count: i64,
    total_count: i64,
}

#[derive(Debug)]
struct AssociationRollup {
    first_row_num: i64,
    row_count: i64,
    values: BTreeSet<String>,
}

struct RowWriter<'a, 'b, F>
where
    F: FnMut(i64, &str),
{
    worksheet: &'a mut Worksheet,
    sheet_name: &'b str,
    excel_row: u32,
    source_rows: &'a mut HashSet<i64>,
    total_rows_written: &'a mut i64,
    on_progress: &'a mut F,
}

impl<'a, 'b, F> RowWriter<'a, 'b, F>
where
    F: FnMut(i64, &str),
{
    fn new(
        worksheet: &'a mut Worksheet,
        sheet_name: &'b str,
        source_rows: &'a mut HashSet<i64>,
        total_rows_written: &'a mut i64,
        on_progress: &'a mut F,
    ) -> Self {
        Self {
            worksheet,
            sheet_name,
            excel_row: 1,
            source_rows,
            total_rows_written,
            on_progress,
        }
    }

    fn write_cells(&mut self, row_num: i64, cells: &[&str]) -> Result<()> {
        self.source_rows.insert(row_num);
        self.worksheet
            .write_number(self.excel_row, 0, row_num as f64)?;
        for (idx, value) in cells.iter().enumerate() {
            write_cell_string(self.worksheet, self.excel_row, (idx + 1) as u16, value)?;
        }
        self.bump_progress();
        Ok(())
    }

    fn write_cells_with_count(&mut self, row_num: i64, cells: &[&str], count: i64) -> Result<()> {
        self.source_rows.insert(row_num);
        self.worksheet
            .write_number(self.excel_row, 0, row_num as f64)?;
        for (idx, value) in cells.iter().enumerate() {
            write_cell_string(self.worksheet, self.excel_row, (idx + 1) as u16, value)?;
        }
        self.worksheet
            .write_number(self.excel_row, (cells.len() + 1) as u16, count as f64)?;
        self.bump_progress();
        Ok(())
    }

    fn bump_progress(&mut self) {
        self.excel_row += 1;
        *self.total_rows_written += 1;
        if *self.total_rows_written % PROGRESS_EVERY == 0 {
            (self.on_progress)(*self.total_rows_written, self.sheet_name);
        }
    }

    fn finish_sheet(&mut self) {
        (self.on_progress)(*self.total_rows_written, self.sheet_name);
    }
}

struct ReportWriteState<'a, F>
where
    F: FnMut(i64, &str),
{
    workbook: &'a mut Workbook,
    source_rows: &'a mut HashSet<i64>,
    total_rows_written: &'a mut i64,
    on_progress: &'a mut F,
}

pub fn export_report(
    conn: &mut Connection,
    columns: &[ColumnMeta],
    dest_path: &Path,
    on_progress: impl FnMut(i64, &str),
) -> Result<ReportExportSummary> {
    export_report_guarded(
        conn,
        columns,
        dest_path,
        on_progress,
        export::publish_completed_export,
    )
}

/// Builds a complete workbook in a unique sibling file and publishes it only after the file is
/// flushed. The caller-supplied publisher can keep dataset-generation validation and the atomic
/// replacement in one critical section. Any write, validation, or publication error removes the
/// temporary file and leaves an existing examiner report untouched.
pub fn export_report_guarded(
    conn: &mut Connection,
    columns: &[ColumnMeta],
    dest_path: &Path,
    mut on_progress: impl FnMut(i64, &str),
    publish: impl FnOnce(&Path, &Path) -> Result<()>,
) -> Result<ReportExportSummary> {
    let parent = dest_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let file_name = dest_path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("log-parser-report.xlsx");
    let sequence = REPORT_TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let temporary_path = parent.join(format!(
        ".{file_name}.log-parser-report-{}-{sequence}.tmp.xlsx",
        std::process::id()
    ));

    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary_path)
        .with_context(|| format!("creating temporary report {}", temporary_path.display()))?;
    let mut pending = PendingReportExport {
        path: temporary_path,
        published: false,
    };

    let mut summary =
        write_report_from_consistent_snapshot(conn, columns, &pending.path, &mut on_progress)?;
    std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&pending.path)
        .with_context(|| format!("opening completed report {}", pending.path.display()))?
        .sync_all()
        .with_context(|| format!("flushing completed report {}", pending.path.display()))?;
    publish(&pending.path, dest_path)?;
    pending.published = true;
    sync_report_parent_directory(parent)?;
    summary.dest_path = dest_path.display().to_string();
    Ok(summary)
}

fn write_report_from_consistent_snapshot<F>(
    conn: &mut Connection,
    columns: &[ColumnMeta],
    dest_path: &Path,
    on_progress: &mut F,
) -> Result<ReportExportSummary>
where
    F: FnMut(i64, &str),
{
    for _ in 0..REPORT_SNAPSHOT_ATTEMPTS {
        on_progress(0, "Semantic evidence archival");
        semantic::complete_required_semantic_audits(conn)
            .context("completing required semantic evidence archival before report export")?;

        let transaction = conn.transaction_with_behavior(TransactionBehavior::Deferred)?;
        // This must remain the first read after BEGIN DEFERRED. It fixes the SQLite snapshot and
        // detects an accepted selection that committed in the narrow gap after archival.
        if semantic::required_semantic_audits_pending(&transaction)? {
            transaction.rollback()?;
            continue;
        }

        let summary = write_report_workbook(&transaction, columns, dest_path, &mut *on_progress)?;
        transaction.commit()?;
        return Ok(summary);
    }

    bail!(
        "report evidence changed repeatedly while the point-in-time snapshot was starting; retry the export"
    )
}

fn write_report_workbook(
    conn: &Connection,
    columns: &[ColumnMeta],
    dest_path: &Path,
    mut on_progress: impl FnMut(i64, &str),
) -> Result<ReportExportSummary> {
    let ranges = load_vpn_ranges()?;
    let roles = load_confirmed_roles(conn, columns)?;
    let has_intel_matches = intel_match_count(conn)? > 0;
    let has_normalized_time = table_has_rows(conn, "_row_time")?;
    let has_timeline_data =
        has_intel_matches && has_normalized_time && intel_matches_have_normalized_time(conn)?;
    let tactics = if has_intel_matches {
        load_tactic_sheets(conn)?
    } else {
        Vec::new()
    };

    let mut workbook = Workbook::new();
    let mut sheets_written = Vec::new();
    let mut source_rows = HashSet::new();
    let mut total_rows_written = 0i64;
    let mut used_sheet_names = HashSet::new();

    {
        let mut write_state = ReportWriteState {
            workbook: &mut workbook,
            source_rows: &mut source_rows,
            total_rows_written: &mut total_rows_written,
            on_progress: &mut on_progress,
        };

        sheets_written.push(write_general_sheet(
            conn,
            columns,
            &roles,
            &ranges,
            &mut write_state,
        )?);
        used_sheet_names.insert("general".to_string());

        if table_has_rows(conn, "_llm_parse_audit")? {
            sheets_written.push(write_llm_audit_sheet(conn, &mut write_state)?);
            used_sheet_names.insert("ai audit".to_string());
        }

        if table_has_rows(conn, "_semantic_retrieval_audit")? {
            sheets_written.push(write_semantic_retrieval_audit_sheet(
                conn,
                &mut write_state,
            )?);
            used_sheet_names.insert("semantic retrieval".to_string());
        }

        if semantic_reportable_snapshots_have_rows(conn)? {
            sheets_written.push(write_semantic_audit_sheet(conn, &mut write_state)?);
            used_sheet_names.insert("semantic audit".to_string());
        }

        let has_activity = table_has_rows(conn, "_row_activity")?;
        if has_activity {
            sheets_written.push(write_activity_summary_sheet(conn, &mut write_state)?);
            used_sheet_names.insert("activity summary".to_string());
        }

        if has_timeline_data {
            sheets_written.push(write_timeline_sheet(
                conn,
                columns,
                &roles,
                &mut write_state,
            )?);
            used_sheet_names.insert("timeline".to_string());
        }

        if has_intel_matches && table_has_rows(conn, "_intel_chain")? {
            sheets_written.push(write_attack_story_sheet(conn, columns, &mut write_state)?);
            used_sheet_names.insert("attack story".to_string());
        }
        // Reserved ahead of the tactic loop so a name collision renames the tactic sheet,
        // not the fixed sheets written afterwards.
        let write_anomalies = table_has_rows(conn, "_anomaly")?;
        if write_anomalies {
            used_sheet_names.insert("anomalies".to_string());
        }
        if has_activity {
            used_sheet_names.insert("row by row".to_string());
        }

        for tactic in tactics {
            let sheet_name = unique_sheet_name(&tactic.tactic_name, &mut used_sheet_names);
            write_tactic_sheet(
                conn,
                columns,
                &tactic,
                &sheet_name,
                has_normalized_time,
                &mut write_state,
            )?;
            sheets_written.push(sheet_name);
        }

        if write_anomalies {
            sheets_written.push(write_anomaly_sheet(conn, columns, &mut write_state)?);
        }

        // Written last: at full-file scale this is by far the largest sheet.
        if has_activity {
            sheets_written.push(write_row_by_row_sheet(conn, &mut write_state)?);
        }
    }

    workbook.save(dest_path)?;

    Ok(ReportExportSummary {
        sheets_written,
        row_count: source_rows.len() as i64,
        dest_path: dest_path.display().to_string(),
    })
}

#[cfg(unix)]
fn sync_report_parent_directory(parent: &Path) -> Result<()> {
    std::fs::File::open(parent)?.sync_all()?;
    Ok(())
}

#[cfg(not(unix))]
fn sync_report_parent_directory(_parent: &Path) -> Result<()> {
    Ok(())
}

fn write_llm_audit_sheet<F>(
    conn: &Connection,
    state: &mut ReportWriteState<'_, F>,
) -> Result<String>
where
    F: FnMut(i64, &str),
{
    let sheet_name = "AI Audit".to_string();
    let worksheet = state.workbook.add_worksheet_with_constant_memory();
    worksheet.set_name(&sheet_name)?;
    let headers = [
        "audit_id",
        "created_at",
        "examiner_decision",
        "decided_at",
        "provider",
        "model_name",
        "model_version",
        "model_sha256",
        "tokenizer_sha256",
        "prompt_template_version",
        "correlation_engine_version",
        "input_sha256",
        "dataset_schema_sha256",
        "dataset_import_sha256",
        "validation_status",
        "validation_detail",
        "generation_parameters_json",
        "artifact_ids_json",
        "raw_model_output",
        "trusted_intent_json",
        "model_load_ms",
        "inference_latency_ms",
    ];
    write_headers(worksheet, &headers)?;
    for column in 0..headers.len() as u16 {
        worksheet.set_column_width(column, if column >= 16 { 60 } else { 24 })?;
    }

    let audit_columns = table_column_names(conn, "_llm_parse_audit")?;
    let dataset_schema = optional_text_column_expression(&audit_columns, "dataset_schema_sha256");
    let dataset_import = optional_text_column_expression(&audit_columns, "dataset_import_sha256");
    let sql = format!(
        "SELECT id, created_at, examiner_decision, COALESCE(decided_at, ''),
                provider, model_name, model_version, model_sha256, tokenizer_sha256,
                prompt_template_version, correlation_engine_version, input_sha256,
                {dataset_schema}, {dataset_import},
                validation_status, COALESCE(validation_detail, ''),
                generation_parameters_json, artifact_ids_json, raw_output,
                trusted_intent_json, load_time_ms, inference_latency_ms
         FROM _llm_parse_audit ORDER BY id"
    );
    let mut stmt = conn.prepare(&sql)?;
    let mut rows = stmt.query([])?;
    let mut excel_row = 1u32;
    while let Some(row) = rows.next()? {
        worksheet.write_number(excel_row, 0, row.get::<_, i64>(0)? as f64)?;
        for column in 1..20usize {
            let value: String = row.get(column)?;
            write_cell_string(worksheet, excel_row, column as u16, &value)?;
        }
        worksheet.write_number(excel_row, 20, row.get::<_, i64>(20)? as f64)?;
        worksheet.write_number(excel_row, 21, row.get::<_, i64>(21)? as f64)?;
        excel_row += 1;
        *state.total_rows_written += 1;
    }
    (state.on_progress)(*state.total_rows_written, &sheet_name);
    Ok(sheet_name)
}

fn table_column_names(conn: &Connection, table_name: &str) -> Result<HashSet<String>> {
    let sql = format!("PRAGMA table_info({})", db::quote_ident(table_name));
    let mut statement = conn.prepare(&sql)?;
    let columns = statement.query_map([], |row| row.get::<_, String>(1))?;
    columns
        .collect::<rusqlite::Result<HashSet<_>>>()
        .map_err(Into::into)
}

fn optional_text_column_expression(columns: &HashSet<String>, column_name: &str) -> String {
    if columns.contains(column_name) {
        format!("COALESCE({}, '')", db::quote_ident(column_name))
    } else {
        "''".to_string()
    }
}

fn optional_integer_column_expression(columns: &HashSet<String>, column_name: &str) -> String {
    if columns.contains(column_name) {
        db::quote_ident(column_name)
    } else {
        "NULL".to_string()
    }
}

fn write_semantic_retrieval_audit_sheet<F>(
    conn: &Connection,
    state: &mut ReportWriteState<'_, F>,
) -> Result<String>
where
    F: FnMut(i64, &str),
{
    let sheet_name = "Semantic Retrieval".to_string();
    let worksheet = state.workbook.add_worksheet_with_constant_memory();
    worksheet.set_name(&sheet_name)?;
    let headers = [
        "retrieval_id",
        "llm_audit_id",
        "input_sha256",
        "dataset_schema_sha256",
        "dataset_import_sha256",
        "semantic_used",
        "outcome_code",
        "detail",
        "selection_id",
        "created_at",
    ];
    write_headers(worksheet, &headers)?;
    for column in 0..headers.len() as u16 {
        worksheet.set_column_width(column, 24)?;
    }
    for column in [2u16, 3, 4, 7, 8] {
        worksheet.set_column_width(column, 56)?;
    }

    let columns = table_column_names(conn, "_semantic_retrieval_audit")?;
    let retrieval_id = optional_integer_column_expression(&columns, "id");
    let llm_audit_id = optional_integer_column_expression(&columns, "llm_audit_id");
    let input_sha256 = optional_text_column_expression(&columns, "input_sha256");
    let dataset_schema = optional_text_column_expression(&columns, "dataset_schema_sha256");
    let dataset_import = optional_text_column_expression(&columns, "dataset_import_sha256");
    let semantic_used = optional_integer_column_expression(&columns, "semantic_used");
    let outcome_code = optional_text_column_expression(&columns, "outcome_code");
    let detail = optional_text_column_expression(&columns, "detail");
    let selection_id = optional_text_column_expression(&columns, "selection_id");
    let created_at = optional_text_column_expression(&columns, "created_at");
    let sql = format!(
        "SELECT {retrieval_id}, {llm_audit_id}, {input_sha256}, {dataset_schema},
                {dataset_import}, {semantic_used}, {outcome_code}, {detail},
                {selection_id}, {created_at}
         FROM _semantic_retrieval_audit
         ORDER BY 1"
    );
    let mut statement = conn.prepare(&sql)?;
    let mut rows = statement.query([])?;
    let mut excel_row = 1u32;
    while let Some(row) = rows.next()? {
        for column in 0..headers.len() {
            write_audit_value(worksheet, excel_row, column as u16, row.get_ref(column)?)?;
        }
        excel_row += 1;
        *state.total_rows_written += 1;
    }
    (state.on_progress)(*state.total_rows_written, &sheet_name);
    Ok(sheet_name)
}

fn write_audit_value(
    worksheet: &mut Worksheet,
    row: u32,
    column: u16,
    value: rusqlite::types::ValueRef<'_>,
) -> Result<()> {
    match value {
        rusqlite::types::ValueRef::Null => {}
        rusqlite::types::ValueRef::Integer(value) => {
            if (-9_007_199_254_740_992..=9_007_199_254_740_992).contains(&value) {
                worksheet.write_number(row, column, value as f64)?;
            } else {
                worksheet.write_string(row, column, value.to_string())?;
            }
        }
        rusqlite::types::ValueRef::Real(value) => {
            worksheet.write_number(row, column, value)?;
        }
        rusqlite::types::ValueRef::Text(value) => {
            let value = std::str::from_utf8(value).context("decoding semantic retrieval audit")?;
            worksheet.write_string(row, column, value)?;
        }
        rusqlite::types::ValueRef::Blob(value) => {
            worksheet.write_string(row, column, lowercase_hex(value))?;
        }
    }
    Ok(())
}

fn write_semantic_audit_sheet<F>(
    conn: &Connection,
    state: &mut ReportWriteState<'_, F>,
) -> Result<String>
where
    F: FnMut(i64, &str),
{
    // The three immutable snapshot tables intentionally share one worksheet. The discriminator
    // makes every exported row self-describing while the common selection_id, mapping digest, and
    // row-set fields keep the forensic chain easy to follow without cross-sheet joins.
    let sheet_name = "Semantic Audit".to_string();
    let worksheet = state.workbook.add_worksheet_with_constant_memory();
    worksheet.set_name(&sheet_name)?;
    let headers = [
        "record_type",
        "selection_id",
        "snapshot_version",
        "build_id",
        "dataset_hash",
        "schema_hash",
        "index_version",
        "normalizer_version",
        "model_name",
        "model_version",
        "model_sha256",
        "tokenizer_sha256",
        "config_sha256",
        "query_sha256",
        "policy_version",
        "minimum_score",
        "maximum_documents",
        "documents_above_threshold",
        "documents_retained",
        "rows_matched",
        "documents_truncated",
        "broad_row_warning",
        "warnings_json",
        "source_rows",
        "index_rows_scanned",
        "index_documents_seen",
        "index_documents_embedded",
        "index_documents_mapped",
        "index_mappings_written",
        "index_documents_skipped",
        "index_mappings_skipped",
        "index_cells_truncated",
        "index_columns_omitted",
        "index_chunks_omitted",
        "candidate_documents",
        "candidate_mappings",
        "candidate_document_limit",
        "candidate_mapping_limit",
        "selected_document_count",
        "mapping_count",
        "mapping_sha256",
        "row_count",
        "row_set_sha256",
        "row_set_encoding",
        "selection_created_at",
        "archived_at",
        "rank",
        "source_doc_id",
        "fingerprint_sha256",
        "kind",
        "column_key",
        "normalized_text",
        "cosine_score",
        "rank_score",
        "chunk_index",
        "first_row_num",
        "last_row_num",
        "encoded_rows_hex",
        "chunk_sha256",
        "mapping_chunk_count",
        "row_chunk_count",
        "sealed",
        "seal_version",
        "archive_status",
    ];
    write_headers(worksheet, &headers)?;
    for column in 0..headers.len() as u16 {
        worksheet.set_column_width(column, 22)?;
    }
    for column in [4u16, 5, 10, 11, 12, 13, 22, 40, 42, 48, 51, 57, 58, 63] {
        worksheet.set_column_width(column, 56)?;
    }

    let mut excel_row = 1u32;
    let snapshot_targets = (1u16..=45).chain(59u16..=63).collect::<Vec<_>>();
    write_semantic_audit_query(
        conn,
        worksheet,
        "snapshot",
        "SELECT selection_id, snapshot_version, build_id, dataset_hash, schema_hash,
                index_version, normalizer_version, model_name, model_version, model_sha256,
                tokenizer_sha256, config_sha256, query_sha256, policy_version, minimum_score,
                maximum_documents, documents_above_threshold, documents_retained, rows_matched,
                documents_truncated, broad_row_warning, warnings_json, source_rows,
                index_rows_scanned, index_documents_seen, index_documents_embedded,
                index_documents_mapped, index_mappings_written, index_documents_skipped,
                index_mappings_skipped, index_cells_truncated, index_columns_omitted,
                index_chunks_omitted, candidate_documents, candidate_mappings,
                candidate_document_limit, candidate_mapping_limit, selected_document_count,
                mapping_count, mapping_sha256, row_count, row_set_sha256, row_set_encoding,
                selection_created_at, archived_at, mapping_chunk_count, row_chunk_count,
                sealed, seal_version, 'sealed_exact_mappings'
         FROM _semantic_v2_audit_snapshot p
         JOIN _semantic_v2_audit_snapshot_complete complete USING(selection_id)
         ORDER BY selection_id",
        &snapshot_targets,
        &mut excel_row,
        state.total_rows_written,
    )?;
    write_semantic_audit_query(
        conn,
        worksheet,
        "document",
        "SELECT selection_id, mapping_count, mapping_sha256, rank, source_doc_id,
                fingerprint_sha256, kind, column_key, normalized_text, cosine_score, rank_score,
                'sealed_exact_mappings'
         FROM _semantic_v2_audit_snapshot_document d
         JOIN _semantic_v2_audit_snapshot_complete complete USING(selection_id)
         ORDER BY selection_id, rank",
        &[1, 39, 40, 46, 47, 48, 49, 50, 51, 52, 53, 63],
        &mut excel_row,
        state.total_rows_written,
    )?;
    write_semantic_audit_query(
        conn,
        worksheet,
        "row_chunk",
        "SELECT selection_id, row_count, chunk_index, first_row_num, last_row_num,
                encoded_rows, chunk_sha256, 'sealed_exact_mappings'
         FROM _semantic_v2_audit_snapshot_row_chunk rc
         JOIN _semantic_v2_audit_snapshot_complete complete USING(selection_id)
         ORDER BY selection_id, chunk_index",
        &[1, 41, 54, 55, 56, 57, 58, 63],
        &mut excel_row,
        state.total_rows_written,
    )?;
    write_semantic_audit_query(
        conn,
        worksheet,
        "mapping_chunk",
        "SELECT selection_id, row_count, source_doc_id, chunk_index, first_row_num,
                last_row_num, encoded_rows, chunk_sha256, 'sealed_exact_mappings'
         FROM _semantic_v2_audit_snapshot_mapping_chunk mc
         JOIN _semantic_v2_audit_snapshot_complete complete USING(selection_id)
         ORDER BY selection_id, chunk_index",
        &[1, 39, 47, 54, 55, 56, 57, 58, 63],
        &mut excel_row,
        state.total_rows_written,
    )?;

    if semantic_audit_view_exists(conn, "_semantic_v2_audit_snapshot_legacy_union")? {
        write_semantic_audit_query(
            conn,
            worksheet,
            "legacy_snapshot_union_only",
            "SELECT selection_id, snapshot_version, build_id, dataset_hash, schema_hash,
                    index_version, normalizer_version, model_name, model_version, model_sha256,
                    tokenizer_sha256, config_sha256, query_sha256, policy_version, minimum_score,
                    maximum_documents, documents_above_threshold, documents_retained, rows_matched,
                    documents_truncated, broad_row_warning, warnings_json, source_rows,
                    index_rows_scanned, index_documents_seen, index_documents_embedded,
                    index_documents_mapped, index_mappings_written, index_documents_skipped,
                    index_mappings_skipped, index_cells_truncated, index_columns_omitted,
                    index_chunks_omitted, candidate_documents, candidate_mappings,
                    candidate_document_limit, candidate_mapping_limit, selected_document_count,
                    mapping_count, mapping_sha256, row_count, row_set_sha256, row_set_encoding,
                    selection_created_at, archived_at, NULL, NULL, sealed, seal_version,
                    'legacy_union_only_mapping_links_unavailable'
             FROM _semantic_v2_audit_snapshot p
             JOIN _semantic_v2_audit_snapshot_legacy_union legacy USING(selection_id)
             ORDER BY selection_id",
            &snapshot_targets,
            &mut excel_row,
            state.total_rows_written,
        )?;
        write_semantic_audit_query(
            conn,
            worksheet,
            "legacy_document_union_only",
            "SELECT selection_id, mapping_count, mapping_sha256, rank, source_doc_id,
                    fingerprint_sha256, kind, column_key, normalized_text, cosine_score, rank_score,
                    'legacy_union_only_mapping_links_unavailable'
             FROM _semantic_v2_audit_snapshot_document d
             JOIN _semantic_v2_audit_snapshot_legacy_union legacy USING(selection_id)
             ORDER BY selection_id, rank",
            &[1, 39, 40, 46, 47, 48, 49, 50, 51, 52, 53, 63],
            &mut excel_row,
            state.total_rows_written,
        )?;
        write_semantic_audit_query(
            conn,
            worksheet,
            "legacy_row_chunk_union_only",
            "SELECT selection_id, row_count, chunk_index, first_row_num, last_row_num,
                    encoded_rows, chunk_sha256,
                    'legacy_union_only_mapping_links_unavailable'
             FROM _semantic_v2_audit_snapshot_row_chunk rc
             JOIN _semantic_v2_audit_snapshot_legacy_union legacy USING(selection_id)
             ORDER BY selection_id, chunk_index",
            &[1, 41, 54, 55, 56, 57, 58, 63],
            &mut excel_row,
            state.total_rows_written,
        )?;
    }

    (state.on_progress)(*state.total_rows_written, &sheet_name);
    Ok(sheet_name)
}

fn write_semantic_audit_query(
    conn: &Connection,
    worksheet: &mut Worksheet,
    record_type: &str,
    sql: &str,
    target_columns: &[u16],
    excel_row: &mut u32,
    total_rows_written: &mut i64,
) -> Result<()> {
    let mut statement = conn.prepare(sql)?;
    let mut rows = statement.query([])?;
    while let Some(row) = rows.next()? {
        write_cell_string(worksheet, *excel_row, 0, record_type)?;
        for (source_column, target_column) in target_columns.iter().copied().enumerate() {
            match row.get_ref(source_column)? {
                rusqlite::types::ValueRef::Null => {}
                rusqlite::types::ValueRef::Integer(value) => {
                    if (-9_007_199_254_740_992..=9_007_199_254_740_992).contains(&value) {
                        worksheet.write_number(*excel_row, target_column, value as f64)?;
                    } else {
                        worksheet.write_string(*excel_row, target_column, value.to_string())?;
                    }
                }
                rusqlite::types::ValueRef::Real(value) => {
                    worksheet.write_number(*excel_row, target_column, value)?;
                }
                rusqlite::types::ValueRef::Text(value) => {
                    let value = std::str::from_utf8(value)
                        .context("decoding semantic audit snapshot text")?;
                    // Snapshot evidence must never use the report's display-oriented truncation.
                    // Let the workbook writer reject an impossible cell instead of silently
                    // changing immutable forensic data.
                    worksheet.write_string(*excel_row, target_column, value)?;
                }
                rusqlite::types::ValueRef::Blob(value) => {
                    worksheet.write_string(*excel_row, target_column, lowercase_hex(value))?;
                }
            }
        }
        *excel_row += 1;
        *total_rows_written += 1;
    }
    Ok(())
}

fn lowercase_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    encoded
}

fn table_has_rows(conn: &Connection, table_name: &str) -> Result<bool> {
    let exists = conn
        .query_row(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?1 LIMIT 1",
            params![table_name],
            |_| Ok(()),
        )
        .optional()?
        .is_some();
    if !exists {
        return Ok(false);
    }

    let sql = format!(
        "SELECT EXISTS(SELECT 1 FROM {} LIMIT 1)",
        db::quote_ident(table_name)
    );
    let has_rows: i64 = conn.query_row(&sql, [], |row| row.get(0))?;
    Ok(has_rows != 0)
}

fn semantic_audit_view_exists(conn: &Connection, view_name: &str) -> Result<bool> {
    conn.query_row(
        "SELECT EXISTS(
            SELECT 1 FROM sqlite_master
            WHERE type = 'view' AND name = ?1
         )",
        [view_name],
        |row| row.get(0),
    )
    .map_err(Into::into)
}

fn semantic_reportable_snapshots_have_rows(conn: &Connection) -> Result<bool> {
    for view_name in [
        "_semantic_v2_audit_snapshot_complete",
        "_semantic_v2_audit_snapshot_legacy_union",
    ] {
        if !semantic_audit_view_exists(conn, view_name)? {
            continue;
        }
        let sql = format!(
            "SELECT EXISTS(SELECT 1 FROM {} LIMIT 1)",
            db::quote_ident(view_name)
        );
        if conn.query_row(&sql, [], |row| row.get::<_, bool>(0))? {
            return Ok(true);
        }
    }
    Ok(false)
}

fn intel_match_count(conn: &Connection) -> Result<i64> {
    if !table_exists(conn, "_intel_match")? {
        return Ok(0);
    }
    conn.query_row("SELECT COUNT(*) FROM _intel_match", [], |row| row.get(0))
        .map_err(Into::into)
}

fn intel_matches_have_normalized_time(conn: &Connection) -> Result<bool> {
    if !table_exists(conn, "_intel_match")? || !table_exists(conn, "_row_time")? {
        return Ok(false);
    }
    let has_rows: i64 = conn.query_row(
        "SELECT EXISTS(
            SELECT 1
            FROM _intel_match m
            JOIN _row_time rt ON rt.row_num = m.row_num
            LIMIT 1
         )",
        [],
        |row| row.get(0),
    )?;
    Ok(has_rows != 0)
}

fn table_exists(conn: &Connection, table_name: &str) -> Result<bool> {
    Ok(conn
        .query_row(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?1 LIMIT 1",
            params![table_name],
            |_| Ok(()),
        )
        .optional()?
        .is_some())
}

fn load_tactic_sheets(conn: &Connection) -> Result<Vec<TacticSheet>> {
    if !table_exists(conn, "_intel_match")? {
        return Ok(Vec::new());
    }
    let mut stmt = conn.prepare(
        "SELECT tactic_id, tactic_name
         FROM _intel_match
         GROUP BY tactic_id, tactic_name
         HAVING COUNT(*) > 0
         ORDER BY tactic_name ASC, tactic_id ASC",
    )?;
    let rows = stmt.query_map([], |row| {
        Ok(TacticSheet {
            tactic_id: row.get(0)?,
            tactic_name: row.get(1)?,
        })
    })?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(Into::into)
}

fn load_confirmed_roles(conn: &Connection, columns: &[ColumnMeta]) -> Result<ConfirmedRoles> {
    Ok(ConfirmedRoles {
        user: confirmed_role_column(conn, columns, "user")?,
        command_line: confirmed_role_column(conn, columns, "command_line")?,
        process_name: confirmed_role_column(conn, columns, "process_name")?,
        file_name: confirmed_role_column(conn, columns, "file_name")?,
        host: confirmed_role_column(conn, columns, "host")?,
        ip: confirmed_role_column(conn, columns, "ip")?,
        text_evidence: confirmed_role_column(conn, columns, "text_evidence")?,
    })
}

fn confirmed_role_column(
    conn: &Connection,
    columns: &[ColumnMeta],
    role: &str,
) -> Result<Option<RoleColumn>> {
    if !table_exists(conn, "_column_roles")? {
        return Ok(None);
    }

    let sql_name = conn
        .query_row(
            "SELECT sql_name FROM _column_roles
             WHERE role = ?1 AND status = 'confirmed'
             LIMIT 1",
            params![role],
            |row| row.get::<_, String>(0),
        )
        .optional()?;

    let Some(sql_name) = sql_name else {
        return Ok(None);
    };
    let column = columns
        .iter()
        .find(|column| column.sql_name == sql_name)
        .ok_or_else(|| anyhow!("confirmed {role} column no longer exists: {sql_name}"))?;
    Ok(Some(RoleColumn {
        sql_name: column.sql_name.clone(),
        original_name: column.original_name.clone(),
    }))
}

fn write_general_sheet<F>(
    conn: &Connection,
    columns: &[ColumnMeta],
    roles: &ConfirmedRoles,
    ranges: &[CompiledVpnRange],
    state: &mut ReportWriteState<'_, F>,
) -> Result<String>
where
    F: FnMut(i64, &str),
{
    let sheet_name = "General".to_string();
    let first_row_num = first_source_row_num(conn)?;
    let worksheet = state.workbook.add_worksheet_with_constant_memory();
    worksheet.set_name(&sheet_name)?;
    write_headers(
        worksheet,
        &[
            "row_num",
            "section",
            "item",
            "value",
            "detail",
            "observed_count",
        ],
    )?;
    worksheet.set_column_width(0, 11)?;
    worksheet.set_column_width(1, 24)?;
    worksheet.set_column_width(2, 28)?;
    worksheet.set_column_width(3, 36)?;
    worksheet.set_column_width(4, 72)?;
    worksheet.set_column_width(5, 16)?;

    let mut writer = RowWriter::new(
        worksheet,
        &sheet_name,
        state.source_rows,
        state.total_rows_written,
        state.on_progress,
    );

    writer.write_cells_with_count(
        first_row_num,
        &[
            "Report note",
            "IP classification caveat",
            "Best-effort offline heuristic",
            "The bundled VPN/hosting ranges are not authoritative, complete, or live-updated. Treat CIDR hits as weak classification signals that need examiner review.",
        ],
        0,
    )?;
    write_date_range(conn, &mut writer)?;
    write_log_type_rollup(columns, roles, first_row_num, &mut writer)?;
    write_distinct_role_values(conn, &mut writer, roles.user.as_ref(), "Users", "user")?;
    write_distinct_role_values(conn, &mut writer, roles.host.as_ref(), "Hosts", "host")?;
    write_ip_rollups(conn, &mut writer, roles.ip.as_ref(), ranges)?;
    write_browser_rollups(conn, columns, first_row_num, &mut writer)?;
    write_user_host_rollups(conn, &mut writer, roles.user.as_ref(), roles.host.as_ref())?;

    writer.finish_sheet();
    Ok(sheet_name)
}

fn write_date_range<F>(conn: &Connection, writer: &mut RowWriter<'_, '_, F>) -> Result<()>
where
    F: FnMut(i64, &str),
{
    if !table_has_rows(conn, "_row_time")? {
        writer.write_cells_with_count(
            first_source_row_num(conn)?,
            &[
                "Date range",
                "normalized_utc",
                "not available",
                "No normalized timestamp data is available. The report still includes raw-data summaries and any independent audit evidence.",
            ],
            0,
        )?;
        return Ok(());
    }

    let min_time: (i64, String, i64) = conn.query_row(
        "SELECT row_num, utc_text, epoch_ms
         FROM _row_time
         ORDER BY epoch_ms ASC, row_num ASC
         LIMIT 1",
        [],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
    )?;
    let max_time: (i64, String, i64) = conn.query_row(
        "SELECT row_num, utc_text, epoch_ms
         FROM _row_time
         ORDER BY epoch_ms DESC, row_num DESC
         LIMIT 1",
        [],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
    )?;
    writer.write_cells_with_count(
        min_time.0,
        &[
            "Date range",
            "start_utc",
            min_time.1.as_str(),
            &format!("epoch_ms: {}", min_time.2),
        ],
        1,
    )?;
    writer.write_cells_with_count(
        max_time.0,
        &[
            "Date range",
            "end_utc",
            max_time.1.as_str(),
            &format!("epoch_ms: {}", max_time.2),
        ],
        1,
    )?;
    Ok(())
}

fn write_log_type_rollup<F>(
    columns: &[ColumnMeta],
    roles: &ConfirmedRoles,
    first_row_num: i64,
    writer: &mut RowWriter<'_, '_, F>,
) -> Result<()>
where
    F: FnMut(i64, &str),
{
    let log_types = infer_log_types(columns, roles);
    if log_types.is_empty() {
        writer.write_cells_with_count(
            first_row_num,
            &[
                "Log types",
                "best_effort",
                "not determined",
                "No strong log-type signal was found from confirmed roles or source column headers.",
            ],
            0,
        )?;
        return Ok(());
    }

    for log_type in log_types {
        writer.write_cells_with_count(
            first_row_num,
            &[
                "Log types",
                "best_effort",
                log_type.as_str(),
                "Inferred from confirmed roles and source column headers; review against the original source data.",
            ],
            0,
        )?;
    }
    Ok(())
}

fn write_distinct_role_values<F>(
    conn: &Connection,
    writer: &mut RowWriter<'_, '_, F>,
    role_column: Option<&RoleColumn>,
    section: &str,
    role_name: &str,
) -> Result<()>
where
    F: FnMut(i64, &str),
{
    let first_row_num = first_source_row_num(conn)?;
    let Some(role_column) = role_column else {
        writer.write_cells_with_count(
            first_row_num,
            &[
                section,
                role_name,
                "not available",
                &format!("No confirmed {role_name} column role was found."),
            ],
            0,
        )?;
        return Ok(());
    };

    let ident = db::quote_ident(&role_column.sql_name);
    let sql = format!(
        "SELECT row_num, {ident}
         FROM rows
         WHERE {ident} IS NOT NULL AND TRIM({ident}) != ''
         ORDER BY row_num ASC"
    );
    let mut stmt = conn.prepare(&sql)?;
    let mut rows = stmt.query([])?;

    // Fold by lowercase so the same real-world entity (e.g. a hostname or username that shows
    // up with inconsistent casing across log sources - Windows identities are case-insensitive)
    // isn't counted and listed as two different "distinct" values. The display casing shown is
    // whichever exact casing occurred most often for that folded value.
    struct Folded {
        first_row_num: i64,
        total_count: i64,
        casing_counts: BTreeMap<String, i64>,
    }
    let mut folded: BTreeMap<String, Folded> = BTreeMap::new();
    while let Some(row) = rows.next()? {
        let row_num: i64 = row.get(0)?;
        let value: String = row.get(1)?;
        let fold_key = value.to_ascii_lowercase();
        let entry = folded.entry(fold_key).or_insert_with(|| Folded {
            first_row_num: row_num,
            total_count: 0,
            casing_counts: BTreeMap::new(),
        });
        entry.total_count += 1;
        entry.first_row_num = entry.first_row_num.min(row_num);
        *entry.casing_counts.entry(value).or_insert(0) += 1;
    }

    let mut wrote_any = false;
    for folded_entry in folded.values() {
        wrote_any = true;
        let mut display_value = String::new();
        let mut best_count = -1i64;
        for (casing, count) in &folded_entry.casing_counts {
            if *count > best_count || (*count == best_count && casing < &display_value) {
                best_count = *count;
                display_value = casing.clone();
            }
        }
        writer.write_cells_with_count(
            folded_entry.first_row_num,
            &[
                section,
                role_column.original_name.as_str(),
                display_value.as_str(),
                "Distinct confirmed role value observed in the source data.",
            ],
            folded_entry.total_count,
        )?;
    }

    if !wrote_any {
        writer.write_cells_with_count(
            first_row_num,
            &[
                section,
                role_column.original_name.as_str(),
                "not available",
                "The confirmed role column contained no non-empty values.",
            ],
            0,
        )?;
    }
    Ok(())
}

fn write_ip_rollups<F>(
    conn: &Connection,
    writer: &mut RowWriter<'_, '_, F>,
    role_column: Option<&RoleColumn>,
    ranges: &[CompiledVpnRange],
) -> Result<()>
where
    F: FnMut(i64, &str),
{
    let first_row_num = first_source_row_num(conn)?;
    let Some(role_column) = role_column else {
        writer.write_cells_with_count(
            first_row_num,
            &[
                "IP addresses",
                "ip",
                "not available",
                "No confirmed ip column role was found.",
            ],
            0,
        )?;
        return Ok(());
    };

    let ident = db::quote_ident(&role_column.sql_name);
    let sql = format!(
        "SELECT row_num, {ident}
         FROM rows
         WHERE {ident} IS NOT NULL AND TRIM({ident}) != ''
         ORDER BY row_num ASC"
    );
    let mut rollups: BTreeMap<String, ValueRollup> = BTreeMap::new();
    let mut stmt = conn.prepare(&sql)?;
    let mut rows = stmt.query([])?;
    while let Some(row) = rows.next()? {
        let row_num: i64 = row.get(0)?;
        let raw: String = row.get(1)?;
        let Some(ip) = parse_ip_candidate(&raw) else {
            continue;
        };
        let key = ip.to_string();
        let entry = rollups.entry(key).or_insert(ValueRollup {
            first_row_num: row_num,
            count: 0,
        });
        entry.count += 1;
    }

    if rollups.is_empty() {
        writer.write_cells_with_count(
            first_row_num,
            &[
                "IP addresses",
                role_column.original_name.as_str(),
                "not available",
                "No parseable IP values were found in the confirmed IP column.",
            ],
            0,
        )?;
        return Ok(());
    }

    let mut vpn_match_rows: Vec<(String, ValueRollup, String)> = Vec::new();
    let mut other_individual_rows: Vec<(String, ValueRollup, String)> = Vec::new();
    let mut private_buckets: BTreeMap<u32, Ipv4SubnetRollup> = BTreeMap::new();

    for (ip_text, rollup) in rollups {
        let ip: IpAddr = ip_text
            .parse()
            .with_context(|| format!("re-parsing normalized IP {ip_text}"))?;
        match ip {
            IpAddr::V4(ipv4) => {
                if let Some(label) = classify_ipv4(ipv4, ranges) {
                    vpn_match_rows.push((
                        ip_text,
                        rollup,
                        format!(
                            "possible VPN/hosting: {label}; best-effort offline heuristic, not authoritative"
                        ),
                    ));
                } else if is_groupable_private_or_reserved_ipv4(ipv4) {
                    let network = ipv4_24_network(ipv4);
                    let entry =
                        private_buckets
                            .entry(network)
                            .or_insert_with(|| Ipv4SubnetRollup {
                                first_row_num: rollup.first_row_num,
                                distinct_count: 0,
                                total_count: 0,
                            });
                    entry.first_row_num = entry.first_row_num.min(rollup.first_row_num);
                    entry.distinct_count += 1;
                    entry.total_count += rollup.count;
                } else {
                    other_individual_rows.push((
                        ip_text,
                        rollup,
                        "no match in bundled offline VPN/hosting ranges".to_string(),
                    ));
                }
            }
            IpAddr::V6(_) => {
                other_individual_rows.push((
                    ip_text,
                    rollup,
                    "not checked; bundled VPN/hosting range dataset is IPv4-only".to_string(),
                ));
            }
        }
    }

    for (ip_text, rollup, detail) in vpn_match_rows.into_iter().chain(other_individual_rows) {
        writer.write_cells_with_count(
            rollup.first_row_num,
            &[
                "IP addresses",
                role_column.original_name.as_str(),
                ip_text.as_str(),
                detail.as_str(),
            ],
            rollup.count,
        )?;
    }

    for (network, rollup) in private_buckets {
        let network_text = format!("{}/24 (private/reserved range)", ipv4_from_u32(network));
        let detail = format!(
            "Distinct private/reserved IPs in this /24: {}; total observations: {}; no VPN/hosting match for any of them; best-effort offline heuristic",
            rollup.distinct_count, rollup.total_count
        );
        writer.write_cells_with_count(
            rollup.first_row_num,
            &[
                "IP addresses",
                role_column.original_name.as_str(),
                network_text.as_str(),
                detail.as_str(),
            ],
            rollup.total_count,
        )?;
    }
    Ok(())
}

fn write_browser_rollups<F>(
    conn: &Connection,
    columns: &[ColumnMeta],
    first_row_num: i64,
    writer: &mut RowWriter<'_, '_, F>,
) -> Result<()>
where
    F: FnMut(i64, &str),
{
    let Some(column) = find_user_agent_column(columns) else {
        writer.write_cells_with_count(
            first_row_num,
            &[
                "Browsers",
                "user_agent",
                "not available",
                "No clear user-agent or browser column was found.",
            ],
            0,
        )?;
        return Ok(());
    };

    let ident = db::quote_ident(&column.sql_name);
    let sql = format!(
        "SELECT row_num, {ident}
         FROM rows
         WHERE {ident} IS NOT NULL AND TRIM({ident}) != ''
         ORDER BY row_num ASC"
    );
    let mut rollups: BTreeMap<String, ValueRollup> = BTreeMap::new();
    let mut stmt = conn.prepare(&sql)?;
    let mut rows = stmt.query([])?;
    while let Some(row) = rows.next()? {
        let row_num: i64 = row.get(0)?;
        let raw: String = row.get(1)?;
        let Some(browser) = detect_browser(&raw) else {
            continue;
        };
        let entry = rollups.entry(browser.to_string()).or_insert(ValueRollup {
            first_row_num: row_num,
            count: 0,
        });
        entry.count += 1;
    }

    if rollups.is_empty() {
        writer.write_cells_with_count(
            first_row_num,
            &[
                "Browsers",
                column.original_name.as_str(),
                "not available",
                "A user-agent-like column was found, but no browser family was detected.",
            ],
            0,
        )?;
        return Ok(());
    }

    for (browser, rollup) in rollups {
        writer.write_cells_with_count(
            rollup.first_row_num,
            &[
                "Browsers",
                column.original_name.as_str(),
                browser.as_str(),
                "Best-effort browser family parsed from a user-agent-like column.",
            ],
            rollup.count,
        )?;
    }
    Ok(())
}

fn write_user_host_rollups<F>(
    conn: &Connection,
    writer: &mut RowWriter<'_, '_, F>,
    user_column: Option<&RoleColumn>,
    host_column: Option<&RoleColumn>,
) -> Result<()>
where
    F: FnMut(i64, &str),
{
    let first_row_num = first_source_row_num(conn)?;
    let (Some(user_column), Some(host_column)) = (user_column, host_column) else {
        writer.write_cells_with_count(
            first_row_num,
            &[
                "User-host rollups",
                "user_host",
                "not available",
                "Confirmed user and host roles are both required for this rollup.",
            ],
            0,
        )?;
        return Ok(());
    };

    let user_ident = db::quote_ident(&user_column.sql_name);
    let host_ident = db::quote_ident(&host_column.sql_name);
    let sql = format!(
        "SELECT row_num, {user_ident}, {host_ident}
         FROM rows
         WHERE {user_ident} IS NOT NULL AND TRIM({user_ident}) != ''
           AND {host_ident} IS NOT NULL AND TRIM({host_ident}) != ''
         ORDER BY row_num ASC"
    );
    let mut by_user: BTreeMap<String, AssociationRollup> = BTreeMap::new();
    let mut by_host: BTreeMap<String, AssociationRollup> = BTreeMap::new();
    let mut stmt = conn.prepare(&sql)?;
    let mut rows = stmt.query([])?;
    while let Some(row) = rows.next()? {
        let row_num: i64 = row.get(0)?;
        let user: String = row.get(1)?;
        let host: String = row.get(2)?;

        let user_entry = by_user
            .entry(user.clone())
            .or_insert_with(|| AssociationRollup {
                first_row_num: row_num,
                row_count: 0,
                values: BTreeSet::new(),
            });
        user_entry.row_count += 1;
        user_entry.values.insert(host.clone());

        let host_entry = by_host.entry(host).or_insert_with(|| AssociationRollup {
            first_row_num: row_num,
            row_count: 0,
            values: BTreeSet::new(),
        });
        host_entry.row_count += 1;
        host_entry.values.insert(user);
    }

    if by_user.is_empty() {
        writer.write_cells_with_count(
            first_row_num,
            &[
                "User-host rollups",
                "user_host",
                "not available",
                "No rows had both confirmed user and host values.",
            ],
            0,
        )?;
        return Ok(());
    }

    for (user, rollup) in by_user {
        let detail = format!("Hosts: {}", join_limited(&rollup.values));
        writer.write_cells_with_count(
            rollup.first_row_num,
            &[
                "User to host rollup",
                user_column.original_name.as_str(),
                user.as_str(),
                detail.as_str(),
            ],
            rollup.row_count,
        )?;
    }

    for (host, rollup) in by_host {
        let detail = format!("Users: {}", join_limited(&rollup.values));
        writer.write_cells_with_count(
            rollup.first_row_num,
            &[
                "Host to user rollup",
                host_column.original_name.as_str(),
                host.as_str(),
                detail.as_str(),
            ],
            rollup.row_count,
        )?;
    }
    Ok(())
}

fn write_timeline_sheet<F>(
    conn: &Connection,
    columns: &[ColumnMeta],
    roles: &ConfirmedRoles,
    state: &mut ReportWriteState<'_, F>,
) -> Result<String>
where
    F: FnMut(i64, &str),
{
    let sheet_name = "Timeline".to_string();
    let worksheet = state.workbook.add_worksheet_with_constant_memory();
    worksheet.set_name(&sheet_name)?;

    // Every event on this sheet is a keyword match, so the same source row can legitimately
    // appear more than once if it matched multiple keywords/techniques - each row here is one
    // match, not one unique source event. The matched_column/evidence pair (same evidence-
    // resolution logic the per-tactic sheets already use) means an examiner never has to go back
    // to the raw grid just to see what text actually triggered the hit.
    let mut headers = vec![
        "row_num",
        "utc_timestamp",
        "tactic_name",
        "technique_id",
        "technique_name",
        "keyword",
        "matched_column",
        "evidence",
    ];
    if let Some(user) = &roles.user {
        headers.push(user.original_name.as_str());
    }
    if let Some(host) = &roles.host {
        headers.push(host.original_name.as_str());
    }
    write_headers(worksheet, &headers)?;
    worksheet.set_column_width(0, 11)?;
    worksheet.set_column_width(1, 24)?;
    worksheet.set_column_width(2, 24)?;
    worksheet.set_column_width(3, 16)?;
    worksheet.set_column_width(4, 36)?;
    worksheet.set_column_width(5, 30)?;
    worksheet.set_column_width(6, 24)?;
    worksheet.set_column_width(7, 80)?;

    let evidence_expr = evidence_case_expression(columns);
    let mut select_exprs = vec![
        "m.row_num".to_string(),
        "COALESCE(rt.utc_text, '')".to_string(),
        "m.tactic_name".to_string(),
        "m.technique_id".to_string(),
        "m.technique_name".to_string(),
        "m.keyword".to_string(),
        "m.column_name".to_string(),
        evidence_expr,
    ];
    if let Some(user) = &roles.user {
        select_exprs.push(format!(
            "COALESCE(r.{}, '')",
            db::quote_ident(&user.sql_name)
        ));
    }
    if let Some(host) = &roles.host {
        select_exprs.push(format!(
            "COALESCE(r.{}, '')",
            db::quote_ident(&host.sql_name)
        ));
    }

    let sql = format!(
        "SELECT {}
         FROM _intel_match m
         LEFT JOIN _row_time rt ON rt.row_num = m.row_num
         JOIN rows r ON r.row_num = m.row_num
         ORDER BY rt.epoch_ms ASC, m.row_num ASC, m.tactic_name ASC, m.technique_id ASC, m.keyword ASC",
        select_exprs.join(", ")
    );

    let mut writer = RowWriter::new(
        worksheet,
        &sheet_name,
        state.source_rows,
        state.total_rows_written,
        state.on_progress,
    );
    let mut stmt = conn.prepare(&sql)?;
    let mut rows = stmt.query([])?;
    while let Some(row) = rows.next()? {
        let row_num: i64 = row.get(0)?;
        let mut values = Vec::new();
        for idx in 1..select_exprs.len() {
            let value: String = row.get(idx)?;
            values.push(value);
        }
        let refs: Vec<&str> = values.iter().map(String::as_str).collect();
        writer.write_cells(row_num, &refs)?;
    }

    writer.finish_sheet();
    Ok(sheet_name)
}

#[derive(Debug)]
struct StoryChain {
    chain_id: i64,
    host: Option<String>,
    start_epoch_ms: Option<i64>,
    end_epoch_ms: Option<i64>,
    first_row: i64,
    last_row: i64,
    tactic_count: i64,
    row_count: i64,
    score: i64,
    tactic_names: Vec<String>,
}

const MAX_STORY_EVENTS_PER_CHAIN: i64 = 500;

fn story_utc_text(epoch_ms: i64) -> String {
    chrono::DateTime::from_timestamp_millis(epoch_ms)
        .map(|dt| dt.format("%Y-%m-%d %H:%M:%S UTC").to_string())
        .unwrap_or_default()
}

/// A plain-language chronological incident narrative rebuilt from the published chain data.
/// Each chain contributes one summary line followed by its events in time order; every line
/// carries the original source row number, so the narrative stays fully pivotable.
fn write_attack_story_sheet<F>(
    conn: &Connection,
    columns: &[ColumnMeta],
    state: &mut ReportWriteState<'_, F>,
) -> Result<String>
where
    F: FnMut(i64, &str),
{
    let sheet_name = "Attack Story".to_string();
    let worksheet = state.workbook.add_worksheet_with_constant_memory();
    worksheet.set_name(&sheet_name)?;
    write_headers(
        worksheet,
        &[
            "row_num",
            "chain",
            "entry_type",
            "utc_timestamp",
            "host",
            "tactic",
            "technique",
            "narrative",
            "evidence",
        ],
    )?;
    worksheet.set_column_width(0, 11)?;
    worksheet.set_column_width(1, 8)?;
    worksheet.set_column_width(2, 14)?;
    worksheet.set_column_width(3, 24)?;
    worksheet.set_column_width(4, 20)?;
    worksheet.set_column_width(5, 22)?;
    worksheet.set_column_width(6, 36)?;
    worksheet.set_column_width(7, 110)?;
    worksheet.set_column_width(8, 80)?;

    let chains = {
        let mut stmt = conn.prepare(
            "SELECT chain_id, host, start_epoch_ms, end_epoch_ms, first_row, last_row,
                    tactic_count, row_count, score, tactic_names
             FROM _intel_chain
             ORDER BY chain_id ASC",
        )?;
        let rows = stmt.query_map([], |row| {
            let tactic_names_json: String = row.get(9)?;
            Ok(StoryChain {
                chain_id: row.get(0)?,
                host: row.get(1)?,
                start_epoch_ms: row.get(2)?,
                end_epoch_ms: row.get(3)?,
                first_row: row.get(4)?,
                last_row: row.get(5)?,
                tactic_count: row.get(6)?,
                row_count: row.get(7)?,
                score: row.get(8)?,
                tactic_names: serde_json::from_str(&tactic_names_json).unwrap_or_default(),
            })
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()?
    };

    let host_column = crate::intel::chains::detect_host_column(conn)?;
    let has_row_time = table_exists(conn, "_row_time")?;
    let evidence_expr = evidence_case_expression(columns);

    let mut writer = RowWriter::new(
        worksheet,
        &sheet_name,
        state.source_rows,
        state.total_rows_written,
        state.on_progress,
    );

    for chain in &chains {
        let where_host = chain.host.as_deref().and_then(|_| host_column.as_deref());
        let host_label = chain.host.as_deref().unwrap_or("(no host mapping)");
        let window_text = match (chain.start_epoch_ms, chain.end_epoch_ms) {
            (Some(start), Some(end)) => format!(
                " between {} and {}",
                story_utc_text(start),
                story_utc_text(end)
            ),
            _ => String::new(),
        };
        let summary = format!(
            "Chain {} on {}: {} tactics ({}) across {} rows{} (score {}).",
            chain.chain_id,
            host_label,
            chain.tactic_count,
            chain.tactic_names.join(" → "),
            chain.row_count,
            window_text,
            chain.score
        );
        writer.write_cells(
            chain.first_row,
            &[
                &chain.chain_id.to_string(),
                "chain summary",
                "",
                host_label,
                "",
                "",
                &summary,
                "",
            ],
        )?;

        // Chain membership is reconstructed with the same host/time semantics the chains
        // were computed under: the host column that grouped them and the published window
        // (row-number range when the dataset has no normalized time).
        let (time_select, time_join, time_order) = if has_row_time {
            (
                "COALESCE(rt.utc_text, '')",
                "LEFT JOIN _row_time rt ON rt.row_num = m.row_num",
                "rt.epoch_ms ASC, ",
            )
        } else {
            ("''", "", "")
        };
        let mut predicates = Vec::new();
        let mut params_vec: Vec<rusqlite::types::Value> = Vec::new();
        match (chain.start_epoch_ms, chain.end_epoch_ms) {
            // Chains with a temporal claim were built from time-windowed events; row numbers
            // are not time-ordered, so membership must use the same epoch window.
            (Some(start), Some(end)) if has_row_time => {
                predicates.push("rt.epoch_ms BETWEEN ? AND ?".to_string());
                params_vec.push(start.into());
                params_vec.push(end.into());
            }
            _ => {
                predicates.push("m.row_num BETWEEN ? AND ?".to_string());
                params_vec.push(chain.first_row.into());
                params_vec.push(chain.last_row.into());
            }
        }
        if let (Some(host_col), Some(host_value)) = (where_host, chain.host.as_deref()) {
            predicates.push(format!("r.{} = ?", db::quote_ident(host_col)));
            params_vec.push(host_value.to_string().into());
        }
        let sql = format!(
            "SELECT m.row_num, {time_select}, m.tactic_name, m.technique_name,
                    m.keyword, m.column_name, MAX(m.score), {evidence_expr}
             FROM _intel_match m
             JOIN rows r ON r.row_num = m.row_num
             {time_join}
             WHERE {}
             GROUP BY m.row_num, m.tactic_id, m.technique_id
             ORDER BY {time_order}m.row_num ASC
             LIMIT {MAX_STORY_EVENTS_PER_CHAIN}",
            predicates.join(" AND ")
        );
        let mut stmt = conn.prepare(&sql)?;
        let mut rows = stmt.query(rusqlite::params_from_iter(params_vec))?;
        while let Some(row) = rows.next()? {
            let row_num: i64 = row.get(0)?;
            let utc_text: String = row.get(1)?;
            let tactic_name: String = row.get(2)?;
            let technique_name: String = row.get(3)?;
            let keyword: String = row.get(4)?;
            let column_name: String = row.get(5)?;
            let evidence: String = row.get(7)?;
            let when_text = if utc_text.is_empty() {
                format!("Row {row_num}")
            } else {
                format!("At {utc_text}")
            };
            let narrative = format!(
                "{when_text}: {tactic_name} activity ({technique_name}) — '{keyword}' seen in column '{column_name}'."
            );
            writer.write_cells(
                row_num,
                &[
                    &chain.chain_id.to_string(),
                    "event",
                    utc_text.as_str(),
                    host_label,
                    tactic_name.as_str(),
                    technique_name.as_str(),
                    &narrative,
                    evidence.as_str(),
                ],
            )?;
        }
    }

    writer.finish_sheet();
    Ok(sheet_name)
}

/// Wide-net heuristic findings. Deliberately labeled as tolerant of false positives so the
/// sheet reads as a review queue, not as verdicts.
fn write_anomaly_sheet<F>(
    conn: &Connection,
    columns: &[ColumnMeta],
    state: &mut ReportWriteState<'_, F>,
) -> Result<String>
where
    F: FnMut(i64, &str),
{
    let sheet_name = "Anomalies".to_string();
    let worksheet = state.workbook.add_worksheet_with_constant_memory();
    worksheet.set_name(&sheet_name)?;
    write_headers(
        worksheet,
        &[
            "row_num",
            "utc_timestamp",
            "category",
            "score",
            "reason",
            "matched_column",
            "evidence",
        ],
    )?;
    worksheet.set_column_width(0, 11)?;
    worksheet.set_column_width(1, 24)?;
    worksheet.set_column_width(2, 26)?;
    worksheet.set_column_width(3, 8)?;
    worksheet.set_column_width(4, 90)?;
    worksheet.set_column_width(5, 24)?;
    worksheet.set_column_width(6, 80)?;

    let has_row_time = table_exists(conn, "_row_time")?;
    let (time_select, time_join) = if has_row_time {
        (
            "COALESCE(rt.utc_text, '')",
            "LEFT JOIN _row_time rt ON rt.row_num = m.row_num",
        )
    } else {
        ("''", "")
    };
    let evidence_expr = evidence_case_expression(columns);
    let sql = format!(
        "SELECT m.row_num, {time_select}, m.category, m.score, m.reason, m.column_name,
                {evidence_expr}
         FROM _anomaly m
         JOIN rows r ON r.row_num = m.row_num
         {time_join}
         ORDER BY m.score DESC, m.row_num ASC, m.category ASC"
    );

    let mut writer = RowWriter::new(
        worksheet,
        &sheet_name,
        state.source_rows,
        state.total_rows_written,
        state.on_progress,
    );
    let mut stmt = conn.prepare(&sql)?;
    let mut rows = stmt.query([])?;
    while let Some(row) = rows.next()? {
        let row_num: i64 = row.get(0)?;
        let utc_text: String = row.get(1)?;
        let category: String = row.get(2)?;
        let score: i64 = row.get(3)?;
        let reason: String = row.get(4)?;
        let matched_column: String = row.get(5)?;
        let evidence: String = row.get(6)?;
        writer.write_cells(
            row_num,
            &[
                utc_text.as_str(),
                crate::intel::anomaly::category_label(&category),
                &score.to_string(),
                reason.as_str(),
                matched_column.as_str(),
                evidence.as_str(),
            ],
        )?;
    }

    writer.finish_sheet();
    Ok(sheet_name)
}

/// One line per activity category: how much of the file each activity type is, with the most
/// common operation values. Anchored to the first source row like the General sheet so the
/// row_num back-reference invariant holds for summary lines too.
fn write_activity_summary_sheet<F>(
    conn: &Connection,
    state: &mut ReportWriteState<'_, F>,
) -> Result<String>
where
    F: FnMut(i64, &str),
{
    let sheet_name = "Activity Summary".to_string();
    let first_row_num = first_source_row_num(conn)?;
    let worksheet = state.workbook.add_worksheet_with_constant_memory();
    worksheet.set_name(&sheet_name)?;
    write_headers(
        worksheet,
        &["row_num", "activity", "rows", "share_pct", "most_common_operations"],
    )?;
    worksheet.set_column_width(0, 11)?;
    worksheet.set_column_width(1, 32)?;
    worksheet.set_column_width(2, 12)?;
    worksheet.set_column_width(3, 10)?;
    worksheet.set_column_width(4, 90)?;

    let total: i64 = conn.query_row("SELECT COUNT(*) FROM _row_activity", [], |row| row.get(0))?;
    let mut writer = RowWriter::new(
        worksheet,
        &sheet_name,
        state.source_rows,
        state.total_rows_written,
        state.on_progress,
    );
    let categories: Vec<(String, i64)> = {
        let mut stmt = conn.prepare(
            "SELECT category, COUNT(*) FROM _row_activity
             GROUP BY category ORDER BY COUNT(*) DESC",
        )?;
        let rows = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        rows
    };
    for (category, count) in categories {
        let top_details: Vec<String> = {
            let mut stmt = conn.prepare_cached(
                "SELECT detail, COUNT(*) FROM _row_activity
                 WHERE category = ?1 AND detail != ''
                 GROUP BY detail ORDER BY COUNT(*) DESC LIMIT 3",
            )?;
            let rows = stmt
                .query_map([&category], |row| {
                    Ok(format!(
                        "{} ({} rows)",
                        row.get::<_, String>(0)?,
                        row.get::<_, i64>(1)?
                    ))
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            rows
        };
        let share = if total > 0 {
            format!("{:.1}", (count as f64 / total as f64) * 100.0)
        } else {
            "0.0".to_string()
        };
        writer.write_cells(
            first_row_num,
            &[
                crate::intel::activity::category_label(&category),
                &count.to_string(),
                &share,
                &top_details.join("; "),
            ],
        )?;
    }
    writer.finish_sheet();
    Ok(sheet_name)
}

/// Excel's hard per-sheet limit is 1,048,576 rows; anything beyond it is written as a truncation
/// notice rather than silently dropped.
const MAX_ROW_BY_ROW_ROWS: i64 = 1_000_000;

/// The full annotated file: every source row with its activity label, plus the MITRE and
/// anomaly annotations where they exist. Streams in row order through a constant-memory
/// worksheet so a 500k-row file cannot balloon the report's memory use.
fn write_row_by_row_sheet<F>(
    conn: &Connection,
    state: &mut ReportWriteState<'_, F>,
) -> Result<String>
where
    F: FnMut(i64, &str),
{
    let sheet_name = "Row by Row".to_string();
    let worksheet = state.workbook.add_worksheet_with_constant_memory();
    worksheet.set_name(&sheet_name)?;
    write_headers(
        worksheet,
        &[
            "row_num",
            "utc_timestamp",
            "activity",
            "detail",
            "mitre_techniques",
            "anomaly_flags",
        ],
    )?;
    worksheet.set_column_width(0, 11)?;
    worksheet.set_column_width(1, 24)?;
    worksheet.set_column_width(2, 30)?;
    worksheet.set_column_width(3, 80)?;
    worksheet.set_column_width(4, 28)?;
    worksheet.set_column_width(5, 30)?;

    let time_parts = if table_exists(conn, "_row_time")? {
        (
            "COALESCE(rt.utc_text, '')",
            "LEFT JOIN _row_time rt ON rt.row_num = a.row_num",
        )
    } else {
        ("''", "")
    };
    let techniques_select = if table_exists(conn, "_intel_match")? {
        "COALESCE((SELECT GROUP_CONCAT(DISTINCT technique_id)
                   FROM _intel_match im WHERE im.row_num = a.row_num), '')"
    } else {
        "''"
    };
    let anomalies_select = if table_exists(conn, "_anomaly")? {
        "COALESCE((SELECT GROUP_CONCAT(DISTINCT category)
                   FROM _anomaly an WHERE an.row_num = a.row_num), '')"
    } else {
        "''"
    };
    let (time_select, time_join) = time_parts;
    let sql = format!(
        "SELECT a.row_num, {time_select}, a.category, a.detail,
                {techniques_select}, {anomalies_select}
         FROM _row_activity a
         {time_join}
         ORDER BY a.row_num ASC
         LIMIT {}",
        MAX_ROW_BY_ROW_ROWS + 1
    );

    let mut writer = RowWriter::new(
        worksheet,
        &sheet_name,
        state.source_rows,
        state.total_rows_written,
        state.on_progress,
    );
    let mut written = 0i64;
    let mut stmt = conn.prepare(&sql)?;
    let mut rows = stmt.query([])?;
    while let Some(row) = rows.next()? {
        let row_num: i64 = row.get(0)?;
        if written >= MAX_ROW_BY_ROW_ROWS {
            writer.write_cells(
                row_num,
                &[
                    "",
                    "TRUNCATED",
                    "sheet reached Excel's row limit; remaining rows are in the app grid",
                    "",
                    "",
                ],
            )?;
            break;
        }
        let utc_text: String = row.get(1)?;
        let category: String = row.get(2)?;
        let detail: String = row.get(3)?;
        let techniques: String = row.get(4)?;
        let anomalies: String = row.get(5)?;
        writer.write_cells(
            row_num,
            &[
                utc_text.as_str(),
                crate::intel::activity::category_label(&category),
                detail.as_str(),
                techniques.as_str(),
                anomalies.as_str(),
            ],
        )?;
        written += 1;
    }
    writer.finish_sheet();
    Ok(sheet_name)
}

fn write_tactic_sheet<F>(
    conn: &Connection,
    columns: &[ColumnMeta],
    tactic: &TacticSheet,
    sheet_name: &str,
    has_normalized_time: bool,
    state: &mut ReportWriteState<'_, F>,
) -> Result<()>
where
    F: FnMut(i64, &str),
{
    let worksheet = state.workbook.add_worksheet_with_constant_memory();
    worksheet.set_name(sheet_name)?;
    write_headers(
        worksheet,
        &[
            "row_num",
            "utc_timestamp",
            "technique_id",
            "technique_name",
            "keyword",
            "matched_column",
            "evidence",
        ],
    )?;
    worksheet.set_column_width(0, 11)?;
    worksheet.set_column_width(1, 24)?;
    worksheet.set_column_width(2, 16)?;
    worksheet.set_column_width(3, 36)?;
    worksheet.set_column_width(4, 30)?;
    worksheet.set_column_width(5, 24)?;
    worksheet.set_column_width(6, 80)?;

    let evidence_expr = evidence_case_expression(columns);
    let (timestamp_expression, timestamp_join, timestamp_order) = if has_normalized_time {
        (
            "COALESCE(rt.utc_text, '')",
            "LEFT JOIN _row_time rt ON rt.row_num = m.row_num",
            "rt.epoch_ms ASC, ",
        )
    } else {
        ("''", "", "")
    };
    let sql = format!(
        "SELECT m.row_num,
                {timestamp_expression},
                m.technique_id,
                m.technique_name,
                m.keyword,
                m.column_name,
                {evidence_expr}
         FROM _intel_match m
         {timestamp_join}
         JOIN rows r ON r.row_num = m.row_num
         WHERE m.tactic_id = ?1
         ORDER BY {timestamp_order}m.row_num ASC, m.technique_id ASC, m.keyword ASC, m.column_name ASC"
    );

    let mut writer = RowWriter::new(
        worksheet,
        sheet_name,
        state.source_rows,
        state.total_rows_written,
        state.on_progress,
    );
    let mut stmt = conn.prepare(&sql)?;
    let mut rows = stmt.query(params![tactic.tactic_id])?;
    while let Some(row) = rows.next()? {
        let row_num: i64 = row.get(0)?;
        let timestamp: String = row.get(1)?;
        let technique_id: String = row.get(2)?;
        let technique_name: String = row.get(3)?;
        let keyword: String = row.get(4)?;
        let matched_column: String = row.get(5)?;
        let evidence: String = row.get(6)?;
        writer.write_cells(
            row_num,
            &[
                timestamp.as_str(),
                technique_id.as_str(),
                technique_name.as_str(),
                keyword.as_str(),
                matched_column.as_str(),
                evidence.as_str(),
            ],
        )?;
    }

    writer.finish_sheet();
    Ok(())
}

fn first_source_row_num(conn: &Connection) -> Result<i64> {
    let row_num = conn
        .query_row("SELECT MIN(row_num) FROM rows", [], |row| {
            row.get::<_, Option<i64>>(0)
        })?
        .ok_or_else(|| anyhow!("no source rows are loaded"))?;
    Ok(row_num)
}

fn infer_log_types(columns: &[ColumnMeta], roles: &ConfirmedRoles) -> Vec<String> {
    let mut out = BTreeSet::new();
    let headers = columns
        .iter()
        .map(|column| {
            format!(
                "{} {}",
                column.sql_name.to_ascii_lowercase(),
                column.original_name.to_ascii_lowercase()
            )
        })
        .collect::<Vec<_>>()
        .join(" ");

    if contains_any(
        &headers,
        &["signin", "sign in", "logon", "login", "authentication"],
    ) {
        out.insert("authentication logs".to_string());
    }
    if contains_any(
        &headers,
        &[
            "operation",
            "audit",
            "activity",
            "cloud",
            "azure",
            "office",
            "m365",
        ],
    ) {
        out.insert("cloud/audit logs".to_string());
    }
    if contains_any(
        &headers,
        &["url", "uri", "http", "request", "response", "useragent"],
    ) {
        out.insert("web/http logs".to_string());
    }
    if contains_any(&headers, &["email", "mail", "sender", "recipient"]) {
        out.insert("email logs".to_string());
    }
    if contains_any(&headers, &["file", "path", "folder"]) || roles.file_name.is_some() {
        out.insert("file activity logs".to_string());
    }
    if roles.command_line.is_some() || roles.process_name.is_some() || roles.host.is_some() {
        out.insert("endpoint/process activity".to_string());
    }
    if roles.ip.is_some() && !out.iter().any(|value| value.contains("web/http")) {
        out.insert("network or access logs".to_string());
    }
    if roles.text_evidence.is_some() {
        out.insert("alert/evidence text logs".to_string());
    }

    out.into_iter().collect()
}

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| haystack.contains(needle))
}

fn find_user_agent_column(columns: &[ColumnMeta]) -> Option<&ColumnMeta> {
    columns.iter().find(|column| {
        let compact = format!("{}{}", column.sql_name, column.original_name)
            .to_ascii_lowercase()
            .chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .collect::<String>();
        compact.contains("useragent")
            || compact.contains("httpuseragent")
            || compact.contains("browser")
    })
}

fn detect_browser(value: &str) -> Option<&'static str> {
    let lower = value.to_ascii_lowercase();
    if lower.contains("edg/") || lower.contains("edge/") {
        Some("Microsoft Edge")
    } else if lower.contains("opr/") || lower.contains("opera") {
        Some("Opera")
    } else if lower.contains("firefox/") {
        Some("Firefox")
    } else if lower.contains("chrome/") || lower.contains("chromium/") {
        Some("Chrome/Chromium")
    } else if lower.contains("safari/") && lower.contains("version/") {
        Some("Safari")
    } else if lower.contains("msie ") || lower.contains("trident/") {
        Some("Internet Explorer")
    } else {
        None
    }
}

fn evidence_case_expression(columns: &[ColumnMeta]) -> String {
    let cases = columns
        .iter()
        .map(|column| {
            format!(
                "WHEN {} THEN r.{}",
                sql_string_literal(&column.sql_name),
                db::quote_ident(&column.sql_name)
            )
        })
        .collect::<Vec<_>>()
        .join(" ");
    format!("COALESCE(CASE m.column_name {cases} ELSE '' END, '')")
}

fn sql_string_literal(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

fn unique_sheet_name(raw: &str, used_lowercase: &mut HashSet<String>) -> String {
    let base = sanitize_sheet_name(raw);
    for idx in 1.. {
        let candidate = if idx == 1 {
            base.clone()
        } else {
            let suffix = format!(" ({idx})");
            let max_base_len = 31usize.saturating_sub(suffix.len());
            format!("{}{}", truncate_chars(&base, max_base_len), suffix)
        };
        if used_lowercase.insert(candidate.to_ascii_lowercase()) {
            return candidate;
        }
    }
    unreachable!("unbounded sheet-name collision loop should always return");
}

fn sanitize_sheet_name(raw: &str) -> String {
    let mut out = String::new();
    for ch in raw.chars() {
        if matches!(ch, ':' | '\\' | '/' | '?' | '*' | '[' | ']') {
            out.push(' ');
        } else {
            out.push(ch);
        }
    }
    let collapsed = out.split_whitespace().collect::<Vec<_>>().join(" ");
    let trimmed = collapsed.trim_matches('\'').trim();
    let value = if trimmed.is_empty() { "Sheet" } else { trimmed };
    truncate_chars(value, 31)
}

fn truncate_chars(value: &str, max_chars: usize) -> String {
    value.chars().take(max_chars).collect()
}

fn write_headers(worksheet: &mut Worksheet, headers: &[&str]) -> Result<()> {
    for (idx, header) in headers.iter().enumerate() {
        write_cell_string(worksheet, 0, idx as u16, header)?;
    }
    Ok(())
}

fn write_cell_string(worksheet: &mut Worksheet, row: u32, col: u16, value: &str) -> Result<()> {
    worksheet.write_string(row, col, excel_safe_string(value))?;
    Ok(())
}

fn excel_safe_string(value: &str) -> Cow<'_, str> {
    if value.len() <= EXCEL_STRING_LIMIT {
        return Cow::Borrowed(value);
    }

    let mut out = value
        .chars()
        .take(EXCEL_STRING_LIMIT - 32)
        .collect::<String>();
    out.push_str("... [truncated for Excel cell]");
    Cow::Owned(out)
}

fn join_limited(values: &BTreeSet<String>) -> String {
    let mut out = String::new();
    let mut included = 0usize;
    for value in values {
        let separator_len = if out.is_empty() { 0 } else { 2 };
        if out.len() + separator_len + value.len() > 30_000 {
            break;
        }
        if !out.is_empty() {
            out.push_str("; ");
        }
        out.push_str(value);
        included += 1;
    }
    let remaining = values.len().saturating_sub(included);
    if remaining > 0 {
        if !out.is_empty() {
            out.push_str("; ");
        }
        out.push_str(&format!("... (+{remaining} more)"));
    }
    out
}

fn parse_ip_candidate(value: &str) -> Option<IpAddr> {
    let trimmed = value
        .trim()
        .trim_matches(|c| matches!(c, '"' | '\'' | '[' | ']'));
    if let Ok(ip) = trimmed.parse::<IpAddr>() {
        return Some(ip);
    }

    let colon_count = trimmed.chars().filter(|&ch| ch == ':').count();
    if colon_count == 1 {
        if let Some((host, port)) = trimmed.rsplit_once(':') {
            if port.chars().all(|ch| ch.is_ascii_digit()) {
                return host.parse::<IpAddr>().ok();
            }
        }
    }

    None
}

fn load_vpn_ranges() -> Result<Vec<CompiledVpnRange>> {
    let raw: VpnRangesFile = serde_json::from_str(VPN_RANGES_JSON)?;
    if raw.schema_version != 1 {
        bail!("unsupported VPN range schemaVersion {}", raw.schema_version);
    }
    raw.ranges
        .into_iter()
        .map(|record| compile_vpn_range(&record.cidr, &record.label))
        .collect()
}

fn compile_vpn_range(cidr: &str, label: &str) -> Result<CompiledVpnRange> {
    let (network, prefix) = parse_ipv4_cidr(cidr)?;
    let mask = prefix_mask(prefix);
    Ok(CompiledVpnRange {
        network: ipv4_to_u32(network) & mask,
        mask,
        label: label.to_string(),
    })
}

fn classify_ipv4(ip: Ipv4Addr, ranges: &[CompiledVpnRange]) -> Option<&str> {
    let raw = ipv4_to_u32(ip);
    ranges
        .iter()
        .find(|range| raw & range.mask == range.network)
        .map(|range| range.label.as_str())
}

fn is_groupable_private_or_reserved_ipv4(ip: Ipv4Addr) -> bool {
    matches!(
        ip.octets(),
        [10, _, _, _] | [172, 16..=31, _, _] | [192, 168, _, _] | [127, _, _, _] | [169, 254, _, _]
    )
}

fn ipv4_24_network(ip: Ipv4Addr) -> u32 {
    ipv4_to_u32(ip) & prefix_mask(24)
}

fn ipv4_from_u32(raw: u32) -> Ipv4Addr {
    Ipv4Addr::from(raw.to_be_bytes())
}

pub fn ipv4_in_cidr(ip: Ipv4Addr, cidr: &str) -> Result<bool> {
    let (network, prefix) = parse_ipv4_cidr(cidr)?;
    let mask = prefix_mask(prefix);
    Ok(ipv4_to_u32(ip) & mask == ipv4_to_u32(network) & mask)
}

fn parse_ipv4_cidr(cidr: &str) -> Result<(Ipv4Addr, u8)> {
    let (network, prefix) = cidr
        .split_once('/')
        .ok_or_else(|| anyhow!("CIDR range is missing a prefix length: {cidr}"))?;
    let network = network
        .parse::<Ipv4Addr>()
        .with_context(|| format!("parsing IPv4 CIDR network {cidr}"))?;
    let prefix = prefix
        .parse::<u8>()
        .with_context(|| format!("parsing IPv4 CIDR prefix {cidr}"))?;
    if prefix > 32 {
        bail!("IPv4 CIDR prefix is outside 0..=32: {cidr}");
    }
    Ok((network, prefix))
}

fn prefix_mask(prefix: u8) -> u32 {
    if prefix == 0 {
        0
    } else {
        u32::MAX << (32 - u32::from(prefix))
    }
}

fn ipv4_to_u32(ip: Ipv4Addr) -> u32 {
    u32::from_be_bytes(ip.octets())
}

#[cfg(test)]
mod tests {
    use super::*;
    use calamine::Reader;

    fn setup_report_fixture(with_matches: bool) -> (Connection, Vec<ColumnMeta>) {
        let conn = Connection::open_in_memory().unwrap();
        let columns = vec![
            ColumnMeta {
                sql_name: "timegenerated".into(),
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
            ColumnMeta {
                sql_name: "device_name".into(),
                original_name: "DeviceName".into(),
                col_index: 2,
                inferred_type: "text".into(),
            },
            ColumnMeta {
                sql_name: "source_ip".into(),
                original_name: "SourceIP".into(),
                col_index: 3,
                inferred_type: "text".into(),
            },
            ColumnMeta {
                sql_name: "processcommandline".into(),
                original_name: "ProcessCommandLine".into(),
                col_index: 4,
                inferred_type: "text".into(),
            },
            ColumnMeta {
                sql_name: "user_agent".into(),
                original_name: "UserAgent".into(),
                col_index: 5,
                inferred_type: "text".into(),
            },
        ];
        db::create_schema(&conn, &columns).unwrap();
        let rows = [
            (
                "2026-01-01T00:01:00Z",
                "CORP\\alice",
                "WKSTN-01",
                "104.16.1.1",
                "powershell.exe -nop -enc SQBFAFgA",
                "Mozilla/5.0 Chrome/120.0 Safari/537.36",
            ),
            (
                "2026-01-01T00:02:00Z",
                "CORP\\alice",
                "WKSTN-02",
                "8.8.8.8",
                "mimikatz sekurlsa::logonpasswords",
                "Mozilla/5.0 Firefox/120.0",
            ),
            (
                "2026-01-01T00:03:00Z",
                "CORP\\bob",
                "WKSTN-02",
                "45.33.1.2",
                "whoami",
                "Mozilla/5.0 Version/17.0 Safari/605.1.15",
            ),
        ];
        for (idx, row) in rows.iter().enumerate() {
            conn.execute(
                "INSERT INTO rows (
                    row_num,
                    timegenerated,
                    account,
                    device_name,
                    source_ip,
                    processcommandline,
                    user_agent
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![(idx as i64) + 1, row.0, row.1, row.2, row.3, row.4, row.5],
            )
            .unwrap();
        }

        db::create_column_roles_table(&conn).unwrap();
        for (role, sql_name) in [
            ("timestamp", "timegenerated"),
            ("user", "account"),
            ("host", "device_name"),
            ("ip", "source_ip"),
            ("command_line", "processcommandline"),
        ] {
            conn.execute(
                "INSERT INTO _column_roles (role, sql_name, confidence, status, reasons_json)
                 VALUES (?1, ?2, 1.0, 'confirmed', '[]')",
                params![role, sql_name],
            )
            .unwrap();
        }

        db::create_row_time_table(&conn).unwrap();
        for (row_num, epoch_ms, utc_text) in [
            (1i64, 1_767_225_660_000i64, "2026-01-01T00:01:00Z"),
            (2, 1_767_225_720_000, "2026-01-01T00:02:00Z"),
            (3, 1_767_225_780_000, "2026-01-01T00:03:00Z"),
        ] {
            conn.execute(
                "INSERT INTO _row_time (row_num, epoch_ms, utc_text, source_text, parse_status)
                 VALUES (?1, ?2, ?3, ?3, 'explicit_offset')",
                params![row_num, epoch_ms, utc_text],
            )
            .unwrap();
        }

        db::create_intel_schema(&conn).unwrap();
        conn.execute(
            "INSERT INTO _intel_scan_info (library_hash, role_hash, completed_at)
             VALUES ('test-library', 'test-roles', '2026-01-01T00:00:00Z')",
            [],
        )
        .unwrap();
        if with_matches {
            conn.execute(
                "INSERT INTO _intel_match (
                    row_num,
                    tactic_id,
                    tactic_name,
                    technique_id,
                    technique_name,
                    pattern_id,
                    keyword,
                    column_name,
                    score
                 ) VALUES
                    (1, 'TA0002', 'Execution', 'T1059.001', 'PowerShell', 'p1', 'powershell -enc', 'processcommandline', 90),
                    (2, 'TA0006', 'Credential Access', 'T1003', 'OS Credential Dumping', 'p2', 'mimikatz', 'processcommandline', 95)",
                [],
            )
            .unwrap();
        }

        (conn, columns)
    }

    fn create_llm_audit_fixture(conn: &Connection, with_dataset_identity: bool) {
        conn.execute_batch(
            "CREATE TABLE _llm_parse_audit (
                id INTEGER PRIMARY KEY,
                provider TEXT NOT NULL,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                model_sha256 TEXT NOT NULL,
                tokenizer_sha256 TEXT NOT NULL,
                prompt_template_version TEXT NOT NULL,
                correlation_engine_version TEXT NOT NULL,
                artifact_ids_json TEXT NOT NULL,
                input_sha256 TEXT NOT NULL,
                generation_parameters_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                load_time_ms INTEGER NOT NULL,
                inference_latency_ms INTEGER NOT NULL,
                raw_output TEXT NOT NULL,
                validation_status TEXT NOT NULL,
                validation_detail TEXT,
                trusted_intent_json TEXT NOT NULL,
                examiner_decision TEXT NOT NULL,
                decided_at TEXT
             );
             INSERT INTO _llm_parse_audit VALUES (
                42, 'local-candle', 'Qwen2.5-1.5B-Instruct', 'Q4_K_M@revision',
                'model-hash', 'tokenizer-hash', 'guided-intent-v2',
                'intel-library:test;matcher:v1',
                '{\"techniqueIds\":[\"T1003.001\"]}', 'input-hash',
                '{\"strategy\":\"greedy_argmax\"}', '2026-07-16T00:00:00Z',
                2945, 13182, '=untrusted model text', 'validated', NULL,
                '{\"intent\":\"techniqueTimeline\"}', 'accepted',
                '2026-07-16T00:01:00Z'
             );",
        )
        .unwrap();
        if with_dataset_identity {
            conn.execute_batch(
                "ALTER TABLE _llm_parse_audit ADD COLUMN dataset_schema_sha256 TEXT;
                 ALTER TABLE _llm_parse_audit ADD COLUMN dataset_import_sha256 TEXT;
                 UPDATE _llm_parse_audit
                 SET dataset_schema_sha256 = 'dataset-schema-hash',
                     dataset_import_sha256 = 'dataset-import-hash';",
            )
            .unwrap();
        }
    }

    fn create_semantic_retrieval_audit_fixture(conn: &Connection) {
        conn.execute_batch(
            "CREATE TABLE _semantic_retrieval_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                llm_audit_id INTEGER,
                input_sha256 TEXT NOT NULL,
                dataset_schema_sha256 TEXT NOT NULL,
                dataset_import_sha256 TEXT NOT NULL,
                semantic_used INTEGER NOT NULL,
                outcome_code TEXT NOT NULL,
                detail TEXT NOT NULL,
                selection_id TEXT,
                created_at TEXT NOT NULL
             );
             INSERT INTO _semantic_retrieval_audit (
                llm_audit_id, input_sha256, dataset_schema_sha256, dataset_import_sha256,
                semantic_used, outcome_code, detail, selection_id, created_at
             ) VALUES
                (42, 'input-one', 'schema-one', 'import-one', 1, 'applied',
                 '=literal semantic match detail', 'selection-one', '2026-07-17T01:00:00Z'),
                (43, 'input-two', 'schema-two', 'import-two', 0, 'no_candidates',
                 'No semantic candidates met the bounded policy', NULL,
                 '2026-07-17T01:01:00Z');",
        )
        .unwrap();
    }

    fn temp_report_path(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "log-parser-report-test-{}-{name}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("report.xlsx")
    }

    fn workbook_sheet_names(path: &Path) -> Vec<String> {
        let workbook: calamine::Sheets<std::io::BufReader<std::fs::File>> =
            calamine::open_workbook_auto(path).unwrap();
        workbook.sheet_names().to_vec()
    }

    fn cell_to_i64(cell: &calamine::Data) -> i64 {
        match cell {
            calamine::Data::Int(value) => *value,
            calamine::Data::Float(value) => *value as i64,
            _ => cell.to_string().parse::<i64>().unwrap(),
        }
    }

    fn cell_to_f64(cell: &calamine::Data) -> f64 {
        match cell {
            calamine::Data::Int(value) => *value as f64,
            calamine::Data::Float(value) => *value,
            _ => cell.to_string().parse::<f64>().unwrap(),
        }
    }

    fn data_row_nums(range: &calamine::Range<calamine::Data>) -> Vec<i64> {
        range
            .rows()
            .skip(1)
            .map(|row| cell_to_i64(&row[0]))
            .collect()
    }

    fn setup_ip_rollup_fixture(ip_values: &[&str]) -> (Connection, RoleColumn) {
        let conn = Connection::open_in_memory().unwrap();
        let columns = vec![ColumnMeta {
            sql_name: "source_ip".into(),
            original_name: "SourceIP".into(),
            col_index: 0,
            inferred_type: "text".into(),
        }];
        db::create_schema(&conn, &columns).unwrap();
        for (idx, value) in ip_values.iter().enumerate() {
            conn.execute(
                "INSERT INTO rows (row_num, source_ip) VALUES (?1, ?2)",
                params![(idx as i64) + 1, value],
            )
            .unwrap();
        }

        (
            conn,
            RoleColumn {
                sql_name: "source_ip".into(),
                original_name: "SourceIP".into(),
            },
        )
    }

    fn write_ip_rollup_sheet(
        ip_values: &[&str],
        ranges: &[CompiledVpnRange],
        name: &str,
    ) -> std::path::PathBuf {
        let (conn, role_column) = setup_ip_rollup_fixture(ip_values);
        let path = temp_report_path(name);
        let mut workbook = Workbook::new();
        {
            let worksheet = workbook.add_worksheet_with_constant_memory();
            worksheet.set_name("General").unwrap();
            write_headers(
                worksheet,
                &[
                    "row_num",
                    "section",
                    "item",
                    "value",
                    "detail",
                    "observed_count",
                ],
            )
            .unwrap();
            let mut source_rows = HashSet::new();
            let mut total_rows_written = 0i64;
            let mut progress = |_: i64, _: &str| {};
            let mut writer = RowWriter::new(
                worksheet,
                "General",
                &mut source_rows,
                &mut total_rows_written,
                &mut progress,
            );
            write_ip_rollups(&conn, &mut writer, Some(&role_column), ranges).unwrap();
            writer.finish_sheet();
        }
        workbook.save(&path).unwrap();
        path
    }

    #[derive(Debug)]
    struct IpRollupRow {
        row_num: i64,
        item: String,
        value: String,
        detail: String,
        count: i64,
    }

    fn ip_rollup_rows(path: &Path) -> Vec<IpRollupRow> {
        let mut workbook: calamine::Sheets<std::io::BufReader<std::fs::File>> =
            calamine::open_workbook_auto(path).unwrap();
        let general = workbook
            .worksheet_range("General")
            .expect("General sheet should exist");
        general
            .rows()
            .skip(1)
            .filter(|row| row[1] == "IP addresses")
            .map(|row| IpRollupRow {
                row_num: cell_to_i64(&row[0]),
                item: row[2].to_string(),
                value: row[3].to_string(),
                detail: row[4].to_string(),
                count: cell_to_i64(&row[5]),
            })
            .collect()
    }

    #[test]
    fn general_sheet_groups_unmatched_private_ipv4s_by_24() {
        let path = write_ip_rollup_sheet(
            &["10.20.30.1", "10.20.30.2", "10.20.30.1", "10.20.30.200"],
            &[],
            "ip-private-one-bucket",
        );

        let rows = ip_rollup_rows(&path);
        assert_eq!(rows.len(), 1, "expected one /24 bucket row, got {rows:?}");
        assert_eq!(rows[0].row_num, 1);
        assert_eq!(rows[0].item, "SourceIP");
        assert_eq!(rows[0].value, "10.20.30.0/24 (private/reserved range)");
        assert_eq!(rows[0].count, 4);
        assert!(
            rows[0]
                .detail
                .contains("Distinct private/reserved IPs in this /24: 3"),
            "detail should report distinct IP count, got: {}",
            rows[0].detail
        );
        assert!(
            rows[0].detail.contains("total observations: 4"),
            "detail should report total observations, got: {}",
            rows[0].detail
        );

        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn general_sheet_keeps_private_vpn_matches_individual() {
        let ranges = vec![compile_vpn_range("10.10.42.8/32", "Test private VPN").unwrap()];
        let path = write_ip_rollup_sheet(
            &["10.10.42.8", "10.10.42.9"],
            &ranges,
            "ip-private-match-individual",
        );

        let rows = ip_rollup_rows(&path);
        assert_eq!(
            rows.len(),
            2,
            "expected one matched individual row and one private bucket row, got {rows:?}"
        );
        assert_eq!(rows[0].row_num, 1);
        assert_eq!(rows[0].value, "10.10.42.8");
        assert_eq!(rows[0].count, 1);
        assert!(
            rows[0]
                .detail
                .contains("possible VPN/hosting: Test private VPN"),
            "matched private IP should keep the individual VPN detail, got: {}",
            rows[0].detail
        );
        assert_eq!(rows[1].row_num, 2);
        assert_eq!(rows[1].value, "10.10.42.0/24 (private/reserved range)");
        assert_eq!(rows[1].count, 1);

        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn general_sheet_keeps_unmatched_public_ipv4_individual() {
        let path = write_ip_rollup_sheet(&["8.8.8.8", "8.8.8.8"], &[], "ip-public-individual");

        let rows = ip_rollup_rows(&path);
        assert_eq!(rows.len(), 1, "expected one individual public IP row");
        assert_eq!(rows[0].row_num, 1);
        assert_eq!(rows[0].value, "8.8.8.8");
        assert_eq!(
            rows[0].detail,
            "no match in bundled offline VPN/hosting ranges"
        );
        assert_eq!(rows[0].count, 2);

        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn general_sheet_splits_private_ipv4s_across_24_buckets() {
        let path = write_ip_rollup_sheet(
            &["10.1.1.1", "10.1.1.2", "10.1.2.1", "10.1.2.5", "10.1.1.1"],
            &[],
            "ip-private-two-buckets",
        );

        let rows = ip_rollup_rows(&path);
        assert_eq!(
            rows.len(),
            2,
            "expected two separate /24 buckets, got {rows:?}"
        );
        assert_eq!(rows[0].row_num, 1);
        assert_eq!(rows[0].value, "10.1.1.0/24 (private/reserved range)");
        assert_eq!(rows[0].count, 3);
        assert!(rows[0]
            .detail
            .contains("Distinct private/reserved IPs in this /24: 2"));
        assert_eq!(rows[1].row_num, 3);
        assert_eq!(rows[1].value, "10.1.2.0/24 (private/reserved range)");
        assert_eq!(rows[1].count, 2);
        assert!(rows[1]
            .detail
            .contains("Distinct private/reserved IPs in this /24: 2"));

        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn report_export_writes_dynamic_sheets_and_source_row_numbers() {
        let (mut conn, columns) = setup_report_fixture(true);
        let path = temp_report_path("with-matches");

        let summary = export_report(&mut conn, &columns, &path, |_, _| {}).unwrap();

        assert_eq!(
            summary.sheets_written,
            vec!["General", "Timeline", "Credential Access", "Execution"]
        );
        assert_eq!(summary.row_count, 3);

        let mut workbook = calamine::open_workbook_auto(&path).unwrap();
        let sheet_names = workbook.sheet_names().to_vec();
        assert_eq!(
            sheet_names,
            vec!["General", "Timeline", "Credential Access", "Execution"]
        );
        let category_sheets = sheet_names
            .iter()
            .filter(|name| !matches!(name.as_str(), "General" | "Timeline"))
            .count();
        assert_eq!(category_sheets, 2);

        for sheet_name in &sheet_names {
            let range = workbook.worksheet_range(sheet_name).unwrap();
            let row_nums = data_row_nums(&range);
            assert!(!row_nums.is_empty(), "{sheet_name} should not be empty");
            for row_num in row_nums {
                assert!(
                    (1..=3).contains(&row_num),
                    "{sheet_name} had non-source row_num {row_num}"
                );
            }
        }

        let timeline = workbook.worksheet_range("Timeline").unwrap();
        assert_eq!(data_row_nums(&timeline), vec![1, 2]);
        let timeline_rows: Vec<_> = timeline.rows().collect();
        assert_eq!(
            timeline_rows[0][6].to_string(),
            "matched_column",
            "Timeline sheet must include a matched_column header"
        );
        assert_eq!(
            timeline_rows[0][7].to_string(),
            "evidence",
            "Timeline sheet must include an evidence header"
        );
        assert_eq!(timeline_rows[1][6].to_string(), "processcommandline");
        assert!(
            timeline_rows[1][7].to_string().contains("powershell"),
            "evidence cell should contain the actual matched command line, got: {}",
            timeline_rows[1][7]
        );
        let credential_access = workbook.worksheet_range("Credential Access").unwrap();
        assert_eq!(data_row_nums(&credential_access), vec![2]);
        let execution = workbook.worksheet_range("Execution").unwrap();
        assert_eq!(data_row_nums(&execution), vec![1]);

        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn report_uses_one_sqlite_snapshot_across_all_sheets() {
        let report_path = temp_report_path("point-in-time-snapshot");
        let db_path = report_path.parent().unwrap().join("evidence.sqlite3");
        let columns = vec![ColumnMeta {
            sql_name: "message".into(),
            original_name: "Message".into(),
            col_index: 0,
            inferred_type: "text".into(),
        }];
        let mut conn = Connection::open(&db_path).unwrap();
        conn.execute_batch("PRAGMA journal_mode = WAL;").unwrap();
        db::create_schema(&conn, &columns).unwrap();
        conn.execute(
            "INSERT INTO rows(row_num, message) VALUES (1, 'initial evidence')",
            [],
        )
        .unwrap();
        create_llm_audit_fixture(&conn, false);

        let (snapshot_ready_tx, snapshot_ready_rx) = std::sync::mpsc::channel();
        let (writer_done_tx, writer_done_rx) = std::sync::mpsc::channel();
        let export_path = report_path.clone();
        let export_columns = columns.clone();
        let export = std::thread::spawn(move || {
            let mut signalled = false;
            export_report(&mut conn, &export_columns, &export_path, |_, sheet| {
                if sheet == "General" && !signalled {
                    signalled = true;
                    snapshot_ready_tx.send(()).unwrap();
                    writer_done_rx
                        .recv_timeout(std::time::Duration::from_secs(10))
                        .unwrap();
                }
            })
            .unwrap()
        });

        snapshot_ready_rx
            .recv_timeout(std::time::Duration::from_secs(10))
            .unwrap();
        let writer = Connection::open(&db_path).unwrap();
        writer
            .busy_timeout(std::time::Duration::from_secs(3))
            .unwrap();
        let write_result = writer.execute_batch(
            "INSERT INTO _llm_parse_audit
             SELECT 43, provider, model_name, model_version, model_sha256, tokenizer_sha256,
                    prompt_template_version, correlation_engine_version, artifact_ids_json,
                    'concurrent-input', generation_parameters_json,
                    '2026-07-17T02:00:00Z', load_time_ms, inference_latency_ms,
                    'concurrent model output', validation_status, validation_detail,
                    trusted_intent_json, examiner_decision, '2026-07-17T02:00:01Z'
             FROM _llm_parse_audit WHERE id = 42;",
        );
        writer_done_tx.send(()).unwrap();
        write_result.unwrap();
        let summary = export.join().unwrap();
        assert!(summary.sheets_written.contains(&"AI Audit".to_string()));

        let mut workbook = calamine::open_workbook_auto(&report_path).unwrap();
        let audit = workbook.worksheet_range("AI Audit").unwrap();
        let rows = audit.rows().collect::<Vec<_>>();
        assert_eq!(
            rows.len(),
            2,
            "concurrent audit must be outside the report snapshot"
        );
        assert_eq!(cell_to_i64(&rows[1][0]), 42);

        let database_audits: i64 = writer
            .query_row("SELECT COUNT(*) FROM _llm_parse_audit", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(
            database_audits, 2,
            "writer must really commit during export"
        );
        drop(workbook);
        drop(writer);
        let _ = std::fs::remove_dir_all(report_path.parent().unwrap());
    }

    #[test]
    fn report_export_writes_complete_ai_and_semantic_audit_sheets() {
        let (mut conn, columns) = setup_report_fixture(true);
        create_llm_audit_fixture(&conn, true);
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS _semantic_v2_audit_snapshot (
                selection_id INTEGER PRIMARY KEY,
                snapshot_version TEXT,
                build_id INTEGER,
                dataset_hash TEXT,
                schema_hash TEXT,
                index_version TEXT,
                normalizer_version TEXT,
                model_name TEXT,
                model_version TEXT,
                model_sha256 TEXT,
                tokenizer_sha256 TEXT,
                config_sha256 TEXT,
                query_sha256 TEXT,
                policy_version TEXT,
                minimum_score REAL,
                maximum_documents INTEGER,
                documents_above_threshold INTEGER,
                documents_retained INTEGER,
                rows_matched INTEGER,
                documents_truncated INTEGER,
                broad_row_warning INTEGER,
                warnings_json TEXT,
                source_rows INTEGER,
                index_rows_scanned INTEGER,
                index_documents_seen INTEGER,
                index_documents_embedded INTEGER,
                index_documents_mapped INTEGER,
                index_mappings_written INTEGER,
                index_documents_skipped INTEGER,
                index_mappings_skipped INTEGER,
                index_cells_truncated INTEGER,
                index_columns_omitted INTEGER,
                index_chunks_omitted INTEGER,
                candidate_documents INTEGER,
                candidate_mappings INTEGER,
                candidate_document_limit INTEGER,
                candidate_mapping_limit INTEGER,
                selected_document_count INTEGER,
                mapping_count INTEGER,
                mapping_sha256 TEXT,
                row_count INTEGER,
                row_set_sha256 TEXT,
                row_set_encoding TEXT,
                mapping_chunk_count INTEGER,
                row_chunk_count INTEGER,
                sealed INTEGER,
                seal_version TEXT,
                selection_created_at TEXT,
                archived_at TEXT
             );
             CREATE TABLE IF NOT EXISTS _semantic_v2_audit_snapshot_document (
                selection_id INTEGER,
                rank INTEGER,
                source_doc_id INTEGER,
                fingerprint_sha256 TEXT,
                kind TEXT,
                column_key TEXT,
                normalized_text TEXT,
                cosine_score REAL,
                rank_score REAL,
                mapping_count INTEGER,
                mapping_sha256 TEXT,
                PRIMARY KEY (selection_id, rank)
             );
             CREATE TABLE IF NOT EXISTS _semantic_v2_audit_snapshot_row_chunk (
                selection_id INTEGER,
                chunk_index INTEGER,
                first_row_num INTEGER,
                last_row_num INTEGER,
                row_count INTEGER,
                encoded_rows BLOB,
                chunk_sha256 TEXT,
                PRIMARY KEY (selection_id, chunk_index)
             );
             CREATE TABLE IF NOT EXISTS _semantic_v2_audit_snapshot_mapping_chunk (
                selection_id INTEGER,
                chunk_index INTEGER,
                source_doc_id INTEGER,
                first_row_num INTEGER,
                last_row_num INTEGER,
                row_count INTEGER,
                encoded_rows BLOB,
                chunk_sha256 TEXT,
                PRIMARY KEY (selection_id, chunk_index)
             );
             INSERT INTO _semantic_v2_audit_snapshot (
                selection_id, snapshot_version, build_id, dataset_hash, schema_hash,
                index_version, normalizer_version, model_name, model_version, model_sha256,
                tokenizer_sha256, config_sha256, query_sha256, policy_version, minimum_score,
                maximum_documents, documents_above_threshold, documents_retained, rows_matched,
                documents_truncated, broad_row_warning, warnings_json, source_rows,
                index_rows_scanned, index_documents_seen, index_documents_embedded,
                index_documents_mapped, index_mappings_written, index_documents_skipped,
                index_mappings_skipped, index_cells_truncated, index_columns_omitted,
                index_chunks_omitted, candidate_documents, candidate_mappings,
                candidate_document_limit, candidate_mapping_limit, selected_document_count,
                mapping_count, mapping_sha256, row_count, row_set_sha256, row_set_encoding,
                mapping_chunk_count, row_chunk_count, sealed, seal_version,
                selection_created_at, archived_at
             ) VALUES (
                9001, 'semantic-audit-snapshot-v2', 77, 'dataset-hash', 'schema-hash',
                'semantic-document-v3', 'dfir-cell-normalizer-v3', 'all-MiniLM-L6-v2',
                'onnx@revision', 'model-hash', 'tokenizer-hash', 'config-hash', 'query-hash',
                'semantic-ranking-v2', 0.42, 250, 10, 4, 7, 0, 1,
                '[\"bounded evidence\"]', 3, 3, 12, 12, 12, 14, 0, 0, 1, 2, 3, 12, 14,
                100000, 6000000, 4, 14, 'mapping-hash', 7, 'row-set-hash',
                'delta-varint-v1', 1, 1, 1, 'semantic-audit-seal-v1',
                '2026-07-16T00:00:30Z', '2026-07-16T00:02:00Z'
             );
             INSERT INTO _semantic_v2_audit_snapshot_document (
                selection_id, rank, source_doc_id, fingerprint_sha256, kind, column_key,
                normalized_text, cosine_score, rank_score, mapping_count, mapping_sha256
             ) VALUES (
                9001, 1, 123, 'fingerprint-hash', 'cell', 'processcommandline',
                '=powershell -enc', 0.875, 0.8125, 3, 'document-mapping-hash'
             );
             INSERT INTO _semantic_v2_audit_snapshot_row_chunk (
                selection_id, chunk_index, first_row_num, last_row_num, row_count,
                encoded_rows, chunk_sha256
             ) VALUES (9001, 0, 1, 3000, 3, x'00017Fff', 'chunk-hash');
             INSERT INTO _semantic_v2_audit_snapshot_mapping_chunk (
                selection_id, chunk_index, source_doc_id, first_row_num, last_row_num,
                row_count, encoded_rows, chunk_sha256
             ) VALUES (9001, 0, 123, 1, 3000, 3, x'00017Fff', 'mapping-chunk-hash');
             INSERT INTO _semantic_v2_audit_snapshot(selection_id, sealed)
                VALUES (9002, 0);
             INSERT INTO _semantic_v2_audit_snapshot_document
                SELECT 9002, rank, source_doc_id, fingerprint_sha256, kind, column_key,
                       normalized_text, cosine_score, rank_score, mapping_count, mapping_sha256
                FROM _semantic_v2_audit_snapshot_document WHERE selection_id = 9001;
             INSERT INTO _semantic_v2_audit_snapshot_mapping_chunk
                SELECT 9002, chunk_index, source_doc_id, first_row_num, last_row_num,
                       row_count, encoded_rows, chunk_sha256
                FROM _semantic_v2_audit_snapshot_mapping_chunk WHERE selection_id = 9001;
             INSERT INTO _semantic_v2_audit_snapshot_row_chunk
                SELECT 9002, chunk_index, first_row_num, last_row_num, row_count,
                       encoded_rows, chunk_sha256
                FROM _semantic_v2_audit_snapshot_row_chunk WHERE selection_id = 9001;
             INSERT INTO _semantic_v2_audit_snapshot
                SELECT 9003, 'semantic-audit-snapshot-v1', build_id, dataset_hash, schema_hash,
                       index_version, normalizer_version, model_name, model_version, model_sha256,
                       tokenizer_sha256, config_sha256, query_sha256, policy_version,
                       minimum_score, maximum_documents, documents_above_threshold,
                       documents_retained, rows_matched, documents_truncated, broad_row_warning,
                       warnings_json, source_rows, index_rows_scanned, index_documents_seen,
                       index_documents_embedded, index_documents_mapped, index_mappings_written,
                       index_documents_skipped, index_mappings_skipped, index_cells_truncated,
                       index_columns_omitted, index_chunks_omitted, candidate_documents,
                       candidate_mappings, candidate_document_limit, candidate_mapping_limit,
                       selected_document_count, mapping_count, mapping_sha256, row_count,
                       row_set_sha256, row_set_encoding, 0, 0, 0, '', selection_created_at,
                       archived_at
                FROM _semantic_v2_audit_snapshot WHERE selection_id = 9001;
             INSERT INTO _semantic_v2_audit_snapshot_document
                SELECT 9003, rank, source_doc_id, fingerprint_sha256, kind, column_key,
                       normalized_text, cosine_score, rank_score, mapping_count, mapping_sha256
                FROM _semantic_v2_audit_snapshot_document WHERE selection_id = 9001;
             INSERT INTO _semantic_v2_audit_snapshot_row_chunk
                SELECT 9003, chunk_index, first_row_num, last_row_num, row_count,
                       encoded_rows, chunk_sha256
                FROM _semantic_v2_audit_snapshot_row_chunk WHERE selection_id = 9001;
             CREATE VIEW _semantic_v2_audit_snapshot_complete AS
                SELECT selection_id FROM _semantic_v2_audit_snapshot
                WHERE snapshot_version = 'semantic-audit-snapshot-v2' AND sealed = 1;
             CREATE VIEW _semantic_v2_audit_snapshot_legacy_union AS
                SELECT selection_id FROM _semantic_v2_audit_snapshot
                WHERE snapshot_version = 'semantic-audit-snapshot-v1' AND sealed = 0;",
        )
        .unwrap();
        let path = temp_report_path("with-ai-audit");

        let summary = export_report(&mut conn, &columns, &path, |_, _| {}).unwrap();
        assert_eq!(
            summary.sheets_written,
            vec![
                "General",
                "AI Audit",
                "Semantic Audit",
                "Timeline",
                "Credential Access",
                "Execution"
            ]
        );

        let mut workbook = calamine::open_workbook_auto(&path).unwrap();
        let audit = workbook.worksheet_range("AI Audit").unwrap();
        let rows: Vec<_> = audit.rows().collect();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0][0].to_string(), "audit_id");
        assert_eq!(rows[0][12].to_string(), "dataset_schema_sha256");
        assert_eq!(rows[0][13].to_string(), "dataset_import_sha256");
        assert_eq!(rows[0][21].to_string(), "inference_latency_ms");
        assert_eq!(cell_to_i64(&rows[1][0]), 42);
        assert_eq!(rows[1][2].to_string(), "accepted");
        assert_eq!(rows[1][4].to_string(), "local-candle");
        assert_eq!(rows[1][5].to_string(), "Qwen2.5-1.5B-Instruct");
        assert_eq!(rows[1][12].to_string(), "dataset-schema-hash");
        assert_eq!(rows[1][13].to_string(), "dataset-import-hash");
        assert_eq!(rows[1][18].to_string(), "=untrusted model text");
        assert_eq!(cell_to_i64(&rows[1][20]), 2945);
        assert_eq!(cell_to_i64(&rows[1][21]), 13182);

        let semantic = workbook.worksheet_range("Semantic Audit").unwrap();
        let semantic_rows: Vec<_> = semantic.rows().collect();
        assert_eq!(semantic_rows.len(), 8);
        assert_eq!(
            semantic_rows
                .iter()
                .skip(1)
                .filter(|row| cell_to_i64(&row[1]) == 9001)
                .count(),
            4
        );
        assert_eq!(
            semantic_rows
                .iter()
                .skip(1)
                .filter(|row| cell_to_i64(&row[1]) == 9003)
                .count(),
            3
        );
        assert!(!semantic_rows
            .iter()
            .skip(1)
            .any(|row| cell_to_i64(&row[1]) == 9002));
        let column = |name: &str| {
            semantic_rows[0]
                .iter()
                .position(|cell| cell.to_string() == name)
                .unwrap_or_else(|| panic!("Semantic Audit header {name} should exist"))
        };
        let snapshot = semantic_rows
            .iter()
            .skip(1)
            .find(|row| row[column("record_type")].to_string() == "snapshot")
            .unwrap();
        assert_eq!(cell_to_i64(&snapshot[column("selection_id")]), 9001);
        assert_eq!(snapshot[column("model_sha256")].to_string(), "model-hash");
        assert_eq!(
            snapshot[column("index_version")].to_string(),
            "semantic-document-v3"
        );
        assert_eq!(
            snapshot[column("normalizer_version")].to_string(),
            "dfir-cell-normalizer-v3"
        );
        assert_eq!(cell_to_i64(&snapshot[column("mapping_chunk_count")]), 1);
        assert_eq!(cell_to_i64(&snapshot[column("row_chunk_count")]), 1);
        assert_eq!(cell_to_i64(&snapshot[column("sealed")]), 1);
        assert_eq!(
            snapshot[column("seal_version")].to_string(),
            "semantic-audit-seal-v1"
        );
        assert_eq!(
            snapshot[column("archive_status")].to_string(),
            "sealed_exact_mappings"
        );

        let document = semantic_rows
            .iter()
            .skip(1)
            .find(|row| row[column("record_type")].to_string() == "document")
            .unwrap();
        assert_eq!(
            document[column("normalized_text")].to_string(),
            "=powershell -enc"
        );
        assert!((cell_to_f64(&document[column("cosine_score")]) - 0.875).abs() < f64::EPSILON);
        assert_eq!(
            document[column("mapping_sha256")].to_string(),
            "document-mapping-hash"
        );

        let row_chunk = semantic_rows
            .iter()
            .skip(1)
            .find(|row| row[column("record_type")].to_string() == "row_chunk")
            .unwrap();
        assert_eq!(
            row_chunk[column("encoded_rows_hex")].to_string(),
            "00017fff"
        );
        let mapping_chunk = semantic_rows
            .iter()
            .skip(1)
            .find(|row| row[column("record_type")].to_string() == "mapping_chunk")
            .unwrap();
        assert_eq!(cell_to_i64(&mapping_chunk[column("source_doc_id")]), 123);
        assert_eq!(cell_to_i64(&mapping_chunk[column("mapping_count")]), 3);
        assert_eq!(
            mapping_chunk[column("encoded_rows_hex")].to_string(),
            "00017fff"
        );
        assert_eq!(row_chunk[column("chunk_sha256")].to_string(), "chunk-hash");

        let legacy = semantic_rows
            .iter()
            .skip(1)
            .find(|row| row[column("record_type")].to_string() == "legacy_snapshot_union_only")
            .unwrap();
        assert_eq!(cell_to_i64(&legacy[column("selection_id")]), 9003);
        assert_eq!(
            legacy[column("archive_status")].to_string(),
            "legacy_union_only_mapping_links_unavailable"
        );
        assert!(semantic_rows.iter().skip(1).any(|row| {
            row[column("record_type")].to_string() == "legacy_row_chunk_union_only"
        }));

        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn report_exports_semantic_retrieval_success_and_fallback_outcomes() {
        let (mut conn, columns) = setup_report_fixture(false);
        create_semantic_retrieval_audit_fixture(&conn);
        let path = temp_report_path("semantic-retrieval");

        let summary = export_report(&mut conn, &columns, &path, |_, _| {}).unwrap();
        assert_eq!(
            summary.sheets_written,
            vec!["General", "Semantic Retrieval"]
        );

        let mut workbook = calamine::open_workbook_auto(&path).unwrap();
        let retrieval = workbook.worksheet_range("Semantic Retrieval").unwrap();
        let rows = retrieval.rows().collect::<Vec<_>>();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0][0].to_string(), "retrieval_id");
        assert_eq!(rows[0][9].to_string(), "created_at");
        assert_eq!(cell_to_i64(&rows[1][1]), 42);
        assert_eq!(rows[1][2].to_string(), "input-one");
        assert_eq!(rows[1][3].to_string(), "schema-one");
        assert_eq!(rows[1][4].to_string(), "import-one");
        assert_eq!(cell_to_i64(&rows[1][5]), 1);
        assert_eq!(rows[1][6].to_string(), "applied");
        assert_eq!(rows[1][7].to_string(), "=literal semantic match detail");
        assert_eq!(rows[1][8].to_string(), "selection-one");
        assert_eq!(cell_to_i64(&rows[2][5]), 0);
        assert_eq!(rows[2][6].to_string(), "no_candidates");
        assert_eq!(rows[2][8].to_string(), "");

        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn semantic_retrieval_sheet_is_migration_safe_for_legacy_columns() {
        let (mut conn, columns) = setup_report_fixture(false);
        conn.execute_batch(
            "CREATE TABLE _semantic_retrieval_audit (selection_id TEXT);
             INSERT INTO _semantic_retrieval_audit(selection_id) VALUES ('legacy-selection');",
        )
        .unwrap();
        let path = temp_report_path("legacy-semantic-retrieval");

        let summary = export_report(&mut conn, &columns, &path, |_, _| {}).unwrap();
        assert_eq!(
            summary.sheets_written,
            vec!["General", "Semantic Retrieval"]
        );
        let mut workbook = calamine::open_workbook_auto(&path).unwrap();
        let retrieval = workbook.worksheet_range("Semantic Retrieval").unwrap();
        let rows = retrieval.rows().collect::<Vec<_>>();
        assert_eq!(rows.len(), 2);
        for column in 0..8 {
            assert_eq!(rows[1][column].to_string(), "");
        }
        assert_eq!(rows[1][8].to_string(), "legacy-selection");
        assert_eq!(rows[1][9].to_string(), "");

        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn oversized_semantic_retrieval_evidence_fails_without_replacing_existing_report() {
        let (mut conn, columns) = setup_report_fixture(false);
        create_semantic_retrieval_audit_fixture(&conn);
        conn.execute(
            "UPDATE _semantic_retrieval_audit SET detail = ?1 WHERE id = 1",
            ["x".repeat(EXCEL_STRING_LIMIT + 1)],
        )
        .unwrap();
        let path = temp_report_path("oversized-semantic-retrieval");
        std::fs::write(&path, b"existing examiner report").unwrap();

        let error = export_report(&mut conn, &columns, &path, |_, _| {}).unwrap_err();
        assert!(error.to_string().contains("Excel"));
        assert_eq!(std::fs::read(&path).unwrap(), b"existing examiner report");
        let siblings = std::fs::read_dir(path.parent().unwrap())
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        assert_eq!(siblings, vec!["report.xlsx"]);

        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn ai_audit_dataset_identity_columns_are_blank_for_legacy_tables() {
        let (mut conn, columns) = setup_report_fixture(false);
        create_llm_audit_fixture(&conn, false);
        let path = temp_report_path("legacy-ai-audit");

        let summary = export_report(&mut conn, &columns, &path, |_, _| {}).unwrap();
        assert_eq!(summary.sheets_written, vec!["General", "AI Audit"]);

        let mut workbook = calamine::open_workbook_auto(&path).unwrap();
        let audit = workbook.worksheet_range("AI Audit").unwrap();
        let rows: Vec<_> = audit.rows().collect();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0][12].to_string(), "dataset_schema_sha256");
        assert_eq!(rows[0][13].to_string(), "dataset_import_sha256");
        assert_eq!(rows[1][12].to_string(), "");
        assert_eq!(rows[1][13].to_string(), "");
        assert_eq!(rows[1][18].to_string(), "=untrusted model text");
        assert_eq!(cell_to_i64(&rows[1][20]), 2945);
        assert_eq!(cell_to_i64(&rows[1][21]), 13182);

        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn report_export_with_zero_matches_writes_no_timeline_or_category_sheets() {
        let (mut conn, columns) = setup_report_fixture(false);
        let path = temp_report_path("zero-matches");

        let summary = export_report(&mut conn, &columns, &path, |_, _| {}).unwrap();

        assert_eq!(summary.sheets_written, vec!["General"]);
        let sheet_names = workbook_sheet_names(&path);
        assert_eq!(sheet_names, vec!["General"]);
        let category_sheets = sheet_names
            .iter()
            .filter(|name| !matches!(name.as_str(), "General" | "Timeline"))
            .count();
        assert_eq!(category_sheets, 0);

        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn report_includes_attack_story_and_anomalies_sheets_when_data_exists() {
        let (mut conn, columns) = setup_report_fixture(true);
        conn.execute(
            "INSERT INTO _intel_chain (
                chain_id, host, start_epoch_ms, end_epoch_ms, first_row, last_row,
                tactic_count, event_count, row_count, score,
                tactic_names, technique_names, sample_rows
             ) VALUES (1, NULL, 1767225660000, 1767225720000, 1, 2, 2, 2, 2, 95,
                       '[\"Execution\",\"Credential Access\"]',
                       '[\"PowerShell\",\"OS Credential Dumping\"]', '[1,2]')",
            [],
        )
        .unwrap();
        db::create_anomaly_schema(&conn).unwrap();
        conn.execute_batch(
            "INSERT INTO _anomaly (row_num, category, score, reason, column_name) VALUES
                (1, 'encoded_blob', 70, 'encoded-command indicator', 'processcommandline'),
                (3, 'off_hours', 20, 'activity at night', '');",
        )
        .unwrap();
        let path = temp_report_path("story-and-anomalies");

        let summary = export_report(&mut conn, &columns, &path, |_, _| {}).unwrap();
        assert!(summary.sheets_written.contains(&"Attack Story".to_string()));
        assert!(summary.sheets_written.contains(&"Anomalies".to_string()));

        let mut workbook = calamine::open_workbook_auto(&path).unwrap();
        let story = workbook.worksheet_range("Attack Story").unwrap();
        let story_rows: Vec<Vec<String>> = story
            .rows()
            .skip(1)
            .map(|row| row.iter().map(|cell| cell.to_string()).collect())
            .collect();
        assert_eq!(story_rows[0][2], "chain summary");
        assert!(story_rows[0][7].contains("Execution → Credential Access"));
        let events: Vec<&Vec<String>> = story_rows
            .iter()
            .filter(|row| row[2] == "event")
            .collect();
        assert_eq!(events.len(), 2, "{story_rows:?}");
        assert!(events
            .iter()
            .all(|row| row[0] == "1" || row[0] == "2"), "{events:?}");
        assert!(events[0][7].contains("Execution activity (PowerShell)"));

        let anomalies = workbook.worksheet_range("Anomalies").unwrap();
        let anomaly_rows: Vec<Vec<String>> = anomalies
            .rows()
            .skip(1)
            .map(|row| row.iter().map(|cell| cell.to_string()).collect())
            .collect();
        assert_eq!(anomaly_rows.len(), 2);
        assert_eq!(anomaly_rows[0][2], "Encoded/obfuscated content");
        assert!(anomaly_rows[0][6].contains("powershell.exe -nop -enc"));
        assert_eq!(anomaly_rows[1][2], "Off-hours activity");

        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn report_includes_activity_summary_and_row_by_row_sheets_when_classified() {
        let (mut conn, columns) = setup_report_fixture(true);
        crate::intel::activity::classify_rows(&mut conn, &columns, |_, _, _| {}).unwrap();
        let path = temp_report_path("activity-sheets");

        let summary = export_report(&mut conn, &columns, &path, |_, _| {}).unwrap();
        assert!(summary
            .sheets_written
            .contains(&"Activity Summary".to_string()));
        assert_eq!(
            summary.sheets_written.last().map(String::as_str),
            Some("Row by Row"),
            "the full-file sheet must be written last: {:?}",
            summary.sheets_written
        );

        let mut workbook = calamine::open_workbook_auto(&path).unwrap();
        let row_by_row = workbook.worksheet_range("Row by Row").unwrap();
        let data: Vec<Vec<String>> = row_by_row
            .rows()
            .skip(1)
            .map(|row| row.iter().map(|cell| cell.to_string()).collect())
            .collect();
        // Completeness: every source row appears exactly once, in order.
        let row_nums: Vec<String> = data.iter().map(|row| row[0].clone()).collect();
        assert_eq!(row_nums, vec!["1", "2", "3"]);
        // Fixture rows all carry command lines → process activity, with timestamps.
        assert!(data.iter().all(|row| row[2] == "Process execution"), "{data:?}");
        assert_eq!(data[0][1], "2026-01-01T00:01:00Z");
        // MITRE annotations land on the matched rows only.
        assert!(data[0][4].contains("T1059.001"), "{data:?}");
        assert!(data[1][4].contains("T1003"), "{data:?}");
        assert_eq!(data[2][4], "");

        let summary_sheet = workbook.worksheet_range("Activity Summary").unwrap();
        let summary_rows: Vec<Vec<String>> = summary_sheet
            .rows()
            .skip(1)
            .map(|row| row.iter().map(|cell| cell.to_string()).collect())
            .collect();
        assert_eq!(summary_rows.len(), 1, "{summary_rows:?}");
        assert_eq!(summary_rows[0][1], "Process execution");
        assert_eq!(summary_rows[0][2], "3");
        assert_eq!(summary_rows[0][3], "100.0");

        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn report_export_without_optional_roles_intel_or_time_writes_raw_and_audit_sheets() {
        let (mut conn, columns) = setup_report_fixture(false);
        conn.execute_batch(
            "DROP TABLE _column_roles;
             DROP TABLE _row_time;
             DROP TABLE _intel_match;
             DROP TABLE _intel_scan_info;",
        )
        .unwrap();
        create_llm_audit_fixture(&conn, true);
        let path = temp_report_path("raw-and-audit-only");

        let summary = export_report(&mut conn, &columns, &path, |_, _| {}).unwrap();
        assert_eq!(summary.sheets_written, vec!["General", "AI Audit"]);
        assert_eq!(summary.dest_path, path.display().to_string());

        let mut workbook = calamine::open_workbook_auto(&path).unwrap();
        let general = workbook.worksheet_range("General").unwrap();
        assert!(general.rows().any(|row| {
            row.get(1).is_some_and(|cell| cell == "Date range")
                && row.get(3).is_some_and(|cell| cell == "not available")
        }));
        assert_eq!(workbook.sheet_names(), &["General", "AI Audit"]);

        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn report_export_without_normalized_time_omits_timeline_but_keeps_tactic_evidence() {
        let (mut conn, columns) = setup_report_fixture(true);
        conn.execute("DELETE FROM _row_time", []).unwrap();
        let path = temp_report_path("missing-time");

        let summary = export_report(&mut conn, &columns, &path, |_, _| {}).unwrap();
        assert_eq!(
            summary.sheets_written,
            vec!["General", "Credential Access", "Execution"]
        );

        let mut workbook = calamine::open_workbook_auto(&path).unwrap();
        assert!(!workbook.sheet_names().iter().any(|name| name == "Timeline"));
        let execution = workbook.worksheet_range("Execution").unwrap();
        let rows = execution.rows().collect::<Vec<_>>();
        assert_eq!(rows[0][1].to_string(), "utc_timestamp");
        assert_eq!(rows[1][1].to_string(), "");
        assert!(rows[1][6].to_string().contains("powershell"));

        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn guarded_report_publish_failure_preserves_destination_and_cleans_temporary_file() {
        let (mut conn, columns) = setup_report_fixture(false);
        let path = temp_report_path("publish-rejected");
        std::fs::write(&path, b"existing examiner report").unwrap();

        let error = export_report_guarded(
            &mut conn,
            &columns,
            &path,
            |_, _| {},
            |_, _| bail!("dataset generation changed before report publication"),
        )
        .unwrap_err();

        assert!(error.to_string().contains("dataset generation changed"));
        assert_eq!(std::fs::read(&path).unwrap(), b"existing examiner report");
        let siblings = std::fs::read_dir(path.parent().unwrap())
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        assert_eq!(siblings, vec!["report.xlsx"]);

        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn general_sheet_case_folds_host_values_before_dedup() {
        // Windows hostnames are case-insensitive - "WKSTN-01.corp.local" and
        // "wkstn-01.corp.local" are the same machine, and must roll up into one General-sheet
        // entry, not two "distinct" hosts.
        let mut conn = Connection::open_in_memory().unwrap();
        let columns = vec![
            ColumnMeta {
                sql_name: "timegenerated".into(),
                original_name: "TimeGenerated".into(),
                col_index: 0,
                inferred_type: "timestamp".into(),
            },
            ColumnMeta {
                sql_name: "device_name".into(),
                original_name: "DeviceName".into(),
                col_index: 1,
                inferred_type: "text".into(),
            },
        ];
        db::create_schema(&conn, &columns).unwrap();
        conn.execute(
            "INSERT INTO rows (row_num, timegenerated, device_name) VALUES
                (1, '2026-01-01T00:01:00Z', 'WKSTN-01.corp.local'),
                (2, '2026-01-01T00:02:00Z', 'wkstn-01.corp.local'),
                (3, '2026-01-01T00:03:00Z', 'WKSTN-01.corp.local')",
            [],
        )
        .unwrap();

        db::create_column_roles_table(&conn).unwrap();
        for (role, sql_name) in [("timestamp", "timegenerated"), ("host", "device_name")] {
            conn.execute(
                "INSERT INTO _column_roles (role, sql_name, confidence, status, reasons_json)
                 VALUES (?1, ?2, 1.0, 'confirmed', '[]')",
                params![role, sql_name],
            )
            .unwrap();
        }

        db::create_row_time_table(&conn).unwrap();
        for (row_num, epoch_ms, utc_text) in [
            (1i64, 1_767_225_660_000i64, "2026-01-01T00:01:00Z"),
            (2, 1_767_225_720_000, "2026-01-01T00:02:00Z"),
            (3, 1_767_225_780_000, "2026-01-01T00:03:00Z"),
        ] {
            conn.execute(
                "INSERT INTO _row_time (row_num, epoch_ms, utc_text, source_text, parse_status)
                 VALUES (?1, ?2, ?3, ?3, 'explicit_offset')",
                params![row_num, epoch_ms, utc_text],
            )
            .unwrap();
        }

        db::create_intel_schema(&conn).unwrap();
        conn.execute(
            "INSERT INTO _intel_scan_info (library_hash, role_hash, completed_at)
             VALUES ('test-library', 'test-roles', '2026-01-01T00:00:00Z')",
            [],
        )
        .unwrap();

        let path = temp_report_path("case-fold-hosts");
        export_report(&mut conn, &columns, &path, |_, _| {}).unwrap();

        let mut workbook: calamine::Sheets<std::io::BufReader<std::fs::File>> =
            calamine::open_workbook_auto(&path).unwrap();
        let general = workbook
            .worksheet_range("General")
            .expect("General sheet should exist");
        let host_rows: Vec<_> = general
            .rows()
            .skip(1)
            .filter(|row| row[1] == "Hosts")
            .collect();
        assert_eq!(
            host_rows.len(),
            1,
            "expected one case-folded Hosts row, got {host_rows:?}"
        );
        assert_eq!(
            cell_to_i64(&host_rows[0][5]),
            3,
            "total observed_count across all casings"
        );

        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn ipv4_cidr_containment_matches_expected_ranges() {
        assert!(ipv4_in_cidr(Ipv4Addr::new(104, 16, 1, 1), "104.16.0.0/13").unwrap());
        assert!(!ipv4_in_cidr(Ipv4Addr::new(8, 8, 8, 8), "104.16.0.0/13").unwrap());
        assert!(ipv4_in_cidr(Ipv4Addr::new(10, 10, 10, 10), "0.0.0.0/0").unwrap());
        assert!(ipv4_in_cidr(Ipv4Addr::new(192, 0, 2, 42), "192.0.2.42/32").unwrap());
        assert!(!ipv4_in_cidr(Ipv4Addr::new(192, 0, 2, 43), "192.0.2.42/32").unwrap());
    }
}
