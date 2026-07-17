use crate::db::{self, ColumnMeta};
use crate::export;
use anyhow::{anyhow, bail, Context, Result};
use rusqlite::{params, Connection, OptionalExtension};
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
    conn: &Connection,
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
    conn: &Connection,
    columns: &[ColumnMeta],
    dest_path: &Path,
    on_progress: impl FnMut(i64, &str),
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

    let mut summary = write_report_workbook(conn, columns, &pending.path, on_progress)?;
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

        if table_has_rows(conn, "_semantic_v2_audit_snapshot")? {
            sheets_written.push(write_semantic_audit_sheet(conn, &mut write_state)?);
            used_sheet_names.insert("semantic audit".to_string());
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
    ];
    write_headers(worksheet, &headers)?;
    for column in 0..headers.len() as u16 {
        worksheet.set_column_width(column, 22)?;
    }
    for column in [4u16, 5, 10, 11, 12, 13, 22, 40, 42, 48, 51, 57, 58] {
        worksheet.set_column_width(column, 56)?;
    }

    let mut excel_row = 1u32;
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
                selection_created_at, archived_at
         FROM _semantic_v2_audit_snapshot ORDER BY selection_id",
        &(1u16..=45).collect::<Vec<_>>(),
        &mut excel_row,
        state.total_rows_written,
    )?;
    write_semantic_audit_query(
        conn,
        worksheet,
        "document",
        "SELECT selection_id, mapping_count, mapping_sha256, rank, source_doc_id,
                fingerprint_sha256, kind, column_key, normalized_text, cosine_score, rank_score
         FROM _semantic_v2_audit_snapshot_document ORDER BY selection_id, rank",
        &[1, 39, 40, 46, 47, 48, 49, 50, 51, 52, 53],
        &mut excel_row,
        state.total_rows_written,
    )?;
    write_semantic_audit_query(
        conn,
        worksheet,
        "row_chunk",
        "SELECT selection_id, row_count, chunk_index, first_row_num, last_row_num,
                encoded_rows, chunk_sha256
         FROM _semantic_v2_audit_snapshot_row_chunk ORDER BY selection_id, chunk_index",
        &[1, 41, 54, 55, 56, 57, 58],
        &mut excel_row,
        state.total_rows_written,
    )?;

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
        let (conn, columns) = setup_report_fixture(true);
        let path = temp_report_path("with-matches");

        let summary = export_report(&conn, &columns, &path, |_, _| {}).unwrap();

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
    fn report_export_writes_complete_ai_and_semantic_audit_sheets() {
        let (conn, columns) = setup_report_fixture(true);
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
                selection_created_at, archived_at
             ) VALUES (
                9001, 'semantic-audit-snapshot-v1', 77, 'dataset-hash', 'schema-hash',
                'semantic-document-v3', 'dfir-cell-normalizer-v3', 'all-MiniLM-L6-v2',
                'onnx@revision', 'model-hash', 'tokenizer-hash', 'config-hash', 'query-hash',
                'semantic-ranking-v2', 0.42, 250, 10, 4, 7, 0, 1,
                '[\"bounded evidence\"]', 3, 3, 12, 12, 12, 14, 0, 0, 1, 2, 3, 12, 14,
                100000, 6000000, 4, 14, 'mapping-hash', 7, 'row-set-hash',
                'delta-varint-v1', '2026-07-16T00:00:30Z', '2026-07-16T00:02:00Z'
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
             ) VALUES (9001, 0, 1, 3000, 3, x'00017Fff', 'chunk-hash');",
        )
        .unwrap();
        let path = temp_report_path("with-ai-audit");

        let summary = export_report(&conn, &columns, &path, |_, _| {}).unwrap();
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
        assert_eq!(semantic_rows.len(), 4);
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
        assert_eq!(row_chunk[column("chunk_sha256")].to_string(), "chunk-hash");

        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn ai_audit_dataset_identity_columns_are_blank_for_legacy_tables() {
        let (conn, columns) = setup_report_fixture(false);
        create_llm_audit_fixture(&conn, false);
        let path = temp_report_path("legacy-ai-audit");

        let summary = export_report(&conn, &columns, &path, |_, _| {}).unwrap();
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
        let (conn, columns) = setup_report_fixture(false);
        let path = temp_report_path("zero-matches");

        let summary = export_report(&conn, &columns, &path, |_, _| {}).unwrap();

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
    fn report_export_without_optional_roles_intel_or_time_writes_raw_and_audit_sheets() {
        let (conn, columns) = setup_report_fixture(false);
        conn.execute_batch(
            "DROP TABLE _column_roles;
             DROP TABLE _row_time;
             DROP TABLE _intel_match;
             DROP TABLE _intel_scan_info;",
        )
        .unwrap();
        create_llm_audit_fixture(&conn, true);
        let path = temp_report_path("raw-and-audit-only");

        let summary = export_report(&conn, &columns, &path, |_, _| {}).unwrap();
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
        let (conn, columns) = setup_report_fixture(true);
        conn.execute("DELETE FROM _row_time", []).unwrap();
        let path = temp_report_path("missing-time");

        let summary = export_report(&conn, &columns, &path, |_, _| {}).unwrap();
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
        let (conn, columns) = setup_report_fixture(false);
        let path = temp_report_path("publish-rejected");
        std::fs::write(&path, b"existing examiner report").unwrap();

        let error = export_report_guarded(
            &conn,
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
        let conn = Connection::open_in_memory().unwrap();
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
        export_report(&conn, &columns, &path, |_, _| {}).unwrap();

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
