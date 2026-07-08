use crate::db::{self, ColumnMeta};
use anyhow::{anyhow, bail, Context, Result};
use rusqlite::{params, Connection, OptionalExtension};
use rust_xlsxwriter::{Workbook, Worksheet};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::net::{IpAddr, Ipv4Addr};
use std::path::Path;

const VPN_RANGES_JSON: &str = include_str!("../resources/intel/vpn_ranges.v1.json");
const PROGRESS_EVERY: i64 = 5000;
const EXCEL_STRING_LIMIT: usize = 32_767;

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
    mut on_progress: impl FnMut(i64, &str),
) -> Result<ReportExportSummary> {
    validate_report_prerequisites(conn)?;

    let ranges = load_vpn_ranges()?;
    let roles = load_confirmed_roles(conn, columns)?;
    let tactics = load_tactic_sheets(conn)?;

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

        if intel_match_count(conn)? > 0 {
            sheets_written.push(write_timeline_sheet(conn, &roles, &mut write_state)?);
            used_sheet_names.insert("timeline".to_string());
        }

        for tactic in tactics {
            let sheet_name = unique_sheet_name(&tactic.tactic_name, &mut used_sheet_names);
            write_tactic_sheet(conn, columns, &tactic, &sheet_name, &mut write_state)?;
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

fn validate_report_prerequisites(conn: &Connection) -> Result<()> {
    if !table_has_rows(conn, "_intel_scan_info")? {
        bail!("intel scan results are not available; run scan_intel_matches before exporting the report");
    }
    if !table_has_rows(conn, "_row_time")? {
        bail!("normalized timestamps are not available; run normalize_timestamp_column before exporting the report");
    }
    Ok(())
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

    for (ip_text, rollup) in rollups {
        let ip: IpAddr = ip_text
            .parse()
            .with_context(|| format!("re-parsing normalized IP {ip_text}"))?;
        let detail = match ip {
            IpAddr::V4(ipv4) => match classify_ipv4(ipv4, ranges) {
                Some(label) => format!(
                    "possible VPN/hosting: {label}; best-effort offline heuristic, not authoritative"
                ),
                None => "no match in bundled offline VPN/hosting ranges".to_string(),
            },
            IpAddr::V6(_) => {
                "not checked; bundled VPN/hosting range dataset is IPv4-only".to_string()
            }
        };
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
    roles: &ConfirmedRoles,
    state: &mut ReportWriteState<'_, F>,
) -> Result<String>
where
    F: FnMut(i64, &str),
{
    let sheet_name = "Timeline".to_string();
    let worksheet = state.workbook.add_worksheet_with_constant_memory();
    worksheet.set_name(&sheet_name)?;

    let mut headers = vec![
        "row_num",
        "utc_timestamp",
        "tactic_name",
        "technique_id",
        "technique_name",
        "keyword",
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

    let mut select_exprs = vec![
        "m.row_num".to_string(),
        "COALESCE(rt.utc_text, '')".to_string(),
        "m.tactic_name".to_string(),
        "m.technique_id".to_string(),
        "m.technique_name".to_string(),
        "m.keyword".to_string(),
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
    let sql = format!(
        "SELECT m.row_num,
                COALESCE(rt.utc_text, ''),
                m.technique_id,
                m.technique_name,
                m.keyword,
                m.column_name,
                {evidence_expr}
         FROM _intel_match m
         LEFT JOIN _row_time rt ON rt.row_num = m.row_num
         JOIN rows r ON r.row_num = m.row_num
         WHERE m.tactic_id = ?1
         ORDER BY rt.epoch_ms ASC, m.row_num ASC, m.technique_id ASC, m.keyword ASC, m.column_name ASC"
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

    fn data_row_nums(range: &calamine::Range<calamine::Data>) -> Vec<i64> {
        range
            .rows()
            .skip(1)
            .map(|row| cell_to_i64(&row[0]))
            .collect()
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
        let credential_access = workbook.worksheet_range("Credential Access").unwrap();
        assert_eq!(data_row_nums(&credential_access), vec![2]);
        let execution = workbook.worksheet_range("Execution").unwrap();
        assert_eq!(data_row_nums(&execution), vec![1]);

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
    fn report_export_fails_before_intel_scan() {
        let (conn, columns) = setup_report_fixture(false);
        conn.execute("DELETE FROM _intel_scan_info", []).unwrap();
        let path = temp_report_path("missing-scan");

        let err = export_report(&conn, &columns, &path, |_, _| {}).unwrap_err();
        assert!(err.to_string().contains("run scan_intel_matches"));

        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn report_export_fails_before_timestamp_normalization() {
        let (conn, columns) = setup_report_fixture(false);
        conn.execute("DELETE FROM _row_time", []).unwrap();
        let path = temp_report_path("missing-time");

        let err = export_report(&conn, &columns, &path, |_, _| {}).unwrap_err();
        assert!(err.to_string().contains("run normalize_timestamp_column"));

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
        assert_eq!(cell_to_i64(&host_rows[0][5]), 3, "total observed_count across all casings");

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
