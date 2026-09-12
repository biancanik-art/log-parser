/// Extracts Indicators of Compromise (IOCs) from imported evidence — IP addresses, domains,
/// URLs, email addresses, and user agents — and enriches IP indicators with the bundled
/// VPN/proxy/hosting range heuristic. The module is deliberately offline: all extraction uses
/// pure parsing, no network lookups.
///
/// Usage: call `extract_iocs` with the cache database connection and column metadata to
/// produce a summary of all unique indicators found across the dataset.

use crate::db::{self, ColumnMeta};
use anyhow::Result;
use rusqlite::Connection;
use serde::Serialize;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::net::{IpAddr, Ipv4Addr};

const SCAN_BATCH_ROWS: i64 = 1000;
const PROGRESS_INTERVAL_ROWS: i64 = 5000;
const MAX_IOCS_PER_TYPE: usize = 10_000;

/// Summary of all extracted IOCs from the dataset.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct IocExtractionSummary {
    pub rows_scanned: i64,
    pub ip_indicators: Vec<IpIndicator>,
    pub domain_indicators: Vec<DomainIndicator>,
    pub url_indicators: Vec<UrlIndicator>,
    pub email_indicators: Vec<EmailIndicator>,
    pub user_agent_indicators: Vec<UserAgentIndicator>,
    #[serde(default)]
    pub correlation_indicators: Vec<CorrelationIndicator>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct IpIndicator {
    pub ip: String,
    pub is_private: bool,
    pub vpn_label: Option<String>,
    pub first_row: i64,
    pub occurrence_count: i64,
    pub source_columns: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DomainIndicator {
    pub domain: String,
    pub first_row: i64,
    pub occurrence_count: i64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UrlIndicator {
    pub url: String,
    pub domain: String,
    pub first_row: i64,
    pub occurrence_count: i64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EmailIndicator {
    pub email: String,
    pub first_row: i64,
    pub occurrence_count: i64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UserAgentIndicator {
    pub user_agent: String,
    pub first_row: i64,
    pub occurrence_count: i64,
}

#[derive(Debug, Clone, Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CorrelationIndicator {
    pub kind: String,
    pub value: String,
    pub first_row: i64,
    pub occurrence_count: i64,
    pub source_column: Option<String>,
}

#[derive(Debug)]
struct IpAccumulator {
    first_row: i64,
    count: i64,
    columns: BTreeSet<String>,
}

#[derive(Debug)]
struct SimpleAccumulator {
    first_row: i64,
    count: i64,
}

impl SimpleAccumulator {
    fn new(row: i64) -> Self {
        Self {
            first_row: row,
            count: 1,
        }
    }

    fn increment(&mut self) {
        self.count += 1;
    }
}

// ----- VPN range matching (reuses the bundled vpn_ranges.v1.json) -----

const VPN_RANGES_JSON: &str = include_str!("../../resources/intel/vpn_ranges.v1.json");

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct VpnRangesFile {
    #[allow(dead_code)]
    schema_version: u32,
    ranges: Vec<VpnRangeRecord>,
}

#[derive(Debug, serde::Deserialize)]
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

fn load_vpn_ranges() -> Result<Vec<CompiledVpnRange>> {
    let file: VpnRangesFile = serde_json::from_str(VPN_RANGES_JSON)?;
    let mut compiled = Vec::new();
    for record in file.ranges {
        if let Some(range) = parse_cidr(&record.cidr) {
            compiled.push(CompiledVpnRange {
                network: range.0,
                mask: range.1,
                label: record.label,
            });
        }
    }
    Ok(compiled)
}

fn parse_cidr(cidr: &str) -> Option<(u32, u32)> {
    let (addr_str, prefix_str) = cidr.split_once('/')?;
    let addr: Ipv4Addr = addr_str.parse().ok()?;
    let prefix: u32 = prefix_str.parse().ok()?;
    if prefix > 32 {
        return None;
    }
    let mask = if prefix == 0 {
        0
    } else {
        !0u32 << (32 - prefix)
    };
    let network = u32::from(addr) & mask;
    Some((network, mask))
}

fn check_vpn(ip_str: &str, ranges: &[CompiledVpnRange]) -> Option<String> {
    let addr: Ipv4Addr = ip_str.parse().ok()?;
    let ip_bits = u32::from(addr);
    for range in ranges {
        if ip_bits & range.mask == range.network {
            return Some(range.label.clone());
        }
    }
    None
}

fn is_private_ip(ip_str: &str) -> bool {
    if let Ok(addr) = ip_str.parse::<IpAddr>() {
        match addr {
            IpAddr::V4(v4) => {
                v4.is_private()
                    || v4.is_loopback()
                    || v4.is_link_local()
                    || v4.octets()[0] == 100 && v4.octets()[1] >= 64 && v4.octets()[1] <= 127
            }
            IpAddr::V6(v6) => v6.is_loopback(),
        }
    } else {
        false
    }
}

// ----- Extraction logic -----

/// Extracts IOCs from all rows in the cache database.
pub fn extract_iocs(
    conn: &Connection,
    columns: &[ColumnMeta],
    mut on_progress: impl FnMut(i64, i64, &str),
) -> Result<IocExtractionSummary> {
    let vpn_ranges = load_vpn_ranges().unwrap_or_default();
    let total_rows: i64 = conn.query_row("SELECT COUNT(*) FROM rows", [], |row| row.get(0))?;

    // Identify which columns are likely to contain IOC-relevant data based on roles.
    let role_columns = load_ioc_relevant_roles(conn)?;

    // Scan all text columns for IOCs.
    let text_columns: Vec<&ColumnMeta> = columns.iter().collect();
    let select_idents: Vec<String> = text_columns
        .iter()
        .map(|column| db::quote_ident(&column.sql_name))
        .collect();
    let select_sql = format!(
        "SELECT row_num, {} FROM rows
         WHERE row_num > ?1
         ORDER BY row_num ASC
         LIMIT ?2",
        select_idents.join(", ")
    );

    let mut ips: HashMap<String, IpAccumulator> = HashMap::new();
    let mut domains: HashMap<String, SimpleAccumulator> = HashMap::new();
    let mut urls: HashMap<String, SimpleAccumulator> = HashMap::new();
    let mut emails: HashMap<String, SimpleAccumulator> = HashMap::new();
    let mut user_agents: HashMap<String, SimpleAccumulator> = HashMap::new();
    let mut correlation_acc: HashMap<(String, String), (i64, i64, Option<String>)> = HashMap::new();

    let mut correlation_cols: HashMap<String, &'static str> = HashMap::new();
    let mut json_cols: HashSet<String> = HashSet::new();

    for col in columns {
        if is_json_payload_column(&col.sql_name) || is_json_payload_column(&col.original_name) {
            json_cols.insert(col.sql_name.clone());
        }
        if let Some(kind) = classify_correlation_column(&col.original_name)
            .or_else(|| classify_correlation_column(&col.sql_name))
        {
            correlation_cols.insert(col.sql_name.clone(), kind);
        }
    }

    let mut rows_scanned = 0i64;
    let mut last_row_num = i64::MIN;
    let mut next_progress_at = PROGRESS_INTERVAL_ROWS;

    on_progress(0, total_rows, "extracting IOCs");

    loop {
        let batch = {
            let mut stmt = conn.prepare(&select_sql)?;
            let mut rows = stmt.query(rusqlite::params![last_row_num, SCAN_BATCH_ROWS])?;
            let mut batch = Vec::new();
            while let Some(row) = rows.next()? {
                let row_num: i64 = row.get(0)?;
                let mut values = Vec::with_capacity(text_columns.len());
                for column_idx in 0..text_columns.len() {
                    values.push(row.get::<_, Option<String>>(column_idx + 1)?);
                }
                batch.push((row_num, values));
            }
            batch
        };

        if batch.is_empty() {
            break;
        }

        for (row_num, values) in &batch {
            last_row_num = *row_num;
            rows_scanned += 1;

            for (column_idx, value) in values.iter().enumerate() {
                let Some(cell) = value.as_deref().map(str::trim).filter(|c| !c.is_empty()) else {
                    continue;
                };
                let col_name = &text_columns[column_idx].sql_name;
                let is_ip_column = role_columns.ip_columns.contains(col_name);
                let is_user_agent_column = role_columns.user_agent_columns.contains(col_name);
                let col_lower = col_name.to_ascii_lowercase();
                let is_ua_like = is_user_agent_column
                    || col_lower.contains("useragent")
                    || col_lower.contains("user_agent")
                    || col_lower.contains("user-agent")
                    || col_lower.contains("http_user_agent")
                    || col_lower.contains("caller_agent")
                    || col_lower.contains("request_agent")
                    || col_lower.contains("http_agent")
                    || col_lower.contains("client_app")
                    || col_lower == "ua"
                    || col_lower.ends_with("_ua")
                    || col_lower.starts_with("ua_");

                // IP extraction — prioritize dedicated IP columns but also scan text (excluding user agents).
                if is_ip_column {
                    // The whole cell is likely an IP address.
                    let trimmed = cell.trim();
                    if trimmed.parse::<IpAddr>().is_ok() && ips.len() < MAX_IOCS_PER_TYPE {
                        let entry = ips.entry(trimmed.to_string()).or_insert_with(|| {
                            IpAccumulator {
                                first_row: *row_num,
                                count: 0,
                                columns: BTreeSet::new(),
                            }
                        });
                        entry.count += 1;
                        entry.columns.insert(col_name.clone());
                    }
                } else if !is_ua_like && !cell.contains("Mozilla/") {
                    // Scan for embedded IPs in text fields.
                    for ip in extract_ipv4_addresses(cell) {
                        if ips.len() >= MAX_IOCS_PER_TYPE {
                            break;
                        }
                        let entry = ips.entry(ip).or_insert_with(|| IpAccumulator {
                            first_row: *row_num,
                            count: 0,
                            columns: BTreeSet::new(),
                        });
                        entry.count += 1;
                        entry.columns.insert(col_name.clone());
                    }
                }

                // User agent extraction — from dedicated columns or cells containing known tool/browser signatures.
                if (is_user_agent_column || is_ua_like) && user_agents.len() < MAX_IOCS_PER_TYPE {
                    for ua in extract_user_agents_from_cell(cell) {
                        if user_agents.len() >= MAX_IOCS_PER_TYPE {
                            break;
                        }
                        user_agents
                            .entry(ua.to_string())
                            .and_modify(|a| a.increment())
                            .or_insert_with(|| SimpleAccumulator::new(*row_num));
                    }
                } else if !is_user_agent_column && !is_ua_like && user_agents.len() < MAX_IOCS_PER_TYPE {
                    for ua in extract_user_agents_from_cell(cell) {
                        if crate::intel::roles::is_known_tool_or_browser_ua(ua) {
                            if user_agents.len() >= MAX_IOCS_PER_TYPE {
                                break;
                            }
                            user_agents
                                .entry(ua.to_string())
                                .and_modify(|a| a.increment())
                                .or_insert_with(|| SimpleAccumulator::new(*row_num));
                        }
                    }
                }

                // Email extraction — from any column. Also extract domain from email!
                if emails.len() < MAX_IOCS_PER_TYPE {
                    for email in extract_emails(cell) {
                        if let Some((_, domain)) = email.split_once('@') {
                            let d = domain.trim().to_lowercase();
                            if is_valid_domain(&d) && domains.len() < MAX_IOCS_PER_TYPE {
                                domains
                                    .entry(d)
                                    .and_modify(|a| a.increment())
                                    .or_insert_with(|| SimpleAccumulator::new(*row_num));
                            }
                        }
                        emails
                            .entry(email)
                            .and_modify(|a| a.increment())
                            .or_insert_with(|| SimpleAccumulator::new(*row_num));
                    }
                }

                // URL extraction — from any column.
                if urls.len() < MAX_IOCS_PER_TYPE {
                    for url in extract_urls(cell) {
                        let domain = extract_domain_from_url(&url).unwrap_or_default();
                        if !domain.is_empty() && is_valid_domain(&domain) && domains.len() < MAX_IOCS_PER_TYPE {
                            domains
                                .entry(domain.clone())
                                .and_modify(|a| a.increment())
                                .or_insert_with(|| SimpleAccumulator::new(*row_num));
                        }
                        urls.entry(url)
                            .and_modify(|a| a.increment())
                            .or_insert_with(|| SimpleAccumulator::new(*row_num));
                    }
                }

                // Standalone domain extraction — from non-UA text columns.
                if !is_ua_like && domains.len() < MAX_IOCS_PER_TYPE {
                    for domain in extract_standalone_domains(cell) {
                        if domains.len() >= MAX_IOCS_PER_TYPE {
                            break;
                        }
                        domains
                            .entry(domain)
                            .and_modify(|a| a.increment())
                            .or_insert_with(|| SimpleAccumulator::new(*row_num));
                    }
                }

                // Correlation & Object ID extraction (DeviceID, SessionID, AppID, UniqueTokenID, CorrelationID/RequestID, MailboxGuid, MessageIDs, FileID, Hashes)
                if json_cols.contains(col_name) || cell.starts_with('{') {
                    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(cell) {
                        let mut extracted = Vec::new();
                        extract_correlation_from_json(&parsed, &mut extracted);
                        for (kind, val) in extracted {
                            if correlation_acc.len() < MAX_IOCS_PER_TYPE * 2 {
                                let entry = correlation_acc
                                    .entry((kind, val))
                                    .or_insert((*row_num, 0, Some(col_name.clone())));
                                entry.1 += 1;
                            }
                        }
                    }
                }

                if let Some(&kind) = correlation_cols.get(col_name) {
                    if kind == "hash" {
                        for h in extract_hashes_from_text(cell) {
                            if correlation_acc.len() < MAX_IOCS_PER_TYPE * 2 {
                                let entry = correlation_acc
                                    .entry(("hash".to_string(), h))
                                    .or_insert((*row_num, 0, Some(col_name.clone())));
                                entry.1 += 1;
                            }
                        }
                    } else if is_valid_correlation_value(kind, cell) {
                        if correlation_acc.len() < MAX_IOCS_PER_TYPE * 2 {
                            let entry = correlation_acc
                                .entry((kind.to_string(), cell.trim().to_string()))
                                .or_insert((*row_num, 0, Some(col_name.clone())));
                            entry.1 += 1;
                        }
                    }
                } else if cell.len() >= 32
                    && (cell.contains("MD5=")
                        || cell.contains("SHA256=")
                        || cell.contains("SHA1=")
                        || cell.len() == 64)
                {
                    for h in extract_hashes_from_text(cell) {
                        if correlation_acc.len() < MAX_IOCS_PER_TYPE * 2 {
                            let entry = correlation_acc
                                .entry(("hash".to_string(), h))
                                .or_insert((*row_num, 0, Some(col_name.clone())));
                            entry.1 += 1;
                        }
                    }
                }
            }
        }

        if rows_scanned >= next_progress_at {
            on_progress(rows_scanned, total_rows, "extracting IOCs");
            while next_progress_at <= rows_scanned {
                next_progress_at += PROGRESS_INTERVAL_ROWS;
            }
        }
    }

    on_progress(rows_scanned, total_rows, "complete");

    // Build final sorted indicator lists.
    let mut ip_indicators: Vec<IpIndicator> = ips
        .into_iter()
        .map(|(ip, acc)| {
            let vpn_label = check_vpn(&ip, &vpn_ranges);
            IpIndicator {
                is_private: is_private_ip(&ip),
                vpn_label,
                ip,
                first_row: acc.first_row,
                occurrence_count: acc.count,
                source_columns: acc.columns.into_iter().collect(),
            }
        })
        .collect();
    ip_indicators.sort_by(|a, b| b.occurrence_count.cmp(&a.occurrence_count));

    let mut domain_indicators: Vec<DomainIndicator> = domains
        .into_iter()
        .map(|(domain, acc)| DomainIndicator {
            domain,
            first_row: acc.first_row,
            occurrence_count: acc.count,
        })
        .collect();
    domain_indicators.sort_by(|a, b| b.occurrence_count.cmp(&a.occurrence_count));

    let mut url_indicators: Vec<UrlIndicator> = urls
        .into_iter()
        .map(|(url, acc)| {
            let domain = extract_domain_from_url(&url).unwrap_or_default();
            UrlIndicator {
                url,
                domain,
                first_row: acc.first_row,
                occurrence_count: acc.count,
            }
        })
        .collect();
    url_indicators.sort_by(|a, b| b.occurrence_count.cmp(&a.occurrence_count));

    let mut email_indicators: Vec<EmailIndicator> = emails
        .into_iter()
        .map(|(email, acc)| EmailIndicator {
            email,
            first_row: acc.first_row,
            occurrence_count: acc.count,
        })
        .collect();
    email_indicators.sort_by(|a, b| b.occurrence_count.cmp(&a.occurrence_count));

    let mut user_agent_indicators: Vec<UserAgentIndicator> = user_agents
        .into_iter()
        .map(|(user_agent, acc)| UserAgentIndicator {
            user_agent,
            first_row: acc.first_row,
            occurrence_count: acc.count,
        })
        .collect();
    user_agent_indicators.sort_by(|a, b| b.occurrence_count.cmp(&a.occurrence_count));

    let mut correlation_indicators: Vec<CorrelationIndicator> = correlation_acc
        .into_iter()
        .map(|((kind, value), (first_row, count, col))| CorrelationIndicator {
            kind,
            value,
            first_row,
            occurrence_count: count,
            source_column: col,
        })
        .collect();
    correlation_indicators.sort_by(|a, b| b.occurrence_count.cmp(&a.occurrence_count));

    Ok(IocExtractionSummary {
        rows_scanned,
        ip_indicators,
        domain_indicators,
        url_indicators,
        email_indicators,
        user_agent_indicators,
        correlation_indicators,
    })
}

// ----- Role-based column identification -----

struct IocRelevantRoles {
    ip_columns: HashSet<String>,
    user_agent_columns: HashSet<String>,
}

fn load_ioc_relevant_roles(conn: &Connection) -> Result<IocRelevantRoles> {
    let exists: i64 = conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = '_column_roles')",
        [],
        |row| row.get(0),
    )?;

    let mut ip_columns = HashSet::new();
    let mut user_agent_columns = HashSet::new();

    if exists != 0 {
        let mut stmt = conn.prepare(
            "SELECT role, sql_name FROM _column_roles WHERE status IN ('suggested', 'confirmed')",
        )?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let role: String = row.get(0)?;
            let sql_name: String = row.get(1)?;
            match role.as_str() {
                "ip" => {
                    ip_columns.insert(sql_name);
                }
                "user_agent" => {
                    user_agent_columns.insert(sql_name);
                }
                _ => {}
            }
        }
    }

    // If no IP column was detected by roles, try header heuristic on all columns.
    if ip_columns.is_empty() {
        if let Ok(columns) = db::load_columns(conn) {
            for col in &columns {
                let lower = col.sql_name.to_ascii_lowercase();
                if lower.contains("ip")
                    || lower.contains("address")
                    || lower.contains("sourceip")
                    || lower.contains("clientip")
                {
                    ip_columns.insert(col.sql_name.clone());
                }
            }
        }
    }

    Ok(IocRelevantRoles {
        ip_columns,
        user_agent_columns,
    })
}

// ----- Parsing helpers -----

/// Extracts plausible user agent values from a user agent cell.
/// Handles single user agents and compound/chained user agents delimited by ";;", newlines, or pipes.
pub(crate) fn extract_user_agents_from_cell(cell: &str) -> Vec<&str> {
    let trimmed = cell.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }
    if trimmed.contains(";;")
        || trimmed.contains('\n')
        || trimmed.contains(" | ")
        || trimmed.contains(" || ")
    {
        trimmed
            .split(|c: char| c == '\n' || c == '\r')
            .flat_map(|line| line.split(";;"))
            .flat_map(|part| part.split(" || "))
            .flat_map(|part| part.split(" | "))
            .map(str::trim)
            .filter(|part| is_plausible_user_agent(part))
            .collect()
    } else if is_plausible_user_agent(trimmed) {
        vec![trimmed]
    } else {
        Vec::new()
    }
}

fn is_plausible_user_agent(val: &str) -> bool {
    let len = val.len();
    if len < 3 || len > 1024 {
        return false;
    }
    let lower = val.to_ascii_lowercase();
    !matches!(
        lower.as_str(),
        "-" | "null" | "none" | "n/a" | "unknown" | "undefined" | "(null)" | "empty" | "default" | "ok" | "true" | "false"
    )
}

/// Extracts IPv4 addresses from free text. Does not extract IPv6 (too noisy in logs).
fn extract_ipv4_addresses(text: &str) -> Vec<String> {
    let mut results = Vec::new();
    let bytes = text.as_bytes();
    let mut index = 0;

    while index < bytes.len() {
        if bytes[index].is_ascii_digit()
            && (index == 0 || (!bytes[index - 1].is_ascii_digit() && bytes[index - 1] != b'.'))
        {
            if let Some((ip, end)) = try_parse_ipv4_at(text, index) {
                let end_ok =
                    end >= bytes.len() || (!bytes[end].is_ascii_digit() && bytes[end] != b'.');
                if end_ok {
                    results.push(ip);
                    index = end;
                    continue;
                }
            }
        }
        index += 1;
    }

    results
}

fn try_parse_ipv4_at(text: &str, start: usize) -> Option<(String, usize)> {
    let bytes = text.as_bytes();
    // If preceded by '/', '\', 'v', 'V' or an alphanumeric character, reject (software version, path, or identifier).
    if start > 0 {
        let prev = bytes[start - 1];
        if prev == b'/' || prev == b'\\' || prev == b'v' || prev == b'V' || prev.is_ascii_alphanumeric() {
            return None;
        }
    }
    let mut cursor = start;
    let mut octets = Vec::new();

    loop {
        let digit_start = cursor;
        while cursor < bytes.len() && bytes[cursor].is_ascii_digit() {
            cursor += 1;
        }
        let digit_len = cursor - digit_start;
        if digit_len == 0 || digit_len > 3 {
            return None;
        }
        let value: u32 = text[digit_start..cursor].parse().ok()?;
        if value > 255 {
            return None;
        }
        octets.push(value);
        if octets.len() == 4 {
            // Reject software versions ending in .0.0.0 (e.g. 151.0.0.0, 149.0.0.0, 140.0.0.0).
            if octets[1] == 0 && octets[2] == 0 && octets[3] == 0 {
                return None;
            }
            // Reject 0.0.0.0 and 255.255.255.255 broadcast / unspecified noise.
            if (octets[0] == 0 && octets[1] == 0 && octets[2] == 0 && octets[3] == 0)
                || (octets[0] == 255 && octets[1] == 255 && octets[2] == 255 && octets[3] == 255)
            {
                return None;
            }
            let ip = format!("{}.{}.{}.{}", octets[0], octets[1], octets[2], octets[3]);
            return Some((ip, cursor));
        }
        if cursor < bytes.len() && bytes[cursor] == b'.' {
            cursor += 1;
        } else {
            return None;
        }
    }
}

/// Extracts email-like patterns from text. Conservative: requires user@domain.tld format.
fn extract_emails(text: &str) -> Vec<String> {
    let mut results = Vec::new();
    // Simple approach: find @ signs and expand outward.
    for (pos, _) in text.match_indices('@') {
        let before = &text[..pos];
        let after = &text[pos + 1..];

        let local_start = before
            .rfind(|c: char| {
                !(c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_' | '+'))
            })
            .map(|i| i + 1)
            .unwrap_or(0);
        let local = &before[local_start..];

        let domain_end = after
            .find(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '.' | '-')))
            .unwrap_or(after.len());
        let domain = &after[..domain_end];

        if !local.is_empty()
            && domain.contains('.')
            && !domain.ends_with('.')
            && !domain.starts_with('.')
            && local.len() >= 2
            && domain.len() >= 4
        {
            let email = format!("{local}@{domain}").to_lowercase();
            if !results.contains(&email) {
                results.push(email);
            }
        }
    }
    results
}

/// Extracts URLs (http:// and https://) from text.
fn extract_urls(text: &str) -> Vec<String> {
    let mut results = Vec::new();
    let lower = text.to_lowercase();

    for prefix in &["https://", "http://"] {
        let mut search_from = 0;
        while let Some(start) = lower[search_from..].find(prefix) {
            let abs_start = search_from + start;
            let url_start = abs_start;
            let remaining = &text[url_start..];
            let url_end = remaining
                .find(|c: char| c.is_whitespace() || matches!(c, '"' | '\'' | '>' | '<' | ')' | ']'))
                .unwrap_or(remaining.len());
            let url = remaining[..url_end].trim_end_matches(|c: char| matches!(c, '.' | ',' | ';'));
            if url.len() > prefix.len() + 3 && !results.contains(&url.to_string()) {
                results.push(url.to_string());
            }
            search_from = url_start + url_end.max(1);
        }
    }
    results
}

const FILE_EXTENSIONS_EXCLUDE: &[&str] = &[
    "exe", "dll", "sys", "drv", "ocx", "bin", "csv", "tsv", "xlsx", "xls", "json", "xml",
    "txt", "log", "png", "jpg", "jpeg", "gif", "ico", "bmp", "pdf", "zip", "tar", "gz",
    "7z", "bak", "tmp", "dat", "ps1", "bat", "cmd", "sh", "py", "rs", "js", "ts", "css",
    "html", "htm", "md", "cfg", "ini", "reg", "inf", "cat", "manifest", "mui", "evtx",
    "etl", "pf", "lnk", "config", "properties", "service", "wasm", "map", "woff", "woff2",
    "ttf", "eot", "svg", "webp", "mp3", "mp4", "wav", "avi", "mov",
];

pub fn is_valid_domain(candidate: &str) -> bool {
    let domain = candidate.trim().trim_matches('.').to_lowercase();
    if domain.len() < 4 || domain.len() > 253 {
        return false;
    }
    if domain.contains("://")
        || domain.contains('@')
        || domain.contains('/')
        || domain.contains('\\')
        || domain.contains(':')
    {
        return false;
    }
    let parts: Vec<&str> = domain.split('.').collect();
    if parts.len() < 2 {
        return false;
    }
    let tld = parts.last().unwrap();
    if tld.len() < 2 || tld.len() > 24 || !tld.chars().all(|c| c.is_ascii_alphabetic()) {
        return false;
    }
    if FILE_EXTENSIONS_EXCLUDE.contains(&tld) {
        return false;
    }
    for part in &parts {
        if part.is_empty() || part.len() > 63 {
            return false;
        }
        if part.starts_with('-') || part.ends_with('-') {
            return false;
        }
        if !part.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
            return false;
        }
    }
    // Reject if every part is numeric (e.g. raw IP address)
    if parts.iter().all(|p| p.chars().all(|c| c.is_ascii_digit())) {
        return false;
    }
    true
}

/// Extracts standalone FQDN domains from text tokens.
fn extract_standalone_domains(text: &str) -> Vec<String> {
    let mut results = Vec::new();
    for token in text.split(|c: char| {
        c.is_whitespace() || matches!(c, '"' | '\'' | '(' | ')' | '<' | '>' | '[' | ']' | '{' | '}' | ',' | ';')
    }) {
        let trimmed = token.trim_matches(|c: char| matches!(c, '.' | ':' | '-' | '/' | '\\' | '?' | '#' | '=' | '&'));
        if is_valid_domain(trimmed) {
            let lower = trimmed.to_lowercase();
            if !results.contains(&lower) {
                results.push(lower);
            }
        }
    }
    results
}

/// Extracts the domain from a URL.
fn extract_domain_from_url(url: &str) -> Option<String> {
    let after_scheme = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))?;
    let domain_end = after_scheme
        .find(|c: char| matches!(c, '/' | ':' | '?' | '#'))
        .unwrap_or(after_scheme.len());
    let domain = &after_scheme[..domain_end];
    let cleaned = domain.trim().trim_matches('.').to_lowercase();
    if is_valid_domain(&cleaned) {
        Some(cleaned)
    } else {
        None
    }
}

pub(crate) fn classify_correlation_column(name: &str) -> Option<&'static str> {
    let lower = name.to_ascii_lowercase();
    let stripped = lower.replace(['_', '-', ' ', '.'], "");

    if stripped.contains("correlationid")
        || stripped.contains("requestid")
        || stripped.contains("activityid")
        || stripped.contains("clientrequestid")
        || stripped.contains("mscv")
    {
        Some("correlation_id")
    } else if stripped.contains("sessionid")
        || stripped.contains("aadsessionid")
        || stripped.contains("logonsessionid")
        || stripped.contains("sessionguid")
        || stripped.contains("usersessionid")
        || stripped == "session"
    {
        Some("session_id")
    } else if stripped.contains("deviceid")
        || stripped.contains("aaddeviceid")
        || stripped.contains("machineid")
        || stripped.contains("endpointid")
    {
        Some("device_id")
    } else if stripped.contains("clientappid")
        || stripped.contains("appid")
        || stripped.contains("applicationid")
        || stripped.contains("clientid")
        || stripped.contains("targetappid")
    {
        Some("app_id")
    } else if stripped.contains("uniquetokenid")
        || stripped.contains("tokenid")
        || stripped.contains("tokenidentifier")
        || stripped == "jti"
    {
        Some("unique_token_id")
    } else if stripped.contains("mailboxguid") || stripped.contains("mailboxid") {
        Some("mailbox_guid")
    } else if stripped.contains("internetmessageid") {
        Some("internet_message_id")
    } else if stripped.contains("networkmessageid") {
        Some("network_message_id")
    } else if stripped.contains("fileid")
        || stripped.contains("driveitemid")
        || (stripped.contains("objectid") && !stripped.contains("userobjectid"))
    {
        Some("file_id")
    } else if stripped.contains("hash")
        || stripped.contains("md5")
        || stripped.contains("sha1")
        || stripped.contains("sha256")
        || stripped.contains("sha512")
        || stripped.contains("imphash")
    {
        Some("hash")
    } else {
        None
    }
}

pub(crate) fn is_json_payload_column(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    let stripped = lower.replace(['_', '-', ' ', '.'], "");
    stripped.contains("auditdata")
        || stripped.contains("parameters")
        || stripped.contains("extendedproperties")
        || stripped.contains("properties")
        || stripped.contains("details")
        || stripped.contains("payload")
        || stripped.contains("rawdata")
        || stripped.contains("eventdata")
}

pub(crate) fn classify_json_key(key: &str) -> Option<&'static str> {
    let lower = key.to_ascii_lowercase();
    let stripped = lower.replace(['_', '-', ' ', '.'], "");

    if stripped.contains("correlationid")
        || stripped.contains("requestid")
        || stripped.contains("activityid")
        || stripped.contains("clientrequestid")
        || stripped.contains("mscv")
    {
        Some("correlation_id")
    } else if stripped.contains("sessionid")
        || stripped.contains("aadsessionid")
        || stripped.contains("logonsessionid")
        || stripped.contains("usersessionid")
        || stripped.contains("sessionguid")
        || stripped == "session"
    {
        Some("session_id")
    } else if stripped.contains("deviceid")
        || stripped.contains("aaddeviceid")
        || stripped.contains("machineid")
        || stripped.contains("endpointid")
    {
        Some("device_id")
    } else if stripped.contains("clientappid")
        || stripped.contains("appid")
        || stripped.contains("applicationid")
        || stripped.contains("clientid")
        || stripped.contains("targetappid")
    {
        Some("app_id")
    } else if stripped.contains("uniquetokenid")
        || stripped.contains("tokenid")
        || stripped.contains("tokenidentifier")
        || stripped == "jti"
    {
        Some("unique_token_id")
    } else if stripped.contains("mailboxguid") || stripped.contains("mailboxid") {
        Some("mailbox_guid")
    } else if stripped.contains("internetmessageid") {
        Some("internet_message_id")
    } else if stripped.contains("networkmessageid") {
        Some("network_message_id")
    } else if stripped.contains("fileid")
        || stripped.contains("driveitemid")
        || (stripped.contains("objectid") && !stripped.contains("userobjectid"))
    {
        Some("file_id")
    } else if stripped.contains("hash")
        || stripped.contains("sha256")
        || stripped.contains("md5")
        || stripped.contains("sha1")
        || stripped.contains("imphash")
    {
        Some("hash")
    } else {
        None
    }
}

pub(crate) fn is_valid_correlation_value(kind: &str, val: &str) -> bool {
    let trimmed = val.trim().trim_matches('"').trim_matches('\'').trim();
    if trimmed.is_empty() || trimmed.len() < 4 || trimmed.len() > 256 {
        return false;
    }
    let lower = trimmed.to_ascii_lowercase();
    if lower == "none"
        || lower == "null"
        || lower == "undefined"
        || lower == "n/a"
        || lower == "na"
        || lower == "unknown"
        || lower == "-"
        || lower == "--"
        || lower == "0"
        || lower == "0x0"
        || lower == "00000000-0000-0000-0000-000000000000"
        || lower == "00000000000000000000000000000000"
        || lower == "ffffffff-ffff-ffff-ffff-ffffffffffff"
        || lower == "{}"
        || lower == "[]"
    {
        return false;
    }
    match kind {
        "hash" => is_valid_hash(&lower),
        "correlation_id" | "device_id" | "app_id" | "network_message_id" | "mailbox_guid" => {
            trimmed.len() >= 6
        }
        _ => trimmed.len() >= 4,
    }
}

pub(crate) fn is_valid_hash(h: &str) -> bool {
    let len = h.len();
    if len != 32 && len != 40 && len != 64 {
        return false;
    }
    if !h.chars().all(|c| c.is_ascii_hexdigit()) {
        return false;
    }
    if h.chars().all(|c| c == '0') || h.chars().all(|c| c == 'f' || c == 'F') {
        return false;
    }
    true
}

pub(crate) fn extract_hashes_from_text(text: &str) -> Vec<String> {
    let mut hashes = Vec::new();
    for part in text.split([',', ';', ' ', '|', '\n', '\r', '\t', '"', '\'']) {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some((k, v)) = trimmed.split_once('=') {
            let k_clean = k.trim().to_ascii_lowercase();
            if k_clean == "md5"
                || k_clean == "sha1"
                || k_clean == "sha256"
                || k_clean == "imphash"
            {
                let h = v.trim().to_ascii_lowercase();
                if is_valid_hash(&h) && !hashes.contains(&h) {
                    hashes.push(h);
                }
            }
        } else if trimmed.len() == 64 || trimmed.len() == 40 || trimmed.len() == 32 {
            let h = trimmed.to_ascii_lowercase();
            if is_valid_hash(&h) && !hashes.contains(&h) {
                hashes.push(h);
            }
        }
    }
    hashes
}

pub(crate) fn extract_correlation_from_json(
    val: &serde_json::Value,
    results: &mut Vec<(String, String)>,
) {
    match val {
        serde_json::Value::Object(map) => {
            for (k, v) in map {
                match v {
                    serde_json::Value::String(s) => {
                        if let Some(kind) = classify_json_key(k) {
                            if kind == "hash" {
                                for h in extract_hashes_from_text(s) {
                                    results.push((kind.to_string(), h));
                                }
                            } else if is_valid_correlation_value(kind, s) {
                                results.push((kind.to_string(), s.trim().to_string()));
                            }
                        }
                    }
                    serde_json::Value::Object(_) | serde_json::Value::Array(_) => {
                        extract_correlation_from_json(v, results);
                    }
                    _ => {}
                }
            }
        }
        serde_json::Value::Array(arr) => {
            for item in arr {
                extract_correlation_from_json(item, results);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_ipv4_basic() {
        let ips = extract_ipv4_addresses("Login from 192.168.1.1 at 10.0.0.5 via proxy");
        assert_eq!(ips, vec!["192.168.1.1", "10.0.0.5"]);
    }

    #[test]
    fn extract_ipv4_skips_invalid() {
        let ips = extract_ipv4_addresses("version 1.2.3 not an ip 999.999.999.999");
        assert!(ips.is_empty());
    }

    #[test]
    fn extract_emails_basic() {
        let emails = extract_emails("sent to user@example.com and admin@corp.local");
        assert_eq!(emails, vec!["user@example.com", "admin@corp.local"]);
    }

    #[test]
    fn extract_urls_basic() {
        let urls = extract_urls("visit https://evil.com/payload.exe and http://test.org/path");
        assert_eq!(urls.len(), 2);
        assert!(urls[0].contains("evil.com"));
        assert!(urls[1].contains("test.org"));
    }

    #[test]
    fn domain_extraction() {
        assert_eq!(
            extract_domain_from_url("https://evil.com/payload"),
            Some("evil.com".to_string())
        );
        assert_eq!(
            extract_domain_from_url("http://sub.domain.org:8080/path"),
            Some("sub.domain.org".to_string())
        );
    }

    #[test]
    fn domain_validation_and_standalone_extraction() {
        assert!(is_valid_domain("example-corp.com"));
        assert!(is_valid_domain("login.microsoftonline.com"));
        assert!(!is_valid_domain("notepad.exe"));
        assert!(!is_valid_domain("data.csv"));
        assert!(!is_valid_domain("192.168.1.1"));

        let domains = extract_standalone_domains(
            "activity from alice at example-corp.com authenticated via login.microsoftonline.com",
        );
        assert!(domains.contains(&"example-corp.com".to_string()));
        assert!(domains.contains(&"login.microsoftonline.com".to_string()));
    }

    #[test]
    fn ip_extraction_skips_browser_versions_and_noise() {
        let ua_ips = extract_ipv4_addresses(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/151.0.0.0 Safari/537.36",
        );
        assert!(ua_ips.is_empty(), "Chrome/151.0.0.0 must not be treated as an IP: {ua_ips:?}");

        let normal = extract_ipv4_addresses("client 198.51.100.156 connected to 10.0.0.1");
        assert_eq!(normal, vec!["198.51.100.156", "10.0.0.1"]);
    }

    #[test]
    fn vpn_range_parsing() {
        let (network, mask) = parse_cidr("10.0.0.0/8").unwrap();
        assert_eq!(network, 0x0A000000);
        assert_eq!(mask, 0xFF000000);
    }

    #[test]
    fn private_ip_detection() {
        assert!(is_private_ip("192.168.1.1"));
        assert!(is_private_ip("10.0.0.1"));
        assert!(is_private_ip("172.16.0.1"));
        assert!(!is_private_ip("8.8.8.8"));
    }

    #[test]
    fn user_agent_extraction_handles_axios_and_delimiters() {
        // Individual short UA
        assert_eq!(extract_user_agents_from_cell("axios"), vec!["axios"]);
        assert_eq!(extract_user_agents_from_cell("curl/8.0"), vec!["curl/8.0"]);

        // Compound UA separated by ;;
        assert_eq!(
            extract_user_agents_from_cell("client rest;;axios"),
            vec!["client rest", "axios"]
        );

        // Skips dashes, nulls, empty
        assert!(extract_user_agents_from_cell("-").is_empty());
        assert!(extract_user_agents_from_cell("null").is_empty());
        assert!(extract_user_agents_from_cell("").is_empty());

        // Newline and pipe delimiters
        assert_eq!(
            extract_user_agents_from_cell("sqlmap/1.7\r\ncurl/8.0"),
            vec!["sqlmap/1.7", "curl/8.0"]
        );
        assert_eq!(
            extract_user_agents_from_cell("nikto | nmap | dirbuster"),
            vec!["nikto", "nmap", "dirbuster"]
        );
    }

    #[test]
    fn tool_and_browser_detection_covers_all_categories() {
        use crate::intel::roles::is_known_tool_or_browser_ua;

        // Browsers
        assert!(is_known_tool_or_browser_ua("Mozilla/5.0 (Windows NT 10.0; Win64; x64)"));
        assert!(is_known_tool_or_browser_ua("Chrome/120.0.0.0"));
        
        // HTTP & CLI libraries
        assert!(is_known_tool_or_browser_ua("curl/7.88.1"));
        assert!(is_known_tool_or_browser_ua("curl"));
        assert!(is_known_tool_or_browser_ua("Wget/1.21.3"));
        assert!(is_known_tool_or_browser_ua("python-requests/2.31.0"));
        assert!(is_known_tool_or_browser_ua("axios/1.6.0"));
        assert!(is_known_tool_or_browser_ua("Go-http-client/1.1"));
        assert!(is_known_tool_or_browser_ua("okhttp/4.9.2"));
        assert!(is_known_tool_or_browser_ua("libwww-perl/6.52"));

        // API / REST tools
        assert!(is_known_tool_or_browser_ua("PostmanRuntime/7.32.3"));
        assert!(is_known_tool_or_browser_ua("client rest"));
        assert!(is_known_tool_or_browser_ua("insomnia/2023.5.8"));
        assert!(is_known_tool_or_browser_ua("HTTPie/3.2.1"));

        // Scanners & Pentest tools
        assert!(is_known_tool_or_browser_ua("sqlmap/1.7.2#stable"));
        assert!(is_known_tool_or_browser_ua("nikto/2.1.6"));
        assert!(is_known_tool_or_browser_ua("Nmap Scripting Engine"));
        assert!(is_known_tool_or_browser_ua("gobuster/3.5"));
        assert!(is_known_tool_or_browser_ua("nuclei - v2.9.8"));
        assert!(is_known_tool_or_browser_ua("BurpSuite"));

        // Offensive & C2
        assert!(is_known_tool_or_browser_ua("metasploit"));
        assert!(is_known_tool_or_browser_ua("CobaltStrike Beacon"));
        assert!(is_known_tool_or_browser_ua("impacket"));

        // Cloud & Admin
        assert!(is_known_tool_or_browser_ua("aws-cli/2.11.0"));
        assert!(is_known_tool_or_browser_ua("rclone/v1.62.2"));
        assert!(is_known_tool_or_browser_ua("kubectl/v1.28.0"));

        // Non-UA random strings
        assert!(!is_known_tool_or_browser_ua("-"));
        assert!(!is_known_tool_or_browser_ua("normal text"));
        assert!(!is_known_tool_or_browser_ua("12345"));
    }

    #[test]
    fn correlation_column_classification_matches_all_variants() {
        assert_eq!(classify_correlation_column("CorrelationId"), Some("correlation_id"));
        assert_eq!(classify_correlation_column("client_request_id"), Some("correlation_id"));
        assert_eq!(classify_correlation_column("ActivityId"), Some("correlation_id"));
        assert_eq!(classify_correlation_column("MSCV"), Some("correlation_id"));

        assert_eq!(classify_correlation_column("AADSessionId"), Some("session_id"));
        assert_eq!(classify_correlation_column("SessionId"), Some("session_id"));
        assert_eq!(classify_correlation_column("logon_session_id"), Some("session_id"));

        assert_eq!(classify_correlation_column("DeviceID"), Some("device_id"));
        assert_eq!(classify_correlation_column("AADDeviceId"), Some("device_id"));
        assert_eq!(classify_correlation_column("machine_id"), Some("device_id"));

        assert_eq!(classify_correlation_column("AppId"), Some("app_id"));
        assert_eq!(classify_correlation_column("client_app_id"), Some("app_id"));
        assert_eq!(classify_correlation_column("application_id"), Some("app_id"));

        assert_eq!(classify_correlation_column("UniqueTokenId"), Some("unique_token_id"));
        assert_eq!(classify_correlation_column("jti"), Some("unique_token_id"));

        assert_eq!(classify_correlation_column("MailboxGuid"), Some("mailbox_guid"));
        assert_eq!(classify_correlation_column("InternetMessageId"), Some("internet_message_id"));
        assert_eq!(classify_correlation_column("NetworkMessageId"), Some("network_message_id"));
        assert_eq!(classify_correlation_column("FileId"), Some("file_id"));
        assert_eq!(classify_correlation_column("ObjectId"), Some("file_id"));

        assert_eq!(classify_correlation_column("SHA256"), Some("hash"));
        assert_eq!(classify_correlation_column("file_hash_md5"), Some("hash"));
        assert_eq!(classify_correlation_column("imphash"), Some("hash"));

        assert_eq!(classify_correlation_column("user_name"), None);
        assert_eq!(classify_correlation_column("status_code"), None);
    }

    #[test]
    fn json_key_classification_and_extraction() {
        let json_payload = serde_json::json!({
            "CorrelationId": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
            "AppId": "00000003-0000-0000-c000-000000000000",
            "DeviceProperties": {
                "DeviceId": "d4e5f6a1-b2c3-4d5e-6f7a-8b9c0d1e2f3a",
                "SessionId": "s-a8b9c0d1-e2f3"
            },
            "MailboxGuid": "12345678-1234-1234-1234-123456789abc",
            "InternetMessageId": "<cab123@mail.protection.outlook.com>",
            "NetworkMessageId": "87654321-4321-4321-4321-cba987654321",
            "FileId": "01ABCDEF9876543210",
            "UniqueTokenId": "token-xyz-1234567",
            "FileHash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        });

        let mut extracted = Vec::new();
        extract_correlation_from_json(&json_payload, &mut extracted);

        let kinds: Vec<&str> = extracted.iter().map(|(k, _)| k.as_str()).collect();
        assert!(kinds.contains(&"correlation_id"));
        assert!(kinds.contains(&"app_id"));
        assert!(kinds.contains(&"device_id"));
        assert!(kinds.contains(&"session_id"));
        assert!(kinds.contains(&"mailbox_guid"));
        assert!(kinds.contains(&"internet_message_id"));
        assert!(kinds.contains(&"network_message_id"));
        assert!(kinds.contains(&"file_id"));
        assert!(kinds.contains(&"unique_token_id"));
        assert!(kinds.contains(&"hash"));

        let hash_val = extracted.iter().find(|(k, _)| k == "hash").unwrap();
        assert_eq!(hash_val.1, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    }

    #[test]
    fn hash_extraction_and_validation() {
        let text = "File dropped with MD5=5d41402abc4b2a76b9719d911017c592 and SHA256=2c26b46b68ffc68ff99b453c1d30413413422d706483bfa0f98a5e886266e7ae";
        let hashes = extract_hashes_from_text(text);
        assert_eq!(hashes.len(), 2);
        assert!(hashes.contains(&"5d41402abc4b2a76b9719d911017c592".to_string()));
        assert!(hashes.contains(&"2c26b46b68ffc68ff99b453c1d30413413422d706483bfa0f98a5e886266e7ae".to_string()));

        // Invalid hashes (all zeros, wrong length, non-hex)
        assert!(!is_valid_hash("00000000000000000000000000000000"));
        assert!(!is_valid_hash("ffffffffffffffffffffffffffffffff"));
        assert!(!is_valid_hash("not_a_hash_at_all"));
        assert!(!is_valid_hash("12345"));
    }

    #[test]
    fn correlation_value_validation_filters_placeholders() {
        assert!(!is_valid_correlation_value("device_id", "-"));
        assert!(!is_valid_correlation_value("device_id", "none"));
        assert!(!is_valid_correlation_value("device_id", "00000000-0000-0000-0000-000000000000"));
        assert!(!is_valid_correlation_value("session_id", "null"));
        assert!(!is_valid_correlation_value("session_id", "N/A"));
        assert!(!is_valid_correlation_value("correlation_id", "{}"));

        assert!(is_valid_correlation_value("device_id", "device-998877-aabbcc"));
        assert!(is_valid_correlation_value("session_id", "session-token-12345"));
        assert!(is_valid_correlation_value("correlation_id", "7c9e6679-7425-40de-944b-e07fc1f90ae7"));
    }
}

