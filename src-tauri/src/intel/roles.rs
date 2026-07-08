use crate::db::{self, ColumnMeta};
use crate::intel::time::{classify_timestamp_text, TimestampValueKind};
use anyhow::{anyhow, Result};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::net::IpAddr;

const SAMPLE_LIMIT: i64 = 500;
const ROLES: [&str; 8] = [
    "timestamp",
    "user",
    "command_line",
    "process_name",
    "file_name",
    "host",
    "ip",
    "text_evidence",
];

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoleDecisionStatus {
    Confirmed,
    Rejected,
}

impl RoleDecisionStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Confirmed => "confirmed",
            Self::Rejected => "rejected",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ColumnRoleSuggestion {
    pub role: String,
    pub sql_name: String,
    pub original_name: String,
    pub confidence: f64,
    pub status: String,
    pub reasons: Vec<String>,
}

#[derive(Debug)]
struct Candidate {
    role: &'static str,
    sql_name: String,
    confidence: f64,
    reasons: Vec<String>,
}

#[derive(Debug)]
struct HeaderProfile {
    text: String,
    compact: String,
}

impl HeaderProfile {
    fn new(column: &ColumnMeta) -> Self {
        let text = format!("{} {}", column.sql_name, column.original_name).to_ascii_lowercase();
        let compact = text.chars().filter(|c| c.is_ascii_alphanumeric()).collect();
        Self { text, compact }
    }

    fn contains_any<'a>(&self, keywords: &'a [&str]) -> Option<&'a str> {
        keywords
            .iter()
            .copied()
            .find(|keyword| self.compact.contains(keyword) || self.text.contains(keyword))
    }

    fn has_token(&self, token: &str) -> bool {
        self.text
            .split(|c: char| !c.is_ascii_alphanumeric())
            .any(|part| part == token)
    }
}

pub fn detect_column_roles(
    conn: &Connection,
    columns: &[ColumnMeta],
) -> Result<Vec<ColumnRoleSuggestion>> {
    db::create_column_roles_table(conn)?;
    let samples = sample_column_values(conn, columns)?;

    for role in ROLES {
        if let Some(candidate) = best_candidate(role, columns, &samples) {
            if candidate.confidence >= threshold_for(role) {
                upsert_suggestion(conn, &candidate)?;
            }
        }
    }

    load_column_roles(conn, columns)
}

pub fn set_column_role_status(
    conn: &Connection,
    columns: &[ColumnMeta],
    role: &str,
    sql_name: &str,
    status: RoleDecisionStatus,
) -> Result<ColumnRoleSuggestion> {
    db::create_column_roles_table(conn)?;
    validate_role(role)?;
    let column = columns
        .iter()
        .find(|column| column.sql_name == sql_name)
        .ok_or_else(|| anyhow!("unknown column for role assignment: {sql_name}"))?;

    let existing = load_role(conn, columns, role).ok();
    let mut reasons = existing
        .as_ref()
        .map(|row| row.reasons.clone())
        .unwrap_or_default();
    let confidence = match status {
        RoleDecisionStatus::Confirmed => {
            if existing
                .as_ref()
                .is_some_and(|row| row.sql_name == column.sql_name)
            {
                reasons.push("examiner confirmed the suggested role assignment".to_string());
            } else {
                reasons.push(format!(
                    "examiner selected '{}' for this role, overriding the suggestion",
                    column.original_name
                ));
            }
            1.0
        }
        RoleDecisionStatus::Rejected => {
            reasons.push("examiner rejected this role assignment".to_string());
            existing.as_ref().map_or(0.0, |row| row.confidence)
        }
    };
    let reasons_json = serde_json::to_string(&reasons)?;

    conn.execute(
        "INSERT INTO _column_roles (role, sql_name, confidence, status, reasons_json)
         VALUES (?1, ?2, ?3, ?4, ?5)
         ON CONFLICT(role) DO UPDATE SET
            sql_name = excluded.sql_name,
            confidence = excluded.confidence,
            status = excluded.status,
            reasons_json = excluded.reasons_json",
        rusqlite::params![
            role,
            column.sql_name,
            confidence,
            status.as_str(),
            reasons_json
        ],
    )?;

    load_role(conn, columns, role)
}

pub fn load_column_roles(
    conn: &Connection,
    columns: &[ColumnMeta],
) -> Result<Vec<ColumnRoleSuggestion>> {
    db::create_column_roles_table(conn)?;
    let mut stmt = conn.prepare(
        "SELECT role, sql_name, confidence, status, reasons_json
         FROM _column_roles
         ORDER BY role",
    )?;
    let rows = stmt.query_map([], |row| {
        let role: String = row.get(0)?;
        let sql_name: String = row.get(1)?;
        let original_name = columns
            .iter()
            .find(|column| column.sql_name == sql_name)
            .map(|column| column.original_name.clone())
            .unwrap_or_else(|| sql_name.clone());
        let reasons_json: String = row.get(4)?;
        let reasons = serde_json::from_str(&reasons_json).unwrap_or_default();
        Ok(ColumnRoleSuggestion {
            role,
            sql_name,
            original_name,
            confidence: row.get(2)?,
            status: row.get(3)?,
            reasons,
        })
    })?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(Into::into)
}

fn load_role(
    conn: &Connection,
    columns: &[ColumnMeta],
    role: &str,
) -> Result<ColumnRoleSuggestion> {
    load_column_roles(conn, columns)?
        .into_iter()
        .find(|row| row.role == role)
        .ok_or_else(|| anyhow!("no column role recorded for {role}"))
}

fn validate_role(role: &str) -> Result<()> {
    if ROLES.contains(&role) {
        Ok(())
    } else {
        Err(anyhow!("unknown column role: {role}"))
    }
}

fn upsert_suggestion(conn: &Connection, candidate: &Candidate) -> Result<()> {
    let reasons_json = serde_json::to_string(&candidate.reasons)?;
    conn.execute(
        "INSERT INTO _column_roles (role, sql_name, confidence, status, reasons_json)
         VALUES (?1, ?2, ?3, 'suggested', ?4)
         ON CONFLICT(role) DO UPDATE SET
            sql_name = excluded.sql_name,
            confidence = excluded.confidence,
            status = 'suggested',
            reasons_json = excluded.reasons_json
         WHERE _column_roles.status = 'suggested'",
        rusqlite::params![
            candidate.role,
            candidate.sql_name,
            candidate.confidence,
            reasons_json
        ],
    )?;
    Ok(())
}

fn sample_column_values(conn: &Connection, columns: &[ColumnMeta]) -> Result<Vec<Vec<String>>> {
    if columns.is_empty() {
        return Ok(Vec::new());
    }

    let select_cols = columns
        .iter()
        .map(|column| db::quote_ident(&column.sql_name))
        .collect::<Vec<_>>()
        .join(", ");
    let sql = format!("SELECT {select_cols} FROM rows ORDER BY row_num ASC LIMIT {SAMPLE_LIMIT}");
    let mut stmt = conn.prepare(&sql)?;
    let mut rows = stmt.query([])?;
    let mut samples = vec![Vec::new(); columns.len()];

    while let Some(row) = rows.next()? {
        for (idx, column_samples) in samples.iter_mut().enumerate() {
            let value: Option<String> = row.get(idx)?;
            let value = value.unwrap_or_default();
            if !value.trim().is_empty() {
                column_samples.push(value);
            }
        }
    }

    Ok(samples)
}

fn best_candidate(
    role: &'static str,
    columns: &[ColumnMeta],
    samples: &[Vec<String>],
) -> Option<Candidate> {
    columns
        .iter()
        .zip(samples.iter())
        .filter_map(|(column, values)| score_column(role, column, values))
        .max_by(|left, right| left.confidence.total_cmp(&right.confidence))
}

fn score_column(role: &'static str, column: &ColumnMeta, values: &[String]) -> Option<Candidate> {
    let header = HeaderProfile::new(column);
    let (confidence, reasons) = match role {
        "timestamp" => score_timestamp(&header, values),
        "user" => score_user(&header, values),
        "command_line" => score_command_line(&header, values),
        "process_name" => score_process_name(&header, values),
        "file_name" => score_file_name(&header, values),
        "host" => score_host(&header, values),
        "ip" => score_ip(&header, values),
        "text_evidence" => score_text_evidence(&header, values),
        _ => return None,
    };

    (!reasons.is_empty()).then(|| Candidate {
        role,
        sql_name: column.sql_name.clone(),
        confidence: confidence.clamp(0.0, 1.0),
        reasons,
    })
}

fn score_timestamp(header: &HeaderProfile, values: &[String]) -> (f64, Vec<String>) {
    let mut score = 0.0;
    let mut reasons = Vec::new();
    if let Some(keyword) = header.contains_any(&[
        "timegenerated",
        "timestamp",
        "eventtime",
        "creationtime",
        "created",
        "datetime",
    ]) {
        score += 0.5;
        reasons.push(format!("header contains timestamp keyword '{keyword}'"));
    } else if let Some(keyword) = header.contains_any(&["date", "time", "utc"]) {
        score += 0.28;
        reasons.push(format!("header contains time-related keyword '{keyword}'"));
    }

    let total = values.len();
    if total > 0 {
        let mut parsed = 0usize;
        let mut explicit_or_epoch = 0usize;
        for value in values {
            match classify_timestamp_text(value) {
                TimestampValueKind::ExplicitOffset | TimestampValueKind::Epoch => {
                    parsed += 1;
                    explicit_or_epoch += 1;
                }
                TimestampValueKind::Naive => parsed += 1,
                TimestampValueKind::Blank | TimestampValueKind::Invalid => {}
            }
        }
        let parsed_ratio = parsed as f64 / total as f64;
        if parsed_ratio >= 0.4 {
            score += parsed_ratio * 0.42;
            reasons.push(format!(
                "{parsed}/{total} sampled values parse as timestamp-like values"
            ));
        }
        if parsed > 0 {
            let explicit_ratio = explicit_or_epoch as f64 / parsed as f64;
            if explicit_ratio >= 0.5 {
                score += 0.08;
                reasons
                    .push("many sampled timestamps include an offset/Z or epoch value".to_string());
            }
        }
    }

    (score, reasons)
}

fn score_user(header: &HeaderProfile, values: &[String]) -> (f64, Vec<String>) {
    let mut score = 0.0;
    let mut reasons = Vec::new();
    if header.compact.contains("useragent") || header.compact.contains("browser") {
        return (0.0, reasons);
    }
    if let Some(keyword) = header.contains_any(&[
        "userprincipalname",
        "targetusername",
        "subjectusername",
        "username",
        "accountname",
        "account",
        "principal",
        "actor",
        "user",
    ]) {
        score += 0.45;
        reasons.push(format!("header contains identity keyword '{keyword}'"));
    } else if let Some(keyword) = header.contains_any(&["owner", "identity"]) {
        score += 0.25;
        reasons.push(format!("header contains weak identity keyword '{keyword}'"));
    }

    let total = values.len();
    if total > 0 {
        let identity_count = values
            .iter()
            .filter(|value| is_identity_like(value))
            .count();
        let ratio = identity_count as f64 / total as f64;
        if ratio >= 0.35 {
            score += ratio * 0.42;
            reasons.push(format!(
                "{identity_count}/{total} sampled values look like users, UPNs, or SIDs"
            ));
        }
    }

    (score, reasons)
}

fn score_command_line(header: &HeaderProfile, values: &[String]) -> (f64, Vec<String>) {
    let mut score = 0.0;
    let mut reasons = Vec::new();
    if let Some(keyword) = header.contains_any(&[
        "initiatingprocesscommandline",
        "processcommandline",
        "commandline",
        "cmdline",
    ]) {
        score += 0.55;
        reasons.push(format!("header contains command-line keyword '{keyword}'"));
    } else if let Some(keyword) = header.contains_any(&["command"]) {
        score += 0.32;
        reasons.push(format!("header contains command keyword '{keyword}'"));
    }

    let total = values.len();
    if total > 0 {
        let command_count = values
            .iter()
            .filter(|value| is_command_line_like(value))
            .count();
        let long_count = values
            .iter()
            .filter(|value| value.trim().len() >= 40)
            .count();
        let command_ratio = command_count as f64 / total as f64;
        if command_ratio >= 0.25 {
            score += command_ratio * 0.36;
            reasons.push(format!(
                "{command_count}/{total} sampled values look like executable command lines"
            ));
        }
        let long_ratio = long_count as f64 / total as f64;
        if long_ratio >= 0.3 {
            score += long_ratio * 0.12;
            reasons.push("sampled values are often long enough to include arguments".to_string());
        }
    }

    (score, reasons)
}

fn score_process_name(header: &HeaderProfile, values: &[String]) -> (f64, Vec<String>) {
    let mut score = 0.0;
    let mut reasons = Vec::new();
    if !header.compact.contains("commandline") {
        if let Some(keyword) = header.contains_any(&[
            "processname",
            "imagename",
            "newprocessname",
            "parentprocessname",
            "process",
            "image",
        ]) {
            score += 0.4;
            reasons.push(format!("header contains process keyword '{keyword}'"));
        }
    }

    let total = values.len();
    if total > 0 {
        let process_count = values
            .iter()
            .filter(|value| is_process_name_like(value))
            .count();
        let ratio = process_count as f64 / total as f64;
        if ratio >= 0.4 {
            score += ratio * 0.45;
            reasons.push(format!(
                "{process_count}/{total} sampled values look like process image names"
            ));
        }
    }

    (score, reasons)
}

fn score_file_name(header: &HeaderProfile, values: &[String]) -> (f64, Vec<String>) {
    let mut score = 0.0;
    let mut reasons = Vec::new();
    if !header.compact.contains("commandline") {
        if let Some(keyword) = header.contains_any(&[
            "targetfilename",
            "filename",
            "filepath",
            "folder",
            "path",
            "file",
        ]) {
            score += 0.38;
            reasons.push(format!("header contains file/path keyword '{keyword}'"));
        }
    }

    let total = values.len();
    if total > 0 {
        let file_count = values.iter().filter(|value| is_file_like(value)).count();
        let ratio = file_count as f64 / total as f64;
        if ratio >= 0.35 {
            score += ratio * 0.42;
            reasons.push(format!(
                "{file_count}/{total} sampled values look like file paths or file names"
            ));
        }
    }

    (score, reasons)
}

fn score_host(header: &HeaderProfile, values: &[String]) -> (f64, Vec<String>) {
    let mut score = 0.0;
    let mut reasons = Vec::new();
    if let Some(keyword) = header.contains_any(&[
        "hostname",
        "computer",
        "devicename",
        "device",
        "workstation",
        "machine",
        "host",
        "dvc",
    ]) {
        score += 0.42;
        reasons.push(format!("header contains host keyword '{keyword}'"));
    }

    let total = values.len();
    if total > 0 {
        let host_count = values.iter().filter(|value| is_host_like(value)).count();
        let ratio = host_count as f64 / total as f64;
        if ratio >= 0.35 {
            score += ratio * 0.38;
            reasons.push(format!(
                "{host_count}/{total} sampled values look like hostnames"
            ));
        }
    }

    (score, reasons)
}

fn score_ip(header: &HeaderProfile, values: &[String]) -> (f64, Vec<String>) {
    let mut score = 0.0;
    let mut reasons = Vec::new();
    if let Some(keyword) = header.contains_any(&[
        "ipaddress",
        "sourceip",
        "destinationip",
        "remoteip",
        "clientip",
        "srcip",
        "dstip",
    ]) {
        score += 0.48;
        reasons.push(format!("header contains IP keyword '{keyword}'"));
    } else if header.has_token("ip") {
        score += 0.42;
        reasons.push("header contains IP token".to_string());
    }

    let total = values.len();
    if total > 0 {
        let ip_count = values
            .iter()
            .filter(|value| parse_ip(value).is_some())
            .count();
        let ratio = ip_count as f64 / total as f64;
        if ratio >= 0.3 {
            score += ratio * 0.45;
            reasons.push(format!(
                "{ip_count}/{total} sampled values parse as IP addresses"
            ));
        }
    }

    (score, reasons)
}

fn score_text_evidence(header: &HeaderProfile, values: &[String]) -> (f64, Vec<String>) {
    let mut score = 0.0;
    let mut reasons = Vec::new();
    if let Some(keyword) = header.contains_any(&[
        "description",
        "eventdata",
        "additionalfields",
        "message",
        "details",
        "activity",
        "operation",
        "action",
        "alert",
        "threat",
        "evidence",
        "summary",
        "raw",
    ]) {
        score += 0.38;
        reasons.push(format!("header contains evidence-text keyword '{keyword}'"));
    }

    let total = values.len();
    if total > 0 {
        let text_count = values
            .iter()
            .filter(|value| {
                let trimmed = value.trim();
                trimmed.len() >= 25 && trimmed.chars().any(char::is_whitespace)
            })
            .count();
        let ratio = text_count as f64 / total as f64;
        if ratio >= 0.35 {
            score += ratio * 0.32;
            reasons.push(format!(
                "{text_count}/{total} sampled values look like descriptive evidence text"
            ));
        }
    }

    (score, reasons)
}

fn threshold_for(role: &str) -> f64 {
    match role {
        "timestamp" | "user" | "command_line" | "ip" => 0.3,
        "process_name" | "file_name" | "host" | "text_evidence" => 0.25,
        _ => 1.0,
    }
}

fn is_identity_like(value: &str) -> bool {
    let trimmed = value.trim();
    if trimmed.is_empty() || trimmed.len() > 256 || looks_like_path(trimmed) {
        return false;
    }
    if trimmed.to_ascii_lowercase().ends_with(".exe") {
        return false;
    }
    is_domain_user(trimmed)
        || is_upn_like(trimmed)
        || is_sid_like(trimmed)
        || is_simple_user(trimmed)
}

fn is_domain_user(value: &str) -> bool {
    let Some((domain, user)) = value.split_once('\\') else {
        return false;
    };
    !domain.is_empty()
        && !user.is_empty()
        && domain.len() <= 64
        && user.len() <= 128
        && domain
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        && user
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_' | '$'))
}

fn is_upn_like(value: &str) -> bool {
    let Some((local, domain)) = value.split_once('@') else {
        return false;
    };
    !local.is_empty()
        && domain.contains('.')
        && !domain.ends_with('.')
        && local
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_' | '+'))
        && domain
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-'))
}

fn is_sid_like(value: &str) -> bool {
    value.starts_with("S-1-")
        && value
            .split('-')
            .skip(1)
            .all(|part| !part.is_empty() && part.chars().all(|c| c.is_ascii_digit()))
}

fn is_simple_user(value: &str) -> bool {
    let trimmed = value.trim();
    (2..=64).contains(&trimmed.len())
        && trimmed.chars().any(|c| c.is_ascii_alphabetic())
        && !trimmed.contains(char::is_whitespace)
        && !trimmed.contains('.')
        && parse_ip(trimmed).is_none()
        && trimmed
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '$'))
}

fn is_command_line_like(value: &str) -> bool {
    let trimmed = value.trim();
    let lower = trimmed.to_ascii_lowercase();
    if trimmed.len() < 12 {
        return false;
    }
    let has_known_tool = [
        "powershell",
        "pwsh",
        "cmd.exe",
        "wmic",
        "rundll32",
        "mshta",
        "regsvr32",
        "certutil",
        "bitsadmin",
        "schtasks",
        "wscript",
        "cscript",
    ]
    .iter()
    .any(|needle| lower.contains(needle));
    let has_args = [
        " /c ",
        " -enc",
        " -encodedcommand",
        " -nop",
        " --",
        " /",
        " -",
        "=\"",
    ]
    .iter()
    .any(|needle| lower.contains(needle));
    let exe_with_args = lower.contains(".exe") && trimmed.split_whitespace().count() >= 2;
    has_known_tool && (has_args || trimmed.len() >= 30) || exe_with_args
}

fn is_process_name_like(value: &str) -> bool {
    let trimmed = value.trim().trim_matches('"');
    if trimmed.is_empty() || trimmed.len() > 180 || is_command_line_like(trimmed) {
        return false;
    }
    let basename = trimmed
        .rsplit(['\\', '/'])
        .next()
        .unwrap_or(trimmed)
        .to_ascii_lowercase();
    basename.split_whitespace().count() == 1
        && [".exe", ".dll", ".ps1", ".bat", ".cmd", ".scr"]
            .iter()
            .any(|suffix| basename.ends_with(suffix))
}

fn is_file_like(value: &str) -> bool {
    let trimmed = value.trim().trim_matches('"');
    if trimmed.is_empty() || is_command_line_like(trimmed) {
        return false;
    }
    let lower = trimmed.to_ascii_lowercase();
    let has_path_separator = trimmed.contains('\\') || trimmed.contains('/');
    let has_drive = trimmed.len() >= 3
        && trimmed.as_bytes()[1] == b':'
        && trimmed.as_bytes()[0].is_ascii_alphabetic();
    let has_extension = lower
        .rsplit(['\\', '/'])
        .next()
        .and_then(|basename| basename.rsplit_once('.'))
        .is_some_and(|(_, ext)| (2..=8).contains(&ext.len()));
    (has_path_separator || has_drive || has_extension) && !trimmed.contains('\n')
}

fn is_host_like(value: &str) -> bool {
    let trimmed = value.trim();
    if trimmed.is_empty()
        || trimmed.len() > 253
        || trimmed.contains(char::is_whitespace)
        || looks_like_path(trimmed)
        || parse_ip(trimmed).is_some()
    {
        return false;
    }
    trimmed.chars().any(|c| c.is_ascii_alphabetic())
        && trimmed
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

fn looks_like_path(value: &str) -> bool {
    value.contains('\\')
        || value.contains('/')
        || (value.len() >= 3
            && value.as_bytes()[1] == b':'
            && value.as_bytes()[0].is_ascii_alphabetic())
}

fn parse_ip(value: &str) -> Option<IpAddr> {
    let trimmed = value.trim().trim_matches(['[', ']']);
    if let Ok(ip) = trimmed.parse::<IpAddr>() {
        return Some(ip);
    }
    let (host, port) = trimmed.rsplit_once(':')?;
    if port.chars().all(|c| c.is_ascii_digit()) {
        host.parse::<IpAddr>().ok()
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_fixture() -> (Connection, Vec<ColumnMeta>) {
        let conn = Connection::open_in_memory().unwrap();
        let columns = vec![
            ColumnMeta {
                sql_name: "timegenerated".into(),
                original_name: "TimeGenerated".into(),
                col_index: 0,
                inferred_type: "text".into(),
            },
            ColumnMeta {
                sql_name: "account".into(),
                original_name: "Account".into(),
                col_index: 1,
                inferred_type: "text".into(),
            },
            ColumnMeta {
                sql_name: "processcommandline".into(),
                original_name: "ProcessCommandLine".into(),
                col_index: 2,
                inferred_type: "text".into(),
            },
            ColumnMeta {
                sql_name: "device_name".into(),
                original_name: "DeviceName".into(),
                col_index: 3,
                inferred_type: "text".into(),
            },
        ];
        db::create_schema(&conn, &columns).unwrap();
        let rows = [
            (
                "2026-01-01T02:30:00+02:00",
                "CORP\\alice",
                r#"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -NoP -EncodedCommand SQBFAFg="#,
                "WKSTN-01",
            ),
            (
                "2026-01-01T03:00:00+02:00",
                "bob@example.com",
                r#"cmd.exe /c whoami && ipconfig /all"#,
                "WKSTN-02",
            ),
        ];
        for (idx, row) in rows.iter().enumerate() {
            conn.execute(
                "INSERT INTO rows (row_num, timegenerated, account, processcommandline, device_name)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params![(idx as i64) + 1, row.0, row.1, row.2, row.3],
            )
            .unwrap();
        }
        (conn, columns)
    }

    fn role<'a>(rows: &'a [ColumnRoleSuggestion], role: &str) -> &'a ColumnRoleSuggestion {
        rows.iter().find(|row| row.role == role).unwrap()
    }

    #[test]
    fn detects_timestamp_user_and_command_line_from_headers_and_sampled_content() {
        let (conn, columns) = setup_fixture();
        let suggestions = detect_column_roles(&conn, &columns).unwrap();

        assert_eq!(role(&suggestions, "timestamp").sql_name, "timegenerated");
        assert_eq!(role(&suggestions, "user").sql_name, "account");
        assert_eq!(
            role(&suggestions, "command_line").sql_name,
            "processcommandline"
        );
    }

    #[test]
    fn command_line_role_remains_suggested_until_examiner_confirms() {
        let (conn, columns) = setup_fixture();
        let suggestions = detect_column_roles(&conn, &columns).unwrap();
        let command_line = role(&suggestions, "command_line");
        assert_eq!(command_line.status, "suggested");

        let confirmed = set_column_role_status(
            &conn,
            &columns,
            "command_line",
            "processcommandline",
            RoleDecisionStatus::Confirmed,
        )
        .unwrap();

        assert_eq!(confirmed.role, "command_line");
        assert_eq!(confirmed.sql_name, "processcommandline");
        assert_eq!(confirmed.status, "confirmed");
    }
}
