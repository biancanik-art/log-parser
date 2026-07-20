use crate::db::{self, ColumnMeta};
use anyhow::Result;
use rusqlite::Connection;
use serde::Serialize;
use std::collections::HashMap;

const SCAN_BATCH_ROWS: i64 = 1000;
const PROGRESS_INTERVAL_ROWS: i64 = 10_000;
const STAGING_TABLE: &str = "temp._row_activity_staging";
const MAX_DETAIL_CHARS: usize = 120;
const MAX_TOP_DETAILS: usize = 3;

/// Column headers (sql_name with underscores stripped) whose per-row value describes what the
/// event *is* — the operation/activity descriptor fields common across Sentinel, Defender,
/// Taegis and O365/Azure audit exports.
const DESCRIPTOR_HEADERS: [&str; 16] = [
    "activity",
    "activitydisplayname",
    "activitytype",
    "operation",
    "operationname",
    "event",
    "eventname",
    "eventtype",
    "eventcategory",
    "action",
    "actiontype",
    "category",
    "taskcategory",
    "recordtype",
    "description",
    "message",
];

const EVENT_ID_HEADERS: [&str; 3] = ["eventid", "eventcode", "eventnumber"];

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ActivityScanSummary {
    pub rows_classified: i64,
    pub categories: Vec<ActivityCategorySummary>,
    pub rows_ignored: i64,
    pub ignored_by_rule: Vec<crate::intel::ignore_rules::IgnoredRuleBreakdown>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ActivityCategorySummary {
    pub category: String,
    pub label: String,
    pub row_count: i64,
    pub top_details: Vec<ActivityDetailCount>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ActivityDetailCount {
    pub detail: String,
    pub row_count: i64,
}

pub fn category_label(category: &str) -> &'static str {
    match category {
        "authentication" => "Authentication / sign-in",
        "account_management" => "Account & group management",
        "process" => "Process execution",
        "file" => "File activity",
        "network" => "Network activity",
        "email" => "Email activity",
        "scheduled_task" => "Scheduled task",
        "registry" => "Registry modification",
        "service_config" => "Service / system configuration",
        "other" => "Other / unclassified",
        _ => "Activity",
    }
}

/// Maps a Windows Security/System event ID to an activity category. Deliberately limited to
/// IDs whose meaning is unambiguous in the channels this tool sees.
fn event_id_category(id: i64) -> Option<&'static str> {
    Some(match id {
        4624 | 4625 | 4634 | 4647 | 4648 | 4768 | 4769 | 4771 | 4776 | 4778 | 4779 => {
            "authentication"
        }
        4720 | 4722 | 4723 | 4724 | 4725 | 4726 | 4728 | 4729 | 4732 | 4733 | 4738 | 4740
        | 4756 | 4757 | 4767 => "account_management",
        4688 | 4689 => "process",
        4656 | 4660 | 4663 | 5140 | 5145 => "file",
        5156 | 5157 => "network",
        4698..=4702 => "scheduled_task",
        4657 => "registry",
        4697 | 7045 | 7036 | 1102 | 4719 => "service_config",
        _ => return None,
    })
}

/// Keyword rules against descriptor-column values, checked in this order — earlier entries win
/// so specific operations ("scheduled task", "inbox rule") are not swallowed by broader ones.
const DESCRIPTOR_RULES: [(&str, &[&str]); 9] = [
    ("scheduled_task", &["scheduled task", "schtask"]),
    ("registry", &["registry"]),
    (
        "email",
        &[
            "mail", "email", "phish", "smtp", "inbox rule", "inboxrule", "forwarding rule",
            "message trace", "send message", "message sent",
        ],
    ),
    (
        "account_management",
        &[
            "user added",
            "member added",
            "member removed",
            "add member",
            "remove member",
            "account created",
            "account was created",
            "account was deleted",
            "account enabled",
            "account disabled",
            "password reset",
            "password change",
            "group membership",
            "role assigned",
            "user created",
            "user deleted",
        ],
    ),
    (
        "authentication",
        &[
            "logon",
            "log on",
            "login",
            "logged in",
            "loggedin",
            "sign-in",
            "signin",
            "sign in",
            "logoff",
            "log off",
            "logged out",
            "authentication",
            "authenticated",
            "kerberos",
            "ntlm",
            "credential validation",
            "mfa",
        ],
    ),
    (
        "process",
        &[
            "process creat",
            "process start",
            "process terminat",
            "process launched",
            "image loaded",
            "executed",
            "execution",
        ],
    ),
    (
        "service_config",
        &[
            "service install",
            "service was installed",
            "service start",
            "service stop",
            "audit policy",
            "policy change",
            "log cleared",
            "audit log",
            "configuration change",
        ],
    ),
    (
        "file",
        &[
            "file",
            "document",
            "sharepoint",
            "onedrive",
            "download",
            "upload",
            "share accessed",
            "folder",
        ],
    ),
    (
        "network",
        &[
            "connection",
            "network",
            "dns",
            "firewall",
            "url",
            "http",
            "remote access",
            "vpn",
        ],
    ),
];

struct SignalColumns {
    /// (index into the SELECT column list, original header) — order preserved from the file.
    event_id: Vec<(usize, String)>,
    descriptors: Vec<(usize, String)>,
    /// role → select index, for the has-data fallback.
    role_columns: Vec<(&'static str, usize, String)>,
    select_idents: Vec<String>,
}

fn plan_signal_columns(conn: &Connection, columns: &[ColumnMeta]) -> Result<SignalColumns> {
    let roles = load_active_roles(conn)?;
    let mut plan = SignalColumns {
        event_id: Vec::new(),
        descriptors: Vec::new(),
        role_columns: Vec::new(),
        select_idents: Vec::new(),
    };
    let mut index_by_sql_name: HashMap<String, usize> = HashMap::new();
    let mut push_column = |plan: &mut SignalColumns, column: &ColumnMeta| -> usize {
        if let Some(&existing) = index_by_sql_name.get(column.sql_name.as_str()) {
            return existing;
        }
        let index = plan.select_idents.len();
        plan.select_idents.push(db::quote_ident(&column.sql_name));
        index_by_sql_name.insert(column.sql_name.clone(), index);
        index
    };

    for column in columns {
        let normalized: String = column
            .sql_name
            .chars()
            .filter(|c| *c != '_')
            .collect();
        if EVENT_ID_HEADERS.contains(&normalized.as_str()) {
            let index = push_column(&mut plan, column);
            plan.event_id.push((index, column.original_name.clone()));
        } else if DESCRIPTOR_HEADERS.contains(&normalized.as_str()) {
            let index = push_column(&mut plan, column);
            plan.descriptors.push((index, column.original_name.clone()));
        }
    }

    // Fallback signals: a row with command-line/process data is process activity even when no
    // descriptor column says so; likewise file and network role columns.
    for (role, category_hint) in [
        ("command_line", "process"),
        ("process_name", "process"),
        ("file_name", "file"),
        ("ip", "network"),
    ] {
        if let Some(sql_name) = roles.get(role) {
            if let Some(column) = columns.iter().find(|column| &column.sql_name == sql_name) {
                let index = push_column(&mut plan, column);
                plan.role_columns
                    .push((category_hint, index, column.original_name.clone()));
            }
        }
    }
    Ok(plan)
}

/// Classifies EVERY non-ignored row into exactly one activity category and atomically publishes
/// the result to `_row_activity`. Deterministic and complete by design: the examiner asked "row
/// by row, what activity is there" — a row the heuristics cannot place still gets a labeled
/// `other` row rather than silence. Rows matching an enabled ignore rule are excluded entirely
/// (see `_ignored_rows`), not classified as `other`.
pub fn classify_rows(
    conn: &mut Connection,
    columns: &[ColumnMeta],
    mut on_progress: impl FnMut(i64, i64, &str),
) -> Result<ActivityScanSummary> {
    db::create_activity_schema(conn)?;
    conn.execute_batch(&format!(
        "DROP TABLE IF EXISTS {STAGING_TABLE};
         CREATE TEMP TABLE _row_activity_staging (
            row_num INTEGER PRIMARY KEY,
            category TEXT NOT NULL,
            detail TEXT NOT NULL,
            source_column TEXT NOT NULL
         );"
    ))?;
    let result = classify_into_staging(conn, columns, &mut on_progress);
    let cleanup = conn.execute_batch(&format!("DROP TABLE IF EXISTS {STAGING_TABLE}"));
    let summary = result?;
    cleanup?;
    Ok(summary)
}

fn classify_into_staging(
    conn: &mut Connection,
    columns: &[ColumnMeta],
    on_progress: &mut impl FnMut(i64, i64, &str),
) -> Result<ActivityScanSummary> {
    let total_rows: i64 = conn.query_row("SELECT COUNT(*) FROM rows", [], |row| row.get(0))?;
    let plan = plan_signal_columns(conn, columns)?;
    crate::intel::ignore_rules::ensure_ignored_rows_computed(conn)?;
    let ignored = crate::intel::ignore_rules::load_ignored_row_set(conn)?;
    on_progress(0, total_rows, "classifying");

    let select_sql = if plan.select_idents.is_empty() {
        "SELECT row_num FROM rows WHERE row_num > ?1 ORDER BY row_num ASC LIMIT ?2".to_string()
    } else {
        format!(
            "SELECT row_num, {} FROM rows WHERE row_num > ?1 ORDER BY row_num ASC LIMIT ?2",
            plan.select_idents.join(", ")
        )
    };

    let mut rows_visited = 0i64;
    let mut rows_classified = 0i64;
    let mut last_row_num = i64::MIN;
    let mut next_progress_at = PROGRESS_INTERVAL_ROWS;

    loop {
        let batch = {
            let mut stmt = conn.prepare(&select_sql)?;
            let mut rows = stmt.query(rusqlite::params![last_row_num, SCAN_BATCH_ROWS])?;
            let mut batch = Vec::new();
            while let Some(row) = rows.next()? {
                let row_num: i64 = row.get(0)?;
                let mut values = Vec::with_capacity(plan.select_idents.len());
                for column_idx in 0..plan.select_idents.len() {
                    values.push(row.get::<_, Option<String>>(column_idx + 1)?);
                }
                batch.push((row_num, values));
            }
            batch
        };
        if batch.is_empty() {
            break;
        }

        let mut tx_rows: Vec<(i64, &'static str, String, String)> = Vec::with_capacity(batch.len());
        for (row_num, values) in &batch {
            last_row_num = *row_num;
            rows_visited += 1;
            if ignored.contains(row_num) {
                continue;
            }
            let (category, detail, source) = classify_row(&plan, values);
            tx_rows.push((*row_num, category, detail, source));
        }
        {
            let mut stmt = conn.prepare_cached(&format!(
                "INSERT INTO {STAGING_TABLE} (row_num, category, detail, source_column)
                 VALUES (?1, ?2, ?3, ?4)"
            ))?;
            for (row_num, category, detail, source) in &tx_rows {
                stmt.execute(rusqlite::params![row_num, category, detail, source])?;
            }
        }
        rows_classified += tx_rows.len() as i64;

        if rows_visited >= next_progress_at {
            on_progress(rows_visited, total_rows, "classifying");
            while next_progress_at <= rows_visited {
                next_progress_at += PROGRESS_INTERVAL_ROWS;
            }
        }
    }

    // Atomic publication: readers see the previous complete classification or the new one.
    let tx = conn.transaction()?;
    tx.execute("DELETE FROM _row_activity", [])?;
    tx.execute("DELETE FROM _row_activity_info", [])?;
    tx.execute(
        &format!(
            "INSERT INTO _row_activity (row_num, category, detail, source_column)
             SELECT row_num, category, detail, source_column FROM {STAGING_TABLE}"
        ),
        [],
    )?;
    tx.execute(
        "INSERT INTO _row_activity_info (rows_classified, completed_at) VALUES (?1, ?2)",
        rusqlite::params![rows_classified, chrono::Utc::now().to_rfc3339()],
    )?;
    tx.commit()?;

    on_progress(rows_visited, total_rows, "complete");
    summarize(conn, rows_classified)
}

fn classify_row(
    plan: &SignalColumns,
    values: &[Option<String>],
) -> (&'static str, String, String) {
    let cell = |index: usize| -> Option<&str> {
        values
            .get(index)
            .and_then(|value| value.as_deref())
            .map(str::trim)
            .filter(|value| !value.is_empty())
    };

    for (index, original_name) in &plan.event_id {
        let Some(raw) = cell(*index) else { continue };
        // Values arrive as "4624", "4624.0" (Excel numeric), or with surrounding text.
        let digits: String = raw.chars().take_while(|c| c.is_ascii_digit()).collect();
        if let Ok(id) = digits.parse::<i64>() {
            if let Some(category) = event_id_category(id) {
                return (
                    category,
                    bounded_detail(&format!("Event ID {id}")),
                    original_name.clone(),
                );
            }
        }
    }

    for (index, original_name) in &plan.descriptors {
        let Some(raw) = cell(*index) else { continue };
        let lower = raw.to_lowercase();
        for (category, keywords) in DESCRIPTOR_RULES {
            if keywords.iter().any(|keyword| lower.contains(keyword)) {
                return (category, bounded_detail(raw), original_name.clone());
            }
        }
    }

    for (category, index, original_name) in &plan.role_columns {
        if let Some(raw) = cell(*index) {
            return (category, bounded_detail(raw), original_name.clone());
        }
    }

    // A descriptor value that matched no rule still names what the event was.
    for (index, original_name) in &plan.descriptors {
        if let Some(raw) = cell(*index) {
            return ("other", bounded_detail(raw), original_name.clone());
        }
    }
    ("other", String::new(), String::new())
}

fn summarize(conn: &Connection, rows_classified: i64) -> Result<ActivityScanSummary> {
    let mut categories = Vec::new();
    let category_rows: Vec<(String, i64)> = {
        let mut stmt = conn.prepare(
            "SELECT category, COUNT(*) FROM _row_activity
             GROUP BY category ORDER BY COUNT(*) DESC",
        )?;
        let rows = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        rows
    };
    for (category, row_count) in category_rows {
        let mut stmt = conn.prepare_cached(
            "SELECT detail, COUNT(*) FROM _row_activity
             WHERE category = ?1 AND detail != ''
             GROUP BY detail ORDER BY COUNT(*) DESC LIMIT ?2",
        )?;
        let top_details = stmt
            .query_map(
                rusqlite::params![category, MAX_TOP_DETAILS as i64],
                |row| {
                    Ok(ActivityDetailCount {
                        detail: row.get(0)?,
                        row_count: row.get(1)?,
                    })
                },
            )?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        categories.push(ActivityCategorySummary {
            label: category_label(&category).to_string(),
            category,
            row_count,
            top_details,
        });
    }
    let (rows_ignored, ignored_by_rule) = crate::intel::ignore_rules::ignored_rows_summary(conn)?;
    Ok(ActivityScanSummary {
        rows_classified,
        categories,
        rows_ignored,
        ignored_by_rule,
    })
}

fn load_active_roles(conn: &Connection) -> Result<HashMap<String, String>> {
    let exists: i64 = conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = '_column_roles')",
        [],
        |row| row.get(0),
    )?;
    let mut roles = HashMap::new();
    if exists == 0 {
        return Ok(roles);
    }
    let mut stmt = conn.prepare(
        "SELECT role, sql_name FROM _column_roles WHERE status IN ('suggested', 'confirmed')",
    )?;
    let mut rows = stmt.query([])?;
    while let Some(row) = rows.next()? {
        roles.insert(row.get(0)?, row.get(1)?);
    }
    Ok(roles)
}

fn bounded_detail(detail: &str) -> String {
    let cleaned = detail.trim();
    if cleaned.chars().count() <= MAX_DETAIL_CHARS {
        return cleaned.to_string();
    }
    let bounded: String = cleaned.chars().take(MAX_DETAIL_CHARS).collect();
    format!("{bounded}…")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn column(sql_name: &str, original: &str) -> ColumnMeta {
        ColumnMeta {
            sql_name: sql_name.to_string(),
            original_name: original.to_string(),
            col_index: 0,
            inferred_type: "text".to_string(),
        }
    }

    fn test_conn(columns: &[&str]) -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        let defs: Vec<String> = columns.iter().map(|c| format!("{c} TEXT")).collect();
        conn.execute_batch(&format!(
            "CREATE TABLE rows (row_num INTEGER PRIMARY KEY, {});",
            defs.join(", ")
        ))
        .unwrap();
        conn
    }

    fn add_role(conn: &Connection, role: &str, sql_name: &str) {
        db::create_column_roles_table(conn).unwrap();
        conn.execute(
            "INSERT INTO _column_roles (role, sql_name, confidence, status, reasons_json)
             VALUES (?1, ?2, 0.9, 'confirmed', '[]')",
            [role, sql_name],
        )
        .unwrap();
    }

    fn category_of(conn: &Connection, row_num: i64) -> String {
        conn.query_row(
            "SELECT category FROM _row_activity WHERE row_num = ?1",
            [row_num],
            |row| row.get(0),
        )
        .unwrap()
    }

    #[test]
    fn windows_event_ids_map_to_activity_categories() {
        let mut conn = test_conn(&["event_id", "message"]);
        let rows = [
            (1, "4624", "An account was successfully logged on"),
            (2, "4720", "A user account was created"),
            (3, "4688", "A new process has been created"),
            (4, "5145", "A network share object was checked"),
            (5, "4698", "A scheduled task was created"),
            (6, "9999", "Unmapped event id falls through to descriptors"),
        ];
        for (row_num, id, msg) in rows {
            conn.execute(
                "INSERT INTO rows (row_num, event_id, message) VALUES (?1, ?2, ?3)",
                rusqlite::params![row_num, id, msg],
            )
            .unwrap();
        }
        let columns = vec![column("event_id", "EventID"), column("message", "Message")];
        let summary = classify_rows(&mut conn, &columns, |_, _, _| {}).unwrap();

        assert_eq!(summary.rows_classified, 6);
        assert_eq!(summary.rows_ignored, 0);
        assert_eq!(category_of(&conn, 1), "authentication");
        assert_eq!(category_of(&conn, 2), "account_management");
        assert_eq!(category_of(&conn, 3), "process");
        assert_eq!(category_of(&conn, 4), "file");
        assert_eq!(category_of(&conn, 5), "scheduled_task");
        // 9999 is unmapped, but the message descriptor still resolves it.
        assert_eq!(category_of(&conn, 6), "other");
    }

    #[test]
    fn ignored_row_is_excluded_from_classification_entirely() {
        let mut conn = test_conn(&["event_id", "processname"]);
        add_role(&conn, "process_name", "processname");
        conn.execute(
            "INSERT INTO rows (row_num, event_id, processname) VALUES
             (1, '4624', 'QualysAgent.exe'),
             (2, '4624', 'winlogon.exe')",
            [],
        )
        .unwrap();
        let columns = vec![
            column("event_id", "EventID"),
            column("processname", "ProcessName"),
        ];
        let summary = classify_rows(&mut conn, &columns, |_, _, _| {}).unwrap();

        assert_eq!(
            summary.rows_classified, 1,
            "the Qualys row is excluded entirely, not classified as 'other'"
        );
        assert_eq!(summary.rows_ignored, 1);
        assert_eq!(summary.ignored_by_rule.len(), 1);
        assert_eq!(summary.ignored_by_rule[0].rule_id, "qualys-agent-activity");
        assert_eq!(category_of(&conn, 2), "authentication");

        let row1_exists: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM _row_activity WHERE row_num = 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(
            row1_exists, 0,
            "ignored row must not appear in _row_activity, not even as 'other'"
        );
    }

    #[test]
    fn cloud_audit_descriptors_classify_without_event_ids() {
        let mut conn = test_conn(&["operation_name"]);
        let rows = [
            (1, "UserLoggedIn"),
            (2, "FileDownloaded"),
            (3, "Add member to role."),
            (4, "Send message"),
            (5, "New-InboxRule created for mailbox"),
            (6, "SomethingUnrecognized"),
        ];
        for (row_num, op) in rows {
            conn.execute(
                "INSERT INTO rows (row_num, operation_name) VALUES (?1, ?2)",
                rusqlite::params![row_num, op],
            )
            .unwrap();
        }
        let columns = vec![column("operation_name", "OperationName")];
        let summary = classify_rows(&mut conn, &columns, |_, _, _| {}).unwrap();

        assert_eq!(category_of(&conn, 1), "authentication");
        assert_eq!(category_of(&conn, 2), "file");
        assert_eq!(category_of(&conn, 3), "account_management");
        assert_eq!(category_of(&conn, 4), "email");
        assert_eq!(category_of(&conn, 5), "email");
        // Unrecognized descriptors stay labeled with their own text under 'other'.
        assert_eq!(category_of(&conn, 6), "other");
        let detail: String = conn
            .query_row(
                "SELECT detail FROM _row_activity WHERE row_num = 6",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(detail, "SomethingUnrecognized");
        assert_eq!(summary.rows_classified, 6);
    }

    #[test]
    fn role_columns_are_the_fallback_and_every_row_is_labeled() {
        let mut conn = test_conn(&["cmd", "src_ip", "note"]);
        add_role(&conn, "command_line", "cmd");
        add_role(&conn, "ip", "src_ip");
        conn.execute(
            "INSERT INTO rows (row_num, cmd, src_ip, note) VALUES
             (1, 'powershell.exe -nop', NULL, NULL),
             (2, NULL, '10.0.0.5', NULL),
             (3, NULL, NULL, 'free text only')",
            [],
        )
        .unwrap();
        let columns = vec![
            column("cmd", "CommandLine"),
            column("src_ip", "SourceIP"),
            column("note", "Note"),
        ];
        let summary = classify_rows(&mut conn, &columns, |_, _, _| {}).unwrap();

        assert_eq!(category_of(&conn, 1), "process");
        assert_eq!(category_of(&conn, 2), "network");
        assert_eq!(category_of(&conn, 3), "other");
        // Completeness: exactly one classification per source row.
        let classified: i64 = conn
            .query_row("SELECT COUNT(*) FROM _row_activity", [], |row| row.get(0))
            .unwrap();
        assert_eq!(classified, 3);
        assert_eq!(summary.rows_classified, 3);
    }

    #[test]
    fn reclassification_replaces_previous_results() {
        let mut conn = test_conn(&["operation_name"]);
        conn.execute(
            "INSERT INTO rows (row_num, operation_name) VALUES (1, 'UserLoggedIn')",
            [],
        )
        .unwrap();
        let columns = vec![column("operation_name", "OperationName")];
        classify_rows(&mut conn, &columns, |_, _, _| {}).unwrap();
        assert_eq!(category_of(&conn, 1), "authentication");

        conn.execute(
            "UPDATE rows SET operation_name = 'FileDownloaded' WHERE row_num = 1",
            [],
        )
        .unwrap();
        classify_rows(&mut conn, &columns, |_, _, _| {}).unwrap();
        assert_eq!(category_of(&conn, 1), "file");
        let info_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM _row_activity_info", [], |row| row.get(0))
            .unwrap();
        assert_eq!(info_count, 1);
    }

    #[test]
    fn summary_reports_counts_and_top_details_per_category() {
        let mut conn = test_conn(&["operation_name"]);
        for row_num in 1..=5i64 {
            conn.execute(
                "INSERT INTO rows (row_num, operation_name) VALUES (?1, 'UserLoggedIn')",
                [row_num],
            )
            .unwrap();
        }
        conn.execute(
            "INSERT INTO rows (row_num, operation_name) VALUES (6, 'FileDownloaded')",
            [],
        )
        .unwrap();
        let columns = vec![column("operation_name", "OperationName")];
        let summary = classify_rows(&mut conn, &columns, |_, _, _| {}).unwrap();

        assert_eq!(summary.rows_classified, 6);
        assert_eq!(summary.categories[0].category, "authentication");
        assert_eq!(summary.categories[0].row_count, 5);
        assert_eq!(summary.categories[0].top_details[0].detail, "UserLoggedIn");
        assert_eq!(summary.categories[0].top_details[0].row_count, 5);
        assert!(summary
            .categories
            .iter()
            .any(|category| category.category == "file" && category.row_count == 1));
    }
}
