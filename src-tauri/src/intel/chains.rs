use anyhow::Result;
use rusqlite::{Connection, OptionalExtension};
use serde::Serialize;
use std::collections::HashMap;

/// Two matched events belong to the same chain when they occur on the same host within this
/// window. Chosen to bridge typical hands-on-keyboard pauses without merging separate days.
const CHAIN_WINDOW_MS: i64 = 60 * 60 * 1000;
/// A chain must progress across at least this many distinct tactics; one or two tactics on a
/// host is ordinary keyword noise, three is a story.
const MIN_CHAIN_TACTICS: usize = 3;
const MAX_CHAIN_EVENTS: i64 = 200_000;
const MAX_SAMPLE_ROWS: usize = 50;
const MAX_TECHNIQUE_NAMES: usize = 8;
const MAX_PUBLISHED_CHAINS: usize = 200;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct IntelChainSummary {
    pub chain_id: i64,
    pub host: Option<String>,
    pub start_epoch_ms: Option<i64>,
    pub end_epoch_ms: Option<i64>,
    pub first_row: i64,
    pub last_row: i64,
    pub tactic_count: i64,
    pub event_count: i64,
    pub row_count: i64,
    pub score: i64,
    pub tactic_names: Vec<String>,
    pub technique_names: Vec<String>,
    pub sample_rows: Vec<i64>,
}

#[derive(Debug, Clone)]
struct ChainEvent {
    row_num: i64,
    host: Option<String>,
    epoch_ms: Option<i64>,
    tactic_id: String,
    tactic_name: String,
    technique_name: String,
    score: i64,
}

/// Computes host-scoped, time-windowed chains from a table of match rows. `match_table` is
/// either the private staging table (during a scan) or `_intel_match`; both share one shape.
///
/// Temporal semantics are strict: with normalized row time available, only rows that parsed
/// get chained and the 60-minute window applies. Without `_row_time`, chains make no time
/// claim — each host group is one window and `startEpochMs`/`endEpochMs` stay null.
pub fn compute_chains(conn: &Connection, match_table: &str) -> Result<Vec<IntelChainSummary>> {
    let host_column = detect_host_column(conn)?;
    let has_row_time = table_exists(conn, "_row_time")?;

    let host_select = match &host_column {
        Some(column) => format!(", r.{}", crate::db::quote_ident(column)),
        None => ", NULL".to_string(),
    };
    let (time_select, time_join) = if has_row_time {
        (
            ", t.epoch_ms".to_string(),
            "LEFT JOIN _row_time t ON t.row_num = m.row_num".to_string(),
        )
    } else {
        (", NULL".to_string(), String::new())
    };
    let sql = format!(
        "SELECT m.row_num, m.tactic_id, m.tactic_name, m.technique_name, MAX(m.score)
                {host_select}{time_select}
         FROM {match_table} m
         JOIN rows r ON r.row_num = m.row_num
         {time_join}
         GROUP BY m.row_num, m.tactic_id, m.technique_id
         ORDER BY m.row_num ASC
         LIMIT {MAX_CHAIN_EVENTS}"
    );

    let mut events: Vec<ChainEvent> = Vec::new();
    {
        let mut stmt = conn.prepare(&sql)?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            events.push(ChainEvent {
                row_num: row.get(0)?,
                tactic_id: row.get(1)?,
                tactic_name: row.get(2)?,
                technique_name: row.get(3)?,
                score: row.get(4)?,
                host: row
                    .get::<_, Option<String>>(5)?
                    .map(|value| value.trim().to_string())
                    .filter(|value| !value.is_empty()),
                epoch_ms: row.get(6)?,
            });
        }
    }

    let mut groups: HashMap<Option<String>, Vec<ChainEvent>> = HashMap::new();
    for event in events {
        // Rows whose timestamp failed to parse cannot support a temporal claim.
        if has_row_time && event.epoch_ms.is_none() {
            continue;
        }
        groups.entry(event.host.clone()).or_default().push(event);
    }

    let mut chains = Vec::new();
    for (_, mut group) in groups {
        group.sort_by_key(|event| (event.epoch_ms.unwrap_or(0), event.row_num));
        if has_row_time {
            let mut start = 0usize;
            while start < group.len() {
                let window_start = group[start].epoch_ms.unwrap_or(0);
                let mut end = start;
                while end + 1 < group.len()
                    && group[end + 1].epoch_ms.unwrap_or(0) - window_start <= CHAIN_WINDOW_MS
                {
                    end += 1;
                }
                match build_chain(&group[start..=end]) {
                    Some(chain) => {
                        chains.push(chain);
                        start = end + 1;
                    }
                    None => start += 1,
                }
            }
        } else if let Some(chain) = build_chain(&group) {
            chains.push(chain);
        }
    }

    chains.sort_by(|a, b| {
        b.tactic_count
            .cmp(&a.tactic_count)
            .then_with(|| b.score.cmp(&a.score))
            .then_with(|| b.event_count.cmp(&a.event_count))
            .then_with(|| a.first_row.cmp(&b.first_row))
    });
    chains.truncate(MAX_PUBLISHED_CHAINS);
    for (index, chain) in chains.iter_mut().enumerate() {
        chain.chain_id = (index as i64) + 1;
    }
    Ok(chains)
}

fn build_chain(window: &[ChainEvent]) -> Option<IntelChainSummary> {
    let mut tactic_names: Vec<String> = Vec::new();
    let mut seen_tactics: Vec<&str> = Vec::new();
    let mut technique_names: Vec<String> = Vec::new();
    let mut sample_rows: Vec<i64> = Vec::new();
    let mut max_score = 0i64;
    for event in window {
        if !seen_tactics.contains(&event.tactic_id.as_str()) {
            seen_tactics.push(&event.tactic_id);
            tactic_names.push(event.tactic_name.clone());
        }
        if !technique_names.contains(&event.technique_name)
            && technique_names.len() < MAX_TECHNIQUE_NAMES
        {
            technique_names.push(event.technique_name.clone());
        }
        if sample_rows.last() != Some(&event.row_num) && sample_rows.len() < MAX_SAMPLE_ROWS {
            sample_rows.push(event.row_num);
        }
        max_score = max_score.max(event.score);
    }
    if seen_tactics.len() < MIN_CHAIN_TACTICS {
        return None;
    }

    let mut distinct_rows: Vec<i64> = window.iter().map(|event| event.row_num).collect();
    distinct_rows.sort_unstable();
    distinct_rows.dedup();

    Some(IntelChainSummary {
        chain_id: 0,
        host: window[0].host.clone(),
        start_epoch_ms: window.iter().filter_map(|event| event.epoch_ms).min(),
        end_epoch_ms: window.iter().filter_map(|event| event.epoch_ms).max(),
        first_row: distinct_rows.first().copied().unwrap_or(0),
        last_row: distinct_rows.last().copied().unwrap_or(0),
        tactic_count: seen_tactics.len() as i64,
        event_count: window.len() as i64,
        row_count: distinct_rows.len() as i64,
        // Severity leads with the strongest single match and rewards breadth of progression.
        score: (max_score + 10 * (seen_tactics.len() as i64 - 1)).min(100),
        tactic_names,
        technique_names,
        sample_rows,
    })
}

/// Writes `chains` as the sole published chain set. Callers run this inside the same
/// transaction that publishes `_intel_match`, so readers never see matches and chains from
/// different scans.
pub fn publish_chains(conn: &Connection, chains: &[IntelChainSummary]) -> Result<()> {
    conn.execute("DELETE FROM _intel_chain", [])?;
    let mut stmt = conn.prepare(
        "INSERT INTO _intel_chain (
            chain_id, host, start_epoch_ms, end_epoch_ms, first_row, last_row,
            tactic_count, event_count, row_count, score,
            tactic_names, technique_names, sample_rows
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
    )?;
    for chain in chains {
        stmt.execute(rusqlite::params![
            chain.chain_id,
            chain.host,
            chain.start_epoch_ms,
            chain.end_epoch_ms,
            chain.first_row,
            chain.last_row,
            chain.tactic_count,
            chain.event_count,
            chain.row_count,
            chain.score,
            serde_json::to_string(&chain.tactic_names)?,
            serde_json::to_string(&chain.technique_names)?,
            serde_json::to_string(&chain.sample_rows)?,
        ])?;
    }
    Ok(())
}

fn detect_host_column(conn: &Connection) -> Result<Option<String>> {
    if !table_exists(conn, "_column_roles")? {
        return Ok(None);
    }
    let column = conn
        .query_row(
            "SELECT sql_name FROM _column_roles
             WHERE role = 'host' AND status IN ('suggested', 'confirmed')",
            [],
            |row| row.get::<_, String>(0),
        )
        .optional()?;
    Ok(column)
}

fn table_exists(conn: &Connection, name: &str) -> Result<bool> {
    let exists: i64 = conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?1)",
        [name],
        |row| row.get(0),
    )?;
    Ok(exists != 0)
}
