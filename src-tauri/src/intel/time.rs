use crate::db::{self, ColumnMeta};
use anyhow::{anyhow, bail, Result};
use chrono::{
    DateTime, FixedOffset, LocalResult, NaiveDate, NaiveDateTime, SecondsFormat, TimeZone, Utc,
};
use chrono_tz::Tz;
use rusqlite::{Connection, OptionalExtension, TransactionBehavior};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

const ROW_TIME_BINDING_VERSION: &str = "row-time-v2";
const ROW_TIME_BATCH_ROWS: usize = 512;
static ROW_TIME_BUILD_COUNTER: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum TimestampValueKind {
    ExplicitOffset,
    Naive,
    Epoch,
    Blank,
    Invalid,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TimestampAnalysis {
    pub timestamp_column: String,
    pub original_name: String,
    pub total_rows: i64,
    pub explicit_count: i64,
    pub epoch_count: i64,
    pub naive_count: i64,
    pub blank_count: i64,
    pub invalid_count: i64,
    pub needs_timezone: bool,
    pub needs_date_convention: bool,
    pub inferred_date_convention: Option<String>,
    pub sample_naive_values: Vec<String>,
    pub sample_invalid_values: Vec<String>,
    pub sample_ambiguous_date_values: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TimestampNormalizationSummary {
    pub timestamp_column: String,
    pub original_name: String,
    pub rows_read: i64,
    pub rows_written: i64,
    pub explicit_count: i64,
    pub epoch_count: i64,
    pub naive_count: i64,
    pub blank_count: i64,
    pub invalid_count: i64,
    pub timezone_applied: Option<String>,
    pub date_convention_applied: Option<String>,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DateConvention {
    MonthFirst,
    DayFirst,
}

impl DateConvention {
    fn from_answer(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "month_first" | "monthfirst" | "mdy" | "us" => Ok(Self::MonthFirst),
            "day_first" | "dayfirst" | "dmy" | "eu" => Ok(Self::DayFirst),
            _ => {
                bail!("date convention must be month_first (MM/DD/YYYY) or day_first (DD/MM/YYYY)")
            }
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::MonthFirst => "month_first",
            Self::DayFirst => "day_first",
        }
    }
}

#[derive(Debug, Clone)]
enum ParsedTimestamp {
    Absolute {
        utc: DateTime<Utc>,
        parse_status: &'static str,
    },
    Naive(NaiveDateTime),
    AmbiguousDate,
    Blank,
    Invalid,
}

#[derive(Debug, Clone)]
enum TimezoneResolver {
    Fixed { offset: FixedOffset, label: String },
    Iana { timezone: Tz, label: String },
}

impl TimezoneResolver {
    fn from_answer(answer: &str) -> Result<Self> {
        let trimmed = answer.trim();
        if trimmed.is_empty() {
            bail!("timezone answer cannot be empty");
        }
        if let Some(offset) = parse_fixed_offset(trimmed) {
            return Ok(Self::Fixed {
                offset,
                label: trimmed.to_string(),
            });
        }
        let timezone = trimmed.parse::<Tz>().map_err(|_| {
            anyhow!(
                "timezone answer must be UTC, a fixed offset like +02:00, or an IANA name like Europe/Bucharest"
            )
        })?;
        Ok(Self::Iana {
            timezone,
            label: trimmed.to_string(),
        })
    }

    fn apply(
        &self,
        naive: NaiveDateTime,
        source_text: &str,
        row_num: i64,
    ) -> Result<DateTime<Utc>> {
        match self {
            Self::Fixed { offset, .. } => local_result_to_utc(
                offset.from_local_datetime(&naive),
                source_text,
                row_num,
                "fixed offset",
            ),
            Self::Iana { timezone, label } => local_result_to_utc(
                timezone.from_local_datetime(&naive),
                source_text,
                row_num,
                label,
            ),
        }
    }

    fn parse_status(&self) -> &'static str {
        match self {
            Self::Fixed { offset, .. } if offset.local_minus_utc() == 0 => "naive_confirmed_utc",
            Self::Fixed { .. } => "naive_with_fixed_offset",
            Self::Iana { .. } => "naive_with_iana_timezone",
        }
    }

    fn label(&self) -> &str {
        match self {
            Self::Fixed { label, .. } | Self::Iana { label, .. } => label,
        }
    }
}

pub fn classify_timestamp_text(value: &str) -> TimestampValueKind {
    match parse_timestamp(value, None) {
        ParsedTimestamp::Absolute {
            parse_status: "epoch",
            ..
        } => TimestampValueKind::Epoch,
        ParsedTimestamp::Absolute { .. } => TimestampValueKind::ExplicitOffset,
        ParsedTimestamp::Naive(_) | ParsedTimestamp::AmbiguousDate => TimestampValueKind::Naive,
        ParsedTimestamp::Blank => TimestampValueKind::Blank,
        ParsedTimestamp::Invalid => TimestampValueKind::Invalid,
    }
}

pub fn analyze_confirmed_timestamp_column(
    conn: &Connection,
    columns: &[ColumnMeta],
) -> Result<TimestampAnalysis> {
    let column = resolved_timestamp_column(conn, columns)?;
    analyze_timestamp_column(conn, &column)
}

pub fn analyze_timestamp_column(
    conn: &Connection,
    column: &ColumnMeta,
) -> Result<TimestampAnalysis> {
    let date_analysis = analyze_date_convention(conn, &column.sql_name)?;
    let mut counts = TimestampCounts::default();

    scan_timestamp_column(
        conn,
        &column.sql_name,
        date_analysis.inferred,
        |_, source_text, parsed| {
            counts.record(&source_text, &parsed);
            Ok(())
        },
    )?;

    Ok(TimestampAnalysis {
        timestamp_column: column.sql_name.clone(),
        original_name: column.original_name.clone(),
        total_rows: counts.total_rows,
        explicit_count: counts.explicit_count,
        epoch_count: counts.epoch_count,
        naive_count: counts.naive_count,
        blank_count: counts.blank_count,
        invalid_count: counts.invalid_count,
        needs_timezone: counts.naive_count > 0,
        needs_date_convention: date_analysis.conflicting
            || (!date_analysis.ambiguous_samples.is_empty() && date_analysis.inferred.is_none()),
        inferred_date_convention: date_analysis
            .inferred
            .map(|convention| convention.label().to_string()),
        sample_naive_values: counts.sample_naive_values,
        sample_invalid_values: counts.sample_invalid_values,
        sample_ambiguous_date_values: date_analysis.ambiguous_samples,
    })
}

pub fn normalize_confirmed_timestamp_column(
    conn: &mut Connection,
    columns: &[ColumnMeta],
    naive_timezone: Option<&str>,
) -> Result<TimestampNormalizationSummary> {
    normalize_timestamp_column_with_options(conn, columns, naive_timezone, None)
}

pub fn normalize_timestamp_column_with_options(
    conn: &mut Connection,
    columns: &[ColumnMeta],
    naive_timezone: Option<&str>,
    date_convention: Option<&str>,
) -> Result<TimestampNormalizationSummary> {
    normalize_timestamp_column_with_progress(
        conn,
        columns,
        naive_timezone,
        date_convention,
        |_, _| Ok(()),
    )
}

fn normalize_timestamp_column_with_progress(
    conn: &mut Connection,
    columns: &[ColumnMeta],
    naive_timezone: Option<&str>,
    date_convention: Option<&str>,
    mut after_batch: impl FnMut(i64, i64) -> Result<()>,
) -> Result<TimestampNormalizationSummary> {
    let column = resolved_timestamp_column(conn, columns)?;
    let date_analysis = analyze_date_convention(conn, &column.sql_name)?;
    if date_analysis.conflicting {
        bail!(
            "timestamp column mixes unambiguous MM/DD/YYYY and DD/MM/YYYY values; separate or correct the source data before normalization"
        );
    }
    let supplied_convention = date_convention
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(DateConvention::from_answer)
        .transpose()?;
    if let (Some(supplied), Some(inferred)) = (supplied_convention, date_analysis.inferred) {
        if supplied != inferred {
            bail!(
                "date convention '{}' contradicts unambiguous source values indicating '{}'",
                supplied.label(),
                inferred.label()
            );
        }
    }
    let resolved_date_convention = supplied_convention.or(date_analysis.inferred);
    if !date_analysis.ambiguous_samples.is_empty() && resolved_date_convention.is_none() {
        bail!(
            "timestamp column contains ambiguous slash dates such as '{}'; supply date_convention month_first or day_first in addition to any timezone",
            date_analysis.ambiguous_samples[0]
        );
    }
    let resolver = naive_timezone
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(TimezoneResolver::from_answer)
        .transpose()?;

    // Validate every conversion before creating staging state. The source scan is paginated, so
    // it never keeps a read statement open while a semantic/audit writer is trying to commit.
    let mut counts = TimestampCounts::default();
    scan_timestamp_column(
        conn,
        &column.sql_name,
        resolved_date_convention,
        |row_num, source_text, parsed| {
            counts.record(&source_text, &parsed);
            match &parsed {
                ParsedTimestamp::Naive(naive) => {
                    if let Some(resolver) = resolver.as_ref() {
                        // Validate DST gaps/overlaps before any staged row is written.
                        resolver.apply(*naive, &source_text, row_num)?;
                    }
                }
                ParsedTimestamp::AmbiguousDate => {
                    bail!("ambiguous slash date escaped date-convention validation")
                }
                ParsedTimestamp::Absolute { .. }
                | ParsedTimestamp::Blank
                | ParsedTimestamp::Invalid => {}
            }
            Ok(())
        },
    )?;

    if counts.naive_count > 0 && resolver.is_none() {
        bail!(
            "timestamp column contains {} naive timestamp value(s); supply a source UTC offset or IANA timezone before normalization",
            counts.naive_count
        );
    }

    db::create_row_time_table(conn)?;
    let binding = current_binding_values(conn, columns)?;
    let stage_name = unique_row_time_object_name("_row_time_stage");
    let stage_index_name = unique_row_time_object_name("idx_row_time_stage_epoch");
    create_row_time_staging_table(conn, &stage_name, &stage_index_name)?;

    let build_result = (|| -> Result<i64> {
        let rows_written = fill_row_time_staging_table(
            conn,
            &stage_name,
            &column.sql_name,
            resolved_date_convention,
            resolver.as_ref(),
            &mut after_batch,
        )?;
        let backup_name = publish_row_time_staging_table(
            conn,
            columns,
            &binding,
            &stage_name,
            &column.sql_name,
            resolved_date_convention,
            resolver.as_ref(),
        )?;

        // Publication is already durable and active. Reclaim the prior generation in bounded
        // transactions so cleanup cannot monopolize SQLite's single writer slot.
        let _ = discard_row_time_table_batched(conn, &backup_name);
        Ok(rows_written)
    })();

    let rows_written = match build_result {
        Ok(rows_written) => rows_written,
        Err(error) => {
            // The active generation was never modified if staging or publication failed.
            let _ = discard_row_time_table_batched(conn, &stage_name);
            return Err(error);
        }
    };

    Ok(TimestampNormalizationSummary {
        timestamp_column: column.sql_name,
        original_name: column.original_name,
        rows_read: counts.total_rows,
        rows_written,
        explicit_count: counts.explicit_count,
        epoch_count: counts.epoch_count,
        naive_count: counts.naive_count,
        blank_count: counts.blank_count,
        invalid_count: counts.invalid_count,
        timezone_applied: resolver.map(|resolver| resolver.label().to_string()),
        date_convention_applied: resolved_date_convention
            .map(|convention| convention.label().to_string()),
    })
}

fn unique_row_time_object_name(prefix: &str) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    let counter = ROW_TIME_BUILD_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}_{}_{}_{}", std::process::id(), nanos, counter)
}

fn create_row_time_staging_table(
    conn: &mut Connection,
    table_name: &str,
    index_name: &str,
) -> Result<()> {
    let table = db::quote_ident(table_name);
    let index = db::quote_ident(index_name);
    let tx = conn.transaction()?;
    tx.execute_batch(&format!(
        "CREATE TABLE {table} (
            row_num INTEGER PRIMARY KEY,
            epoch_ms INTEGER NOT NULL,
            utc_text TEXT NOT NULL,
            source_text TEXT NOT NULL,
            parse_status TEXT NOT NULL
         );
         CREATE INDEX {index} ON {table}(epoch_ms, row_num);"
    ))?;
    tx.commit()?;
    Ok(())
}

fn fill_row_time_staging_table(
    conn: &mut Connection,
    stage_name: &str,
    source_column: &str,
    date_convention: Option<DateConvention>,
    resolver: Option<&TimezoneResolver>,
    after_batch: &mut impl FnMut(i64, i64) -> Result<()>,
) -> Result<i64> {
    let stage = db::quote_ident(stage_name);
    let mut cursor = 0_i64;
    let mut rows_read = 0_i64;
    let mut rows_written = 0_i64;

    loop {
        let source_rows = load_timestamp_source_batch(conn, source_column, cursor)?;
        if source_rows.is_empty() {
            return Ok(rows_written);
        }
        cursor = source_rows
            .last()
            .map(|(row_num, _)| *row_num)
            .unwrap_or(cursor);
        rows_read += source_rows.len() as i64;

        let mut records = Vec::with_capacity(source_rows.len());
        for (row_num, source_text) in source_rows {
            match parse_timestamp(&source_text, date_convention) {
                ParsedTimestamp::Absolute { utc, parse_status } => {
                    records.push(row_time_record(row_num, utc, &source_text, parse_status));
                }
                ParsedTimestamp::Naive(naive) => {
                    let resolver = resolver.ok_or_else(|| {
                        anyhow!("timestamp validation changed while building normalized rows")
                    })?;
                    let utc = resolver.apply(naive, &source_text, row_num)?;
                    records.push(row_time_record(
                        row_num,
                        utc,
                        &source_text,
                        resolver.parse_status(),
                    ));
                }
                ParsedTimestamp::AmbiguousDate => {
                    bail!("ambiguous slash date escaped date-convention validation")
                }
                ParsedTimestamp::Blank | ParsedTimestamp::Invalid => {}
            }
        }

        let tx = conn.transaction()?;
        {
            let mut insert = tx.prepare(&format!(
                "INSERT INTO {stage} (row_num, epoch_ms, utc_text, source_text, parse_status)
                 VALUES (?1, ?2, ?3, ?4, ?5)"
            ))?;
            for record in &records {
                insert.execute(rusqlite::params![
                    record.row_num,
                    record.epoch_ms,
                    record.utc_text,
                    record.source_text,
                    record.parse_status
                ])?;
            }
        }
        tx.commit()?;
        rows_written += records.len() as i64;

        // Deliberately outside the transaction. Tests use this boundary to prove an unrelated
        // writer can commit between normalization batches.
        after_batch(rows_read, rows_written)?;
    }
}

fn publish_row_time_staging_table(
    conn: &mut Connection,
    columns: &[ColumnMeta],
    expected_binding: &BindingValues,
    stage_name: &str,
    source_column: &str,
    date_convention: Option<DateConvention>,
    resolver: Option<&TimezoneResolver>,
) -> Result<String> {
    let backup_name = unique_row_time_object_name("_row_time_previous");
    let stage = db::quote_ident(stage_name);
    let backup = db::quote_ident(&backup_name);
    let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let current_binding = current_binding_values(&tx, columns)?;
    if &current_binding != expected_binding {
        bail!("source dataset changed while timestamps were being normalized; the previous normalization remains active");
    }

    tx.execute_batch(&format!(
        "ALTER TABLE _row_time RENAME TO {backup};
         ALTER TABLE {stage} RENAME TO _row_time;"
    ))?;
    tx.execute("DELETE FROM _row_time_info", [])?;
    tx.execute(
        "INSERT INTO _row_time_info (
            binding_version, source_column, schema_sha256, import_sha256, row_count,
            date_convention, timezone_applied, completed_at
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        rusqlite::params![
            ROW_TIME_BINDING_VERSION,
            source_column,
            current_binding.schema_sha256,
            current_binding.import_sha256,
            current_binding.row_count,
            date_convention.map(DateConvention::label),
            resolver.map(TimezoneResolver::label),
            chrono::Utc::now().to_rfc3339(),
        ],
    )?;
    tx.commit()?;
    Ok(backup_name)
}

fn discard_row_time_table_batched(conn: &mut Connection, table_name: &str) -> Result<()> {
    let table = db::quote_ident(table_name);
    loop {
        let tx = conn.transaction()?;
        let deleted = tx.execute(
            &format!(
                "DELETE FROM {table}
                 WHERE row_num IN (
                    SELECT row_num FROM {table} ORDER BY row_num LIMIT {ROW_TIME_BATCH_ROWS}
                 )"
            ),
            [],
        )?;
        tx.commit()?;
        if deleted == 0 {
            break;
        }
    }
    conn.execute_batch(&format!("DROP TABLE IF EXISTS {table}"))?;
    Ok(())
}

#[derive(Debug, Default)]
struct TimestampCounts {
    total_rows: i64,
    explicit_count: i64,
    epoch_count: i64,
    naive_count: i64,
    blank_count: i64,
    invalid_count: i64,
    sample_naive_values: Vec<String>,
    sample_invalid_values: Vec<String>,
}

impl TimestampCounts {
    fn record(&mut self, source_text: &str, parsed: &ParsedTimestamp) {
        self.total_rows += 1;
        match parsed {
            ParsedTimestamp::Absolute {
                parse_status: "epoch",
                ..
            } => self.epoch_count += 1,
            ParsedTimestamp::Absolute { .. } => self.explicit_count += 1,
            ParsedTimestamp::Naive(_) | ParsedTimestamp::AmbiguousDate => {
                self.naive_count += 1;
                push_sample(&mut self.sample_naive_values, source_text);
            }
            ParsedTimestamp::Blank => self.blank_count += 1,
            ParsedTimestamp::Invalid => {
                self.invalid_count += 1;
                push_sample(&mut self.sample_invalid_values, source_text);
            }
        }
    }
}

#[derive(Debug)]
struct RowTimeRecord {
    row_num: i64,
    epoch_ms: i64,
    utc_text: String,
    source_text: String,
    parse_status: &'static str,
}

fn row_time_record(
    row_num: i64,
    utc: DateTime<Utc>,
    source_text: &str,
    parse_status: &'static str,
) -> RowTimeRecord {
    RowTimeRecord {
        row_num,
        epoch_ms: utc.timestamp_millis(),
        utc_text: utc.to_rfc3339_opts(SecondsFormat::AutoSi, true),
        source_text: source_text.to_string(),
        parse_status,
    }
}

#[derive(Debug, Eq, PartialEq)]
struct BindingValues {
    schema_sha256: String,
    import_sha256: String,
    row_count: i64,
}

#[derive(Debug)]
struct StoredBinding {
    binding_version: String,
    source_column: String,
    schema_sha256: String,
    import_sha256: String,
    row_count: i64,
}

pub fn row_time_is_bound_to(
    conn: &Connection,
    columns: &[ColumnMeta],
    source_column: &str,
) -> Result<bool> {
    let Some(stored) = load_row_time_binding(conn)? else {
        return Ok(false);
    };
    let current = current_binding_values(conn, columns)?;
    Ok(stored.binding_version == ROW_TIME_BINDING_VERSION
        && stored.source_column == source_column
        && stored.schema_sha256 == current.schema_sha256
        && stored.import_sha256 == current.import_sha256
        && stored.row_count == current.row_count)
}

pub fn require_row_time_binding(
    conn: &Connection,
    columns: &[ColumnMeta],
    source_column: &str,
) -> Result<()> {
    let Some(stored) = load_row_time_binding(conn)? else {
        bail!(
            "normalized timeline metadata is missing; normalize timestamp column '{source_column}' for this import"
        );
    };
    if stored.binding_version != ROW_TIME_BINDING_VERSION {
        bail!("timestamp normalization was created by an older unbound format; normalize it again");
    }
    if stored.source_column != source_column {
        bail!(
            "timestamp normalization is bound to column '{}', not '{}'; normalize the selected timeline column",
            stored.source_column,
            source_column
        );
    }
    let current = current_binding_values(conn, columns)?;
    if stored.schema_sha256 != current.schema_sha256
        || stored.import_sha256 != current.import_sha256
        || stored.row_count != current.row_count
    {
        bail!("timestamp normalization is stale for the current import; normalize it again");
    }
    Ok(())
}

fn load_row_time_binding(conn: &Connection) -> Result<Option<StoredBinding>> {
    if !table_exists(conn, "_row_time")? || !table_exists(conn, "_row_time_info")? {
        return Ok(None);
    }
    conn.query_row(
        "SELECT binding_version, source_column, schema_sha256, import_sha256, row_count
         FROM _row_time_info ORDER BY rowid DESC LIMIT 1",
        [],
        |row| {
            Ok(StoredBinding {
                binding_version: row.get(0)?,
                source_column: row.get(1)?,
                schema_sha256: row.get(2)?,
                import_sha256: row.get(3)?,
                row_count: row.get(4)?,
            })
        },
    )
    .optional()
    .map_err(Into::into)
}

fn current_binding_values(conn: &Connection, columns: &[ColumnMeta]) -> Result<BindingValues> {
    let schema_json = serde_json::to_string(columns)?;
    let import_info = db::load_import_info(conn).optional()?;
    let import_json = serde_json::to_string(&import_info)?;
    let row_count = conn.query_row("SELECT COUNT(*) FROM rows", [], |row| row.get(0))?;
    Ok(BindingValues {
        schema_sha256: sha256_text(&schema_json),
        import_sha256: sha256_text(&import_json),
        row_count,
    })
}

fn sha256_text(value: &str) -> String {
    Sha256::digest(value.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[derive(Debug, Default)]
struct DateConventionAnalysis {
    inferred: Option<DateConvention>,
    conflicting: bool,
    ambiguous_samples: Vec<String>,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum SlashDateEvidence {
    MonthFirst,
    DayFirst,
    Ambiguous,
}

fn analyze_date_convention(conn: &Connection, sql_name: &str) -> Result<DateConventionAnalysis> {
    let mut month_first = false;
    let mut day_first = false;
    let mut ambiguous_samples = Vec::new();
    let mut cursor = 0_i64;
    loop {
        let batch = load_timestamp_source_batch(conn, sql_name, cursor)?;
        if batch.is_empty() {
            break;
        }
        cursor = batch.last().map(|(row_num, _)| *row_num).unwrap_or(cursor);
        for (_, value) in batch {
            match slash_date_evidence(&value) {
                Some(SlashDateEvidence::MonthFirst) => month_first = true,
                Some(SlashDateEvidence::DayFirst) => day_first = true,
                Some(SlashDateEvidence::Ambiguous) => push_sample(&mut ambiguous_samples, &value),
                None => {}
            }
        }
    }
    let conflicting = month_first && day_first;
    let inferred = if conflicting {
        None
    } else if month_first {
        Some(DateConvention::MonthFirst)
    } else if day_first {
        Some(DateConvention::DayFirst)
    } else {
        None
    };
    Ok(DateConventionAnalysis {
        inferred,
        conflicting,
        ambiguous_samples,
    })
}

fn slash_date_evidence(value: &str) -> Option<SlashDateEvidence> {
    let date = value
        .trim()
        .split(|character: char| character.is_whitespace() || character == 'T')
        .next()?;
    let parts = date.split('/').collect::<Vec<_>>();
    if parts.len() != 3 || parts[2].len() != 4 {
        return None;
    }
    let first = parts[0].parse::<u32>().ok()?;
    let second = parts[1].parse::<u32>().ok()?;
    let year = parts[2].parse::<u32>().ok()?;
    if year < 1000 || first == 0 || second == 0 || first > 31 || second > 31 {
        return None;
    }
    match (first <= 12, second <= 12) {
        (true, true) => Some(SlashDateEvidence::Ambiguous),
        (true, false) => Some(SlashDateEvidence::MonthFirst),
        (false, true) => Some(SlashDateEvidence::DayFirst),
        (false, false) => None,
    }
}

/// Resolves the timestamp mapping without forcing an examiner through a blocking role-review
/// workflow. An explicit confirmation always wins. A high-confidence automatic suggestion is
/// safe to use for analysis because normalization still parses every value and refuses naive
/// timestamps until a timezone is supplied. As a final fallback, a single imported column whose
/// inferred type is timestamp-like is unambiguous enough to analyze; multiple candidates require
/// the user to choose in the optional Data mapping panel.
fn resolved_timestamp_column(conn: &Connection, columns: &[ColumnMeta]) -> Result<ColumnMeta> {
    const AUTOMATIC_CONFIDENCE: f64 = 0.75;
    db::create_column_roles_table(conn)?;
    let recorded: Option<(String, String, f64)> = conn
        .query_row(
            "SELECT sql_name, status, confidence FROM _column_roles
             WHERE role = 'timestamp' AND status != 'rejected'
             ORDER BY CASE status WHEN 'confirmed' THEN 0 ELSE 1 END, confidence DESC
             LIMIT 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .optional()?;

    if let Some((sql_name, status, confidence)) = recorded {
        if status == "confirmed" || confidence >= AUTOMATIC_CONFIDENCE {
            return columns
                .iter()
                .find(|column| column.sql_name == sql_name)
                .cloned()
                .ok_or_else(|| anyhow!("timestamp mapping no longer exists: {sql_name}"));
        }
    }

    let mut inferred = columns
        .iter()
        .filter(|column| column.inferred_type == "timestamp");
    let only = inferred.next().cloned();
    if only.is_some() && inferred.next().is_none() {
        return Ok(only.expect("one inferred timestamp"));
    }

    bail!(
        "timestamp mapping is ambiguous; choose the event-time column in Data mapping before building a timeline"
    )
}

fn scan_timestamp_column(
    conn: &Connection,
    sql_name: &str,
    date_convention: Option<DateConvention>,
    mut on_row: impl FnMut(i64, String, ParsedTimestamp) -> Result<()>,
) -> Result<()> {
    let mut cursor = 0_i64;
    loop {
        let batch = load_timestamp_source_batch(conn, sql_name, cursor)?;
        if batch.is_empty() {
            return Ok(());
        }
        cursor = batch.last().map(|(row_num, _)| *row_num).unwrap_or(cursor);
        for (row_num, source_text) in batch {
            let parsed = parse_timestamp(&source_text, date_convention);
            on_row(row_num, source_text, parsed)?;
        }
    }
}

fn load_timestamp_source_batch(
    conn: &Connection,
    sql_name: &str,
    after_row_num: i64,
) -> Result<Vec<(i64, String)>> {
    let ident = db::quote_ident(sql_name);
    let sql = format!(
        "SELECT row_num, {ident} FROM rows
         WHERE row_num > ?1 ORDER BY row_num ASC LIMIT {ROW_TIME_BATCH_ROWS}"
    );
    let mut stmt = conn.prepare(&sql)?;
    let mut rows = stmt.query([after_row_num])?;
    let mut batch = Vec::with_capacity(ROW_TIME_BATCH_ROWS);
    while let Some(row) = rows.next()? {
        let row_num: i64 = row.get(0)?;
        let source_text: Option<String> = row.get(1)?;
        batch.push((row_num, source_text.unwrap_or_default()));
    }
    Ok(batch)
}

fn parse_timestamp(value: &str, date_convention: Option<DateConvention>) -> ParsedTimestamp {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return ParsedTimestamp::Blank;
    }
    if let Some(utc) = parse_epoch(trimmed) {
        return ParsedTimestamp::Absolute {
            utc,
            parse_status: "epoch",
        };
    }
    if let Some(utc) = parse_explicit_offset(trimmed) {
        return ParsedTimestamp::Absolute {
            utc,
            parse_status: "explicit_offset",
        };
    }
    if slash_date_evidence(trimmed) == Some(SlashDateEvidence::Ambiguous)
        && date_convention.is_none()
    {
        return ParsedTimestamp::AmbiguousDate;
    }
    if let Some(naive) = parse_naive(trimmed, date_convention) {
        return ParsedTimestamp::Naive(naive);
    }
    ParsedTimestamp::Invalid
}

fn parse_epoch(value: &str) -> Option<DateTime<Utc>> {
    if !value.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    match value.len() {
        10 => {
            let seconds = value.parse::<i64>().ok()?;
            Utc.timestamp_opt(seconds, 0).single()
        }
        13 => {
            let millis = value.parse::<i64>().ok()?;
            Utc.timestamp_millis_opt(millis).single()
        }
        _ => None,
    }
}

fn parse_explicit_offset(value: &str) -> Option<DateTime<Utc>> {
    if let Ok(parsed) = DateTime::parse_from_rfc3339(value) {
        return Some(parsed.with_timezone(&Utc));
    }

    if value.ends_with('Z') || value.ends_with('z') {
        let without_z = &value[..value.len() - 1];
        if let Some(naive) = parse_naive(without_z.trim_end(), None) {
            return Some(Utc.from_utc_datetime(&naive));
        }
    }

    const FORMATS: &[&str] = &[
        "%Y-%m-%d %H:%M:%S%.f %:z",
        "%Y-%m-%d %H:%M:%S%.f%:z",
        "%Y-%m-%dT%H:%M:%S%.f%:z",
        "%Y/%m/%d %H:%M:%S%.f %:z",
        "%Y/%m/%d %H:%M:%S%.f%:z",
        "%Y-%m-%d %H:%M:%S%.f %z",
        "%Y-%m-%d %H:%M:%S%.f%z",
        "%Y-%m-%dT%H:%M:%S%.f%z",
    ];
    FORMATS.iter().find_map(|format| {
        DateTime::parse_from_str(value, format)
            .ok()
            .map(|parsed| parsed.with_timezone(&Utc))
    })
}

fn parse_naive(value: &str, date_convention: Option<DateConvention>) -> Option<NaiveDateTime> {
    const DATETIME_FORMATS: &[&str] = &[
        "%Y-%m-%dT%H:%M:%S%.f",
        "%Y-%m-%d %H:%M:%S%.f",
        "%Y/%m/%d %H:%M:%S%.f",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M",
    ];
    if let Some(parsed) = DATETIME_FORMATS
        .iter()
        .find_map(|format| NaiveDateTime::parse_from_str(value, format).ok())
    {
        return Some(parsed);
    }

    const DATE_FORMATS: &[&str] = &["%Y-%m-%d", "%Y/%m/%d"];
    if let Some(parsed) = DATE_FORMATS.iter().find_map(|format| {
        NaiveDate::parse_from_str(value, format)
            .ok()
            .and_then(|date| date.and_hms_opt(0, 0, 0))
    }) {
        return Some(parsed);
    }

    let convention = date_convention.or_else(|| match slash_date_evidence(value) {
        Some(SlashDateEvidence::MonthFirst) => Some(DateConvention::MonthFirst),
        Some(SlashDateEvidence::DayFirst) => Some(DateConvention::DayFirst),
        Some(SlashDateEvidence::Ambiguous) | None => None,
    })?;
    let (datetime_formats, date_format): (&[&str], &str) = match convention {
        DateConvention::MonthFirst => (&["%m/%d/%Y %H:%M:%S%.f", "%m/%d/%Y %H:%M"], "%m/%d/%Y"),
        DateConvention::DayFirst => (&["%d/%m/%Y %H:%M:%S%.f", "%d/%m/%Y %H:%M"], "%d/%m/%Y"),
    };
    datetime_formats
        .iter()
        .find_map(|format| NaiveDateTime::parse_from_str(value, format).ok())
        .or_else(|| {
            NaiveDate::parse_from_str(value, date_format)
                .ok()
                .and_then(|date| date.and_hms_opt(0, 0, 0))
        })
}

fn parse_fixed_offset(value: &str) -> Option<FixedOffset> {
    let upper = value.to_ascii_uppercase();
    if matches!(upper.as_str(), "UTC" | "Z" | "ETC/UTC" | "GMT") {
        return FixedOffset::east_opt(0);
    }

    let raw = upper
        .strip_prefix("UTC")
        .or_else(|| upper.strip_prefix("GMT"))
        .unwrap_or(upper.as_str());
    let sign = match raw.as_bytes().first()? {
        b'+' => 1,
        b'-' => -1,
        _ => return None,
    };
    let rest = &raw[1..];
    let (hours, minutes) = if let Some((hours, minutes)) = rest.split_once(':') {
        (hours.parse::<i32>().ok()?, minutes.parse::<i32>().ok()?)
    } else if rest.len() == 2 {
        (rest.parse::<i32>().ok()?, 0)
    } else if rest.len() == 4 {
        (
            rest[..2].parse::<i32>().ok()?,
            rest[2..].parse::<i32>().ok()?,
        )
    } else {
        return None;
    };
    if hours > 23 || minutes > 59 {
        return None;
    }
    FixedOffset::east_opt(sign * ((hours * 3600) + (minutes * 60)))
}

fn local_result_to_utc<Tz: TimeZone>(
    result: LocalResult<DateTime<Tz>>,
    source_text: &str,
    row_num: i64,
    zone_label: &str,
) -> Result<DateTime<Utc>> {
    match result {
        LocalResult::Single(value) => Ok(value.with_timezone(&Utc)),
        LocalResult::Ambiguous(_, _) => bail!(
            "row {row_num} timestamp '{source_text}' is ambiguous in {zone_label}; supply a fixed UTC offset"
        ),
        LocalResult::None => bail!(
            "row {row_num} timestamp '{source_text}' does not exist in {zone_label}; supply a fixed UTC offset"
        ),
    }
}

fn push_sample(samples: &mut Vec<String>, value: &str) {
    if samples.len() < 5 {
        samples.push(value.to_string());
    }
}

fn table_exists(conn: &Connection, table: &str) -> rusqlite::Result<bool> {
    conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?1)",
        [table],
        |row| row.get::<_, i64>(0),
    )
    .map(|value| value != 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    struct TestDbFile(PathBuf);

    impl TestDbFile {
        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TestDbFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
            for suffix in ["-journal", "-wal", "-shm"] {
                let _ = std::fs::remove_file(format!("{}{suffix}", self.0.display()));
            }
        }
    }

    fn setup_with_timestamp(values: &[&str]) -> (Connection, Vec<ColumnMeta>) {
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
        ];
        db::create_schema(&conn, &columns).unwrap();
        db::create_column_roles_table(&conn).unwrap();
        conn.execute(
            "INSERT INTO _column_roles (role, sql_name, confidence, status, reasons_json)
             VALUES ('timestamp', 'timegenerated', 1.0, 'confirmed', '[]')",
            [],
        )
        .unwrap();
        for (idx, value) in values.iter().enumerate() {
            conn.execute(
                "INSERT INTO rows (row_num, timegenerated, account) VALUES (?1, ?2, 'alice')",
                rusqlite::params![(idx as i64) + 1, value],
            )
            .unwrap();
        }
        (conn, columns)
    }

    fn setup_file_with_naive_timestamps(row_count: i64) -> (TestDbFile, Vec<ColumnMeta>) {
        let unique = ROW_TIME_BUILD_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "log-parser-row-time-{}-{}-{unique}.sqlite3",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let mut conn = db::open(&path).unwrap();
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
        ];
        db::create_schema(&conn, &columns).unwrap();
        db::create_column_roles_table(&conn).unwrap();
        conn.execute(
            "INSERT INTO _column_roles (role, sql_name, confidence, status, reasons_json)
             VALUES ('timestamp', 'timegenerated', 1.0, 'confirmed', '[]')",
            [],
        )
        .unwrap();
        conn.execute_batch(
            "CREATE TABLE _test_audit (
                id INTEGER PRIMARY KEY,
                action TEXT NOT NULL
             );",
        )
        .unwrap();
        let tx = conn.transaction().unwrap();
        {
            let mut insert = tx
                .prepare(
                    "INSERT INTO rows (row_num, timegenerated, account)
                     VALUES (?1, '2026-01-01 02:30:00', 'alice')",
                )
                .unwrap();
            for row_num in 1..=row_count {
                insert.execute([row_num]).unwrap();
            }
        }
        tx.commit().unwrap();
        drop(conn);
        (TestDbFile(path), columns)
    }

    fn inactive_row_time_table_count(conn: &Connection) -> i64 {
        conn.query_row(
            "SELECT COUNT(*) FROM sqlite_master
             WHERE type = 'table'
               AND (name GLOB '_row_time_stage_*'
                    OR name GLOB '_row_time_previous_*')",
            [],
            |row| row.get(0),
        )
        .unwrap()
    }

    #[test]
    fn normalizes_iso8601_offsets_to_utc_epoch_ms() {
        let (mut conn, columns) = setup_with_timestamp(&["2026-01-01T02:30:00+02:00"]);

        let analysis = analyze_confirmed_timestamp_column(&conn, &columns).unwrap();
        assert!(!analysis.needs_timezone);
        assert_eq!(analysis.explicit_count, 1);

        let summary = normalize_confirmed_timestamp_column(&mut conn, &columns, None).unwrap();
        assert_eq!(summary.rows_written, 1);

        let (epoch_ms, utc_text): (i64, String) = conn
            .query_row(
                "SELECT epoch_ms, utc_text FROM _row_time WHERE row_num = 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        let expected = Utc
            .with_ymd_and_hms(2026, 1, 1, 0, 30, 0)
            .single()
            .unwrap()
            .timestamp_millis();
        assert_eq!(epoch_ms, expected);
        assert_eq!(utc_text, "2026-01-01T00:30:00Z");
    }

    #[test]
    fn high_confidence_automatic_timestamp_mapping_does_not_need_confirmation() {
        let (mut conn, columns) = setup_with_timestamp(&["2026-01-01T02:30:00+02:00"]);
        conn.execute(
            "UPDATE _column_roles
             SET confidence = 0.93, status = 'suggested'
             WHERE role = 'timestamp'",
            [],
        )
        .unwrap();

        let analysis = analyze_confirmed_timestamp_column(&conn, &columns).unwrap();
        assert_eq!(analysis.timestamp_column, "timegenerated");
        assert!(!analysis.needs_timezone);
        let normalized = normalize_confirmed_timestamp_column(&mut conn, &columns, None).unwrap();
        assert_eq!(normalized.rows_written, 1);
    }

    #[test]
    fn weak_automatic_timestamp_mapping_requests_a_data_mapping_choice() {
        let (conn, columns) = setup_with_timestamp(&["2026-01-01T02:30:00+02:00"]);
        conn.execute(
            "UPDATE _column_roles
             SET confidence = 0.40, status = 'suggested'
             WHERE role = 'timestamp'",
            [],
        )
        .unwrap();

        let error = analyze_confirmed_timestamp_column(&conn, &columns).unwrap_err();
        assert!(error.to_string().contains("Data mapping"));
    }

    #[test]
    fn naive_timestamps_are_flagged_ambiguous_without_examiner_timezone() {
        let (mut conn, columns) = setup_with_timestamp(&["2026-01-01 02:30:00"]);

        let analysis = analyze_confirmed_timestamp_column(&conn, &columns).unwrap();
        assert!(analysis.needs_timezone);
        assert_eq!(analysis.naive_count, 1);

        let err = normalize_confirmed_timestamp_column(&mut conn, &columns, None)
            .expect_err("naive timestamps must not be normalized without examiner input");
        assert!(err.to_string().contains("supply a source UTC offset"));
    }

    #[test]
    fn examiner_fixed_offset_normalizes_naive_timestamps() {
        let (mut conn, columns) = setup_with_timestamp(&["2026-01-01 02:30:00"]);

        let summary =
            normalize_confirmed_timestamp_column(&mut conn, &columns, Some("+02:00")).unwrap();
        assert_eq!(summary.rows_written, 1);
        assert_eq!(summary.timezone_applied.as_deref(), Some("+02:00"));

        let epoch_ms: i64 = conn
            .query_row(
                "SELECT epoch_ms FROM _row_time WHERE row_num = 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        let expected = Utc
            .with_ymd_and_hms(2026, 1, 1, 0, 30, 0)
            .single()
            .unwrap()
            .timestamp_millis();
        assert_eq!(epoch_ms, expected);
    }

    #[test]
    fn infers_month_first_from_unambiguous_us_values() {
        let (mut conn, columns) = setup_with_timestamp(&["03/14/2026 01:00", "03/04/2026 02:00"]);

        let analysis = analyze_confirmed_timestamp_column(&conn, &columns).unwrap();
        assert_eq!(
            analysis.inferred_date_convention.as_deref(),
            Some("month_first")
        );
        assert!(!analysis.needs_date_convention);
        let summary =
            normalize_timestamp_column_with_options(&mut conn, &columns, Some("UTC"), None)
                .unwrap();
        assert_eq!(
            summary.date_convention_applied.as_deref(),
            Some("month_first")
        );
        let utc_text: String = conn
            .query_row(
                "SELECT utc_text FROM _row_time WHERE row_num = 2",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(utc_text, "2026-03-04T02:00:00Z");
        let stored: String = conn
            .query_row("SELECT date_convention FROM _row_time_info", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(stored, "month_first");
    }

    #[test]
    fn infers_day_first_from_unambiguous_european_values() {
        let (mut conn, columns) = setup_with_timestamp(&["14/03/2026 01:00", "03/04/2026 02:00"]);

        let analysis = analyze_confirmed_timestamp_column(&conn, &columns).unwrap();
        assert_eq!(
            analysis.inferred_date_convention.as_deref(),
            Some("day_first")
        );
        assert!(!analysis.needs_date_convention);
        normalize_timestamp_column_with_options(&mut conn, &columns, Some("UTC"), None).unwrap();
        let utc_text: String = conn
            .query_row(
                "SELECT utc_text FROM _row_time WHERE row_num = 2",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(utc_text, "2026-04-03T02:00:00Z");
    }

    #[test]
    fn ambiguous_slash_date_requires_convention_even_with_timezone() {
        let (mut conn, columns) = setup_with_timestamp(&["03/04/2026 02:00"]);

        let analysis = analyze_confirmed_timestamp_column(&conn, &columns).unwrap();
        assert!(analysis.needs_date_convention);
        assert!(analysis.needs_timezone);
        let error = normalize_timestamp_column_with_options(&mut conn, &columns, Some("UTC"), None)
            .expect_err("timezone must not silently choose the date order");
        assert!(error.to_string().contains("date_convention"));

        let summary = normalize_timestamp_column_with_options(
            &mut conn,
            &columns,
            Some("UTC"),
            Some("month_first"),
        )
        .unwrap();
        assert_eq!(
            summary.date_convention_applied.as_deref(),
            Some("month_first")
        );
    }

    #[test]
    fn mixed_unambiguous_slash_conventions_are_rejected() {
        let (mut conn, columns) = setup_with_timestamp(&["03/14/2026 01:00", "14/03/2026 01:00"]);

        let analysis = analyze_confirmed_timestamp_column(&conn, &columns).unwrap();
        assert!(analysis.needs_date_convention);
        let error = normalize_timestamp_column_with_options(
            &mut conn,
            &columns,
            Some("UTC"),
            Some("month_first"),
        )
        .expect_err("conflicting source conventions cannot be normalized safely");
        assert!(error.to_string().contains("mixes unambiguous"));
    }

    #[test]
    fn normalization_releases_writer_lock_between_batches_and_publishes_atomically() {
        let row_count = (ROW_TIME_BATCH_ROWS as i64) + 1;
        let (db_file, columns) = setup_file_with_naive_timestamps(row_count);
        let mut initial = db::open(db_file.path()).unwrap();
        normalize_timestamp_column_with_options(&mut initial, &columns, Some("UTC"), None).unwrap();
        drop(initial);

        let normalize_path = db_file.path().to_path_buf();
        let normalize_columns = columns.clone();
        let (paused_tx, paused_rx) = mpsc::channel();
        let (resume_tx, resume_rx) = mpsc::channel();
        let normalize_thread = thread::spawn(move || {
            let mut conn = db::open(&normalize_path).unwrap();
            let mut paused = false;
            normalize_timestamp_column_with_progress(
                &mut conn,
                &normalize_columns,
                Some("+02:00"),
                None,
                |rows_read, _| {
                    if rows_read >= ROW_TIME_BATCH_ROWS as i64 && !paused {
                        paused = true;
                        paused_tx.send(()).unwrap();
                        resume_rx.recv().unwrap();
                    }
                    Ok(())
                },
            )
        });

        paused_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("normalization should pause after its first committed batch");
        let observer = db::open(db_file.path()).unwrap();
        let still_published: String = observer
            .query_row(
                "SELECT utc_text FROM _row_time WHERE row_num = 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(still_published, "2026-01-01T02:30:00Z");
        observer
            .execute("INSERT INTO _test_audit (action) VALUES ('accepted')", [])
            .expect("timestamp normalization must not retain a write transaction between batches");
        resume_tx.send(()).unwrap();

        let summary = normalize_thread.join().unwrap().unwrap();
        assert_eq!(summary.rows_read, row_count);
        assert_eq!(summary.rows_written, row_count);
        let newly_published: String = observer
            .query_row(
                "SELECT utc_text FROM _row_time WHERE row_num = 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(newly_published, "2026-01-01T00:30:00Z");
        let stored_timezone: String = observer
            .query_row("SELECT timezone_applied FROM _row_time_info", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(stored_timezone, "+02:00");
        let audit_count: i64 = observer
            .query_row("SELECT COUNT(*) FROM _test_audit", [], |row| row.get(0))
            .unwrap();
        assert_eq!(audit_count, 1);
        assert_eq!(inactive_row_time_table_count(&observer), 0);
    }

    #[test]
    fn failed_staged_normalization_preserves_previous_result_and_cleans_up() {
        let row_count = (ROW_TIME_BATCH_ROWS as i64) + 1;
        let (db_file, columns) = setup_file_with_naive_timestamps(row_count);
        let mut conn = db::open(db_file.path()).unwrap();
        normalize_timestamp_column_with_options(&mut conn, &columns, Some("UTC"), None).unwrap();

        let error = normalize_timestamp_column_with_progress(
            &mut conn,
            &columns,
            Some("+02:00"),
            None,
            |rows_read, _| {
                if rows_read >= ROW_TIME_BATCH_ROWS as i64 {
                    return Err(anyhow::anyhow!("injected failure after committed batch"));
                }
                Ok(())
            },
        )
        .expect_err("injected staging failure must abort normalization");
        assert!(error.to_string().contains("injected failure"));

        let still_published: String = conn
            .query_row(
                "SELECT utc_text FROM _row_time WHERE row_num = 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(still_published, "2026-01-01T02:30:00Z");
        let stored_timezone: String = conn
            .query_row("SELECT timezone_applied FROM _row_time_info", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(stored_timezone, "UTC");
        assert_eq!(inactive_row_time_table_count(&conn), 0);
    }

    #[test]
    fn row_time_binding_detects_wrong_column_stale_rows_and_old_tables() {
        let (mut conn, columns) = setup_with_timestamp(&["2026-01-01T00:00:00Z"]);
        normalize_confirmed_timestamp_column(&mut conn, &columns, None).unwrap();
        assert!(row_time_is_bound_to(&conn, &columns, "timegenerated").unwrap());
        assert!(!row_time_is_bound_to(&conn, &columns, "account").unwrap());
        let wrong = require_row_time_binding(&conn, &columns, "account").unwrap_err();
        assert!(wrong
            .to_string()
            .contains("bound to column 'timegenerated'"));

        conn.execute(
            "INSERT INTO rows (row_num, timegenerated, account)
             VALUES (2, '2026-01-02T00:00:00Z', 'alice')",
            [],
        )
        .unwrap();
        assert!(!row_time_is_bound_to(&conn, &columns, "timegenerated").unwrap());
        assert!(require_row_time_binding(&conn, &columns, "timegenerated")
            .unwrap_err()
            .to_string()
            .contains("stale"));

        conn.execute("DELETE FROM _row_time_info", []).unwrap();
        assert!(!row_time_is_bound_to(&conn, &columns, "timegenerated").unwrap());
        assert!(require_row_time_binding(&conn, &columns, "timegenerated")
            .unwrap_err()
            .to_string()
            .contains("metadata is missing"));
    }
}
