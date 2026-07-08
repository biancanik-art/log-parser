use crate::db::{self, ColumnMeta};
use anyhow::{anyhow, bail, Result};
use chrono::{
    DateTime, FixedOffset, LocalResult, NaiveDate, NaiveDateTime, SecondsFormat, TimeZone, Utc,
};
use chrono_tz::Tz;
use rusqlite::Connection;
use serde::Serialize;

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
    pub sample_naive_values: Vec<String>,
    pub sample_invalid_values: Vec<String>,
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
}

#[derive(Debug, Clone)]
enum ParsedTimestamp {
    Absolute {
        utc: DateTime<Utc>,
        parse_status: &'static str,
    },
    Naive(NaiveDateTime),
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
    match parse_timestamp(value) {
        ParsedTimestamp::Absolute {
            parse_status: "epoch",
            ..
        } => TimestampValueKind::Epoch,
        ParsedTimestamp::Absolute { .. } => TimestampValueKind::ExplicitOffset,
        ParsedTimestamp::Naive(_) => TimestampValueKind::Naive,
        ParsedTimestamp::Blank => TimestampValueKind::Blank,
        ParsedTimestamp::Invalid => TimestampValueKind::Invalid,
    }
}

pub fn analyze_confirmed_timestamp_column(
    conn: &Connection,
    columns: &[ColumnMeta],
) -> Result<TimestampAnalysis> {
    let column = confirmed_timestamp_column(conn, columns)?;
    let mut counts = TimestampCounts::default();

    scan_timestamp_column(conn, &column.sql_name, |_, source_text, parsed| {
        counts.record(&source_text, &parsed);
        Ok(())
    })?;

    Ok(TimestampAnalysis {
        timestamp_column: column.sql_name,
        original_name: column.original_name,
        total_rows: counts.total_rows,
        explicit_count: counts.explicit_count,
        epoch_count: counts.epoch_count,
        naive_count: counts.naive_count,
        blank_count: counts.blank_count,
        invalid_count: counts.invalid_count,
        needs_timezone: counts.naive_count > 0,
        sample_naive_values: counts.sample_naive_values,
        sample_invalid_values: counts.sample_invalid_values,
    })
}

pub fn normalize_confirmed_timestamp_column(
    conn: &mut Connection,
    columns: &[ColumnMeta],
    naive_timezone: Option<&str>,
) -> Result<TimestampNormalizationSummary> {
    let column = confirmed_timestamp_column(conn, columns)?;
    let resolver = naive_timezone
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(TimezoneResolver::from_answer)
        .transpose()?;

    let mut counts = TimestampCounts::default();
    let mut rows_to_write = Vec::new();

    scan_timestamp_column(conn, &column.sql_name, |row_num, source_text, parsed| {
        counts.record(&source_text, &parsed);
        match parsed {
            ParsedTimestamp::Absolute { utc, parse_status } => {
                rows_to_write.push(row_time_record(row_num, utc, &source_text, parse_status));
            }
            ParsedTimestamp::Naive(naive) => {
                if let Some(resolver) = resolver.as_ref() {
                    let utc = resolver.apply(naive, &source_text, row_num)?;
                    rows_to_write.push(row_time_record(
                        row_num,
                        utc,
                        &source_text,
                        resolver.parse_status(),
                    ));
                }
            }
            ParsedTimestamp::Blank | ParsedTimestamp::Invalid => {}
        }
        Ok(())
    })?;

    if counts.naive_count > 0 && resolver.is_none() {
        bail!(
            "timestamp column contains {} naive timestamp value(s); supply a source UTC offset or IANA timezone before normalization",
            counts.naive_count
        );
    }

    db::create_row_time_table(conn)?;
    let tx = conn.transaction()?;
    tx.execute("DELETE FROM _row_time", [])?;
    {
        let mut stmt = tx.prepare(
            "INSERT INTO _row_time (row_num, epoch_ms, utc_text, source_text, parse_status)
             VALUES (?1, ?2, ?3, ?4, ?5)",
        )?;
        for record in &rows_to_write {
            stmt.execute(rusqlite::params![
                record.row_num,
                record.epoch_ms,
                record.utc_text,
                record.source_text,
                record.parse_status
            ])?;
        }
    }
    tx.commit()?;

    Ok(TimestampNormalizationSummary {
        timestamp_column: column.sql_name,
        original_name: column.original_name,
        rows_read: counts.total_rows,
        rows_written: rows_to_write.len() as i64,
        explicit_count: counts.explicit_count,
        epoch_count: counts.epoch_count,
        naive_count: counts.naive_count,
        blank_count: counts.blank_count,
        invalid_count: counts.invalid_count,
        timezone_applied: resolver.map(|resolver| resolver.label().to_string()),
    })
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
            ParsedTimestamp::Naive(_) => {
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

fn confirmed_timestamp_column(conn: &Connection, columns: &[ColumnMeta]) -> Result<ColumnMeta> {
    db::create_column_roles_table(conn)?;
    let sql_name: String = conn
        .query_row(
            "SELECT sql_name FROM _column_roles
             WHERE role = 'timestamp' AND status = 'confirmed'
             LIMIT 1",
            [],
            |row| row.get(0),
        )
        .map_err(|_| anyhow!("confirm a timestamp column role before timestamp normalization"))?;

    columns
        .iter()
        .find(|column| column.sql_name == sql_name)
        .cloned()
        .ok_or_else(|| anyhow!("confirmed timestamp column no longer exists: {sql_name}"))
}

fn scan_timestamp_column(
    conn: &Connection,
    sql_name: &str,
    mut on_row: impl FnMut(i64, String, ParsedTimestamp) -> Result<()>,
) -> Result<()> {
    let ident = db::quote_ident(sql_name);
    let sql = format!("SELECT row_num, {ident} FROM rows ORDER BY row_num ASC");
    let mut stmt = conn.prepare(&sql)?;
    let mut rows = stmt.query([])?;
    while let Some(row) = rows.next()? {
        let row_num: i64 = row.get(0)?;
        let source_text: Option<String> = row.get(1)?;
        let source_text = source_text.unwrap_or_default();
        let parsed = parse_timestamp(&source_text);
        on_row(row_num, source_text, parsed)?;
    }
    Ok(())
}

fn parse_timestamp(value: &str) -> ParsedTimestamp {
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
    if let Some(naive) = parse_naive(trimmed) {
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
        if let Some(naive) = parse_naive(without_z.trim_end()) {
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

fn parse_naive(value: &str) -> Option<NaiveDateTime> {
    const DATETIME_FORMATS: &[&str] = &[
        "%Y-%m-%dT%H:%M:%S%.f",
        "%Y-%m-%d %H:%M:%S%.f",
        "%Y/%m/%d %H:%M:%S%.f",
        "%m/%d/%Y %H:%M:%S%.f",
        "%d/%m/%Y %H:%M:%S%.f",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M",
        "%m/%d/%Y %H:%M",
        "%d/%m/%Y %H:%M",
    ];
    if let Some(parsed) = DATETIME_FORMATS
        .iter()
        .find_map(|format| NaiveDateTime::parse_from_str(value, format).ok())
    {
        return Some(parsed);
    }

    const DATE_FORMATS: &[&str] = &["%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y"];
    DATE_FORMATS.iter().find_map(|format| {
        NaiveDate::parse_from_str(value, format)
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
