use crate::db;
use crate::intel::library::{
    self, normalize_header_token, RuleCondition, RULE_CONDITION_ROLES,
};
use crate::intel::rule_conditions;
use anyhow::{anyhow, bail, Context, Result};
use rusqlite::{Connection, OptionalExtension, TransactionBehavior};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

const IGNORE_SCAN_BATCH_ROWS: i64 = 1000;
const MAX_MATCH_VALUE_CHARS: usize = 48;

/// Bounds a matched cell value before it's stored in `_ignored_rows` — mirrors
/// `matcher::bounded_match_value` so an arbitrarily long free-text cell can't bloat the table.
fn bounded_match_value(value: &str) -> String {
    if value.chars().count() <= MAX_MATCH_VALUE_CHARS {
        value.to_string()
    } else {
        let bounded: String = value.chars().take(MAX_MATCH_VALUE_CHARS).collect();
        format!("{bounded}…")
    }
}

static BUILTIN_IGNORE_RULES_PATH: OnceLock<PathBuf> = OnceLock::new();
const BUILTIN_IGNORE_RULES_SHA256: &str =
    "455490e5628c7a221ae945fe9a756d76400b5f0dba576fafdc647176eef2bd43";

const MAX_RULE_CONDITIONS: usize = 8;
const MAX_RULE_VALUES: usize = 64;

/// Configures the bundled ignore-rules resource shipped with the app, mirroring
/// `library::configure_builtin_library_path`. Verified on load and again on every read so a
/// resource changed after startup cannot silently alter which rows get suppressed.
pub fn configure_builtin_ignore_rules_path(path: PathBuf) -> Result<()> {
    if !path.is_file() {
        bail!(
            "bundled ignore-rules resource was not found at {}",
            path.display()
        );
    }
    let _ = read_verified_builtin_ignore_rules(&path)?;
    BUILTIN_IGNORE_RULES_PATH
        .set(path)
        .map_err(|_| anyhow!("built-in ignore-rules path was already configured"))
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct IgnoreRuleFile {
    pub schema_version: u32,
    pub rule_set_id: String,
    pub rules: Vec<IgnoreRule>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct IgnoreRule {
    pub id: String,
    pub name: String,
    #[serde(default = "default_true")]
    pub enabled: bool,
    pub conditions: Vec<RuleCondition>,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuleSource {
    Builtin,
    Custom,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EffectiveIgnoreRule {
    pub id: String,
    pub name: String,
    pub enabled: bool,
    pub source: RuleSource,
    pub conditions: Vec<RuleCondition>,
}

#[derive(Debug, Clone)]
pub struct MergedIgnoreRules {
    pub rules: Vec<EffectiveIgnoreRule>,
    /// Content hash over every source file that fed this merge (built-in + custom, if present).
    /// Two merges with the same `rules_hash` produced byte-identical effective rules.
    pub rules_hash: String,
    pub custom_rules_error: Option<String>,
}

impl MergedIgnoreRules {
    pub fn enabled_rules(&self) -> impl Iterator<Item = &EffectiveIgnoreRule> {
        self.rules.iter().filter(|rule| rule.enabled)
    }
}

pub fn load_builtin_ignore_rules() -> Result<IgnoreRuleFile> {
    let raw = builtin_ignore_rules_json()?;
    parse_ignore_rule_file("built-in ignore rules", raw.as_ref())
}

/// Merges the built-in rule catalog with this file's own overrides/custom rules. Global vs.
/// per-file split: the built-in rule *definitions* (what Qualys/msedge-crashpad match on) are
/// one shared, checksum-pinned catalog — that's just "what's available," not "what's active."
/// Which built-ins are enabled and any rules an examiner adds live in `conn`'s own database
/// (`_ignore_rule_overrides`/`_custom_ignore_rules`, created via `db::create_ignore_rule_state_schema`),
/// exactly like `_column_roles` — a rule toggled or added while working one file never touches
/// any other file.
pub fn load_merged_ignore_rules(conn: &Connection) -> Result<MergedIgnoreRules> {
    let builtin_raw = builtin_ignore_rules_json()?;
    let builtin = parse_ignore_rule_file("built-in ignore rules", builtin_raw.as_ref())?;

    db::create_ignore_rule_state_schema(conn)?;

    let mut overrides: HashMap<String, bool> = HashMap::new();
    {
        let mut stmt = conn.prepare("SELECT rule_id, enabled FROM _ignore_rule_overrides")?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let rule_id: String = row.get(0)?;
            let enabled: i64 = row.get(1)?;
            overrides.insert(rule_id, enabled != 0);
        }
    }

    let mut custom_rules: Vec<IgnoreRule> = Vec::new();
    let mut custom_rules_error = None;
    {
        let mut stmt = conn.prepare(
            "SELECT id, name, enabled, conditions_json FROM _custom_ignore_rules ORDER BY id",
        )?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let id: String = row.get(0)?;
            let name: String = row.get(1)?;
            let enabled: i64 = row.get(2)?;
            let conditions_json: String = row.get(3)?;
            match serde_json::from_str::<Vec<RuleCondition>>(&conditions_json) {
                Ok(conditions) => custom_rules.push(IgnoreRule {
                    id,
                    name,
                    enabled: enabled != 0,
                    conditions,
                }),
                Err(err) => {
                    // Writers are our own code, so this should never happen — but one
                    // unreadable row must not take every other rule down with it.
                    let message = format!("custom ignore rule '{id}' has unreadable conditions: {err}");
                    eprintln!("{message}");
                    custom_rules_error = Some(message);
                }
            }
        }
    }

    let builtin_ids: HashSet<&str> = builtin.rules.iter().map(|rule| rule.id.as_str()).collect();
    if let Err(err) = validate_ignore_rules("custom ignore rules", &custom_rules, &builtin_ids) {
        // Shouldn't happen either, given every writer validates before storing — but if the
        // stored state is ever bad, fail safe (built-ins only) rather than apply a broken rule.
        let message = format!("stored custom ignore rules are invalid: {err}");
        eprintln!("{message}");
        custom_rules_error = Some(match custom_rules_error {
            Some(existing) => format!("{existing}; {message}"),
            None => message,
        });
        custom_rules.clear();
    }

    let mut rules: Vec<EffectiveIgnoreRule> = builtin
        .rules
        .into_iter()
        .map(|rule| {
            let enabled = overrides.get(&rule.id).copied().unwrap_or(rule.enabled);
            EffectiveIgnoreRule {
                id: rule.id,
                name: rule.name,
                enabled,
                source: RuleSource::Builtin,
                conditions: rule.conditions,
            }
        })
        .collect();
    rules.extend(custom_rules.into_iter().map(|rule| EffectiveIgnoreRule {
        id: rule.id,
        name: rule.name,
        enabled: rule.enabled,
        source: RuleSource::Custom,
        conditions: rule.conditions,
    }));

    let rules_hash = hash_ignore_rule_state(&builtin_raw, &rules);
    Ok(MergedIgnoreRules {
        rules,
        rules_hash,
        custom_rules_error,
    })
}

#[cfg(not(test))]
fn builtin_ignore_rules_json() -> Result<Cow<'static, str>> {
    let path = BUILTIN_IGNORE_RULES_PATH.get().cloned().unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("resources")
            .join("intel")
            .join("ignore_rules.v1.json")
    });
    read_verified_builtin_ignore_rules(&path).map(Cow::Owned)
}

#[cfg(test)]
fn builtin_ignore_rules_json() -> Result<Cow<'static, str>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("resources")
        .join("intel")
        .join("ignore_rules.v1.json");
    read_verified_builtin_ignore_rules(&path).map(Cow::Owned)
}

fn read_verified_builtin_ignore_rules(path: &Path) -> Result<String> {
    let raw =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    verify_builtin_ignore_rules_checksum(path, &raw)?;
    Ok(raw)
}

fn verify_builtin_ignore_rules_checksum(path: &Path, raw: &str) -> Result<()> {
    let normalized = library::normalize_line_endings(raw);
    let actual = format!("{:x}", Sha256::digest(normalized.as_bytes()));
    if actual != BUILTIN_IGNORE_RULES_SHA256 {
        bail!(
            "bundled ignore-rules checksum mismatch for {}: expected {}, got {}",
            path.display(),
            BUILTIN_IGNORE_RULES_SHA256,
            actual
        );
    }
    Ok(())
}

fn parse_ignore_rule_file(label: &str, raw: &str) -> Result<IgnoreRuleFile> {
    let file: IgnoreRuleFile =
        serde_json::from_str(raw).with_context(|| format!("parsing {label}"))?;
    if file.schema_version != 1 {
        bail!(
            "{label} has unsupported schemaVersion {}",
            file.schema_version
        );
    }
    if file.rule_set_id.trim().is_empty() {
        bail!("{label} has an empty ruleSetId");
    }
    validate_ignore_rules(label, &file.rules, &HashSet::new())?;
    Ok(file)
}

/// `reserved_ids` are ids this rule set must not reuse (the built-in set, when validating
/// stored custom rules; empty when validating the built-in file against itself).
fn validate_ignore_rules(
    label: &str,
    rules: &[IgnoreRule],
    reserved_ids: &HashSet<&str>,
) -> Result<()> {
    let mut seen_rule_ids = HashSet::new();
    for rule in rules {
        let rule_label = if rule.id.trim().is_empty() {
            "<empty rule id>"
        } else {
            rule.id.as_str()
        };
        if rule.id.trim().is_empty() {
            bail!("{label} has an ignore rule with an empty id");
        }
        if !seen_rule_ids.insert(rule.id.as_str()) {
            bail!("{label} has a duplicate ignore rule id: {rule_label}");
        }
        if reserved_ids.contains(rule.id.as_str()) {
            bail!("{label} rule id '{rule_label}' collides with a built-in rule id");
        }
        if rule.name.trim().is_empty() {
            bail!("{label} ignore rule {rule_label} has an empty name");
        }
        if rule.conditions.is_empty() || rule.conditions.len() > MAX_RULE_CONDITIONS {
            bail!("{label} ignore rule {rule_label} needs 1..={MAX_RULE_CONDITIONS} conditions");
        }
        for condition in &rule.conditions {
            match (&condition.role, condition.header_any_of.is_empty()) {
                (Some(role), true) => {
                    if !RULE_CONDITION_ROLES.contains(&role.as_str()) {
                        bail!(
                            "{label} ignore rule {rule_label} uses unsupported role '{role}'"
                        );
                    }
                }
                (None, false) => {
                    if condition
                        .header_any_of
                        .iter()
                        .any(|candidate| normalize_header_token(candidate).is_empty())
                    {
                        bail!("{label} ignore rule {rule_label} has an empty header candidate");
                    }
                }
                _ => bail!(
                    "{label} ignore rule {rule_label} conditions need exactly one of role or headerAnyOf"
                ),
            }
            if condition.values.is_empty()
                || condition.values.len() > MAX_RULE_VALUES
                || condition.values.iter().any(|value| value.trim().is_empty())
            {
                bail!(
                    "{label} ignore rule {rule_label} needs 1..={MAX_RULE_VALUES} non-empty condition values"
                );
            }
        }
    }
    Ok(())
}

/// Content hash of the merged, *effective* rule set — the built-in resource plus this file's
/// own overrides/custom rules already applied. Hashing the effective result rather than raw
/// source bytes means the cache key is exactly "what would change `_ignored_rows`", nothing
/// more, nothing less. Sorted by id first so the hash doesn't depend on `HashMap`/SQL row
/// iteration order.
fn hash_ignore_rule_state(builtin_raw: &str, rules: &[EffectiveIgnoreRule]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"log-parser-ignore-rule-state-v2\0");
    let normalized_builtin = library::normalize_line_endings(builtin_raw);
    let builtin_bytes = normalized_builtin.as_bytes();
    hasher.update((builtin_bytes.len() as u64).to_le_bytes());
    hasher.update(builtin_bytes);

    let mut sorted: Vec<&EffectiveIgnoreRule> = rules.iter().collect();
    sorted.sort_by(|a, b| a.id.cmp(&b.id));
    for rule in sorted {
        hasher.update((rule.id.len() as u64).to_le_bytes());
        hasher.update(rule.id.as_bytes());
        hasher.update([rule.enabled as u8]);
        let conditions_json = serde_json::to_string(&rule.conditions).unwrap_or_default();
        hasher.update((conditions_json.len() as u64).to_le_bytes());
        hasher.update(conditions_json.as_bytes());
    }
    format!("{:x}", hasher.finalize())
}

/// Content hash of everything that determines which rows `_ignored_rows` should contain: the
/// merged rule set plus every confirmed role assignment a role-scoped condition could resolve
/// through. Two calls with the same hash would recompute an identical `_ignored_rows` — used to
/// skip that recomputation when nothing has actually changed.
pub fn rules_state_hash(conn: &Connection, merged: &MergedIgnoreRules) -> Result<String> {
    let mut hasher = Sha256::new();
    hasher.update(b"log-parser-ignore-rules-state-v1\0");
    hasher.update(merged.rules_hash.as_bytes());

    let roles_exist: i64 = conn.query_row(
        "SELECT EXISTS(
            SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = '_column_roles'
         )",
        [],
        |row| row.get(0),
    )?;
    if roles_exist != 0 {
        let placeholders = RULE_CONDITION_ROLES
            .iter()
            .map(|_| "?")
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            "SELECT role, sql_name FROM _column_roles
             WHERE status = 'confirmed' AND role IN ({placeholders})
             ORDER BY role, sql_name"
        );
        let mut stmt = conn.prepare(&sql)?;
        let mut rows = stmt.query(rusqlite::params_from_iter(RULE_CONDITION_ROLES.iter()))?;
        while let Some(row) = rows.next()? {
            let role: String = row.get(0)?;
            let sql_name: String = row.get(1)?;
            hasher.update((role.len() as u64).to_le_bytes());
            hasher.update(role.as_bytes());
            hasher.update((sql_name.len() as u64).to_le_bytes());
            hasher.update(sql_name.as_bytes());
        }
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Ensures `_ignored_rows` reflects the current merged rule set and confirmed role mappings.
/// Hash-guarded: a repeat call with nothing changed is a cheap metadata read, not a rescan —
/// this is called defensively at the top of every stage that must respect ignore rules (MITRE
/// matcher, activity classifier, anomaly scanner, semantic index build), so it must stay cheap
/// on the no-op path regardless of dataset size.
pub fn ensure_ignored_rows_computed(conn: &mut Connection) -> Result<()> {
    db::create_ignore_rows_schema(conn)?;
    let merged = load_merged_ignore_rules(conn)?;
    let state_hash = rules_state_hash(conn, &merged)?;

    let current: Option<String> = conn
        .query_row(
            "SELECT rules_hash FROM _ignore_rules_meta WHERE singleton = 1",
            [],
            |row| row.get(0),
        )
        .optional()?;
    if current.as_deref() == Some(state_hash.as_str()) {
        return Ok(());
    }

    recompute_ignored_rows(conn, &merged, &state_hash)
}

fn recompute_ignored_rows(
    conn: &mut Connection,
    merged: &MergedIgnoreRules,
    state_hash: &str,
) -> Result<()> {
    let enabled_rules: Vec<&EffectiveIgnoreRule> = merged.enabled_rules().collect();
    let condition_lists: Vec<&[RuleCondition]> = enabled_rules
        .iter()
        .map(|rule| rule.conditions.as_slice())
        .collect();
    // No fixed "always scan these" set like the MITRE matcher's evidence columns — an ignore
    // rule only ever needs whatever columns its own conditions resolve to, so every resolved
    // column comes back as "extra" here.
    let (select_columns, resolved_rules) =
        rule_conditions::resolve_rule_conditions(conn, &condition_lists, &[], &["confirmed"])?;

    conn.execute_batch(
        "DROP TABLE IF EXISTS temp._ignored_rows_staging;
         CREATE TEMP TABLE _ignored_rows_staging (
            row_num INTEGER PRIMARY KEY,
            rule_id TEXT NOT NULL,
            rule_name TEXT NOT NULL,
            matched_column TEXT NOT NULL,
            matched_value TEXT NOT NULL
         );",
    )?;

    let mut rows_ignored = 0i64;
    if !resolved_rules.is_empty() {
        let select_idents: Vec<String> = select_columns
            .iter()
            .map(|column| db::quote_ident(column))
            .collect();
        let select_sql = format!(
            "SELECT row_num, {} FROM rows WHERE row_num > ?1 ORDER BY row_num ASC LIMIT ?2",
            select_idents.join(", ")
        );
        let mut last_row_num = i64::MIN;
        loop {
            let batch = {
                let mut stmt = conn.prepare(&select_sql)?;
                let mut rows = stmt.query(rusqlite::params![last_row_num, IGNORE_SCAN_BATCH_ROWS])?;
                let mut batch = Vec::new();
                while let Some(row) = rows.next()? {
                    let row_num: i64 = row.get(0)?;
                    let mut values = Vec::with_capacity(select_columns.len());
                    for column_idx in 0..select_columns.len() {
                        values.push(row.get::<_, Option<String>>(column_idx + 1)?);
                    }
                    batch.push((row_num, values));
                }
                batch
            };
            if batch.is_empty() {
                break;
            }

            let mut pending: Vec<(i64, &str, &str, String, String)> = Vec::new();
            for (row_num, values) in &batch {
                last_row_num = *row_num;
                'rules: for resolved in &resolved_rules {
                    let mut first_hit: Option<(usize, &str)> = None;
                    for condition in &resolved.conditions {
                        match rule_conditions::condition_match(condition, values) {
                            Some(hit) => {
                                if first_hit.is_none() {
                                    first_hit = Some(hit);
                                }
                            }
                            None => continue 'rules,
                        }
                    }
                    if let Some((column_idx, value)) = first_hit {
                        let rule = enabled_rules[resolved.rule_idx];
                        pending.push((
                            *row_num,
                            rule.id.as_str(),
                            rule.name.as_str(),
                            select_columns[column_idx].clone(),
                            bounded_match_value(value),
                        ));
                        break 'rules;
                    }
                }
            }

            if !pending.is_empty() {
                let mut stmt = conn.prepare_cached(
                    "INSERT INTO temp._ignored_rows_staging
                        (row_num, rule_id, rule_name, matched_column, matched_value)
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                )?;
                for (row_num, rule_id, rule_name, column, value) in &pending {
                    stmt.execute(rusqlite::params![row_num, rule_id, rule_name, column, value])?;
                    rows_ignored += 1;
                }
            }
        }
    }

    // Atomic publication: readers see the previous complete ignore set or the new one, and
    // Immediate rather than Deferred fails fast under contention instead of a handler-free
    // SQLITE_BUSY if another writer interleaves mid-upgrade.
    let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
    tx.execute("DELETE FROM _ignored_rows", [])?;
    tx.execute(
        "INSERT INTO _ignored_rows (row_num, rule_id, rule_name, matched_column, matched_value)
         SELECT row_num, rule_id, rule_name, matched_column, matched_value
         FROM temp._ignored_rows_staging",
        [],
    )?;
    tx.execute("DELETE FROM _ignore_rules_meta", [])?;
    tx.execute(
        "INSERT INTO _ignore_rules_meta (singleton, rules_hash, rule_count, rows_ignored, computed_at)
         VALUES (1, ?1, ?2, ?3, ?4)",
        rusqlite::params![
            state_hash,
            enabled_rules.len() as i64,
            rows_ignored,
            chrono::Utc::now().to_rfc3339(),
        ],
    )?;
    tx.commit()?;
    conn.execute_batch("DROP TABLE IF EXISTS temp._ignored_rows_staging")?;
    Ok(())
}

/// Loads the current ignored-row set. Callers on the hot per-row path should call this once per
/// scan and check membership with `HashSet::contains`, not requery per row.
pub fn load_ignored_row_set(conn: &Connection) -> Result<HashSet<i64>> {
    db::create_ignore_rows_schema(conn)?;
    let mut stmt = conn.prepare("SELECT row_num FROM _ignored_rows")?;
    let rows = stmt
        .query_map([], |row| row.get::<_, i64>(0))?
        .collect::<rusqlite::Result<HashSet<i64>>>()?;
    Ok(rows)
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct IgnoredRuleBreakdown {
    pub rule_id: String,
    pub rule_name: String,
    pub row_count: i64,
}

/// (total ignored rows, per-rule breakdown) for surfacing in a stage's summary. Dataset-wide,
/// not stage-specific — every stage reports the same numbers for the same dataset state, which
/// is correct: `_ignored_rows` is one fact about the dataset, not four independent ones.
pub fn ignored_rows_summary(conn: &Connection) -> Result<(i64, Vec<IgnoredRuleBreakdown>)> {
    db::create_ignore_rows_schema(conn)?;
    let mut stmt = conn.prepare(
        "SELECT rule_id, rule_name, COUNT(*) FROM _ignored_rows
         GROUP BY rule_id, rule_name ORDER BY COUNT(*) DESC",
    )?;
    let mut by_rule = Vec::new();
    let mut rows = stmt.query([])?;
    let mut total = 0i64;
    while let Some(row) = rows.next()? {
        let row_count: i64 = row.get(2)?;
        total += row_count;
        by_rule.push(IgnoredRuleBreakdown {
            rule_id: row.get(0)?,
            rule_name: row.get(1)?,
            row_count,
        });
    }
    Ok((total, by_rule))
}

// --- Command-facing mutation API: reads/writes THIS file's own ignore-rule state — the two
// tables `db::create_ignore_rule_state_schema` creates. Per-file by design (like `_column_roles`):
// a rule toggled or added while working one case never touches any other file's database.

/// One rule as shown/edited in the UI: `id`/`name`/`enabled`/`source`/`conditions`, matching
/// `EffectiveIgnoreRule` — kept as a thin public alias rather than a duplicate type so the wire
/// shape and the internal merge result never drift apart.
pub type IgnoreRuleView = EffectiveIgnoreRule;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct IgnoreRulesListing {
    pub rules: Vec<IgnoreRuleView>,
    pub custom_rules_error: Option<String>,
}

pub fn list_ignore_rules(conn: &Connection) -> Result<IgnoreRulesListing> {
    let merged = load_merged_ignore_rules(conn)?;
    Ok(IgnoreRulesListing {
        rules: merged.rules,
        custom_rules_error: merged.custom_rules_error,
    })
}

/// What the "add a rule" form collects. Mirrors `RuleCondition` plus a display name; a rule has
/// exactly one condition (the form builds one condition at a time — multi-condition rules are
/// only ever built-in today).
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NewIgnoreRuleInput {
    pub name: String,
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub header_any_of: Vec<String>,
    pub op: crate::intel::library::ConditionOp,
    pub values: Vec<String>,
}

pub fn add_custom_ignore_rule(conn: &Connection, input: NewIgnoreRuleInput) -> Result<IgnoreRulesListing> {
    let name = input.name.trim().to_string();
    if name.is_empty() {
        bail!("rule name must not be empty");
    }
    db::create_ignore_rule_state_schema(conn)?;
    let builtin = load_builtin_ignore_rules()?;

    let mut reserved: HashSet<String> = builtin.rules.iter().map(|rule| rule.id.clone()).collect();
    {
        let mut stmt = conn.prepare("SELECT id FROM _custom_ignore_rules")?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            reserved.insert(row.get::<_, String>(0)?);
        }
    }
    let id = unique_rule_id(&slugify(&name), &reserved);

    let new_rule = IgnoreRule {
        id: id.clone(),
        name: name.clone(),
        enabled: true,
        conditions: vec![RuleCondition {
            role: input.role,
            header_any_of: input.header_any_of,
            op: input.op,
            values: input.values,
        }],
    };
    let builtin_ids: HashSet<&str> = builtin.rules.iter().map(|rule| rule.id.as_str()).collect();
    validate_ignore_rules(
        "new ignore rule",
        std::slice::from_ref(&new_rule),
        &builtin_ids,
    )?;

    let conditions_json =
        serde_json::to_string(&new_rule.conditions).context("serializing rule conditions")?;
    conn.execute(
        "INSERT INTO _custom_ignore_rules (id, name, enabled, conditions_json)
         VALUES (?1, ?2, 1, ?3)",
        rusqlite::params![id, name, conditions_json],
    )?;

    list_ignore_rules(conn)
}

pub fn delete_custom_ignore_rule(conn: &Connection, rule_id: &str) -> Result<IgnoreRulesListing> {
    db::create_ignore_rule_state_schema(conn)?;
    let changed = conn.execute(
        "DELETE FROM _custom_ignore_rules WHERE id = ?1",
        rusqlite::params![rule_id],
    )?;
    if changed == 0 {
        bail!("no custom ignore rule with id '{rule_id}' (built-in rules can be disabled, not deleted)");
    }
    // A deleted custom rule's id becomes reusable; drop any override that referenced it too,
    // though overrides only ever target built-in ids so this is nearly always a no-op.
    conn.execute(
        "DELETE FROM _ignore_rule_overrides WHERE rule_id = ?1",
        rusqlite::params![rule_id],
    )?;
    list_ignore_rules(conn)
}

pub fn set_ignore_rule_enabled(conn: &Connection, rule_id: &str, enabled: bool) -> Result<IgnoreRulesListing> {
    db::create_ignore_rule_state_schema(conn)?;

    let custom_changed = conn.execute(
        "UPDATE _custom_ignore_rules SET enabled = ?2 WHERE id = ?1",
        rusqlite::params![rule_id, enabled],
    )?;
    if custom_changed == 0 {
        let builtin = load_builtin_ignore_rules()?;
        if !builtin.rules.iter().any(|rule| rule.id == rule_id) {
            bail!("no ignore rule with id '{rule_id}'");
        }
        conn.execute(
            "INSERT INTO _ignore_rule_overrides (rule_id, enabled) VALUES (?1, ?2)
             ON CONFLICT(rule_id) DO UPDATE SET enabled = excluded.enabled",
            rusqlite::params![rule_id, enabled],
        )?;
    }

    list_ignore_rules(conn)
}

fn slugify(name: &str) -> String {
    let mut slug = String::new();
    let mut last_was_dash = true;
    for ch in name.to_lowercase().chars() {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch);
            last_was_dash = false;
        } else if !last_was_dash {
            slug.push('-');
            last_was_dash = true;
        }
    }
    let trimmed = slug.trim_end_matches('-');
    if trimmed.is_empty() {
        "custom-rule".to_string()
    } else {
        trimmed.to_string()
    }
}

fn unique_rule_id(base: &str, reserved: &HashSet<String>) -> String {
    if !reserved.contains(base) {
        return base.to_string();
    }
    let mut suffix = 2u32;
    loop {
        let candidate = format!("{base}-{suffix}");
        if !reserved.contains(&candidate) {
            return candidate;
        }
        suffix += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intel::library::ConditionOp;

    #[test]
    fn builtin_ignore_rules_loads_real_resource() {
        let file = load_builtin_ignore_rules().unwrap();
        assert_eq!(file.rule_set_id, "builtin_ignore_rules_v1");
        assert_eq!(file.rules.len(), 2);
        assert!(file.rules.iter().any(|r| r.id == "qualys-agent-activity"));
        assert!(file
            .rules
            .iter()
            .any(|r| r.id == "msedge-crashpad-handler"));
        assert!(file.rules.iter().all(|r| r.enabled));
    }

    fn unique_temp_dir(label: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "log-parser-ignore-rules-{label}-{}-{nonce}",
            std::process::id()
        ))
    }

    fn custom_rules_conn() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        db::create_ignore_rule_state_schema(&conn).unwrap();
        conn
    }

    fn insert_custom_rule(conn: &Connection, id: &str, name: &str, enabled: bool, conditions: &str) {
        conn.execute(
            "INSERT INTO _custom_ignore_rules (id, name, enabled, conditions_json)
             VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![id, name, enabled, conditions],
        )
        .unwrap();
    }

    fn insert_override(conn: &Connection, rule_id: &str, enabled: bool) {
        conn.execute(
            "INSERT INTO _ignore_rule_overrides (rule_id, enabled) VALUES (?1, ?2)",
            rusqlite::params![rule_id, enabled],
        )
        .unwrap();
    }

    #[test]
    fn builtin_checksum_rejects_tampered_content() {
        let tampered = r#"{"schemaVersion":1,"ruleSetId":"x","rules":[{"id":"a","name":"a","conditions":[{"role":"process_name","op":"contains_any","values":["x"]}]}]}"#;
        let dir = unique_temp_dir("tamper");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ignore_rules.v1.json");
        std::fs::write(&path, tampered).unwrap();

        let err = read_verified_builtin_ignore_rules(&path).unwrap_err();
        assert!(err.to_string().contains("checksum mismatch"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn merge_applies_custom_additions_and_overrides() {
        let conn = custom_rules_conn();
        insert_custom_rule(
            &conn,
            "custom-noisy-event-id",
            "Custom noisy Event ID",
            true,
            r#"[{"headerAnyOf": ["EventID"], "op": "equals_any", "values": ["9999"]}]"#,
        );
        insert_override(&conn, "msedge-crashpad-handler", false);

        let merged = load_merged_ignore_rules(&conn).unwrap();
        assert!(merged.custom_rules_error.is_none());

        let msedge = merged
            .rules
            .iter()
            .find(|r| r.id == "msedge-crashpad-handler")
            .unwrap();
        assert!(!msedge.enabled, "override should disable the built-in rule");

        let custom = merged
            .rules
            .iter()
            .find(|r| r.id == "custom-noisy-event-id")
            .unwrap();
        assert!(custom.enabled);
        assert_eq!(custom.source, RuleSource::Custom);

        let qualys = merged
            .rules
            .iter()
            .find(|r| r.id == "qualys-agent-activity")
            .unwrap();
        assert!(qualys.enabled, "rules without an override keep their default");
    }

    #[test]
    fn unreadable_custom_rule_row_is_skipped_not_fatal() {
        let conn = custom_rules_conn();
        insert_custom_rule(&conn, "broken", "Broken Rule", true, "not valid json");

        let merged = load_merged_ignore_rules(&conn).unwrap();
        assert!(merged.custom_rules_error.is_some());
        // Built-ins still load even though the one bad row was rejected.
        assert_eq!(merged.rules.len(), 2);
    }

    #[test]
    fn no_custom_rules_stored_is_not_an_error() {
        let conn = custom_rules_conn();

        let merged = load_merged_ignore_rules(&conn).unwrap();
        assert!(merged.custom_rules_error.is_none());
        assert_eq!(merged.rules.len(), 2);
    }

    #[test]
    fn custom_rule_id_colliding_with_builtin_is_rejected() {
        let conn = custom_rules_conn();
        // Bypasses `add_custom_ignore_rule`'s own collision check, simulating state that
        // somehow ended up bad — `load_merged_ignore_rules` must still catch it on read.
        insert_custom_rule(
            &conn,
            "qualys-agent-activity",
            "Collides with builtin",
            true,
            r#"[{"role": "host", "op": "equals_any", "values": ["x"]}]"#,
        );

        let merged = load_merged_ignore_rules(&conn).unwrap();
        assert!(merged
            .custom_rules_error
            .as_ref()
            .is_some_and(|err| err.contains("collides")));
    }

    #[test]
    fn rules_hash_is_stable_and_changes_with_content() {
        let rule = |id: &str, enabled: bool| EffectiveIgnoreRule {
            id: id.to_string(),
            name: "Test".to_string(),
            enabled,
            source: RuleSource::Custom,
            conditions: vec![RuleCondition {
                role: Some("process_name".to_string()),
                header_any_of: vec![],
                op: ConditionOp::ContainsAny,
                values: vec!["x".to_string()],
            }],
        };
        let a = hash_ignore_rule_state("{}", &[rule("r1", true)]);
        let b = hash_ignore_rule_state("{}", &[rule("r1", true)]);
        let c = hash_ignore_rule_state("{}", &[rule("r1", false)]);
        let d = hash_ignore_rule_state("{\"different\":true}", &[rule("r1", true)]);
        assert_eq!(a, b, "same inputs must hash identically");
        assert_ne!(a, c, "enabled state must affect the hash");
        assert_ne!(a, d, "builtin resource content must affect the hash");
    }

    #[test]
    fn ignore_rule_state_is_isolated_per_connection() {
        let conn_a = custom_rules_conn();
        let conn_b = custom_rules_conn();
        insert_custom_rule(
            &conn_a,
            "file-a-only-rule",
            "Only visible in file A",
            true,
            r#"[{"headerAnyOf": ["EventID"], "op": "equals_any", "values": ["1"]}]"#,
        );
        insert_override(&conn_a, "qualys-agent-activity", false);

        let listing_a = list_ignore_rules(&conn_a).unwrap();
        let listing_b = list_ignore_rules(&conn_b).unwrap();

        assert!(listing_a.rules.iter().any(|r| r.id == "file-a-only-rule"));
        assert!(
            !listing_b.rules.iter().any(|r| r.id == "file-a-only-rule"),
            "a custom rule added to one file's database must not appear in another's"
        );

        let qualys_a = listing_a
            .rules
            .iter()
            .find(|r| r.id == "qualys-agent-activity")
            .unwrap();
        let qualys_b = listing_b
            .rules
            .iter()
            .find(|r| r.id == "qualys-agent-activity")
            .unwrap();
        assert!(!qualys_a.enabled, "file A disabled Qualys for itself");
        assert!(
            qualys_b.enabled,
            "disabling a built-in rule in file A must not affect file B's default-enabled state"
        );
    }

    #[test]
    fn validate_rejects_condition_with_neither_role_nor_header() {
        let rule = IgnoreRule {
            id: "bad".into(),
            name: "bad".into(),
            enabled: true,
            conditions: vec![RuleCondition {
                role: None,
                header_any_of: vec![],
                op: ConditionOp::EqualsAny,
                values: vec!["x".into()],
            }],
        };
        let err = validate_ignore_rules("test", &[rule], &HashSet::new()).unwrap_err();
        assert!(err.to_string().contains("exactly one of role or headerAnyOf"));
    }

    fn process_name_test_db(rows: &[(i64, &str)]) -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        let columns = vec![db::ColumnMeta {
            sql_name: "processname".into(),
            original_name: "ProcessName".into(),
            col_index: 0,
            inferred_type: "text".into(),
        }];
        db::create_schema(&conn, &columns).unwrap();
        for (row_num, process_name) in rows {
            conn.execute(
                "INSERT INTO rows (row_num, processname) VALUES (?1, ?2)",
                rusqlite::params![row_num, process_name],
            )
            .unwrap();
        }
        conn
    }

    fn set_process_name_role(conn: &Connection, status: &str) {
        db::create_column_roles_table(conn).unwrap();
        conn.execute(
            "INSERT INTO _column_roles (role, sql_name, confidence, status, reasons_json)
             VALUES ('process_name', 'processname', 1.0, ?1, '[]')",
            rusqlite::params![status],
        )
        .unwrap();
    }

    #[test]
    fn ensure_ignored_rows_matches_builtin_qualys_rule_and_reports_breakdown() {
        let mut conn = process_name_test_db(&[(1, "QualysAgent.exe"), (2, "notepad.exe")]);
        set_process_name_role(&conn, "confirmed");

        ensure_ignored_rows_computed(&mut conn).unwrap();
        let ignored = load_ignored_row_set(&conn).unwrap();
        assert_eq!(ignored, HashSet::from([1]));

        let (total, by_rule) = ignored_rows_summary(&conn).unwrap();
        assert_eq!(total, 1);
        assert_eq!(by_rule.len(), 1);
        assert_eq!(by_rule[0].rule_id, "qualys-agent-activity");
        assert_eq!(by_rule[0].row_count, 1);
    }

    #[test]
    fn ignore_rules_require_confirmed_role_not_merely_suggested() {
        let mut conn = process_name_test_db(&[(1, "QualysAgent.exe")]);
        set_process_name_role(&conn, "suggested");

        ensure_ignored_rows_computed(&mut conn).unwrap();
        assert!(
            load_ignored_row_set(&conn).unwrap().is_empty(),
            "a merely-suggested role must not drive ignore-rule suppression, unlike MITRE matching"
        );
    }

    #[test]
    fn recompute_triggers_when_role_confirmation_changes_but_not_otherwise() {
        let mut conn = process_name_test_db(&[(1, "QualysAgent.exe")]);
        set_process_name_role(&conn, "suggested");

        ensure_ignored_rows_computed(&mut conn).unwrap();
        assert!(load_ignored_row_set(&conn).unwrap().is_empty());
        let computed_at_1: String = conn
            .query_row(
                "SELECT computed_at FROM _ignore_rules_meta WHERE singleton = 1",
                [],
                |row| row.get(0),
            )
            .unwrap();

        // Unchanged inputs: a second call must be a no-op, not a rescan.
        ensure_ignored_rows_computed(&mut conn).unwrap();
        let computed_at_2: String = conn
            .query_row(
                "SELECT computed_at FROM _ignore_rules_meta WHERE singleton = 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(
            computed_at_1, computed_at_2,
            "unchanged rules and roles must not trigger a recompute"
        );

        // Confirming the role changes the state hash and must trigger a real recompute.
        conn.execute(
            "UPDATE _column_roles SET status = 'confirmed' WHERE role = 'process_name'",
            [],
        )
        .unwrap();
        ensure_ignored_rows_computed(&mut conn).unwrap();
        assert_eq!(load_ignored_row_set(&conn).unwrap(), HashSet::from([1]));
    }

    fn header_input(name: &str, header: &str, op: ConditionOp, value: &str) -> NewIgnoreRuleInput {
        NewIgnoreRuleInput {
            name: name.to_string(),
            role: None,
            header_any_of: vec![header.to_string()],
            op,
            values: vec![value.to_string()],
        }
    }

    #[test]
    fn add_custom_rule_stores_it_for_this_file_and_slugifies_the_id() {
        let conn = custom_rules_conn();
        let listing = add_custom_ignore_rule(
            &conn,
            header_input("Noisy Event ID!", "EventID", ConditionOp::EqualsAny, "9999"),
        )
        .unwrap();

        let added = listing
            .rules
            .iter()
            .find(|rule| rule.name == "Noisy Event ID!")
            .unwrap();
        assert_eq!(added.id, "noisy-event-id");
        assert_eq!(added.source, RuleSource::Custom);
        assert!(added.enabled);
        // Built-ins are still present alongside the new custom rule.
        assert_eq!(listing.rules.len(), 3);
    }

    #[test]
    fn add_custom_rule_disambiguates_id_collision_with_builtin() {
        let conn = custom_rules_conn();
        // Slugifies to "qualys-agent-activity", identical to the built-in rule's id.
        let listing = add_custom_ignore_rule(
            &conn,
            header_input(
                "Qualys Agent Activity",
                "EventID",
                ConditionOp::EqualsAny,
                "1",
            ),
        )
        .unwrap();

        let custom_rule = listing
            .rules
            .iter()
            .find(|rule| rule.source == RuleSource::Custom)
            .unwrap();
        assert_eq!(custom_rule.id, "qualys-agent-activity-2");
    }

    #[test]
    fn add_custom_rule_rejects_empty_name() {
        let conn = custom_rules_conn();
        let err = add_custom_ignore_rule(
            &conn,
            header_input("   ", "EventID", ConditionOp::EqualsAny, "1"),
        )
        .unwrap_err();
        assert!(err.to_string().contains("empty"));
    }

    #[test]
    fn delete_custom_rule_removes_it_but_not_builtins() {
        let conn = custom_rules_conn();
        add_custom_ignore_rule(
            &conn,
            header_input("Temp Rule", "EventID", ConditionOp::EqualsAny, "1"),
        )
        .unwrap();

        let listing = delete_custom_ignore_rule(&conn, "temp-rule").unwrap();
        assert_eq!(listing.rules.len(), 2);
        assert!(listing.rules.iter().all(|rule| rule.id != "temp-rule"));

        let err = delete_custom_ignore_rule(&conn, "qualys-agent-activity").unwrap_err();
        assert!(err.to_string().contains("no custom ignore rule"));
    }

    #[test]
    fn set_enabled_overrides_builtin_and_toggles_custom_directly() {
        let conn = custom_rules_conn();
        add_custom_ignore_rule(
            &conn,
            header_input("Temp Rule", "EventID", ConditionOp::EqualsAny, "1"),
        )
        .unwrap();

        let listing = set_ignore_rule_enabled(&conn, "qualys-agent-activity", false).unwrap();
        let qualys = listing
            .rules
            .iter()
            .find(|rule| rule.id == "qualys-agent-activity")
            .unwrap();
        assert!(!qualys.enabled);
        assert_eq!(qualys.source, RuleSource::Builtin);

        let listing = set_ignore_rule_enabled(&conn, "temp-rule", false).unwrap();
        let temp = listing.rules.iter().find(|rule| rule.id == "temp-rule").unwrap();
        assert!(!temp.enabled);

        let err = set_ignore_rule_enabled(&conn, "does-not-exist", true).unwrap_err();
        assert!(err.to_string().contains("no ignore rule"));
    }

    #[test]
    fn list_ignore_rules_surfaces_a_corrupted_custom_row_without_failing() {
        let conn = custom_rules_conn();
        insert_custom_rule(&conn, "broken", "Broken Rule", true, "not valid json");
        let listing = list_ignore_rules(&conn).unwrap();
        assert!(listing.custom_rules_error.is_some());
        assert_eq!(listing.rules.len(), 2, "built-ins still load");
    }
}
