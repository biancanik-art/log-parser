use crate::db;
use crate::intel::library::{self, ConditionOp, RuleCondition, RULE_CONDITION_ROLES};
use anyhow::Result;
use rusqlite::Connection;
use std::collections::{HashMap, HashSet};

/// One rule's conditions bound to this dataset's actual columns. `column_indices` index into
/// the combined per-row value vector (scan columns first, extra rule-only columns appended).
#[derive(Debug, Clone)]
pub struct ResolvedCondition {
    pub column_indices: Vec<usize>,
    pub op: ConditionOp,
    pub values_lower: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ResolvedRuleConditions {
    /// Index into the `rules_conditions` slice passed to `resolve_rule_conditions`.
    pub rule_idx: usize,
    pub conditions: Vec<ResolvedCondition>,
}

/// Binds each rule's conditions to this dataset. Role conditions resolve against
/// `_column_roles` entries whose `status` is in `role_statuses`; header conditions match the
/// normalized original header or SQL name regardless of role status. Rules with any
/// unresolvable condition are skipped for this dataset — a condition that cannot see its
/// column must not silently pass. `scan_columns` seeds the combined column-index space;
/// condition-only columns not already in it are appended and returned as the first tuple
/// element.
pub fn resolve_rule_conditions(
    conn: &Connection,
    rules_conditions: &[&[RuleCondition]],
    scan_columns: &[String],
    role_statuses: &[&str],
) -> Result<(Vec<String>, Vec<ResolvedRuleConditions>)> {
    if rules_conditions.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    // `db::load_columns` reads `_meta`, which only exists once a file has actually been
    // imported. Skip it when every condition resolves by role — a rule set with no
    // `headerAnyOf` condition (e.g. the built-in ignore rules, which are role-only) never reads
    // `all_columns`, so requiring `_meta` to exist for it would be an unnecessary dependency.
    let needs_header_columns = rules_conditions
        .iter()
        .flat_map(|conditions| conditions.iter())
        .any(|condition| condition.role.is_none());
    let all_columns = if needs_header_columns {
        db::load_columns(conn)?
    } else {
        Vec::new()
    };
    let roles_exist: i64 = conn.query_row(
        "SELECT EXISTS(
            SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = '_column_roles'
         )",
        [],
        |row| row.get(0),
    )?;
    let mut role_columns: HashMap<String, Vec<String>> = HashMap::new();
    if roles_exist != 0 {
        let placeholders = role_statuses
            .iter()
            .map(|_| "?")
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            "SELECT role, sql_name FROM _column_roles
             WHERE status IN ({placeholders})
             ORDER BY sql_name"
        );
        let mut stmt = conn.prepare(&sql)?;
        let mut query = stmt.query(rusqlite::params_from_iter(role_statuses.iter()))?;
        while let Some(row) = query.next()? {
            let role: String = row.get(0)?;
            let sql_name: String = row.get(1)?;
            if RULE_CONDITION_ROLES.contains(&role.as_str()) {
                role_columns.entry(role).or_default().push(sql_name);
            }
        }
    }

    let mut combined: Vec<String> = scan_columns.to_vec();
    let mut combined_index: HashMap<String, usize> = combined
        .iter()
        .enumerate()
        .map(|(index, name)| (name.clone(), index))
        .collect();
    let mut resolved_rules = Vec::new();
    for (rule_idx, conditions_for_rule) in rules_conditions.iter().enumerate() {
        let mut conditions = Vec::with_capacity(conditions_for_rule.len());
        let mut resolvable = true;
        for condition in conditions_for_rule.iter() {
            let target_columns: Vec<String> = if let Some(role) = &condition.role {
                role_columns.get(role).cloned().unwrap_or_default()
            } else {
                let wanted: HashSet<String> = condition
                    .header_any_of
                    .iter()
                    .map(|candidate| library::normalize_header_token(candidate))
                    .collect();
                all_columns
                    .iter()
                    .filter(|column| {
                        wanted.contains(&library::normalize_header_token(&column.original_name))
                            || wanted.contains(&library::normalize_header_token(&column.sql_name))
                    })
                    .map(|column| column.sql_name.clone())
                    .collect()
            };
            if target_columns.is_empty() {
                resolvable = false;
                break;
            }
            let column_indices = target_columns
                .iter()
                .map(|name| {
                    *combined_index.entry(name.clone()).or_insert_with(|| {
                        combined.push(name.clone());
                        combined.len() - 1
                    })
                })
                .collect();
            conditions.push(ResolvedCondition {
                column_indices,
                op: condition.op,
                values_lower: condition
                    .values
                    .iter()
                    .map(|value| value.trim().to_lowercase())
                    .collect(),
            });
        }
        if resolvable {
            resolved_rules.push(ResolvedRuleConditions {
                rule_idx,
                conditions,
            });
        }
    }
    let extra = combined.split_off(scan_columns.len());
    Ok((extra, resolved_rules))
}

/// Returns the first (column index, cell value) satisfying the condition on this row.
pub fn condition_match<'row>(
    condition: &ResolvedCondition,
    values: &'row [Option<String>],
) -> Option<(usize, &'row str)> {
    for &column_idx in &condition.column_indices {
        let Some(value) = values.get(column_idx).and_then(|value| value.as_deref()) else {
            continue;
        };
        let cell = value.trim();
        if cell.is_empty() {
            continue;
        }
        let cell_lower = cell.to_lowercase();
        let satisfied = match condition.op {
            ConditionOp::EqualsAny => condition
                .values_lower
                .iter()
                .any(|wanted| cell_lower == *wanted),
            ConditionOp::ContainsAny => condition
                .values_lower
                .iter()
                .any(|wanted| cell_lower.contains(wanted.as_str())),
            ConditionOp::EndsWithAny => condition
                .values_lower
                .iter()
                .any(|wanted| cell_lower.ends_with(wanted.as_str())),
        };
        if satisfied {
            return Some((column_idx, cell));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_column(sql_name: &str, original_name: &str, col_index: usize) -> db::ColumnMeta {
        db::ColumnMeta {
            sql_name: sql_name.to_string(),
            original_name: original_name.to_string(),
            col_index,
            inferred_type: "text".to_string(),
        }
    }

    fn open_test_db_with_columns(columns: &[db::ColumnMeta]) -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        db::create_schema(&conn, columns).unwrap();
        conn
    }

    fn set_role(conn: &Connection, role: &str, sql_name: &str, status: &str) {
        db::create_column_roles_table(conn).unwrap();
        conn.execute(
            "INSERT INTO _column_roles (role, sql_name, confidence, status, reasons_json)
             VALUES (?1, ?2, 1.0, ?3, '[]')",
            rusqlite::params![role, sql_name, status],
        )
        .unwrap();
    }

    fn condition(role: Option<&str>, header_any_of: &[&str], op: ConditionOp, values: &[&str]) -> RuleCondition {
        RuleCondition {
            role: role.map(|r| r.to_string()),
            header_any_of: header_any_of.iter().map(|s| s.to_string()).collect(),
            op,
            values: values.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn header_only_condition_resolves_by_normalized_header() {
        let conn = open_test_db_with_columns(&[test_column("event_id", "Event ID", 0)]);
        let cond = condition(None, &["EventID"], ConditionOp::EqualsAny, &["4688"]);
        let rules: Vec<&[RuleCondition]> = vec![std::slice::from_ref(&cond)];
        let (extra, resolved) =
            resolve_rule_conditions(&conn, &rules, &[], &["confirmed"]).unwrap();
        assert_eq!(extra, vec!["event_id".to_string()]);
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].conditions.len(), 1);
    }

    #[test]
    fn role_condition_requires_matching_status() {
        let conn = open_test_db_with_columns(&[test_column("processname", "ProcessName", 0)]);
        set_role(&conn, "process_name", "processname", "suggested");
        let cond = condition(
            Some("process_name"),
            &[],
            ConditionOp::ContainsAny,
            &["qualys"],
        );
        let rules: Vec<&[RuleCondition]> = vec![std::slice::from_ref(&cond)];

        // Suggested-only role should not resolve when only "confirmed" is allowed.
        let (_, resolved_confirmed_only) =
            resolve_rule_conditions(&conn, &rules, &[], &["confirmed"]).unwrap();
        assert!(resolved_confirmed_only.is_empty());

        // Same role resolves when "suggested" is allowed.
        let (_, resolved_with_suggested) =
            resolve_rule_conditions(&conn, &rules, &[], &["suggested", "confirmed"]).unwrap();
        assert_eq!(resolved_with_suggested.len(), 1);
    }

    #[test]
    fn unresolvable_condition_skips_the_whole_rule() {
        let conn = open_test_db_with_columns(&[test_column("host", "Host", 0)]);
        let cond = condition(Some("process_name"), &[], ConditionOp::ContainsAny, &["x"]);
        let rules: Vec<&[RuleCondition]> = vec![std::slice::from_ref(&cond)];
        let (extra, resolved) =
            resolve_rule_conditions(&conn, &rules, &[], &["confirmed"]).unwrap();
        assert!(extra.is_empty());
        assert!(resolved.is_empty());
    }

    #[test]
    fn condition_match_is_case_insensitive_and_trims() {
        let resolved = ResolvedCondition {
            column_indices: vec![0],
            op: ConditionOp::ContainsAny,
            values_lower: vec!["qualys".to_string()],
        };
        let values = vec![Some("  QualysAgent.exe  ".to_string())];
        let result = condition_match(&resolved, &values);
        assert!(result.is_some());
    }

    #[test]
    fn condition_match_ops_equals_contains_ends_with() {
        let values = vec![Some("msedge_crashpad_handler.exe".to_string())];

        let equals = ResolvedCondition {
            column_indices: vec![0],
            op: ConditionOp::EqualsAny,
            values_lower: vec!["msedge_crashpad_handler.exe".to_string()],
        };
        assert!(condition_match(&equals, &values).is_some());

        let contains = ResolvedCondition {
            column_indices: vec![0],
            op: ConditionOp::ContainsAny,
            values_lower: vec!["crashpad".to_string()],
        };
        assert!(condition_match(&contains, &values).is_some());

        let ends_with = ResolvedCondition {
            column_indices: vec![0],
            op: ConditionOp::EndsWithAny,
            values_lower: vec!["handler.exe".to_string()],
        };
        assert!(condition_match(&ends_with, &values).is_some());

        let no_match = ResolvedCondition {
            column_indices: vec![0],
            op: ConditionOp::EqualsAny,
            values_lower: vec!["notepad.exe".to_string()],
        };
        assert!(condition_match(&no_match, &values).is_none());
    }
}
