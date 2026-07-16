use crate::prompt::MockContext;
use crate::schema::{GuidedIntent, GuidedSort};
use serde::Deserialize;
use std::collections::BTreeSet;

#[derive(Debug, Deserialize)]
#[serde(tag = "intent", rename_all = "camelCase", deny_unknown_fields)]
enum ModelIntent {
    SuspiciousScan {
        #[serde(rename = "tacticIds")]
        tactic_ids: Vec<String>,
        #[serde(rename = "techniqueIds")]
        technique_ids: Vec<String>,
        #[serde(default, rename = "sort")]
        _sort: Option<GuidedSort>,
    },
    UserTechniqueTimeline {
        #[serde(rename = "userValue")]
        user_value: String,
        #[serde(default, rename = "userColumn")]
        user_column: Option<String>,
        #[serde(rename = "techniqueIds")]
        technique_ids: Vec<String>,
        #[serde(default, rename = "sort")]
        _sort: Option<GuidedSort>,
    },
    TechniqueTimeline {
        #[serde(rename = "techniqueIds")]
        technique_ids: Vec<String>,
        #[serde(default, rename = "sort")]
        _sort: Option<GuidedSort>,
    },
    Unknown {
        message: String,
        suggestions: Vec<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseOutcome {
    Parsed(GuidedIntent),
    InvalidJson {
        raw: String,
        error: String,
    },
    HallucinatedReference {
        intent: GuidedIntent,
        detail: String,
    },
}

/// Parses raw model output into a GuidedIntent and validates it against the
/// case's injected MockContext. Fail-closed, no retry-as-freeform: an
/// invalid-JSON or hallucinated-reference result is a scored failure, not
/// something to patch up and re-ask.
pub fn parse_and_validate(raw: &str, context: &MockContext) -> ParseOutcome {
    let wire: ModelIntent = match serde_json::from_str(raw.trim()) {
        Ok(intent) => intent,
        Err(e) => {
            return ParseOutcome::InvalidJson {
                raw: raw.to_string(),
                error: e.to_string(),
            }
        }
    };

    let known_technique_ids: std::collections::HashSet<&str> = context
        .techniques
        .iter()
        .map(|t| t.technique_id.as_str())
        .collect();
    let known_tactic_ids: std::collections::HashSet<&str> = context
        .techniques
        .iter()
        .map(|t| t.tactic_id.as_str())
        .collect();
    let known_user_columns: std::collections::HashSet<&str> = context
        .confirmed_roles
        .iter()
        .filter(|r| r.role == "user")
        .map(|r| r.sql_name.as_str())
        .collect();

    let sort = if context.has_normalized_time {
        GuidedSort::ChronologicalAsc
    } else {
        GuidedSort::RowNumAsc
    };
    let intent = match wire {
        ModelIntent::SuspiciousScan {
            tactic_ids,
            technique_ids,
            ..
        } => GuidedIntent::SuspiciousScan {
            tactic_ids: dedup(tactic_ids),
            technique_ids: dedup(technique_ids),
            sort,
        },
        ModelIntent::UserTechniqueTimeline {
            user_value,
            user_column,
            technique_ids,
            ..
        } => {
            let user_column = user_column.unwrap_or_else(|| {
                if known_user_columns.len() == 1 {
                    known_user_columns
                        .iter()
                        .next()
                        .copied()
                        .unwrap_or_default()
                        .to_string()
                } else {
                    String::new()
                }
            });
            GuidedIntent::UserTechniqueTimeline {
                user_value,
                user_column,
                technique_ids: dedup(technique_ids),
                sort,
            }
        }
        ModelIntent::TechniqueTimeline { technique_ids, .. } => GuidedIntent::TechniqueTimeline {
            technique_ids: dedup(technique_ids),
            sort,
        },
        ModelIntent::Unknown {
            message,
            suggestions,
        } => GuidedIntent::Unknown {
            message,
            suggestions,
        },
    };

    let mut hallucination_detail: Option<String> = None;
    for tid in intent.referenced_technique_ids() {
        if !known_technique_ids.contains(tid) {
            hallucination_detail = Some(format!("technique_id '{tid}' not in injected library"));
            break;
        }
    }
    if hallucination_detail.is_none() {
        for tid in intent.referenced_tactic_ids() {
            if !known_tactic_ids.contains(tid) {
                hallucination_detail = Some(format!("tactic_id '{tid}' not in injected library"));
                break;
            }
        }
    }
    if hallucination_detail.is_none() {
        if let Some(col) = intent.referenced_user_column() {
            if !known_user_columns.contains(col) {
                hallucination_detail = Some(format!("user_column '{col}' not in confirmed roles"));
            }
        }
    }

    match hallucination_detail {
        Some(detail) => ParseOutcome::HallucinatedReference { intent, detail },
        None => ParseOutcome::Parsed(intent),
    }
}

fn dedup(values: Vec<String>) -> Vec<String> {
    values
        .into_iter()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_wrappers_prose_and_multiple_objects() {
        let context = MockContext::default();
        let json = r#"{"intent":"unknown","message":"clarify","suggestions":[]}"#;
        for raw in [
            format!("```json\n{json}\n```"),
            format!("result: {json}"),
            format!("{json}\nextra text"),
            format!("{json}\n{json}"),
            format!("<|im_start|>assistant\n{json}"),
            format!("{json}<|im_start|>"),
            format!("{json}<|im_end|>"),
        ] {
            assert!(
                matches!(
                    parse_and_validate(&raw, &context),
                    ParseOutcome::InvalidJson { .. }
                ),
                "raw output unexpectedly validated: {raw}"
            );
        }
    }

    #[test]
    fn accepts_one_object_with_json_whitespace_only() {
        let context = MockContext::default();
        let raw = " \r\n{\"intent\":\"unknown\",\"message\":\"clarify\",\"suggestions\":[]}\t ";
        assert!(matches!(
            parse_and_validate(raw, &context),
            ParseOutcome::Parsed(GuidedIntent::Unknown { .. })
        ));
    }

    #[test]
    fn trusted_fields_are_assigned_like_production() {
        let context = MockContext {
            techniques: vec![],
            confirmed_roles: vec![crate::prompt::MockRole {
                role: "user".into(),
                sql_name: "Account".into(),
            }],
            has_normalized_time: true,
        };
        let raw = r#"{"intent":"userTechniqueTimeline","userValue":"alice","techniqueIds":[]}"#;
        assert!(matches!(
            parse_and_validate(raw, &context),
            ParseOutcome::Parsed(GuidedIntent::UserTechniqueTimeline {
                user_column,
                sort: GuidedSort::ChronologicalAsc,
                ..
            }) if user_column == "Account"
        ));
    }
}
