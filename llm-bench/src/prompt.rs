use serde::{Deserialize, Serialize};

pub const ASSISTANT_JSON_PREFIX: &str = "{";

/// A real technique entry pulled from src-tauri/resources/intel/mitre_core.v1.json,
/// trimmed to just what the parser prompt needs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MockTechnique {
    pub technique_id: String,
    pub name: String,
    pub tactic_id: String,
    pub tactic_name: String,
    pub aliases: Vec<String>,
}

/// A confirmed column role, mirroring src-tauri/src/intel/roles.rs's
/// ColumnRoleSuggestion once status == "confirmed".
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MockRole {
    pub role: String,
    pub sql_name: String,
}

/// Everything the real parser has available at query time, mocked for
/// eval purposes: the subset of the MITRE library relevant to a case, and
/// whichever column roles the (simulated) examiner has confirmed so far.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MockContext {
    pub techniques: Vec<MockTechnique>,
    pub confirmed_roles: Vec<MockRole>,
    #[serde(default)]
    pub has_normalized_time: bool,
}

const SCHEMA_INSTRUCTIONS: &str = r#"You translate a DFIR examiner's free-text search into exactly one JSON object matching this schema. Output ONLY the JSON object, no prose, no markdown fences, no explanation.

Schema (exactly one of these four shapes, tagged by "intent"):

{"intent": "suspiciousScan", "tacticIds": [<tactic id from the library below>], "techniqueIds": [<technique id from the library below>]}

{"intent": "userTechniqueTimeline", "userValue": <the user identity string from the query, exactly as written>, "techniqueIds": [<technique id(s) from the library below>]}

{"intent": "techniqueTimeline", "techniqueIds": [<technique id(s) from the library below>]}

{"intent": "unknown", "message": <short explanation of what's missing or ambiguous>, "suggestions": [<short follow-up questions or example phrasings>]}

Hard rules:
- Only ever reference technique/tactic IDs that appear in the library below. Never invent or guess an ID.
- Only ever use "userTechniqueTimeline" if a role with role="user" exists in confirmed roles below. If the query names a user but no user column is confirmed, return "unknown" and ask for one.
- If the query is ambiguous (matches multiple techniques and doesn't disambiguate), vague (no technique/tactic/user signal at all), or asks something out of scope (e.g. explaining *how* an attack happened, causality, root cause), return "unknown". Do not guess a plausible-looking but unconfirmed answer.
- Do not emit userColumn or sort. As in production, trusted Rust code assigns the sole confirmed user column and the safe sort order after validation.
"#;

pub fn build_prompt(context: &MockContext, query_text: &str) -> String {
    let library_json = serde_json::to_string_pretty(&context.techniques).unwrap_or_default();
    let roles_json = serde_json::to_string_pretty(&context.confirmed_roles).unwrap_or_default();

    let user_message = format!(
        "{SCHEMA_INSTRUCTIONS}\n\nMITRE library (techniques available for this case):\n{library_json}\n\nConfirmed column roles:\n{roles_json}\n\nhas_normalized_time: {}\n\nExaminer query: {query_text}",
        context.has_normalized_time,
    );

    format!(
        "<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n{ASSISTANT_JSON_PREFIX}"
    )
}

pub fn complete_assistant_output(generated_suffix: String) -> String {
    let mut output = String::with_capacity(ASSISTANT_JSON_PREFIX.len() + generated_suffix.len());
    output.push_str(ASSISTANT_JSON_PREFIX);
    output.push_str(&generated_suffix);
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_prefills_json_and_saved_output_is_complete() {
        let prompt = build_prompt(&MockContext::default(), "mimikatz alice");
        assert!(prompt.ends_with("<|im_start|>assistant\n{"));
        assert_eq!(
            complete_assistant_output(r#""intent":"unknown"}"#.to_string()),
            r#"{"intent":"unknown"}"#
        );
    }
}
