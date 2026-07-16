use crate::prompt::MockContext;
use crate::schema::GuidedIntent;
use serde::{Deserialize, Serialize};

// Externally tagged (serde's default representation), not internally
// tagged: serde cannot deserialize an internally tagged enum (tag="kind")
// that has a field whose type is itself another internally tagged enum
// (GuidedIntent, tag="intent") -- confirmed empirically, this is a known
// serde content-buffering limitation. GuidedIntent must stay byte-identical
// to the real production type in src-tauri, so this harness-only type
// changes representation instead.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpectedOutcome {
    /// Must parse to exactly this intent.
    Exact { intent: GuidedIntent },
    /// Any of these intents is an acceptable parse (multiple valid phrasings
    /// of the same underlying request).
    AnyOf { intents: Vec<GuidedIntent> },
    /// Must NOT confidently produce a structured guess -- either `Unknown`,
    /// or a validation failure (invalid JSON / hallucinated reference) also
    /// counts as "did not confidently guess wrong," scored separately from
    /// invalid-output rate. See metrics.rs for how these are split out.
    MustBeUnknownOrClarify,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalCase {
    pub id: String,
    pub query_text: String,
    pub mock_context: MockContext,
    pub expected: ExpectedOutcome,
    #[serde(default)]
    pub notes: String,
}

pub fn load_eval_set(path: &std::path::Path) -> anyhow::Result<Vec<EvalCase>> {
    let raw = std::fs::read_to_string(path)?;
    let cases: Vec<EvalCase> = serde_json::from_str(&raw)?;
    Ok(cases)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression guard: serde 1.0.228 / serde_json 1.0.150 (the exact
    /// versions pinned here and in src-tauri/Cargo.lock) silently no-op
    /// rename_all on struct-variant fields of an internally tagged enum.
    /// schema::GuidedIntent works around it with explicit per-field
    /// #[serde(rename = ...)]; this test guards against that workaround
    /// silently regressing (e.g. if a field's #[serde(rename)] gets
    /// dropped during a future manual sync with src-tauri's copy).
    #[test]
    fn guided_intent_round_trips_true_camel_case() {
        let original = GuidedIntent::UserTechniqueTimeline {
            user_value: "alice".into(),
            user_column: "Account".into(),
            technique_ids: vec!["T1003.001".into()],
            sort: crate::schema::GuidedSort::ChronologicalAsc,
        };
        let serialized = serde_json::to_string(&original).unwrap();
        assert_eq!(
            serialized,
            r#"{"intent":"userTechniqueTimeline","userValue":"alice","userColumn":"Account","techniqueIds":["T1003.001"],"sort":"chronological_asc"}"#
        );
        let round_tripped: GuidedIntent = serde_json::from_str(&serialized).unwrap();
        assert_eq!(original, round_tripped);
    }

    #[test]
    fn full_eval_set_parses() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval_set.json");
        let cases = load_eval_set(&path).expect("eval_set.json should parse");
        assert_eq!(cases.len(), 18);
    }
}
