use serde::{Deserialize, Serialize};

// Hand-copied from src-tauri/src/intel/parser.rs (GuidedIntent, GuidedSort).
// Must stay in sync manually -- this is a deliberate, accepted drift risk
// for a throwaway Phase 1 benchmark harness, not an oversight. Do not
// depend on src-tauri as a path dependency to avoid this duplication: that
// would pull rusqlite/tauri/calamine into this crate's build and defeat
// the point of keeping llm-bench isolated from the shipped app.
//
// NOTE ON #[serde(rename = "...")] PER FIELD INSTEAD OF rename_all: the
// pinned serde/serde_json in this workspace (confirmed identical versions
// in src-tauri/Cargo.lock: serde 1.0.228, serde_json 1.0.150) silently do
// not apply rename_all to struct-variant fields on an internally tagged
// enum (tag = "..."), in *either* direction -- verified empirically with a
// minimal repro (see eval.rs test module). Variant-name renaming (via
// rename_all on the enum itself) is unaffected and works correctly. This
// is currently harmless in src-tauri because GuidedIntent's JSON only ever
// round-trips through its own Rust encode/decode (intel/parser.rs
// encode_intent/intent_from_token) and is never inspected by the frontend
// or any external consumer as camelCase -- but it means production's
// `rename_all = "camelCase"` on GuidedIntent is a silent no-op today, not
// actually producing camelCase wire JSON. Worth fixing there too at some
// point, but out of scope for this benchmark harness. Explicit per-field
// rename (used below) is unaffected by the bug and produces genuinely
// correct camelCase JSON, which this harness needs since the LLM prompt
// instructs (and the eval set's ground truth uses) real camelCase keys.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "intent", rename_all = "camelCase")]
pub enum GuidedIntent {
    SuspiciousScan {
        #[serde(rename = "tacticIds")]
        tactic_ids: Vec<String>,
        #[serde(rename = "techniqueIds")]
        technique_ids: Vec<String>,
        sort: GuidedSort,
    },
    UserTechniqueTimeline {
        #[serde(rename = "userValue")]
        user_value: String,
        #[serde(rename = "userColumn")]
        user_column: String,
        #[serde(rename = "techniqueIds")]
        technique_ids: Vec<String>,
        sort: GuidedSort,
    },
    TechniqueTimeline {
        #[serde(rename = "techniqueIds")]
        technique_ids: Vec<String>,
        sort: GuidedSort,
    },
    Unknown {
        message: String,
        suggestions: Vec<String>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GuidedSort {
    ChronologicalAsc,
    RowNumAsc,
}

impl GuidedIntent {
    /// Every technique/tactic ID this intent references, for hallucination
    /// checking against the eval case's injected MockContext.
    pub fn referenced_technique_ids(&self) -> Vec<&str> {
        match self {
            GuidedIntent::SuspiciousScan { technique_ids, .. } => {
                technique_ids.iter().map(String::as_str).collect()
            }
            GuidedIntent::UserTechniqueTimeline { technique_ids, .. } => {
                technique_ids.iter().map(String::as_str).collect()
            }
            GuidedIntent::TechniqueTimeline { technique_ids, .. } => {
                technique_ids.iter().map(String::as_str).collect()
            }
            GuidedIntent::Unknown { .. } => Vec::new(),
        }
    }

    pub fn referenced_tactic_ids(&self) -> Vec<&str> {
        match self {
            GuidedIntent::SuspiciousScan { tactic_ids, .. } => {
                tactic_ids.iter().map(String::as_str).collect()
            }
            _ => Vec::new(),
        }
    }

    pub fn referenced_user_column(&self) -> Option<&str> {
        match self {
            GuidedIntent::UserTechniqueTimeline { user_column, .. } => Some(user_column.as_str()),
            _ => None,
        }
    }

    pub fn is_unknown(&self) -> bool {
        matches!(self, GuidedIntent::Unknown { .. })
    }
}
