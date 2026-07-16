/// Registry of candidate models to benchmark. Swappable by design -- add a
/// new entry here to eval a third candidate without touching main.rs.
#[derive(Debug, Clone, Copy)]
pub struct ModelSpec {
    pub key: &'static str,
    pub display_name: &'static str,
    /// HuggingFace repo id ("owner/name") holding the GGUF file.
    pub gguf_owner: &'static str,
    pub gguf_repo: &'static str,
    pub gguf_filename: &'static str,
    /// HuggingFace repo id holding tokenizer.json (the base instruct repo,
    /// not the GGUF repo -- GGUF repos don't ship tokenizer.json).
    pub tokenizer_owner: &'static str,
    pub tokenizer_repo: &'static str,
    pub quant_label: &'static str,
}

pub const SMALL: ModelSpec = ModelSpec {
    key: "small",
    display_name: "Qwen2.5-1.5B-Instruct (Q4_K_M)",
    gguf_owner: "Qwen",
    gguf_repo: "Qwen2.5-1.5B-Instruct-GGUF",
    gguf_filename: "qwen2.5-1.5b-instruct-q4_k_m.gguf",
    tokenizer_owner: "Qwen",
    tokenizer_repo: "Qwen2.5-1.5B-Instruct",
    quant_label: "Q4_K_M",
};

// Deliberately not the official Qwen/Qwen2.5-7B-Instruct-GGUF repo: its
// Q4_K_M quant is split into two shards (-00001-of-00002 / -00002-of-00002),
// which candle's GGUF reader cannot load directly. bartowski's repackaging
// ships Q4_K_M as a single file.
pub const MID: ModelSpec = ModelSpec {
    key: "mid",
    display_name: "Qwen2.5-7B-Instruct (Q4_K_M)",
    gguf_owner: "bartowski",
    gguf_repo: "Qwen2.5-7B-Instruct-GGUF",
    gguf_filename: "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
    tokenizer_owner: "Qwen",
    tokenizer_repo: "Qwen2.5-7B-Instruct",
    quant_label: "Q4_K_M",
};

pub fn by_key(key: &str) -> Option<ModelSpec> {
    match key {
        "small" => Some(SMALL),
        "mid" => Some(MID),
        _ => None,
    }
}
