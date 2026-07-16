use crate::db::{self, ColumnMeta};
use anyhow::{bail, Context, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use rusqlite::{params, Connection, OptionalExtension};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::cmp::{Ordering, Reverse};
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, HashSet};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::time::Instant;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

pub const MODEL_RESOURCE_PATH: &str = "models/all-minilm-l6-v2-model.safetensors";
pub const TOKENIZER_RESOURCE_PATH: &str = "models/all-minilm-l6-v2-tokenizer.json";
pub const CONFIG_RESOURCE_PATH: &str = "models/all-minilm-l6-v2-config.json";
pub const MODEL_NAME: &str = "sentence-transformers/all-MiniLM-L6-v2";
pub const MODEL_VERSION: &str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41";
pub const MODEL_SHA256: &str = "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db";
pub const TOKENIZER_SHA256: &str =
    "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037";
pub const CONFIG_SHA256: &str = "953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41";

pub const V2_INDEX_VERSION: &str = "semantic-document-v2";
pub const V2_NORMALIZER_VERSION: &str = "dfir-cell-normalizer-v1";
pub const V2_SOURCE_BATCH_ROWS: usize = 256;
pub const V2_EMBED_BATCH_DOCUMENTS: usize = 16;
pub const V2_DEFAULT_DOCUMENT_CANDIDATES: usize = 256;
pub const V2_MAX_DOCUMENT_CANDIDATES: usize = 1_024;
pub const V2_DEFAULT_MINIMUM_SCORE: f32 = 0.38;
pub const V2_BROAD_MINIMUM_SCORE: f32 = 0.30;
const MAX_TOKENS: usize = 256;
const MAX_QUERY_CHARS: usize = 4_096;
const MAX_TOP_K: usize = 1_000;
const EMBEDDING_DIMENSIONS: usize = 384;

pub struct SemanticModel {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    pub load_time_ms: u128,
}

impl SemanticModel {
    pub fn load(model_path: &Path, tokenizer_path: &Path, config_path: &Path) -> Result<Self> {
        let started = Instant::now();
        verify_sha256(model_path, MODEL_SHA256, "semantic model")?;
        verify_sha256(tokenizer_path, TOKENIZER_SHA256, "semantic tokenizer")?;
        verify_sha256(config_path, CONFIG_SHA256, "semantic model config")?;

        let config_text = std::fs::read_to_string(config_path)
            .with_context(|| format!("reading semantic config {}", config_path.display()))?;
        let config: Config =
            serde_json::from_str(&config_text).context("parsing all-MiniLM-L6-v2 configuration")?;
        if config.hidden_size != EMBEDDING_DIMENSIONS {
            bail!(
                "semantic model hidden size changed: expected {EMBEDDING_DIMENSIONS}, got {}",
                config.hidden_size
            );
        }

        let device = Device::Cpu;
        let builder = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_path], DTYPE, &device)
                .context("memory-mapping semantic model weights")?
        };
        let model = BertModel::load(builder, &config).context("loading semantic BERT model")?;
        let mut tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|error| anyhow::anyhow!("loading semantic tokenizer: {error}"))?;
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            ..Default::default()
        }));
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: MAX_TOKENS,
                ..Default::default()
            }))
            .map_err(|error| anyhow::anyhow!("configuring semantic tokenizer: {error}"))?;

        Ok(Self {
            model,
            tokenizer,
            device,
            load_time_ms: started.elapsed().as_millis(),
        })
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut rows = self.embed_batch(&[text.to_string()])?;
        rows.pop()
            .ok_or_else(|| anyhow::anyhow!("semantic model returned no embedding"))
    }

    pub fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|error| anyhow::anyhow!("tokenizing semantic-search text: {error}"))?;
        let token_ids = encodings
            .iter()
            .map(|encoding| {
                Tensor::new(encoding.get_ids(), &self.device)
                    .context("building semantic token tensor")
            })
            .collect::<Result<Vec<_>>>()?;
        let attention_masks = encodings
            .iter()
            .map(|encoding| {
                Tensor::new(encoding.get_attention_mask(), &self.device)
                    .context("building semantic attention tensor")
            })
            .collect::<Result<Vec<_>>>()?;
        let token_ids = Tensor::stack(&token_ids, 0).context("stacking semantic token tensors")?;
        let attention_mask =
            Tensor::stack(&attention_masks, 0).context("stacking semantic attention masks")?;
        let token_type_ids = token_ids
            .zeros_like()
            .context("building semantic token-type tensor")?;
        let embeddings = self
            .model
            .forward(&token_ids, &token_type_ids, Some(&attention_mask))
            .context("running semantic model inference")?;
        let pooling_mask = attention_mask
            .to_dtype(DTYPE)
            .context("converting semantic pooling mask")?
            .unsqueeze(2)
            .context("expanding semantic pooling mask")?;
        let sum_mask = pooling_mask
            .sum(1)
            .context("summing semantic pooling mask")?;
        let pooled = embeddings
            .broadcast_mul(&pooling_mask)
            .context("masking semantic token embeddings")?
            .sum(1)
            .context("pooling semantic token embeddings")?
            .broadcast_div(&sum_mask)
            .context("averaging semantic token embeddings")?;
        let normalized = pooled
            .broadcast_div(
                &pooled
                    .sqr()
                    .context("squaring semantic embeddings")?
                    .sum_keepdim(1)
                    .context("summing semantic embedding norms")?
                    .sqrt()
                    .context("normalizing semantic embedding norms")?,
            )
            .context("normalizing semantic embeddings")?;
        normalized
            .to_vec2::<f32>()
            .context("reading semantic embeddings")
    }
}

/// Small abstraction used by the resumable builder and deterministic tests. Production uses the
/// pinned MiniLM model; tests can prove deduplication/cancellation without loading 90 MB weights.
pub trait SemanticEmbedder {
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;

    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut rows = self.embed_batch(&[text.to_string()])?;
        rows.pop()
            .ok_or_else(|| anyhow::anyhow!("semantic embedder returned no embedding"))
    }
}

impl SemanticEmbedder for SemanticModel {
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        SemanticModel::embed_batch(self, texts)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ColumnMode {
    ExactOnly,
    Categorical,
    Text,
}

impl ColumnMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::ExactOnly => "exact_only",
            Self::Categorical => "categorical",
            Self::Text => "text",
        }
    }

    fn parse(value: &str) -> Result<Self> {
        match value {
            "exact_only" => Ok(Self::ExactOnly),
            "categorical" => Ok(Self::Categorical),
            "text" => Ok(Self::Text),
            _ => bail!("semantic column plan contains unknown mode {value}"),
        }
    }
}

#[derive(Debug, Clone)]
struct ColumnPlan {
    col_index: usize,
    sql_name: String,
    original_name: String,
    mode: ColumnMode,
}

#[derive(Debug, Clone)]
struct NormalizedDocument {
    kind: &'static str,
    column_key: String,
    text: String,
    rows: BTreeSet<i64>,
}

const V2_COLUMN_SAMPLE_ROWS: usize = 4_096;
const V2_PRIMARY_CHUNK_WORDS: usize = 72;
const V2_CHUNK_OVERLAP_WORDS: usize = 12;
const V2_MAX_CHUNKS_PER_CELL: usize = 4;
const V2_MAX_ADDITIONAL_CHUNKS_PER_ROW: usize = 32;

fn header_key(column: &ColumnMeta) -> String {
    format!("{} {}", column.sql_name, column.original_name)
        .to_ascii_lowercase()
        .replace(['_', '-'], " ")
}

fn exact_only_header(column: &ColumnMeta) -> bool {
    if matches!(
        column.inferred_type.to_ascii_lowercase().as_str(),
        "timestamp" | "ip" | "identifier" | "number" | "numeric"
    ) {
        return true;
    }
    let key = header_key(column);
    [
        "timestamp",
        "time generated",
        "date",
        "source ip",
        "src ip",
        "destination ip",
        "dst ip",
        "address",
        "hash",
        "guid",
        "uuid",
        " sid",
        "account",
        "user name",
        "username",
        "computer",
        "device name",
        "host name",
        "hostname",
        "report id",
        "session id",
        "port",
    ]
    .iter()
    .any(|needle| key.contains(needle))
        || key.ends_with(" id")
        || key.contains(" event id")
}

fn force_text_header(column: &ColumnMeta) -> bool {
    let key = header_key(column);
    [
        "message",
        "description",
        "command",
        "script",
        "process",
        "parent",
        "file name",
        "filename",
        "alert",
        "activity",
        "status",
        "severity",
        "rule",
        "threat",
        "action",
        "operation",
    ]
    .iter()
    .any(|needle| key.contains(needle))
}

fn value_is_dynamic_identifier(value: &str) -> bool {
    let value = value.trim();
    if value.is_empty() {
        return true;
    }
    if value.parse::<f64>().is_ok() || value.parse::<std::net::IpAddr>().is_ok() {
        return true;
    }
    let folded = value.to_ascii_lowercase();
    if folded.starts_with("s-1-")
        && folded
            .chars()
            .all(|ch| ch.is_ascii_digit() || ch == '-' || ch == 's')
    {
        return true;
    }
    let compact = value.chars().filter(|ch| ch.is_ascii_hexdigit()).count();
    let separators = value.chars().filter(|ch| *ch == '-').count();
    (value.len() >= 16 && compact == value.len())
        || (value.len() == 36 && compact == 32 && separators == 4)
}

fn normalized_token(token: &str) -> String {
    let trimmed =
        token.trim_matches(|ch: char| ch.is_ascii_punctuation() && ch != '.' && ch != '/');
    if trimmed.parse::<std::net::IpAddr>().is_ok() {
        return token.replacen(trimmed, "<ip>", 1).to_ascii_lowercase();
    }
    let folded = trimmed.to_ascii_lowercase();
    if folded.starts_with("s-1-") {
        return token.replacen(trimmed, "<sid>", 1).to_ascii_lowercase();
    }
    let hex_count = trimmed.chars().filter(|ch| ch.is_ascii_hexdigit()).count();
    let dash_count = trimmed.chars().filter(|ch| *ch == '-').count();
    if trimmed.len() == 36 && hex_count == 32 && dash_count == 4 {
        return token.replacen(trimmed, "<guid>", 1).to_ascii_lowercase();
    }
    if trimmed.len() >= 16 && hex_count == trimmed.len() {
        return token.replacen(trimmed, "<hash>", 1).to_ascii_lowercase();
    }

    let mut output = String::with_capacity(token.len());
    let mut in_digits = false;
    for ch in token.chars().flat_map(char::to_lowercase) {
        if ch.is_ascii_digit() {
            if !in_digits {
                output.push_str("<number>");
                in_digits = true;
            }
        } else {
            in_digits = false;
            if ch.is_control() {
                output.push(' ');
            } else {
                output.push(ch);
            }
        }
    }
    output
}

fn normalize_text(value: &str) -> String {
    value
        .split_whitespace()
        .map(normalized_token)
        .filter(|token| !token.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn is_informative_text(value: &str) -> bool {
    value
        .chars()
        .filter(|ch| ch.is_alphabetic())
        .take(3)
        .count()
        >= 3
}

fn word_chunks(value: &str) -> Vec<String> {
    let words = value.split_whitespace().collect::<Vec<_>>();
    if words.is_empty() {
        return Vec::new();
    }
    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < words.len() && chunks.len() < V2_MAX_CHUNKS_PER_CELL {
        let end = (start + V2_PRIMARY_CHUNK_WORDS).min(words.len());
        chunks.push(words[start..end].join(" "));
        if end == words.len() {
            break;
        }
        start = end.saturating_sub(V2_CHUNK_OVERLAP_WORDS);
    }
    chunks
}

fn classify_columns(conn: &Connection, columns: &[ColumnMeta]) -> Result<Vec<ColumnPlan>> {
    if columns.is_empty() {
        return Ok(Vec::new());
    }
    #[derive(Default)]
    struct Stats {
        nonempty: usize,
        dynamic: usize,
        distinct: HashSet<String>,
    }
    let identifiers = columns
        .iter()
        .map(|column| db::quote_ident(&column.sql_name))
        .collect::<Vec<_>>()
        .join(", ");
    let sql =
        format!("SELECT {identifiers} FROM rows ORDER BY row_num LIMIT {V2_COLUMN_SAMPLE_ROWS}");
    let mut stats = (0..columns.len())
        .map(|_| Stats::default())
        .collect::<Vec<_>>();
    let mut stmt = conn.prepare(&sql)?;
    let mut rows = stmt.query([])?;
    while let Some(row) = rows.next()? {
        for (index, stat) in stats.iter_mut().enumerate() {
            let value: Option<String> = row.get(index)?;
            let Some(value) = value.as_deref().map(str::trim).filter(|v| !v.is_empty()) else {
                continue;
            };
            stat.nonempty += 1;
            if value_is_dynamic_identifier(value) {
                stat.dynamic += 1;
            }
            if stat.distinct.len() <= V2_COLUMN_SAMPLE_ROWS {
                stat.distinct.insert(value.to_ascii_lowercase());
            }
        }
    }

    Ok(columns
        .iter()
        .zip(stats)
        .enumerate()
        .map(|(index, (column, stat))| {
            let distinct_ratio = if stat.nonempty == 0 {
                0.0
            } else {
                stat.distinct.len() as f64 / stat.nonempty as f64
            };
            let dynamic_ratio = if stat.nonempty == 0 {
                0.0
            } else {
                stat.dynamic as f64 / stat.nonempty as f64
            };
            let mode = if exact_only_header(column) {
                ColumnMode::ExactOnly
            } else if force_text_header(column) {
                ColumnMode::Text
            } else if stat.nonempty >= 16 && dynamic_ratio >= 0.80 {
                ColumnMode::ExactOnly
            } else if stat.distinct.len() <= 128 || distinct_ratio <= 0.10 {
                ColumnMode::Categorical
            } else {
                ColumnMode::Text
            };
            ColumnPlan {
                col_index: index,
                sql_name: column.sql_name.clone(),
                original_name: column.original_name.clone(),
                mode,
            }
        })
        .collect())
}

fn normalized_label(plan: &ColumnPlan) -> String {
    let label = plan.original_name.trim();
    if label.is_empty() {
        plan.sql_name.replace('_', " ")
    } else {
        label.to_ascii_lowercase()
    }
}

fn row_documents_v2(
    plans: &[ColumnPlan],
    values: &[Option<String>],
) -> Vec<(&'static str, String, String)> {
    let mut cell_chunks: Vec<(ColumnMode, String, String, Vec<String>)> = Vec::new();
    let mut context_parts = Vec::new();
    for (plan, value) in plans.iter().zip(values) {
        if plan.mode == ColumnMode::ExactOnly {
            continue;
        }
        let Some(value) = value
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        else {
            continue;
        };
        let normalized = normalize_text(value);
        if !is_informative_text(&normalized) {
            continue;
        }
        let label = normalized_label(plan);
        let chunks = word_chunks(&normalized);
        if chunks.is_empty() {
            continue;
        }
        if plan.mode == ColumnMode::Categorical {
            context_parts.push(format!("{label}: {}", chunks[0]));
        }
        cell_chunks.push((plan.mode, plan.sql_name.clone(), label, chunks));
    }

    // Fairness: every eligible column contributes its first chunk before any early column may
    // contribute a second. Further rounds are bounded without starving later columns.
    let mut documents = Vec::new();
    for (_, column_key, label, chunks) in &cell_chunks {
        documents.push((
            "cell",
            column_key.clone(),
            format!("{label}: {}", chunks[0]),
        ));
    }
    let mut additional = 0usize;
    for round in 1..V2_MAX_CHUNKS_PER_CELL {
        for (_, column_key, label, chunks) in &cell_chunks {
            if additional == V2_MAX_ADDITIONAL_CHUNKS_PER_ROW {
                break;
            }
            if let Some(chunk) = chunks.get(round) {
                documents.push((
                    "cell_chunk",
                    column_key.clone(),
                    format!("{label}: {chunk}"),
                ));
                additional += 1;
            }
        }
    }
    if context_parts.len() >= 2 {
        documents.push(("row_context", String::new(), context_parts.join("; ")));
    }
    documents
}

fn text_sha256(kind: &str, column_key: &str, text_value: &str) -> String {
    let mut hasher = Sha256::new();
    for value in [kind, column_key, text_value] {
        hasher.update((value.len() as u64).to_le_bytes());
        hasher.update(value.as_bytes());
    }
    bytes_to_hex(&hasher.finalize())
}

fn collect_normalized_documents(
    plans: &[ColumnPlan],
    source_rows: &[(i64, Vec<Option<String>>)],
) -> BTreeMap<String, NormalizedDocument> {
    let mut documents = BTreeMap::<String, NormalizedDocument>::new();
    for (row_num, values) in source_rows {
        for (kind, column_key, text_value) in row_documents_v2(plans, values) {
            let hash = text_sha256(kind, &column_key, &text_value);
            let document = documents.entry(hash).or_insert_with(|| NormalizedDocument {
                kind,
                column_key,
                text: text_value,
                rows: BTreeSet::new(),
            });
            document.rows.insert(*row_num);
        }
    }
    documents
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticIndexSummary {
    pub rows_indexed: i64,
    pub documents_indexed: i64,
    pub mappings_written: i64,
    pub elapsed_ms: u128,
    pub from_cache: bool,
    pub resumed: bool,
    pub cancelled: bool,
    pub model_name: &'static str,
    pub model_version: &'static str,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticCandidate {
    pub row_num: i64,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticBuildProgress {
    pub build_id: i64,
    pub phase: String,
    pub rows_scanned: i64,
    pub rows_total: i64,
    pub documents_embedded: i64,
    pub mappings_written: i64,
    pub resumed_from_row: i64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticSelectionSummary {
    pub selection_id: String,
    pub documents_above_threshold: usize,
    pub documents_retained: usize,
    pub rows_matched: i64,
    pub documents_truncated: bool,
    pub broad_row_warning: bool,
    pub warnings: Vec<String>,
}

pub fn semantic_schema_hash(columns: &[ColumnMeta]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(V2_INDEX_VERSION.as_bytes());
    hasher.update(V2_NORMALIZER_VERSION.as_bytes());
    for column in columns {
        for value in [
            column.sql_name.as_str(),
            column.original_name.as_str(),
            column.inferred_type.as_str(),
        ] {
            hasher.update((value.len() as u64).to_le_bytes());
            hasher.update(value.as_bytes());
        }
        hasher.update((column.col_index as u64).to_le_bytes());
    }
    bytes_to_hex(&hasher.finalize())
}

/// Stable identity for semantic artifacts. Raw imported rows are immutable; the cache import
/// record, schema, and row count therefore identify the dataset without depending on optional
/// role, timestamp, or intelligence-enrichment tables.
pub fn semantic_dataset_hash(conn: &Connection, columns: &[ColumnMeta]) -> Result<String> {
    let mut hasher = Sha256::new();
    hasher.update(V2_INDEX_VERSION.as_bytes());
    hasher.update(semantic_schema_hash(columns).as_bytes());
    let row_count: i64 = conn.query_row("SELECT COUNT(*) FROM rows", [], |row| row.get(0))?;
    hasher.update(row_count.to_le_bytes());
    if let Ok(info) = db::load_import_info(conn) {
        for value in [
            info.source_path.as_str(),
            info.sheet_name.as_str(),
            info.imported_at.as_str(),
        ] {
            hasher.update((value.len() as u64).to_le_bytes());
            hasher.update(value.as_bytes());
        }
        hasher.update(info.row_count.to_le_bytes());
    }
    Ok(bytes_to_hex(&hasher.finalize()))
}

/// Creates only empty v2 structures. Existing v1 artifacts remain readable until a complete v2
/// build is atomically published through `_semantic_v2_active`.
pub fn create_semantic_v2_schema(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS _semantic_v2_build (
            build_id INTEGER PRIMARY KEY,
            dataset_hash TEXT NOT NULL,
            schema_hash TEXT NOT NULL,
            model_sha256 TEXT NOT NULL,
            normalizer_version TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('building','paused','ready','cancelled','failed')),
            source_rows INTEGER NOT NULL,
            cursor_row_num INTEGER NOT NULL DEFAULT 0,
            rows_scanned INTEGER NOT NULL DEFAULT 0,
            documents_seen INTEGER NOT NULL DEFAULT 0,
            documents_embedded INTEGER NOT NULL DEFAULT 0,
            mappings_written INTEGER NOT NULL DEFAULT 0,
            started_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            completed_at TEXT,
            error TEXT
         );
         CREATE INDEX IF NOT EXISTS _semantic_v2_build_identity
            ON _semantic_v2_build(dataset_hash, schema_hash, model_sha256, normalizer_version, status);
         CREATE UNIQUE INDEX IF NOT EXISTS _semantic_v2_build_unique_identity
            ON _semantic_v2_build(dataset_hash, schema_hash, model_sha256, normalizer_version);
         CREATE TABLE IF NOT EXISTS _semantic_v2_active (
            singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
            build_id INTEGER NOT NULL
         );
         CREATE TABLE IF NOT EXISTS _semantic_v2_column_plan (
            build_id INTEGER NOT NULL,
            col_index INTEGER NOT NULL,
            mode TEXT NOT NULL CHECK(mode IN ('exact_only','categorical','text')),
            sql_name TEXT NOT NULL,
            original_name TEXT NOT NULL,
            PRIMARY KEY(build_id, col_index)
         ) WITHOUT ROWID;
         CREATE TABLE IF NOT EXISTS _semantic_v2_document (
            doc_id INTEGER PRIMARY KEY,
            model_sha256 TEXT NOT NULL,
            normalizer_version TEXT NOT NULL,
            kind TEXT NOT NULL,
            column_key TEXT NOT NULL,
            text_sha256 TEXT NOT NULL,
            normalized_text TEXT NOT NULL,
            embedding BLOB NOT NULL,
            UNIQUE(model_sha256, normalizer_version, text_sha256)
         );
         CREATE TABLE IF NOT EXISTS _semantic_v2_mapping (
            build_id INTEGER NOT NULL,
            doc_id INTEGER NOT NULL,
            row_num INTEGER NOT NULL,
            PRIMARY KEY(build_id, doc_id, row_num)
         ) WITHOUT ROWID;
         CREATE TABLE IF NOT EXISTS _semantic_v2_selection (
            selection_id TEXT PRIMARY KEY,
            build_id INTEGER NOT NULL,
            dataset_hash TEXT NOT NULL,
            query_sha256 TEXT NOT NULL,
            policy_version TEXT NOT NULL,
            minimum_score REAL NOT NULL,
            documents_above_threshold INTEGER NOT NULL,
            documents_retained INTEGER NOT NULL,
            rows_matched INTEGER NOT NULL,
            documents_truncated INTEGER NOT NULL,
            broad_row_warning INTEGER NOT NULL,
            warnings_json TEXT NOT NULL,
            created_at TEXT NOT NULL
         );
         CREATE TABLE IF NOT EXISTS _semantic_v2_selection_doc (
            selection_id TEXT NOT NULL,
            doc_id INTEGER NOT NULL,
            cosine_score REAL NOT NULL,
            rank_score REAL NOT NULL,
            PRIMARY KEY(selection_id, doc_id)
         ) WITHOUT ROWID;",
    )
}

fn active_v2_build(
    conn: &Connection,
    dataset_hash: &str,
    schema_hash: &str,
) -> Result<Option<(i64, i64, i64, i64)>> {
    conn.query_row(
        "SELECT b.build_id, b.rows_scanned, b.documents_embedded, b.mappings_written
         FROM _semantic_v2_active a
         JOIN _semantic_v2_build b ON b.build_id = a.build_id
         WHERE a.singleton = 1 AND b.status = 'ready'
           AND b.dataset_hash = ?1 AND b.schema_hash = ?2
           AND b.model_sha256 = ?3 AND b.normalizer_version = ?4",
        params![
            dataset_hash,
            schema_hash,
            MODEL_SHA256,
            V2_NORMALIZER_VERSION
        ],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
    )
    .optional()
    .map_err(Into::into)
}

fn load_column_plans(conn: &Connection, build_id: i64) -> Result<Vec<ColumnPlan>> {
    let mut stmt = conn.prepare(
        "SELECT col_index, sql_name, original_name, mode
         FROM _semantic_v2_column_plan WHERE build_id = ?1 ORDER BY col_index",
    )?;
    let rows = stmt.query_map([build_id], |row| {
        let mode: String = row.get(3)?;
        Ok((
            row.get::<_, i64>(0)? as usize,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            mode,
        ))
    })?;
    let plans = rows
        .map(|row| {
            let (col_index, sql_name, original_name, mode) = row?;
            Ok(ColumnPlan {
                col_index,
                sql_name,
                original_name,
                mode: ColumnMode::parse(&mode)?,
            })
        })
        .collect();
    plans
}

fn prepare_v2_build(
    conn: &mut Connection,
    columns: &[ColumnMeta],
    dataset_hash: &str,
    schema_hash: &str,
    rows_total: i64,
) -> Result<(i64, i64, Vec<ColumnPlan>, bool)> {
    if let Some((build_id, cursor)) = conn
        .query_row(
            "SELECT build_id, cursor_row_num FROM _semantic_v2_build
             WHERE dataset_hash = ?1 AND schema_hash = ?2 AND model_sha256 = ?3
               AND normalizer_version = ?4 AND status IN ('building','paused','failed','ready')
             ORDER BY build_id DESC LIMIT 1",
            params![
                dataset_hash,
                schema_hash,
                MODEL_SHA256,
                V2_NORMALIZER_VERSION
            ],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)),
        )
        .optional()?
    {
        conn.execute(
            "UPDATE _semantic_v2_build SET status = 'building', updated_at = ?2, error = NULL
             WHERE build_id = ?1 AND status IN ('building','paused','failed')",
            params![build_id, chrono::Utc::now().to_rfc3339()],
        )?;
        return Ok((
            build_id,
            cursor,
            load_column_plans(conn, build_id)?,
            cursor > 0,
        ));
    }

    let plans = classify_columns(conn, columns)?;
    let now = chrono::Utc::now().to_rfc3339();
    let tx = conn.transaction()?;
    let inserted = tx.execute(
        "INSERT OR IGNORE INTO _semantic_v2_build (
            dataset_hash, schema_hash, model_sha256, normalizer_version, status,
            source_rows, started_at, updated_at
         ) VALUES (?1, ?2, ?3, ?4, 'building', ?5, ?6, ?6)",
        params![
            dataset_hash,
            schema_hash,
            MODEL_SHA256,
            V2_NORMALIZER_VERSION,
            rows_total,
            now,
        ],
    )?;
    if inserted == 0 {
        // Another connection won the identity race while this connection classified columns.
        // Its build and column plan are now committed because SQLite serializes these writers.
        tx.commit()?;
        return prepare_v2_build(conn, columns, dataset_hash, schema_hash, rows_total);
    }
    let build_id = tx.last_insert_rowid();
    {
        let mut insert = tx.prepare(
            "INSERT INTO _semantic_v2_column_plan (
                build_id, col_index, mode, sql_name, original_name
             ) VALUES (?1, ?2, ?3, ?4, ?5)",
        )?;
        for plan in &plans {
            insert.execute(params![
                build_id,
                plan.col_index as i64,
                plan.mode.as_str(),
                plan.sql_name,
                plan.original_name,
            ])?;
        }
    }
    tx.commit()?;
    Ok((build_id, 0, plans, false))
}

fn build_summary(
    conn: &Connection,
    build_id: i64,
    started: Instant,
    from_cache: bool,
    resumed: bool,
    cancelled: bool,
) -> Result<SemanticIndexSummary> {
    let (rows, documents, mappings) = conn.query_row(
        "SELECT rows_scanned, documents_embedded, mappings_written
         FROM _semantic_v2_build WHERE build_id = ?1",
        [build_id],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
    )?;
    Ok(SemanticIndexSummary {
        rows_indexed: rows,
        documents_indexed: documents,
        mappings_written: mappings,
        elapsed_ms: started.elapsed().as_millis(),
        from_cache,
        resumed,
        cancelled,
        model_name: MODEL_NAME,
        model_version: MODEL_VERSION,
    })
}

/// Resumable semantic-document builder. All model inference occurs outside a transaction; each
/// source-row batch is published to staging with a short idempotent transaction. `is_cancelled`
/// is checked around every potentially expensive phase so a stale loaded-file generation cannot
/// publish itself as active.
pub fn ensure_semantic_index_v2<E, C, P>(
    conn: &mut Connection,
    columns: &[ColumnMeta],
    embedder: &E,
    is_cancelled: C,
    mut on_progress: P,
) -> Result<SemanticIndexSummary>
where
    E: SemanticEmbedder + ?Sized,
    C: Fn() -> bool,
    P: FnMut(SemanticBuildProgress),
{
    let started = Instant::now();
    conn.busy_timeout(std::time::Duration::from_secs(3))?;
    create_semantic_v2_schema(conn)?;
    let rows_total: i64 = conn.query_row("SELECT COUNT(*) FROM rows", [], |row| row.get(0))?;
    let dataset_hash = semantic_dataset_hash(conn, columns)?;
    let schema_hash = semantic_schema_hash(columns);
    if let Some((build_id, _, _, _)) = active_v2_build(conn, &dataset_hash, &schema_hash)? {
        return build_summary(conn, build_id, started, true, false, false);
    }

    let (build_id, mut cursor, plans, resumed) =
        prepare_v2_build(conn, columns, &dataset_hash, &schema_hash, rows_total)?;
    let resumed_from_row = cursor;
    let identifiers = columns
        .iter()
        .map(|column| db::quote_ident(&column.sql_name))
        .collect::<Vec<_>>()
        .join(", ");

    let result = (|| -> Result<SemanticIndexSummary> {
        loop {
            if is_cancelled() {
                conn.execute(
                    "UPDATE _semantic_v2_build SET status = 'paused', updated_at = ?2
                 WHERE build_id = ?1 AND status = 'building'",
                    params![build_id, chrono::Utc::now().to_rfc3339()],
                )?;
                return build_summary(conn, build_id, started, false, resumed, true);
            }

            let source_rows = {
                let sql = format!(
                    "SELECT row_num, {identifiers} FROM rows
                 WHERE row_num > ?1 ORDER BY row_num LIMIT {V2_SOURCE_BATCH_ROWS}"
                );
                let mut stmt = conn.prepare(&sql)?;
                let mut rows = stmt.query([cursor])?;
                let mut batch = Vec::with_capacity(V2_SOURCE_BATCH_ROWS);
                while let Some(row) = rows.next()? {
                    let row_num: i64 = row.get(0)?;
                    let values = (0..columns.len())
                        .map(|index| row.get::<_, Option<String>>(index + 1))
                        .collect::<rusqlite::Result<Vec<_>>>()?;
                    batch.push((row_num, values));
                }
                batch
            };

            if source_rows.is_empty() {
                if is_cancelled() {
                    continue;
                }
                let now = chrono::Utc::now().to_rfc3339();
                let tx = conn.transaction()?;
                tx.execute(
                    "UPDATE _semantic_v2_build
                 SET status = 'ready', updated_at = ?2, completed_at = ?2
                 WHERE build_id = ?1 AND status = 'building'",
                    params![build_id, now],
                )?;
                tx.execute(
                    "INSERT INTO _semantic_v2_active(singleton, build_id) VALUES (1, ?1)
                 ON CONFLICT(singleton) DO UPDATE SET build_id = excluded.build_id",
                    [build_id],
                )?;
                tx.commit()?;
                let summary = build_summary(conn, build_id, started, false, resumed, false)?;
                on_progress(SemanticBuildProgress {
                    build_id,
                    phase: "ready".to_string(),
                    rows_scanned: summary.rows_indexed,
                    rows_total,
                    documents_embedded: summary.documents_indexed,
                    mappings_written: summary.mappings_written,
                    resumed_from_row,
                });
                return Ok(summary);
            }

            let documents = collect_normalized_documents(&plans, &source_rows);
            let mut existing = HashMap::<String, i64>::new();
            {
                let mut lookup = conn.prepare(
                    "SELECT doc_id FROM _semantic_v2_document
                 WHERE model_sha256 = ?1 AND normalizer_version = ?2 AND text_sha256 = ?3",
                )?;
                for hash in documents.keys() {
                    if let Some(doc_id) = lookup
                        .query_row(params![MODEL_SHA256, V2_NORMALIZER_VERSION, hash], |row| {
                            row.get(0)
                        })
                        .optional()?
                    {
                        existing.insert(hash.clone(), doc_id);
                    }
                }
            }

            let unknown = documents
                .iter()
                .filter(|(hash, _)| !existing.contains_key(*hash))
                .map(|(hash, document)| (hash.clone(), document.text.clone()))
                .collect::<Vec<_>>();
            let mut embeddings = HashMap::<String, Vec<u8>>::new();
            for chunk in unknown.chunks(V2_EMBED_BATCH_DOCUMENTS) {
                if is_cancelled() {
                    break;
                }
                let texts = chunk
                    .iter()
                    .map(|(_, text)| text.clone())
                    .collect::<Vec<_>>();
                let vectors = embedder.embed_batch(&texts)?;
                if is_cancelled() {
                    conn.execute(
                        "UPDATE _semantic_v2_build SET status = 'paused', updated_at = ?2
                     WHERE build_id = ?1 AND status = 'building'",
                        params![build_id, chrono::Utc::now().to_rfc3339()],
                    )?;
                    return build_summary(conn, build_id, started, false, resumed, true);
                }
                if vectors.len() != chunk.len() {
                    bail!("semantic model returned the wrong batch size");
                }
                for ((hash, _), vector) in chunk.iter().zip(vectors) {
                    if vector.len() != EMBEDDING_DIMENSIONS {
                        bail!(
                        "semantic model returned {} dimensions, expected {EMBEDDING_DIMENSIONS}",
                        vector.len()
                    );
                    }
                    embeddings.insert(hash.clone(), vector_to_blob(&vector));
                }
            }
            if is_cancelled() {
                conn.execute(
                    "UPDATE _semantic_v2_build SET status = 'paused', updated_at = ?2
                 WHERE build_id = ?1 AND status = 'building'",
                    params![build_id, chrono::Utc::now().to_rfc3339()],
                )?;
                return build_summary(conn, build_id, started, false, resumed, true);
            }

            let last_row = source_rows.last().map(|row| row.0).unwrap_or(cursor);
            let tx = conn.transaction()?;
            let claimed = tx.execute(
                "UPDATE _semantic_v2_build SET
                cursor_row_num = ?2,
                rows_scanned = rows_scanned + ?3,
                documents_seen = documents_seen + ?4,
                updated_at = ?5
             WHERE build_id = ?1 AND status = 'building' AND cursor_row_num = ?6",
                params![
                    build_id,
                    last_row,
                    source_rows.len() as i64,
                    documents.len() as i64,
                    chrono::Utc::now().to_rfc3339(),
                    cursor,
                ],
            )?;
            if claimed == 0 {
                tx.rollback()?;
                cursor = conn.query_row(
                    "SELECT cursor_row_num FROM _semantic_v2_build WHERE build_id = ?1",
                    [build_id],
                    |row| row.get(0),
                )?;
                continue;
            }
            let mut doc_ids = existing;
            {
                let mut insert_doc = tx.prepare(
                    "INSERT OR IGNORE INTO _semantic_v2_document (
                    model_sha256, normalizer_version, kind, column_key, text_sha256,
                    normalized_text, embedding
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                )?;
                let mut lookup_doc = tx.prepare(
                    "SELECT doc_id FROM _semantic_v2_document
                 WHERE model_sha256 = ?1 AND normalizer_version = ?2 AND text_sha256 = ?3",
                )?;
                for (hash, document) in &documents {
                    if !doc_ids.contains_key(hash) {
                        let embedding = embeddings.get(hash).ok_or_else(|| {
                            anyhow::anyhow!("missing semantic document embedding")
                        })?;
                        insert_doc.execute(params![
                            MODEL_SHA256,
                            V2_NORMALIZER_VERSION,
                            document.kind,
                            document.column_key,
                            hash,
                            document.text,
                            embedding,
                        ])?;
                        let doc_id = lookup_doc.query_row(
                            params![MODEL_SHA256, V2_NORMALIZER_VERSION, hash],
                            |row| row.get(0),
                        )?;
                        doc_ids.insert(hash.clone(), doc_id);
                    }
                }
            }
            let mut mappings_added = 0i64;
            {
                let mut insert_mapping = tx.prepare(
                    "INSERT OR IGNORE INTO _semantic_v2_mapping(build_id, doc_id, row_num)
                 VALUES (?1, ?2, ?3)",
                )?;
                for (hash, document) in &documents {
                    let doc_id = *doc_ids
                        .get(hash)
                        .ok_or_else(|| anyhow::anyhow!("semantic document ID was not resolved"))?;
                    for row_num in &document.rows {
                        mappings_added +=
                            insert_mapping.execute(params![build_id, doc_id, row_num])? as i64;
                    }
                }
            }
            tx.execute(
                "UPDATE _semantic_v2_build SET
                documents_embedded = documents_embedded + ?2,
                mappings_written = mappings_written + ?3,
                updated_at = ?4
             WHERE build_id = ?1",
                params![
                    build_id,
                    embeddings.len() as i64,
                    mappings_added,
                    chrono::Utc::now().to_rfc3339(),
                ],
            )?;
            tx.commit()?;
            cursor = last_row;
            let summary = build_summary(conn, build_id, started, false, resumed, false)?;
            on_progress(SemanticBuildProgress {
                build_id,
                phase: "indexing".to_string(),
                rows_scanned: summary.rows_indexed,
                rows_total,
                documents_embedded: summary.documents_indexed,
                mappings_written: summary.mappings_written,
                resumed_from_row,
            });
        }
    })();

    if let Err(error) = &result {
        // Embedding and staging failures are resumable. The active pointer is intentionally left
        // untouched, so a prior complete generation remains the only queryable semantic index.
        let mut message = format!("{error:#}");
        message.truncate(2_048);
        let _ = conn.execute(
            "UPDATE _semantic_v2_build SET status = 'paused', error = ?2, updated_at = ?3
             WHERE build_id = ?1 AND status = 'building'",
            params![build_id, message, chrono::Utc::now().to_rfc3339()],
        );
    }
    result
}

pub fn semantic_index_ready(conn: &Connection, columns: &[ColumnMeta]) -> Result<bool> {
    if !table_exists(conn, "_semantic_v2_active")? || !table_exists(conn, "_semantic_v2_build")? {
        return Ok(false);
    }
    let dataset_hash = semantic_dataset_hash(conn, columns)?;
    Ok(active_v2_build(conn, &dataset_hash, &semantic_schema_hash(columns))?.is_some())
}

pub fn ensure_semantic_index(
    conn: &mut Connection,
    columns: &[ColumnMeta],
    model: &SemanticModel,
) -> Result<SemanticIndexSummary> {
    ensure_semantic_index_v2(conn, columns, model, || false, |_| {})
}

#[derive(Debug, Clone, Copy)]
pub struct SemanticSearchPolicy {
    pub maximum_documents: usize,
    pub minimum_score: f32,
}

impl Default for SemanticSearchPolicy {
    fn default() -> Self {
        Self {
            maximum_documents: V2_DEFAULT_DOCUMENT_CANDIDATES,
            minimum_score: V2_DEFAULT_MINIMUM_SCORE,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ScoredDocument {
    doc_id: i64,
    score: f32,
}

impl PartialEq for ScoredDocument {
    fn eq(&self, other: &Self) -> bool {
        self.doc_id == other.doc_id && self.score.to_bits() == other.score.to_bits()
    }
}

impl Eq for ScoredDocument {}

impl PartialOrd for ScoredDocument {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredDocument {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| other.doc_id.cmp(&self.doc_id))
    }
}

fn active_build_identity(conn: &Connection) -> Result<(i64, String, i64)> {
    conn.query_row(
        "SELECT b.build_id, b.dataset_hash, b.source_rows
         FROM _semantic_v2_active a
         JOIN _semantic_v2_build b ON b.build_id = a.build_id
         WHERE a.singleton = 1 AND b.status = 'ready'",
        [],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
    )
    .map_err(Into::into)
}

fn rank_semantic_documents<E: SemanticEmbedder + ?Sized>(
    conn: &Connection,
    embedder: &E,
    build_id: i64,
    query: &str,
    policy: SemanticSearchPolicy,
) -> Result<(Vec<ScoredDocument>, usize)> {
    let query = query.trim();
    if query.is_empty() {
        bail!("semantic search query is empty");
    }
    if query.chars().count() > MAX_QUERY_CHARS {
        bail!("semantic search query exceeds {MAX_QUERY_CHARS} characters");
    }
    if !table_exists(conn, "_semantic_v2_active")? {
        bail!("semantic index is not ready");
    }
    let maximum_documents = policy
        .maximum_documents
        .clamp(1, V2_MAX_DOCUMENT_CANDIDATES);
    // Cell documents replace volatile literals with stable placeholders. Applying the same
    // normalization to the semantic query aligns dynamic-ID templates; exact literals remain
    // available to the independent FTS/structured branches.
    let normalized_query = normalize_text(query);
    let query_embedding = embedder.embed(if normalized_query.is_empty() {
        query
    } else {
        &normalized_query
    })?;
    let mut heap: BinaryHeap<Reverse<ScoredDocument>> =
        BinaryHeap::with_capacity(maximum_documents + 1);
    let mut above_threshold = 0usize;
    let mut stmt = conn.prepare(
        "SELECT d.doc_id, d.embedding
         FROM _semantic_v2_document d
         WHERE EXISTS (
            SELECT 1 FROM _semantic_v2_mapping m
            WHERE m.build_id = ?1 AND m.doc_id = d.doc_id
         )",
    )?;
    let mut rows = stmt.query([build_id])?;
    while let Some(row) = rows.next()? {
        let doc_id: i64 = row.get(0)?;
        let blob: Vec<u8> = row.get(1)?;
        let score = dot_blob(&query_embedding, &blob)?;
        if !score.is_finite() || score < policy.minimum_score {
            continue;
        }
        above_threshold += 1;
        let candidate = ScoredDocument { doc_id, score };
        if heap.len() < maximum_documents {
            heap.push(Reverse(candidate));
        } else if heap.peek().is_some_and(|smallest| candidate > smallest.0) {
            heap.pop();
            heap.push(Reverse(candidate));
        }
    }
    let mut documents = heap
        .into_iter()
        .map(|Reverse(document)| document)
        .collect::<Vec<_>>();
    documents.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.doc_id.cmp(&right.doc_id))
    });
    Ok((documents, above_threshold))
}

fn new_selection_id(dataset_hash: &str, query: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(dataset_hash.as_bytes());
    hasher.update(query.as_bytes());
    hasher.update(chrono::Utc::now().to_rfc3339().as_bytes());
    hasher.update(std::process::id().to_le_bytes());
    bytes_to_hex(&hasher.finalize())
}

pub fn create_semantic_selection<E: SemanticEmbedder + ?Sized>(
    conn: &mut Connection,
    columns: &[ColumnMeta],
    embedder: &E,
    query: &str,
    policy: SemanticSearchPolicy,
) -> Result<SemanticSelectionSummary> {
    if !semantic_index_ready(conn, columns)? {
        bail!("semantic index is not ready");
    }
    let (build_id, dataset_hash, source_rows) = active_build_identity(conn)?;
    if semantic_dataset_hash(conn, columns)? != dataset_hash {
        bail!("semantic index belongs to a different dataset");
    }
    let (documents, above_threshold) =
        rank_semantic_documents(conn, embedder, build_id, query, policy)?;
    let truncated = above_threshold > documents.len();
    let selection_id = new_selection_id(&dataset_hash, query);
    let query_hash = {
        let mut hasher = Sha256::new();
        hasher.update(query.as_bytes());
        bytes_to_hex(&hasher.finalize())
    };
    let tx = conn.transaction()?;
    tx.execute(
        "INSERT INTO _semantic_v2_selection (
            selection_id, build_id, dataset_hash, query_sha256, policy_version,
            minimum_score, documents_above_threshold, documents_retained, rows_matched,
            documents_truncated, broad_row_warning, warnings_json, created_at
         ) VALUES (?1, ?2, ?3, ?4, 'semantic-doc-search-v1', ?5, ?6, ?7, 0, ?8, 0, '[]', ?9)",
        params![
            selection_id,
            build_id,
            dataset_hash,
            query_hash,
            policy.minimum_score,
            above_threshold as i64,
            documents.len() as i64,
            i64::from(truncated),
            chrono::Utc::now().to_rfc3339(),
        ],
    )?;
    {
        let mut insert = tx.prepare(
            "INSERT INTO _semantic_v2_selection_doc (
                selection_id, doc_id, cosine_score, rank_score
             ) VALUES (?1, ?2, ?3, ?3)",
        )?;
        for document in &documents {
            insert.execute(params![selection_id, document.doc_id, document.score])?;
        }
    }
    let rows_matched: i64 = tx.query_row(
        "SELECT COUNT(DISTINCT m.row_num)
         FROM _semantic_v2_selection_doc sd
         JOIN _semantic_v2_mapping m ON m.build_id = ?2 AND m.doc_id = sd.doc_id
         WHERE sd.selection_id = ?1",
        params![selection_id, build_id],
        |row| row.get(0),
    )?;
    let broad = rows_matched > 10_000 || rows_matched.saturating_mul(4) > source_rows;
    let mut warnings = Vec::new();
    warnings.push(
        "Semantic retrieval uses bounded normalized cell/chunk documents; lexical and structured conditions remain complete."
            .to_string(),
    );
    if truncated {
        warnings.push(format!(
            "Semantic document candidates were limited to {}; lexical and structured matches remain complete.",
            documents.len()
        ));
    }
    if broad {
        warnings.push(format!(
            "Semantic expansion matched {rows_matched} rows; refine the request if this is too broad."
        ));
    }
    tx.execute(
        "UPDATE _semantic_v2_selection SET rows_matched = ?2, broad_row_warning = ?3,
            warnings_json = ?4 WHERE selection_id = ?1",
        params![
            selection_id,
            rows_matched,
            i64::from(broad),
            serde_json::to_string(&warnings)?,
        ],
    )?;
    tx.commit()?;
    Ok(SemanticSelectionSummary {
        selection_id,
        documents_above_threshold: above_threshold,
        documents_retained: documents.len(),
        rows_matched,
        documents_truncated: truncated,
        broad_row_warning: broad,
        warnings,
    })
}

pub fn validate_semantic_selection(
    conn: &Connection,
    columns: &[ColumnMeta],
    selection_id: &str,
) -> Result<()> {
    if selection_id.len() != 64 || !selection_id.chars().all(|ch| ch.is_ascii_hexdigit()) {
        bail!("semantic selection ID is invalid");
    }
    let expected = semantic_dataset_hash(conn, columns)?;
    let valid: bool = conn.query_row(
        "SELECT EXISTS(
            SELECT 1 FROM _semantic_v2_selection s
            JOIN _semantic_v2_active a ON a.singleton = 1 AND a.build_id = s.build_id
            JOIN _semantic_v2_build b ON b.build_id = s.build_id
            WHERE s.selection_id = ?1 AND s.dataset_hash = ?2 AND b.status = 'ready'
         )",
        params![selection_id, expected],
        |row| row.get::<_, i64>(0).map(|value| value != 0),
    )?;
    if !valid {
        bail!("semantic selection is unknown, stale, or belongs to another dataset");
    }
    Ok(())
}

pub fn semantic_selection_reasons(
    conn: &Connection,
    selection_id: &str,
    row_numbers: &[i64],
) -> Result<HashMap<i64, Vec<String>>> {
    if row_numbers.is_empty() {
        return Ok(HashMap::new());
    }
    let mut reasons = HashMap::<i64, Vec<String>>::new();
    let mut stmt = conn.prepare(
        "SELECT m.row_num, d.normalized_text, sd.cosine_score
         FROM _semantic_v2_selection s
         JOIN _semantic_v2_selection_doc sd ON sd.selection_id = s.selection_id
         JOIN _semantic_v2_mapping m ON m.build_id = s.build_id AND m.doc_id = sd.doc_id
         JOIN _semantic_v2_document d ON d.doc_id = sd.doc_id
         WHERE s.selection_id = ?1 AND m.row_num = ?2
         ORDER BY sd.cosine_score DESC LIMIT 3",
    )?;
    for row_num in row_numbers {
        let matches = stmt
            .query_map(params![selection_id, row_num], |row| {
                let text: String = row.get(1)?;
                let score: f32 = row.get(2)?;
                Ok(format!("semantic {:.3}: {text}", score))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        if !matches.is_empty() {
            reasons.insert(*row_num, matches);
        }
    }
    Ok(reasons)
}

/// Compatibility API retained for existing tests/callers. Production v2 persists a document
/// selection and expands all of its mapped rows through QueryExpression instead of using this
/// bounded row projection.
pub fn semantic_search(
    conn: &Connection,
    model: &SemanticModel,
    query: &str,
    top_k: usize,
    minimum_score: f32,
) -> Result<Vec<SemanticCandidate>> {
    let (build_id, _, _) = active_build_identity(conn)?;
    let (documents, _) = rank_semantic_documents(
        conn,
        model,
        build_id,
        query,
        SemanticSearchPolicy {
            maximum_documents: top_k.clamp(1, MAX_TOP_K),
            minimum_score,
        },
    )?;
    if documents.is_empty() {
        return Ok(Vec::new());
    }
    let mut scores = HashMap::<i64, f32>::new();
    let mut stmt = conn
        .prepare("SELECT row_num FROM _semantic_v2_mapping WHERE build_id = ?1 AND doc_id = ?2")?;
    for document in documents {
        for row in stmt.query_map(params![build_id, document.doc_id], |row| {
            row.get::<_, i64>(0)
        })? {
            let row_num = row?;
            scores
                .entry(row_num)
                .and_modify(|score| *score = score.max(document.score))
                .or_insert(document.score);
        }
    }
    let mut candidates = scores
        .into_iter()
        .map(|(row_num, score)| SemanticCandidate { row_num, score })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.row_num.cmp(&right.row_num))
    });
    candidates.truncate(top_k.clamp(1, MAX_TOP_K));
    Ok(candidates)
}

fn vector_to_blob(vector: &[f32]) -> Vec<u8> {
    let mut blob = Vec::with_capacity(vector.len() * std::mem::size_of::<f32>());
    for value in vector {
        blob.extend_from_slice(&value.to_le_bytes());
    }
    blob
}

fn dot_blob(query: &[f32], blob: &[u8]) -> Result<f32> {
    if blob.len() != query.len() * std::mem::size_of::<f32>() {
        bail!("semantic index contains an invalid embedding length");
    }
    Ok(query
        .iter()
        .zip(blob.chunks_exact(4))
        .map(|(left, bytes)| {
            let right = f32::from_le_bytes(bytes.try_into().expect("four-byte chunk"));
            left * right
        })
        .sum())
}

fn verify_sha256(path: &Path, expected: &str, label: &str) -> Result<()> {
    let actual =
        sha256_file(path).with_context(|| format!("hashing {label} {}", path.display()))?;
    if actual != expected {
        bail!("{label} checksum mismatch: expected {expected}, got {actual}");
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(bytes_to_hex(&hasher.finalize()))
}

fn bytes_to_hex(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(output, "{byte:02x}");
    }
    output
}

fn table_exists(conn: &Connection, table: &str) -> rusqlite::Result<bool> {
    conn.query_row(
        "SELECT EXISTS(
            SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?1
         )",
        [table],
        |row| row.get::<_, i64>(0),
    )
    .map(|value| value != 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering};
    use std::sync::{Arc, Barrier, Mutex};

    #[derive(Default)]
    struct FakeEmbedder {
        calls: AtomicUsize,
        fail_on_call: Option<usize>,
        cancel_after_call: Option<Arc<AtomicBool>>,
        first_call_barrier: Option<Arc<Barrier>>,
        concurrent_write_path: Option<std::path::PathBuf>,
        seen: Mutex<Vec<String>>,
    }

    impl FakeEmbedder {
        fn vector() -> Vec<f32> {
            let mut vector = vec![0.0; EMBEDDING_DIMENSIONS];
            vector[0] = 1.0;
            vector
        }

        fn call_count(&self) -> usize {
            self.calls.load(AtomicOrdering::SeqCst)
        }
    }

    impl SemanticEmbedder for FakeEmbedder {
        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            let call = self.calls.fetch_add(1, AtomicOrdering::SeqCst) + 1;
            if call == 1 {
                if let Some(barrier) = &self.first_call_barrier {
                    barrier.wait();
                }
                if let Some(path) = &self.concurrent_write_path {
                    let audit = Connection::open(path)?;
                    audit.busy_timeout(std::time::Duration::from_millis(500))?;
                    audit.execute(
                        "INSERT INTO _semantic_test_audit(note) VALUES ('during inference')",
                        [],
                    )?;
                }
            }
            self.seen.lock().unwrap().extend(texts.iter().cloned());
            if self.fail_on_call == Some(call) {
                bail!("synthetic embedding interruption on call {call}");
            }
            if let Some(cancelled) = &self.cancel_after_call {
                cancelled.store(true, AtomicOrdering::SeqCst);
            }
            Ok(texts.iter().map(|_| Self::vector()).collect())
        }
    }

    fn text_column(sql_name: &str, original_name: &str, col_index: usize) -> ColumnMeta {
        ColumnMeta {
            sql_name: sql_name.to_string(),
            original_name: original_name.to_string(),
            col_index,
            inferred_type: "text".to_string(),
        }
    }

    fn alphabetic_id(mut value: usize) -> String {
        let mut output = String::new();
        loop {
            output.push((b'a' + (value % 26) as u8) as char);
            value /= 26;
            if value == 0 {
                break;
            }
        }
        output
    }

    fn populate_messages(
        conn: &mut Connection,
        columns: &[ColumnMeta],
        count: usize,
        unique: bool,
    ) {
        db::create_schema(conn, columns).unwrap();
        let tx = conn.transaction().unwrap();
        {
            let mut insert = tx
                .prepare("INSERT INTO rows(row_num, event_id, message) VALUES (?1, ?2, ?3)")
                .unwrap();
            for index in 0..count {
                let message = if unique {
                    format!(
                        "suspicious process execution variant {}",
                        alphabetic_id(index)
                    )
                } else {
                    "credential dumping process observed".to_string()
                };
                insert
                    .execute(params![index as i64 + 1, format!("evt-{index}"), message])
                    .unwrap();
            }
        }
        tx.commit().unwrap();
    }

    fn message_columns() -> Vec<ColumnMeta> {
        vec![
            text_column("event_id", "Event ID", 0),
            text_column("message", "Message", 1),
        ]
    }

    fn temporary_database_path(label: &str) -> std::path::PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "log-parser-{label}-{}-{nonce}.sqlite",
            std::process::id()
        ))
    }

    fn columns() -> Vec<ColumnMeta> {
        vec![
            ColumnMeta {
                sql_name: "timestamp".into(),
                original_name: "Time Generated".into(),
                col_index: 0,
                inferred_type: "timestamp".into(),
            },
            ColumnMeta {
                sql_name: "message".into(),
                original_name: "Message".into(),
                col_index: 1,
                inferred_type: "text".into(),
            },
        ]
    }

    #[test]
    fn vector_blob_dot_product_round_trips() {
        let query = vec![0.5f32, -0.25, 0.75];
        let candidate = vec![0.25f32, 0.5, 0.75];
        let score = dot_blob(&query, &vector_to_blob(&candidate)).unwrap();
        assert!((score - 0.5625).abs() < 1e-6);
        assert!(dot_blob(&query, &[0, 1]).is_err());
    }

    #[test]
    fn semantic_schema_hash_is_stable_and_sensitive_to_column_meaning() {
        let first = semantic_schema_hash(&columns());
        let second = semantic_schema_hash(&columns());
        assert_eq!(first, second);
        let mut changed = columns();
        changed[1].original_name = "Event Description".into();
        assert_ne!(first, semantic_schema_hash(&changed));
    }

    #[test]
    fn semantic_v2_schema_is_additive_and_dataset_bound() {
        let conn = Connection::open_in_memory().unwrap();
        let columns = columns();
        db::create_schema(&conn, &columns).unwrap();
        conn.execute(
            "INSERT INTO rows (row_num, timestamp, message) VALUES (1, '2026-01-01', 'event')",
            [],
        )
        .unwrap();
        create_semantic_v2_schema(&conn).unwrap();
        for table in [
            "_semantic_v2_build",
            "_semantic_v2_active",
            "_semantic_v2_column_plan",
            "_semantic_v2_document",
            "_semantic_v2_mapping",
            "_semantic_v2_selection",
            "_semantic_v2_selection_doc",
        ] {
            assert!(table_exists(&conn, table).unwrap(), "missing {table}");
        }
        let first = semantic_dataset_hash(&conn, &columns).unwrap();
        conn.execute(
            "INSERT INTO rows (row_num, timestamp, message) VALUES (2, '2026-01-02', 'event')",
            [],
        )
        .unwrap();
        assert_ne!(first, semantic_dataset_hash(&conn, &columns).unwrap());
        assert!(!table_exists(&conn, "_semantic_index").unwrap());
    }

    #[test]
    fn v2_normalizer_deduplicates_dynamic_ids_and_never_starves_the_final_column() {
        let conn = Connection::open_in_memory().unwrap();
        let columns = vec![
            text_column("event_id", "Event GUID", 0),
            text_column("verbose_message", "Message", 1),
            text_column("final_evidence", "Evidence Description", 2),
        ];
        db::create_schema(&conn, &columns).unwrap();
        let long_prefix = (0..300)
            .map(|index| format!("background{}", alphabetic_id(index)))
            .collect::<Vec<_>>()
            .join(" ");
        for row_num in 1..=32i64 {
            conn.execute(
                "INSERT INTO rows(row_num, event_id, verbose_message, final_evidence)
                 VALUES (?1, ?2, ?3, ?4)",
                params![
                    row_num,
                    format!("550e8400-e29b-41d4-a716-{row_num:012}"),
                    format!("{long_prefix} process {row_num}"),
                    "credential dumping observed in final field",
                ],
            )
            .unwrap();
        }

        let plans = classify_columns(&conn, &columns).unwrap();
        assert_eq!(plans[0].mode, ColumnMode::ExactOnly);
        let values = vec![
            Some("550e8400-e29b-41d4-a716-446655440000".to_string()),
            Some(format!("{long_prefix} process 9384")),
            Some("credential dumping observed in final field".to_string()),
        ];
        let documents = row_documents_v2(&plans, &values);
        assert_eq!(documents[0].1, "verbose_message");
        assert_eq!(documents[1].1, "final_evidence");
        assert_eq!(documents[2].0, "cell_chunk");

        let source_rows = vec![
            (1, values.clone()),
            (
                2,
                vec![
                    Some("550e8400-e29b-41d4-a716-446655449999".to_string()),
                    Some(format!("{long_prefix} process 12001")),
                    Some("credential dumping observed in final field".to_string()),
                ],
            ),
        ];
        let deduplicated = collect_normalized_documents(&plans, &source_rows);
        assert!(deduplicated
            .values()
            .all(|document| document.rows.len() == 2));
        assert!(deduplicated
            .values()
            .all(|document| !document.text.contains("9384") && !document.text.contains("12001")));
    }

    #[test]
    fn v2_selection_expands_every_mapping_beyond_legacy_row_caps() {
        let mut conn = Connection::open_in_memory().unwrap();
        let columns = message_columns();
        populate_messages(&mut conn, &columns, 1_601, false);
        let embedder = FakeEmbedder::default();

        let summary =
            ensure_semantic_index_v2(&mut conn, &columns, &embedder, || false, |_| {}).unwrap();
        assert_eq!(summary.rows_indexed, 1_601);
        assert_eq!(summary.documents_indexed, 1);
        assert_eq!(summary.mappings_written, 1_601);

        embedder.seen.lock().unwrap().clear();
        let selection = create_semantic_selection(
            &mut conn,
            &columns,
            &embedder,
            "credential dumping from 10.20.30.40 process 9842",
            SemanticSearchPolicy {
                maximum_documents: 1,
                minimum_score: -1.0,
            },
        )
        .unwrap();
        assert_eq!(selection.documents_retained, 1);
        assert_eq!(selection.rows_matched, 1_601);
        assert!(selection.broad_row_warning);
        assert!(selection.warnings[0].contains("bounded"));
        validate_semantic_selection(&conn, &columns, &selection.selection_id).unwrap();
        let seen = embedder.seen.lock().unwrap();
        assert_eq!(
            seen.last().unwrap(),
            "credential dumping from <ip> process <number>"
        );
    }

    #[test]
    fn v2_cancel_resumes_after_a_short_committed_batch_and_reports_real_totals() {
        let mut conn = Connection::open_in_memory().unwrap();
        let columns = message_columns();
        populate_messages(&mut conn, &columns, 600, false);
        let embedder = FakeEmbedder::default();
        let cancelled = Arc::new(AtomicBool::new(false));
        let cancel_from_progress = Arc::clone(&cancelled);
        let first = ensure_semantic_index_v2(
            &mut conn,
            &columns,
            &embedder,
            || cancelled.load(AtomicOrdering::SeqCst),
            move |progress| {
                if progress.rows_scanned >= V2_SOURCE_BATCH_ROWS as i64 {
                    cancel_from_progress.store(true, AtomicOrdering::SeqCst);
                }
            },
        )
        .unwrap();
        assert!(first.cancelled);
        assert_eq!(first.rows_indexed, V2_SOURCE_BATCH_ROWS as i64);
        assert!(!semantic_index_ready(&conn, &columns).unwrap());

        let progress = Arc::new(Mutex::new(Vec::<SemanticBuildProgress>::new()));
        let progress_sink = Arc::clone(&progress);
        let resumed = ensure_semantic_index_v2(
            &mut conn,
            &columns,
            &embedder,
            || false,
            move |update| progress_sink.lock().unwrap().push(update),
        )
        .unwrap();
        assert!(resumed.resumed);
        assert_eq!(resumed.rows_indexed, 600);
        assert_eq!(resumed.documents_indexed, 1);
        assert_eq!(resumed.mappings_written, 600);
        let ready = progress.lock().unwrap().last().unwrap().clone();
        assert_eq!(ready.phase, "ready");
        assert_eq!(ready.documents_embedded, 1);
        assert_eq!(ready.mappings_written, 600);
    }

    #[test]
    fn v2_cancellation_after_inference_exits_without_reembedding() {
        let mut conn = Connection::open_in_memory().unwrap();
        let columns = message_columns();
        populate_messages(&mut conn, &columns, 300, true);
        let cancelled = Arc::new(AtomicBool::new(false));
        let embedder = FakeEmbedder {
            cancel_after_call: Some(Arc::clone(&cancelled)),
            ..Default::default()
        };
        let summary = ensure_semantic_index_v2(
            &mut conn,
            &columns,
            &embedder,
            || cancelled.load(AtomicOrdering::SeqCst),
            |_| {},
        )
        .unwrap();
        assert!(summary.cancelled);
        assert_eq!(summary.rows_indexed, 0);
        assert_eq!(embedder.call_count(), 1);
    }

    #[test]
    fn v2_embedding_failure_is_persisted_then_resumes_without_duplicate_progress() {
        let mut conn = Connection::open_in_memory().unwrap();
        let columns = message_columns();
        populate_messages(&mut conn, &columns, 300, true);
        let failing = FakeEmbedder {
            fail_on_call: Some(17),
            ..Default::default()
        };
        let error =
            ensure_semantic_index_v2(&mut conn, &columns, &failing, || false, |_| {}).unwrap_err();
        assert!(error
            .to_string()
            .contains("synthetic embedding interruption"));
        let (status, cursor, stored_error): (String, i64, Option<String>) = conn
            .query_row(
                "SELECT status, cursor_row_num, error FROM _semantic_v2_build",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(status, "paused");
        assert_eq!(cursor, V2_SOURCE_BATCH_ROWS as i64);
        assert!(stored_error
            .unwrap()
            .contains("synthetic embedding interruption"));
        assert!(!semantic_index_ready(&conn, &columns).unwrap());

        let healthy = FakeEmbedder::default();
        let summary =
            ensure_semantic_index_v2(&mut conn, &columns, &healthy, || false, |_| {}).unwrap();
        assert!(summary.resumed);
        assert_eq!(summary.rows_indexed, 300);
        assert_eq!(summary.mappings_written, 300);

        let selection = create_semantic_selection(
            &mut conn,
            &columns,
            &healthy,
            "suspicious process",
            SemanticSearchPolicy {
                maximum_documents: 5,
                minimum_score: -1.0,
            },
        )
        .unwrap();
        assert_eq!(selection.documents_above_threshold, 300);
        assert_eq!(selection.documents_retained, 5);
        assert!(selection.documents_truncated);
        assert!(selection
            .warnings
            .iter()
            .any(|warning| warning.contains("limited")));
    }

    #[test]
    fn semantic_selection_must_belong_to_the_current_active_build_and_dataset() {
        let mut conn = Connection::open_in_memory().unwrap();
        let columns = message_columns();
        populate_messages(&mut conn, &columns, 2, false);
        let embedder = FakeEmbedder::default();
        ensure_semantic_index_v2(&mut conn, &columns, &embedder, || false, |_| {}).unwrap();
        let selection = create_semantic_selection(
            &mut conn,
            &columns,
            &embedder,
            "credential dumping",
            SemanticSearchPolicy {
                maximum_documents: 1,
                minimum_score: -1.0,
            },
        )
        .unwrap();
        validate_semantic_selection(&conn, &columns, &selection.selection_id).unwrap();

        let (dataset_hash, schema_hash): (String, String) = conn
            .query_row(
                "SELECT dataset_hash, schema_hash FROM _semantic_v2_build",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        conn.execute(
            "INSERT INTO _semantic_v2_build (
                dataset_hash, schema_hash, model_sha256, normalizer_version, status,
                source_rows, cursor_row_num, rows_scanned, started_at, updated_at, completed_at
             ) VALUES (?1, ?2, ?3, 'different-normalizer', 'ready', 2, 2, 2, ?4, ?4, ?4)",
            params![
                dataset_hash,
                schema_hash,
                MODEL_SHA256,
                chrono::Utc::now().to_rfc3339()
            ],
        )
        .unwrap();
        let replacement = conn.last_insert_rowid();
        conn.execute(
            "UPDATE _semantic_v2_active SET build_id = ?1",
            [replacement],
        )
        .unwrap();
        assert!(validate_semantic_selection(&conn, &columns, &selection.selection_id).is_err());

        conn.execute("DELETE FROM rows WHERE row_num = 2", [])
            .unwrap();
        assert!(validate_semantic_selection(&conn, &columns, &selection.selection_id).is_err());
        assert!(validate_semantic_selection(&conn, &columns, "forged").is_err());
    }

    #[test]
    fn overlapping_v2_builders_claim_each_batch_once() {
        let path = temporary_database_path("semantic-overlap");
        let columns = message_columns();
        {
            let mut conn = Connection::open(&path).unwrap();
            populate_messages(&mut conn, &columns, 520, true);
        }
        let barrier = Arc::new(Barrier::new(2));
        let mut workers = Vec::new();
        for _ in 0..2 {
            let path = path.clone();
            let columns = columns.clone();
            let barrier = Arc::clone(&barrier);
            workers.push(std::thread::spawn(move || {
                let mut conn = Connection::open(path).unwrap();
                let embedder = FakeEmbedder {
                    first_call_barrier: Some(barrier),
                    ..Default::default()
                };
                ensure_semantic_index_v2(&mut conn, &columns, &embedder, || false, |_| {})
            }));
        }
        for worker in workers {
            worker.join().unwrap().unwrap();
        }

        let conn = Connection::open(&path).unwrap();
        let (builds, rows_scanned, mappings): (i64, i64, i64) = conn
            .query_row(
                "SELECT COUNT(*), MAX(rows_scanned), MAX(mappings_written)
                 FROM _semantic_v2_build",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(builds, 1);
        assert_eq!(rows_scanned, 520);
        assert_eq!(mappings, 520);
        let mapped: i64 = conn
            .query_row("SELECT COUNT(*) FROM _semantic_v2_mapping", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(mapped, 520);
        drop(conn);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn v2_model_inference_does_not_hold_a_database_write_lock() {
        let path = temporary_database_path("semantic-audit-write");
        let columns = message_columns();
        let mut conn = Connection::open(&path).unwrap();
        populate_messages(&mut conn, &columns, 300, true);
        conn.execute_batch("CREATE TABLE _semantic_test_audit(note TEXT NOT NULL);")
            .unwrap();
        let embedder = FakeEmbedder {
            concurrent_write_path: Some(path.clone()),
            ..Default::default()
        };
        let summary =
            ensure_semantic_index_v2(&mut conn, &columns, &embedder, || false, |_| {}).unwrap();
        assert_eq!(summary.rows_indexed, 300);
        let audit_rows: i64 = conn
            .query_row("SELECT COUNT(*) FROM _semantic_test_audit", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(audit_rows, 1);
        drop(conn);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    #[ignore = "loads the pinned all-MiniLM-L6-v2 model"]
    fn real_semantic_model_places_related_security_text_closer() {
        let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let resources = manifest.join("resources");
        let model = SemanticModel::load(
            &resources.join(MODEL_RESOURCE_PATH),
            &resources.join(TOKENIZER_RESOURCE_PATH),
            &resources.join(CONFIG_RESOURCE_PATH),
        )
        .unwrap();
        let embeddings = model
            .embed_batch(&[
                "failed interactive login for alice".to_string(),
                "user authentication was denied for alice".to_string(),
                "printer toner level is normal".to_string(),
            ])
            .unwrap();
        let related = embeddings[0]
            .iter()
            .zip(&embeddings[1])
            .map(|(left, right)| left * right)
            .sum::<f32>();
        let unrelated = embeddings[0]
            .iter()
            .zip(&embeddings[2])
            .map(|(left, right)| left * right)
            .sum::<f32>();
        assert!(
            related > unrelated + 0.15,
            "{related} <= {unrelated} + 0.15"
        );
    }

    #[test]
    #[ignore = "loads the pinned model and builds a real semantic row index"]
    fn real_semantic_index_retrieves_paraphrased_evidence_from_raw_rows() {
        let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let resources = manifest.join("resources");
        let model = SemanticModel::load(
            &resources.join(MODEL_RESOURCE_PATH),
            &resources.join(TOKENIZER_RESOURCE_PATH),
            &resources.join(CONFIG_RESOURCE_PATH),
        )
        .unwrap();
        let mut conn = Connection::open_in_memory().unwrap();
        let columns = columns();
        db::create_schema(&conn, &columns).unwrap();
        for (row_num, timestamp, message) in [
            (
                1i64,
                "2026-01-01T00:00:00Z",
                "interactive sign-in rejected for alice",
            ),
            (2, "2026-01-01T00:01:00Z", "printer toner level is normal"),
            (
                3,
                "2026-01-01T00:02:00Z",
                "authentication was denied for account alice",
            ),
            (4, "2026-01-01T00:03:00Z", "successful backup completed"),
        ] {
            conn.execute(
                "INSERT INTO rows (row_num, timestamp, message) VALUES (?1, ?2, ?3)",
                params![row_num, timestamp, message],
            )
            .unwrap();
        }

        let summary = ensure_semantic_index(&mut conn, &columns, &model).unwrap();
        assert_eq!(summary.rows_indexed, 4);
        assert!(!summary.from_cache);
        let cached = ensure_semantic_index(&mut conn, &columns, &model).unwrap();
        assert!(cached.from_cache);

        let candidates = semantic_search(&conn, &model, "failed login for alice", 3, -1.0).unwrap();
        let related_positions = [1i64, 3i64]
            .into_iter()
            .map(|row_num| {
                candidates
                    .iter()
                    .position(|candidate| candidate.row_num == row_num)
                    .expect("related evidence should be in the top three")
            })
            .collect::<Vec<_>>();
        let printer_position = candidates
            .iter()
            .position(|candidate| candidate.row_num == 2);
        assert!(
            printer_position.is_none()
                || related_positions
                    .iter()
                    .all(|related| *related < printer_position.unwrap()),
            "candidates={candidates:?}"
        );
    }
}
