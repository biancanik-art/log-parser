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
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
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

pub const V2_INDEX_VERSION: &str = "semantic-document-v3";
pub const V2_NORMALIZER_VERSION: &str = "dfir-cell-normalizer-v3";
const V2_LEGACY_UNRECORDED_IDENTITY: &str = "legacy-unrecorded";
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
static V2_WORKER_SEQUENCE: AtomicU64 = AtomicU64::new(1);

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
const V2_MAX_PRIMARY_DOCUMENTS_PER_ROW: usize = 48;
const V2_MAX_ADDITIONAL_CHUNKS_PER_ROW: usize = 15;
const V2_MAX_DOCUMENTS_PER_ROW: usize = 64;
const V2_MAX_CELL_INPUT_CHARS: usize = 16_384;
const V2_MAX_DOCUMENT_CHARS: usize = 2_048;
const V2_MAX_MAPPED_DOCUMENTS_PER_BUILD: i64 = 100_000;
const V2_MAX_MAPPINGS_PER_BUILD: i64 = 6_000_000;
const V2_MAX_SELECTIONS_PER_BUILD: i64 = 512;
const V2_MAX_SELECTION_CLEANUP_PER_REQUEST: i64 = 64;
const V2_STALE_SELECTION_DOC_PRUNE_BATCH: usize = 4_096;
const V2_STALE_SELECTION_PRUNE_BATCH: usize = 64;
const V2_STALE_MAPPING_PRUNE_BATCH: usize = 16_384;
const V2_STALE_COLUMN_PRUNE_BATCH: usize = 256;
const V2_STALE_BUILD_PRUNE_BATCH: usize = 32;
const V2_ORPHAN_DOCUMENT_PRUNE_BATCH: usize = 4_096;
const V1_INDEX_PRUNE_BATCH: usize = 1_024;
const V2_PRUNE_PASSES_PER_INVOCATION: usize = 16;
const V2_PRUNE_TIME_BUDGET_MS: u128 = 150;
const V2_SELECTION_POLICY_VERSION: &str = "semantic-doc-search-v2";
const V2_AUDIT_SNAPSHOT_VERSION: &str = "semantic-audit-snapshot-v1";
const V2_AUDIT_ROW_SET_ENCODING: &str = "sorted-positive-delta-varint-chunks-v1";
const V2_AUDIT_MAPPING_BATCH: usize = 8_192;
const V2_AUDIT_ROW_CHUNK_ROWS: usize = 1_024;
const V2_AUDIT_ARCHIVE_STEPS_PER_SLICE: usize = 4;
const V2_AUDIT_ARCHIVE_SLICE_TIME_BUDGET_MS: u128 = 150;

#[derive(Debug, Clone, Copy)]
struct SemanticResourceLimits {
    mapped_documents: i64,
    mappings: i64,
}

impl SemanticResourceLimits {
    const PRODUCTION: Self = Self {
        mapped_documents: V2_MAX_MAPPED_DOCUMENTS_PER_BUILD,
        mappings: V2_MAX_MAPPINGS_PER_BUILD,
    };
}

fn header_key(column: &ColumnMeta) -> String {
    format!(
        "{} {}",
        normalized_header_component(&column.sql_name),
        normalized_header_component(&column.original_name)
    )
}

fn normalized_header_component(value: &str) -> String {
    let characters = value.chars().collect::<Vec<_>>();
    let mut normalized = String::with_capacity(value.len() + 4);
    for (index, character) in characters.iter().copied().enumerate() {
        let previous = index
            .checked_sub(1)
            .and_then(|at| characters.get(at))
            .copied();
        let next = characters.get(index + 1).copied();
        let camel_boundary = character.is_uppercase()
            && previous.is_some_and(|value| value.is_lowercase() || value.is_ascii_digit());
        let acronym_boundary = character.is_uppercase()
            && previous.is_some_and(char::is_uppercase)
            && next.is_some_and(char::is_lowercase);
        if camel_boundary || acronym_boundary {
            normalized.push(' ');
        }
        if character.is_alphanumeric() {
            normalized.extend(character.to_lowercase());
        } else {
            normalized.push(' ');
        }
    }
    normalized.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn header_has_phrase(key: &str, phrase: &str) -> bool {
    let padded_key = format!(" {key} ");
    let padded_phrase = format!(" {phrase} ");
    padded_key.contains(&padded_phrase)
}

fn exact_only_header(column: &ColumnMeta) -> bool {
    exact_only_header_name(column)
        || matches!(
            column.inferred_type.to_ascii_lowercase().as_str(),
            "timestamp" | "ip" | "identifier" | "number" | "numeric"
        )
}

fn exact_only_header_name(column: &ColumnMeta) -> bool {
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
        "sid",
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
    .any(|phrase| header_has_phrase(&key, phrase))
        || header_has_phrase(&key, "id")
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
        "detail",
        "summary",
        "narrative",
        "evidence",
        "payload",
        "reason",
    ]
    .iter()
    .any(|phrase| header_has_phrase(&key, phrase))
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

struct TextChunk {
    text: String,
    truncated: bool,
}

struct WordChunks {
    chunks: Vec<TextChunk>,
    omitted: i64,
}

fn word_chunks(value: &str) -> WordChunks {
    let words = value.split_whitespace().collect::<Vec<_>>();
    if words.is_empty() {
        return WordChunks {
            chunks: Vec::new(),
            omitted: 0,
        };
    }
    let mut chunks = Vec::new();
    let mut omitted = 0i64;
    let mut start = 0usize;
    while start < words.len() {
        let end = (start + V2_PRIMARY_CHUNK_WORDS).min(words.len());
        if chunks.len() < V2_MAX_CHUNKS_PER_CELL {
            let unbounded = words[start..end].join(" ");
            let mut characters = unbounded.chars();
            let text = characters.by_ref().take(V2_MAX_DOCUMENT_CHARS).collect();
            chunks.push(TextChunk {
                text,
                truncated: characters.next().is_some(),
            });
        } else {
            omitted += 1;
        }
        if end == words.len() {
            break;
        }
        start = end.saturating_sub(V2_CHUNK_OVERLAP_WORDS);
    }
    WordChunks { chunks, omitted }
}

fn balanced_indices(length: usize, maximum: usize) -> Vec<usize> {
    if length <= maximum {
        return (0..length).collect();
    }
    if maximum == 0 {
        return Vec::new();
    }
    if maximum == 1 {
        return vec![length - 1];
    }
    (0..maximum)
        .map(|slot| slot * (length - 1) / (maximum - 1))
        .collect()
}

fn classify_columns(conn: &Connection, columns: &[ColumnMeta]) -> Result<Vec<ColumnPlan>> {
    if columns.is_empty() {
        return Ok(Vec::new());
    }
    #[derive(Default)]
    struct Stats {
        nonempty: usize,
        dynamic: usize,
        total_chars: usize,
        total_words: usize,
        distinct: HashSet<String>,
    }
    let identifiers = columns
        .iter()
        .map(|column| {
            format!(
                "substr({}, 1, {V2_MAX_CELL_INPUT_CHARS})",
                db::quote_ident(&column.sql_name)
            )
        })
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
            stat.total_chars += value.chars().count();
            stat.total_words += value.split_whitespace().count();
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
            let average_chars = stat.total_chars as f64 / stat.nonempty.max(1) as f64;
            let average_words = stat.total_words as f64 / stat.nonempty.max(1) as f64;
            let high_cardinality_entity = stat.nonempty >= 16
                && distinct_ratio >= 0.80
                && average_chars <= 128.0
                && average_words <= 3.0;
            // Strong free-text headers correct weak importer type inference (for example, a
            // Description column once misclassified as IP). Explicit identifier/timestamp/hash
            // names still win for ambiguous names such as ProcessId.
            let mode = if force_text_header(column) && !exact_only_header_name(column) {
                ColumnMode::Text
            } else if exact_only_header(column) {
                ColumnMode::ExactOnly
            } else if stat.nonempty >= 16 && (dynamic_ratio >= 0.80 || high_cardinality_entity) {
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

fn bounded_labelled_document(label: &str, chunk: &TextChunk) -> (String, bool) {
    let unbounded = format!("{label}: {}", chunk.text);
    let mut characters = unbounded.chars();
    let document = characters.by_ref().take(V2_MAX_DOCUMENT_CHARS).collect();
    (document, chunk.truncated || characters.next().is_some())
}

struct RowDocumentsV2 {
    documents: Vec<(&'static str, String, String)>,
    eligible_columns_omitted: i64,
    chunks_omitted: i64,
}

fn row_documents_with_stats_v2(plans: &[ColumnPlan], values: &[Option<String>]) -> RowDocumentsV2 {
    let mut cell_chunks: Vec<(ColumnMode, String, String, WordChunks)> = Vec::new();
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
        let bounded_value = value
            .chars()
            .take(V2_MAX_CELL_INPUT_CHARS)
            .collect::<String>();
        let normalized = normalize_text(&bounded_value);
        if !is_informative_text(&normalized) {
            continue;
        }
        let label = normalized_label(plan);
        let chunks = word_chunks(&normalized);
        if chunks.chunks.is_empty() {
            continue;
        }
        cell_chunks.push((plan.mode, plan.sql_name.clone(), label, chunks));
    }

    // Fairness: every eligible column contributes its first chunk before any early column may
    // contribute a second. Wide tables are sampled evenly across their full width, always keeping
    // the last eligible column; the lexical/structured query remains complete for omitted columns.
    let selected = balanced_indices(cell_chunks.len(), V2_MAX_PRIMARY_DOCUMENTS_PER_ROW);
    let eligible_columns_omitted = cell_chunks.len().saturating_sub(selected.len()) as i64;
    let mut chunks_omitted = selected
        .iter()
        .map(|index| cell_chunks[*index].3.omitted)
        .sum::<i64>();
    let mut documents = Vec::new();
    for index in &selected {
        let (mode, column_key, label, chunks) = &cell_chunks[*index];
        let (document, truncated) = bounded_labelled_document(label, &chunks.chunks[0]);
        chunks_omitted += i64::from(truncated);
        documents.push(("cell", column_key.clone(), document.clone()));
        if *mode == ColumnMode::Categorical {
            context_parts.push(document);
        }
    }
    let mut additional = 0usize;
    let available_additional = selected
        .iter()
        .map(|index| cell_chunks[*index].3.chunks.len().saturating_sub(1))
        .sum::<usize>();
    for round in 1..V2_MAX_CHUNKS_PER_CELL {
        for index in &selected {
            if additional == V2_MAX_ADDITIONAL_CHUNKS_PER_ROW
                || documents.len() == V2_MAX_DOCUMENTS_PER_ROW
            {
                break;
            }
            let (_, column_key, label, chunks) = &cell_chunks[*index];
            if let Some(chunk) = chunks.chunks.get(round) {
                let (document, truncated) = bounded_labelled_document(label, chunk);
                chunks_omitted += i64::from(truncated);
                documents.push(("cell_chunk", column_key.clone(), document));
                additional += 1;
            }
        }
    }
    chunks_omitted += available_additional.saturating_sub(additional) as i64;
    if context_parts.len() >= 2 && documents.len() < V2_MAX_DOCUMENTS_PER_ROW {
        let context = context_parts.join("; ");
        let mut characters = context.chars();
        let bounded = characters.by_ref().take(V2_MAX_DOCUMENT_CHARS).collect();
        chunks_omitted += i64::from(characters.next().is_some());
        documents.push(("row_context", String::new(), bounded));
    }
    debug_assert!(documents.len() <= V2_MAX_DOCUMENTS_PER_ROW);
    RowDocumentsV2 {
        documents,
        eligible_columns_omitted,
        chunks_omitted,
    }
}

fn row_documents_v2(
    plans: &[ColumnPlan],
    values: &[Option<String>],
) -> Vec<(&'static str, String, String)> {
    row_documents_with_stats_v2(plans, values).documents
}

fn text_sha256(kind: &str, column_key: &str, text_value: &str) -> String {
    let mut hasher = Sha256::new();
    for value in [kind, column_key, text_value] {
        hasher.update((value.len() as u64).to_le_bytes());
        hasher.update(value.as_bytes());
    }
    bytes_to_hex(&hasher.finalize())
}

#[cfg(test)]
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

#[derive(Debug)]
struct BudgetedDocuments {
    documents: BTreeMap<String, NormalizedDocument>,
    documents_seen: i64,
    documents_mapped: i64,
    documents_skipped: i64,
    mappings_skipped: i64,
    columns_omitted: i64,
    chunks_omitted: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SemanticResourcePolicy {
    documents_over_limit: bool,
    mappings_over_limit: bool,
}

fn cumulative_balanced_budget(rows_processed: i64, rows_total: i64, limit: i64) -> i64 {
    if rows_processed <= 0 || rows_total <= 0 || limit <= 0 {
        return 0;
    }
    let processed = rows_processed.min(rows_total) as i128;
    ((processed * limit as i128) / rows_total as i128) as i64
}

/// Applies build-wide limits as cumulative targets instead of consuming them from the first rows.
/// The targets advance uniformly across the full source-row count and reach their final increment
/// on the final row. Within a row, balanced selection likewise keeps the final eligible document.
fn budget_documents_v2(
    plans: &[ColumnPlan],
    source_rows: &[(i64, Vec<Option<String>>)],
    known_build_documents: &HashMap<String, i64>,
    rows_before: i64,
    mapped_documents_before: i64,
    mappings_before: i64,
    rows_total: i64,
    limits: SemanticResourceLimits,
    policy: SemanticResourcePolicy,
) -> BudgetedDocuments {
    let mut retained = BTreeMap::<String, NormalizedDocument>::new();
    let mut newly_known = HashSet::<String>::new();
    let mut documents_seen = 0i64;
    let mut documents_mapped = 0i64;
    let mut documents_skipped = 0i64;
    let mut mappings_retained = 0i64;
    let mut mappings_skipped = 0i64;
    let mut columns_omitted = 0i64;
    let mut chunks_omitted = 0i64;

    for (offset, (row_num, values)) in source_rows.iter().enumerate() {
        let mut row_hashes = HashSet::<String>::new();
        let mut candidates = Vec::new();
        let row_documents = row_documents_with_stats_v2(plans, values);
        columns_omitted += row_documents.eligible_columns_omitted;
        chunks_omitted += row_documents.chunks_omitted;
        for (kind, column_key, text) in row_documents.documents {
            let hash = text_sha256(kind, &column_key, &text);
            if row_hashes.insert(hash.clone()) {
                candidates.push((hash, kind, column_key, text));
            }
        }
        documents_seen += candidates.len() as i64;

        let row_position = rows_before + offset as i64 + 1;
        let mapping_allowance = if policy.mappings_over_limit {
            let mapping_target =
                cumulative_balanced_budget(row_position, rows_total, limits.mappings.max(0));
            mapping_target
                .saturating_sub(mappings_before + mappings_retained)
                .min(candidates.len() as i64)
                .max(0) as usize
        } else {
            candidates.len()
        };
        let mapping_indices = balanced_indices(candidates.len(), mapping_allowance);
        mappings_skipped += candidates.len().saturating_sub(mapping_indices.len()) as i64;

        let unknown_indices = mapping_indices
            .iter()
            .copied()
            .filter(|index| {
                !known_build_documents.contains_key(&candidates[*index].0)
                    && !newly_known.contains(&candidates[*index].0)
            })
            .collect::<Vec<_>>();
        let admitted_unknown = if policy.documents_over_limit {
            let document_target = cumulative_balanced_budget(
                row_position,
                rows_total,
                limits.mapped_documents.max(0),
            );
            let document_allowance = document_target
                .saturating_sub(mapped_documents_before + documents_mapped)
                .min(unknown_indices.len() as i64)
                .max(0) as usize;
            balanced_indices(unknown_indices.len(), document_allowance)
                .into_iter()
                .map(|index| unknown_indices[index])
                .collect::<HashSet<_>>()
        } else {
            unknown_indices.iter().copied().collect::<HashSet<_>>()
        };

        for index in mapping_indices {
            let (hash, kind, column_key, text) = &candidates[index];
            let already_known =
                known_build_documents.contains_key(hash) || newly_known.contains(hash);
            if !already_known && !admitted_unknown.contains(&index) {
                documents_skipped += 1;
                mappings_skipped += 1;
                continue;
            }
            if !already_known && newly_known.insert(hash.clone()) {
                documents_mapped += 1;
            }
            mappings_retained += 1;
            retained
                .entry(hash.clone())
                .or_insert_with(|| NormalizedDocument {
                    kind,
                    column_key: column_key.clone(),
                    text: text.clone(),
                    rows: BTreeSet::new(),
                })
                .rows
                .insert(*row_num);
        }
    }

    debug_assert!(
        !policy.documents_over_limit
            || mapped_documents_before + documents_mapped <= limits.mapped_documents.max(0)
    );
    debug_assert!(
        !policy.mappings_over_limit
            || mappings_before + mappings_retained <= limits.mappings.max(0)
    );
    BudgetedDocuments {
        documents: retained,
        documents_seen,
        documents_mapped,
        documents_skipped,
        mappings_skipped,
        columns_omitted,
        chunks_omitted,
    }
}

struct BoundedSourceBatch {
    rows: Vec<(i64, Vec<Option<String>>)>,
    cells_truncated: i64,
}

fn source_select_expressions(columns: &[ColumnMeta]) -> String {
    columns
        .iter()
        .map(|column| {
            let identifier = db::quote_ident(&column.sql_name);
            format!("substr({identifier}, 1, {})", V2_MAX_CELL_INPUT_CHARS + 1)
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn load_bounded_source_batch(
    conn: &Connection,
    columns: &[ColumnMeta],
    source_expressions: &str,
    cursor: i64,
) -> Result<BoundedSourceBatch> {
    let sql = format!(
        "SELECT row_num, {source_expressions} FROM rows
         WHERE row_num > ?1 ORDER BY row_num LIMIT {V2_SOURCE_BATCH_ROWS}"
    );
    let mut statement = conn.prepare(&sql)?;
    let mut rows = statement.query([cursor])?;
    let mut batch = Vec::with_capacity(V2_SOURCE_BATCH_ROWS);
    let mut cells_truncated = 0i64;
    while let Some(row) = rows.next()? {
        let row_num: i64 = row.get(0)?;
        let values = (0..columns.len())
            .map(|index| {
                let value = row.get::<_, Option<String>>(index + 1)?;
                Ok(value.map(|value| {
                    let mut characters = value.chars();
                    let bounded = characters
                        .by_ref()
                        .take(V2_MAX_CELL_INPUT_CHARS)
                        .collect::<String>();
                    if characters.next().is_some() {
                        cells_truncated += 1;
                    }
                    bounded
                }))
            })
            .collect::<rusqlite::Result<Vec<_>>>()?;
        batch.push((row_num, values));
    }
    Ok(BoundedSourceBatch {
        rows: batch,
        cells_truncated,
    })
}

fn determine_semantic_resource_policy<C>(
    conn: &Connection,
    columns: &[ColumnMeta],
    plans: &[ColumnPlan],
    build_id: i64,
    limits: SemanticResourceLimits,
    is_cancelled: &C,
) -> Result<Option<SemanticResourcePolicy>>
where
    C: Fn() -> bool,
{
    let stored = conn.query_row(
        "SELECT candidate_documents, candidate_mappings, candidate_document_limit,
                candidate_mapping_limit
         FROM _semantic_v2_build WHERE build_id = ?1",
        [build_id],
        |row| {
            Ok((
                row.get::<_, Option<i64>>(0)?,
                row.get::<_, Option<i64>>(1)?,
                row.get::<_, Option<i64>>(2)?,
                row.get::<_, Option<i64>>(3)?,
            ))
        },
    )?;
    if let (Some(documents), Some(mappings), Some(document_limit), Some(mapping_limit)) = stored {
        if document_limit == limits.mapped_documents && mapping_limit == limits.mappings {
            return Ok(Some(SemanticResourcePolicy {
                documents_over_limit: documents > document_limit,
                mappings_over_limit: mappings > mapping_limit,
            }));
        }
    }

    let source_expressions = source_select_expressions(columns);
    let mut cursor = 0i64;
    let mut distinct_documents = HashSet::<String>::new();
    let mut document_count = 0i64;
    let mut mapping_count = 0i64;
    let document_sentinel = limits.mapped_documents.max(0).saturating_add(1);
    let mapping_sentinel = limits.mappings.max(0).saturating_add(1);
    loop {
        if is_cancelled() {
            return Ok(None);
        }
        let batch = load_bounded_source_batch(conn, columns, &source_expressions, cursor)?;
        if batch.rows.is_empty() {
            break;
        }
        for (row_num, values) in &batch.rows {
            cursor = *row_num;
            let mut row_hashes = HashSet::<String>::new();
            for (kind, column_key, text) in row_documents_v2(plans, values) {
                let hash = text_sha256(kind, &column_key, &text);
                if !row_hashes.insert(hash.clone()) {
                    continue;
                }
                if mapping_count < mapping_sentinel {
                    mapping_count += 1;
                }
                if document_count < document_sentinel && distinct_documents.insert(hash) {
                    document_count += 1;
                }
            }
        }
        if document_count >= document_sentinel && mapping_count >= mapping_sentinel {
            break;
        }
    }
    if is_cancelled() {
        return Ok(None);
    }
    conn.execute(
        "UPDATE _semantic_v2_build SET candidate_documents = ?2, candidate_mappings = ?3,
                candidate_document_limit = ?4, candidate_mapping_limit = ?5,
                updated_at = ?6
         WHERE build_id = ?1",
        params![
            build_id,
            document_count,
            mapping_count,
            limits.mapped_documents,
            limits.mappings,
            chrono::Utc::now().to_rfc3339(),
        ],
    )?;
    Ok(Some(SemanticResourcePolicy {
        documents_over_limit: document_count > limits.mapped_documents.max(0),
        mappings_over_limit: mapping_count > limits.mappings.max(0),
    }))
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticIndexSummary {
    pub rows_indexed: i64,
    pub documents_indexed: i64,
    pub documents_mapped: i64,
    pub documents_skipped: i64,
    pub mappings_written: i64,
    pub mappings_skipped: i64,
    pub cells_truncated: i64,
    pub columns_omitted: i64,
    pub chunks_omitted: i64,
    pub truncated: bool,
    pub warnings: Vec<String>,
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
    pub documents_skipped: i64,
    pub mappings_skipped: i64,
    pub cells_truncated: i64,
    pub columns_omitted: i64,
    pub chunks_omitted: i64,
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
    pub index_documents_skipped: i64,
    pub index_mappings_skipped: i64,
    pub index_cells_truncated: i64,
    pub index_columns_omitted: i64,
    pub index_chunks_omitted: i64,
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
            index_version TEXT NOT NULL DEFAULT 'legacy-unrecorded',
            model_name TEXT NOT NULL DEFAULT 'legacy-unrecorded',
            model_version TEXT NOT NULL DEFAULT 'legacy-unrecorded',
            model_sha256 TEXT NOT NULL,
            tokenizer_sha256 TEXT NOT NULL DEFAULT 'legacy-unrecorded',
            config_sha256 TEXT NOT NULL DEFAULT 'legacy-unrecorded',
            normalizer_version TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('building','paused','ready','cancelled','failed')),
            worker_token TEXT,
            source_rows INTEGER NOT NULL,
            cursor_row_num INTEGER NOT NULL DEFAULT 0,
            rows_scanned INTEGER NOT NULL DEFAULT 0,
            documents_seen INTEGER NOT NULL DEFAULT 0,
            documents_embedded INTEGER NOT NULL DEFAULT 0,
            documents_mapped INTEGER NOT NULL DEFAULT 0,
            documents_skipped INTEGER NOT NULL DEFAULT 0,
            mappings_written INTEGER NOT NULL DEFAULT 0,
            mappings_skipped INTEGER NOT NULL DEFAULT 0,
            cells_truncated INTEGER NOT NULL DEFAULT 0,
            columns_omitted INTEGER NOT NULL DEFAULT 0,
            chunks_omitted INTEGER NOT NULL DEFAULT 0,
            candidate_documents INTEGER,
            candidate_mappings INTEGER,
            candidate_document_limit INTEGER,
            candidate_mapping_limit INTEGER,
            started_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            completed_at TEXT,
            error TEXT
         );
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
            tokenizer_sha256 TEXT NOT NULL DEFAULT 'legacy-unrecorded',
            config_sha256 TEXT NOT NULL DEFAULT 'legacy-unrecorded',
            normalizer_version TEXT NOT NULL,
            kind TEXT NOT NULL,
            column_key TEXT NOT NULL,
            text_sha256 TEXT NOT NULL,
            normalized_text TEXT NOT NULL,
            embedding BLOB NOT NULL,
            UNIQUE(
                model_sha256, tokenizer_sha256, config_sha256, normalizer_version, text_sha256
            )
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
            maximum_documents INTEGER NOT NULL,
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
         ) WITHOUT ROWID;
         CREATE TABLE IF NOT EXISTS _semantic_v2_audit_snapshot (
            selection_id TEXT PRIMARY KEY,
            snapshot_version TEXT NOT NULL,
            build_id INTEGER NOT NULL,
            dataset_hash TEXT NOT NULL,
            schema_hash TEXT NOT NULL,
            index_version TEXT NOT NULL,
            normalizer_version TEXT NOT NULL,
            model_name TEXT NOT NULL,
            model_version TEXT NOT NULL,
            model_sha256 TEXT NOT NULL,
            tokenizer_sha256 TEXT NOT NULL,
            config_sha256 TEXT NOT NULL,
            query_sha256 TEXT NOT NULL,
            policy_version TEXT NOT NULL,
            minimum_score REAL NOT NULL,
            maximum_documents INTEGER NOT NULL,
            documents_above_threshold INTEGER NOT NULL,
            documents_retained INTEGER NOT NULL,
            rows_matched INTEGER NOT NULL,
            documents_truncated INTEGER NOT NULL,
            broad_row_warning INTEGER NOT NULL,
            warnings_json TEXT NOT NULL,
            source_rows INTEGER NOT NULL,
            index_rows_scanned INTEGER NOT NULL,
            index_documents_seen INTEGER NOT NULL,
            index_documents_embedded INTEGER NOT NULL,
            index_documents_mapped INTEGER NOT NULL,
            index_mappings_written INTEGER NOT NULL,
            index_documents_skipped INTEGER NOT NULL,
            index_mappings_skipped INTEGER NOT NULL,
            index_cells_truncated INTEGER NOT NULL,
            index_columns_omitted INTEGER NOT NULL,
            index_chunks_omitted INTEGER NOT NULL,
            candidate_documents INTEGER NOT NULL,
            candidate_mappings INTEGER NOT NULL,
            candidate_document_limit INTEGER NOT NULL,
            candidate_mapping_limit INTEGER NOT NULL,
            selected_document_count INTEGER NOT NULL,
            mapping_count INTEGER NOT NULL,
            mapping_sha256 TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            row_set_sha256 TEXT NOT NULL,
            row_set_encoding TEXT NOT NULL,
            selection_created_at TEXT NOT NULL,
            archived_at TEXT NOT NULL
         ) WITHOUT ROWID;
         CREATE TABLE IF NOT EXISTS _semantic_v2_audit_snapshot_document (
            selection_id TEXT NOT NULL,
            rank INTEGER NOT NULL,
            source_doc_id INTEGER NOT NULL,
            fingerprint_sha256 TEXT NOT NULL,
            kind TEXT NOT NULL,
            column_key TEXT NOT NULL,
            normalized_text TEXT NOT NULL,
            cosine_score REAL NOT NULL,
            rank_score REAL NOT NULL,
            mapping_count INTEGER NOT NULL,
            mapping_sha256 TEXT NOT NULL,
            PRIMARY KEY(selection_id, rank),
            UNIQUE(selection_id, fingerprint_sha256)
         ) WITHOUT ROWID;
         CREATE TABLE IF NOT EXISTS _semantic_v2_audit_snapshot_row_chunk (
            selection_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            first_row_num INTEGER NOT NULL,
            last_row_num INTEGER NOT NULL,
            row_count INTEGER NOT NULL,
            encoded_rows BLOB NOT NULL,
            chunk_sha256 TEXT NOT NULL,
            PRIMARY KEY(selection_id, chunk_index)
         ) WITHOUT ROWID;
         CREATE TRIGGER IF NOT EXISTS _semantic_v2_audit_snapshot_no_update
         BEFORE UPDATE ON _semantic_v2_audit_snapshot
         BEGIN
            SELECT RAISE(ABORT, 'semantic audit snapshots are immutable');
         END;
         CREATE TRIGGER IF NOT EXISTS _semantic_v2_audit_snapshot_no_delete
         BEFORE DELETE ON _semantic_v2_audit_snapshot
         BEGIN
            SELECT RAISE(ABORT, 'semantic audit snapshots are immutable');
         END;
         CREATE TRIGGER IF NOT EXISTS _semantic_v2_audit_snapshot_document_no_update
         BEFORE UPDATE ON _semantic_v2_audit_snapshot_document
         BEGIN
            SELECT RAISE(ABORT, 'semantic audit snapshots are immutable');
         END;
         CREATE TRIGGER IF NOT EXISTS _semantic_v2_audit_snapshot_document_no_delete
         BEFORE DELETE ON _semantic_v2_audit_snapshot_document
         BEGIN
            SELECT RAISE(ABORT, 'semantic audit snapshots are immutable');
         END;
         CREATE TRIGGER IF NOT EXISTS _semantic_v2_audit_snapshot_row_no_update
         BEFORE UPDATE ON _semantic_v2_audit_snapshot_row_chunk
         BEGIN
            SELECT RAISE(ABORT, 'semantic audit snapshots are immutable');
         END;
         CREATE TRIGGER IF NOT EXISTS _semantic_v2_audit_snapshot_row_no_delete
         BEFORE DELETE ON _semantic_v2_audit_snapshot_row_chunk
         BEGIN
            SELECT RAISE(ABORT, 'semantic audit snapshots are immutable');
         END;
         CREATE TABLE IF NOT EXISTS _semantic_v2_audit_snapshot_stage (
            selection_id TEXT PRIMARY KEY,
            build_id INTEGER NOT NULL,
            phase TEXT NOT NULL CHECK(phase IN ('mappings','rows')),
            cursor_doc_id INTEGER NOT NULL DEFAULT 0,
            cursor_row_num INTEGER NOT NULL DEFAULT 0,
            next_mapping_chunk INTEGER NOT NULL DEFAULT 0,
            next_row_chunk INTEGER NOT NULL DEFAULT 0,
            mappings_seen INTEGER NOT NULL DEFAULT 0,
            rows_seen INTEGER NOT NULL DEFAULT 0,
            started_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
         ) WITHOUT ROWID;
         CREATE TABLE IF NOT EXISTS _semantic_v2_audit_snapshot_stage_document (
            selection_id TEXT NOT NULL,
            doc_id INTEGER NOT NULL,
            rank INTEGER NOT NULL,
            fingerprint_sha256 TEXT NOT NULL,
            kind TEXT NOT NULL,
            column_key TEXT NOT NULL,
            normalized_text TEXT NOT NULL,
            cosine_score REAL NOT NULL,
            rank_score REAL NOT NULL,
            PRIMARY KEY(selection_id, doc_id),
            UNIQUE(selection_id, rank)
         ) WITHOUT ROWID;
         CREATE TABLE IF NOT EXISTS _semantic_v2_audit_snapshot_stage_mapping_chunk (
            selection_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            doc_id INTEGER NOT NULL,
            first_row_num INTEGER NOT NULL,
            last_row_num INTEGER NOT NULL,
            row_count INTEGER NOT NULL,
            encoded_rows BLOB NOT NULL,
            chunk_sha256 TEXT NOT NULL,
            PRIMARY KEY(selection_id, chunk_index)
         ) WITHOUT ROWID;
         CREATE TABLE IF NOT EXISTS _semantic_v2_audit_snapshot_stage_row (
            selection_id TEXT NOT NULL,
            row_num INTEGER NOT NULL,
            PRIMARY KEY(selection_id, row_num)
         ) WITHOUT ROWID;
         CREATE TABLE IF NOT EXISTS _semantic_v2_audit_snapshot_stage_row_chunk (
            selection_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            first_row_num INTEGER NOT NULL,
            last_row_num INTEGER NOT NULL,
            row_count INTEGER NOT NULL,
            encoded_rows BLOB NOT NULL,
            chunk_sha256 TEXT NOT NULL,
            PRIMARY KEY(selection_id, chunk_index)
         ) WITHOUT ROWID;
         CREATE TRIGGER IF NOT EXISTS _semantic_v2_audit_snapshot_document_no_insert
         BEFORE INSERT ON _semantic_v2_audit_snapshot_document
         WHEN NOT EXISTS (
                SELECT 1 FROM _semantic_v2_audit_snapshot p
                WHERE p.selection_id = NEW.selection_id
              )
           OR NOT EXISTS (
                SELECT 1
                FROM _semantic_v2_audit_snapshot_stage st
                JOIN _semantic_v2_audit_snapshot_stage_document sd
                  ON sd.selection_id = st.selection_id
                WHERE st.selection_id = NEW.selection_id
                  AND sd.rank = NEW.rank
                  AND sd.doc_id = NEW.source_doc_id
                  AND sd.fingerprint_sha256 = NEW.fingerprint_sha256
                  AND sd.kind = NEW.kind
                  AND sd.column_key = NEW.column_key
                  AND sd.normalized_text = NEW.normalized_text
                  AND sd.cosine_score = NEW.cosine_score
                  AND sd.rank_score = NEW.rank_score
              )
         BEGIN
            SELECT RAISE(ABORT, 'semantic audit snapshot children are immutable');
         END;
         CREATE TRIGGER IF NOT EXISTS _semantic_v2_audit_snapshot_row_no_insert
         BEFORE INSERT ON _semantic_v2_audit_snapshot_row_chunk
         WHEN NOT EXISTS (
                SELECT 1 FROM _semantic_v2_audit_snapshot p
                WHERE p.selection_id = NEW.selection_id
              )
           OR NOT EXISTS (
                SELECT 1
                FROM _semantic_v2_audit_snapshot_stage st
                JOIN _semantic_v2_audit_snapshot_stage_row_chunk sr
                  ON sr.selection_id = st.selection_id
                WHERE st.selection_id = NEW.selection_id
                  AND sr.chunk_index = NEW.chunk_index
                  AND sr.first_row_num = NEW.first_row_num
                  AND sr.last_row_num = NEW.last_row_num
                  AND sr.row_count = NEW.row_count
                  AND sr.encoded_rows = NEW.encoded_rows
                  AND sr.chunk_sha256 = NEW.chunk_sha256
              )
         BEGIN
            SELECT RAISE(ABORT, 'semantic audit snapshot children are immutable');
         END;",
    )?;
    ensure_semantic_v2_build_columns(conn)?;
    ensure_semantic_v2_document_identity_schema(conn)
}

fn semantic_v2_build_has_column(conn: &Connection, expected: &str) -> rusqlite::Result<bool> {
    sqlite_table_has_columns(conn, "_semantic_v2_build", &[expected])
}

fn sqlite_table_has_columns(
    conn: &Connection,
    table_name: &str,
    expected: &[&str],
) -> rusqlite::Result<bool> {
    let mut statement = conn.prepare(&format!(
        "PRAGMA table_info({})",
        db::quote_ident(table_name)
    ))?;
    let columns = statement
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<rusqlite::Result<HashSet<_>>>()?;
    Ok(expected.iter().all(|column| columns.contains(*column)))
}

fn ensure_semantic_v2_build_columns(conn: &Connection) -> rusqlite::Result<()> {
    for (name, definition) in [
        ("index_version", "TEXT NOT NULL DEFAULT 'legacy-unrecorded'"),
        ("model_name", "TEXT NOT NULL DEFAULT 'legacy-unrecorded'"),
        ("model_version", "TEXT NOT NULL DEFAULT 'legacy-unrecorded'"),
        (
            "tokenizer_sha256",
            "TEXT NOT NULL DEFAULT 'legacy-unrecorded'",
        ),
        ("config_sha256", "TEXT NOT NULL DEFAULT 'legacy-unrecorded'"),
        ("worker_token", "TEXT"),
        ("documents_mapped", "INTEGER NOT NULL DEFAULT 0"),
        ("documents_skipped", "INTEGER NOT NULL DEFAULT 0"),
        ("mappings_skipped", "INTEGER NOT NULL DEFAULT 0"),
        ("cells_truncated", "INTEGER NOT NULL DEFAULT 0"),
        ("columns_omitted", "INTEGER NOT NULL DEFAULT 0"),
        ("chunks_omitted", "INTEGER NOT NULL DEFAULT 0"),
        ("candidate_documents", "INTEGER"),
        ("candidate_mappings", "INTEGER"),
        ("candidate_document_limit", "INTEGER"),
        ("candidate_mapping_limit", "INTEGER"),
    ] {
        if semantic_v2_build_has_column(conn, name)? {
            continue;
        }
        let sql = format!(
            "ALTER TABLE _semantic_v2_build ADD COLUMN {} {definition}",
            db::quote_ident(name)
        );
        match conn.execute(&sql, []) {
            Ok(_) => {}
            Err(_) if semantic_v2_build_has_column(conn, name)? => {}
            Err(error) => return Err(error),
        }
    }
    ensure_semantic_v2_build_identity_indexes(conn)?;
    ensure_semantic_v2_selection_columns(conn)
}

fn ensure_semantic_v2_build_identity_indexes(conn: &Connection) -> rusqlite::Result<()> {
    const IDENTITY_COLUMNS: &[&str] = &[
        "dataset_hash",
        "schema_hash",
        "index_version",
        "normalizer_version",
        "model_name",
        "model_version",
        "model_sha256",
        "tokenizer_sha256",
        "config_sha256",
    ];
    let mut lookup_columns = IDENTITY_COLUMNS.to_vec();
    lookup_columns.push("status");
    if sqlite_index_columns_match(
        sqlite_index_columns(conn, "_semantic_v2_build_identity")?,
        &lookup_columns,
    ) && sqlite_index_columns_match(
        sqlite_index_columns(conn, "_semantic_v2_build_unique_identity")?,
        IDENTITY_COLUMNS,
    ) {
        return Ok(());
    }
    // The original identity predated persisted tokenizer/configuration metadata. Replacing the
    // indexes permits a current build to coexist with a migrated legacy row while ensuring that
    // no legacy-unrecorded build can win a current cache/resume lookup.
    let result = conn.execute_batch(
        "SAVEPOINT semantic_v2_build_identity_index_migration;
         DROP INDEX IF EXISTS _semantic_v2_build_unique_identity;
         DROP INDEX IF EXISTS _semantic_v2_build_identity;
         CREATE INDEX _semantic_v2_build_identity ON _semantic_v2_build(
            dataset_hash, schema_hash, index_version, normalizer_version, model_name,
            model_version, model_sha256, tokenizer_sha256, config_sha256, status
         );
         CREATE UNIQUE INDEX _semantic_v2_build_unique_identity ON _semantic_v2_build(
            dataset_hash, schema_hash, index_version, normalizer_version, model_name,
            model_version, model_sha256, tokenizer_sha256, config_sha256
         );
         RELEASE semantic_v2_build_identity_index_migration;",
    );
    if result.is_err() {
        let _ = conn.execute_batch(
            "ROLLBACK TO semantic_v2_build_identity_index_migration;
             RELEASE semantic_v2_build_identity_index_migration;",
        );
    }
    result
}

fn sqlite_index_columns(
    conn: &Connection,
    index_name: &str,
) -> rusqlite::Result<Option<Vec<String>>> {
    let exists: bool = conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = ?1)",
        [index_name],
        |row| row.get(0),
    )?;
    if !exists {
        return Ok(None);
    }
    let mut statement = conn.prepare(&format!(
        "PRAGMA index_info({})",
        db::quote_ident(index_name)
    ))?;
    let columns = statement
        .query_map([], |row| row.get::<_, String>(2))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(Some(columns))
}

fn sqlite_index_columns_match(actual: Option<Vec<String>>, expected: &[&str]) -> bool {
    actual.is_some_and(|columns| {
        columns
            .iter()
            .map(String::as_str)
            .eq(expected.iter().copied())
    })
}

fn ensure_semantic_v2_document_identity_schema(conn: &Connection) -> rusqlite::Result<()> {
    const IDENTITY_COLUMNS: &[&str] = &[
        "model_sha256",
        "tokenizer_sha256",
        "config_sha256",
        "normalizer_version",
        "text_sha256",
    ];
    let tokenizer_recorded = semantic_v2_document_has_column(conn, "tokenizer_sha256")?;
    let config_recorded = semantic_v2_document_has_column(conn, "config_sha256")?;
    if tokenizer_recorded
        && config_recorded
        && semantic_v2_document_has_unique_columns(conn, IDENTITY_COLUMNS)?
    {
        return Ok(());
    }

    let tokenizer_source = if tokenizer_recorded {
        db::quote_ident("tokenizer_sha256")
    } else {
        format!("'{}'", V2_LEGACY_UNRECORDED_IDENTITY)
    };
    let config_source = if config_recorded {
        db::quote_ident("config_sha256")
    } else {
        format!("'{}'", V2_LEGACY_UNRECORDED_IDENTITY)
    };
    let sql = format!(
        "SAVEPOINT semantic_v2_document_identity_migration;
         DROP TABLE IF EXISTS _semantic_v2_document_identity_migration;
         CREATE TABLE _semantic_v2_document_identity_migration (
            doc_id INTEGER PRIMARY KEY,
            model_sha256 TEXT NOT NULL,
            tokenizer_sha256 TEXT NOT NULL DEFAULT 'legacy-unrecorded',
            config_sha256 TEXT NOT NULL DEFAULT 'legacy-unrecorded',
            normalizer_version TEXT NOT NULL,
            kind TEXT NOT NULL,
            column_key TEXT NOT NULL,
            text_sha256 TEXT NOT NULL,
            normalized_text TEXT NOT NULL,
            embedding BLOB NOT NULL,
            UNIQUE(
                model_sha256, tokenizer_sha256, config_sha256, normalizer_version, text_sha256
            )
         );
         INSERT INTO _semantic_v2_document_identity_migration (
            doc_id, model_sha256, tokenizer_sha256, config_sha256, normalizer_version,
            kind, column_key, text_sha256, normalized_text, embedding
         )
         SELECT doc_id, model_sha256, {tokenizer_source}, {config_source}, normalizer_version,
                kind, column_key, text_sha256, normalized_text, embedding
         FROM _semantic_v2_document;
         DROP TABLE _semantic_v2_document;
         ALTER TABLE _semantic_v2_document_identity_migration RENAME TO _semantic_v2_document;
         RELEASE semantic_v2_document_identity_migration;"
    );
    let result = conn.execute_batch(&sql);
    if result.is_err() {
        let _ = conn.execute_batch(
            "ROLLBACK TO semantic_v2_document_identity_migration;
             RELEASE semantic_v2_document_identity_migration;",
        );
    }
    result
}

fn semantic_v2_document_has_column(conn: &Connection, expected: &str) -> rusqlite::Result<bool> {
    let mut statement = conn.prepare("PRAGMA table_info(_semantic_v2_document)")?;
    let mut rows = statement.query([])?;
    while let Some(row) = rows.next()? {
        if row.get::<_, String>(1)? == expected {
            return Ok(true);
        }
    }
    Ok(false)
}

fn semantic_v2_document_has_unique_columns(
    conn: &Connection,
    expected: &[&str],
) -> rusqlite::Result<bool> {
    let indexes = {
        let mut statement = conn.prepare("PRAGMA index_list(_semantic_v2_document)")?;
        let collected = statement
            .query_map([], |row| {
                Ok((row.get::<_, String>(1)?, row.get::<_, bool>(2)?))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        collected
    };
    for (name, unique) in indexes {
        if unique && sqlite_index_columns_match(sqlite_index_columns(conn, &name)?, expected) {
            return Ok(true);
        }
    }
    Ok(false)
}

fn semantic_v2_selection_has_column(conn: &Connection, expected: &str) -> rusqlite::Result<bool> {
    let mut statement = conn.prepare("PRAGMA table_info(_semantic_v2_selection)")?;
    let mut rows = statement.query([])?;
    while let Some(row) = rows.next()? {
        if row.get::<_, String>(1)? == expected {
            return Ok(true);
        }
    }
    Ok(false)
}

fn ensure_semantic_v2_selection_columns(conn: &Connection) -> rusqlite::Result<()> {
    if semantic_v2_selection_has_column(conn, "maximum_documents")? {
        return Ok(());
    }
    match conn.execute(
        "ALTER TABLE _semantic_v2_selection ADD COLUMN maximum_documents INTEGER NOT NULL DEFAULT 256",
        [],
    ) {
        Ok(_) => Ok(()),
        Err(_) if semantic_v2_selection_has_column(conn, "maximum_documents")? => Ok(()),
        Err(error) => Err(error),
    }
}

#[derive(Debug, Clone)]
struct AuditSnapshotStage {
    selection_id: String,
    build_id: i64,
    phase: String,
    cursor_doc_id: i64,
    cursor_row_num: i64,
    next_mapping_chunk: i64,
    next_row_chunk: i64,
    mappings_seen: i64,
    rows_seen: i64,
}

#[derive(Debug, Clone)]
struct AuditSnapshotDocument {
    doc_id: i64,
    rank: i64,
    fingerprint_sha256: String,
    kind: String,
    column_key: String,
    normalized_text: String,
    cosine_score: f64,
    rank_score: f64,
}

#[derive(Debug)]
struct AuditSnapshotSource {
    build_id: i64,
    dataset_hash: String,
    schema_hash: String,
    index_version: String,
    model_name: String,
    model_version: String,
    model_sha256: String,
    tokenizer_sha256: String,
    config_sha256: String,
    normalizer_version: String,
    query_sha256: String,
    policy_version: String,
    minimum_score: f64,
    maximum_documents: i64,
    documents_above_threshold: i64,
    documents_retained: i64,
    rows_matched: i64,
    documents_truncated: i64,
    broad_row_warning: i64,
    warnings_json: String,
    source_rows: i64,
    rows_scanned: i64,
    documents_seen: i64,
    documents_embedded: i64,
    documents_mapped: i64,
    mappings_written: i64,
    documents_skipped: i64,
    mappings_skipped: i64,
    cells_truncated: i64,
    columns_omitted: i64,
    chunks_omitted: i64,
    candidate_documents: i64,
    candidate_mappings: i64,
    candidate_document_limit: i64,
    candidate_mapping_limit: i64,
    selection_created_at: String,
}

fn update_snapshot_digest(hasher: &mut Sha256, value: &[u8]) {
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value);
}

fn encode_sorted_positive_delta_varints(row_numbers: &[i64]) -> Result<Vec<u8>> {
    let mut output = Vec::with_capacity(row_numbers.len().saturating_mul(2));
    let mut previous = 0u64;
    for row_num in row_numbers {
        let current = u64::try_from(*row_num)
            .map_err(|_| anyhow::anyhow!("semantic audit row numbers must be positive"))?;
        if current == 0 || current <= previous {
            bail!("semantic audit row numbers must be strictly increasing and positive");
        }
        let mut delta = current - previous;
        loop {
            let mut byte = (delta & 0x7f) as u8;
            delta >>= 7;
            if delta != 0 {
                byte |= 0x80;
            }
            output.push(byte);
            if delta == 0 {
                break;
            }
        }
        previous = current;
    }
    Ok(output)
}

#[cfg(test)]
fn decode_sorted_positive_delta_varints(encoded: &[u8]) -> Result<Vec<i64>> {
    let mut output = Vec::new();
    let mut previous = 0u64;
    let mut value = 0u64;
    let mut shift = 0u32;
    for byte in encoded {
        let payload = u64::from(byte & 0x7f);
        if shift >= 64 || (shift == 63 && payload > 1) {
            bail!("semantic audit row-set varint overflow");
        }
        value |= payload << shift;
        if byte & 0x80 != 0 {
            shift += 7;
            if shift >= 64 {
                bail!("semantic audit row-set varint overflow");
            }
            continue;
        }
        if value == 0 {
            bail!("semantic audit row-set contains a zero delta");
        }
        previous = previous
            .checked_add(value)
            .ok_or_else(|| anyhow::anyhow!("semantic audit row-set delta overflow"))?;
        let row_num = i64::try_from(previous)
            .map_err(|_| anyhow::anyhow!("semantic audit row number exceeds SQLite range"))?;
        output.push(row_num);
        value = 0;
        shift = 0;
    }
    if shift != 0 {
        bail!("semantic audit row-set ends with an incomplete varint");
    }
    Ok(output)
}

fn audit_snapshot_chunk_sha256(
    domain: &str,
    first_row_num: i64,
    last_row_num: i64,
    row_count: i64,
    encoded_rows: &[u8],
) -> String {
    let mut hasher = Sha256::new();
    update_snapshot_digest(&mut hasher, V2_AUDIT_SNAPSHOT_VERSION.as_bytes());
    update_snapshot_digest(&mut hasher, domain.as_bytes());
    hasher.update(first_row_num.to_le_bytes());
    hasher.update(last_row_num.to_le_bytes());
    hasher.update(row_count.to_le_bytes());
    update_snapshot_digest(&mut hasher, encoded_rows);
    bytes_to_hex(&hasher.finalize())
}

fn pending_audit_snapshot_selection(conn: &Connection) -> Result<Option<String>> {
    if !table_exists(conn, "_llm_parse_audit")?
        || !sqlite_table_has_columns(
            conn,
            "_llm_parse_audit",
            &["trusted_intent_json", "examiner_decision"],
        )?
    {
        return Ok(None);
    }
    conn.query_row(
        "SELECT s.selection_id
         FROM _semantic_v2_selection s
         JOIN _semantic_v2_build b ON b.build_id = s.build_id
         LEFT JOIN _semantic_v2_audit_snapshot_stage st
           ON st.selection_id = s.selection_id
         WHERE NOT EXISTS (
                SELECT 1 FROM _semantic_v2_audit_snapshot snap
                WHERE snap.selection_id = s.selection_id
             )
           AND (
                st.selection_id IS NOT NULL
                OR EXISTS (
                    SELECT 1
                    FROM _llm_parse_audit l,
                         json_tree(CASE WHEN json_valid(l.trusted_intent_json)
                                        THEN l.trusted_intent_json ELSE '{}' END) j
                    WHERE l.examiner_decision = 'accepted'
                      AND j.key = 'semanticSelectionId'
                      AND j.value = s.selection_id
                )
                OR (
                    NOT EXISTS (
                        SELECT 1 FROM _semantic_v2_active a WHERE a.build_id = b.build_id
                    )
                    AND EXISTS (
                        SELECT 1
                        FROM _llm_parse_audit l,
                             json_tree(CASE WHEN json_valid(l.trusted_intent_json)
                                            THEN l.trusted_intent_json ELSE '{}' END) j
                        WHERE l.examiner_decision = 'unreviewed'
                          AND j.key = 'semanticSelectionId'
                          AND j.value = s.selection_id
                    )
                )
             )
         ORDER BY CASE WHEN st.selection_id IS NULL THEN 1 ELSE 0 END,
                  CASE WHEN EXISTS (
                    SELECT 1
                    FROM _llm_parse_audit l,
                         json_tree(CASE WHEN json_valid(l.trusted_intent_json)
                                        THEN l.trusted_intent_json ELSE '{}' END) j
                    WHERE l.examiner_decision = 'accepted'
                      AND j.key = 'semanticSelectionId'
                      AND j.value = s.selection_id
                  ) THEN 0 ELSE 1 END,
                  s.build_id, s.selection_id
         LIMIT 1",
        [],
        |row| row.get(0),
    )
    .optional()
    .map_err(Into::into)
}

fn load_audit_snapshot_stage(
    conn: &Connection,
    selection_id: &str,
) -> Result<Option<AuditSnapshotStage>> {
    conn.query_row(
        "SELECT selection_id, build_id, phase, cursor_doc_id, cursor_row_num,
                next_mapping_chunk, next_row_chunk, mappings_seen, rows_seen
         FROM _semantic_v2_audit_snapshot_stage WHERE selection_id = ?1",
        [selection_id],
        |row| {
            Ok(AuditSnapshotStage {
                selection_id: row.get(0)?,
                build_id: row.get(1)?,
                phase: row.get(2)?,
                cursor_doc_id: row.get(3)?,
                cursor_row_num: row.get(4)?,
                next_mapping_chunk: row.get(5)?,
                next_row_chunk: row.get(6)?,
                mappings_seen: row.get(7)?,
                rows_seen: row.get(8)?,
            })
        },
    )
    .optional()
    .map_err(Into::into)
}

fn initialize_audit_snapshot_stage(conn: &Connection, selection_id: &str) -> Result<()> {
    let (build_id, expected_documents): (i64, i64) = conn.query_row(
        "SELECT s.build_id, s.documents_retained
         FROM _semantic_v2_selection s
         JOIN _semantic_v2_build b ON b.build_id = s.build_id
         WHERE s.selection_id = ?1",
        [selection_id],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;
    let documents = {
        let mut statement = conn.prepare(
            "SELECT sd.doc_id, d.text_sha256, d.kind, d.column_key, d.normalized_text,
                    sd.cosine_score, sd.rank_score
             FROM _semantic_v2_selection_doc sd
             JOIN _semantic_v2_document d ON d.doc_id = sd.doc_id
             WHERE sd.selection_id = ?1
             ORDER BY sd.rank_score DESC, sd.cosine_score DESC, sd.doc_id",
        )?;
        let collected = statement
            .query_map([selection_id], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, f64>(5)?,
                    row.get::<_, f64>(6)?,
                ))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        collected
    };
    if documents.len() as i64 != expected_documents {
        bail!(
            "semantic audit snapshot is missing selected documents; preserving stale live artifacts"
        );
    }
    let now = chrono::Utc::now().to_rfc3339();
    conn.execute(
        "INSERT INTO _semantic_v2_audit_snapshot_stage (
            selection_id, build_id, phase, started_at, updated_at
         ) VALUES (?1, ?2, 'mappings', ?3, ?3)",
        params![selection_id, build_id, now],
    )?;
    let mut insert = conn.prepare(
        "INSERT INTO _semantic_v2_audit_snapshot_stage_document (
            selection_id, doc_id, rank, fingerprint_sha256, kind, column_key,
            normalized_text, cosine_score, rank_score
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
    )?;
    for (index, document) in documents.into_iter().enumerate() {
        insert.execute(params![
            selection_id,
            document.0,
            index as i64 + 1,
            document.1,
            document.2,
            document.3,
            document.4,
            document.5,
            document.6,
        ])?;
    }
    Ok(())
}

fn advance_audit_snapshot_mappings(conn: &Connection, stage: &AuditSnapshotStage) -> Result<()> {
    advance_audit_snapshot_mappings_with_hook(conn, stage, || Ok(()))
}

fn advance_audit_snapshot_mappings_with_hook(
    conn: &Connection,
    stage: &AuditSnapshotStage,
    before_cursor_update: impl FnOnce() -> Result<()>,
) -> Result<()> {
    let mappings = {
        let mut statement = conn.prepare(&format!(
            "SELECT m.doc_id, m.row_num
             FROM _semantic_v2_audit_snapshot_stage_document sd
             JOIN _semantic_v2_mapping m
               ON m.build_id = ?2 AND m.doc_id = sd.doc_id
             WHERE sd.selection_id = ?1
               AND (m.doc_id > ?3 OR (m.doc_id = ?3 AND m.row_num > ?4))
             ORDER BY m.doc_id, m.row_num
             LIMIT {V2_AUDIT_MAPPING_BATCH}"
        ))?;
        let collected = statement
            .query_map(
                params![
                    stage.selection_id,
                    stage.build_id,
                    stage.cursor_doc_id,
                    stage.cursor_row_num,
                ],
                |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)),
            )?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        collected
    };
    if mappings.is_empty() {
        conn.execute(
            "UPDATE _semantic_v2_audit_snapshot_stage
             SET phase = 'rows', cursor_doc_id = 0, cursor_row_num = 0,
                 updated_at = ?2
             WHERE selection_id = ?1 AND phase = 'mappings'",
            params![stage.selection_id, chrono::Utc::now().to_rfc3339()],
        )?;
        return Ok(());
    }

    let mut next_chunk = stage.next_mapping_chunk;
    let mut at = 0usize;
    let mut insert_chunk = conn.prepare(
        "INSERT INTO _semantic_v2_audit_snapshot_stage_mapping_chunk (
            selection_id, chunk_index, doc_id, first_row_num, last_row_num,
            row_count, encoded_rows, chunk_sha256
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
    )?;
    let mut insert_row = conn.prepare(
        "INSERT OR IGNORE INTO _semantic_v2_audit_snapshot_stage_row(selection_id, row_num)
         VALUES (?1, ?2)",
    )?;
    while at < mappings.len() {
        let doc_id = mappings[at].0;
        let end = mappings[at..]
            .iter()
            .position(|(candidate, _)| *candidate != doc_id)
            .map(|offset| at + offset)
            .unwrap_or(mappings.len());
        let row_numbers = mappings[at..end]
            .iter()
            .map(|(_, row_num)| *row_num)
            .collect::<Vec<_>>();
        let encoded = encode_sorted_positive_delta_varints(&row_numbers)?;
        let first = *row_numbers
            .first()
            .ok_or_else(|| anyhow::anyhow!("semantic audit mapping chunk was empty"))?;
        let last = *row_numbers
            .last()
            .ok_or_else(|| anyhow::anyhow!("semantic audit mapping chunk was empty"))?;
        let count = row_numbers.len() as i64;
        let digest = audit_snapshot_chunk_sha256("mapping", first, last, count, &encoded);
        insert_chunk.execute(params![
            stage.selection_id,
            next_chunk,
            doc_id,
            first,
            last,
            count,
            encoded,
            digest,
        ])?;
        for row_num in row_numbers {
            insert_row.execute(params![stage.selection_id, row_num])?;
        }
        next_chunk += 1;
        at = end;
    }
    drop(insert_row);
    drop(insert_chunk);
    let (cursor_doc_id, cursor_row_num) = *mappings
        .last()
        .ok_or_else(|| anyhow::anyhow!("semantic audit mapping batch was empty"))?;
    before_cursor_update()?;
    conn.execute(
        "UPDATE _semantic_v2_audit_snapshot_stage
         SET cursor_doc_id = ?2, cursor_row_num = ?3, next_mapping_chunk = ?4,
             mappings_seen = mappings_seen + ?5, updated_at = ?6
         WHERE selection_id = ?1 AND phase = 'mappings'",
        params![
            stage.selection_id,
            cursor_doc_id,
            cursor_row_num,
            next_chunk,
            mappings.len() as i64,
            chrono::Utc::now().to_rfc3339(),
        ],
    )?;
    Ok(())
}

fn advance_audit_snapshot_rows(conn: &Connection, stage: &AuditSnapshotStage) -> Result<bool> {
    let row_numbers = {
        let mut statement = conn.prepare(&format!(
            "SELECT row_num FROM _semantic_v2_audit_snapshot_stage_row
             WHERE selection_id = ?1 AND row_num > ?2
             ORDER BY row_num LIMIT {V2_AUDIT_ROW_CHUNK_ROWS}"
        ))?;
        let collected = statement
            .query_map(params![stage.selection_id, stage.cursor_row_num], |row| {
                row.get::<_, i64>(0)
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        collected
    };
    if row_numbers.is_empty() {
        return Ok(false);
    }
    let encoded = encode_sorted_positive_delta_varints(&row_numbers)?;
    let first = *row_numbers
        .first()
        .ok_or_else(|| anyhow::anyhow!("semantic audit row chunk was empty"))?;
    let last = *row_numbers
        .last()
        .ok_or_else(|| anyhow::anyhow!("semantic audit row chunk was empty"))?;
    let count = row_numbers.len() as i64;
    let digest = audit_snapshot_chunk_sha256("row-set", first, last, count, &encoded);
    conn.execute(
        "INSERT INTO _semantic_v2_audit_snapshot_stage_row_chunk (
            selection_id, chunk_index, first_row_num, last_row_num, row_count,
            encoded_rows, chunk_sha256
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![
            stage.selection_id,
            stage.next_row_chunk,
            first,
            last,
            count,
            encoded,
            digest,
        ],
    )?;
    conn.execute(
        "UPDATE _semantic_v2_audit_snapshot_stage
         SET cursor_row_num = ?2, next_row_chunk = next_row_chunk + 1,
             rows_seen = rows_seen + ?3, updated_at = ?4
         WHERE selection_id = ?1 AND phase = 'rows'",
        params![
            stage.selection_id,
            last,
            count,
            chrono::Utc::now().to_rfc3339(),
        ],
    )?;
    Ok(true)
}

fn load_audit_snapshot_source(
    conn: &Connection,
    selection_id: &str,
) -> Result<AuditSnapshotSource> {
    conn.query_row(
        "SELECT s.build_id, s.dataset_hash, b.schema_hash, b.index_version,
                b.normalizer_version, b.model_name, b.model_version, b.model_sha256,
                b.tokenizer_sha256, b.config_sha256, s.query_sha256, s.policy_version,
                s.minimum_score,
                s.maximum_documents, s.documents_above_threshold, s.documents_retained,
                s.rows_matched, s.documents_truncated, s.broad_row_warning, s.warnings_json,
                b.source_rows, b.rows_scanned, b.documents_seen, b.documents_embedded,
                b.documents_mapped, b.mappings_written, b.documents_skipped,
                b.mappings_skipped, b.cells_truncated, b.columns_omitted, b.chunks_omitted,
                COALESCE(b.candidate_documents, -1), COALESCE(b.candidate_mappings, -1),
                COALESCE(b.candidate_document_limit, -1),
                COALESCE(b.candidate_mapping_limit, -1), s.created_at
         FROM _semantic_v2_selection s
         JOIN _semantic_v2_build b ON b.build_id = s.build_id
         WHERE s.selection_id = ?1",
        [selection_id],
        |row| {
            Ok(AuditSnapshotSource {
                build_id: row.get(0)?,
                dataset_hash: row.get(1)?,
                schema_hash: row.get(2)?,
                index_version: row.get(3)?,
                normalizer_version: row.get(4)?,
                model_name: row.get(5)?,
                model_version: row.get(6)?,
                model_sha256: row.get(7)?,
                tokenizer_sha256: row.get(8)?,
                config_sha256: row.get(9)?,
                query_sha256: row.get(10)?,
                policy_version: row.get(11)?,
                minimum_score: row.get(12)?,
                maximum_documents: row.get(13)?,
                documents_above_threshold: row.get(14)?,
                documents_retained: row.get(15)?,
                rows_matched: row.get(16)?,
                documents_truncated: row.get(17)?,
                broad_row_warning: row.get(18)?,
                warnings_json: row.get(19)?,
                source_rows: row.get(20)?,
                rows_scanned: row.get(21)?,
                documents_seen: row.get(22)?,
                documents_embedded: row.get(23)?,
                documents_mapped: row.get(24)?,
                mappings_written: row.get(25)?,
                documents_skipped: row.get(26)?,
                mappings_skipped: row.get(27)?,
                cells_truncated: row.get(28)?,
                columns_omitted: row.get(29)?,
                chunks_omitted: row.get(30)?,
                candidate_documents: row.get(31)?,
                candidate_mappings: row.get(32)?,
                candidate_document_limit: row.get(33)?,
                candidate_mapping_limit: row.get(34)?,
                selection_created_at: row.get(35)?,
            })
        },
    )
    .map_err(Into::into)
}

fn finalize_audit_snapshot(conn: &Connection, stage: &AuditSnapshotStage) -> Result<()> {
    let source = load_audit_snapshot_source(conn, &stage.selection_id)?;
    if source.build_id != stage.build_id {
        bail!("semantic audit snapshot build identity changed; preserving stale live artifacts");
    }
    let mismatched_document_provenance: i64 = conn.query_row(
        "SELECT COUNT(*)
         FROM _semantic_v2_audit_snapshot_stage_document sd
         JOIN _semantic_v2_document d ON d.doc_id = sd.doc_id
         WHERE sd.selection_id = ?1
           AND (d.model_sha256 <> ?2 OR d.tokenizer_sha256 <> ?3
                OR d.config_sha256 <> ?4 OR d.normalizer_version <> ?5)",
        params![
            stage.selection_id,
            source.model_sha256,
            source.tokenizer_sha256,
            source.config_sha256,
            source.normalizer_version,
        ],
        |row| row.get(0),
    )?;
    if mismatched_document_provenance != 0 {
        bail!(
            "semantic audit document provenance differs from its build; preserving stale live artifacts"
        );
    }
    let documents = {
        let mut statement = conn.prepare(
            "SELECT doc_id, rank, fingerprint_sha256, kind, column_key, normalized_text,
                    cosine_score, rank_score
             FROM _semantic_v2_audit_snapshot_stage_document
             WHERE selection_id = ?1 ORDER BY rank",
        )?;
        let collected = statement
            .query_map([&stage.selection_id], |row| {
                Ok(AuditSnapshotDocument {
                    doc_id: row.get(0)?,
                    rank: row.get(1)?,
                    fingerprint_sha256: row.get(2)?,
                    kind: row.get(3)?,
                    column_key: row.get(4)?,
                    normalized_text: row.get(5)?,
                    cosine_score: row.get(6)?,
                    rank_score: row.get(7)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        collected
    };
    if documents.len() as i64 != source.documents_retained {
        bail!(
            "semantic audit snapshot selected-document count changed; preserving stale live artifacts"
        );
    }

    let mut final_documents = Vec::with_capacity(documents.len());
    let mut aggregate_mapping_hasher = Sha256::new();
    update_snapshot_digest(
        &mut aggregate_mapping_hasher,
        V2_AUDIT_SNAPSHOT_VERSION.as_bytes(),
    );
    update_snapshot_digest(&mut aggregate_mapping_hasher, b"selected-document-mappings");
    let mut mapping_total = 0i64;
    let mut mapping_statement = conn.prepare(
        "SELECT first_row_num, last_row_num, row_count, encoded_rows, chunk_sha256
         FROM _semantic_v2_audit_snapshot_stage_mapping_chunk
         WHERE selection_id = ?1 AND doc_id = ?2 ORDER BY chunk_index",
    )?;
    for document in documents {
        let chunks = mapping_statement
            .query_map(params![stage.selection_id, document.doc_id], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, Vec<u8>>(3)?,
                    row.get::<_, String>(4)?,
                ))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        let mut document_hasher = Sha256::new();
        update_snapshot_digest(&mut document_hasher, V2_AUDIT_SNAPSHOT_VERSION.as_bytes());
        update_snapshot_digest(&mut document_hasher, b"document-mappings");
        update_snapshot_digest(&mut document_hasher, document.fingerprint_sha256.as_bytes());
        let mut document_mapping_count = 0i64;
        for (first, last, count, encoded, stored_digest) in chunks {
            let expected = audit_snapshot_chunk_sha256("mapping", first, last, count, &encoded);
            if expected != stored_digest {
                bail!(
                    "semantic audit mapping chunk failed integrity validation; preserving stale live artifacts"
                );
            }
            document_mapping_count = document_mapping_count
                .checked_add(count)
                .ok_or_else(|| anyhow::anyhow!("semantic audit mapping count overflow"))?;
            document_hasher.update(first.to_le_bytes());
            document_hasher.update(last.to_le_bytes());
            document_hasher.update(count.to_le_bytes());
            update_snapshot_digest(&mut document_hasher, stored_digest.as_bytes());
        }
        if document_mapping_count == 0 {
            bail!(
                "semantic audit selected document has no mappings; preserving stale live artifacts"
            );
        }
        let mapping_sha256 = bytes_to_hex(&document_hasher.finalize());
        mapping_total = mapping_total
            .checked_add(document_mapping_count)
            .ok_or_else(|| anyhow::anyhow!("semantic audit mapping count overflow"))?;
        aggregate_mapping_hasher.update(document.rank.to_le_bytes());
        update_snapshot_digest(
            &mut aggregate_mapping_hasher,
            document.fingerprint_sha256.as_bytes(),
        );
        aggregate_mapping_hasher.update(document_mapping_count.to_le_bytes());
        update_snapshot_digest(&mut aggregate_mapping_hasher, mapping_sha256.as_bytes());
        final_documents.push((document, document_mapping_count, mapping_sha256));
    }
    drop(mapping_statement);
    if mapping_total != stage.mappings_seen {
        bail!("semantic audit mapping total changed; preserving stale live artifacts");
    }
    let mapping_sha256 = bytes_to_hex(&aggregate_mapping_hasher.finalize());

    let row_chunks = {
        let mut statement = conn.prepare(
            "SELECT chunk_index, first_row_num, last_row_num, row_count, encoded_rows,
                    chunk_sha256
             FROM _semantic_v2_audit_snapshot_stage_row_chunk
             WHERE selection_id = ?1 ORDER BY chunk_index",
        )?;
        let collected = statement
            .query_map([&stage.selection_id], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, i64>(3)?,
                    row.get::<_, Vec<u8>>(4)?,
                    row.get::<_, String>(5)?,
                ))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        collected
    };
    let mut row_set_hasher = Sha256::new();
    update_snapshot_digest(&mut row_set_hasher, V2_AUDIT_SNAPSHOT_VERSION.as_bytes());
    update_snapshot_digest(&mut row_set_hasher, V2_AUDIT_ROW_SET_ENCODING.as_bytes());
    let mut row_total = 0i64;
    for (chunk_index, first, last, count, encoded, stored_digest) in &row_chunks {
        let expected = audit_snapshot_chunk_sha256("row-set", *first, *last, *count, encoded);
        if expected != *stored_digest {
            bail!(
                "semantic audit row-set chunk failed integrity validation; preserving stale live artifacts"
            );
        }
        row_total = row_total
            .checked_add(*count)
            .ok_or_else(|| anyhow::anyhow!("semantic audit row count overflow"))?;
        row_set_hasher.update(chunk_index.to_le_bytes());
        row_set_hasher.update(first.to_le_bytes());
        row_set_hasher.update(last.to_le_bytes());
        row_set_hasher.update(count.to_le_bytes());
        update_snapshot_digest(&mut row_set_hasher, stored_digest.as_bytes());
    }
    if row_total != stage.rows_seen || row_total != source.rows_matched {
        bail!("semantic audit expanded row set changed; preserving stale live artifacts");
    }
    let row_set_sha256 = bytes_to_hex(&row_set_hasher.finalize());
    let archived_at = chrono::Utc::now().to_rfc3339();
    conn.execute(
        "INSERT INTO _semantic_v2_audit_snapshot (
            selection_id, snapshot_version, build_id, dataset_hash, schema_hash,
            index_version, normalizer_version, model_name, model_version, model_sha256,
            tokenizer_sha256, config_sha256, query_sha256, policy_version, minimum_score,
            maximum_documents, documents_above_threshold, documents_retained, rows_matched,
            documents_truncated, broad_row_warning, warnings_json, source_rows,
            index_rows_scanned, index_documents_seen, index_documents_embedded,
            index_documents_mapped, index_mappings_written, index_documents_skipped,
            index_mappings_skipped, index_cells_truncated, index_columns_omitted,
            index_chunks_omitted, candidate_documents, candidate_mappings,
            candidate_document_limit, candidate_mapping_limit, selected_document_count,
            mapping_count, mapping_sha256, row_count, row_set_sha256, row_set_encoding,
            selection_created_at, archived_at
         ) VALUES (
            ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10,
            ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20,
            ?21, ?22, ?23, ?24, ?25, ?26, ?27, ?28, ?29, ?30,
            ?31, ?32, ?33, ?34, ?35, ?36, ?37, ?38, ?39, ?40,
            ?41, ?42, ?43, ?44, ?45
         )",
        params![
            stage.selection_id,
            V2_AUDIT_SNAPSHOT_VERSION,
            source.build_id,
            source.dataset_hash,
            source.schema_hash,
            source.index_version,
            source.normalizer_version,
            source.model_name,
            source.model_version,
            source.model_sha256,
            source.tokenizer_sha256,
            source.config_sha256,
            source.query_sha256,
            source.policy_version,
            source.minimum_score,
            source.maximum_documents,
            source.documents_above_threshold,
            source.documents_retained,
            source.rows_matched,
            source.documents_truncated,
            source.broad_row_warning,
            source.warnings_json,
            source.source_rows,
            source.rows_scanned,
            source.documents_seen,
            source.documents_embedded,
            source.documents_mapped,
            source.mappings_written,
            source.documents_skipped,
            source.mappings_skipped,
            source.cells_truncated,
            source.columns_omitted,
            source.chunks_omitted,
            source.candidate_documents,
            source.candidate_mappings,
            source.candidate_document_limit,
            source.candidate_mapping_limit,
            final_documents.len() as i64,
            mapping_total,
            mapping_sha256,
            row_total,
            row_set_sha256,
            V2_AUDIT_ROW_SET_ENCODING,
            source.selection_created_at,
            archived_at,
        ],
    )?;

    let mut insert_document = conn.prepare(
        "INSERT INTO _semantic_v2_audit_snapshot_document (
            selection_id, rank, source_doc_id, fingerprint_sha256, kind, column_key,
            normalized_text, cosine_score, rank_score, mapping_count, mapping_sha256
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
    )?;
    for (document, mapping_count, document_mapping_sha256) in final_documents {
        insert_document.execute(params![
            stage.selection_id,
            document.rank,
            document.doc_id,
            document.fingerprint_sha256,
            document.kind,
            document.column_key,
            document.normalized_text,
            document.cosine_score,
            document.rank_score,
            mapping_count,
            document_mapping_sha256,
        ])?;
    }
    drop(insert_document);
    let mut insert_row_chunk = conn.prepare(
        "INSERT INTO _semantic_v2_audit_snapshot_row_chunk (
            selection_id, chunk_index, first_row_num, last_row_num, row_count,
            encoded_rows, chunk_sha256
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
    )?;
    for (chunk_index, first, last, count, encoded, digest) in row_chunks {
        insert_row_chunk.execute(params![
            stage.selection_id,
            chunk_index,
            first,
            last,
            count,
            encoded,
            digest,
        ])?;
    }
    drop(insert_row_chunk);

    for table in [
        "_semantic_v2_audit_snapshot_stage_mapping_chunk",
        "_semantic_v2_audit_snapshot_stage_row_chunk",
        "_semantic_v2_audit_snapshot_stage_row",
        "_semantic_v2_audit_snapshot_stage_document",
        "_semantic_v2_audit_snapshot_stage",
    ] {
        conn.execute(
            &format!(
                "DELETE FROM {} WHERE selection_id = ?1",
                db::quote_ident(table)
            ),
            [&stage.selection_id],
        )?;
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticAuditArchiveProgress {
    pub steps_advanced: usize,
    pub snapshots_completed: usize,
    pub pending: bool,
}

#[derive(Debug, Clone, Copy)]
struct RequiredAuditSnapshotStep {
    snapshot_completed: bool,
}

/// Advances at most one bounded archive step. While this returns work, stale v2 deletion is
/// deliberately backpressured: no selected documents, mappings, or embeddings can disappear
/// before an accepted plan (or stale unreviewed plan) has a complete immutable evidence snapshot.
fn advance_required_audit_snapshot(conn: &Connection) -> Result<Option<RequiredAuditSnapshotStep>> {
    let Some(selection_id) = pending_audit_snapshot_selection(conn)? else {
        return Ok(None);
    };
    let Some(stage) = load_audit_snapshot_stage(conn, &selection_id)? else {
        initialize_audit_snapshot_stage(conn, &selection_id)?;
        return Ok(Some(RequiredAuditSnapshotStep {
            snapshot_completed: false,
        }));
    };
    let mut snapshot_completed = false;
    match stage.phase.as_str() {
        "mappings" => {
            advance_audit_snapshot_mappings(conn, &stage)?;
        }
        "rows" => {
            if !advance_audit_snapshot_rows(conn, &stage)? {
                finalize_audit_snapshot(conn, &stage)?;
                snapshot_completed = true;
            }
        }
        other => bail!("semantic audit snapshot has invalid stage phase {other}"),
    }
    Ok(Some(RequiredAuditSnapshotStep { snapshot_completed }))
}

fn prepare_required_semantic_audit_archive(conn: &Connection) -> Result<bool> {
    if !table_exists(conn, "_llm_parse_audit")?
        || !table_exists(conn, "_semantic_v2_build")?
        || !table_exists(conn, "_semantic_v2_selection")?
    {
        return Ok(false);
    }
    create_semantic_v2_schema(conn)?;
    Ok(true)
}

fn advance_required_semantic_audit_archive_transaction(
    conn: &mut Connection,
) -> Result<SemanticAuditArchiveProgress> {
    let tx = conn.transaction()?;
    let advanced = advance_required_audit_snapshot(&tx)?;
    let pending = pending_audit_snapshot_selection(&tx)?.is_some();
    let progress = SemanticAuditArchiveProgress {
        steps_advanced: usize::from(advanced.is_some()),
        snapshots_completed: usize::from(advanced.is_some_and(|step| step.snapshot_completed)),
        pending,
    };
    tx.commit()?;
    Ok(progress)
}

/// Advances required accepted/stale-unreviewed semantic evidence archival for a short bounded
/// slice. Every step commits independently, including its staging cursor, so callers can safely
/// retry while `pending` is true without holding a long writer transaction.
pub fn archive_required_semantic_audits_slice(
    conn: &mut Connection,
) -> Result<SemanticAuditArchiveProgress> {
    if !prepare_required_semantic_audit_archive(conn)? {
        return Ok(SemanticAuditArchiveProgress::default());
    }
    let started = Instant::now();
    let mut total = SemanticAuditArchiveProgress::default();
    for _ in 0..V2_AUDIT_ARCHIVE_STEPS_PER_SLICE {
        let progress = advance_required_semantic_audit_archive_transaction(conn)?;
        total.steps_advanced += progress.steps_advanced;
        total.snapshots_completed += progress.snapshots_completed;
        total.pending = progress.pending;
        if !total.pending || started.elapsed().as_millis() >= V2_AUDIT_ARCHIVE_SLICE_TIME_BUDGET_MS
        {
            break;
        }
    }
    Ok(total)
}

/// Completes every currently required immutable semantic evidence snapshot. Work may take time on
/// very large evidence sets, but no transaction contains more than one bounded archive step.
pub fn complete_required_semantic_audits(
    conn: &mut Connection,
) -> Result<SemanticAuditArchiveProgress> {
    if !prepare_required_semantic_audit_archive(conn)? {
        return Ok(SemanticAuditArchiveProgress::default());
    }
    let mut total = SemanticAuditArchiveProgress::default();
    loop {
        let progress = advance_required_semantic_audit_archive_transaction(conn)?;
        total.steps_advanced += progress.steps_advanced;
        total.snapshots_completed += progress.snapshots_completed;
        total.pending = progress.pending;
        if !total.pending {
            return Ok(total);
        }
        if progress.steps_advanced == 0 {
            bail!("semantic audit archival remained pending without making progress");
        }
    }
}

/// Removes a bounded slice of superseded semantic artifacts after a current build is active.
/// Repeated cache hits converge without turning any one request into a long writer transaction.
fn prune_stale_semantic_artifacts_pass(conn: &mut Connection) -> Result<usize> {
    let has_v2 = table_exists(conn, "_semantic_v2_build")?;
    let has_v1_index = table_exists(conn, "_semantic_index")?;
    let has_v1_info = table_exists(conn, "_semantic_index_info")?;
    if !has_v2 && !has_v1_index && !has_v1_info {
        return Ok(0);
    }
    let tx = conn.transaction()?;
    let mut removed = 0usize;

    if has_v1_index {
        removed += tx.execute(
            &format!(
                "DELETE FROM _semantic_index WHERE row_num IN (
                    SELECT row_num FROM _semantic_index ORDER BY row_num
                    LIMIT {V1_INDEX_PRUNE_BATCH}
                 )"
            ),
            [],
        )?;
    }
    if has_v1_info {
        let v1_rows = if has_v1_index {
            tx.query_row("SELECT COUNT(*) FROM _semantic_index", [], |row| {
                row.get::<_, i64>(0)
            })?
        } else {
            0
        };
        if v1_rows == 0 {
            removed += tx.execute(
                "DELETE FROM _semantic_index_info WHERE rowid IN (
                    SELECT rowid FROM _semantic_index_info ORDER BY rowid LIMIT 16
                 )",
                [],
            )?;
        }
    }

    if has_v2 {
        // Every archive step and its cursor publication share this transaction. Returning here
        // also globally backpressures stale v2 deletion: a failed or incomplete required
        // snapshot leaves every source selection document, mapping, build, and embedding live
        // for the next bounded retry.
        if advance_required_audit_snapshot(&tx)?.is_some() {
            tx.commit()?;
            return Ok(removed.saturating_add(1));
        }
        removed += tx.execute(
            &format!(
                "DELETE FROM _semantic_v2_selection_doc
                 WHERE (selection_id, doc_id) IN (
                    SELECT sd.selection_id, sd.doc_id
                    FROM _semantic_v2_selection_doc sd
                    JOIN _semantic_v2_selection s ON s.selection_id = sd.selection_id
                    JOIN _semantic_v2_build b ON b.build_id = s.build_id
                    WHERE NOT EXISTS (
                        SELECT 1 FROM _semantic_v2_active a WHERE a.build_id = b.build_id
                    )
                    ORDER BY s.build_id, sd.selection_id, sd.doc_id
                    LIMIT {V2_STALE_SELECTION_DOC_PRUNE_BATCH}
                 )"
            ),
            [],
        )?;
        removed += tx.execute(
            &format!(
                "DELETE FROM _semantic_v2_selection WHERE selection_id IN (
                    SELECT s.selection_id
                    FROM _semantic_v2_selection s
                    JOIN _semantic_v2_build b ON b.build_id = s.build_id
                    WHERE NOT EXISTS (
                        SELECT 1 FROM _semantic_v2_active a WHERE a.build_id = b.build_id
                    )
                      AND NOT EXISTS (
                        SELECT 1 FROM _semantic_v2_selection_doc sd
                        WHERE sd.selection_id = s.selection_id
                      )
                    ORDER BY s.build_id, s.selection_id
                    LIMIT {V2_STALE_SELECTION_PRUNE_BATCH}
                 )"
            ),
            [],
        )?;
        removed += tx.execute(
            &format!(
                "DELETE FROM _semantic_v2_mapping
                 WHERE (build_id, doc_id, row_num) IN (
                    SELECT m.build_id, m.doc_id, m.row_num
                    FROM _semantic_v2_mapping m
                    JOIN _semantic_v2_build b ON b.build_id = m.build_id
                    WHERE NOT EXISTS (
                        SELECT 1 FROM _semantic_v2_active a WHERE a.build_id = b.build_id
                    )
                    ORDER BY m.build_id, m.doc_id, m.row_num
                    LIMIT {V2_STALE_MAPPING_PRUNE_BATCH}
                 )"
            ),
            [],
        )?;
        removed += tx.execute(
            &format!(
                "DELETE FROM _semantic_v2_column_plan WHERE (build_id, col_index) IN (
                    SELECT p.build_id, p.col_index
                    FROM _semantic_v2_column_plan p
                    JOIN _semantic_v2_build b ON b.build_id = p.build_id
                    WHERE NOT EXISTS (
                        SELECT 1 FROM _semantic_v2_active a WHERE a.build_id = b.build_id
                    )
                    ORDER BY p.build_id, p.col_index
                    LIMIT {V2_STALE_COLUMN_PRUNE_BATCH}
                 )"
            ),
            [],
        )?;
        removed += tx.execute(
            &format!(
                "DELETE FROM _semantic_v2_build WHERE build_id IN (
                    SELECT b.build_id FROM _semantic_v2_build b
                    WHERE NOT EXISTS (
                        SELECT 1 FROM _semantic_v2_active a WHERE a.build_id = b.build_id
                    )
                      AND NOT EXISTS (
                        SELECT 1 FROM _semantic_v2_mapping m WHERE m.build_id = b.build_id
                      )
                      AND NOT EXISTS (
                        SELECT 1 FROM _semantic_v2_column_plan p WHERE p.build_id = b.build_id
                      )
                      AND NOT EXISTS (
                        SELECT 1 FROM _semantic_v2_selection s WHERE s.build_id = b.build_id
                      )
                    ORDER BY b.build_id LIMIT {V2_STALE_BUILD_PRUNE_BATCH}
                 )"
            ),
            [],
        )?;
        let stale_builds_remaining: i64 = tx.query_row(
            "SELECT COUNT(*) FROM _semantic_v2_build b
             WHERE NOT EXISTS (
                SELECT 1 FROM _semantic_v2_active a WHERE a.build_id = b.build_id
             )",
            [],
            |row| row.get(0),
        )?;
        if stale_builds_remaining == 0 {
            // Once every stale build and reference is gone, the active build's (build_id, doc_id)
            // primary-key lookup identifies reusable documents without constructing a large
            // secondary index during migration.
            removed += tx.execute(
                &format!(
                    "DELETE FROM _semantic_v2_document WHERE doc_id IN (
                        SELECT d.doc_id FROM _semantic_v2_document d
                        WHERE NOT EXISTS (
                            SELECT 1
                            FROM _semantic_v2_active a
                            JOIN _semantic_v2_mapping m
                              ON m.build_id = a.build_id AND m.doc_id = d.doc_id
                        )
                        ORDER BY d.doc_id LIMIT {V2_ORPHAN_DOCUMENT_PRUNE_BATCH}
                     )"
                ),
                [],
            )?;
        }
    }
    tx.commit()?;
    Ok(removed)
}

fn prune_stale_semantic_artifacts(conn: &mut Connection) -> Result<()> {
    let started = Instant::now();
    for _ in 0..V2_PRUNE_PASSES_PER_INVOCATION {
        if prune_stale_semantic_artifacts_pass(conn)? == 0 {
            break;
        }
        if started.elapsed().as_millis() >= V2_PRUNE_TIME_BUDGET_MS {
            break;
        }
    }
    Ok(())
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
           AND b.index_version = ?3 AND b.normalizer_version = ?4
           AND b.model_name = ?5 AND b.model_version = ?6 AND b.model_sha256 = ?7
           AND b.tokenizer_sha256 = ?8 AND b.config_sha256 = ?9",
        params![
            dataset_hash,
            schema_hash,
            V2_INDEX_VERSION,
            V2_NORMALIZER_VERSION,
            MODEL_NAME,
            MODEL_VERSION,
            MODEL_SHA256,
            TOKENIZER_SHA256,
            CONFIG_SHA256,
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

fn new_worker_token(dataset_hash: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(dataset_hash.as_bytes());
    hasher.update(chrono::Utc::now().to_rfc3339().as_bytes());
    hasher.update(std::process::id().to_le_bytes());
    hasher.update(
        V2_WORKER_SEQUENCE
            .fetch_add(1, AtomicOrdering::Relaxed)
            .to_le_bytes(),
    );
    bytes_to_hex(&hasher.finalize())
}

enum PreparedV2Build {
    Ready(i64),
    Work {
        build_id: i64,
        cursor: i64,
        plans: Vec<ColumnPlan>,
        resumed: bool,
    },
}

fn prepare_v2_build(
    conn: &mut Connection,
    columns: &[ColumnMeta],
    dataset_hash: &str,
    schema_hash: &str,
    rows_total: i64,
    worker_token: &str,
) -> Result<PreparedV2Build> {
    if let Some((build_id, cursor, status, previous_worker)) = conn
        .query_row(
            "SELECT build_id, cursor_row_num, status, worker_token FROM _semantic_v2_build
             WHERE dataset_hash = ?1 AND schema_hash = ?2 AND index_version = ?3
               AND normalizer_version = ?4 AND model_name = ?5 AND model_version = ?6
               AND model_sha256 = ?7 AND tokenizer_sha256 = ?8 AND config_sha256 = ?9
             ORDER BY build_id DESC LIMIT 1",
            params![
                dataset_hash,
                schema_hash,
                V2_INDEX_VERSION,
                V2_NORMALIZER_VERSION,
                MODEL_NAME,
                MODEL_VERSION,
                MODEL_SHA256,
                TOKENIZER_SHA256,
                CONFIG_SHA256,
            ],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, Option<String>>(3)?,
                ))
            },
        )
        .optional()?
    {
        if status == "ready" {
            return Ok(PreparedV2Build::Ready(build_id));
        }
        let claimed = conn.execute(
            "UPDATE _semantic_v2_build SET status = 'building', updated_at = ?2, error = NULL,
                worker_token = ?3
             WHERE build_id = ?1 AND status = ?4 AND worker_token IS ?5",
            params![
                build_id,
                chrono::Utc::now().to_rfc3339(),
                worker_token,
                status,
                previous_worker,
            ],
        )?;
        if claimed == 0 {
            return prepare_v2_build(
                conn,
                columns,
                dataset_hash,
                schema_hash,
                rows_total,
                worker_token,
            );
        }
        return Ok(PreparedV2Build::Work {
            build_id,
            cursor,
            plans: load_column_plans(conn, build_id)?,
            resumed: cursor > 0,
        });
    }

    let plans = classify_columns(conn, columns)?;
    let now = chrono::Utc::now().to_rfc3339();
    let tx = conn.transaction()?;
    let inserted = tx.execute(
        "INSERT OR IGNORE INTO _semantic_v2_build (
            dataset_hash, schema_hash, index_version, normalizer_version, model_name,
            model_version, model_sha256, tokenizer_sha256, config_sha256, status,
            worker_token, source_rows, started_at, updated_at
         ) VALUES (
            ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, 'building', ?10, ?11, ?12, ?12
         )",
        params![
            dataset_hash,
            schema_hash,
            V2_INDEX_VERSION,
            V2_NORMALIZER_VERSION,
            MODEL_NAME,
            MODEL_VERSION,
            MODEL_SHA256,
            TOKENIZER_SHA256,
            CONFIG_SHA256,
            worker_token,
            rows_total,
            now,
        ],
    )?;
    if inserted == 0 {
        // Another connection won the identity race while this connection classified columns.
        // Its build and column plan are now committed because SQLite serializes these writers.
        tx.commit()?;
        return prepare_v2_build(
            conn,
            columns,
            dataset_hash,
            schema_hash,
            rows_total,
            worker_token,
        );
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
    Ok(PreparedV2Build::Work {
        build_id,
        cursor: 0,
        plans,
        resumed: false,
    })
}

fn build_summary(
    conn: &Connection,
    build_id: i64,
    started: Instant,
    from_cache: bool,
    resumed: bool,
    cancelled: bool,
) -> Result<SemanticIndexSummary> {
    let (
        rows,
        documents,
        documents_mapped,
        documents_skipped,
        mappings,
        mappings_skipped,
        cells_truncated,
        columns_omitted,
        chunks_omitted,
    ) = conn.query_row(
        "SELECT rows_scanned, documents_embedded, documents_mapped, documents_skipped,
                mappings_written, mappings_skipped, cells_truncated, columns_omitted,
                chunks_omitted
         FROM _semantic_v2_build WHERE build_id = ?1",
        [build_id],
        |row| {
            Ok((
                row.get(0)?,
                row.get(1)?,
                row.get(2)?,
                row.get(3)?,
                row.get(4)?,
                row.get(5)?,
                row.get(6)?,
                row.get(7)?,
                row.get(8)?,
            ))
        },
    )?;
    let truncated = documents_skipped > 0
        || mappings_skipped > 0
        || cells_truncated > 0
        || columns_omitted > 0
        || chunks_omitted > 0;
    let warnings = if truncated {
        vec![format!(
            "Semantic indexing was bounded: {cells_truncated} oversized cell(s) were truncated, {columns_omitted} eligible wide-row column value(s) were omitted, {chunks_omitted} chunk document(s) were omitted or truncated, {documents_skipped} new-document candidate(s) were skipped, and {mappings_skipped} document-to-row mapping candidate(s) were skipped. Exact lexical and structured matching remains complete across all {rows} scanned row(s)."
        )]
    } else {
        Vec::new()
    };
    Ok(SemanticIndexSummary {
        rows_indexed: rows,
        documents_indexed: documents,
        documents_mapped,
        documents_skipped,
        mappings_written: mappings,
        mappings_skipped,
        cells_truncated,
        columns_omitted,
        chunks_omitted,
        truncated,
        warnings,
        elapsed_ms: started.elapsed().as_millis(),
        from_cache,
        resumed,
        cancelled,
        model_name: MODEL_NAME,
        model_version: MODEL_VERSION,
    })
}

fn pause_owned_build(conn: &Connection, build_id: i64, worker_token: &str) -> Result<bool> {
    Ok(conn.execute(
        "UPDATE _semantic_v2_build SET status = 'paused', updated_at = ?3
         WHERE build_id = ?1 AND status = 'building' AND worker_token = ?2",
        params![build_id, worker_token, chrono::Utc::now().to_rfc3339()],
    )? > 0)
}

fn build_is_owned(conn: &Connection, build_id: i64, worker_token: &str) -> Result<bool> {
    conn.query_row(
        "SELECT status = 'building' AND worker_token = ?2
         FROM _semantic_v2_build WHERE build_id = ?1",
        params![build_id, worker_token],
        |row| row.get::<_, bool>(0),
    )
    .map_err(Into::into)
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
    on_progress: P,
) -> Result<SemanticIndexSummary>
where
    E: SemanticEmbedder + ?Sized,
    C: Fn() -> bool,
    P: FnMut(SemanticBuildProgress),
{
    ensure_semantic_index_v2_with_limits(
        conn,
        columns,
        embedder,
        is_cancelled,
        on_progress,
        SemanticResourceLimits::PRODUCTION,
    )
}

fn ensure_semantic_index_v2_with_limits<E, C, P>(
    conn: &mut Connection,
    columns: &[ColumnMeta],
    embedder: &E,
    is_cancelled: C,
    mut on_progress: P,
    limits: SemanticResourceLimits,
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
        // A valid active index remains usable even if bounded stale-cache reclamation is
        // interrupted or encounters a recoverable database error. Future cache hits retry it.
        let _ = prune_stale_semantic_artifacts(conn);
        let summary = build_summary(conn, build_id, started, true, false, false)?;
        on_progress(SemanticBuildProgress {
            build_id,
            phase: "ready".to_string(),
            rows_scanned: summary.rows_indexed,
            rows_total,
            documents_embedded: summary.documents_indexed,
            mappings_written: summary.mappings_written,
            documents_skipped: summary.documents_skipped,
            mappings_skipped: summary.mappings_skipped,
            cells_truncated: summary.cells_truncated,
            columns_omitted: summary.columns_omitted,
            chunks_omitted: summary.chunks_omitted,
            resumed_from_row: summary.rows_indexed,
        });
        return Ok(summary);
    }

    let worker_token = new_worker_token(&dataset_hash);
    let prepared = prepare_v2_build(
        conn,
        columns,
        &dataset_hash,
        &schema_hash,
        rows_total,
        &worker_token,
    )?;
    let (build_id, mut cursor, plans, resumed) = match prepared {
        PreparedV2Build::Ready(build_id) => {
            conn.execute(
                "INSERT INTO _semantic_v2_active(singleton, build_id) VALUES (1, ?1)
                 ON CONFLICT(singleton) DO UPDATE SET build_id = excluded.build_id",
                [build_id],
            )?;
            let _ = prune_stale_semantic_artifacts(conn);
            let summary = build_summary(conn, build_id, started, true, false, false)?;
            on_progress(SemanticBuildProgress {
                build_id,
                phase: "ready".to_string(),
                rows_scanned: summary.rows_indexed,
                rows_total,
                documents_embedded: summary.documents_indexed,
                mappings_written: summary.mappings_written,
                documents_skipped: summary.documents_skipped,
                mappings_skipped: summary.mappings_skipped,
                cells_truncated: summary.cells_truncated,
                columns_omitted: summary.columns_omitted,
                chunks_omitted: summary.chunks_omitted,
                resumed_from_row: summary.rows_indexed,
            });
            return Ok(summary);
        }
        PreparedV2Build::Work {
            build_id,
            cursor,
            plans,
            resumed,
        } => (build_id, cursor, plans, resumed),
    };
    let resumed_from_row = cursor;
    let resource_policy = match determine_semantic_resource_policy(
        conn,
        columns,
        &plans,
        build_id,
        limits,
        &is_cancelled,
    )? {
        Some(policy) => policy,
        None => {
            pause_owned_build(conn, build_id, &worker_token)?;
            return build_summary(conn, build_id, started, false, resumed, true);
        }
    };
    let source_expressions = source_select_expressions(columns);
    let mut build_doc_ids = HashMap::<String, i64>::new();
    {
        let mut statement = conn.prepare(
            "SELECT DISTINCT d.text_sha256, d.doc_id
             FROM _semantic_v2_mapping m
             JOIN _semantic_v2_document d ON d.doc_id = m.doc_id
             WHERE m.build_id = ?1
             ORDER BY d.doc_id",
        )?;
        let rows = statement.query_map([build_id], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
        })?;
        for row in rows {
            let (hash, doc_id) = row?;
            if build_doc_ids.len() as i64 >= limits.mapped_documents.max(0) {
                bail!("semantic build exceeds its mapped-document resource limit");
            }
            build_doc_ids.insert(hash, doc_id);
        }
    }
    let persisted_documents_mapped: i64 = conn.query_row(
        "SELECT documents_mapped FROM _semantic_v2_build WHERE build_id = ?1",
        [build_id],
        |row| row.get(0),
    )?;
    if persisted_documents_mapped != build_doc_ids.len() as i64 {
        bail!(
            "semantic build document counter mismatch: stored {persisted_documents_mapped}, resolved {}",
            build_doc_ids.len()
        );
    }

    let result = (|| -> Result<SemanticIndexSummary> {
        loop {
            if is_cancelled() {
                pause_owned_build(conn, build_id, &worker_token)?;
                return build_summary(conn, build_id, started, false, resumed, true);
            }

            let source_batch =
                load_bounded_source_batch(conn, columns, &source_expressions, cursor)?;
            let source_rows = source_batch.rows;
            let cells_truncated = source_batch.cells_truncated;

            if source_rows.is_empty() {
                if is_cancelled() {
                    continue;
                }
                let now = chrono::Utc::now().to_rfc3339();
                let tx = conn.transaction()?;
                let published = tx.execute(
                    "UPDATE _semantic_v2_build
                 SET status = 'ready', updated_at = ?3, completed_at = ?3, worker_token = NULL
                 WHERE build_id = ?1 AND status = 'building' AND worker_token = ?2",
                    params![build_id, worker_token, now],
                )?;
                if published == 0 {
                    tx.rollback()?;
                    return build_summary(conn, build_id, started, false, resumed, true);
                }
                tx.execute(
                    "INSERT INTO _semantic_v2_active(singleton, build_id) VALUES (1, ?1)
                 ON CONFLICT(singleton) DO UPDATE SET build_id = excluded.build_id",
                    [build_id],
                )?;
                tx.commit()?;
                let _ = prune_stale_semantic_artifacts(conn);
                let summary = build_summary(conn, build_id, started, false, resumed, false)?;
                on_progress(SemanticBuildProgress {
                    build_id,
                    phase: "ready".to_string(),
                    rows_scanned: summary.rows_indexed,
                    rows_total,
                    documents_embedded: summary.documents_indexed,
                    mappings_written: summary.mappings_written,
                    documents_skipped: summary.documents_skipped,
                    mappings_skipped: summary.mappings_skipped,
                    cells_truncated: summary.cells_truncated,
                    columns_omitted: summary.columns_omitted,
                    chunks_omitted: summary.chunks_omitted,
                    resumed_from_row,
                });
                return Ok(summary);
            }

            let (rows_before, mapped_documents_before, mappings_before): (i64, i64, i64) = conn
                .query_row(
                    "SELECT rows_scanned, documents_mapped, mappings_written
                     FROM _semantic_v2_build WHERE build_id = ?1",
                    [build_id],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
                )?;
            let budgeted = budget_documents_v2(
                &plans,
                &source_rows,
                &build_doc_ids,
                rows_before,
                mapped_documents_before,
                mappings_before,
                rows_total,
                limits,
                resource_policy,
            );
            let documents = budgeted.documents;
            let mut existing = documents
                .keys()
                .filter_map(|hash| {
                    build_doc_ids
                        .get(hash)
                        .map(|doc_id| (hash.clone(), *doc_id))
                })
                .collect::<HashMap<_, _>>();
            {
                let mut lookup = conn.prepare(
                    "SELECT doc_id FROM _semantic_v2_document
                     WHERE model_sha256 = ?1 AND tokenizer_sha256 = ?2
                       AND config_sha256 = ?3 AND normalizer_version = ?4
                       AND text_sha256 = ?5",
                )?;
                let unresolved = documents
                    .keys()
                    .filter(|hash| !existing.contains_key(*hash))
                    .cloned()
                    .collect::<Vec<_>>();
                for hash in unresolved {
                    if let Some(doc_id) = lookup
                        .query_row(
                            params![
                                MODEL_SHA256,
                                TOKENIZER_SHA256,
                                CONFIG_SHA256,
                                V2_NORMALIZER_VERSION,
                                &hash,
                            ],
                            |row| row.get(0),
                        )
                        .optional()?
                    {
                        existing.insert(hash, doc_id);
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
                    pause_owned_build(conn, build_id, &worker_token)?;
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
                pause_owned_build(conn, build_id, &worker_token)?;
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
             WHERE build_id = ?1 AND status = 'building' AND cursor_row_num = ?6
               AND worker_token = ?7",
                params![
                    build_id,
                    last_row,
                    source_rows.len() as i64,
                    budgeted.documents_seen,
                    chrono::Utc::now().to_rfc3339(),
                    cursor,
                    worker_token,
                ],
            )?;
            if claimed == 0 {
                tx.rollback()?;
                if !build_is_owned(conn, build_id, &worker_token)? {
                    return build_summary(conn, build_id, started, false, resumed, true);
                }
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
                    model_sha256, tokenizer_sha256, config_sha256, normalizer_version,
                    kind, column_key, text_sha256, normalized_text, embedding
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                )?;
                let mut lookup_doc = tx.prepare(
                    "SELECT doc_id FROM _semantic_v2_document
                 WHERE model_sha256 = ?1 AND tokenizer_sha256 = ?2
                   AND config_sha256 = ?3 AND normalizer_version = ?4
                   AND text_sha256 = ?5",
                )?;
                for (hash, document) in &documents {
                    if !doc_ids.contains_key(hash) {
                        let embedding = embeddings.get(hash).ok_or_else(|| {
                            anyhow::anyhow!("missing semantic document embedding")
                        })?;
                        insert_doc.execute(params![
                            MODEL_SHA256,
                            TOKENIZER_SHA256,
                            CONFIG_SHA256,
                            V2_NORMALIZER_VERSION,
                            document.kind,
                            document.column_key,
                            hash,
                            document.text,
                            embedding,
                        ])?;
                        let doc_id = lookup_doc.query_row(
                            params![
                                MODEL_SHA256,
                                TOKENIZER_SHA256,
                                CONFIG_SHA256,
                                V2_NORMALIZER_VERSION,
                                hash,
                            ],
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
            let expected_mappings = documents
                .values()
                .map(|document| document.rows.len() as i64)
                .sum::<i64>();
            if mappings_added != expected_mappings {
                bail!(
                    "semantic mapping publication wrote {mappings_added} rows, expected {expected_mappings}"
                );
            }
            let updated = tx.execute(
                "UPDATE _semantic_v2_build SET
                 documents_embedded = documents_embedded + ?2,
                 documents_mapped = documents_mapped + ?3,
                 documents_skipped = documents_skipped + ?4,
                 mappings_written = mappings_written + ?5,
                 mappings_skipped = mappings_skipped + ?6,
                 cells_truncated = cells_truncated + ?7,
                 columns_omitted = columns_omitted + ?8,
                 chunks_omitted = chunks_omitted + ?9,
                 updated_at = ?10
              WHERE build_id = ?1 AND status = 'building' AND worker_token = ?11",
                params![
                    build_id,
                    embeddings.len() as i64,
                    budgeted.documents_mapped,
                    budgeted.documents_skipped,
                    mappings_added,
                    budgeted.mappings_skipped,
                    cells_truncated,
                    budgeted.columns_omitted,
                    budgeted.chunks_omitted,
                    chrono::Utc::now().to_rfc3339(),
                    worker_token,
                ],
            )?;
            if updated != 1 {
                bail!("semantic build ownership changed during batch publication");
            }
            tx.commit()?;
            build_doc_ids.extend(doc_ids);
            cursor = last_row;
            let summary = build_summary(conn, build_id, started, false, resumed, false)?;
            on_progress(SemanticBuildProgress {
                build_id,
                phase: "indexing".to_string(),
                rows_scanned: summary.rows_indexed,
                rows_total,
                documents_embedded: summary.documents_indexed,
                mappings_written: summary.mappings_written,
                documents_skipped: summary.documents_skipped,
                mappings_skipped: summary.mappings_skipped,
                cells_truncated: summary.cells_truncated,
                columns_omitted: summary.columns_omitted,
                chunks_omitted: summary.chunks_omitted,
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
            "UPDATE _semantic_v2_build SET status = 'paused', error = ?3, updated_at = ?4
             WHERE build_id = ?1 AND status = 'building' AND worker_token = ?2",
            params![
                build_id,
                worker_token,
                message,
                chrono::Utc::now().to_rfc3339()
            ],
        );
    }
    result
}

fn semantic_v2_status_schema_is_readable(conn: &Connection) -> rusqlite::Result<bool> {
    const ACTIVE_COLUMNS: &[&str] = &["singleton", "build_id"];
    const BUILD_COLUMNS: &[&str] = &[
        "build_id",
        "dataset_hash",
        "schema_hash",
        "index_version",
        "normalizer_version",
        "model_name",
        "model_version",
        "model_sha256",
        "tokenizer_sha256",
        "config_sha256",
        "status",
        "rows_scanned",
        "documents_embedded",
        "mappings_written",
        "documents_skipped",
        "mappings_skipped",
        "cells_truncated",
        "columns_omitted",
        "chunks_omitted",
    ];
    Ok(
        sqlite_table_has_columns(conn, "_semantic_v2_active", ACTIVE_COLUMNS)?
            && sqlite_table_has_columns(conn, "_semantic_v2_build", BUILD_COLUMNS)?,
    )
}

pub fn semantic_index_ready(conn: &Connection, columns: &[ColumnMeta]) -> Result<bool> {
    if !table_exists(conn, "_semantic_v2_active")? || !table_exists(conn, "_semantic_v2_build")? {
        return Ok(false);
    }
    if !semantic_v2_status_schema_is_readable(conn)? {
        return Ok(false);
    }
    let dataset_hash = semantic_dataset_hash(conn, columns)?;
    Ok(active_v2_build(conn, &dataset_hash, &semantic_schema_hash(columns))?.is_some())
}

pub fn semantic_indexed_rows(conn: &Connection, columns: &[ColumnMeta]) -> Result<i64> {
    if !table_exists(conn, "_semantic_v2_active")? || !table_exists(conn, "_semantic_v2_build")? {
        return Ok(0);
    }
    if !semantic_v2_status_schema_is_readable(conn)? {
        return Ok(0);
    }
    let dataset_hash = semantic_dataset_hash(conn, columns)?;
    Ok(
        active_v2_build(conn, &dataset_hash, &semantic_schema_hash(columns))?
            .map(|(_, rows, _, _)| rows)
            .unwrap_or(0),
    )
}

#[derive(Debug, Clone, Copy, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct SemanticIndexCoverage {
    pub documents_skipped: i64,
    pub mappings_skipped: i64,
    pub cells_truncated: i64,
    pub columns_omitted: i64,
    pub chunks_omitted: i64,
}

pub fn semantic_index_coverage(
    conn: &Connection,
    columns: &[ColumnMeta],
) -> Result<Option<SemanticIndexCoverage>> {
    if !table_exists(conn, "_semantic_v2_active")? || !table_exists(conn, "_semantic_v2_build")? {
        return Ok(None);
    }
    if !semantic_v2_status_schema_is_readable(conn)? {
        return Ok(None);
    }
    let dataset_hash = semantic_dataset_hash(conn, columns)?;
    let schema_hash = semantic_schema_hash(columns);
    conn.query_row(
        "SELECT b.documents_skipped, b.mappings_skipped, b.cells_truncated,
                b.columns_omitted, b.chunks_omitted
         FROM _semantic_v2_active a
         JOIN _semantic_v2_build b ON b.build_id = a.build_id
         WHERE a.singleton = 1 AND b.status = 'ready'
           AND b.dataset_hash = ?1 AND b.schema_hash = ?2
           AND b.index_version = ?3 AND b.normalizer_version = ?4
           AND b.model_name = ?5 AND b.model_version = ?6 AND b.model_sha256 = ?7
           AND b.tokenizer_sha256 = ?8 AND b.config_sha256 = ?9",
        params![
            dataset_hash,
            schema_hash,
            V2_INDEX_VERSION,
            V2_NORMALIZER_VERSION,
            MODEL_NAME,
            MODEL_VERSION,
            MODEL_SHA256,
            TOKENIZER_SHA256,
            CONFIG_SHA256,
        ],
        |row| {
            Ok(SemanticIndexCoverage {
                documents_skipped: row.get(0)?,
                mappings_skipped: row.get(1)?,
                cells_truncated: row.get(2)?,
                columns_omitted: row.get(3)?,
                chunks_omitted: row.get(4)?,
            })
        },
    )
    .optional()
    .map_err(Into::into)
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
         WHERE a.singleton = 1 AND b.status = 'ready'
           AND b.index_version = ?1 AND b.normalizer_version = ?2
           AND b.model_name = ?3 AND b.model_version = ?4 AND b.model_sha256 = ?5
           AND b.tokenizer_sha256 = ?6 AND b.config_sha256 = ?7",
        params![
            V2_INDEX_VERSION,
            V2_NORMALIZER_VERSION,
            MODEL_NAME,
            MODEL_VERSION,
            MODEL_SHA256,
            TOKENIZER_SHA256,
            CONFIG_SHA256,
        ],
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
    if !policy.minimum_score.is_finite() {
        bail!("semantic search minimum score must be finite");
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
    let normalized_query = semantic_query_input(query);
    let query_embedding = embedder.embed(&normalized_query)?;
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

fn query_sha256(query: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(query.as_bytes());
    bytes_to_hex(&hasher.finalize())
}

fn semantic_query_input(query: &str) -> String {
    let trimmed = query.trim();
    let normalized = normalize_text(trimmed);
    if normalized.is_empty() {
        trimmed.to_string()
    } else {
        normalized
    }
}

fn new_selection_id(
    build_id: i64,
    dataset_hash: &str,
    query_hash: &str,
    policy: SemanticSearchPolicy,
) -> String {
    let mut hasher = Sha256::new();
    for value in [
        V2_SELECTION_POLICY_VERSION.as_bytes(),
        dataset_hash.as_bytes(),
        query_hash.as_bytes(),
    ] {
        hasher.update((value.len() as u64).to_le_bytes());
        hasher.update(value);
    }
    hasher.update(build_id.to_le_bytes());
    hasher.update((policy.maximum_documents as u64).to_le_bytes());
    hasher.update(policy.minimum_score.to_bits().to_le_bytes());
    bytes_to_hex(&hasher.finalize())
}

fn load_semantic_selection(
    conn: &Connection,
    selection_id: &str,
) -> Result<Option<SemanticSelectionSummary>> {
    let stored = conn
        .query_row(
            "SELECT s.documents_above_threshold, s.documents_retained, s.rows_matched,
                    s.documents_truncated, s.broad_row_warning, s.warnings_json,
                    b.documents_skipped, b.mappings_skipped, b.cells_truncated,
                    b.columns_omitted, b.chunks_omitted
             FROM _semantic_v2_selection s
             JOIN _semantic_v2_build b ON b.build_id = s.build_id
             WHERE s.selection_id = ?1",
            [selection_id],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, bool>(3)?,
                    row.get::<_, bool>(4)?,
                    row.get::<_, String>(5)?,
                    row.get::<_, i64>(6)?,
                    row.get::<_, i64>(7)?,
                    row.get::<_, i64>(8)?,
                    row.get::<_, i64>(9)?,
                    row.get::<_, i64>(10)?,
                ))
            },
        )
        .optional()?;
    let Some((
        documents_above_threshold,
        documents_retained,
        rows_matched,
        documents_truncated,
        broad_row_warning,
        warnings_json,
        index_documents_skipped,
        index_mappings_skipped,
        index_cells_truncated,
        index_columns_omitted,
        index_chunks_omitted,
    )) = stored
    else {
        return Ok(None);
    };
    Ok(Some(SemanticSelectionSummary {
        selection_id: selection_id.to_string(),
        documents_above_threshold: documents_above_threshold.max(0) as usize,
        documents_retained: documents_retained.max(0) as usize,
        rows_matched,
        documents_truncated,
        index_documents_skipped,
        index_mappings_skipped,
        index_cells_truncated,
        index_columns_omitted,
        index_chunks_omitted,
        broad_row_warning,
        warnings: serde_json::from_str(&warnings_json)
            .context("reading semantic selection warnings")?,
    }))
}

fn cleanup_semantic_selections(
    conn: &Connection,
    build_id: i64,
    protected_selection_id: &str,
    maximum_unreferenced: i64,
) -> Result<usize> {
    let keep_other = maximum_unreferenced.saturating_sub(1).max(0);
    let llm_audit_exists = table_exists(conn, "_llm_parse_audit")?;
    let mut protected_clauses = Vec::new();
    if llm_audit_exists {
        protected_clauses.push(
            "EXISTS (
                SELECT 1
                FROM _llm_parse_audit l,
                     json_tree(CASE WHEN json_valid(l.trusted_intent_json)
                                    THEN l.trusted_intent_json ELSE '{}' END) j
                WHERE l.examiner_decision IN ('unreviewed', 'accepted')
                  AND j.key = 'semanticSelectionId'
                  AND j.value = s.selection_id
             )",
        );
    }
    let audit_guard = if protected_clauses.is_empty() {
        String::new()
    } else {
        format!("AND NOT ({})", protected_clauses.join(" OR "))
    };
    let sql = format!(
        "SELECT s.selection_id FROM _semantic_v2_selection s
         WHERE s.build_id = ?1 AND s.selection_id <> ?2 {audit_guard}
         ORDER BY s.created_at DESC, s.selection_id DESC
         LIMIT ?4 OFFSET ?3"
    );
    let victims = {
        let mut statement = conn.prepare(&sql)?;
        let victims = statement
            .query_map(
                params![
                    build_id,
                    protected_selection_id,
                    keep_other,
                    V2_MAX_SELECTION_CLEANUP_PER_REQUEST,
                ],
                |row| row.get::<_, String>(0),
            )?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        victims
    };
    let mut removed = 0usize;
    for selection_id in victims {
        conn.execute(
            "DELETE FROM _semantic_v2_selection_doc WHERE selection_id = ?1",
            [&selection_id],
        )?;
        removed += conn.execute(
            "DELETE FROM _semantic_v2_selection WHERE selection_id = ?1",
            [&selection_id],
        )?;
    }
    Ok(removed)
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
    let query = query.trim();
    if query.is_empty() {
        bail!("semantic search query is empty");
    }
    if query.chars().count() > MAX_QUERY_CHARS {
        bail!("semantic search query exceeds {MAX_QUERY_CHARS} characters");
    }
    if !policy.minimum_score.is_finite() {
        bail!("semantic search minimum score must be finite");
    }
    let policy = SemanticSearchPolicy {
        maximum_documents: policy
            .maximum_documents
            .clamp(1, V2_MAX_DOCUMENT_CANDIDATES),
        minimum_score: policy.minimum_score,
    };
    let query_hash = query_sha256(&semantic_query_input(query));
    let selection_id = new_selection_id(build_id, &dataset_hash, &query_hash, policy);
    if let Some(selection) = load_semantic_selection(conn, &selection_id)? {
        let _ = prune_stale_semantic_artifacts(conn);
        return Ok(selection);
    }
    let (documents, above_threshold) =
        rank_semantic_documents(conn, embedder, build_id, query, policy)?;
    let truncated = above_threshold > documents.len();
    let tx = conn.transaction()?;
    let inserted = tx.execute(
        "INSERT INTO _semantic_v2_selection (
            selection_id, build_id, dataset_hash, query_sha256, policy_version,
            minimum_score, maximum_documents, documents_above_threshold, documents_retained,
            rows_matched, documents_truncated, broad_row_warning, warnings_json, created_at
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, 0, ?10, 0, '[]', ?11)
         ON CONFLICT(selection_id) DO NOTHING",
        params![
            selection_id,
            build_id,
            dataset_hash,
            query_hash,
            V2_SELECTION_POLICY_VERSION,
            policy.minimum_score,
            policy.maximum_documents as i64,
            above_threshold as i64,
            documents.len() as i64,
            i64::from(truncated),
            chrono::Utc::now().to_rfc3339(),
        ],
    )?;
    if inserted == 0 {
        tx.rollback()?;
        return load_semantic_selection(conn, &selection_id)?.ok_or_else(|| {
            anyhow::anyhow!("semantic selection identity collision was not readable")
        });
    }
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
    let (
        index_documents_skipped,
        index_mappings_skipped,
        index_cells_truncated,
        index_columns_omitted,
        index_chunks_omitted,
    ): (i64, i64, i64, i64, i64) = tx.query_row(
        "SELECT documents_skipped, mappings_skipped, cells_truncated, columns_omitted,
                chunks_omitted
         FROM _semantic_v2_build WHERE build_id = ?1",
        [build_id],
        |row| {
            Ok((
                row.get(0)?,
                row.get(1)?,
                row.get(2)?,
                row.get(3)?,
                row.get(4)?,
            ))
        },
    )?;
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
    if index_documents_skipped > 0
        || index_mappings_skipped > 0
        || index_cells_truncated > 0
        || index_columns_omitted > 0
        || index_chunks_omitted > 0
    {
        warnings.push(format!(
            "Semantic indexing was bounded: {index_cells_truncated} oversized cell(s) were truncated, {index_columns_omitted} eligible wide-row column value(s) were omitted, {index_chunks_omitted} chunk document(s) were omitted or truncated, {index_documents_skipped} new-document candidate(s) were skipped, and {index_mappings_skipped} document-to-row mapping candidate(s) were skipped. Exact lexical and structured matching remains complete across every raw row."
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
    cleanup_semantic_selections(&tx, build_id, &selection_id, V2_MAX_SELECTIONS_PER_BUILD)?;
    tx.commit()?;
    let _ = prune_stale_semantic_artifacts(conn);
    Ok(SemanticSelectionSummary {
        selection_id,
        documents_above_threshold: above_threshold,
        documents_retained: documents.len(),
        rows_matched,
        documents_truncated: truncated,
        index_documents_skipped,
        index_mappings_skipped,
        index_cells_truncated,
        index_columns_omitted,
        index_chunks_omitted,
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
    if !table_exists(conn, "_semantic_v2_selection")?
        || !table_exists(conn, "_semantic_v2_active")?
        || !table_exists(conn, "_semantic_v2_build")?
        || !semantic_v2_status_schema_is_readable(conn)?
    {
        bail!("semantic selection is unknown, stale, or belongs to another dataset");
    }
    let expected = semantic_dataset_hash(conn, columns)?;
    let valid: bool = conn.query_row(
        "SELECT EXISTS(
            SELECT 1 FROM _semantic_v2_selection s
            JOIN _semantic_v2_active a ON a.singleton = 1 AND a.build_id = s.build_id
            JOIN _semantic_v2_build b ON b.build_id = s.build_id
            WHERE s.selection_id = ?1 AND s.dataset_hash = ?2 AND b.status = 'ready'
              AND b.index_version = ?3 AND b.normalizer_version = ?4
              AND b.model_name = ?5 AND b.model_version = ?6 AND b.model_sha256 = ?7
              AND b.tokenizer_sha256 = ?8 AND b.config_sha256 = ?9
         )",
        params![
            selection_id,
            expected,
            V2_INDEX_VERSION,
            V2_NORMALIZER_VERSION,
            MODEL_NAME,
            MODEL_VERSION,
            MODEL_SHA256,
            TOKENIZER_SHA256,
            CONFIG_SHA256,
        ],
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
    if selection_id.len() != 64 || !selection_id.chars().all(|ch| ch.is_ascii_hexdigit()) {
        bail!("semantic selection ID is invalid");
    }
    if !table_exists(conn, "_semantic_v2_selection")?
        || !table_exists(conn, "_semantic_v2_active")?
        || !table_exists(conn, "_semantic_v2_build")?
        || !semantic_v2_status_schema_is_readable(conn)?
    {
        bail!("semantic selection is unknown or stale");
    }
    let current: bool = conn.query_row(
        "SELECT EXISTS(
            SELECT 1 FROM _semantic_v2_selection s
            JOIN _semantic_v2_active a ON a.singleton = 1 AND a.build_id = s.build_id
            JOIN _semantic_v2_build b ON b.build_id = s.build_id
            WHERE s.selection_id = ?1 AND b.status = 'ready'
              AND b.index_version = ?2 AND b.normalizer_version = ?3
              AND b.model_name = ?4 AND b.model_version = ?5 AND b.model_sha256 = ?6
              AND b.tokenizer_sha256 = ?7 AND b.config_sha256 = ?8
         )",
        params![
            selection_id,
            V2_INDEX_VERSION,
            V2_NORMALIZER_VERSION,
            MODEL_NAME,
            MODEL_VERSION,
            MODEL_SHA256,
            TOKENIZER_SHA256,
            CONFIG_SHA256,
        ],
        |row| row.get::<_, bool>(0),
    )?;
    if !current {
        bail!("semantic selection is unknown or stale");
    }
    if row_numbers.is_empty() {
        return Ok(HashMap::new());
    }
    let mut reasons = HashMap::<i64, Vec<String>>::new();
    let mut stmt = conn.prepare(
        "SELECT m.row_num, d.normalized_text, sd.cosine_score
         FROM _semantic_v2_selection s
         JOIN _semantic_v2_active a ON a.singleton = 1 AND a.build_id = s.build_id
         JOIN _semantic_v2_build b ON b.build_id = s.build_id
         JOIN _semantic_v2_selection_doc sd ON sd.selection_id = s.selection_id
         JOIN _semantic_v2_mapping m ON m.build_id = s.build_id AND m.doc_id = sd.doc_id
         JOIN _semantic_v2_document d ON d.doc_id = sd.doc_id
         WHERE s.selection_id = ?1 AND m.row_num = ?2 AND b.status = 'ready'
           AND b.index_version = ?3 AND b.normalizer_version = ?4
           AND b.model_name = ?5 AND b.model_version = ?6 AND b.model_sha256 = ?7
           AND b.tokenizer_sha256 = ?8 AND b.config_sha256 = ?9
           AND d.model_sha256 = ?7 AND d.normalizer_version = ?4
           AND d.tokenizer_sha256 = ?8 AND d.config_sha256 = ?9
         ORDER BY sd.cosine_score DESC LIMIT 3",
    )?;
    for row_num in row_numbers {
        let matches = stmt
            .query_map(
                params![
                    selection_id,
                    row_num,
                    V2_INDEX_VERSION,
                    V2_NORMALIZER_VERSION,
                    MODEL_NAME,
                    MODEL_VERSION,
                    MODEL_SHA256,
                    TOKENIZER_SHA256,
                    CONFIG_SHA256,
                ],
                |row| {
                    let text: String = row.get(1)?;
                    let score: f32 = row.get(2)?;
                    Ok(format!("semantic {:.3}: {text}", score))
                },
            )?
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
    fn semantic_audit_row_set_codec_is_compact_deterministic_and_strict() {
        let rows = vec![1, 2, 127, 128, 16_384, i64::MAX];
        let first = encode_sorted_positive_delta_varints(&rows).unwrap();
        let second = encode_sorted_positive_delta_varints(&rows).unwrap();
        assert_eq!(first, second);
        assert_eq!(decode_sorted_positive_delta_varints(&first).unwrap(), rows);
        assert!(first.len() < rows.len() * std::mem::size_of::<i64>());
        assert!(encode_sorted_positive_delta_varints(&[1, 1]).is_err());
        assert!(encode_sorted_positive_delta_varints(&[0]).is_err());
        assert!(decode_sorted_positive_delta_varints(&[0]).is_err());
        assert!(decode_sorted_positive_delta_varints(&[0x80]).is_err());
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
    fn header_classification_uses_exact_tokens_across_snake_camel_and_acronyms() {
        for (sql_name, original_name) in [
            ("process_id", "ProcessId"),
            ("security_sid", "SID"),
            ("event_guid", "EventGUID"),
            ("source_ip", "SourceIP"),
        ] {
            assert!(
                exact_only_header_name(&text_column(sql_name, original_name, 0)),
                "{sql_name}/{original_name} should be exact-only"
            );
        }
        for (sql_name, original_name) in [
            ("identity_note", "Identity Narrative"),
            ("candidate_status", "Candidate Status"),
            ("rapid_detail", "Rapid Detail"),
        ] {
            assert!(
                !exact_only_header_name(&text_column(sql_name, original_name, 0)),
                "{sql_name}/{original_name} must not match the id token"
            );
        }
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
    fn semantic_v2_identity_migration_is_honest_and_idempotent() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE _semantic_v2_build (
                build_id INTEGER PRIMARY KEY,
                dataset_hash TEXT NOT NULL,
                schema_hash TEXT NOT NULL,
                model_sha256 TEXT NOT NULL,
                normalizer_version TEXT NOT NULL,
                status TEXT NOT NULL,
                worker_token TEXT,
                source_rows INTEGER NOT NULL,
                cursor_row_num INTEGER NOT NULL DEFAULT 0,
                rows_scanned INTEGER NOT NULL DEFAULT 0,
                documents_seen INTEGER NOT NULL DEFAULT 0,
                documents_embedded INTEGER NOT NULL DEFAULT 0,
                documents_mapped INTEGER NOT NULL DEFAULT 0,
                documents_skipped INTEGER NOT NULL DEFAULT 0,
                mappings_written INTEGER NOT NULL DEFAULT 0,
                mappings_skipped INTEGER NOT NULL DEFAULT 0,
                cells_truncated INTEGER NOT NULL DEFAULT 0,
                columns_omitted INTEGER NOT NULL DEFAULT 0,
                chunks_omitted INTEGER NOT NULL DEFAULT 0,
                candidate_documents INTEGER,
                candidate_mappings INTEGER,
                candidate_document_limit INTEGER,
                candidate_mapping_limit INTEGER,
                started_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT,
                error TEXT
             );
             CREATE INDEX _semantic_v2_build_identity ON _semantic_v2_build(
                dataset_hash, schema_hash, model_sha256, normalizer_version, status
             );
             CREATE UNIQUE INDEX _semantic_v2_build_unique_identity ON _semantic_v2_build(
                dataset_hash, schema_hash, model_sha256, normalizer_version
             );
             INSERT INTO _semantic_v2_build (
                dataset_hash, schema_hash, model_sha256, normalizer_version, status,
                source_rows, started_at, updated_at
             ) VALUES ('legacy-dataset', 'legacy-schema', 'legacy-model',
                       'legacy-normalizer', 'ready', 1, 'then', 'then');
             CREATE TABLE _semantic_v2_document (
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
             INSERT INTO _semantic_v2_document (
                model_sha256, normalizer_version, kind, column_key, text_sha256,
                normalized_text, embedding
             ) VALUES ('legacy-model', 'legacy-normalizer', 'cell', 'message',
                       'same-text', 'legacy evidence', X'00000000');",
        )
        .unwrap();

        create_semantic_v2_schema(&conn).unwrap();
        let build_identity: (String, String, String, String, String) = conn
            .query_row(
                "SELECT index_version, model_name, model_version, tokenizer_sha256,
                        config_sha256
                 FROM _semantic_v2_build WHERE dataset_hash = 'legacy-dataset'",
                [],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                    ))
                },
            )
            .unwrap();
        assert_eq!(
            build_identity,
            (
                V2_LEGACY_UNRECORDED_IDENTITY.to_string(),
                V2_LEGACY_UNRECORDED_IDENTITY.to_string(),
                V2_LEGACY_UNRECORDED_IDENTITY.to_string(),
                V2_LEGACY_UNRECORDED_IDENTITY.to_string(),
                V2_LEGACY_UNRECORDED_IDENTITY.to_string(),
            )
        );
        let expected_identity = vec![
            "dataset_hash",
            "schema_hash",
            "index_version",
            "normalizer_version",
            "model_name",
            "model_version",
            "model_sha256",
            "tokenizer_sha256",
            "config_sha256",
        ];
        assert_eq!(
            sqlite_index_columns(&conn, "_semantic_v2_build_unique_identity")
                .unwrap()
                .unwrap(),
            expected_identity
        );
        let document_identity: (String, String) = conn
            .query_row(
                "SELECT tokenizer_sha256, config_sha256
                 FROM _semantic_v2_document WHERE text_sha256 = 'same-text'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(
            document_identity,
            (
                V2_LEGACY_UNRECORDED_IDENTITY.to_string(),
                V2_LEGACY_UNRECORDED_IDENTITY.to_string(),
            )
        );
        conn.execute(
            "INSERT INTO _semantic_v2_document (
                model_sha256, tokenizer_sha256, config_sha256, normalizer_version,
                kind, column_key, text_sha256, normalized_text, embedding
             ) VALUES ('legacy-model', ?1, ?2, 'legacy-normalizer', 'cell', 'message',
                       'same-text', 'fresh evidence', X'00000000')",
            params![TOKENIZER_SHA256, CONFIG_SHA256],
        )
        .unwrap();
        assert_eq!(
            conn.query_row("SELECT COUNT(*) FROM _semantic_v2_document", [], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap(),
            2
        );

        let schema_version: i64 = conn
            .query_row("PRAGMA schema_version", [], |row| row.get(0))
            .unwrap();
        create_semantic_v2_schema(&conn).unwrap();
        let repeated_schema_version: i64 = conn
            .query_row("PRAGMA schema_version", [], |row| row.get(0))
            .unwrap();
        assert_eq!(repeated_schema_version, schema_version);
    }

    #[test]
    fn legacy_status_reads_not_ready_then_build_path_migrates_and_rebuilds() {
        let mut conn = Connection::open_in_memory().unwrap();
        let columns = message_columns();
        populate_messages(&mut conn, &columns, 1, false);
        let dataset_hash = semantic_dataset_hash(&conn, &columns).unwrap();
        let schema_hash = semantic_schema_hash(&columns);
        conn.execute_batch(
            "CREATE TABLE _semantic_v2_build (
                build_id INTEGER PRIMARY KEY,
                dataset_hash TEXT NOT NULL,
                schema_hash TEXT NOT NULL,
                model_sha256 TEXT NOT NULL,
                normalizer_version TEXT NOT NULL,
                status TEXT NOT NULL,
                worker_token TEXT,
                source_rows INTEGER NOT NULL,
                cursor_row_num INTEGER NOT NULL DEFAULT 0,
                rows_scanned INTEGER NOT NULL DEFAULT 0,
                documents_seen INTEGER NOT NULL DEFAULT 0,
                documents_embedded INTEGER NOT NULL DEFAULT 0,
                documents_mapped INTEGER NOT NULL DEFAULT 0,
                documents_skipped INTEGER NOT NULL DEFAULT 0,
                mappings_written INTEGER NOT NULL DEFAULT 0,
                mappings_skipped INTEGER NOT NULL DEFAULT 0,
                cells_truncated INTEGER NOT NULL DEFAULT 0,
                columns_omitted INTEGER NOT NULL DEFAULT 0,
                chunks_omitted INTEGER NOT NULL DEFAULT 0,
                candidate_documents INTEGER,
                candidate_mappings INTEGER,
                candidate_document_limit INTEGER,
                candidate_mapping_limit INTEGER,
                started_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT,
                error TEXT
             );
             CREATE UNIQUE INDEX _semantic_v2_build_unique_identity ON _semantic_v2_build(
                dataset_hash, schema_hash, model_sha256, normalizer_version
             );
             CREATE TABLE _semantic_v2_active (
                singleton INTEGER PRIMARY KEY,
                build_id INTEGER NOT NULL
             );
             CREATE TABLE _semantic_v2_document (
                doc_id INTEGER PRIMARY KEY,
                model_sha256 TEXT NOT NULL,
                normalizer_version TEXT NOT NULL,
                kind TEXT NOT NULL,
                column_key TEXT NOT NULL,
                text_sha256 TEXT NOT NULL,
                normalized_text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                UNIQUE(model_sha256, normalizer_version, text_sha256)
             );",
        )
        .unwrap();
        conn.execute(
            "INSERT INTO _semantic_v2_build (
                dataset_hash, schema_hash, model_sha256, normalizer_version, status,
                source_rows, cursor_row_num, rows_scanned, started_at, updated_at,
                completed_at
             ) VALUES (?1, ?2, ?3, ?4, 'ready', 1, 1, 1, 'then', 'then', 'then')",
            params![
                dataset_hash,
                schema_hash,
                MODEL_SHA256,
                V2_NORMALIZER_VERSION,
            ],
        )
        .unwrap();
        let legacy_build = conn.last_insert_rowid();
        conn.execute(
            "INSERT INTO _semantic_v2_active(singleton, build_id) VALUES (1, ?1)",
            [legacy_build],
        )
        .unwrap();

        assert!(!semantic_index_ready(&conn, &columns).unwrap());
        assert_eq!(semantic_indexed_rows(&conn, &columns).unwrap(), 0);
        assert!(semantic_index_coverage(&conn, &columns).unwrap().is_none());

        let embedder = FakeEmbedder::default();
        let summary =
            ensure_semantic_index_v2(&mut conn, &columns, &embedder, || false, |_| {}).unwrap();
        assert!(!summary.from_cache);
        assert!(semantic_index_ready(&conn, &columns).unwrap());
        assert_eq!(semantic_indexed_rows(&conn, &columns).unwrap(), 1);
        assert!(semantic_index_coverage(&conn, &columns).unwrap().is_some());
        assert_ne!(active_build_identity(&conn).unwrap().0, legacy_build);
    }

    #[test]
    fn v2_normalizer_deduplicates_dynamic_ids_and_never_starves_the_final_column() {
        let conn = Connection::open_in_memory().unwrap();
        let mut description = text_column("final_evidence", "Evidence Description", 2);
        description.inferred_type = "ip".to_string();
        let columns = vec![
            text_column("event_id", "Event GUID", 0),
            text_column("verbose_message", "Message", 1),
            description,
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
        assert_eq!(plans[2].mode, ColumnMode::Text);
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
    fn v2_classifier_keeps_unknown_high_cardinality_entities_exact_only() {
        let conn = Connection::open_in_memory().unwrap();
        let columns = vec![
            text_column("principal", "Principal", 0),
            text_column("details", "Event Description", 1),
        ];
        db::create_schema(&conn, &columns).unwrap();
        for index in 0..32 {
            conn.execute(
                "INSERT INTO rows(row_num, principal, details) VALUES (?1, ?2, ?3)",
                params![
                    index as i64 + 1,
                    format!("user{}", alphabetic_id(index as usize)),
                    format!(
                        "investigator narrative describes credential access variant {} in detail",
                        alphabetic_id(index as usize)
                    )
                ],
            )
            .unwrap();
        }
        let plans = classify_columns(&conn, &columns).unwrap();
        assert_eq!(plans[0].mode, ColumnMode::ExactOnly);
        assert_eq!(plans[1].mode, ColumnMode::Text);
    }

    #[test]
    fn v2_wide_rows_report_omissions_and_keep_the_final_eligible_column() {
        let plans = (0..60)
            .map(|index| ColumnPlan {
                col_index: index,
                sql_name: format!("field_{index}"),
                original_name: format!("Narrative Field {index}"),
                mode: ColumnMode::Text,
            })
            .collect::<Vec<_>>();
        let values = (0..60)
            .map(|index| Some(format!("security evidence value {index}")))
            .collect::<Vec<_>>();
        let output = row_documents_with_stats_v2(&plans, &values);
        assert_eq!(output.eligible_columns_omitted, 12);
        assert_eq!(output.documents.len(), V2_MAX_PRIMARY_DOCUMENTS_PER_ROW);
        assert_eq!(output.documents.last().unwrap().1, "field_59");
    }

    #[test]
    fn v2_reports_cell_and_row_chunk_omissions_exactly() {
        let single_plan = vec![ColumnPlan {
            col_index: 0,
            sql_name: "message".to_string(),
            original_name: "Message".to_string(),
            mode: ColumnMode::Text,
        }];
        let long_under_input_cap = (0..400)
            .map(|index| format!("evidence{}", alphabetic_id(index)))
            .collect::<Vec<_>>()
            .join(" ");
        assert!(long_under_input_cap.chars().count() < V2_MAX_CELL_INPUT_CHARS);
        let single = row_documents_with_stats_v2(&single_plan, &[Some(long_under_input_cap)]);
        assert_eq!(single.chunks_omitted, 3);

        let plans = (0..10)
            .map(|index| ColumnPlan {
                col_index: index,
                sql_name: format!("message_{index}"),
                original_name: format!("Message {index}"),
                mode: ColumnMode::Text,
            })
            .collect::<Vec<_>>();
        let multi_chunk = (0..200)
            .map(|index| format!("token{}", alphabetic_id(index)))
            .collect::<Vec<_>>()
            .join(" ");
        let values = (0..plans.len())
            .map(|_| Some(multi_chunk.clone()))
            .collect::<Vec<_>>();
        let multiple = row_documents_with_stats_v2(&plans, &values);
        assert_eq!(multiple.chunks_omitted, 15);
        assert_eq!(multiple.documents.len(), 25);
    }

    #[test]
    fn v2_persists_and_discloses_chunk_omissions() {
        let mut conn = Connection::open_in_memory().unwrap();
        let columns = message_columns();
        db::create_schema(&conn, &columns).unwrap();
        let long_under_input_cap = (0..400)
            .map(|index| format!("evidence{}", alphabetic_id(index)))
            .collect::<Vec<_>>()
            .join(" ");
        conn.execute(
            "INSERT INTO rows(row_num, event_id, message) VALUES (1, 'event-one', ?1)",
            [&long_under_input_cap],
        )
        .unwrap();
        let embedder = FakeEmbedder::default();
        let summary =
            ensure_semantic_index_v2(&mut conn, &columns, &embedder, || false, |_| {}).unwrap();
        assert_eq!(summary.cells_truncated, 0);
        assert_eq!(summary.chunks_omitted, 3);
        assert!(summary.truncated);
        assert_eq!(
            semantic_index_coverage(&conn, &columns)
                .unwrap()
                .unwrap()
                .chunks_omitted,
            3
        );
        let selection = create_semantic_selection(
            &mut conn,
            &columns,
            &embedder,
            "security evidence",
            SemanticSearchPolicy {
                maximum_documents: 4,
                minimum_score: -1.0,
            },
        )
        .unwrap();
        assert_eq!(selection.index_chunks_omitted, 3);
        assert!(selection
            .warnings
            .iter()
            .any(|warning| warning.contains("3 chunk document")));
    }

    #[test]
    fn v2_under_cap_deduplicated_dataset_has_no_candidate_skips() {
        let mut conn = Connection::open_in_memory().unwrap();
        let columns = message_columns();
        db::create_schema(&conn, &columns).unwrap();
        let tx = conn.transaction().unwrap();
        for index in 0..120i64 {
            tx.execute(
                "INSERT INTO rows(row_num, event_id, message) VALUES (?1, ?2, ?3)",
                params![
                    index + 1,
                    format!("event-{index}"),
                    format!(
                        "credential access evidence variant {}",
                        alphabetic_id(index as usize % 15)
                    )
                ],
            )
            .unwrap();
        }
        tx.commit().unwrap();
        let embedder = FakeEmbedder::default();
        let summary = ensure_semantic_index_v2_with_limits(
            &mut conn,
            &columns,
            &embedder,
            || false,
            |_| {},
            SemanticResourceLimits {
                mapped_documents: 15,
                mappings: 120,
            },
        )
        .unwrap();
        assert_eq!(summary.documents_mapped, 15);
        assert_eq!(summary.mappings_written, 120);
        assert_eq!(summary.documents_skipped, 0);
        assert_eq!(summary.mappings_skipped, 0);
    }

    #[test]
    fn v2_build_caps_are_balanced_to_the_final_row_and_report_exact_skips() {
        let mut conn = Connection::open_in_memory().unwrap();
        let columns = message_columns();
        populate_messages(&mut conn, &columns, 5, true);
        let embedder = FakeEmbedder::default();
        let summary = ensure_semantic_index_v2_with_limits(
            &mut conn,
            &columns,
            &embedder,
            || false,
            |_| {},
            SemanticResourceLimits {
                mapped_documents: 2,
                mappings: 5,
            },
        )
        .unwrap();
        assert_eq!(summary.rows_indexed, 5);
        assert_eq!(summary.documents_mapped, 2);
        assert_eq!(summary.mappings_written, 2);
        assert_eq!(summary.documents_skipped, 3);
        assert_eq!(summary.mappings_skipped, 3);
        assert!(summary.truncated);
        let mapped_rows = conn
            .prepare("SELECT DISTINCT row_num FROM _semantic_v2_mapping ORDER BY row_num")
            .unwrap()
            .query_map([], |row| row.get::<_, i64>(0))
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap();
        assert_eq!(mapped_rows, vec![3, 5]);

        let selection = create_semantic_selection(
            &mut conn,
            &columns,
            &embedder,
            "suspicious process",
            SemanticSearchPolicy {
                maximum_documents: 2,
                minimum_score: -1.0,
            },
        )
        .unwrap();
        assert_eq!(selection.index_documents_skipped, 3);
        assert_eq!(selection.index_mappings_skipped, 3);
        assert!(selection
            .warnings
            .iter()
            .any(|warning| warning.contains("3 new-document")
                && warning.contains("3 document-to-row")));
    }

    #[test]
    fn v2_fetch_and_documents_bound_oversized_cells_before_embedding() {
        let mut conn = Connection::open_in_memory().unwrap();
        let columns = message_columns();
        db::create_schema(&conn, &columns).unwrap();
        let huge = (0..50_000)
            .map(|index| format!("evidence{index}"))
            .collect::<Vec<_>>()
            .join(" ");
        conn.execute(
            "INSERT INTO rows(row_num, event_id, message) VALUES (1, 'event-one', ?1)",
            [&huge],
        )
        .unwrap();
        let embedder = FakeEmbedder::default();
        let summary =
            ensure_semantic_index_v2(&mut conn, &columns, &embedder, || false, |_| {}).unwrap();
        assert_eq!(summary.cells_truncated, 1);
        assert!(summary.truncated);
        assert!(embedder
            .seen
            .lock()
            .unwrap()
            .iter()
            .all(|text| text.chars().count() <= V2_MAX_DOCUMENT_CHARS));
        let longest: i64 = conn
            .query_row(
                "SELECT MAX(length(normalized_text)) FROM _semantic_v2_document",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(longest <= V2_MAX_DOCUMENT_CHARS as i64);
    }

    #[test]
    fn stale_normalizer_build_is_not_reused_or_published() {
        let mut conn = Connection::open_in_memory().unwrap();
        let columns = message_columns();
        populate_messages(&mut conn, &columns, 1, false);
        create_semantic_v2_schema(&conn).unwrap();
        let dataset_hash = semantic_dataset_hash(&conn, &columns).unwrap();
        let schema_hash = semantic_schema_hash(&columns);
        let now = chrono::Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO _semantic_v2_build (
                dataset_hash, schema_hash, model_sha256, normalizer_version, status,
                source_rows, cursor_row_num, rows_scanned, started_at, updated_at, completed_at
             ) VALUES (?1, ?2, ?3, 'dfir-cell-normalizer-v2', 'ready', 1, 1, 1, ?4, ?4, ?4)",
            params![dataset_hash, schema_hash, MODEL_SHA256, now],
        )
        .unwrap();
        let stale_build = conn.last_insert_rowid();
        conn.execute(
            "INSERT INTO _semantic_v2_active(singleton, build_id) VALUES (1, ?1)",
            [stale_build],
        )
        .unwrap();

        let embedder = FakeEmbedder::default();
        let summary =
            ensure_semantic_index_v2(&mut conn, &columns, &embedder, || false, |_| {}).unwrap();
        assert!(!summary.from_cache);
        let (active_build, active_version): (i64, String) = conn
            .query_row(
                "SELECT b.build_id, b.normalizer_version
                 FROM _semantic_v2_active a JOIN _semantic_v2_build b ON b.build_id = a.build_id",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_ne!(active_build, stale_build);
        assert_eq!(active_version, V2_NORMALIZER_VERSION);
    }

    #[test]
    fn legacy_unrecorded_pipeline_is_rebuilt_without_reusing_its_embeddings() {
        let mut conn = Connection::open_in_memory().unwrap();
        let columns = message_columns();
        populate_messages(&mut conn, &columns, 1, false);
        create_semantic_v2_schema(&conn).unwrap();
        let dataset_hash = semantic_dataset_hash(&conn, &columns).unwrap();
        let schema_hash = semantic_schema_hash(&columns);
        let now = chrono::Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO _semantic_v2_build (
                dataset_hash, schema_hash, model_sha256, normalizer_version, status,
                source_rows, cursor_row_num, rows_scanned, documents_seen,
                documents_embedded, documents_mapped, mappings_written,
                started_at, updated_at, completed_at
             ) VALUES (?1, ?2, ?3, ?4, 'ready', 1, 1, 1, 1, 1, 1, 1, ?5, ?5, ?5)",
            params![
                dataset_hash,
                schema_hash,
                MODEL_SHA256,
                V2_NORMALIZER_VERSION,
                now,
            ],
        )
        .unwrap();
        let legacy_build = conn.last_insert_rowid();
        let legacy_fingerprint =
            text_sha256("cell", "message", "credential dumping process observed");
        conn.execute(
            "INSERT INTO _semantic_v2_document (
                model_sha256, normalizer_version, kind, column_key, text_sha256,
                normalized_text, embedding
             ) VALUES (?1, ?2, 'cell', 'message', ?3, ?4, ?5)",
            params![
                MODEL_SHA256,
                V2_NORMALIZER_VERSION,
                legacy_fingerprint,
                "credential dumping process observed",
                vector_to_blob(&FakeEmbedder::vector()),
            ],
        )
        .unwrap();
        let legacy_document = conn.last_insert_rowid();
        conn.execute(
            "INSERT INTO _semantic_v2_mapping(build_id, doc_id, row_num) VALUES (?1, ?2, 1)",
            params![legacy_build, legacy_document],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO _semantic_v2_active(singleton, build_id) VALUES (1, ?1)",
            [legacy_build],
        )
        .unwrap();

        let embedder = FakeEmbedder::default();
        let summary =
            ensure_semantic_index_v2(&mut conn, &columns, &embedder, || false, |_| {}).unwrap();
        assert!(!summary.from_cache);
        assert!(embedder.call_count() > 0);
        let (active_build, index_version, tokenizer_sha256, config_sha256): (
            i64,
            String,
            String,
            String,
        ) = conn
            .query_row(
                "SELECT b.build_id, b.index_version, b.tokenizer_sha256, b.config_sha256
                 FROM _semantic_v2_active a
                 JOIN _semantic_v2_build b ON b.build_id = a.build_id
                 WHERE a.singleton = 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .unwrap();
        assert_ne!(active_build, legacy_build);
        assert_eq!(index_version, V2_INDEX_VERSION);
        assert_eq!(tokenizer_sha256, TOKENIZER_SHA256);
        assert_eq!(config_sha256, CONFIG_SHA256);
        let active_document: i64 = conn
            .query_row(
                "SELECT d.doc_id
                 FROM _semantic_v2_mapping m
                 JOIN _semantic_v2_document d ON d.doc_id = m.doc_id
                 WHERE m.build_id = ?1 AND d.model_sha256 = ?2
                   AND d.tokenizer_sha256 = ?3 AND d.config_sha256 = ?4
                   AND d.normalizer_version = ?5
                 LIMIT 1",
                params![
                    active_build,
                    MODEL_SHA256,
                    TOKENIZER_SHA256,
                    CONFIG_SHA256,
                    V2_NORMALIZER_VERSION,
                ],
                |row| row.get(0),
            )
            .unwrap();
        assert_ne!(active_document, legacy_document);
    }

    #[test]
    fn stale_audited_selection_is_snapshotted_exactly_before_bounded_reclamation() {
        let mut conn = Connection::open_in_memory().unwrap();
        create_semantic_v2_schema(&conn).unwrap();
        conn.execute_batch(
            "CREATE TABLE _llm_parse_audit (
                trusted_intent_json TEXT NOT NULL,
                examiner_decision TEXT NOT NULL
             );
             CREATE TABLE _semantic_retrieval_audit(selection_id TEXT);",
        )
        .unwrap();
        let now = "2026-07-17T00:00:00Z";
        let snapshot_index = "snapshot-index";
        let snapshot_normalizer = "snapshot-normalizer";
        let snapshot_model_name = "snapshot-model";
        let snapshot_model_version = "snapshot-model-version";
        let snapshot_model_sha = "e".repeat(64);
        let snapshot_tokenizer_sha = "f".repeat(64);
        let snapshot_config_sha = "0".repeat(64);
        let insert_build = |conn: &Connection, dataset: &str, schema: &str| -> i64 {
            conn.execute(
                "INSERT INTO _semantic_v2_build (
                    dataset_hash, schema_hash, index_version, normalizer_version,
                    model_name, model_version, model_sha256, tokenizer_sha256,
                    config_sha256, status, source_rows, cursor_row_num, rows_scanned, documents_seen,
                    documents_embedded, documents_mapped, mappings_written,
                    documents_skipped, mappings_skipped, cells_truncated, columns_omitted,
                    chunks_omitted, started_at, updated_at, completed_at
                 ) VALUES (
                    ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, 'ready', 8, 8, 8, 4, 4, 4, 8,
                    2, 3, 4, 5, 6, ?10, ?10, ?10
                 )",
                params![
                    dataset,
                    schema,
                    snapshot_index,
                    snapshot_normalizer,
                    snapshot_model_name,
                    snapshot_model_version,
                    snapshot_model_sha,
                    snapshot_tokenizer_sha,
                    snapshot_config_sha,
                    now,
                ],
            )
            .unwrap();
            conn.last_insert_rowid()
        };
        let accepted_build = insert_build(&conn, "accepted-dataset", "accepted-schema");
        let rejected_build = insert_build(&conn, "rejected-dataset", "rejected-schema");
        let retrieval_build = insert_build(&conn, "retrieval-dataset", "retrieval-schema");
        let active_build = insert_build(&conn, "active-dataset", "active-schema");
        conn.execute(
            "INSERT INTO _semantic_v2_active(singleton, build_id) VALUES (1, ?1)",
            [active_build],
        )
        .unwrap();

        let embedding = vector_to_blob(&FakeEmbedder::vector());
        let insert_document = |conn: &Connection, fingerprint: &str, text: &str| -> i64 {
            conn.execute(
                "INSERT INTO _semantic_v2_document (
                        model_sha256, tokenizer_sha256, config_sha256, normalizer_version,
                        kind, column_key, text_sha256, normalized_text, embedding
                     ) VALUES (?1, ?2, ?3, ?4, 'cell', 'message', ?5, ?6, ?7)",
                params![
                    snapshot_model_sha,
                    snapshot_tokenizer_sha,
                    snapshot_config_sha,
                    snapshot_normalizer,
                    fingerprint,
                    text,
                    embedding,
                ],
            )
            .unwrap();
            conn.last_insert_rowid()
        };
        let accepted_doc_one = insert_document(
            &conn,
            &format!("{:064x}", 1),
            "credential dumping process observed",
        );
        let accepted_doc_two = insert_document(
            &conn,
            &format!("{:064x}", 2),
            "powershell downloaded a payload",
        );
        let rejected_doc =
            insert_document(&conn, &format!("{:064x}", 3), "rejected semantic document");
        let retrieval_doc = insert_document(
            &conn,
            &format!("{:064x}", 4),
            "retrieval audit only document",
        );
        for (build_id, doc_id, rows) in [
            (accepted_build, accepted_doc_one, vec![1, 3, 5]),
            (accepted_build, accepted_doc_two, vec![2, 3, 8]),
            (rejected_build, rejected_doc, vec![4]),
            (retrieval_build, retrieval_doc, vec![6]),
        ] {
            for row_num in rows {
                conn.execute(
                    "INSERT INTO _semantic_v2_mapping(build_id, doc_id, row_num)
                     VALUES (?1, ?2, ?3)",
                    params![build_id, doc_id, row_num],
                )
                .unwrap();
            }
        }

        let accepted_selection = "a".repeat(64);
        let rejected_selection = "b".repeat(64);
        let retrieval_selection = "c".repeat(64);
        let insert_selection = |conn: &Connection,
                                selection_id: &str,
                                build_id: i64,
                                dataset: &str,
                                query_sha: &str,
                                documents: i64,
                                rows: i64| {
            conn.execute(
                "INSERT INTO _semantic_v2_selection (
                    selection_id, build_id, dataset_hash, query_sha256, policy_version,
                    minimum_score, maximum_documents, documents_above_threshold,
                    documents_retained, rows_matched, documents_truncated,
                    broad_row_warning, warnings_json, created_at
                 ) VALUES (?1, ?2, ?3, ?4, 'snapshot-policy', 0.25, 7, ?5, ?5, ?6,
                           1, 1, '[\"bounded semantic evidence\"]', ?7)",
                params![
                    selection_id,
                    build_id,
                    dataset,
                    query_sha,
                    documents,
                    rows,
                    now,
                ],
            )
            .unwrap();
        };
        insert_selection(
            &conn,
            &accepted_selection,
            accepted_build,
            "accepted-dataset",
            &"d".repeat(64),
            2,
            5,
        );
        insert_selection(
            &conn,
            &rejected_selection,
            rejected_build,
            "rejected-dataset",
            &"e".repeat(64),
            1,
            1,
        );
        insert_selection(
            &conn,
            &retrieval_selection,
            retrieval_build,
            "retrieval-dataset",
            &"f".repeat(64),
            1,
            1,
        );
        for (selection_id, doc_id, cosine, rank) in [
            (&accepted_selection, accepted_doc_one, 0.91, 0.95),
            (&accepted_selection, accepted_doc_two, 0.82, 0.85),
            (&rejected_selection, rejected_doc, 0.75, 0.75),
            (&retrieval_selection, retrieval_doc, 0.70, 0.70),
        ] {
            conn.execute(
                "INSERT INTO _semantic_v2_selection_doc (
                    selection_id, doc_id, cosine_score, rank_score
                 ) VALUES (?1, ?2, ?3, ?4)",
                params![selection_id, doc_id, cosine, rank],
            )
            .unwrap();
        }
        for (selection_id, decision) in [
            (&accepted_selection, "accepted"),
            (&rejected_selection, "rejected"),
        ] {
            conn.execute(
                "INSERT INTO _llm_parse_audit(trusted_intent_json, examiner_decision)
                 VALUES (?1, ?2)",
                params![
                    serde_json::json!({
                        "intent": "rawEvidenceSearch",
                        "semanticSelectionId": selection_id,
                    })
                    .to_string(),
                    decision,
                ],
            )
            .unwrap();
        }
        conn.execute(
            "INSERT INTO _semantic_retrieval_audit(selection_id) VALUES (?1)",
            [&retrieval_selection],
        )
        .unwrap();

        // Initialize the accepted archive, then inject a failure after mapping chunks/row-union
        // inserts but before cursor publication. The encompassing transaction must make that
        // interruption indistinguishable from never starting the batch.
        {
            let tx = conn.transaction().unwrap();
            initialize_audit_snapshot_stage(&tx, &accepted_selection).unwrap();
            tx.commit().unwrap();
        }
        {
            let tx = conn.transaction().unwrap();
            let stage = load_audit_snapshot_stage(&tx, &accepted_selection)
                .unwrap()
                .unwrap();
            let error = advance_audit_snapshot_mappings_with_hook(&tx, &stage, || {
                bail!("synthetic interruption before snapshot cursor publication")
            })
            .unwrap_err();
            assert!(error.to_string().contains("synthetic interruption"));
            tx.rollback().unwrap();
        }
        let stage_after_rollback = load_audit_snapshot_stage(&conn, &accepted_selection)
            .unwrap()
            .unwrap();
        assert_eq!(stage_after_rollback.cursor_doc_id, 0);
        assert_eq!(stage_after_rollback.cursor_row_num, 0);
        assert_eq!(stage_after_rollback.mappings_seen, 0);
        for table in [
            "_semantic_v2_audit_snapshot_stage_mapping_chunk",
            "_semantic_v2_audit_snapshot_stage_row",
        ] {
            let count: i64 = conn
                .query_row(
                    &format!("SELECT COUNT(*) FROM {}", db::quote_ident(table)),
                    [],
                    |row| row.get(0),
                )
                .unwrap();
            assert_eq!(count, 0, "{table} must roll back with its cursor");
        }

        let mut converged = false;
        for _ in 0..64 {
            if prune_stale_semantic_artifacts_pass(&mut conn).unwrap() == 0 {
                converged = true;
                break;
            }
        }
        assert!(converged, "bounded snapshot/prune passes must converge");

        let header: (
            String,
            i64,
            String,
            String,
            i64,
            i64,
            String,
            String,
            String,
        ) = conn
            .query_row(
                "SELECT snapshot_version, build_id, dataset_hash, query_sha256,
                        selected_document_count, mapping_count, mapping_sha256,
                        row_set_sha256, warnings_json
                 FROM _semantic_v2_audit_snapshot WHERE selection_id = ?1",
                [&accepted_selection],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                        row.get(5)?,
                        row.get(6)?,
                        row.get(7)?,
                        row.get(8)?,
                    ))
                },
            )
            .unwrap();
        assert_eq!(header.0, V2_AUDIT_SNAPSHOT_VERSION);
        assert_eq!(header.1, accepted_build);
        assert_eq!(header.2, "accepted-dataset");
        assert_eq!(header.3, "d".repeat(64));
        assert_eq!(header.4, 2);
        assert_eq!(header.5, 6);
        assert_eq!(header.6.len(), 64);
        assert_eq!(header.7.len(), 64);
        assert_eq!(header.8, "[\"bounded semantic evidence\"]");
        let identity: (String, String, String, String, String, String, String) = conn
            .query_row(
                "SELECT index_version, normalizer_version, model_name, model_version,
                        model_sha256, tokenizer_sha256, config_sha256
                 FROM _semantic_v2_audit_snapshot WHERE selection_id = ?1",
                [&accepted_selection],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                        row.get(5)?,
                        row.get(6)?,
                    ))
                },
            )
            .unwrap();
        assert_eq!(identity.0, snapshot_index);
        assert_eq!(identity.1, snapshot_normalizer);
        assert_eq!(identity.2, snapshot_model_name);
        assert_eq!(identity.3, snapshot_model_version);
        assert_eq!(identity.4, snapshot_model_sha);
        assert_eq!(identity.5, snapshot_tokenizer_sha);
        assert_eq!(identity.6, snapshot_config_sha);

        let documents = {
            let mut statement = conn
                .prepare(
                    "SELECT rank, fingerprint_sha256, normalized_text, cosine_score,
                            rank_score, mapping_count, mapping_sha256
                     FROM _semantic_v2_audit_snapshot_document
                     WHERE selection_id = ?1 ORDER BY rank",
                )
                .unwrap();
            let collected = statement
                .query_map([&accepted_selection], |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, f64>(3)?,
                        row.get::<_, f64>(4)?,
                        row.get::<_, i64>(5)?,
                        row.get::<_, String>(6)?,
                    ))
                })
                .unwrap()
                .collect::<rusqlite::Result<Vec<_>>>()
                .unwrap();
            collected
        };
        assert_eq!(documents.len(), 2);
        assert_eq!(documents[0].0, 1);
        assert_eq!(documents[0].1, format!("{:064x}", 1));
        assert_eq!(documents[0].2, "credential dumping process observed");
        assert!((documents[0].3 - 0.91).abs() < f64::EPSILON);
        assert!((documents[0].4 - 0.95).abs() < f64::EPSILON);
        assert_eq!(documents[0].5, 3);
        assert_eq!(documents[0].6.len(), 64);
        assert_eq!(documents[1].5, 3);

        let encoded_chunks = {
            let mut statement = conn
                .prepare(
                    "SELECT encoded_rows FROM _semantic_v2_audit_snapshot_row_chunk
                     WHERE selection_id = ?1 ORDER BY chunk_index",
                )
                .unwrap();
            let collected = statement
                .query_map([&accepted_selection], |row| row.get::<_, Vec<u8>>(0))
                .unwrap()
                .collect::<rusqlite::Result<Vec<_>>>()
                .unwrap();
            collected
        };
        let decoded_rows = encoded_chunks
            .iter()
            .flat_map(|chunk| decode_sorted_positive_delta_varints(chunk).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(decoded_rows, vec![1, 2, 3, 5, 8]);

        for selection_id in [&rejected_selection, &retrieval_selection] {
            let snapshotted: bool = conn
                .query_row(
                    "SELECT EXISTS(
                        SELECT 1 FROM _semantic_v2_audit_snapshot WHERE selection_id = ?1
                     )",
                    [selection_id],
                    |row| row.get(0),
                )
                .unwrap();
            assert!(
                !snapshotted,
                "{selection_id} must not receive a protected snapshot"
            );
        }
        for table in [
            "_semantic_v2_selection",
            "_semantic_v2_selection_doc",
            "_semantic_v2_mapping",
        ] {
            let count: i64 = conn
                .query_row(
                    &format!("SELECT COUNT(*) FROM {}", db::quote_ident(table)),
                    [],
                    |row| row.get(0),
                )
                .unwrap();
            assert_eq!(count, 0, "stale live {table} rows must be reclaimed");
        }
        assert_eq!(
            conn.query_row("SELECT COUNT(*) FROM _semantic_v2_document", [], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap(),
            0,
            "embeddings must be reclaimable after immutable archival"
        );
        for build_id in [accepted_build, rejected_build, retrieval_build] {
            let retained: bool = conn
                .query_row(
                    "SELECT EXISTS(SELECT 1 FROM _semantic_v2_build WHERE build_id = ?1)",
                    [build_id],
                    |row| row.get(0),
                )
                .unwrap();
            assert!(!retained);
        }

        let immutable_before: (String, String, String) = conn
            .query_row(
                "SELECT mapping_sha256, row_set_sha256, archived_at
                 FROM _semantic_v2_audit_snapshot WHERE selection_id = ?1",
                [&accepted_selection],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        for _ in 0..3 {
            assert_eq!(prune_stale_semantic_artifacts_pass(&mut conn).unwrap(), 0);
        }
        let immutable_after: (String, String, String) = conn
            .query_row(
                "SELECT mapping_sha256, row_set_sha256, archived_at
                 FROM _semantic_v2_audit_snapshot WHERE selection_id = ?1",
                [&accepted_selection],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(immutable_after, immutable_before);
        let child_counts_before: (i64, i64) = conn
            .query_row(
                "SELECT
                    (SELECT COUNT(*) FROM _semantic_v2_audit_snapshot_document
                     WHERE selection_id = ?1),
                    (SELECT COUNT(*) FROM _semantic_v2_audit_snapshot_row_chunk
                     WHERE selection_id = ?1)",
                [&accepted_selection],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert!(conn
            .execute(
                "INSERT INTO _semantic_v2_audit_snapshot_document (
                    selection_id, rank, source_doc_id, fingerprint_sha256, kind, column_key,
                    normalized_text, cosine_score, rank_score, mapping_count, mapping_sha256
                 ) VALUES (?1, 999, 999, ?2, 'cell', 'message', 'late extension',
                           0.0, 0.0, 1, ?3)",
                params![accepted_selection, "9".repeat(64), "8".repeat(64)],
            )
            .is_err());
        assert!(conn
            .execute(
                "UPDATE _semantic_v2_audit_snapshot_document
                 SET normalized_text = 'tampered' WHERE selection_id = ?1 AND rank = 1",
                [&accepted_selection],
            )
            .is_err());
        assert!(conn
            .execute(
                "DELETE FROM _semantic_v2_audit_snapshot_document
                 WHERE selection_id = ?1 AND rank = 1",
                [&accepted_selection],
            )
            .is_err());
        assert!(conn
            .execute(
                "INSERT INTO _semantic_v2_audit_snapshot_row_chunk (
                    selection_id, chunk_index, first_row_num, last_row_num, row_count,
                    encoded_rows, chunk_sha256
                 ) VALUES (?1, 999, 999, 999, 1, X'01', ?2)",
                params![accepted_selection, "7".repeat(64)],
            )
            .is_err());
        assert!(conn
            .execute(
                "UPDATE _semantic_v2_audit_snapshot_row_chunk
                 SET chunk_sha256 = ?2 WHERE selection_id = ?1 AND chunk_index = 0",
                params![accepted_selection, "6".repeat(64)],
            )
            .is_err());
        assert!(conn
            .execute(
                "DELETE FROM _semantic_v2_audit_snapshot_row_chunk
                 WHERE selection_id = ?1 AND chunk_index = 0",
                [&accepted_selection],
            )
            .is_err());
        let orphan_selection = "f".repeat(64);
        assert!(conn
            .execute(
                "INSERT INTO _semantic_v2_audit_snapshot_document (
                    selection_id, rank, source_doc_id, fingerprint_sha256, kind, column_key,
                    normalized_text, cosine_score, rank_score, mapping_count, mapping_sha256
                 ) VALUES (?1, 1, 1, ?2, 'cell', 'message', 'orphan', 0.0, 0.0, 1, ?3)",
                params![orphan_selection, "5".repeat(64), "4".repeat(64)],
            )
            .is_err());
        assert!(conn
            .execute(
                "INSERT INTO _semantic_v2_audit_snapshot_row_chunk (
                    selection_id, chunk_index, first_row_num, last_row_num, row_count,
                    encoded_rows, chunk_sha256
                 ) VALUES (?1, 0, 1, 1, 1, X'01', ?2)",
                params![orphan_selection, "3".repeat(64)],
            )
            .is_err());
        let child_counts_after: (i64, i64) = conn
            .query_row(
                "SELECT
                    (SELECT COUNT(*) FROM _semantic_v2_audit_snapshot_document
                     WHERE selection_id = ?1),
                    (SELECT COUNT(*) FROM _semantic_v2_audit_snapshot_row_chunk
                     WHERE selection_id = ?1)",
                [&accepted_selection],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(child_counts_after, child_counts_before);
        assert!(conn
            .execute(
                "UPDATE _semantic_v2_audit_snapshot SET warnings_json = '[]'
                 WHERE selection_id = ?1",
                [&accepted_selection],
            )
            .is_err());
    }

    #[test]
    fn active_accepted_selection_archival_is_bounded_exact_and_idempotent() {
        let mut conn = Connection::open_in_memory().unwrap();
        let columns = message_columns();
        populate_messages(&mut conn, &columns, 9_000, false);
        let embedder = FakeEmbedder::default();
        ensure_semantic_index_v2(&mut conn, &columns, &embedder, || false, |_| {}).unwrap();
        let active_build = active_build_identity(&conn).unwrap().0;
        let policy = SemanticSearchPolicy {
            maximum_documents: 1,
            minimum_score: -1.0,
        };
        let accepted = create_semantic_selection(
            &mut conn,
            &columns,
            &embedder,
            "accepted active evidence",
            policy,
        )
        .unwrap();
        let unreviewed = create_semantic_selection(
            &mut conn,
            &columns,
            &embedder,
            "unreviewed active evidence",
            policy,
        )
        .unwrap();
        let rejected = create_semantic_selection(
            &mut conn,
            &columns,
            &embedder,
            "rejected active evidence",
            policy,
        )
        .unwrap();
        let retrieval_only = create_semantic_selection(
            &mut conn,
            &columns,
            &embedder,
            "retrieval only active evidence",
            policy,
        )
        .unwrap();
        conn.execute_batch(
            "CREATE TABLE _llm_parse_audit (
                trusted_intent_json TEXT NOT NULL,
                examiner_decision TEXT NOT NULL
             );
             CREATE TABLE _semantic_retrieval_audit(selection_id TEXT NOT NULL);",
        )
        .unwrap();
        for (selection_id, decision) in [
            (&accepted.selection_id, "accepted"),
            (&unreviewed.selection_id, "unreviewed"),
            (&rejected.selection_id, "rejected"),
        ] {
            conn.execute(
                "INSERT INTO _llm_parse_audit(trusted_intent_json, examiner_decision)
                 VALUES (?1, ?2)",
                params![
                    serde_json::json!({ "semanticSelectionId": selection_id }).to_string(),
                    decision,
                ],
            )
            .unwrap();
        }
        conn.execute(
            "INSERT INTO _semantic_retrieval_audit(selection_id) VALUES (?1)",
            [&retrieval_only.selection_id],
        )
        .unwrap();

        let first_slice = archive_required_semantic_audits_slice(&mut conn).unwrap();
        assert!(first_slice.steps_advanced > 0);
        assert!(first_slice.pending);
        assert_eq!(first_slice.snapshots_completed, 0);

        let completed = complete_required_semantic_audits(&mut conn).unwrap();
        assert!(completed.steps_advanced > 0);
        assert_eq!(completed.snapshots_completed, 1);
        assert!(!completed.pending);
        let repeated = complete_required_semantic_audits(&mut conn).unwrap();
        assert_eq!(repeated, SemanticAuditArchiveProgress::default());

        let header: (i64, i64, i64, i64) = conn
            .query_row(
                "SELECT build_id, selected_document_count, mapping_count, row_count
                 FROM _semantic_v2_audit_snapshot WHERE selection_id = ?1",
                [&accepted.selection_id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .unwrap();
        assert_eq!(header, (active_build, 1, 9_000, 9_000));
        let document_mapping_count: i64 = conn
            .query_row(
                "SELECT mapping_count FROM _semantic_v2_audit_snapshot_document
                 WHERE selection_id = ?1",
                [&accepted.selection_id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(document_mapping_count, 9_000);
        let encoded_chunks = {
            let mut statement = conn
                .prepare(
                    "SELECT encoded_rows FROM _semantic_v2_audit_snapshot_row_chunk
                     WHERE selection_id = ?1 ORDER BY chunk_index",
                )
                .unwrap();
            let collected = statement
                .query_map([&accepted.selection_id], |row| row.get::<_, Vec<u8>>(0))
                .unwrap()
                .collect::<rusqlite::Result<Vec<_>>>()
                .unwrap();
            collected
        };
        let decoded_rows = encoded_chunks
            .iter()
            .flat_map(|chunk| decode_sorted_positive_delta_varints(chunk).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(decoded_rows, (1..=9_000).collect::<Vec<_>>());

        for selection_id in [
            &unreviewed.selection_id,
            &rejected.selection_id,
            &retrieval_only.selection_id,
        ] {
            let archived_or_staged: bool = conn
                .query_row(
                    "SELECT EXISTS(
                        SELECT 1 FROM _semantic_v2_audit_snapshot WHERE selection_id = ?1
                        UNION ALL
                        SELECT 1 FROM _semantic_v2_audit_snapshot_stage WHERE selection_id = ?1
                     )",
                    [selection_id],
                    |row| row.get(0),
                )
                .unwrap();
            assert!(
                !archived_or_staged,
                "{selection_id} must remain unsnapshotted"
            );
        }
        assert_eq!(active_build_identity(&conn).unwrap().0, active_build);
        validate_semantic_selection(&conn, &columns, &accepted.selection_id).unwrap();
    }

    #[test]
    fn active_cache_hits_reclaim_stale_artifacts_but_retain_audit_metadata() {
        let mut conn = Connection::open_in_memory().unwrap();
        let columns = message_columns();
        populate_messages(&mut conn, &columns, 1, false);
        let embedder = FakeEmbedder::default();
        ensure_semantic_index_v2(&mut conn, &columns, &embedder, || false, |_| {}).unwrap();
        let active_build = active_build_identity(&conn).unwrap().0;
        let now = chrono::Utc::now().to_rfc3339();

        conn.execute_batch(
            "CREATE TABLE _semantic_index(row_num INTEGER PRIMARY KEY, embedding BLOB NOT NULL);
             CREATE TABLE _semantic_index_info(note TEXT NOT NULL);
             CREATE TABLE _semantic_retrieval_audit(selection_id TEXT);
             CREATE TABLE _llm_parse_audit(
                trusted_intent_json TEXT NOT NULL,
                examiner_decision TEXT NOT NULL
             );",
        )
        .unwrap();
        let tx = conn.transaction().unwrap();
        for row_num in 1..=1_100i64 {
            tx.execute(
                "INSERT INTO _semantic_index(row_num, embedding) VALUES (?1, X'00')",
                [row_num],
            )
            .unwrap();
        }
        tx.execute("INSERT INTO _semantic_index_info(note) VALUES ('v1')", [])
            .unwrap();
        tx.commit().unwrap();

        let insert_stale_build = |conn: &Connection, normalizer: &str, suffix: &str| {
            conn.execute(
                "INSERT INTO _semantic_v2_build (
                    dataset_hash, schema_hash, model_sha256, normalizer_version, status,
                    source_rows, cursor_row_num, rows_scanned, started_at, updated_at, completed_at
                 ) VALUES (?1, ?2, ?3, ?4, 'ready', 600, 600, 600, ?5, ?5, ?5)",
                params![
                    format!("stale-dataset-{suffix}"),
                    format!("stale-schema-{suffix}"),
                    MODEL_SHA256,
                    normalizer,
                    now,
                ],
            )
            .unwrap();
            conn.last_insert_rowid()
        };
        let stale_build = insert_stale_build(&conn, "dfir-cell-normalizer-v1", "plain");
        let audited_build = insert_stale_build(&conn, "dfir-cell-normalizer-v2", "audited");
        for build_id in [stale_build, audited_build] {
            conn.execute(
                "INSERT INTO _semantic_v2_column_plan(
                    build_id, col_index, mode, sql_name, original_name
                 ) VALUES (?1, 0, 'text', 'message', 'Message')",
                [build_id],
            )
            .unwrap();
        }

        let ordinary_selection = "a".repeat(64);
        let audited_selection = "b".repeat(64);
        let tx = conn.transaction().unwrap();
        let embedding = vector_to_blob(&FakeEmbedder::vector());
        let mut stale_doc_ids = Vec::new();
        for index in 0..4_200i64 {
            tx.execute(
                "INSERT INTO _semantic_v2_document(
                    model_sha256, normalizer_version, kind, column_key, text_sha256,
                    normalized_text, embedding
                 ) VALUES (?1, ?2, 'cell', 'message', ?3, ?4, ?5)",
                params![
                    MODEL_SHA256,
                    V2_NORMALIZER_VERSION,
                    format!("{index:064x}"),
                    format!("stale document {index}"),
                    embedding,
                ],
            )
            .unwrap();
            let doc_id = tx.last_insert_rowid();
            stale_doc_ids.push(doc_id);
            for mapping_index in 0..4i64 {
                let row_num = index * 4 + mapping_index + 1;
                tx.execute(
                    "INSERT INTO _semantic_v2_mapping(build_id, doc_id, row_num)
                     VALUES (?1, ?2, ?3)",
                    params![stale_build, doc_id, row_num],
                )
                .unwrap();
            }
        }
        tx.execute(
            "INSERT INTO _semantic_v2_selection(
                selection_id, build_id, dataset_hash, query_sha256, policy_version,
                minimum_score, maximum_documents, documents_above_threshold,
                documents_retained, rows_matched, documents_truncated,
                broad_row_warning, warnings_json, created_at
             ) VALUES (?1, ?2, 'stale', ?3, 'old', 0.0, 4200, 4200, 4200, 16800, 0, 0, '[]', ?4)",
            params![ordinary_selection, stale_build, "c".repeat(64), now],
        )
        .unwrap();
        for doc_id in &stale_doc_ids {
            tx.execute(
                "INSERT INTO _semantic_v2_selection_doc(
                    selection_id, doc_id, cosine_score, rank_score
                 ) VALUES (?1, ?2, 1.0, 1.0)",
                params![ordinary_selection, doc_id],
            )
            .unwrap();
        }

        tx.execute(
            "INSERT INTO _semantic_v2_document(
                model_sha256, normalizer_version, kind, column_key, text_sha256,
                normalized_text, embedding
             ) VALUES (?1, ?2, 'cell', 'message', ?3, 'audited document', ?4)",
            params![
                MODEL_SHA256,
                "dfir-cell-normalizer-v2",
                "d".repeat(64),
                embedding
            ],
        )
        .unwrap();
        let audited_doc = tx.last_insert_rowid();
        tx.execute(
            "INSERT INTO _semantic_v2_mapping(build_id, doc_id, row_num) VALUES (?1, ?2, 1)",
            params![audited_build, audited_doc],
        )
        .unwrap();
        tx.execute(
            "INSERT INTO _semantic_v2_selection(
                selection_id, build_id, dataset_hash, query_sha256, policy_version,
                minimum_score, maximum_documents, documents_above_threshold,
                documents_retained, rows_matched, documents_truncated,
                broad_row_warning, warnings_json, created_at
             ) VALUES (?1, ?2, 'audited', ?3, 'old', 0.0, 1, 1, 1, 1, 0, 0, '[]', ?4)",
            params![audited_selection, audited_build, "e".repeat(64), now],
        )
        .unwrap();
        tx.execute(
            "INSERT INTO _semantic_v2_selection_doc(
                selection_id, doc_id, cosine_score, rank_score
             ) VALUES (?1, ?2, 1.0, 1.0)",
            params![audited_selection, audited_doc],
        )
        .unwrap();
        tx.commit().unwrap();
        conn.execute(
            "INSERT INTO _semantic_retrieval_audit(selection_id) VALUES (?1)",
            [&ordinary_selection],
        )
        .unwrap();
        let trusted = serde_json::json!({
            "intent": "rawEvidenceSearch",
            "semanticSelectionId": audited_selection,
        })
        .to_string();
        conn.execute(
            "INSERT INTO _llm_parse_audit(trusted_intent_json, examiner_decision)
             VALUES (?1, 'accepted')",
            [trusted],
        )
        .unwrap();

        let first_pass_removed = prune_stale_semantic_artifacts_pass(&mut conn).unwrap();
        assert!(first_pass_removed > 0);
        let v1_after_first: i64 = conn
            .query_row("SELECT COUNT(*) FROM _semantic_index", [], |row| row.get(0))
            .unwrap();
        let mappings_after_first: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM _semantic_v2_mapping WHERE build_id = ?1",
                [stale_build],
                |row| row.get(0),
            )
            .unwrap();
        let selection_docs_after_first: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM _semantic_v2_selection_doc WHERE selection_id = ?1",
                [&ordinary_selection],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(v1_after_first, 1_100 - V1_INDEX_PRUNE_BATCH as i64);
        assert_eq!(
            mappings_after_first, 16_800,
            "accepted-plan archival must backpressure every stale v2 deletion"
        );
        assert_eq!(
            selection_docs_after_first, 4_200,
            "snapshot initialization must commit before stale selection docs are reclaimed"
        );

        let cached =
            ensure_semantic_index_v2(&mut conn, &columns, &embedder, || false, |_| {}).unwrap();
        assert!(cached.from_cache);

        for _ in 0..8 {
            ensure_semantic_index_v2(&mut conn, &columns, &embedder, || false, |_| {}).unwrap();
        }
        let exists = |table: &str, column: &str, value: &dyn rusqlite::ToSql| -> bool {
            conn.query_row(
                &format!("SELECT EXISTS(SELECT 1 FROM {table} WHERE {column} = ?1)"),
                [value],
                |row| row.get(0),
            )
            .unwrap()
        };
        assert!(!exists("_semantic_v2_build", "build_id", &stale_build));
        assert!(!exists(
            "_semantic_v2_selection",
            "selection_id",
            &ordinary_selection
        ));
        assert!(exists(
            "_semantic_retrieval_audit",
            "selection_id",
            &ordinary_selection
        ));
        assert!(exists("_semantic_v2_build", "build_id", &active_build));
        assert!(!exists("_semantic_v2_build", "build_id", &audited_build));
        assert!(!exists(
            "_semantic_v2_selection",
            "selection_id",
            &audited_selection
        ));
        assert!(!exists("_semantic_v2_document", "doc_id", &audited_doc));
        assert!(exists(
            "_semantic_v2_audit_snapshot",
            "selection_id",
            &audited_selection
        ));
        assert!(!exists(
            "_semantic_v2_audit_snapshot",
            "selection_id",
            &ordinary_selection
        ));
        assert!(conn
            .query_row(
                "SELECT EXISTS(
                    SELECT 1 FROM _llm_parse_audit
                    WHERE instr(trusted_intent_json, ?1) > 0
                 )",
                [&audited_selection],
                |row| row.get::<_, bool>(0),
            )
            .unwrap());
        assert_eq!(
            conn.query_row("SELECT COUNT(*) FROM _semantic_index", [], |row| row
                .get::<_, i64>(0))
                .unwrap(),
            0
        );
        assert_eq!(
            conn.query_row("SELECT COUNT(*) FROM _semantic_index_info", [], |row| row
                .get::<_, i64>(
                0
            ))
            .unwrap(),
            0
        );
    }

    #[test]
    fn cleanup_failure_never_hides_a_valid_active_index() {
        let mut conn = Connection::open_in_memory().unwrap();
        let columns = message_columns();
        populate_messages(&mut conn, &columns, 1, false);
        let embedder = FakeEmbedder::default();
        ensure_semantic_index_v2(&mut conn, &columns, &embedder, || false, |_| {}).unwrap();
        conn.execute_batch("CREATE TABLE _semantic_index(unexpected_column TEXT);")
            .unwrap();
        let cached =
            ensure_semantic_index_v2(&mut conn, &columns, &embedder, || false, |_| {}).unwrap();
        assert!(cached.from_cache);
        assert!(semantic_index_ready(&conn, &columns).unwrap());
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
    fn semantic_selection_reuses_normalized_query_and_policy_identity() {
        let mut conn = Connection::open_in_memory().unwrap();
        let columns = message_columns();
        populate_messages(&mut conn, &columns, 2, false);
        let embedder = FakeEmbedder::default();
        ensure_semantic_index_v2(&mut conn, &columns, &embedder, || false, |_| {}).unwrap();
        let policy = SemanticSearchPolicy {
            maximum_documents: 4,
            minimum_score: -1.0,
        };
        let first = create_semantic_selection(
            &mut conn,
            &columns,
            &embedder,
            "  CREDENTIAL   Dumping  ",
            policy,
        )
        .unwrap();
        let calls_after_first = embedder.call_count();
        let reused =
            create_semantic_selection(&mut conn, &columns, &embedder, "credential dumping", policy)
                .unwrap();
        assert_eq!(reused.selection_id, first.selection_id);
        assert_eq!(embedder.call_count(), calls_after_first);

        let different_policy = create_semantic_selection(
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
        assert_ne!(different_policy.selection_id, first.selection_id);
    }

    #[test]
    fn selection_cleanup_is_bounded_and_preserves_accepted_and_unreviewed_audits() {
        let mut conn = Connection::open_in_memory().unwrap();
        let columns = message_columns();
        populate_messages(&mut conn, &columns, 1, false);
        let embedder = FakeEmbedder::default();
        ensure_semantic_index_v2(&mut conn, &columns, &embedder, || false, |_| {}).unwrap();
        let policy = SemanticSearchPolicy {
            maximum_documents: 1,
            minimum_score: -1.0,
        };
        let accepted = create_semantic_selection(
            &mut conn,
            &columns,
            &embedder,
            "accepted audit query",
            policy,
        )
        .unwrap();
        let unreviewed = create_semantic_selection(
            &mut conn,
            &columns,
            &embedder,
            "unreviewed audit query",
            policy,
        )
        .unwrap();
        let retrieval_only = create_semantic_selection(
            &mut conn,
            &columns,
            &embedder,
            "ordinary retrieval audit query",
            policy,
        )
        .unwrap();
        conn.execute_batch(
            "CREATE TABLE _llm_parse_audit (
                trusted_intent_json TEXT NOT NULL,
                examiner_decision TEXT NOT NULL
             );
             CREATE TABLE _semantic_retrieval_audit (
                selection_id TEXT
             );",
        )
        .unwrap();
        for (selection_id, decision) in [
            (&accepted.selection_id, "accepted"),
            (&unreviewed.selection_id, "unreviewed"),
        ] {
            let trusted = serde_json::json!({
                "intent": "rawEvidenceSearch",
                "semanticSelectionId": selection_id,
            })
            .to_string();
            conn.execute(
                "INSERT INTO _llm_parse_audit(trusted_intent_json, examiner_decision)
                 VALUES (?1, ?2)",
                params![trusted, decision],
            )
            .unwrap();
        }
        conn.execute(
            "INSERT INTO _semantic_retrieval_audit(selection_id) VALUES (?1)",
            [&retrieval_only.selection_id],
        )
        .unwrap();
        let mut newest = String::new();
        for index in 0..70 {
            newest = create_semantic_selection(
                &mut conn,
                &columns,
                &embedder,
                &format!("cleanup candidate {}", alphabetic_id(index)),
                policy,
            )
            .unwrap()
            .selection_id;
        }
        let build_id = active_build_identity(&conn).unwrap().0;
        let first_pass = cleanup_semantic_selections(&conn, build_id, &newest, 2).unwrap();
        assert_eq!(first_pass, V2_MAX_SELECTION_CLEANUP_PER_REQUEST as usize);
        let second_pass = cleanup_semantic_selections(&conn, build_id, &newest, 2).unwrap();
        assert!(second_pass <= V2_MAX_SELECTION_CLEANUP_PER_REQUEST as usize);
        for selection_id in [&accepted.selection_id, &unreviewed.selection_id] {
            let retained: bool = conn
                .query_row(
                    "SELECT EXISTS(SELECT 1 FROM _semantic_v2_selection WHERE selection_id = ?1)",
                    [selection_id],
                    |row| row.get(0),
                )
                .unwrap();
            assert!(retained, "audited selection {selection_id} was deleted");
        }
        let retrieval_retained: bool = conn
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM _semantic_v2_selection WHERE selection_id = ?1)",
                [&retrieval_only.selection_id],
                |row| row.get(0),
            )
            .unwrap();
        assert!(
            !retrieval_retained,
            "ordinary retrieval audit rows must not pin live selections"
        );
        let unaudited: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM _semantic_v2_selection s
                 WHERE NOT EXISTS (
                    SELECT 1 FROM _llm_parse_audit l,
                         json_tree(l.trusted_intent_json) j
                    WHERE l.examiner_decision IN ('unreviewed', 'accepted')
                      AND j.key = 'semanticSelectionId' AND j.value = s.selection_id
                 )",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(unaudited, 2);
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

        let cached_progress = Arc::new(Mutex::new(Vec::<SemanticBuildProgress>::new()));
        let cached_sink = Arc::clone(&cached_progress);
        let cached = ensure_semantic_index_v2(
            &mut conn,
            &columns,
            &embedder,
            || false,
            move |update| cached_sink.lock().unwrap().push(update),
        )
        .unwrap();
        assert!(cached.from_cache);
        let cached_ready = cached_progress.lock().unwrap().last().unwrap().clone();
        assert_eq!(cached_ready.phase, "ready");
        assert_eq!(cached_ready.rows_scanned, 600);
        assert_eq!(cached_ready.documents_embedded, 1);
        assert_eq!(cached_ready.mappings_written, 600);
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
        let current_build = active_build_identity(&conn).unwrap().0;
        assert!(
            !semantic_selection_reasons(&conn, &selection.selection_id, &[1])
                .unwrap()
                .is_empty()
        );
        conn.execute(
            "UPDATE _semantic_v2_build SET tokenizer_sha256 = ?2 WHERE build_id = ?1",
            params![current_build, V2_LEGACY_UNRECORDED_IDENTITY],
        )
        .unwrap();
        assert!(validate_semantic_selection(&conn, &columns, &selection.selection_id).is_err());
        assert!(semantic_selection_reasons(&conn, &selection.selection_id, &[1]).is_err());
        assert!(semantic_selection_reasons(&conn, &selection.selection_id, &[]).is_err());
        conn.execute(
            "UPDATE _semantic_v2_build SET tokenizer_sha256 = ?2 WHERE build_id = ?1",
            params![current_build, TOKENIZER_SHA256],
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
        let results = workers
            .into_iter()
            .map(|worker| worker.join().unwrap().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            results.iter().filter(|summary| summary.cancelled).count(),
            1,
            "the worker that lost ownership must exit without pausing the winner"
        );

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
        let (status, worker_token): (String, Option<String>) = conn
            .query_row(
                "SELECT status, worker_token FROM _semantic_v2_build",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(status, "ready");
        assert!(worker_token.is_none());

        // Simulate the narrow observer race where the identity row becomes ready after the
        // caller's active-pointer check but before build preparation. Preparation must return the
        // ready identity instead of retrying INSERT OR IGNORE forever.
        conn.execute("DELETE FROM _semantic_v2_active", []).unwrap();
        drop(conn);
        let mut conn = Connection::open(&path).unwrap();
        let cache_embedder = FakeEmbedder::default();
        let cached =
            ensure_semantic_index_v2(&mut conn, &columns, &cache_embedder, || false, |_| {})
                .unwrap();
        assert!(cached.from_cache);
        assert_eq!(cache_embedder.call_count(), 0);
        assert!(semantic_index_ready(&conn, &columns).unwrap());
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
