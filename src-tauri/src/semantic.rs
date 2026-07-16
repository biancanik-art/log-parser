use crate::db::{self, ColumnMeta};
use anyhow::{bail, Context, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use rusqlite::{params, Connection, OptionalExtension};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
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

const INDEX_VERSION: &str = "semantic-row-v1";
pub const V2_INDEX_VERSION: &str = "semantic-document-v2";
pub const V2_NORMALIZER_VERSION: &str = "dfir-cell-normalizer-v1";
pub const V2_SOURCE_BATCH_ROWS: usize = 256;
pub const V2_EMBED_BATCH_DOCUMENTS: usize = 16;
pub const V2_DEFAULT_DOCUMENT_CANDIDATES: usize = 256;
pub const V2_MAX_DOCUMENT_CANDIDATES: usize = 1_024;
pub const V2_DEFAULT_MINIMUM_SCORE: f32 = 0.38;
pub const V2_BROAD_MINIMUM_SCORE: f32 = 0.30;
const MAX_TOKENS: usize = 256;
const INDEX_BATCH_SIZE: usize = 32;
const MAX_DOCUMENT_CHARS: usize = 4_096;
const MAX_CELL_CHARS: usize = 1_024;
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

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticIndexSummary {
    pub rows_indexed: i64,
    pub elapsed_ms: u128,
    pub from_cache: bool,
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
    hasher.update(INDEX_VERSION.as_bytes());
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

pub fn semantic_index_ready(conn: &Connection, columns: &[ColumnMeta]) -> Result<bool> {
    if !table_exists(conn, "_semantic_index")? || !table_exists(conn, "_semantic_index_info")? {
        return Ok(false);
    }
    let expected_schema = semantic_schema_hash(columns);
    let expected_rows: i64 = conn.query_row("SELECT COUNT(*) FROM rows", [], |row| row.get(0))?;
    let info: Option<(String, String, i64)> = conn
        .query_row(
            "SELECT schema_hash, model_sha256, rows_indexed
             FROM _semantic_index_info ORDER BY rowid DESC LIMIT 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .optional()?;
    Ok(info.is_some_and(|(schema, model, rows)| {
        schema == expected_schema && model == MODEL_SHA256 && rows == expected_rows
    }))
}

pub fn ensure_semantic_index(
    conn: &mut Connection,
    columns: &[ColumnMeta],
    model: &SemanticModel,
) -> Result<SemanticIndexSummary> {
    let started = Instant::now();
    if semantic_index_ready(conn, columns)? {
        let rows_indexed = conn.query_row(
            "SELECT rows_indexed FROM _semantic_index_info ORDER BY rowid DESC LIMIT 1",
            [],
            |row| row.get(0),
        )?;
        return Ok(SemanticIndexSummary {
            rows_indexed,
            elapsed_ms: started.elapsed().as_millis(),
            from_cache: true,
            model_name: MODEL_NAME,
            model_version: MODEL_VERSION,
        });
    }

    let tx = conn.transaction()?;
    create_semantic_schema(&tx)?;
    tx.execute("DELETE FROM _semantic_index", [])?;
    tx.execute("DELETE FROM _semantic_index_info", [])?;

    let identifiers = columns
        .iter()
        .map(|column| db::quote_ident(&column.sql_name))
        .collect::<Vec<_>>()
        .join(", ");
    let select_sql = format!("SELECT row_num, {identifiers} FROM rows ORDER BY row_num");
    let mut select = tx.prepare(&select_sql)?;
    let mut source_rows = select.query([])?;
    let mut insert =
        tx.prepare("INSERT INTO _semantic_index (row_num, embedding) VALUES (?1, ?2)")?;
    let mut rows_indexed = 0i64;

    loop {
        let mut row_numbers = Vec::with_capacity(INDEX_BATCH_SIZE);
        let mut documents = Vec::with_capacity(INDEX_BATCH_SIZE);
        while row_numbers.len() < INDEX_BATCH_SIZE {
            let Some(row) = source_rows.next()? else {
                break;
            };
            let row_num: i64 = row.get(0)?;
            let values = (0..columns.len())
                .map(|index| row.get::<_, Option<String>>(index + 1))
                .collect::<rusqlite::Result<Vec<_>>>()?;
            row_numbers.push(row_num);
            documents.push(row_document(columns, &values));
        }
        if row_numbers.is_empty() {
            break;
        }
        let embeddings = model.embed_batch(&documents)?;
        if embeddings.len() != row_numbers.len() {
            bail!("semantic model returned the wrong batch size");
        }
        for (row_num, embedding) in row_numbers.into_iter().zip(embeddings) {
            if embedding.len() != EMBEDDING_DIMENSIONS {
                bail!(
                    "semantic model returned {} dimensions, expected {EMBEDDING_DIMENSIONS}",
                    embedding.len()
                );
            }
            insert.execute(params![row_num, vector_to_blob(&embedding)])?;
            rows_indexed += 1;
        }
    }
    drop(source_rows);
    drop(select);
    drop(insert);

    let schema_hash = semantic_schema_hash(columns);
    tx.execute(
        "INSERT INTO _semantic_index_info (
            schema_hash, index_version, model_name, model_version, model_sha256,
            rows_indexed, dimensions, completed_at
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        params![
            schema_hash,
            INDEX_VERSION,
            MODEL_NAME,
            MODEL_VERSION,
            MODEL_SHA256,
            rows_indexed,
            EMBEDDING_DIMENSIONS as i64,
            chrono::Utc::now().to_rfc3339(),
        ],
    )?;
    tx.commit()?;

    Ok(SemanticIndexSummary {
        rows_indexed,
        elapsed_ms: started.elapsed().as_millis(),
        from_cache: false,
        model_name: MODEL_NAME,
        model_version: MODEL_VERSION,
    })
}

pub fn semantic_search(
    conn: &Connection,
    model: &SemanticModel,
    query: &str,
    top_k: usize,
    minimum_score: f32,
) -> Result<Vec<SemanticCandidate>> {
    let query = query.trim();
    if query.is_empty() {
        bail!("semantic search query is empty");
    }
    if query.chars().count() > MAX_QUERY_CHARS {
        bail!("semantic search query exceeds {MAX_QUERY_CHARS} characters");
    }
    if !table_exists(conn, "_semantic_index")? {
        bail!("semantic index is not ready");
    }
    let top_k = top_k.clamp(1, MAX_TOP_K);
    let query_embedding = model.embed(query)?;
    let mut heap: BinaryHeap<Reverse<ScoredRow>> = BinaryHeap::with_capacity(top_k + 1);
    let mut stmt = conn.prepare("SELECT row_num, embedding FROM _semantic_index")?;
    let mut rows = stmt.query([])?;
    while let Some(row) = rows.next()? {
        let row_num: i64 = row.get(0)?;
        let blob: Vec<u8> = row.get(1)?;
        let score = dot_blob(&query_embedding, &blob)?;
        if !score.is_finite() || score < minimum_score {
            continue;
        }
        let candidate = ScoredRow { row_num, score };
        if heap.len() < top_k {
            heap.push(Reverse(candidate));
        } else if heap.peek().is_some_and(|smallest| candidate > smallest.0) {
            heap.pop();
            heap.push(Reverse(candidate));
        }
    }
    let mut candidates = heap
        .into_iter()
        .map(|Reverse(row)| SemanticCandidate {
            row_num: row.row_num,
            score: row.score,
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.row_num.cmp(&right.row_num))
    });
    Ok(candidates)
}

fn create_semantic_schema(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS _semantic_index (
            row_num INTEGER PRIMARY KEY,
            embedding BLOB NOT NULL
         );
         CREATE TABLE IF NOT EXISTS _semantic_index_info (
            schema_hash TEXT NOT NULL,
            index_version TEXT NOT NULL,
            model_name TEXT NOT NULL,
            model_version TEXT NOT NULL,
            model_sha256 TEXT NOT NULL,
            rows_indexed INTEGER NOT NULL,
            dimensions INTEGER NOT NULL,
            completed_at TEXT NOT NULL
         );",
    )
}

fn row_document(columns: &[ColumnMeta], values: &[Option<String>]) -> String {
    let mut document = String::new();
    for (column, value) in columns.iter().zip(values) {
        let Some(value) = value
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        else {
            continue;
        };
        let value = truncate_chars(value, MAX_CELL_CHARS);
        let separator_chars = if document.is_empty() { 0 } else { 2 };
        let label_chars = column.original_name.chars().count() + 2;
        let remaining = MAX_DOCUMENT_CHARS.saturating_sub(document.chars().count());
        if remaining <= separator_chars + label_chars {
            break;
        }
        if !document.is_empty() {
            document.push_str("; ");
        }
        document.push_str(&column.original_name);
        document.push_str(": ");
        let remaining = MAX_DOCUMENT_CHARS.saturating_sub(document.chars().count());
        document.push_str(&truncate_chars(&value, remaining));
    }
    if document.is_empty() {
        "empty log row".to_string()
    } else {
        document
    }
}

fn truncate_chars(value: &str, maximum: usize) -> String {
    value.chars().take(maximum).collect()
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

#[derive(Debug, Clone, Copy)]
struct ScoredRow {
    row_num: i64,
    score: f32,
}

impl PartialEq for ScoredRow {
    fn eq(&self, other: &Self) -> bool {
        self.row_num == other.row_num && self.score.to_bits() == other.score.to_bits()
    }
}

impl Eq for ScoredRow {}

impl PartialOrd for ScoredRow {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredRow {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| other.row_num.cmp(&self.row_num))
    }
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
    fn row_documents_are_labeled_bounded_and_skip_blanks() {
        let values = vec![
            Some("2026-01-01T00:00:00Z".into()),
            Some("  powershell.exe  ".into()),
        ];
        let document = row_document(&columns(), &values);
        assert_eq!(
            document,
            "Time Generated: 2026-01-01T00:00:00Z; Message: powershell.exe"
        );
        assert!(document.chars().count() <= MAX_DOCUMENT_CHARS);
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
