use anyhow::{Context, Result};
use candle_core::quantized::gguf_file;
use candle_core::Device;
use candle_transformers::models::quantized_qwen2::ModelWeights;
use hf_hub::HFClientSync;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tokenizers::Tokenizer;

use crate::models::ModelSpec;

pub struct LoadedModel {
    pub weights: ModelWeights,
    pub tokenizer: Tokenizer,
    pub device: Device,
    pub gguf_path: PathBuf,
    pub gguf_size_bytes: u64,
    pub load_time_ms: u128,
}

/// Downloads (if not already cached under `cache_dir`) and loads a model.
/// Network access here is a one-time, explicit, local-machine benchmarking
/// step -- not part of the shipped app's runtime, which stays zero-network.
pub fn load(spec: ModelSpec, cache_dir: &Path) -> Result<LoadedModel> {
    let start = Instant::now();
    let client = HFClientSync::new().context("creating HF client")?;

    let gguf_dir = cache_dir.join(format!("{}--{}", spec.gguf_owner, spec.gguf_repo));
    std::fs::create_dir_all(&gguf_dir).context("creating gguf cache dir")?;
    let gguf_path = client
        .model(spec.gguf_owner, spec.gguf_repo)
        .download_file()
        .filename(spec.gguf_filename)
        .local_dir(gguf_dir)
        .send()
        .with_context(|| format!("downloading {}", spec.gguf_filename))?;

    let tokenizer_dir =
        cache_dir.join(format!("{}--{}", spec.tokenizer_owner, spec.tokenizer_repo));
    std::fs::create_dir_all(&tokenizer_dir).context("creating tokenizer cache dir")?;
    let tokenizer_path = client
        .model(spec.tokenizer_owner, spec.tokenizer_repo)
        .download_file()
        .filename("tokenizer.json")
        .local_dir(tokenizer_dir)
        .send()
        .context("downloading tokenizer.json")?;

    let gguf_size_bytes = std::fs::metadata(&gguf_path)
        .context("reading gguf file metadata")?
        .len();

    let device = Device::Cpu;
    let mut file = File::open(&gguf_path).context("opening gguf file")?;
    let content = gguf_file::Content::read(&mut file).context("reading gguf header")?;
    let weights = ModelWeights::from_gguf(content, &mut file, &device)
        .context("loading model weights from gguf")?;

    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("loading tokenizer.json: {e}"))?;

    Ok(LoadedModel {
        weights,
        tokenizer,
        device,
        gguf_path,
        gguf_size_bytes,
        load_time_ms: start.elapsed().as_millis(),
    })
}
