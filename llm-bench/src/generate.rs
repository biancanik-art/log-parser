use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_qwen2;
use tokenizers::Tokenizer;

const MAX_NEW_TOKENS: usize = 256;

/// Greedy (temperature=None -> ArgMax) generation: deterministic output is
/// what we want for scoring a structured-JSON-parsing eval, not creative
/// sampling diversity.
pub fn generate(
    weights: &mut quantized_qwen2::ModelWeights,
    tokenizer: &Tokenizer,
    device: &Device,
    prompt: &str,
) -> Result<String> {
    weights.clear_kv_cache();

    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!("tokenizing prompt: {e}"))?;
    let mut tokens: Vec<u32> = encoding.get_ids().to_vec();
    if tokens.is_empty() {
        anyhow::bail!("prompt encoded to zero tokens");
    }

    let eos_id = tokenizer
        .token_to_id("<|im_end|>")
        .context("tokenizer is missing <|im_end|>")?;

    let mut logits_processor = LogitsProcessor::new(299792458, None, None);
    let mut generated_ids: Vec<u32> = Vec::new();
    let mut index_pos = 0usize;

    for step in 0..MAX_NEW_TOKENS {
        let (context, context_index) = if step == 0 {
            (tokens.as_slice(), 0usize)
        } else {
            (&tokens[tokens.len() - 1..], index_pos)
        };
        let input = Tensor::new(context, device)
            .context("building input tensor")?
            .unsqueeze(0)
            .context("unsqueezing input tensor")?;
        let logits = weights
            .forward(&input, context_index)
            .context("model forward pass")?;
        let logits = logits.squeeze(0).context("squeezing logits")?;

        let next_token = logits_processor
            .sample(&logits)
            .context("sampling next token")?;
        index_pos += context.len();
        tokens.push(next_token);

        if next_token == eos_id {
            break;
        }
        generated_ids.push(next_token);
    }

    let text = tokenizer
        // EOS is consumed above rather than appended. Preserve any other generated special
        // token so strict validation and the saved benchmark output expose the violation.
        .decode(&generated_ids, false)
        .map_err(|e| anyhow::anyhow!("decoding generated tokens: {e}"))?;
    Ok(text)
}
