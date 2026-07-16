mod eval;
mod generate;
mod load;
mod metrics;
mod models;
mod prompt;
mod schema;
mod validate;

use anyhow::{Context, Result};
use clap::Parser;
use metrics::{score_case, CaseResult, RunReport};
use std::path::PathBuf;
use std::time::Instant;

/// Standalone benchmark harness: evaluates a candidate local LLM as the
/// log-parser intel guided-query parser. Not part of the shipped app.
#[derive(Parser, Debug)]
struct Cli {
    /// Which model to benchmark.
    #[arg(long, value_parser = ["small", "mid"])]
    model: String,

    /// Path to the eval case set.
    #[arg(long, default_value = "eval_set.json")]
    eval_set: PathBuf,

    /// Local model/tokenizer download cache (never committed).
    #[arg(long, default_value = ".model-cache")]
    cache_dir: PathBuf,

    /// Where to write the JSON/Markdown reports (never committed).
    #[arg(long, default_value = "results")]
    results_dir: PathBuf,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let spec = models::by_key(&cli.model).context("unknown --model value")?;

    println!(
        "Loading {} from HuggingFace (cached under {})...",
        spec.display_name,
        cli.cache_dir.display()
    );
    let mut loaded = load::load(spec, &cli.cache_dir)?;
    println!(
        "Loaded {} in {:.1} s ({:.2} GB)",
        loaded.gguf_path.display(),
        loaded.load_time_ms as f64 / 1000.0,
        loaded.gguf_size_bytes as f64 / 1e9
    );

    let cases = eval::load_eval_set(&cli.eval_set)
        .with_context(|| format!("loading eval set from {}", cli.eval_set.display()))?;
    println!("Running {} eval cases...", cases.len());

    let mut case_results = Vec::with_capacity(cases.len());
    for case in &cases {
        let prompt_text = prompt::build_prompt(&case.mock_context, &case.query_text);

        let start = Instant::now();
        let generated_suffix = generate::generate(
            &mut loaded.weights,
            &loaded.tokenizer,
            &loaded.device,
            &prompt_text,
        )
        .with_context(|| format!("generating for case {}", case.id))?;
        let latency_ms = start.elapsed().as_millis();
        let raw_output = prompt::complete_assistant_output(generated_suffix);

        let outcome = validate::parse_and_validate(&raw_output, &case.mock_context);
        let verdict = score_case(&case.expected, &outcome);

        println!("  [{latency_ms:>6} ms] {:<45} -> {:?}", case.id, verdict);

        case_results.push(CaseResult {
            case_id: case.id.clone(),
            query_text: case.query_text.clone(),
            raw_output,
            verdict,
            latency_ms,
            notes: case.notes.clone(),
        });
    }

    let report = RunReport {
        model_key: spec.key.to_string(),
        model_display_name: spec.display_name.to_string(),
        quant_label: spec.quant_label.to_string(),
        gguf_size_bytes: loaded.gguf_size_bytes,
        load_time_ms: loaded.load_time_ms,
        case_results,
    };

    report.print_summary();

    std::fs::create_dir_all(&cli.results_dir).context("creating results dir")?;
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let json_path = cli
        .results_dir
        .join(format!("{timestamp}_{}_results.json", spec.key));
    let md_path = cli
        .results_dir
        .join(format!("{timestamp}_{}_results.md", spec.key));
    report.write_json(&json_path)?;
    report.write_markdown(&md_path)?;
    println!("\nWrote {} and {}", json_path.display(), md_path.display());

    Ok(())
}
