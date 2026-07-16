use crate::eval::ExpectedOutcome;
use crate::validate::ParseOutcome;
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// Parsed and matched (or was one of) the expected intent(s).
    CorrectMatch,
    /// Was correctly required to refuse, and did (Unknown, or a caught
    /// invalid/hallucinated output -- either way, it did not silently
    /// produce a confident wrong structured answer).
    CorrectRefusal,
    /// Parsed to a valid, non-hallucinated intent that does not match any
    /// expected intent.
    WrongIntent,
    /// Was required to refuse but instead confidently produced a valid,
    /// non-hallucinated (but wrong) structured intent. The worst outcome:
    /// undetectable by the schema/library validator.
    FailedToRefuse,
    /// Model output did not parse as valid GuidedIntent JSON.
    InvalidJson,
    /// Parsed, but referenced a technique/tactic/column not present in the
    /// case's injected context.
    HallucinatedReference,
}

/// Scores one case's model output against its expected outcome. Fail-closed
/// throughout -- no retry-as-freeform happens before this point (see
/// validate::parse_and_validate).
pub fn score_case(expected: &ExpectedOutcome, outcome: &ParseOutcome) -> Verdict {
    match (expected, outcome) {
        (ExpectedOutcome::MustBeUnknownOrClarify, ParseOutcome::Parsed(intent)) => {
            if intent.is_unknown() {
                Verdict::CorrectRefusal
            } else {
                Verdict::FailedToRefuse
            }
        }
        (ExpectedOutcome::MustBeUnknownOrClarify, ParseOutcome::InvalidJson { .. }) => {
            Verdict::CorrectRefusal
        }
        (ExpectedOutcome::MustBeUnknownOrClarify, ParseOutcome::HallucinatedReference { .. }) => {
            Verdict::CorrectRefusal
        }
        (
            ExpectedOutcome::Exact {
                intent: expected_intent,
            },
            ParseOutcome::Parsed(actual),
        ) => {
            if actual == expected_intent {
                Verdict::CorrectMatch
            } else {
                Verdict::WrongIntent
            }
        }
        (ExpectedOutcome::AnyOf { intents }, ParseOutcome::Parsed(actual)) => {
            if intents.contains(actual) {
                Verdict::CorrectMatch
            } else {
                Verdict::WrongIntent
            }
        }
        (
            ExpectedOutcome::Exact { .. } | ExpectedOutcome::AnyOf { .. },
            ParseOutcome::InvalidJson { .. },
        ) => Verdict::InvalidJson,
        (
            ExpectedOutcome::Exact { .. } | ExpectedOutcome::AnyOf { .. },
            ParseOutcome::HallucinatedReference { .. },
        ) => Verdict::HallucinatedReference,
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct CaseResult {
    pub case_id: String,
    pub query_text: String,
    pub raw_output: String,
    pub verdict: Verdict,
    pub latency_ms: u128,
    pub notes: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct RunReport {
    pub model_key: String,
    pub model_display_name: String,
    pub quant_label: String,
    pub gguf_size_bytes: u64,
    pub load_time_ms: u128,
    pub case_results: Vec<CaseResult>,
}

impl RunReport {
    fn scoreable_count(&self, matches: impl Fn(Verdict) -> bool) -> (usize, usize) {
        let hits = self
            .case_results
            .iter()
            .filter(|c| matches(c.verdict))
            .count();
        (hits, self.case_results.len())
    }

    pub fn accuracy(&self) -> (usize, usize) {
        // denominator: cases scored CorrectMatch or WrongIntent (i.e. cases
        // with an Exact/AnyOf expectation that produced valid, non-
        // hallucinated JSON) plus InvalidJson/HallucinatedReference against
        // an Exact/AnyOf expectation (still counts against accuracy).
        let relevant: Vec<&CaseResult> = self
            .case_results
            .iter()
            .filter(|c| {
                matches!(
                    c.verdict,
                    Verdict::CorrectMatch
                        | Verdict::WrongIntent
                        | Verdict::InvalidJson
                        | Verdict::HallucinatedReference
                )
            })
            .collect();
        let hits = relevant
            .iter()
            .filter(|c| c.verdict == Verdict::CorrectMatch)
            .count();
        (hits, relevant.len())
    }

    pub fn refusal_rate(&self) -> (usize, usize) {
        let relevant: Vec<&CaseResult> = self
            .case_results
            .iter()
            .filter(|c| matches!(c.verdict, Verdict::CorrectRefusal | Verdict::FailedToRefuse))
            .collect();
        let hits = relevant
            .iter()
            .filter(|c| c.verdict == Verdict::CorrectRefusal)
            .count();
        (hits, relevant.len())
    }

    pub fn invalid_output_rate(&self) -> (usize, usize) {
        self.scoreable_count(|v| matches!(v, Verdict::InvalidJson | Verdict::HallucinatedReference))
    }

    pub fn print_summary(&self) {
        println!(
            "\n=== {} ({}) ===",
            self.model_display_name, self.quant_label
        );
        println!("GGUF size: {:.2} GB", self.gguf_size_bytes as f64 / 1e9);
        println!("Load time: {:.1} s", self.load_time_ms as f64 / 1000.0);

        let (acc_hits, acc_total) = self.accuracy();
        let (ref_hits, ref_total) = self.refusal_rate();
        let (inv_hits, inv_total) = self.invalid_output_rate();
        println!(
            "Accuracy (exact/any_of cases): {acc_hits}/{acc_total} ({:.0}%)",
            pct(acc_hits, acc_total)
        );
        println!(
            "Correct-refusal rate (must-refuse cases): {ref_hits}/{ref_total} ({:.0}%)",
            pct(ref_hits, ref_total)
        );
        println!(
            "Invalid-output rate (all cases): {inv_hits}/{inv_total} ({:.0}%)",
            pct(inv_hits, inv_total)
        );

        let latencies: Vec<u128> = self.case_results.iter().map(|c| c.latency_ms).collect();
        if let Some((cold, rest)) = latencies.split_first() {
            let warm_mean = if !rest.is_empty() {
                rest.iter().sum::<u128>() as f64 / rest.len() as f64
            } else {
                0.0
            };
            println!("Cold latency (first case): {cold} ms");
            println!(
                "Warm latency (mean of remaining {} cases): {warm_mean:.0} ms",
                rest.len()
            );
        }

        println!("\n{:<45} {:<22} {:>8}", "case_id", "verdict", "ms");
        for case in &self.case_results {
            println!(
                "{:<45} {:<22} {:>8}",
                case.case_id,
                format!("{:?}", case.verdict),
                case.latency_ms
            );
        }
    }

    pub fn write_json(&self, path: &std::path::Path) -> anyhow::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn write_markdown(&self, path: &std::path::Path) -> anyhow::Result<()> {
        let (acc_hits, acc_total) = self.accuracy();
        let (ref_hits, ref_total) = self.refusal_rate();
        let (inv_hits, inv_total) = self.invalid_output_rate();
        let mut md = format!(
            "# {} ({})\n\n- GGUF size: {:.2} GB\n- Load time: {:.1} s\n- Accuracy: {acc_hits}/{acc_total} ({:.0}%)\n- Correct-refusal rate: {ref_hits}/{ref_total} ({:.0}%)\n- Invalid-output rate: {inv_hits}/{inv_total} ({:.0}%)\n\n| case_id | verdict | latency_ms | query |\n|---|---|---|---|\n",
            self.model_display_name,
            self.quant_label,
            self.gguf_size_bytes as f64 / 1e9,
            self.load_time_ms as f64 / 1000.0,
            pct(acc_hits, acc_total),
            pct(ref_hits, ref_total),
            pct(inv_hits, inv_total),
        );
        for case in &self.case_results {
            md.push_str(&format!(
                "| {} | {:?} | {} | {} |\n",
                case.case_id, case.verdict, case.latency_ms, case.query_text
            ));
        }
        std::fs::write(path, md)?;
        Ok(())
    }
}

fn pct(hits: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        100.0 * hits as f64 / total as f64
    }
}
