# log-parser

A fast, fully offline desktop viewer for large SIEM/EDR log exports (Excel
and CSV) — built for DFIR examiners. Handles 100k+ row files, dynamic
column detection, keyset-paginated filtering/search, AI-assisted evidence
retrieval, optional MITRE ATT&CK enrichment, and multi-sheet report export.
**No runtime network calls** — imported evidence and AI inference stay on
your machine.

See the [wiki](../../wiki) for a full user guide.

## Features

- Import large Excel (`.xlsx`/`.xls`) or CSV exports from tools like
  Microsoft Sentinel, Secureworks Taegis, and Microsoft Defender.
- Dynamic column detection — no per-source schema required.
- Fast filtering, full-text search, sorting, and CSV/XLSX export, all
  keyset-paginated for large files.
- Automatic data mapping for timestamps and common evidence fields, with
  optional manual overrides. Mapping is metadata for timelines and threat
  enrichment; it never limits which raw rows the AI can search.
- UTC timestamp normalization, with an explicit prompt when a timestamp's
  timezone is ambiguous.
- An offline, built-in MITRE ATT&CK-style keyword library, scanned via
  Aho-Corasick pattern matching, extensible with your own custom
  categories.
- A local AI evidence search powered by embedded Qwen2.5-1.5B-Instruct and
  all-MiniLM-L6-v2 models. Describe the evidence in plain language (for
  example, *"show failed logins followed by PowerShell activity for alice,
  chronologically"*). The AI plans a bounded query over the complete raw
  table and combines exact/full-text conditions with semantic candidates;
  it is not restricted to rows found by the optional threat scan.
- A validated, examiner-visible query plan before execution, plus a
  `Why matched` explanation on returned rows. The models cannot execute SQL,
  read arbitrary files, launch processes, or access the network.
- Background semantic indexing after import. Lexical AI queries remain
  available while the index is building, and semantic recall is added when
  it becomes ready.
- One-click multi-sheet XLSX report export: a case-summary sheet, a
  chronological MITRE-mapped timeline, and one sheet per matched
  technique category — every row traceable back to its original source
  row.

## Try it

Sample data is included under `testdata/`:
- `sentinel_sample_120k.xlsx` — a 120,000-row synthetic Sentinel-style
  export, for trying the tool at realistic scale.
- `multi_sheet_sample.xlsx` — a small 3-sheet workbook.

## Building from source

Requires [Rust](https://rustup.rs/), Python 3, and the
[Tauri v2 prerequisites](https://v2.tauri.app/start/prerequisites/) for
your platform (Windows, macOS, or Linux). Fetch the checksum-pinned models,
tokenizers, and configuration once before building; this is an explicit
build-time download, not an application runtime download:

```sh
python scripts/fetch_llm_resources.py
cd src-tauri
cargo tauri build
```

This produces a native installer/bundle for your current platform in
`src-tauri/target/release/bundle/`. Cross-compiling for other platforms
follows the standard Tauri process — see the
[Tauri distribution docs](https://v2.tauri.app/distribute/).

To run in development mode:

```sh
python scripts/fetch_llm_resources.py
cd src-tauri
cargo tauri dev
```

The embedded AI resources add about 1.22 GB (1.13 GiB) to an unpacked
application. Current
x86-64 builds require AVX2 and FMA CPU support; Apple Silicon uses its native
ARM SIMD baseline.

## Stack

Rust + [Tauri v2](https://v2.tauri.app/) · [calamine](https://github.com/tafia/calamine)
(Excel parsing) · SQLite via [rusqlite](https://github.com/rusqlite/rusqlite)
(bundled, FTS5 full-text search) · [aho-corasick](https://github.com/BurntSushi/aho-corasick)
(keyword matching) · [rust_xlsxwriter](https://github.com/jmcnamara/rust_xlsxwriter)
(report export) · [Candle](https://github.com/huggingface/candle) with
[Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
for local query planning and
[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
for semantic retrieval · plain HTML/CSS/vanilla JS frontend
([Tabulator.js](https://tabulator.info/) for the grid), no build step.

## Downloads

Pre-built installers for Windows, macOS (Apple Silicon and Intel), and
Linux are published on the [Releases](../../releases) page.

**macOS note:** these builds are not yet code-signed/notarized with an
Apple Developer ID, so Gatekeeper will refuse to open them with a
"log-parser is damaged and should be moved to the Trash" message — the
app isn't actually damaged, this is just Gatekeeper blocking an
unsigned/unnotarized download. After moving `log-parser.app` to
`/Applications`, clear the quarantine flag once:

```sh
xattr -cr /Applications/log-parser.app
```

Then it opens normally.

## Credits

log-parser is a team effort: concept, requirements, and product direction
by [biancanik-art](https://github.com/biancanik-art); engineering by
[Claude](https://www.anthropic.com/claude) (Anthropic) and
[Codex](https://openai.com/index/introducing-codex/) (OpenAI), working as
independent AI engineers/reviewers throughout the build — implementing
features in parallel, cross-checking each other's work, and running live
end-to-end verification against the real app.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
