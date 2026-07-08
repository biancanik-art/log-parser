# log-parser

A fast, fully offline desktop viewer for large SIEM/EDR log exports (Excel
and CSV) — built for DFIR examiners. Handles 100k+ row files, dynamic
column detection, keyset-paginated filtering/search, MITRE ATT&CK-mapped
keyword scanning, a guided natural-language search box, and multi-sheet
report export. **No network calls, ever** — your data never leaves your
machine.

## Features

- Import large Excel (`.xlsx`/`.xls`) or CSV exports from tools like
  Microsoft Sentinel, Secureworks Taegis, and Microsoft Defender.
- Dynamic column detection — no per-source schema required.
- Fast filtering, full-text search, sorting, and CSV/XLSX export, all
  keyset-paginated for large files.
- Column-role detection (user, timestamp, command line, host, IP, etc.)
  with suggest-then-confirm — the tool never assumes, it always asks.
- UTC timestamp normalization, with an explicit prompt when a timestamp's
  timezone is ambiguous.
- An offline, built-in MITRE ATT&CK-style keyword library, scanned via
  Aho-Corasick pattern matching, extensible with your own custom
  categories.
- A local, rule-based "guided search" box — describe what you're looking
  for in plain language (e.g. *"show credential access for alice
  chronologically"*) and it previews its interpretation before running
  anything. No LLM, no network call, ever.
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

Requires [Rust](https://rustup.rs/) and the
[Tauri v2 prerequisites](https://v2.tauri.app/start/prerequisites/) for
your platform (Windows, macOS, or Linux).

```sh
cd src-tauri
cargo tauri build
```

This produces a native installer/bundle for your current platform in
`src-tauri/target/release/bundle/`. Cross-compiling for other platforms
follows the standard Tauri process — see the
[Tauri distribution docs](https://v2.tauri.app/distribute/).

To run in development mode:

```sh
cd src-tauri
cargo tauri dev
```

## Stack

Rust + [Tauri v2](https://v2.tauri.app/) · [calamine](https://github.com/tafia/calamine)
(Excel parsing) · SQLite via [rusqlite](https://github.com/rusqlite/rusqlite)
(bundled, FTS5 full-text search) · [aho-corasick](https://github.com/BurntSushi/aho-corasick)
(keyword matching) · [rust_xlsxwriter](https://github.com/jmcnamara/rust_xlsxwriter)
(report export) · plain HTML/CSS/vanilla JS frontend
([Tabulator.js](https://tabulator.info/) for the grid), no build step.

## License

No license file yet — all rights reserved by default until one is added.
