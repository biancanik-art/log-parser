//! Slow, real-file end-to-end checks against synthetic SIEM/EDR-style fixtures.
//! Not run by default `cargo test` — run explicitly:
//!   cargo test --test fixture_e2e -- --ignored --nocapture

use log_parser_lib::{db, excel_import, export, query, semantic};
use rusqlite::params;
use rust_xlsxwriter::Workbook;
use std::path::{Path, PathBuf};
use std::time::Instant;

const NUM_ROWS: usize = 120_000;
const MARKER: &str = "forensic_test_marker_XYZ";
const MARKER_ROW: usize = 60_000; // 1-indexed data row where the marker appears
const SEMANTIC_MARKER_TEXT: &str =
    "LSASS process memory access detected during operating system credential dumping";

fn testdata_dir() -> PathBuf {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("testdata");
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn headers() -> Vec<&'static str> {
    vec![
        "TimeGenerated",
        "Computer",
        "EventID",
        "Account",
        "SrcIpAddress",
        "DstIpAddress",
        "CommandLine",
        "ProcessName",
        "ParentProcessName",
        "LogonType",
        "TargetUserName",
        "SubjectUserName",
        "Activity",
        "AlertSeverity",
        "AlertName",
        "DeviceName",
        "FileName",
        "FileHash",
        "DestinationPort",
        "Protocol",
        "Description",
        "RuleName",
        "Status",
        "ThreatName",
        "ReportId",
        "Account", // deliberate duplicate header
        "",        // deliberate blank header
    ]
}

fn generate_large_fixture(path: &Path, num_rows: usize) {
    if path.exists() {
        return;
    }
    let mut workbook = Workbook::new();
    let worksheet = workbook.add_worksheet_with_constant_memory();
    for (col, h) in headers().iter().enumerate() {
        worksheet.write_string(0, col as u16, *h).unwrap();
    }

    for i in 0..num_rows {
        let row = (i as u32) + 1;
        let account = if i + 1 == MARKER_ROW {
            MARKER.to_string()
        } else {
            format!("user{}", i % 500)
        };

        worksheet
            .write_string(
                row,
                0,
                format!("2026-01-01T00:{:02}:{:02}Z", (i / 60) % 60, i % 60),
            )
            .unwrap();
        worksheet.write_string(row, 1, format!("HOST-{:04}", i % 300)).unwrap();
        worksheet.write_number(row, 2, (4624 + (i % 20)) as f64).unwrap();
        worksheet.write_string(row, 3, &account).unwrap();
        worksheet
            .write_string(row, 4, format!("10.0.{}.{}", (i / 256) % 256, i % 256))
            .unwrap();
        worksheet
            .write_string(row, 5, format!("192.168.{}.{}", (i / 256) % 256, i % 256))
            .unwrap();
        if i % 37 != 0 {
            worksheet
                .write_string(row, 6, format!("cmd.exe /c task{}", i % 100))
                .unwrap();
        }
        worksheet.write_string(row, 7, "cmd.exe").unwrap();
        worksheet.write_string(row, 8, "explorer.exe").unwrap();
        worksheet.write_number(row, 9, (i % 10) as f64).unwrap();
        worksheet.write_string(row, 10, format!("target{}", i % 50)).unwrap();
        worksheet.write_string(row, 11, format!("subject{}", i % 50)).unwrap();
        worksheet.write_string(row, 12, "Logon").unwrap();
        worksheet
            .write_string(row, 13, if i % 1000 == 0 { "High" } else { "Low" })
            .unwrap();
        worksheet.write_string(row, 14, "SuspiciousActivity").unwrap();
        worksheet.write_string(row, 15, format!("DEV-{:04}", i % 300)).unwrap();
        worksheet.write_string(row, 16, format!("file{}.exe", i % 200)).unwrap();
        worksheet.write_string(row, 17, format!("{:064x}", i)).unwrap();
        worksheet.write_number(row, 18, (443 + (i % 1000)) as f64).unwrap();
        worksheet
            .write_string(row, 19, if i % 2 == 0 { "TCP" } else { "UDP" })
            .unwrap();
        worksheet
            .write_string(
                row,
                20,
                if i + 1 == MARKER_ROW {
                    SEMANTIC_MARKER_TEXT
                } else {
                    "Synthetic test event description text"
                },
            )
            .unwrap();
        worksheet.write_string(row, 21, "RuleA").unwrap();
        worksheet.write_string(row, 22, "Resolved").unwrap();
        worksheet.write_string(row, 23, "TestThreat").unwrap();
        worksheet.write_string(row, 24, format!("R{i}")).unwrap();
        worksheet.write_string(row, 25, &account).unwrap(); // duplicate "Account" column
                                                              // column 26 (blank header) deliberately left empty for every row
    }

    workbook.save(path).unwrap();
}

fn generate_multi_sheet_fixture(path: &Path) {
    if path.exists() {
        return;
    }
    let mut workbook = Workbook::new();
    for (idx, name) in ["Sentinel", "Defender", "Taegis"].iter().enumerate() {
        let worksheet = workbook.add_worksheet();
        worksheet.set_name(*name).unwrap();
        worksheet.write_string(0, 0, "Col1").unwrap();
        worksheet.write_string(0, 1, "Col2").unwrap();
        for r in 1..=20u32 {
            worksheet.write_string(r, 0, format!("{name}-{r}")).unwrap();
            worksheet
                .write_number(r, 1, (idx as f64) * 100.0 + r as f64)
                .unwrap();
        }
    }
    workbook.save(path).unwrap();
}

#[test]
#[ignore]
fn large_fixture_import_and_query() {
    let dir = testdata_dir();
    let xlsx_path = dir.join("sentinel_sample_120k.xlsx");
    generate_large_fixture(&xlsx_path, NUM_ROWS);

    let db_path = std::env::temp_dir().join("log-parser-fixture-e2e.sqlite3");
    let _ = std::fs::remove_file(&db_path);

    let start = Instant::now();
    let result = excel_import::import_into_db(&xlsx_path, "Sheet1", &db_path, |done, total| {
        if done % 30000 == 0 {
            println!("import progress: {done}/{total}");
        }
    })
    .expect("import should succeed");
    let elapsed = start.elapsed();
    println!("Imported {} rows in {:?}", result.row_count, elapsed);
    assert_eq!(result.row_count, NUM_ROWS as i64);

    let conn = db::open(&db_path).unwrap();

    // full-text search finds exactly the marker row
    let mut spec = query::QuerySpec {
        search: Some(MARKER.to_string()),
        filters: vec![],
        expression: None,
        sort: None,
        cursor: None,
        limit: 50,
    };
    let page = query::query_rows(&conn, &result.columns, &spec).unwrap();
    assert_eq!(
        page.rows.len(),
        1,
        "expected exactly one marker row via full-text search"
    );
    assert_eq!(page.rows[0]["account"], serde_json::json!(MARKER));

    // equivalent column filter finds the same row
    spec.search = None;
    spec.filters.push(query::ColumnFilter {
        column: "account".to_string(),
        op: query::FilterOp::Equals,
        value: MARKER.to_string(),
    });
    let page = query::query_rows(&conn, &result.columns, &spec).unwrap();
    assert_eq!(page.rows.len(), 1);

    // AND'd with an unrelated non-matching filter -> zero rows
    spec.filters.push(query::ColumnFilter {
        column: "protocol".to_string(),
        op: query::FilterOp::Equals,
        value: "NOT_A_REAL_PROTOCOL".to_string(),
    });
    let page = query::query_rows(&conn, &result.columns, &spec).unwrap();
    assert_eq!(page.rows.len(), 0);

    // numeric filter against known distribution: destinationport is always in [443, 1442]
    let count_spec = query::QuerySpec {
        search: None,
        filters: vec![query::ColumnFilter {
            column: "destinationport".to_string(),
            op: query::FilterOp::GreaterThan,
            value: "10000".to_string(),
        }],
        expression: None,
        sort: None,
        cursor: None,
        limit: 50,
    };
    let count = query::count_rows(&conn, &result.columns, &count_spec).unwrap();
    assert_eq!(count, 0, "destinationport never exceeds 1442 by construction");

    // export a filtered subset to CSV and confirm it round-trips
    let export_dir = std::env::temp_dir().join("log-parser-fixture-export");
    std::fs::create_dir_all(&export_dir).unwrap();
    let csv_path = export_dir.join("marker_export.csv");
    let export_spec = query::QuerySpec {
        search: Some(MARKER.to_string()),
        filters: vec![],
        expression: None,
        sort: None,
        cursor: None,
        limit: 50,
    };
    let summary =
        export::export_csv(&conn, &result.columns, &export_spec, &csv_path, |_| {}).unwrap();
    assert_eq!(summary.row_count, 1);
    let contents = std::fs::read_to_string(&csv_path).unwrap();
    assert!(contents.contains(MARKER));

    let _ = std::fs::remove_file(&db_path);
    let _ = std::fs::remove_dir_all(&export_dir);
}

/// Release-only scale benchmark for the pinned all-MiniLM-L6-v2 deduplicated document index. It
/// imports the tracked 120,000-row, 27-column fixture and defaults to the complete dataset. The
/// known evidence row gets a paraphrasable marker in a real text column so retrieval quality is
/// asserted independently from exact Account/identifier matching.
///
/// Override the default sample with `LOG_PARSER_SEMANTIC_BENCH_ROWS`, up to all 120,000 rows.
#[test]
#[ignore = "loads the pinned MiniLM model and builds a real semantic index"]
fn semantic_fixture_build_and_search_benchmark() {
    const DEFAULT_BENCH_ROWS: usize = NUM_ROWS;

    let requested_rows = std::env::var("LOG_PARSER_SEMANTIC_BENCH_ROWS")
        .map(|value| {
            value
                .parse::<usize>()
                .expect("LOG_PARSER_SEMANTIC_BENCH_ROWS must be a positive integer")
        })
        .unwrap_or(DEFAULT_BENCH_ROWS);
    assert!(
        requested_rows > 0,
        "semantic benchmark needs at least one row"
    );
    let bench_rows = requested_rows.min(NUM_ROWS);
    if requested_rows > NUM_ROWS {
        println!(
            "semantic bench row request {requested_rows} exceeds fixture size; clamped to {NUM_ROWS}"
        );
    }

    let fixture_path = testdata_dir().join("sentinel_sample_120k.xlsx");
    generate_large_fixture(&fixture_path, NUM_ROWS);

    let unique = format!("{}-{}", std::process::id(), bench_rows);
    let temp_dir = std::env::temp_dir();
    let source_db_path = temp_dir.join(format!("log-parser-semantic-source-{unique}.sqlite3"));
    let bench_db_path = temp_dir.join(format!("log-parser-semantic-bench-{unique}.sqlite3"));
    let _ = std::fs::remove_file(&source_db_path);
    let _ = std::fs::remove_file(&bench_db_path);

    let import_started = Instant::now();
    let imported =
        excel_import::import_into_db(&fixture_path, "Sheet1", &source_db_path, |_, _| {})
            .expect("semantic benchmark fixture import should succeed");
    let import_elapsed = import_started.elapsed();
    assert_eq!(imported.row_count, NUM_ROWS as i64);
    assert_eq!(headers().len(), 27, "fixture declaration changed");
    assert_eq!(
        imported.columns.len(),
        26,
        "the fixture's deliberately blank trailing column should be omitted"
    );
    println!(
        "semantic bench import: rows={} declared_columns={} searchable_columns={} elapsed_ms={} rows_per_second={:.1}",
        imported.row_count,
        headers().len(),
        imported.columns.len(),
        import_elapsed.as_millis(),
        imported.row_count as f64 / import_elapsed.as_secs_f64()
    );

    // Copy the bounded sample into a clean on-disk database. Keeping the marker and the first
    // N-1 rows makes the benchmark deterministic while retaining realistic 27-column documents.
    let mut conn = db::open(&bench_db_path).expect("opening semantic benchmark database");
    db::create_schema(&conn, &imported.columns).expect("creating semantic benchmark schema");
    conn.execute(
        "ATTACH DATABASE ?1 AS source",
        params![source_db_path.to_string_lossy().as_ref()],
    )
    .expect("attaching imported fixture database");
    let identifiers = imported
        .columns
        .iter()
        .map(|column| db::quote_ident(&column.sql_name))
        .collect::<Vec<_>>()
        .join(", ");
    let insert_prefix = format!("INSERT INTO rows (row_num, {identifiers})");
    if bench_rows == NUM_ROWS {
        conn.execute_batch(&format!(
            "{insert_prefix} SELECT row_num, {identifiers} FROM source.rows ORDER BY row_num"
        ))
        .expect("copying complete fixture into semantic benchmark database");
    } else {
        conn.execute(
            &format!(
                "{insert_prefix} SELECT row_num, {identifiers} FROM source.rows WHERE row_num = ?1"
            ),
            [MARKER_ROW as i64],
        )
        .expect("copying forensic marker row into semantic benchmark database");
        if bench_rows > 1 {
            conn.execute(
                &format!(
                    "{insert_prefix}
                     SELECT row_num, {identifiers} FROM source.rows
                     WHERE row_num != ?1 ORDER BY row_num LIMIT ?2"
                ),
                params![MARKER_ROW as i64, (bench_rows - 1) as i64],
            )
            .expect("copying bounded fixture rows into semantic benchmark database");
        }
    }
    conn.execute_batch("DETACH DATABASE source")
        .expect("detaching imported fixture database");
    // Older checked-in/generated fixture copies predate the semantic text marker. Keep the
    // benchmark deterministic without weakening the independent exact Account marker assertion.
    conn.execute(
        "UPDATE rows SET description = ?1 WHERE row_num = ?2",
        params![SEMANTIC_MARKER_TEXT, MARKER_ROW as i64],
    )
    .expect("placing semantic marker in the Description text column");
    db::populate_fts(&conn, &imported.columns)
        .expect("populating FTS for a production-shaped benchmark database");

    let copied_rows: i64 = conn
        .query_row("SELECT COUNT(*) FROM rows", [], |row| row.get(0))
        .unwrap();
    assert_eq!(copied_rows, bench_rows as i64);
    let marker_account: String = conn
        .query_row(
            "SELECT account FROM rows WHERE row_num = ?1",
            [MARKER_ROW as i64],
            |row| row.get(0),
        )
        .expect("bounded fixture must retain its known evidence row");
    assert_eq!(marker_account, MARKER);
    let semantic_marker: String = conn
        .query_row(
            "SELECT description FROM rows WHERE row_num = ?1",
            [MARKER_ROW as i64],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(semantic_marker, SEMANTIC_MARKER_TEXT);
    let exact_marker_count = query::count_rows(
        &conn,
        &imported.columns,
        &query::QuerySpec {
            search: Some(MARKER.to_string()),
            ..query::QuerySpec::default()
        },
    )
    .expect("exact FTS should remain independently usable");
    assert_eq!(exact_marker_count, 1);

    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let resources = manifest.join("resources");
    let model_started = Instant::now();
    let model = semantic::SemanticModel::load(
        &resources.join(semantic::MODEL_RESOURCE_PATH),
        &resources.join(semantic::TOKENIZER_RESOURCE_PATH),
        &resources.join(semantic::CONFIG_RESOURCE_PATH),
    )
    .expect("loading pinned MiniLM semantic model");
    println!(
        "semantic bench model: load_ms={} measured_ms={} name={} version={}",
        model.load_time_ms,
        model_started.elapsed().as_millis(),
        semantic::MODEL_NAME,
        semantic::MODEL_VERSION
    );

    let database_bytes_before = std::fs::metadata(&bench_db_path).unwrap().len();
    let build_started = Instant::now();
    let summary = semantic::ensure_semantic_index(&mut conn, &imported.columns, &model)
        .expect("building semantic benchmark index");
    let build_elapsed = build_started.elapsed();
    assert_eq!(summary.rows_indexed, bench_rows as i64);
    assert!(!summary.from_cache);

    let embedding_bytes: i64 = conn
        .query_row(
            "SELECT COALESCE(SUM(length(embedding)), 0) FROM _semantic_v2_document",
            [],
            |row| row.get(0),
        )
        .unwrap();
    let unique_documents: i64 = conn
        .query_row("SELECT COUNT(*) FROM _semantic_v2_document", [], |row| {
            row.get(0)
        })
        .unwrap();
    let mappings: i64 = conn
        .query_row("SELECT COUNT(*) FROM _semantic_v2_mapping", [], |row| {
            row.get(0)
        })
        .unwrap();
    let database_bytes_after = std::fs::metadata(&bench_db_path).unwrap().len();
    let index_file_growth_bytes = database_bytes_after.saturating_sub(database_bytes_before);
    let build_seconds = build_elapsed.as_secs_f64();
    let build_throughput = summary.rows_indexed as f64 / build_seconds;
    println!(
        "semantic bench build: rows={} unique_documents={} mappings={} elapsed_ms={} summary_elapsed_ms={} rows_per_second={:.2} embedding_bytes={} database_growth_bytes={} database_bytes={}",
        summary.rows_indexed,
        unique_documents,
        mappings,
        build_elapsed.as_millis(),
        summary.elapsed_ms,
        build_throughput,
        embedding_bytes,
        index_file_growth_bytes,
        database_bytes_after
    );
    assert_eq!(summary.documents_indexed, unique_documents);
    assert_eq!(summary.mappings_written, mappings);
    if bench_rows == NUM_ROWS {
        assert!(
            build_elapsed <= std::time::Duration::from_secs(120),
            "full semantic v2 build exceeded the two-minute target: {build_elapsed:?}"
        );
    }

    let search_started = Instant::now();
    let selection = semantic::create_semantic_selection(
        &mut conn,
        &imported.columns,
        &model,
        "dump credentials by reading LSASS memory",
        semantic::SemanticSearchPolicy::default(),
    )
    .expect("creating production semantic document selection");
    let search_elapsed = search_started.elapsed();
    let selected_documents = conn
        .prepare(
            "SELECT d.normalized_text, sd.cosine_score
             FROM _semantic_v2_selection_doc sd
             JOIN _semantic_v2_document d ON d.doc_id = sd.doc_id
             WHERE sd.selection_id = ?1
             ORDER BY sd.cosine_score DESC, sd.doc_id ASC",
        )
        .unwrap()
        .query_map([&selection.selection_id], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, f32>(1)?))
        })
        .unwrap()
        .collect::<rusqlite::Result<Vec<_>>>()
        .unwrap();
    let (marker_rank, marker_document) = selected_documents
        .iter()
        .enumerate()
        .find(|(_, (text, _))| text.contains("lsass process memory access"))
        .map(|(index, document)| (index + 1, document))
        .unwrap_or_else(|| {
            panic!("semantic marker document was not retained; selected={selected_documents:?}")
        });
    assert!(
        marker_rank <= 5,
        "semantic marker ranked too low: rank={marker_rank} selected={selected_documents:?}"
    );
    let selection_spec = query::QuerySpec {
        expression: Some(query::QueryExpression::SemanticSelection {
            selection_id: selection.selection_id.clone(),
        }),
        cursor: Some(query::Cursor {
            sort_value: None,
            row_num: MARKER_ROW as i64 - 1,
        }),
        limit: 1,
        ..query::QuerySpec::default()
    };
    let marker_page = query::query_rows(&conn, &imported.columns, &selection_spec)
        .expect("querying the trusted semantic selection");
    assert_eq!(marker_page.rows[0]["row_num"], MARKER_ROW as i64);
    let marker_reasons =
        semantic::semantic_selection_reasons(&conn, &selection.selection_id, &[MARKER_ROW as i64])
            .unwrap();
    assert!(marker_reasons[&(MARKER_ROW as i64)]
        .iter()
        .any(|reason| reason.contains("lsass process memory access")));
    println!(
        "semantic bench search: elapsed_ms={} documents_above_threshold={} documents_retained={} rows_matched={} marker_rank={} marker_score={:.6} warnings={:?} top={:?}",
        search_elapsed.as_millis(),
        selection.documents_above_threshold,
        selection.documents_retained,
        selection.rows_matched,
        marker_rank,
        marker_document.1,
        selection.warnings,
        selected_documents.iter().take(10).collect::<Vec<_>>()
    );

    drop(conn);
    let _ = std::fs::remove_file(&source_db_path);
    let _ = std::fs::remove_file(&bench_db_path);
}

#[test]
#[ignore]
fn multi_sheet_listing_and_import() {
    let dir = testdata_dir();
    let xlsx_path = dir.join("multi_sheet_sample.xlsx");
    generate_multi_sheet_fixture(&xlsx_path);

    let sheets = excel_import::list_sheet_names(&xlsx_path).unwrap();
    assert_eq!(sheets, vec!["Sentinel", "Defender", "Taegis"]);

    for sheet in &sheets {
        let db_path = std::env::temp_dir().join(format!("log-parser-multisheet-{sheet}.sqlite3"));
        let _ = std::fs::remove_file(&db_path);
        let result = excel_import::import_into_db(&xlsx_path, sheet, &db_path, |_, _| {}).unwrap();
        assert_eq!(result.row_count, 20);
        let _ = std::fs::remove_file(&db_path);
    }
}
