//! Slow, real-file end-to-end checks against synthetic SIEM/EDR-style fixtures.
//! Not run by default `cargo test` — run explicitly:
//!   cargo test --test fixture_e2e -- --ignored --nocapture

use log_parser_lib::{db, excel_import, export, query};
use rust_xlsxwriter::Workbook;
use std::path::{Path, PathBuf};
use std::time::Instant;

const NUM_ROWS: usize = 120_000;
const MARKER: &str = "forensic_test_marker_XYZ";
const MARKER_ROW: usize = 60_000; // 1-indexed data row where the marker appears

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
            .write_string(row, 20, "Synthetic test event description text")
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
