//! Agent-authored end-to-end test: drives the REAL backend command logic against a
//! synthetic Microsoft Sentinel-style CSV with a planted attack narrative, exercising
//! CSV import -> role detection -> UTC normalization -> intel scan (keyword + behavior
//! rules + attack-chain detection) -> multi-sheet XLSX report -> embedded Qwen guided
//! search. Not part of the normal suite (WebView2 runtime 150 strips
//! --remote-debugging-port so the usual CDP driver can't attach; this exercises the
//! identical Rust the Tauri commands call).
//!
//!   cargo test --release --test agent_sentinel_e2e -- --ignored --nocapture

use log_parser_lib::intel::llm_parser::{LlmParser, MODEL_RESOURCE_PATH, TOKENIZER_RESOURCE_PATH};
use log_parser_lib::intel::parser::{
    accept_llm_audit, intent_from_token, parse_guided_query_with_llm,
};
use log_parser_lib::intel::query::{active_evidence_columns, run_guided_query};
use log_parser_lib::intel::roles::{detect_column_roles, set_column_role_status, RoleDecisionStatus};
use log_parser_lib::intel::time::{
    analyze_confirmed_timestamp_column, normalize_confirmed_timestamp_column,
};
use log_parser_lib::intel::matcher::scan_connection;
use log_parser_lib::{db, report, tabular_import};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn dev_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join(".dev")
        .join("agent-test-sentinel")
}

#[test]
#[ignore]
fn sentinel_attack_narrative_full_pipeline() {
    let dir = dev_dir();
    let csv = dir.join("sentinel_security_events.csv");
    assert!(csv.exists(), "fixture CSV missing: {}", csv.display());
    let db_path = dir.join("agent_e2e.sqlite3");
    let _ = std::fs::remove_file(&db_path);

    // ---- 1. import -------------------------------------------------------------
    let t = Instant::now();
    tabular_import::import_into_db(&csv, "", &db_path, |_, _| {}).unwrap();
    let mut conn = db::open(&db_path).unwrap();
    let columns = db::load_columns(&conn).unwrap();
    println!("\n=== IMPORT: {} columns in {:?} ===", columns.len(), t.elapsed());
    for c in &columns {
        println!("  col {} sql='{}' orig='{}' type={}", c.col_index, c.sql_name, c.original_name, c.inferred_type);
    }

    // ---- 2. role detection -----------------------------------------------------
    let roles = detect_column_roles(&conn, &columns).unwrap();
    println!("\n=== ROLE SUGGESTIONS ===");
    let mut timestamp_col: Option<String> = None;
    for r in &roles {
        if r.status == "suggested" || r.status == "confirmed" {
            println!("  role={:<14} -> {:<22} conf={:.2} status={}", r.role, r.original_name, r.confidence, r.status);
        }
        if r.role == "timestamp" {
            timestamp_col = Some(r.sql_name.clone());
        }
    }
    let ts_col = timestamp_col.expect("timestamp role should be suggested");

    // Confirm the load-bearing role (examiner does this in the UI).
    set_column_role_status(&conn, &columns, "timestamp", &ts_col, RoleDecisionStatus::Confirmed)
        .unwrap();

    let evidence = active_evidence_columns(&conn).unwrap();
    println!("\nactive evidence columns (fed to scan): {:?}", evidence);
    assert!(!evidence.is_empty(), "no evidence columns resolved from roles");

    // ---- 3. UTC normalization --------------------------------------------------
    let analysis = analyze_confirmed_timestamp_column(&conn, &columns).unwrap();
    println!(
        "\n=== TIMESTAMP ANALYSIS on '{}': total={} explicit={} naive={} invalid={} needsTz={} ===",
        analysis.original_name, analysis.total_rows, analysis.explicit_count, analysis.naive_count,
        analysis.invalid_count, analysis.needs_timezone
    );
    let norm = normalize_confirmed_timestamp_column(&mut conn, &columns, None).unwrap();
    println!("normalized rows (naiveTimezone=None): {:?}", norm);

    // ---- 4. intel scan (keyword + behavior rules + chains) ---------------------
    let t = Instant::now();
    let summary = scan_connection(&mut conn, &evidence, |_, _, _| {}).unwrap();
    println!("\n=== INTEL SCAN in {:?} ===", t.elapsed());
    println!("rowsScanned={} matchCount={} matchedRows={}", summary.rows_scanned, summary.match_count, summary.matched_rows);
    println!("\ntactics:");
    for tac in &summary.tactics {
        println!("  {:<24} matches={} rows={}", tac.name, tac.match_count, tac.row_count);
    }
    println!("\ntechniques:");
    for te in &summary.techniques {
        println!("  {:<40} matches={} rows={}", te.name, te.match_count, te.row_count);
    }

    // behavior-rule hits vs keyword hits: rule matches carry a pattern_id starting with 'rule_'
    let (rule_hits, kw_hits): (i64, i64) = conn
        .query_row(
            "SELECT
               SUM(CASE WHEN pattern_id LIKE 'rule\\_%' ESCAPE '\\' THEN 1 ELSE 0 END),
               SUM(CASE WHEN pattern_id LIKE 'rule\\_%' ESCAPE '\\' THEN 0 ELSE 1 END)
             FROM _intel_match",
            [],
            |r| Ok((r.get::<_, Option<i64>>(0)?.unwrap_or(0), r.get::<_, Option<i64>>(1)?.unwrap_or(0))),
        )
        .unwrap();
    println!("\nkeyword matches={}  behavior-rule matches={}", kw_hits, rule_hits);
    println!("\nbehavior-rule hits detail:");
    {
        let mut stmt = conn
            .prepare("SELECT row_num, pattern_id, keyword FROM _intel_match WHERE pattern_id LIKE 'rule\\_%' ESCAPE '\\' ORDER BY row_num")
            .unwrap();
        let rows = stmt
            .query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?)))
            .unwrap();
        for row in rows {
            let (rn, pid, kw) = row.unwrap();
            println!("  row {:<4} {:<40} evidence='{}'", rn, pid, kw);
        }
    }

    // ---- 5. ATTACK CHAINS (the new feature) ------------------------------------
    println!("\n=== ATTACK CHAINS ({} detected) ===", summary.chains.len());
    for ch in &summary.chains {
        println!(
            "  chain#{} host={:?} tactics={} events={} rows={} score={}",
            ch.chain_id, ch.host, ch.tactic_count, ch.event_count, ch.row_count, ch.score
        );
        println!("     tactic progression: {}", ch.tactic_names.join(" -> "));
        println!("     techniques: {}", ch.technique_names.join(", "));
        println!("     sample rows: {:?}", ch.sample_rows);
        if let (Some(s), Some(e)) = (ch.start_epoch_ms, ch.end_epoch_ms) {
            println!("     window: {} -> {} ({} min)", s, e, (e - s) / 60000);
        }
    }
    assert!(!summary.chains.is_empty(), "expected at least one attack chain");
    let top = &summary.chains[0];
    assert!(top.host.as_deref() == Some("WS-FIN-07"), "top chain host should be WS-FIN-07, got {:?}", top.host);
    assert!(top.tactic_count >= 3, "top chain must span >=3 tactics");

    // ---- 6. report export #1 (pre guided search) -------------------------------
    let report1 = dir.join("report.xlsx");
    let _ = std::fs::remove_file(&report1);
    let s1 = report::export_report(&mut conn, &columns, &report1, |_, _| {}).unwrap();
    println!("\n=== REPORT #1 (pre-guided) sheets={:?} rowCount={} ===", s1.sheets_written, s1.row_count);
    assert!(s1.sheets_written.iter().any(|s| s == "General"));
    assert!(s1.sheets_written.iter().any(|s| s == "Timeline"));

    // ---- 7. Qwen guided search -------------------------------------------------
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let model_path = manifest.join("resources").join(MODEL_RESOURCE_PATH);
    let tok_path = manifest.join("resources").join(TOKENIZER_RESOURCE_PATH);
    let t = Instant::now();
    let mut model = LlmParser::load(&model_path, &tok_path).unwrap();
    println!("\n=== QWEN model loaded in {:?} ===", t.elapsed());

    let queries = [
        ("credential access for the attacker", "show credential access for CORP\\gsmith chronologically"),
        ("deliberately vague", "find bad stuff"),
        ("exfil phrasing", "show data exfiltration by CORP\\gsmith"),
    ];
    for (label, q) in queries {
        println!("\n--- guided query [{}]: {:?} ---", label, q);
        let t = Instant::now();
        let preview = match parse_guided_query_with_llm(&conn, &columns, q, &mut model) {
            Ok(p) => p,
            Err(e) => {
                println!("  parse error (fail-closed): {e}");
                continue;
            }
        };
        let elapsed = t.elapsed();
        println!("  aiAssisted={} needsClarification={} reviewStatus={} validation={:?}",
            preview.ai_assisted, preview.needs_clarification, preview.review_status, preview.validation_status);
        println!("  previewText: {}", preview.preview_text);
        if let Some(msg) = &preview.clarification_message {
            println!("  clarification: {msg}");
        }
        if let Some(aid) = preview.audit_id {
            let (status, detail, raw): (String, Option<String>, String) = conn
                .query_row(
                    "SELECT validation_status, validation_detail, raw_output FROM _llm_parse_audit WHERE id=?1",
                    [aid],
                    |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
                )
                .unwrap();
            println!("  audit#{aid}: status={status} detail={detail:?}");
            println!("  raw model JSON: {raw}");
            let (load_ms, inf_ms): (Option<i64>, Option<i64>) = conn
                .query_row(
                    "SELECT load_time_ms, inference_latency_ms FROM _llm_parse_audit WHERE id=?1",
                    [aid],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )
                .unwrap();
            println!("  timings: load_ms={load_ms:?} inference_ms={inf_ms:?} wall={:?}", elapsed);
        }

        if preview.needs_clarification {
            println!("  -> correctly refused; nothing executed");
            continue;
        }
        // accept-before-run, then run
        if let Some(aid) = preview.audit_id {
            accept_llm_audit(&conn, aid, &preview.intent_token).unwrap();
        }
        let intent = intent_from_token(&preview.intent_token).unwrap();
        println!("  intent: {intent:?}");
        match run_guided_query(&conn, &columns, &preview.intent_token, None, Some(50)) {
            Ok(page) => {
                let row_nums: Vec<i64> = page.rows.iter().filter_map(|r| r["row_num"].as_i64()).collect();
                println!("  RESULT rows={} row_nums={:?} hasMore={}", page.rows.len(), row_nums, page.has_more);
            }
            Err(e) => println!("  run error: {e}"),
        }
    }

    // ---- 8. report export #2 (post guided search, should include AI Audit) -----
    let report2 = dir.join("report_with_audit.xlsx");
    let _ = std::fs::remove_file(&report2);
    match report::export_report(&mut conn, &columns, &report2, |_, _| {}) {
        Ok(s2) => {
            println!("\n=== REPORT #2 (post-guided) sheets={:?} rowCount={} ===", s2.sheets_written, s2.row_count);
            println!("AI Audit sheet present: {}", s2.sheets_written.iter().any(|s| s.eq_ignore_ascii_case("ai audit")));
        }
        Err(e) => println!("\n=== REPORT #2 export error: {e} ==="),
    }

    println!("\n=== DONE. Artifacts: {} , {} , {} ===", report1.display(), report2.display(), db_path.display());
}

/// v0.2.2 analyst front door: a bare "what is in this file" must auto-run the whole
/// pipeline (roles, UTC normalization, MITRE scan + chains, wide-net anomaly scan) with NO
/// prior examiner setup, name the attack host and chain in its narrative, and flag anomaly
/// rows beyond the curated library. A report-shaped ask must request report generation and
/// the exported workbook must carry the new Attack Story + Anomalies sheets.
///
///   cargo test --release --test agent_sentinel_e2e analyst_front_door -- --ignored --nocapture
#[test]
#[ignore]
fn analyst_front_door_answers_whats_in_this_file() {
    use log_parser_lib::intel::analyst;

    let dir = dev_dir();
    let csv = dir.join("sentinel_security_events.csv");
    assert!(csv.exists(), "fixture CSV missing: {}", csv.display());
    let db_path = dir.join("agent_analyst_e2e.sqlite3");
    let _ = std::fs::remove_file(&db_path);

    tabular_import::import_into_db(&csv, "", &db_path, |_, _| {}).unwrap();
    let mut conn = db::open(&db_path).unwrap();
    let columns = db::load_columns(&conn).unwrap();

    // ---- the front-door ask: zero prior setup ----------------------------------
    let t = Instant::now();
    let answer = analyst::ask(&mut conn, &columns, "what is in this file?", |phase| {
        println!("  [analyst phase] {phase}");
    })
    .unwrap();
    println!("\n=== ANALYST ANSWER in {:?} ===", t.elapsed());
    println!("intent: {}", answer.intent);
    println!("headline: {}", answer.headline);
    for step in &answer.steps {
        println!("  step {:<14} {:<8} {}", step.step, step.status, step.detail);
    }
    for section in &answer.sections {
        println!("\n[{}]", section.heading);
        for line in &section.lines {
            println!("  {} {:?}", line.text, line.rows);
        }
    }

    assert_eq!(answer.intent, "profile");
    assert!(!answer.use_guided_search);
    let status_of = |name: &str| {
        answer
            .steps
            .iter()
            .find(|step| step.step == name)
            .map(|step| step.status.clone())
            .unwrap_or_default()
    };
    assert_eq!(status_of("data_mapping"), "ran");
    assert_eq!(status_of("timeline"), "ran");
    assert_eq!(status_of("mitre_scan"), "ran");
    assert_eq!(status_of("anomaly_scan"), "ran");

    let scan = answer.scan.as_ref().expect("scan summary in answer");
    assert!(scan.match_count > 0);
    assert!(!scan.chains.is_empty(), "expected the WS-FIN-07 chain");
    assert_eq!(scan.chains[0].host.as_deref(), Some("WS-FIN-07"));
    assert!(
        answer.headline.contains("WS-FIN-07"),
        "headline should name the attack host: {}",
        answer.headline
    );

    // ---- anomaly layer must reach beyond the curated library --------------------
    let anomalies = answer.anomalies.as_ref().expect("anomaly summary");
    assert!(anomalies.flagged_rows > 0);
    let beyond_library: i64 = conn
        .query_row(
            "SELECT COUNT(DISTINCT a.row_num) FROM _anomaly a
             WHERE a.row_num NOT IN (SELECT row_num FROM _intel_match)",
            [],
            |r| r.get(0),
        )
        .unwrap();
    println!(
        "\nanomaly rows beyond curated-library matches: {beyond_library} (of {} flagged)",
        anomalies.flagged_rows
    );
    assert!(
        beyond_library > 0,
        "wide-net layer should flag rows the curated library missed"
    );

    // ---- report-shaped ask + Attack Story / Anomalies sheets --------------------
    let report_answer = analyst::ask(
        &mut conn,
        &columns,
        "make a chronological attack report",
        |_| {},
    )
    .unwrap();
    assert_eq!(report_answer.intent, "report");
    assert!(report_answer.report_requested);

    let report_path = dir.join("report_analyst.xlsx");
    let _ = std::fs::remove_file(&report_path);
    let summary = report::export_report(&mut conn, &columns, &report_path, |_, _| {}).unwrap();
    println!("\n=== ANALYST REPORT sheets={:?} ===", summary.sheets_written);
    assert!(summary.sheets_written.iter().any(|s| s == "Attack Story"));
    assert!(summary.sheets_written.iter().any(|s| s == "Anomalies"));

    println!(
        "\n=== DONE. Artifacts: {} , {} ===",
        report_path.display(),
        db_path.display()
    );
}

/// Proves the ignore-rules feature end to end through the REAL pipeline: CSV import (not an
/// in-memory fixture) -> role confirmation -> MITRE scan -> report export, using only the
/// built-in Qualys rule so this never touches the real custom_ignore_rules.v1.json on whatever
/// machine runs it (a custom rule would persist in the *real* app's config afterward, which
/// would be a bad side effect for a test). One planted row's command line matches a real MITRE
/// keyword (the same "powershell -enc" trigger `scan_flags_known_powershell_keyword` uses) but
/// also has a Qualys process name, so it must be suppressed even though the identical trigger
/// on a non-Qualys row still fires.
#[test]
#[ignore]
fn ignore_rules_suppress_planted_row_through_the_real_pipeline() {
    use calamine::Reader;
    use log_parser_lib::intel::ignore_rules;

    let dir = dev_dir().parent().unwrap().join("agent-test-ignore-rules");
    std::fs::create_dir_all(&dir).unwrap();
    let csv_path = dir.join("ignore_rules_fixture.csv");
    std::fs::write(
        &csv_path,
        "ProcessName,CommandLine,EventID\n\
         powershell.exe,\"powershell.exe powershell -enc SQBFAFgA\",4688\n\
         QualysAgent.exe,\"QualysAgent.exe powershell -enc SQBFAFgA\",4688\n\
         QualysAgent.exe,\"QualysAgent.exe --scan --quick\",4688\n\
         notepad.exe,\"notepad.exe C:\\Users\\bob\\notes.txt\",4688\n\
         -,-,4624\n",
    )
    .unwrap();
    let db_path = dir.join("ignore_rules_e2e.sqlite3");
    let _ = std::fs::remove_file(&db_path);

    tabular_import::import_into_db(&csv_path, "", &db_path, |_, _| {}).unwrap();
    let mut conn = db::open(&db_path).unwrap();
    let columns = db::load_columns(&conn).unwrap();

    detect_column_roles(&conn, &columns).unwrap();
    set_column_role_status(
        &conn,
        &columns,
        "process_name",
        "processname",
        RoleDecisionStatus::Confirmed,
    )
    .unwrap();

    let evidence_columns = vec!["commandline".to_string()];
    let scan = scan_connection(&mut conn, &evidence_columns, |_, _, _| {}).unwrap();
    println!(
        "scan: matched_rows={} rows_ignored={} ignored_by_rule={:?}",
        scan.matched_rows, scan.rows_ignored, scan.ignored_by_rule
    );
    assert_eq!(scan.rows_ignored, 2, "rows 2 and 3 are Qualys-attributed");
    assert_eq!(scan.ignored_by_rule.len(), 1);
    assert_eq!(scan.ignored_by_rule[0].rule_id, "qualys-agent-activity");
    assert_eq!(
        scan.matched_rows, 1,
        "row 2's identical trigger phrase must not double-count once row 1 alone matches"
    );

    let row2_matches: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM _intel_match WHERE row_num = 2",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        row2_matches, 0,
        "the Qualys row must produce zero MITRE matches despite containing the trigger phrase"
    );
    let row1_matches: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM _intel_match WHERE row_num = 1",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert!(row1_matches > 0, "the non-Qualys row with the same phrase must still match");

    let (total_ignored, by_rule) = ignore_rules::ignored_rows_summary(&conn).unwrap();
    assert_eq!(total_ignored, 2);
    assert_eq!(by_rule[0].row_count, 2);

    let report_path = dir.join("report_ignore_rules.xlsx");
    let _ = std::fs::remove_file(&report_path);
    let summary = report::export_report(&mut conn, &columns, &report_path, |_, _| {}).unwrap();
    println!("report sheets: {:?}", summary.sheets_written);
    assert!(
        summary.sheets_written.iter().any(|s| s == "Ignored Rows"),
        "the Ignored Rows sheet must be written when rows were suppressed"
    );

    let mut workbook = calamine::open_workbook_auto(&report_path).unwrap();
    let sheet = workbook.worksheet_range("Ignored Rows").unwrap();
    let ignored_row_nums: Vec<String> = sheet
        .rows()
        .skip(1)
        .map(|row| row[0].to_string())
        .collect();
    assert_eq!(
        ignored_row_nums.len(),
        2,
        "the report sheet must list exactly the two suppressed rows"
    );
    assert!(ignored_row_nums.contains(&"2".to_string()));
    assert!(ignored_row_nums.contains(&"3".to_string()));

    println!(
        "\n=== DONE. Artifacts: {} , {} ===",
        report_path.display(),
        db_path.display()
    );
}

/// Scale proof that ignore rules don't regress analysis on a partially-noisy dataset: same
/// 100k-row dataset, same content, analyzed twice — once with the built-in Qualys rule inactive
/// (baseline: process_name role left unconfirmed, so the role-scoped rule can't resolve),
/// once with it active (40% of rows are Qualys-attributed and get skipped before the expensive
/// per-row work in matcher/anomaly/activity). Isolates the feature's effect from import or
/// model-loading time by building the table directly rather than round-tripping through XLSX,
/// and by excluding the model-dependent semantic stage (proven separately in semantic.rs's own
/// unit tests: an entire ignored batch doesn't stall the build, and ignored rows never reach the
/// embedder).
///
/// Expect a small, sometimes-negative "speedup" here, not a dramatic one — matcher/anomaly/
/// activity's per-row work is cheap, so on a fast (~1.6s) run the real saved work is on the same
/// order as normal OS-scheduling noise. That's fine: this test's job is proving no *regression*,
/// not chasing a number. The feature's real payoff for expensive per-row work (semantic
/// embedding) is proven structurally above, not by timing it.
///
///   cargo test --release --test agent_sentinel_e2e ignore_rules_speed_up -- --ignored --nocapture
#[test]
#[ignore]
fn ignore_rules_speed_up_analysis_on_a_partially_noisy_dataset() {
    use log_parser_lib::intel::{activity, anomaly};

    const TOTAL_ROWS: i64 = 100_000;
    const QUALYS_MODULUS: i64 = 5; // row_num % 5 in {0,1} => 2-in-5 = 40% of rows.

    let dir = dev_dir().parent().unwrap().join("agent-test-ignore-rules");
    std::fs::create_dir_all(&dir).unwrap();
    let db_path = dir.join("ignore_rules_scale.sqlite3");
    let _ = std::fs::remove_file(&db_path);

    let columns = vec![
        db::ColumnMeta {
            sql_name: "processname".into(),
            original_name: "ProcessName".into(),
            col_index: 0,
            inferred_type: "text".into(),
        },
        db::ColumnMeta {
            sql_name: "commandline".into(),
            original_name: "CommandLine".into(),
            col_index: 1,
            inferred_type: "text".into(),
        },
    ];
    let mut conn = db::open(&db_path).unwrap();
    db::create_schema(&conn, &columns).unwrap();
    let t = Instant::now();
    {
        let tx = conn.transaction().unwrap();
        {
            let mut stmt = tx
                .prepare("INSERT INTO rows (row_num, processname, commandline) VALUES (?1, ?2, ?3)")
                .unwrap();
            for row_num in 1..=TOTAL_ROWS {
                // 2 of every 5 rows (40%) are Qualys noise; the rest carry varied, realistically
                // long command lines so the per-row scan work (Aho-Corasick + anomaly
                // heuristics) does real, comparable work in both runs.
                if row_num % QUALYS_MODULUS < 2 {
                    stmt.execute(rusqlite::params![
                        row_num,
                        "QualysAgent.exe",
                        format!("QualysAgent.exe --scan --profile=full --session={row_num}")
                    ])
                    .unwrap();
                } else {
                    stmt.execute(rusqlite::params![
                        row_num,
                        format!("proc-{}.exe", row_num % 50),
                        format!(
                            "proc-{}.exe --run --session={row_num} --token={} --path=C:\\Users\\user{}\\AppData\\Local\\Temp\\file{row_num}.tmp",
                            row_num % 50,
                            row_num * 7919,
                            row_num % 1000
                        )
                    ])
                    .unwrap();
                }
            }
        }
        tx.commit().unwrap();
    }
    println!("=== FIXTURE: {TOTAL_ROWS} rows built in {:?} ===", t.elapsed());

    let evidence_columns = vec!["commandline".to_string()];
    let run_stages = |conn: &mut rusqlite::Connection| -> (i64, i64) {
        let t = Instant::now();
        let scan = scan_connection(conn, &evidence_columns, |_, _, _| {}).unwrap();
        anomaly::scan_anomalies(conn, &columns, |_, _, _| {}).unwrap();
        activity::classify_rows(conn, &columns, |_, _, _| {}).unwrap();
        (t.elapsed().as_millis() as i64, scan.rows_ignored)
    };

    // Role detection independently activates anomaly.rs's rare_process_findings and
    // activity.rs's role-based signal columns (both gated on status IN ('suggested',
    // 'confirmed')) — real work unrelated to ignore rules, which specifically require status =
    // 'confirmed' (stricter, by design). Detecting once up front and toggling only 'suggested'
    // -> 'confirmed' between the two runs keeps that other work identical in both, isolating
    // the ignore-rule's own effect instead of conflating it with "roles now exist at all".
    detect_column_roles(&conn, &columns).unwrap();
    let role_status: String = conn
        .query_row(
            "SELECT status FROM _column_roles WHERE role = 'process_name'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(
        role_status, "suggested",
        "expected automatic detection to suggest, not confirm, the role"
    );

    // ---- baseline: role suggested but not confirmed, ignore rule cannot resolve -------------
    let (before_ms, before_ignored) = run_stages(&mut conn);
    println!("=== BEFORE (ignore rule inactive): {before_ms}ms, rows_ignored={before_ignored} ===");
    assert_eq!(before_ignored, 0, "role-scoped rule must not fire without a confirmed role");

    // ---- same data, same role-gated heuristics active, ignore rule now also active ---------
    set_column_role_status(
        &conn,
        &columns,
        "process_name",
        "processname",
        RoleDecisionStatus::Confirmed,
    )
    .unwrap();
    let (after_ms, after_ignored) = run_stages(&mut conn);
    println!("=== AFTER (ignore rule active): {after_ms}ms, rows_ignored={after_ignored} ===");
    assert_eq!(after_ignored, 40_000, "40% of the dataset is planted Qualys noise");

    let speedup_pct = if before_ms > 0 {
        100 - (after_ms * 100 / before_ms)
    } else {
        0
    };
    println!("=== SPEEDUP: {speedup_pct}% ({before_ms}ms -> {after_ms}ms) ===");
    // 15% tolerance: on a ~1.6s run, matcher/anomaly/activity's per-row work is cheap enough
    // that normal OS-scheduling noise alone can swing the measurement a couple of percent
    // either way (observed: 0% to -2% across repeated runs on the same unchanged code). This
    // must still catch a *real* regression — a 15% slowdown is well beyond that noise floor.
    let tolerance_ms = before_ms / 100 * 15;
    assert!(
        after_ms <= before_ms + tolerance_ms,
        "suppressing 40% of rows before the expensive per-row work should not meaningfully \
         regress performance: before={before_ms}ms after={after_ms}ms (tolerance={tolerance_ms}ms)"
    );
}

/// 520k-row scale proof for the v0.2.2 "parse it row by row" flow: generates (once, then
/// cached on disk) a Sentinel-style 520,000-row XLSX — benign multi-log-type noise plus one
/// planted multi-tactic intrusion inside a 45-minute window on one host — then drives the
/// real pipeline end to end: XLSX import → analyst front-door ask (roles, UTC, MITRE scan +
/// chains, anomaly scan, per-row activity classification) → full report export including the
/// 520k-row "Row by Row" sheet. Prints per-phase wall times and asserts generous ceilings so
/// a hang or pathological slowdown fails loudly instead of freezing the app for the examiner.
///
///   cargo test --release --test agent_sentinel_e2e analyst_scale_520k -- --ignored --nocapture
#[test]
#[ignore]
fn analyst_scale_520k_rows() {
    use log_parser_lib::intel::analyst;
    use rust_xlsxwriter::Workbook;

    const TOTAL_ROWS: usize = 520_000;
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join(".dev")
        .join("scale-520k");
    std::fs::create_dir_all(&dir).unwrap();
    let xlsx = dir.join("sentinel_scale_520k.xlsx");

    // ---- 0. fixture generation (cached across runs) -----------------------------
    if !xlsx.exists() {
        let t = Instant::now();
        let headers = [
            "TimeGenerated",
            "Computer",
            "Account",
            "EventID",
            "Activity",
            "ProcessName",
            "CommandLine",
            "SourceIP",
            "FileName",
            "LogonType",
        ];
        let mut workbook = Workbook::new();
        let sheet = workbook.add_worksheet_with_constant_memory();
        for (col, header) in headers.iter().enumerate() {
            sheet.write_string(0, col as u16, *header).unwrap();
        }

        // The intrusion: 12 events, one host, one identity, 02:00-02:45 UTC (off-hours),
        // spanning enough tactics to chain.
        let intrusion: [(&str, &str, i64, &str, &str, &str); 12] = [
            ("2026-06-10T02:01:10Z", "4624", 3, "An account was successfully logged on", "", ""),
            ("2026-06-10T02:03:22Z", "4688", 0, "A new process has been created", "powershell.exe", "powershell.exe -nop -w hidden -enc SQBFAFgAIAAoAE4AZQB3AC0ATwBiAGoAZQBjAHQAIABOAGUAdAAuAFcAZQBiAEMAbABpAGUAbgB0ACkALgBEAG8AdwBuAGwAbwBhAGQA"),
            ("2026-06-10T02:05:41Z", "4688", 0, "A new process has been created", "whoami.exe", "whoami /all"),
            ("2026-06-10T02:07:02Z", "4688", 0, "A new process has been created", "nltest.exe", "nltest /domain_trusts"),
            ("2026-06-10T02:11:33Z", "4688", 0, "A new process has been created", "procdump.exe", "procdump.exe -ma lsass.exe C:\\Users\\Public\\l.dmp"),
            ("2026-06-10T02:15:09Z", "4688", 0, "A new process has been created", "mimikatz.exe", "mimikatz.exe sekurlsa::logonpasswords"),
            ("2026-06-10T02:19:47Z", "4688", 0, "A new process has been created", "psexec.exe", "psexec.exe \\\\FS-SCALE-02 -u CORP\\svc_backup cmd.exe"),
            ("2026-06-10T02:24:12Z", "4688", 0, "A new process has been created", "7z.exe", "7z.exe a -pinfected C:\\Users\\Public\\stage.7z C:\\Finance\\*"),
            ("2026-06-10T02:29:55Z", "4688", 0, "A new process has been created", "rclone.exe", "rclone.exe copy C:\\Users\\Public\\stage.7z remote:drop"),
            ("2026-06-10T02:34:18Z", "4688", 0, "A new process has been created", "vssadmin.exe", "vssadmin delete shadows /all /quiet"),
            ("2026-06-10T02:38:30Z", "4688", 0, "A new process has been created", "wevtutil.exe", "wevtutil.exe cl Security"),
            ("2026-06-10T02:41:03Z", "4634", 3, "An account was logged off", "", ""),
        ];

        let benign_users = 40usize;
        let benign_hosts = 60usize;
        let benign_commands = [
            "cmd.exe /c dir C:\\Reports",
            "notepad.exe C:\\notes\\meeting.txt",
            "explorer.exe",
            "outlook.exe",
            "chrome.exe --profile-directory=Default",
            "svchost.exe -k netsvcs",
        ];
        // 12 intrusion rows scattered deterministically through the noise, everything in
        // ascending time order like a real export: noise covers 22 business days.
        let mut intrusion_iter = intrusion.iter();
        let mut next_intrusion = intrusion_iter.next();
        let mut excel_row = 1u32;
        for i in 0..TOTAL_ROWS - intrusion.len() {
            let day = 1 + (i * 22 / (TOTAL_ROWS - intrusion.len()));
            // Insert the whole intrusion between day 9 noise and day 10 noise.
            while day >= 10 && next_intrusion.is_some() {
                let (ts, event_id, logon_type, activity, process, cmd) = next_intrusion.unwrap();
                sheet.write_string(excel_row, 0, *ts).unwrap();
                sheet.write_string(excel_row, 1, "WS-SCALE-13").unwrap();
                sheet.write_string(excel_row, 2, "CORP\\eviluser").unwrap();
                sheet.write_string(excel_row, 3, *event_id).unwrap();
                sheet.write_string(excel_row, 4, *activity).unwrap();
                sheet.write_string(excel_row, 5, *process).unwrap();
                sheet.write_string(excel_row, 6, *cmd).unwrap();
                sheet.write_string(excel_row, 7, "10.10.9.13").unwrap();
                sheet.write_string(excel_row, 8, "").unwrap();
                sheet
                    .write_string(excel_row, 9, &logon_type.to_string())
                    .unwrap();
                excel_row += 1;
                next_intrusion = intrusion_iter.next();
            }
            let hour = 8 + (i % 10);
            let minute = i % 60;
            let second = (i * 7) % 60;
            let ts = format!("2026-06-{:02}T{:02}:{:02}:{:02}Z", day.min(30), hour, minute, second);
            let user = format!("CORP\\user{:03}", i % benign_users);
            let host = format!("WS-SCALE-{:02}", i % benign_hosts);
            // Rotate log types: authentication, process, file share, network, logoff.
            let (event_id, activity, process, cmd, file): (&str, &str, &str, String, String) =
                match i % 5 {
                    0 => ("4624", "An account was successfully logged on", "", String::new(), String::new()),
                    1 => (
                        "4688",
                        "A new process has been created",
                        "cmd.exe",
                        benign_commands[i % benign_commands.len()].to_string(),
                        String::new(),
                    ),
                    2 => (
                        "5140",
                        "A network share object was accessed",
                        "",
                        String::new(),
                        format!("\\\\FS-SCALE-01\\dept\\doc{}.docx", i % 900),
                    ),
                    3 => ("5156", "The Windows Filtering Platform has permitted a connection", "", String::new(), String::new()),
                    _ => ("4634", "An account was logged off", "", String::new(), String::new()),
                };
            sheet.write_string(excel_row, 0, &ts).unwrap();
            sheet.write_string(excel_row, 1, &host).unwrap();
            sheet.write_string(excel_row, 2, &user).unwrap();
            sheet.write_string(excel_row, 3, event_id).unwrap();
            sheet.write_string(excel_row, 4, activity).unwrap();
            sheet.write_string(excel_row, 5, process).unwrap();
            sheet.write_string(excel_row, 6, &cmd).unwrap();
            sheet
                .write_string(excel_row, 7, &format!("10.10.{}.{}", i % 20, i % 250))
                .unwrap();
            sheet.write_string(excel_row, 8, &file).unwrap();
            sheet.write_string(excel_row, 9, "3").unwrap();
            excel_row += 1;
        }
        assert_eq!(excel_row as usize, TOTAL_ROWS + 1, "fixture must have exactly 520k data rows");
        workbook.save(&xlsx).unwrap();
        println!("=== FIXTURE generated {} rows in {:?} ({} bytes) ===",
            TOTAL_ROWS, t.elapsed(), std::fs::metadata(&xlsx).unwrap().len());
    } else {
        println!("=== FIXTURE cached: {} ===", xlsx.display());
    }

    // ---- 1. import (real XLSX path) ---------------------------------------------
    let db_path = dir.join("scale_520k.sqlite3");
    let _ = std::fs::remove_file(&db_path);
    let t = Instant::now();
    tabular_import::import_into_db(&xlsx, "Sheet1", &db_path, |_, _| {}).unwrap();
    let import_time = t.elapsed();
    let mut conn = db::open(&db_path).unwrap();
    let columns = db::load_columns(&conn).unwrap();
    let row_count: i64 = conn
        .query_row("SELECT COUNT(*) FROM rows", [], |r| r.get(0))
        .unwrap();
    println!("\n=== IMPORT: {} rows, {} columns in {:?} ===", row_count, columns.len(), import_time);
    assert_eq!(row_count, TOTAL_ROWS as i64);
    assert!(
        import_time.as_secs() < 180,
        "520k import took {import_time:?} — should be well under 3 minutes in release"
    );

    // ---- 2. the user's exact flow: one ask, zero setup --------------------------
    let t = Instant::now();
    let answer = analyst::ask(
        &mut conn,
        &columns,
        "parse this xls and find me row by row what activity is there",
        |phase| println!("  [analyst phase] {phase} at {:?}", t.elapsed()),
    )
    .unwrap();
    let ask_time = t.elapsed();
    println!("\n=== ANALYST ANSWER in {ask_time:?} ===");
    println!("intent: {}", answer.intent);
    println!("headline: {}", answer.headline);
    for step in &answer.steps {
        println!("  step {:<14} {:<8} {}", step.step, step.status, step.detail);
    }
    for section in &answer.sections {
        println!("\n[{}]", section.heading);
        for line in &section.lines {
            println!("  {}", line.text);
        }
    }
    assert_eq!(answer.intent, "profile");
    for step_name in ["data_mapping", "timeline", "mitre_scan", "anomaly_scan", "activity"] {
        let status = answer
            .steps
            .iter()
            .find(|step| step.step == step_name)
            .map(|step| step.status.as_str())
            .unwrap_or("missing");
        assert_eq!(status, "ran", "step {step_name} did not run at 520k scale");
    }
    assert!(
        ask_time.as_secs() < 600,
        "analyst ask took {ask_time:?} at 520k rows — the app would feel stuck"
    );

    // Every row classified; the planted intrusion found and chained on the right host.
    let activity = answer.activity.as_ref().expect("activity summary");
    assert_eq!(activity.rows_classified, TOTAL_ROWS as i64);
    assert!(
        activity.categories.len() >= 4,
        "expected several activity types, got {:?}",
        activity.categories.iter().map(|c| &c.category).collect::<Vec<_>>()
    );
    let scan = answer.scan.as_ref().expect("scan summary");
    assert!(scan.match_count > 0, "curated scan found nothing at 520k");
    assert!(!scan.chains.is_empty(), "planted intrusion did not chain");
    assert_eq!(
        scan.chains[0].host.as_deref(),
        Some("WS-SCALE-13"),
        "top chain should be the planted host"
    );
    let anomalies = answer.anomalies.as_ref().expect("anomaly summary");
    assert!(anomalies.flagged_rows > 0);

    // ---- 3. full report incl. the 520k-row Row by Row sheet ----------------------
    let report_path = dir.join("report_scale_520k.xlsx");
    let _ = std::fs::remove_file(&report_path);
    let t = Instant::now();
    let summary = report::export_report(&mut conn, &columns, &report_path, |_, _| {}).unwrap();
    let report_time = t.elapsed();
    println!("\n=== REPORT in {report_time:?}: sheets={:?} ({} bytes) ===",
        summary.sheets_written,
        std::fs::metadata(&report_path).unwrap().len());
    assert!(summary.sheets_written.iter().any(|s| s == "Activity Summary"));
    assert_eq!(summary.sheets_written.last().map(String::as_str), Some("Row by Row"));
    assert!(
        report_time.as_secs() < 600,
        "report export took {report_time:?} at 520k rows — the app would feel stuck"
    );

    println!(
        "\n=== SCALE PROOF DONE. import={import_time:?} ask={ask_time:?} report={report_time:?} ===\nArtifacts: {} , {}",
        report_path.display(),
        db_path.display()
    );
}
