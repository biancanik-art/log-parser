(() => {
  const { invoke } = window.__TAURI__.core;
  const { listen } = window.__TAURI__.event;

  // -- element refs -----------------------------------------------------
  const openFileBtn = document.getElementById("open-file-btn");
  const removeFileBtn = document.getElementById("remove-file-btn");
  const fileInfo = document.getElementById("file-info");
  const searchBox = document.getElementById("search-box");
  const guidedSearchForm = document.getElementById("guided-search-form");
  const guidedSearchBox = document.getElementById("guided-search-box");
  const guidedSearchSubmit = document.getElementById("guided-search-submit");
  const reportExportBtn = document.getElementById("report-export-btn");
  const exportCsvBtn = document.getElementById("export-csv-btn");
  const exportXlsxBtn = document.getElementById("export-xlsx-btn");

  const progressWrap = document.getElementById("progress-bar-wrap");
  const progressFill = document.getElementById("progress-fill");
  const progressLabel = document.getElementById("progress-label");

  const guidedQueryPanel = document.getElementById("guided-query-panel");
  const guidedAiStatus = document.getElementById("guided-ai-status");
  const guidedPreviewText = document.getElementById("guided-preview-text");
  const guidedClarification = document.getElementById("guided-clarification");
  const guidedRunBtn = document.getElementById("guided-run-btn");
  const guidedRejectBtn = document.getElementById("guided-reject-btn");
  const guidedResetBtn = document.getElementById("guided-reset-btn");
  const guidedPanelClose = document.getElementById("guided-panel-close");

  const roleReviewPanel = document.getElementById("role-review-panel");
  const roleList = document.getElementById("role-list");
  const rolePanelStatus = document.getElementById("role-panel-status");
  const rolePanelClose = document.getElementById("role-panel-close");

  const timezonePanel = document.getElementById("timezone-panel");
  const timezoneSummary = document.getElementById("timezone-summary");
  const timezoneSamples = document.getElementById("timezone-samples");
  const timezoneInput = document.getElementById("timezone-input");
  const timezoneUtcBtn = document.getElementById("timezone-utc-btn");
  const timezoneNormalizeBtn = document.getElementById("timezone-normalize-btn");
  const timezonePanelClose = document.getElementById("timezone-panel-close");

  const reportSummaryPanel = document.getElementById("report-summary-panel");
  const reportSummaryText = document.getElementById("report-summary-text");
  const reportSummaryClose = document.getElementById("report-summary-close");

  const sheetPicker = document.getElementById("sheet-picker");
  const sheetSelect = document.getElementById("sheet-select");
  const sheetLoadBtn = document.getElementById("sheet-load-btn");

  const sortColumn = document.getElementById("sort-column");
  const sortDirection = document.getElementById("sort-direction");
  const filterList = document.getElementById("filter-list");
  const addFilterBtn = document.getElementById("add-filter-btn");
  const applyBtn = document.getElementById("apply-btn");
  const clearBtn = document.getElementById("clear-btn");
  const filterRowTemplate = document.getElementById("filter-row-template");
  const suspiciousScanBtn = document.getElementById("suspicious-scan-btn");
  const reviewRolesBtn = document.getElementById("review-roles-btn");
  const evidenceColumnsLabel = document.getElementById("evidence-columns-label");
  const intelScanSummary = document.getElementById("intel-scan-summary");

  const rowCountLabel = document.getElementById("row-count-label");
  const pageLabel = document.getElementById("page-label");
  const prevPageBtn = document.getElementById("prev-page-btn");
  const nextPageBtn = document.getElementById("next-page-btn");

  // -- state --------------------------------------------------------------
  const PAGE_SIZE = 300;

  let columns = []; // ColumnMeta[] from ImportSummary
  let table = null;
  let currentPath = null;
  let currentSheet = null;

  let spec = { search: null, filters: [], sort: null, cursor: null, limit: PAGE_SIZE };
  let cursorStack = []; // for Prev navigation
  let nextCursor = null;
  let hasMore = false;
  let pageIndex = 1;
  let totalCount = null;

  let queryMode = "normal";
  let guidedParseResult = null;
  let guidedIntentToken = null;
  let guidedAuditId = null;
  let guidedReviewStatus = null;
  let guidedContextRevision = 0;
  let guidedParseRequestSequence = 0;
  let guidedActiveParse = null;
  let guidedActionSequence = 0;
  let guidedActiveAction = null;
  let guidedActiveQuery = null;
  let sheetLoadInFlight = false;
  let controlsEnabled = false;

  let columnRoleSuggestions = [];
  let timestampAnalysis = null;
  let timestampNormalizationSummary = null;
  let intelScanSummaryResult = null;
  let reportSummaryResult = null;
  let roleDetectionInFlight = false;
  let roleDetectionError = null;
  let intelScanInFlight = false;

  const EVIDENCE_ROLES = new Set([
    "command_line",
    "process_name",
    "file_name",
    "host",
    "text_evidence",
  ]);

  // -- helpers --------------------------------------------------------------

  function setControlsEnabled(enabled) {
    controlsEnabled = enabled;
    removeFileBtn.disabled = !enabled;
    searchBox.disabled = !enabled;
    guidedSearchBox.disabled = !enabled;
    guidedSearchSubmit.disabled = !enabled;
    reportExportBtn.disabled = !enabled;
    exportCsvBtn.disabled = !enabled;
    exportXlsxBtn.disabled = !enabled;
    addFilterBtn.disabled = !enabled;
    applyBtn.disabled = !enabled;
    clearBtn.disabled = !enabled;
    reviewRolesBtn.disabled = !enabled;
    if (enabled) {
      updateEvidenceColumnsUi();
    } else {
      suspiciousScanBtn.disabled = true;
    }
    updateGuidedInteractionControls();
  }

  function showProgress(label, fraction) {
    progressWrap.classList.remove("hidden");
    progressLabel.textContent = label;
    progressFill.style.width = `${Math.max(0, Math.min(1, fraction)) * 100}%`;
  }

  function hideProgress() {
    progressWrap.classList.add("hidden");
  }

  function updateGuidedInteractionControls() {
    const parsing = guidedActiveParse !== null;
    const actionInFlight = guidedActiveAction !== null;
    const queryInFlight = guidedActiveQuery !== null;
    guidedSearchSubmit.disabled =
      !controlsEnabled || columns.length === 0 || sheetLoadInFlight || parsing || actionInFlight || queryInFlight;
    guidedRunBtn.disabled = actionInFlight || queryInFlight;
    guidedRejectBtn.disabled = actionInFlight || queryInFlight;
    guidedResetBtn.disabled = actionInFlight || queryInFlight;
    // Keep Close available while parsing so it can cancel a slow preview, but do not let it
    // race the decision implicit in Run or an explicit Reject/Edit request.
    guidedPanelClose.disabled = actionInFlight || queryInFlight;
  }

  function invalidateGuidedContext() {
    guidedContextRevision += 1;
    guidedActiveParse = null;
  }

  function resetGuidedQueryUi() {
    invalidateGuidedContext();
    queryMode = "normal";
    guidedParseResult = null;
    guidedIntentToken = null;
    guidedAuditId = null;
    guidedReviewStatus = null;

    guidedSearchBox.value = "";
    guidedQueryPanel.classList.add("hidden");
    guidedPreviewText.textContent = "";
    guidedAiStatus.textContent = "";
    guidedAiStatus.classList.add("hidden");
    guidedClarification.textContent = "";
    guidedClarification.classList.add("hidden");
    guidedRunBtn.textContent = "Run";
    guidedRunBtn.classList.add("hidden");
    guidedRejectBtn.classList.add("hidden");
    guidedResetBtn.classList.add("hidden");
    updateGuidedInteractionControls();
  }

  function resetIntelUiState() {
    resetGuidedQueryUi();
    columnRoleSuggestions = [];
    timestampAnalysis = null;
    timestampNormalizationSummary = null;
    intelScanSummaryResult = null;
    reportSummaryResult = null;
    roleDetectionInFlight = false;
    roleDetectionError = null;
    intelScanInFlight = false;

    roleList.innerHTML = "";
    rolePanelStatus.textContent = "";
    roleReviewPanel.classList.add("hidden");

    timezoneInput.value = "";
    timezoneSummary.textContent = "";
    timezoneSamples.textContent = "";
    timezoneSamples.classList.add("hidden");
    timezonePanel.classList.add("hidden");

    reportSummaryText.textContent = "";
    reportSummaryPanel.classList.add("hidden");

    renderScanSummary(null);
    updateEvidenceColumnsUi();
  }

  function guidedParseIsCurrent(request) {
    return (
      guidedActiveParse === request &&
      guidedContextRevision === request.contextRevision &&
      currentPath === request.path &&
      currentSheet === request.sheet
    );
  }

  function cancelActiveGuidedParse() {
    if (guidedActiveParse === null) return;
    guidedActiveParse = null;
    hideProgress();
    updateGuidedInteractionControls();
  }

  function beginGuidedAction(type, { allowDuringParse = false } = {}) {
    if (
      guidedActiveAction !== null ||
      guidedActiveQuery !== null ||
      (!allowDuringParse && guidedActiveParse !== null)
    ) {
      return null;
    }
    const action = {
      id: ++guidedActionSequence,
      type,
      contextRevision: guidedContextRevision,
      auditId: guidedAuditId,
      intentToken: guidedIntentToken,
    };
    guidedActiveAction = action;
    updateGuidedInteractionControls();
    return action;
  }

  function guidedActionIsCurrent(action) {
    return (
      guidedActiveAction === action &&
      guidedContextRevision === action.contextRevision &&
      guidedAuditId === action.auditId &&
      guidedIntentToken === action.intentToken
    );
  }

  function finishGuidedAction(action) {
    if (guidedActiveAction === action) {
      guidedActiveAction = null;
      updateGuidedInteractionControls();
    }
  }

  function setGuidedReviewStatus(status) {
    guidedReviewStatus = status;
    if (guidedParseResult) {
      guidedParseResult = { ...guidedParseResult, reviewStatus: status };
    }
    guidedAiStatus.textContent = `AI-assisted interpretation \u2022 ${status} by examiner \u2022 processed locally`;
  }

  function formatRoleName(role) {
    return role.replace(/_/g, " ");
  }

  function columnDisplayName(sqlName) {
    const column = columns.find((c) => c.sqlName === sqlName);
    return column ? column.originalName : sqlName;
  }

  function upsertRoleSuggestion(updated) {
    const idx = columnRoleSuggestions.findIndex((row) => row.role === updated.role);
    if (idx === -1) {
      columnRoleSuggestions.push(updated);
    } else {
      columnRoleSuggestions[idx] = updated;
    }
  }

  function confirmedEvidenceColumns() {
    const out = [];
    columnRoleSuggestions.forEach((row) => {
      if (row.status === "confirmed" && EVIDENCE_ROLES.has(row.role) && !out.includes(row.sqlName)) {
        out.push(row.sqlName);
      }
    });
    return out;
  }

  function updateEvidenceColumnsUi() {
    const evidenceColumns = confirmedEvidenceColumns();
    suspiciousScanBtn.disabled = columns.length === 0 || evidenceColumns.length === 0 || intelScanInFlight;
    if (columns.length === 0) {
      evidenceColumnsLabel.textContent = "Load a file first.";
    } else if (evidenceColumns.length === 0) {
      evidenceColumnsLabel.textContent = "Confirm evidence roles first.";
    } else {
      evidenceColumnsLabel.textContent = `Evidence: ${evidenceColumns
        .map(columnDisplayName)
        .join(", ")}`;
    }
  }

  function renderRoleSuggestions() {
    roleList.innerHTML = "";

    if (columnRoleSuggestions.length === 0) {
      roleReviewPanel.classList.remove("hidden");
      rolePanelStatus.textContent = roleDetectionInFlight
        ? "Detecting column roles..."
        : roleDetectionError
          ? `Column role detection failed: ${roleDetectionError}`
        : "No column role suggestions were found.";
      updateEvidenceColumnsUi();
      return;
    }

    columnRoleSuggestions.forEach((suggestion) => {
      const row = document.createElement("div");
      row.className = "role-row";

      const roleTitle = document.createElement("div");
      const roleName = document.createElement("div");
      roleName.className = "role-title";
      roleName.textContent = formatRoleName(suggestion.role);
      roleTitle.appendChild(roleName);
      if (suggestion.role === "command_line" && suggestion.status !== "confirmed") {
        const warning = document.createElement("div");
        warning.className = "role-warning";
        warning.textContent = "Requires examiner confirmation before trusted.";
        roleTitle.appendChild(warning);
      }

      const column = document.createElement("div");
      column.className = "role-column";
      column.textContent = `${suggestion.originalName || columnDisplayName(suggestion.sqlName)} (${suggestion.sqlName})`;

      const meta = document.createElement("div");
      const badge = document.createElement("span");
      badge.className = `role-badge ${suggestion.status}`;
      badge.textContent = suggestion.status;
      meta.appendChild(badge);
      const confidence = document.createElement("div");
      confidence.className = "role-confidence";
      confidence.textContent = `${Math.round((suggestion.confidence || 0) * 100)}% confidence`;
      meta.appendChild(confidence);

      const actions = document.createElement("div");
      actions.className = "role-actions";
      const confirmBtn = document.createElement("button");
      confirmBtn.className = "btn btn-small";
      confirmBtn.textContent = "Confirm";
      confirmBtn.disabled = suggestion.status === "confirmed";
      confirmBtn.addEventListener("click", () => {
        setColumnRoleStatus(suggestion.role, suggestion.sqlName, "confirmed").catch((err) =>
          alert(`Role update failed: ${err}`)
        );
      });
      const rejectBtn = document.createElement("button");
      rejectBtn.className = "btn btn-small";
      rejectBtn.textContent = "Reject";
      rejectBtn.disabled = suggestion.status === "rejected";
      rejectBtn.addEventListener("click", () => {
        setColumnRoleStatus(suggestion.role, suggestion.sqlName, "rejected").catch((err) =>
          alert(`Role update failed: ${err}`)
        );
      });
      actions.append(confirmBtn, rejectBtn);

      row.append(roleTitle, column, meta, actions);

      if (suggestion.reasons && suggestion.reasons.length > 0) {
        const reasons = document.createElement("div");
        reasons.className = "role-reasons";
        reasons.textContent = suggestion.reasons.join("; ");
        row.appendChild(reasons);
      }

      roleList.appendChild(row);
    });

    const commandLine = columnRoleSuggestions.find((row) => row.role === "command_line");
    rolePanelStatus.textContent =
      commandLine && commandLine.status !== "confirmed"
        ? "Command-line evidence is not trusted until confirmed."
        : "Confirmed evidence roles can be scanned for suspicious matches.";
    updateEvidenceColumnsUi();

    // Once every suggestion has been explicitly confirmed or rejected, there's nothing
    // left for the examiner to decide — collapse the panel automatically so the sheet
    // gets full screen room. "Review column roles" in the sidebar brings it back.
    const allResolved = columnRoleSuggestions.every((row) => row.status !== "suggested");
    roleReviewPanel.classList.toggle("hidden", allResolved);
  }

  function renderScanSummary(summary) {
    intelScanSummary.innerHTML = "";
    if (!summary) return;

    const header = document.createElement("div");
    header.className = "sidebar-note";
    header.textContent = `${summary.matchedRows.toLocaleString()} matched rows, ${summary.matchCount.toLocaleString()} matches`;
    intelScanSummary.appendChild(header);

    if (summary.customLibraryError) {
      const warning = document.createElement("div");
      warning.className = "sidebar-note";
      warning.textContent = `Custom library skipped: ${summary.customLibraryError}`;
      intelScanSummary.appendChild(warning);
    }

    const tactics = summary.tactics || [];
    if (tactics.length === 0) {
      const empty = document.createElement("div");
      empty.className = "sidebar-note";
      empty.textContent = "No tactic matches found.";
      intelScanSummary.appendChild(empty);
      return;
    }

    tactics.slice(0, 10).forEach((tactic) => {
      const row = document.createElement("div");
      row.className = "scan-summary-row";
      const name = document.createElement("span");
      name.textContent = tactic.name;
      const count = document.createElement("span");
      count.className = "scan-summary-count";
      count.textContent = `${tactic.rowCount.toLocaleString()} rows`;
      row.append(name, count);
      intelScanSummary.appendChild(row);
    });
  }

  function renderGuidedPreview(result) {
    guidedParseResult = result;
    guidedIntentToken = result.intentToken || null;
    guidedAuditId = Number.isInteger(result.auditId) ? result.auditId : null;
    guidedReviewStatus = result.reviewStatus || null;
    guidedRunBtn.textContent = "Run";
    guidedQueryPanel.classList.remove("hidden");
    guidedPreviewText.textContent = result.previewText || "No preview was returned.";

    if (result.aiAssisted) {
      const validation = result.validationStatus ? ` • ${result.validationStatus.replace(/_/g, " ")}` : "";
      guidedAiStatus.textContent = `AI-assisted interpretation • ${guidedReviewStatus || "unreviewed"}${validation} • processed locally`;
      guidedAiStatus.classList.remove("hidden");
      guidedRejectBtn.classList.toggle("hidden", guidedReviewStatus !== "unreviewed");
    } else {
      guidedAiStatus.textContent = "Deterministic safety check • no model inference used";
      guidedAiStatus.classList.remove("hidden");
      guidedRejectBtn.classList.add("hidden");
    }

    if (
      result.needsClarification ||
      guidedAuditId === null ||
      !["unreviewed", "accepted"].includes(guidedReviewStatus)
    ) {
      guidedClarification.textContent = result.clarificationMessage || "Clarification is needed before this can run.";
      guidedClarification.classList.remove("hidden");
      guidedRunBtn.classList.add("hidden");
    } else {
      guidedClarification.textContent = "";
      guidedClarification.classList.add("hidden");
      guidedRunBtn.classList.remove("hidden");
    }
    guidedResetBtn.classList.toggle("hidden", queryMode !== "guided");
    updateGuidedInteractionControls();
  }

  function renderReportSummary(summary) {
    reportSummaryResult = summary;
    const sheets = summary.sheetsWritten && summary.sheetsWritten.length > 0
      ? summary.sheetsWritten.join(", ")
      : "(none reported)";
    reportSummaryText.textContent = `Wrote ${summary.rowCount.toLocaleString()} rows to ${summary.destPath}. Sheets: ${sheets}.`;
    reportSummaryPanel.classList.remove("hidden");
  }

  function showTimezonePrompt(analysis) {
    timestampAnalysis = analysis;
    timezoneSummary.textContent = `${analysis.originalName} has ${analysis.naiveCount.toLocaleString()} timestamp value(s) without an explicit timezone.`;
    if (analysis.sampleNaiveValues && analysis.sampleNaiveValues.length > 0) {
      timezoneSamples.textContent = `Samples: ${analysis.sampleNaiveValues.join("; ")}`;
      timezoneSamples.classList.remove("hidden");
    } else {
      timezoneSamples.textContent = "";
      timezoneSamples.classList.add("hidden");
    }
    timezoneInput.value = "";
    timezonePanel.classList.remove("hidden");
  }

  function currentFilterValues() {
    const rows = filterList.querySelectorAll(".filter-row");
    const out = [];
    rows.forEach((row) => {
      const column = row.querySelector(".filter-column").value;
      const op = row.querySelector(".filter-op").value;
      const value = row.querySelector(".filter-value").value;
      if (column) {
        out.push({ column, op, value });
      }
    });
    return out;
  }

  function addFilterRow() {
    const frag = filterRowTemplate.content.cloneNode(true);
    const row = frag.querySelector(".filter-row");
    const colSelect = row.querySelector(".filter-column");
    columns.forEach((c) => {
      const opt = document.createElement("option");
      opt.value = c.sqlName;
      opt.textContent = c.originalName;
      colSelect.appendChild(opt);
    });
    row.querySelector(".filter-remove-btn").addEventListener("click", () => {
      row.remove();
    });
    filterList.appendChild(row);
  }

  function resetPagination() {
    spec.cursor = null;
    cursorStack = [];
    nextCursor = null;
    hasMore = false;
    pageIndex = 1;
  }

  function buildSpecFromControls(forExport) {
    const s = {
      search: searchBox.value.trim() || null,
      filters: currentFilterValues(),
      sort: sortColumn.value
        ? { column: sortColumn.value, direction: sortDirection.value }
        : null,
      cursor: forExport ? null : spec.cursor,
      limit: PAGE_SIZE,
    };
    return s;
  }

  async function refreshCount() {
    if (queryMode === "guided") {
      totalCount = null;
      updateRowCountLabel();
      return;
    }

    try {
      totalCount = await invoke("count_rows", { spec });
      updateRowCountLabel();
    } catch (err) {
      console.error("count_rows failed", err);
    }
  }

  function updateRowCountLabel() {
    const shown = table ? table.getDataCount() : 0;
    if (totalCount === null) {
      rowCountLabel.textContent =
        queryMode === "guided" ? `${shown} guided rows on this page` : `${shown} rows on this page`;
    } else {
      rowCountLabel.textContent = `${totalCount.toLocaleString()} matching rows`;
    }
    pageLabel.textContent = `page ${pageIndex}`;
  }

  async function refreshData() {
    // Disabled synchronously (before the first await below) so a rapid double-click on
    // Prev/Next can't fire a second query_rows() while this one is still in flight and read a
    // stale nextCursor — the button is unclickable for the whole round trip either way.
    const modeAtStart = queryMode;
    const isGuidedRequest = modeAtStart === "guided";
    if (isGuidedRequest && guidedActiveQuery !== null) return null;

    const guidedQueryRequest = isGuidedRequest
      ? {
          contextRevision: guidedContextRevision,
          auditId: guidedAuditId,
          intentToken: guidedIntentToken,
          cursor: spec.cursor,
          limit: spec.limit,
          table,
        }
      : null;
    if (guidedQueryRequest) {
      guidedActiveQuery = guidedQueryRequest;
      updateGuidedInteractionControls();
    }

    const guidedQueryIsCurrent = () =>
      !guidedQueryRequest ||
      (guidedContextRevision === guidedQueryRequest.contextRevision &&
        queryMode === "guided" &&
        guidedAuditId === guidedQueryRequest.auditId &&
        guidedIntentToken === guidedQueryRequest.intentToken &&
        table === guidedQueryRequest.table);

    prevPageBtn.disabled = true;
    nextPageBtn.disabled = true;
    showProgress(isGuidedRequest ? "Running guided query..." : "Filtering...", 0.5);
    try {
      const page =
        isGuidedRequest
          ? await invoke("run_guided_query", {
              intentToken: guidedQueryRequest.intentToken,
              auditId: guidedQueryRequest.auditId,
              cursor: guidedQueryRequest.cursor,
              limit: guidedQueryRequest.limit,
            })
          : await invoke("query_rows", { spec });
      if (!guidedQueryIsCurrent() || !table) return null;
      table.setData(page.rows);
      nextCursor = page.nextCursor;
      hasMore = page.hasMore;
      updateRowCountLabel();
      return page;
    } catch (err) {
      if (!guidedQueryIsCurrent()) return null;
      console.error(`${isGuidedRequest ? "run_guided_query" : "query_rows"} failed`, err);
      alert(`Query failed: ${err}`);
      return null;
    } finally {
      const stillCurrent = guidedQueryIsCurrent();
      if (guidedActiveQuery === guidedQueryRequest) {
        guidedActiveQuery = null;
        updateGuidedInteractionControls();
      }
      if (stillCurrent) {
        prevPageBtn.disabled = cursorStack.length === 0;
        nextPageBtn.disabled = !hasMore;
        hideProgress();
      }
    }
  }

  function applyControlsAndReload() {
    queryMode = "normal";
    guidedIntentToken = null;
    guidedResetBtn.classList.add("hidden");
    spec.search = searchBox.value.trim() || null;
    spec.filters = currentFilterValues();
    spec.sort = sortColumn.value
      ? { column: sortColumn.value, direction: sortDirection.value }
      : null;
    resetPagination();
    refreshData();
    refreshCount();
  }

  let searchDebounceHandle = null;
  function debouncedApply() {
    if (searchDebounceHandle) clearTimeout(searchDebounceHandle);
    searchDebounceHandle = setTimeout(applyControlsAndReload, 300);
  }

  async function detectColumnRolesForLoadedFile({ throwOnError = false } = {}) {
    roleDetectionInFlight = true;
    roleDetectionError = null;
    renderRoleSuggestions();
    try {
      columnRoleSuggestions = await invoke("detect_column_roles");
      return columnRoleSuggestions;
    } catch (err) {
      console.error("detect_column_roles failed", err);
      roleDetectionError = err;
      roleReviewPanel.classList.remove("hidden");
      rolePanelStatus.textContent = `Column role detection failed: ${err}`;
      if (throwOnError) throw err;
      return [];
    } finally {
      roleDetectionInFlight = false;
      renderRoleSuggestions();
    }
  }

  async function setColumnRoleStatus(role, sqlName, status) {
    rolePanelStatus.textContent = `Updating ${formatRoleName(role)}...`;
    try {
      const updated = await invoke("set_column_role_status", { role, sqlName, status });
      upsertRoleSuggestion(updated);
      renderRoleSuggestions();
      if (role === "timestamp" && status === "confirmed") {
        await handleTimestampConfirmed();
      }
      return updated;
    } catch (err) {
      console.error("set_column_role_status failed", err);
      rolePanelStatus.textContent = `Role update failed: ${err}`;
      throw err;
    }
  }

  async function handleTimestampConfirmed() {
    rolePanelStatus.textContent = "Analyzing timestamp column...";
    try {
      const analysis = await invoke("analyze_timestamp_column");
      timestampAnalysis = analysis;
      if (analysis.needsTimezone) {
        showTimezonePrompt(analysis);
        rolePanelStatus.textContent = "Timestamp normalization needs examiner timezone input.";
        return null;
      }
      const summary = await normalizeTimestampColumn(null);
      rolePanelStatus.textContent = `Timestamp normalized to UTC: ${summary.rowsWritten.toLocaleString()} rows written.`;
      return summary;
    } catch (err) {
      console.error("timestamp analysis/normalization failed", err);
      rolePanelStatus.textContent = `Timestamp normalization failed: ${err}`;
      throw err;
    }
  }

  async function normalizeTimestampColumn(naiveTimezone) {
    timezoneNormalizeBtn.disabled = true;
    timezoneUtcBtn.disabled = true;
    try {
      const summary = await invoke("normalize_timestamp_column", { naiveTimezone });
      timestampNormalizationSummary = summary;
      timezonePanel.classList.add("hidden");
      rolePanelStatus.textContent = `Timestamp normalized to UTC: ${summary.rowsWritten.toLocaleString()} rows written.`;
      return summary;
    } catch (err) {
      console.error("normalize_timestamp_column failed", err);
      timezoneSummary.textContent = `Normalization failed: ${err}`;
      throw err;
    } finally {
      timezoneNormalizeBtn.disabled = false;
      timezoneUtcBtn.disabled = false;
    }
  }

  async function runIntelScan(evidenceColumns = confirmedEvidenceColumns()) {
    if (!evidenceColumns || evidenceColumns.length === 0) {
      throw new Error("confirm at least one evidence column role before scanning");
    }

    intelScanInFlight = true;
    updateEvidenceColumnsUi();
    showProgress("Scanning suspicious matches...", 0);
    try {
      const summary = await invoke("scan_intel_matches", { evidenceColumns });
      intelScanSummaryResult = summary;
      renderScanSummary(summary);
      return summary;
    } catch (err) {
      console.error("scan_intel_matches failed", err);
      throw err;
    } finally {
      hideProgress();
      intelScanInFlight = false;
      updateEvidenceColumnsUi();
    }
  }

  async function previewGuidedQuery(queryText = guidedSearchBox.value) {
    const trimmed = queryText.trim();
    if (!trimmed) return null;
    if (guidedActiveAction !== null || guidedActiveQuery !== null || sheetLoadInFlight) return null;

    const request = {
      id: ++guidedParseRequestSequence,
      contextRevision: guidedContextRevision,
      path: currentPath,
      sheet: currentSheet,
    };
    // Replacing this object immediately makes any older inference response stale, even when a
    // test/debug caller starts a second preview without waiting for the first one.
    guidedActiveParse = request;

    updateGuidedInteractionControls();
    guidedQueryPanel.classList.remove("hidden");
    guidedPreviewText.textContent = "Parsing guided query...";
    guidedClarification.classList.add("hidden");
    guidedRunBtn.classList.add("hidden");
    guidedRejectBtn.classList.add("hidden");
    showProgress("Parsing guided query...", 0.3);
    try {
      if (guidedAuditId !== null && guidedReviewStatus === "unreviewed") {
        const edited = await decideGuidedParse("edited", { allowDuringParse: true });
        if (!edited || !guidedParseIsCurrent(request)) return null;
      }
      const result = await invoke("parse_guided_query", { queryText: trimmed });
      if (!guidedParseIsCurrent(request)) return null;
      renderGuidedPreview(result);
      return result;
    } catch (err) {
      if (!guidedParseIsCurrent(request)) return null;
      console.error("parse_guided_query failed", err);
      guidedPreviewText.textContent = `Guided query preview failed: ${err}`;
      throw err;
    } finally {
      const stillCurrent = guidedParseIsCurrent(request);
      if (guidedActiveParse === request) {
        guidedActiveParse = null;
        updateGuidedInteractionControls();
      }
      if (stillCurrent) hideProgress();
    }
  }

  async function runGuidedQuery(intentToken = guidedIntentToken) {
    if (!intentToken || guidedAuditId === null) {
      throw new Error("no reviewed AI-assisted intent is ready to run");
    }
    if (!["unreviewed", "accepted"].includes(guidedReviewStatus)) {
      throw new Error(`AI-assisted interpretation was ${guidedReviewStatus || "not reviewable"} and cannot be run`);
    }
    guidedIntentToken = intentToken;
    const action = beginGuidedAction("run");
    if (!action) return null;

    try {
      guidedRunBtn.textContent = "Accepting...";
      // Make the audit transition explicit so an acceptance failure (stale scan/library,
      // rejected preview, or token mismatch) cannot be displayed as accepted. The query command
      // repeats this check idempotently for direct-IPC safety.
      await invoke("accept_guided_query", {
        intentToken: action.intentToken,
        auditId: action.auditId,
      });
      if (!guidedActionIsCurrent(action)) return null;

      setGuidedReviewStatus("accepted");
      guidedRejectBtn.classList.add("hidden");
      queryMode = "guided";
      totalCount = null;
      resetPagination();
      guidedResetBtn.classList.remove("hidden");
      guidedRunBtn.textContent = "Running...";

      const page = await refreshData();
      if (guidedActionIsCurrent(action)) {
        guidedRunBtn.textContent = page ? "Run" : "Retry";
        guidedRunBtn.classList.remove("hidden");
        refreshCount();
      }
      return page;
    } catch (err) {
      if (guidedActionIsCurrent(action) && guidedReviewStatus !== "accepted") {
        guidedRunBtn.textContent = "Run";
        guidedRunBtn.classList.remove("hidden");
        guidedRejectBtn.classList.remove("hidden");
      }
      throw err;
    } finally {
      finishGuidedAction(action);
    }
  }

  async function decideGuidedParse(decision, { allowDuringParse = false } = {}) {
    if (guidedAuditId === null || guidedReviewStatus !== "unreviewed") return false;
    const action = beginGuidedAction(decision, { allowDuringParse });
    if (!action) return false;
    try {
      await invoke("set_guided_parse_decision", {
        auditId: action.auditId,
        intentToken: action.intentToken,
        decision,
      });
      if (!guidedActionIsCurrent(action)) return false;
      setGuidedReviewStatus(decision);
      guidedRunBtn.classList.add("hidden");
      guidedRejectBtn.classList.add("hidden");
      return true;
    } finally {
      finishGuidedAction(action);
    }
  }

  async function generateReport(destPath) {
    showProgress("Generating report workbook...", 0);
    try {
      const summary = await invoke("export_report", { destPath });
      renderReportSummary(summary);
      return summary;
    } catch (err) {
      console.error("export_report failed", err);
      throw err;
    } finally {
      hideProgress();
    }
  }

  // -- import flow --------------------------------------------------------------

  async function pickAndOpenFile() {
    const path = await invoke("plugin:dialog|open", {
      options: {
        multiple: false,
        filters: [{ name: "Tabular files", extensions: ["xlsx", "xls", "xlsb", "ods", "csv"] }],
      },
    });
    if (!path) return;

    currentPath = path;
    sheetLoadInFlight = true;
    // Choosing a different source invalidates any inference that was started for the previous
    // file, even before the native sheet listing/import round trip completes.
    resetGuidedQueryUi();
    hideProgress();
    let sheets;
    try {
      sheets = await invoke("list_sheets", { path });
    } catch (err) {
      sheetLoadInFlight = false;
      updateGuidedInteractionControls();
      alert(`Could not read workbook: ${err}`);
      return;
    }

    if (sheets.length === 1) {
      await loadSheet(sheets[0]);
    } else {
      sheetSelect.innerHTML = "";
      sheets.forEach((name) => {
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = name;
        sheetSelect.appendChild(opt);
      });
      sheetPicker.classList.remove("hidden");
    }
  }

  async function loadSheet(sheet) {
    sheetLoadInFlight = true;
    resetGuidedQueryUi();
    sheetPicker.classList.add("hidden");
    currentSheet = sheet;
    showProgress(`Reading "${sheet}"…`, 0);

    try {
      const summary = await invoke("import_sheet", { path: currentPath, sheet });
      sheetLoadInFlight = false;
      hideProgress();
      onImportComplete(summary);
      return summary;
    } catch (err) {
      sheetLoadInFlight = false;
      updateGuidedInteractionControls();
      hideProgress();
      alert(`Import failed: ${err}`);
      throw err;
    }
  }

  function onImportComplete(summary) {
    columns = summary.columns;
    fileInfo.textContent = `${currentPath.split(/[\\/]/).pop()} — ${summary.rowCount.toLocaleString()} rows, ${columns.length} columns${summary.fromCache ? " (cached)" : ""}`;

    // reset controls
    resetIntelUiState();
    searchBox.value = "";
    filterList.innerHTML = "";
    sortColumn.innerHTML = '<option value="">(row order)</option>';
    columns.forEach((c) => {
      const opt = document.createElement("option");
      opt.value = c.sqlName;
      opt.textContent = c.originalName;
      sortColumn.appendChild(opt);
    });

    spec = { search: null, filters: [], sort: null, cursor: null, limit: PAGE_SIZE };
    resetPagination();

    const tabulatorColumns = [
      { title: "#", field: "row_num", width: 70, headerSort: false, frozen: true },
      ...columns.map((c) => ({
        title: c.originalName,
        field: c.sqlName,
        headerSort: false,
        resizable: true,
      })),
    ];

    if (table) {
      table.destroy();
    }
    table = new Tabulator("#grid", {
      data: [],
      columns: tabulatorColumns,
      layout: "fitDataFill",
      height: "100%",
      placeholder: "No matching rows",
    });

    setControlsEnabled(true);
    detectColumnRolesForLoadedFile();
    table.on("tableBuilt", () => {
      refreshData();
      refreshCount();
    });
  }

  function removeFile() {
    sheetLoadInFlight = false;
    if (table) {
      table.destroy();
      table = null;
    }
    columns = [];
    currentPath = null;
    currentSheet = null;
    fileInfo.textContent = "No file loaded";
    sortColumn.innerHTML = '<option value="">(row order)</option>';
    filterList.innerHTML = "";
    searchBox.value = "";
    spec = { search: null, filters: [], sort: null, cursor: null, limit: PAGE_SIZE };
    resetPagination();
    resetIntelUiState();
    setControlsEnabled(false);
    hideProgress();
    rowCountLabel.textContent = "—";
    pageLabel.textContent = "";
    return invoke("clear_loaded_file").catch((err) => {
      // The local generation/UI have already been invalidated synchronously. Keep the app in
      // that safe empty state and surface a backend-clear failure for diagnostics.
      console.error("clear_loaded_file failed", err);
    });
  }

  // -- export flow --------------------------------------------------------------

  async function doExport(format) {
    const ext = format === "csv" ? "csv" : "xlsx";
    const destPath = await invoke("plugin:dialog|save", {
      options: {
        filters: [{ name: format.toUpperCase(), extensions: [ext] }],
        defaultPath: `log-parser-export.${ext}`,
      },
    });
    if (!destPath) return;

    const exportSpec = buildSpecFromControls(true);
    showProgress(`Exporting to ${ext.toUpperCase()}…`, 0);
    try {
      const result = await invoke("export_data", { spec: exportSpec, format, destPath });
      hideProgress();
      alert(`Exported ${result.rowCount.toLocaleString()} rows to ${result.destPath}`);
    } catch (err) {
      hideProgress();
      alert(`Export failed: ${err}`);
    }
  }

  async function doReportExport() {
    const destPath = await invoke("plugin:dialog|save", {
      options: {
        filters: [{ name: "Excel Workbook", extensions: ["xlsx"] }],
        defaultPath: "log-parser-report.xlsx",
      },
    });
    if (!destPath) return;

    try {
      await generateReport(destPath);
    } catch (err) {
      alert(`Report export failed: ${err}`);
    }
  }

  // -- event wiring --------------------------------------------------------------

  openFileBtn.addEventListener("click", () => {
    pickAndOpenFile().catch((err) => alert(`Error: ${err}`));
  });

  removeFileBtn.addEventListener("click", () => {
    removeFile();
  });

  reviewRolesBtn.addEventListener("click", () => {
    roleReviewPanel.classList.remove("hidden");
  });

  sheetLoadBtn.addEventListener("click", () => {
    loadSheet(sheetSelect.value);
  });

  searchBox.addEventListener("input", debouncedApply);
  guidedSearchForm.addEventListener("submit", (event) => {
    event.preventDefault();
    previewGuidedQuery().catch((err) => alert(`Guided query preview failed: ${err}`));
  });
  guidedRunBtn.addEventListener("click", () => {
    runGuidedQuery().catch((err) => alert(`Guided query failed: ${err}`));
  });
  guidedRejectBtn.addEventListener("click", () => {
    decideGuidedParse("rejected").catch((err) => alert(`Could not record decision: ${err}`));
  });
  guidedResetBtn.addEventListener("click", () => {
    queryMode = "normal";
    guidedIntentToken = null;
    guidedResetBtn.classList.add("hidden");
    resetPagination();
    refreshData();
    refreshCount();
  });
  guidedPanelClose.addEventListener("click", () => {
    cancelActiveGuidedParse();
    decideGuidedParse("rejected").catch((err) => console.error("set_guided_parse_decision failed", err));
    guidedQueryPanel.classList.add("hidden");
  });

  addFilterBtn.addEventListener("click", addFilterRow);
  applyBtn.addEventListener("click", applyControlsAndReload);
  clearBtn.addEventListener("click", () => {
    searchBox.value = "";
    filterList.innerHTML = "";
    sortColumn.value = "";
    applyControlsAndReload();
  });

  suspiciousScanBtn.addEventListener("click", () => {
    runIntelScan().catch((err) => alert(`Suspicious scan failed: ${err}`));
  });

  rolePanelClose.addEventListener("click", () => {
    roleReviewPanel.classList.add("hidden");
  });

  timezoneUtcBtn.addEventListener("click", () => {
    normalizeTimestampColumn("UTC").catch((err) => alert(`Timestamp normalization failed: ${err}`));
  });
  timezoneNormalizeBtn.addEventListener("click", () => {
    const answer = timezoneInput.value.trim();
    if (!answer) {
      alert("Enter a UTC offset or IANA timezone, or choose Already UTC.");
      return;
    }
    normalizeTimestampColumn(answer).catch((err) => alert(`Timestamp normalization failed: ${err}`));
  });
  timezonePanelClose.addEventListener("click", () => {
    timezonePanel.classList.add("hidden");
  });

  reportSummaryClose.addEventListener("click", () => {
    reportSummaryPanel.classList.add("hidden");
  });

  reportExportBtn.addEventListener("click", doReportExport);
  exportCsvBtn.addEventListener("click", () => doExport("csv"));
  exportXlsxBtn.addEventListener("click", () => doExport("xlsx"));

  prevPageBtn.addEventListener("click", () => {
    if (cursorStack.length === 0) return;
    spec.cursor = cursorStack.pop();
    pageIndex -= 1;
    refreshData();
  });

  nextPageBtn.addEventListener("click", () => {
    if (!hasMore) return;
    cursorStack.push(spec.cursor);
    spec.cursor = nextCursor;
    pageIndex += 1;
    refreshData();
  });

  listen("import-progress", (event) => {
    const { rowsDone, rowsTotal, phase } = event.payload;
    const fraction = rowsTotal > 0 ? rowsDone / rowsTotal : 0;
    const label =
      phase === "indexing"
        ? "Building search index…"
        : `Reading rows… ${rowsDone.toLocaleString()} / ${rowsTotal.toLocaleString()}`;
    showProgress(label, fraction);
  });

  listen("export-progress", (event) => {
    const { rowsDone } = event.payload;
    showProgress(`Exporting… ${rowsDone.toLocaleString()} rows written`, 0.5);
  });

  listen("intel-scan-progress", (event) => {
    const { rowsDone, rowsTotal, phase } = event.payload;
    const fraction = rowsTotal > 0 ? rowsDone / rowsTotal : 0;
    const label =
      phase === "complete"
        ? "Suspicious scan complete"
        : `Scanning suspicious matches... ${rowsDone.toLocaleString()} / ${rowsTotal.toLocaleString()}`;
    showProgress(label, fraction);
  });

  listen("report-export-progress", (event) => {
    const { rowsDone, sheet } = event.payload;
    const sheetLabel = sheet ? ` (${sheet})` : "";
    showProgress(`Writing report${sheetLabel}... ${rowsDone.toLocaleString()} rows`, 0.5);
  });

  // Debug hook: lets automated/CDP-driven testing open a file by path directly,
  // bypassing the native OS file-picker dialog (which can't be scripted).
  // Harmless in normal use — withGlobalTauri already exposes the raw invoke()
  // surface to page scripts, so this adds no new capability, just convenience.
  window.__logParserDebug = window.__logParserDebug || {};
  Object.assign(window.__logParserDebug, {
    loadSheetForTest(path, sheet) {
      if (!path) {
        throw new Error("loadSheetForTest(path, sheet): path is required");
      }
      if (!sheet) {
        throw new Error(
          "loadSheetForTest(path, sheet): sheet is required - call listSheetsForTest(path) first if you don't know the sheet name"
        );
      }
      currentPath = path;
      return loadSheet(sheet);
    },
    async listSheetsForTest(path) {
      if (!path) {
        throw new Error("listSheetsForTest(path): path is required");
      }
      return invoke("list_sheets", { path });
    },
    getState() {
      return { spec, hasMore, pageIndex, totalCount, columns };
    },
    getIntelState() {
      return {
        columnRoleSuggestions,
        timestampAnalysis,
        timestampNormalizationSummary,
        evidenceColumns: confirmedEvidenceColumns(),
        intelScanSummary: intelScanSummaryResult,
        reportSummary: reportSummaryResult,
      };
    },
    getGuidedState() {
      return {
        queryMode,
        guidedParseResult,
        guidedIntentToken,
        guidedAuditId,
        guidedReviewStatus,
        parseInFlight: guidedActiveParse !== null,
        actionInFlight: guidedActiveAction ? guidedActiveAction.type : null,
        queryInFlight: guidedActiveQuery !== null,
        hasMore,
        pageIndex,
        totalCount,
        rows: table ? table.getData() : [],
      };
    },
    detectRolesForTest() {
      return detectColumnRolesForLoadedFile({ throwOnError: true });
    },
    setColumnRoleStatusForTest(role, sqlName, status) {
      return setColumnRoleStatus(role, sqlName, status);
    },
    analyzeTimestampForTest() {
      return handleTimestampConfirmed();
    },
    normalizeTimestampForTest(naiveTimezone = null) {
      return normalizeTimestampColumn(naiveTimezone);
    },
    scanIntelForTest(evidenceColumns = confirmedEvidenceColumns()) {
      return runIntelScan(evidenceColumns);
    },
    previewGuidedQueryForTest(queryText) {
      guidedSearchBox.value = queryText;
      return previewGuidedQuery(queryText);
    },
    runGuidedQueryForTest(intentToken = guidedIntentToken) {
      return runGuidedQuery(intentToken);
    },
    decideGuidedParseForTest(decision) {
      return decideGuidedParse(decision);
    },
    generateReportForTest(destPath) {
      return generateReport(destPath);
    },
    removeFileForTest() {
      removeFile();
    },
  });
})();
