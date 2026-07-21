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
  const aiSearchAvailability = document.getElementById("ai-search-availability");
  const semanticIndexStatus = document.getElementById("semantic-index-status");
  const reportExportBtn = document.getElementById("report-export-btn");
  const exportCsvBtn = document.getElementById("export-csv-btn");
  const exportXlsxBtn = document.getElementById("export-xlsx-btn");

  const progressWrap = document.getElementById("progress-bar-wrap");
  const progressFill = document.getElementById("progress-fill");
  const progressLabel = document.getElementById("progress-label");

  const analystPanel = document.getElementById("analyst-panel");
  const analystHeadline = document.getElementById("analyst-headline");
  const analystStatus = document.getElementById("analyst-status");
  const analystSections = document.getElementById("analyst-sections");
  const analystSteps = document.getElementById("analyst-steps");
  const analystReportBtn = document.getElementById("analyst-report-btn");
  const analystPanelClose = document.getElementById("analyst-panel-close");

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
  const dataMappingSummary = document.getElementById("data-mapping-summary");

  const ignoreRulesPanel = document.getElementById("ignore-rules-panel");
  const ignoreRulesSummary = document.getElementById("ignore-rules-summary");
  const ignoreRuleList = document.getElementById("ignore-rule-list");
  const ignoreRulePanelStatus = document.getElementById("ignore-rule-panel-status");
  const ignoreRulesPanelClose = document.getElementById("ignore-rules-panel-close");
  const manageIgnoreRulesBtn = document.getElementById("manage-ignore-rules-btn");
  const addIgnoreRuleForm = document.getElementById("add-ignore-rule-form");
  const ignoreRuleNameInput = document.getElementById("ignore-rule-name");
  const ignoreRuleTargetType = document.getElementById("ignore-rule-target-type");
  const ignoreRuleRoleSelect = document.getElementById("ignore-rule-role");
  const ignoreRuleHeaderInput = document.getElementById("ignore-rule-header");
  const ignoreRuleOpSelect = document.getElementById("ignore-rule-op");
  const ignoreRuleValuesInput = document.getElementById("ignore-rule-values");

  const timezonePanel = document.getElementById("timezone-panel");
  const timezoneSummary = document.getElementById("timezone-summary");
  const timezoneSamples = document.getElementById("timezone-samples");
  const timezoneInput = document.getElementById("timezone-input");
  const dateConventionWrap = document.getElementById("date-convention-wrap");
  const dateConventionSelect = document.getElementById("date-convention-select");
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
  // Above this many columns, the grid switches to a cheaper layout/render mode (see the
  // Tabulator setup below). Normal files here run ~50-300 columns; this exists for outliers
  // (a real 1,824-column export made "fitDataFill" + non-virtualized rendering measure every
  // cell of every column synchronously and stall the whole app).
  const WIDE_GRID_COLUMN_THRESHOLD = 400;

  let columns = []; // ColumnMeta[] from ImportSummary
  let table = null;
  let currentPath = null;
  let currentSheet = null;

  // Per-file, like columnRoleSuggestions: IgnoreRuleView[] from list_ignore_rules, reset on
  // file removal and refetched after each import.
  let ignoreRules = [];
  let ignoreRulesLoaded = false;
  let ignoreRulesInFlight = false;

  let spec = { search: null, filters: [], sort: null, expression: null, cursor: null, limit: PAGE_SIZE };
  let cursorStack = []; // for Prev navigation
  let nextCursor = null;
  let hasMore = false;
  let pageIndex = 1;
  let totalCount = null;

  let queryMode = "normal";
  // The plan that actually produced the rows currently shown. A new AI interpretation is kept
  // separate until its first page succeeds, so failed/ambiguous searches cannot relabel or page
  // the previous table through an unexecuted plan.
  let activeEvidenceQuery = null;
  let guidedParseResult = null;
  let guidedIntentToken = null;
  let guidedAuditId = null;
  let guidedReviewStatus = null;
  let guidedQuerySpec = null;
  let guidedMatchExplanation = [];
  let guidedPreviewQueryText = null;
  let guidedContextRevision = 0;
  let guidedParseRequestSequence = 0;
  let guidedActiveParse = null;
  let guidedActionSequence = 0;
  let guidedActiveAction = null;
  let guidedActiveQuery = null;
  let dataRequestSequence = 0;
  let activeDataRequest = null;
  let countRequestSequence = 0;
  let activeCountRequest = null;
  let sheetLoadInFlight = false;
  let sourceLoadSequence = 0;
  let activeSourceLoad = null;
  let activeSheetImport = null;
  let controlsEnabled = false;

  let columnRoleSuggestions = [];
  let cachedColumnOptionsRef = null;
  let cachedColumnOptionsTemplate = null;
  let timestampAnalysis = null;
  let timestampNormalizationSummary = null;
  let intelScanSummaryResult = null;
  let reportSummaryResult = null;
  let reportExportSequence = 0;
  let activeReportExport = null;
  let analystRequestSequence = 0;
  let activeAnalystRequest = null;
  let roleDetectionInFlight = false;
  let roleDetectionError = null;
  let roleDetectionRequestSequence = 0;
  let activeRoleDetectionRequest = null;
  let mappingRequestSequence = 0;
  const activeMappingRequests = new Map();
  let automaticTimestampInFlight = false;
  let automaticTimestampSqlName = null;
  let timestampOperationSequence = 0;
  let activeTimestampOperation = null;
  let semanticIndexState = {
    status: "idle",
    phase: null,
    buildId: null,
    rowsIndexed: 0,
    documentsEmbedded: 0,
    mappingsWritten: 0,
    documentsSkipped: 0,
    mappingsSkipped: 0,
    cellsTruncated: 0,
    columnsOmitted: 0,
    chunksOmitted: 0,
    resumedFromRow: 0,
    summary: null,
    error: null,
  };
  let semanticIndexRequestSequence = 0;
  let activeSemanticIndexRequest = null;
  let pendingSemanticSearch = null;
  let intelScanInFlight = false;

  const EVIDENCE_ROLES = new Set([
    "command_line",
    "process_name",
    "file_name",
    "host",
    "text_evidence",
  ]);

  const MAPPING_ROLES = [
    "timestamp",
    "user",
    "command_line",
    "process_name",
    "file_name",
    "host",
    "ip",
    "text_evidence",
  ];

  // Ignore-rule conditions can key off any data-mapping role except timestamp — matches the
  // backend's RULE_CONDITION_ROLES (library.rs).
  const IGNORE_RULE_ROLES = MAPPING_ROLES.filter((role) => role !== "timestamp");
  const IGNORE_RULE_OP_LABELS = {
    contains_any: "contains",
    equals_any: "equals",
    ends_with_any: "ends with",
  };

  const FILTER_OPERATORS = new Set([
    "equals",
    "notEquals",
    "contains",
    "notContains",
    "startsWith",
    "endsWith",
    "isEmpty",
    "isNotEmpty",
    "greaterThan",
    "lessThan",
  ]);

  // -- helpers --------------------------------------------------------------

  function setControlsEnabled(enabled) {
    controlsEnabled = enabled;
    removeFileBtn.disabled = !enabled;
    searchBox.disabled = !enabled;
    guidedSearchBox.disabled = !enabled;
    guidedSearchSubmit.disabled = !enabled;
    reportExportBtn.disabled = !enabled || sheetLoadInFlight || activeReportExport !== null;
    exportCsvBtn.disabled = !enabled;
    exportXlsxBtn.disabled = !enabled;
    addFilterBtn.disabled = !enabled;
    applyBtn.disabled = !enabled;
    clearBtn.disabled = !enabled;
    reviewRolesBtn.disabled = !enabled;
    manageIgnoreRulesBtn.disabled = !enabled;
    aiSearchAvailability.textContent = enabled
      ? "Ready to search every imported row. No enrichment scan is required."
      : "Import a file to search its evidence.";
    aiSearchAvailability.classList.toggle("ready", enabled);
    if (enabled) {
      updateEvidenceColumnsUi();
    } else {
      suspiciousScanBtn.disabled = true;
    }
    updateGuidedInteractionControls();
  }

  function setSourceLoadInFlight(inFlight) {
    sheetLoadInFlight = inFlight;
    openFileBtn.disabled = inFlight;
    sheetLoadBtn.disabled = inFlight;
    searchBox.disabled = inFlight || !controlsEnabled;
    reportExportBtn.disabled = inFlight || !controlsEnabled || activeReportExport !== null;
    exportCsvBtn.disabled = inFlight || !controlsEnabled;
    exportXlsxBtn.disabled = inFlight || !controlsEnabled;
    addFilterBtn.disabled = inFlight || !controlsEnabled;
    applyBtn.disabled = inFlight || !controlsEnabled;
    clearBtn.disabled = inFlight || !controlsEnabled;
    reviewRolesBtn.disabled = inFlight || !controlsEnabled;
    suspiciousScanBtn.disabled = inFlight || !controlsEnabled;
    manageIgnoreRulesBtn.disabled = inFlight || !controlsEnabled;
    if (inFlight) {
      prevPageBtn.disabled = true;
      nextPageBtn.disabled = true;
    } else {
      prevPageBtn.disabled = cursorStack.length === 0;
      nextPageBtn.disabled = !hasMore;
      updateEvidenceColumnsUi();
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

  function guidedWorkInFlight() {
    return (
      guidedActiveParse !== null ||
      guidedActiveAction !== null ||
      guidedActiveQuery !== null
    );
  }

  function tableTransitionInFlight() {
    return guidedWorkInFlight() || activeDataRequest !== null;
  }

  function updateGuidedInteractionControls() {
    const parsing = guidedActiveParse !== null;
    const actionInFlight = guidedActiveAction !== null;
    const queryInFlight = guidedActiveQuery !== null;
    const tableTransition = parsing || actionInFlight || queryInFlight || activeDataRequest !== null;
    const tableControlsBlocked = !controlsEnabled || sheetLoadInFlight || tableTransition;
    guidedSearchBox.disabled =
      !controlsEnabled ||
      columns.length === 0 ||
      sheetLoadInFlight ||
      parsing ||
      actionInFlight ||
      queryInFlight ||
      activeDataRequest !== null ||
      activeReportExport !== null;
    guidedSearchSubmit.disabled =
      !controlsEnabled ||
      columns.length === 0 ||
      sheetLoadInFlight ||
      parsing ||
      actionInFlight ||
      queryInFlight ||
      activeDataRequest !== null ||
      activeReportExport !== null;
    guidedRunBtn.disabled = tableTransition || activeReportExport !== null;
    guidedRejectBtn.disabled = tableTransition || activeReportExport !== null;
    guidedResetBtn.disabled = tableTransition || activeReportExport !== null;
    searchBox.disabled = tableControlsBlocked;
    exportCsvBtn.disabled = tableControlsBlocked;
    exportXlsxBtn.disabled = tableControlsBlocked;
    reportExportBtn.disabled = tableControlsBlocked || activeReportExport !== null;
    addFilterBtn.disabled = tableControlsBlocked;
    applyBtn.disabled = tableControlsBlocked;
    clearBtn.disabled = tableControlsBlocked;
    prevPageBtn.disabled = tableControlsBlocked || cursorStack.length === 0;
    nextPageBtn.disabled = tableControlsBlocked || !hasMore;
    // Keep Close available while parsing so it can cancel a slow preview, but do not let it
    // race the decision implicit in Run or an explicit Reject/Edit request.
    guidedPanelClose.disabled = actionInFlight || queryInFlight;
  }

  function invalidateGuidedContext() {
    guidedContextRevision += 1;
    guidedActiveParse = null;
  }

  function resetGuidedQueryUi({ invalidateDataset = true } = {}) {
    cancelSearchDebounce();
    if (invalidateDataset) {
      invalidateGuidedContext();
      activeAnalystRequest = null;
      hideAnalystPanel();
    }
    queryMode = "normal";
    activeEvidenceQuery = null;
    guidedParseResult = null;
    guidedIntentToken = null;
    guidedAuditId = null;
    guidedReviewStatus = null;
    guidedQuerySpec = null;
    guidedMatchExplanation = [];
    guidedPreviewQueryText = null;
    pendingSemanticSearch = null;

    guidedSearchBox.value = "";
    guidedQueryPanel.classList.add("hidden");
    guidedPreviewText.textContent = "";
    guidedAiStatus.textContent = "";
    guidedAiStatus.classList.add("hidden");
    guidedClarification.textContent = "";
    guidedClarification.classList.add("hidden");
    guidedRunBtn.textContent = "Search evidence";
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
    activeRoleDetectionRequest = null;
    activeMappingRequests.clear();
    automaticTimestampInFlight = false;
    automaticTimestampSqlName = null;
    activeTimestampOperation = null;
    activeSemanticIndexRequest = null;
    semanticIndexState = {
      status: "idle",
      phase: null,
      buildId: null,
      rowsIndexed: 0,
      documentsEmbedded: 0,
      mappingsWritten: 0,
      documentsSkipped: 0,
      mappingsSkipped: 0,
      cellsTruncated: 0,
      columnsOmitted: 0,
      chunksOmitted: 0,
      resumedFromRow: 0,
      summary: null,
      error: null,
    };
    semanticIndexStatus.className = "semantic-index-status";
    semanticIndexStatus.textContent = "Semantic matching starts automatically after import.";
    intelScanInFlight = false;
    activeDataRequest = null;
    activeCountRequest = null;

    roleList.innerHTML = "";
    rolePanelStatus.textContent = "";
    roleReviewPanel.classList.add("hidden");

    // Ignore rules are per-file (stored in this file's own database), so stale rows from the
    // previous file must not linger in the panel while a new one loads.
    ignoreRules = [];
    ignoreRulesLoaded = false;
    ignoreRuleList.innerHTML = "";
    ignoreRulePanelStatus.textContent = "";
    ignoreRulesSummary.textContent = "Loading…";
    ignoreRulesPanel.classList.add("hidden");
    ignoreRulesPanel.open = false;
    roleReviewPanel.open = false;
    dataMappingSummary.textContent = "Waiting for a file";

    timezoneInput.value = "";
    dateConventionSelect.value = "";
    dateConventionWrap.classList.add("hidden");
    timezoneSummary.textContent = "";
    timezoneSamples.textContent = "";
    timezoneSamples.classList.add("hidden");
    timezonePanel.classList.add("hidden");
    timezoneNormalizeBtn.disabled = false;
    timezoneNormalizeBtn.textContent = "Use timezone";
    timezoneUtcBtn.disabled = false;

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
      currentSheet === request.sheet &&
      guidedSearchBox.value.trim() === request.queryText
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
      activeDataRequest !== null ||
      activeReportExport !== null ||
      sheetLoadInFlight ||
      (!allowDuringParse && guidedActiveParse !== null)
    ) {
      return null;
    }
    const action = {
      id: ++guidedActionSequence,
      type,
      contextRevision: guidedContextRevision,
      path: currentPath,
      sheet: currentSheet,
      queryText: guidedPreviewQueryText,
      auditId: guidedAuditId,
      intentToken: guidedIntentToken,
      querySpec: guidedQuerySpec,
    };
    guidedActiveAction = action;
    updateGuidedInteractionControls();
    return action;
  }

  function guidedActionIsCurrent(action) {
    return (
      guidedActiveAction === action &&
      loadedContextIsCurrent(action) &&
      guidedPreviewQueryText === action.queryText &&
      guidedSearchBox.value.trim() === action.queryText &&
      guidedAuditId === action.auditId &&
      guidedIntentToken === action.intentToken &&
      guidedQuerySpec === action.querySpec
    );
  }

  function guidedDecisionIsCurrent(action) {
    return (
      guidedActiveAction === action &&
      loadedContextIsCurrent(action) &&
      guidedAuditId === action.auditId &&
      guidedIntentToken === action.intentToken &&
      guidedQuerySpec === action.querySpec
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
    guidedAiStatus.textContent = `Offline AI interpretation \u2022 ${status} \u2022 processed locally`;
  }

  function guidedPlanIsReadyToRun() {
    if (!guidedParseResult || guidedParseResult.needsClarification) {
      return false;
    }
    if (!guidedParseResult.aiAssisted) {
      return guidedQuerySpec !== null && guidedAuditId === null;
    }
    // A validated MITRE-mapping plan carries no querySpec: the audited intent token is the
    // backend-validated authority and executes through run_guided_query.
    return (
      guidedIntentToken !== null &&
      guidedAuditId !== null &&
      ["unreviewed", "accepted"].includes(guidedReviewStatus)
    );
  }

  function formatRoleName(role) {
    const labels = {
      timestamp: "Timestamp",
      user: "User / account",
      command_line: "Command line",
      process_name: "Process",
      file_name: "File",
      host: "Host / device",
      ip: "IP address",
      text_evidence: "Evidence text",
    };
    return labels[role] || role.replace(/_/g, " ");
  }

  function columnDisplayName(sqlName) {
    const column = columns.find((c) => c.sqlName === sqlName);
    return column ? column.originalName : sqlName;
  }

  function describeIgnoreRuleConditions(rule) {
    return rule.conditions
      .map((condition) => {
        const target = condition.role
          ? formatRoleName(condition.role)
          : (condition.headerAnyOf || []).join(" / ") || "(any column)";
        const opLabel = IGNORE_RULE_OP_LABELS[condition.op] || condition.op;
        return `${target} ${opLabel}: ${condition.values.join(", ")}`;
      })
      .join(" AND ");
  }

  function renderIgnoreRules() {
    ignoreRuleList.innerHTML = "";
    const activeCount = ignoreRules.filter((rule) => rule.enabled).length;
    ignoreRulesSummary.textContent = ignoreRules.length
      ? `${activeCount} of ${ignoreRules.length} active`
      : "No rules";

    ignoreRules.forEach((rule) => {
      const row = document.createElement("div");
      row.className = "role-row ignore-rule-row";

      const titleWrap = document.createElement("div");
      const title = document.createElement("div");
      title.className = "role-title";
      title.textContent = rule.name;
      titleWrap.appendChild(title);
      const sourceBadge = document.createElement("span");
      sourceBadge.className = `role-badge ${rule.source === "custom" ? "custom" : "builtin"}`;
      sourceBadge.textContent = rule.source === "custom" ? "custom" : "built-in";
      titleWrap.appendChild(sourceBadge);
      if (!rule.enabled) {
        const disabledBadge = document.createElement("span");
        disabledBadge.className = "role-badge disabled-rule";
        disabledBadge.textContent = "disabled";
        titleWrap.appendChild(disabledBadge);
      }

      const condition = document.createElement("div");
      condition.className = "ignore-rule-condition";
      condition.textContent = describeIgnoreRuleConditions(rule);

      const actions = document.createElement("div");
      actions.className = "role-actions";
      const toggleBtn = document.createElement("button");
      toggleBtn.className = "btn btn-small";
      toggleBtn.textContent = rule.enabled ? "Disable" : "Enable";
      toggleBtn.addEventListener("click", () => {
        toggleBtn.disabled = true;
        setIgnoreRuleEnabled(rule.id, !rule.enabled).finally(() => {
          toggleBtn.disabled = false;
        });
      });
      actions.appendChild(toggleBtn);
      if (rule.source === "custom") {
        const deleteBtn = document.createElement("button");
        deleteBtn.className = "btn btn-small";
        deleteBtn.textContent = "Delete";
        deleteBtn.addEventListener("click", () => {
          if (!confirm(`Delete ignore rule "${rule.name}"?`)) return;
          deleteBtn.disabled = true;
          toggleBtn.disabled = true;
          deleteIgnoreRule(rule.id).finally(() => {
            deleteBtn.disabled = false;
            toggleBtn.disabled = false;
          });
        });
        actions.appendChild(deleteBtn);
      }

      row.append(titleWrap, condition, actions);
      ignoreRuleList.appendChild(row);
    });
  }

  async function loadIgnoreRules() {
    if (ignoreRulesInFlight) return;
    ignoreRulesInFlight = true;
    try {
      const listing = await invoke("list_ignore_rules");
      ignoreRules = listing.rules;
      ignoreRulesLoaded = true;
      ignoreRulePanelStatus.textContent = listing.customRulesError
        ? `Your custom ignore-rules file could not be read, so only built-in rules are active: ${listing.customRulesError}`
        : "";
      renderIgnoreRules();
    } catch (err) {
      console.error("list_ignore_rules failed", err);
      ignoreRulePanelStatus.textContent = `Could not load ignore rules: ${err}`;
    } finally {
      ignoreRulesInFlight = false;
    }
  }

  async function setIgnoreRuleEnabled(ruleId, enabled) {
    try {
      const listing = await invoke("set_ignore_rule_enabled", { ruleId, enabled });
      ignoreRules = listing.rules;
      renderIgnoreRules();
    } catch (err) {
      console.error("set_ignore_rule_enabled failed", err);
      ignoreRulePanelStatus.textContent = `Could not update ignore rule: ${err}`;
    }
  }

  async function deleteIgnoreRule(ruleId) {
    try {
      const listing = await invoke("delete_custom_ignore_rule", { ruleId });
      ignoreRules = listing.rules;
      renderIgnoreRules();
    } catch (err) {
      console.error("delete_custom_ignore_rule failed", err);
      ignoreRulePanelStatus.textContent = `Could not delete ignore rule: ${err}`;
    }
  }

  async function addIgnoreRule(input) {
    try {
      const listing = await invoke("add_custom_ignore_rule", { input });
      ignoreRules = listing.rules;
      ignoreRulePanelStatus.textContent = "";
      renderIgnoreRules();
      return true;
    } catch (err) {
      console.error("add_custom_ignore_rule failed", err);
      ignoreRulePanelStatus.textContent = `Could not add ignore rule: ${err}`;
      return false;
    }
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

  function inferredEvidenceColumns() {
    const out = [];
    columnRoleSuggestions.forEach((row) => {
      if (
        row.status !== "rejected" &&
        EVIDENCE_ROLES.has(row.role) &&
        row.sqlName &&
        !out.includes(row.sqlName)
      ) {
        out.push(row.sqlName);
      }
    });
    return out;
  }

  function updateEvidenceColumnsUi() {
    const evidenceColumns = inferredEvidenceColumns();
    suspiciousScanBtn.disabled =
      columns.length === 0 ||
      evidenceColumns.length === 0 ||
      roleDetectionInFlight ||
      intelScanInFlight;
    if (columns.length === 0) {
      evidenceColumnsLabel.textContent = "Automatic evidence mapping starts after import.";
    } else if (roleDetectionInFlight) {
      evidenceColumnsLabel.textContent = "Detecting optional evidence mappings...";
    } else if (evidenceColumns.length === 0) {
      evidenceColumnsLabel.textContent = "No evidence mapping was inferred. AI search is still available.";
    } else {
      evidenceColumnsLabel.textContent = `Enrichment will inspect: ${evidenceColumns
        .map(columnDisplayName)
        .join(", ")}`;
    }
  }

  // Building this <select>'s <option> list is the same 1-per-column DOM work for every one of
  // the 8 roles, every time the panel renders (including once per single confirm/reject). On a
  // very wide file (1,800+ columns) that's tens of thousands of createElement/appendChild calls
  // per render. `columns` is reassigned wholesale on every import (never mutated in place), so
  // reference equality is a safe, free cache-invalidation signal: build the template once per
  // loaded file and hand out cheap native clones instead of rebuilding from scratch every time.
  function columnOptionsTemplate() {
    if (cachedColumnOptionsRef !== columns) {
      const template = document.createElement("select");
      const emptyOption = document.createElement("option");
      emptyOption.value = "";
      emptyOption.textContent = "(not mapped)";
      template.appendChild(emptyOption);
      columns.forEach((candidate) => {
        const option = document.createElement("option");
        option.value = candidate.sqlName;
        option.textContent = candidate.originalName;
        template.appendChild(option);
      });
      cachedColumnOptionsTemplate = template;
      cachedColumnOptionsRef = columns;
    }
    return cachedColumnOptionsTemplate.cloneNode(true);
  }

  function renderRoleSuggestions() {
    roleList.innerHTML = "";
    roleReviewPanel.classList.toggle("hidden", columns.length === 0);

    if (roleDetectionInFlight) {
      dataMappingSummary.textContent = "Detecting likely columns...";
      rolePanelStatus.textContent = "Automatic mapping is running in the background. AI evidence search is ready now.";
      updateEvidenceColumnsUi();
      return;
    }

    if (roleDetectionError) {
      dataMappingSummary.textContent = "Automatic mapping unavailable";
      rolePanelStatus.textContent = `Automatic mapping failed: ${roleDetectionError}. AI evidence search is unaffected.`;
    }

    MAPPING_ROLES.forEach((role) => {
      const suggestion = columnRoleSuggestions.find((row) => row.role === role) || {
        role,
        sqlName: "",
        originalName: "",
        confidence: 0,
        status: "unmapped",
        reasons: [],
      };
      const row = document.createElement("div");
      row.className = "role-row";

      const roleTitle = document.createElement("div");
      roleTitle.className = "role-title";
      roleTitle.textContent = formatRoleName(role);

      const columnSelect = columnOptionsTemplate();
      columnSelect.className = "mapping-column-select";
      columnSelect.setAttribute("aria-label", `Column mapped to ${formatRoleName(role)}`);
      columnSelect.value = suggestion.sqlName || "";

      const meta = document.createElement("div");
      const badge = document.createElement("span");
      badge.className = `role-badge ${suggestion.status}`;
      badge.textContent =
        suggestion.status === "suggested"
          ? "automatic"
          : suggestion.status === "rejected"
            ? "ignored"
            : suggestion.status;
      meta.appendChild(badge);
      const confidence = document.createElement("div");
      confidence.className = "role-confidence";
      confidence.textContent = suggestion.sqlName
        ? `${Math.round((suggestion.confidence || 0) * 100)}% confidence`
        : "No automatic match";
      meta.appendChild(confidence);

      const actions = document.createElement("div");
      actions.className = "role-actions";
      const confirmBtn = document.createElement("button");
      confirmBtn.className = "btn btn-small";
      const updateConfirmButton = () => {
        const isSameConfirmed = suggestion.status === "confirmed" && columnSelect.value === suggestion.sqlName;
        confirmBtn.textContent =
          suggestion.sqlName && columnSelect.value && columnSelect.value !== suggestion.sqlName
            ? "Use override"
            : suggestion.sqlName
              ? "Confirm"
              : "Use mapping";
        confirmBtn.disabled = !columnSelect.value || isSameConfirmed;
      };
      updateConfirmButton();
      columnSelect.addEventListener("change", updateConfirmButton);
      confirmBtn.addEventListener("click", () => {
        const selectedColumn = columnSelect.value;
        if (!selectedColumn) return;
        columnSelect.disabled = true;
        confirmBtn.disabled = true;
        rejectBtn.disabled = true;
        setColumnRoleStatus(role, selectedColumn, "confirmed").catch((err) =>
          alert(`Data mapping update failed: ${err}`)
        ).finally(() => {
          if (!row.isConnected) return;
          columnSelect.disabled = false;
          updateConfirmButton();
          rejectBtn.disabled = !suggestion.sqlName || suggestion.status === "rejected";
        });
      });

      const rejectBtn = document.createElement("button");
      rejectBtn.className = "btn btn-small";
      rejectBtn.textContent = "Reject";
      rejectBtn.disabled = !suggestion.sqlName || suggestion.status === "rejected";
      rejectBtn.addEventListener("click", () => {
        columnSelect.disabled = true;
        confirmBtn.disabled = true;
        rejectBtn.disabled = true;
        setColumnRoleStatus(role, suggestion.sqlName, "rejected").catch((err) =>
          alert(`Data mapping update failed: ${err}`)
        ).finally(() => {
          if (!row.isConnected) return;
          columnSelect.disabled = false;
          updateConfirmButton();
          rejectBtn.disabled = !suggestion.sqlName || suggestion.status === "rejected";
        });
      });
      actions.append(confirmBtn, rejectBtn);
      if (
        role === "timestamp" &&
        timestampAnalysis &&
        (timestampAnalysis.needsTimezone || timestampAnalysis.needsDateConvention)
      ) {
        const timezoneBtn = document.createElement("button");
        timezoneBtn.className = "btn btn-small";
        timezoneBtn.textContent = "Time format...";
        timezoneBtn.addEventListener("click", () => showTimezonePrompt(timestampAnalysis));
        actions.appendChild(timezoneBtn);
      }

      row.append(roleTitle, columnSelect, meta, actions);
      if (suggestion.reasons && suggestion.reasons.length > 0) {
        const reasons = document.createElement("div");
        reasons.className = "role-reasons";
        reasons.textContent = suggestion.reasons.join("; ");
        row.appendChild(reasons);
      }
      roleList.appendChild(row);
    });

    if (!roleDetectionError) {
      const automaticCount = columnRoleSuggestions.filter((row) => row.status === "suggested").length;
      const confirmedCount = columnRoleSuggestions.filter((row) => row.status === "confirmed").length;
      const mappedCount = columnRoleSuggestions.filter((row) => row.status !== "rejected").length;
      dataMappingSummary.textContent = `${mappedCount} inferred${confirmedCount ? `, ${confirmedCount} confirmed` : ""}`;
      rolePanelStatus.textContent = automaticCount
        ? "Automatic mappings are active for optional enrichment and timeline hints. Confirm only when you want to lock in an override."
        : "Mappings are optional. AI evidence search always searches the imported table directly.";
    }
    updateEvidenceColumnsUi();
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

    const chains = summary.chains || [];
    if (chains.length > 0) {
      const chainHeader = document.createElement("div");
      chainHeader.className = "sidebar-note";
      chainHeader.textContent = `${chains.length.toLocaleString()} attack chain${chains.length === 1 ? "" : "s"} (multi-tactic sequences)`;
      intelScanSummary.appendChild(chainHeader);
      chains.slice(0, 5).forEach((chain) => {
        const row = document.createElement("div");
        row.className = "scan-summary-row";
        const name = document.createElement("span");
        name.textContent = `${chain.host || "all rows"} — ${chain.tacticCount} tactics`;
        name.title = `${(chain.tacticNames || []).join(" → ")}\nTechniques: ${(chain.techniqueNames || []).join(", ")}\nRows ${chain.firstRow}–${chain.lastRow}`;
        const count = document.createElement("span");
        count.className = "scan-summary-count";
        count.textContent = `${chain.rowCount.toLocaleString()} rows, score ${chain.score}`;
        row.append(name, count);
        intelScanSummary.appendChild(row);
      });
    }
  }

  function normalizeQueryExpression(expression, depth = 0, state = { nodes: 0 }) {
    if (expression == null) return null;
    state.nodes += 1;
    if (depth > 8 || state.nodes > 128 || typeof expression !== "object" || Array.isArray(expression)) {
      throw new Error("AI search plan contains an invalid expression");
    }

    switch (expression.type) {
      case "and":
      case "or": {
        if (
          !Array.isArray(expression.children) ||
          expression.children.length === 0 ||
          expression.children.length > 128
        ) {
          throw new Error("AI search plan contains an invalid expression group");
        }
        return {
          type: expression.type,
          children: expression.children.map((child) => normalizeQueryExpression(child, depth + 1, state)),
        };
      }
      case "not": {
        const child = normalizeQueryExpression(expression.child, depth + 1, state);
        if (!child) throw new Error("AI search plan contains an empty NOT expression");
        return { type: "not", child };
      }
      case "search":
        if (typeof expression.value !== "string" || expression.value.length > 4096) {
          throw new Error("AI search plan contains an invalid search term");
        }
        return { type: "search", value: expression.value };
      case "predicate":
        if (
          !columns.some((column) => column.sqlName === expression.column) ||
          !FILTER_OPERATORS.has(expression.op) ||
          typeof expression.value !== "string" ||
          expression.value.length > 4096
        ) {
          throw new Error("AI search plan contains an invalid column predicate");
        }
        return {
          type: "predicate",
          column: expression.column,
          op: expression.op,
          value: expression.value,
        };
      case "rowIds":
        if (
          !Array.isArray(expression.values) ||
          expression.values.length === 0 ||
          expression.values.length > 1000 ||
          !expression.values.every((value) => Number.isSafeInteger(value) && value > 0)
        ) {
          throw new Error("AI search plan contains invalid row candidates");
        }
        // Row IDs are only accepted by copying a trusted backend-built QuerySpec. They are
        // never derived from the request text or synthesized in the frontend.
        return { type: "rowIds", values: [...expression.values] };
      case "matchNone":
        return { type: "matchNone" };
      case "semanticSelection":
        if (
          typeof expression.selectionId !== "string" ||
          !/^[0-9a-fA-F]{64}$/.test(expression.selectionId)
        ) {
          throw new Error("AI search plan contains an invalid semantic selection");
        }
        // Selection IDs are opaque backend capabilities. SQLite revalidates the current
        // dataset/build before every page, count, timeline, and export.
        return { type: "semanticSelection", selectionId: expression.selectionId };
      default:
        throw new Error("AI search plan contains an unknown expression type");
    }
  }

  function normalizeBackendQuerySpec(candidate) {
    if (!candidate || typeof candidate !== "object" || Array.isArray(candidate)) return null;
    const normalized = {
      search: candidate.search == null ? null : candidate.search,
      filters: [],
      sort: null,
      expression: normalizeQueryExpression(candidate.expression),
      cursor: null,
      limit: PAGE_SIZE,
    };
    if (
      normalized.search !== null &&
      (typeof normalized.search !== "string" || normalized.search.length > 4096)
    ) {
      throw new Error("AI search plan contains an invalid full-table search");
    }
    const candidateFilters = candidate.filters == null ? [] : candidate.filters;
    if (!Array.isArray(candidateFilters) || candidateFilters.length > 128) {
      throw new Error("AI search plan contains invalid filters");
    }
    normalized.filters = candidateFilters.map((filter) => {
      if (
        !filter ||
        !columns.some((column) => column.sqlName === filter.column) ||
        !FILTER_OPERATORS.has(filter.op) ||
        typeof filter.value !== "string" ||
        filter.value.length > 4096
      ) {
        throw new Error("AI search plan contains an invalid filter");
      }
      return { column: filter.column, op: filter.op, value: filter.value };
    });
    if (candidate.sort != null) {
      if (
        !columns.some((column) => column.sqlName === candidate.sort.column) ||
        !["asc", "desc"].includes(candidate.sort.direction)
      ) {
        throw new Error("AI search plan contains an invalid sort");
      }
      normalized.sort = {
        column: candidate.sort.column,
        direction: candidate.sort.direction,
      };
    }
    return normalized;
  }

  function expressionUsesSemanticSelection(expression) {
    if (!expression || typeof expression !== "object") return false;
    if (expression.type === "matchNone") return false;
    if (expression.type === "semanticSelection") return true;
    if (expression.type === "not") return expressionUsesSemanticSelection(expression.child);
    if (["and", "or"].includes(expression.type) && Array.isArray(expression.children)) {
      return expression.children.some(expressionUsesSemanticSelection);
    }
    return false;
  }

  function querySpecUsesSemanticSelection(spec) {
    return Boolean(spec && expressionUsesSemanticSelection(spec.expression));
  }

  function isPositiveSemanticClaim(message) {
    return [
      "Semantic matching was used:",
      "Semantic recall:",
      "Semantic retrieval uses",
      "Semantic document candidates",
      "Semantic expansion matched",
      "Semantic selection retained",
    ].some((prefix) => message.startsWith(prefix));
  }

  function renderGuidedPreview(result, queryText, { showReadyPlan = true } = {}) {
    guidedParseResult = result;
    guidedIntentToken = typeof result.intentToken === "string" && result.intentToken ? result.intentToken : null;
    guidedAuditId = Number.isInteger(result.auditId) ? result.auditId : null;
    guidedReviewStatus = result.reviewStatus || null;
    guidedMatchExplanation = Array.isArray(result.matchExplanation)
      ? result.matchExplanation.filter((item) => typeof item === "string" && item.trim())
      : [];
    guidedPreviewQueryText = queryText;
    try {
      guidedQuerySpec = normalizeBackendQuerySpec(result.querySpec);
    } catch (error) {
      guidedQuerySpec = null;
      result = {
        ...result,
        needsClarification: true,
        clarificationMessage: `The returned search plan was rejected by the frontend safety check: ${error.message}`,
      };
      guidedParseResult = result;
    }

    const semanticApplied = querySpecUsesSemanticSelection(guidedQuerySpec);
    const semanticFallbackReported = guidedMatchExplanation.some((item) =>
      item.startsWith("Semantic matching was not used:")
    );
    if (!semanticApplied && guidedMatchExplanation.some(isPositiveSemanticClaim)) {
      // A semantic status sentence alone is never proof that retrieval affects this plan. Only
      // the normalized, backend-issued semanticSelection expression can establish that.
      guidedMatchExplanation = guidedMatchExplanation.filter((item) => !isPositiveSemanticClaim(item));
      if (!semanticFallbackReported) {
        guidedMatchExplanation.push(
          "Semantic matching was not used: the frontend could not verify a trusted semantic selection in this preview. Exact and structured conditions remain available."
        );
      }
    }

    guidedRunBtn.textContent = "Search evidence";
    const previewLines = [result.previewText || "No search plan was returned."];
    const searchNotes = guidedMatchExplanation.filter((item) => item.startsWith("Semantic "));
    const matchRules = guidedMatchExplanation.filter((item) => !searchNotes.includes(item));
    if (matchRules.length > 0) {
      previewLines.push("", "Why rows will match:", ...matchRules.map((item) => `\u2022 ${item}`));
    }
    if (searchNotes.length > 0) {
      previewLines.push("", "Semantic search notes:", ...searchNotes.map((item) => `\u2022 ${item}`));
    }
    guidedPreviewText.textContent = previewLines.join("\n");

    if (result.aiAssisted) {
      const validation = result.validationStatus ? ` \u2022 ${result.validationStatus.replace(/_/g, " ")}` : "";
      const semanticStatus = semanticApplied
        ? " \u2022 semantic matching used"
        : guidedMatchExplanation.some((item) => item.startsWith("Semantic matching was not used:"))
          ? " \u2022 semantic matching not used"
          : "";
      guidedAiStatus.textContent = `Offline AI interpretation \u2022 ${guidedReviewStatus || "unreviewed"}${validation}${semanticStatus}`;
      guidedAiStatus.classList.remove("hidden");
      guidedRejectBtn.classList.toggle(
        "hidden",
        !showReadyPlan || guidedReviewStatus !== "unreviewed"
      );
    } else {
      guidedAiStatus.textContent = "Deterministic local search plan \u2022 no model inference";
      guidedAiStatus.classList.remove("hidden");
      guidedRejectBtn.classList.add("hidden");
    }

    const planReady = guidedPlanIsReadyToRun();
    if (!planReady) {
      guidedQueryPanel.classList.remove("hidden");
      if (!showReadyPlan) {
        guidedAiStatus.classList.add("hidden");
        guidedPreviewText.textContent = "I could not safely turn that request into a table search yet.";
      }
      guidedClarification.textContent =
        result.clarificationMessage ||
        "More detail is needed before a safe evidence search can run.";
      guidedClarification.classList.remove("hidden");
      guidedRunBtn.classList.add("hidden");
      if (table && table.getDataCount() > 0) {
        rowCountLabel.textContent = "Previous table results shown — they do not answer this unresolved request.";
      }
    } else {
      guidedClarification.textContent = "";
      guidedClarification.classList.add("hidden");
      guidedRunBtn.classList.toggle("hidden", !showReadyPlan);
      guidedQueryPanel.classList.toggle("hidden", !showReadyPlan);
    }
    guidedResetBtn.classList.toggle("hidden", !["guided", "querySpec"].includes(queryMode));
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
    const needsTimezone = Boolean(analysis.needsTimezone);
    const needsDateConvention = Boolean(analysis.needsDateConvention);
    const requirements = [];
    if (needsTimezone) requirements.push("a source timezone");
    if (needsDateConvention) requirements.push("the slash-date order");
    timezoneSummary.textContent = `${analysis.originalName} needs ${requirements.join(" and ")} before chronological ordering is safe.`;
    const samples = needsDateConvention
      ? analysis.sampleAmbiguousDateValues || []
      : analysis.sampleNaiveValues || [];
    if (samples.length > 0) {
      timezoneSamples.textContent = `Samples: ${samples.join("; ")}`;
      timezoneSamples.classList.remove("hidden");
    } else {
      timezoneSamples.textContent = "";
      timezoneSamples.classList.add("hidden");
    }
    timezoneInput.value = "";
    dateConventionSelect.value = analysis.inferredDateConvention || "";
    dateConventionWrap.classList.toggle("hidden", !needsDateConvention);
    timezoneInput.classList.toggle("hidden", !needsTimezone);
    timezoneUtcBtn.classList.toggle("hidden", !needsTimezone);
    timezoneNormalizeBtn.textContent = needsTimezone ? "Use timestamp details" : "Use date order";
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
      expression: null,
      cursor: forExport ? null : spec.cursor,
      limit: PAGE_SIZE,
    };
    return s;
  }

  function snapshotQuerySpec(source = spec) {
    return JSON.parse(JSON.stringify(source));
  }

  async function refreshCount() {
    const evidenceQuery = activeEvidenceQuery;
    const isAcceptedGuidedQuery =
      queryMode === "guided" &&
      evidenceQuery?.mode === "guided" &&
      evidenceQuery.auditId !== null &&
      evidenceQuery.intentToken !== null &&
      evidenceQuery.querySpec !== null;
    if (queryMode === "guided" && !isAcceptedGuidedQuery) {
      activeCountRequest = null;
      countRequestSequence += 1;
      totalCount = null;
      updateRowCountLabel();
      return;
    }

    // The accepted intent is executed through run_guided_query for paging, while its
    // backend-issued QuerySpec is safe to reuse for COUNT: the predicate compiler revalidates
    // any semantic selection against the current dataset and active semantic build.
    const countSpec = isAcceptedGuidedQuery ? evidenceQuery.querySpec : spec;
    const request = {
      id: ++countRequestSequence,
      contextRevision: guidedContextRevision,
      path: currentPath,
      sheet: currentSheet,
      mode: queryMode,
      auditId: evidenceQuery?.auditId ?? null,
      intentToken: evidenceQuery?.intentToken ?? null,
      table,
      evidenceQuery,
      spec: snapshotQuerySpec(countSpec),
    };
    activeCountRequest = request;
    totalCount = null;
    updateRowCountLabel();
    const isCurrent = () =>
      activeCountRequest === request &&
      loadedContextIsCurrent(request) &&
      queryMode === request.mode &&
      table === request.table &&
      (request.mode === "normal" || activeEvidenceQuery === request.evidenceQuery);
    try {
      const count = await invoke("count_rows", { spec: request.spec });
      if (!isCurrent()) return;
      totalCount = count;
      updateRowCountLabel();
    } catch (err) {
      if (!isCurrent()) return;
      console.error("count_rows failed", err);
    } finally {
      if (activeCountRequest === request) activeCountRequest = null;
    }
  }

  function updateRowCountLabel() {
    const shown = table ? table.getDataCount() : 0;
    if (totalCount === null) {
      rowCountLabel.textContent =
        ["guided", "querySpec"].includes(queryMode)
          ? `${shown} AI evidence rows on this page`
          : `${shown} rows on this page`;
    } else {
      rowCountLabel.textContent = `${totalCount.toLocaleString()} ${["guided", "querySpec"].includes(queryMode) ? "evidence" : "matching"} rows`;
    }
    pageLabel.textContent = `page ${pageIndex}`;
  }

  function setAiMatchColumnVisible(visible) {
    if (!table) return;
    const matchColumn = table.getColumn("__aiMatch");
    if (!matchColumn) return;
    if (visible) {
      matchColumn.show();
    } else {
      matchColumn.hide();
    }
  }

  async function refreshData() {
    if (guidedActiveParse !== null || guidedActiveAction !== null || activeDataRequest !== null) {
      return null;
    }
    // Disabled synchronously (before the first await below) so a rapid double-click on
    // Prev/Next can't fire a second query_rows() while this one is still in flight and read a
    // stale nextCursor — the button is unclickable for the whole round trip either way.
    const modeAtStart = queryMode;
    const isGuidedRequest = modeAtStart === "guided";
    const isQuerySpecRequest = modeAtStart === "querySpec";
    const isTrackedEvidenceRequest = isGuidedRequest || isQuerySpecRequest;
    if (isTrackedEvidenceRequest && guidedActiveQuery !== null) return null;
    const evidenceQuery = isTrackedEvidenceRequest ? activeEvidenceQuery : null;
    if (isTrackedEvidenceRequest && evidenceQuery?.mode !== modeAtStart) return null;
    setAiMatchColumnVisible(isTrackedEvidenceRequest);

    const request = {
      id: ++dataRequestSequence,
      contextRevision: guidedContextRevision,
      path: currentPath,
      sheet: currentSheet,
      mode: modeAtStart,
      auditId: evidenceQuery?.auditId ?? null,
      intentToken: evidenceQuery?.intentToken ?? null,
      evidenceQuery,
      cursor: spec.cursor,
      limit: spec.limit,
      spec: isGuidedRequest ? null : snapshotQuerySpec(),
      table,
    };
    activeDataRequest = request;
    updateGuidedInteractionControls();
    if (isTrackedEvidenceRequest) {
      guidedActiveQuery = request;
      updateGuidedInteractionControls();
    }

    const requestIsCurrent = () =>
      activeDataRequest === request &&
      loadedContextIsCurrent(request) &&
      queryMode === request.mode &&
      (!isTrackedEvidenceRequest || activeEvidenceQuery === request.evidenceQuery) &&
      table === request.table;

    prevPageBtn.disabled = true;
    nextPageBtn.disabled = true;
    showProgress(
      isGuidedRequest
        ? "Searching evidence..."
        : queryMode === "querySpec"
          ? "Applying evidence search plan..."
          : "Filtering...",
      0.5
    );
    try {
      const page =
        isGuidedRequest
          ? await invoke("run_guided_query", {
              intentToken: request.intentToken,
              auditId: request.auditId,
              cursor: request.cursor,
              limit: request.limit,
            })
          : await invoke("query_rows", { spec: request.spec });
      if (!requestIsCurrent() || !table) return null;
      await request.table.replaceData(page.rows);
      if (!requestIsCurrent() || !table) return null;
      nextCursor = page.nextCursor;
      hasMore = page.hasMore;
      updateRowCountLabel();
      return page;
    } catch (err) {
      if (!requestIsCurrent()) return null;
      console.error(`${isGuidedRequest ? "run_guided_query" : "query_rows"} failed`, err);
      alert(`Query failed: ${err}`);
      return null;
    } finally {
      const stillCurrent = requestIsCurrent();
      if (activeDataRequest === request) {
        activeDataRequest = null;
        updateGuidedInteractionControls();
      }
      if (guidedActiveQuery === request) {
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

  function discardGuidedPlanForTableAction() {
    const auditId = guidedAuditId;
    const intentToken = guidedIntentToken;
    if (
      auditId !== null &&
      typeof intentToken === "string" &&
      guidedReviewStatus === "unreviewed"
    ) {
      invoke("set_guided_parse_decision", {
        auditId,
        intentToken,
        decision: "edited",
      }).catch((err) => console.error("could not retire AI interpretation before table filtering", err));
    }

    pendingSemanticSearch = null;
    guidedParseResult = null;
    guidedIntentToken = null;
    guidedAuditId = null;
    guidedReviewStatus = null;
    guidedQuerySpec = null;
    guidedMatchExplanation = [];
    guidedPreviewQueryText = null;
    guidedSearchBox.value = "";
    guidedQueryPanel.classList.add("hidden");
    guidedPreviewText.textContent = "";
    guidedAiStatus.textContent = "";
    guidedAiStatus.classList.add("hidden");
    guidedClarification.textContent = "";
    guidedClarification.classList.add("hidden");
    guidedRunBtn.textContent = "Search evidence";
    guidedRunBtn.classList.add("hidden");
    guidedRejectBtn.classList.add("hidden");
    guidedResetBtn.classList.add("hidden");
    aiSearchAvailability.textContent = "Ready to search every imported row. No enrichment scan is required.";
    aiSearchAvailability.classList.add("ready");
  }

  function applyControlsAndReload() {
    cancelSearchDebounce();
    if (sheetLoadInFlight || tableTransitionInFlight()) return null;
    discardGuidedPlanForTableAction();
    queryMode = "normal";
    activeEvidenceQuery = null;
    setAiMatchColumnVisible(false);
    guidedResetBtn.classList.add("hidden");
    spec.search = searchBox.value.trim() || null;
    spec.filters = currentFilterValues();
    spec.sort = sortColumn.value
      ? { column: sortColumn.value, direction: sortDirection.value }
      : null;
    spec.expression = null;
    resetPagination();
    const page = refreshData();
    refreshCount();
    return page;
  }

  let searchDebounceHandle = null;
  function cancelSearchDebounce() {
    if (searchDebounceHandle) clearTimeout(searchDebounceHandle);
    searchDebounceHandle = null;
  }

  function debouncedApply() {
    cancelSearchDebounce();
    pendingSemanticSearch = null;
    if (sheetLoadInFlight || tableTransitionInFlight()) return;
    const request = {
      contextRevision: guidedContextRevision,
      path: currentPath,
      sheet: currentSheet,
    };
    searchDebounceHandle = setTimeout(() => {
      searchDebounceHandle = null;
      if (
        loadedContextIsCurrent(request) &&
        controlsEnabled &&
        !sheetLoadInFlight &&
        !tableTransitionInFlight()
      ) {
        applyControlsAndReload();
      }
    }, 300);
  }

  function loadedContextIsCurrent(request) {
    return (
      guidedContextRevision === request.contextRevision &&
      currentPath === request.path &&
      currentSheet === request.sheet
    );
  }

  function reportExportIsCurrent(request) {
    return (
      activeReportExport === request &&
      currentPath === request.path &&
      currentSheet === request.sheet
    );
  }

  function updateReportExportButton() {
    reportExportBtn.disabled =
      !controlsEnabled ||
      sheetLoadInFlight ||
      activeReportExport !== null ||
      tableTransitionInFlight();
  }

  function semanticIndexRequestIsCurrent(request) {
    return activeSemanticIndexRequest === request && loadedContextIsCurrent(request);
  }

  function semanticBoundedIndexNote(state) {
    const limitations = [
      [state.cellsTruncated, "oversized cells truncated"],
      [state.columnsOmitted, "eligible wide-row values omitted"],
      [state.chunksOmitted, "chunk documents omitted or truncated"],
      [state.documentsSkipped, "new document candidates skipped"],
      [state.mappingsSkipped, "document-to-row mappings skipped"],
    ]
      .filter(([count]) => Number.isSafeInteger(count) && count > 0)
      .map(([count, label]) => `${count.toLocaleString()} ${label}`);
    return limitations.length ? ` Bounded-index notes: ${limitations.join("; ")}.` : "";
  }

  function renderSemanticIndexState() {
    semanticIndexStatus.className = `semantic-index-status ${semanticIndexState.status}`;
    const boundedNote = semanticBoundedIndexNote(semanticIndexState);
    if (semanticIndexState.status === "ready") {
      const documentCount = semanticIndexState.summary?.documentsMapped;
      const documentDetail = Number.isFinite(documentCount)
        ? ` from ${documentCount.toLocaleString()} deduplicated document${documentCount === 1 ? "" : "s"}`
        : "";
      semanticIndexStatus.textContent = `Semantic matching ready (${semanticIndexState.rowsIndexed.toLocaleString()} raw rows processed${documentDetail}).${boundedNote}${boundedNote ? " Exact and structured matching still covers every raw row." : ""}`;
    } else if (semanticIndexState.status === "building") {
      if (semanticIndexState.phase === "loadingModel") {
        semanticIndexStatus.textContent = "Loading the local semantic model. Exact and structured AI search are ready now.";
      } else if (semanticIndexState.phase === "preparing") {
        semanticIndexStatus.textContent = "Preparing the semantic index. Exact and structured AI search are ready now.";
      } else if (semanticIndexState.phase === "estimating") {
        semanticIndexStatus.textContent = "Estimating semantic index size from a sample of rows. Exact and structured AI search are ready now.";
      } else {
        const progressParts = [];
        if (semanticIndexState.rowsIndexed) {
          progressParts.push(`${semanticIndexState.rowsIndexed.toLocaleString()} raw rows processed`);
        }
        if (semanticIndexState.documentsEmbedded) {
          progressParts.push(`${semanticIndexState.documentsEmbedded.toLocaleString()} documents embedded`);
        }
        if (semanticIndexState.mappingsWritten) {
          progressParts.push(`${semanticIndexState.mappingsWritten.toLocaleString()} row mappings saved`);
        }
        const progress = progressParts.length ? ` ${progressParts.join("; ")}.` : "";
        const resumed = semanticIndexState.resumedFromRow
          ? ` Resumed after row ${semanticIndexState.resumedFromRow.toLocaleString()}.`
          : "";
        semanticIndexStatus.textContent = `Semantic matching is preparing in resumable batches.${progress}${resumed}${boundedNote} Exact and structured AI search are ready now.`;
      }
    } else if (semanticIndexState.status === "error") {
      semanticIndexStatus.textContent = "Semantic matching is unavailable; exact and structured AI search remain ready.";
    } else {
      semanticIndexStatus.textContent = "Semantic matching starts automatically after import.";
    }
  }

  function previewMissedSemanticIndex(result = guidedParseResult) {
    return (
      result?.semanticStatus === "index_not_ready" &&
      !querySpecUsesSemanticSelection(guidedQuerySpec)
    );
  }

  function pendingSemanticSearchIsCurrent(request) {
    return (
      loadedContextIsCurrent(request) &&
      guidedSearchBox.value.trim() === request.queryText &&
      activeEvidenceQuery?.auditId === request.auditId &&
      activeEvidenceQuery?.intentToken === request.intentToken &&
      activeEvidenceQuery?.querySpec === request.querySpec &&
      ["guided", "querySpec"].includes(queryMode)
    );
  }

  function refreshPendingSemanticSearch() {
    if (semanticIndexState.status !== "ready" || pendingSemanticSearch === null) return;
    const request = pendingSemanticSearch;
    if (activeReportExport !== null || sheetLoadInFlight) {
      setTimeout(refreshPendingSemanticSearch, 250);
      return;
    }
    if (!pendingSemanticSearchIsCurrent(request)) {
      pendingSemanticSearch = null;
      return;
    }
    if (
      guidedActiveParse !== null ||
      guidedActiveAction !== null ||
      guidedActiveQuery !== null ||
      activeDataRequest !== null
    ) {
      setTimeout(refreshPendingSemanticSearch, 250);
      return;
    }
    pendingSemanticSearch = null;
    aiSearchAvailability.textContent = "Semantic matching is ready. Refreshing the evidence results...";
    aiSearchAvailability.classList.remove("ready");
    searchGuidedQuery(request.queryText, { semanticRetry: true }).catch((err) =>
      console.error("automatic semantic evidence refresh failed", err)
    );
  }

  function queueSemanticSearchRefresh(request, plan) {
    pendingSemanticSearch = {
      ...request,
      auditId: plan.auditId,
      intentToken: plan.intentToken,
      querySpec: plan.querySpec,
    };
    if (semanticIndexState.status === "ready") {
      refreshPendingSemanticSearch();
    }
  }

  function failPendingSemanticSearch() {
    if (pendingSemanticSearch && pendingSemanticSearchIsCurrent(pendingSemanticSearch)) {
      aiSearchAvailability.textContent =
        "Exact AI results remain visible. Semantic matching could not be prepared for this file.";
      aiSearchAvailability.classList.add("ready");
    }
    pendingSemanticSearch = null;
  }

  async function startSemanticIndexForLoadedFile() {
    const request = {
      id: ++semanticIndexRequestSequence,
      contextRevision: guidedContextRevision,
      path: currentPath,
      sheet: currentSheet,
    };
    activeSemanticIndexRequest = request;
    semanticIndexState = {
      status: "building",
      phase: "loadingModel",
      buildId: null,
      rowsIndexed: 0,
      documentsEmbedded: 0,
      mappingsWritten: 0,
      documentsSkipped: 0,
      mappingsSkipped: 0,
      cellsTruncated: 0,
      columnsOmitted: 0,
      chunksOmitted: 0,
      resumedFromRow: 0,
      summary: null,
      error: null,
    };
    renderSemanticIndexState();
    try {
      const status = await invoke("semantic_index_status");
      if (!semanticIndexRequestIsCurrent(request)) return null;
      if (status.ready) {
        semanticIndexState = {
          status: "ready",
          phase: "ready",
          buildId: null,
          rowsIndexed: status.rowsIndexed || 0,
          documentsEmbedded: 0,
          mappingsWritten: 0,
          documentsSkipped: status.documentsSkipped || 0,
          mappingsSkipped: status.mappingsSkipped || 0,
          cellsTruncated: status.cellsTruncated || 0,
          columnsOmitted: status.columnsOmitted || 0,
          chunksOmitted: status.chunksOmitted || 0,
          resumedFromRow: 0,
          summary: null,
          error: null,
        };
        renderSemanticIndexState();
        refreshPendingSemanticSearch();
        return status;
      }

      const summary = await invoke("build_semantic_index");
      if (!semanticIndexRequestIsCurrent(request)) return null;
      if (summary.cancelled) {
        throw new Error("semantic preparation was superseded before publication");
      }
      semanticIndexState = {
        status: "ready",
        phase: "ready",
        buildId: semanticIndexState.buildId,
        rowsIndexed: summary.rowsIndexed || 0,
        documentsEmbedded: summary.documentsIndexed || 0,
        mappingsWritten: summary.mappingsWritten || 0,
        documentsSkipped: summary.documentsSkipped || 0,
        mappingsSkipped: summary.mappingsSkipped || 0,
        cellsTruncated: summary.cellsTruncated || 0,
        columnsOmitted: summary.columnsOmitted || 0,
        chunksOmitted: summary.chunksOmitted || 0,
        resumedFromRow: summary.resumed ? semanticIndexState.resumedFromRow : 0,
        summary,
        error: null,
      };
      renderSemanticIndexState();
      refreshPendingSemanticSearch();
      return summary;
    } catch (err) {
      if (!semanticIndexRequestIsCurrent(request)) return null;
      console.error("semantic index build failed", err);
      semanticIndexState = {
        status: "error",
        phase: null,
        buildId: null,
        rowsIndexed: 0,
        documentsEmbedded: 0,
        mappingsWritten: 0,
        documentsSkipped: 0,
        mappingsSkipped: 0,
        cellsTruncated: 0,
        columnsOmitted: 0,
        chunksOmitted: 0,
        resumedFromRow: 0,
        summary: null,
        error: String(err),
      };
      renderSemanticIndexState();
      failPendingSemanticSearch();
      return null;
    } finally {
      if (semanticIndexRequestIsCurrent(request)) activeSemanticIndexRequest = null;
    }
  }

  function automaticTimestampMappingIsCurrent(suggestion, request) {
    const current = columnRoleSuggestions.find((row) => row.role === "timestamp");
    return (
      loadedContextIsCurrent(request) &&
      current &&
      current.sqlName === suggestion.sqlName &&
      current.status !== "rejected"
    );
  }

  function timestampOperationIsCurrent(operation) {
    return activeTimestampOperation === operation && loadedContextIsCurrent(operation);
  }

  async function analyzeAutomaticTimestampMapping(suggestion, request) {
    if (
      automaticTimestampInFlight ||
      !suggestion ||
      suggestion.status === "rejected" ||
      suggestion.confidence < 0.75 ||
      !automaticTimestampMappingIsCurrent(suggestion, request)
    ) {
      return null;
    }

    const operation = {
      id: ++timestampOperationSequence,
      contextRevision: request.contextRevision,
      path: request.path,
      sheet: request.sheet,
      kind: "automatic",
    };
    activeTimestampOperation = operation;
    automaticTimestampInFlight = true;
    automaticTimestampSqlName = suggestion.sqlName;
    dataMappingSummary.textContent = "Checking timestamp format...";
    rolePanelStatus.textContent = "The high-confidence timestamp mapping is being checked in the background.";
    try {
      const analysis = await invoke("analyze_timestamp_column");
      if (!automaticTimestampMappingIsCurrent(suggestion, request) || !timestampOperationIsCurrent(operation)) return null;
      timestampAnalysis = analysis;
      if (analysis.needsTimezone || analysis.needsDateConvention) {
        renderRoleSuggestions();
        showTimezonePrompt(analysis);
        dataMappingSummary.textContent = "Timestamp details needed";
        rolePanelStatus.textContent = "Confirm timestamp details only if chronological ordering is needed. AI evidence search remains available.";
        return null;
      }

      const summary = await invoke("normalize_timestamp_column", { naiveTimezone: null });
      if (!automaticTimestampMappingIsCurrent(suggestion, request) || !timestampOperationIsCurrent(operation)) return null;
      timestampNormalizationSummary = summary;
      timezonePanel.classList.add("hidden");
      dataMappingSummary.textContent = `${columnRoleSuggestions.filter((row) => row.status !== "rejected").length} inferred, time ready`;
      rolePanelStatus.textContent = `Unambiguous timestamp values (explicit timezone or epoch) were normalized to UTC automatically (${summary.rowsWritten.toLocaleString()} rows).`;
      return summary;
    } catch (err) {
      if (!automaticTimestampMappingIsCurrent(suggestion, request) || !timestampOperationIsCurrent(operation)) return null;
      console.error("automatic timestamp analysis/normalization failed", err);
      rolePanelStatus.textContent = `Automatic timestamp preparation was skipped: ${err}. AI evidence search is unaffected.`;
      return null;
    } finally {
      if (loadedContextIsCurrent(request) && automaticTimestampSqlName === suggestion.sqlName) {
        automaticTimestampInFlight = false;
        automaticTimestampSqlName = null;
      }
      if (activeTimestampOperation === operation) activeTimestampOperation = null;
    }
  }

  async function detectColumnRolesForLoadedFile({ throwOnError = false } = {}) {
    const request = {
      id: ++roleDetectionRequestSequence,
      contextRevision: guidedContextRevision,
      path: currentPath,
      sheet: currentSheet,
    };
    activeRoleDetectionRequest = request;
    roleDetectionInFlight = true;
    roleDetectionError = null;
    renderRoleSuggestions();
    try {
      const suggestions = await invoke("detect_column_roles");
      if (activeRoleDetectionRequest !== request || !loadedContextIsCurrent(request)) return [];
      columnRoleSuggestions = suggestions;
      return suggestions;
    } catch (err) {
      if (activeRoleDetectionRequest !== request || !loadedContextIsCurrent(request)) return [];
      console.error("detect_column_roles failed", err);
      roleDetectionError = err;
      if (throwOnError) throw err;
      return [];
    } finally {
      if (activeRoleDetectionRequest === request && loadedContextIsCurrent(request)) {
        activeRoleDetectionRequest = null;
        roleDetectionInFlight = false;
        renderRoleSuggestions();
        const timestampSuggestion = columnRoleSuggestions.find(
          (row) => row.role === "timestamp" && row.status !== "rejected" && row.confidence >= 0.75
        );
        if (timestampSuggestion) {
          await analyzeAutomaticTimestampMapping(timestampSuggestion, request);
        }
      }
    }
  }

  async function setColumnRoleStatus(role, sqlName, status) {
    const request = {
      id: ++mappingRequestSequence,
      contextRevision: guidedContextRevision,
      path: currentPath,
      sheet: currentSheet,
      role,
      sqlName,
      status,
    };
    activeMappingRequests.set(role, request);
    rolePanelStatus.textContent = `Updating ${formatRoleName(role)} mapping...`;
    let updated;
    try {
      updated = await invoke("set_column_role_status", { role, sqlName, status });
      if (activeMappingRequests.get(role) !== request || !loadedContextIsCurrent(request)) return null;
      upsertRoleSuggestion(updated);
      renderRoleSuggestions();
    } catch (err) {
      if (activeMappingRequests.get(role) !== request || !loadedContextIsCurrent(request)) return null;
      console.error("set_column_role_status failed", err);
      rolePanelStatus.textContent = `Data mapping update failed: ${err}`;
      throw err;
    } finally {
      if (activeMappingRequests.get(role) === request) activeMappingRequests.delete(role);
    }

    if (
      role === "timestamp" &&
      status === "confirmed" &&
      (!automaticTimestampInFlight || automaticTimestampSqlName !== sqlName)
    ) {
      try {
        await handleTimestampConfirmed(request);
      } catch (err) {
        if (loadedContextIsCurrent(request)) {
          console.error("timestamp preparation after mapping failed", err);
          rolePanelStatus.textContent = `Timestamp mapping was saved, but timeline preparation failed: ${err}`;
        }
      }
    }
    return updated;
  }

  async function handleTimestampConfirmed(context = {
    contextRevision: guidedContextRevision,
    path: currentPath,
    sheet: currentSheet,
  }) {
    const operation = {
      id: ++timestampOperationSequence,
      contextRevision: context.contextRevision,
      path: context.path,
      sheet: context.sheet,
      kind: "manual-analysis",
    };
    activeTimestampOperation = operation;
    rolePanelStatus.textContent = "Analyzing timestamp column...";
    try {
      const analysis = await invoke("analyze_timestamp_column");
      if (!timestampOperationIsCurrent(operation)) return null;
      timestampAnalysis = analysis;
      if (analysis.needsTimezone || analysis.needsDateConvention) {
        renderRoleSuggestions();
        showTimezonePrompt(analysis);
        dataMappingSummary.textContent = "Timestamp details needed";
        rolePanelStatus.textContent = "Timestamp normalization needs examiner input.";
        return null;
      }
      const summary = await normalizeTimestampColumn(null, operation);
      if (!timestampOperationIsCurrent(operation)) return null;
      rolePanelStatus.textContent = `Timestamp normalized to UTC: ${summary.rowsWritten.toLocaleString()} rows written.`;
      return summary;
    } catch (err) {
      if (!timestampOperationIsCurrent(operation)) return null;
      console.error("timestamp analysis/normalization failed", err);
      rolePanelStatus.textContent = `Timestamp normalization failed: ${err}`;
      throw err;
    } finally {
      if (activeTimestampOperation === operation) activeTimestampOperation = null;
    }
  }

  async function normalizeTimestampColumn(naiveTimezone, existingOperation = null, dateConvention = null) {
    const operation = existingOperation || {
      id: ++timestampOperationSequence,
      contextRevision: guidedContextRevision,
      path: currentPath,
      sheet: currentSheet,
      kind: "manual-normalization",
    };
    if (!existingOperation) activeTimestampOperation = operation;
    timezoneNormalizeBtn.disabled = true;
    timezoneUtcBtn.disabled = true;
    try {
      const summary = await invoke("normalize_timestamp_column", { naiveTimezone, dateConvention });
      if (!timestampOperationIsCurrent(operation)) return null;
      timestampNormalizationSummary = summary;
      if (timestampAnalysis) {
        timestampAnalysis = {
          ...timestampAnalysis,
          needsTimezone: false,
          needsDateConvention: false,
        };
      }
      timezonePanel.classList.add("hidden");
      renderRoleSuggestions();
      dataMappingSummary.textContent = `${columnRoleSuggestions.filter((row) => row.status !== "rejected").length} inferred, time ready`;
      rolePanelStatus.textContent = `Timestamp normalized to UTC: ${summary.rowsWritten.toLocaleString()} rows written.`;
      return summary;
    } catch (err) {
      if (!timestampOperationIsCurrent(operation)) return null;
      console.error("normalize_timestamp_column failed", err);
      timezoneSummary.textContent = `Normalization failed: ${err}`;
      throw err;
    } finally {
      if (activeTimestampOperation === operation && !existingOperation) {
        activeTimestampOperation = null;
      }
      if (activeTimestampOperation === operation || activeTimestampOperation === null) {
        timezoneNormalizeBtn.disabled = false;
        timezoneUtcBtn.disabled = false;
      }
    }
  }

  async function runIntelScan(evidenceColumns = inferredEvidenceColumns()) {
    if (!evidenceColumns || evidenceColumns.length === 0) {
      throw new Error("no evidence columns were inferred; choose columns in Data mapping first");
    }

    intelScanInFlight = true;
    updateEvidenceColumnsUi();
    showProgress("Running optional threat enrichment...", 0);
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

  async function previewGuidedQuery(
    queryText = guidedSearchBox.value,
    { showReadyPlan = true } = {}
  ) {
    const trimmed = queryText.trim();
    if (!trimmed) return null;
    if (
      guidedActiveAction !== null ||
      guidedActiveQuery !== null ||
      activeDataRequest !== null ||
      activeReportExport !== null ||
      sheetLoadInFlight
    ) return null;
    cancelSearchDebounce();

    const request = {
      id: ++guidedParseRequestSequence,
      contextRevision: guidedContextRevision,
      path: currentPath,
      sheet: currentSheet,
      queryText: trimmed,
    };
    // Replacing this object immediately makes any older inference response stale, even when a
    // test/debug caller starts a second preview without waiting for the first one.
    guidedActiveParse = request;

    updateGuidedInteractionControls();
    guidedQueryPanel.classList.toggle("hidden", !showReadyPlan);
    guidedPreviewText.textContent = "Understanding the evidence request...";
    guidedClarification.classList.add("hidden");
    guidedRunBtn.classList.add("hidden");
    guidedRejectBtn.classList.add("hidden");
    showProgress("Local AI is planning the evidence search...", 0.3);
    try {
      if (guidedAuditId !== null && guidedReviewStatus === "unreviewed") {
        const edited = await decideGuidedParse("edited", { allowDuringParse: true });
        if (!edited || !guidedParseIsCurrent(request)) return null;
      }
      const result = await invoke("parse_guided_query", { queryText: trimmed });
      if (!guidedParseIsCurrent(request)) {
        // If only the text changed while inference was running, retire the now-invisible audit
        // record. A source change is handled against the old database by the Rust command.
        if (
          loadedContextIsCurrent(request) &&
          result?.aiAssisted &&
          Number.isInteger(result.auditId) &&
          typeof result.intentToken === "string"
        ) {
          invoke("set_guided_parse_decision", {
            auditId: result.auditId,
            intentToken: result.intentToken,
            decision: "edited",
          }).catch((err) => console.error("could not retire stale AI interpretation", err));
        }
        return null;
      }
      renderGuidedPreview(result, request.queryText, { showReadyPlan });
      return result;
    } catch (err) {
      if (!guidedParseIsCurrent(request)) return null;
      console.error("parse_guided_query failed", err);
      guidedPreviewText.textContent = `Evidence search preview failed: ${err}`;
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

  function guidedSearchResultIsCurrent(request, plan) {
    return (
      loadedContextIsCurrent(request) &&
      guidedSearchBox.value.trim() === request.queryText &&
      guidedPreviewQueryText === request.queryText &&
      guidedAuditId === plan.auditId &&
      guidedIntentToken === plan.intentToken &&
      guidedQuerySpec === plan.querySpec &&
      guidedPlanIsReadyToRun()
    );
  }

  function showGuidedSearchFailure(message, { canRetry = false } = {}) {
    guidedQueryPanel.classList.remove("hidden");
    guidedAiStatus.classList.add("hidden");
    guidedPreviewText.textContent = "The AI search could not be completed.";
    guidedClarification.textContent = message;
    guidedClarification.classList.remove("hidden");
    guidedRunBtn.textContent = "Retry search";
    guidedRunBtn.classList.toggle("hidden", !canRetry);
    guidedRejectBtn.classList.add("hidden");
    aiSearchAvailability.textContent = "The last AI search did not change the table.";
    aiSearchAvailability.classList.remove("ready");
  }

  async function searchGuidedQuery(
    queryText = guidedSearchBox.value,
    { semanticRetry = false } = {}
  ) {
    const trimmed = queryText.trim();
    if (!trimmed) return null;
    if (
      guidedActiveParse !== null ||
      guidedActiveAction !== null ||
      guidedActiveQuery !== null ||
      activeDataRequest !== null ||
      activeReportExport !== null ||
      sheetLoadInFlight
    ) {
      return null;
    }
    pendingSemanticSearch = null;
    const request = {
      contextRevision: guidedContextRevision,
      path: currentPath,
      sheet: currentSheet,
      queryText: trimmed,
    };
    aiSearchAvailability.textContent = "Understanding the request and searching the imported table...";
    aiSearchAvailability.classList.remove("ready");

    let result;
    try {
      result = await previewGuidedQuery(trimmed, { showReadyPlan: false });
    } catch (err) {
      if (loadedContextIsCurrent(request) && guidedSearchBox.value.trim() === request.queryText) {
        showGuidedSearchFailure(`The request could not be understood: ${err}`);
      }
      throw err;
    }
    const plan = {
      auditId: guidedAuditId,
      intentToken: guidedIntentToken,
      querySpec: guidedQuerySpec,
    };
    if (!result || !guidedSearchResultIsCurrent(request, plan)) {
      if (loadedContextIsCurrent(request) && guidedParseResult?.needsClarification) {
        aiSearchAvailability.textContent = "Add the requested detail, then search again.";
        aiSearchAvailability.classList.remove("ready");
      }
      return result;
    }

    const missedSemanticIndex = previewMissedSemanticIndex(result);
    if (missedSemanticIndex && semanticIndexState.status === "ready" && !semanticRetry) {
      if (guidedAuditId !== null && guidedReviewStatus === "unreviewed") {
        let retired;
        try {
          retired = await decideGuidedParse("edited");
        } catch (err) {
          showGuidedSearchFailure(`The first search plan could not be refreshed safely: ${err}`);
          throw err;
        }
        if (!retired || !loadedContextIsCurrent(request) || guidedSearchBox.value.trim() !== request.queryText) {
          return result;
        }
      }
      return searchGuidedQuery(trimmed, { semanticRetry: true });
    }

    guidedQueryPanel.classList.add("hidden");
    try {
      const page = await runGuidedQuery(guidedIntentToken);
      if (!guidedSearchResultIsCurrent(request, plan)) return result;
      if (page === null) {
        showGuidedSearchFailure("The table query failed. You can retry the same request.", {
          canRetry: true,
        });
        return result;
      }
      const shown = Array.isArray(page.rows) ? page.rows.length : 0;
      const resultMessage =
        shown === 0
          ? 'Search complete. No evidence rows matched this request. Use "Clear AI search" to return to the full table.'
          : `Showing ${shown.toLocaleString()} AI evidence row${shown === 1 ? "" : "s"}${page.hasMore ? " on this page" : ""}.`;
      if (missedSemanticIndex && semanticIndexState.status === "ready" && !semanticRetry) {
        return searchGuidedQuery(trimmed, { semanticRetry: true });
      }
      if (missedSemanticIndex && semanticIndexState.status === "building") {
        queueSemanticSearchRefresh(request, plan);
        aiSearchAvailability.textContent =
          shown === 0
            ? "No exact rows matched yet. Semantic matching is still preparing; results will refresh automatically."
            : `${resultMessage} Semantic matching is still preparing; results will refresh automatically.`;
      } else if (missedSemanticIndex) {
        aiSearchAvailability.textContent = `${resultMessage} Semantic matching was not available for this run.`;
      } else {
        aiSearchAvailability.textContent = resultMessage;
      }
      aiSearchAvailability.classList.add("ready");
      guidedQueryPanel.classList.add("hidden");
      guidedResetBtn.classList.remove("hidden");
      return result;
    } catch (err) {
      if (loadedContextIsCurrent(request) && guidedSearchBox.value.trim() === request.queryText) {
        showGuidedSearchFailure(`The search could not start: ${err}`, {
          canRetry: guidedPlanIsReadyToRun(),
        });
      }
      throw err;
    }
  }

  async function requestInitialEvidencePage(action, mode) {
    const targetTable = table;
    if (!targetTable) throw new Error("the evidence table is not available");
    const previousRows = targetTable.getData();
    prevPageBtn.disabled = true;
    nextPageBtn.disabled = true;
    showProgress("Searching evidence...", 0.5);
    try {
      const page =
        mode === "guided"
          ? await invoke("run_guided_query", {
              intentToken: action.intentToken,
              auditId: action.auditId,
              cursor: null,
              limit: PAGE_SIZE,
            })
          : await invoke("query_rows", {
              spec: { ...action.querySpec, cursor: null, limit: PAGE_SIZE },
            });
      if (!guidedActionIsCurrent(action) || table !== targetTable) return null;
      if (!page || !Array.isArray(page.rows)) {
        throw new Error("the evidence query returned an invalid page");
      }
      await targetTable.replaceData(page.rows);
      if (!guidedActionIsCurrent(action) || table !== targetTable) {
        if (table === targetTable && loadedContextIsCurrent(action)) {
          await targetTable.replaceData(previousRows);
        }
        return null;
      }
      return page;
    } finally {
      if (guidedActionIsCurrent(action)) {
        prevPageBtn.disabled = cursorStack.length === 0;
        nextPageBtn.disabled = !hasMore;
        hideProgress();
      }
    }
  }

  function publishInitialEvidencePage(action, mode, page) {
    if (mode === "querySpec") {
      spec = { ...action.querySpec, cursor: null, limit: PAGE_SIZE };
    }
    queryMode = mode;
    totalCount = null;
    resetPagination();
    nextCursor = page.nextCursor;
    hasMore = Boolean(page.hasMore);
    activeEvidenceQuery = {
      mode,
      auditId: action.auditId,
      intentToken: action.intentToken,
      querySpec: action.querySpec,
    };
    setAiMatchColumnVisible(true);
    guidedResetBtn.classList.remove("hidden");
    guidedRunBtn.textContent = "Search evidence";
    guidedRunBtn.classList.remove("hidden");
    prevPageBtn.disabled = true;
    nextPageBtn.disabled = !hasMore;
    updateRowCountLabel();
    refreshCount();
  }

  async function runGuidedQuery(intentToken = guidedIntentToken) {
    cancelSearchDebounce();
    const deterministicPlanReady =
      guidedParseResult &&
      !guidedParseResult.aiAssisted &&
      guidedAuditId === null &&
      guidedQuerySpec !== null;
    if (deterministicPlanReady) {
      const action = beginGuidedAction("run-deterministic");
      if (!action) return null;
      try {
        guidedRunBtn.textContent = "Searching...";
        const page = await requestInitialEvidencePage(action, "querySpec");
        if (page && guidedActionIsCurrent(action)) {
          publishInitialEvidencePage(action, "querySpec", page);
        }
        return page;
      } finally {
        finishGuidedAction(action);
      }
    }

    if (!intentToken || guidedAuditId === null) {
      throw new Error("no safe evidence search plan is ready to run");
    }
    if (!["unreviewed", "accepted"].includes(guidedReviewStatus)) {
      throw new Error(`AI-assisted interpretation was ${guidedReviewStatus || "not reviewable"} and cannot be run`);
    }
    guidedIntentToken = intentToken;
    const action = beginGuidedAction("run");
    if (!action) return null;

    try {
      guidedRunBtn.textContent = "Starting search...";
      // Submitting "Search evidence" is the examiner's acceptance. The backend repeats this
      // transition idempotently before every direct execution and validates the exact token.
      await invoke("accept_guided_query", {
        intentToken: action.intentToken,
        auditId: action.auditId,
      });
      if (!guidedActionIsCurrent(action)) return null;

      setGuidedReviewStatus("accepted");
      guidedRejectBtn.classList.add("hidden");
      guidedRunBtn.textContent = "Searching...";
      const page = await requestInitialEvidencePage(action, "guided");
      if (page && guidedActionIsCurrent(action)) {
        publishInitialEvidencePage(action, "guided", page);
      }
      return page;
    } catch (err) {
      if (guidedActionIsCurrent(action) && guidedReviewStatus !== "accepted") {
        guidedRunBtn.textContent = "Search evidence";
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
      if (!guidedDecisionIsCurrent(action)) return false;
      setGuidedReviewStatus(decision);
      guidedRunBtn.classList.add("hidden");
      guidedRejectBtn.classList.add("hidden");
      return true;
    } finally {
      finishGuidedAction(action);
    }
  }

  async function generateReport(destPath, request) {
    if (!reportExportIsCurrent(request)) return null;
    showProgress("Generating report workbook...", 0);
    try {
      const summary = await invoke("export_report", { destPath, requestId: request.id });
      if (!reportExportIsCurrent(request)) return null;
      renderReportSummary(summary);
      return summary;
    } catch (err) {
      console.error("export_report failed", err);
      throw err;
    }
  }

  // -- import flow --------------------------------------------------------------

  async function pickAndOpenFile() {
    if (sheetLoadInFlight) return null;
    setSourceLoadInFlight(true);
    const previousPath = currentPath;
    const previousSheet = currentSheet;
    let sourceRequest = null;
    try {
      const path = await invoke("plugin:dialog|open", {
        options: {
          multiple: false,
          filters: [{ name: "Tabular files", extensions: ["xlsx", "xls", "xlsb", "ods", "csv"] }],
        },
      });
      if (!path) {
        setSourceLoadInFlight(false);
        return null;
      }

      sourceRequest = {
        id: ++sourceLoadSequence,
        path,
        previousPath,
        previousSheet,
      };
      activeSourceLoad = sourceRequest;
      sheetPicker.classList.add("hidden");
      hideProgress();
      const sheets = await invoke("list_sheets", { path });
      if (activeSourceLoad !== sourceRequest) return null;

      if (sheets.length === 1) {
        return await loadSheet(sheets[0], sourceRequest);
      }

      sheetSelect.innerHTML = "";
      sheets.forEach((name) => {
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = name;
        sheetSelect.appendChild(opt);
      });
      sheetPicker.classList.remove("hidden");
      setSourceLoadInFlight(false);
      return sheets;
    } catch (err) {
      if (sourceRequest && activeSourceLoad !== sourceRequest) return null;
      if (sourceRequest && activeSourceLoad === sourceRequest) {
        currentPath = sourceRequest.previousPath;
        currentSheet = sourceRequest.previousSheet;
        activeSourceLoad = null;
      }
      setSourceLoadInFlight(false);
      alert(`Could not read workbook: ${err}`);
      return null;
    }
  }

  async function loadSheet(sheet, sourceRequest = activeSourceLoad) {
    if (!sourceRequest || sourceRequest !== activeSourceLoad || !sheet) return null;
    const importRequest = {
      id: ++sourceLoadSequence,
      sourceRequest,
      path: sourceRequest.path,
      sheet,
    };
    activeSheetImport = importRequest;
    setSourceLoadInFlight(true);
    resetGuidedQueryUi();
    sheetPicker.classList.add("hidden");
    currentPath = importRequest.path;
    currentSheet = sheet;
    showProgress(`Reading "${sheet}"…`, 0);

    try {
      const summary = await invoke("import_sheet", { path: importRequest.path, sheet });
      if (
        activeSheetImport !== importRequest ||
        activeSourceLoad !== sourceRequest ||
        currentPath !== importRequest.path ||
        currentSheet !== sheet
      ) {
        return null;
      }
      activeSheetImport = null;
      activeSourceLoad = null;
      setSourceLoadInFlight(false);
      hideProgress();
      onImportComplete(summary, importRequest.path, sheet);
      return summary;
    } catch (err) {
      if (activeSheetImport !== importRequest) return null;
      activeSheetImport = null;
      activeSourceLoad = null;
      setSourceLoadInFlight(false);
      hideProgress();
      // import_sheet clears the backend's prior loaded state before it starts. A failed import
      // therefore cannot safely restore the old table UI; clear it so frontend and backend agree.
      await removeFile();
      alert(`Import failed: ${err}`);
      throw err;
    }
  }

  function onImportComplete(summary, importedPath, importedSheet) {
    currentPath = importedPath;
    currentSheet = importedSheet;
    columns = summary.columns;
    fileInfo.textContent = `${importedPath.split(/[\\/]/).pop()} — ${summary.rowCount.toLocaleString()} rows, ${columns.length} columns${summary.fromCache ? " (cached)" : ""}`;

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

    spec = { search: null, filters: [], sort: null, expression: null, cursor: null, limit: PAGE_SIZE };
    resetPagination();

    const isWideGrid = columns.length > WIDE_GRID_COLUMN_THRESHOLD;
    const tabulatorColumns = [
      { title: "#", field: "row_num", width: 70, headerSort: false, frozen: true },
      {
        title: "Why matched",
        field: "__aiMatch",
        width: 250,
        minWidth: 180,
        headerSort: false,
        visible: false,
        formatter(cell) {
          const rawReasons = cell.getValue();
          const reasons = Array.isArray(rawReasons)
            ? rawReasons.filter((reason) => typeof reason === "string" && reason.trim())
            : [];
          const element = document.createElement("div");
          element.className = "ai-match-reason";
          if (reasons.length === 0) {
            element.textContent = "AI search plan match";
            return element;
          }
          element.textContent = reasons.length > 1 ? `${reasons[0]} (+${reasons.length - 1})` : reasons[0];
          element.title = reasons.join("\n");
          return element;
        },
      },
      ...columns.map((c) => ({
        title: c.originalName,
        field: c.sqlName,
        headerSort: false,
        resizable: true,
        ...(isWideGrid ? { width: 160 } : {}),
      })),
    ];

    if (table) {
      table.destroy();
    }
    table = new Tabulator("#grid", {
      data: [],
      columns: tabulatorColumns,
      // "fitDataFill" measures every rendered cell of every column to size columns to content
      // (Tabulator's reinitializeWidth()/fitToData()) - on a very wide file that's real,
      // synchronous, per-cell DOM measurement work repeated on every page turn. "fitColumns"
      // instead sizes columns from fixed width/grow/shrink config with no content measurement,
      // and "virtual" horizontal rendering avoids building DOM cells for off-screen columns.
      // Below the threshold this is unchanged from before (fitDataFill + basic rendering).
      layout: isWideGrid ? "fitColumns" : "fitDataFill",
      renderHorizontal: isWideGrid ? "virtual" : "basic",
      height: "100%",
      placeholder: "No matching rows",
    });

    setControlsEnabled(true);
    // Semantic preparation is independent of optional mapping. Start both immediately so a
    // slow or failed role detector can never delay AI recall.
    startSemanticIndexForLoadedFile().catch((err) =>
      console.error("semantic index preparation failed", err)
    );
    detectColumnRolesForLoadedFile().catch((err) =>
      console.error("data mapping preparation failed", err)
    );
    loadIgnoreRules().catch((err) => console.error("ignore rules load failed", err));
    const builtTable = table;
    const tableContext = {
      contextRevision: guidedContextRevision,
      path: importedPath,
      sheet: importedSheet,
    };
    table.on("tableBuilt", () => {
      if (table !== builtTable || !loadedContextIsCurrent(tableContext) || sheetLoadInFlight) return;
      refreshData();
      refreshCount();
    });
  }

  function removeFile() {
    sourceLoadSequence += 1;
    activeSourceLoad = null;
    activeSheetImport = null;
    setSourceLoadInFlight(false);
    sheetPicker.classList.add("hidden");
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
    spec = { search: null, filters: [], sort: null, expression: null, cursor: null, limit: PAGE_SIZE };
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
    if (sheetLoadInFlight || !controlsEnabled || tableTransitionInFlight()) return;
    if (
      format === "csv" &&
      !window.confirm(
        "CSV preserves raw cell text. Spreadsheet programs may interpret values beginning with =, +, - or @ as formulas. Export Excel is safer for opening in a spreadsheet. Continue with raw CSV?"
      )
    ) {
      return;
    }
    const tableIdentityBeforeDialog = {
      contextRevision: guidedContextRevision,
      path: currentPath,
      sheet: currentSheet,
      table,
      mode: queryMode,
      evidenceQuery: activeEvidenceQuery,
      spec: JSON.stringify(
        queryMode === "querySpec"
          ? { ...snapshotQuerySpec(), cursor: null }
          : buildSpecFromControls(true)
      ),
    };
    const ext = format === "csv" ? "csv" : "xlsx";
    const destPath = await invoke("plugin:dialog|save", {
      options: {
        filters: [{ name: format.toUpperCase(), extensions: [ext] }],
        defaultPath: `log-parser-export.${ext}`,
      },
    });
    const currentExportSpec = JSON.stringify(
      queryMode === "querySpec"
        ? { ...snapshotQuerySpec(), cursor: null }
        : buildSpecFromControls(true)
    );
    if (
      !destPath ||
      tableTransitionInFlight() ||
      guidedContextRevision !== tableIdentityBeforeDialog.contextRevision ||
      currentPath !== tableIdentityBeforeDialog.path ||
      currentSheet !== tableIdentityBeforeDialog.sheet ||
      table !== tableIdentityBeforeDialog.table ||
      queryMode !== tableIdentityBeforeDialog.mode ||
      activeEvidenceQuery !== tableIdentityBeforeDialog.evidenceQuery ||
      currentExportSpec !== tableIdentityBeforeDialog.spec
    ) return;

    const modeAtStart = queryMode;
    const evidenceQuery = activeEvidenceQuery;
    const context = {
      contextRevision: guidedContextRevision,
      path: currentPath,
      sheet: currentSheet,
      table,
    };
    const exportSpec =
      modeAtStart === "querySpec"
        ? { ...snapshotQuerySpec(), cursor: null }
        : buildSpecFromControls(true);
    showProgress(`Exporting to ${ext.toUpperCase()}…`, 0);
    try {
      const result =
        modeAtStart === "guided"
          ? await invoke("export_guided_data", {
              intentToken: evidenceQuery?.intentToken,
              auditId: evidenceQuery?.auditId,
              format,
              destPath,
            })
          : await invoke("export_data", { spec: exportSpec, format, destPath });
      if (
        tableTransitionInFlight() ||
        !loadedContextIsCurrent(context) ||
        table !== context.table ||
        queryMode !== modeAtStart ||
        (["guided", "querySpec"].includes(modeAtStart) && activeEvidenceQuery !== evidenceQuery)
      ) return;
      hideProgress();
      alert(`Exported ${result.rowCount.toLocaleString()} rows to ${result.destPath}`);
    } catch (err) {
      if (
        tableTransitionInFlight() ||
        !loadedContextIsCurrent(context) ||
        table !== context.table ||
        queryMode !== modeAtStart ||
        (["guided", "querySpec"].includes(modeAtStart) && activeEvidenceQuery !== evidenceQuery)
      ) return;
      hideProgress();
      alert(`Export failed: ${err}`);
    }
  }

  async function doReportExport() {
    if (
      sheetLoadInFlight ||
      !controlsEnabled ||
      activeReportExport !== null ||
      tableTransitionInFlight()
    ) return;
    const request = {
      id: ++reportExportSequence,
      path: currentPath,
      sheet: currentSheet,
    };
    activeReportExport = request;
    updateReportExportButton();
    updateGuidedInteractionControls();

    try {
      const destPath = await invoke("plugin:dialog|save", {
        options: {
          filters: [{ name: "Excel Workbook", extensions: ["xlsx"] }],
          defaultPath: "log-parser-report.xlsx",
        },
      });
      if (!destPath || !reportExportIsCurrent(request) || tableTransitionInFlight()) return;
      await generateReport(destPath, request);
    } catch (err) {
      if (!reportExportIsCurrent(request)) return;
      alert(`Report export failed: ${err}`);
    } finally {
      const shouldHideProgress = reportExportIsCurrent(request);
      if (activeReportExport === request) {
        activeReportExport = null;
        if (shouldHideProgress) hideProgress();
        updateReportExportButton();
        updateGuidedInteractionControls();
      }
    }
  }

  // -- AI analyst ----------------------------------------------------------------

  const ANALYST_PHASE_LABELS = {
    mapping: "Detecting column roles...",
    timeline: "Normalizing timestamps...",
    "mitre-scan": "Scanning for MITRE-mapped activity and chains...",
    "anomaly-scan": "Running the wide-net anomaly scan...",
    activity: "Classifying every row's activity...",
    compose: "Writing the answer...",
  };

  const ANALYST_STEP_LABELS = {
    data_mapping: "Data mapping",
    timeline: "Timeline",
    mitre_scan: "MITRE scan",
    anomaly_scan: "Anomaly scan",
    activity: "Row-by-row activity",
  };

  function analystRequestIsCurrent(request) {
    return (
      activeAnalystRequest === request &&
      currentPath === request.path &&
      currentSheet === request.sheet
    );
  }

  function hideAnalystPanel() {
    analystPanel.classList.add("hidden");
    analystReportBtn.classList.add("hidden");
  }

  function scrollGridToRow(rowNum) {
    if (!table) return false;
    const target = table
      .getRows()
      .find((row) => row.getData().row_num === rowNum);
    if (!target) return false;
    table.scrollToRow(target, "center", false);
    target.getElement().classList.add("analyst-row-flash");
    setTimeout(() => target.getElement().classList.remove("analyst-row-flash"), 1600);
    return true;
  }

  function renderAnalystAnswer(answer) {
    analystHeadline.textContent = answer.headline || "AI analyst";
    analystSections.innerHTML = "";
    (answer.sections || []).forEach((section) => {
      const heading = document.createElement("h4");
      heading.className = "analyst-section-heading";
      heading.textContent = section.heading;
      analystSections.appendChild(heading);
      (section.lines || []).forEach((line) => {
        const paragraph = document.createElement("p");
        paragraph.className = "analyst-line";
        paragraph.appendChild(document.createTextNode(line.text + " "));
        (line.rows || []).forEach((rowNum) => {
          const chip = document.createElement("button");
          chip.type = "button";
          chip.className = "analyst-row-chip";
          chip.textContent = `row ${rowNum}`;
          chip.title = "Scroll the grid to this source row (when it is on the current page)";
          chip.addEventListener("click", () => {
            if (!scrollGridToRow(rowNum)) {
              aiSearchAvailability.textContent = `Row ${rowNum} is not on the current grid page. Clear filters or page to it; the row number always refers to the imported sheet.`;
              aiSearchAvailability.classList.remove("ready");
            }
          });
          paragraph.appendChild(chip);
        });
        analystSections.appendChild(paragraph);
      });
    });

    const steps = answer.steps || [];
    if (steps.length > 0) {
      analystSteps.textContent = `Pipeline: ${steps
        .map((step) => `${ANALYST_STEP_LABELS[step.step] || step.step} ${step.status} (${step.detail})`)
        .join(" · ")}`;
      analystSteps.classList.remove("hidden");
    } else {
      analystSteps.classList.add("hidden");
    }

    analystStatus.classList.add("hidden");
    analystReportBtn.classList.toggle("hidden", !answer.reportRequested);
    analystPanel.classList.remove("hidden");

    if (answer.scan) {
      renderScanSummary(answer.scan);
    }
  }

  async function askAnalyst(trimmed) {
    const request = {
      id: ++analystRequestSequence,
      path: currentPath,
      sheet: currentSheet,
    };
    activeAnalystRequest = request;
    guidedSearchSubmit.disabled = true;
    analystHeadline.textContent = "AI analyst";
    analystSections.innerHTML = "";
    analystSteps.classList.add("hidden");
    analystReportBtn.classList.add("hidden");
    analystStatus.textContent = "The analyst is looking at the file...";
    analystStatus.classList.remove("hidden");
    analystPanel.classList.remove("hidden");
    try {
      const answer = await invoke("ask_analyst", {
        askText: trimmed,
        requestId: request.id,
      });
      if (!analystRequestIsCurrent(request)) return null;
      return answer;
    } finally {
      if (activeAnalystRequest === request) {
        activeAnalystRequest = null;
      }
      guidedSearchSubmit.disabled = !controlsEnabled;
      analystStatus.classList.add("hidden");
    }
  }

  async function routeAnalystAsk() {
    const trimmed = guidedSearchBox.value.trim();
    if (!trimmed) return;
    if (
      activeAnalystRequest !== null ||
      guidedActiveParse !== null ||
      guidedActiveAction !== null ||
      guidedActiveQuery !== null ||
      activeDataRequest !== null ||
      activeReportExport !== null ||
      sheetLoadInFlight
    ) {
      return;
    }
    let answer;
    try {
      answer = await askAnalyst(trimmed);
    } catch (err) {
      hideAnalystPanel();
      throw err;
    }
    if (!answer) return;
    if (answer.useGuidedSearch) {
      // Filter-shaped asks keep the existing audited preview/run search flow.
      hideAnalystPanel();
      await searchGuidedQuery();
      return;
    }
    renderAnalystAnswer(answer);
    // The pipeline may have added role suggestions and a normalized timeline; refresh the
    // mapping panel so the sidebar reflects what actually ran.
    if ((answer.steps || []).some((step) => step.step === "data_mapping" && step.status === "ran")) {
      detectColumnRolesForLoadedFile().catch((err) =>
        console.error("post-analyst mapping refresh failed", err)
      );
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
    roleReviewPanel.open = true;
  });

  manageIgnoreRulesBtn.addEventListener("click", () => {
    ignoreRulesPanel.classList.remove("hidden");
    ignoreRulesPanel.open = true;
    if (!ignoreRulesLoaded) loadIgnoreRules();
  });

  sheetLoadBtn.addEventListener("click", () => {
    loadSheet(sheetSelect.value).catch((err) => console.error("import_sheet failed", err));
  });

  searchBox.addEventListener("input", debouncedApply);
  guidedSearchBox.addEventListener("input", () => {
    const currentText = guidedSearchBox.value.trim();
    pendingSemanticSearch = null;
    if (guidedActiveParse && currentText !== guidedActiveParse.queryText) {
      cancelActiveGuidedParse();
      guidedQueryPanel.classList.remove("hidden");
      guidedPreviewText.textContent = "The evidence request changed before planning finished.";
    }
    if (guidedPreviewQueryText !== null && currentText !== guidedPreviewQueryText) {
      guidedPreviewQueryText = null;
      guidedRunBtn.classList.add("hidden");
      guidedRejectBtn.classList.add("hidden");
      guidedClarification.textContent = "Request changed. Search again to use the updated wording.";
      guidedClarification.classList.remove("hidden");
      if (guidedAuditId !== null && guidedReviewStatus === "unreviewed") {
        decideGuidedParse("edited").catch((err) =>
          console.error("could not mark the previous AI interpretation as edited", err)
        );
      }
    }
  });
  guidedSearchForm.addEventListener("submit", (event) => {
    event.preventDefault();
    routeAnalystAsk().catch((err) => alert(`AI analyst failed: ${err}`));
  });
  analystPanelClose.addEventListener("click", hideAnalystPanel);
  analystReportBtn.addEventListener("click", () => {
    doReportExport();
  });
  guidedRunBtn.addEventListener("click", () => {
    const retrying = guidedRunBtn.textContent === "Retry search";
    const action = retrying ? searchGuidedQuery() : runGuidedQuery();
    action.catch((err) => alert(`Evidence search failed: ${err}`));
  });
  guidedRejectBtn.addEventListener("click", () => {
    decideGuidedParse("rejected").catch((err) => alert(`Could not record decision: ${err}`));
  });
  guidedResetBtn.addEventListener("click", () => {
    // This only returns to the ordinary table view. The imported dataset is unchanged, so keep
    // semantic indexing, role detection, and timestamp analysis bound to their current revision.
    resetGuidedQueryUi({ invalidateDataset: false });
    aiSearchAvailability.textContent = "Ready to search every imported row. No enrichment scan is required.";
    aiSearchAvailability.classList.add("ready");
    applyControlsAndReload();
  });
  guidedPanelClose.addEventListener("click", () => {
    cancelActiveGuidedParse();
    decideGuidedParse("rejected").catch((err) => console.error("set_guided_parse_decision failed", err));
    guidedQueryPanel.classList.add("hidden");
  });

  addFilterBtn.addEventListener("click", () => {
    if (sheetLoadInFlight || tableTransitionInFlight()) return;
    addFilterRow();
  });
  applyBtn.addEventListener("click", applyControlsAndReload);
  clearBtn.addEventListener("click", () => {
    if (sheetLoadInFlight || tableTransitionInFlight()) return;
    searchBox.value = "";
    filterList.innerHTML = "";
    sortColumn.value = "";
    applyControlsAndReload();
  });

  suspiciousScanBtn.addEventListener("click", () => {
    runIntelScan().catch((err) => alert(`Threat enrichment failed: ${err}`));
  });

  rolePanelClose.addEventListener("click", () => {
    roleReviewPanel.open = false;
  });

  ignoreRulesPanelClose.addEventListener("click", () => {
    ignoreRulesPanel.open = false;
  });

  ignoreRuleTargetType.addEventListener("change", () => {
    const isHeader = ignoreRuleTargetType.value === "header";
    ignoreRuleHeaderInput.classList.toggle("hidden", !isHeader);
    ignoreRuleRoleSelect.classList.toggle("hidden", isHeader);
  });

  addIgnoreRuleForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const name = ignoreRuleNameInput.value.trim();
    const values = ignoreRuleValuesInput.value
      .split(",")
      .map((value) => value.trim())
      .filter(Boolean);
    const isHeader = ignoreRuleTargetType.value === "header";
    const headerAnyOf = isHeader ? [ignoreRuleHeaderInput.value.trim()].filter(Boolean) : [];
    if (!name || !values.length || (isHeader && !headerAnyOf.length)) return;

    const submitBtn = addIgnoreRuleForm.querySelector('button[type="submit"]');
    submitBtn.disabled = true;
    const ok = await addIgnoreRule({
      name,
      role: isHeader ? null : ignoreRuleRoleSelect.value,
      headerAnyOf,
      op: ignoreRuleOpSelect.value,
      values,
    });
    submitBtn.disabled = false;
    if (ok) {
      addIgnoreRuleForm.reset();
      ignoreRuleHeaderInput.classList.add("hidden");
      ignoreRuleRoleSelect.classList.remove("hidden");
    }
  });

  timezoneUtcBtn.addEventListener("click", () => {
    const dateConvention = dateConventionSelect.value || null;
    if (timestampAnalysis?.needsDateConvention && !dateConvention) {
      alert("Choose whether slash dates are month-first or day-first.");
      return;
    }
    normalizeTimestampColumn("UTC", null, dateConvention).catch((err) =>
      alert(`Timestamp normalization failed: ${err}`)
    );
  });
  timezoneNormalizeBtn.addEventListener("click", () => {
    const answer = timezoneInput.value.trim();
    const dateConvention = dateConventionSelect.value || null;
    if (timestampAnalysis?.needsTimezone && !answer) {
      alert("Enter a UTC offset or IANA timezone, or choose Already UTC.");
      return;
    }
    if (timestampAnalysis?.needsDateConvention && !dateConvention) {
      alert("Choose whether slash dates are month-first or day-first.");
      return;
    }
    normalizeTimestampColumn(answer || null, null, dateConvention).catch((err) =>
      alert(`Timestamp normalization failed: ${err}`)
    );
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
    if (sheetLoadInFlight || tableTransitionInFlight()) return;
    if (cursorStack.length === 0) return;
    spec.cursor = cursorStack.pop();
    pageIndex -= 1;
    refreshData();
  });

  nextPageBtn.addEventListener("click", () => {
    if (sheetLoadInFlight || tableTransitionInFlight()) return;
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
        ? "Optional threat enrichment complete"
        : `Enriching threat matches... ${rowsDone.toLocaleString()} / ${rowsTotal.toLocaleString()}`;
    showProgress(label, fraction);
  });

  listen("semantic-index-progress", (event) => {
    if (semanticIndexState.status !== "building" || !activeSemanticIndexRequest) return;
    const {
      buildId,
      rowsDone,
      documentsEmbedded,
      mappingsWritten,
      documentsSkipped,
      mappingsSkipped,
      cellsTruncated,
      columnsOmitted,
      chunksOmitted,
      resumedFromRow,
      phase,
    } = event.payload;
    semanticIndexState = {
      ...semanticIndexState,
      // Progress events are process-global and carry no file generation. Only the scoped
      // command response is authoritative for readiness; a late event from a previous file
      // must never mark the current file ready.
      status: "building",
      phase: typeof phase === "string" ? phase : semanticIndexState.phase,
      buildId: Number.isSafeInteger(buildId) && buildId > 0 ? buildId : semanticIndexState.buildId,
      rowsIndexed: rowsDone || 0,
      documentsEmbedded: documentsEmbedded || 0,
      mappingsWritten: mappingsWritten || 0,
      documentsSkipped: documentsSkipped || 0,
      mappingsSkipped: mappingsSkipped || 0,
      cellsTruncated: cellsTruncated || 0,
      columnsOmitted: columnsOmitted || 0,
      chunksOmitted: chunksOmitted || 0,
      resumedFromRow: resumedFromRow || 0,
    };
    renderSemanticIndexState();
  });

  listen("analyst-progress", (event) => {
    const payload = event?.payload;
    if (!payload || activeAnalystRequest === null || payload.requestId !== activeAnalystRequest.id) {
      return;
    }
    analystStatus.textContent =
      ANALYST_PHASE_LABELS[payload.phase] || `Working: ${payload.phase}...`;
    analystStatus.classList.remove("hidden");
  });

  listen("report-export-progress", (event) => {
    const { requestId, rowsDone, sheet } = event.payload;
    if (
      !activeReportExport ||
      activeReportExport.id !== requestId ||
      !reportExportIsCurrent(activeReportExport)
    ) {
      return;
    }
    const sheetLabel = sheet ? ` (${sheet})` : "";
    showProgress(`Writing report${sheetLabel}... ${rowsDone.toLocaleString()} rows`, 0.5);
  });

  // The role dropdown's option list is static (independent of which file is loaded), so it can
  // be populated once here — the rules themselves are per-file and load after import instead
  // (see the `manageIgnoreRulesBtn`-adjacent call in the import-success handler).
  IGNORE_RULE_ROLES.forEach((role) => {
    const option = document.createElement("option");
    option.value = role;
    option.textContent = formatRoleName(role);
    ignoreRuleRoleSelect.appendChild(option);
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
      const sourceRequest = {
        id: ++sourceLoadSequence,
        path,
        previousPath: currentPath,
        previousSheet: currentSheet,
      };
      activeSourceLoad = sourceRequest;
      currentPath = path;
      currentSheet = null;
      return loadSheet(sheet, sourceRequest);
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
    async askAnalystForTest(text) {
      if (!text) {
        throw new Error("askAnalystForTest(text): text is required");
      }
      const answer = await askAnalyst(String(text));
      if (answer && !answer.useGuidedSearch) renderAnalystAnswer(answer);
      return answer;
    },
    getIntelState() {
      return {
        columnRoleSuggestions,
        timestampAnalysis,
        timestampNormalizationSummary,
        evidenceColumns: confirmedEvidenceColumns(),
        inferredEvidenceColumns: inferredEvidenceColumns(),
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
        guidedQuerySpec,
        guidedMatchExplanation,
        guidedPreviewQueryText,
        parseInFlight: guidedActiveParse !== null,
        actionInFlight: guidedActiveAction ? guidedActiveAction.type : null,
        queryInFlight: guidedActiveQuery !== null,
        dataRequestInFlight: activeDataRequest !== null,
        countRequestInFlight: activeCountRequest !== null,
        sourceLoadInFlight: sheetLoadInFlight,
        aiMatchColumnVisible: Boolean(table && table.getColumn("__aiMatch")?.isVisible()),
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
    normalizeTimestampForTest(naiveTimezone = null, dateConvention = null) {
      return normalizeTimestampColumn(naiveTimezone, null, dateConvention);
    },
    scanIntelForTest(evidenceColumns = inferredEvidenceColumns()) {
      return runIntelScan(evidenceColumns);
    },
    previewGuidedQueryForTest(queryText) {
      guidedSearchBox.value = queryText;
      return previewGuidedQuery(queryText);
    },
    runGuidedQueryForTest(intentToken = guidedIntentToken) {
      return runGuidedQuery(intentToken);
    },
    previewAiEvidenceQueryForTest(queryText) {
      guidedSearchBox.value = queryText;
      return previewGuidedQuery(queryText);
    },
    searchAiEvidenceForTest(queryText) {
      guidedSearchBox.value = queryText;
      return searchGuidedQuery(queryText);
    },
    runAiEvidenceQueryForTest(intentToken = guidedIntentToken) {
      return runGuidedQuery(intentToken);
    },
    getAiSearchState() {
      return window.__logParserDebug.getGuidedState();
    },
    getSemanticIndexState() {
      return { ...semanticIndexState, inFlight: activeSemanticIndexRequest !== null };
    },
    buildSemanticIndexForTest() {
      return startSemanticIndexForLoadedFile();
    },
    openDataMappingForTest() {
      roleReviewPanel.classList.remove("hidden");
      roleReviewPanel.open = true;
    },
    setMappingForTest(role, sqlName, status = "confirmed") {
      return setColumnRoleStatus(role, sqlName, status);
    },
    openIgnoreRulesForTest() {
      ignoreRulesPanel.classList.remove("hidden");
      ignoreRulesPanel.open = true;
      return loadIgnoreRules();
    },
    getIgnoreRulesForTest() {
      return ignoreRules;
    },
    addIgnoreRuleForTest(input) {
      return addIgnoreRule(input);
    },
    setIgnoreRuleEnabledForTest(ruleId, enabled) {
      return setIgnoreRuleEnabled(ruleId, enabled);
    },
    deleteIgnoreRuleForTest(ruleId) {
      return deleteIgnoreRule(ruleId);
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
