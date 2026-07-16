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

  let columns = []; // ColumnMeta[] from ImportSummary
  let table = null;
  let currentPath = null;
  let currentSheet = null;

  let spec = { search: null, filters: [], sort: null, expression: null, cursor: null, limit: PAGE_SIZE };
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
  let timestampAnalysis = null;
  let timestampNormalizationSummary = null;
  let intelScanSummaryResult = null;
  let reportSummaryResult = null;
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
  let semanticIndexState = { status: "idle", rowsIndexed: 0, summary: null, error: null };
  let semanticIndexRequestSequence = 0;
  let activeSemanticIndexRequest = null;
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
    reportExportBtn.disabled = !enabled;
    exportCsvBtn.disabled = !enabled;
    exportXlsxBtn.disabled = !enabled;
    addFilterBtn.disabled = !enabled;
    applyBtn.disabled = !enabled;
    clearBtn.disabled = !enabled;
    reviewRolesBtn.disabled = !enabled;
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
    reportExportBtn.disabled = inFlight || !controlsEnabled;
    exportCsvBtn.disabled = inFlight || !controlsEnabled;
    exportXlsxBtn.disabled = inFlight || !controlsEnabled;
    addFilterBtn.disabled = inFlight || !controlsEnabled;
    applyBtn.disabled = inFlight || !controlsEnabled;
    clearBtn.disabled = inFlight || !controlsEnabled;
    reviewRolesBtn.disabled = inFlight || !controlsEnabled;
    suspiciousScanBtn.disabled = inFlight || !controlsEnabled;
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

  function updateGuidedInteractionControls() {
    const parsing = guidedActiveParse !== null;
    const actionInFlight = guidedActiveAction !== null;
    const queryInFlight = guidedActiveQuery !== null;
    guidedSearchBox.disabled =
      !controlsEnabled ||
      columns.length === 0 ||
      sheetLoadInFlight ||
      parsing ||
      actionInFlight ||
      queryInFlight;
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
    cancelSearchDebounce();
    invalidateGuidedContext();
    queryMode = "normal";
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
    semanticIndexState = { status: "idle", rowsIndexed: 0, summary: null, error: null };
    semanticIndexStatus.className = "semantic-index-status";
    semanticIndexStatus.textContent = "Semantic matching starts automatically after import.";
    intelScanInFlight = false;
    activeDataRequest = null;
    activeCountRequest = null;

    roleList.innerHTML = "";
    rolePanelStatus.textContent = "";
    roleReviewPanel.classList.add("hidden");
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
      querySpec: guidedQuerySpec,
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

      const columnSelect = document.createElement("select");
      columnSelect.className = "mapping-column-select";
      columnSelect.setAttribute("aria-label", `Column mapped to ${formatRoleName(role)}`);
      const emptyOption = document.createElement("option");
      emptyOption.value = "";
      emptyOption.textContent = "(not mapped)";
      columnSelect.appendChild(emptyOption);
      columns.forEach((candidate) => {
        const option = document.createElement("option");
        option.value = candidate.sqlName;
        option.textContent = candidate.originalName;
        columnSelect.appendChild(option);
      });
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

  function renderGuidedPreview(result, queryText) {
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

    guidedRunBtn.textContent = "Search evidence";
    guidedQueryPanel.classList.remove("hidden");
    const previewLines = [result.previewText || "No search plan was returned."];
    if (guidedMatchExplanation.length > 0) {
      previewLines.push("", "Why rows will match:", ...guidedMatchExplanation.map((item) => `\u2022 ${item}`));
    }
    guidedPreviewText.textContent = previewLines.join("\n");

    if (result.aiAssisted) {
      const validation = result.validationStatus ? ` \u2022 ${result.validationStatus.replace(/_/g, " ")}` : "";
      guidedAiStatus.textContent = `Offline AI interpretation \u2022 ${guidedReviewStatus || "unreviewed"}${validation}`;
      guidedAiStatus.classList.remove("hidden");
      guidedRejectBtn.classList.toggle("hidden", guidedReviewStatus !== "unreviewed");
    } else {
      guidedAiStatus.textContent = "Deterministic local search plan \u2022 no model inference";
      guidedAiStatus.classList.remove("hidden");
      guidedRejectBtn.classList.add("hidden");
    }

    const auditedPlanReady =
      result.aiAssisted &&
      guidedIntentToken !== null &&
      guidedAuditId !== null &&
      ["unreviewed", "accepted"].includes(guidedReviewStatus);
    const deterministicPlanReady = !result.aiAssisted && guidedAuditId === null && guidedQuerySpec !== null;
    if (result.needsClarification || (!auditedPlanReady && !deterministicPlanReady)) {
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
      guidedRunBtn.classList.remove("hidden");
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
    if (queryMode === "guided") {
      activeCountRequest = null;
      countRequestSequence += 1;
      totalCount = null;
      updateRowCountLabel();
      return;
    }

    const request = {
      id: ++countRequestSequence,
      contextRevision: guidedContextRevision,
      path: currentPath,
      sheet: currentSheet,
      mode: queryMode,
      table,
      querySpec: guidedQuerySpec,
      spec: snapshotQuerySpec(),
    };
    activeCountRequest = request;
    totalCount = null;
    updateRowCountLabel();
    const isCurrent = () =>
      activeCountRequest === request &&
      loadedContextIsCurrent(request) &&
      queryMode === request.mode &&
      table === request.table &&
      (request.mode === "normal" || guidedQuerySpec === request.querySpec);
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
      rowCountLabel.textContent = `${totalCount.toLocaleString()} ${queryMode === "querySpec" ? "evidence" : "matching"} rows`;
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
    // Disabled synchronously (before the first await below) so a rapid double-click on
    // Prev/Next can't fire a second query_rows() while this one is still in flight and read a
    // stale nextCursor — the button is unclickable for the whole round trip either way.
    const modeAtStart = queryMode;
    const isGuidedRequest = modeAtStart === "guided";
    const isQuerySpecRequest = modeAtStart === "querySpec";
    const isTrackedEvidenceRequest = isGuidedRequest || isQuerySpecRequest;
    if (isTrackedEvidenceRequest && guidedActiveQuery !== null) return null;
    setAiMatchColumnVisible(isTrackedEvidenceRequest);

    const request = {
      id: ++dataRequestSequence,
      contextRevision: guidedContextRevision,
      path: currentPath,
      sheet: currentSheet,
      mode: modeAtStart,
      auditId: guidedAuditId,
      intentToken: guidedIntentToken,
      querySpec: guidedQuerySpec,
      cursor: spec.cursor,
      limit: spec.limit,
      spec: isGuidedRequest ? null : snapshotQuerySpec(),
      table,
    };
    activeDataRequest = request;
    if (isTrackedEvidenceRequest) {
      guidedActiveQuery = request;
      updateGuidedInteractionControls();
    }

    const requestIsCurrent = () =>
      activeDataRequest === request &&
      loadedContextIsCurrent(request) &&
      queryMode === request.mode &&
      (!isTrackedEvidenceRequest ||
        (guidedAuditId === request.auditId &&
          guidedIntentToken === request.intentToken &&
          guidedQuerySpec === request.querySpec)) &&
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
      if (activeDataRequest === request) activeDataRequest = null;
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

  function applyControlsAndReload() {
    cancelSearchDebounce();
    queryMode = "normal";
    setAiMatchColumnVisible(false);
    guidedResetBtn.classList.add("hidden");
    spec.search = searchBox.value.trim() || null;
    spec.filters = currentFilterValues();
    spec.sort = sortColumn.value
      ? { column: sortColumn.value, direction: sortDirection.value }
      : null;
    spec.expression = null;
    resetPagination();
    refreshData();
    refreshCount();
  }

  let searchDebounceHandle = null;
  function cancelSearchDebounce() {
    if (searchDebounceHandle) clearTimeout(searchDebounceHandle);
    searchDebounceHandle = null;
  }

  function debouncedApply() {
    cancelSearchDebounce();
    const request = {
      contextRevision: guidedContextRevision,
      path: currentPath,
      sheet: currentSheet,
    };
    searchDebounceHandle = setTimeout(() => {
      searchDebounceHandle = null;
      if (loadedContextIsCurrent(request) && controlsEnabled && !sheetLoadInFlight) {
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

  function semanticIndexRequestIsCurrent(request) {
    return activeSemanticIndexRequest === request && loadedContextIsCurrent(request);
  }

  function renderSemanticIndexState() {
    semanticIndexStatus.className = `semantic-index-status ${semanticIndexState.status}`;
    if (semanticIndexState.status === "ready") {
      semanticIndexStatus.textContent = `Semantic matching ready (${semanticIndexState.rowsIndexed.toLocaleString()} rows indexed).`;
    } else if (semanticIndexState.status === "building") {
      const progress = semanticIndexState.rowsIndexed
        ? ` ${semanticIndexState.rowsIndexed.toLocaleString()} rows processed.`
        : "";
      semanticIndexStatus.textContent = `Semantic matching is building in the background.${progress} Exact and structured AI search are ready now.`;
    } else if (semanticIndexState.status === "error") {
      semanticIndexStatus.textContent = "Semantic matching is unavailable; exact and structured AI search remain ready.";
    } else {
      semanticIndexStatus.textContent = "Semantic matching starts automatically after import.";
    }
  }

  function requireFreshPreviewForSemanticIndex() {
    if (
      !guidedParseResult ||
      guidedPreviewQueryText === null ||
      guidedReviewStatus !== "unreviewed" ||
      ["guided", "querySpec"].includes(queryMode)
    ) {
      return;
    }
    guidedPreviewQueryText = null;
    guidedRunBtn.classList.add("hidden");
    guidedRejectBtn.classList.add("hidden");
    guidedClarification.textContent = "Semantic matching is now ready. Preview again to include semantic evidence candidates.";
    guidedClarification.classList.remove("hidden");
    decideGuidedParse("edited").catch((err) =>
      console.error("could not retire the pre-semantic AI interpretation", err)
    );
  }

  async function startSemanticIndexForLoadedFile() {
    const request = {
      id: ++semanticIndexRequestSequence,
      contextRevision: guidedContextRevision,
      path: currentPath,
      sheet: currentSheet,
    };
    activeSemanticIndexRequest = request;
    semanticIndexState = { status: "building", rowsIndexed: 0, summary: null, error: null };
    renderSemanticIndexState();
    try {
      const status = await invoke("semantic_index_status");
      if (!semanticIndexRequestIsCurrent(request)) return null;
      if (status.ready) {
        semanticIndexState = {
          status: "ready",
          rowsIndexed: status.rowsIndexed || 0,
          summary: null,
          error: null,
        };
        renderSemanticIndexState();
        requireFreshPreviewForSemanticIndex();
        return status;
      }

      const summary = await invoke("build_semantic_index");
      if (!semanticIndexRequestIsCurrent(request)) return null;
      semanticIndexState = {
        status: "ready",
        rowsIndexed: summary.rowsIndexed || 0,
        summary,
        error: null,
      };
      renderSemanticIndexState();
      requireFreshPreviewForSemanticIndex();
      return summary;
    } catch (err) {
      if (!semanticIndexRequestIsCurrent(request)) return null;
      console.error("semantic index build failed", err);
      semanticIndexState = { status: "error", rowsIndexed: 0, summary: null, error: String(err) };
      renderSemanticIndexState();
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

  async function previewGuidedQuery(queryText = guidedSearchBox.value) {
    const trimmed = queryText.trim();
    if (!trimmed) return null;
    if (guidedActiveAction !== null || guidedActiveQuery !== null || sheetLoadInFlight) return null;
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
    guidedQueryPanel.classList.remove("hidden");
    guidedPreviewText.textContent = "Building an evidence search plan...";
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
      renderGuidedPreview(result, request.queryText);
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
        queryMode = "querySpec";
        spec = { ...guidedQuerySpec, cursor: null, limit: PAGE_SIZE };
        totalCount = null;
        resetPagination();
        guidedResetBtn.classList.remove("hidden");
        guidedRunBtn.textContent = "Searching...";
        const page = await refreshData();
        if (guidedActionIsCurrent(action)) {
          guidedRunBtn.textContent = page ? "Search evidence" : "Retry";
          guidedRunBtn.classList.remove("hidden");
          refreshCount();
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
      guidedRunBtn.textContent = "Accepting plan...";
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
      guidedRunBtn.textContent = "Searching...";

      const page = await refreshData();
      if (guidedActionIsCurrent(action)) {
        guidedRunBtn.textContent = page ? "Search evidence" : "Retry";
        guidedRunBtn.classList.remove("hidden");
        refreshCount();
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
    // Semantic preparation is independent of optional mapping. Start both immediately so a
    // slow or failed role detector can never delay AI recall.
    startSemanticIndexForLoadedFile().catch((err) =>
      console.error("semantic index preparation failed", err)
    );
    detectColumnRolesForLoadedFile().catch((err) =>
      console.error("data mapping preparation failed", err)
    );
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
    if (sheetLoadInFlight || !controlsEnabled) return;
    if (
      format === "csv" &&
      !window.confirm(
        "CSV preserves raw cell text. Spreadsheet programs may interpret values beginning with =, +, - or @ as formulas. Export Excel is safer for opening in a spreadsheet. Continue with raw CSV?"
      )
    ) {
      return;
    }
    const ext = format === "csv" ? "csv" : "xlsx";
    const destPath = await invoke("plugin:dialog|save", {
      options: {
        filters: [{ name: format.toUpperCase(), extensions: [ext] }],
        defaultPath: `log-parser-export.${ext}`,
      },
    });
    if (!destPath) return;

    const modeAtStart = queryMode;
    const context = {
      contextRevision: guidedContextRevision,
      path: currentPath,
      sheet: currentSheet,
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
              intentToken: guidedIntentToken,
              auditId: guidedAuditId,
              format,
              destPath,
            })
          : await invoke("export_data", { spec: exportSpec, format, destPath });
      if (!loadedContextIsCurrent(context) || queryMode !== modeAtStart) return;
      hideProgress();
      alert(`Exported ${result.rowCount.toLocaleString()} rows to ${result.destPath}`);
    } catch (err) {
      if (!loadedContextIsCurrent(context) || queryMode !== modeAtStart) return;
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
    roleReviewPanel.open = true;
  });

  sheetLoadBtn.addEventListener("click", () => {
    loadSheet(sheetSelect.value).catch((err) => console.error("import_sheet failed", err));
  });

  searchBox.addEventListener("input", debouncedApply);
  guidedSearchBox.addEventListener("input", () => {
    const currentText = guidedSearchBox.value.trim();
    if (guidedActiveParse && currentText !== guidedActiveParse.queryText) {
      cancelActiveGuidedParse();
      guidedQueryPanel.classList.remove("hidden");
      guidedPreviewText.textContent = "The evidence request changed before planning finished.";
    }
    if (guidedPreviewQueryText !== null && currentText !== guidedPreviewQueryText) {
      guidedPreviewQueryText = null;
      guidedRunBtn.classList.add("hidden");
      guidedRejectBtn.classList.add("hidden");
      guidedClarification.textContent = "Request changed. Preview the updated search before running it.";
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
    previewGuidedQuery().catch((err) => alert(`Evidence search preview failed: ${err}`));
  });
  guidedRunBtn.addEventListener("click", () => {
    runGuidedQuery().catch((err) => alert(`Evidence search failed: ${err}`));
  });
  guidedRejectBtn.addEventListener("click", () => {
    decideGuidedParse("rejected").catch((err) => alert(`Could not record decision: ${err}`));
  });
  guidedResetBtn.addEventListener("click", () => {
    applyControlsAndReload();
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
    runIntelScan().catch((err) => alert(`Threat enrichment failed: ${err}`));
  });

  rolePanelClose.addEventListener("click", () => {
    roleReviewPanel.open = false;
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
        ? "Optional threat enrichment complete"
        : `Enriching threat matches... ${rowsDone.toLocaleString()} / ${rowsTotal.toLocaleString()}`;
    showProgress(label, fraction);
  });

  listen("semantic-index-progress", (event) => {
    if (semanticIndexState.status !== "building" || !activeSemanticIndexRequest) return;
    const { rowsDone } = event.payload;
    semanticIndexState = {
      ...semanticIndexState,
      // Progress events are process-global and carry no file generation. Only the scoped
      // command response is authoritative for readiness; a late event from a previous file
      // must never mark the current file ready.
      status: "building",
      rowsIndexed: rowsDone || 0,
    };
    renderSemanticIndexState();
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
