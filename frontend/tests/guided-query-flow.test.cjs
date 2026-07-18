"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const APP_PATH = path.resolve(__dirname, "..", "app.js");
const APP_SOURCE = fs.readFileSync(APP_PATH, "utf8");

class FakeClassList {
  constructor(element) {
    this.element = element;
  }

  add(...names) {
    names.forEach((name) => this.element._classes.add(name));
  }

  remove(...names) {
    names.forEach((name) => this.element._classes.delete(name));
  }

  contains(name) {
    return this.element._classes.has(name);
  }

  toggle(name, force) {
    if (force === true) {
      this.add(name);
      return true;
    }
    if (force === false) {
      this.remove(name);
      return false;
    }
    if (this.contains(name)) {
      this.remove(name);
      return false;
    }
    this.add(name);
    return true;
  }
}

class FakeElement {
  constructor(tagName = "div", id = "") {
    this.tagName = tagName.toUpperCase();
    this.id = id;
    this._classes = new Set();
    this.classList = new FakeClassList(this);
    this._innerHTML = "";
    this.textContent = "";
    this.value = "";
    this.disabled = false;
    this.open = false;
    this.title = "";
    this.style = {};
    this.dataset = {};
    this.attributes = new Map();
    this.children = [];
    this.parentElement = null;
    this.isConnected = true;
    this.listeners = new Map();
    if (this.tagName === "TEMPLATE") {
      this.content = new FakeElement("fragment");
    }
  }

  get className() {
    return [...this._classes].join(" ");
  }

  set className(value) {
    this._classes = new Set(String(value || "").split(/\s+/).filter(Boolean));
  }

  get innerHTML() {
    return this._innerHTML;
  }

  set innerHTML(value) {
    this._innerHTML = String(value);
    this.children = [];
  }

  get firstElementChild() {
    return this.children[0] || null;
  }

  setAttribute(name, value) {
    this.attributes.set(name, String(value));
  }

  removeAttribute(name) {
    this.attributes.delete(name);
  }

  addEventListener(type, listener) {
    const listeners = this.listeners.get(type) || [];
    listeners.push(listener);
    this.listeners.set(type, listeners);
  }

  dispatchEvent(event) {
    event.target = event.target || this;
    for (const listener of this.listeners.get(event.type) || []) {
      listener.call(this, event);
    }
    return true;
  }

  appendChild(child) {
    if (child == null) return child;
    child.parentElement = this;
    child.isConnected = this.isConnected;
    this.children.push(child);
    return child;
  }

  append(...children) {
    children.forEach((child) => this.appendChild(child));
  }

  cloneNode(deep = false) {
    const clone = new FakeElement(this.tagName, this.id);
    clone.className = this.className;
    clone.textContent = this.textContent;
    clone.value = this.value;
    clone.disabled = this.disabled;
    clone._innerHTML = this._innerHTML;
    if (deep) this.children.forEach((child) => clone.appendChild(child.cloneNode(true)));
    return clone;
  }

  querySelector(selector) {
    return this.querySelectorAll(selector)[0] || null;
  }

  querySelectorAll(selector) {
    const matches = [];
    const visit = (element) => {
      for (const child of element.children) {
        const matchesClass = selector.startsWith(".") && child.classList.contains(selector.slice(1));
        const matchesId = selector.startsWith("#") && child.id === selector.slice(1);
        const matchesTag = !selector.startsWith(".") && !selector.startsWith("#") &&
          child.tagName.toLowerCase() === selector.toLowerCase();
        if (matchesClass || matchesId || matchesTag) matches.push(child);
        visit(child);
      }
    };
    visit(this);
    return matches;
  }
}

class FakeDocument {
  constructor() {
    this.elements = new Map();
  }

  getElementById(id) {
    if (!this.elements.has(id)) {
      const tagName = id === "filter-row-template" ? "template" : "div";
      this.elements.set(id, new FakeElement(tagName, id));
    }
    return this.elements.get(id);
  }

  createElement(tagName) {
    return new FakeElement(tagName);
  }
}

class FakeTabulatorColumn {
  constructor(visible) {
    this.visible = visible;
  }

  show() {
    this.visible = true;
  }

  hide() {
    this.visible = false;
  }

  isVisible() {
    return this.visible;
  }
}

class FakeTabulator {
  constructor(_selector, options) {
    this.data = [...(options.data || [])];
    this.destroyed = false;
    this.columns = new Map(
      (options.columns || []).map((column) => [
        column.field,
        new FakeTabulatorColumn(column.visible !== false),
      ])
    );
  }

  on(eventName, listener) {
    if (eventName === "tableBuilt") {
      queueMicrotask(() => {
        if (!this.destroyed) listener();
      });
    }
  }

  async replaceData(rows) {
    this.data = [...rows];
  }

  getData() {
    return this.data;
  }

  getDataCount() {
    return this.data.length;
  }

  getColumn(field) {
    return this.columns.get(field) || null;
  }

  destroy() {
    this.destroyed = true;
  }
}

const BASELINE_ROW = { row_num: 1, commandline: "cmd.exe /c echo baseline" };
const EVIDENCE_ROW = {
  row_num: 9,
  commandline: "powershell.exe -enc AAAA",
  __aiMatch: ["CommandLine contains powershell"],
};
const EXACT_ROW = {
  row_num: 10,
  commandline: "powershell.exe exact lexical match",
  __aiMatch: ["Exact match for powershell"],
};
const SEMANTIC_ROW = {
  row_num: 11,
  commandline: "encoded script interpreter activity",
  __aiMatch: ["Semantically similar to PowerShell activity"],
};
const REPLACEMENT_ROW = {
  row_num: 12,
  commandline: "rundll32.exe javascript:replacement",
  __aiMatch: ["CommandLine contains rundll32"],
};
const NORMAL_FILTER_ROW = {
  row_num: 13,
  commandline: "ordinary table filter result",
};

function page(rows) {
  return { rows, nextCursor: null, hasMore: false };
}

function validAiPreview(overrides = {}) {
  return {
    intentToken: '{"intent":"rawEvidenceSearch","terms":["powershell"]}',
    previewText: "Search every imported row for PowerShell evidence.",
    needsClarification: false,
    clarificationMessage: null,
    aiAssisted: true,
    auditId: 7,
    reviewStatus: "unreviewed",
    validationStatus: "validated",
    querySpec: {
      search: "powershell",
      filters: [],
      expression: null,
      sort: null,
      cursor: null,
      limit: 300,
    },
    matchExplanation: ["Any searchable column contains powershell."],
    ...overrides,
  };
}

function clarificationPreview() {
  return {
    intentToken: '{"intent":"unknown"}',
    previewText: "No evidence search was run.",
    needsClarification: true,
    clarificationMessage: "Say which activity, account, host, or value to find.",
    aiAssisted: false,
    auditId: null,
    reviewStatus: "not_applicable",
    validationStatus: null,
    querySpec: null,
    matchExplanation: [],
  };
}

function deterministicPreview() {
  return {
    intentToken: '{"intent":"rawEvidenceSearch","terms":["powershell"]}',
    previewText: "Search every imported row for PowerShell evidence.",
    needsClarification: false,
    clarificationMessage: null,
    aiAssisted: false,
    auditId: null,
    reviewStatus: "not_applicable",
    validationStatus: null,
    querySpec: {
      search: "powershell",
      filters: [],
      expression: null,
      sort: null,
      cursor: null,
      limit: 300,
    },
    matchExplanation: ["Any searchable column contains powershell."],
  };
}

function semanticAppliedPreview() {
  return validAiPreview({
    intentToken: '{"intent":"semanticEvidenceSearch","terms":["powershell"]}',
    auditId: 8,
    semanticStatus: "applied",
    querySpec: {
      search: null,
      filters: [],
      expression: {
        type: "semanticSelection",
        selectionId: "a".repeat(64),
      },
      sort: null,
      cursor: null,
      limit: 300,
    },
    matchExplanation: ["Semantic matching was used: similar script-interpreter activity."],
  });
}

function semanticPendingMatchNonePreview() {
  return validAiPreview({
    semanticStatus: "index_not_ready",
    querySpec: {
      search: null,
      filters: [],
      expression: { type: "matchNone" },
      sort: null,
      cursor: null,
      limit: 300,
    },
    matchExplanation: ["Semantic matching was not used: the index is still preparing."],
  });
}

function semanticIndexSummary() {
  return {
    cancelled: false,
    rowsIndexed: 1,
    documentsIndexed: 1,
    documentsMapped: 1,
    mappingsWritten: 1,
    documentsSkipped: 0,
    mappingsSkipped: 0,
    cellsTruncated: 0,
    columnsOmitted: 0,
    chunksOmitted: 0,
    resumed: false,
  };
}

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

function bootApp({ commandHandlers = {} } = {}) {
  const calls = [];
  const document = new FakeDocument();

  const defaults = {
    import_sheet: async () => ({
      rowCount: 1,
      fromCache: false,
      columns: [{ originalName: "CommandLine", sqlName: "commandline" }],
    }),
    semantic_index_status: async () => ({ ready: true, rowsIndexed: 1 }),
    detect_column_roles: async () => [],
    query_rows: async () => page([BASELINE_ROW]),
    count_rows: async () => 1,
    accept_guided_query: async () => null,
    run_guided_query: async () => page([EVIDENCE_ROW]),
    set_guided_parse_decision: async () => null,
  };

  const invoke = async (command, args = {}) => {
    calls.push({ command, args });
    const handler = commandHandlers[command] || defaults[command];
    if (!handler) throw new Error(`Unexpected Tauri command in frontend test: ${command}`);
    return handler(args, calls);
  };

  const window = {
    __TAURI__: {
      core: { invoke },
      event: { listen: async () => () => {} },
    },
    __logParserDebug: {},
    confirm: () => true,
  };
  window.window = window;

  const sandbox = {
    window,
    document,
    Tabulator: FakeTabulator,
    alert: () => {},
    console: { error: () => {}, warn: () => {}, log: () => {} },
    setTimeout,
    clearTimeout,
  };

  vm.runInNewContext(APP_SOURCE, sandbox, { filename: APP_PATH });
  return { calls, debug: window.__logParserDebug, document };
}

async function settleFrontend(turns = 8) {
  for (let turn = 0; turn < turns; turn += 1) {
    await new Promise((resolve) => setImmediate(resolve));
  }
}

async function waitForCommand(app, command, expectedCount = 1) {
  for (let attempt = 0; attempt < 20; attempt += 1) {
    const count = app.calls.filter((call) => call.command === command).length;
    if (count >= expectedCount) return;
    await settleFrontend(1);
  }
  assert.fail(`Timed out waiting for ${expectedCount} ${command} call(s)`);
}

async function loadFixture(app) {
  await app.debug.loadSheetForTest("C:\\fixtures\\events.csv", "events");
  await settleFrontend();
}

function guidedCommands(app) {
  return app.calls
    .map(({ command }) => command)
    .filter((command) => [
      "parse_guided_query",
      "accept_guided_query",
      "run_guided_query",
    ].includes(command));
}

test("one AI search call accepts and executes a valid plan, then shows its rows", async () => {
  const app = bootApp({
    commandHandlers: {
      parse_guided_query: async () => validAiPreview(),
    },
  });
  await loadFixture(app);

  await app.debug.searchAiEvidenceForTest("find PowerShell evidence");
  await settleFrontend();

  assert.deepEqual(guidedCommands(app), [
    "parse_guided_query",
    "accept_guided_query",
    "run_guided_query",
  ]);
  const state = app.debug.getAiSearchState();
  assert.equal(state.queryMode, "guided");
  assert.equal(state.guidedReviewStatus, "accepted");
  assert.equal(state.rows.length, 1);
  assert.equal(state.rows[0].row_num, EVIDENCE_ROW.row_num);
  assert.equal(state.rows[0].commandline, EVIDENCE_ROW.commandline);
});

test("a clarification response never accepts or executes a plan", async () => {
  const app = bootApp({
    commandHandlers: {
      parse_guided_query: async () => clarificationPreview(),
    },
  });
  await loadFixture(app);

  await app.debug.searchAiEvidenceForTest("find bad stuff");
  await settleFrontend();

  assert.deepEqual(guidedCommands(app), ["parse_guided_query"]);
  const state = app.debug.getAiSearchState();
  assert.equal(state.queryMode, "normal");
  assert.equal(state.rows[0].row_num, BASELINE_ROW.row_num);
});

test("a malformed backend plan is shown as clarification and never executes", async () => {
  const malformed = validAiPreview({
    querySpec: {
      search: null,
      filters: [],
      expression: { type: "sql", value: "DROP TABLE evidence" },
      sort: null,
      cursor: null,
      limit: 300,
    },
  });
  const app = bootApp({
    commandHandlers: {
      parse_guided_query: async () => malformed,
    },
  });
  await loadFixture(app);

  await app.debug.searchAiEvidenceForTest("run an unsafe plan");
  await settleFrontend();

  assert.deepEqual(guidedCommands(app), ["parse_guided_query"]);
  const state = app.debug.getAiSearchState();
  assert.equal(state.queryMode, "normal");
  assert.equal(state.guidedQuerySpec, null);
  assert.equal(state.guidedParseResult.needsClarification, true);
  assert.equal(state.rows[0].row_num, BASELINE_ROW.row_num);
});

test("a parse failure never accepts or executes a plan", async () => {
  const app = bootApp({
    commandHandlers: {
      parse_guided_query: async () => {
        throw new Error("model unavailable");
      },
    },
  });
  await loadFixture(app);

  await app.debug.searchAiEvidenceForTest("find PowerShell evidence").catch(() => null);
  await settleFrontend();

  assert.deepEqual(guidedCommands(app), ["parse_guided_query"]);
  const state = app.debug.getAiSearchState();
  assert.equal(state.queryMode, "normal");
  assert.equal(state.rows[0].row_num, BASELINE_ROW.row_num);
});

test("an acceptance failure never runs the query", async () => {
  const app = bootApp({
    commandHandlers: {
      parse_guided_query: async () => validAiPreview(),
      accept_guided_query: async () => {
        throw new Error("audit acceptance rejected");
      },
    },
  });
  await loadFixture(app);

  await app.debug.searchAiEvidenceForTest("find PowerShell evidence").catch(() => null);
  await settleFrontend();

  assert.deepEqual(guidedCommands(app), ["parse_guided_query", "accept_guided_query"]);
  const state = app.debug.getAiSearchState();
  assert.notEqual(state.guidedReviewStatus, "accepted");
  assert.equal(state.rows[0].row_num, BASELINE_ROW.row_num);
});

test("a deterministic non-AI plan executes through query_rows without acceptance", async () => {
  const app = bootApp({
    commandHandlers: {
      parse_guided_query: async () => deterministicPreview(),
      query_rows: async ({ spec }) =>
        spec.search === "powershell" ? page([EVIDENCE_ROW]) : page([BASELINE_ROW]),
    },
  });
  await loadFixture(app);

  await app.debug.searchAiEvidenceForTest("find PowerShell evidence");
  await settleFrontend();

  assert.deepEqual(guidedCommands(app), ["parse_guided_query"]);
  const queryCalls = app.calls.filter((call) => call.command === "query_rows");
  assert.equal(queryCalls.length, 2, "expected the initial table load and deterministic evidence query");
  assert.equal(queryCalls[1].args.spec.search, "powershell");
  const state = app.debug.getAiSearchState();
  assert.equal(state.queryMode, "querySpec");
  assert.equal(state.rows.length, 1);
  assert.equal(state.rows[0].row_num, EVIDENCE_ROW.row_num);
});

test("a superseded parse cannot accept a stale plan or publish stale rows", async () => {
  const firstParse = deferred();
  const app = bootApp({
    commandHandlers: {
      parse_guided_query: async ({ queryText }) =>
        queryText === "first request" ? firstParse.promise : clarificationPreview(),
    },
  });
  await loadFixture(app);

  const staleSearch = app.debug.searchAiEvidenceForTest("first request");
  await waitForCommand(app, "parse_guided_query");
  const searchBox = app.document.getElementById("guided-search-box");
  searchBox.value = "replacement request";
  searchBox.dispatchEvent({ type: "input" });
  firstParse.resolve(validAiPreview());
  await staleSearch;
  await settleFrontend();

  assert.deepEqual(guidedCommands(app), ["parse_guided_query"]);
  const retirement = app.calls.find((call) => call.command === "set_guided_parse_decision");
  assert.equal(retirement?.args.decision, "edited");
  const state = app.debug.getAiSearchState();
  assert.equal(state.queryMode, "normal");
  assert.equal(state.rows.length, 1);
  assert.equal(state.rows[0].row_num, BASELINE_ROW.row_num);
});

test("an edited completed preview retires cleanly before the replacement search runs", async () => {
  let parseCount = 0;
  const replacementPreview = validAiPreview({
    intentToken: '{"intent":"rawEvidenceSearch","terms":["rundll32"]}',
    auditId: 8,
    previewText: "Search every imported row for rundll32 evidence.",
    querySpec: {
      search: "rundll32",
      filters: [],
      expression: null,
      sort: null,
      cursor: null,
      limit: 300,
    },
    matchExplanation: ["Any searchable column contains rundll32."],
  });
  const app = bootApp({
    commandHandlers: {
      parse_guided_query: async ({ queryText }) => {
        parseCount += 1;
        return queryText === "first preview" ? validAiPreview() : replacementPreview;
      },
      run_guided_query: async ({ auditId }) => {
        assert.equal(auditId, 8, "only the replacement audit may execute");
        return page([REPLACEMENT_ROW]);
      },
    },
  });
  await loadFixture(app);

  await app.debug.previewAiEvidenceQueryForTest("first preview");
  let state = app.debug.getAiSearchState();
  assert.equal(state.guidedReviewStatus, "unreviewed");

  const searchBox = app.document.getElementById("guided-search-box");
  searchBox.value = "replacement search";
  searchBox.dispatchEvent({ type: "input" });
  await waitForCommand(app, "set_guided_parse_decision");
  await settleFrontend();

  const decisions = app.calls.filter((call) => call.command === "set_guided_parse_decision");
  assert.equal(decisions.length, 1);
  assert.equal(decisions[0].args.auditId, 7);
  assert.equal(decisions[0].args.decision, "edited");
  state = app.debug.getAiSearchState();
  assert.equal(state.guidedReviewStatus, "edited");
  assert.equal(state.actionInFlight, null);

  await app.debug.searchAiEvidenceForTest("replacement search");
  await settleFrontend();

  assert.equal(parseCount, 2);
  assert.deepEqual(guidedCommands(app), [
    "parse_guided_query",
    "parse_guided_query",
    "accept_guided_query",
    "run_guided_query",
  ]);
  const acceptance = app.calls.find((call) => call.command === "accept_guided_query");
  assert.equal(acceptance?.args.auditId, 8);
  state = app.debug.getAiSearchState();
  assert.equal(state.queryMode, "guided");
  assert.equal(state.guidedReviewStatus, "accepted");
  assert.equal(state.rows.length, 1);
  assert.equal(state.rows[0].row_num, REPLACEMENT_ROW.row_num);
});

test("semantic-index readiness during acceptance does not retire or block the direct search", async () => {
  const semanticBuild = deferred();
  const acceptance = deferred();
  const app = bootApp({
    commandHandlers: {
      semantic_index_status: async () => ({ ready: false, rowsIndexed: 0 }),
      build_semantic_index: async () => semanticBuild.promise,
      parse_guided_query: async () => validAiPreview(),
      accept_guided_query: async () => acceptance.promise,
    },
  });
  await loadFixture(app);

  const search = app.debug.searchAiEvidenceForTest("find PowerShell evidence");
  await waitForCommand(app, "accept_guided_query");
  semanticBuild.resolve({
    cancelled: false,
    rowsIndexed: 1,
    documentsIndexed: 1,
    documentsMapped: 1,
    mappingsWritten: 1,
    documentsSkipped: 0,
    mappingsSkipped: 0,
    cellsTruncated: 0,
    columnsOmitted: 0,
    chunksOmitted: 0,
    resumed: false,
  });
  await settleFrontend();
  acceptance.resolve(null);
  await search;
  await settleFrontend();

  assert.deepEqual(guidedCommands(app), [
    "parse_guided_query",
    "accept_guided_query",
    "run_guided_query",
  ]);
  assert.equal(
    app.calls.some((call) => call.command === "set_guided_parse_decision"),
    false,
    "semantic readiness must not retire the direct search plan"
  );
  const state = app.debug.getAiSearchState();
  assert.equal(state.queryMode, "guided");
  assert.equal(state.guidedReviewStatus, "accepted");
  assert.equal(state.rows.length, 1);
  assert.equal(state.rows[0].row_num, EVIDENCE_ROW.row_num);
});

test("a guided query failure preserves baseline rows and exposes a visible Retry action", async () => {
  const app = bootApp({
    commandHandlers: {
      parse_guided_query: async () => validAiPreview(),
      run_guided_query: async () => {
        throw new Error("database query failed");
      },
    },
  });
  await loadFixture(app);

  await app.debug.searchAiEvidenceForTest("find PowerShell evidence").catch(() => null);
  await settleFrontend();

  assert.deepEqual(guidedCommands(app), [
    "parse_guided_query",
    "accept_guided_query",
    "run_guided_query",
  ]);
  const state = app.debug.getAiSearchState();
  assert.equal(state.queryMode, "normal");
  assert.equal(state.rows.length, 1);
  assert.equal(state.rows[0].row_num, BASELINE_ROW.row_num);
  assert.equal(state.aiMatchColumnVisible, false);
  assert.equal(
    app.calls.filter((call) => call.command === "count_rows").length,
    1,
    "a failed plan must not start a new evidence count"
  );
  const panel = app.document.getElementById("guided-query-panel");
  const retry = app.document.getElementById("guided-run-btn");
  assert.equal(panel.classList.contains("hidden"), false, "the failure panel should be visible");
  assert.equal(retry.classList.contains("hidden"), false, "the Retry button should be visible");
  assert.match(retry.textContent, /Retry/i);
});

test("an index-not-ready parse reparses once when semantic readiness wins the race", async () => {
  const semanticBuild = deferred();
  let parseCount = 0;
  const app = bootApp({
    commandHandlers: {
      semantic_index_status: async () => ({ ready: false, rowsIndexed: 0 }),
      build_semantic_index: async () => semanticBuild.promise,
      parse_guided_query: async () => {
        parseCount += 1;
        if (parseCount === 1) {
          semanticBuild.resolve(semanticIndexSummary());
          await new Promise((resolve) => setImmediate(resolve));
          return validAiPreview({ semanticStatus: "index_not_ready" });
        }
        return semanticAppliedPreview();
      },
      run_guided_query: async () => page([SEMANTIC_ROW]),
    },
  });
  await loadFixture(app);

  await app.debug.searchAiEvidenceForTest("find related script activity");
  await settleFrontend();

  assert.equal(parseCount, 2);
  assert.deepEqual(guidedCommands(app), [
    "parse_guided_query",
    "parse_guided_query",
    "accept_guided_query",
    "run_guided_query",
  ]);
  const decisions = app.calls.filter((call) => call.command === "set_guided_parse_decision");
  assert.equal(decisions.length, 1);
  assert.equal(decisions[0].args.decision, "edited");
  const state = app.debug.getAiSearchState();
  assert.equal(state.queryMode, "guided");
  assert.equal(state.guidedParseResult.semanticStatus, "applied");
  assert.equal(state.rows.length, 1);
  assert.equal(state.rows[0].row_num, SEMANTIC_ROW.row_num);
});

test("building semantic search publishes exact rows, then automatically refreshes them", async () => {
  const semanticBuild = deferred();
  let parseCount = 0;
  const app = bootApp({
    commandHandlers: {
      semantic_index_status: async () => ({ ready: false, rowsIndexed: 0 }),
      build_semantic_index: async () => semanticBuild.promise,
      parse_guided_query: async () => {
        parseCount += 1;
        return parseCount === 1
          ? validAiPreview({ semanticStatus: "index_not_ready" })
          : semanticAppliedPreview();
      },
      run_guided_query: async ({ auditId }) =>
        auditId === 7 ? page([EXACT_ROW]) : page([SEMANTIC_ROW]),
    },
  });
  await loadFixture(app);

  await app.debug.searchAiEvidenceForTest("find related script activity");
  await settleFrontend();

  let state = app.debug.getAiSearchState();
  assert.equal(parseCount, 1);
  assert.equal(state.rows.length, 1);
  assert.equal(state.rows[0].row_num, EXACT_ROW.row_num);
  assert.match(
    app.document.getElementById("ai-search-availability").textContent,
    /refresh automatically/i
  );

  semanticBuild.resolve(semanticIndexSummary());
  await waitForCommand(app, "parse_guided_query", 2);
  await waitForCommand(app, "run_guided_query", 2);
  await settleFrontend();

  assert.equal(parseCount, 2);
  assert.deepEqual(guidedCommands(app), [
    "parse_guided_query",
    "accept_guided_query",
    "run_guided_query",
    "parse_guided_query",
    "accept_guided_query",
    "run_guided_query",
  ]);
  state = app.debug.getAiSearchState();
  assert.equal(state.queryMode, "guided");
  assert.equal(state.guidedParseResult.semanticStatus, "applied");
  assert.equal(state.rows.length, 1);
  assert.equal(state.rows[0].row_num, SEMANTIC_ROW.row_num);
  assert.doesNotMatch(
    app.document.getElementById("ai-search-availability").textContent,
    /refresh automatically/i
  );
});

test("semantic readiness stays pending during report export and refreshes afterward", async () => {
  const semanticBuild = deferred();
  const reportDestination = deferred();
  let parseCount = 0;
  const app = bootApp({
    commandHandlers: {
      semantic_index_status: async () => ({ ready: false, rowsIndexed: 0 }),
      build_semantic_index: async () => semanticBuild.promise,
      parse_guided_query: async () => {
        parseCount += 1;
        return parseCount === 1
          ? validAiPreview({ semanticStatus: "index_not_ready" })
          : semanticAppliedPreview();
      },
      run_guided_query: async ({ auditId }) =>
        auditId === 7 ? page([EXACT_ROW]) : page([SEMANTIC_ROW]),
      "plugin:dialog|save": async () => reportDestination.promise,
      export_report: async ({ destPath }) => ({
        rowCount: 1,
        destPath,
        sheetsWritten: ["General"],
      }),
    },
  });
  await loadFixture(app);

  await app.debug.searchAiEvidenceForTest("find related script activity");
  await settleFrontend();
  assert.equal(parseCount, 1);
  assert.equal(app.debug.getAiSearchState().rows[0].row_num, EXACT_ROW.row_num);

  app.document.getElementById("report-export-btn").dispatchEvent({ type: "click" });
  await waitForCommand(app, "plugin:dialog|save");
  for (const id of [
    "guided-search-submit",
    "guided-run-btn",
    "guided-reject-btn",
    "guided-reset-btn",
  ]) {
    assert.equal(app.document.getElementById(id).disabled, true, `${id} must be report-guarded`);
  }
  const blockedPreview = await app.debug.previewAiEvidenceQueryForTest("find related script activity");
  assert.equal(blockedPreview, null);
  assert.equal(parseCount, 1);

  semanticBuild.resolve(semanticIndexSummary());
  await settleFrontend();
  assert.equal(parseCount, 1, "semantic refresh must remain pending while report export is active");

  reportDestination.resolve("C:\\tmp\\report.xlsx");
  await waitForCommand(app, "export_report");
  await settleFrontend();
  await new Promise((resolve) => setTimeout(resolve, 320));
  await waitForCommand(app, "parse_guided_query", 2);
  await waitForCommand(app, "run_guided_query", 2);
  await settleFrontend();

  assert.equal(parseCount, 2);
  assert.deepEqual(guidedCommands(app), [
    "parse_guided_query",
    "accept_guided_query",
    "run_guided_query",
    "parse_guided_query",
    "accept_guided_query",
    "run_guided_query",
  ]);
  const state = app.debug.getAiSearchState();
  assert.equal(state.guidedParseResult.semanticStatus, "applied");
  assert.equal(state.rows.length, 1);
  assert.equal(state.rows[0].row_num, SEMANTIC_ROW.row_num);
});

test("readiness during first acceptance executes exact once, then semantic once", async () => {
  const semanticBuild = deferred();
  const firstAcceptance = deferred();
  let parseCount = 0;
  let acceptanceCount = 0;
  const app = bootApp({
    commandHandlers: {
      semantic_index_status: async () => ({ ready: false, rowsIndexed: 0 }),
      build_semantic_index: async () => semanticBuild.promise,
      parse_guided_query: async () => {
        parseCount += 1;
        return parseCount === 1
          ? validAiPreview({ semanticStatus: "index_not_ready" })
          : semanticAppliedPreview();
      },
      accept_guided_query: async () => {
        acceptanceCount += 1;
        return acceptanceCount === 1 ? firstAcceptance.promise : null;
      },
      run_guided_query: async ({ auditId }) =>
        auditId === 7 ? page([EXACT_ROW]) : page([SEMANTIC_ROW]),
    },
  });
  await loadFixture(app);

  const search = app.debug.searchAiEvidenceForTest("find related script activity");
  await waitForCommand(app, "accept_guided_query");
  semanticBuild.resolve(semanticIndexSummary());
  await settleFrontend();
  assert.equal(parseCount, 1, "readiness must wait for the accepted exact execution to finish");
  assert.equal(
    app.calls.some((call) => call.command === "set_guided_parse_decision"),
    false,
    "an audit already entering execution must not be retired as edited"
  );

  firstAcceptance.resolve(null);
  await search;
  await settleFrontend();

  assert.equal(parseCount, 2);
  assert.equal(acceptanceCount, 2);
  assert.deepEqual(guidedCommands(app), [
    "parse_guided_query",
    "accept_guided_query",
    "run_guided_query",
    "parse_guided_query",
    "accept_guided_query",
    "run_guided_query",
  ]);
  assert.equal(
    app.calls.some((call) => call.command === "set_guided_parse_decision"),
    false
  );
  const state = app.debug.getAiSearchState();
  assert.equal(state.guidedParseResult.semanticStatus, "applied");
  assert.equal(state.rows.length, 1);
  assert.equal(state.rows[0].row_num, SEMANTIC_ROW.row_num);
});

test("a semantic retry that is still not ready stops after two parses", async () => {
  let parseCount = 0;
  const app = bootApp({
    commandHandlers: {
      parse_guided_query: async () => {
        parseCount += 1;
        return validAiPreview({
          semanticStatus: "index_not_ready",
          auditId: parseCount === 1 ? 7 : 8,
          intentToken: parseCount === 1
            ? '{"intent":"rawEvidenceSearch","attempt":1}'
            : '{"intent":"rawEvidenceSearch","attempt":2}',
        });
      },
      run_guided_query: async () => page([EXACT_ROW]),
    },
  });
  await loadFixture(app);

  await app.debug.searchAiEvidenceForTest("find related script activity");
  await settleFrontend(20);

  assert.equal(parseCount, 2, "semantic retry must be bounded to one additional parse");
  assert.deepEqual(guidedCommands(app), [
    "parse_guided_query",
    "parse_guided_query",
    "accept_guided_query",
    "run_guided_query",
  ]);
  const decisions = app.calls.filter((call) => call.command === "set_guided_parse_decision");
  assert.equal(decisions.length, 1);
  assert.equal(decisions[0].args.auditId, 7);
  assert.equal(decisions[0].args.decision, "edited");
  const state = app.debug.getAiSearchState();
  assert.equal(state.guidedParseResult.semanticStatus, "index_not_ready");
  assert.equal(state.rows.length, 1);
  assert.equal(state.rows[0].row_num, EXACT_ROW.row_num);
  assert.doesNotMatch(
    app.document.getElementById("ai-search-availability").textContent,
    /refresh automatically/i,
    "a completed bounded retry must not queue another semantic refresh"
  );
});

test("matchNone safely publishes zero rows until semantic results automatically replace it", async () => {
  const semanticBuild = deferred();
  let parseCount = 0;
  const app = bootApp({
    commandHandlers: {
      semantic_index_status: async () => ({ ready: false, rowsIndexed: 0 }),
      build_semantic_index: async () => semanticBuild.promise,
      parse_guided_query: async () => {
        parseCount += 1;
        return parseCount === 1
          ? semanticPendingMatchNonePreview()
          : semanticAppliedPreview();
      },
      run_guided_query: async ({ auditId }) =>
        auditId === 7 ? page([]) : page([SEMANTIC_ROW]),
    },
  });
  await loadFixture(app);

  await app.debug.searchAiEvidenceForTest("find the attack path");
  await settleFrontend();

  let state = app.debug.getAiSearchState();
  assert.equal(parseCount, 1);
  assert.equal(state.queryMode, "guided");
  assert.equal(state.guidedQuerySpec.expression.type, "matchNone");
  assert.equal(state.rows.length, 0);
  assert.match(
    app.document.getElementById("ai-search-availability").textContent,
    /No exact rows matched yet.*refresh automatically/i
  );

  semanticBuild.resolve(semanticIndexSummary());
  await waitForCommand(app, "parse_guided_query", 2);
  await waitForCommand(app, "run_guided_query", 2);
  await settleFrontend();

  assert.equal(parseCount, 2);
  assert.deepEqual(guidedCommands(app), [
    "parse_guided_query",
    "accept_guided_query",
    "run_guided_query",
    "parse_guided_query",
    "accept_guided_query",
    "run_guided_query",
  ]);
  state = app.debug.getAiSearchState();
  assert.equal(state.guidedParseResult.semanticStatus, "applied");
  assert.equal(state.guidedQuerySpec.expression.type, "semanticSelection");
  assert.equal(state.rows.length, 1);
  assert.equal(state.rows[0].row_num, SEMANTIC_ROW.row_num);
});

test("manual guided reset preserves same-dataset semantic, role, and timestamp work", async () => {
  const semanticBuild = deferred();
  const refreshedRoles = deferred();
  const timestampCheck = deferred();
  let roleDetectionCount = 0;
  let attackPathParseCount = 0;
  const timestampSuggestion = {
    role: "timestamp",
    sqlName: "commandline",
    originalName: "CommandLine",
    confidence: 0.95,
    status: "suggested",
    reasons: ["test timestamp mapping"],
  };
  const hostSuggestion = {
    role: "host",
    sqlName: "commandline",
    originalName: "CommandLine",
    confidence: 0.8,
    status: "suggested",
    reasons: ["test host mapping"],
  };
  const pendingAttackPath = {
    ...semanticPendingMatchNonePreview(),
    auditId: 8,
    intentToken: '{"intent":"attackPath","attempt":1}',
  };
  const semanticAttackPath = {
    ...semanticAppliedPreview(),
    auditId: 9,
    intentToken: '{"intent":"attackPath","attempt":2}',
  };
  const app = bootApp({
    commandHandlers: {
      semantic_index_status: async () => ({ ready: false, rowsIndexed: 0 }),
      build_semantic_index: async () => semanticBuild.promise,
      detect_column_roles: async () => {
        roleDetectionCount += 1;
        return roleDetectionCount === 1
          ? [timestampSuggestion]
          : refreshedRoles.promise;
      },
      analyze_timestamp_column: async () => timestampCheck.promise,
      parse_guided_query: async ({ queryText }) => {
        if (queryText === "first evidence search") return validAiPreview();
        attackPathParseCount += 1;
        return attackPathParseCount === 1 ? pendingAttackPath : semanticAttackPath;
      },
      run_guided_query: async ({ auditId }) => {
        if (auditId === 7) return page([EVIDENCE_ROW]);
        if (auditId === 8) return page([]);
        return page([SEMANTIC_ROW]);
      },
    },
  });
  await loadFixture(app);
  await waitForCommand(app, "analyze_timestamp_column");

  await app.debug.searchAiEvidenceForTest("first evidence search");
  await settleFrontend();
  const reset = app.document.getElementById("guided-reset-btn");
  assert.equal(reset.classList.contains("hidden"), false);
  assert.equal(reset.disabled, false);

  const roleRequest = app.debug.detectRolesForTest();
  await waitForCommand(app, "detect_column_roles", 2);
  reset.dispatchEvent({ type: "click" });
  await waitForCommand(app, "query_rows", 2);
  await settleFrontend();
  let state = app.debug.getAiSearchState();
  assert.equal(state.queryMode, "normal");
  assert.equal(app.debug.getSemanticIndexState().status, "building");
  assert.equal(app.debug.getSemanticIndexState().inFlight, true);

  refreshedRoles.resolve([timestampSuggestion, hostSuggestion]);
  await roleRequest;
  timestampCheck.resolve({
    originalName: "CommandLine",
    needsTimezone: true,
    needsDateConvention: false,
    sampleNaiveValues: ["2026-07-17 10:00:00"],
    sampleAmbiguousDateValues: [],
  });
  await settleFrontend();
  const intelState = app.debug.getIntelState();
  assert.equal(intelState.columnRoleSuggestions.some((row) => row.role === "host"), true);
  assert.equal(intelState.timestampAnalysis?.needsTimezone, true);

  await app.debug.searchAiEvidenceForTest("find the attack path");
  await settleFrontend();
  state = app.debug.getAiSearchState();
  assert.equal(state.guidedQuerySpec.expression.type, "matchNone");
  assert.equal(state.rows.length, 0);

  semanticBuild.resolve(semanticIndexSummary());
  await waitForCommand(app, "parse_guided_query", 3);
  await waitForCommand(app, "run_guided_query", 3);
  await settleFrontend();

  assert.equal(attackPathParseCount, 2);
  assert.equal(app.debug.getSemanticIndexState().status, "ready");
  state = app.debug.getAiSearchState();
  assert.equal(state.guidedParseResult.semanticStatus, "applied");
  assert.equal(state.guidedQuerySpec.expression.type, "semanticSelection");
  assert.equal(state.rows.length, 1);
  assert.equal(state.rows[0].row_num, SEMANTIC_ROW.row_num);
});

test("text changed during deferred acceptance cannot publish the stale evidence page", async () => {
  const acceptance = deferred();
  const app = bootApp({
    commandHandlers: {
      parse_guided_query: async () => validAiPreview(),
      accept_guided_query: async () => acceptance.promise,
    },
  });
  await loadFixture(app);

  const search = app.debug.searchAiEvidenceForTest("find PowerShell evidence");
  await waitForCommand(app, "accept_guided_query");
  const searchBox = app.document.getElementById("guided-search-box");
  searchBox.value = "find a different account instead";
  searchBox.dispatchEvent({ type: "input" });
  acceptance.resolve(null);
  await search;
  await settleFrontend();

  assert.deepEqual(guidedCommands(app), ["parse_guided_query", "accept_guided_query"]);
  const state = app.debug.getAiSearchState();
  assert.equal(state.queryMode, "normal");
  assert.equal(state.rows.length, 1);
  assert.equal(state.rows[0].row_num, BASELINE_ROW.row_num);
  assert.equal(state.aiMatchColumnVisible, false);
  assert.equal(app.calls.filter((call) => call.command === "count_rows").length, 1);
});

test("Apply cannot interrupt acceptance or first-page publication", async () => {
  const firstPage = deferred();
  const app = bootApp({
    commandHandlers: {
      parse_guided_query: async () => validAiPreview(),
      run_guided_query: async () => firstPage.promise,
    },
  });
  await loadFixture(app);

  const search = app.debug.searchAiEvidenceForTest("find PowerShell evidence");
  await waitForCommand(app, "run_guided_query");

  const apply = app.document.getElementById("apply-btn");
  assert.equal(apply.disabled, true, "Apply should be disabled while the first evidence page is pending");
  for (const id of [
    "clear-btn",
    "search-box",
    "export-csv-btn",
    "export-xlsx-btn",
    "report-export-btn",
    "prev-page-btn",
    "next-page-btn",
  ]) {
    assert.equal(
      app.document.getElementById(id).disabled,
      true,
      `${id} should be disabled while the first evidence page is pending`
    );
  }
  const quickSearch = app.document.getElementById("search-box");
  quickSearch.value = "must not be cleared by a competing event";
  app.document.getElementById("clear-btn").dispatchEvent({ type: "click" });
  assert.equal(quickSearch.value, "must not be cleared by a competing event");
  apply.dispatchEvent({ type: "click" });
  await settleFrontend();
  assert.equal(
    app.calls.filter((call) => call.command === "query_rows").length,
    1,
    "a programmatic Apply event must not bypass the in-flight guard"
  );
  assert.equal(app.debug.getAiSearchState().actionInFlight, "run");

  firstPage.resolve(page([EVIDENCE_ROW]));
  await search;
  await settleFrontend();

  assert.deepEqual(guidedCommands(app), [
    "parse_guided_query",
    "accept_guided_query",
    "run_guided_query",
  ]);
  const state = app.debug.getAiSearchState();
  assert.equal(state.queryMode, "guided");
  assert.equal(state.rows.length, 1);
  assert.equal(state.rows[0].row_num, EVIDENCE_ROW.row_num);
});

test("an active ordinary page request blocks parsing until its table publication finishes", async () => {
  const ordinaryPage = deferred();
  let queryCallCount = 0;
  const app = bootApp({
    commandHandlers: {
      query_rows: async () => {
        queryCallCount += 1;
        return queryCallCount === 1 ? page([BASELINE_ROW]) : ordinaryPage.promise;
      },
      parse_guided_query: async () => validAiPreview(),
    },
  });
  await loadFixture(app);

  app.document.getElementById("search-box").value = "ordinary filter";
  app.document.getElementById("apply-btn").dispatchEvent({ type: "click" });
  await waitForCommand(app, "query_rows", 2);
  assert.equal(app.debug.getAiSearchState().dataRequestInFlight, true);
  assert.equal(app.document.getElementById("guided-search-submit").disabled, true);

  const blockedSearch = await app.debug.searchAiEvidenceForTest("find PowerShell evidence");
  assert.equal(blockedSearch, null);
  assert.equal(
    app.calls.filter((call) => call.command === "parse_guided_query").length,
    0,
    "AI parsing must not overlap an unpublished ordinary page"
  );

  ordinaryPage.resolve(page([NORMAL_FILTER_ROW]));
  await settleFrontend();
  let state = app.debug.getAiSearchState();
  assert.equal(state.dataRequestInFlight, false);
  assert.equal(state.queryMode, "normal");
  assert.equal(state.rows.length, 1);
  assert.equal(state.rows[0].row_num, NORMAL_FILTER_ROW.row_num);

  await app.debug.searchAiEvidenceForTest("find PowerShell evidence");
  await settleFrontend();
  assert.deepEqual(guidedCommands(app), [
    "parse_guided_query",
    "accept_guided_query",
    "run_guided_query",
  ]);
  state = app.debug.getAiSearchState();
  assert.equal(state.queryMode, "guided");
  assert.equal(state.rows.length, 1);
  assert.equal(state.rows[0].row_num, EVIDENCE_ROW.row_num);
});
