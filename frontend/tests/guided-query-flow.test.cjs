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
