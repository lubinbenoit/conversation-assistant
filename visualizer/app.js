(function () {
  "use strict";

  const state = {
    runs: [],
    activeRunId: null,
    activeTurnIndex: 0,
    activeTab: "exchange",
  };

  const elements = {
    fileInput: document.querySelector("#file-input"),
    openButton: document.querySelector("#open-button"),
    emptyOpenButton: document.querySelector("#empty-open-button"),
    themeButton: document.querySelector("#theme-button"),
    rawRunButton: document.querySelector("#raw-run-button"),
    runList: document.querySelector("#run-list"),
    runCount: document.querySelector("#run-count"),
    dropZone: document.querySelector("#drop-zone"),
    runView: document.querySelector("#run-view"),
    runMetadata: document.querySelector("#run-metadata"),
    timeline: document.querySelector("#turn-timeline"),
    tabs: document.querySelector("#turn-tabs"),
    toastRegion: document.querySelector("#toast-region"),
    rawDialog: document.querySelector("#raw-dialog"),
    rawDialogTitle: document.querySelector("#raw-dialog-title"),
    rawJson: document.querySelector("#raw-json"),
    closeRawButton: document.querySelector("#close-raw-button"),
  };

  function activeRun() {
    return state.runs.find((run) => run.id === state.activeRunId) || null;
  }

  function makeId(file, data) {
    const identity = [file.name, file.size, file.lastModified, data.scenario, data.policy, data.records.length].join("|");
    let hash = 0;
    for (let index = 0; index < identity.length; index += 1) {
      hash = ((hash << 5) - hash + identity.charCodeAt(index)) | 0;
    }
    return `${file.name}:${hash}`;
  }

  function validateRun(data) {
    if (!data || typeof data !== "object" || Array.isArray(data)) {
      throw new Error("The top-level value must be a JSON object.");
    }
    if (!Array.isArray(data.records)) {
      throw new Error('Missing required field: "records" must be an array.');
    }
    if (data.records.length === 0) {
      throw new Error('The "records" array is empty.');
    }
    data.records.forEach((record, index) => {
      if (!record || typeof record !== "object" || Array.isArray(record)) {
        throw new Error(`Record ${index + 1} must be an object.`);
      }
      if (!Array.isArray(record.messages)) {
        throw new Error(`Record ${index + 1} is missing a "messages" array.`);
      }
    });
  }

  async function addFiles(fileList) {
    const files = Array.from(fileList).filter((file) => file.name.toLowerCase().endsWith(".json") || file.type === "application/json");
    if (files.length === 0) {
      showToast("No JSON files were selected.", true);
      return;
    }

    let added = 0;
    for (const file of files) {
      try {
        const data = JSON.parse(await file.text());
        validateRun(data);
        const id = makeId(file, data);
        if (state.runs.some((run) => run.id === id)) {
          showToast(`${file.name} is already loaded.`);
          continue;
        }
        state.runs.push({ id, filename: file.name, data });
        added += 1;
        if (!state.activeRunId) state.activeRunId = id;
      } catch (error) {
        const detail = error instanceof SyntaxError ? `Invalid JSON: ${error.message}` : error.message;
        showToast(`Could not open ${file.name}. ${detail}`, true);
      }
    }

    if (added > 0) {
      const newest = state.runs[state.runs.length - 1];
      selectRun(newest.id);
      showToast(`${added} evaluation ${added === 1 ? "run" : "runs"} loaded.`);
    } else {
      render();
    }
    elements.fileInput.value = "";
  }

  function selectRun(id) {
    state.activeRunId = id;
    state.activeTurnIndex = 0;
    state.activeTab = "exchange";
    render();
  }

  function removeRun(id) {
    const index = state.runs.findIndex((run) => run.id === id);
    if (index === -1) return;
    state.runs.splice(index, 1);
    if (state.activeRunId === id) {
      const replacement = state.runs[Math.min(index, state.runs.length - 1)];
      state.activeRunId = replacement ? replacement.id : null;
      state.activeTurnIndex = 0;
    }
    render();
  }

  function setTurn(index) {
    const run = activeRun();
    if (!run) return;
    state.activeTurnIndex = Math.max(0, Math.min(index, run.data.records.length - 1));
    renderTurn(run);
  }

  function render() {
    renderRunList();
    const run = activeRun();
    elements.dropZone.hidden = Boolean(run);
    elements.runView.hidden = !run;
    elements.rawRunButton.disabled = !run;
    if (run) renderRun(run);
  }

  function renderRunList() {
    elements.runList.replaceChildren();
    elements.runCount.textContent = String(state.runs.length);

    state.runs.forEach((run) => {
      const row = node("div", "run-row");
      const button = node("button", `run-item${run.id === state.activeRunId ? " is-active" : ""}`);
      button.type = "button";
      button.append(node("strong", "", run.filename));
      button.append(node("span", "", displayValue(run.data.policy, "Unknown policy")));
      button.addEventListener("click", () => selectRun(run.id));

      const remove = node("button", "remove-run", "×");
      remove.type = "button";
      remove.setAttribute("aria-label", `Remove ${run.filename}`);
      remove.addEventListener("click", () => removeRun(run.id));
      row.append(button, remove);
      elements.runList.append(row);
    });
  }

  function renderRun(run) {
    const data = run.data;

    const metadata = [
      ["Model", data.model],
      ["Context window", formatNumber(data.num_ctx)],
      ["Recent turns", data.recent_turns],
      ["Generation limit", formatNumber(data.num_predict)],
      ["Temperature", data.temperature],
      ["Seed", data.seed],
      ["Thinking mode", data.thinking_mode],
      ["Policy", data.policy],
    ];
    elements.runMetadata.replaceChildren(...metadata.map(([label, value]) => definition(label, value)));
    renderTimeline(run);
    renderTurn(run);
  }

  function renderTimeline(run) {
    elements.timeline.replaceChildren();
    run.data.records.forEach((record, index) => {
      const button = node("button", `turn-button${index === state.activeTurnIndex ? " is-active" : ""}`);
      button.type = "button";
      const utilization = finiteNumber(record.prompt_utilization);
      const height = utilization === null ? 3 : Math.max(3, Math.min(100, utilization * 100));
      const frame = node("span", "usage-bar-frame");
      const bar = node("span", "usage-bar");
      bar.style.height = `${height}%`;
      frame.append(bar);
      button.append(frame, node("span", "turn-number", String(displayValue(record.turn, index + 1))));
      const percentage = utilization === null ? "not recorded" : formatPercent(utilization);
      button.title = `Turn ${displayValue(record.turn, index + 1)}: ${percentage} context utilization`;
      button.setAttribute("aria-label", button.title);
      button.addEventListener("click", () => setTurn(index));
      elements.timeline.append(button);
    });
  }

  function renderTurn(run) {
    const records = run.data.records;
    const record = records[state.activeTurnIndex];
    if (!record) return;
    renderTimeline(run);
    renderExchange(record);
    renderModelInput(record);
    renderPolicyState(record);
    renderUsage(record);
    activateTab(state.activeTab);
  }

  function renderExchange(record) {
    const panel = document.querySelector("#panel-exchange");
    const stack = node("div", "content-stack");
    stack.append(messageCard("user", displayValue(record.user, "")));
    stack.append(messageCard("assistant", displayValue(record.assistant, "")));
    panel.replaceChildren(stack);
  }

  function renderModelInput(record) {
    const panel = document.querySelector("#panel-input");
    const stack = node("div", "content-stack");
    const sourceSection = node("section");
    sourceSection.append(node("h3", "", "Selected source IDs"));
    const ids = node("div", "source-ids");
    if (Array.isArray(record.source_ids) && record.source_ids.length > 0) {
      record.source_ids.forEach((id) => ids.append(node("span", "source-id", String(id))));
    } else {
      ids.append(node("span", "empty-value", "No source IDs recorded"));
    }
    sourceSection.append(ids);
    stack.append(sourceSection);

    const messages = Array.isArray(record.messages) ? record.messages : [];
    messages.forEach((message, index) => {
      const role = typeof message.role === "string" ? message.role : "message";
      stack.append(messageCard(role, displayValue(message.content, ""), `Message ${index + 1}`));
    });
    panel.replaceChildren(stack);
  }

  function renderPolicyState(record) {
    const panel = document.querySelector("#panel-state");
    const content = node("div", "content-stack");
    if (record.maintenance_error !== null && record.maintenance_error !== undefined) {
      content.append(node("div", "error-banner", String(record.maintenance_error)));
    }
    const columns = node("div", "two-column");
    columns.append(stateCard("Before model call", record.manager_state_before));
    columns.append(stateCard("After maintenance", record.manager_state_after));
    content.append(columns);
    content.append(maintenanceCard(record.maintenance_calls));
    panel.replaceChildren(content);
  }

  function renderUsage(record) {
    const panel = document.querySelector("#panel-usage");
    const content = node("div", "content-stack");
    const grid = node("div", "usage-grid primary-usage-grid");
    grid.append(utilizationCard(record.prompt_utilization));
    grid.append(usageCard("Prompt tokens", record.prompt_eval_count));
    grid.append(usageCard("Generated tokens", record.eval_count));
    grid.append(usageCard("Completion reason", record.done_reason));
    content.append(grid);

    const maintenance = record.maintenance_calls;
    const section = node("section", "maintenance-section");
    section.append(node("h3", "", "Maintenance usage"));
    const maintenanceGrid = node("div", "usage-grid");
    maintenanceGrid.append(usageCard("Calls", maintenance && maintenance.call_count));
    maintenanceGrid.append(usageCard("Prompt tokens", maintenance && maintenance.prompt_eval_count));
    maintenanceGrid.append(usageCard("Generated tokens", maintenance && maintenance.eval_count));
    section.append(maintenanceGrid);
    content.append(section);
    panel.replaceChildren(content);
  }

  function messageCard(role, content, annotation) {
    const safeRole = String(role).toLowerCase();
    const card = node("article", `message-card role-${safeRole.replace(/[^a-z0-9_-]/g, "-")}`);
    const header = node("div", "message-header");
    header.append(node("span", "", role));
    if (annotation) header.append(node("span", "message-meta", annotation));
    const body = node("div", "message-body");
    renderMarkdown(body, String(content));
    card.append(header, body);
    return card;
  }

  function stateCard(title, value) {
    const card = node("section", "state-card");
    card.append(node("div", "card-header", title));
    card.append(renderJson(value));
    return card;
  }

  function maintenanceCard(value) {
    const card = node("section", "maintenance-card");
    card.append(node("div", "card-header", "Maintenance calls"));
    if (!value || typeof value !== "object") {
      const empty = node("div", "maintenance-summary");
      empty.append(node("span", "empty-value", "Not recorded"));
      card.append(empty);
      return card;
    }

    const summary = node("div", "maintenance-summary");
    [
      ["Calls", value.call_count],
      ["Successful", value.successful_call_count],
      ["Failed", value.failed_call_count],
      ["Prompt tokens", value.prompt_eval_count],
      ["Generated tokens", value.eval_count],
    ].forEach(([label, metric]) => {
      const item = node("div");
      item.append(node("span", "metric-label", label), node("span", "metric-value", formatRecorded(metric)));
      summary.append(item);
    });
    card.append(summary);

    if (Array.isArray(value.calls) && value.calls.length > 0) {
      const callList = node("div", "call-list");
      value.calls.forEach((call, index) => {
        const details = document.createElement("details");
        const summaryNode = document.createElement("summary");
        summaryNode.textContent = `Call ${index + 1}`;
        const detailGrid = node("div", "call-details");
        Object.entries(call).forEach(([key, itemValue]) => {
          const item = node("div");
          item.append(node("span", "metric-label", humanizeKey(key)), node("span", "metric-value", formatRecorded(itemValue)));
          detailGrid.append(item);
        });
        details.append(summaryNode, detailGrid);
        callList.append(details);
      });
      card.append(callList);
    }
    return card;
  }

  function usageCard(label, value) {
    const card = node("div", "usage-card");
    card.append(node("div", "label", label), node("div", "value", formatRecorded(value)));
    return card;
  }

  function utilizationCard(value) {
    const utilization = finiteNumber(value);
    const card = usageCard("Context utilization", utilization === null ? null : formatPercent(utilization));
    const meter = node("div", "usage-meter");
    const fill = node("span");
    fill.style.width = `${utilization === null ? 0 : Math.min(100, Math.max(0, utilization * 100))}%`;
    meter.append(fill);
    card.append(meter);
    return card;
  }

  function renderJson(value) {
    const root = node("ul", "json-tree");
    appendJsonValue(root, value, null);
    return root;
  }

  function appendJsonValue(parent, value, key) {
    const item = document.createElement("li");
    if (key !== null) item.append(node("span", "json-key", `${humanizeKey(key)}: `));

    if (Array.isArray(value)) {
      item.append(node("span", "collection-label", value.length === 0 ? "Empty list" : `${value.length} ${value.length === 1 ? "item" : "items"}`));
      if (value.length > 0) {
        const list = document.createElement("ul");
        value.forEach((entry, index) => appendJsonValue(list, entry, String(index + 1)));
        item.append(list);
      }
    } else if (value && typeof value === "object") {
      const entries = Object.entries(value);
      item.append(node("span", "collection-label", entries.length === 0 ? "Empty object" : ""));
      if (entries.length > 0) {
        const list = document.createElement("ul");
        entries.forEach(([entryKey, entryValue]) => appendJsonValue(list, entryValue, entryKey));
        item.append(list);
      }
    } else {
      item.append(jsonScalar(value));
    }
    parent.append(item);
  }

  function jsonScalar(value) {
    if (value === null || value === undefined) return node("span", "json-value json-null", value === null ? "null" : "Not recorded");
    if (typeof value === "boolean") return node("span", "json-value json-boolean", String(value));
    if (typeof value === "number") return node("span", "json-value json-number", String(value));
    return node("span", "json-value json-string", String(value));
  }

  const safeMarkdownTags = new Set([
    "A", "BLOCKQUOTE", "BR", "CODE", "DEL", "EM", "H1", "H2", "H3", "H4",
    "H5", "H6", "HR", "LI", "OL", "P", "PRE", "STRONG", "TABLE", "TBODY",
    "TD", "TH", "THEAD", "TR", "UL",
  ]);

  // Marked provides the Markdown structure. We copy its result into a fresh DOM
  // using an explicit element allowlist before rendering mathematics with KaTeX.
  function renderMarkdown(container, markdown) {
    marked.use({ gfm: true, breaks: true });
    const template = document.createElement("template");
    template.innerHTML = marked.parse(markdown, { async: false });
    template.content.childNodes.forEach((child) => {
      const safeChild = sanitizeMarkdownNode(child);
      if (safeChild) container.append(safeChild);
    });
    renderMath(container);
    if (!container.childNodes.length) container.append(node("span", "empty-value", "No content recorded"));
  }

  function sanitizeMarkdownNode(source) {
    if (source.nodeType === Node.TEXT_NODE) return document.createTextNode(source.textContent);
    if (source.nodeType !== Node.ELEMENT_NODE || !safeMarkdownTags.has(source.tagName)) return null;
    const clean = document.createElement(source.tagName.toLowerCase());
    if (source.tagName === "A") {
      const href = source.getAttribute("href") || "";
      if (/^(https?:|mailto:|#)/i.test(href)) {
        clean.setAttribute("href", href);
        clean.setAttribute("rel", "noopener noreferrer");
        if (/^https?:/i.test(href)) clean.setAttribute("target", "_blank");
      }
    }
    source.childNodes.forEach((child) => {
      const safeChild = sanitizeMarkdownNode(child);
      if (safeChild) clean.append(safeChild);
    });
    return clean;
  }

  function renderMath(container) {
    const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
    const textNodes = [];
    while (walker.nextNode()) {
      const parent = walker.currentNode.parentElement;
      if (parent && !parent.closest("code, pre")) textNodes.push(walker.currentNode);
    }
    textNodes.forEach(renderMathInTextNode);
  }

  function renderMathInTextNode(textNode) {
    const text = textNode.textContent;
    const pattern = /(\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\]|\\\(([\s\S]+?)\\\)|\$(?!\d)([^$\n]+?)\$)/g;
    const fragment = document.createDocumentFragment();
    let lastIndex = 0;
    let found = false;
    for (const match of text.matchAll(pattern)) {
      found = true;
      if (match.index > lastIndex) fragment.append(document.createTextNode(text.slice(lastIndex, match.index)));
      const displayMode = match[0].startsWith("$$") || match[0].startsWith("\\[");
      const expression = match[2] ?? match[3] ?? match[4] ?? match[5];
      const math = node("span", displayMode ? "math-display" : "math-inline");
      katex.render(expression, math, { displayMode, throwOnError: false, strict: false });
      fragment.append(math);
      lastIndex = match.index + match[0].length;
    }
    if (!found) return;
    if (lastIndex < text.length) fragment.append(document.createTextNode(text.slice(lastIndex)));
    textNode.replaceWith(fragment);
  }

  function activateTab(tabName) {
    state.activeTab = tabName;
    document.querySelectorAll(".tab").forEach((tab) => {
      const active = tab.dataset.tab === tabName;
      tab.classList.toggle("is-active", active);
      tab.setAttribute("aria-selected", String(active));
    });
    document.querySelectorAll(".tab-panel").forEach((panel) => {
      panel.hidden = panel.id !== `panel-${tabName}`;
    });
  }

  function showRawRun() {
    const run = activeRun();
    if (!run) return;
    elements.rawDialogTitle.textContent = run.filename;
    elements.rawJson.textContent = JSON.stringify(run.data, null, 2);
    elements.rawDialog.showModal();
  }

  function showToast(message, isError = false) {
    const toast = node("div", `toast${isError ? " error" : ""}`, message);
    elements.toastRegion.append(toast);
    window.setTimeout(() => toast.remove(), 4500);
  }

  function definition(label, value) {
    const wrapper = document.createElement("div");
    wrapper.append(node("dt", "", label), node("dd", "", formatRecorded(value)));
    return wrapper;
  }

  function node(tag, className = "", text) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = text;
    return element;
  }

  function humanizeKey(key) {
    const text = String(key).replace(/[_-]+/g, " ").trim();
    return text ? text.charAt(0).toUpperCase() + text.slice(1) : String(key);
  }

  function displayValue(value, fallback) {
    return value === null || value === undefined || value === "" ? fallback : value;
  }

  function finiteNumber(value) {
    const number = Number(value);
    return value !== null && value !== "" && Number.isFinite(number) ? number : null;
  }

  function formatNumber(value) {
    const number = finiteNumber(value);
    return number === null ? "Not recorded" : new Intl.NumberFormat().format(number);
  }

  function formatPercent(value) {
    const number = finiteNumber(value);
    return number === null ? "Not recorded" : `${(number * 100).toFixed(1)}%`;
  }

  function formatRecorded(value) {
    if (value === null || value === undefined || value === "") return "Not recorded";
    if (typeof value === "number") return formatNumber(value);
    if (typeof value === "boolean") return value ? "True" : "False";
    return String(value);
  }

  function openPicker() { elements.fileInput.click(); }

  function applyTheme(theme, persist = true) {
    document.documentElement.dataset.theme = theme;
    const dark = theme === "dark";
    elements.themeButton.textContent = dark ? "Light mode" : "Dark mode";
    elements.themeButton.setAttribute("aria-pressed", String(dark));
    if (persist) {
      try { localStorage.setItem("evaluation-viewer-theme", theme); } catch (_) { /* Storage may be unavailable for local files. */ }
    }
  }

  elements.openButton.addEventListener("click", openPicker);
  elements.emptyOpenButton.addEventListener("click", openPicker);
  elements.themeButton.addEventListener("click", () => {
    applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
  });
  elements.fileInput.addEventListener("change", () => addFiles(elements.fileInput.files));
  elements.rawRunButton.addEventListener("click", showRawRun);
  elements.closeRawButton.addEventListener("click", () => elements.rawDialog.close());
  elements.rawDialog.addEventListener("click", (event) => {
    if (event.target === elements.rawDialog) elements.rawDialog.close();
  });
  elements.tabs.addEventListener("click", (event) => {
    const tab = event.target.closest("[data-tab]");
    if (tab) activateTab(tab.dataset.tab);
  });

  ["dragenter", "dragover"].forEach((eventName) => {
    document.addEventListener(eventName, (event) => {
      event.preventDefault();
      elements.dropZone.classList.add("is-dragging");
      if (activeRun()) elements.dropZone.hidden = false;
    });
  });
  ["dragleave", "drop"].forEach((eventName) => {
    document.addEventListener(eventName, (event) => {
      event.preventDefault();
      elements.dropZone.classList.remove("is-dragging");
      if (eventName === "drop") addFiles(event.dataTransfer.files);
      else if (activeRun() && event.relatedTarget === null) elements.dropZone.hidden = true;
    });
  });

  applyTheme(document.documentElement.dataset.theme || "light", false);
  render();
})();
