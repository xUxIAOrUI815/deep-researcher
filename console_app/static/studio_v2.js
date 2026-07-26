const runId = document.body.dataset.runId;
const root = document.querySelector("#view-root");
const detail = document.querySelector("#detail-root");
const loadMore = document.querySelector("#load-more");
const title = document.querySelector("#view-title");
const kicker = document.querySelector("#view-kicker");

const state = {
  view: "tasks",
  cursor: null,
  schedulerDiffSequence: 0,
  evidenceDiffSequence: 0,
  runtimeErrorSequence: 0,
  schedulerErrorSequence: 0,
  schedulerDiffMore: true,
  evidenceDiffMore: true,
  runtimeErrorMore: true,
  schedulerErrorMore: true,
  nodes: [],
  edges: [],
  records: [],
};

const viewMeta = {
  tasks: ["ORCHESTRATION", "Task DAG"],
  evidence: ["KNOWLEDGE", "Evidence Graph"],
  diffs: ["EVENT REPLAY", "State Diff"],
  errors: ["RECOVERY", "Errors & Retries"],
  conflicts: ["VERIFICATION", "Conflict Navigation"],
  components: ["VERSIONING", "Current Components"],
};

function esc(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

async function api(path) {
  const response = await fetch(path);
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || `${response.status} ${response.statusText}`);
  }
  return response.json();
}

function showDetail(value) {
  const links = value.links || [];
  detail.className = "";
  detail.innerHTML = `
    <pre>${esc(JSON.stringify(value, null, 2))}</pre>
    <div class="links">
      ${links.map((link) => `<a href="${esc(link.href)}" target="_blank" rel="noreferrer">${esc(link.kind)} · ${esc(link.label)}</a>`).join("")}
    </div>
  `;
}

function bindDetails() {
  root.querySelectorAll("[data-detail]").forEach((element) => {
    element.addEventListener("click", () => {
      const bucket = element.dataset.bucket;
      const index = Number(element.dataset.detail);
      showDetail(state[bucket][index]);
    });
  });
}

function renderGraph() {
  if (!state.nodes.length) {
    root.innerHTML = `<div class="empty">此运行尚无该视图的数据。Studio 不会从旧状态或底层表猜测节点。</div>`;
    return;
  }
  root.innerHTML = `
    <div class="graph">
      ${state.nodes.map((node, index) => `
        <button class="node ${esc(node.status)}" data-bucket="nodes" data-detail="${index}">
          <span class="type">${esc(node.node_type)}</span>
          <strong>${esc(node.label)}</strong>
          <span class="status">${esc(node.status || "n/a")}</span>
        </button>
      `).join("")}
      <div class="edge-list">
        ${state.edges.map((edge, index) => `
          <div class="edge" data-bucket="edges" data-detail="${index}">
            <b>${esc(edge.edge_type)}</b>
            <code>${esc(edge.source_node_id)} → ${esc(edge.target_node_id)}</code>
          </div>
        `).join("")}
      </div>
    </div>
  `;
  bindDetails();
}

function renderRecords() {
  if (!state.records.length) {
    root.innerHTML = `<div class="empty">没有匹配的事件记录。</div>`;
    return;
  }
  root.innerHTML = `<div class="records">
    ${state.records.map((record, index) => {
      const changes = record.changes || [];
      const error = record.error;
      return `
        <button class="record ${error ? "failed" : ""}" data-bucket="records" data-detail="${index}">
          <span class="type">${esc(record.domain || record.event_type || record.scope)}</span>
          <strong>${esc(record.event_type || record.name || record.version_id)}</strong>
          <span>${esc(record.task_id || record.kind || "")}</span>
          ${changes.map((change) => `
            <div class="change">
              ${esc(change.entity_kind)}.${esc(change.field)}:
              <del>${esc(JSON.stringify(change.before))}</del>
              → <ins>${esc(JSON.stringify(change.after))}</ins>
            </div>`).join("")}
          ${error ? `<div class="change"><del>${esc(error.code)} · ${esc(error.message)}</del></div>` : ""}
        </button>
      `;
    }).join("")}
  </div>`;
  bindDetails();
}

async function loadMetrics() {
  try {
    const data = await api(`/api/studio/v2/runs/${runId}/metrics`);
    const values = [
      ["Tokens", data.totals.total_tokens],
      ["Cost", `$${Number(data.totals.cost_usd).toFixed(4)}`],
      ["Latency", `${Number(data.totals.latency_ms).toFixed(0)} ms`],
      ["Retries", data.totals.retries],
      ["Budget risks", data.budget_health.filter((item) => item.exceeded_dimensions.length).length],
    ];
    document.querySelector("#metric-strip").innerHTML = values.map(([label, value]) => `
      <div class="metric"><span>${esc(label)}</span><strong>${esc(value)}</strong></div>
    `).join("");
  } catch (error) {
    document.querySelector("#metric-strip").innerHTML =
      `<div class="metric"><span>Metrics</span><strong>${esc(error.message)}</strong></div>`;
  }
}

async function loadView({ append = false } = {}) {
  root.innerHTML = `<div class="empty">读取公开投影…</div>`;
  try {
    let data;
    if (state.view === "tasks") {
      const query = state.cursor ? `?cursor=${encodeURIComponent(state.cursor)}` : "";
      data = await api(`/api/studio/v2/runs/${runId}/task-graph${query}`);
      state.nodes = append ? [...state.nodes, ...data.nodes] : data.nodes;
      state.edges = append ? [...state.edges, ...data.edges] : data.edges;
      state.cursor = data.next_cursor;
      renderGraph();
    } else if (state.view === "evidence") {
      const query = state.cursor ? `?cursor=${encodeURIComponent(state.cursor)}` : "";
      data = await api(`/api/studio/v2/runs/${runId}/evidence-graph${query}`);
      state.nodes = append ? [...state.nodes, ...data.nodes] : data.nodes;
      state.edges = append ? [...state.edges, ...data.edges] : data.edges;
      state.cursor = data.next_cursor;
      renderGraph();
    } else if (state.view === "conflicts") {
      const query = state.cursor ? `?cursor=${encodeURIComponent(state.cursor)}` : "";
      data = await api(`/api/studio/v2/runs/${runId}/conflicts${query}`);
      state.nodes = append ? [...state.nodes, ...data.nodes] : data.nodes;
      state.edges = append ? [...state.edges, ...data.edges] : data.edges;
      state.cursor = data.next_cursor;
      renderGraph();
    } else if (state.view === "diffs") {
      const scheduler = state.schedulerDiffMore
        ? await api(`/api/studio/v2/runs/${runId}/state-diff?domain=scheduler&after_sequence=${state.schedulerDiffSequence}`)
        : { items: [], next_after_sequence: null };
      const evidence = state.evidenceDiffMore
        ? await api(`/api/studio/v2/runs/${runId}/state-diff?domain=evidence&after_sequence=${state.evidenceDiffSequence}`)
        : { items: [], next_after_sequence: null };
      const records = [...scheduler.items, ...evidence.items]
        .sort((left, right) => left.occurred_at.localeCompare(right.occurred_at));
      state.records = append ? [...state.records, ...records] : records;
      state.schedulerDiffMore = scheduler.next_after_sequence !== null;
      state.evidenceDiffMore = evidence.next_after_sequence !== null;
      if (scheduler.next_after_sequence !== null) {
        state.schedulerDiffSequence = scheduler.next_after_sequence;
      }
      if (evidence.next_after_sequence !== null) {
        state.evidenceDiffSequence = evidence.next_after_sequence;
      }
      state.cursor = state.schedulerDiffMore || state.evidenceDiffMore ? "more" : null;
      renderRecords();
    } else if (state.view === "errors") {
      const runtime = state.runtimeErrorMore
        ? await api(`/api/studio/v2/runs/${runId}/error-retry-chain?domain=runtime&after_sequence=${state.runtimeErrorSequence}`)
        : { items: [], next_after_sequence: null };
      const scheduler = state.schedulerErrorMore
        ? await api(`/api/studio/v2/runs/${runId}/error-retry-chain?domain=scheduler&after_sequence=${state.schedulerErrorSequence}`)
        : { items: [], next_after_sequence: null };
      const records = [
        ...runtime.items.map((item) => ({ ...item, domain: "runtime" })),
        ...scheduler.items.map((item) => ({ ...item, domain: "scheduler" })),
      ];
      state.records = append ? [...state.records, ...records] : records;
      state.runtimeErrorMore = runtime.next_after_sequence !== null;
      state.schedulerErrorMore = scheduler.next_after_sequence !== null;
      if (runtime.next_after_sequence !== null) {
        state.runtimeErrorSequence = runtime.next_after_sequence;
      }
      if (scheduler.next_after_sequence !== null) {
        state.schedulerErrorSequence = scheduler.next_after_sequence;
      }
      state.cursor = state.runtimeErrorMore || state.schedulerErrorMore ? "more" : null;
      renderRecords();
    } else {
      data = await api(`/api/studio/v2/runs/${runId}/components`);
      state.records = data.items;
      state.cursor = null;
      renderRecords();
    }
    loadMore.hidden = !state.cursor;
  } catch (error) {
    root.innerHTML = `<div class="error">${esc(error.message)}</div>`;
    loadMore.hidden = true;
  }
}

function resetView(view) {
  state.view = view;
  state.cursor = null;
  state.schedulerDiffSequence = 0;
  state.evidenceDiffSequence = 0;
  state.runtimeErrorSequence = 0;
  state.schedulerErrorSequence = 0;
  state.schedulerDiffMore = true;
  state.evidenceDiffMore = true;
  state.runtimeErrorMore = true;
  state.schedulerErrorMore = true;
  state.nodes = [];
  state.edges = [];
  state.records = [];
  [kicker.textContent, title.textContent] = viewMeta[view];
  detail.className = "detail-empty";
  detail.textContent = "选择任一节点、边、差异或错误，即可查看其字段和可追溯事件/制品。";
  document.querySelectorAll(".tab").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === view);
  });
  loadView();
}

document.querySelectorAll(".tab").forEach((button) => {
  button.addEventListener("click", () => resetView(button.dataset.view));
});
document.querySelector("#refresh-view").addEventListener("click", () => resetView(state.view));
loadMore.addEventListener("click", () => loadView({ append: true }));

loadMetrics();
loadView();
