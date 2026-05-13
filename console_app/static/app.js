const app = document.getElementById("app");

const state = {
  route: resolveRoute(window.location.pathname),
  runs: [],
  console: null,
  report: null,
  debug: null,
  loading: false,
  error: "",
  debugOpen: false,
  selectedTaskId: null,
  selectedSourceTab: "sources",
  pollTimer: null,
};

function resolveRoute(pathname) {
  if (pathname.startsWith("/console/")) {
    return { page: "console", researchId: pathname.split("/")[2] || "" };
  }
  if (pathname.startsWith("/report/")) {
    return { page: "report", researchId: pathname.split("/")[2] || "" };
  }
  return { page: "landing", researchId: document.body.dataset.researchId || "" };
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `HTTP ${response.status}`);
  }
  return response.json();
}

function navigate(path) {
  history.pushState({}, "", path);
  state.route = resolveRoute(path);
  clearPoll();
  bootstrap();
}

window.addEventListener("popstate", () => {
  state.route = resolveRoute(window.location.pathname);
  clearPoll();
  bootstrap();
});

function clearPoll() {
  if (state.pollTimer) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
}

async function bootstrap() {
  state.error = "";
  if (state.route.page === "landing") {
    await loadLanding();
  } else if (state.route.page === "console") {
    await loadConsole();
  } else if (state.route.page === "report") {
    await loadReport();
  }
}

async function loadLanding() {
  state.loading = true;
  render();
  try {
    state.runs = await api("/api/runs");
  } catch (error) {
    state.error = String(error.message || error);
  } finally {
    state.loading = false;
    render();
  }
}

async function loadConsole() {
  state.loading = true;
  render();
  try {
    state.console = await api(`/api/runs/${state.route.researchId}/console`);
    state.debug = await api(`/api/runs/${state.route.researchId}/debug`);
    state.selectedTaskId = state.console.active_task_id || state.console.root_task_id || firstTaskId(state.console.task_tree);
    if (["initializing", "running"].includes(state.console.status) || ["planning", "researching", "knowledge_updating", "writing"].includes(state.console.current_stage)) {
      state.pollTimer = setInterval(async () => {
        try {
          state.console = await api(`/api/runs/${state.route.researchId}/console`);
          state.debug = await api(`/api/runs/${state.route.researchId}/debug`);
          render();
          if (["completed", "failed"].includes(state.console.status)) {
            clearPoll();
          }
        } catch (error) {
          state.error = String(error.message || error);
          clearPoll();
          render();
        }
      }, 2000);
    }
  } catch (error) {
    state.error = String(error.message || error);
  } finally {
    state.loading = false;
    render();
  }
}

async function loadReport() {
  state.loading = true;
  render();
  try {
    state.report = await api(`/api/runs/${state.route.researchId}/report`);
  } catch (error) {
    state.error = String(error.message || error);
  } finally {
    state.loading = false;
    render();
  }
}

function firstTaskId(taskTree) {
  return Object.keys(taskTree || {})[0] || null;
}

function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function stageLabel(stage) {
  return {
    initializing: "初始化",
    planning: "规划",
    researching: "检索",
    knowledge_updating: "知识更新",
    writing: "写作",
    completed: "已完成",
    failed: "失败",
  }[stage] || stage;
}

function statusLabel(status) {
  return {
    initializing: "初始化",
    running: "运行中",
    active: "运行中",
    completed: "已完成",
    failed: "失败",
    pending: "待处理",
    ready: "就绪",
    skipped: "已跳过",
  }[status] || status;
}

function taskStatusLabel(status) {
  return {
    pending: "待处理",
    running: "运行中",
    completed: "已完成",
    failed: "失败",
    blocked: "阻塞",
    deferred: "暂缓",
    skipped: "已跳过",
    unknown: "未知",
  }[status] || status;
}

function progressStrip(currentStage) {
  const stages = ["planning", "researching", "knowledge_updating", "writing", "completed"];
  const currentIndex = Math.max(0, stages.indexOf(currentStage));
  return `
    <div class="progress-strip">
      ${stages.map((stage, index) => `
        <div class="progress-step ${index <= currentIndex ? "active" : ""}">
          <div class="progress-dot"></div>
          <span>${stageLabel(stage)}</span>
        </div>
      `).join("")}
    </div>
  `;
}

function landingView() {
  return `
    <main class="landing-shell">
      <section class="hero-card">
        <p class="eyebrow">DeepResearcher</p>
        <h1>研究控制台</h1>
        <p class="lede">
          用于观察规划、检索、知识积累和报告生成流程的多智能体研究控制台。
        </p>
        <form id="query-form" class="query-form">
          <label>
            研究问题
            <textarea name="query" rows="4" placeholder="比较 SK hynix、Samsung 和 Micron 在 HBM4 商业化进展上的差异。"></textarea>
          </label>
          <label>
            可选指令
            <textarea name="instructions" rows="3" placeholder="优先使用一手来源，关注尚未解决的冲突和时间线差异。"></textarea>
          </label>
          <label>
            研究深度
            <select name="depth">
              <option value="standard">标准</option>
              <option value="quick">快速</option>
              <option value="deep">深入</option>
            </select>
          </label>
          <button type="submit" class="primary-btn">开始研究</button>
        </form>
      </section>
      <section class="examples-grid">
        <div class="card">
          <h2>运行流程</h2>
          <ul class="flat-list">
            <li>Planner 将问题拆成任务树。</li>
            <li>Researcher 检索候选来源和段落。</li>
            <li>Distiller 抽取结构化 claims、facts、evidence、packs、gaps 和 conflicts。</li>
            <li>Session knowledge 跨轮次积累，并为检索上下文提供依据。</li>
            <li>Writer 基于会话中的 section packs 组装报告。</li>
          </ul>
        </div>
        <div class="card">
          <h2>示例问题</h2>
          <div class="example-list">
            ${[
              "比较 SK hynix、Samsung 和 Micron 关于 HBM4 量产时间线的说法。",
              "评估 TSMC、Samsung 和 Intel 的 2nm 制程进展说法是否与官方及行业来源一致。",
              "梳理当前 AI 加速器供应瓶颈，以及不同厂商之间尚未解决的证据缺口。"
            ].map(example => `<button class="example-chip" data-example="${escapeHtml(example)}">${escapeHtml(example)}</button>`).join("")}
          </div>
        </div>
        <div class="card">
          <h2>最近会话</h2>
          ${state.loading ? `<p class="muted">正在加载最近会话...</p>` : state.runs.length ? `
            <ul class="run-list">
              ${state.runs.map(run => `
                <li>
                  <button class="link-btn" data-nav="/console/${run.research_id}">${escapeHtml(run.query)}</button>
                  <span class="badge">${escapeHtml(statusLabel(run.status))}</span>
                </li>
              `).join("")}
            </ul>
          ` : `<p class="muted">暂无最近会话。</p>`}
        </div>
      </section>
    </main>
  `;
}

function consoleView() {
  const data = state.console;
  if (!data) {
    return emptyState("控制台数据尚不可用。");
  }
  const selectedTask = (data.task_tree || {})[state.selectedTaskId] || {};
  return `
    <main class="console-shell">
      <header class="console-header card">
        <div>
          <p class="eyebrow">研究控制台</p>
          <h1>${escapeHtml(data.query)}</h1>
          <div class="header-meta">
            <span class="badge ${data.status}">${escapeHtml(statusLabel(data.status))}</span>
            <span>第 ${data.current_round} 轮</span>
            <span>${stageLabel(data.current_stage)}</span>
            <span>已用 ${Math.round(data.elapsed_seconds)} 秒</span>
            ${data.resumed ? `<span class="badge resumed">已恢复</span>` : ""}
          </div>
        </div>
        <div class="header-actions">
          <button class="secondary-btn" ${data.has_report ? "" : "disabled"} data-nav="/report/${data.research_id}">打开报告</button>
          <button class="secondary-btn" data-action="toggle-debug">${state.debugOpen ? "关闭调试" : "打开调试"}</button>
        </div>
      </header>
      ${progressStrip(data.current_stage)}
      <section class="console-grid">
        <aside class="column column-left">
          <div class="card">
            <h2>任务树</h2>
            ${renderTaskTree(data.task_tree)}
          </div>
          <div class="card">
            <h2>选中任务</h2>
            ${selectedTask.id ? `
              <dl class="detail-list">
                <div><dt>标题</dt><dd>${escapeHtml(selectedTask.title || selectedTask.query || selectedTask.id)}</dd></div>
                <div><dt>状态</dt><dd>${escapeHtml(taskStatusLabel(selectedTask.status || "unknown"))}</dd></div>
                <div><dt>深度</dt><dd>${escapeHtml(selectedTask.depth)}</dd></div>
                <div><dt>类型</dt><dd>${escapeHtml(selectedTask.node_type || "unknown")}</dd></div>
                <div><dt>依据</dt><dd>${escapeHtml(selectedTask.rationale || "")}</dd></div>
              </dl>
            ` : `<p class="muted">请选择一个任务查看详情。</p>`}
          </div>
        </aside>
        <section class="column column-center">
          <div class="card">
            <h2>当前智能体</h2>
            <div class="agent-card">
              <p><strong>${escapeHtml(data.active_agent.name)}</strong></p>
              <p>${escapeHtml(statusLabel(data.active_agent.status))}</p>
              <p class="muted">${escapeHtml(data.active_agent.target)}</p>
              <p>${escapeHtml(data.active_agent.last_output_summary)}</p>
            </div>
          </div>
          <div class="card">
            <h2>当前上下文</h2>
            <div class="context-grid">
              ${contextCard("Planner 上下文", data.context_summary.planner)}
              ${contextCard("Researcher 上下文", data.context_summary.researcher)}
              ${contextCard("Writer 上下文", data.context_summary.writer)}
            </div>
          </div>
          <div class="card">
            <h2>时间线</h2>
            ${data.timeline.length ? `
              <div class="timeline-list">
                ${data.timeline.map(item => `
                  <div class="timeline-item">
                    <div class="timeline-top">
                      <span class="badge">${escapeHtml(item.event_type)}</span>
                      <time>${escapeHtml(item.timestamp)}</time>
                    </div>
                    <p>${escapeHtml(item.message || item.event_type)}</p>
                    ${Object.keys(item.payload || {}).length ? `<details><summary>详情</summary><pre>${escapeHtml(JSON.stringify(item.payload, null, 2))}</pre></details>` : ""}
                  </div>
                `).join("")}
              </div>
            ` : `<p class="muted">暂无事件时间线；在 trace 数据出现前，控制台会根据状态推断进度。</p>`}
          </div>
        </section>
        <aside class="column column-right">
          <div class="card">
            <h2>知识摘要</h2>
            <div class="metric-grid">
              ${metric("来源", data.knowledge_summary.source_count)}
              ${metric("主张", data.knowledge_summary.claim_count)}
              ${metric("事实", data.knowledge_summary.fact_count)}
              ${metric("证据", data.knowledge_summary.evidence_count)}
              ${metric("冲突", data.knowledge_summary.conflict_count)}
              ${metric("开放缺口", data.knowledge_summary.open_gap_count)}
              ${metric("章节证据包", data.knowledge_summary.section_pack_count)}
            </div>
          </div>
          <div class="card">
            <h2>覆盖率 / 缺口 / 冲突</h2>
            <div class="tabs">
              <div class="mini-section">
                <h3>覆盖率</h3>
                ${renderCoverage(data.latest_coverage_snapshot, data.section_packs)}
              </div>
              <div class="mini-section">
                <h3>开放缺口</h3>
                ${data.open_gaps.length ? `<ul class="flat-list">${data.open_gaps.map(gap => `<li>${escapeHtml(gap.gap_text || "")}</li>`).join("")}</ul>` : `<p class="muted">暂无开放缺口。</p>`}
              </div>
              <div class="mini-section">
                <h3>冲突</h3>
                ${data.conflicts.length ? `<ul class="flat-list">${data.conflicts.map(conflict => `<li>${escapeHtml(conflict.description || conflict.id || "")}</li>`).join("")}</ul>` : `<p class="muted">暂无活跃冲突。</p>`}
              </div>
            </div>
          </div>
          <div class="card">
            <h2>来源 / 证据包 / 详情</h2>
            <div class="segmented">
              <button class="${state.selectedSourceTab === "sources" ? "active" : ""}" data-tab="sources">来源</button>
              <button class="${state.selectedSourceTab === "packs" ? "active" : ""}" data-tab="packs">证据包</button>
              <button class="${state.selectedSourceTab === "detail" ? "active" : ""}" data-tab="detail">详情</button>
            </div>
            ${renderRightPanel(data)}
          </div>
        </aside>
      </section>
      ${state.debugOpen ? debugDrawer() : ""}
    </main>
  `;
}

function reportView() {
  const data = state.report;
  if (!data) {
    return emptyState("报告数据尚不可用。");
  }
  const sections = ((data.outline || {}).sections || []);
  return `
    <main class="report-shell">
      <header class="card report-header">
        <div>
          <p class="eyebrow">报告查看器</p>
          <h1>${escapeHtml(data.title || data.query)}</h1>
          <div class="header-meta">
            <span class="badge ${data.status}">${escapeHtml(statusLabel(data.status))}</span>
            <span>${escapeHtml(data.query)}</span>
          </div>
        </div>
        <div class="header-actions">
          <button class="secondary-btn" data-nav="/console/${data.research_id}">返回控制台</button>
        </div>
      </header>
      <section class="report-grid">
        <aside class="card">
          <h2>报告大纲</h2>
          ${sections.length ? `
            <ul class="outline-list">
              ${sections.map(section => `<li><a href="#${section.section_id}">${escapeHtml(section.title)}</a></li>`).join("")}
            </ul>
          ` : `<p class="muted">暂无结构化大纲。</p>`}
          <h2>报告元信息</h2>
          <div class="metric-grid">
            ${metric("来源", data.knowledge_summary.source_count)}
            ${metric("主张", data.knowledge_summary.claim_count)}
            ${metric("证据", data.knowledge_summary.evidence_count)}
            ${metric("开放缺口", data.knowledge_summary.open_gap_count)}
          </div>
        </aside>
        <article class="card report-content">
          ${sections.length ? sections.map(section => `
            <section id="${section.section_id}" class="report-section">
              <h2>${escapeHtml(section.title)}</h2>
              <p class="muted">${escapeHtml(section.goal || "")}</p>
              ${evidenceNote(section.section_id, data.section_packs)}
            </section>
          `).join("") : ""}
          <section class="markdown-body">
            ${renderMarkdown(data.markdown)}
          </section>
        </article>
      </section>
    </main>
  `;
}

function renderTaskTree(taskTree) {
  const tasks = Object.values(taskTree || {});
  if (!tasks.length) return `<p class="muted">暂无任务树。</p>`;
  return `
    <ul class="task-tree">
      ${tasks.sort((a, b) => (a.depth || 0) - (b.depth || 0)).map(task => `
        <li class="task-node ${state.selectedTaskId === task.id ? "selected" : ""}" style="--depth:${task.depth || 0}">
          <button data-task-id="${task.id}">
            <span class="task-title">${escapeHtml(task.title || task.query || task.id)}</span>
            <span class="badge">${escapeHtml(taskStatusLabel(task.status || "unknown"))}</span>
          </button>
        </li>
      `).join("")}
    </ul>
  `;
}

function renderCoverage(snapshot, packs) {
  if (!snapshot && !(packs || []).length) return `<p class="muted">暂无覆盖率数据。</p>`;
  const avg = snapshot?.avg_section_coverage ?? 0;
  return `
    <div class="coverage-box">
      <div class="coverage-bar"><span style="width:${Math.round(avg * 100)}%"></span></div>
      <p>平均覆盖率：${Math.round(avg * 100)}%</p>
      <p class="muted">充分性：${escapeHtml(snapshot?.sufficiency_level || "未知")}</p>
      ${(packs || []).length ? `<ul class="flat-list">${packs.slice(0, 5).map(pack => `<li>${escapeHtml(pack.section_title || pack.section_id)}: ${Math.round((pack.coverage_score || 0) * 100)}%</li>`).join("")}</ul>` : ""}
    </div>
  `;
}

function renderRightPanel(data) {
  if (state.selectedSourceTab === "sources") {
    return data.sources.length ? `<ul class="flat-list">${data.sources.slice(0, 8).map(source => `<li>${escapeHtml(source.title || source.url || source.source_id)}</li>`).join("")}</ul>` : `<p class="muted">暂无来源。</p>`;
  }
  if (state.selectedSourceTab === "packs") {
    return data.section_packs.length ? `<ul class="flat-list">${data.section_packs.slice(0, 8).map(pack => `<li>${escapeHtml(pack.section_title || pack.section_id)} · ${Math.round((pack.coverage_score || 0) * 100)}%</li>`).join("")}</ul>` : `<p class="muted">暂无证据包。</p>`;
  }
  return `
    <dl class="detail-list">
      <div><dt>当前轮次</dt><dd>${escapeHtml(data.current_round)}</dd></div>
      <div><dt>Planner 动作</dt><dd>${escapeHtml((data.planner_state || {}).action || "n/a")}</dd></div>
      <div><dt>当前任务</dt><dd>${escapeHtml(data.active_task_id || "n/a")}</dd></div>
    </dl>
  `;
}

function contextCard(title, payload) {
  return `
    <div class="mini-card">
      <h3>${escapeHtml(title)}</h3>
      ${Object.keys(payload || {}).length ? `<pre>${escapeHtml(JSON.stringify(payload, null, 2))}</pre>` : `<p class="muted">暂无上下文摘要。</p>`}
    </div>
  `;
}

function metric(label, value) {
  return `<div class="metric"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`;
}

function evidenceNote(sectionId, packs) {
  const pack = (packs || []).find(item => item.section_id === sectionId);
  if (!pack) return "";
  return `
    <div class="evidence-note">
      <p><strong>证据关联</strong></p>
      <p>${(pack.claim_ids || []).length} 条主张 · ${(pack.evidence_ids || []).length} 条证据 · ${(pack.conflict_ids || []).length} 条冲突</p>
    </div>
  `;
}

function renderMarkdown(markdown) {
  if (!markdown) return `<p class="muted">报告尚未生成。</p>`;
  return markdown
    .split("\n")
    .map(line => {
      if (line.startsWith("# ")) return `<h1>${escapeHtml(line.slice(2))}</h1>`;
      if (line.startsWith("## ")) return `<h2>${escapeHtml(line.slice(3))}</h2>`;
      if (line.startsWith("### ")) return `<h3>${escapeHtml(line.slice(4))}</h3>`;
      if (line.startsWith("- ")) return `<li>${escapeHtml(line.slice(2))}</li>`;
      return line.trim() ? `<p>${escapeHtml(line)}</p>` : "";
    })
    .join("");
}

function debugDrawer() {
  const debug = state.debug;
  if (!debug) return "";
  return `
    <aside class="debug-drawer">
      <div class="debug-header">
        <h2>调试 / Trace</h2>
        <button class="secondary-btn" data-action="toggle-debug">关闭</button>
      </div>
      <div class="debug-grid">
        <section class="card">
          <h3>状态</h3>
          <pre>${escapeHtml(JSON.stringify(debug.state_summary, null, 2))}</pre>
        </section>
        <section class="card">
          <h3>上下文</h3>
          <pre>${escapeHtml(JSON.stringify(debug.context_summary, null, 2))}</pre>
        </section>
        <section class="card">
          <h3>Trace</h3>
          <div class="timeline-list">
            ${(debug.trace || []).map(item => `
              <div class="timeline-item">
                <strong>${escapeHtml(item.event_type)}</strong>
                <p>${escapeHtml(item.message || "")}</p>
                <time>${escapeHtml(item.timestamp)}</time>
              </div>
            `).join("")}
          </div>
        </section>
      </div>
    </aside>
  `;
}

function emptyState(message) {
  return `<main class="empty-shell"><div class="card"><p class="muted">${escapeHtml(message)}</p></div></main>`;
}

function render() {
  if (state.error) {
    app.innerHTML = `<main class="empty-shell"><div class="card error-card"><h1>出现错误</h1><pre>${escapeHtml(state.error)}</pre></div></main>`;
    bindEvents();
    return;
  }
  if (state.loading && state.route.page !== "landing") {
    app.innerHTML = `<main class="empty-shell"><div class="card"><p class="muted">正在加载 ${escapeHtml(state.route.page)}...</p></div></main>`;
    bindEvents();
    return;
  }
  if (state.route.page === "landing") {
    app.innerHTML = landingView();
  } else if (state.route.page === "console") {
    app.innerHTML = consoleView();
  } else {
    app.innerHTML = reportView();
  }
  bindEvents();
}

function bindEvents() {
  document.querySelectorAll("[data-example]").forEach(button => {
    button.onclick = () => {
      const textarea = document.querySelector("textarea[name='query']");
      if (textarea) textarea.value = button.dataset.example || "";
    };
  });
  const form = document.getElementById("query-form");
  if (form) {
    form.onsubmit = async (event) => {
      event.preventDefault();
      const formData = new FormData(form);
      const query = String(formData.get("query") || "").trim();
      if (!query) return;
      try {
        const result = await api("/api/runs", {
          method: "POST",
          body: JSON.stringify({
            query,
            instructions: String(formData.get("instructions") || ""),
            depth: String(formData.get("depth") || "standard"),
          }),
        });
        navigate(result.console_url);
      } catch (error) {
        state.error = String(error.message || error);
        render();
      }
    };
  }
  document.querySelectorAll("[data-nav]").forEach(button => {
    button.onclick = () => navigate(button.dataset.nav);
  });
  document.querySelectorAll("[data-action='toggle-debug']").forEach(button => {
    button.onclick = () => {
      state.debugOpen = !state.debugOpen;
      render();
    };
  });
  document.querySelectorAll("[data-task-id]").forEach(button => {
    button.onclick = () => {
      state.selectedTaskId = button.dataset.taskId;
      render();
    };
  });
  document.querySelectorAll("[data-tab]").forEach(button => {
    button.onclick = () => {
      state.selectedSourceTab = button.dataset.tab;
      render();
    };
  });
}

bootstrap();
