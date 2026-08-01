import {
  arrayOf,
  describeApiError,
  escapeHtml,
  eventLabel,
  filterRuns,
  filterTasks,
  formatDateTime,
  formatDuration,
  formatNumber,
  formatPercent,
  formatUsd,
  itemStatusTone,
  isNamespacedIdentifier,
  normalizeReport,
  normalizeRunList,
  normalizeWorkspace,
  objectOf,
  pollingDelay,
  renderMarkdownSafe,
  reviewDimensionLabel,
  roleLabel,
  runStatusTone,
  safeExternalUrl,
  safeInternalPath,
  shouldPoll,
  stageLabel,
  statusLabel,
  taskChildren,
  taskKindLabel,
  taskRoots,
} from "./console_models.js?v=console-workspace-2-3";

const app = document.getElementById("app");

const state = {
  route: resolveRoute(window.location.pathname),
  system: null,
  runs: [],
  workspace: null,
  report: null,
  loading: true,
  refreshing: false,
  fatalError: null,
  banner: null,
  toast: null,
  activeView: readViewFromUrl(),
  selectedTaskId: null,
  runSearch: "",
  runStatus: "all",
  taskSearch: "",
  taskStatus: "all",
  evidenceView: "sections",
  timeline: {
    items: null,
    next: null,
    search: "",
    eventTypes: "",
    errorOnly: false,
    loading: false,
  },
  modal: null,
  actionPending: false,
  pollTimer: null,
  pollFailures: 0,
  requestGeneration: 0,
};

class ApiError extends Error {
  constructor(message, status, detail) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

function resolveRoute(pathname) {
  const parts = pathname.split("/").filter(Boolean);
  if (parts[0] === "console" && parts[1]) {
    return { page: "console", researchId: decodeURIComponent(parts[1]) };
  }
  if (parts[0] === "report" && parts[1]) {
    return { page: "report", researchId: decodeURIComponent(parts[1]) };
  }
  return { page: "landing", researchId: "" };
}

function readViewFromUrl() {
  const view = new URLSearchParams(window.location.search).get("view");
  return ["overview", "tasks", "evidence", "report", "timeline"].includes(view)
    ? view
    : "overview";
}

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (options.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  headers.set("Accept", "application/json");
  const response = await fetch(path, {
    credentials: "same-origin",
    ...options,
    headers,
  });
  const contentType = response.headers.get("content-type") || "";
  let payload = null;
  if (contentType.includes("application/json")) {
    payload = await response.json();
  } else {
    const text = await response.text();
    payload = text ? { detail: text } : null;
  }
  if (!response.ok) {
    const detail = payload?.detail || payload?.message || `HTTP ${response.status}`;
    throw new ApiError(String(detail), response.status, payload);
  }
  return payload;
}

function navigate(path) {
  const target = safeInternalPath(path);
  const url = new URL(target, window.location.origin);
  history.pushState({}, "", `${url.pathname}${url.search}`);
  clearPoll();
  state.route = resolveRoute(url.pathname);
  state.activeView = readViewFromUrl();
  state.fatalError = null;
  state.banner = null;
  state.modal = null;
  state.timeline.items = null;
  state.timeline.next = null;
  state.workspace = null;
  state.report = null;
  bootstrap();
}

window.addEventListener("popstate", () => {
  clearPoll();
  state.route = resolveRoute(window.location.pathname);
  state.activeView = readViewFromUrl();
  state.fatalError = null;
  state.banner = null;
  state.modal = null;
  state.timeline.items = null;
  state.workspace = null;
  state.report = null;
  bootstrap();
});

document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    clearPoll();
    return;
  }
  if (state.route.page === "console" && shouldPoll(state.workspace)) {
    refreshWorkspace({ quiet: true, immediate: true });
  }
});

function clearPoll() {
  if (state.pollTimer) {
    window.clearTimeout(state.pollTimer);
    state.pollTimer = null;
  }
}

function schedulePoll() {
  clearPoll();
  if (
    state.route.page !== "console"
    || document.hidden
    || !shouldPoll(state.workspace)
  ) {
    return;
  }
  const generation = state.requestGeneration;
  state.pollTimer = window.setTimeout(async () => {
    if (generation !== state.requestGeneration) return;
    await refreshWorkspace({ quiet: true });
  }, pollingDelay(state.workspace, state.pollFailures));
}

async function bootstrap() {
  const generation = ++state.requestGeneration;
  state.loading = true;
  render();
  try {
    if (state.route.page === "landing") {
      const [system, runs] = await Promise.all([
        api("/api/health"),
        api("/api/runs"),
      ]);
      if (generation !== state.requestGeneration) return;
      state.system = system;
      state.runs = normalizeRunList(runs);
    } else if (state.route.page === "console") {
      const payload = await api(
        `/api/runs/${encodeURIComponent(state.route.researchId)}/console`,
      );
      if (generation !== state.requestGeneration) return;
      applyWorkspace(normalizeWorkspace(payload));
    } else {
      const payload = await api(
        `/api/runs/${encodeURIComponent(state.route.researchId)}/report`,
      );
      if (generation !== state.requestGeneration) return;
      state.report = normalizeReport(payload);
    }
  } catch (error) {
    if (generation !== state.requestGeneration) return;
    state.fatalError = error;
  } finally {
    if (generation !== state.requestGeneration) return;
    state.loading = false;
    render();
    schedulePoll();
  }
}

function applyWorkspace(workspace) {
  state.workspace = workspace;
  const tasks = workspace.scheduler.tasks;
  const selectedStillExists = tasks.some(
    (task) => task.task_id === state.selectedTaskId,
  );
  if (!selectedStillExists) {
    state.selectedTaskId = (
      workspace.runtime.active_task_id
      || tasks[0]?.task_id
      || null
    );
  }
}

async function refreshWorkspace({ quiet = false, immediate = false } = {}) {
  if (!state.route.researchId || state.refreshing) return;
  clearPoll();
  captureTransientInputs();
  state.refreshing = true;
  if (!quiet) render();
  try {
    const payload = await api(
      `/api/runs/${encodeURIComponent(state.route.researchId)}/console`,
      { cache: "no-store" },
    );
    applyWorkspace(normalizeWorkspace(payload));
    state.pollFailures = 0;
    state.banner = null;
    render({ preserveFocus: true });
  } catch (error) {
    state.pollFailures += 1;
    state.banner = {
      tone: "danger",
      title: "暂时无法刷新运行投影",
      message: describeApiError(error),
      retry: true,
    };
    render({ preserveFocus: true });
  } finally {
    state.refreshing = false;
    if (!immediate || shouldPoll(state.workspace)) schedulePoll();
  }
}

function captureTransientInputs() {
  state.runSearch = document.querySelector("#run-search")?.value ?? state.runSearch;
  state.runStatus = document.querySelector("#run-status")?.value ?? state.runStatus;
  state.taskSearch = document.querySelector("#task-search")?.value ?? state.taskSearch;
  state.taskStatus = document.querySelector("#task-status")?.value ?? state.taskStatus;
  state.timeline.search = document.querySelector("#timeline-search")?.value ?? state.timeline.search;
  state.timeline.eventTypes = document.querySelector("#timeline-types")?.value ?? state.timeline.eventTypes;
  state.timeline.errorOnly = document.querySelector("#timeline-errors")?.checked ?? state.timeline.errorOnly;
}

function chromeHeader(context = "") {
  return `
    <header class="app-topbar">
      <a class="brand" href="/" data-nav="/">
        <span class="brand-mark" aria-hidden="true"><i></i><i></i><i></i></span>
        <span>
          <strong>DeepResearcher</strong>
          <small>Evidence-first runtime</small>
        </span>
      </a>
      <div class="topbar-context">
        ${context ? `<span class="context-label">${escapeHtml(context)}</span>` : ""}
        <span class="schema-chip">ConsoleWorkspace@2</span>
      </div>
    </header>
  `;
}

function landingView() {
  const providers = state.system?.live_providers || {};
  const providerReady = providers.model_configured && providers.search_configured;
  const visibleRuns = filterRuns(state.runs, state.runSearch, state.runStatus);
  return `
    <div class="app-frame landing-frame">
      ${chromeHeader("研究入口")}
      <main id="main-content" class="landing-main">
        <section class="landing-hero">
          <div class="hero-copy">
            <p class="eyebrow">BACKGROUND001 · NATIVE RESEARCH RUNTIME</p>
            <h1>从问题出发，<br /><span>以证据结束。</span></h1>
            <p class="hero-lede">
              监督器动态拆解任务，受控工作池检索材料，独立验证器核验主张，
              合成器只读取已验证证据，审查器在报告交付前执行完整质量门。
            </p>
            <div class="architecture-rail" aria-label="Background001 五个运行角色">
              ${[
                ["01", "研究监督器", "计划与收敛"],
                ["02", "研究工作池", "检索与抽取"],
                ["03", "证据验证器", "独立核验"],
                ["04", "证据合成器", "限定写作"],
                ["05", "报告审查器", "评分与修复"],
              ].map(([number, title, detail]) => `
                <div class="architecture-node">
                  <span>${number}</span>
                  <strong>${title}</strong>
                  <small>${detail}</small>
                </div>
              `).join("")}
            </div>
          </div>
          <section class="create-panel" aria-labelledby="create-heading">
            <div class="panel-kicker">
              <span class="live-dot" aria-hidden="true"></span>
              <span>${providerReady ? "实时提供方已就绪" : "本地控制台已就绪"}</span>
            </div>
            <h2 id="create-heading">创建新的研究运行</h2>
            <p class="panel-intro">每个运行拥有独立任务 DAG、证据图、报告修订和可重放 Trace。</p>
            ${!providerReady ? `
              <div class="inline-notice warning">
                <strong>实时运行凭据尚不完整</strong>
                <span>页面可正常使用；创建实时研究前需配置模型与至少一个搜索提供方。</span>
              </div>
            ` : ""}
            <form id="create-run-form" class="stack-form">
              <label>
                <span>研究问题</span>
                <textarea
                  name="query"
                  rows="5"
                  maxlength="8000"
                  required
                  placeholder="例如：比较主要 HBM4 厂商的量产时间线，并核验各项说法的来源与冲突。"
                ></textarea>
                <small>清楚描述需要比较、验证或解释的核心问题。</small>
              </label>
              <label>
                <span>研究约束与偏好 <em>可选</em></span>
                <textarea
                  name="instructions"
                  rows="3"
                  maxlength="16000"
                  placeholder="优先采用一手来源；保留未解决冲突；明确区分事实与推断。"
                ></textarea>
              </label>
              <fieldset class="depth-fieldset">
                <legend>研究深度</legend>
                ${[
                  ["quick", "快速", "更紧预算，优先形成有界结论"],
                  ["standard", "标准", "平衡来源覆盖与运行成本"],
                  ["deep", "深入", "扩大检索、核验和修复预算"],
                ].map(([value, title, detail], index) => `
                  <label class="depth-option">
                    <input type="radio" name="depth" value="${value}" ${index === 1 ? "checked" : ""} />
                    <span><strong>${title}</strong><small>${detail}</small></span>
                  </label>
                `).join("")}
              </fieldset>
              <button class="primary-action wide" type="submit">
                <span>启动证据研究</span><span aria-hidden="true">↗</span>
              </button>
              <p class="form-error" id="create-error" role="alert"></p>
            </form>
          </section>
        </section>

        <section class="runs-section" aria-labelledby="runs-heading">
          <div class="section-heading">
            <div>
              <p class="eyebrow">DURABLE RUN CATALOG</p>
              <h2 id="runs-heading">最近运行</h2>
              <p>运行、证据和报告状态均从持久化投影读取。</p>
            </div>
            <button class="quiet-action" type="button" data-action="refresh-runs">
              ${state.refreshing ? "正在刷新…" : "刷新列表"}
            </button>
          </div>
          <div class="filter-bar">
            <label class="search-control">
              <span class="sr-only">搜索运行</span>
              <input id="run-search" type="search" value="${escapeHtml(state.runSearch)}" placeholder="搜索研究问题" />
            </label>
            <label>
              <span class="sr-only">筛选运行状态</span>
              <select id="run-status">
                ${[
                  ["all", "全部状态"],
                  ["queued", "排队中"],
                  ["running", "运行中"],
                  ["waiting_approval", "等待审批"],
                  ["completed", "已完成"],
                  ["failed", "失败"],
                  ["cancelled", "已取消"],
                ].map(([value, label]) => `<option value="${value}" ${state.runStatus === value ? "selected" : ""}>${label}</option>`).join("")}
              </select>
            </label>
            <span class="filter-count">${visibleRuns.length} / ${state.runs.length}</span>
          </div>
          ${visibleRuns.length ? `
            <div class="run-catalog">
              ${visibleRuns.map(runCard).join("")}
            </div>
          ` : emptyPanel(
            state.runs.length ? "没有匹配的运行" : "还没有研究运行",
            state.runs.length ? "调整搜索条件或状态筛选。" : "使用上方表单创建第一个证据研究运行。",
          )}
        </section>
      </main>
      ${footer()}
    </div>
  `;
}

function runCard(run) {
  return `
    <article class="run-card">
      <div class="run-card-top">
        ${badge(statusLabel(run.status), runStatusTone(run.status))}
        <time datetime="${escapeHtml(run.updated_at)}">${formatDateTime(run.updated_at)}</time>
      </div>
      <h3><a href="${safeInternalPath(run.console_url)}" data-nav="${safeInternalPath(run.console_url)}">${escapeHtml(run.query)}</a></h3>
      <div class="run-card-meta">
        <span>${depthLabel(run.depth)}</span>
        <span>第 ${formatNumber(run.current_round)} 轮</span>
        <span>${escapeHtml(stageLabel(run.current_stage))}</span>
        ${run.resumed ? `<span>已恢复</span>` : ""}
      </div>
      <div class="run-card-footer">
        <code>${escapeHtml(run.run_id)}</code>
        <div>
          ${run.has_report ? `<a class="text-action" href="${safeInternalPath(run.report_url)}" data-nav="${safeInternalPath(run.report_url)}">报告</a>` : ""}
          <a class="text-action" href="${safeInternalPath(run.console_url)}" data-nav="${safeInternalPath(run.console_url)}">打开运行 →</a>
        </div>
      </div>
    </article>
  `;
}

function consoleView() {
  const data = state.workspace;
  if (!data) return fatalOrEmpty("运行投影尚不可用。");
  const identity = data.identity;
  const runtime = data.runtime;
  return `
    <div class="app-frame">
      ${chromeHeader("运行工作台")}
      <main id="main-content" class="console-main">
        ${renderBanner()}
        <header class="run-masthead">
          <div class="run-heading">
            <div class="run-heading-line">
              ${badge(statusLabel(runtime.status), runStatusTone(runtime.status))}
              <span>${escapeHtml(stageLabel(runtime.current_stage))}</span>
              ${identity.resumed ? badge("恢复运行", "warning") : ""}
            </div>
            <h1>${escapeHtml(identity.query)}</h1>
            <div class="identity-line">
              <span>${depthLabel(identity.depth)}</span>
              <span>第 ${formatNumber(runtime.current_round)} 轮</span>
              <span>${formatDuration(runtime.elapsed_seconds)}</span>
              <button class="copy-id" type="button" data-copy="${escapeHtml(identity.run_id)}" title="复制 Run ID">
                Run ${escapeHtml(shortId(identity.run_id))}
              </button>
              <span>投影 r${formatNumber(data.scheduler.projection_revision)}</span>
            </div>
          </div>
          <div class="masthead-actions">
            <button class="quiet-action" type="button" data-action="refresh-workspace" ${state.refreshing ? "disabled" : ""}>
              ${state.refreshing ? "刷新中…" : "刷新"}
            </button>
            ${data.actions.can_approve ? `<button class="warning-action" type="button" data-action="open-approve">处理审批</button>` : ""}
            ${data.actions.can_cancel ? `<button class="danger-action ghost" type="button" data-action="open-cancel">取消运行</button>` : ""}
            ${data.reporting.artifact_ready ? `<a class="secondary-action" href="${safeInternalPath(data.navigation.report_url)}" data-nav="${safeInternalPath(data.navigation.report_url)}">打开报告</a>` : ""}
            <a class="primary-action" href="${safeInternalPath(data.navigation.studio_url)}">高级 Studio ↗</a>
          </div>
        </header>

        ${approvalCallout(data)}
        ${runtimeErrorCallout(runtime, data.scheduler)}
        ${progressRail(runtime.progress)}
        ${roleRail(runtime.roles, runtime.active_role_id)}

        <nav class="workspace-tabs" aria-label="运行工作台视图">
          ${[
            ["overview", "总览"],
            ["tasks", `任务 ${data.scheduler.tasks.length}`],
            ["evidence", `证据 ${data.evidence.knowledge.evidence_count}`],
            ["report", `报告 r${data.reporting.revision_count}`],
            ["timeline", `时间线 ${data.timeline.length}`],
          ].map(([view, label]) => `
            <button
              type="button"
              data-view="${view}"
              class="${state.activeView === view ? "active" : ""}"
              aria-current="${state.activeView === view ? "page" : "false"}"
            >${escapeHtml(label)}</button>
          `).join("")}
        </nav>

        <section class="workspace-view">
          ${renderConsoleView(data)}
        </section>
      </main>
      ${footer(data)}
      ${actionDialog(data)}
      ${toast()}
    </div>
  `;
}

function renderConsoleView(data) {
  if (state.activeView === "tasks") return tasksView(data);
  if (state.activeView === "evidence") return evidenceView(data);
  if (state.activeView === "report") return reportLifecycleView(data);
  if (state.activeView === "timeline") return timelineView(data);
  return overviewView(data);
}

function overviewView(data) {
  const { runtime, scheduler, evidence, reporting } = data;
  const coverage = evidence.coverage;
  const activeRole = runtime.roles.find((role) => role.role_id === runtime.active_role_id);
  const activeTask = scheduler.tasks.find((task) => task.task_id === runtime.active_task_id);
  const blockers = (
    coverage.gap_section_ids.length
    + coverage.blocked_high_impact_claim_ids.length
    + coverage.severe_conflict_ids.length
  );
  return `
    <div class="overview-grid">
      <section class="metric-band" aria-label="运行关键指标">
        ${metricCard("调度器", statusLabel(scheduler.status), `${scheduler.tasks.length} 个任务 · 并发 ${scheduler.max_concurrency}`, itemStatusTone(scheduler.status))}
        ${metricCard("章节就绪", formatPercent(coverage.completion_ratio), `${coverage.complete_count} / ${coverage.required_count} 个必需章节`, coverage.ready_for_reporting ? "success" : "warning")}
        ${metricCard("验证证据", evidence.knowledge.evidence_count, `${evidence.knowledge.claim_count} 条主张 · ${evidence.packets.reduce((total, packet) => total + packet.citations.length, 0)} 条引用`, "info")}
        ${metricCard("报告修订", reporting.revision_count, reporting.latest_review ? `最近审查：${statusLabel(reporting.latest_review.decision)}` : "尚未进入审查", reporting.artifact_ready ? "success" : "neutral")}
        ${metricCard("开放阻塞", blockers, `${evidence.gaps.length} 缺口 · ${evidence.conflicts.length} 冲突`, blockers ? "danger" : "success")}
      </section>

      <div class="overview-columns">
        <div class="overview-primary">
          <section class="surface-card live-operation">
            <div class="card-heading">
              <div>
                <p class="eyebrow">LIVE OPERATION</p>
                <h2>当前运行责任</h2>
              </div>
              ${activeRole ? badge(statusLabel(activeRole.status), itemStatusTone(activeRole.status)) : badge(statusLabel(runtime.status), runStatusTone(runtime.status))}
            </div>
            ${activeRole ? `
              <div class="active-role-block">
                <div class="role-monogram">${escapeHtml(roleInitials(activeRole.role_id))}</div>
                <div>
                  <h3>${escapeHtml(roleLabel(activeRole.role_id, activeRole.label))}</h3>
                  <p>${escapeHtml(activeTask?.title || activeRole.target || stageLabel(runtime.current_stage))}</p>
                  ${activeTask ? `<button class="text-action" type="button" data-open-task="${escapeHtml(activeTask.task_id)}">查看任务详情 →</button>` : ""}
                </div>
              </div>
            ` : `
              <div class="active-role-block terminal">
                <div class="role-monogram">✓</div>
                <div><h3>${escapeHtml(statusLabel(runtime.status))}</h3><p>${escapeHtml(stageLabel(runtime.current_stage))}</p></div>
              </div>
            `}
            <div class="decision-box">
              <span>最近收敛决策</span>
              <strong>${escapeHtml(runtime.decision ? statusLabel(runtime.decision) : "尚未形成")}</strong>
              ${runtime.decision_reasons.length ? `<ul>${runtime.decision_reasons.map((reason) => `<li>${escapeHtml(reason)}</li>`).join("")}</ul>` : `<p>监督器完成首轮评估后，这里会显示结构化决策依据。</p>`}
            </div>
          </section>

          <section class="surface-card">
            <div class="card-heading">
              <div>
                <p class="eyebrow">SECTION READINESS</p>
                <h2>章节覆盖与引用就绪度</h2>
              </div>
              <button class="text-action" type="button" data-view="evidence">打开证据视图 →</button>
            </div>
            ${sectionReadinessList(evidence.sections)}
          </section>
        </div>

        <aside class="overview-secondary">
          <section class="surface-card">
            <div class="card-heading compact">
              <div><p class="eyebrow">SCHEDULER</p><h2>任务分布</h2></div>
              <span>r${formatNumber(scheduler.projection_revision)}</span>
            </div>
            ${taskDistribution(scheduler.task_counts, scheduler.tasks.length)}
            <dl class="compact-details">
              <div><dt>运行状态</dt><dd>${escapeHtml(statusLabel(scheduler.status))}</dd></div>
              <div><dt>活跃任务</dt><dd>${scheduler.active_task_ids.length}</dd></div>
              <div><dt>就绪任务</dt><dd>${scheduler.ready_task_ids.length}</dd></div>
              <div><dt>等待审批</dt><dd>${scheduler.waiting_approval_task_ids.length}</dd></div>
            </dl>
          </section>
          <section class="surface-card">
            <div class="card-heading compact">
              <div><p class="eyebrow">BLOCKERS</p><h2>收敛阻塞项</h2></div>
              ${badge(blockers ? `${blockers} 项` : "无阻塞", blockers ? "danger" : "success")}
            </div>
            ${blockerSummary(evidence)}
          </section>
          <section class="surface-card">
            <div class="card-heading compact">
              <div><p class="eyebrow">RUN IDENTITY</p><h2>运行边界</h2></div>
            </div>
            ${identityDetails(data.identity)}
          </section>
        </aside>
      </div>
    </div>
  `;
}

function tasksView(data) {
  const tasks = data.scheduler.tasks;
  const filtered = filterTasks(tasks, state.taskSearch, state.taskStatus);
  const selected = tasks.find((task) => task.task_id === state.selectedTaskId) || null;
  const filtering = Boolean(state.taskSearch.trim()) || state.taskStatus !== "all";
  return `
    <div class="tasks-layout">
      <section class="surface-card task-browser">
        <div class="card-heading">
          <div>
            <p class="eyebrow">DURABLE TASK DAG</p>
            <h2>任务结构</h2>
            <p>${filtered.length} / ${tasks.length} 个调度任务</p>
          </div>
          ${badge(statusLabel(data.scheduler.status), itemStatusTone(data.scheduler.status))}
        </div>
        <div class="filter-bar task-filters">
          <label class="search-control">
            <span class="sr-only">搜索任务</span>
            <input id="task-search" type="search" value="${escapeHtml(state.taskSearch)}" placeholder="标题、目标、标签或 ID" />
          </label>
          <label>
            <span class="sr-only">筛选任务状态</span>
            <select id="task-status">
              ${taskStatusOptions(tasks)}
            </select>
          </label>
        </div>
        <div class="task-dag" role="tree">
          ${filtered.length ? (
            filtering
              ? filtered.map((task) => taskRow(task, false)).join("")
              : renderTaskTree(tasks)
          ) : emptyPanel("没有匹配的任务", "调整任务搜索或状态筛选。", true)}
        </div>
      </section>
      <aside class="surface-card task-inspector">
        ${selected ? taskInspector(selected, tasks) : emptyPanel("请选择任务", "从任务结构中选择一个节点查看调度、预算和制品详情。", true)}
      </aside>
    </div>
  `;
}

function renderTaskTree(tasks) {
  const children = taskChildren(tasks);
  const roots = taskRoots(tasks);
  const renderBranch = (task, visited = new Set()) => {
    if (visited.has(task.task_id)) return "";
    const nextVisited = new Set(visited);
    nextVisited.add(task.task_id);
    const descendants = children.get(task.task_id) || [];
    return `
      <div class="task-branch" role="treeitem" aria-level="${Number(task.depth || 0) + 1}">
        ${taskRow(task, descendants.length > 0)}
        ${descendants.length ? `<div class="task-children" role="group">${descendants.map((child) => renderBranch(child, nextVisited)).join("")}</div>` : ""}
      </div>
    `;
  };
  return roots.map((task) => renderBranch(task)).join("");
}

function taskRow(task, hasChildren) {
  return `
    <button
      type="button"
      class="task-row ${state.selectedTaskId === task.task_id ? "selected" : ""}"
      data-task-id="${escapeHtml(task.task_id)}"
      aria-selected="${state.selectedTaskId === task.task_id}"
    >
      <span class="task-connector" aria-hidden="true">${hasChildren ? "◇" : "·"}</span>
      <span class="task-row-copy">
        <strong>${escapeHtml(task.title)}</strong>
        <small>${escapeHtml(taskKindLabel(task.kind))} · 尝试 ${task.attempt}/${task.max_attempts}</small>
      </span>
      ${badge(statusLabel(task.status), itemStatusTone(task.status))}
    </button>
  `;
}

function taskInspector(task, tasks) {
  const parent = tasks.find((item) => item.task_id === task.parent_task_id);
  const dependencies = task.dependency_task_ids
    .map((id) => tasks.find((item) => item.task_id === id))
    .filter(Boolean);
  return `
    <div class="inspector-heading">
      <div>
        <p class="eyebrow">${escapeHtml(taskKindLabel(task.kind))}</p>
        <h2>${escapeHtml(task.title)}</h2>
      </div>
      ${badge(statusLabel(task.status), itemStatusTone(task.status))}
    </div>
    <p class="task-goal">${escapeHtml(task.goal)}</p>
    ${task.error_ref ? `<div class="inline-notice danger"><strong>任务错误</strong><code>${escapeHtml(task.error_ref)}</code></div>` : ""}
    ${task.approval ? approvalDetail(task.approval) : ""}
    <section class="inspector-section">
      <h3>调度关系</h3>
      <dl class="detail-grid">
        <div><dt>任务 ID</dt><dd><code>${escapeHtml(task.task_id)}</code></dd></div>
        <div><dt>父任务</dt><dd>${parent ? `<button class="inline-code-button" type="button" data-task-id="${escapeHtml(parent.task_id)}">${escapeHtml(parent.title)}</button>` : `<code>${escapeHtml(task.parent_task_id || "root")}</code>`}</dd></div>
        <div><dt>优先级</dt><dd>${formatNumber(task.priority, 2)}</dd></div>
        <div><dt>创建者</dt><dd>${escapeHtml(task.created_by)}</dd></div>
        <div><dt>分配角色</dt><dd>${escapeHtml(task.assigned_actor_id || "由工作池动态分配")}</dd></div>
        <div><dt>租约</dt><dd>${task.lease_owner ? `${escapeHtml(task.lease_owner)} · ${formatDateTime(task.lease_expires_at)}` : "无活动租约"}</dd></div>
        <div><dt>输出契约</dt><dd><code>${escapeHtml(task.expected_output_schema)}</code></dd></div>
        <div><dt>结果</dt><dd><code>${escapeHtml(task.result_id || "尚未生成")}</code></dd></div>
      </dl>
      ${dependencies.length ? `<div class="dependency-list"><span>依赖</span>${dependencies.map((item) => `<button type="button" data-task-id="${escapeHtml(item.task_id)}">${escapeHtml(item.title)}</button>`).join("")}</div>` : ""}
    </section>
    <section class="inspector-section">
      <h3>预算与消耗</h3>
      ${budgetTable(task.budget, task.budget_usage)}
    </section>
    <section class="inspector-section">
      <h3>制品与输入</h3>
      ${artifactList("输入制品", task.input_artifact_ids)}
      ${artifactList("输出制品", task.output_artifact_ids)}
    </section>
    <section class="inspector-section">
      <h3>约束与工具契约</h3>
      <details class="json-details">
        <summary>查看结构化约束</summary>
        <pre>${escapeHtml(JSON.stringify(objectOf(task.constraints), null, 2))}</pre>
      </details>
      ${task.tags.length ? `<div class="tag-list">${task.tags.map((tag) => `<span>${escapeHtml(tag)}</span>`).join("")}</div>` : ""}
    </section>
  `;
}

function evidenceView(data) {
  const evidence = data.evidence;
  const coverage = evidence.coverage;
  return `
    <div class="evidence-layout">
      <section class="coverage-hero surface-card">
        <div>
          <p class="eyebrow">VERIFIED-ONLY READ BOUNDARY</p>
          <h2>${coverage.ready_for_reporting ? "证据门已通过" : "证据仍在收敛"}</h2>
          <p>
            ${coverage.complete_count} / ${coverage.required_count} 个必需章节完成；
            ${coverage.blocked_high_impact_claim_ids.length} 条高影响主张受阻；
            ${coverage.severe_conflict_ids.length} 条严重冲突未消解。
          </p>
        </div>
        <div class="coverage-orbit ${coverage.ready_for_reporting ? "ready" : ""}" style="--coverage:${Math.round(coverage.completion_ratio * 360)}deg">
          <div><strong>${formatPercent(coverage.completion_ratio)}</strong><span>章节完成</span></div>
        </div>
      </section>

      <nav class="subtabs" aria-label="证据视图">
        ${[
          ["sections", `章节 ${evidence.sections.length}`],
          ["verified", `已验证主张 ${evidence.packets.reduce((sum, packet) => sum + packet.verified_claims.length, 0)}`],
          ["sources", `来源 ${evidence.sources.length}`],
          ["risks", `缺口与冲突 ${evidence.gaps.length + evidence.conflicts.length}`],
        ].map(([view, label]) => `<button type="button" data-evidence-view="${view}" class="${state.evidenceView === view ? "active" : ""}">${escapeHtml(label)}</button>`).join("")}
      </nav>

      <section class="surface-card evidence-content">
        ${renderEvidenceContent(evidence)}
      </section>
    </div>
  `;
}

function renderEvidenceContent(evidence) {
  if (state.evidenceView === "verified") return verifiedPackets(evidence.packets);
  if (state.evidenceView === "sources") return sourcesView(evidence.sources);
  if (state.evidenceView === "risks") return risksView(evidence);
  return `
    <div class="card-heading">
      <div><p class="eyebrow">SECTION COVERAGE</p><h2>报告章节就绪度</h2></div>
      ${badge(evidence.coverage.ready_for_reporting ? "允许进入报告" : "尚未通过证据门", evidence.coverage.ready_for_reporting ? "success" : "warning")}
    </div>
    ${sectionCards(evidence.sections)}
  `;
}

function verifiedPackets(packets) {
  if (!packets.length) return emptyPanel("尚无已验证证据包", "独立验证完成后，写作边界会在这里发布只包含已验证主张的证据包。", true);
  return `
    <div class="card-heading">
      <div><p class="eyebrow">WRITER EVIDENCE PACKETS</p><h2>合成器可读取的证据</h2></div>
      ${badge(`${packets.length} 个不可变包`, "success")}
    </div>
    <div class="packet-list">
      ${packets.map((packet) => `
        <article class="packet-card">
          <div class="packet-header">
            <div>
              <strong>${escapeHtml(packet.packet_id)}</strong>
              <span>${formatDateTime(packet.created_at)}</span>
            </div>
            <span>${packet.verified_claims.length} 主张 · ${packet.citations.length} 引用</span>
          </div>
          <div class="verified-claim-list">
            ${packet.verified_claims.length ? packet.verified_claims.map((claim) => `
              <details class="verified-claim">
                <summary>
                  <span>${escapeHtml(claim.statement)}</span>
                  ${claim.high_impact ? badge("高影响", "warning") : badge(`${claim.citation_ids.length} 引用`, "success")}
                </summary>
                <div class="claim-citations">
                  ${packet.citations.filter((citation) => claim.citation_ids.includes(citation.citation_id)).map(citationCard).join("") || `<p class="muted-copy">该主张未在当前包中关联可显示引用。</p>`}
                </div>
              </details>
            `).join("") : `<p class="muted-copy">该包不包含可写入的已验证主张。</p>`}
          </div>
        </article>
      `).join("")}
    </div>
  `;
}

function citationCard(citation) {
  const url = safeExternalUrl(citation.canonical_url);
  return `
    <article class="citation-card">
      <div>
        <strong>${escapeHtml(citation.source_title)}</strong>
        <span>${escapeHtml(citation.publisher || citation.locator)}</span>
      </div>
      <blockquote>“${escapeHtml(citation.quote)}”</blockquote>
      ${url ? `<a href="${escapeHtml(url)}" target="_blank" rel="noreferrer noopener">查看来源 ↗</a>` : `<code>${escapeHtml(citation.canonical_url)}</code>`}
    </article>
  `;
}

function sourcesView(sources) {
  if (!sources.length) return emptyPanel("尚未持久化来源", "工作池读取并持久化来源快照后，这里会显示来源级别、权威度和可追溯任务。", true);
  return `
    <div class="card-heading">
      <div><p class="eyebrow">GOVERNED SOURCES</p><h2>来源目录</h2></div>
      <span>${sources.length} 个来源</span>
    </div>
    <div class="source-table" role="table" aria-label="研究来源">
      <div class="source-row source-header" role="row">
        <span role="columnheader">来源</span><span role="columnheader">级别</span><span role="columnheader">权威度</span><span role="columnheader">状态</span>
      </div>
      ${sources.map((source) => {
        const url = safeExternalUrl(source.canonical_url);
        return `
          <div class="source-row" role="row">
            <div role="cell">
              ${url ? `<a href="${escapeHtml(url)}" target="_blank" rel="noreferrer noopener">${escapeHtml(source.title || source.canonical_url)}</a>` : `<strong>${escapeHtml(source.title || source.source_id)}</strong>`}
              <small>${escapeHtml(source.publisher || source.canonical_url)}</small>
            </div>
            <span role="cell">${escapeHtml(source.source_level)} · ${escapeHtml(source.source_type)}</span>
            <span role="cell">${formatPercent(source.authority_score)}</span>
            <span role="cell">${badge(statusLabel(source.status), itemStatusTone(source.status))}</span>
          </div>
        `;
      }).join("")}
    </div>
  `;
}

function risksView(evidence) {
  return `
    <div class="risk-columns">
      <section>
        <div class="card-heading compact"><div><p class="eyebrow">OPEN GAPS</p><h2>章节缺口</h2></div>${badge(`${evidence.gaps.length}`, evidence.gaps.length ? "warning" : "success")}</div>
        ${evidence.gaps.length ? `<div class="risk-list">${evidence.gaps.map((gap) => `
          <article class="risk-card warning">
            <div><strong>${escapeHtml(gap.section_title)}</strong>${badge(statusLabel(gap.coverage_status), "warning")}</div>
            <p>覆盖 ${formatPercent(gap.coverage_score)} · 引用 ${formatPercent(gap.citation_score)}</p>
            <small>${gap.unsupported_claim_ids.length} 条必需主张仍未获得充分支持</small>
          </article>
        `).join("")}</div>` : emptyPanel("没有开放章节缺口", "所有必需章节均达到覆盖与引用门槛。", true)}
      </section>
      <section>
        <div class="card-heading compact"><div><p class="eyebrow">PRESERVED CONFLICTS</p><h2>证据冲突</h2></div>${badge(`${evidence.conflicts.length}`, evidence.conflicts.length ? "danger" : "success")}</div>
        ${evidence.conflicts.length ? `<div class="risk-list">${evidence.conflicts.map((conflict) => `
          <article class="risk-card danger">
            <div><strong>${escapeHtml(conflict.summary)}</strong>${badge(statusLabel(conflict.severity), conflict.high_impact ? "danger" : "warning")}</div>
            <p>${conflict.claim_ids.length} 条相关主张 · ${escapeHtml(statusLabel(conflict.status))}</p>
            ${conflict.resolution ? `<small>${escapeHtml(conflict.resolution)}</small>` : `<small>冲突保持可见，尚未形成可验证的解决结论。</small>`}
          </article>
        `).join("")}</div>` : emptyPanel("没有活跃证据冲突", "当前证据图未记录需要披露的冲突。", true)}
      </section>
    </div>
  `;
}

function reportLifecycleView(data) {
  const reporting = data.reporting;
  const revision = reporting.latest_revision;
  const review = reporting.latest_review;
  const outcome = reporting.outcome;
  return `
    <div class="report-lifecycle">
      <section class="surface-card report-status-card">
        <div>
          <p class="eyebrow">SYNTHESIS & REVIEW LOOP</p>
          <h2>${escapeHtml(reporting.title)}</h2>
          <p>${outcome ? escapeHtml(outcome.summary) : "证据门通过后，合成器与审查器会在有界循环中形成、评分并修复报告。"}</p>
          <div class="report-status-meta">
            ${badge(reporting.artifact_ready ? "最终制品可用" : "报告尚未就绪", reporting.artifact_ready ? "success" : "neutral")}
            <span>${reporting.revision_count} 次修订</span>
            <span>${review ? `审查决策：${statusLabel(review.decision)}` : "等待审查"}</span>
          </div>
        </div>
        ${reporting.artifact_ready ? `<a class="primary-action" href="${safeInternalPath(data.navigation.report_url)}" data-nav="${safeInternalPath(data.navigation.report_url)}">阅读完整报告 →</a>` : ""}
      </section>
      <div class="report-lifecycle-grid">
        <section class="surface-card">
          <div class="card-heading compact"><div><p class="eyebrow">REVISION</p><h2>最近修订</h2></div>${revision ? badge(`r${revision.revision}`, "info") : ""}</div>
          ${revision ? revisionDetails(revision) : emptyPanel("尚无报告修订", "报告合成器完成首版后，这里会显示不可变制品和使用量。", true)}
        </section>
        <section class="surface-card">
          <div class="card-heading compact"><div><p class="eyebrow">REVIEW RUBRIC</p><h2>审查评分</h2></div>${review ? badge(statusLabel(review.decision), review.decision === "accept" ? "success" : "warning") : ""}</div>
          ${review ? reviewScores(review) : emptyPanel("尚无审查结果", "报告审查器执行完整八维质量门后，这里会显示评分与依据。", true)}
        </section>
      </div>
      ${review && (review.findings.length || review.repair_actions.length) ? `
        <div class="report-lifecycle-grid">
          <section class="surface-card">${reviewFindings(review.findings)}</section>
          <section class="surface-card">${repairActions(review.repair_actions)}</section>
        </div>
      ` : ""}
      <section class="surface-card">
        <div class="card-heading"><div><p class="eyebrow">REPORT OUTLINE</p><h2>章节生命周期</h2></div></div>
        ${sectionReadinessList(reporting.outline)}
      </section>
    </div>
  `;
}

function timelineView(data) {
  const events = state.timeline.items || data.timeline;
  return `
    <div class="timeline-layout">
      <section class="surface-card">
        <div class="card-heading">
          <div>
            <p class="eyebrow">APPEND-ONLY TRACE</p>
            <h2>运行时间线</h2>
            <p>展示结构化决策、模型/工具、证据、报告、预算、权限和错误事件。</p>
          </div>
          <div class="timeline-export">
            <a class="quiet-action" href="${safeInternalPath(data.navigation.trace_export_json_url)}">导出 JSON</a>
            <a class="quiet-action" href="${safeInternalPath(data.navigation.trace_export_ndjson_url)}">导出 NDJSON</a>
          </div>
        </div>
        <form id="timeline-filter-form" class="timeline-filters">
          <label>
            <span class="sr-only">搜索时间线</span>
            <input id="timeline-search" type="search" value="${escapeHtml(state.timeline.search)}" placeholder="搜索事件文本" />
          </label>
          <label>
            <span class="sr-only">事件类型</span>
            <input id="timeline-types" value="${escapeHtml(state.timeline.eventTypes)}" placeholder="事件类型，逗号分隔" />
          </label>
          <label class="checkbox-control">
            <input id="timeline-errors" type="checkbox" ${state.timeline.errorOnly ? "checked" : ""} />
            <span>仅错误</span>
          </label>
          <button class="secondary-action" type="submit" ${state.timeline.loading ? "disabled" : ""}>${state.timeline.loading ? "查询中…" : "应用筛选"}</button>
          ${state.timeline.items ? `<button class="quiet-action" type="button" data-action="clear-timeline-filter">清除</button>` : ""}
        </form>
        ${events.length ? `
          <div class="event-stream">
            ${events.map(timelineEvent).join("")}
          </div>
          ${state.timeline.next ? `<button class="secondary-action timeline-more" type="button" data-action="timeline-more" data-after="${state.timeline.next}" ${state.timeline.loading ? "disabled" : ""}>加载下一页</button>` : ""}
        ` : emptyPanel("没有匹配的时间线事件", "调整筛选条件或等待运行产生新的事件。", true)}
      </section>
    </div>
  `;
}

function timelineEvent(event) {
  const usage = objectOf(event.usage);
  const hasUsage = Object.values(usage).some((value) => Number(value) > 0);
  const artifacts = [
    ...arrayOf(event.input_artifact_ids),
    ...arrayOf(event.output_artifact_ids),
  ];
  return `
    <article class="event-item ${event.error ? "has-error" : ""}">
      <div class="event-sequence">${formatNumber(event.sequence_no)}</div>
      <div class="event-body">
        <div class="event-heading">
          <div>
            ${badge(eventLabel(event.event_type), event.error ? "danger" : itemStatusTone(event.status))}
            <strong>${escapeHtml(roleLabelFromActor(event.actor_id))}</strong>
          </div>
          <time datetime="${escapeHtml(event.timestamp || event.occurred_at || "")}">${formatDateTime(event.timestamp || event.occurred_at)}</time>
        </div>
        <p>${escapeHtml(event.message || event.payload?.message || eventLabel(event.event_type))}</p>
        <div class="event-meta">
          <span>${escapeHtml(event.span_kind || "event")}</span>
          <span>${escapeHtml(statusLabel(event.status))}</span>
          <span>${formatNumber(event.latency_ms, 1)} ms</span>
          <span>尝试 ${formatNumber(event.attempt || 1)}</span>
          ${event.task_id ? `<button type="button" data-open-task="${escapeHtml(event.task_id)}">任务 ${escapeHtml(shortId(event.task_id))}</button>` : ""}
        </div>
        ${event.error ? `<div class="inline-notice danger"><strong>${escapeHtml(event.error.code || "运行错误")}</strong><span>${escapeHtml(event.error.message || JSON.stringify(event.error))}</span></div>` : ""}
        ${(hasUsage || artifacts.length || Object.keys(objectOf(event.component_versions)).length || Object.keys(objectOf(event.permissions)).length) ? `
          <details class="event-details">
            <summary>使用量、制品、版本与权限</summary>
            <div class="event-detail-grid">
              <div><span>Tokens</span><strong>${formatNumber(usage.input_tokens || 0)} / ${formatNumber(usage.output_tokens || 0)}</strong></div>
              <div><span>成本</span><strong>${formatUsd(usage.cost_usd || 0)}</strong></div>
              <div><span>制品</span><strong>${artifacts.length}</strong></div>
              <div><span>重试</span><strong>${formatNumber(usage.retries || 0)}</strong></div>
            </div>
            <pre>${escapeHtml(JSON.stringify({
              input_artifact_ids: event.input_artifact_ids,
              output_artifact_ids: event.output_artifact_ids,
              state_artifact_id: event.state_artifact_id,
              component_versions: event.component_versions,
              permissions: event.permissions,
            }, null, 2))}</pre>
          </details>
        ` : ""}
        ${Object.keys(objectOf(event.payload)).length ? `
          <details class="event-details">
            <summary>事件载荷</summary>
            <pre>${escapeHtml(JSON.stringify(event.payload, null, 2))}</pre>
          </details>
        ` : ""}
      </div>
    </article>
  `;
}

function reportView() {
  const data = state.report;
  if (!data) return fatalOrEmpty("报告投影尚不可用。");
  const reporting = data.reporting;
  const review = reporting.latest_review;
  return `
    <div class="app-frame report-frame">
      ${chromeHeader("报告工作台")}
      <main id="main-content" class="report-main">
        ${renderBanner()}
        <header class="report-masthead">
          <div>
            <div class="run-heading-line">
              ${badge(statusLabel(data.runtime.status), runStatusTone(data.runtime.status))}
              ${reporting.latest_revision ? badge(`修订 r${reporting.latest_revision.revision}`, "info") : ""}
              ${review ? badge(`审查：${statusLabel(review.decision)}`, review.decision === "accept" ? "success" : "warning") : ""}
            </div>
            <p class="eyebrow">VERIFIED EVIDENCE REPORT</p>
            <h1>${escapeHtml(reporting.title || data.identity.query)}</h1>
            <p>${escapeHtml(data.identity.query)}</p>
          </div>
          <div class="masthead-actions">
            <a class="secondary-action" href="${safeInternalPath(data.navigation.console_url)}" data-nav="${safeInternalPath(data.navigation.console_url)}">返回运行</a>
            <a class="primary-action" href="${safeInternalPath(data.navigation.studio_url)}">在 Studio 中检查 ↗</a>
          </div>
        </header>

        <div class="report-workspace">
          <aside class="report-sidebar">
            <section class="surface-card sticky-card">
              <p class="eyebrow">OUTLINE</p>
              <h2>报告大纲</h2>
              ${reporting.outline.length ? `<nav class="outline-nav">${reporting.outline.map((section) => `<a href="#${escapeHtml(sectionAnchor(section))}"><span>${String(section.order + 1).padStart(2, "0")}</span>${escapeHtml(section.title)}</a>`).join("")}</nav>` : `<p class="muted-copy">尚无结构化大纲。</p>`}
              <div class="sidebar-divider"></div>
              <p class="eyebrow">EVIDENCE</p>
              <dl class="compact-details">
                <div><dt>来源</dt><dd>${data.evidence.knowledge.source_count}</dd></div>
                <div><dt>主张</dt><dd>${data.evidence.knowledge.claim_count}</dd></div>
                <div><dt>证据</dt><dd>${data.evidence.knowledge.evidence_count}</dd></div>
                <div><dt>章节覆盖</dt><dd>${formatPercent(data.evidence.coverage.completion_ratio)}</dd></div>
                <div><dt>开放缺口</dt><dd>${data.evidence.knowledge.open_gap_count}</dd></div>
                <div><dt>冲突</dt><dd>${data.evidence.knowledge.conflict_count}</dd></div>
              </dl>
            </section>
          </aside>
          <article class="report-document surface-card">
            ${data.markdown ? renderMarkdownSafe(data.markdown) : reportUnavailable(data)}
          </article>
          <aside class="review-sidebar">
            <section class="surface-card">
              <p class="eyebrow">REVIEW</p>
              <h2>报告质量门</h2>
              ${review ? reviewScores(review, true) : `<p class="muted-copy">报告尚未完成审查。</p>`}
              ${reporting.outcome ? `<div class="outcome-note"><strong>${escapeHtml(statusLabel(reporting.outcome.status))}</strong><p>${escapeHtml(reporting.outcome.summary)}</p></div>` : ""}
            </section>
            ${review?.findings.length ? `<section class="surface-card">${reviewFindings(review.findings)}</section>` : ""}
          </aside>
        </div>
      </main>
      ${footer({ navigation: data.navigation, identity: data.identity })}
      ${toast()}
    </div>
  `;
}

function reportUnavailable(data) {
  const status = data.runtime.status;
  const messages = {
    queued: "运行仍在排队，尚未进入证据研究。",
    running: "报告尚未生成。研究与独立验证完成后，合成器才会读取已验证证据。",
    waiting_approval: "运行正在等待人工审批；处理后才能继续研究或报告修复。",
    failed: "运行失败，未生成最终报告。已产生的事件与证据仍可在运行工作台和 Studio 中检查。",
    cancelled: "运行已取消，未生成最终报告。",
  };
  return `
    <div class="report-empty">
      <span class="report-empty-mark">DR</span>
      <h2>报告制品尚不可用</h2>
      <p>${escapeHtml(messages[status] || "报告仍在生成。")}</p>
      <a class="secondary-action" href="${safeInternalPath(data.navigation.console_url)}" data-nav="${safeInternalPath(data.navigation.console_url)}">返回运行工作台</a>
    </div>
  `;
}

function progressRail(steps) {
  return `
    <section class="progress-rail" aria-label="运行阶段">
      ${steps.map((step, index) => `
        <div class="progress-item ${escapeHtml(step.status)}">
          <span class="progress-number">${String(index + 1).padStart(2, "0")}</span>
          <span class="progress-line" aria-hidden="true"></span>
          <div>
            <strong>${escapeHtml(progressLabel(step.step_id, step.label))}</strong>
            <small>${escapeHtml(statusLabel(step.status))}</small>
          </div>
        </div>
      `).join("")}
    </section>
  `;
}

function roleRail(roles, activeRoleId) {
  return `
    <section class="role-rail" aria-label="Background001 五个运行角色">
      ${roles.map((role, index) => `
        <article class="role-tile ${role.role_id === activeRoleId ? "active" : ""} ${escapeHtml(role.status)}">
          <span class="role-index">${String(index + 1).padStart(2, "0")}</span>
          <div>
            <strong>${escapeHtml(roleLabel(role.role_id, role.label))}</strong>
            <small>${escapeHtml(roleResponsibility(role.role_id))}</small>
          </div>
          ${badge(statusLabel(role.status), itemStatusTone(role.status))}
        </article>
      `).join("")}
    </section>
  `;
}

function approvalCallout(data) {
  if (!data.actions.approvals.length) return "";
  return `
    <section class="operator-callout">
      <div>
        <span class="callout-icon">!</span>
        <div>
          <p class="eyebrow">HUMAN-IN-THE-LOOP</p>
          <h2>${data.actions.approvals.length} 个任务等待审批</h2>
          <p>${escapeHtml(data.actions.approvals.map((approval) => `${approval.task_title}：${approval.reason}`).join("；"))}</p>
        </div>
      </div>
      ${data.actions.can_approve ? `<button class="warning-action" type="button" data-action="open-approve">审查并批准</button>` : ""}
    </section>
  `;
}

function runtimeErrorCallout(runtime, scheduler) {
  if (runtime.status !== "failed" && runtime.status !== "cancelled") return "";
  const isFailure = runtime.status === "failed";
  return `
    <section class="operator-callout ${isFailure ? "danger" : "muted"}">
      <div>
        <span class="callout-icon">${isFailure ? "×" : "—"}</span>
        <div>
          <p class="eyebrow">${isFailure ? "RUN FAILED" : "RUN CANCELLED"}</p>
          <h2>${escapeHtml(runtime.error_code || statusLabel(runtime.status))}</h2>
          <p>${escapeHtml(runtime.error_message || scheduler.cancellation_reason || "运行已进入终止状态，已生成的事件、证据与制品均被保留。")}</p>
          ${runtime.causal_errors?.length ? `
            <ol class="causal-error-chain" aria-label="运行错误因果链">
              ${runtime.causal_errors.map((error, index) => `
                <li class="${error.is_primary ? "primary" : "downstream"}">
                  <strong>${escapeHtml(error.code || `error-${index + 1}`)}</strong>
                  <span>${escapeHtml(error.message || "运行错误")}</span>
                  <small>#${escapeHtml(error.sequence_no)} · ${escapeHtml(roleLabelFromActor(error.actor_id))} · ${escapeHtml(error.category)}</small>
                </li>
              `).join("")}
            </ol>
          ` : ""}
        </div>
      </div>
    </section>
  `;
}

function actionDialog(data) {
  if (!state.modal) return "";
  if (state.modal === "approve") {
    return `
      <dialog id="action-dialog" class="action-dialog" aria-labelledby="dialog-title">
        <form id="approve-form" method="dialog">
          <div class="dialog-heading">
            <div><p class="eyebrow">HUMAN APPROVAL</p><h2 id="dialog-title">批准等待中的任务</h2></div>
            <button type="button" class="dialog-close" data-action="close-dialog" aria-label="关闭">×</button>
          </div>
          <div class="dialog-task-list">
            ${data.actions.approvals.map((approval) => `
              <article>
                <strong>${escapeHtml(approval.task_title)}</strong>
                <p>${escapeHtml(approval.reason)}</p>
                <small>请求者：${escapeHtml(approval.requested_by)} · ${formatDateTime(approval.requested_at)}</small>
              </article>
            `).join("")}
          </div>
          <label>
            <span>审批人 ID</span>
            <input
              name="approved_by"
              maxlength="200"
              pattern="[a-z][a-z0-9_]*_[A-Za-z0-9][A-Za-z0-9_.:-]*"
              required
              placeholder="例如：user_research_owner"
              aria-describedby="approved-by-help"
            />
            <small id="approved-by-help">使用带命名空间的审计身份，例如 user_research_owner。</small>
          </label>
          <label><span>审批说明</span><textarea name="note" rows="4" maxlength="2000" required placeholder="说明批准依据、风险判断和允许继续的边界。"></textarea></label>
          <p class="form-error" id="dialog-error" role="alert"></p>
          <div class="dialog-actions">
            <button class="quiet-action" type="button" data-action="close-dialog">返回</button>
            <button class="warning-action" type="submit" ${state.actionPending ? "disabled" : ""}>${state.actionPending ? "正在提交…" : "批准并继续运行"}</button>
          </div>
        </form>
      </dialog>
    `;
  }
  return `
    <dialog id="action-dialog" class="action-dialog danger-dialog" aria-labelledby="dialog-title">
      <form id="cancel-form" method="dialog">
        <div class="dialog-heading">
          <div><p class="eyebrow">CANCEL RUN</p><h2 id="dialog-title">取消当前研究运行</h2></div>
          <button type="button" class="dialog-close" data-action="close-dialog" aria-label="关闭">×</button>
        </div>
        <div class="inline-notice danger">
          <strong>取消会终止尚未完成的调度任务</strong>
          <span>已持久化的事件、证据、来源和报告修订会保留，运行不能恢复为活动状态。</span>
        </div>
        <label><span>取消原因</span><textarea name="reason" rows="5" maxlength="2000" required placeholder="记录取消原因，便于后续审计和 Studio 检查。"></textarea></label>
        <p class="form-error" id="dialog-error" role="alert"></p>
        <div class="dialog-actions">
          <button class="quiet-action" type="button" data-action="close-dialog">保留运行</button>
          <button class="danger-action" type="submit" ${state.actionPending ? "disabled" : ""}>${state.actionPending ? "正在取消…" : "确认取消运行"}</button>
        </div>
      </form>
    </dialog>
  `;
}

function renderBanner() {
  if (!state.banner) return "";
  return `
    <section class="status-banner ${escapeHtml(state.banner.tone || "info")}" role="status">
      <div><strong>${escapeHtml(state.banner.title)}</strong><span>${escapeHtml(state.banner.message)}</span></div>
      ${state.banner.retry ? `<button type="button" data-action="retry-workspace">重试</button>` : `<button type="button" data-action="dismiss-banner">关闭</button>`}
    </section>
  `;
}

function toast() {
  if (!state.toast) return "";
  return `<div class="toast ${escapeHtml(state.toast.tone || "success")}" role="status">${escapeHtml(state.toast.message)}</div>`;
}

function fatalOrEmpty(message) {
  if (state.fatalError) return fatalView(state.fatalError);
  return emptyPanel("数据尚不可用", message);
}

function fatalView(error) {
  const notFound = error?.status === 404;
  return `
    <div class="app-frame">
      ${chromeHeader("运行工作台")}
      <main id="main-content" class="fatal-main">
        <section class="fatal-card">
          <span class="fatal-code">${notFound ? "404" : "ERR"}</span>
          <p class="eyebrow">${notFound ? "RUN NOT FOUND" : "CONSOLE REQUEST FAILED"}</p>
          <h1>${notFound ? "找不到这次研究运行" : "无法读取运行工作台"}</h1>
          <p>${escapeHtml(describeApiError(error))}</p>
          <div>
            <a class="secondary-action" href="/" data-nav="/">返回运行列表</a>
            ${!notFound ? `<button class="primary-action" type="button" data-action="retry-bootstrap">重新加载</button>` : ""}
          </div>
        </section>
      </main>
    </div>
  `;
}

function loadingView() {
  return `
    <div class="app-frame">
      ${chromeHeader("正在连接运行投影")}
      <main id="main-content" class="loading-main">
        <div class="loading-mark"><i></i><i></i><i></i></div>
        <p class="eyebrow">READING DURABLE PROJECTIONS</p>
        <h1>${state.route.page === "landing" ? "正在读取运行目录" : state.route.page === "report" ? "正在组装报告工作台" : "正在组装运行工作台"}</h1>
        <p>任务、证据、报告与 Trace 从各自的持久化所有者读取。</p>
      </main>
    </div>
  `;
}

function render(options = {}) {
  const focusSnapshot = options.preserveFocus ? captureFocus() : null;
  if (state.loading) {
    app.innerHTML = loadingView();
  } else if (state.fatalError && !state.workspace && !state.report && state.route.page !== "landing") {
    app.innerHTML = fatalView(state.fatalError);
  } else if (state.route.page === "landing") {
    app.innerHTML = state.fatalError ? fatalView(state.fatalError) : landingView();
  } else if (state.route.page === "console") {
    app.innerHTML = consoleView();
  } else {
    app.innerHTML = reportView();
  }
  bindEvents();
  if (state.modal) {
    const dialog = document.getElementById("action-dialog");
    if (dialog && !dialog.open) dialog.showModal();
  }
  if (focusSnapshot) restoreFocus(focusSnapshot);
}

function captureFocus() {
  const active = document.activeElement;
  if (!active || active === document.body) return null;
  const selector = active.id
    ? `#${CSS.escape(active.id)}`
    : active.name
      ? `[name="${CSS.escape(active.name)}"]`
      : null;
  return {
    selector,
    start: typeof active.selectionStart === "number" ? active.selectionStart : null,
    end: typeof active.selectionEnd === "number" ? active.selectionEnd : null,
    scrollX: window.scrollX,
    scrollY: window.scrollY,
  };
}

function restoreFocus(snapshot) {
  window.requestAnimationFrame(() => {
    const target = snapshot.selector ? document.querySelector(snapshot.selector) : null;
    if (target) {
      target.focus({ preventScroll: true });
      if (snapshot.start !== null && typeof target.setSelectionRange === "function") {
        target.setSelectionRange(snapshot.start, snapshot.end);
      }
    }
    window.scrollTo(snapshot.scrollX, snapshot.scrollY);
  });
}

function bindEvents() {
  document.querySelectorAll("[data-nav]").forEach((element) => {
    element.addEventListener("click", (event) => {
      if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      navigate(element.dataset.nav || element.getAttribute("href") || "/");
    });
  });
  document.querySelectorAll("[data-view]").forEach((button) => {
    button.addEventListener("click", () => {
      state.activeView = button.dataset.view;
      const url = new URL(window.location.href);
      if (state.activeView === "overview") url.searchParams.delete("view");
      else url.searchParams.set("view", state.activeView);
      history.replaceState({}, "", `${url.pathname}${url.search}`);
      render();
    });
  });
  document.querySelectorAll("[data-evidence-view]").forEach((button) => {
    button.addEventListener("click", () => {
      state.evidenceView = button.dataset.evidenceView;
      render();
    });
  });
  document.querySelectorAll("[data-task-id]").forEach((button) => {
    button.addEventListener("click", () => {
      state.selectedTaskId = button.dataset.taskId;
      state.activeView = "tasks";
      render();
    });
  });
  document.querySelectorAll("[data-open-task]").forEach((button) => {
    button.addEventListener("click", () => {
      state.selectedTaskId = button.dataset.openTask;
      state.activeView = "tasks";
      const url = new URL(window.location.href);
      url.searchParams.set("view", "tasks");
      history.replaceState({}, "", `${url.pathname}${url.search}`);
      render();
    });
  });
  document.querySelectorAll("[data-copy]").forEach((button) => {
    button.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(button.dataset.copy || "");
        showToast("ID 已复制。");
      } catch {
        showToast("浏览器未允许复制，请手动选择 ID。", "warning");
      }
    });
  });

  bindLandingEvents();
  bindConsoleEvents();
  bindActionDialog();
}

function bindLandingEvents() {
  const form = document.getElementById("create-run-form");
  if (form) {
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const errorRoot = document.getElementById("create-error");
      const submit = form.querySelector("button[type='submit']");
      const values = new FormData(form);
      const query = String(values.get("query") || "").trim();
      const instructions = String(values.get("instructions") || "").trim();
      const depth = String(values.get("depth") || "standard");
      if (!query) {
        errorRoot.textContent = "请输入研究问题。";
        form.querySelector("[name='query']")?.focus();
        return;
      }
      errorRoot.textContent = "";
      submit.disabled = true;
      submit.querySelector("span").textContent = "正在创建运行…";
      try {
        const created = await api("/api/runs", {
          method: "POST",
          body: JSON.stringify({ query, instructions, depth }),
        });
        navigate(created.console_url);
      } catch (error) {
        errorRoot.textContent = describeApiError(error);
        submit.disabled = false;
        submit.querySelector("span").textContent = "启动证据研究";
      }
    });
  }
  const runSearch = document.getElementById("run-search");
  if (runSearch) {
    runSearch.addEventListener("input", () => {
      state.runSearch = runSearch.value;
      render({ preserveFocus: true });
    });
  }
  const runStatus = document.getElementById("run-status");
  if (runStatus) {
    runStatus.addEventListener("change", () => {
      state.runStatus = runStatus.value;
      render({ preserveFocus: true });
    });
  }
  document.querySelector("[data-action='refresh-runs']")?.addEventListener("click", refreshRuns);
}

function bindConsoleEvents() {
  document.querySelector("[data-action='refresh-workspace']")?.addEventListener("click", () => refreshWorkspace());
  document.querySelector("[data-action='retry-workspace']")?.addEventListener("click", () => refreshWorkspace());
  document.querySelector("[data-action='retry-bootstrap']")?.addEventListener("click", bootstrap);
  document.querySelector("[data-action='dismiss-banner']")?.addEventListener("click", () => {
    state.banner = null;
    render();
  });
  document.querySelector("[data-action='open-approve']")?.addEventListener("click", () => {
    state.modal = "approve";
    render();
  });
  document.querySelector("[data-action='open-cancel']")?.addEventListener("click", () => {
    state.modal = "cancel";
    render();
  });

  const taskSearch = document.getElementById("task-search");
  if (taskSearch) {
    taskSearch.addEventListener("input", () => {
      state.taskSearch = taskSearch.value;
      render({ preserveFocus: true });
    });
  }
  const taskStatus = document.getElementById("task-status");
  if (taskStatus) {
    taskStatus.addEventListener("change", () => {
      state.taskStatus = taskStatus.value;
      render({ preserveFocus: true });
    });
  }
  document.getElementById("timeline-filter-form")?.addEventListener("submit", async (event) => {
    event.preventDefault();
    captureTransientInputs();
    await loadTimeline(0, false);
  });
  document.querySelector("[data-action='timeline-more']")?.addEventListener("click", async (event) => {
    await loadTimeline(Number(event.currentTarget.dataset.after || 0), true);
  });
  document.querySelector("[data-action='clear-timeline-filter']")?.addEventListener("click", () => {
    state.timeline = {
      ...state.timeline,
      items: null,
      next: null,
      search: "",
      eventTypes: "",
      errorOnly: false,
    };
    render();
  });
}

function bindActionDialog() {
  const dialog = document.getElementById("action-dialog");
  if (!dialog) return;
  dialog.addEventListener("cancel", (event) => {
    event.preventDefault();
    if (!state.actionPending) closeDialog();
  });
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog && !state.actionPending) closeDialog();
  });
  document.querySelectorAll("[data-action='close-dialog']").forEach((button) => {
    button.addEventListener("click", closeDialog);
  });
  document.getElementById("approve-form")?.addEventListener("submit", submitApproval);
  document.getElementById("cancel-form")?.addEventListener("submit", submitCancellation);
}

function closeDialog() {
  if (state.actionPending) return;
  document.getElementById("action-dialog")?.close();
  state.modal = null;
  render();
}

async function submitApproval(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const values = new FormData(form);
  const approvedBy = String(values.get("approved_by") || "").trim();
  const note = String(values.get("note") || "").trim();
  if (!approvedBy || !note) {
    document.getElementById("dialog-error").textContent = "审批人和审批说明均为必填项。";
    return;
  }
  if (!isNamespacedIdentifier(approvedBy)) {
    document.getElementById("dialog-error").textContent = "审批人 ID 必须是带命名空间的审计身份，例如 user_research_owner。";
    return;
  }
  await submitRunAction("approve", { approved_by: approvedBy, note }, "审批已提交，运行正在恢复。");
}

async function submitCancellation(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const reason = String(new FormData(form).get("reason") || "").trim();
  if (!reason) {
    document.getElementById("dialog-error").textContent = "请填写取消原因。";
    return;
  }
  await submitRunAction("cancel", { reason }, "运行已取消，持久化数据已保留。");
}

async function submitRunAction(action, payload, successMessage) {
  state.actionPending = true;
  render();
  try {
    await api(
      `/api/runs/${encodeURIComponent(state.route.researchId)}/${action}`,
      { method: "POST", body: JSON.stringify(payload) },
    );
    state.modal = null;
    state.actionPending = false;
    state.toast = { tone: "success", message: successMessage };
    await refreshWorkspace({ quiet: true, immediate: true });
    scheduleToastClear();
  } catch (error) {
    state.actionPending = false;
    render();
    const errorRoot = document.getElementById("dialog-error");
    if (errorRoot) errorRoot.textContent = describeApiError(error);
  }
}

async function refreshRuns() {
  if (state.refreshing) return;
  captureTransientInputs();
  state.refreshing = true;
  render({ preserveFocus: true });
  try {
    state.runs = normalizeRunList(await api("/api/runs", { cache: "no-store" }));
    state.fatalError = null;
  } catch (error) {
    state.banner = {
      tone: "danger",
      title: "无法刷新运行目录",
      message: describeApiError(error),
    };
  } finally {
    state.refreshing = false;
    render({ preserveFocus: true });
  }
}

async function loadTimeline(afterSequence, append) {
  if (!state.workspace || state.timeline.loading) return;
  state.timeline.loading = true;
  render({ preserveFocus: true });
  const params = new URLSearchParams({
    after_sequence: String(afterSequence || 0),
    limit: "100",
  });
  if (state.timeline.search) params.set("text", state.timeline.search);
  if (state.timeline.eventTypes) params.set("event_types", state.timeline.eventTypes);
  if (state.timeline.errorOnly) params.set("error_only", "true");
  try {
    const page = await api(
      `/api/studio/runs/${encodeURIComponent(state.workspace.identity.run_id)}/timeline?${params}`,
    );
    const items = arrayOf(page.items);
    state.timeline.items = append
      ? [...arrayOf(state.timeline.items), ...items]
      : items;
    state.timeline.next = page.next_after_sequence ?? null;
    state.banner = null;
  } catch (error) {
    state.banner = {
      tone: "danger",
      title: "时间线查询失败",
      message: describeApiError(error),
    };
  } finally {
    state.timeline.loading = false;
    render({ preserveFocus: true });
  }
}

function showToast(message, tone = "success") {
  state.toast = { message, tone };
  render({ preserveFocus: true });
  scheduleToastClear();
}

function scheduleToastClear() {
  window.setTimeout(() => {
    state.toast = null;
    render({ preserveFocus: true });
  }, 2600);
}

function metricCard(label, value, detail, tone = "neutral") {
  return `
    <article class="metric-card ${escapeHtml(tone)}">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(value)}</strong>
      <small>${escapeHtml(detail)}</small>
    </article>
  `;
}

function badge(label, tone = "neutral") {
  return `<span class="status-badge ${escapeHtml(tone)}"><i aria-hidden="true"></i>${escapeHtml(label)}</span>`;
}

function emptyPanel(title, message, compact = false) {
  return `
    <div class="empty-panel ${compact ? "compact" : ""}">
      <span aria-hidden="true">○</span>
      <h3>${escapeHtml(title)}</h3>
      <p>${escapeHtml(message)}</p>
    </div>
  `;
}

function sectionReadinessList(sections) {
  if (!sections.length) return emptyPanel("尚无报告章节", "监督器建立报告结构后，这里会显示逐章节就绪度。", true);
  return `
    <div class="section-readiness-list">
      ${sections.map((section) => `
        <article>
          <div class="section-order">${String(section.order + 1).padStart(2, "0")}</div>
          <div class="section-copy">
            <div><strong>${escapeHtml(section.title)}</strong>${badge(statusLabel(section.coverage_status), section.coverage_status === "complete" ? "success" : "warning")}</div>
            <p>${escapeHtml(section.goal)}</p>
            <div class="dual-progress">
              <span><i style="width:${Math.round(section.coverage_score * 100)}%"></i></span>
              <small>覆盖 ${formatPercent(section.coverage_score)}</small>
              <span><i style="width:${Math.round(section.citation_score * 100)}%"></i></span>
              <small>引用 ${formatPercent(section.citation_score)}</small>
            </div>
          </div>
        </article>
      `).join("")}
    </div>
  `;
}

function sectionCards(sections) {
  if (!sections.length) return emptyPanel("尚无章节", "报告结构尚未建立。", true);
  return `<div class="section-card-grid">${sections.map((section) => `
    <article class="section-card">
      <div class="section-card-heading">
        <span>${String(section.order + 1).padStart(2, "0")}</span>
        ${badge(statusLabel(section.coverage_status), section.coverage_status === "complete" ? "success" : "warning")}
      </div>
      <h3>${escapeHtml(section.title)}</h3>
      <p>${escapeHtml(section.goal)}</p>
      <div class="section-score-grid">
        <div><span>覆盖</span><strong>${formatPercent(section.coverage_score)}</strong></div>
        <div><span>引用</span><strong>${formatPercent(section.citation_score)}</strong></div>
        <div><span>必需主张</span><strong>${section.required_claim_ids.length}</strong></div>
        <div><span>未支持</span><strong>${section.unsupported_claim_ids.length}</strong></div>
      </div>
    </article>
  `).join("")}</div>`;
}

function taskDistribution(counts, total) {
  const entries = Object.entries(objectOf(counts));
  if (!entries.length) return `<p class="muted-copy">调度器尚未创建任务。</p>`;
  return `<div class="distribution-list">${entries.map(([status, count]) => `
    <div>
      <span>${escapeHtml(statusLabel(status))}</span>
      <div><i class="${escapeHtml(itemStatusTone(status))}" style="width:${total ? Math.max(4, (count / total) * 100) : 0}%"></i></div>
      <strong>${count}</strong>
    </div>
  `).join("")}</div>`;
}

function blockerSummary(evidence) {
  const items = [
    ["章节缺口", evidence.coverage.gap_section_ids.length, "warning"],
    ["高影响主张", evidence.coverage.blocked_high_impact_claim_ids.length, "danger"],
    ["严重冲突", evidence.coverage.severe_conflict_ids.length, "danger"],
  ];
  return `<div class="blocker-list">${items.map(([label, count, tone]) => `
    <button type="button" data-view="evidence">
      <span class="blocker-dot ${tone}"></span>
      <span>${label}</span>
      <strong>${count}</strong>
    </button>
  `).join("")}</div>`;
}

function identityDetails(identity) {
  return `
    <dl class="compact-details identity-details">
      <div><dt>Research</dt><dd><code>${escapeHtml(identity.research_id)}</code></dd></div>
      <div><dt>Thread</dt><dd><code>${escapeHtml(identity.thread_id)}</code></dd></div>
      <div><dt>Session</dt><dd><code>${escapeHtml(identity.session_id)}</code></dd></div>
      <div><dt>Run</dt><dd><code>${escapeHtml(identity.run_id)}</code></dd></div>
      <div><dt>创建时间</dt><dd>${formatDateTime(identity.created_at)}</dd></div>
      <div><dt>更新时间</dt><dd>${formatDateTime(identity.updated_at)}</dd></div>
    </dl>
    ${identity.instructions ? `<details class="instruction-details"><summary>用户约束</summary><p>${escapeHtml(identity.instructions)}</p></details>` : ""}
  `;
}

function budgetTable(budget, usage) {
  const rows = [
    ["Tokens", "max_tokens", "input_tokens", (value) => formatNumber(value)],
    ["模型调用", "max_model_calls", "model_calls", (value) => formatNumber(value)],
    ["工具调用", "max_tool_calls", "tool_calls", (value) => formatNumber(value)],
    ["搜索调用", "max_search_calls", "search_calls", (value) => formatNumber(value)],
    ["重试", "max_retries", "retries", (value) => formatNumber(value)],
    ["错误", "max_errors", "errors", (value) => formatNumber(value)],
    ["成本", "max_cost_usd", "cost_usd", (value) => value == null ? "不限" : formatUsd(value)],
    ["墙钟时间", "max_wall_time_seconds", "wall_time_seconds", (value) => value == null ? "不限" : formatDuration(value)],
  ];
  return `
    <div class="budget-table">
      ${rows.map(([label, maxKey, usedKey, formatter]) => {
        const maximum = budget?.[maxKey];
        const used = usage?.[usedKey] || 0;
        const ratio = Number(maximum) > 0 ? Math.min(1, Number(used) / Number(maximum)) : 0;
        return `
          <div>
            <span>${label}</span>
            <div class="budget-track"><i style="width:${Math.round(ratio * 100)}%"></i></div>
            <strong>${formatter(used)} / ${formatter(maximum)}</strong>
          </div>
        `;
      }).join("")}
    </div>
  `;
}

function artifactList(label, ids) {
  return `
    <div class="artifact-group">
      <span>${escapeHtml(label)}</span>
      ${ids.length ? `<div>${ids.map((id) => `<code>${escapeHtml(id)}</code>`).join("")}</div>` : `<small>无</small>`}
    </div>
  `;
}

function approvalDetail(approval) {
  return `
    <div class="inline-notice warning approval-detail">
      <strong>等待人工审批</strong>
      <span>${escapeHtml(approval.reason)}</span>
      <small>请求者 ${escapeHtml(approval.requested_by)} · ${formatDateTime(approval.requested_at)}</small>
    </div>
  `;
}

function revisionDetails(revision) {
  const usage = objectOf(revision.usage);
  return `
    <dl class="detail-grid">
      <div><dt>修订 ID</dt><dd><code>${escapeHtml(revision.revision_id)}</code></dd></div>
      <div><dt>创建时间</dt><dd>${formatDateTime(revision.created_at)}</dd></div>
      <div><dt>陈述数量</dt><dd>${revision.statement_count}</dd></div>
      <div><dt>引用数量</dt><dd>${revision.citation_count}</dd></div>
      <div><dt>模型调用</dt><dd>${formatNumber(usage.model_calls || 0)}</dd></div>
      <div><dt>Tokens</dt><dd>${formatNumber((usage.input_tokens || 0) + (usage.output_tokens || 0))}</dd></div>
    </dl>
    <details class="json-details">
      <summary>不可变制品引用</summary>
      <pre>${escapeHtml(JSON.stringify({
        report_artifact_id: revision.report_artifact_id,
        draft_artifact_id: revision.draft_artifact_id,
        citation_map_artifact_id: revision.citation_map_artifact_id,
        evidence_packet_artifact_id: revision.evidence_packet_artifact_id,
        parent_revision_id: revision.parent_revision_id,
      }, null, 2))}</pre>
    </details>
  `;
}

function reviewScores(review, compact = false) {
  return `
    <div class="review-score-list ${compact ? "compact" : ""}">
      ${review.scores.map((score) => `
        <details class="review-score">
          <summary>
            <span>${escapeHtml(reviewDimensionLabel(score.dimension))}</span>
            <span class="score-track"><i style="width:${Math.round(score.score * 100)}%"></i></span>
            <strong>${formatPercent(score.score)}</strong>
          </summary>
          <p>${escapeHtml(score.rationale)}</p>
        </details>
      `).join("")}
    </div>
    ${!compact ? `<div class="decision-box"><span>审查结论</span><strong>${escapeHtml(statusLabel(review.decision))}</strong><p>${escapeHtml(review.decision_summary)}</p></div>` : ""}
  `;
}

function reviewFindings(findings) {
  return `
    <div class="card-heading compact"><div><p class="eyebrow">FINDINGS</p><h2>审查发现</h2></div>${badge(`${findings.length}`, findings.length ? "warning" : "success")}</div>
    ${findings.length ? `<div class="finding-list">${findings.map((finding) => `
      <article>
        <div>${badge(statusLabel(finding.severity), finding.severity === "critical" || finding.severity === "high" ? "danger" : "warning")}<span>${escapeHtml(reviewDimensionLabel(finding.dimension))}</span></div>
        <p>${escapeHtml(finding.message)}</p>
      </article>
    `).join("")}</div>` : `<p class="muted-copy">没有审查发现。</p>`}
  `;
}

function repairActions(actions) {
  return `
    <div class="card-heading compact"><div><p class="eyebrow">REPAIR ACTIONS</p><h2>有界修复动作</h2></div>${badge(`${actions.length}`, actions.length ? "warning" : "success")}</div>
    ${actions.length ? `<div class="finding-list">${actions.map((action) => `
      <article>
        <div>${badge(statusLabel(action.kind), "info")}<span>${action.section_ids.length} 个章节</span></div>
        <p>${escapeHtml(action.reason)}</p>
      </article>
    `).join("")}</div>` : `<p class="muted-copy">没有要求执行额外修复。</p>`}
  `;
}

function taskStatusOptions(tasks) {
  const statuses = [...new Set(tasks.map((task) => task.status))].sort();
  return [
    `<option value="all" ${state.taskStatus === "all" ? "selected" : ""}>全部状态</option>`,
    ...statuses.map((status) => `<option value="${escapeHtml(status)}" ${state.taskStatus === status ? "selected" : ""}>${escapeHtml(statusLabel(status))}</option>`),
  ].join("");
}

function footer(data = null) {
  return `
    <footer class="app-footer">
      <div><strong>DeepResearcher</strong><span>Background001 native runtime · RL excluded</span></div>
      <div>
        ${data?.navigation?.studio_url ? `<a href="${safeInternalPath(data.navigation.studio_url)}">Studio V4</a>` : ""}
        ${data?.identity?.run_id ? `<code>${escapeHtml(data.identity.run_id)}</code>` : `<span>Append-only events · Immutable artifacts</span>`}
      </div>
    </footer>
  `;
}

function shortId(value) {
  const text = String(value || "");
  return text.length > 14 ? `${text.slice(0, 7)}…${text.slice(-5)}` : text;
}

function depthLabel(value) {
  return { quick: "快速研究", standard: "标准研究", deep: "深入研究" }[value] || String(value || "标准研究");
}

function progressLabel(stepId, fallback) {
  return {
    queued: "运行准备",
    research: "监督与研究",
    verification: "独立验证",
    synthesis: "证据合成",
    review: "报告审查",
    complete: "终止结果",
  }[stepId] || fallback;
}

function roleResponsibility(roleId) {
  return {
    research_supervisor: "动态计划、预算与语义收敛",
    research_worker_pool: "受控检索、读取、抽取与比较",
    evidence_verifier: "主张支持、引用与冲突独立核验",
    synthesis_writer: "仅依据已验证证据形成报告",
    report_reviewer: "八维评分与有界修复",
  }[roleId] || "结构化运行角色";
}

function roleInitials(roleId) {
  return {
    research_supervisor: "RS",
    research_worker_pool: "RW",
    evidence_verifier: "EV",
    synthesis_writer: "SW",
    report_reviewer: "RR",
  }[roleId] || "DR";
}

function roleLabelFromActor(actorId) {
  const value = String(actorId || "");
  if (value.includes("research_supervisor")) return roleLabel("research_supervisor");
  if (value.includes("research_worker")) return roleLabel("research_worker_pool");
  if (value.includes("evidence_verifier")) return roleLabel("evidence_verifier");
  if (value.includes("synthesis_writer")) return roleLabel("synthesis_writer");
  if (value.includes("report_reviewer")) return roleLabel("report_reviewer");
  return value || "运行时";
}

function sectionAnchor(section) {
  const slug = String(section.title || "")
    .normalize("NFKC")
    .toLocaleLowerCase()
    .replace(/[^\p{Letter}\p{Number}]+/gu, "-")
    .replace(/^-|-$/g, "");
  return slug || `section-${section.order + 1}`;
}

bootstrap();

export {
  api,
  resolveRoute,
  state,
};
