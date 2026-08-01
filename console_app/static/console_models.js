export const CONSOLE_SCHEMA_VERSION = "ConsoleWorkspace@2";
export const REPORT_SCHEMA_VERSION = "ReportWorkspace@2";

export const TERMINAL_RUN_STATUSES = new Set([
  "completed",
  "failed",
  "cancelled",
]);

export const LIVE_RUN_STATUSES = new Set([
  "queued",
  "running",
  "waiting_approval",
]);

const STATUS_LABELS = {
  queued: "排队中",
  running: "运行中",
  waiting_approval: "等待审批",
  completed: "已完成",
  complete: "完整",
  failed: "失败",
  cancelled: "已取消",
  pending: "待处理",
  ready: "就绪",
  paused: "已暂停",
  deferred: "已暂缓",
  pruned: "已裁剪",
  merged: "已合并",
  active: "进行中",
  blocked: "受阻",
  waiting: "等待中",
  succeeded: "成功",
  not_created: "尚未创建",
  not_assessed: "未评估",
  insufficient: "不充分",
  accept: "已接受",
  accepted: "已接受",
  reject: "已拒绝",
  supported: "已支持",
  partially_supported: "部分支持",
  contradicted: "已反驳",
  conflicted: "存在冲突",
  unsupported: "未支持",
  stale: "已过期",
  accessible: "可访问",
  inaccessible: "不可访问",
  low: "低",
  medium: "中",
  high: "高",
  critical: "严重",
  continue_research: "继续研究",
  complete_with_gaps: "有界完成（含证据缺口）",
  stop_low_gain: "低增益停止",
  stop_budget: "预算停止",
  stop_conflict: "冲突停止",
  start_reporting: "进入报告",
  targeted_research: "定向研究",
  citation_repair: "引用修复",
  local_rewrite: "局部改写",
  structural_rewrite: "结构改写",
};

const STAGE_LABELS = {
  queued: "运行排队",
  initializing: "运行时初始化",
  researching: "监督、检索与独立验证",
  reporting: "证据合成与报告审查",
  waiting_approval: "等待人工审批",
  completed: "报告已完成",
  failed: "运行失败",
  cancelled: "运行已取消",
};

const TASK_KIND_LABELS = {
  root: "根任务",
  source_discovery: "来源发现",
  research: "研究",
  gap: "缺口补全",
  conflict: "冲突核查",
  verification: "独立验证",
  section_support: "章节支持",
  synthesis: "报告合成",
  review: "报告审查",
  repair: "定向修复",
};

const ROLE_LABELS = {
  research_supervisor: "研究监督器",
  research_worker_pool: "研究工作池",
  evidence_verifier: "证据验证器",
  synthesis_writer: "证据合成器",
  report_reviewer: "报告审查器",
};

const REVIEW_DIMENSION_LABELS = {
  completeness: "完整性",
  support: "证据支持",
  citation: "引用质量",
  conflicts: "冲突披露",
  instruction_following: "指令遵循",
  depth: "研究深度",
  organization: "组织结构",
  readability: "可读性",
};

const EVENT_LABELS = {
  run_started: "运行开始",
  run_completed: "运行完成",
  run_failed: "运行失败",
  run_cancelled: "运行取消",
  span_started: "角色开始",
  span_completed: "角色完成",
  span_failed: "角色失败",
  model_started: "模型调用开始",
  model_completed: "模型调用完成",
  model_failed: "模型调用失败",
  command_proposed: "命令提出",
  decision_recorded: "结构化决策",
  policy_decided: "策略检查",
  tool_started: "工具调用开始",
  tool_completed: "工具调用完成",
  tool_failed: "工具调用失败",
  verification_completed: "验证完成",
  evidence_changed: "证据状态更新",
  report_changed: "报告状态更新",
  budget_updated: "预算更新",
};

export function invariant(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

export function isNamespacedIdentifier(value) {
  return /^[a-z][a-z0-9_]*_[A-Za-z0-9][A-Za-z0-9_.:-]*$/.test(String(value || "").trim());
}

export function normalizeWorkspace(payload) {
  invariant(payload && typeof payload === "object", "Console 响应不是对象。");
  invariant(
    payload.schema_version === CONSOLE_SCHEMA_VERSION,
    `Console 契约版本不匹配：期望 ${CONSOLE_SCHEMA_VERSION}，收到 ${String(payload.schema_version || "missing")}。`,
  );
  for (const key of [
    "identity",
    "runtime",
    "actions",
    "scheduler",
    "evidence",
    "reporting",
    "navigation",
  ]) {
    invariant(payload[key] && typeof payload[key] === "object", `Console 响应缺少 ${key}。`);
  }
  return {
    ...payload,
    runtime: {
      ...payload.runtime,
      progress: arrayOf(payload.runtime.progress),
      roles: arrayOf(payload.runtime.roles),
      decision_reasons: arrayOf(payload.runtime.decision_reasons),
      causal_errors: arrayOf(payload.runtime.causal_errors),
    },
    actions: {
      ...payload.actions,
      approvals: arrayOf(payload.actions.approvals),
      waiting_approval_task_ids: arrayOf(payload.actions.waiting_approval_task_ids),
    },
    scheduler: {
      ...payload.scheduler,
      tasks: arrayOf(payload.scheduler.tasks),
      active_task_ids: arrayOf(payload.scheduler.active_task_ids),
      ready_task_ids: arrayOf(payload.scheduler.ready_task_ids),
      waiting_approval_task_ids: arrayOf(payload.scheduler.waiting_approval_task_ids),
      task_counts: objectOf(payload.scheduler.task_counts),
    },
    evidence: {
      ...payload.evidence,
      sections: arrayOf(payload.evidence.sections),
      gaps: arrayOf(payload.evidence.gaps),
      conflicts: arrayOf(payload.evidence.conflicts),
      packets: arrayOf(payload.evidence.packets),
      sources: arrayOf(payload.evidence.sources),
    },
    timeline: arrayOf(payload.timeline),
  };
}

export function normalizeReport(payload) {
  invariant(payload && typeof payload === "object", "报告响应不是对象。");
  invariant(
    payload.schema_version === REPORT_SCHEMA_VERSION,
    `报告契约版本不匹配：期望 ${REPORT_SCHEMA_VERSION}，收到 ${String(payload.schema_version || "missing")}。`,
  );
  for (const key of [
    "identity",
    "runtime",
    "evidence",
    "reporting",
    "navigation",
  ]) {
    invariant(payload[key] && typeof payload[key] === "object", `报告响应缺少 ${key}。`);
  }
  return {
    ...payload,
    evidence: {
      ...payload.evidence,
      sections: arrayOf(payload.evidence.sections),
      gaps: arrayOf(payload.evidence.gaps),
      conflicts: arrayOf(payload.evidence.conflicts),
      packets: arrayOf(payload.evidence.packets),
      sources: arrayOf(payload.evidence.sources),
    },
    markdown: String(payload.markdown || ""),
  };
}

export function normalizeRunList(payload) {
  invariant(Array.isArray(payload), "最近运行响应不是列表。");
  return payload
    .filter((item) => item && item.schema_version === "ConsoleRunListItem@2")
    .map((item) => ({ ...item }))
    .sort((left, right) => String(right.updated_at).localeCompare(String(left.updated_at)));
}

export function arrayOf(value) {
  return Array.isArray(value) ? value : [];
}

export function objectOf(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? value : {};
}

export function shouldPoll(workspace) {
  return Boolean(workspace && LIVE_RUN_STATUSES.has(workspace.runtime?.status));
}

export function pollingDelay(workspace, failures = 0) {
  const status = workspace?.runtime?.status;
  const base = status === "waiting_approval" ? 5000 : status === "queued" ? 2500 : 1800;
  return Math.min(15000, base * Math.max(1, 2 ** Math.min(failures, 3)));
}

export function statusLabel(value) {
  return STATUS_LABELS[String(value || "")] || humanize(value);
}

export function stageLabel(value) {
  return STAGE_LABELS[String(value || "")] || humanize(value);
}

export function taskKindLabel(value) {
  return TASK_KIND_LABELS[String(value || "")] || humanize(value);
}

export function roleLabel(value, fallback = "") {
  return ROLE_LABELS[String(value || "")] || fallback || humanize(value);
}

export function reviewDimensionLabel(value) {
  return REVIEW_DIMENSION_LABELS[String(value || "")] || humanize(value);
}

export function eventLabel(value) {
  return EVENT_LABELS[String(value || "")] || humanize(value);
}

export function humanize(value) {
  const normalized = String(value || "").trim();
  if (!normalized) return "—";
  return normalized
    .replaceAll("_", " ")
    .replaceAll("-", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

export function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

export function safeInternalPath(value, fallback = "/") {
  const path = String(value || "");
  if (!path.startsWith("/") || path.startsWith("//") || path.includes("\\")) {
    return fallback;
  }
  return path;
}

export function safeExternalUrl(value) {
  try {
    const parsed = new URL(String(value || ""));
    return ["http:", "https:"].includes(parsed.protocol) ? parsed.href : null;
  } catch {
    return null;
  }
}

export function formatDateTime(value) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(date);
}

export function formatDuration(seconds) {
  const total = Math.max(0, Math.round(Number(seconds) || 0));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const remainder = total % 60;
  if (hours) return `${hours} 时 ${String(minutes).padStart(2, "0")} 分`;
  if (minutes) return `${minutes} 分 ${String(remainder).padStart(2, "0")} 秒`;
  return `${remainder} 秒`;
}

export function formatNumber(value, maximumFractionDigits = 0) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  return new Intl.NumberFormat("zh-CN", { maximumFractionDigits }).format(number);
}

export function formatPercent(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  return `${Math.round(Math.max(0, Math.min(1, number)) * 100)}%`;
}

export function formatUsd(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  return `$${number.toFixed(number < 0.01 ? 4 : 2)}`;
}

export function truncate(value, length = 120) {
  const text = String(value || "");
  return text.length > length ? `${text.slice(0, Math.max(0, length - 1))}…` : text;
}

export function taskRoots(tasks) {
  const values = arrayOf(tasks);
  const ids = new Set(values.map((item) => item.task_id));
  return values.filter((item) => !item.parent_task_id || !ids.has(item.parent_task_id));
}

export function taskChildren(tasks) {
  const children = new Map();
  for (const task of arrayOf(tasks)) {
    const key = task.parent_task_id || "__root__";
    const items = children.get(key) || [];
    items.push(task);
    children.set(key, items);
  }
  for (const items of children.values()) {
    items.sort((left, right) => {
      if ((right.priority || 0) !== (left.priority || 0)) {
        return (right.priority || 0) - (left.priority || 0);
      }
      return String(left.created_at).localeCompare(String(right.created_at));
    });
  }
  return children;
}

export function filterRuns(runs, query, status) {
  const needle = String(query || "").trim().toLocaleLowerCase();
  return arrayOf(runs).filter((run) => {
    const queryMatches = !needle || String(run.query || "").toLocaleLowerCase().includes(needle);
    const statusMatches = !status || status === "all" || run.status === status;
    return queryMatches && statusMatches;
  });
}

export function filterTasks(tasks, query, status) {
  const needle = String(query || "").trim().toLocaleLowerCase();
  return arrayOf(tasks).filter((task) => {
    const haystack = [
      task.task_id,
      task.title,
      task.goal,
      task.kind,
      task.assigned_actor_id,
      ...(task.tags || []),
    ].join(" ").toLocaleLowerCase();
    const queryMatches = !needle || haystack.includes(needle);
    const statusMatches = !status || status === "all" || task.status === status;
    return queryMatches && statusMatches;
  });
}

export function runStatusTone(status) {
  return {
    completed: "success",
    running: "info",
    queued: "neutral",
    waiting_approval: "warning",
    failed: "danger",
    cancelled: "muted",
  }[status] || "neutral";
}

export function itemStatusTone(status) {
  return {
    completed: "success",
    succeeded: "success",
    running: "info",
    active: "info",
    ready: "info",
    waiting_approval: "warning",
    blocked: "warning",
    failed: "danger",
    cancelled: "muted",
    pruned: "muted",
    merged: "muted",
    deferred: "neutral",
    paused: "warning",
  }[status] || "neutral";
}

export function describeApiError(error) {
  if (!error) return "发生未知错误。";
  if (typeof error === "string") return error;
  return String(error.message || error.detail || error);
}

export function renderMarkdownSafe(markdown) {
  const lines = String(markdown || "").replaceAll("\r\n", "\n").split("\n");
  const output = [];
  let index = 0;
  let inCode = false;
  let codeLanguage = "";
  let codeLines = [];
  let listType = null;
  let listItems = [];

  const flushList = () => {
    if (!listType || !listItems.length) return;
    output.push(`<${listType}>${listItems.map((item) => `<li>${renderInline(item)}</li>`).join("")}</${listType}>`);
    listType = null;
    listItems = [];
  };

  while (index < lines.length) {
    const line = lines[index];
    const fence = line.match(/^```([\w-]*)\s*$/);
    if (fence) {
      flushList();
      if (!inCode) {
        inCode = true;
        codeLanguage = fence[1] || "";
        codeLines = [];
      } else {
        output.push(
          `<pre class="report-code" data-language="${escapeHtml(codeLanguage)}"><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`,
        );
        inCode = false;
        codeLanguage = "";
        codeLines = [];
      }
      index += 1;
      continue;
    }
    if (inCode) {
      codeLines.push(line);
      index += 1;
      continue;
    }
    if (!line.trim()) {
      flushList();
      index += 1;
      continue;
    }
    const heading = line.match(/^(#{1,4})\s+(.+)$/);
    if (heading) {
      flushList();
      const level = heading[1].length;
      const id = slugify(heading[2], index);
      output.push(`<h${level} id="${escapeHtml(id)}">${renderInline(heading[2])}</h${level}>`);
      index += 1;
      continue;
    }
    const unordered = line.match(/^\s*[-*+]\s+(.+)$/);
    const ordered = line.match(/^\s*\d+[.)]\s+(.+)$/);
    if (unordered || ordered) {
      const nextType = unordered ? "ul" : "ol";
      if (listType && listType !== nextType) flushList();
      listType = nextType;
      listItems.push((unordered || ordered)[1]);
      index += 1;
      continue;
    }
    const quote = line.match(/^>\s?(.*)$/);
    if (quote) {
      flushList();
      output.push(`<blockquote>${renderInline(quote[1])}</blockquote>`);
      index += 1;
      continue;
    }
    if (/^\s*([-*_])\1\1+\s*$/.test(line)) {
      flushList();
      output.push("<hr />");
      index += 1;
      continue;
    }

    flushList();
    const paragraph = [line.trim()];
    index += 1;
    while (
      index < lines.length
      && lines[index].trim()
      && !/^(#{1,4})\s+/.test(lines[index])
      && !/^\s*[-*+]\s+/.test(lines[index])
      && !/^\s*\d+[.)]\s+/.test(lines[index])
      && !/^>\s?/.test(lines[index])
      && !/^```/.test(lines[index])
    ) {
      paragraph.push(lines[index].trim());
      index += 1;
    }
    output.push(`<p>${renderInline(paragraph.join(" "))}</p>`);
  }
  flushList();
  if (inCode) {
    output.push(
      `<pre class="report-code" data-language="${escapeHtml(codeLanguage)}"><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`,
    );
  }
  return output.join("");
}

export function renderInline(value) {
  const raw = String(value || "");
  const linkPattern = /\[([^\]]+)\]\(([^)\s]+)(?:\s+"[^"]*")?\)/g;
  const output = [];
  let cursor = 0;
  for (const match of raw.matchAll(linkPattern)) {
    output.push(formatPlainInline(raw.slice(cursor, match.index)));
    const url = safeExternalUrl(match[2]);
    if (url) {
      output.push(
        `<a href="${escapeHtml(url)}" target="_blank" rel="noreferrer noopener">${formatPlainInline(match[1])}</a>`,
      );
    } else {
      output.push(formatPlainInline(match[0]));
    }
    cursor = Number(match.index) + match[0].length;
  }
  output.push(formatPlainInline(raw.slice(cursor)));
  return output.join("");
}

function formatPlainInline(value) {
  const raw = String(value || "");
  const urlPattern = /https?:\/\/[^\s<>()\[\]{}"'，。；：（）]+/g;
  const output = [];
  let cursor = 0;
  for (const match of raw.matchAll(urlPattern)) {
    output.push(formatEscapedInline(raw.slice(cursor, match.index)));
    const url = safeExternalUrl(match[0]);
    if (url) {
      output.push(
        `<a href="${escapeHtml(url)}" target="_blank" rel="noreferrer noopener">${escapeHtml(match[0])}</a>`,
      );
    } else {
      output.push(formatEscapedInline(match[0]));
    }
    cursor = Number(match.index) + match[0].length;
  }
  output.push(formatEscapedInline(raw.slice(cursor)));
  return output.join("");
}

function formatEscapedInline(value) {
  return escapeHtml(value)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/__([^_]+)__/g, "<strong>$1</strong>")
    .replace(/(?<!\*)\*([^*]+)\*(?!\*)/g, "<em>$1</em>");
}

function slugify(value, fallback) {
  const slug = String(value || "")
    .normalize("NFKC")
    .toLocaleLowerCase()
    .replace(/[^\p{Letter}\p{Number}]+/gu, "-")
    .replace(/^-|-$/g, "");
  return slug || `section-${fallback}`;
}
