const htmlEscapeMap = {
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
  "'": "&#39;",
};

function escapeHTML(value) {
  return String(value ?? "").replace(/[&<>"']/g, char => htmlEscapeMap[char] || char);
}

function coerceNumber(value) {
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
}

function formatScore(value) {
  const num = coerceNumber(value);
  if (num === null) return "–";
  return num.toFixed(2);
}

function formatDelta(value) {
  const num = coerceNumber(value);
  if (num === null) return "–";
  const sign = num > 0 ? "+" : "";
  return `${sign}${num.toFixed(2)}`;
}

function formatGain(value) {
  const num = coerceNumber(value);
  if (num === null) return "–";
  return num.toFixed(3);
}

function formatConfidence(value) {
  const num = coerceNumber(value);
  if (num === null) return "–";
  return num.toFixed(2);
}

const FLAG_LABELS = {
  low_confidence: "Low confidence",
  high_hints: "High hint use",
  regression: "Regression",
};

const APPLIED_BY_STORAGE_KEY = "ailb-admin-applied-by";

function formatRelativeTimestamp(text) {
  if (!text) return "–";
  const date = new Date(text);
  if (Number.isNaN(date.getTime())) {
    return text;
  }
  const now = Date.now();
  const diffSeconds = Math.max(0, Math.round((now - date.getTime()) / 1000));
  if (diffSeconds < 60) return "just now";
  if (diffSeconds < 3600) return `${Math.round(diffSeconds / 60)} min ago`;
  if (diffSeconds < 86400) return `${Math.round(diffSeconds / 3600)} h ago`;
  if (diffSeconds < 86400 * 7) return `${Math.round(diffSeconds / 86400)} d ago`;
  return date.toLocaleString();
}

function formatAbsoluteTimestamp(text) {
  if (!text) return "–";
  const date = new Date(text);
  if (Number.isNaN(date.getTime())) {
    return text;
  }
  return date.toLocaleString();
}

function createBadge(label, variant = "default") {
  const badge = document.createElement("span");
  badge.className = variant === "danger" ? "badge badge-danger" : "badge";
  badge.textContent = label;
  return badge;
}

function normaliseEntry(entry) {
  const topic = entry.topic || entry.subject_id || null;
  const preScore = coerceNumber(entry.pre_score ?? entry.pretest_score);
  const postScore = coerceNumber(entry.post_score ?? entry.posttest_score);
  let scoreDelta = coerceNumber(entry.score_delta);
  if (scoreDelta === null && preScore !== null && postScore !== null) {
    scoreDelta = Number((postScore - preScore).toFixed(4));
  }
  const normalizedGain = coerceNumber(entry.normalized_gain);
  const preMax = coerceNumber(entry.pre_max_score);
  const postMax = coerceNumber(entry.post_max_score);
  const historyTail = Array.isArray(entry.history_tail) ? entry.history_tail : [];
  const levels = entry && typeof entry.levels === "object" && entry.levels !== null ? entry.levels : {};
  const manualOverride = entry && typeof entry.manual_override === "object" && entry.manual_override !== null ? entry.manual_override : null;
  const manualOverrides = Array.isArray(entry.manual_overrides) ? entry.manual_overrides : [];
  const flaggedRaw = Array.isArray(entry.flagged_reasons)
    ? entry.flagged_reasons.map(reason => String(reason))
    : entry.flagged_reasons
    ? [String(entry.flagged_reasons)]
    : [];
  const flaggedUnique = Array.from(new Set(flaggedRaw));
  const flagLow = Boolean(entry.flag_low_confidence) || flaggedUnique.includes("low_confidence");
  const flagHighHints = Boolean(entry.flag_high_hints) || flaggedUnique.includes("high_hints");
  const flagRegression = Boolean(entry.flag_regression) || flaggedUnique.includes("regression");
  const recentConfidence = coerceNumber(entry.recent_confidence);
  const confidenceTrend = coerceNumber(entry.confidence_trend);
  const recentScore = coerceNumber(entry.recent_score);
  const ciLower = coerceNumber(entry.confidence_interval_lower);
  const ciUpper = coerceNumber(entry.confidence_interval_upper);
  const ciMean = coerceNumber(entry.confidence_interval_mean);
  const ciMargin = coerceNumber(entry.confidence_interval_margin);
  const ciWidth = coerceNumber(entry.confidence_interval_width);
  const ciConfidence = coerceNumber(entry.confidence_interval_confidence_level);
  const ciSampleSize = Number.isFinite(Number(entry.confidence_interval_sample_size))
    ? Number(entry.confidence_interval_sample_size)
    : null;
  const hintCount = Number.isFinite(Number(entry.hint_count)) ? Number(entry.hint_count) : 0;
  const lowConfidenceCount = Number.isFinite(Number(entry.low_confidence_count))
    ? Number(entry.low_confidence_count)
    : 0;
  return {
    ...entry,
    topic,
    pre_score: preScore,
    post_score: postScore,
    score_delta: scoreDelta,
    normalized_gain: normalizedGain,
    pre_max_score: preMax,
    post_max_score: postMax,
    history_tail: historyTail,
    levels,
    manual_override: manualOverride,
    manual_overrides: manualOverrides,
    flagged_reasons: flaggedUnique,
    flag_low_confidence: flagLow,
    flag_high_hints: flagHighHints,
    flag_regression: flagRegression,
    stuck_flag: Boolean(entry.stuck_flag) || flagLow || flagHighHints || flagRegression,
    recent_confidence: recentConfidence,
    confidence_trend: confidenceTrend,
    recent_score: recentScore,
    confidence_interval_lower: ciLower,
    confidence_interval_upper: ciUpper,
    confidence_interval_mean: ciMean,
    confidence_interval_margin: ciMargin,
    confidence_interval_width: ciWidth,
    confidence_interval_confidence_level: ciConfidence,
    confidence_interval_sample_size: ciSampleSize,
    hint_count: hintCount,
    low_confidence_count: lowConfidenceCount,
  };
}

document.addEventListener("DOMContentLoaded", () => {
  const form = document.querySelector("#analyticsFilters");
  const subjectInput = document.querySelector("#filterSubject");
  const windowInput = document.querySelector("#filterWindow");
  const limitInput = document.querySelector("#filterLimit");
  const flaggedInput = document.querySelector("#filterFlagged");
  const statusEl = document.querySelector("#dashboardStatus");
  const tbody = document.querySelector("#analyticsBody");
  const sortButton = document.querySelector("#sortGain");
  const overlay = document.querySelector("#timelineOverlay");
  const timelineBody = document.querySelector("#timelineBody");
  const timelineTitle = document.querySelector("#timelineTitle");
  const timelineClose = document.querySelector("#timelineClose");

  const params = new URLSearchParams(window.location.search);
  if (params.get("subject_id")) {
    subjectInput.value = params.get("subject_id");
  }
  if (params.get("window_days")) {
    windowInput.value = params.get("window_days");
  }
  if (params.get("limit")) {
    limitInput.value = params.get("limit");
  }
  if (params.get("only_flagged") === "true") {
    flaggedInput.checked = true;
  }

  let analyticsData = [];
  let sortDirection = -1; // -1 => descending, 1 => ascending

  function updateQueryString(query) {
    const next = `${window.location.pathname}?${query.toString()}`;
    window.history.replaceState({}, "", next);
  }

  function updateSortIndicator() {
    if (!sortButton) return;
    const arrow = sortButton.querySelector(".arrow");
    if (arrow) {
      arrow.textContent = sortDirection === -1 ? "⇣" : "⇡";
    }
    const directionLabel = sortDirection === -1 ? "descending" : "ascending";
    sortButton.setAttribute("aria-label", `Sort by normalized gain (${directionLabel})`);
  }

  function ensureAppliedBy() {
    try {
      const stored = window.localStorage?.getItem(APPLIED_BY_STORAGE_KEY);
      if (stored) {
        return stored;
      }
    } catch (error) {
      console.warn("localStorage unavailable", error);
    }
    const input = window.prompt("Override applied by (email or name)", "") || "";
    const value = input.trim() || "teacher-dashboard";
    try {
      window.localStorage?.setItem(APPLIED_BY_STORAGE_KEY, value);
    } catch (error) {
      console.warn("Failed to store applied-by value", error);
    }
    return value;
  }

  function renderFlagBadges(entry) {
    const container = document.createElement("div");
    container.className = "flag-badges";
    if (!Array.isArray(entry.flagged_reasons) || entry.flagged_reasons.length === 0) {
      container.textContent = "—";
      return container;
    }
    entry.flagged_reasons.forEach(reason => {
      const label = FLAG_LABELS[reason] || reason;
      container.appendChild(createBadge(label, "danger"));
    });
    return container;
  }

  function renderEmptyRow(message) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 9;
    cell.className = "timeline-empty";
    cell.textContent = message;
    row.appendChild(cell);
    tbody.appendChild(row);
  }

  function applySort(data) {
    const sorted = [...data];
    sorted.sort((a, b) => {
      if (a.stuck_flag !== b.stuck_flag) {
        return a.stuck_flag ? -1 : 1;
      }
      if ((a.flagged_reasons?.length || 0) !== (b.flagged_reasons?.length || 0)) {
        return (b.flagged_reasons?.length || 0) - (a.flagged_reasons?.length || 0);
      }
      if ((a.hint_count || 0) !== (b.hint_count || 0)) {
        return (b.hint_count || 0) - (a.hint_count || 0);
      }
      const aVal = coerceNumber(a.normalized_gain);
      const bVal = coerceNumber(b.normalized_gain);
      const aScore = aVal === null ? -Infinity : aVal;
      const bScore = bVal === null ? -Infinity : bVal;
      if (aScore !== bScore) {
        return sortDirection === -1 ? bScore - aScore : aScore - bScore;
      }
      const aDelta = coerceNumber(a.score_delta);
      const bDelta = coerceNumber(b.score_delta);
      const aDeltaScore = aDelta === null ? -Infinity : aDelta;
      const bDeltaScore = bDelta === null ? -Infinity : bDelta;
      if (aDeltaScore !== bDeltaScore) {
        return sortDirection === -1 ? bDeltaScore - aDeltaScore : aDeltaScore - bDeltaScore;
      }
      const aUser = String(a.user_id || "");
      const bUser = String(b.user_id || "");
      if (aUser !== bUser) {
        return aUser.localeCompare(bUser);
      }
      const aTopic = String(a.topic || "");
      const bTopic = String(b.topic || "");
      return aTopic.localeCompare(bTopic);
    });
    return sorted;
  }

  function buildRow(entry) {
    const tr = document.createElement("tr");
    tr.tabIndex = 0;
    tr.setAttribute("role", "button");
    tr.setAttribute("aria-label", `Open timeline for ${entry.user_id || "learner"} in ${entry.topic || entry.subject_id || "topic"}`);
    if (entry.stuck_flag) {
      tr.classList.add("is-flagged");
    }

    const learnerCell = document.createElement("td");
    learnerCell.dataset.label = "Learner";
    const learnerName = document.createElement("div");
    learnerName.innerHTML = `<strong>${escapeHTML(entry.user_id || "–")}</strong>`;
    learnerCell.appendChild(learnerName);
    if (entry.stuck_flag) {
      learnerCell.appendChild(createBadge("Flagged", "danger"));
    }
    const learnerMeta = document.createElement("div");
    learnerMeta.className = "override-meta";
    const learnerMetaParts = [];
    if (entry.state_updated_at) {
      learnerMetaParts.push(`State ${formatRelativeTimestamp(entry.state_updated_at)}`);
    }
    if (entry.manual_override?.applied_by) {
      learnerMetaParts.push(`Override by ${entry.manual_override.applied_by}`);
    }
    learnerMeta.textContent = learnerMetaParts.join(" · ") || " ";
    learnerCell.appendChild(learnerMeta);
    tr.appendChild(learnerCell);

    const topicCell = document.createElement("td");
    topicCell.dataset.label = "Topic";
    topicCell.innerHTML = `<strong>${escapeHTML(entry.topic || entry.subject_id || "–")}</strong>`;
    const topicMeta = document.createElement("div");
    topicMeta.className = "override-meta";
    const parts = [];
    if (entry.window_days) parts.push(`Window ${entry.window_days}d`);
    if (entry.gain_strategy) parts.push(`Strategy ${entry.gain_strategy}`);
    if (entry.analytics_updated_at) parts.push(`Analytics ${formatRelativeTimestamp(entry.analytics_updated_at)}`);
    topicMeta.textContent = parts.join(" · ") || " ";
    topicCell.appendChild(topicMeta);
    tr.appendChild(topicCell);

    const levelCell = document.createElement("td");
    levelCell.dataset.label = "Current level";
    levelCell.innerHTML = `<strong>${escapeHTML(entry.current_level || "–")}</strong>`;
    const levelMeta = document.createElement("div");
    levelMeta.className = "override-meta";
    const levelParts = [];
    if (entry.recent_confidence !== null && entry.recent_confidence !== undefined) {
      levelParts.push(`Conf ${formatConfidence(entry.recent_confidence)}`);
    }
    if (
      entry.confidence_interval_lower !== null &&
      entry.confidence_interval_lower !== undefined &&
      entry.confidence_interval_upper !== null &&
      entry.confidence_interval_upper !== undefined
    ) {
      levelParts.push(
        `CI ${formatConfidence(entry.confidence_interval_lower)}–${formatConfidence(entry.confidence_interval_upper)}`,
      );
    }
    if (entry.recent_score !== null && entry.recent_score !== undefined) {
      levelParts.push(`Score ${formatScore(entry.recent_score)}`);
    }
    if (entry.manual_override?.target_level) {
      levelParts.push(`Override → ${entry.manual_override.target_level}`);
    }
    levelMeta.textContent = levelParts.join(" · ") || " ";
    levelCell.appendChild(levelMeta);
    tr.appendChild(levelCell);

    const flagsCell = document.createElement("td");
    flagsCell.dataset.label = "Flags";
    flagsCell.appendChild(renderFlagBadges(entry));
    const flagsMeta = document.createElement("div");
    flagsMeta.className = "override-meta";
    const flagParts = [];
    flagParts.push(`Hints ${entry.hint_count || 0}`);
    flagParts.push(`Low conf ${entry.low_confidence_count || 0}`);
    if (entry.confidence_trend !== null && entry.confidence_trend !== undefined) {
      flagParts.push(`Trend ${formatDelta(entry.confidence_trend)}`);
    }
    flagsMeta.textContent = flagParts.join(" · ") || " ";
    flagsCell.appendChild(flagsMeta);
    tr.appendChild(flagsCell);

    const preCell = document.createElement("td");
    preCell.dataset.label = "Pre score";
    preCell.textContent = formatScore(entry.pre_score);
    tr.appendChild(preCell);

    const postCell = document.createElement("td");
    postCell.dataset.label = "Post score";
    postCell.textContent = formatScore(entry.post_score);
    tr.appendChild(postCell);

    const deltaCell = document.createElement("td");
    deltaCell.dataset.label = "Δ score";
    deltaCell.textContent = formatDelta(entry.score_delta);
    tr.appendChild(deltaCell);

    const gainCell = document.createElement("td");
    gainCell.dataset.label = "Normalized gain";
    gainCell.textContent = formatGain(entry.normalized_gain);
    tr.appendChild(gainCell);

    const actionsCell = document.createElement("td");
    actionsCell.dataset.label = "Actions";
    actionsCell.className = "actions-cell";
    const statusDisplay = document.createElement("div");
    statusDisplay.className = "action-status";
    const forceButton = document.createElement("button");
    forceButton.type = "button";
    forceButton.textContent = "Force Next Step";
    forceButton.addEventListener("click", event => {
      event.preventDefault();
      event.stopPropagation();
      handleForceNextStep(entry, statusDisplay);
    });
    actionsCell.appendChild(forceButton);
    actionsCell.appendChild(statusDisplay);
    tr.appendChild(actionsCell);

    tr.addEventListener("click", () => openTimeline(entry));
    tr.addEventListener("keydown", event => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        openTimeline(entry);
      }
    });

    return tr;
  }

  function renderRows() {
    tbody.innerHTML = "";
    if (!analyticsData.length) {
      renderEmptyRow("No cohorts match the current filters.");
      if (statusEl) {
        statusEl.textContent = "No cohorts match the current filters.";
      }
      return;
    }
    const sorted = applySort(analyticsData);
    sorted.forEach(entry => {
      tbody.appendChild(buildRow(entry));
    });
    if (statusEl) {
      const direction = sortDirection === -1 ? "↓" : "↑";
      const flaggedCount = sorted.filter(item => item.stuck_flag).length;
      statusEl.textContent = `${sorted.length} cohort${sorted.length === 1 ? "" : "s"} · ${flaggedCount} flagged · sorted by g ${direction}.`;
    }
  }

  function showOverlay() {
    if (!overlay) return;
    overlay.classList.add("is-visible");
    overlay.setAttribute("aria-hidden", "false");
    document.body.style.overflow = "hidden";
    if (timelineClose) {
      timelineClose.focus({ preventScroll: true });
    }
  }

  function hideOverlay() {
    if (!overlay) return;
    overlay.classList.remove("is-visible");
    overlay.setAttribute("aria-hidden", "true");
    document.body.style.overflow = "";
  }

  function renderTimelineSummary(entry, data) {
    const wrapper = document.createElement("div");
    wrapper.className = "timeline-summary";
    const summary = data?.summary || {};
    const chips = [
      `Sessions ${summary.total_sessions ?? 0}`,
      `Open ${summary.open_sessions ?? 0}`,
      `Events in sessions ${summary.events_in_sessions ?? 0}`,
      `Loose events ${summary.loose_events ?? 0}`,
    ];
    if (entry.score_delta !== null && entry.score_delta !== undefined) {
      chips.push(`Δ ${formatDelta(entry.score_delta)}`);
    }
    if (entry.normalized_gain !== null && entry.normalized_gain !== undefined) {
      chips.push(`g ${formatGain(entry.normalized_gain)}`);
    }
    if (entry.current_level) {
      chips.push(`Level ${entry.current_level}`);
    }
    if (entry.recent_confidence !== null && entry.recent_confidence !== undefined) {
      chips.push(`Conf ${formatConfidence(entry.recent_confidence)}`);
    }
    if (entry.confidence_trend !== null && entry.confidence_trend !== undefined) {
      chips.push(`Trend ${formatDelta(entry.confidence_trend)}`);
    }
    if (
      entry.confidence_interval_lower !== null &&
      entry.confidence_interval_lower !== undefined &&
      entry.confidence_interval_upper !== null &&
      entry.confidence_interval_upper !== undefined
    ) {
      chips.push(
        `CI ${formatConfidence(entry.confidence_interval_lower)}–${formatConfidence(entry.confidence_interval_upper)}`,
      );
    }
    if (entry.confidence_interval_sample_size) {
      chips.push(`n ${entry.confidence_interval_sample_size}`);
    }
    if (entry.hint_count) {
      chips.push(`Hints ${entry.hint_count}`);
    }
    if (entry.low_confidence_count) {
      chips.push(`Low conf hits ${entry.low_confidence_count}`);
    }
    if (Array.isArray(entry.flagged_reasons)) {
      entry.flagged_reasons.forEach(reason => {
        chips.push(`⚠ ${FLAG_LABELS[reason] || reason}`);
      });
    }
    if (entry.manual_override?.target_level) {
      chips.push(`Override → ${entry.manual_override.target_level}`);
    }
    chips.forEach(text => {
      const span = document.createElement("span");
      span.textContent = text;
      wrapper.appendChild(span);
    });
    return wrapper;
  }

  function renderSessionEvents(events) {
    if (!Array.isArray(events) || !events.length) {
      const empty = document.createElement("div");
      empty.className = "timeline-empty";
      empty.textContent = "No events recorded in this session.";
      return empty;
    }
    const list = document.createElement("ul");
    events.slice(0, 25).forEach(event => {
      const item = document.createElement("li");
      const label = document.createElement("span");
      label.textContent = event.event_type || "event";
      const meta = document.createElement("span");
      const parts = [formatAbsoluteTimestamp(event.recorded_at)];
      if (event.score !== undefined && event.score !== null) {
        parts.push(`score ${formatScore(event.score)}`);
      }
      if (event.lesson_id) {
        parts.push(`lesson ${event.lesson_id}`);
      }
      meta.textContent = parts.join(" · ");
      item.appendChild(label);
      item.appendChild(meta);
      list.appendChild(item);
    });
    return list;
  }

  function renderSessions(sessions) {
    if (!Array.isArray(sessions) || !sessions.length) {
      const empty = document.createElement("div");
      empty.className = "timeline-empty";
      empty.textContent = "No sessions recorded in this window.";
      return empty;
    }
    const container = document.createElement("div");
    container.className = "timeline-sessions";
    sessions.forEach(session => {
      const card = document.createElement("div");
      card.className = "timeline-session";
      const header = document.createElement("header");
      const title = document.createElement("div");
      title.textContent = `${session.session_type || "Session"} · ${formatAbsoluteTimestamp(session.started_at)}`;
      header.appendChild(title);
      const status = document.createElement("div");
      const details = [];
      if (session.subject_id) details.push(session.subject_id);
      if (session.session_id) details.push(session.session_id);
      details.push(session.ended_at ? `Ended ${formatAbsoluteTimestamp(session.ended_at)}` : "In progress");
      status.textContent = details.join(" · ");
      header.appendChild(status);
      card.appendChild(header);
      card.appendChild(renderSessionEvents(session.events));
      container.appendChild(card);
    });
    return container;
  }

  function renderLooseEvents(events) {
    if (!Array.isArray(events) || !events.length) {
      return null;
    }
    const container = document.createElement("div");
    container.className = "timeline-session";
    const header = document.createElement("header");
    header.textContent = "Loose events";
    container.appendChild(header);
    container.appendChild(renderSessionEvents(events));
    return container;
  }

  async function handleForceNextStep(entry, statusNode) {
    if (!entry || !entry.user_id) {
      return;
    }
    const subject = entry.topic || entry.subject_id;
    if (!subject) {
      window.alert("Unable to determine subject for override.");
      return;
    }
    const levelKeys = Object.keys(entry.levels || {});
    const levelHint = levelKeys.length ? `Available: ${levelKeys.join(", ")}` : "";
    const defaultLevel = entry.current_level || levelKeys[0] || "";
    const targetInput = window.prompt(
      `Target Bloom level for ${entry.user_id} · ${subject}. ${levelHint}`.trim(),
      defaultLevel
    );
    if (targetInput === null) {
      return;
    }
    const targetLevel = targetInput.trim();
    if (!targetLevel) {
      window.alert("Override cancelled: no level provided.");
      return;
    }
    const notesDefault = entry.flagged_reasons?.length
      ? `Flagged (${entry.flagged_reasons.join(", ")})`
      : "";
    const notesInput = window.prompt("Notes for override (optional)", notesDefault || "");
    const notes = notesInput ? notesInput.trim() : undefined;
    const appliedBy = ensureAppliedBy();
    const windowDays = Math.max(1, parseInt(windowInput.value, 10) || 7);
    const payload = {
      user_id: entry.user_id,
      subject_id: subject,
      target_level: targetLevel,
      notes: notes || undefined,
      applied_by: appliedBy,
      metadata: {
        source: "admin-ui",
        flagged_reasons: entry.flagged_reasons,
        hint_count: entry.hint_count,
        low_confidence_count: entry.low_confidence_count,
      },
      window_days: windowDays,
    };
    try {
      if (statusNode) {
        statusNode.textContent = "Applying override…";
      }
      if (statusEl) {
        statusEl.textContent = `Applying override for ${entry.user_id}…`;
      }
      const res = await fetch("/teacher/learning-path/override", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      const data = await res.json();
      if (data?.analytics) {
        const normalized = normaliseEntry(data.analytics);
        const idx = analyticsData.findIndex(
          item =>
            item.user_id === normalized.user_id &&
            (item.topic || item.subject_id) === (normalized.topic || normalized.subject_id)
        );
        if (idx !== -1) {
          analyticsData[idx] = normalized;
        } else {
          analyticsData.push(normalized);
        }
      }
      if (statusNode) {
        statusNode.textContent = "Override saved, refreshing…";
      }
      await loadAnalytics();
      if (statusEl) {
        statusEl.textContent = `Override applied for ${entry.user_id} → ${targetLevel}.`;
      }
    } catch (error) {
      console.error(error);
      if (statusNode) {
        statusNode.textContent = `Override failed: ${error.message}`;
      }
      if (statusEl) {
        statusEl.textContent = `Override failed: ${error.message}`;
      }
    }
  }

  async function openTimeline(entry) {
    if (!entry || !entry.user_id) {
      return;
    }
    if (timelineBody) {
      timelineBody.innerHTML = '<div class="timeline-loading">Loading timeline…</div>';
    }
    if (timelineTitle) {
      timelineTitle.textContent = `${entry.user_id} · ${entry.topic || entry.subject_id || "topic"}`;
    }
    showOverlay();
    try {
      const query = new URLSearchParams({
        user_id: entry.user_id,
        limit_sessions: "12",
        limit_events: "40",
      });
      if (entry.topic) {
        query.set("subject_id", entry.topic);
      }
      const res = await fetch(`/journey/timeline?${query.toString()}`);
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      const data = await res.json();
      if (timelineBody) {
        timelineBody.innerHTML = "";
        timelineBody.appendChild(renderTimelineSummary(entry, data));
        timelineBody.appendChild(renderSessions(data.sessions));
        const loose = renderLooseEvents(data.loose_events);
        if (loose) {
          timelineBody.appendChild(loose);
        }
      }
    } catch (error) {
      if (timelineBody) {
        timelineBody.innerHTML = `<div class="timeline-empty">Failed to load timeline: ${escapeHTML(error.message)}</div>`;
      }
    }
  }

  async function loadAnalytics() {
    const subject = subjectInput.value.trim();
    const windowDays = Math.max(1, parseInt(windowInput.value, 10) || 7);
    const limit = Math.max(1, parseInt(limitInput.value, 10) || 50);
    const flaggedOnly = flaggedInput.checked;

    const query = new URLSearchParams();
    query.set("window_days", String(windowDays));
    query.set("limit", String(limit));
    if (subject) query.set("subject_id", subject);
    if (flaggedOnly) query.set("only_flagged", "true");
    updateQueryString(query);

    if (statusEl) {
      statusEl.textContent = "Loading analytics…";
    }
    tbody.innerHTML = "";

    try {
      const res = await fetch(`/teacher/analytics?${query.toString()}`);
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      const data = await res.json();
      if (!Array.isArray(data) || data.length === 0) {
        analyticsData = [];
        renderRows();
        return;
      }
      analyticsData = data.map(normaliseEntry);
      renderRows();
    } catch (error) {
      console.error(error);
      analyticsData = [];
      renderRows();
      if (statusEl) {
        statusEl.textContent = `Failed to load analytics: ${error.message}`;
      }
    }
  }

  if (form) {
    form.addEventListener("submit", event => {
      event.preventDefault();
      loadAnalytics();
    });
  }

  if (sortButton) {
    sortButton.addEventListener("click", () => {
      sortDirection = sortDirection === -1 ? 1 : -1;
      updateSortIndicator();
      renderRows();
    });
  }

  if (timelineClose) {
    timelineClose.addEventListener("click", hideOverlay);
  }
  if (overlay) {
    overlay.addEventListener("click", event => {
      if (event.target === overlay) {
        hideOverlay();
      }
    });
  }
  document.addEventListener("keydown", event => {
    if (event.key === "Escape" && overlay?.classList.contains("is-visible")) {
      hideOverlay();
    }
  });

  updateSortIndicator();
  loadAnalytics();
});
