// ===== App Boot =====
document.addEventListener("DOMContentLoaded", () => {
  const THEMES = [
    { id: "business", label: "Business Process", topic: "business_process" },
    { id: "language", label: "Language", topic: "language" },
    { id: "math", label: "Mathematics", topic: "mathematics" }
  ];

  // ===== Store =====
  const store = {
    token: localStorage.getItem("auth_token") || null,
    userId: localStorage.getItem("user_id") || null,
    mode: "simple",
    applyMode: "auto",
    maxTokens: "",
    sessions: {},
    currentSessionId: null,
    initialized: false,
    defaultSessions(){
      const base = {};
      THEMES.forEach(theme => { base[theme.id] = []; });
      return base;
    },
    prefKey(name){
      return this.userId ? `${name}_${this.userId}` : name;
    },
    sessionKey(){
      return this.userId ? `sessions_${this.userId}` : "sessions";
    },
    transcriptKey(sessionId){
      return this.userId ? `chat_${this.userId}_${sessionId}` : `chat_${sessionId}`;
    },
    ensureThemeBuckets(){
      THEMES.forEach(theme => {
        if(!Array.isArray(this.sessions[theme.id])){
          this.sessions[theme.id] = [];
        }
      });
    },
    findSession(sessionId){
      for(const theme of THEMES){
        const arr = this.sessions[theme.id] || [];
        const session = arr.find(s => s.id === sessionId);
        if(session){
          return { session, theme };
        }
      }
      return { session: null, theme: null };
    },
    createSession(themeId){
      const theme = THEMES.find(t => t.id === themeId) || THEMES[0];
      const session = {
        id: `${theme.id}-${Date.now()}`,
        themeId: theme.id,
        topic: theme.topic,
        title: "New chat",
        createdAt: Date.now()
      };
      this.sessions[theme.id].unshift(session);
      this.saveSessions();
      this.setCurrentSession(session.id);
      this.setTranscript(session.id, []);
      return session;
    },
    ensureDefaultSession(){
      this.ensureThemeBuckets();
      let firstSession = null;
      THEMES.forEach(theme => {
        if((this.sessions[theme.id] || []).length === 0){
          const session = this.createSession(theme.id);
          if(!firstSession) firstSession = session;
        }
      });
      if(!firstSession){
        const theme = THEMES[0];
        firstSession = (this.sessions[theme.id] || [])[0];
        if(!firstSession){
          firstSession = this.createSession(theme.id);
        }
      }
      return firstSession;
    },
    initForUser(userId){
      this.userId = userId;
      const storedSessions = localStorage.getItem(this.sessionKey());
      try{
        this.sessions = storedSessions ? JSON.parse(storedSessions) : this.defaultSessions();
      }catch{
        this.sessions = this.defaultSessions();
      }
      this.ensureThemeBuckets();
      const storedMode = localStorage.getItem(this.prefKey("mode"));
      const storedApply = localStorage.getItem(this.prefKey("apply_mode"));
      const storedMax = localStorage.getItem(this.prefKey("max_tokens"));
      this.mode = storedMode || "simple";
      this.applyMode = storedApply || "auto";
      this.maxTokens = storedMax || "";

      const savedSession = localStorage.getItem(this.prefKey("current_session"));
      if(savedSession && this.findSession(savedSession).session){
        this.currentSessionId = savedSession;
      }else{
        const first = this.ensureDefaultSession();
        this.currentSessionId = first?.id || null;
      }
      this.initialized = true;
    },
    saveSessions(){
      if(this.userId){
        localStorage.setItem(this.sessionKey(), JSON.stringify(this.sessions));
      }
    },
    savePreferences(){
      if(!this.userId) return;
      localStorage.setItem(this.prefKey("mode"), this.mode);
      localStorage.setItem(this.prefKey("apply_mode"), this.applyMode);
      localStorage.setItem(this.prefKey("max_tokens"), this.maxTokens);
      if(this.currentSessionId){
        localStorage.setItem(this.prefKey("current_session"), this.currentSessionId);
      }
    },
    setCurrentSession(sessionId){
      const { session } = this.findSession(sessionId);
      if(!session) return;
      this.currentSessionId = sessionId;
      this.savePreferences();
    },
    setTranscript(sessionId, arr){
      localStorage.setItem(this.transcriptKey(sessionId), JSON.stringify(arr));
    },
    getTranscript(sessionId){
      try{
        const raw = localStorage.getItem(this.transcriptKey(sessionId));
        return raw ? JSON.parse(raw) : [];
      }catch{
        return [];
      }
    },
    updateSessionTitle(sessionId, title){
      const { session, theme } = this.findSession(sessionId);
      if(!session || !theme) return;
      session.title = title;
      this.saveSessions();
    }
  };

  // ===== DOM =====
  const $ = sel => document.querySelector(sel);
  const authScreen = $("#authScreen");
  const appScreen = $("#appScreen");
  const historyUser = $("#historyUser");
  const sessionUser = $("#sessionUser");
  const activeThemeEl = $("#activeTheme");
  const currentChatTitle = $("#currentChatTitle");
  const activeTopic = $("#activeTopic");
  const chatLog = $("#chatLog");
  const statusEl = $("#status");
  const msgInput = $("#msgInput");
  const sendBtn = $("#sendBtn");
  const modeSelect = $("#modeSelect");
  const applyModeEl = $("#applyMode");
  const maxTokensEl = $("#maxTokens");
  const applyRow = $("#applyRow");
  const btnDiag = $("#btnDiag");
  const btnRefreshDB = $("#btnRefreshDB");
  const btnOpenAdmin = $("#btnOpenAdmin");
  const btnClearChat = $("#btnClearChat");
  const btnLogout = $("#btnLogout");
  const oversightCard = $("#teacherOversight");
  const oversightStatus = $("#oversightStatus");
  const oversightTableBody = $("#oversightTableBody");
  const refreshOversightBtn = $("#refreshOversight");
  const toast = $("#toast");
  const viewButtons = document.querySelectorAll(".tab-btn[data-view]");
  const chatView = $("#chatView");
  const profileView = $("#profileView");
  const profileStatus = $("#profileStatus");
  const profileTableBody = $("#profileTableBody");
  const profileEmpty = $("#profileEmpty");
  const profileNextList = $("#profileNextList");
  const refreshProfileBtn = $("#refreshProfile");

  const btnLogin = $("#btnLogin");
  const btnRegister = $("#btnRegister");
  const authMsg = $("#authMsg");

  const historyLists = {
    business: $("#history-business"),
    language: $("#history-language"),
    math: $("#history-math")
  };

  const themeMap = THEMES.reduce((acc, t) => { acc[t.id] = t; return acc; }, {});

  let activeView = "chat";
  let profileData = null;
  let teacherOversightTimer = null;

  function resetProfileView(){
    profileData = null;
    if(profileView) profileView.removeAttribute("data-loaded");
    if(profileTableBody) profileTableBody.innerHTML = "";
    if(profileNextList) profileNextList.innerHTML = "";
    if(profileEmpty) profileEmpty.classList.remove("hidden");
    if(profileStatus) profileStatus.textContent = "No data loaded yet.";
  }

  function renderOversightPlaceholder(text){
    if(!oversightTableBody) return;
    oversightTableBody.innerHTML = "";
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 4;
    cell.className = "oversight-empty";
    cell.textContent = text;
    row.appendChild(cell);
    oversightTableBody.appendChild(row);
  }

  function resetTeacherOversight(message){
    const note = message || "Sign in to load teacher analytics.";
    if(oversightStatus) oversightStatus.textContent = note;
    renderOversightPlaceholder(message || "Sign in to view teacher oversight.");
  }

  function stopTeacherOversightUpdates(){
    if(teacherOversightTimer){
      clearInterval(teacherOversightTimer);
      teacherOversightTimer = null;
    }
  }

  // ===== Helpers =====
  function showAuth(){
    authScreen?.classList.remove("hidden");
    appScreen?.classList.add("hidden");
    setActiveView("chat");
  }

  function showApp(){
    authScreen?.classList.add("hidden");
    appScreen?.classList.remove("hidden");
    setActiveView("chat");
  }

  function showToast(text, ms=2500){
    if(!toast) return;
    toast.textContent = text;
    toast.hidden = false;
    setTimeout(() => { toast.hidden = true; }, ms);
  }

  function setAuthFeedback(msg){
    if(authMsg) authMsg.textContent = msg;
  }

  function setBusy(isBusy, label="generating…"){
    if(sendBtn) sendBtn.disabled = isBusy;
    if(msgInput) msgInput.disabled = isBusy;
    if(statusEl) statusEl.innerHTML = isBusy ? `<span class="loader"></span> ${label}` : "ready";
  }

  function formatDateTime(value){
    if(!value) return "–";
    const date = new Date(value);
    if(Number.isNaN(date.getTime())) return String(value);
    return date.toLocaleString();
  }

  function stripThink(text){
    return (text || "").replace(/<think>[\s\S]*?<\/think>/g, "").trim();
  }

  function escapeHtml(str){
    return String(str ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function createSparkline(history){
    const span = document.createElement("span");
    span.className = "sparkline";
    if(!Array.isArray(history) || history.length === 0){
      span.classList.add("sparkline-muted");
      span.textContent = "–";
      span.title = "No progress history";
      return span;
    }
    const glyphs = ["▁","▂","▃","▄","▅","▆","▇","█"];
    const values = history.map((item, index) => {
      const delta = typeof item?.delta === "number" ? item.delta : 0;
      const confidence = typeof item?.confidence === "number" ? item.confidence : 0.5;
      const correctPenalty = item?.correct === false ? -0.3 : 0;
      return delta + confidence + correctPenalty + index * 0.05;
    });
    const min = Math.min.apply(null, values);
    const max = Math.max.apply(null, values);
    const range = max - min || 1;
    span.textContent = values
      .map(val => {
        const normalized = (val - min) / range;
        const idx = Math.max(0, Math.min(glyphs.length - 1, Math.round(normalized * (glyphs.length - 1))));
        return glyphs[idx];
      })
      .join("");
    const tooltipLines = history.map(item => {
      const level = item?.bloom_level || item?.level || "–";
      const delta = typeof item?.delta === "number" ? Number(item.delta).toFixed(2) : "0.00";
      const confidence = typeof item?.confidence === "number" ? Number(item.confidence).toFixed(2) : "–";
      const mark = item?.correct === false ? "✕" : "✓";
      return `${level}: Δ${delta} ${mark} (conf ${confidence})`;
    });
    span.title = escapeHtml(tooltipLines.join("\n"));
    if(values[values.length - 1] < values[0]){
      span.style.color = "var(--danger)";
    }
    return span;
  }

  function createBloomChips(levels){
    const wrap = document.createElement("div");
    wrap.className = "bloom-chip-row";
    const entries = Object.entries(levels || {}).filter(([, value]) => typeof value === "number" && Number.isFinite(value));
    if(entries.length === 0){
      wrap.classList.add("muted");
      wrap.textContent = "No Bloom data";
      return wrap;
    }
    entries.sort((a, b) => String(a[0] || "").localeCompare(String(b[0] || "")));
    entries.forEach(([level, value]) => {
      const chip = document.createElement("span");
      chip.className = "bloom-chip";
      const label = document.createElement("span");
      label.className = "chip-label";
      label.textContent = level || "–";
      const score = document.createElement("span");
      score.className = "chip-score";
      const percent = Math.round(Number(value) * 100);
      score.textContent = `${Number.isFinite(percent) ? percent : 0}%`;
      chip.title = `Mastery ${(Number(value) * 100).toFixed(1)}%`;
      chip.append(label, score);
      wrap.appendChild(chip);
    });
    return wrap;
  }

  function renderTeacherOversight(entries){
    if(!oversightTableBody) return;
    oversightTableBody.innerHTML = "";
    if(!Array.isArray(entries) || entries.length === 0){
      renderOversightPlaceholder("No stuck learners 🎉");
      return;
    }
    entries.forEach(entry => {
      const row = document.createElement("tr");
      if(entry?.stuck_flag) row.classList.add("stuck");

      const learnerCell = document.createElement("td");
      const header = document.createElement("div");
      header.className = "oversight-learner";
      const userId = entry?.user_id || "–";
      const subjectId = entry?.subject_id || "–";
      header.textContent = `${userId} · ${subjectId}`;
      learnerCell.appendChild(header);

      const meta = document.createElement("div");
      meta.className = "oversight-meta";
      const stats = [];
      if(entry?.hint_count) stats.push(`Hints ×${entry.hint_count}`);
      if(entry?.low_confidence_count) stats.push(`Low confidence ×${entry.low_confidence_count}`);
      if(typeof entry?.recent_score === "number") stats.push(`Score ${(entry.recent_score * 100).toFixed(0)}%`);
      if(typeof entry?.recent_confidence === "number") stats.push(`Conf ${(entry.recent_confidence * 100).toFixed(0)}%`);
      meta.textContent = stats.length ? stats.join(" • ") : "Flagged for review";
      learnerCell.appendChild(meta);

      const reasonList = document.createElement("div");
      reasonList.className = "reason-list";
      const reasonChips = [];
      if(Array.isArray(entry?.hint_events) && entry.hint_events.length){
        const lastHint = entry.hint_events[entry.hint_events.length - 1];
        const hintStamp = lastHint?.created_at ? formatDateTime(lastHint.created_at) : null;
        reasonChips.push(hintStamp ? `Last hint ${hintStamp}` : "Hints recently");
      }
      if(entry?.manual_override){
        reasonChips.push("Manual override active");
      }
      if(entry?.state_updated_at){
        reasonChips.push(`State ${formatDateTime(entry.state_updated_at)}`);
      }
      reasonChips.filter(Boolean).forEach(text => {
        const chip = document.createElement("span");
        chip.className = "reason-chip";
        chip.textContent = text;
        reasonList.appendChild(chip);
      });
      if(reasonList.childElementCount) learnerCell.appendChild(reasonList);
      row.appendChild(learnerCell);

      const overrideCell = document.createElement("td");
      const actions = document.createElement("div");
      actions.className = "oversight-actions";
      const overrideBtn = document.createElement("button");
      overrideBtn.type = "button";
      overrideBtn.className = "btn ghost small-btn";
      overrideBtn.textContent = "Override";
      if(userId && userId !== "–"){
        overrideBtn.setAttribute("data-override-user", userId);
        overrideBtn.setAttribute("data-override-subject", subjectId !== "–" ? subjectId : "");
      }else{
        overrideBtn.disabled = true;
        overrideBtn.title = "Missing user identifier";
      }
      actions.appendChild(overrideBtn);
      const overrideMeta = document.createElement("div");
      overrideMeta.className = "override-meta";
      if(entry?.manual_override){
        const override = entry.manual_override;
        const target = override?.target_level ? `→ ${override.target_level}` : "Manual override";
        const parts = [target];
        if(override?.notes) parts.push(override.notes);
        if(override?.applied_by) parts.push(`by ${override.applied_by}`);
        overrideMeta.textContent = parts.join(" · ");
        if(override?.created_at) overrideMeta.title = `Applied ${formatDateTime(override.created_at)}`;
      }else if(entry?.current_level){
        overrideMeta.textContent = `Current level ${entry.current_level}`;
      }else{
        overrideMeta.textContent = "No override applied.";
      }
      actions.appendChild(overrideMeta);
      overrideCell.appendChild(actions);
      row.appendChild(overrideCell);

      const sparkCell = document.createElement("td");
      sparkCell.appendChild(createSparkline(entry?.history_tail || []));
      row.appendChild(sparkCell);

      const bloomCell = document.createElement("td");
      bloomCell.appendChild(createBloomChips(entry?.levels || {}));
      row.appendChild(bloomCell);

      oversightTableBody.appendChild(row);
    });
  }

  async function loadTeacherOversight(force=false){
    if(!oversightTableBody){
      return;
    }
    const token = store.token;
    if(!token){
      resetTeacherOversight();
      return;
    }
    if(oversightCard?.dataset.loading === "true" && !force){
      return;
    }
    if(force){
      renderOversightPlaceholder("Loading…");
    }
    if(oversightStatus) oversightStatus.textContent = "Loading stuck learners…";
    oversightCard?.setAttribute("data-loading", "true");
    try{
      const headers = { Authorization: `Bearer ${token}` };
      const res = await fetch(`/teacher/analytics?only_flagged=true&limit=50`, { headers });
      if(!res.ok){
        const msg = await res.text();
        throw new Error(msg || "Teacher analytics failed");
      }
      const data = await res.json();
      if(store.token !== token){
        resetTeacherOversight();
        return;
      }
      const entries = Array.isArray(data) ? data : [];
      renderTeacherOversight(entries);
      if(oversightStatus){
        oversightStatus.textContent = entries.length ? `Updated ${new Date().toLocaleTimeString()}` : "No stuck learners detected.";
      }
      if(oversightCard) oversightCard.dataset.loaded = "true";
    }catch(err){
      console.error(err);
      if(oversightStatus) oversightStatus.textContent = "Could not load teacher analytics.";
      renderOversightPlaceholder("Unable to load oversight.");
    }finally{
      oversightCard?.removeAttribute("data-loading");
    }
  }

  function startTeacherOversightUpdates(){
    if(!oversightTableBody){
      return;
    }
    stopTeacherOversightUpdates();
    if(!store.token){
      resetTeacherOversight();
      return;
    }
    loadTeacherOversight(true);
    teacherOversightTimer = setInterval(() => loadTeacherOversight(false), 60000);
  }

  resetProfileView();
  resetTeacherOversight();

  function formatInline(text){
    const escaped = escapeHtml(text);
    const segments = escaped.split(/`([^`]+)`/g);
    let out = "";
    for(let i = 0; i < segments.length; i++){
      if(i % 2 === 1){
        out += `<code>${segments[i]}</code>`;
      }else{
        let segment = segments[i];
        segment = segment.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
        segment = segment.replace(/__(.+?)__/g, "<strong>$1</strong>");
        segment = segment.replace(/\*(.+?)\*/g, "<em>$1</em>");
        segment = segment.replace(/_(.+?)_/g, "<em>$1</em>");
        out += segment;
      }
    }
    return out;
  }

  async function loadProfile(force=false){
    if(!store.userId){
      return showToast("Please sign in first.");
    }
    if(profileView?.dataset.loading === "true") return;
    if(!force && profileView?.dataset.loaded === "true" && profileData){
      renderProfile(profileData);
      return;
    }
    if(profileStatus) profileStatus.textContent = "Loading profile…";
    if(profileView) profileView.dataset.loading = "true";
    try{
      const res = await fetch(`/profile?user_id=${encodeURIComponent(store.userId)}`);
      if(!res.ok){
        const msg = await res.text();
        throw new Error(msg || "Profile could not be loaded");
      }
      const data = await res.json();
      profileData = data;
      renderProfile(data);
      if(profileView) profileView.dataset.loaded = "true";
    }catch(err){
      console.error(err);
      if(profileStatus) profileStatus.textContent = "Could not load profile.";
    }finally{
      if(profileView) profileView.removeAttribute("data-loading");
    }
  }

  function renderProfile(data){
    if(!profileTableBody) return;
    if(profileStatus) profileStatus.textContent = "";
    profileTableBody.innerHTML = "";
    const entries = Object.entries((data?.skills) || {});
    if(entries.length === 0){
      profileEmpty?.classList.remove("hidden");
    }else{
      profileEmpty?.classList.add("hidden");
      entries.sort((a, b) => a[0].localeCompare(b[0]));
      entries.forEach(([skill, detail]) => {
        const row = document.createElement("tr");
        const skillCell = document.createElement("td");
        skillCell.textContent = skill;
        const masteryCell = document.createElement("td");
        const masteryValue = detail?.theta ?? detail?.p_know ?? detail?.level ?? null;
        masteryCell.textContent = masteryValue == null ? "–" : Number(masteryValue).toFixed(3);
        const bloomCell = document.createElement("td");
        const bloomInfo = (data?.bloom_progress || {})[skill] || {};
        const bloomLevel = detail?.bloom_level || bloomInfo?.current_level;
        bloomCell.textContent = bloomLevel || "–";
        const updatedCell = document.createElement("td");
        const updatedSource = detail?.bloom_updated_at || detail?.updated_at || bloomInfo?.updated_at;
        updatedCell.textContent = formatDateTime(updatedSource);
        const actionCell = document.createElement("td");
        const link = document.createElement("button");
        link.type = "button";
        link.className = "link-btn";
        link.textContent = "View evidence";
        link.addEventListener("click", () => showEvidenceForSkill(skill));
        actionCell.appendChild(link);
        row.append(skillCell, masteryCell, bloomCell, updatedCell, actionCell);
        profileTableBody.appendChild(row);
      });
    }

    if(profileNextList){
      profileNextList.innerHTML = "";
      const due = Array.isArray(data?.next_due) ? data.next_due : [];
      if(due.length === 0){
        const li = document.createElement("li");
        li.className = "muted";
        li.textContent = "No recommendations available.";
        profileNextList.appendChild(li);
      }else{
        due.forEach(item => {
          const li = document.createElement("li");
          const title = document.createElement("div");
          title.className = "next-title";
          title.textContent = item?.skill || item?.activity || "Recommendation";
          const reason = document.createElement("div");
          reason.className = "next-reason";
          reason.textContent = item?.reason || "–";
          const time = document.createElement("div");
          time.className = "next-time";
          time.textContent = formatDateTime(item?.created_at);
          li.append(title, reason, time);

          if(item?.modality){
            const modality = document.createElement("div");
            modality.className = "next-modality";
            modality.textContent = `Modality: ${item.modality}`;
            li.appendChild(modality);
          }

          const highlights = Array.isArray(item?.preference_highlights) ? item.preference_highlights : [];
          if(highlights.length){
            const list = document.createElement("ul");
            list.className = "next-preference-highlights";
            highlights.forEach(text => {
              const highlightItem = document.createElement("li");
              highlightItem.textContent = text;
              list.appendChild(highlightItem);
            });
            li.appendChild(list);
          }
          profileNextList.appendChild(li);
        });
      }
    }
  }

  async function showEvidenceForSkill(skill){
    if(!store.userId){
      return showToast("Please sign in first.");
    }
    if(!skill){
      return showToast("Unknown skill.");
    }
    const bloomInfo = (profileData?.bloom_progress || {})[skill] || {};
    const historyEntries = Array.isArray(bloomInfo?.history) ? bloomInfo.history : [];
    const historyLines = historyEntries.map(item => {
      const timestamp = formatDateTime(item?.created_at);
      const fromLevel = item?.previous_level || "–";
      const toLevel = item?.new_level || "–";
      const kLevel = item?.k_level ? ` [K-level ${item.k_level}]` : "";
      const metrics = [];
      if(typeof item?.average_score === "number"){
        metrics.push(`avg ${(item.average_score ?? 0).toFixed(2)}`);
      }
      if(typeof item?.attempts_considered === "number"){
        metrics.push(`${item.attempts_considered} attempts`);
      }
      const metricText = metrics.length ? ` (${metrics.join(", ")})` : "";
      const rationale = item?.reason ? ` — ${item.reason}` : "";
      return `[${timestamp}] ${fromLevel} → ${toLevel}${kLevel}${metricText}${rationale}`;
    });
    try{
      if(profileStatus) profileStatus.textContent = `Loading evidence for ${skill}…`;
      const res = await fetch(`/db/journey?user_id=${encodeURIComponent(store.userId)}&limit=200`);
      if(!res.ok){
        throw new Error("journey_fetch_failed");
      }
      const data = await res.json();
      const relevant = Array.isArray(data) ? data.filter(entry => {
        const payload = entry?.payload || {};
        const details = payload?.details || {};
        const hints = [
          payload?.skill,
          payload?.topic,
          details?.skill,
          details?.target_skill,
          details?.topic,
          entry?.event_type,
        ].filter(Boolean);
        return hints.some(val => typeof val === "string" && val.toLowerCase().includes(String(skill).toLowerCase()));
      }) : [];
      if(profileStatus) profileStatus.textContent = "";
      const sections = [];
      if(historyLines.length){
        sections.push(`Bloom level history for ${skill}:\n${historyLines.join("\n")}`);
      }
      if(relevant.length === 0){
        sections.push(`No evidence found for ${skill}.`);
      }else{
        sections.push(`Evidence for ${skill}:\n` + JSON.stringify(relevant, null, 2));
      }
      renderMessage("meta", sections.join("\n\n"));
    }catch(err){
      console.error(err);
      showToast("Could not load evidence.");
      if(profileStatus) profileStatus.textContent = "";
    }
  }

  function setActiveView(view){
    activeView = view === "profile" ? "profile" : "chat";
    viewButtons.forEach(btn => {
      const target = btn.getAttribute("data-view") || "chat";
      btn.classList.toggle("active", target === activeView);
    });
    if(activeView === "profile"){
      chatView?.classList.add("hidden");
      profileView?.classList.remove("hidden");
      loadProfile(false);
    }else{
      chatView?.classList.remove("hidden");
      profileView?.classList.add("hidden");
    }
  }

  function markdownToHtml(md){
    if(md == null) return "";
    const lines = String(md).replace(/\r\n/g, "\n").split(/\n/);
    let html = "";
    let inList = false;
    let inCode = false;
    let codeLines = [];

    const flushList = () => {
      if(inList){
        html += "</ul>";
        inList = false;
      }
    };

    const flushCode = () => {
      if(inCode){
        html += `<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`;
        codeLines = [];
        inCode = false;
      }
    };

    for(const rawLine of lines){
      const line = rawLine;
      const fenceMatch = line.trim().match(/^```/);
      if(fenceMatch){
        if(inCode){
          flushCode();
        }else{
          flushList();
          inCode = true;
          codeLines = [];
        }
        continue;
      }

      if(inCode){
        codeLines.push(line);
        continue;
      }

      const listMatch = line.match(/^\s*[-*+]\s+(.*)$/);
      if(listMatch){
        if(!inList){
          flushCode();
          html += "<ul>";
          inList = true;
        }
        html += `<li>${formatInline(listMatch[1])}</li>`;
        continue;
      }

      flushList();

      const headingMatch = line.match(/^(#{1,6})\s+(.*)$/);
      if(headingMatch){
        const level = headingMatch[1].length;
        html += `<h${level}>${formatInline(headingMatch[2])}</h${level}>`;
        continue;
      }

      if(line.trim() === ""){
        html += "<br/>";
      }else{
        html += `<p>${formatInline(line)}</p>`;
      }
    }

    flushList();
    flushCode();

    if(!html){
      return `<p>${escapeHtml(md)}</p>`;
    }
    return html;
  }

  function setBubbleContent(bubble, role, text){
    if(!bubble) return;
    const raw = typeof text === "string" ? text : (text == null ? "" : String(text));
    if(role === "bot"){
      const html = markdownToHtml(raw);
      bubble.innerHTML = html || "";
    }else{
      bubble.textContent = raw;
    }
  }

  function truncate(text, max=40){
    if(text.length <= max) return text;
    return text.slice(0, max).trim() + "…";
  }

  function renderMessage(role, text){
    if(!chatLog) return null;
    const wrap = document.createElement("div");
    wrap.className = "msg " + (role === "user" ? "user" : role === "bot" ? "bot" : "meta");
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    setBubbleContent(bubble, role, text);
    wrap.appendChild(bubble);
    chatLog.appendChild(wrap);
    chatLog.scrollTop = chatLog.scrollHeight;
    return bubble;
  }

  function persistMessage(role, text){
    if(!store.currentSessionId) return;
    const arr = store.getTranscript(store.currentSessionId);
    arr.push({ role, text, ts: Date.now() });
    store.setTranscript(store.currentSessionId, arr);
    if(role === "user"){
      const { session } = store.findSession(store.currentSessionId);
      if(session && (!session.title || session.title === "New chat" || session.title === "Neuer Chat")){
        const newTitle = truncate(text);
        store.updateSessionTitle(store.currentSessionId, newTitle || "New chat");
        renderHistory();
      }
    }
  }

  function updateLastBotMessage(text){
    if(!store.currentSessionId) return;
    const arr = store.getTranscript(store.currentSessionId);
    for(let i = arr.length - 1; i >= 0; i--){
      if(arr[i].role === "bot"){
        arr[i].text = text;
        break;
      }
    }
    store.setTranscript(store.currentSessionId, arr);
  }

  function renderTranscript(){
    if(!chatLog || !store.currentSessionId) return;
    chatLog.innerHTML = "";
    const transcript = store.getTranscript(store.currentSessionId);
    if(transcript.length === 0){
      renderMessage("meta", "New chat started.");
    }else{
      transcript.forEach(entry => renderMessage(entry.role, entry.text));
    }
  }

  function renderHistory(){
    THEMES.forEach(theme => {
      const listEl = historyLists[theme.id];
      if(!listEl) return;
      listEl.innerHTML = "";
      const sessions = store.sessions[theme.id] || [];
      if(sessions.length === 0){
        const empty = document.createElement("div");
        empty.className = "muted";
        empty.textContent = "No chats yet";
        listEl.appendChild(empty);
        return;
      }
      sessions.forEach(session => {
        const item = document.createElement("div");
        item.className = "history-item" + (session.id === store.currentSessionId ? " active" : "");
        const title = document.createElement("div");
        title.className = "history-item-title";
        title.textContent = session.title || "New chat";
        const meta = document.createElement("div");
        meta.className = "history-item-meta";
        const date = new Date(session.createdAt);
        meta.textContent = date.toLocaleDateString();
        item.appendChild(title);
        item.appendChild(meta);
        item.onclick = () => {
          store.setCurrentSession(session.id);
          renderActiveSession();
          renderTranscript();
        };
        listEl.appendChild(item);
      });
    });
  }

  function renderActiveSession(){
    const { session, theme } = store.findSession(store.currentSessionId || "");
    if(!session || !theme){
      activeThemeEl && (activeThemeEl.textContent = "–");
      currentChatTitle && (currentChatTitle.textContent = "New chat");
      activeTopic && (activeTopic.textContent = "no topic");
      return;
    }
    if(activeThemeEl) activeThemeEl.textContent = theme.label;
    if(currentChatTitle) currentChatTitle.textContent = session.title || "New chat";
    if(activeTopic) activeTopic.textContent = session.topic || theme.topic || "no topic";
    renderHistory();
  }

  function toggleApplyRow(){
    if(applyRow) applyRow.style.display = store.mode === "tutor" ? "flex" : "none";
  }

  function syncControlsFromStore(){
    if(modeSelect) modeSelect.value = store.mode;
    if(applyModeEl) applyModeEl.value = store.applyMode;
    if(maxTokensEl) maxTokensEl.value = store.maxTokens;
    toggleApplyRow();
  }

  // ===== Auth Actions =====
  if(btnRegister) btnRegister.onclick = async () => {
    const user = ($("#authUser")?.value || "").trim();
    const email = ($("#authEmail")?.value || "").trim();
    const pass = $("#authPass")?.value || "";
    if(!user || !pass){
      return setAuthFeedback("User ID and password required.");
    }
    setAuthFeedback("");
    try{
      const res = await fetch("/auth/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: user, email: email || null, password: pass })
      });
      const data = await res.json();
      if(!res.ok) throw new Error(data.detail || "Registration failed");
      showToast("Registration successful. Please sign in.");
    }catch(err){
      setAuthFeedback(err.message);
    }
  };

  if(btnLogin) btnLogin.onclick = async () => {
    const user = ($("#authUser")?.value || "").trim();
    const pass = $("#authPass")?.value || "";
    if(!user || !pass){
      return setAuthFeedback("User ID and password required.");
    }
    setAuthFeedback("");
    try{
      const res = await fetch("/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: user, password: pass })
      });
      const data = await res.json();
      if(!res.ok) throw new Error(data.detail || "Login failed");
      store.token = data.token;
      store.userId = data.user_id;
      localStorage.setItem("auth_token", store.token);
      localStorage.setItem("user_id", store.userId);
      store.initForUser(store.userId);
      syncControlsFromStore();
      renderActiveSession();
      renderTranscript();
      updateUserBadges();
      resetProfileView();
      resetTeacherOversight();
      showApp();
      startTeacherOversightUpdates();
      showToast("Signed in.");
    }catch(err){
      setAuthFeedback(err.message);
    }
  };

  if(btnLogout) btnLogout.onclick = () => {
    store.token = null;
    store.userId = null;
    store.currentSessionId = null;
    localStorage.removeItem("auth_token");
    localStorage.removeItem("user_id");
    resetProfileView();
    stopTeacherOversightUpdates();
    resetTeacherOversight();
    if(sessionUser) sessionUser.textContent = "not signed in";
    if(historyUser) historyUser.textContent = "–";
    showAuth();
    showToast("Signed out.");
  };

  function updateUserBadges(){
    if(sessionUser) sessionUser.textContent = store.userId || "not signed in";
    if(historyUser) historyUser.textContent = store.userId || "–";
  }

  // ===== History actions =====
  document.querySelectorAll(".history-section .icon-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const themeId = btn.getAttribute("data-theme") || THEMES[0].id;
      const session = store.createSession(themeId);
      renderActiveSession();
      renderTranscript();
      showToast(`${themeMap[themeId]?.label || "Chat"} started.`);
    });
  });

  viewButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      const view = btn.getAttribute("data-view") || "chat";
      setActiveView(view);
    });
  });

  if(refreshProfileBtn){
    refreshProfileBtn.onclick = () => loadProfile(true);
  }

  // ===== Controls =====
  if(refreshOversightBtn){
    refreshOversightBtn.onclick = () => {
      if(!store.token){
        showToast("Please sign in first.");
        resetTeacherOversight();
        return;
      }
      loadTeacherOversight(true);
    };
  }

  if(oversightTableBody){
    oversightTableBody.addEventListener("click", evt => {
      const target = evt.target.closest("button[data-override-user]");
      if(!target) return;
      const userId = target.getAttribute("data-override-user") || "";
      if(!userId){
        return;
      }
      const subjectId = target.getAttribute("data-override-subject") || "";
      const params = new URLSearchParams({ user_id: userId });
      if(subjectId) params.set("subject_id", subjectId);
      const url = `/static/admin.html?${params.toString()}`;
      window.open(url, "_blank", "noopener");
    });
  }

  if(modeSelect){
    modeSelect.onchange = e => {
      store.mode = e.target.value;
      store.savePreferences();
      toggleApplyRow();
    };
  }
  if(applyModeEl){
    applyModeEl.onchange = e => {
      store.applyMode = e.target.value;
      store.savePreferences();
    };
  }
  if(maxTokensEl){
    maxTokensEl.oninput = e => {
      store.maxTokens = e.target.value;
      store.savePreferences();
    };
  }

  if(btnClearChat) btnClearChat.onclick = () => {
    if(!store.currentSessionId) return;
    store.setTranscript(store.currentSessionId, []);
    renderTranscript();
    showToast("Chat history cleared");
  };

  if(btnDiag) btnDiag.onclick = async () => {
    if(!store.userId){
      return showToast("Please sign in first.");
    }
    try{
      setBusy(true, "running diagnostics…");
      const res = await fetch("/diagnose/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: store.userId })
      });
      const data = await res.json();
      renderMessage("meta", "Diagnostic items:\n" + JSON.stringify(data.diagnostic_items, null, 2));
    }catch{
      showToast("Diagnostics failed");
    }finally{
      setBusy(false);
    }
  };

  function renderDbSnapshot(label, data, emptyHint){
    const isArray = Array.isArray(data);
    const isEmptyArray = isArray && data.length === 0;
    const isEmptyObject = !isArray && data && typeof data === "object" && Object.keys(data).length === 0;
    const isNullish = data == null;

    if(isEmptyArray || isEmptyObject || isNullish){
      const pretty = isArray ? "[]" : "{}";
      renderMessage("meta", `${label}: ${pretty}\n(no records yet – ${emptyHint})`);
      return;
    }

    renderMessage("meta", `${label}:\n` + JSON.stringify(data, null, 2));
  }

  if(btnRefreshDB) btnRefreshDB.onclick = async () => {
    if(!store.userId){
      return showToast("Please sign in first.");
    }
    try{
      setBusy(true, "loading database…");
      const [mastery, prompts, journey, bloom, chatOps] = await Promise.all([
        fetch("/db/mastery").then(r => r.json()),
        fetch(`/db/prompts?topic=${encodeURIComponent(store.findSession(store.currentSessionId || "").session?.topic || "")}`).then(r => r.json()),
        fetch(`/db/journey?user_id=${encodeURIComponent(store.userId)}`).then(r => r.json()),
        fetch(`/db/bloom?user_id=${encodeURIComponent(store.userId)}`).then(r => r.json()),
        fetch(`/db/chat_ops?user_id=${encodeURIComponent(store.userId)}`).then(r => r.json())
      ]);
      renderDbSnapshot("Database mastery", mastery, "complete graded tasks to populate mastery levels");
      renderDbSnapshot("Prompts", prompts, "no saved prompts for the current topic yet");
      renderDbSnapshot("Journey", journey, "start or complete a session to log journey events");
      renderDbSnapshot("Bloom", bloom, "submit Bloom-scored activities to populate this table");
      renderDbSnapshot("Chat operations", chatOps, "no tutor turns have been logged yet");
    }catch{
      showToast("Database load failed");
    }finally{
      setBusy(false);
    }
  };

  if(btnOpenAdmin) btnOpenAdmin.onclick = () => {
    const targetUser = store.userId ? encodeURIComponent(store.userId) : "";
    const url = targetUser ? `/static/admin.html?user_id=${targetUser}` : "/static/admin.html";
    window.open(url, "_blank", "noopener");
  };

  function asIntOrNull(v){
    if(!v || String(v).trim() === "") return null;
    const n = parseInt(v, 10);
    return Number.isNaN(n) ? null : n;
  }

  let timerId = null;
  function startTimer(){
    const t0 = Date.now();
    const tick = () => {
      const s = Math.floor((Date.now() - t0) / 1000);
      if(statusEl) statusEl.innerHTML = `<span class="loader"></span> thinking… ${s}s`;
    };
    tick();
    timerId = setInterval(tick, 500);
  }
  function stopTimer(){
    if(timerId){
      clearInterval(timerId);
      timerId = null;
    }
    if(statusEl) statusEl.textContent = "ready";
  }

  async function handleSend(){
    const txt = msgInput?.value.trim();
    if(!txt) return;
    if(!store.userId){
      showToast("Please sign in first.");
      return;
    }
    if(!store.currentSessionId){
      showToast("No active chat.");
      return;
    }
    const { session, theme } = store.findSession(store.currentSessionId);
    if(!session){
      showToast("No active chat.");
      return;
    }
    msgInput.value = "";
    persistMessage("user", txt);
    renderMessage("user", txt);
    const botBubble = renderMessage("bot", "…");
    persistMessage("bot", "…");

    const mode = store.mode;
    const topic = session.topic || theme.topic;
    const body = mode === "simple"
      ? { user_id: store.userId, text: txt, max_tokens: asIntOrNull(store.maxTokens) }
      : { user_id: store.userId, topic, text: txt, apply_mode: store.applyMode, max_tokens: asIntOrNull(store.maxTokens) };
    const url = mode === "simple" ? "/chat_simple" : "/chat";

    try{
      setBusy(true, "thinking…");
      startTimer();
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      });
      const data = await res.json();
      if(!res.ok) throw new Error(data.detail || "Server error");
      const finalText = stripThink(data.answer_text || "");
      if(botBubble) setBubbleContent(botBubble, "bot", finalText || "(no answer)");
      updateLastBotMessage(finalText);
    }catch(err){
      const errorText = `⚠️ ${err.message}`;
      if(botBubble) setBubbleContent(botBubble, "bot", errorText);
      updateLastBotMessage(errorText);
      showToast(err.message, 4000);
    }finally{
      setBusy(false);
      stopTimer();
    }
  }

  if(sendBtn) sendBtn.onclick = handleSend;
  if(msgInput) msgInput.addEventListener("keydown", e => {
    if(e.key === "Enter" && !e.shiftKey){
      e.preventDefault();
      handleSend();
    }
  });

  // ===== Init =====
  if(store.token && store.userId){
    store.initForUser(store.userId);
    syncControlsFromStore();
    renderActiveSession();
    renderTranscript();
    updateUserBadges();
    showApp();
    startTeacherOversightUpdates();
  }else{
    showAuth();
    resetTeacherOversight();
  }

  renderHistory();
});
