// session_tabs.js — horizontal session tab strip: poll sessions-status, highlight current,
// show running/ready badges, auto-refresh current session on running->ready transition.

let els = {};
let apiBase, base, sessionId;
let onCurrentReady = () => {};
let pollTimer = null;
let prevRunning = null; // running state of current session on the last poll
let sessions = [];

export function init({ containerEl, apiBase: api, base: b, sessionId: sid, onCurrentReady: cb }) {
  els.container = containerEl;
  apiBase = api;
  base = b;
  sessionId = sid;
  onCurrentReady = cb || onCurrentReady;

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) stopPolling();
    else startPolling();
  });

  refresh();
  startPolling();
}

function startPolling() {
  if (pollTimer) return;
  pollTimer = setInterval(refresh, 5000);
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

async function refresh() {
  let data;
  try {
    const res = await fetch(`${apiBase}/sessions-status`);
    if (!res.ok) return;
    data = await res.json();
  } catch (e) {
    return;
  }
  sessions = data.sessions || [];
  render();

  const current = sessions.find((s) => s.id === sessionId);
  if (current) {
    if (prevRunning === true && current.running === false) {
      // running -> not running: refresh session content
      await markSeen(sessionId);
      onCurrentReady();
    } else if (current.ready && prevRunning === null) {
      // arriving at a session that's already ready
      await markSeen(sessionId);
    }
    prevRunning = current.running;
  }
}

async function markSeen(id) {
  try {
    await fetch(`${apiBase}/session/${id}/seen`, { method: "POST" });
  } catch (e) { /* best-effort */ }
}

function render() {
  const container = els.container;
  container.innerHTML = "";

  for (const s of sessions) {
    const tab = document.createElement("a");
    tab.className = "session-tab" + (s.id === sessionId ? " active" : "");
    tab.href = `${base}/s/${s.id}`;

    const label = document.createElement("span");
    label.className = "session-tab-label";
    const name = s.label || s.source_name || s.id;
    label.textContent = truncate(name, 16);
    tab.appendChild(label);

    const steps = document.createElement("span");
    steps.className = "session-tab-steps";
    steps.textContent = String(s.steps != null ? s.steps : 0);
    tab.appendChild(steps);

    if (s.running) {
      const dot = document.createElement("span");
      dot.className = "session-tab-spinner";
      tab.appendChild(dot);
    }
    if (s.ready) {
      const badge = document.createElement("span");
      badge.className = "session-tab-ready";
      badge.textContent = "●";
      tab.appendChild(badge);
    }

    container.appendChild(tab);
  }

  const addTab = document.createElement("a");
  addTab.className = "session-tab session-tab-add";
  addTab.href = `${base}/`;
  addTab.textContent = "＋";
  container.appendChild(addTab);
}

function truncate(s, n) {
  if (!s) return "";
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}
