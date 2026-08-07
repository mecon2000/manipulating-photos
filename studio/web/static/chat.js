// chat.js — chat message list + NDJSON streaming.

let messagesEl, inputEl, formEl;
let onGraphChanged = () => {};
let sending = false;

export function init({ messagesEl: m, inputEl: i, formEl: f, apiBase, sessionId, api, getContext, onGraphChanged: cb }) {
  messagesEl = m;
  inputEl = i;
  formEl = f;
  onGraphChanged = cb || onGraphChanged;

  formEl.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (sending) return;
    const text = inputEl.value.trim();
    if (!text) return;
    inputEl.value = "";
    addMessage("user", text);
    await send(text, apiBase, sessionId, getContext);
  });

  // restore chat history so a phone refresh loses nothing
  fetch(`${apiBase}/session/${sessionId}/chat-history`)
    .then((r) => (r.ok ? r.json() : { messages: [] }))
    .then((h) => {
      for (const m of h.messages || []) {
        addMessage(m.role === "user" ? "user" : "agent", m.text);
      }
    })
    .catch(() => {});
}

function addMessage(kind, text) {
  const div = document.createElement("div");
  div.className = kind === "user" ? "msg msg-user" : kind === "system" ? "msg msg-system" : "msg msg-agent";
  div.textContent = text;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return div;
}

function addToolChip(name, detail) {
  const div = document.createElement("div");
  div.className = "msg msg-tool";
  div.textContent = `⚙ ${name}: ${detail || ""}`;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return div;
}

function setSending(v) {
  sending = v;
  inputEl.disabled = v;
  const btn = formEl.querySelector("button");
  if (btn) btn.disabled = v;
}

async function send(text, apiBase, sessionId, getContext) {
  setSending(true);
  let agentDiv = null;
  let agentText = "";
  const toolChips = {};

  try {
    const res = await fetch(`${apiBase}/session/${sessionId}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, context: getContext ? getContext() : {} }),
    });
    if (!res.ok || !res.body) {
      addMessage("system", `Error: ${res.statusText}`);
      setSending(false);
      return;
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, idx).trim();
        buf = buf.slice(idx + 1);
        if (!line) continue;
        handleEvent(JSON.parse(line));
      }
    }
    if (buf.trim()) handleEvent(JSON.parse(buf.trim()));
  } catch (err) {
    addMessage("system", `Error: ${err.message}`);
  } finally {
    setSending(false);
  }

  function handleEvent(ev) {
    if (ev.type === "text") {
      if (!agentDiv) agentDiv = addMessage("agent", "");
      agentText += ev.delta;
      agentDiv.textContent = agentText;
      messagesEl.scrollTop = messagesEl.scrollHeight;
    } else if (ev.type === "tool_start") {
      toolChips[ev.name + ":" + ev.detail] = addToolChip(ev.name, ev.detail);
    } else if (ev.type === "tool_end") {
      const key = ev.name + ":" + ev.detail;
      const existing = Object.entries(toolChips).find(([k]) => k.startsWith(ev.name + ":"));
      const chip = existing ? existing[1] : addToolChip(ev.name, "");
      chip.textContent = `⚙ ${ev.name}: ${ev.detail || "done"}`;
    } else if (ev.type === "graph_changed") {
      onGraphChanged();
    } else if (ev.type === "error") {
      addMessage("system", ev.message);
    } else if (ev.type === "done") {
      // no-op
    }
  }
}
