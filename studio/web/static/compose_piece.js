// compose_piece.js — one piece: canvas (final.jpg), layer panel, choice cards,
// pinned notes (→ agent), history, chat. Reuses canvas.js + chat.js.
import * as canvas from "./canvas.js";
import { init as initChat } from "./chat.js";

const BASE = window.COMPOSE.base || "";
const PIECE = window.COMPOSE.piece;
const API = `${BASE}/api/compose`;
const P_API = `${API}/piece/${encodeURIComponent(PIECE)}`;
const $ = (id) => document.getElementById(id);

let state = null;      // last /piece summary
let busy = 0;          // in-flight mutations from this page
let pendingPin = null; // marker awaiting text
let pinContext = null; // one-shot chat context for the next (pin) message
const openNotes = new Set();

let toastTimer = null;
function toast(msg, ms = 3500) {
  const t = $("toast");
  t.textContent = msg;
  t.classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.add("hidden"), ms);
}

async function api(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch (e) {}
    throw new Error(msg);
  }
  return res.json();
}
const post = (path, body) => api(`${P_API}${path}`, {
  method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body || {}),
});
const fileUrl = (rel, w) => `${P_API}/file/${rel}?v=${state ? state.final_v : 0}${w ? `&w=${w}` : ""}`;

function setBusy(on, text = "rendering…") {
  busy = Math.max(0, busy + (on ? 1 : -1));
  const show = busy > 0 || (state && state.busy);
  $("busy-chip").classList.toggle("hidden", !show);
  $("busy-text").textContent = text;
  $("layers-body").classList.toggle("panel-disabled", busy > 0);
}

// Run a mutation that returns a fresh summary (set/pick/reseed/render).
async function mutate(path, body, label) {
  setBusy(true, label);
  try {
    const res = await post(path, body);
    await apply(res.summary || res);
    return res;
  } catch (e) {
    toast(`Error: ${e.message}`, 6000);
  } finally {
    setBusy(false);
  }
}

// ---- state → UI ----
let lastFinal = -1;
let lastKey = "";
async function apply(s) {
  const key = JSON.stringify({ ...s, busy: null });
  const same = key === lastKey;
  lastKey = key;
  state = s;
  $("busy-chip").classList.toggle("hidden", !(busy > 0 || s.busy));
  if (same) return; // poll with no change: don't clobber half-typed comments
  if (s.final_v !== lastFinal) {
    lastFinal = s.final_v;
    try { await canvas.loadImage(fileUrl("final.jpg")); } catch (e) { /* no render yet */ }
  }
  renderLayers();
  renderQuestions();
  renderPins();
  renderHistory();
  $("psd-link").classList.toggle("hidden", !s.has_psd);
  $("psd-link").href = `${P_API}/file/piece.psd`;
  $("busy-chip").classList.toggle("hidden", !(busy > 0 || s.busy));
}

async function load() {
  try { await apply(await api(P_API)); } catch (e) { toast(`Error: ${e.message}`); }
}

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

function renderLayers() {
  const body = $("layers-body");
  body.innerHTML = "";
  // top of the stack first reads like Photoshop's layer panel
  for (const L of state.layers.slice().reverse()) {
    if (L.id === "subject") {
      body.appendChild(el("div", "layer-row subject-row", "— SUBJECT —"));
      continue;
    }
    const row = el("div", "layer-row" + (L.enabled ? "" : " off"));
    const cb = el("input");
    cb.type = "checkbox";
    cb.checked = L.enabled;
    cb.title = "on / off";
    cb.onchange = () => mutate("/set", { kv: [`${L.id}.enabled=${cb.checked}`],
                                         note: `${cb.checked ? "on" : "off"}: ${L.id}` },
                               cb.checked ? "adding…" : "removing…");
    const name = el("button", "layer-name", L.id);
    name.appendChild(el("span", "layer-kind", L.kind));
    if (L.riff) name.appendChild(el("span", "riff-badge", "riff"));
    name.onclick = () => {
      openNotes.has(L.id) ? openNotes.delete(L.id) : openNotes.add(L.id);
      renderLayers();
    };
    const z = el("button", "zbtn", L.z === "over" ? "over ↑" : "under ↓");
    z.title = "move over / under her";
    z.onclick = () => mutate("/set", { kv: [`${L.id}.z=${L.z === "over" ? "under" : "over"}`],
                                       note: `move ${L.id} ${L.z === "over" ? "under" : "over"}` });
    const seed = el("button", "seedbtn", "🎲");
    seed.title = "re-seed (new random layout)";
    seed.onclick = () => mutate("/reseed", { layer: L.id }, "re-rolling…");
    row.append(cb, name, z, seed);
    body.appendChild(row);
    if (openNotes.has(L.id)) {
      body.appendChild(el("div", "layer-note",
        (L.note || "(no note)") + (L.region ? `  · region: ${L.region}` : "")));
    }
  }
}

function renderQuestions() {
  const box = $("questions");
  box.innerHTML = "";
  for (const q of state.questions.filter((x) => x.status === "open")) {
    const card = el("div", "qcard");
    card.appendChild(el("div", "q-text", q.question));
    card.appendChild(el("div", "q-key", q.key));
    const opts = el("div", "q-opts");
    const comment = el("input", "text-in");
    comment.placeholder = "comment (optional)";
    for (const o of q.options) {
      const col = el("div", "q-opt");
      const img = el("img");
      img.loading = "lazy";
      img.src = `${P_API}/file/questions/${o.file}?w=600`;
      img.onclick = () => lightbox(`${P_API}/file/questions/${o.file}`, `${o.letter} — ${o.value}`);
      const btn = el("button", "btn btn-accent", o.letter);
      btn.onclick = () => mutate("/pick", { question: q.id, choice: o.letter,
                                            comment: comment.value.trim() }, `picking ${o.letter}…`);
      col.append(img, btn, el("div", "q-val", String(o.value)));
      opts.appendChild(col);
    }
    card.append(opts, comment);
    box.appendChild(card);
  }
}

function openPins() { return state.comments.filter((c) => c.status === "open"); }

function renderPins() {
  const pins = openPins();
  if (!pendingPin) {
    canvas.setMarkers(pins.map((c, i) => ({ n: i + 1, x: c.x, y: c.y, label: c.text.slice(0, 40) })));
  }
  $("pins-toggle").textContent = `Notes (${pins.length}) ${$("pins-body").classList.contains("hidden") ? "▸" : "▾"}`;
  const body = $("pins-body");
  body.innerHTML = "";
  const all = state.comments.slice().reverse();
  if (!all.length) body.appendChild(el("div", "empty-hint", "Tap “Pin a note”, then the canvas."));
  for (const c of all) {
    const row = el("div", "pin-row" + (c.status === "open" ? "" : " done"));
    const idx = pins.indexOf(c);
    row.appendChild(el("span", "pin-n", idx >= 0 ? String(idx + 1) : "✓"));
    const t = el("div", "pin-text", c.text);
    if (c.label) t.appendChild(el("div", "pin-label", `near ${c.label}`));
    row.appendChild(t);
    const b = el("button", "btn btn-small", c.status === "open" ? "done" : "reopen");
    b.onclick = async () => {
      try {
        await post(`/comments/${c.id}`, { status: c.status === "open" ? "done" : "open" });
        load();
      } catch (e) { toast(e.message); }
    };
    row.appendChild(b);
    body.appendChild(row);
  }
}

function describeDecision(d) {
  if (d.init) return ["init", `wildness ${d.init.wildness}`];
  if (d.question) return [d.question, `→ ${d.choice}${d.comment ? ` (“${d.comment}”)` : ""}`];
  if (d.set) return [d.note || "set", Object.entries(d.set).map(([k, v]) => `${k}=${v}`).join(", ")];
  return ["", JSON.stringify(d)];
}

function renderHistory() {
  const body = $("hist-body");
  body.innerHTML = "";
  const answered = new Map(state.questions.filter((q) => q.status === "answered")
                                         .map((q) => [q.question, q]));
  for (const d of state.decisions.slice().reverse()) {
    const row = el("div", "hist-row");
    const [head, tail] = describeDecision(d);
    const q = d.question && answered.get(d.question);
    if (q) {
      const o = q.options.find((x) => x.letter === q.choice);
      if (o) {
        const img = el("img");
        img.loading = "lazy";
        img.src = `${P_API}/file/questions/${o.file}?w=160`;
        img.onclick = () => lightbox(`${P_API}/file/questions/${q.id}_sheet.jpg`, q.question);
        row.appendChild(img);
      }
    }
    row.appendChild(el("span", "", `${(d.t || "").slice(5, 16)} `));
    row.appendChild(el("b", "", head));
    row.appendChild(el("span", "", ` ${tail}`));
    body.appendChild(row);
  }
}

function lightbox(src, cap) {
  $("lb-img").src = src;
  $("lb-cap").textContent = cap || "";
  $("lightbox").classList.remove("hidden");
}

// ---- pins ----
function onMarkerAdded(m) {
  if (pendingPin) return;
  pendingPin = m;
  $("pin-text").value = "";
  $("pin-sheet").classList.remove("hidden");
  setTimeout(() => $("pin-text").focus(), 50);
}

function closePin() {
  pendingPin = null;
  $("pin-sheet").classList.add("hidden");
  renderPins();
}

async function sendPin() {
  const text = $("pin-text").value.trim();
  if (!text || !pendingPin) return;
  const { x, y } = pendingPin;
  $("pin-send").disabled = true;
  setBusy(true, "pinning…");
  try {
    const c = await post("/comments", { x, y, text });
    closePin();
    await load();
    pinContext = { pin: c.id };
    const input = $("chat-input");
    input.value = c.note;
    $("chat-form").requestSubmit();
  } catch (e) {
    toast(`Error: ${e.message}`);
  } finally {
    $("pin-send").disabled = false;
    setBusy(false);
  }
}

// ---- boot ----
function wireCollapsible(btnId, bodyId, label) {
  $(btnId).onclick = () => {
    const b = $(bodyId);
    b.classList.toggle("hidden");
    if (label) $(btnId).textContent = `${label()} ${b.classList.contains("hidden") ? "▸" : "▾"}`;
  };
}

function boot() {
  canvas.init($("stage-container"));
  canvas.on("markerAdded", onMarkerAdded);
  // markers are server-owned: a local tap-to-delete just re-syncs from state
  canvas.on("markersChanged", () => setTimeout(() => { if (!pendingPin && state) renderPins(); }, 0));
  $("mode-toggle").addEventListener("click", (e) => {
    const b = e.target.closest("[data-mode]");
    if (!b) return;
    $("mode-toggle").querySelectorAll(".mode-btn").forEach((x) => x.classList.toggle("active", x === b));
    canvas.setMode(b.dataset.mode);
  });
  $("pin-cancel").onclick = closePin;
  $("pin-send").onclick = sendPin;
  $("lightbox").onclick = () => $("lightbox").classList.add("hidden");
  $("full-btn").onclick = () => mutate("/render", { full: true }, "full-res + PSD…");
  wireCollapsible("layers-toggle", "layers-body", () => "Layers");
  wireCollapsible("pins-toggle", "pins-body", () => `Notes (${state ? openPins().length : 0})`);
  wireCollapsible("hist-toggle", "hist-body", () => "History");

  initChat({
    messagesEl: $("chat-messages"), inputEl: $("chat-input"), formEl: $("chat-form"),
    apiBase: API, sessionId: encodeURIComponent(PIECE),
    getContext: () => { const c = pinContext || {}; pinContext = null; return c; },
    onGraphChanged: load,
  });

  load();
  // the agent (or another device) may change the piece — cheap poll while visible
  setInterval(() => { if (!document.hidden && busy === 0) load(); }, 6000);
}

boot();
