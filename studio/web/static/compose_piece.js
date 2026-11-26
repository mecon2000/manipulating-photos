// compose_piece.js — one piece, hands-on: the browser composites layers/manifest.json
// so toggles / reorders / variant previews are instant; edits persist with no-render
// calls and the server re-renders in the background (debounced) — a "syncing" dot
// shows until final.jpg/history catch up. Plus choice cards, pinned notes → agent,
// savepoints with wipe/side-by-side compare, chat. Reuses canvas.js + chat.js.
import * as canvas from "./canvas.js";
import { init as initChat } from "./chat.js";
import { composite, loadImg } from "./compose_layers.js";
import { renderPanel, isDragging } from "./compose_panel.js";
import { openPins, renderHistory, renderPinList, renderQuestions } from "./compose_side.js";

const BASE = window.COMPOSE.base || "";
const PIECE = window.COMPOSE.piece;
const API = `${BASE}/api/compose`;
const P_API = `${API}/piece/${encodeURIComponent(PIECE)}`;
const $ = (id) => document.getElementById(id);

let state = null;           // last full summary from the server
let man = null;             // working manifest (optimistic local edits applied)
const overrides = {};       // layer id -> previewed variant seed
const rolling = new Set();  // layer ids with a variant render in flight
const openNotes = new Set();
const usedSeeds = {};       // layer id -> seed chosen with "use", until the manifest catches up
const localVariants = {};   // layer id -> seeds rolled here (manifest only learns them on render)
let inflight = 0;           // local requests not yet acknowledged
let serverPending = false;
let compare = null;         // {sp, img, mode: "hold"|"side", showing}
let pendingPin = null;
let pinContext = null;

// ---- plumbing ----
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
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}),
});

// what the busy chip says while a change syncs: instant changes are only being saved
let syncLabel = "rendering…";
const isInstant = (path, body) => path === "/order" ||
  (path === "/quick-set" && (body.kv || []).every((kv) => /\.(enabled|removed|seed)=/.test(kv)));

function updateDots() {
  const syncing = inflight > 0 || serverPending;
  // a visible "rendering…" chip on the picture while a change is saving / rendering
  const chip = $("busy-chip"), txt = $("busy-text");
  if (syncing && chip.classList.contains("hidden")) {
    txt.textContent = syncLabel;
    chip.classList.remove("hidden");
    chip.dataset.sync = "1";
  } else if (!syncing && chip.dataset.sync) {
    delete chip.dataset.sync;
    if (txt.textContent === "rendering…" || txt.textContent === "saving…") chip.classList.add("hidden");
  }
  $("sync-dot").classList.toggle("hidden", !syncing);
  $("sync-dot").title = state && state.sync && state.sync.error ? `render error: ${state.sync.error}` : "saving — the picture you see is already up to date";
}

// Fire-and-forget persistence: the UI never waits on it.
async function persist(path, body, failMsg) {
  syncLabel = isInstant(path, body) ? "saving…" : "rendering…";
  inflight++;
  updateDots();
  try {
    const r = await post(path, body);
    if (r && "pending" in r) serverPending = r.pending;
    return r;
  } catch (e) {
    toast(`${failMsg || "Save failed"}: ${e.message}`, 6000);
    load(true);
  } finally {
    inflight--;
    updateDots();
    pollSoon();
  }
}

// Blocking mutation (full-render endpoints: pick, full render, restore).
async function mutate(path, body, label) {
  $("busy-chip").classList.remove("hidden");
  $("busy-text").textContent = label || "rendering…";
  try {
    const res = await post(path, body);
    await apply(res.summary || res, true);
    return res;
  } catch (e) {
    toast(`Error: ${e.message}`, 6000);
  } finally {
    $("busy-chip").classList.add("hidden");
  }
}

// ---- compositing ----
const fileUrl = (rel, v) => `${P_API}/file/${rel}${v != null ? `?v=${v}` : ""}`;

function layerSrc(L) {
  const ov = overrides[L.id];
  if (ov != null && ov !== L.seed) return fileUrl(`layers/variants/${L.id}/s${ov}.png`, ov);
  const f = String(L.file || "");
  return fileUrl(f.startsWith("layers/") ? f : `layers/${f}`, L.v || man.rev);
}

let drawing = false, redrawAgain = false, firstPainted = false, layersShown = false;
async function redraw() {
  if (!man) return;
  if (drawing) { redrawAgain = true; return; }
  drawing = true;
  try {
    const live = painting ? { id: painting, mask: canvas.getMaskCanvas() } : liveHold;
    const layers = man.layers.map((L) => {
      if (live && L.id === live.id && live.mask) {
        const raw = L.raw ? fileUrl(`layers/${L.raw}`, L.v || man.rev) : layerSrc(L);
        return { ...L, enabled: true, removed: false, url: raw, liveMask: live.mask };
      }
      return { ...L, url: layerSrc(L) };
    });
    // first paint: the finished render (one JPEG) while the ~20 layer PNGs download
    if (!firstPainted) {
      firstPainted = true;
      const fin = new Image();
      fin.onload = () => { if (!layersShown) canvas.setImageSource(fin); };
      fin.src = fileUrl("final.jpg", man.rev);
    }
    const chip = $("busy-chip"), txt = $("busy-text");
    const c = await composite(man.W, man.H, layers, compare, (n, total) => {
      if (n < total) { chip.classList.remove("hidden"); txt.textContent = `loading layers ${n}/${total}…`; }
      else if (txt.textContent.startsWith("loading layers")) chip.classList.add("hidden");
    });
    if (txt.textContent.startsWith("loading layers")) chip.classList.add("hidden");
    layersShown = true;
    if (revealGate) {             // rendered change: wait for the cue, then show + low beep
      const g = revealGate;
      await g;
      if (revealGate === g) revealGate = null;
      cueReveal();
    }
    canvas.setImageSource(c);
  } catch (e) {
    toast(`draw: ${e.message}`);
  } finally {
    drawing = false;
    if (redrawAgain) { redrawAgain = false; redraw(); }
  }
}

// ---- render cue ----
// A server render (any layer image changed) announces itself: two quick high beeps, then
// ~700ms later the new picture appears with a lower beep — time to look where the change
// should land. Instant changes (toggle, reorder, picking a variant) show at once, silently.
let audioCtx = null;
function audio() {
  try {
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    if (audioCtx.state === "suspended") audioCtx.resume();
  } catch (e) { /* no audio */ }
  return audioCtx;
}
document.addEventListener("pointerdown", audio, { passive: true });   // unlock (iOS needs a tap)
function beep(freq, dur, t0) {
  const c = audioCtx;
  if (!c) return;
  const o = c.createOscillator(), g = c.createGain();
  o.type = "sine";
  o.frequency.value = freq;
  g.gain.setValueAtTime(0.0001, t0);
  g.gain.exponentialRampToValueAtTime(0.3, t0 + 0.01);
  g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
  o.connect(g).connect(c.destination);
  o.start(t0);
  o.stop(t0 + dur + 0.03);
}
let revealGate = null;          // set while a rendered change waits to be shown
function cueRender() {
  const c = audio();
  if (c) { const t = c.currentTime; beep(1760, 0.07, t); beep(1760, 0.07, t + 0.12); }
  revealGate = new Promise((r) => setTimeout(r, 190 + 700));
}
function cueReveal() { const c = audio(); if (c) beep(880, 0.14, c.currentTime); }
const layerVersions = (m) => Object.fromEntries((m ? m.layers : []).map((L) => [L.id, L.v]));
function wasRendered(oldMan, newMan) {
  if (!oldMan || !newMan) return false;
  const a = layerVersions(oldMan);
  const changed = newMan.layers.filter((L) => L.v && L.v !== a[L.id]).map((L) => L.id);
  const loud = changed.filter((id) => !quietLayers.has(id));
  changed.forEach((id) => {          // the saved mask arrived: stop holding the live preview
    if (quietLayers.has(id)) { quietLayers.delete(id); if (liveHold && liveHold.id === id) liveHold = null; }
  });
  return loud.length > 0;
}

// ---- state ----
let lastKey = "";
async function apply(s, force = false) {
  if (s.manifest && wasRendered(man, s.manifest)) cueRender();
  state = s;
  serverPending = !!(s.sync && s.sync.pending);
  updateDots();
  const key = JSON.stringify({ ...s, busy: null, sync: null });
  if (key === lastKey && !force) return;
  lastKey = key;
  if (s.manifest) {
    man = JSON.parse(JSON.stringify(s.manifest));
    for (const L of man.layers) {
      if (overrides[L.id] === L.seed) delete overrides[L.id];
      const extra = (localVariants[L.id] || []).filter((x) => !(L.variants || []).includes(x));
      if (extra.length) L.variants = [...(L.variants || []), ...extra];
      if (usedSeeds[L.id] != null && usedSeeds[L.id] !== L.seed) L._pendingSeed = usedSeeds[L.id];
    }
    redraw();
  } else {
    man = null;
    try { await canvas.loadImage(fileUrl("final.jpg", s.final_v)); } catch (e) {}
  }
  renderAll();
}

async function load(force = false) {
  try { await apply(await api(P_API), force); } catch (e) { toast(`Error: ${e.message}`); }
}

// Poll the cheap /sync until the background render is done, then reload once.
let pollT = null;
let panelStale = false;
function pollSoon(ms = 1500) {
  clearTimeout(pollT);
  pollT = setTimeout(poll, ms);
}
async function poll() {
  try {
    const s = await api(`${P_API}/sync`);
    serverPending = s.pending;
    updateDots();
    if (!s.pending && inflight === 0) {
      if (man && s.rev !== man.rev) await load(true);
      else if (s.error) toast(`render error: ${s.error.slice(-200)}`, 8000);
      pollSoon(document.hidden ? 15000 : 3000);   // idle watch: agent edits / other devices
      return;
    }
  } catch (e) { /* transient */ }
  pollSoon();
}

// ---- panel handlers ----
// "Only model + paper": remembers which layers were on, so the button flips between the
// clean base and that set. Any manual toggle while clean starts a NEW set (forget the old).
let cleanSaved = null;
const contentLayers = () => (man ? man.layers.filter((L) => (L.z === "under" || L.z === "over") && !L.removed) : []);

function syncCleanBtn() {
  const b = $("clean-btn");
  if (!b) return;
  const bare = !cleanSaved && man && !contentLayers().some((L) => L.enabled);
  b.textContent = cleanSaved ? `↩ Previous layers (${cleanSaved.length})`
    : bare ? "All layers on" : "Only model + paper";
  b.classList.toggle("btn-accent", !!cleanSaved);
}

function onCleanClick() {
  if (!man) return;
  const kv = [];
  if (cleanSaved) {
    const want = new Set(cleanSaved);
    for (const L of contentLayers()) {
      const on = want.has(L.id);
      if (L.enabled !== on) { L.enabled = on; kv.push(`${L.id}.enabled=${on}`); }
    }
    cleanSaved = null;
  } else {
    cleanSaved = contentLayers().filter((L) => L.enabled).map((L) => L.id);
    if (!cleanSaved.length) {          // already bare and nothing to go back to: everything on
      cleanSaved = null;
      for (const L of contentLayers()) { L.enabled = true; kv.push(`${L.id}.enabled=true`); }
      syncCleanBtn(); redraw(); renderAll();
      if (kv.length) persist("/quick-set", { kv, note: "all layers on" });
      return;
    }
    for (const L of contentLayers()) {
      if (L.enabled) { L.enabled = false; kv.push(`${L.id}.enabled=false`); }
    }
  }
  syncCleanBtn();
  redraw();
  renderAll();
  if (kv.length) persist("/quick-set", { kv, note: cleanSaved ? "only model + paper" : "back to previous layers" });
}

// ---- paint a layer's mask on the canvas ----
// Painting is all in the browser: the mask shows red, the layer previews live through it
// (unmasked layer PNG × the mask), and only Apply goes to the server — silently, since
// the picture on screen is already the result.
let painting = null;            // layer id being painted
const quietLayers = new Set();  // layers whose next server version must not beep
async function startPaint(L) {
  if (painting) endPaint();
  const m = man && man.layers.find((x) => x.id === L.id);
  let init = null;
  const P = (m && m.params) || L.params || {};
  if (P.paint_mask) {
    try { init = await loadImgEl(fileUrl(P.paint_mask, P.paint_mask_v || 0)); } catch (e) { init = null; }
  }
  painting = L.id;
  canvas.setBrushSizePct(Number($("paint-size").value));
  canvas.startMaskEdit(init);          // first time: empty — paint where the layer should show
  $("paint-color").value = localStorage.getItem("maskColor") || "red";      // remembered per device
  $("paint-feather").value = localStorage.getItem("maskFeather") || "0.5";
  canvas.setMaskColor($("paint-color").value);
  canvas.setMaskFeather(Number($("paint-feather").value));
  $("paint-label").textContent = `“${L.id.replace(/_/g, " ")}”`;
  $("paint-bar").classList.remove("hidden");
  $("mode-toggle").classList.add("hidden");
  setTool("paint");
  if (!init) toast("Paint where this layer should show. Erase to take it back. Apply when done.", 3500);
  redraw();
}
function setTool(t) {
  canvas.setMaskTool(t);
  $("paint-tool-paint").classList.toggle("active", t === "paint");
  $("paint-tool-erase").classList.toggle("active", t === "erase");
}
function endPaint() {
  painting = null;
  canvas.endMaskEdit();
  $("paint-bar").classList.add("hidden");
  $("mode-toggle").classList.remove("hidden");
  setMode("pan");
  redraw();
}
async function applyPaint(remove = false) {
  const id = painting;
  if (!id) return;
  const png = remove ? "" : canvas.maskPNG();
  quietLayers.add(id);
  // keep the live result on screen until the saved version arrives
  liveHold = remove ? null : { id, mask: canvas.getMaskCanvas() };
  painting = null;
  canvas.endMaskEdit();
  $("paint-bar").classList.add("hidden");
  $("mode-toggle").classList.remove("hidden");
  setMode("pan");
  syncLabel = "saving…";
  inflight++; updateDots();
  try {
    const r = await post("/paint-mask", { layer: id, png });
    if (r && "pending" in r) serverPending = r.pending;
  } catch (e) { toast(`Mask failed: ${e.message}`, 6000); quietLayers.delete(id); liveHold = null; }
  finally { inflight--; updateDots(); pollSoon(500); redraw(); }
}
// ---- free-transform the torn strip ----
let shaping = null;
function startShape(L) {
  if (painting) endPaint();
  const m = man && man.layers.find((x) => x.id === L.id);
  const g = m && m.geom && m.geom.shape;
  if (!g) { toast("Render the piece once first — the tear's handles come from the last render.", 4000); return; }
  shaping = L.id;
  shapeLayer = { L, g, m };
  canvas.startShapeEdit(g);
  setShapeMode("distort");
  setShapeTab("strip");
  $("shape-tab-under").classList.toggle("hidden", !(m.geom && m.geom.under));
  $("shape-label").textContent = `“${L.id.replace(/_/g, " ")}”`;
  $("shape-bar").classList.remove("hidden");
  $("mode-toggle").classList.add("hidden");
}
let shapeLayer = null, shapeTab = "strip", stripEdits = null, underEdits = null;
async function setShapeTab(tab) {
  if (!shapeLayer) return;
  // remember what was changed on the tab we leave
  if (shapeTab === "strip" && tab !== "strip") stripEdits = canvas.getShape();
  if (shapeTab === "under" && tab !== "under") underEdits = canvas.getUnderTf();
  shapeTab = tab;
  $("shape-tab-strip").classList.toggle("active", tab === "strip");
  $("shape-tab-under").classList.toggle("active", tab === "under");
  $("strip-tools").classList.toggle("hidden", tab !== "strip");
  $("under-op-wrap").classList.toggle("hidden", tab !== "under");
  if (tab === "strip") {
    canvas.endUnderEdit();
    canvas.startShapeEdit(stripEdits || shapeLayer.g);
  } else {
    canvas.endShapeEdit();
    const P = shapeLayer.m.params || {};
    const tf = underEdits || ((String(P.under_tf || "").match(/-?\d+(\.\d+)?/g) || []).map(Number));
    try {
      const img = await loadImgEl(fileUrl(shapeLayer.m.geom.under, shapeLayer.m.v || Date.now()));
      canvas.startUnderEdit(img, tf.length === 4 ? tf : null);
      canvas.setUnderOpacity(Number($("under-op").value));
    } catch (e) { toast("Couldn't load the photo below — render once and try again.", 4000); }
  }
}
function setShapeMode(m) {
  canvas.setShapeMode(m);
  $("shape-distort").classList.toggle("active", m === "distort");
  $("shape-scale").classList.toggle("active", m === "scale");
}
function endShape() {
  shaping = null;
  canvas.endShapeEdit();
  canvas.endUnderEdit();
  shapeLayer = null; shapeTab = "strip"; stripEdits = null; underEdits = null;
  $("shape-bar").classList.add("hidden");
  $("mode-toggle").classList.remove("hidden");
  setMode("pan");
}
function applyShape(reset = false) {
  const id = shaping;
  if (!id) return;
  if (shapeTab === "strip") stripEdits = canvas.getShape(); else underEdits = canvas.getUnderTf();
  const kv = [];
  if (reset) {
    kv.push(shapeTab === "under" ? `${id}.under_tf=` : `${id}.shape=`);
  } else {
    if (stripEdits) kv.push(`${id}.shape=${stripEdits.map((p) => p.join(",")).join(";")}`);
    if (underEdits) kv.push(`${id}.under_tf=${underEdits.join(",")}`);
  }
  endShape();
  if (!kv.length) return;
  persist("/quick-set", { kv, note: reset ? `tear ${id}: back to automatic` : `tear ${id}: reshaped` });
}

let liveHold = null;            // {id, mask}: applied mask shown live until the server's PNG lands
function loadImgEl(url) {
  return new Promise((res, rej) => { const i = new Image(); i.crossOrigin = "anonymous";
    i.onload = () => res(i); i.onerror = rej; i.src = url; });
}

const panelCtx = () => ({
  onPaint(L) { startPaint(L); },
  onShape(L) { startShape(L); },
  layers: man ? man.layers : [], overrides, rolling, openNotes,
  onRemove(L) {
    L.removed = true;
    redraw();
    renderAll();
    persist("/quick-set", { kv: [`${L.id}.removed=true`], note: `removed: ${L.id}` });
  },
  onRestore(L) {
    L.removed = false;
    redraw();
    renderAll();
    persist("/quick-set", { kv: [`${L.id}.removed=false`], note: `restored: ${L.id}` });
  },
  onToggle(L, on) {
    if (cleanSaved) { cleanSaved = null; syncCleanBtn(); }   // building a new set from clean
    L.enabled = on;
    redraw();
    renderAll();
    persist("/quick-set", { kv: [`${L.id}.enabled=${on}`], note: `${on ? "on" : "off"}: ${L.id}` });
  },
  onReorder(ids) {
    const byId = new Map(man.layers.map((L) => [L.z === "subject" ? "subject" : L.id, L]));
    const content = ids.map((id) => byId.get(id)).filter(Boolean);
    let z = "under";
    for (const L of content) {
      if (L.id === "subject" || L.z === "subject") { z = "over"; continue; }
      L.z = z;
    }
    // removed layers aren't in the drag list — keep them (server keeps their slots too)
    man.layers = [...man.layers.filter((L) => L.z === "base"), ...content,
                  ...man.layers.filter((L) => L.removed && !content.includes(L)),
                  ...man.layers.filter((L) => L.z === "finish")];
    redraw();
    renderAll();
    persist("/order", { ids: content.map((L) => (L.z === "subject" ? "subject" : L.id)) },
            "Reorder failed");
  },
  async onRoll(L) {
    rolling.add(L.id);
    renderAll();
    try {
      const r = await post("/variant", { layer: L.id, count: 3 });   // 🎲 = 3 new versions
      const seeds = r.seeds || [r.seed];
      localVariants[L.id] = [...(localVariants[L.id] || []), ...seeds];
      L.variants = [...(L.variants || []).filter((s) => !seeds.includes(s)), ...seeds];
      overrides[L.id] = seeds[0];
      await loadImg(layerSrc(L));
      redraw();
    } catch (e) {
      toast(`Roll failed: ${e.message}`, 6000);
    } finally {
      rolling.delete(L.id);
      renderAll();
    }
  },
  onChip(L, seed) {
    // picking a chip = using it (no separate "use" step)
    const active = usedSeeds[L.id] ?? L._pendingSeed ?? L.seed;
    if (seed === active) return;
    this.onUse(L, seed);
    redraw();
  },
  onUse(L, seed) {
    // keep showing the variant PNG (override) until the background render
    // refreshes layers/ and the manifest's seed catches up
    overrides[L.id] = seed;
    L._pendingSeed = seed;
    usedSeeds[L.id] = seed;
    renderAll();
    // the original version may have no seed (null): send it empty, never the text "null"
    persist("/quick-set", { kv: [`${L.id}.seed=${seed == null ? "" : seed}`], note: `use variant seed ${seed ?? "original"} for ${L.id}` });
  },
  onDragEnd() { if (panelStale) renderAll(); },
  onNote(id) {
    openNotes.has(id) ? openNotes.delete(id) : openNotes.add(id);
    renderAll();
  },
  onRefresh() { renderAll(); },
  onParam(L, key, value) {
    persist("/quick-set", { kv: [`${L.id}.${key}=${value}`], note: `${L.id}: ${key} ${value}` });
  },
});

// ---- render the side UI ----
function renderAll() {
  if (!state) return;
  if (man) {
    if (isDragging()) panelStale = true;          // rebuild after the drop, not mid-drag
    else {
      const lb = $("layers-body"), sc = lb.closest(".collapsible-body") || lb;
      const top = sc.scrollTop;                   // keep the scroll position across rebuilds
      renderPanel(lb, panelCtx());
      sc.scrollTop = top;
      syncCleanBtn();
      panelStale = false;
    }
  } else $("layers-body").innerHTML = '<div class="empty-hint">Building layers… (first render after the update)</div>';
  const lb = (src, cap) => lightbox(src, cap);
  renderQuestions($("questions"), state, {
    api: P_API, lightbox: lb,
    onPick: (q, letter, comment) => mutate("/pick", { question: q.id, choice: letter, comment }, `picking ${letter}…`),
  });
  renderPalettes();
  renderPins();
  renderHistory($("hist-body"), state, { api: P_API, lightbox: lb });
  renderSavepoints();
  const psd = state.psd;
  $("psd-link").classList.toggle("hidden", !psd);
  if (psd) {
    $("psd-link").href = `${P_API}/file/${encodeURIComponent(psd.rel)}`;
    $("psd-link").textContent = `PSD ${psd.mb}MB`;
    $("psd-link").title = psd.path;
  }
}

// Palette picker: a recipe can ship several named colour schemes (recipe.palettes);
// base.palette_name picks one. Hidden when the recipe has only one.
function renderPalettes() {
  const row = $("palette-row");
  const pals = state.palettes || {};
  const names = Object.keys(pals);
  row.classList.toggle("hidden", names.length < 2);
  if (names.length < 2) return;
  row.innerHTML = '<span class="pal-label">Palette</span>';
  for (const n of names) {
    const b = document.createElement("button");
    b.className = "pal-chip" + (n === state.palette_name ? " active" : "");
    b.title = n;
    b.innerHTML = pals[n].map((c) => `<i style="background:${c}"></i>`).join("") + `<span>${n}</span>`;
    b.onclick = () => {
      if (n === state.palette_name) return;
      mutate("/set", { kv: [`base.palette_name=${n}`], note: `palette → ${n}` }, `palette ${n}…`);
    };
    row.appendChild(b);
  }
}

function renderPins() {
  const pins = openPins(state);
  if (!pendingPin && !compare) {
    canvas.setMarkers(pins.map((c, i) => ({ n: i + 1, x: c.x, y: c.y, label: c.text.slice(0, 40) })));
  }
  $("pins-toggle").textContent = `Notes (${pins.length}) ${$("pins-body").classList.contains("hidden") ? "▸" : "▾"}`;
  renderPinList($("pins-body"), state, {
    onStatus: async (c, status) => {
      try { await post(`/comments/${c.id}`, { status }); load(); } catch (e) { toast(e.message); }
    },
  });
}

// Lightbox. lightbox(src, cap) for one image, or lightbox(items, i) with items =
// [{src, cap, pick?}] to flip between them: ‹ › buttons, tap the left/right third,
// arrow keys. No animation — an instant swap makes the differences jump out.
let lbItems = [], lbIdx = 0;
function lightbox(src, cap) {
  if (Array.isArray(src)) { lbItems = src; lbIdx = cap || 0; }
  else { lbItems = [{ src, cap }]; lbIdx = 0; }
  lbItems.forEach((it) => { const im = new Image(); im.src = it.src; });   // preload: instant flips
  lbShow();
  $("lightbox").classList.remove("hidden");
}
function lbShow() {
  const it = lbItems[lbIdx];
  if (!it) return;
  $("lb-img").src = it.src;
  const n = lbItems.length;
  $("lb-cap").textContent = (n > 1 ? `${lbIdx + 1}/${n}  ` : "") + (it.cap || "");
  $("lb-prev").classList.toggle("hidden", n < 2);
  $("lb-next").classList.toggle("hidden", n < 2);
  const pk = $("lb-pick");
  pk.classList.toggle("hidden", !it.pick);
  if (it.pick) { pk.textContent = it.pickLabel || "Pick"; pk.onclick = (e) => { e.stopPropagation(); lbClose(); it.pick(); }; }
}
function lbStep(d) {
  if (lbItems.length < 2) return;
  lbIdx = (lbIdx + d + lbItems.length) % lbItems.length;
  lbShow();
}
function lbClose() { $("lightbox").classList.add("hidden"); }
function wireLightbox() {
  const lb = $("lightbox");
  lb.onclick = (e) => {
    if (lbItems.length > 1 && e.target.closest("#lb-img, .lb-stage")) {
      const r = lb.getBoundingClientRect(), x = (e.clientX - r.left) / r.width;
      if (x < 0.33) return lbStep(-1);
      if (x > 0.67) return lbStep(1);
    }
    lbClose();
  };
  $("lb-prev").onclick = (e) => { e.stopPropagation(); lbStep(-1); };
  $("lb-next").onclick = (e) => { e.stopPropagation(); lbStep(1); };
  document.addEventListener("keydown", (e) => {
    if (lb.classList.contains("hidden")) return;
    if (e.key === "ArrowLeft") { e.preventDefault(); lbStep(-1); }
    else if (e.key === "ArrowRight") { e.preventDefault(); lbStep(1); }
    else if (e.key === "Escape") lbClose();
  });
}

// ---- savepoints + compare ----
function renderSavepoints() {
  const sel = $("sp-select");
  const cur = sel.value;
  sel.innerHTML = "";
  const sps = state.savepoints || [];
  sel.appendChild(new Option(sps.length ? `Savepoints (${sps.length})` : "No savepoints", ""));
  const start = sps.filter((s) => s.kind === "start" || s.n === 0);
  const rest = sps.filter((s) => !start.includes(s)).reverse();
  for (const s of [...start, ...rest]) {
    const mark = s.kind === "fav" ? "★ " : s.kind === "start" ? "⏮ " : "";
    sel.appendChild(new Option(`${mark}${s.n} · ${s.name}`, String(s.n)));
  }
  $("vs-start-btn").classList.toggle("hidden", !start.length);
  sel.value = compare ? String(compare.sp.n) : (cur && sps.some((s) => String(s.n) === cur) ? cur : "");
}

async function openCompare(n) {
  const sp = (state.savepoints || []).find((s) => String(s.n) === String(n));
  if (!sp) return closeCompare();
  try {
    const img = await loadImg(fileUrl(`${sp.rel}/final.jpg`, sp.t));
    compare = { sp, img, mode: compare ? compare.mode : "hold", showing: false };
  } catch (e) { toast(`no image for savepoint ${sp.n}`); return; }
  setMode("pan");
  canvas.setMarkers([]);
  $("sp-select").value = String(sp.n);
  $("compare-bar").classList.remove("hidden");
  $("cmp-label").textContent = `savepoint: ${sp.name}`;
  $("cmp-hold").textContent = `Hold to see “${sp.name}”`;
  syncCompareUi();
  redraw();
}

function closeCompare() {
  compare = null;
  $("compare-bar").classList.add("hidden");
  $("sp-select").value = "";
  redraw();
  renderPins();
}

function syncCompareUi() {
  $("cmp-mode").textContent = compare.mode === "hold" ? "Side by side" : "Hold to flip";
  $("cmp-hold").classList.toggle("hidden", compare.mode !== "hold");
}

// ---- fav (full-res + PSD + favorites, ~20s in the background) ----
async function startFav() {
  const btn = $("fav-btn");
  btn.disabled = true;
  try {
    const f = await post("/fav", {});
    if (f.state === "running") toast("Faving — full-res + PSD, ~20s (you'll get a ping)", 5000);
    pollFav();
  } catch (e) {
    toast(`Fav failed: ${e.message}`, 6000);
    btn.disabled = false;
  }
}

async function pollFav() {
  try {
    const s = await api(`${P_API}/sync`);
    const f = s.fav;
    if (f && f.state === "running") { setTimeout(pollFav, 2000); return; }
    $("fav-btn").disabled = false;
    if (f && f.state === "done") {
      toast(`★ Faved. PSD: ${f.psd}${f.psd_mb ? ` (${f.psd_mb}MB)` : ""}`, 12000);
      load(true);
    } else if (f && f.state === "failed") {
      toast(`Fav failed: ${f.error.slice(-200)}`, 10000);
    }
  } catch (e) { setTimeout(pollFav, 4000); }
}

// ---- pins ----
function pinMarkers() {
  return openPins(state).map((c, i) => ({ n: i + 1, x: c.x, y: c.y, label: c.text.slice(0, 40) }));
}
function onMarkerAdded(m) {
  if (compare) return;
  const saved = pinMarkers();
  if (pendingPin) {
    // tapping again: move the pending pin (or, once something is typed, keep it where it is)
    if (!$("pin-text").value.trim()) pendingPin = { ...m, n: saved.length + 1 };
    canvas.setMarkers([...saved, pendingPin]);
    return;
  }
  pendingPin = { ...m, n: saved.length + 1 };
  canvas.setMarkers([...saved, pendingPin]);
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
  try {
    const c = await post("/comments", { x, y, text });
    closePin();
    await load();
    pinContext = { pin: c.id };
    $("chat-input").value = c.note;
    $("chat-form").requestSubmit();
  } catch (e) {
    toast(`Error: ${e.message}`);
  } finally {
    $("pin-send").disabled = false;
  }
}

// ---- boot ----
function setMode(m) {
  $("mode-toggle").querySelectorAll(".mode-btn").forEach((x) => x.classList.toggle("active", x.dataset.mode === m));
  canvas.setMode(m);
}

function wireCollapsible(btnId, bodyId, label) {
  $(btnId).onclick = () => {
    const b = $(bodyId);
    b.classList.toggle("hidden");
    // collapsed sections dock at the bottom of the panel (VS Code style) so Layers gets the room
    b.parentElement.classList.toggle("docked", b.classList.contains("hidden"));
    $(btnId).textContent = `${label()} ${b.classList.contains("hidden") ? "▸" : "▾"}`;
  };
}

// Phone: a draggable divider between the photo and the panel. Height is remembered
// per device; double-tap resets. Desktop keeps its side-by-side layout (hidden there).
function wireSplit() {
  const wrap = $("canvas-wrap");
  const bar = document.createElement("div");
  bar.className = "split-handle";
  bar.title = "drag to resize the photo (double-tap to reset)";
  bar.appendChild(document.createElement("span"));
  document.querySelector(".canvas-col").after(bar);
  const KEY = "compose.photoH";
  const clamp = (h) => Math.max(window.innerHeight * 0.2, Math.min(window.innerHeight * 0.85, h));
  const apply = (h) => {
    if (h) wrap.style.height = `${Math.round(clamp(h))}px`;
    else wrap.style.removeProperty("height");
    window.dispatchEvent(new Event("resize"));      // canvas refits to the new box
  };
  try { const v = +localStorage.getItem(KEY); if (v) apply(v); } catch (e) { /* no storage */ }
  bar.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    bar.setPointerCapture(e.pointerId);
    bar.classList.add("active");
    const y0 = e.clientY, h0 = wrap.getBoundingClientRect().height;
    let raf = 0, y = y0;
    const move = (ev) => {
      y = ev.clientY;
      if (!raf) raf = requestAnimationFrame(() => { raf = 0; apply(h0 + (y - y0)); });
    };
    const up = () => {
      bar.removeEventListener("pointermove", move);
      bar.removeEventListener("pointerup", up);
      bar.removeEventListener("pointercancel", up);
      bar.classList.remove("active");
      try { localStorage.setItem(KEY, String(Math.round(wrap.getBoundingClientRect().height))); } catch (e) { /* */ }
    };
    bar.addEventListener("pointermove", move);
    bar.addEventListener("pointerup", up);
    bar.addEventListener("pointercancel", up);
  });
  bar.addEventListener("dblclick", () => {
    try { localStorage.removeItem(KEY); } catch (e) { /* */ }
    apply(0);
  });
}

function boot() {
  canvas.init($("stage-container"));
  $("paint-size").oninput = (e) => canvas.setBrushSizePct(Number(e.target.value));
  $("paint-tool-paint").onclick = () => setTool("paint");
  $("paint-tool-erase").onclick = () => setTool("erase");
  $("paint-clear").onclick = () => canvas.clearMask();
  $("paint-all").onclick = () => canvas.fillMask();
  $("paint-tint").onchange = (e) => canvas.setMaskTint(e.target.checked);
  $("paint-color").onchange = (e) => { canvas.setMaskColor(e.target.value); localStorage.setItem("maskColor", e.target.value); };
  $("paint-feather").oninput = (e) => { canvas.setMaskFeather(Number(e.target.value)); localStorage.setItem("maskFeather", e.target.value); };
  $("shape-distort").onclick = () => setShapeMode("distort");
  $("shape-scale").onclick = () => setShapeMode("scale");
  $("shape-cancel").onclick = endShape;
  $("shape-tab-strip").onclick = () => setShapeTab("strip");
  $("shape-tab-under").onclick = () => setShapeTab("under");
  $("under-op").oninput = (e) => canvas.setUnderOpacity(Number(e.target.value));
  $("shape-apply").onclick = () => applyShape(false);
  $("shape-reset").onclick = () => applyShape(true);
  $("paint-cancel").onclick = endPaint;
  $("paint-apply").onclick = () => applyPaint(false);
  $("paint-remove").onclick = () => applyPaint(true);
  canvas.on("maskChanged", () => { if (painting) redraw(); });
  wireSplit();
  canvas.on("markerAdded", onMarkerAdded);
  canvas.on("markersChanged", () => setTimeout(() => { if (!pendingPin && state && !compare) renderPins(); }, 0));
  $("mode-toggle").addEventListener("click", (e) => {
    const b = e.target.closest("[data-mode]");
    if (!b) return;
    if (compare && b.dataset.mode === "markers") { toast("Close compare to pin notes"); return; }
    setMode(b.dataset.mode);
  });
  $("pin-cancel").onclick = closePin;
  $("pin-send").onclick = sendPin;
  wireLightbox();
  $("fav-btn").onclick = startFav;
  $("vs-start-btn").onclick = () => {
    const st = (state.savepoints || []).find((s) => s.kind === "start" || s.n === 0);
    if (!st) return;
    if (compare && compare.sp.n === st.n) closeCompare(); else openCompare(st.n);
  };
  $("save-btn").onclick = async () => {
    try {
      const r = await post("/save", {});      // numbered automatically (#1, #2 …)
      state.savepoints = r.savepoints;
      renderSavepoints();
      const last = r.savepoints[r.savepoints.length - 1];
      toast(`Saved ${last ? last.name : ""}`);
    } catch (e) { toast(`Save failed: ${e.message}`, 6000); }
  };
  $("sp-select").onchange = (e) => (e.target.value ? openCompare(e.target.value) : closeCompare());
  $("cmp-close").onclick = closeCompare;
  $("cmp-mode").onclick = () => { compare.mode = compare.mode === "hold" ? "side" : "hold"; compare.showing = false; syncCompareUi(); redraw(); };
  // press-and-hold flips to the savepoint; release = back to now (button or Space)
  const hold = (on) => {
    if (!compare || compare.mode !== "hold" || compare.showing === on) return;
    compare.showing = on;
    $("cmp-hold").classList.toggle("active", on);
    redraw();
  };
  const hb = $("cmp-hold");
  hb.addEventListener("pointerdown", (e) => { e.preventDefault(); hb.setPointerCapture?.(e.pointerId); hold(true); });
  ["pointerup", "pointercancel", "lostpointercapture"].forEach((ev) => hb.addEventListener(ev, () => hold(false)));
  hb.addEventListener("contextmenu", (e) => e.preventDefault());
  window.addEventListener("keydown", (e) => {
    if (e.code === "Space" && compare && !/INPUT|TEXTAREA/.test(document.activeElement?.tagName || "")) {
      e.preventDefault(); hold(true);
    }
  });
  window.addEventListener("keyup", (e) => { if (e.code === "Space") hold(false); });
  $("cmp-restore").onclick = async () => {
    if (!compare || !confirm(`Restore “${compare.sp.name}”? (current state stays in history)`)) return;
    const n = compare.sp.n;
    closeCompare();
    await mutate("/restore", { n }, "restoring…");
  };
  wireCollapsible("layers-toggle", "layers-body", () => "Layers");
  $("clean-btn").onclick = onCleanClick;
  syncCleanBtn();
  wireCollapsible("pins-toggle", "pins-body", () => `Notes (${state ? openPins(state).length : 0})`);
  wireCollapsible("hist-toggle", "hist-body", () => "History");

  initChat({
    messagesEl: $("chat-messages"), inputEl: $("chat-input"), formEl: $("chat-form"),
    apiBase: API, sessionId: encodeURIComponent(PIECE),
    getContext: () => { const c = pinContext || {}; pinContext = null; return c; },
    onGraphChanged: () => load(true),
  });

  load(true).then(() => pollSoon(serverPending ? 1500 : 3000));
  // the agent (or another device) may change the piece — cheap poll while idle
  setInterval(() => {
    if (!document.hidden && inflight === 0 && !serverPending && rolling.size === 0 && !isDragging()) load();
  }, 6000);
}

boot();
