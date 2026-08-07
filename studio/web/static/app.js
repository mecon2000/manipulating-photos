import * as canvas from "./canvas.js";
import { renderSteps } from "./steps.js";
import { renderToolGrid, renderParamForm } from "./params.js";
import { init as initChat } from "./chat.js";

const BASE = window.STUDIO.base || "";
const SESSION_ID = window.STUDIO.sessionId;
const API = `${BASE}/api`;

const state = {
  tools: {},
  session: null,
  selectedNodeId: null, // node selected in steps strip (null = head/new-step mode)
  addingTool: null, // tool name chosen in "add step" flow, or null
  formHandle: null,
  mode: "pan",
};

const toastEl = document.getElementById("toast");
let toastTimer = null;
export function toast(msg, ms = 3000) {
  toastEl.textContent = msg;
  toastEl.classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toastEl.classList.add("hidden"), ms);
}

async function api(path, opts) {
  try {
    const res = await fetch(`${API}${path}`, opts);
    if (!res.ok) {
      let msg = res.statusText;
      try { const j = await res.json(); if (j.error) msg = j.error; } catch (e) {}
      toast(`Error: ${msg}`);
      throw new Error(msg);
    }
    const ct = res.headers.get("content-type") || "";
    if (ct.includes("application/json")) return await res.json();
    return null;
  } catch (err) {
    if (!/^Error:/.test(String(err.message))) toast(`Error: ${err.message}`);
    throw err;
  }
}

function objectUrl(ref) {
  if (!ref) return null;
  return `${API}/object/${ref}`;
}

// ---- DOM refs ----
const els = {
  undoBtn: document.getElementById("undo-btn"),
  redoBtn: document.getElementById("redo-btn"),
  costChip: document.getElementById("cost-chip"),
  stageContainer: document.getElementById("stage-container"),
  modeToggle: document.getElementById("mode-toggle"),
  brushControls: document.getElementById("brush-controls"),
  brushSize: document.getElementById("brush-size"),
  brushClear: document.getElementById("brush-clear"),
  brushUse: document.getElementById("brush-use"),
  maskChip: document.getElementById("mask-chip"),
  stepsToggle: document.getElementById("steps-toggle"),
  stepsBody: document.getElementById("steps-body"),
  stepsStrip: document.getElementById("steps-strip"),
  paramsToggle: document.getElementById("params-toggle"),
  paramsBody: document.getElementById("params-body"),
  paramsContent: document.getElementById("params-content"),
  chatMessages: document.getElementById("chat-messages"),
  chatInput: document.getElementById("chat-input"),
  chatForm: document.getElementById("chat-form"),
};

let brushMaskRef = null;

// ---- init ----
async function main() {
  canvas.init(els.stageContainer);
  canvas.on("markersChanged", debounce(syncMarkers, 500));

  wireTopbar();
  wireModeBar();
  wireCollapsibles();

  state.tools = await api("/tools");
  await refreshSession();
  await refreshCost();

  initChat({
    messagesEl: els.chatMessages,
    inputEl: els.chatInput,
    formEl: els.chatForm,
    apiBase: API,
    sessionId: SESSION_ID,
    getContext: () => ({ brush_mask: brushMaskRef, markers: canvas.getMarkers() }),
    onGraphChanged: async () => {
      await refreshSession();
      await refreshCost();
    },
  });

  renderParamsPanel();
}

function wireTopbar() {
  els.undoBtn.addEventListener("click", async () => {
    await api(`/session/${SESSION_ID}/undo`, { method: "POST" });
    await refreshSession();
  });
  els.redoBtn.addEventListener("click", async () => {
    await api(`/session/${SESSION_ID}/redo`, { method: "POST" });
    await refreshSession();
  });
}

function wireModeBar() {
  els.modeToggle.querySelectorAll(".mode-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const m = btn.dataset.mode;
      state.mode = m;
      canvas.setMode(m);
      els.modeToggle.querySelectorAll(".mode-btn").forEach((b) => b.classList.toggle("active", b === btn));
      els.brushControls.classList.toggle("hidden", m !== "brush");
    });
  });

  els.brushSize.addEventListener("input", () => {
    canvas.setBrushSizePct(parseFloat(els.brushSize.value));
  });
  els.brushClear.addEventListener("click", () => canvas.clearBrush());
  els.brushUse.addEventListener("click", async () => {
    const dataUrl = canvas.getBrushMaskDataURL();
    if (!dataUrl) return;
    const res = await api(`/session/${SESSION_ID}/brush-mask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ png: dataUrl }),
    });
    brushMaskRef = res.mask;
    els.maskChip.classList.remove("hidden");
    toast("Mask saved");
  });
}

function wireCollapsibles() {
  els.stepsToggle.addEventListener("click", () => {
    els.stepsBody.classList.toggle("hidden");
  });
  els.paramsToggle.addEventListener("click", () => {
    els.paramsBody.classList.toggle("hidden");
  });
}

// ---- session refresh ----
async function refreshSession() {
  const session = await api(`/session/${SESSION_ID}`);
  state.session = session;

  els.undoBtn.disabled = false;
  els.redoBtn.disabled = false;

  const canvasRef = state.selectedNodeId && session.outputs[state.selectedNodeId]
    ? session.outputs[state.selectedNodeId]
    : session.canvas_ref;
  if (canvasRef) {
    await canvas.loadImage(objectUrl(canvasRef));
  }

  canvas.setMarkers(session.ui && session.ui.markers ? session.ui.markers : []);

  els.stepsToggle.textContent = `Steps (${session.chain.length}) ▾`;
  renderSteps(els.stepsStrip, session.chain, session.graph.nodes, state.selectedNodeId, onSelectStep);

  renderParamsPanel();
}

async function refreshCost() {
  const c = await api("/costs/today");
  els.costChip.textContent = `$${Number(c.total || 0).toFixed(2)}`;
}

function onSelectStep(nodeId) {
  state.selectedNodeId = nodeId === state.selectedNodeId ? null : nodeId;
  state.addingTool = null;
  const session = state.session;
  renderSteps(els.stepsStrip, session.chain, session.graph.nodes, state.selectedNodeId, onSelectStep);
  const canvasRef = state.selectedNodeId && session.outputs[state.selectedNodeId]
    ? session.outputs[state.selectedNodeId]
    : session.canvas_ref;
  canvas.loadImage(objectUrl(canvasRef));
  renderParamsPanel();
}

// ---- params panel ----
function renderParamsPanel() {
  const container = els.paramsContent;
  container.innerHTML = "";

  const selectedNode = state.selectedNodeId ? state.session.graph.nodes[state.selectedNodeId] : null;

  if (selectedNode) {
    renderEditForm(container, selectedNode);
    return;
  }

  if (state.addingTool) {
    renderAddStepForm(container, state.addingTool);
    return;
  }

  const addBtn = document.createElement("button");
  addBtn.type = "button";
  addBtn.className = "btn btn-accent";
  addBtn.textContent = "＋ Add step";
  addBtn.addEventListener("click", () => {
    state.addingTool = "__choose__";
    renderParamsPanel();
  });
  container.appendChild(addBtn);

  if (state.addingTool === "__choose__") {
    const grid = renderToolGrid(state.tools, (name) => {
      state.addingTool = name;
      renderParamsPanel();
    }, null);
    container.appendChild(grid);
  }
}

function renderAddStepForm(container, toolName) {
  if (toolName === "__choose__") return;
  const schema = state.tools[toolName];
  if (!schema) return;

  const title = document.createElement("div");
  title.className = "param-label";
  title.innerHTML = `<span>${schema.label || toolName}</span>`;
  container.appendChild(title);

  const formArea = document.createElement("div");
  container.appendChild(formArea);
  const handle = renderParamForm(formArea, schema, {});

  const runRow = document.createElement("div");
  runRow.className = "run-row";
  const backBtn = document.createElement("button");
  backBtn.type = "button";
  backBtn.className = "btn";
  backBtn.textContent = "Back";
  backBtn.addEventListener("click", () => { state.addingTool = "__choose__"; renderParamsPanel(); });
  const runBtn = document.createElement("button");
  runBtn.type = "button";
  runBtn.className = "btn btn-accent";
  runBtn.textContent = "Run";
  runRow.appendChild(backBtn);
  runRow.appendChild(runBtn);
  container.appendChild(runRow);

  const progress = document.createElement("div");
  progress.className = "progress-line hidden";
  container.appendChild(progress);

  runBtn.addEventListener("click", async () => {
    const values = handle.getValues();
    const params = { ...values.params };
    if (values.preset) params.preset = values.preset;
    if (values.artifact) params.artifact = values.artifact;
    runBtn.disabled = true;
    progress.classList.remove("hidden");
    progress.textContent = "Adding step…";
    try {
      const { node } = await api(`/session/${SESSION_ID}/step`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tool: toolName, params, seed: values.seed, flags: values.flags }),
      });
      progress.textContent = "Running (this can take a while)…";
      await api(`/session/${SESSION_ID}/eval`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ node_id: node.id }),
      });
      state.addingTool = null;
      state.selectedNodeId = null;
      await refreshSession();
      await refreshCost();
      toast("Step complete");
    } catch (err) {
      progress.textContent = `Failed: ${err.message}`;
    } finally {
      runBtn.disabled = false;
    }
  });
}

function renderEditForm(container, node) {
  const schema = state.tools[node.tool];
  if (!schema) {
    container.textContent = `Unknown tool: ${node.tool}`;
    return;
  }

  const title = document.createElement("div");
  title.className = "param-label";
  title.innerHTML = `<span>${schema.label || node.tool}</span>`;
  container.appendChild(title);

  const formArea = document.createElement("div");
  container.appendChild(formArea);
  const handle = renderParamForm(formArea, schema, {
    params: node.params,
    flags: node.flags,
    seed: node.seed,
  });

  const runRow = document.createElement("div");
  runRow.className = "run-row";
  const applyBtn = document.createElement("button");
  applyBtn.type = "button";
  applyBtn.className = "btn btn-accent";
  applyBtn.textContent = "Apply changes";
  runRow.appendChild(applyBtn);
  container.appendChild(runRow);

  const progress = document.createElement("div");
  progress.className = "progress-line hidden";
  container.appendChild(progress);

  applyBtn.addEventListener("click", async () => {
    const values = handle.getValues();
    const params = { ...values.params };
    if (values.preset) params.preset = values.preset;
    if (values.artifact) params.artifact = values.artifact;
    applyBtn.disabled = true;
    progress.classList.remove("hidden");
    progress.textContent = "Applying…";
    try {
      const { head } = await api(`/session/${SESSION_ID}/edit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ node_id: node.id, params, seed: values.seed }),
      });
      progress.textContent = "Running (this can take a while)…";
      await api(`/session/${SESSION_ID}/eval`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ node_id: head }),
      });
      await refreshSession();
      await refreshCost();
      toast("Updated");
    } catch (err) {
      progress.textContent = `Failed: ${err.message}`;
    } finally {
      applyBtn.disabled = false;
    }
  });
}

// ---- markers sync ----
async function syncMarkers(markers) {
  try {
    await api(`/session/${SESSION_ID}/ui`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ markers }),
    });
  } catch (e) { /* toast already shown by api() */ }
}

function debounce(fn, ms) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

main();
