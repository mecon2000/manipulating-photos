// params.js — render a param form from a tool schema, read back values.
// Also handles: hidden-params drawer + search, usage logging (appear/touch),
// and agent-proposed param highlighting.

export function renderToolGrid(tools, onPick, activeTool) {
  const wrap = document.createElement("div");
  wrap.className = "tool-grid";
  Object.keys(tools).forEach((name) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "tool-grid-btn" + (name === activeTool ? " active" : "");
    btn.textContent = tools[name].label || name;
    btn.addEventListener("click", () => onPick(name));
    wrap.appendChild(btn);
  });
  return wrap;
}

function logUsage(apiBase, tool, param, kind) {
  if (!apiBase || !tool || !param) return;
  fetch(`${apiBase}/params-usage`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tool, param, kind }),
  }).catch(() => {});
}

function debounce(fn, ms) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

// Renders the param form into `container`. Returns a `getValues()` accessor.
// opts: { apiBase, toolName, proposedParams: [{tool,param,note,ts}], pinned: Set<string>, onProposedDismiss(key) }
export function renderParamForm(container, schema, initial, opts) {
  container.innerHTML = "";
  initial = initial || {};
  opts = opts || {};
  const apiBase = opts.apiBase;
  const toolName = opts.toolName;
  const pinned = opts.pinned || new Set();
  let proposed = (opts.proposedParams || []).slice();

  const state = {
    params: {},
    flags: new Set(initial.flags || []),
    seed: initial.seed ?? null,
    preset: initial.preset ?? null,
    artifact: initial.artifact ?? null,
  };

  const touchLoggers = {}; // key -> debounced fn

  function logTouch(key) {
    if (!touchLoggers[key]) touchLoggers[key] = debounce(() => logUsage(apiBase, toolName, key, "touch"), 400);
    touchLoggers[key]();
  }

  const params = schema.params || {};
  const keys = Object.keys(params);
  const proposedByKey = new Map(proposed.map((p) => [p.param, p]));

  function dismissProposed(key) {
    proposed = proposed.filter((p) => p.param !== key);
    proposedByKey.delete(key);
    opts.onProposedDismiss && opts.onProposedDismiss(key);
  }

  function makeOnChange(key) {
    return (v) => {
      state.params[key] = v;
      logTouch(key);
      if (proposedByKey.has(key)) dismissProposed(key);
    };
  }

  // ---- ✨ proposed params (always shown, at top, real schema params) ----
  const proposedSection = document.createElement("div");
  proposedSection.className = "proposed-section";
  container.appendChild(proposedSection);

  // ---- normal + hidden params ----
  const formList = document.createElement("div");
  formList.className = "param-list";
  container.appendChild(formList);

  const hiddenEntries = [];
  const searchableRows = []; // {row, text}

  keys.forEach((key) => {
    const def = params[key];
    const val = initial.params && initial.params[key] !== undefined ? initial.params[key] : def.default;
    state.params[key] = val;

    const isProposed = proposedByKey.has(key);
    if (isProposed) {
      const row = renderParamRow(key, def, val, makeOnChange(key), {
        badge: "✨ new",
        note: proposedByKey.get(key).note,
        onDismiss: () => { dismissProposed(key); renderParamForm(container, schema, buildCurrentInitial(), opts); },
      });
      proposedSection.appendChild(row);
      searchableRows.push({ row, text: searchText(key, def) });
      logUsage(apiBase, toolName, key, "appear");
      return;
    }

    const isHidden = !!def.hidden && !pinned.has(key);
    if (isHidden) {
      hiddenEntries.push({ key, def, val });
      return;
    }

    const row = renderParamRow(key, def, val, makeOnChange(key), {
      pin: !!def.hidden, // was learned-hidden but currently force-shown (pinned) — allow un-pin
      pinned: pinned.has(key),
      onPin: () => { pinned.delete(key); renderParamForm(container, schema, buildCurrentInitial(), opts); },
    });
    formList.appendChild(row);
    searchableRows.push({ row, text: searchText(key, def) });
    logUsage(apiBase, toolName, key, "appear");
  });

  function buildCurrentInitial() {
    return { params: { ...state.params }, flags: Array.from(state.flags), seed: state.seed, preset: state.preset, artifact: state.artifact };
  }

  // ---- hidden-params drawer ----
  let drawerOpen = false;
  let hiddenAppearLogged = false;
  if (hiddenEntries.length) {
    const drawerToggle = document.createElement("button");
    drawerToggle.type = "button";
    drawerToggle.className = "param-drawer-toggle";
    drawerToggle.textContent = `more params… (${hiddenEntries.length} hidden)`;
    container.appendChild(drawerToggle);

    const drawerBody = document.createElement("div");
    drawerBody.className = "param-drawer hidden";
    container.appendChild(drawerBody);

    const searchRow = document.createElement("input");
    searchRow.type = "text";
    searchRow.className = "param-search";
    searchRow.placeholder = "Search params…";
    drawerBody.appendChild(searchRow);

    const hiddenList = document.createElement("div");
    drawerBody.appendChild(hiddenList);

    hiddenEntries.forEach(({ key, def, val }) => {
      const row = renderParamRow(key, def, val, makeOnChange(key), {
        pin: true,
        pinned: false,
        onPin: () => { pinned.add(key); renderParamForm(container, schema, buildCurrentInitial(), opts); },
      });
      hiddenList.appendChild(row);
      searchableRows.push({ row, text: searchText(key, def) });
    });

    drawerToggle.addEventListener("click", () => {
      drawerOpen = !drawerOpen;
      drawerBody.classList.toggle("hidden", !drawerOpen);
      if (drawerOpen && !hiddenAppearLogged) {
        hiddenAppearLogged = true;
        hiddenEntries.forEach(({ key }) => logUsage(apiBase, toolName, key, "appear"));
      }
    });

    searchRow.addEventListener("input", () => {
      const q = searchRow.value.trim().toLowerCase();
      filterRows(searchableRows, q);
      // auto-open drawer if a hidden row matches
      if (q) { drawerOpen = true; drawerBody.classList.remove("hidden"); }
    });
  }

  if (schema.presets && schema.presets.length) {
    const row = document.createElement("div");
    row.className = "param-row";
    const label = document.createElement("div");
    label.className = "param-label";
    label.textContent = "Preset";
    const select = document.createElement("select");
    const emptyOpt = document.createElement("option");
    emptyOpt.value = "";
    emptyOpt.textContent = "(none)";
    select.appendChild(emptyOpt);
    schema.presets.forEach((p) => {
      const opt = document.createElement("option");
      opt.value = p;
      opt.textContent = p;
      if (p === state.preset) opt.selected = true;
      select.appendChild(opt);
    });
    select.addEventListener("change", () => { state.preset = select.value || null; });
    row.appendChild(label);
    row.appendChild(select);
    container.appendChild(row);
  }

  if (schema.artifacts && schema.artifacts.length) {
    const row = document.createElement("div");
    row.className = "param-row";
    const label = document.createElement("div");
    label.className = "param-label";
    label.textContent = "Artifact";
    const select = document.createElement("select");
    const emptyOpt = document.createElement("option");
    emptyOpt.value = "";
    emptyOpt.textContent = "(none)";
    select.appendChild(emptyOpt);
    schema.artifacts.forEach((a) => {
      const opt = document.createElement("option");
      opt.value = a;
      opt.textContent = a;
      if (a === state.artifact) opt.selected = true;
      select.appendChild(opt);
    });
    select.addEventListener("change", () => { state.artifact = select.value || null; });
    row.appendChild(label);
    row.appendChild(select);
    container.appendChild(row);
  }

  if (schema.flags && schema.flags.length) {
    const row = document.createElement("div");
    row.className = "param-row";
    const label = document.createElement("div");
    label.className = "param-label";
    label.textContent = "Flags";
    row.appendChild(label);
    schema.flags.forEach((flag) => {
      const line = document.createElement("label");
      line.className = "param-flag";
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = state.flags.has(flag);
      cb.addEventListener("change", () => {
        if (cb.checked) state.flags.add(flag); else state.flags.delete(flag);
      });
      const span = document.createElement("span");
      span.textContent = (schema.flag_descriptions && schema.flag_descriptions[flag]) || flag;
      line.appendChild(cb);
      line.appendChild(span);
      row.appendChild(line);
    });
    container.appendChild(row);
  }

  // Seed
  const seedRow = document.createElement("div");
  seedRow.className = "param-row";
  const seedLabel = document.createElement("div");
  seedLabel.className = "param-label";
  seedLabel.textContent = "Seed";
  const seedInner = document.createElement("div");
  seedInner.className = "seed-row";
  const seedInput = document.createElement("input");
  seedInput.type = "number";
  seedInput.value = state.seed ?? "";
  seedInput.placeholder = "random";
  seedInput.addEventListener("input", () => {
    state.seed = seedInput.value === "" ? null : Number(seedInput.value);
  });
  const diceBtn = document.createElement("button");
  diceBtn.type = "button";
  diceBtn.className = "btn btn-small";
  diceBtn.textContent = "🎲";
  diceBtn.addEventListener("click", () => {
    const v = Math.floor(Math.random() * 2147483647);
    seedInput.value = v;
    state.seed = v;
  });
  seedInner.appendChild(seedInput);
  seedInner.appendChild(diceBtn);
  seedRow.appendChild(seedLabel);
  seedRow.appendChild(seedInner);
  container.appendChild(seedRow);

  if (schema.cost_estimate_usd != null || schema.wall_time_estimate_sec != null) {
    const est = document.createElement("div");
    est.className = "param-desc";
    const parts = [];
    if (schema.cost_estimate_usd != null) parts.push(`~$${Number(schema.cost_estimate_usd).toFixed(2)}`);
    if (schema.wall_time_estimate_sec != null) parts.push(`~${schema.wall_time_estimate_sec}s`);
    est.textContent = parts.join(" · ");
    container.appendChild(est);
  }

  return {
    getValues() {
      return {
        params: { ...state.params },
        flags: Array.from(state.flags),
        seed: state.seed,
        preset: state.preset,
        artifact: state.artifact,
      };
    },
  };
}

function searchText(key, def) {
  return `${key} ${def.description || ""}`.toLowerCase();
}

function filterRows(searchableRows, q) {
  searchableRows.forEach(({ row, text }) => {
    const match = !q || text.includes(q);
    row.classList.toggle("param-row-dim", !match);
    row.classList.toggle("param-row-match", !!q && match);
  });
}

// decorations: { badge, note, onDismiss, pin, pinned, onPin }
function renderParamRow(key, def, val, onChange, decorations) {
  decorations = decorations || {};
  const row = document.createElement("div");
  row.className = "param-row";
  if (decorations.badge) row.classList.add("param-row-proposed");

  const label = document.createElement("div");
  label.className = "param-label";
  const labelLeft = document.createElement("span");
  labelLeft.className = "param-label-left";
  const labelText = document.createElement("span");
  labelText.textContent = key;
  labelLeft.appendChild(labelText);

  if (decorations.badge) {
    const badge = document.createElement("span");
    badge.className = "param-badge-new";
    badge.textContent = decorations.badge;
    if (decorations.note) badge.title = decorations.note;
    labelLeft.appendChild(badge);
  }
  if (decorations.pin) {
    const pinBtn = document.createElement("button");
    pinBtn.type = "button";
    pinBtn.className = "param-pin" + (decorations.pinned ? " active" : "");
    pinBtn.title = decorations.pinned ? "Unpin" : "Pin visible";
    pinBtn.textContent = "📌";
    pinBtn.addEventListener("click", (e) => { e.stopPropagation(); decorations.onPin && decorations.onPin(); });
    labelLeft.appendChild(pinBtn);
  }
  label.appendChild(labelLeft);

  if (decorations.onDismiss) {
    const dismissBtn = document.createElement("button");
    dismissBtn.type = "button";
    dismissBtn.className = "param-dismiss";
    dismissBtn.title = "Dismiss";
    dismissBtn.textContent = "✕";
    dismissBtn.addEventListener("click", (e) => { e.stopPropagation(); decorations.onDismiss(); });
    label.appendChild(dismissBtn);
  }

  if (def.type === "float" || def.type === "int") {
    const readout = document.createElement("span");
    readout.textContent = val;
    label.appendChild(readout);
    row.appendChild(label);

    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = def.min ?? 0;
    slider.max = def.max ?? 1;
    slider.step = def.type === "int" ? 1 : (def.step || 0.01);
    slider.value = val ?? def.default ?? def.min ?? 0;
    slider.addEventListener("input", () => {
      const v = def.type === "int" ? parseInt(slider.value, 10) : parseFloat(slider.value);
      readout.textContent = v;
      onChange(v);
    });
    row.appendChild(slider);
  } else {
    row.appendChild(label);
    const input = document.createElement("input");
    input.type = "text";
    input.value = val ?? "";
    input.addEventListener("input", () => onChange(input.value));
    row.appendChild(input);
  }

  if (decorations.note && !decorations.badge) {
    // note without badge context still surfaces as a subtitle
  }
  if (decorations.note) {
    const noteEl = document.createElement("div");
    noteEl.className = "param-note";
    noteEl.textContent = decorations.note;
    row.appendChild(noteEl);
  }

  if (def.description) {
    const desc = document.createElement("div");
    desc.className = "param-desc";
    desc.textContent = def.description;
    row.appendChild(desc);
  }
  return row;
}
