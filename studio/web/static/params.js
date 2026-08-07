// params.js — render a param form from a tool schema, read back values.

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

// Renders the param form into `container`. Returns a `getValues()` accessor.
export function renderParamForm(container, schema, initial) {
  container.innerHTML = "";
  initial = initial || {};
  const state = {
    params: {},
    flags: new Set(initial.flags || []),
    seed: initial.seed ?? null,
    preset: initial.preset ?? null,
    artifact: initial.artifact ?? null,
  };

  const params = schema.params || {};
  Object.keys(params).forEach((key) => {
    const def = params[key];
    const val = initial.params && initial.params[key] !== undefined ? initial.params[key] : def.default;
    state.params[key] = val;
    container.appendChild(renderParamRow(key, def, val, (v) => { state.params[key] = v; }));
  });

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

function renderParamRow(key, def, val, onChange) {
  const row = document.createElement("div");
  row.className = "param-row";
  const label = document.createElement("div");
  label.className = "param-label";
  const labelText = document.createElement("span");
  labelText.textContent = key;
  label.appendChild(labelText);

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

  if (def.description) {
    const desc = document.createElement("div");
    desc.className = "param-desc";
    desc.textContent = def.description;
    row.appendChild(desc);
  }
  return row;
}
