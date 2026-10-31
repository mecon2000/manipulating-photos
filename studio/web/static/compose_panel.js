// compose_panel.js — layer panel: one drag-sortable list (shown top→bottom),
// "subject" as a drag-through divider (crossing it flips under/over), per-row
// on/off, 🎲 variant roll with seed chips, notes on tap.

const RIFF_TIP = "riff = added by the wildness step, not in the reference — based on this photo/model";

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

// ctx: {layers (bottom→top), overrides {id: seed}, rolling Set, openNotes Set,
//       onToggle(L, on), onReorder(ids), onRoll(L), onChip(L, seed), onUse(L, seed), onNote(id)}
export function renderPanel(body, ctx) {
  body.innerHTML = "";
  const top = ctx.layers.slice().reverse();
  const fixedTop = top.filter((L) => L.z === "finish");
  const fixedBottom = top.filter((L) => L.z === "base");
  const movable = top.filter((L) => !["finish", "base"].includes(L.z));

  fixedTop.forEach((L) => body.appendChild(fixedRow(L)));
  const list = el("div", "drag-list");
  movable.forEach((L) => {
    const wrap = el("div", "drag-item");
    wrap.dataset.id = L.id;
    if (L.id === "subject" || L.z === "subject") {
      wrap.dataset.id = "subject";
      wrap.appendChild(el("div", "layer-row subject-row", "— SUBJECT —  (drag layers across)"));
    } else {
      layerRows(wrap, L, ctx, list);
    }
    list.appendChild(wrap);
  });
  body.appendChild(list);
  fixedBottom.forEach((L) => body.appendChild(fixedRow(L)));
  body.appendChild(el("div", "panel-legend", `top of the list = top of the stack · ${RIFF_TIP}`));
}

function pretty(id) {
  return String(id).replace(/_/g, " ");
}

function fixedRow(L) {
  const row = el("div", "layer-row fixed-row");
  row.appendChild(el("span", "drag-handle dim", "·"));
  const n = el("span", "layer-name", L.z === "base" ? "paper" : "grain & scratches");
  n.title = L.z === "base" ? "the paper background" : `finish (${L.blend || "normal"})`;
  row.appendChild(n);
  return row;
}

function layerRows(wrap, L, ctx, list) {
  const row = el("div", "layer-row" + (L.enabled ? "" : " off"));
  const handle = el("span", "drag-handle", "≡");
  handle.title = "drag to reorder";
  wireDrag(handle, wrap, list, ctx);
  const cb = el("input");
  cb.type = "checkbox";
  cb.checked = !!L.enabled;
  cb.onchange = () => ctx.onToggle(L, cb.checked);
  // one readable name; the generator type + blend live in the tooltip
  const name = el("button", "layer-name", pretty(L.id));
  name.title = `${L.id} · ${L.kind || ""}${L.blend && L.blend !== "normal" ? " · " + L.blend : ""} — tap for the note`;
  if (L.riff) {
    const b = el("span", "riff-badge", "riff");
    b.title = RIFF_TIP;
    name.appendChild(b);
  }
  name.onclick = () => ctx.onNote(L.id);
  const roll = el("button", "seedbtn", ctx.rolling.has(L.id) ? "" : "🎲");
  roll.title = "roll a variant (new random layout)";
  if (ctx.rolling.has(L.id)) { roll.appendChild(el("span", "spin")); roll.disabled = true; }
  roll.onclick = () => ctx.onRoll(L);
  row.append(handle, cb, name);
  wrap.appendChild(row);

  const seeds = [L.seed, ...(L.variants || []).filter((s) => s !== L.seed)];
  if (seeds.length > 1) {
    const chips = el("span", "seed-chips");   // inline, on the layer's own line
    const active = ctx.overrides[L.id] ?? L.seed;
    seeds.forEach((s, i) => {
      const c = el("button", "seed-chip" + (s === active ? " active" : "") + (s === L.seed ? " current" : ""),
                    String(i + 1));
      c.title = s === L.seed ? `seed ${s} (in use)` : `seed ${s}`;
      c.onclick = () => ctx.onChip(L, s);
      chips.appendChild(c);
    });
    if (active !== L.seed && L._pendingSeed !== active) {
      const use = el("button", "btn btn-small btn-accent", "use");
      use.onclick = () => ctx.onUse(L, active);
      chips.appendChild(use);
    }
    row.appendChild(chips);
  }
  row.appendChild(roll);
  if (ctx.openNotes.has(L.id)) {
    wrap.appendChild(el("div", "layer-note", L.note || "(no note)"));
  }
}

// Re-place `item` before `before` (null = end) by moving the OTHER rows around it:
// removing the dragged node itself from the DOM would drop its pointer capture.
function placeAt(list, item, sibs, before) {
  const k = before ? sibs.indexOf(before) : sibs.length;
  sibs.slice(0, k).forEach((n) => list.insertBefore(n, item));
  sibs.slice(k).forEach((n) => list.appendChild(n));
}

// Pointer-based (touch-safe) drag: the item follows the finger by being
// re-inserted among its siblings at the midpoint it crosses.
function wireDrag(handle, item, list, ctx) {
  handle.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    handle.setPointerCapture(e.pointerId);
    item.classList.add("dragging");
    const startOrder = [...list.children].map((c) => c.dataset.id).join(",");
    const move = (ev) => {
      const sibs = [...list.children].filter((c) => c !== item);
      let before = null;
      for (const s of sibs) {
        const r = s.getBoundingClientRect();
        if (ev.clientY < r.top + r.height / 2) { before = s; break; }
      }
      if (before !== item.nextSibling) placeAt(list, item, sibs, before);
      const lr = list.getBoundingClientRect();
      const sc = list.closest(".studio-layout") || list.parentElement;
      if (ev.clientY < lr.top + 30 || ev.clientY > window.innerHeight - 30) {
        sc.scrollBy(0, ev.clientY < window.innerHeight / 2 ? -12 : 12);
      }
    };
    const up = () => {
      handle.removeEventListener("pointermove", move);
      handle.removeEventListener("pointerup", up);
      handle.removeEventListener("pointercancel", up);
      item.classList.remove("dragging");
      const ids = [...list.children].map((c) => c.dataset.id);
      if (ids.join(",") !== startOrder) ctx.onReorder(ids.reverse()); // → bottom→top
    };
    handle.addEventListener("pointermove", move);
    handle.addEventListener("pointerup", up);
    handle.addEventListener("pointercancel", up);
  });
}
