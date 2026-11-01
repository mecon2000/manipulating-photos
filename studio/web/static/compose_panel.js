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
  const movable = top.filter((L) => !["finish", "base"].includes(L.z) && !L.removed);
  const removed = top.filter((L) => L.removed);

  fixedTop.forEach((L) => body.appendChild(fixedRow(L, ctx)));
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
  fixedBottom.forEach((L) => body.appendChild(fixedRow(L, ctx)));
  if (removed.length) body.appendChild(removedSection(removed, ctx));
  body.appendChild(el("div", "panel-legend", `top of the list = top of the stack · ${RIFF_TIP}`));
}

let removedOpen = false;   // survives panel re-renders
function removedSection(removed, ctx) {
  const det = el("details", "removed-sec");
  det.open = removedOpen;
  det.ontoggle = () => { removedOpen = det.open; };
  det.appendChild(el("summary", "removed-head", `Removed (${removed.length})`));
  removed.forEach((L) => {
    const row = el("div", "layer-row removed-row");
    const n = el("span", "layer-name", pretty(L.id));
    n.title = `${L.id} · ${L.kind || ""}`;
    const back = el("button", "btn btn-small", "restore");
    back.title = "put it back in the stack where it was";
    back.onclick = () => ctx.onRestore(L);
    row.append(n, back);
    det.appendChild(row);
  });
  return det;
}

// Sliders per layer kind: [param, label, min, max, step, default]. Released → one quick-set.
const SLIDERS = {
  embers: [["saturation", "colour", 0, 1, 0.02, 0.12], ["opacity", "visibility", 0, 1, 0.02, 0.55],
           ["spread", "spread", 0.05, 0.8, 0.01, 0.3], ["chaos", "wind wildness", 0, 1, 0.02, 0.45],
           ["count", "amount", 40, 800, 10, 160], ["size", "spark size", 0.4, 3, 0.05, 1],
           ["trail", "trail length", 0, 2, 0.05, 0.5], ["inside", "into her outline", 0, 0.1, 0.005, 0.025]],
  motion_smear: [["strength", "strength", 0, 5, 0.1, 1], ["length", "length", 0, 0.6, 0.01, 0.25],
                 ["opacity", "visibility", 0, 1, 0.02, 0.8]],
  ghost: [["strength", "strength", 0, 5, 0.1, 1], ["opacity", "visibility", 0, 1, 0.02, 0.4],
          ["offset", "offset", 0, 0.2, 0.005, 0.05]],
  light_leak: [["opacity", "visibility", 0, 1, 0.02, 0.5], ["width", "width", 0.1, 2, 0.05, 1]],
  haze: [["opacity", "visibility", 0, 1, 0.02, 0.5]],
  vignette: [["opacity", "visibility", 0, 1, 0.02, 0.6]],
};
const openSliders = new Set();   // survives panel re-renders

function sliderBox(L, ctx) {
  const box = el("div", "layer-sliders");
  const P = L.params || (L.params = {});
  for (const [k, label, lo, hi, step, def] of SLIDERS[L.kind]) {
    const row = el("label", "slider-row");
    const val = el("span", "slider-val");
    const r = el("input");
    r.type = "range"; r.min = lo; r.max = hi; r.step = step;
    r.value = P[k] ?? def;
    val.textContent = r.value;
    r.oninput = () => { val.textContent = r.value; };
    // while a finger is on the slider the page must not rebuild the panel (a rebuild
    // replaces the input and the drag dies — the 6s poll / render updates did that)
    r.addEventListener("pointerdown", () => { _sliding = true; });
    const end = () => { _sliding = false; };
    r.addEventListener("pointerup", end);
    r.addEventListener("pointercancel", end);
    r.addEventListener("blur", end);
    r.onchange = () => { end(); P[k] = Number(r.value); ctx.onParam(L, k, Number(r.value)); };
    row.append(el("span", "slider-label", label), r, val);
    box.appendChild(row);
  }
  return box;
}

function pretty(id) {
  return String(id).replace(/_/g, " ");
}

function fixedRow(L, ctx) {
  const isBase = L.z === "base";
  const row = el("div", "layer-row fixed-row" + (isBase || L.enabled ? "" : " off"));
  row.appendChild(el("span", "drag-handle dim", "·"));
  if (!isBase) {                  // grain & scratches can be switched off; paper can't
    const cb = el("input");
    cb.type = "checkbox";
    cb.checked = !!L.enabled;
    cb.title = "show / hide grain & scratches";
    cb.onchange = () => ctx.onToggle(L, cb.checked);
    row.appendChild(cb);
  }
  const n = el("span", "layer-name", isBase ? "paper" : "grain & scratches");
  n.title = isBase ? "the paper background" : `finish (${L.blend || "normal"})`;
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
  const rm = el("button", "rmbtn", "✕");
  rm.title = "remove (goes to Removed at the bottom — restorable)";
  rm.onclick = () => ctx.onRemove(L);

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
    row.appendChild(chips);
  }
  if (SLIDERS[L.kind]) {
    const tune = el("button", "seedbtn" + (openSliders.has(L.id) ? " active" : ""), "⚙");
    tune.title = "sliders: colour, visibility, spread …";
    tune.onclick = () => {
      openSliders.has(L.id) ? openSliders.delete(L.id) : openSliders.add(L.id);
      ctx.onRefresh();
    };
    row.appendChild(tune);
  }
  row.appendChild(roll);
  row.appendChild(rm);
  if (openSliders.has(L.id) && SLIDERS[L.kind]) wrap.appendChild(sliderBox(L, ctx));
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

// True while a row is being dragged — the page must not rebuild the list then
// (a rebuild destroys the dragged node and silently ends the drag).
let _dragging = false;
let _sliding = false;
export const isDragging = () => _dragging || _sliding;

function scrollerOf(node) {
  for (let n = node.parentElement; n; n = n.parentElement) {
    const oy = getComputedStyle(n).overflowY;
    if ((oy === "auto" || oy === "scroll") && n.scrollHeight > n.clientHeight) return n;
  }
  return document.scrollingElement;
}

// Pointer-based (touch-safe) drag: the item follows the finger by being
// re-inserted among its siblings at the midpoint it crosses. Holding near the
// top/bottom edge of the scrolling panel keeps scrolling (rAF), even without moving.
function wireDrag(handle, item, list, ctx) {
  handle.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    handle.setPointerCapture(e.pointerId);
    _dragging = true;
    item.classList.add("dragging");
    const startOrder = [...list.children].map((c) => c.dataset.id).join(",");
    const sc = scrollerOf(list);
    let y = e.clientY, raf = 0;
    const place = () => {
      const sibs = [...list.children].filter((c) => c !== item);
      let before = null;
      for (const s of sibs) {
        const r = s.getBoundingClientRect();
        if (y < r.top + r.height / 2) { before = s; break; }
      }
      if (before !== item.nextSibling) {
        const keep = sc.scrollTop;              // scroll anchoring would shift it a row —
        placeAt(list, item, sibs, before);      // only our edge tick may scroll
        sc.scrollTop = keep;
      }
    };
    // Edge auto-scroll, speed ∝ how far the finger is pulled toward/past the edge:
    // inside the thin edge band it creeps (after a ~350ms rest); past the edge it
    // speeds up with distance (pointer capture keeps reporting outside the panel).
    let edgeSince = 0, lastT = 0, carry = 0;
    const tick = (t) => {
      raf = 0;
      const r = sc === document.scrollingElement
        ? { top: 0, bottom: window.innerHeight } : sc.getBoundingClientRect();
      const zone = Math.min(28, (r.bottom - r.top) / 6);
      let d = 0;                                   // px into the band / past the edge
      if (y < r.top + zone) d = -(r.top + zone - y);
      else if (y > r.bottom - zone) d = y - (r.bottom - zone);
      if (!d) { edgeSince = 0; lastT = 0; carry = 0; return; }
      if (!edgeSince) edgeSince = t;
      const past = Math.abs(d) - zone;             // >0 once outside the panel
      if (past > 0 || t - edgeSince > 350) {
        // px/s: 30 creeping in the band → 150 max once ~120px past the edge.
        // Time-based with a fractional carry so it's frame-rate independent.
        const pxs = past > 0 ? Math.min(150, 30 + past) : 30;
        const dt = lastT ? Math.min(50, t - lastT) : 16;
        carry += Math.sign(d) * pxs * dt / 1000;
        const step = Math.trunc(carry);
        carry -= step;
        lastT = t;
        if (step) {
          const before = sc.scrollTop;
          sc.scrollTop += step;
          if (sc.scrollTop === before) { lastT = 0; carry = 0; return; }
          place();
        }
      } else lastT = 0;
      raf = requestAnimationFrame(tick);
    };
    const move = (ev) => {
      y = ev.clientY;
      place();
      if (!raf) raf = requestAnimationFrame(tick);
    };
    const up = () => {
      handle.removeEventListener("pointermove", move);
      handle.removeEventListener("pointerup", up);
      handle.removeEventListener("pointercancel", up);
      if (raf) cancelAnimationFrame(raf);
      _dragging = false;
      item.classList.remove("dragging");
      const ids = [...list.children].map((c) => c.dataset.id);
      if (ids.join(",") !== startOrder) ctx.onReorder(ids.reverse()); // → bottom→top
      else ctx.onDragEnd?.();
    };
    handle.addEventListener("pointermove", move);
    handle.addEventListener("pointerup", up);
    handle.addEventListener("pointercancel", up);
  });
}
