// compose_side.js — choice cards, pinned-note list, history (render-only helpers).

export function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

// h: {api (P_API base), onPick(q, letter, comment), lightbox(src, cap)}
export function renderQuestions(box, state, h) {
  box.innerHTML = "";
  for (const q of state.questions.filter((x) => x.status === "open")) {
    const card = el("div", "qcard");
    card.appendChild(el("div", "q-text", q.question));
    card.appendChild(el("div", "q-key", q.key));
    const opts = el("div", "q-opts");
    const comment = el("input", "text-in");
    comment.placeholder = "comment (optional)";
    const items = q.options.map((o) => ({
      src: `${h.api}/file/questions/${o.file}`, cap: `${o.letter} — ${o.value}`,
      pick: () => h.onPick(q, o.letter, comment.value.trim()), pickLabel: `Pick ${o.letter}`,
    }));
    q.options.forEach((o, i) => {
      const col = el("div", "q-opt");
      const img = el("img");
      img.loading = "lazy";
      img.src = `${h.api}/file/questions/${o.file}?w=600`;
      img.onclick = () => h.lightbox(items, i);
      const btn = el("button", "btn btn-accent", o.letter);
      btn.onclick = () => h.onPick(q, o.letter, comment.value.trim());
      col.append(img, btn, el("div", "q-val", String(o.value)));
      opts.appendChild(col);
    });
    card.append(opts, comment);
    box.appendChild(card);
  }
}

export const openPins = (state) => state.comments.filter((c) => c.status === "open");

// h: {onStatus(c, status)}
export function renderPinList(body, state, h) {
  const pins = openPins(state);
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
    b.onclick = () => h.onStatus(c, c.status === "open" ? "done" : "open");
    row.appendChild(b);
    body.appendChild(row);
  }
}

function describeDecision(d) {
  if (d.init) return ["init", `wildness ${d.init.wildness}`];
  if (d.question) return [d.question, `→ ${d.choice}${d.comment ? ` (“${d.comment}”)` : ""}`];
  if (d.set) return [d.note || "set", Object.entries(d.set).map(([k, v]) => `${k}=${v}`).join(", ")];
  if (d.order) return ["reorder", Array.isArray(d.order) ? d.order.join(" › ") : String(d.order)];
  if (d.restore != null) return ["restore", `savepoint ${d.restore}`];
  const [k] = Object.keys(d).filter((x) => x !== "t");
  return [k || "", JSON.stringify(d[k] ?? "")];
}

export function renderHistory(body, state, h) {
  body.innerHTML = "";
  const answered = new Map(state.questions.filter((q) => q.status === "answered")
                                         .map((q) => [q.question, q]));
  for (const d of state.decisions.slice().reverse()) {
    const row = el("div", "hist-row");
    const [head, tail] = describeDecision(d);
    const q = d.question && answered.get(d.question);
    const o = q && q.options.find((x) => x.letter === q.choice);
    if (o) {
      const img = el("img");
      img.loading = "lazy";
      img.src = `${h.api}/file/questions/${o.file}?w=160`;
      img.onclick = () => h.lightbox(`${h.api}/file/questions/${q.id}_sheet.jpg`, q.question);
      row.appendChild(img);
    }
    row.appendChild(el("span", "", `${(d.t || "").slice(5, 16)} `));
    row.appendChild(el("b", "", head));
    row.appendChild(el("span", "", ` ${tail}`));
    body.appendChild(row);
  }
}
