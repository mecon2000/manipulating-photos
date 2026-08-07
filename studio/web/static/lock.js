// lock.js — "Lock ⭐" dialog: upscale the exact result or re-render at full res.

export function initLock({ apiBase, sessionId, buttonEl, toast, onLocked }) {
  buttonEl.addEventListener("click", () => openDialog({ apiBase, sessionId, toast, onLocked }));
}

function openDialog({ apiBase, sessionId, toast, onLocked }) {
  const overlay = document.createElement("div");
  overlay.className = "lock-overlay";

  const box = document.createElement("div");
  box.className = "lock-box";
  overlay.appendChild(box);

  const title = document.createElement("div");
  title.className = "lock-title";
  title.textContent = "Lock this result";
  box.appendChild(title);

  let mode = "upscale";
  let scale = 4;

  const modeRow = document.createElement("div");
  modeRow.className = "lock-mode-row";
  const upscaleBtn = mkChoiceBtn("Upscale this exact result", true);
  const rerenderBtn = mkChoiceBtn("Re-render at full res", false);
  modeRow.appendChild(upscaleBtn);
  modeRow.appendChild(rerenderBtn);
  box.appendChild(modeRow);

  const scaleRow = document.createElement("div");
  scaleRow.className = "lock-scale-row";
  const s2 = mkChoiceBtn("2×", false, true);
  const s4 = mkChoiceBtn("4×", true, true);
  scaleRow.appendChild(s2);
  scaleRow.appendChild(s4);
  box.appendChild(scaleRow);

  const warning = document.createElement("div");
  warning.className = "lock-warning hidden";
  warning.textContent = "generative steps may come out different";
  box.appendChild(warning);

  const errEl = document.createElement("div");
  errEl.className = "lock-error hidden";
  box.appendChild(errEl);

  const spinner = document.createElement("div");
  spinner.className = "lock-spinner hidden";
  spinner.textContent = "Working… this can take a while.";
  box.appendChild(spinner);

  const actionRow = document.createElement("div");
  actionRow.className = "run-row";
  const cancelBtn = document.createElement("button");
  cancelBtn.type = "button";
  cancelBtn.className = "btn";
  cancelBtn.textContent = "Cancel";
  cancelBtn.addEventListener("click", close);
  const confirmBtn = document.createElement("button");
  confirmBtn.type = "button";
  confirmBtn.className = "btn btn-accent";
  confirmBtn.textContent = "Lock ⭐";
  actionRow.appendChild(cancelBtn);
  actionRow.appendChild(confirmBtn);
  box.appendChild(actionRow);

  function mkChoiceBtn(label, active, small) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "btn" + (small ? " btn-small" : "") + (active ? " active" : "");
    b.textContent = label;
    return b;
  }

  function setMode(m) {
    mode = m;
    upscaleBtn.classList.toggle("active", m === "upscale");
    rerenderBtn.classList.toggle("active", m === "rerender");
    scaleRow.classList.toggle("hidden", m !== "upscale");
    warning.classList.toggle("hidden", m !== "rerender");
  }
  upscaleBtn.addEventListener("click", () => setMode("upscale"));
  rerenderBtn.addEventListener("click", () => setMode("rerender"));
  s2.addEventListener("click", () => { scale = 2; s2.classList.add("active"); s4.classList.remove("active"); });
  s4.addEventListener("click", () => { scale = 4; s4.classList.add("active"); s2.classList.remove("active"); });

  document.body.appendChild(overlay);

  function close() { overlay.remove(); }

  confirmBtn.addEventListener("click", async () => {
    errEl.classList.add("hidden");
    confirmBtn.disabled = true;
    cancelBtn.disabled = true;
    spinner.classList.remove("hidden");
    try {
      const body = { mode };
      if (mode === "upscale") body.scale = scale;
      const res = await fetch(`${apiBase}/session/${sessionId}/lock`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        let msg = res.statusText;
        try { const j = await res.json(); if (j.error) msg = j.error; } catch (e) {}
        throw new Error(msg);
      }
      const data = await res.json();
      close();
      toast("Locked → favorites ⭐");
      if (onLocked) onLocked(data);
    } catch (err) {
      errEl.textContent = err.message;
      errEl.classList.remove("hidden");
      spinner.classList.add("hidden");
      confirmBtn.disabled = false;
      cancelBtn.disabled = false;
      return;
    }
  });
}
