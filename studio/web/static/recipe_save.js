// recipe_save.js — "Save recipe" dialog: fetch a suggested recipe draft from
// the current session, let the user edit the name and toggle per-step
// "general" (travels to other photos) vs photo-specific (dropped on apply).

export function initRecipeSave({ apiBase, base, sessionId, buttonEl, toast }) {
  buttonEl.addEventListener("click", () => openDialog({ apiBase, base, sessionId, toast }));
}

async function openDialog({ apiBase, base, sessionId, toast }) {
  let suggestion;
  try {
    const res = await fetch(`${apiBase}/recipes/suggest?session_id=${encodeURIComponent(sessionId)}`);
    if (!res.ok) {
      let msg = res.statusText;
      try { const j = await res.json(); if (j.error) msg = j.error; } catch (e) {}
      throw new Error(msg);
    }
    suggestion = await res.json();
  } catch (err) {
    toast(`Error: ${err.message}`);
    return;
  }

  const overlay = document.createElement("div");
  overlay.className = "lock-overlay";

  const box = document.createElement("div");
  box.className = "lock-box recipe-save-box";
  overlay.appendChild(box);

  const title = document.createElement("div");
  title.className = "lock-title";
  title.textContent = "Save recipe";
  box.appendChild(title);

  const nameRow = document.createElement("div");
  nameRow.className = "param-row";
  const nameLabel = document.createElement("div");
  nameLabel.className = "param-label";
  nameLabel.innerHTML = "<span>Name</span>";
  nameRow.appendChild(nameLabel);
  const nameInput = document.createElement("input");
  nameInput.type = "text";
  nameInput.value = suggestion.name || "";
  nameRow.appendChild(nameInput);
  box.appendChild(nameRow);

  const stepsLabel = document.createElement("div");
  stepsLabel.className = "param-label";
  stepsLabel.innerHTML = "<span>Steps (uncheck to mark photo-specific — excluded on apply)</span>";
  box.appendChild(stepsLabel);

  const stepsList = document.createElement("div");
  stepsList.className = "recipe-save-steps";
  const checkboxes = [];
  (suggestion.steps || []).forEach((step) => {
    const row = document.createElement("label");
    row.className = "param-flag recipe-save-step";

    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = step.general !== false;
    checkboxes.push(cb);

    const info = document.createElement("div");
    info.className = "recipe-save-step-info";
    const toolName = document.createElement("div");
    toolName.textContent = step.tool;
    info.appendChild(toolName);
    const params = Object.entries(step.params || {})
      .map(([k, v]) => `${k}=${v}`)
      .join(", ");
    if (params) {
      const paramsEl = document.createElement("div");
      paramsEl.className = "recipe-save-step-params";
      paramsEl.textContent = params;
      info.appendChild(paramsEl);
    }

    row.appendChild(cb);
    row.appendChild(info);
    stepsList.appendChild(row);
  });
  box.appendChild(stepsList);

  const errEl = document.createElement("div");
  errEl.className = "lock-error hidden";
  box.appendChild(errEl);

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
  confirmBtn.textContent = "Save recipe";
  actionRow.appendChild(cancelBtn);
  actionRow.appendChild(confirmBtn);
  box.appendChild(actionRow);

  document.body.appendChild(overlay);
  function close() { overlay.remove(); }

  confirmBtn.addEventListener("click", async () => {
    errEl.classList.add("hidden");
    confirmBtn.disabled = true;
    cancelBtn.disabled = true;
    try {
      const body = {
        session_id: sessionId,
        name: nameInput.value.trim() || suggestion.name,
        general_marks: checkboxes.map((cb) => cb.checked),
      };
      const res = await fetch(`${apiBase}/recipes`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        let msg = res.statusText;
        try { const j = await res.json(); if (j.error) msg = j.error; } catch (e) {}
        throw new Error(msg);
      }
      close();
      const link = `${base}/recipes`;
      toast(`Recipe saved — see Recipes view (${link})`);
    } catch (err) {
      errEl.textContent = err.message;
      errEl.classList.remove("hidden");
      confirmBtn.disabled = false;
      cancelBtn.disabled = false;
    }
  });
}
