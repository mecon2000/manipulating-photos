// recipes.js — Recipes view: grid of saved recipes, apply-to-batch dialog,
// import-from-favorites section.

const BASE = window.STUDIO.base || "";
const API = `${BASE}/api`;

const toastEl = document.getElementById("toast");
let toastTimer = null;
function toast(msg, ms = 3000) {
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

const grid = document.getElementById("recipe-grid");
const favList = document.getElementById("fav-list");

function fmtDate(ts) {
  if (!ts) return "";
  return new Date(ts * 1000).toLocaleDateString(undefined, { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" });
}

function stepChain(steps) {
  if (!steps || !steps.length) return "(empty)";
  return steps.map((s) => s.tool).join(" → ");
}

function recipeCard(r) {
  const card = document.createElement("div");
  card.className = "recipe-card";

  const thumb = document.createElement("div");
  thumb.className = "recipe-thumb";
  if (r.thumbnail) {
    const img = document.createElement("img");
    img.src = `${API}/recipes/${r.slug}/thumb`;
    img.alt = r.name;
    img.onerror = () => { thumb.classList.add("recipe-thumb-placeholder"); img.remove(); };
    thumb.appendChild(img);
  } else {
    thumb.classList.add("recipe-thumb-placeholder");
  }
  card.appendChild(thumb);

  const body = document.createElement("div");
  body.className = "recipe-body";

  const name = document.createElement("div");
  name.className = "recipe-name";
  name.textContent = r.name;
  body.appendChild(name);

  const chain = document.createElement("div");
  chain.className = "recipe-chain";
  chain.textContent = stepChain(r.steps);
  body.appendChild(chain);

  const meta = document.createElement("div");
  meta.className = "recipe-meta";
  meta.textContent = fmtDate(r.created);
  body.appendChild(meta);

  const actions = document.createElement("div");
  actions.className = "recipe-actions";

  const renameBtn = document.createElement("button");
  renameBtn.type = "button";
  renameBtn.className = "btn btn-small";
  renameBtn.textContent = "Rename";
  renameBtn.addEventListener("click", async () => {
    const next = prompt("New name", r.name);
    if (!next || !next.trim() || next.trim() === r.name) return;
    try {
      const updated = await api(`/recipes/${r.slug}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: next.trim() }),
      });
      r.name = updated.name;
      name.textContent = r.name;
      toast("Renamed");
    } catch (e) { /* toasted */ }
  });

  const deleteBtn = document.createElement("button");
  deleteBtn.type = "button";
  deleteBtn.className = "btn btn-small";
  deleteBtn.textContent = "Delete";
  deleteBtn.addEventListener("click", async () => {
    if (!confirm(`Delete recipe "${r.name}"?`)) return;
    try {
      await api(`/recipes/${r.slug}`, { method: "DELETE" });
      card.remove();
      toast("Deleted");
    } catch (e) { /* toasted */ }
  });

  const applyBtn = document.createElement("button");
  applyBtn.type = "button";
  applyBtn.className = "btn btn-small btn-accent";
  applyBtn.textContent = "Apply to…";
  applyBtn.addEventListener("click", () => openApplyDialog(r));

  actions.appendChild(renameBtn);
  actions.appendChild(deleteBtn);
  actions.appendChild(applyBtn);
  body.appendChild(actions);

  card.appendChild(body);
  return card;
}

function openApplyDialog(r) {
  const overlay = document.createElement("div");
  overlay.className = "lock-overlay";

  const box = document.createElement("div");
  box.className = "lock-box";
  overlay.appendChild(box);

  const title = document.createElement("div");
  title.className = "lock-title";
  title.textContent = `Apply "${r.name}" to…`;
  box.appendChild(title);

  const nRow = document.createElement("div");
  nRow.className = "param-row";
  const nLabel = document.createElement("div");
  nLabel.className = "param-label";
  nLabel.innerHTML = "<span>Number of candidates</span>";
  nRow.appendChild(nLabel);
  const nInput = document.createElement("input");
  nInput.type = "number";
  nInput.min = "1";
  nInput.max = "12";
  nInput.value = "4";
  nRow.appendChild(nInput);
  box.appendChild(nRow);

  const mRow = document.createElement("div");
  mRow.className = "param-row";
  const mLabel = document.createElement("div");
  mLabel.className = "param-label";
  mLabel.innerHTML = "<span>Model name contains (optional)</span>";
  mRow.appendChild(mLabel);
  const mInput = document.createElement("input");
  mInput.type = "text";
  mInput.placeholder = "e.g. anya";
  mRow.appendChild(mInput);
  box.appendChild(mRow);

  const errEl = document.createElement("div");
  errEl.className = "lock-error hidden";
  box.appendChild(errEl);

  const actionRow = document.createElement("div");
  actionRow.className = "run-row";
  const cancelBtn = document.createElement("button");
  cancelBtn.type = "button";
  cancelBtn.className = "btn";
  cancelBtn.textContent = "Cancel";
  cancelBtn.addEventListener("click", () => overlay.remove());
  const confirmBtn = document.createElement("button");
  confirmBtn.type = "button";
  confirmBtn.className = "btn btn-accent";
  confirmBtn.textContent = "N random candidates";
  actionRow.appendChild(cancelBtn);
  actionRow.appendChild(confirmBtn);
  box.appendChild(actionRow);

  confirmBtn.addEventListener("click", async () => {
    errEl.classList.add("hidden");
    let n = parseInt(nInput.value, 10);
    if (!n || n < 1) n = 1;
    if (n > 12) n = 12;
    confirmBtn.disabled = true;
    cancelBtn.disabled = true;
    try {
      const body = { n };
      if (mInput.value.trim()) body.model = mInput.value.trim();
      const res = await fetch(`${API}/recipes/${r.slug}/apply`, {
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
      overlay.remove();
      toast(`Batch started (job ${data.job}) — results appear in Batches + hub Jobs tab`);
    } catch (err) {
      errEl.textContent = err.message;
      errEl.classList.remove("hidden");
      confirmBtn.disabled = false;
      cancelBtn.disabled = false;
    }
  });

  document.body.appendChild(overlay);
}

function favRow(f) {
  const row = document.createElement("div");
  row.className = "fav-row";

  const info = document.createElement("div");
  info.className = "fav-info";
  const name = document.createElement("div");
  name.className = "fav-name";
  name.textContent = f.file || "(unknown)";
  const tool = document.createElement("div");
  tool.className = "fav-tool";
  tool.textContent = f.tool || "";
  info.appendChild(name);
  info.appendChild(tool);
  row.appendChild(info);

  const importBtn = document.createElement("button");
  importBtn.type = "button";
  importBtn.className = "btn btn-small btn-accent";
  importBtn.textContent = "Import";
  importBtn.addEventListener("click", async () => {
    importBtn.disabled = true;
    try {
      const recipe = await api("/recipes/import-favorite", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ file: f.file }),
      });
      grid.querySelector(".empty-hint")?.remove();
      grid.prepend(recipeCard(recipe));
      toast("Imported");
    } catch (e) {
      importBtn.disabled = false;
    }
  });
  row.appendChild(importBtn);

  return row;
}

async function main() {
  try {
    const { recipes } = await api("/recipes");
    grid.innerHTML = "";
    if (!recipes.length) {
      grid.innerHTML = '<div class="empty-hint">No recipes yet. Save one from a Studio session.</div>';
    } else {
      recipes.forEach((r) => grid.appendChild(recipeCard(r)));
    }
  } catch (e) {
    grid.innerHTML = '<div class="empty-hint">Failed to load recipes.</div>';
  }

  try {
    const { favorites } = await api("/favorites");
    favList.innerHTML = "";
    if (!favorites.length) {
      favList.innerHTML = '<div class="empty-hint">No favorites yet.</div>';
    } else {
      favorites.forEach((f) => favList.appendChild(favRow(f)));
    }
  } catch (e) {
    favList.innerHTML = '<div class="empty-hint">Failed to load favorites.</div>';
  }
}

main();
