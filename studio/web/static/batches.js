// batches.js — contact-sheet review of recipe batches.

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

async function api(path) {
  try {
    const res = await fetch(`${API}${path}`);
    if (!res.ok) {
      let msg = res.statusText;
      try { const j = await res.json(); if (j.error) msg = j.error; } catch (e) {}
      toast(`Error: ${msg}`);
      throw new Error(msg);
    }
    return await res.json();
  } catch (err) {
    if (!/^Error:/.test(String(err.message))) toast(`Error: ${err.message}`);
    throw err;
  }
}

const list = document.getElementById("batch-list");

function fmtDate(ts) {
  if (!ts) return "";
  return new Date(ts * 1000).toLocaleDateString(undefined, { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" });
}

function entryEl(batch, entry) {
  if (entry.ok && entry.output) {
    const a = document.createElement("a");
    a.className = "contact-cell";
    a.href = entry.session ? `${BASE}/s/${entry.session}` : "#";
    const img = document.createElement("img");
    img.src = `${API}/batches/${batch.id}/${entry.output}`;
    img.alt = entry.source || "";
    img.loading = "lazy";
    a.appendChild(img);
    return a;
  }
  const err = document.createElement("div");
  err.className = "contact-cell contact-cell-error";
  err.title = entry.error || "failed";
  err.textContent = "✕";
  return err;
}

function batchSection(batch) {
  const section = document.createElement("section");
  section.className = "batch-section";

  const header = document.createElement("div");
  header.className = "batch-header";
  const okCount = (batch.entries || []).filter((e) => e.ok).length;
  header.innerHTML = `
    <div class="batch-title">${batch.recipe_name || batch.recipe || "recipe"}</div>
    <div class="batch-meta">${fmtDate(batch.created)} · ${okCount}/${(batch.entries || []).length} ok</div>
  `;
  section.appendChild(header);

  const sheet = document.createElement("div");
  sheet.className = "contact-sheet";
  (batch.entries || []).forEach((e) => sheet.appendChild(entryEl(batch, e)));
  section.appendChild(sheet);

  return section;
}

async function main() {
  try {
    const { batches } = await api("/batches");
    list.innerHTML = "";
    if (!batches.length) {
      list.innerHTML = '<div class="empty-hint">No batches yet. Apply a recipe from the Recipes view.</div>';
      return;
    }
    batches.forEach((b) => list.appendChild(batchSection(b)));
  } catch (e) {
    list.innerHTML = '<div class="empty-hint">Failed to load batches.</div>';
  }
}

main();
