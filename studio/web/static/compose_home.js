// compose_home.js — pieces list + "new piece" (recipe × photo × wildness → background init).

const BASE = window.COMPOSE.base || "";
const API = `${BASE}/api/compose`;
const $ = (id) => document.getElementById(id);

const sel = { recipe: null, picked: null, photo: null };
const watching = new Set(); // job ids started from this page

let toastTimer = null;
function toast(msg, ms = 3500) {
  const t = $("toast");
  t.textContent = msg;
  t.classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.add("hidden"), ms);
}

async function api(path, opts) {
  const res = await fetch(`${API}${path}`, opts);
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch (e) {}
    throw new Error(msg);
  }
  return res.json();
}

const thumbUrl = (path, w = 240) => `${API}/thumb?w=${w}&path=${encodeURIComponent(path)}`;
const pieceUrl = (p) => `${BASE}/compose/p/${encodeURIComponent(p)}`;
const baseName = (p) => (p || "").split("/").pop();

// ---- pieces + jobs ----
async function refresh() {
  let data;
  try { data = await api("/pieces"); } catch (e) { toast(`Error: ${e.message}`); return; }
  const grid = $("pieces");
  grid.innerHTML = "";
  if (!data.pieces.length) grid.innerHTML = '<div class="empty-hint">No pieces yet.</div>';
  for (const p of data.pieces) {
    const a = document.createElement("a");
    a.className = "piece-card";
    a.href = pieceUrl(p.piece);
    const img = p.final_v
      ? `<img loading="lazy" src="${API}/piece/${encodeURIComponent(p.piece)}/file/final.jpg?w=400&v=${p.final_v}">`
      : '<img alt="">';
    a.innerHTML = `${img}<div class="pc-body">
      <div class="pc-photo"></div><div class="pc-dim pc-recipe"></div>
      <div class="pc-dim pc-last"></div></div>`;
    a.querySelector(".pc-photo").textContent = p.photo;
    a.querySelector(".pc-recipe").textContent = p.recipe;
    a.querySelector(".pc-last").textContent = `${p.changed} · ${p.last || ""}`;
    if (p.open_q) {
      const b = document.createElement("span");
      b.className = "badge-q";
      b.textContent = `${p.open_q} to pick`;
      a.querySelector(".pc-recipe").append(" ", b);
    }
    grid.appendChild(a);
  }
  renderJobs(data.jobs || []);
}

function renderJobs(jobs) {
  const box = $("jobs");
  box.innerHTML = "";
  let running = false;
  for (const j of jobs.slice().reverse()) {
    if (j.state === "done" && !watching.has(j.id)) continue;
    const row = document.createElement("div");
    row.className = "job-row" + (j.state === "failed" ? " failed" : "");
    if (j.state === "running") {
      running = true;
      const secs = Math.round(Date.now() / 1000 - j.started);
      row.innerHTML = `<span class="spin"></span><span></span>`;
      row.lastChild.textContent = `building ${j.piece}… ${secs}s (you'll get a ping)`;
    } else if (j.state === "done") {
      row.innerHTML = `✓ <a></a>`;
      row.querySelector("a").href = pieceUrl(j.piece);
      row.querySelector("a").textContent = `${j.piece} ready — open`;
    } else {
      row.innerHTML = `<div><b>failed:</b> <span></span><div class="job-err"></div></div>`;
      row.querySelector("span").textContent = j.piece;
      row.querySelector(".job-err").textContent = j.error;
    }
    box.appendChild(row);
  }
  clearTimeout(refresh._t);
  refresh._t = setTimeout(refresh, running ? 4000 : 20000);
}

// ---- new piece ----
async function loadRecipes() {
  const { recipes } = await api("/recipes");
  const box = $("recipe-pick");
  box.innerHTML = "";
  if (!recipes.length) box.innerHTML = '<div class="empty-hint">No recipes — run deconstruct.py first.</div>';
  for (const r of recipes) {
    const b = document.createElement("button");
    b.className = "recipe-opt";
    b.title = r.summary;
    b.innerHTML = (r.breakdown ? `<img loading="lazy" src="${API}/recipes/${encodeURIComponent(r.name)}/breakdown?w=360">` : '<img alt="">')
      + `<span></span>`;
    b.querySelector("span").textContent = `${r.name} · ${r.layers} layers`;
    b.onclick = () => {
      sel.recipe = r.name;
      box.querySelectorAll(".recipe-opt").forEach((x) => x.classList.toggle("sel", x === b));
      updateSummary();
    };
    box.appendChild(b);
    if (recipes.length === 1) b.click();
  }
}

async function loadPhotos(src) {
  const grid = $("photo-grid");
  grid.innerHTML = '<div class="empty-hint">loading…</div>';
  try {
    const { photos } = await api(`/photos?src=${src}`);
    grid.innerHTML = photos.length ? "" : '<div class="empty-hint">Nothing here.</div>';
    for (const p of photos) {
      const img = document.createElement("img");
      img.loading = "lazy";
      img.src = thumbUrl(p, 200);
      img.title = baseName(p);
      if (p === sel.photo) img.classList.add("sel");
      img.onclick = () => {
        sel.picked = p;
        $("photo-path").value = "";
        grid.querySelectorAll("img").forEach((x) => x.classList.toggle("sel", x === img));
        updateSummary();
      };
      grid.appendChild(img);
    }
  } catch (e) {
    grid.innerHTML = `<div class="empty-hint">${e.message}</div>`;
  }
}

function updateSummary() {
  sel.photo = $("photo-path").value.trim() || sel.picked;
  $("sel-summary").textContent = sel.recipe && sel.photo
    ? `${baseName(sel.photo)} × ${sel.recipe}`
    : "pick a technique + photo";
  $("start-btn").disabled = !(sel.recipe && sel.photo);
}

async function start() {
  const btn = $("start-btn");
  btn.disabled = true;
  const crop = $("crop").value.trim().replace(/\s+/g, "");
  try {
    const job = await api("/init", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ photo: sel.photo, recipe: sel.recipe,
                             wildness: parseFloat($("wildness").value),
                             custom: crop || null }),
    });
    watching.add(job.id);
    toast("Building — ~1 min. Stay or leave, you'll get a ping.");
    $("new-box").classList.add("hidden");
    refresh();
  } catch (e) {
    if (/piece exists/.test(e.message)) {
      const name = e.message.split("piece exists: ").pop();
      toast("Already exists — opening it");
      setTimeout(() => { location.href = pieceUrl(name); }, 800);
    } else {
      toast(`Error: ${e.message}`, 6000);
    }
  } finally {
    btn.disabled = false;
  }
}

function init() {
  $("new-btn").onclick = () => {
    const box = $("new-box");
    box.classList.toggle("hidden");
    if (!box.dataset.loaded) {
      box.dataset.loaded = "1";
      loadRecipes().catch((e) => toast(e.message));
      loadPhotos("candidates");
    }
  };
  $("src-tabs").addEventListener("click", (e) => {
    const b = e.target.closest("[data-src]");
    if (!b) return;
    $("src-tabs").querySelectorAll(".mode-btn").forEach((x) => x.classList.toggle("active", x === b));
    loadPhotos(b.dataset.src);
  });
  $("photo-path").addEventListener("input", updateSummary);
  $("wildness").addEventListener("input", (e) => { $("wild-val").textContent = e.target.value; });
  $("start-btn").onclick = start;
  refresh();
  if ($("photo-path").value.trim()) { $("new-btn").click(); updateSummary(); }
}

init();
