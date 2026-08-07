// variants.js — canvas-overlay left/right arrows + swipe-to-flip for variant groups.
// Rendering of the variant STACK badge/list in the steps strip lives in steps.js;
// this module owns the on-canvas flip affordance (arrows + swipe + "vN/M" chip).

let prevBtn, nextBtn, chipEl;
let currentGroup = null; // ordered array of sibling node ids, or null
let currentIndex = -1;
let onPick = null;

export function initVariantNav({ canvasWrapEl, canvasModule, onFlip }) {
  onPick = onFlip;

  prevBtn = document.createElement("button");
  prevBtn.type = "button";
  prevBtn.className = "variant-arrow variant-arrow-left hidden";
  prevBtn.textContent = "‹";
  prevBtn.title = "Previous variant";
  prevBtn.addEventListener("click", () => flip(-1));

  nextBtn = document.createElement("button");
  nextBtn.type = "button";
  nextBtn.className = "variant-arrow variant-arrow-right hidden";
  nextBtn.textContent = "›";
  nextBtn.title = "Next variant";
  nextBtn.addEventListener("click", () => flip(1));

  chipEl = document.createElement("div");
  chipEl.className = "variant-chip hidden";

  canvasWrapEl.appendChild(prevBtn);
  canvasWrapEl.appendChild(nextBtn);
  canvasWrapEl.appendChild(chipEl);

  canvasModule.on("swipe", ({ direction }) => flip(direction === "next" ? 1 : -1));
}

function flip(dir) {
  if (!currentGroup || currentIndex < 0) return;
  const newIndex = currentIndex + dir;
  if (newIndex < 0 || newIndex >= currentGroup.length) return;
  onPick && onPick(currentGroup[newIndex]);
}

// Call after every session refresh / step selection with the node id currently shown
// on the canvas, and the session's variants map ({chainNodeId: [siblingId, ...]}).
export function updateVariantNav(currentNodeId, variants) {
  variants = variants || {};
  let group = null;
  for (const key of Object.keys(variants)) {
    const g = variants[key];
    if (g.includes(currentNodeId)) { group = g; break; }
  }
  if (!group || group.length < 2) {
    currentGroup = null;
    currentIndex = -1;
    prevBtn.classList.add("hidden");
    nextBtn.classList.add("hidden");
    chipEl.classList.add("hidden");
    return;
  }
  currentGroup = group;
  currentIndex = group.indexOf(currentNodeId);
  prevBtn.classList.toggle("hidden", currentIndex <= 0);
  nextBtn.classList.toggle("hidden", currentIndex >= group.length - 1);
  chipEl.textContent = `v${currentIndex + 1}/${group.length}`;
  chipEl.classList.remove("hidden");
}
