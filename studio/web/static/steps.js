// steps.js — render the horizontal chain-of-nodes strip, incl. variant stacks.

let expandedGroup = null; // chain node id whose variant stack is currently expanded

export function renderSteps(container, chain, nodes, activeId, onSelect, variants, onVariantPick) {
  container.innerHTML = "";
  variants = variants || {};

  // Baseline chip: the untouched source photo, before step 1.
  const srcChip = document.createElement("button");
  srcChip.type = "button";
  srcChip.className = "step-chip step-chip-source" + (activeId === "__source__" ? " active" : "");
  srcChip.textContent = "0 source";
  srcChip.addEventListener("click", () => onSelect("__source__"));
  container.appendChild(srcChip);
  chain.forEach((nodeId, i) => {
    const node = nodes[nodeId];
    if (!node) return;

    const wrap = document.createElement("div");
    wrap.className = "step-chip-wrap";

    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "step-chip" + (nodeId === activeId ? " active" : "");
    chip.textContent = `${i + 1} ${node.tool}`;
    chip.addEventListener("click", () => onSelect(nodeId));
    wrap.appendChild(chip);

    const group = variants[nodeId];
    if (group && group.length > 1) {
      const idx = group.indexOf(nodeId);
      const badge = document.createElement("button");
      badge.type = "button";
      badge.className = "variant-badge";
      badge.title = "Variants of this step";
      badge.textContent = `${idx >= 0 ? idx + 1 : 1}/${group.length}`;
      badge.addEventListener("click", (e) => {
        e.stopPropagation();
        expandedGroup = expandedGroup === nodeId ? null : nodeId;
        renderSteps(container, chain, nodes, activeId, onSelect, variants, onVariantPick);
      });
      wrap.appendChild(badge);

      if (expandedGroup === nodeId) {
        const list = document.createElement("div");
        list.className = "variant-list";
        group.forEach((sibId, si) => {
          const sibNode = nodes[sibId];
          const item = document.createElement("button");
          item.type = "button";
          item.className = "variant-item" + (sibId === nodeId ? " current" : "");
          const seedTxt = sibNode && sibNode.seed != null ? ` · seed ${sibNode.seed}` : "";
          item.textContent = `v${si + 1}${seedTxt}`;
          item.addEventListener("click", () => onVariantPick && onVariantPick(sibId));
          list.appendChild(item);
        });
        wrap.appendChild(list);
      }
    }

    container.appendChild(wrap);
  });
}
