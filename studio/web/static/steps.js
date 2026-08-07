// steps.js — render the horizontal chain-of-nodes strip.

export function renderSteps(container, chain, nodes, activeId, onSelect) {
  container.innerHTML = "";
  chain.forEach((nodeId, i) => {
    const node = nodes[nodeId];
    if (!node) return;
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "step-chip" + (nodeId === activeId ? " active" : "");
    chip.textContent = `${i + 1} ${node.tool}`;
    chip.addEventListener("click", () => onSelect(nodeId));
    container.appendChild(chip);
  });
}
