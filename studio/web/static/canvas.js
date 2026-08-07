// canvas.js — Konva-based image viewer with pan/zoom, brush mask, markers.

let stage, imgLayer, overlayLayer, selectLayer, markerLayer;
let konvaImage = null;
let imgEl = null;
let natW = 0, natH = 0;
let fitScale = 1, fitX = 0, fitY = 0;
let mode = "pan";
let brushSizePct = 2.5; // % of image width
let strokes = []; // Konva.Line[] in image coords
let currentLine = null;
let selectOverlayNode = null; // Konva.Image, tinted mask
let markers = []; // {n, x, y, label} normalized 0..1
let markerNodes = new Map(); // n -> Konva.Group
let container = null;
let listeners = {};

export function on(event, cb) {
  (listeners[event] = listeners[event] || []).push(cb);
}

function emit(event, payload) {
  (listeners[event] || []).forEach((cb) => cb(payload));
}

export function init(el) {
  container = el;
  const w = el.clientWidth || 300;
  const h = el.clientHeight || 300;
  stage = new Konva.Stage({ container: el, width: w, height: h });
  imgLayer = new Konva.Layer();
  overlayLayer = new Konva.Layer();
  selectLayer = new Konva.Layer();
  markerLayer = new Konva.Layer();
  stage.add(imgLayer);
  stage.add(overlayLayer);
  stage.add(selectLayer);
  stage.add(markerLayer);

  stage.draggable(false);

  wireZoomPan();
  wireBrush();
  wireSelect();
  wireMarkers();
  wireSwipe();

  window.addEventListener("resize", refit);
}

// ---- Swipe (variant flip) ----
// Horizontal-swipe detection on the raw container, independent of pan/zoom mode —
// listeners decide whether to act on the emitted event (e.g. only when a variant
// group is active for the currently shown node).
function wireSwipe() {
  let startX = 0, startY = 0, startTime = 0;
  container.addEventListener("touchstart", (e) => {
    if (e.touches.length !== 1) return;
    startX = e.touches[0].clientX;
    startY = e.touches[0].clientY;
    startTime = Date.now();
  }, { passive: true });
  container.addEventListener("touchend", (e) => {
    if (!startTime) return;
    const dt = Date.now() - startTime;
    startTime = 0;
    if (!e.changedTouches || !e.changedTouches.length) return;
    const dx = e.changedTouches[0].clientX - startX;
    const dy = e.changedTouches[0].clientY - startY;
    if (dt < 600 && Math.abs(dx) > 40 && Math.abs(dx) > Math.abs(dy) * 1.5) {
      emit("swipe", { direction: dx < 0 ? "next" : "prev" });
    }
  }, { passive: true });
}

function stageContentGroupScale() {
  // Konva stage itself is scaled/positioned for pan+zoom (whole-content transform).
  return stage.scaleX();
}

function toImageCoords(pointerPos) {
  // pointerPos is stage-local (already accounts for stage scale/pos via getRelativePointerPosition)
  const x = (pointerPos.x - fitX) / fitScale;
  const y = (pointerPos.y - fitY) / fitScale;
  return { x, y };
}

function toStageCoords(imgPt) {
  return { x: imgPt.x * fitScale + fitX, y: imgPt.y * fitScale + fitY };
}

export async function loadImage(refOrUrl) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      imgEl = img;
      natW = img.naturalWidth;
      natH = img.naturalHeight;
      if (konvaImage) konvaImage.destroy();
      konvaImage = new Konva.Image({ image: img, x: 0, y: 0, width: natW, height: natH });
      imgLayer.add(konvaImage);
      refit();
      resolve();
    };
    img.onerror = reject;
    img.src = refOrUrl;
  });
}

function refit() {
  if (!container) return;
  const w = container.clientWidth || stage.width();
  const h = container.clientHeight || stage.height();
  stage.width(w);
  stage.height(h);
  if (!natW || !natH) return;
  const scale = Math.min(w / natW, h / natH);
  fitScale = scale;
  fitX = (w - natW * scale) / 2;
  fitY = (h - natH * scale) / 2;
  if (konvaImage) {
    konvaImage.width(natW);
    konvaImage.height(natH);
  }
  // Reset stage transform baseline to identity; content positioned via fit values above,
  // stage scale/pos is used only for user pan/zoom on top of that.
  applyBaseTransform();
  rescaleMarkers();
}

// Base transform bakes fit into layer positions; stage scale/pos handle user zoom/pan.
function applyBaseTransform() {
  [imgLayer, overlayLayer, selectLayer, markerLayer].forEach((layer) => {
    layer.offsetX(0);
    layer.offsetY(0);
  });
  if (konvaImage) {
    konvaImage.position({ x: fitX, y: fitY });
    konvaImage.size({ width: natW * fitScale, height: natH * fitScale });
  }
  if (selectOverlayNode) {
    selectOverlayNode.position({ x: fitX, y: fitY });
    selectOverlayNode.size({ width: natW * fitScale, height: natH * fitScale });
  }
  imgLayer.batchDraw();
  selectLayer.batchDraw();
}

function wireZoomPan() {
  let lastDist = null;
  let lastCenter = null;

  stage.on("wheel", (e) => {
    e.evt.preventDefault();
    const scaleBy = 1.05;
    const oldScale = stage.scaleX();
    const pointer = stage.getPointerPosition();
    if (!pointer) return;
    const mousePointTo = {
      x: (pointer.x - stage.x()) / oldScale,
      y: (pointer.y - stage.y()) / oldScale,
    };
    const direction = e.evt.deltaY > 0 ? -1 : 1;
    const newScale = clamp(direction > 0 ? oldScale * scaleBy : oldScale / scaleBy, 0.3, 8);
    stage.scale({ x: newScale, y: newScale });
    stage.position({
      x: pointer.x - mousePointTo.x * newScale,
      y: pointer.y - mousePointTo.y * newScale,
    });
    stage.batchDraw();
    rescaleMarkers();
  });

  stage.on("touchmove", (e) => {
    const touches = e.evt.touches;
    if (touches.length !== 2) return;
    e.evt.preventDefault();
    const [t1, t2] = touches;
    const p1 = { x: t1.clientX, y: t1.clientY };
    const p2 = { x: t2.clientX, y: t2.clientY };
    const dist = Math.hypot(p2.x - p1.x, p2.y - p1.y);
    const center = { x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2 };
    if (lastDist) {
      const oldScale = stage.scaleX();
      const newScale = clamp(oldScale * (dist / lastDist), 0.3, 8);
      const rect = stage.container().getBoundingClientRect();
      const localCenter = { x: center.x - rect.left, y: center.y - rect.top };
      const mousePointTo = {
        x: (localCenter.x - stage.x()) / oldScale,
        y: (localCenter.y - stage.y()) / oldScale,
      };
      stage.scale({ x: newScale, y: newScale });
      stage.position({
        x: localCenter.x - mousePointTo.x * newScale,
        y: localCenter.y - mousePointTo.y * newScale,
      });
      stage.batchDraw();
      rescaleMarkers();
    }
    lastDist = dist;
    lastCenter = center;
  });
  stage.on("touchend", () => { lastDist = null; lastCenter = null; });
}

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

export function setMode(m) {
  mode = m;
  stage.draggable(mode === "pan");
}

// ---- Brush ----
function brushPxRadius() {
  return (brushSizePct / 100) * natW / 2;
}

export function setBrushSizePct(pct) {
  brushSizePct = pct;
}

function wireBrush() {
  let drawing = false;

  stage.on("mousedown touchstart", (e) => {
    if (mode !== "brush") return;
    drawing = true;
    const pos = stage.getRelativePointerPosition();
    const imgPt = toImageCoords(pos);
    currentLine = new Konva.Line({
      points: [imgPt.x, imgPt.y],
      stroke: "#ff3b30",
      strokeWidth: brushPxRadius() * 2,
      opacity: 0.45,
      lineCap: "round",
      lineJoin: "round",
      globalCompositeOperation: "source-over",
      x: fitX, y: fitY, scaleX: fitScale, scaleY: fitScale,
    });
    overlayLayer.add(currentLine);
    strokes.push(currentLine);
  });

  stage.on("mousemove touchmove", () => {
    if (mode !== "brush" || !drawing || !currentLine) return;
    const pos = stage.getRelativePointerPosition();
    const imgPt = toImageCoords(pos);
    const newPoints = currentLine.points().concat([imgPt.x, imgPt.y]);
    currentLine.points(newPoints);
    overlayLayer.batchDraw();
  });

  stage.on("mouseup touchend", () => {
    drawing = false;
    currentLine = null;
  });
}

export function clearBrush() {
  strokes.forEach((s) => s.destroy());
  strokes = [];
  overlayLayer.batchDraw();
}

export function getBrushMaskDataURL() {
  if (!natW || !natH) return null;
  const off = document.createElement("canvas");
  off.width = natW;
  off.height = natH;
  const ctx = off.getContext("2d");
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, natW, natH);
  ctx.strokeStyle = "white";
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.globalAlpha = 1.0;
  strokes.forEach((line) => {
    const pts = line.points();
    if (pts.length < 4) return;
    ctx.lineWidth = line.strokeWidth();
    ctx.beginPath();
    ctx.moveTo(pts[0], pts[1]);
    for (let i = 2; i < pts.length; i += 2) ctx.lineTo(pts[i], pts[i + 1]);
    ctx.stroke();
  });
  return off.toDataURL("image/png");
}

// ---- Select (tap-to-mask) ----
function wireSelect() {
  stage.on("click tap", (e) => {
    if (mode !== "select") return;
    if (e.target !== stage && e.target.getLayer() === markerLayer) return;
    const pos = stage.getRelativePointerPosition();
    const imgPt = toImageCoords(pos);
    if (imgPt.x < 0 || imgPt.y < 0 || imgPt.x > natW || imgPt.y > natH) return;
    const shift = !!(e.evt && e.evt.shiftKey);
    emit("selectPoint", { x: imgPt.x / natW, y: imgPt.y / natH, shift });
  });
}

// Loads a white-on-black L-mode mask PNG and tints it teal, translucent-over-transparent.
function tintMaskImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      const off = document.createElement("canvas");
      off.width = img.naturalWidth;
      off.height = img.naturalHeight;
      const ctx = off.getContext("2d");
      ctx.drawImage(img, 0, 0);
      const data = ctx.getImageData(0, 0, off.width, off.height);
      const px = data.data;
      for (let i = 0; i < px.length; i += 4) {
        const v = px[i]; // grayscale mask stored in R (=G=B)
        px[i] = 45; px[i + 1] = 212; px[i + 2] = 191; // teal
        px[i + 3] = Math.round((v / 255) * 0.4 * 255);
      }
      ctx.putImageData(data, 0, 0);
      resolve(off);
    };
    img.onerror = reject;
    img.src = url;
  });
}

export async function renderMaskOverlay(url) {
  const tinted = await tintMaskImage(url);
  if (selectOverlayNode) selectOverlayNode.destroy();
  selectOverlayNode = new Konva.Image({
    image: tinted,
    x: fitX, y: fitY,
    width: natW * fitScale, height: natH * fitScale,
    listening: false,
  });
  selectLayer.add(selectOverlayNode);
  selectLayer.batchDraw();
}

export function clearSelectOverlay() {
  if (selectOverlayNode) {
    selectOverlayNode.destroy();
    selectOverlayNode = null;
    selectLayer.batchDraw();
  }
}

// ---- Markers ----
function wireMarkers() {
  stage.on("click tap", (e) => {
    if (mode !== "markers") return;
    // ignore clicks that landed on an existing marker (handled by its own handler)
    if (e.target !== stage && e.target.getLayer() === markerLayer) return;
    const pos = stage.getRelativePointerPosition();
    const imgPt = toImageCoords(pos);
    if (imgPt.x < 0 || imgPt.y < 0 || imgPt.x > natW || imgPt.y > natH) return;
    const n = nextMarkerNumber();
    const marker = { n, x: imgPt.x / natW, y: imgPt.y / natH };
    markers.push(marker);
    drawMarker(marker);
    emitMarkersChanged();
    emit("markerAdded", marker);
  });
}

function nextMarkerNumber() {
  let n = 1;
  const used = new Set(markers.map((m) => m.n));
  while (used.has(n)) n++;
  return n;
}

function markerVisualScale() {
  return 1 / stage.scaleX();
}

function drawMarker(marker) {
  const group = new Konva.Group({ draggable: true });
  const circle = new Konva.Circle({ radius: 16, fill: "#2dd4bf", stroke: "#06201c", strokeWidth: 2 });
  const text = new Konva.Text({
    text: String(marker.n),
    fontSize: 16,
    fontStyle: "bold",
    fill: "#06201c",
    width: 32,
    height: 32,
    offsetX: 16,
    offsetY: 16,
    align: "center",
    verticalAlign: "middle",
  });
  const labelText = new Konva.Text({
    text: marker.label || "",
    fontSize: 11,
    fill: "#eafff9",
    width: 140,
    offsetX: 70,
    y: 20,
    align: "center",
    visible: !!marker.label,
    shadowColor: "#06201c",
    shadowBlur: 3,
    shadowOpacity: 0.9,
  });
  group.add(circle);
  group.add(text);
  group.add(labelText);
  group.setAttr("labelNode", labelText);
  const stagePt = toStageCoordsUnscaled(marker);
  group.position(stagePt);
  group.scale({ x: markerVisualScale(), y: markerVisualScale() });

  group.on("dragend", () => {
    const localPt = { x: group.x(), y: group.y() };
    const imgPt = toImageCoords(localPt);
    marker.x = clamp(imgPt.x / natW, 0, 1);
    marker.y = clamp(imgPt.y / natH, 0, 1);
    emitMarkersChanged();
  });

  group.on("click tap", (e) => {
    if (mode !== "markers") return;
    e.cancelBubble = true;
    markers = markers.filter((m) => m.n !== marker.n);
    group.destroy();
    markerNodes.delete(marker.n);
    markerLayer.batchDraw();
    emitMarkersChanged();
  });

  markerLayer.add(group);
  markerNodes.set(marker.n, group);
  markerLayer.batchDraw();
}

function toStageCoordsUnscaled(marker) {
  // position relative to stage's own coordinate system (pre stage-scale), matching layer content
  return { x: marker.x * natW * fitScale + fitX, y: marker.y * natH * fitScale + fitY };
}

function rescaleMarkers() {
  const s = markerVisualScale();
  markerNodes.forEach((group) => group.scale({ x: s, y: s }));
  markerLayer.batchDraw();
}

function redrawAllMarkers() {
  markerNodes.forEach((group) => group.destroy());
  markerNodes.clear();
  markers.forEach(drawMarker);
}

function emitMarkersChanged() {
  emit("markersChanged", getMarkers());
}

export function getMarkers() {
  return markers.map((m) => ({ n: m.n, x: m.x, y: m.y, label: m.label }));
}

export function setMarkers(list) {
  markers = (list || []).map((m) => ({ n: m.n, x: m.x, y: m.y, label: m.label }));
  redrawAllMarkers();
}

export function setMarkerLabel(n, label) {
  const m = markers.find((mm) => mm.n === n);
  if (m) m.label = label;
  const group = markerNodes.get(n);
  if (!group) return;
  const labelNode = group.getAttr("labelNode");
  if (!labelNode) return;
  labelNode.text(label || "");
  labelNode.visible(!!label);
  markerLayer.batchDraw();
}
