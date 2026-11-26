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
  wireMaskEdit();
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

// Show an in-memory canvas/image (Compose composites layers client-side).
// Same size → swap pixels only, keeping the user's zoom/pan; new size → refit.
export function setImageSource(el) {
  const w = el.width, h = el.height;
  if (konvaImage && w === natW && h === natH) {
    konvaImage.image(el);
    imgLayer.batchDraw();
    return;
  }
  imgEl = el;
  natW = w;
  natH = h;
  if (konvaImage) konvaImage.destroy();
  konvaImage = new Konva.Image({ image: el, x: 0, y: 0, width: natW, height: natH });
  imgLayer.add(konvaImage);
  refit();
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
  if (maskNode) { maskNode.position({ x: fitX, y: fitY }); maskNode.scale({ x: fitScale, y: fitScale }); }
  if (shapePts) drawShape();
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

  // Two fingers = pinch-zoom AND pan (midpoint drift), one finger = Konva
  // drag-pan. Pinch runs on NATIVE container listeners, not Konva events —
  // with a draggable stage Konva's drag machinery swallows/reorders touch
  // events unpredictably, which is why pinch kept dying in Pan mode.
  let pinchActive = false;

  function touchPair(touches) {
    const [t1, t2] = touches;
    return {
      dist: Math.hypot(t2.clientX - t1.clientX, t2.clientY - t1.clientY),
      center: { x: (t1.clientX + t2.clientX) / 2,
                y: (t1.clientY + t2.clientY) / 2 },
    };
  }

  container.addEventListener("touchstart", (e) => {
    if (e.touches.length < 2) return;
    pinchActive = true;
    if (stage.isDragging()) stage.stopDrag();
    stage.draggable(false);
    const { dist, center } = touchPair(e.touches);
    lastDist = dist;
    lastCenter = center;
  }, { passive: true });

  container.addEventListener("touchmove", (e) => {
    if (!pinchActive || e.touches.length !== 2) return;
    e.preventDefault();
    const { dist, center } = touchPair(e.touches);
    if (lastDist && lastCenter) {
      const oldScale = stage.scaleX();
      const newScale = clamp(oldScale * (dist / lastDist), 0.3, 8);
      const rect = container.getBoundingClientRect();
      const localCenter = { x: center.x - rect.left, y: center.y - rect.top };
      const lastLocal = { x: lastCenter.x - rect.left, y: lastCenter.y - rect.top };
      // anchor on where the midpoint WAS: scaling stays centered between the
      // fingers, and midpoint drift translates the stage (two-finger pan)
      const pointTo = {
        x: (lastLocal.x - stage.x()) / oldScale,
        y: (lastLocal.y - stage.y()) / oldScale,
      };
      stage.scale({ x: newScale, y: newScale });
      stage.position({
        x: localCenter.x - pointTo.x * newScale,
        y: localCenter.y - pointTo.y * newScale,
      });
      stage.batchDraw();
      rescaleMarkers();
    }
    lastDist = dist;
    lastCenter = center;
  }, { passive: false });

  function endPinch(e) {
    if (!pinchActive || e.touches.length >= 2) return;
    pinchActive = false;
    lastDist = null;
    lastCenter = null;
    stage.draggable(mode === "pan");
  }
  container.addEventListener("touchend", endPinch, { passive: true });
  container.addEventListener("touchcancel", endPinch, { passive: true });

  // Double-tap (Pan mode): reset zoom/pan if transformed, else zoom in 2.5×
  // at the tap point. Other modes keep their tap semantics untouched.
  stage.on("dbltap dblclick", () => {
    if (mode !== "pan") return;
    const transformed = Math.abs(stage.scaleX() - 1) > 0.01 ||
      Math.abs(stage.x()) > 1 || Math.abs(stage.y()) > 1;
    if (transformed) {
      stage.scale({ x: 1, y: 1 });
      stage.position({ x: 0, y: 0 });
    } else {
      const pointer = stage.getPointerPosition() ||
        { x: stage.width() / 2, y: stage.height() / 2 };
      const z = 2.5;
      stage.scale({ x: z, y: z });
      stage.position({ x: pointer.x - pointer.x * z,
                       y: pointer.y - pointer.y * z });
    }
    stage.batchDraw();
    rescaleMarkers();
  });
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


// ---- Mask edit (Compose: paint where a layer shows) ----
// The mask lives in its own canvas (alpha = mask) at the picture's pixel size. It is shown
// as a red overlay, edited with a round brush (paint / erase), and a circle the size of the
// brush follows the pointer. "maskChanged" fires (throttled) so the page can preview live.
let maskCanvas = null, maskTint = null, maskNode = null, cursorNode = null;
let maskTool = "paint", maskShowTint = true, maskLast = null, maskTimer = null;
let maskColor = "rgba(255, 59, 48, 0.42)", maskFeather = 0.5, stampCanvas = null, stampKey = "";
let innerNode = null;
const MASK_COLORS = { red: "rgba(255, 59, 48, 0.42)", green: "rgba(52, 199, 89, 0.42)",
                      blue: "rgba(10, 132, 255, 0.45)" };

function retint() {
  if (!maskCanvas) return;
  const t = maskTint.getContext("2d");
  t.globalCompositeOperation = "source-over";
  t.clearRect(0, 0, maskTint.width, maskTint.height);
  if (maskShowTint) {
    t.drawImage(maskCanvas, 0, 0);
    t.globalCompositeOperation = "source-in";
    t.fillStyle = maskColor;
    t.fillRect(0, 0, maskTint.width, maskTint.height);
  }
  maskNode.getLayer() && maskNode.getLayer().batchDraw();
}

function maskChangedSoon() {
  if (maskTimer) return;
  maskTimer = setTimeout(() => { maskTimer = null; emit("maskChanged"); }, 90);
}

// Soft round brush: a radial-gradient dab (full strength inside (1-feather)·r, fading to 0
// at r), stamped along the stroke every ~r/5 — paint adds, erase removes, like Photoshop.
function stamp() {
  const r = Math.max(1, brushPxRadius());
  const key = `${r.toFixed(1)}|${maskFeather}`;
  if (stampKey === key && stampCanvas) return stampCanvas;
  const d = Math.ceil(r * 2) + 2;
  stampCanvas = document.createElement("canvas");
  stampCanvas.width = stampCanvas.height = d;
  const c = stampCanvas.getContext("2d");
  const g = c.createRadialGradient(d / 2, d / 2, 0, d / 2, d / 2, r);
  const hard = Math.max(0, Math.min(0.999, 1 - maskFeather));
  g.addColorStop(0, "rgba(255,255,255,1)");
  g.addColorStop(hard, "rgba(255,255,255,1)");
  g.addColorStop(1, "rgba(255,255,255,0)");
  c.fillStyle = g;
  c.fillRect(0, 0, d, d);
  stampKey = key;
  return stampCanvas;
}

function maskDot(a, b) {
  const c = maskCanvas.getContext("2d");
  c.globalCompositeOperation = maskTool === "erase" ? "destination-out" : "source-over";
  const st = stamp(), h = st.width / 2;
  const r = Math.max(1, brushPxRadius());
  const dist = Math.hypot(b.x - a.x, b.y - a.y);
  const step = Math.max(1, r * (0.12 + 0.15 * (1 - maskFeather)));
  const n = Math.max(1, Math.ceil(dist / step));
  // soft brushes build up: lower per-dab strength so one pass ≈ one Photoshop stroke
  c.globalAlpha = maskFeather > 0.05 ? 0.35 + 0.65 * (1 - maskFeather) : 1;
  for (let i = (a === b ? 0 : 1); i <= n; i++) {
    const t = i / n;
    c.drawImage(st, a.x + (b.x - a.x) * t - h, a.y + (b.y - a.y) * t - h);
  }
  c.globalAlpha = 1;
}

function moveCursor() {
  if (!cursorNode) return;
  const pos = stage.getRelativePointerPosition();
  if (!pos) return;
  cursorNode.position(pos);
  cursorNode.radius(brushPxRadius() * fitScale);
  cursorNode.strokeWidth(1.5 / stage.scaleX());
  cursorNode.visible(true);
  if (innerNode) {               // where the brush is at full strength (hardness)
    innerNode.position(pos);
    innerNode.radius(Math.max(0.5, brushPxRadius() * fitScale * (1 - maskFeather)));
    innerNode.strokeWidth(1 / stage.scaleX());
    innerNode.visible(maskFeather > 0.05);
  }
  overlayLayer.batchDraw();
}

function wireMaskEdit() {
  stage.on("mousedown touchstart", (e) => {
    if (mode !== "mask" || !maskCanvas) return;
    if (e.evt.touches && e.evt.touches.length > 1) return;     // two fingers = zoom, not paint
    maskLast = toImageCoords(stage.getRelativePointerPosition());
    maskDot(maskLast, maskLast);
    retint(); maskChangedSoon(); moveCursor();
  });
  stage.on("mousemove touchmove", (e) => {
    if (mode !== "mask" || !maskCanvas) return;
    moveCursor();
    if (!maskLast || (e.evt.touches && e.evt.touches.length > 1)) return;
    const p = toImageCoords(stage.getRelativePointerPosition());
    maskDot(maskLast, p);
    maskLast = p;
    retint(); maskChangedSoon();
  });
  stage.on("mouseup touchend mouseleave", () => {
    if (mode !== "mask") return;
    if (maskLast) { maskLast = null; emit("maskChanged"); }
  });
  stage.on("mouseleave", () => {
    if (cursorNode) { cursorNode.visible(false); if (innerNode) innerNode.visible(false); overlayLayer.batchDraw(); }
  });
}

export function startMaskEdit(initial) {
  maskCanvas = document.createElement("canvas");
  maskCanvas.width = natW; maskCanvas.height = natH;
  if (initial) {                       // grey PNG (white = show) -> alpha
    const c = maskCanvas.getContext("2d");
    c.drawImage(initial, 0, 0, natW, natH);
    const d = c.getImageData(0, 0, natW, natH);
    for (let i = 0; i < d.data.length; i += 4) {
      const v = d.data[i];
      d.data[i] = d.data[i + 1] = d.data[i + 2] = 255;
      d.data[i + 3] = v;
    }
    c.putImageData(d, 0, 0);
  }
  maskTint = document.createElement("canvas");
  maskTint.width = natW; maskTint.height = natH;
  maskNode = new Konva.Image({ image: maskTint, x: fitX, y: fitY, scaleX: fitScale, scaleY: fitScale, listening: false });
  overlayLayer.add(maskNode);
  cursorNode = new Konva.Circle({ radius: 10, stroke: "#fff", strokeWidth: 1.5, dash: [4, 3],
                                  shadowColor: "#000", shadowBlur: 2, shadowOpacity: 0.9,
                                  listening: false, visible: false });
  overlayLayer.add(cursorNode);
  innerNode = new Konva.Circle({ radius: 5, stroke: "rgba(255,255,255,0.55)", strokeWidth: 1,
                                 listening: false, visible: false });
  overlayLayer.add(innerNode);
  container.style.cursor = "none";
  maskTool = "paint"; maskShowTint = true;
  setMode("mask");
  retint();
}
export function setMaskTool(t) { maskTool = t; }
export function setMaskColor(name) { maskColor = MASK_COLORS[name] || MASK_COLORS.red; retint(); }
export function setMaskFeather(f) { maskFeather = Math.max(0, Math.min(1, f)); }
export function setMaskTint(on) { maskShowTint = on; retint(); }
export function clearMask() {
  if (!maskCanvas) return;
  maskCanvas.getContext("2d").clearRect(0, 0, natW, natH);
  retint(); emit("maskChanged");
}
export function fillMask() {
  if (!maskCanvas) return;
  const c = maskCanvas.getContext("2d");
  c.globalCompositeOperation = "source-over"; c.fillStyle = "#fff"; c.fillRect(0, 0, natW, natH);
  retint(); emit("maskChanged");
}
export const getMaskCanvas = () => maskCanvas;
// Grey PNG for the server: white where the layer shows.
export function maskPNG() {
  const o = document.createElement("canvas");
  o.width = natW; o.height = natH;
  const c = o.getContext("2d");
  c.fillStyle = "#000"; c.fillRect(0, 0, natW, natH);
  c.drawImage(maskCanvas, 0, 0);
  return o.toDataURL("image/png");
}
export function endMaskEdit() {
  if (maskNode) maskNode.destroy();
  if (cursorNode) cursorNode.destroy();
  if (innerNode) innerNode.destroy();
  maskNode = cursorNode = innerNode = maskCanvas = maskTint = null;
  container.style.cursor = "";
  overlayLayer.batchDraw();
}


// ---- Shape edit (Compose: free-transform the torn strip) ----
// 6 points in image px: TL, TM, TR, BR, BM, BL. Corners distort (or scale evenly in "scale"
// mode), the two middles bend, ✥ moves all, ↻ rotates all around the centre.
let shapePts = null, shapeNodes = [], shapeLine = null, shapeMode = "distort";
const QUAD = (p0, pm, p1, n = 32) => {
  const c = { x: 2 * pm.x - 0.5 * (p0.x + p1.x), y: 2 * pm.y - 0.5 * (p0.y + p1.y) };
  const out = [];
  for (let i = 0; i <= n; i++) {
    const t = i / n, a = (1 - t) * (1 - t), b = 2 * (1 - t) * t, d = t * t;
    out.push(a * p0.x + b * c.x + d * p1.x, a * p0.y + b * c.y + d * p1.y);
  }
  return out;
};
const centre = () => ({ x: shapePts.reduce((a, p) => a + p.x, 0) / 6, y: shapePts.reduce((a, p) => a + p.y, 0) / 6 });

function drawShape() {
  if (!shapePts) return;
  const P = shapePts.map((p) => toStageCoords(p));
  const top = QUAD(P[0], P[1], P[2]), bot = QUAD(P[3], P[4], P[5]);
  shapeLine.points(top.concat(bot));
  const k = 1 / stage.scaleX();
  shapeLine.strokeWidth(2 * k);
  const c = toStageCoords(centre());
  const up = { x: (P[0].x + P[2].x) / 2 - c.x, y: (P[0].y + P[2].y) / 2 - c.y };
  const L = Math.hypot(up.x, up.y) || 1;
  const rot = { x: c.x + up.x / L * (L + 40 * k), y: c.y + up.y / L * (L + 40 * k) };
  const where = P.concat([c, rot]);
  shapeNodes.forEach((n, i) => { n.position(where[i]); n.scale({ x: k, y: k }); });
  overlayLayer.batchDraw();
}

function iconHandle(glyph) {         // round white button with a symbol: clearer than a coloured dot
  const g = new Konva.Group();
  // a generous invisible hit disc so the whole button (and a margin) grabs, glyph included
  g.add(new Konva.Circle({ radius: 26, fill: "rgba(0,0,0,0.002)" }));
  g.add(new Konva.Circle({ radius: 15, fill: "#fff", stroke: "#000", strokeWidth: 1.5,
                           shadowColor: "#000", shadowBlur: 4, shadowOpacity: 0.6 }));
  g.add(new Konva.Text({ text: glyph, fontSize: 20, fontStyle: "bold", fill: "#111", width: 30, height: 30,
                         offsetX: 15, offsetY: 14, align: "center", verticalAlign: "middle", listening: false }));
  return g;
}

// drag anywhere inside the strip = move it (like Photoshop's free transform)
let bodyDrag = null;
function wireShapeBody() {
  shapeLine.on("mousedown touchstart", (e) => {
    if (e.evt.touches && e.evt.touches.length > 1) return;
    e.cancelBubble = true;
    bodyDrag = toImageCoords(stage.getRelativePointerPosition());
    container.style.cursor = "move";
  });
  stage.on("mousemove.shape touchmove.shape", () => {
    if (!bodyDrag || !shapePts) return;
    const p = toImageCoords(stage.getRelativePointerPosition());
    const dx = p.x - bodyDrag.x, dy = p.y - bodyDrag.y;
    shapePts.forEach((q) => { q.x += dx; q.y += dy; });
    bodyDrag = p;
    drawShape();
  });
  stage.on("mouseup.shape touchend.shape mouseleave.shape", () => {
    if (bodyDrag) { bodyDrag = null; container.style.cursor = ""; }
  });
  shapeLine.on("mouseenter", () => { if (!bodyDrag) container.style.cursor = "move"; });
  shapeLine.on("mouseleave", () => { if (!bodyDrag) container.style.cursor = ""; });
}

export function startShapeEdit(norm) {
  endShapeEdit();
  shapePts = norm.map(([x, y]) => ({ x: x * natW, y: y * natH }));
  shapeLine = new Konva.Line({ points: [], closed: true, stroke: "#fff", dash: [6, 4], listening: true,
                               shadowColor: "#000", shadowBlur: 3, shadowOpacity: 0.9, fill: "rgba(255,255,255,0.08)" });
  overlayLayer.add(shapeLine);
  wireShapeBody();
  const mk = (i) => {
    const kind = i < 6 ? ([1, 4].includes(i) ? "bend" : "corner") : (i === 6 ? "move" : "rotate");
    const n = kind === "corner" ? new Konva.Circle({ radius: 9, fill: "#fff", stroke: "#000", strokeWidth: 1.5 })
      : kind === "bend" ? new Konva.Rect({ width: 14, height: 14, offsetX: 7, offsetY: 7, rotation: 45, fill: "#ffd60a", stroke: "#000", strokeWidth: 1.5 })
      : iconHandle(kind === "move" ? "✥" : "↻");
    n.draggable(true);
    if (kind === "move" || kind === "rotate") n.moveToTop();
    let last = null, c0 = null, a0 = 0, d0 = 1, start = null;
    n.on("dragstart", () => {
      last = toImageCoords(n.position());
      c0 = centre(); start = shapePts.map((p) => ({ ...p }));
      a0 = Math.atan2(last.y - c0.y, last.x - c0.x);
      d0 = Math.hypot(last.x - c0.x, last.y - c0.y) || 1;
    });
    n.on("dragmove", (e) => {
      const p = toImageCoords(n.position());
      if (kind === "move") {
        const dx = p.x - last.x, dy = p.y - last.y;
        shapePts.forEach((q) => { q.x += dx; q.y += dy; });
      } else if (kind === "rotate") {
        const a = Math.atan2(p.y - c0.y, p.x - c0.x) - a0, cs = Math.cos(a), sn = Math.sin(a);
        shapePts = start.map((q) => ({ x: c0.x + (q.x - c0.x) * cs - (q.y - c0.y) * sn,
                                       y: c0.y + (q.x - c0.x) * sn + (q.y - c0.y) * cs }));
      } else if (kind === "corner" && (shapeMode === "scale" || (e.evt && e.evt.shiftKey))) {
        const k = Math.hypot(p.x - c0.x, p.y - c0.y) / d0;
        shapePts = start.map((q) => ({ x: c0.x + (q.x - c0.x) * k, y: c0.y + (q.y - c0.y) * k }));
      } else {
        shapePts[i] = p;
      }
      last = p;
      drawShape();
    });
    n.on("dragend", drawShape);
    n.on("mouseenter", () => { container.style.cursor = kind === "move" ? "move" : "pointer"; });
    n.on("mouseleave", () => { container.style.cursor = ""; });
    overlayLayer.add(n);
    return n;
  };
  shapeNodes = [0, 1, 2, 3, 4, 5, 6, 7].map(mk);
  setMode("shape");
  drawShape();
}
export function setShapeMode(m) { shapeMode = m; }
export function getShape() {
  return shapePts ? shapePts.map((p) => [+(p.x / natW).toFixed(4), +(p.y / natH).toFixed(4)]) : null;
}
export function endShapeEdit() {
  shapeNodes.forEach((n) => n.destroy());
  if (shapeLine) shapeLine.destroy();
  shapeNodes = []; shapeLine = null; shapePts = null; bodyDrag = null;
  if (stage) stage.off(".shape");
  if (overlayLayer) overlayLayer.batchDraw();
}


// ---- Photo-below edit (Compose: nudge the sibling photo under the tear) ----
// The eye-aligned photo shows half-transparent over the picture; Konva's transformer moves,
// rotates and scales it. tf = [dx, dy (fractions of the canvas), rot (deg, clockwise), scale].
let underNode = null, underTr = null;
export function startUnderEdit(img, tf) {
  endUnderEdit();
  const [dx, dy, rot, sc] = tf || [0, 0, 0, 1];
  underNode = new Konva.Image({ image: img, width: natW, height: natH, offsetX: natW / 2, offsetY: natH / 2,
                                opacity: 0.55, draggable: true });
  const c = toStageCoords({ x: natW / 2 + dx * natW, y: natH / 2 + dy * natH });
  underNode.position(c);
  underNode.rotation(rot);
  underNode.scale({ x: sc * fitScale, y: sc * fitScale });
  overlayLayer.add(underNode);
  underTr = new Konva.Transformer({ nodes: [underNode], keepRatio: true, rotateEnabled: true,
                                    enabledAnchors: ["top-left", "top-right", "bottom-left", "bottom-right"],
                                    rotateAnchorOffset: 30, anchorSize: 14, borderStroke: "#fff",
                                    anchorStroke: "#000", anchorFill: "#fff" });
  overlayLayer.add(underTr);
  underNode.on("mouseenter", () => { container.style.cursor = "move"; });
  underNode.on("mouseleave", () => { container.style.cursor = ""; });
  setMode("under");
  overlayLayer.batchDraw();
}
export function setUnderOpacity(o) { if (underNode) { underNode.opacity(o); overlayLayer.batchDraw(); } }
export function getUnderTf() {
  if (!underNode) return null;
  const p = toImageCoords(underNode.position());
  return [+((p.x - natW / 2) / natW).toFixed(4), +((p.y - natH / 2) / natH).toFixed(4),
          +underNode.rotation().toFixed(2), +(underNode.scaleX() / fitScale).toFixed(4)];
}
export function endUnderEdit() {
  if (underTr) underTr.destroy();
  if (underNode) underNode.destroy();
  underTr = underNode = null;
  if (overlayLayer) overlayLayer.batchDraw();
}
