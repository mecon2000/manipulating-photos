// compose_layers.js — client-side compositor for layers/manifest.json.
// Draws enabled layers bottom→top with per-layer blend; supports per-layer
// source overrides (variant previews) and a compare mode against a savepoint.

const imgCache = new Map(); // url -> Promise<HTMLImageElement>
const out = document.createElement("canvas");
const ctx = out.getContext("2d");

const BLEND = { normal: "source-over", multiply: "multiply", overlay: "overlay",
                screen: "screen", "soft-light": "soft-light" };

export function loadImg(url) {
  if (!imgCache.has(url)) {
    imgCache.set(url, new Promise((resolve, reject) => {
      const im = new Image();
      im.onload = () => resolve(im);
      im.onerror = () => { imgCache.delete(url); reject(new Error(`load ${url}`)); };
      im.src = url;
    }));
  }
  return imgCache.get(url);
}

// layers: manifest-shaped rows in bottom→top order, each with .url (resolved src).
// compare: null | {img, mode: "hold"|"side", showing: bool}
//   hold: the canvas shows the savepoint only while the hold button/space is down
export async function composite(W, H, layers, compare) {
  if (compare && compare.mode === "hold" && compare.showing && compare.img) {
    if (out.width !== W || out.height !== H) { out.width = W; out.height = H; }
    ctx.globalCompositeOperation = "source-over";
    ctx.drawImage(compare.img, 0, 0, W, H);
    return out;
  }
  const imgs = await Promise.all(layers.map((L) =>
    L.enabled && !L.removed ? loadImg(L.url).catch(() => null) : Promise.resolve(null)));
  const side = compare && compare.mode === "side";
  const cw = side ? W * 2 + Math.round(W * 0.02) : W;
  if (out.width !== cw || out.height !== H) { out.width = cw; out.height = H; }
  ctx.globalCompositeOperation = "source-over";
  ctx.clearRect(0, 0, cw, H);
  const x0 = side ? cw - W : 0;
  ctx.save();
  ctx.beginPath();
  ctx.rect(x0, 0, W, H);
  ctx.clip();
  ctx.fillStyle = "#fff";
  ctx.fillRect(x0, 0, W, H);
  layers.forEach((L, i) => {
    const im = imgs[i];
    if (!im) return;
    ctx.globalCompositeOperation = BLEND[L.blend] || "source-over";
    ctx.drawImage(im, x0, 0, W, H);
  });
  ctx.restore();
  ctx.globalCompositeOperation = "source-over";
  if (compare && compare.img) {
    if (side) {
      ctx.fillStyle = "#111";
      ctx.fillRect(W, 0, cw - 2 * W, H);
      ctx.drawImage(compare.img, 0, 0, W, H);
    }
  }
  return out;
}
