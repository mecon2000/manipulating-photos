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
// compare: null | {img, mode: "wipe"|"side", pos: 0..1}
export async function composite(W, H, layers, compare) {
  const imgs = await Promise.all(layers.map((L) =>
    L.enabled ? loadImg(L.url).catch(() => null) : Promise.resolve(null)));
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
    } else {
      const px = Math.round(W * compare.pos);
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, 0, px, H);
      ctx.clip();
      ctx.drawImage(compare.img, 0, 0, W, H);
      ctx.restore();
      ctx.fillStyle = "#2dd4bf";
      ctx.fillRect(px - Math.max(2, W * 0.002), 0, Math.max(4, W * 0.004), H);
    }
  }
  return out;
}
