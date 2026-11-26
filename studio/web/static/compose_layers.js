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
export const isLoaded = (url) => loadedUrls.has(url);
const loadedUrls = new Set();

const tmp = document.createElement("canvas");
const tctx = tmp.getContext("2d");

export async function composite(W, H, layers, compare, onProgress) {
  if (compare && compare.mode === "hold" && compare.showing && compare.img) {
    if (out.width !== W || out.height !== H) { out.width = W; out.height = H; }
    ctx.globalCompositeOperation = "source-over";
    ctx.drawImage(compare.img, 0, 0, W, H);
    return out;
  }
  const want = layers.filter((L) => L.enabled && !L.removed);
  let done = want.filter((L) => loadedUrls.has(L.url)).length;
  if (onProgress && done < want.length) onProgress(done, want.length);
  const imgs = await Promise.all(layers.map((L) => {
    if (!(L.enabled && !L.removed)) return Promise.resolve(null);
    const had = loadedUrls.has(L.url);
    return loadImg(L.url).then((im) => {
      if (!had) { loadedUrls.add(L.url); done++; if (onProgress) onProgress(done, want.length); }
      return im;
    }).catch(() => null);
  }));
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
    let src = im;
    if (L.liveMask) {               // mask being painted: clip the unmasked layer in the browser
      if (tmp.width !== W || tmp.height !== H) { tmp.width = W; tmp.height = H; }
      tctx.globalCompositeOperation = "source-over";
      tctx.filter = "none";
      tctx.clearRect(0, 0, W, H);
      tctx.drawImage(im, 0, 0, W, H);
      tctx.globalCompositeOperation = "destination-in";
      tctx.filter = `blur(${Math.max(0.5, 0.0015 * Math.min(W, H))}px)`;   // = the server's feather
      tctx.drawImage(L.liveMask, 0, 0, W, H);
      tctx.filter = "none";
      src = tmp;
    }
    ctx.globalCompositeOperation = BLEND[L.blend] || "source-over";
    ctx.drawImage(src, x0, 0, W, H);
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
