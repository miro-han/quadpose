import {
  DEFAULT_MAX_SINGLE_MAP_PIXELS,
  GeometryCacheStats,
  GeometryMapCache
} from "./geometryCache";
import { lensAmplitude } from "./thickLens";

export type FieldProfile = "panel" | "convex" | "rim";

export interface Shape {
  width: number;
  height: number;
  radius: number;
  band: number;
  profile?: FieldProfile;
}

const sharedCache = new GeometryMapCache();

function smooth(t: number): number {
  const x = Math.max(0, Math.min(1, t));
  return x * x * (3 - 2 * x);
}

export function shapeCacheKey(s: Shape): string {
  const w = Math.round(s.width);
  const h = Math.round(s.height);
  const r = Math.round(Math.min(s.radius, w / 2, h / 2));
  const b = Math.round(s.band);
  const p = s.profile ?? "sine";
  return `${w}:${h}:${r}:${b}:${p}`;
}

/**
 * Signed inward distance to the rounded-rectangle boundary.
 * Positive inside the silhouette, 0 on the boundary, negative outside.
 */
export function distance(x: number, y: number, s: Shape): number {
  const hw = s.width / 2;
  const hh = s.height / 2;
  const r = Math.max(0, Math.min(s.radius, hw, hh));
  const px = Math.abs(x - hw) - (hw - r);
  const py = Math.abs(y - hh) - (hh - r);
  const ox = Math.max(px, 0);
  const oy = Math.max(py, 0);
  const outside = Math.hypot(ox, oy) + Math.min(Math.max(px, py), 0) - r;
  return -outside;
}

/**
 * Evaluates the 2D displacement vector [dx, dy] in [-1, 1] at pixel (x, y).
 * Follows the inward boundary normal within [0, s.band] and returns [0, 0]
 * outside the silhouette or inside the calm interior.
 */
export function field(x: number, y: number, s: Shape): [number, number] {
  const d = distance(x, y, s);
  if (d <= 0 || d >= s.band || s.band <= 0) {
    return [0, 0];
  }

  const eps = 0.25;
  const gx = distance(x + eps, y, s) - distance(x - eps, y, s);
  const gy = distance(x, y + eps, s) - distance(x, y - eps, s);
  const len = Math.hypot(gx, gy);
  if (len < 1e-5) {
    return [0, 0];
  }

  const nx = gx / len;
  const ny = gy / len;
  const t = d / s.band;

  let amp: number;
  if (s.profile === "convex" || s.profile === "rim") {
    const effectiveBand = Math.min(s.band, s.height / 2, s.width / 2);
    amp = lensAmplitude(d, effectiveBand, s.profile);
  } else if (s.profile === "panel") {
    amp = t < 0.22 ? smooth(t / 0.22) : 1 - smooth((t - 0.22) / 0.78);
  } else {
    amp = Math.sin(Math.PI * t);
    if (s.band > s.height / 2) {
      const axisGuard =
        1 - smooth(Math.max(0, Math.min(1, (d - (s.height / 2 - 3)) / 3)));
      amp *= axisGuard;
    }
  }

  return [nx * amp, ny * amp];
}

/**
 * Encodes the displacement vector field into a RGBA buffer (R=dx, G=dy, B=128, A=255)
 * with neutral zero-displacement encoded at 128.
 */
export function generateDisplacementBytes(s: Shape): Uint8ClampedArray {
  const w = Math.max(1, Math.round(s.width));
  const h = Math.max(1, Math.round(s.height));
  const data = new Uint8ClampedArray(w * h * 4);

  for (let i = 0; i < data.length; i += 4) {
    data[i] = 128;
    data[i + 1] = 128;
    data[i + 2] = 128;
    data[i + 3] = 255;
  }

  const r = Math.min(s.radius, w / 2, h / 2);
  const calmInset = Math.max(s.band, r);

  for (let y = 0; y < h; y++) {
    const inCalmY = y >= s.band && y < h - s.band;
    for (let x = 0; x < w; x++) {
      if (inCalmY && x >= calmInset && x < w - calmInset) {
        continue;
      }
      const [dx, dy] = field(x + 0.5, y + 0.5, { ...s, width: w, height: h });
      if (dx === 0 && dy === 0) continue;
      const idx = (y * w + x) * 4;
      data[idx] = Math.max(0, Math.min(255, Math.round(128 + 127 * dx)));
      data[idx + 1] = Math.max(0, Math.min(255, Math.round(128 + 127 * dy)));
    }
  }

  return data;
}

/**
 * Generates or retrieves a cached PNG data URL encoding the boundary-normal displacement field.
 */
export function generateMap(s: Shape): string {
  const w = Math.round(s.width);
  const h = Math.round(s.height);
  if (w <= 0 || h <= 0 || w * h > DEFAULT_MAX_SINGLE_MAP_PIXELS) {
    return "";
  }

  const key = shapeCacheKey(s);
  return sharedCache.getOrCreate(key, w, h, () => {
    if (typeof document === "undefined") {
      return "";
    }
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return "";
    }
    const bytes = generateDisplacementBytes({ ...s, width: w, height: h });
    const imgData = ctx.createImageData(w, h);
    imgData.data.set(bytes);
    ctx.putImageData(imgData, 0, 0);
    return canvas.toDataURL("image/png");
  });
}

export function clearGeometryMapCache(): void {
  sharedCache.clear();
}

export function getGeometryMapCacheStats(): GeometryCacheStats {
  return sharedCache.stats();
}
