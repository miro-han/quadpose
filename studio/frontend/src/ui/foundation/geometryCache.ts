export interface GeometryCacheOptions {
  maxEntries?: number;
  maxPixels?: number;
  maxSingleMapPixels?: number;
}

export interface GeometryCacheStats {
  entries: number;
  pixels: number;
  hits: number;
  misses: number;
  evictions: number;
  oversizedSkips: number;
}

interface CacheRecord {
  key: string;
  url: string;
  pixels: number;
}

export const DEFAULT_MAX_ENTRIES = 12;
export const DEFAULT_MAX_PIXELS = 262_144;
export const DEFAULT_MAX_SINGLE_MAP_PIXELS = 1_048_576;

/**
 * Bounded LRU cache for generated SVG displacement normal maps.
 * Enforces both entry-count and total-pixel budgets to prevent unbounded memory growth.
 */
export class GeometryMapCache {
  private readonly maxEntries: number;
  private readonly maxPixels: number;
  private readonly maxSingleMapPixels: number;
  private readonly map = new Map<string, CacheRecord>();
  private totalPixels = 0;
  private hits = 0;
  private misses = 0;
  private evictions = 0;
  private oversizedSkips = 0;

  constructor(options: GeometryCacheOptions = {}) {
    this.maxEntries = options.maxEntries ?? DEFAULT_MAX_ENTRIES;
    this.maxPixels = options.maxPixels ?? DEFAULT_MAX_PIXELS;
    this.maxSingleMapPixels = options.maxSingleMapPixels ?? DEFAULT_MAX_SINGLE_MAP_PIXELS;
  }

  getOrCreate(
    key: string,
    width: number,
    height: number,
    factory: () => string
  ): string {
    const w = Math.max(0, Math.round(width));
    const h = Math.max(0, Math.round(height));
    const pixels = w * h;

    if (pixels <= 0 || pixels > this.maxSingleMapPixels) {
      this.oversizedSkips += 1;
      return "";
    }

    const existing = this.map.get(key);
    if (existing) {
      this.hits += 1;
      this.map.delete(key);
      this.map.set(key, existing);
      return existing.url;
    }

    this.misses += 1;
    const url = factory();
    if (!url) {
      return "";
    }

    if (pixels > this.maxPixels) {
      return url;
    }

    while (
      this.map.size > 0 &&
      (this.map.size >= this.maxEntries || this.totalPixels + pixels > this.maxPixels)
    ) {
      const oldestKey = this.map.keys().next().value;
      if (oldestKey === undefined) break;
      const oldest = this.map.get(oldestKey);
      this.map.delete(oldestKey);
      if (oldest) {
        this.totalPixels -= oldest.pixels;
        this.evictions += 1;
      }
    }

    this.map.set(key, { key, url, pixels });
    this.totalPixels += pixels;
    return url;
  }

  clear(): void {
    this.map.clear();
    this.totalPixels = 0;
  }

  stats(): GeometryCacheStats {
    return {
      entries: this.map.size,
      pixels: this.totalPixels,
      hits: this.hits,
      misses: this.misses,
      evictions: this.evictions,
      oversizedSkips: this.oversizedSkips
    };
  }
}
