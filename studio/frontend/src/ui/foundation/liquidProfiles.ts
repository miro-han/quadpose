export type MaterialStyle = "flat" | "frosted" | "liquid";
export type LiquidProfileName = "compact" | "shell" | "panel" | "overlay";

export interface LiquidProfile {
  /** Refraction band thickness in pixels inward from the silhouette edge. */
  band: number;
  /** Displacement scale multiplier for SVG feDisplacementMap. */
  strength: number;
  /** Interior veil opacity percentage (0-100) mixed with --panel. */
  veil: number;
  /** Backdrop blur radius in pixels. */
  blur: number;
}

export const liquidProfiles: Record<LiquidProfileName, LiquidProfile> = {
  compact: { band: 15, strength: 5, veil: 48, blur: 3 },
  shell: { band: 10, strength: 2, veil: 0, blur: 1.5 },
  panel: { band: 34, strength: 2, veil: 66, blur: 4 },
  overlay: { band: 10, strength: 2, veil: 78, blur: 6 }
};
