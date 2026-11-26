export type ThemeMode = "light" | "dark" | "system";
export type DensityMode = "compact" | "comfortable";

export interface ColorPalette {
  bg: string;
  panel: string;
  soft: string;
  text: string;
  muted: string;
  line: string;
  accent: string;
  accentSoft: string;
  paper: string;
  success: string;
  warning: string;
  danger: string;
  dangerSurface: string;
}

export const LIGHT_PALETTE: ColorPalette = {
  bg: "#f5f5f2",
  panel: "#fafaf8",
  soft: "#f0f1f3",
  text: "#272930",
  muted: "#686d78",
  line: "#e2e4e9",
  accent: "#505d70",
  accentSoft: "#e9ebee",
  paper: "#e6e7eb",
  success: "#568065",
  warning: "#92763f",
  danger: "#a64e44",
  dangerSurface: "#f3e7e3"
};

export const DARK_PALETTE: ColorPalette = {
  bg: "#0e0f11",
  panel: "#151619",
  soft: "#202126",
  text: "#eeeef1",
  muted: "#a2a4ad",
  line: "#2c2e35",
  accent: "#c0cad8",
  accentSoft: "#25282d",
  paper: "#090a0c",
  success: "#9cbda7",
  warning: "#c7b17d",
  danger: "#e5aaa0",
  dangerSurface: "#332624"
};

export const SPACING_TOKENS = {
  space1: "0.25rem",
  space2: "0.5rem",
  space3: "0.75rem",
  space4: "1rem",
  space5: "1.25rem",
  space6: "1.5rem",
  space8: "2rem",
  space10: "2.5rem",
  space12: "3rem"
} as const;

export const RADIUS_TOKENS = {
  control: "3px",
  floatingRow: "8px",
  popover: "12px",
  panel: "18px",
  dialog: "24px",
  badge: "999px",
  compactControl: "999px"
} as const;

export const TYPOGRAPHY_TOKENS = {
  uiFamily: '"Segoe UI Variable", "Segoe UI", -apple-system, BlinkMacSystemFont, sans-serif',
  editorFamily: '"Cascadia Code", Consolas, monospace',
  typeMeta: "0.6875rem",
  typeSmall: "0.75rem",
  typeUi: "0.8125rem",
  typeBody: "0.9375rem",
  typeHeading: "1.25rem",
  typeTitle: "2.5rem",
  leadingUi: 1.5,
  leadingEditor: 1.85,
  trackingIndex: "0.075em"
} as const;

/**
 * SIDE-derived reference preset for compact capsule chrome (`CompactSelector`, `CompactAction`,
 * `CompactPill`, `CompactSegmented`). Invariant across content density modes.
 */
export const COMPACT_CHROME_TOKENS = {
  labelSize: "0.6875rem",
  labelWeight: 600,
  iconSize: "14px",
  controlHeight: "2.375rem",
  controlRadius: "999px",
  controlInset: "2px",
  workspaceControlInset: "4px",
  beadHeight: "calc(var(--miro-compact-control-height) - 6px)",
  shellClearance:
    "calc(var(--miro-compact-control-height) + (var(--miro-workspace-control-inset) * 2))",
  workspaceBottomClearance: "var(--miro-space-4)"
} as const;

export const DENSITY_TYPOGRAPHY_PRESETS = {
  compact: {
    typeMeta: "0.6875rem",
    typeSmall: "0.75rem",
    typeUi: "0.8125rem",
    typeBody: "0.9375rem",
    controlHeight: "2rem"
  },
  comfortable: {
    typeMeta: "0.75rem",
    typeSmall: "0.8125rem",
    typeUi: "0.875rem",
    typeBody: "0.9375rem",
    controlHeight: "2.25rem"
  }
} as const;

export const MIN_UI_SCALE = 0.8;
export const MAX_UI_SCALE = 1.6;
export const UI_SCALE_STEP = 0.1;

/**
 * Clamps and normalizes UI zoom scale to SIDE's [0.8, 1.6] range in 0.1 increments.
 */
export function normalizeScale(scale: number): number {
  if (!Number.isFinite(scale)) return 1.0;
  const clamped = Math.max(MIN_UI_SCALE, Math.min(MAX_UI_SCALE, scale));
  return Math.round(clamped * 10) / 10;
}
