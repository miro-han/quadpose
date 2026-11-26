export const MOTION = {
  instant: 80,
  fast: 140,
  normal: 220,
  selection: 220,
  slow: 280,
  popoverEnter: 160,
  popoverExit: 140,
  navigationPush: 200,
  workspaceTransition: 220,
  semanticFlash: 600,
  zoomHold: 750,
  zoomExit: 180
} as const;

export const EASING = {
  selection: "ease",
  out: "cubic-bezier(0.16, 1, 0.3, 1)",
  standard: "cubic-bezier(0.2, 0, 0, 1)"
} as const;

export function isReducedMotionActive(root?: HTMLElement | null): boolean {
  const target =
    root ?? (typeof document !== "undefined" ? document.documentElement : null);
  if (target?.dataset.reduceMotion === "true") {
    return true;
  }
  if (
    typeof window !== "undefined" &&
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches
  ) {
    return true;
  }
  return false;
}

export function resolveDuration(ms: number, root?: HTMLElement | null): number {
  return isReducedMotionActive(root) ? 0 : ms;
}
