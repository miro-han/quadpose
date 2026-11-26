export interface SidebarLayoutMetrics {
  collapsed: boolean;
  width: number;
  railWidth: number;
  contentInsetLeft: number;
  indentStep: number;
}

export const SIDEBAR_DEFAULTS = {
  expandedWidth: 264,
  minWidth: 210,
  maxWidth: 420,
  railWidth: 44,
  indentStep: 14,
  baseRowPadding: 8
} as const;

export function clampSidebarWidth(width: number): number {
  if (!Number.isFinite(width)) return SIDEBAR_DEFAULTS.expandedWidth;
  return Math.max(
    SIDEBAR_DEFAULTS.minWidth,
    Math.min(SIDEBAR_DEFAULTS.maxWidth, Math.round(width))
  );
}

export function computeSidebarMetrics(
  collapsed: boolean,
  requestedWidth: number = SIDEBAR_DEFAULTS.expandedWidth
): SidebarLayoutMetrics {
  const clamped = clampSidebarWidth(requestedWidth);
  return {
    collapsed,
    width: collapsed ? SIDEBAR_DEFAULTS.railWidth : clamped,
    railWidth: SIDEBAR_DEFAULTS.railWidth,
    contentInsetLeft: collapsed ? 0 : clamped,
    indentStep: SIDEBAR_DEFAULTS.indentStep
  };
}

export function computeTreeRowIndent(depth: number): number {
  const safeDepth = Math.max(0, Math.floor(depth));
  return SIDEBAR_DEFAULTS.baseRowPadding + safeDepth * SIDEBAR_DEFAULTS.indentStep;
}
