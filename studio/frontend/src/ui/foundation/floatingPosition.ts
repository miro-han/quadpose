export interface RectLike {
  left: number;
  top: number;
  width: number;
  height: number;
}

export interface ViewportSize {
  width: number;
  height: number;
}

export interface FloatingPlacementOptions {
  anchor: RectLike;
  floating: { width: number; height: number };
  viewport: ViewportSize;
  preferredPlacement?: "top" | "bottom";
  align?: "start" | "center" | "end";
  offset?: number;
  margin?: number;
}

export interface FloatingPlacementResult {
  left: number;
  top: number;
  placement: "top" | "bottom";
}

/**
 * Computes viewport-clamped floating overlay coordinates relative to an anchor rect.
 * Flips placement between top and bottom when insufficient space exists on the preferred side.
 */
export function computeFloatingPosition(
  options: FloatingPlacementOptions
): FloatingPlacementResult {
  const {
    anchor,
    floating,
    viewport,
    preferredPlacement = "bottom",
    align = "center",
    offset = 8,
    margin = 12
  } = options;

  let rawLeft: number;
  if (align === "start") {
    rawLeft = anchor.left;
  } else if (align === "end") {
    rawLeft = anchor.left + anchor.width - floating.width;
  } else {
    rawLeft = anchor.left + anchor.width / 2 - floating.width / 2;
  }

  const maxLeft = Math.max(margin, viewport.width - floating.width - margin);
  const left = Math.max(margin, Math.min(maxLeft, Math.round(rawLeft)));

  const belowTop = anchor.top + anchor.height + offset;
  const aboveTop = anchor.top - floating.height - offset;
  const fitsBelow = belowTop + floating.height <= viewport.height - margin;
  const fitsAbove = aboveTop >= margin;

  let placement: "top" | "bottom" = preferredPlacement;
  if (preferredPlacement === "bottom" && !fitsBelow && fitsAbove) {
    placement = "top";
  } else if (preferredPlacement === "top" && !fitsAbove && fitsBelow) {
    placement = "bottom";
  }

  const rawTop = placement === "bottom" ? belowTop : aboveTop;
  const maxTop = Math.max(margin, viewport.height - floating.height - margin);
  const top = Math.max(margin, Math.min(maxTop, Math.round(rawTop)));

  return { left, top, placement };
}
