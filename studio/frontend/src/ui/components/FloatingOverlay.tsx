import React, { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import {
  computeFloatingPosition,
  FloatingPlacementResult,
  RectLike,
  ViewportSize
} from "../foundation/floatingPosition";
import { MaterialStyle } from "../foundation/liquidProfiles";
import { MaterialSurface } from "./MaterialSurface";

export type AnchorRectSource = RectLike | (() => RectLike | null) | null;
export type AnchorElementSource =
  | HTMLElement
  | null
  | React.RefObject<HTMLElement | null>;

export interface FloatingOverlayProps {
  open: boolean;
  /**
   * Anchor bounding rectangle or getter function returning the current rectangle.
   */
  anchorRect?: AnchorRectSource;
  /**
   * Optional live DOM element or ref to track as the anchor.
   * When provided, `FloatingOverlay` automatically reads `getBoundingClientRect()`
   * on scroll, window resize, and anchor resize.
   */
  anchorEl?: AnchorElementSource;
  viewport?: ViewportSize;
  preferredPlacement?: "top" | "bottom";
  align?: "start" | "center" | "end";
  offset?: number;
  margin?: number;
  inline?: boolean;
  mode?: MaterialStyle;
  ariaLabel?: string;
  onPositionChange?: (position: FloatingPlacementResult) => void;
  className?: string;
  children: React.ReactNode;
}

function resolveAnchorElement(
  source?: AnchorElementSource
): HTMLElement | null {
  if (!source) return null;
  if ("current" in source) {
    return source.current;
  }
  return source;
}

function resolveAnchorRect(
  anchorRect?: AnchorRectSource,
  anchorEl?: AnchorElementSource
): RectLike | null {
  const el = resolveAnchorElement(anchorEl);
  if (el && typeof el.getBoundingClientRect === "function") {
    const r = el.getBoundingClientRect();
    return {
      left: r.left,
      top: r.top,
      width: r.width,
      height: r.height
    };
  }
  if (typeof anchorRect === "function") {
    return anchorRect();
  }
  return anchorRect ?? null;
}

/**
 * Viewport-clamped floating overlay (`overlay` Liquid Glass profile)
 * used for contextual selection toolboxes, file action popovers, and inline inspectors.
 *
 * Automatically recalculates placement on window resize, ancestor scroll,
 * anchor element resize/movement, and overlay content resize.
 */
export function FloatingOverlay({
  open,
  anchorRect,
  anchorEl,
  viewport,
  preferredPlacement = "top",
  align = "center",
  offset = 8,
  margin = 12,
  inline = false,
  mode,
  ariaLabel,
  onPositionChange,
  className = "",
  children
}: FloatingOverlayProps) {
  const ref = useRef<HTMLDivElement | null>(null);
  const onPositionChangeRef = useRef(onPositionChange);
  onPositionChangeRef.current = onPositionChange;

  const [coords, setCoords] = useState<FloatingPlacementResult | null>(null);

  const recalculate = useCallback(() => {
    if (!open || inline) {
      setCoords(null);
      return;
    }
    const resolvedAnchor = resolveAnchorRect(anchorRect, anchorEl);
    if (!resolvedAnchor) {
      setCoords(null);
      return;
    }

    const el = ref.current;
    const rect =
      el && typeof el.getBoundingClientRect === "function"
        ? el.getBoundingClientRect()
        : null;
    const width = el?.offsetWidth || (rect && rect.width > 0 ? rect.width : 220);
    const height =
      el?.offsetHeight || (rect && rect.height > 0 ? rect.height : 44);

    const vp: ViewportSize = viewport ?? {
      width: typeof window !== "undefined" ? window.innerWidth : 1280,
      height: typeof window !== "undefined" ? window.innerHeight : 800
    };

    const next = computeFloatingPosition({
      anchor: resolvedAnchor,
      floating: { width, height },
      viewport: vp,
      preferredPlacement,
      align,
      offset,
      margin
    });

    setCoords((prev) => {
      if (
        prev &&
        prev.left === next.left &&
        prev.top === next.top &&
        prev.placement === next.placement
      ) {
        return prev;
      }
      onPositionChangeRef.current?.(next);
      return next;
    });
  }, [
    open,
    inline,
    anchorRect,
    anchorEl,
    viewport,
    preferredPlacement,
    align,
    offset,
    margin
  ]);

  useLayoutEffect(() => {
    recalculate();
  }, [recalculate]);

  useEffect(() => {
    if (!open || inline || typeof window === "undefined") {
      return;
    }

    const handleViewportOrScroll = () => {
      recalculate();
    };

    window.addEventListener("resize", handleViewportOrScroll, { passive: true });
    window.addEventListener("scroll", handleViewportOrScroll, {
      capture: true,
      passive: true
    });

    let ro: ResizeObserver | null = null;
    if (typeof ResizeObserver !== "undefined") {
      ro = new ResizeObserver(() => {
        recalculate();
      });
      if (ref.current) {
        ro.observe(ref.current);
      }
      const targetAnchorEl = resolveAnchorElement(anchorEl);
      if (targetAnchorEl) {
        ro.observe(targetAnchorEl);
      }
    }

    return () => {
      window.removeEventListener("resize", handleViewportOrScroll);
      window.removeEventListener("scroll", handleViewportOrScroll, true);
      ro?.disconnect();
    };
  }, [open, inline, anchorEl, recalculate]);

  if (!open) return null;

  return (
    <MaterialSurface
      ref={ref}
      role="toolbar"
      aria-label={ariaLabel}
      mode={mode}
      profile="overlay"
      radius={12}
      data-inline={inline ? "true" : undefined}
      data-placement={coords?.placement ?? preferredPlacement}
      className={["miro-floating-overlay", "floating-overlay", className]
        .filter(Boolean)
        .join(" ")}
      style={
        !inline && coords
          ? { left: `${coords.left}px`, top: `${coords.top}px` }
          : undefined
      }
    >
      {children}
    </MaterialSurface>
  );
}

export interface FloatingToolbarItem {
  id: string;
  label: string;
  icon?: React.ReactNode;
  active?: boolean;
  disabled?: boolean;
  onClick: () => void;
}

export interface FloatingToolbarProps {
  items: Array<FloatingToolbarItem | "divider">;
  mode?: MaterialStyle;
  ariaLabel?: string;
  className?: string;
}

/**
 * Contextual authoring / action toolbar reproducing SIDE's `AuthoringToolbox` and `FloatingFileActions`.
 */
export function FloatingToolbar({
  items,
  mode,
  ariaLabel = "Contextual actions",
  className = ""
}: FloatingToolbarProps) {
  return (
    <MaterialSurface
      role="toolbar"
      aria-label={ariaLabel}
      mode={mode}
      profile="overlay"
      radius={12}
      className={["miro-floating-toolbar", "floating-toolbar", className]
        .filter(Boolean)
        .join(" ")}
    >
      {items.map((item, idx) => {
        if (item === "divider") {
          return (
            <span
              key={`div-${idx}`}
              className="miro-floating-toolbar-divider floating-toolbar-divider"
              aria-hidden="true"
            />
          );
        }
        return (
          <button
            key={item.id}
            type="button"
            disabled={item.disabled}
            className="miro-utility-action utility-action"
            data-active={item.active ? "true" : undefined}
            aria-pressed={item.active ? true : undefined}
            onClick={item.onClick}
            style={{ height: "1.85rem", padding: "0 0.55rem" }}
          >
            {item.icon}
            <span>{item.label}</span>
          </button>
        );
      })}
    </MaterialSurface>
  );
}
