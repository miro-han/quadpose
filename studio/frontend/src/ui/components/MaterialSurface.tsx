import React, { useId, useLayoutEffect, useMemo, useRef, useState } from "react";
import {
  FieldProfile,
  generateMap,
  Shape
} from "../foundation/geometryField";
import {
  LiquidProfileName,
  liquidProfiles,
  MaterialStyle
} from "../foundation/liquidProfiles";
import { useDesignSystem } from "../foundation/theme";

export type MaterialTag =
  | "div"
  | "span"
  | "nav"
  | "dialog"
  | "section"
  | "aside"
  | "header";

export interface MaterialSurfaceProps
  extends Omit<React.HTMLAttributes<HTMLElement>, "as"> {
  as?: MaterialTag;
  mode?: MaterialStyle;
  profile?: LiquidProfileName;
  fieldProfile?: FieldProfile;
  radius?: number | "capsule" | "css";
  band?: number;
  strength?: number;
  veil?: number;
  blur?: number;
  opticalEffects?: boolean;
  disabled?: boolean;
  open?: boolean;
  children?: React.ReactNode;
}

/**
 * Resolves the active MaterialStyle, prioritizing:
 * 1. Explicit `mode` prop on the component
 * 2. Authoritative React Context (`contextMode`) when inside a `DesignSystemProvider` (`hasProvider === true`)
 * 3. Fallback DOM dataset inspection only when rendered outside any `DesignSystemProvider`
 */
export function resolveMaterialStyle(
  explicitMode?: MaterialStyle,
  contextMode?: MaterialStyle,
  hasProvider: boolean = true
): MaterialStyle {
  if (explicitMode) return explicitMode;
  if (hasProvider && contextMode) {
    return contextMode;
  }
  if (typeof document !== "undefined") {
    const attr = document.documentElement.dataset.surfaceStyle as
      | MaterialStyle
      | undefined;
    if (attr === "flat" || attr === "frosted" || attr === "liquid") {
      return attr;
    }
  }
  return contextMode ?? "liquid";
}

/**
 * Resolves whether optical boundary displacement is enabled, prioritizing:
 * 1. Explicit `opticalEffects` prop on the component
 * 2. Authoritative React Context (`contextOptical`) when inside a `DesignSystemProvider` (`hasProvider === true`)
 * 3. Fallback DOM dataset inspection only when rendered outside any `DesignSystemProvider`
 */
export function resolveOpticalEnabled(
  explicitOptical?: boolean,
  contextOptical?: boolean,
  hasProvider: boolean = true
): boolean {
  if (explicitOptical !== undefined) return explicitOptical;
  if (hasProvider && contextOptical !== undefined) {
    return contextOptical;
  }
  if (typeof document !== "undefined") {
    const attr = document.documentElement.dataset.optical;
    if (attr === "off") return false;
    if (attr === "on") return true;
  }
  return contextOptical ?? true;
}

export const MaterialSurface = React.forwardRef<HTMLElement, MaterialSurfaceProps>(
  function MaterialSurface(
    {
      as: Tag = "div",
      mode,
      profile = "panel",
      fieldProfile,
      radius = "css",
      band,
      strength,
      veil,
      blur,
      opticalEffects,
      disabled = false,
      style,
      className = "",
      children,
      ...rest
    },
    forwardedRef
  ) {
    const rawId = useId();
    const filterId = `miro-liquid-${rawId.replace(/[^a-zA-Z0-9_-]/g, "")}`;
    const localRef = useRef<HTMLElement | null>(null);
    const ds = useDesignSystem();

    const preset = liquidProfiles[profile];
    const effectiveBand = band ?? preset.band;
    const effectiveStrength = strength ?? preset.strength;
    const effectiveVeil = veil ?? preset.veil;
    const effectiveBlur = blur ?? preset.blur;
    const effectiveFieldProfile: FieldProfile | undefined =
      fieldProfile ?? (profile === "panel" ? "panel" : undefined);

    const resolvedMode = resolveMaterialStyle(
      mode,
      ds.surfaceStyle,
      ds.hasProvider
    );
    const resolvedOptical = resolveOpticalEnabled(
      opticalEffects,
      ds.opticalEffects,
      ds.hasProvider
    );

    const [shape, setShape] = useState<Shape | null>(null);

    useLayoutEffect(() => {
      const node = localRef.current;
      if (!node || resolvedMode !== "liquid" || !resolvedOptical || disabled) {
        setShape(null);
        return;
      }

      const measure = () => {
        const w = Math.round(node.clientWidth);
        const h = Math.round(node.clientHeight);
        if (w <= 0 || h <= 0) {
          setShape(null);
          return;
        }
        let r: number;
        if (radius === "capsule") {
          r = h / 2;
        } else if (radius === "css") {
          const computed =
            typeof window !== "undefined"
              ? parseFloat(window.getComputedStyle(node).borderTopLeftRadius) || 0
              : 0;
          r = Math.min(computed, w / 2, h / 2);
        } else {
          r = Math.min(radius, w / 2, h / 2);
        }

        setShape((prev) => {
          if (
            prev &&
            prev.width === w &&
            prev.height === h &&
            prev.radius === r &&
            prev.band === effectiveBand &&
            prev.profile === effectiveFieldProfile
          ) {
            return prev;
          }
          return {
            width: w,
            height: h,
            radius: r,
            band: effectiveBand,
            profile: effectiveFieldProfile
          };
        });
      };

      measure();

      if (typeof ResizeObserver !== "undefined") {
        const ro = new ResizeObserver(measure);
        ro.observe(node);
        return () => ro.disconnect();
      }
      return undefined;
    }, [
      resolvedMode,
      resolvedOptical,
      disabled,
      radius,
      effectiveBand,
      effectiveFieldProfile
    ]);

    const mapUrl = useMemo(() => {
      if (!shape || resolvedMode !== "liquid" || !resolvedOptical || disabled) {
        return "";
      }
      return generateMap(shape);
    }, [shape, resolvedMode, resolvedOptical, disabled]);

    const mergedStyle: React.CSSProperties = {
      ...style,
      ["--miro-material-veil" as string]: `${effectiveVeil}%`,
      ["--miro-material-blur" as string]: `${effectiveBlur}px`,
      ["--material-veil" as string]: `${effectiveVeil}%`,
      ["--material-blur" as string]: `${effectiveBlur}px`,
      ["--overlay-veil" as string]: `${effectiveVeil}%`,
      ["--overlay-blur" as string]: `${effectiveBlur}px`
    };

    const setRefs = (el: HTMLElement | null) => {
      localRef.current = el;
      if (typeof forwardedRef === "function") {
        forwardedRef(el);
      } else if (forwardedRef) {
        (forwardedRef as React.MutableRefObject<HTMLElement | null>).current = el;
      }
    };

    const classes = [
      "miro-surface",
      `miro-surface-${resolvedMode}`,
      "material-surface",
      `material-${resolvedMode}`,
      className
    ]
      .filter(Boolean)
      .join(" ");

    const showLiquidOptics =
      resolvedMode === "liquid" &&
      resolvedOptical &&
      !disabled &&
      shape !== null;

    const backdropFilterValue =
      showLiquidOptics && mapUrl
        ? `${effectiveBlur ? `blur(${effectiveBlur}px) ` : ""}url("#${filterId}")`
        : effectiveBlur
          ? `blur(${effectiveBlur}px) saturate(1.06)`
          : undefined;

    const Element = Tag as unknown as React.ElementType;

    return (
      <Element
        ref={setRefs}
        className={classes}
        data-profile={profile}
        data-surface-mode={resolvedMode}
        data-optical-enabled={resolvedOptical ? "true" : "false"}
        data-disabled={disabled ? "true" : undefined}
        style={mergedStyle}
        {...rest}
      >
        {showLiquidOptics && mapUrl && shape ? (
          <svg className="miro-filter material-filter" aria-hidden="true">
            <defs>
              <filter
                id={filterId}
                x="0%"
                y="0%"
                width="100%"
                height="100%"
                colorInterpolationFilters="sRGB"
              >
                <feImage
                  href={mapUrl}
                  x="0"
                  y="0"
                  width={shape.width}
                  height={shape.height}
                  preserveAspectRatio="none"
                  result="map"
                />
                <feComponentTransfer in="map" result="unbiased">
                  <feFuncR type="linear" slope="1" intercept={-0.5 / 255} />
                  <feFuncG type="linear" slope="1" intercept={-0.5 / 255} />
                </feComponentTransfer>
                <feDisplacementMap
                  in="SourceGraphic"
                  in2="unbiased"
                  scale={effectiveStrength * 2}
                  xChannelSelector="R"
                  yChannelSelector="G"
                />
              </filter>
            </defs>
          </svg>
        ) : null}
        {resolvedMode === "liquid" ? (
          <span
            className="miro-optics material-optics"
            aria-hidden="true"
            style={
              backdropFilterValue
                ? {
                    backdropFilter: backdropFilterValue,
                    WebkitBackdropFilter: backdropFilterValue
                  }
                : undefined
            }
          />
        ) : null}
        {children}
      </Element>
    );
  }
);
