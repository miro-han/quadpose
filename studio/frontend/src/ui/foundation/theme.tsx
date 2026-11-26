import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState
} from "react";
import { MaterialStyle } from "./liquidProfiles";
import { DensityMode, normalizeScale, ThemeMode } from "./tokens";

export interface DesignSystemAppearance {
  theme: ThemeMode;
  surfaceStyle: MaterialStyle;
  opticalEffects: boolean;
  reduceMotion: boolean;
  uiScale: number;
  density?: DensityMode;
}

export interface DesignSystemContextValue extends DesignSystemAppearance {
  density: DensityMode;
  resolvedTheme: "light" | "dark";
  hasProvider: boolean;
  rootElement: HTMLElement | null;
  rootFontScaled?: boolean;
  effectiveZoom?: number;
  setTheme: (theme: ThemeMode) => void;
  setSurfaceStyle: (style: MaterialStyle) => void;
  setOpticalEffects: (enabled: boolean) => void;
  setReduceMotion: (reduce: boolean) => void;
  setUiScale: (scale: number) => void;
  stepUiScale: (delta: number) => void;
  resetUiScale: () => void;
  setDensity: (density: DensityMode) => void;
}

export const DEFAULT_APPEARANCE: DesignSystemAppearance & {
  density: DensityMode;
} = {
  theme: "light",
  surfaceStyle: "liquid",
  opticalEffects: true,
  reduceMotion: false,
  uiScale: 1.0,
  density: "compact"
};

export interface ApplyAppearanceOptions {
  syncFontSize?: boolean;
}

/**
 * Synchronizes the design system appearance state onto a target root element
 * (defaults to `document.documentElement`).
 * By default `DesignSystemProvider` uses a locally scoped `.miro-root` container
 * and only calls this on `document.documentElement` when `syncDocumentRoot={true}` is explicitly set.
 */
export function applyAppearanceToDocument(
  appearance: Partial<DesignSystemAppearance>,
  root?: HTMLElement | null,
  options: ApplyAppearanceOptions = {}
): void {
  const target =
    root ?? (typeof document !== "undefined" ? document.documentElement : null);
  if (!target) return;

  const { syncFontSize = true } = options;

  if (target.dataset) {
    target.dataset.miroGlobal = "true";
    if (appearance.theme !== undefined) {
      target.dataset.theme = appearance.theme;
    }
    if (appearance.surfaceStyle !== undefined) {
      target.dataset.surfaceStyle = appearance.surfaceStyle;
    }
    if (appearance.opticalEffects !== undefined) {
      target.dataset.optical = appearance.opticalEffects ? "on" : "off";
    }
    if (appearance.reduceMotion !== undefined) {
      target.dataset.reduceMotion = appearance.reduceMotion ? "true" : "false";
    }
    if (appearance.density !== undefined) {
      target.dataset.density = appearance.density;
    }
  }

  if (appearance.uiScale !== undefined && target.style) {
    const normalized = normalizeScale(appearance.uiScale);
    if (syncFontSize) {
      target.style.fontSize = `${16 * normalized}px`;
    }
    if (typeof target.style.setProperty === "function") {
      target.style.setProperty("--miro-ui-scale", String(normalized));
    }
  }
}

export function resolveEffectiveTheme(theme: ThemeMode): "light" | "dark" {
  if (theme === "light" || theme === "dark") {
    return theme;
  }
  if (
    typeof window !== "undefined" &&
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-color-scheme: dark)").matches
  ) {
    return "dark";
  }
  return "light";
}

const DesignSystemContext = createContext<DesignSystemContextValue>({
  ...DEFAULT_APPEARANCE,
  resolvedTheme: "light",
  hasProvider: false,
  rootElement: null,
  setTheme: () => {},
  setSurfaceStyle: () => {},
  setOpticalEffects: () => {},
  setReduceMotion: () => {},
  setUiScale: () => {},
  stepUiScale: () => {},
  resetUiScale: () => {},
  setDensity: () => {}
});

export interface DesignSystemProviderProps {
  initialAppearance?: Partial<DesignSystemAppearance>;
  storageKey?: string | null;
  /**
   * Explicit opt-in to synchronize appearance attributes onto `document.documentElement`
   * (or `targetRoot`). Defaults to `false` so MIRO does not mutate host globals by default.
   */
  syncDocumentRoot?: boolean;
  /**
   * Explicit opt-in to mutate root `style.fontSize` when `syncDocumentRoot` is enabled.
   * Defaults to `false` so host `rem` sizing is preserved.
   */
  syncRootFontSize?: boolean;
  /**
   * Whether to render a locally scoped `.miro-root` container element around children.
   * Defaults to `true`.
   */
  scopedRoot?: boolean;
  targetRoot?: HTMLElement | null;
  className?: string;
  style?: React.CSSProperties;
  children: React.ReactNode;
}

export function DesignSystemProvider({
  initialAppearance,
  storageKey = null,
  syncDocumentRoot = false,
  syncRootFontSize = false,
  scopedRoot = true,
  targetRoot,
  className = "",
  style,
  children
}: DesignSystemProviderProps) {
  const parent = useContext(DesignSystemContext);
  const [rootElement, setRootElement] = useState<HTMLElement | null>(null);

  const [overrides, setOverrides] = useState<Partial<DesignSystemAppearance>>(
    () => {
      const initial: Partial<DesignSystemAppearance> = {};
      if (initialAppearance) {
        if (initialAppearance.theme !== undefined)
          initial.theme = initialAppearance.theme;
        if (initialAppearance.surfaceStyle !== undefined)
          initial.surfaceStyle = initialAppearance.surfaceStyle;
        if (initialAppearance.opticalEffects !== undefined)
          initial.opticalEffects = initialAppearance.opticalEffects;
        if (initialAppearance.reduceMotion !== undefined)
          initial.reduceMotion = initialAppearance.reduceMotion;
        if (initialAppearance.uiScale !== undefined)
          initial.uiScale = normalizeScale(initialAppearance.uiScale);
        if (initialAppearance.density !== undefined)
          initial.density = initialAppearance.density;
      }

      if (storageKey && typeof window !== "undefined" && window.localStorage) {
        try {
          const raw = window.localStorage.getItem(storageKey);
          if (raw) {
            const parsed = JSON.parse(raw) as Partial<DesignSystemAppearance>;
            if (parsed.theme !== undefined) initial.theme = parsed.theme;
            if (parsed.surfaceStyle !== undefined)
              initial.surfaceStyle = parsed.surfaceStyle;
            if (parsed.opticalEffects !== undefined)
              initial.opticalEffects = parsed.opticalEffects;
            if (parsed.reduceMotion !== undefined)
              initial.reduceMotion = parsed.reduceMotion;
            if (parsed.uiScale !== undefined)
              initial.uiScale = normalizeScale(parsed.uiScale);
            if (parsed.density !== undefined)
              initial.density = parsed.density;
          }
        } catch {
          // Ignore storage read errors
        }
      }
      return initial;
    }
  );

  const prevInitialRef = useRef<Partial<DesignSystemAppearance> | undefined>(
    initialAppearance
  );

  useEffect(() => {
    const prev = prevInitialRef.current;
    prevInitialRef.current = initialAppearance;
    if (!initialAppearance) return;
    setOverrides((curr) => {
      let changed = false;
      const next = { ...curr };
      if (
        initialAppearance.theme !== undefined &&
        initialAppearance.theme !== prev?.theme
      ) {
        next.theme = initialAppearance.theme;
        changed = true;
      }
      if (
        initialAppearance.surfaceStyle !== undefined &&
        initialAppearance.surfaceStyle !== prev?.surfaceStyle
      ) {
        next.surfaceStyle = initialAppearance.surfaceStyle;
        changed = true;
      }
      if (
        initialAppearance.opticalEffects !== undefined &&
        initialAppearance.opticalEffects !== prev?.opticalEffects
      ) {
        next.opticalEffects = initialAppearance.opticalEffects;
        changed = true;
      }
      if (
        initialAppearance.reduceMotion !== undefined &&
        initialAppearance.reduceMotion !== prev?.reduceMotion
      ) {
        next.reduceMotion = initialAppearance.reduceMotion;
        changed = true;
      }
      if (
        initialAppearance.uiScale !== undefined &&
        initialAppearance.uiScale !== prev?.uiScale
      ) {
        next.uiScale = normalizeScale(initialAppearance.uiScale);
        changed = true;
      }
      if (
        initialAppearance.density !== undefined &&
        initialAppearance.density !== prev?.density
      ) {
        next.density = initialAppearance.density;
        changed = true;
      }
      return changed ? next : curr;
    });
  }, [
    initialAppearance,
    initialAppearance?.theme,
    initialAppearance?.surfaceStyle,
    initialAppearance?.opticalEffects,
    initialAppearance?.reduceMotion,
    initialAppearance?.uiScale,
    initialAppearance?.density
  ]);

  const baseAppearance: DesignSystemAppearance & { density: DensityMode } =
    parent.hasProvider
      ? {
          theme: parent.theme,
          surfaceStyle: parent.surfaceStyle,
          opticalEffects: parent.opticalEffects,
          reduceMotion: parent.reduceMotion,
          uiScale: parent.uiScale,
          density: parent.density
        }
      : DEFAULT_APPEARANCE;

  const state: DesignSystemAppearance & { density: DensityMode } = useMemo(
    () => ({
      theme: overrides.theme ?? baseAppearance.theme,
      surfaceStyle: overrides.surfaceStyle ?? baseAppearance.surfaceStyle,
      opticalEffects: overrides.opticalEffects ?? baseAppearance.opticalEffects,
      reduceMotion: overrides.reduceMotion ?? baseAppearance.reduceMotion,
      uiScale: normalizeScale(overrides.uiScale ?? baseAppearance.uiScale),
      density: overrides.density ?? baseAppearance.density
    }),
    [overrides, baseAppearance]
  );

  const [systemDark, setSystemDark] = useState<boolean>(
    () => resolveEffectiveTheme("system") === "dark"
  );

  useEffect(() => {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
      return;
    }
    const mql = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = (e: MediaQueryListEvent) => setSystemDark(e.matches);
    mql.addEventListener?.("change", onChange);
    return () => mql.removeEventListener?.("change", onChange);
  }, []);

  useEffect(() => {
    if (syncDocumentRoot) {
      applyAppearanceToDocument(state, targetRoot, {
        syncFontSize: syncRootFontSize
      });
    } else if (targetRoot && targetRoot.style && !syncRootFontSize) {
      targetRoot.style.fontSize = "";
    }
    if (storageKey && typeof window !== "undefined" && window.localStorage) {
      try {
        window.localStorage.setItem(storageKey, JSON.stringify(state));
      } catch {
        // Ignore storage write errors
      }
    }
  }, [state, syncDocumentRoot, syncRootFontSize, targetRoot, storageKey]);

  const resolvedTheme: "light" | "dark" =
    state.theme === "system" ? (systemDark ? "dark" : "light") : state.theme;

  const setRootRef = useCallback((node: HTMLDivElement | null) => {
    setRootElement(node);
  }, []);

  const rootFontScaled = Boolean(
    (syncDocumentRoot && syncRootFontSize) || parent.rootFontScaled
  );
  const parentEffectiveZoom = parent.hasProvider
    ? (parent.effectiveZoom ?? parent.uiScale ?? 1.0)
    : 1.0;
  const shouldApplyCssZoom =
    !rootFontScaled &&
    Math.abs(state.uiScale - parentEffectiveZoom) > 1e-4;
  const localZoom = shouldApplyCssZoom
    ? Number((state.uiScale / parentEffectiveZoom).toFixed(4))
    : 1.0;
  const effectiveZoom = rootFontScaled
    ? 1.0
    : shouldApplyCssZoom
      ? state.uiScale
      : parentEffectiveZoom;

  const value = useMemo<DesignSystemContextValue>(
    () => ({
      ...state,
      resolvedTheme,
      hasProvider: true,
      rootElement: rootElement ?? parent.rootElement,
      rootFontScaled,
      effectiveZoom,
      setTheme: (theme) => setOverrides((prev) => ({ ...prev, theme })),
      setSurfaceStyle: (surfaceStyle) =>
        setOverrides((prev) => ({ ...prev, surfaceStyle })),
      setOpticalEffects: (opticalEffects) =>
        setOverrides((prev) => ({ ...prev, opticalEffects })),
      setReduceMotion: (reduceMotion) =>
        setOverrides((prev) => ({ ...prev, reduceMotion })),
      setUiScale: (uiScale) =>
        setOverrides((prev) => ({ ...prev, uiScale: normalizeScale(uiScale) })),
      stepUiScale: (delta) =>
        setOverrides((prev) => ({
          ...prev,
          uiScale: normalizeScale((prev.uiScale ?? state.uiScale) + delta)
        })),
      resetUiScale: () =>
        setOverrides((prev) => ({ ...prev, uiScale: 1.0 })),
      setDensity: (density) =>
        setOverrides((prev) => ({ ...prev, density }))
    }),
    [
      state,
      resolvedTheme,
      rootElement,
      parent.rootElement,
      rootFontScaled,
      effectiveZoom
    ]
  );

  const rootStyle: React.CSSProperties = {
    ...style,
    ["--miro-ui-scale" as string]: String(state.uiScale),
    ...(localZoom !== 1.0 ? { zoom: localZoom } : {})
  };

  return (
    <DesignSystemContext.Provider value={value}>
      {scopedRoot ? (
        <div
          ref={setRootRef}
          className={["miro-root", className].filter(Boolean).join(" ")}
          data-miro-root="true"
          data-theme={resolvedTheme}
          data-theme-mode={state.theme}
          data-surface-style={state.surfaceStyle}
          data-optical={state.opticalEffects ? "on" : "off"}
          data-reduce-motion={state.reduceMotion ? "true" : "false"}
          data-density={state.density}
          style={rootStyle}
        >
          {children}
        </div>
      ) : (
        children
      )}
    </DesignSystemContext.Provider>
  );
}

export function useDesignSystem(): DesignSystemContextValue {
  return useContext(DesignSystemContext);
}
