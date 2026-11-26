import React, { useEffect, useId, useRef, useState } from "react";
import { Check, ChevronDown } from "lucide-react";
import { MaterialStyle } from "../foundation/liquidProfiles";
import { MaterialSurface } from "./MaterialSurface";

export interface CompactOption<T extends string = string> {
  value: T;
  label: string;
  meta?: string;
  icon?: React.ReactNode;
  disabled?: boolean;
}

export interface CompactSelectorProps<T extends string = string> {
  value: T;
  options: CompactOption<T>[];
  onChange: (value: T) => void;
  ariaLabel?: string;
  prefixIcon?: React.ReactNode;
  align?: "start" | "end";
  mode?: MaterialStyle;
  disabled?: boolean;
  className?: string;
}

/**
 * Capsule dropdown selector with Liquid Glass trigger and overlay menu.
 * Supports full keyboard navigation (ArrowDown, ArrowUp, Home, End, Enter, Space, Escape).
 */
export function CompactSelector<T extends string = string>({
  value,
  options,
  onChange,
  ariaLabel,
  prefixIcon,
  align = "start",
  mode,
  disabled = false,
  className = ""
}: CompactSelectorProps<T>) {
  const [open, setOpen] = useState(false);
  const [highlightedIndex, setHighlightedIndex] = useState<number>(() =>
    Math.max(
      0,
      options.findIndex((o) => o.value === value && !o.disabled)
    )
  );
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const triggerRef = useRef<HTMLButtonElement | null>(null);
  const optionRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const menuId = useId();

  const selectedOption =
    options.find((o) => o.value === value) ?? options[0];

  useEffect(() => {
    if (!open) return;
    const activeIdx = options.findIndex((o) => o.value === value && !o.disabled);
    const nextIdx =
      activeIdx >= 0
        ? activeIdx
        : options.findIndex((o) => !o.disabled);
    setHighlightedIndex(Math.max(0, nextIdx));
  }, [open, value, options]);

  useEffect(() => {
    if (!open) return;
    const onPointerDown = (e: MouseEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", onPointerDown);
    return () => document.removeEventListener("mousedown", onPointerDown);
  }, [open]);

  const moveHighlight = (step: 1 | -1) => {
    const enabledIndices = options
      .map((opt, i) => (opt.disabled ? -1 : i))
      .filter((i) => i >= 0);
    if (enabledIndices.length === 0) return;
    const currentPos = enabledIndices.indexOf(highlightedIndex);
    const nextPos =
      currentPos === -1
        ? 0
        : (currentPos + step + enabledIndices.length) % enabledIndices.length;
    const targetIdx = enabledIndices[nextPos];
    setHighlightedIndex(targetIdx);
    optionRefs.current[targetIdx]?.focus();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (disabled) return;
    if (!open) {
      if (e.key === "ArrowDown" || e.key === "ArrowUp" || e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        setOpen(true);
      }
      return;
    }

    if (e.key === "Escape") {
      e.preventDefault();
      setOpen(false);
      triggerRef.current?.focus();
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      moveHighlight(1);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      moveHighlight(-1);
    } else if (e.key === "Home") {
      e.preventDefault();
      const first = options.findIndex((o) => !o.disabled);
      if (first >= 0) {
        setHighlightedIndex(first);
        optionRefs.current[first]?.focus();
      }
    } else if (e.key === "End") {
      e.preventDefault();
      for (let i = options.length - 1; i >= 0; i--) {
        if (!options[i].disabled) {
          setHighlightedIndex(i);
          optionRefs.current[i]?.focus();
          break;
        }
      }
    } else if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      const candidate = options[highlightedIndex];
      if (candidate && !candidate.disabled) {
        onChange(candidate.value);
        setOpen(false);
        triggerRef.current?.focus();
      }
    }
  };

  return (
    <div
      ref={wrapRef}
      className={[
        "miro-compact-selector-wrap",
        "compact-selector-wrap",
        className
      ]
        .filter(Boolean)
        .join(" ")}
      onKeyDown={handleKeyDown}
    >
      <MaterialSurface
        as="span"
        mode={mode}
        profile="compact"
        radius="capsule"
        disabled={disabled}
        style={{
          display: "inline-flex",
          borderRadius: "var(--miro-compact-control-radius, 999px)"
        }}
      >
        <button
          ref={triggerRef}
          type="button"
          className="miro-compact-selector-trigger compact-selector-trigger"
          aria-haspopup="listbox"
          aria-expanded={open}
          aria-controls={open ? menuId : undefined}
          aria-label={ariaLabel}
          disabled={disabled}
          onClick={() => {
            if (!disabled) setOpen((prev) => !prev);
          }}
        >
          <span className="miro-compact-selector-label compact-selector-label">
            {prefixIcon ?? selectedOption?.icon}
            <span>{selectedOption?.label ?? value}</span>
          </span>
          <ChevronDown
            size={14}
            className="miro-compact-selector-caret compact-selector-caret"
            aria-hidden="true"
          />
        </button>
      </MaterialSurface>

      {open ? (
        <MaterialSurface
          id={menuId}
          role="listbox"
          aria-label={ariaLabel}
          mode={mode}
          profile="overlay"
          radius={12}
          className="miro-compact-selector-menu compact-selector-menu"
          data-align={align}
        >
          <div className="miro-compact-selector-list">
            {options.map((opt, idx) => {
              const isSelected = opt.value === value;
              const isHighlighted = idx === highlightedIndex;
              return (
                <button
                  key={opt.value}
                  ref={(el) => {
                    optionRefs.current[idx] = el;
                  }}
                  type="button"
                  role="option"
                  aria-selected={isSelected}
                  disabled={opt.disabled}
                  data-highlighted={isHighlighted ? "true" : undefined}
                  className="miro-compact-selector-option compact-selector-option"
                  onMouseEnter={() => {
                    if (!opt.disabled) setHighlightedIndex(idx);
                  }}
                  onClick={() => {
                    if (opt.disabled) return;
                    onChange(opt.value);
                    setOpen(false);
                    triggerRef.current?.focus();
                  }}
                >
                  <span className="miro-compact-selector-label compact-selector-label">
                    {opt.icon}
                    <span>{opt.label}</span>
                  </span>
                  <span
                    style={{
                      display: "inline-flex",
                      alignItems: "center",
                      gap: "0.35rem",
                      flexShrink: 0
                    }}
                  >
                    {opt.meta ? (
                      <span className="miro-compact-selector-option-meta compact-selector-option-meta">
                        {opt.meta}
                      </span>
                    ) : null}
                    {isSelected ? <Check size={13} aria-hidden="true" /> : null}
                  </span>
                </button>
              );
            })}
          </div>
        </MaterialSurface>
      ) : null}
    </div>
  );
}

export interface CompactActionProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  icon?: React.ReactNode;
  label?: string;
  iconOnly?: boolean;
  active?: boolean;
  tone?: "default" | "danger";
  mode?: MaterialStyle;
}

/**
 * Capsule action button wrapped in a `compact` Liquid Glass `MaterialSurface`.
 */
export function CompactAction({
  icon,
  label,
  iconOnly = false,
  active = false,
  tone = "default",
  mode,
  disabled = false,
  className = "",
  children,
  ...rest
}: CompactActionProps) {
  return (
    <MaterialSurface
      as="span"
      mode={mode}
      profile="compact"
      radius="capsule"
      disabled={disabled}
      style={{
        display: "inline-flex",
        borderRadius: "var(--miro-compact-control-radius, 999px)"
      }}
    >
      <button
        type="button"
        disabled={disabled}
        data-active={active ? "true" : undefined}
        data-tone={tone !== "default" ? tone : undefined}
        aria-pressed={active ? true : undefined}
        aria-label={iconOnly ? label ?? rest["aria-label"] : rest["aria-label"]}
        className={[
          "miro-compact-action",
          "compact-action",
          iconOnly ? "miro-icon-only icon-only" : "",
          className
        ]
          .filter(Boolean)
          .join(" ")}
        {...rest}
      >
        <span className="miro-compact-action-label compact-action-label">
          {icon}
          {!iconOnly && (label ?? children) ? <span>{label ?? children}</span> : null}
        </span>
      </button>
    </MaterialSurface>
  );
}

export interface CompactPillProps extends React.HTMLAttributes<HTMLDivElement> {
  mode?: MaterialStyle;
  children: React.ReactNode;
}

/**
 * Capsule container for grouping compact icon or segmented controls inside a single Liquid Glass pill.
 */
export function CompactPill({
  mode,
  className = "",
  children,
  ...rest
}: CompactPillProps) {
  return (
    <MaterialSurface
      mode={mode}
      profile="compact"
      radius="capsule"
      className={[
        "miro-floating-control-pill",
        "floating-control-pill",
        className
      ]
        .filter(Boolean)
        .join(" ")}
      {...rest}
    >
      {children}
    </MaterialSurface>
  );
}

export interface CompactSegmentedOption<T extends string = string> {
  value: T;
  label: string;
  icon?: React.ReactNode;
  disabled?: boolean;
}

export interface CompactSegmentedProps<T extends string = string> {
  value: T;
  options: CompactSegmentedOption<T>[];
  onChange: (value: T) => void;
  ariaLabel?: string;
  panelIdPrefix?: string;
  mode?: MaterialStyle;
  className?: string;
  style?: React.CSSProperties;
}

/**
 * Segmented capsule control with a sliding active selection thumb (`--miro-motion-selection`).
 */
export function CompactSegmented<T extends string = string>({
  value,
  options,
  onChange,
  ariaLabel,
  panelIdPrefix,
  mode,
  className = "",
  style
}: CompactSegmentedProps<T>) {
  const activeIndex = Math.max(
    0,
    options.findIndex((o) => o.value === value)
  );
  const count = Math.max(1, options.length);

  return (
    <MaterialSurface
      role="tablist"
      aria-label={ariaLabel}
      onKeyDown={panelIdPrefix ? (event) => {
        const tabs = [...event.currentTarget.querySelectorAll<HTMLButtonElement>('[role="tab"]:not(:disabled)')];
        const index = tabs.indexOf(event.target as HTMLButtonElement);
        if (index < 0) return;
        const next = { ArrowLeft: (index + tabs.length - 1) % tabs.length, ArrowRight: (index + 1) % tabs.length, Home: 0, End: tabs.length - 1 }[event.key];
        if (next === undefined) return;
        event.preventDefault();
        tabs[next].click();
        tabs[next].focus();
      } : undefined}
      mode={mode}
      profile="compact"
      radius="capsule"
      style={style}
      className={["miro-compact-segmented", "compact-segmented", className]
        .filter(Boolean)
        .join(" ")}
    >
      <span
        className="miro-compact-segmented-thumb compact-segmented-thumb"
        aria-hidden="true"
        style={{
          width: `calc((100% - (var(--miro-compact-control-inset, 2px) * 2) - ${(count - 1) * 2}px) / ${count})`,
          transform: `translateX(calc(${activeIndex} * (100% + 2px)))`
        }}
      />
      {options.map((opt) => {
        const selected = opt.value === value;
        return (
          <button
            key={opt.value}
            type="button"
            role="tab"
            id={panelIdPrefix ? `${panelIdPrefix}-tab-${opt.value}` : undefined}
            aria-controls={panelIdPrefix ? `${panelIdPrefix}-${opt.value}` : undefined}
            tabIndex={panelIdPrefix ? (selected ? 0 : -1) : undefined}
            aria-selected={selected}
            disabled={opt.disabled}
            className="miro-compact-segmented-item compact-segmented-item"
            onClick={() => {
              if (!opt.disabled) onChange(opt.value);
            }}
          >
            {opt.icon}
            <span>{opt.label}</span>
          </button>
        );
      })}
    </MaterialSurface>
  );
}
