import React from "react";
import { MaterialStyle } from "../foundation/liquidProfiles";
import { normalizeScale } from "../foundation/tokens";
import { MaterialSurface } from "./MaterialSurface";

export interface UtilityActionProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  icon?: React.ReactNode;
  active?: boolean;
  variant?: "ghost" | "solid" | "danger";
}

export function UtilityAction({
  icon,
  active = false,
  variant = "ghost",
  className = "",
  children,
  ...rest
}: UtilityActionProps) {
  return (
    <button
      type="button"
      data-active={active ? "true" : undefined}
      data-variant={variant !== "ghost" ? variant : undefined}
      aria-pressed={active ? true : undefined}
      className={["miro-utility-action", "utility-action", className]
        .filter(Boolean)
        .join(" ")}
      {...rest}
    >
      {icon}
      {children ? <span>{children}</span> : null}
    </button>
  );
}

export interface IconActionProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  label: string;
  active?: boolean;
  children: React.ReactNode;
}

export function IconAction({
  label,
  active = false,
  className = "",
  children,
  ...rest
}: IconActionProps) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      data-active={active ? "true" : undefined}
      aria-pressed={active ? true : undefined}
      className={["miro-icon-action", "icon-action", className]
        .filter(Boolean)
        .join(" ")}
      {...rest}
    >
      {children}
    </button>
  );
}

export type UtilityInputProps = React.InputHTMLAttributes<HTMLInputElement>;

export const UtilityInput = React.forwardRef<HTMLInputElement, UtilityInputProps>(
  function UtilityInput({ className = "", ...rest }, ref) {
    return (
      <input
        ref={ref}
        className={["miro-utility-input", "utility-input", className]
          .filter(Boolean)
          .join(" ")}
        {...rest}
      />
    );
  }
);

export type UtilitySelectProps = React.SelectHTMLAttributes<HTMLSelectElement>;

export const UtilitySelect = React.forwardRef<
  HTMLSelectElement,
  UtilitySelectProps
>(function UtilitySelect({ className = "", children, ...rest }, ref) {
  return (
    <select
      ref={ref}
      className={["miro-utility-select", "utility-select", className]
        .filter(Boolean)
        .join(" ")}
      {...rest}
    >
      {children}
    </select>
  );
});

export type UtilityTextareaProps =
  React.TextareaHTMLAttributes<HTMLTextAreaElement>;

export const UtilityTextarea = React.forwardRef<
  HTMLTextAreaElement,
  UtilityTextareaProps
>(function UtilityTextarea({ className = "", ...rest }, ref) {
  return (
    <textarea
      ref={ref}
      className={["miro-utility-textarea", "utility-textarea", className]
        .filter(Boolean)
        .join(" ")}
      {...rest}
    />
  );
});

export interface StatusPillProps {
  label: string;
  tone?: "neutral" | "success" | "warning" | "danger";
  mode?: MaterialStyle;
  className?: string;
}

export function StatusPill({
  label,
  tone = "neutral",
  mode,
  className = ""
}: StatusPillProps) {
  return (
    <MaterialSurface
      as="span"
      mode={mode}
      profile="compact"
      radius="capsule"
      className={["miro-status-pill", "status-pill", className]
        .filter(Boolean)
        .join(" ")}
    >
      <span
        className="miro-status-dot status-dot"
        data-tone={tone !== "neutral" ? tone : undefined}
        aria-hidden="true"
      />
      <span>{label}</span>
    </MaterialSurface>
  );
}

export interface ZoomIndicatorProps {
  scale: number;
  onReset?: () => void;
  mode?: MaterialStyle;
  className?: string;
}

/**
 * Displays current UI scale percentage in a compact Liquid Glass pill (`ZoomIndicator`).
 */
export function ZoomIndicator({
  scale,
  onReset,
  mode,
  className = ""
}: ZoomIndicatorProps) {
  const normalized = normalizeScale(scale);
  const percent = Math.round(normalized * 100);

  return (
    <MaterialSurface
      as="span"
      mode={mode}
      profile="compact"
      radius="capsule"
      className={["miro-zoom-indicator", "zoom-indicator", className]
        .filter(Boolean)
        .join(" ")}
    >
      <span>{percent}%</span>
      {onReset && normalized !== 1.0 ? (
        <button
          type="button"
          className="miro-utility-action utility-action"
          style={{
            height: "1.35rem",
            padding: "0 0.4rem",
            fontSize: "var(--miro-type-meta, 0.6875rem)"
          }}
          onClick={onReset}
        >
          Reset
        </button>
      ) : null}
    </MaterialSurface>
  );
}
