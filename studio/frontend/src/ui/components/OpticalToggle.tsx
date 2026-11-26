import React from "react";

export interface OpticalToggleProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
  label?: string;
  id?: string;
  className?: string;
}

/**
 * Capsule switch component matching SIDE's `OpticalToggle` (`role="switch"`).
 */
export function OpticalToggle({
  checked,
  onChange,
  disabled = false,
  label,
  id,
  className = ""
}: OpticalToggleProps) {
  return (
    <button
      id={id}
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      disabled={disabled}
      className={["miro-optical-toggle", "optical-toggle", className]
        .filter(Boolean)
        .join(" ")}
      onClick={() => {
        if (!disabled) {
          onChange(!checked);
        }
      }}
    >
      <span
        className="miro-optical-toggle-thumb optical-toggle-thumb"
        aria-hidden="true"
      />
    </button>
  );
}
