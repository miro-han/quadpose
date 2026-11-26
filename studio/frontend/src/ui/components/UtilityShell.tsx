import React, { useId } from "react";
import { X } from "lucide-react";
import { MaterialStyle } from "../foundation/liquidProfiles";
import { SurfaceDialog } from "./SurfaceDialog";
import { IconAction } from "./UtilityControls";

export interface UtilityNavTab {
  id: string;
  label: string;
  icon?: React.ReactNode;
}

export interface UtilityShellProps {
  open?: boolean;
  onClose?: () => void;
  kicker?: string;
  title: string;
  subtitle?: string;
  tabs?: UtilityNavTab[];
  activeTab?: string;
  onTabChange?: (tabId: string) => void;
  headerActions?: React.ReactNode;
  mode?: MaterialStyle;
  inline?: boolean;
  className?: string;
  children: React.ReactNode;
}

/**
 * Structured utility modal/surface shell reproducing SIDE's `UtilityShell`
 * (used for Appearance, Fonts, AI Settings, and Workspace inspectors).
 */
export function UtilityShell({
  open = true,
  onClose,
  kicker,
  title,
  subtitle,
  tabs,
  activeTab,
  onTabChange,
  headerActions,
  mode,
  inline = false,
  className = "",
  children
}: UtilityShellProps) {
  const titleId = useId();

  return (
    <SurfaceDialog
      open={open}
      onClose={onClose}
      ariaLabelledBy={titleId}
      mode={mode}
      inline={inline}
      className={className}
    >
      <header className="miro-utility-shell-header utility-shell-header">
        <div className="miro-utility-shell-title-group utility-shell-title-group">
          {kicker ? (
            <span className="miro-utility-shell-kicker utility-shell-kicker">
              {kicker}
            </span>
          ) : null}
          <h2 id={titleId} className="miro-utility-shell-title utility-shell-title">
            {title}
          </h2>
          {subtitle ? (
            <p className="miro-utility-shell-subtitle utility-shell-subtitle">
              {subtitle}
            </p>
          ) : null}
        </div>
        <div style={{ display: "inline-flex", alignItems: "center", gap: "0.35rem" }}>
          {headerActions}
          {onClose ? (
            <IconAction label="Close utility dialog" onClick={onClose}>
              <X size={16} aria-hidden="true" />
            </IconAction>
          ) : null}
        </div>
      </header>

      {tabs && tabs.length > 0 ? (
        <div className="miro-utility-shell-grid utility-shell-grid">
          <nav
            className="miro-utility-shell-nav utility-shell-nav"
            aria-label={`${title} sections`}
          >
            {tabs.map((tab) => {
              const selected = tab.id === activeTab;
              return (
                <button
                  key={tab.id}
                  type="button"
                  data-active={selected ? "true" : undefined}
                  aria-selected={selected}
                  className="miro-utility-shell-nav-item utility-shell-nav-item"
                  onClick={() => onTabChange?.(tab.id)}
                >
                  {tab.icon}
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </nav>
          <div className="miro-utility-shell-body utility-shell-body">
            {children}
          </div>
        </div>
      ) : (
        <div className="miro-utility-shell-body utility-shell-body">
          {children}
        </div>
      )}
    </SurfaceDialog>
  );
}

export interface UtilitySectionProps {
  label?: string;
  children: React.ReactNode;
}

export function UtilitySection({ label, children }: UtilitySectionProps) {
  return (
    <section className="miro-utility-section utility-section">
      {label ? (
        <h3 className="miro-utility-section-label utility-section-label">
          {label}
        </h3>
      ) : null}
      <div>{children}</div>
    </section>
  );
}

export interface UtilityRowProps {
  title: string;
  description?: string;
  control: React.ReactNode;
}

export function UtilityRow({ title, description, control }: UtilityRowProps) {
  return (
    <div className="miro-utility-row utility-row">
      <div className="miro-utility-row-info utility-row-info">
        <span className="miro-utility-row-title utility-row-title">{title}</span>
        {description ? (
          <span className="miro-utility-row-desc utility-row-desc">
            {description}
          </span>
        ) : null}
      </div>
      <div>{control}</div>
    </div>
  );
}
