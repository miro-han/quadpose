import React from "react";
import { ChevronRight, Copy, Minus, Square, X } from "lucide-react";
import { MaterialStyle } from "../foundation/liquidProfiles";
import {
  computeSidebarMetrics,
  computeTreeRowIndent
} from "../foundation/sidebarGeometry";
import { MaterialSurface } from "./MaterialSurface";

export type WorkspaceLayoutVariant = "band" | "floating";

export interface WorkspaceShellProps {
  /**
   * Workspace composition mode:
   * - `"band"` (default): Full-width upper bar with bottom border and elevated panel sidebar.
   * - `"floating"`: SIDE-faithful continuous canvas with floating upper control clusters,
   *   `--miro-shell-clearance` top clearance, and quiet tonal sidebar rail.
   */
  variant?: WorkspaceLayoutVariant;
  upperBar?: React.ReactNode;
  sidebar?: React.ReactNode;
  children?: React.ReactNode;
  className?: string;
  bodyClassName?: string;
  mainClassName?: string;
  style?: React.CSSProperties;
}

/**
 * Top-level workspace layout container supporting both Banded (`variant="band"`, default)
 * and SIDE-faithful Floating (`variant="floating"`, opt-in) composition modes.
 */
export function WorkspaceShell({
  variant = "band",
  upperBar,
  sidebar,
  children,
  className = "",
  bodyClassName = "",
  mainClassName = "",
  style
}: WorkspaceShellProps) {
  const hasStructuredSlots = upperBar !== undefined || sidebar !== undefined;

  return (
    <div
      className={["miro-workspace-shell", "workspace-shell", className]
        .filter(Boolean)
        .join(" ")}
      data-variant={variant}
      style={style}
    >
      {hasStructuredSlots ? (
        <>
          {upperBar}
          <div
            className={[
              "miro-workspace-body",
              "workspace-body",
              bodyClassName
            ]
              .filter(Boolean)
              .join(" ")}
            data-variant={variant}
          >
            {sidebar}
            <main
              className={[
                "miro-workspace-main",
                "workspace-main",
                mainClassName
              ]
                .filter(Boolean)
                .join(" ")}
              data-variant={variant}
            >
              {children}
            </main>
          </div>
        </>
      ) : (
        children
      )}
    </div>
  );
}

export interface WorkspaceUpperBarProps {
  left?: React.ReactNode;
  center?: React.ReactNode;
  right?: React.ReactNode;
  mode?: MaterialStyle;
  /**
   * Upper bar composition variant:
   * - `"band"` (default): Full-width material band with bottom border.
   * - `"floating"`: Transparent, borderless floating header container where only the
   *   left/center/right capsule controls carry material surfaces.
   */
  variant?: WorkspaceLayoutVariant;
  onDragRegionMouseDown?: React.MouseEventHandler<HTMLElement>;
  onDragRegionDoubleClick?: React.MouseEventHandler<HTMLElement>;
  className?: string;
  style?: React.CSSProperties;
}

/**
 * Top workspace bar reproducing SIDE's upper bar with left/center/right compact clusters
 * and optional native window drag region hooks.
 * Supports `"band"` (default) and `"floating"` (opt-in SIDE continuous canvas) variants.
 */
export function WorkspaceUpperBar({
  left,
  center,
  right,
  mode,
  variant = "band",
  onDragRegionMouseDown,
  onDragRegionDoubleClick,
  className = "",
  style
}: WorkspaceUpperBarProps) {
  const classes = [
    "miro-workspace-upper-bar",
    "workspace-upper-bar",
    className
  ]
    .filter(Boolean)
    .join(" ");

  const hasDragRegion = Boolean(
    onDragRegionMouseDown || onDragRegionDoubleClick
  );

  if (variant === "floating") {
    return (
      <header
        className={classes}
        data-variant="floating"
        data-drag-region={hasDragRegion ? "true" : undefined}
        style={style}
        onMouseDown={onDragRegionMouseDown}
        onDoubleClick={onDragRegionDoubleClick}
      >
        <div className="miro-workspace-upper-cluster workspace-upper-cluster">
          {left}
        </div>
        <div className="miro-workspace-upper-cluster workspace-upper-cluster">
          {center}
        </div>
        <div className="miro-workspace-upper-cluster workspace-upper-cluster">
          {right}
        </div>
      </header>
    );
  }

  return (
    <MaterialSurface
      as="header"
      mode={mode}
      profile="shell"
      radius={0}
      className={classes}
      data-variant="band"
      style={style}
      onMouseDown={onDragRegionMouseDown}
      onDoubleClick={onDragRegionDoubleClick}
    >
      <div className="miro-workspace-upper-cluster workspace-upper-cluster">
        {left}
      </div>
      <div className="miro-workspace-upper-cluster workspace-upper-cluster">
        {center}
      </div>
      <div className="miro-workspace-upper-cluster workspace-upper-cluster">
        {right}
      </div>
    </MaterialSurface>
  );
}

export interface WindowControlsProps {
  maximized?: boolean;
  onMinimize?: () => void;
  onToggleMaximize?: () => void;
  onClose?: () => void;
  className?: string;
}

/**
 * Platform-independent window decoration buttons (Minimize, Maximize/Restore, Close).
 * Works standalone in browsers or can be wired to the optional `@miro-design/ui/platform` adapter.
 */
export function WindowControls({
  maximized = false,
  onMinimize,
  onToggleMaximize,
  onClose,
  className = ""
}: WindowControlsProps) {
  return (
    <div
      className={["miro-window-controls", "window-controls", className]
        .filter(Boolean)
        .join(" ")}
      role="group"
      aria-label="Window controls"
    >
      <button
        type="button"
        className="miro-window-control-btn window-control-btn"
        data-action="minimize"
        aria-label="Minimize window"
        onClick={onMinimize}
      >
        <Minus size={14} aria-hidden="true" />
      </button>
      <button
        type="button"
        className="miro-window-control-btn window-control-btn"
        data-action="maximize"
        aria-label={maximized ? "Restore window" : "Maximize window"}
        onClick={onToggleMaximize}
      >
        {maximized ? (
          <Copy size={12} aria-hidden="true" />
        ) : (
          <Square size={12} aria-hidden="true" />
        )}
      </button>
      <button
        type="button"
        className="miro-window-control-btn window-control-btn"
        data-action="close"
        aria-label="Close window"
        onClick={onClose}
      >
        <X size={14} aria-hidden="true" />
      </button>
    </div>
  );
}

export interface SidebarTreeNode {
  id: string;
  label: string;
  depth?: number;
  badge?: string;
  icon?: React.ReactNode;
}

export interface WorkspaceSidebarProps {
  collapsed?: boolean;
  width?: number;
  /**
   * Sidebar surface variant:
   * - `"band"` (default): Standard opaque panel sidebar (`--miro-panel`).
   * - `"floating"`: SIDE-faithful quiet tonal sidebar with top shell clearance for floating upper controls.
   */
  variant?: WorkspaceLayoutVariant;
  header?: React.ReactNode;
  footer?: React.ReactNode;
  items: SidebarTreeNode[];
  activeId?: string;
  onSelect?: (id: string) => void;
  className?: string;
  style?: React.CSSProperties;
}

/**
 * Collapsible workspace sidebar reproducing SIDE's `WorkspaceSidebar` geometry and tree rows.
 */
export function WorkspaceSidebar({
  collapsed = false,
  width,
  variant = "band",
  header,
  footer,
  items,
  activeId,
  onSelect,
  className = "",
  style
}: WorkspaceSidebarProps) {
  const metrics = computeSidebarMetrics(collapsed, width);

  if (metrics.collapsed) {
    return null;
  }

  return (
    <aside
      className={["miro-workspace-sidebar", "workspace-sidebar", className]
        .filter(Boolean)
        .join(" ")}
      data-variant={variant}
      style={{ width: `${metrics.width}px`, ...style }}
    >
      {header ? <div>{header}</div> : null}
      <nav
        className="miro-sidebar-tree sidebar-tree"
        aria-label="Workspace files"
      >
        {items.map((item) => {
          const active = item.id === activeId;
          const indent = computeTreeRowIndent(item.depth ?? 0);
          return (
            <button
              key={item.id}
              type="button"
              data-active={active ? "true" : undefined}
              aria-current={active ? "page" : undefined}
              className="miro-sidebar-tree-row sidebar-tree-row"
              style={{ paddingLeft: `${indent}px` }}
              onClick={() => onSelect?.(item.id)}
            >
              <span
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  gap: "0.4rem",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap"
                }}
              >
                {item.icon}
                <span>{item.label}</span>
              </span>
              {item.badge ? (
                <span className="miro-sidebar-tree-badge sidebar-tree-badge">
                  {item.badge}
                </span>
              ) : null}
            </button>
          );
        })}
      </nav>
      {footer ? <div style={{ marginTop: "auto" }}>{footer}</div> : null}
    </aside>
  );
}

export interface ActiveFilePathProps {
  segments: string[];
  onSegmentClick?: (index: number, segment: string) => void;
  className?: string;
}

/**
 * Quiet interactive breadcrumb trail reproducing SIDE's `ActiveFilePath`.
 */
export function ActiveFilePath({
  segments,
  onSegmentClick,
  className = ""
}: ActiveFilePathProps) {
  return (
    <nav
      aria-label="Active file path"
      className={["miro-active-file-path", "active-file-path", className]
        .filter(Boolean)
        .join(" ")}
    >
      {segments.map((segment, index) => {
        const isLast = index === segments.length - 1;
        return (
          <React.Fragment key={`${segment}-${index}`}>
            {index > 0 ? (
              <ChevronRight size={12} aria-hidden="true" />
            ) : null}
            <button
              type="button"
              className="miro-active-file-segment active-file-segment"
              data-current={isLast ? "true" : undefined}
              aria-current={isLast ? "location" : undefined}
              onClick={() => onSegmentClick?.(index, segment)}
            >
              {segment}
            </button>
          </React.Fragment>
        );
      })}
    </nav>
  );
}
