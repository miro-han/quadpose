import React, { useEffect, useId, useLayoutEffect, useRef } from "react";
import { createPortal } from "react-dom";
import { LiquidProfileName, MaterialStyle } from "../foundation/liquidProfiles";
import { useDesignSystem } from "../foundation/theme";
import { MaterialSurface } from "./MaterialSurface";

export interface SurfaceDialogProps {
  open: boolean;
  onClose?: () => void;
  ariaLabel?: string;
  ariaLabelledBy?: string;
  ariaDescribedBy?: string;
  mode?: MaterialStyle;
  profile?: LiquidProfileName;
  inline?: boolean;
  /**
   * Optional ref to receive focus when the modal opens.
   * Defaults to the first tabbable descendant or the dialog surface itself.
   */
  initialFocusRef?: React.RefObject<HTMLElement | null>;
  /**
   * Optional ref to receive focus when the modal closes.
   * Defaults to the element that was focused immediately prior to opening.
   */
  finalFocusRef?: React.RefObject<HTMLElement | null>;
  /**
   * Whether to render the modal backdrop inside a React Portal (default: `true` for non-inline modals).
   * Portals into `portalContainer`, the nearest scoped `.miro-root`, or `document.body`.
   */
  portal?: boolean;
  portalContainer?: Element | null;
  closeOnEscape?: boolean;
  closeOnBackdropClick?: boolean;
  className?: string;
  children: React.ReactNode;
}

const FOCUSABLE_SELECTOR = [
  "a[href]",
  "button:not(:disabled)",
  "input:not(:disabled):not([type='hidden'])",
  "select:not(:disabled)",
  "textarea:not(:disabled)",
  "[tabindex]:not([tabindex='-1'])"
].join(", ");

function getTabbableElements(container: HTMLElement): HTMLElement[] {
  const nodes = Array.from(
    container.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR)
  );
  return nodes.filter((el) => {
    if (el.hasAttribute("disabled") || el.getAttribute("aria-hidden") === "true") {
      return false;
    }
    return true;
  });
}

// Active modal stack so nested dialogs only trap focus and handle Escape on the topmost modal.
const activeModalStack: string[] = [];

function isTopmostModal(id: string): boolean {
  return (
    activeModalStack.length > 0 &&
    activeModalStack[activeModalStack.length - 1] === id
  );
}

function isolateBackgroundSiblings(backdropEl: HTMLElement): () => void {
  if (typeof document === "undefined") return () => {};
  const restored: Array<{
    el: Element;
    prevAriaHidden: string | null;
    hadInert: boolean;
  }> = [];

  let current: HTMLElement | null = backdropEl;
  while (current && current !== document.body && current.parentElement) {
    const parent: HTMLElement = current.parentElement;
    for (const sibling of Array.from(parent.children)) {
      if (
        sibling === current ||
        sibling.tagName === "SCRIPT" ||
        sibling.tagName === "STYLE" ||
        sibling.classList.contains("miro-optical-svg-defs") ||
        sibling.classList.contains("optical-svg-defs") ||
        sibling.classList.contains("miro-surface-dialog-backdrop") ||
        sibling.classList.contains("surface-dialog-backdrop")
      ) {
        continue;
      }
      restored.push({
        el: sibling,
        prevAriaHidden: sibling.getAttribute("aria-hidden"),
        hadInert: sibling.hasAttribute("inert")
      });
      sibling.setAttribute("aria-hidden", "true");
      sibling.setAttribute("inert", "");
    }
    current = parent;
  }

  return () => {
    for (const item of restored) {
      if (item.prevAriaHidden === null) {
        item.el.removeAttribute("aria-hidden");
      } else {
        item.el.setAttribute("aria-hidden", item.prevAriaHidden);
      }
      if (!item.hadInert) {
        item.el.removeAttribute("inert");
      }
    }
  };
}

/**
 * Accessible modal or inline dialog container wrapped in a Liquid Glass `MaterialSurface` (`panel` profile by default).
 *
 * When `inline={false}` (default), enforces full modal semantics:
 * - Portals into the enclosing `.miro-root` scope (or `portalContainer` / `document.body`)
 * - Initial focus placement into `initialFocusRef`, first tabbable element, or dialog container
 * - Focus containment (`Tab` / `Shift+Tab` cycle + `focusin` guard)
 * - Focus restoration to trigger element upon dismissal or unmount
 * - Background sibling isolation (`aria-hidden="true"` and `inert`)
 * - Nested modal stack coordination (only topmost modal responds to `Escape` and traps focus)
 */
export function SurfaceDialog({
  open,
  onClose,
  ariaLabel,
  ariaLabelledBy,
  ariaDescribedBy,
  mode,
  profile = "panel",
  inline = false,
  initialFocusRef,
  finalFocusRef,
  portal = true,
  portalContainer,
  closeOnEscape = true,
  closeOnBackdropClick = true,
  className = "",
  children
}: SurfaceDialogProps) {
  const modalId = useId();
  const ds = useDesignSystem();
  const backdropRef = useRef<HTMLDivElement | null>(null);
  const dialogRef = useRef<HTMLElement | null>(null);
  const restoreFocusRef = useRef<HTMLElement | null>(null);
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;

  useLayoutEffect(() => {
    if (!open || inline || typeof document === "undefined") {
      return;
    }

    const prevActive = document.activeElement as HTMLElement | null;
    restoreFocusRef.current = prevActive;

    activeModalStack.push(modalId);

    const dialogEl = dialogRef.current;
    const backdropEl = backdropRef.current;
    const restoreSiblings = backdropEl
      ? isolateBackgroundSiblings(backdropEl)
      : () => {};

    if (dialogEl) {
      const preferred = initialFocusRef?.current;
      if (preferred && typeof preferred.focus === "function") {
        preferred.focus();
      } else {
        const tabbables = getTabbableElements(dialogEl);
        if (tabbables.length > 0) {
          tabbables[0].focus();
        } else {
          dialogEl.focus();
        }
      }
    }

    return () => {
      restoreSiblings();
      const idx = activeModalStack.lastIndexOf(modalId);
      if (idx !== -1) {
        activeModalStack.splice(idx, 1);
      }

      const targetRestore = finalFocusRef?.current ?? restoreFocusRef.current;
      if (
        targetRestore &&
        typeof targetRestore.focus === "function" &&
        targetRestore.isConnected !== false
      ) {
        try {
          targetRestore.focus();
        } catch {
          // Ignore focus errors on detached nodes
        }
      }
      restoreFocusRef.current = null;
    };
  }, [open, inline, modalId, initialFocusRef, finalFocusRef]);

  useEffect(() => {
    if (!open || inline || typeof window === "undefined") {
      return;
    }

    const onKeyDown = (e: KeyboardEvent) => {
      if (!isTopmostModal(modalId)) return;

      if (e.key === "Escape" && closeOnEscape && onCloseRef.current) {
        e.stopPropagation();
        e.preventDefault();
        onCloseRef.current();
        return;
      }

      if (e.key === "Tab") {
        const dialogEl = dialogRef.current;
        if (!dialogEl) return;
        const tabbables = getTabbableElements(dialogEl);
        if (tabbables.length === 0) {
          e.preventDefault();
          dialogEl.focus();
          return;
        }
        const first = tabbables[0];
        const last = tabbables[tabbables.length - 1];
        const active = document.activeElement as HTMLElement | null;

        if (e.shiftKey) {
          if (
            !active ||
            active === first ||
            active === dialogEl ||
            !dialogEl.contains(active)
          ) {
            e.preventDefault();
            last.focus();
          }
        } else {
          if (!active || active === last || !dialogEl.contains(active)) {
            e.preventDefault();
            first.focus();
          }
        }
      }
    };

    const onFocusIn = (e: FocusEvent) => {
      if (!isTopmostModal(modalId)) return;
      const dialogEl = dialogRef.current;
      const target = e.target as Node | null;
      if (!dialogEl || !target) return;
      if (!dialogEl.contains(target)) {
        const tabbables = getTabbableElements(dialogEl);
        if (tabbables.length > 0) {
          tabbables[0].focus();
        } else {
          dialogEl.focus();
        }
      }
    };

    window.addEventListener("keydown", onKeyDown, true);
    document.addEventListener("focusin", onFocusIn, true);
    return () => {
      window.removeEventListener("keydown", onKeyDown, true);
      document.removeEventListener("focusin", onFocusIn, true);
    };
  }, [open, inline, modalId, closeOnEscape]);

  if (!open) return null;

  const resolvedAriaLabel =
    ariaLabelledBy ? ariaLabel : ariaLabel ?? "Dialog";

  const content = (
    <div
      ref={backdropRef}
      className="miro-surface-dialog-backdrop surface-dialog-backdrop"
      data-inline={inline ? "true" : undefined}
      onMouseDown={(e) => {
        if (
          !inline &&
          closeOnBackdropClick &&
          e.target === e.currentTarget &&
          isTopmostModal(modalId) &&
          onClose
        ) {
          onClose();
        }
      }}
    >
      <MaterialSurface
        ref={dialogRef}
        role="dialog"
        tabIndex={-1}
        aria-modal={inline ? undefined : true}
        aria-label={resolvedAriaLabel}
        aria-labelledby={ariaLabelledBy}
        aria-describedby={ariaDescribedBy}
        mode={mode}
        profile={profile}
        radius={24}
        className={[
          "miro-surface-dialog",
          "surface-dialog",
          className
        ]
          .filter(Boolean)
          .join(" ")}
      >
        {children}
      </MaterialSurface>
    </div>
  );

  const resolvedPortalTarget =
    !inline && portal && typeof document !== "undefined"
      ? portalContainer ?? ds.rootElement ?? document.body
      : null;

  if (resolvedPortalTarget) {
    return createPortal(content, resolvedPortalTarget);
  }

  return content;
}
