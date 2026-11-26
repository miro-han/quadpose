import React, { useEffect, useId, useMemo, useRef, useState } from "react";
import { Search } from "lucide-react";

export interface CommandItemData {
  id: string;
  title: string;
  category?: string;
  shortcut?: string;
  icon?: React.ReactNode;
  keywords?: string[];
  disabled?: boolean;
  onSelect: () => void;
}

export interface CommandPaletteProps {
  items: CommandItemData[];
  placeholder?: string;
  emptyMessage?: string;
  initialQuery?: string;
  /** Optional controlled query value. */
  query?: string;
  /** Optional callback fired whenever the search query changes. */
  onQueryChange?: (query: string) => void;
  /** Optional callback fired when Escape is pressed. */
  onEscape?: () => void;
  /** Whether pressing Escape clears a non-empty search query first (default: true). */
  clearOnEscape?: boolean;
  className?: string;
}

function findFirstEnabledIndex(list: CommandItemData[]): number {
  return list.findIndex((item) => !item.disabled);
}

function findLastEnabledIndex(list: CommandItemData[]): number {
  for (let i = list.length - 1; i >= 0; i--) {
    if (!list[i].disabled) return i;
  }
  return -1;
}

/**
 * Keyboard-first command index and quick-switcher implementing the WAI-ARIA 1.2
 * Combobox + Listbox pattern (`role="combobox"`, `aria-activedescendant`, `aria-controls`).
 */
export function CommandPalette({
  items,
  placeholder = "Filter commands, symbols, or files...",
  emptyMessage = "No matching commands.",
  initialQuery = "",
  query: controlledQuery,
  onQueryChange,
  onEscape,
  clearOnEscape = true,
  className = ""
}: CommandPaletteProps) {
  const baseId = useId().replace(/[^a-zA-Z0-9_-]/g, "");
  const listboxId = `miro-cmd-list-${baseId}`;
  const statusId = `miro-cmd-status-${baseId}`;

  const [internalQuery, setInternalQuery] = useState(initialQuery);
  const [activeIndex, setActiveIndex] = useState<number>(0);
  const optionRefs = useRef<Array<HTMLElement | null>>([]);
  const keyboardNavRef = useRef(false);

  // Sync internalQuery when initialQuery prop updates externally
  useEffect(() => {
    setInternalQuery(initialQuery);
  }, [initialQuery]);

  const effectiveQuery =
    controlledQuery !== undefined ? controlledQuery : internalQuery;

  const updateQuery = (next: string) => {
    if (controlledQuery === undefined) {
      setInternalQuery(next);
    }
    onQueryChange?.(next);
  };

  const filtered = useMemo(() => {
    const q = effectiveQuery.trim().toLowerCase();
    if (!q) return items;
    return items.filter((item) => {
      const haystack = [
        item.title,
        item.category ?? "",
        ...(item.keywords ?? [])
      ]
        .join(" ")
        .toLowerCase();
      return haystack.includes(q);
    });
  }, [items, effectiveQuery]);

  // Resolve a valid enabled activeIndex
  const safeIndex = useMemo(() => {
    if (filtered.length === 0) return -1;
    const candidate = filtered[activeIndex];
    if (candidate && !candidate.disabled) {
      return activeIndex;
    }
    return findFirstEnabledIndex(filtered);
  }, [filtered, activeIndex]);

  // Reset activeIndex to first enabled item when query or items list changes
  useEffect(() => {
    const firstEnabled = findFirstEnabledIndex(filtered);
    setActiveIndex(firstEnabled >= 0 ? firstEnabled : 0);
  }, [effectiveQuery, items, filtered]);

  // Keep active item visible inside scrollable list during keyboard navigation
  useEffect(() => {
    if (!keyboardNavRef.current) return;
    keyboardNavRef.current = false;
    if (safeIndex >= 0) {
      const el = optionRefs.current[safeIndex];
      if (el && typeof el.scrollIntoView === "function") {
        el.scrollIntoView({ block: "nearest" });
      }
    }
  }, [safeIndex]);

  const moveEnabledHighlight = (step: 1 | -1) => {
    const enabledIndices = filtered
      .map((item, idx) => (item.disabled ? -1 : idx))
      .filter((idx) => idx >= 0);
    if (enabledIndices.length === 0) return;

    const currentPos = enabledIndices.indexOf(safeIndex);
    let nextPos: number;
    if (currentPos === -1) {
      nextPos = step === 1 ? 0 : enabledIndices.length - 1;
    } else {
      nextPos =
        (currentPos + step + enabledIndices.length) % enabledIndices.length;
    }
    keyboardNavRef.current = true;
    setActiveIndex(enabledIndices[nextPos]);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      moveEnabledHighlight(1);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      moveEnabledHighlight(-1);
    } else if (e.key === "Home") {
      const first = findFirstEnabledIndex(filtered);
      if (first >= 0) {
        e.preventDefault();
        keyboardNavRef.current = true;
        setActiveIndex(first);
      }
    } else if (e.key === "End") {
      const last = findLastEnabledIndex(filtered);
      if (last >= 0) {
        e.preventDefault();
        keyboardNavRef.current = true;
        setActiveIndex(last);
      }
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (safeIndex >= 0) {
        const target = filtered[safeIndex];
        if (target && !target.disabled) {
          target.onSelect();
        }
      }
    } else if (e.key === "Escape") {
      e.preventDefault();
      if (clearOnEscape && effectiveQuery.length > 0) {
        updateQuery("");
      }
      onEscape?.();
    }
  };

  const activeItem = safeIndex >= 0 ? filtered[safeIndex] : undefined;
  const activeOptionId = activeItem
    ? `miro-cmd-opt-${baseId}-${activeItem.id}`
    : undefined;

  return (
    <div
      className={["miro-command-palette", "command-palette", className]
        .filter(Boolean)
        .join(" ")}
      onKeyDown={handleKeyDown}
    >
      <div className="miro-command-search-row command-search-row">
        <Search
          size={15}
          style={{ color: "var(--miro-muted, var(--muted))" }}
          aria-hidden="true"
        />
        <input
          type="text"
          role="combobox"
          aria-autocomplete="list"
          aria-expanded={true}
          aria-controls={listboxId}
          aria-activedescendant={activeOptionId}
          aria-describedby={statusId}
          className="miro-command-search-input command-search-input"
          placeholder={placeholder}
          aria-label={placeholder}
          value={effectiveQuery}
          onChange={(e) => {
            updateQuery(e.target.value);
          }}
        />
      </div>

      <div
        id={listboxId}
        className="miro-command-list command-list"
        role="listbox"
        aria-label="Commands"
      >
        {filtered.length === 0 ? (
          <div
            id={statusId}
            role="status"
            aria-live="polite"
            style={{
              padding: "var(--miro-space-3, 0.75rem)",
              color: "var(--miro-muted, var(--muted))",
              fontSize: "var(--miro-type-small, 0.75rem)"
            }}
          >
            {emptyMessage}
          </div>
        ) : (
          filtered.map((item, idx) => {
            const active = idx === safeIndex && !item.disabled;
            const optionId = `miro-cmd-opt-${baseId}-${item.id}`;
            return (
              <div
                key={item.id}
                id={optionId}
                ref={(el) => {
                  optionRefs.current[idx] = el;
                }}
                role="option"
                tabIndex={-1}
                aria-selected={active}
                aria-disabled={item.disabled ? true : undefined}
                data-active={active ? "true" : undefined}
                data-disabled={item.disabled ? "true" : undefined}
                className="miro-command-item command-item"
                onMouseEnter={() => {
                  if (!item.disabled) {
                    setActiveIndex(idx);
                  }
                }}
                onClick={() => {
                  if (!item.disabled) {
                    item.onSelect();
                  }
                }}
              >
                <span
                  style={{
                    display: "inline-flex",
                    alignItems: "center",
                    gap: "var(--miro-space-2, 0.5rem)"
                  }}
                >
                  {item.icon}
                  <span>{item.title}</span>
                  {item.category ? (
                    <span
                      style={{
                        color: "var(--miro-muted, var(--muted))",
                        fontSize: "var(--miro-type-small, 0.75rem)"
                      }}
                    >
                      · {item.category}
                    </span>
                  ) : null}
                </span>
                {item.shortcut ? (
                  <kbd className="miro-command-shortcut command-shortcut">
                    {item.shortcut}
                  </kbd>
                ) : null}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
