/**
 * Optional Platform Integration Layer (`@miro-design/ui/platform`)
 *
 * Isolates Tauri / Windows desktop window shell bindings from the core UI library.
 * Has zero compile-time or runtime dependency on `@tauri-apps/*` or SIDE.
 */

export interface NativeWindowState {
  maximized: boolean;
  fullscreen: boolean;
  focused: boolean;
  platform: "windows" | "macos" | "linux" | "browser";
}

export interface NativeShellTransport {
  invoke?: <T = unknown>(cmd: string, args?: Record<string, unknown>) => Promise<T>;
  minimize?: () => Promise<void> | void;
  toggleMaximize?: () => Promise<boolean | void> | boolean | void;
  close?: () => Promise<void> | void;
  startDragging?: () => Promise<void> | void;
  getState?: () => Promise<Partial<NativeWindowState>> | Partial<NativeWindowState>;
}

interface TauriGlobalLike {
  __TAURI_INTERNALS__?: {
    invoke?: <T = unknown>(cmd: string, args?: Record<string, unknown>) => Promise<T>;
  };
}

export function isTauriEnvironment(win?: unknown): boolean {
  const target = (win ?? (typeof window !== "undefined" ? window : undefined)) as
    | TauriGlobalLike
    | undefined;
  return Boolean(target?.__TAURI_INTERNALS__?.invoke);
}

/**
 * Checks whether a pointer target is inside an interactive element that should NOT initiate
 * native window dragging when the user clicks or drags on the upper bar.
 */
export function isInteractiveDragExclusion(target: EventTarget | null): boolean {
  if (!target || typeof (target as Element).closest !== "function") {
    return false;
  }
  const el = target as Element;
  return Boolean(
    el.closest(
      'button, a, input, select, textarea, [role="button"], [role="listbox"], [role="option"], [role="switch"], [role="tab"], [data-no-drag="true"]'
    )
  );
}

export interface NativeShellAdapter {
  readonly isNative: boolean;
  minimize: () => Promise<void>;
  toggleMaximize: () => Promise<boolean>;
  close: () => Promise<void>;
  startDragging: () => Promise<void>;
  handleTitlebarMouseDown: (event: {
    button?: number;
    detail?: number;
    target: EventTarget | null;
  }) => void;
}

export function createNativeShellAdapter(
  customTransport?: NativeShellTransport
): NativeShellAdapter {
  let maximizedState = false;

  const getGlobalInvoke = () => {
    if (customTransport?.invoke) return customTransport.invoke;
    if (typeof window !== "undefined") {
      const w = window as unknown as TauriGlobalLike;
      return w.__TAURI_INTERNALS__?.invoke;
    }
    return undefined;
  };

  const isNative = Boolean(
    customTransport || isTauriEnvironment()
  );

  const minimize = async (): Promise<void> => {
    if (customTransport?.minimize) {
      await customTransport.minimize();
      return;
    }
    const invoke = getGlobalInvoke();
    if (invoke) {
      await invoke("plugin:window|minimize");
    }
  };

  const toggleMaximize = async (): Promise<boolean> => {
    if (customTransport?.toggleMaximize) {
      const result = await customTransport.toggleMaximize();
      if (typeof result === "boolean") {
        maximizedState = result;
      } else {
        maximizedState = !maximizedState;
      }
      return maximizedState;
    }
    const invoke = getGlobalInvoke();
    if (invoke) {
      await invoke("plugin:window|toggle_maximize");
    }
    maximizedState = !maximizedState;
    return maximizedState;
  };

  const close = async (): Promise<void> => {
    if (customTransport?.close) {
      await customTransport.close();
      return;
    }
    const invoke = getGlobalInvoke();
    if (invoke) {
      await invoke("plugin:window|close");
    }
  };

  const startDragging = async (): Promise<void> => {
    if (customTransport?.startDragging) {
      await customTransport.startDragging();
      return;
    }
    const invoke = getGlobalInvoke();
    if (invoke) {
      await invoke("plugin:window|start_dragging");
    }
  };

  const handleTitlebarMouseDown = (event: {
    button?: number;
    detail?: number;
    target: EventTarget | null;
  }) => {
    if (event.button !== undefined && event.button !== 0) return;
    if (isInteractiveDragExclusion(event.target)) return;
    if (event.detail && event.detail >= 2) {
      void toggleMaximize();
      return;
    }
    void startDragging();
  };

  return {
    isNative,
    minimize,
    toggleMaximize,
    close,
    startDragging,
    handleTitlebarMouseDown
  };
}
