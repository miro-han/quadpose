/**
 * JSON-RPC client preserving the schema_version=1 contract from the provided Tauri asset,
 * routed to the local Python Studio backend (works in both browser and Tauri WebView).
 */

const isTauriOrFile =
  typeof window !== "undefined" &&
  ("__TAURI_INTERNALS__" in window || window.location.protocol === "file:");

export const API_BASE =
  (import.meta as unknown as { env?: { VITE_STUDIO_API?: string } }).env
    ?.VITE_STUDIO_API ??
  (isTauriOrFile ? "http://127.0.0.1:8765" : "");

export async function rpc<T = unknown>(
  method: string,
  params: Record<string, unknown> = {},
): Promise<T> {
  const id =
    typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
      ? crypto.randomUUID()
      : `${Date.now()}-${Math.random().toString(16).slice(2)}`;

  const res = await fetch(`${API_BASE}/api/rpc`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ schema_version: 1, id, method, params }),
  });

  if (!res.ok) {
    throw new Error(`HTTP ${res.status}: ${res.statusText}`);
  }

  const response = (await res.json()) as {
    schema_version: 1;
    id: string;
    result: T;
    error: { code: string; message: string } | null;
  };

  if (response.schema_version !== 1) {
    throw new Error("Runtime schema version mismatch");
  }
  if (response.error) {
    throw new Error(response.error.message);
  }
  return response.result;
}

export function cameraStreamUrl(slot: number, bust?: number): string {
  const q = bust ? `?t=${bust}` : "";
  return `${API_BASE}/api/cameras/${slot}/stream${q}`;
}

export function samFrameUrl(
  mode: "raw" | "overlay" | "edges",
  rev: number,
): string {
  return `${API_BASE}/api/sam/frame.jpg?mode=${mode}&rev=${rev}`;
}

export function pipelinePreviewUrl(rev: number): string {
  return `${API_BASE}/api/pipeline/preview.jpg?rev=${rev}`;
}
