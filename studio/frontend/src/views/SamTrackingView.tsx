import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  Crosshair,
  FolderOpen,
  Play,
  RotateCcw,
  Sparkles,
  Square,
  Undo2,
} from "lucide-react";
import {
  CompactAction,
  CompactSegmented,
  CompactSelector,
  MaterialSurface,
  StatusPill,
  UtilityAction,
  UtilityInput,
} from "../ui";
import { rpc, samFrameUrl } from "../api";

interface VideoItem {
  path: string;
  label: string;
  name: string;
  category: string;
  size_mb: number;
}

interface SamStatus {
  busy: boolean;
  phase: string;
  message: string;
  error: string;
  progress: { completed: number; total: number; elapsed_s: number };
  video: {
    path?: string;
    total_frames?: number;
    fps?: number;
    width?: number;
    height?: number;
    start?: number;
    length?: number;
  };
  has_frame: boolean;
  has_mask: boolean;
  mask_area_px: number;
  frame_rev: number;
  output_folder: string | null;
  compute: { cuda: boolean; device: string; name: string };
}

type PromptMode = "box" | "positive" | "negative";
type DisplayMode = "raw" | "overlay" | "edges";

const MODE_OPTIONS: Array<{ value: PromptMode; label: string }> = [
  { value: "box", label: "박스" },
  { value: "positive", label: "포함(+)" },
  { value: "negative", label: "제외(-)" },
];

const VIEW_OPTIONS: Array<{ value: DisplayMode; label: string }> = [
  { value: "raw", label: "원본" },
  { value: "overlay", label: "마스크" },
  { value: "edges", label: "에지" },
];

export function SamTrackingView({
  initialVideoPath,
}: {
  initialVideoPath?: string;
}) {
  const [videos, setVideos] = useState<VideoItem[]>([]);
  const [videoPath, setVideoPath] = useState(initialVideoPath || "");
  const [startFrame, setStartFrame] = useState(0);
  const [clipLength, setClipLength] = useState(30);
  const [promptMode, setPromptMode] = useState<PromptMode>("box");
  const [displayMode, setDisplayMode] = useState<DisplayMode>("overlay");
  const [status, setStatus] = useState<SamStatus | null>(null);
  const [box, setBox] = useState<[number, number, number, number] | null>(null);
  const [points, setPoints] = useState<Array<{ x: number; y: number; label: 1 | 0 }>>([]);
  const [dragOrigin, setDragOrigin] = useState<{ x: number; y: number } | null>(null);
  const [error, setError] = useState("");
  const imgRef = useRef<HTMLImageElement | null>(null);

  const loadFrame = useCallback(
    async (path: string, start: number, length: number) => {
      if (!path) return;
      setError("");
      try {
        const res = await rpc<SamStatus>("sam.load_frame", {
          video_path: path,
          start,
          length,
        });
        setStatus(res);
        setBox(null);
        setPoints([]);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    },
    [],
  );

  useEffect(() => {
    void (async () => {
      try {
        const res = await rpc<{ videos: VideoItem[] }>("sam.videos");
        setVideos(res.videos);
        const target = initialVideoPath || res.videos[0]?.path || "";
        if (target) {
          setVideoPath(target);
          void loadFrame(target, 0, 30);
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    })();
  }, [initialVideoPath, loadFrame]);

  useEffect(() => {
    const timer = window.setInterval(async () => {
      try {
        const s = await rpc<SamStatus>("sam.status");
        setStatus(s);
      } catch {
        // ignore transient poll errors
      }
    }, 500);
    return () => window.clearInterval(timer);
  }, []);

  const toImageCoords = (e: React.MouseEvent): { x: number; y: number } | null => {
    const img = imgRef.current;
    const vw = status?.video.width;
    const vh = status?.video.height;
    if (!img || !vw || !vh) return null;
    const rect = img.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return null;
    const nx = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const ny = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
    return { x: Math.round(nx * vw), y: Math.round(ny * vh) };
  };

  const handlePointerDown = (e: React.MouseEvent) => {
    if (status?.busy || !status?.has_frame) return;
    const pt = toImageCoords(e);
    if (!pt) return;
    if (promptMode === "box") {
      setDragOrigin(pt);
      setBox([pt.x, pt.y, pt.x, pt.y]);
    } else {
      setPoints((prev) => [
        ...prev,
        { x: pt.x, y: pt.y, label: promptMode === "positive" ? 1 : 0 },
      ]);
    }
  };

  const handlePointerMove = (e: React.MouseEvent) => {
    if (!dragOrigin) return;
    const pt = toImageCoords(e);
    if (!pt) return;
    setBox([
      Math.min(dragOrigin.x, pt.x),
      Math.min(dragOrigin.y, pt.y),
      Math.max(dragOrigin.x, pt.x),
      Math.max(dragOrigin.y, pt.y),
    ]);
  };

  const handlePointerUp = () => {
    if (!dragOrigin) return;
    setDragOrigin(null);
    if (box && (box[2] - box[0] < 6 || box[3] - box[1] < 6)) {
      setBox(null);
    }
  };

  const runPreview = async () => {
    if (!box) {
      setError("먼저 영상 위에서 대상 물체에 박스를 드래그하세요.");
      return;
    }
    setError("");
    try {
      const s = await rpc<SamStatus>("sam.preview", {
        box,
        points: points.map((p) => [p.x, p.y]),
        labels: points.map((p) => p.label),
      });
      setStatus(s);
      setDisplayMode("overlay");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const runTrack = async () => {
    setError("");
    try {
      const s = await rpc<SamStatus>("sam.track");
      setStatus(s);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const cancelTask = async () => {
    try {
      const s = await rpc<SamStatus>("sam.cancel");
      setStatus(s);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const totalFrames = status?.video.total_frames ?? 0;
  const maxStart = Math.max(0, totalFrames - 1);
  const rangePct = maxStart > 0 ? Math.round((startFrame / maxStart) * 100) : 0;
  const imgW = status?.video.width ?? 1280;
  const imgH = status?.video.height ?? 720;
  const pct =
    status && status.progress.total > 0
      ? Math.round((status.progress.completed / status.progress.total) * 100)
      : 0;
  const posCount = points.filter((p) => p.label === 1).length;
  const negCount = points.filter((p) => p.label === 0).length;

  return (
    <div className="studio-main">
      {error || status?.error ? (
        <div className="studio-error-banner">{error || status?.error}</div>
      ) : null}

      <div className="studio-split">
        <MaterialSurface className="sam-stage-card">
          <div className="sam-stage-head">
            <span
              style={{
                fontSize: "var(--miro-type-small)",
                color: "var(--miro-muted)",
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
              }}
            >
              {status?.video.width
                ? `${status.video.width}×${status.video.height} · ${status.video.fps} fps · 프레임 ${status.video.start ?? 0}`
                : "영상을 불러오세요"}
            </span>
            <CompactSegmented
              ariaLabel="Display Layer"
              value={displayMode}
              options={VIEW_OPTIONS}
              onChange={(val) => setDisplayMode(val as DisplayMode)}
            />
          </div>

          <div className="sam-canvas-wrap">
            {status?.has_frame ? (
              <div
                className="sam-canvas-inner"
                onMouseDown={handlePointerDown}
                onMouseMove={handlePointerMove}
                onMouseUp={handlePointerUp}
                onMouseLeave={handlePointerUp}
              >
                <img
                  ref={imgRef}
                  src={samFrameUrl(
                    status.has_mask ? displayMode : "raw",
                    status.frame_rev,
                  )}
                  alt="SAM 2.1 frame"
                  draggable={false}
                />
                <svg
                  className="sam-svg-overlay"
                  viewBox={`0 0 ${imgW} ${imgH}`}
                  preserveAspectRatio="none"
                >
                  {box ? (
                    <rect
                      x={box[0]}
                      y={box[1]}
                      width={Math.max(1, box[2] - box[0])}
                      height={Math.max(1, box[3] - box[1])}
                      fill="rgba(80, 93, 112, 0.18)"
                      stroke="#9eb4cf"
                      strokeWidth={Math.max(2, imgW * 0.0025)}
                      strokeDasharray="8 5"
                    />
                  ) : null}
                  {points.map((pt, idx) => (
                    <circle
                      key={idx}
                      cx={pt.x}
                      cy={pt.y}
                      r={Math.max(6, imgW * 0.007)}
                      fill={pt.label === 1 ? "#568065" : "#a64e44"}
                      stroke="#ffffff"
                      strokeWidth={Math.max(2, imgW * 0.002)}
                    />
                  ))}
                </svg>
              </div>
            ) : (
              <div className="camera-placeholder">
                우측에서 영상을 선택해 프레임을 불러오세요.
              </div>
            )}
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 5, flexShrink: 0 }}>
            <div style={{ fontSize: "var(--miro-type-small)", color: "var(--miro-text)" }}>
              {status?.message}
            </div>
            {status?.phase === "tracking" || status?.progress.total ? (
              <>
                <div className="studio-progress-track">
                  <div className="studio-progress-fill" style={{ width: `${pct}%` }} />
                </div>
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    fontSize: "var(--miro-type-meta)",
                    color: "var(--miro-muted)",
                  }}
                >
                  <span>
                    진행률 {status.progress.completed} / {status.progress.total} ({pct}%)
                  </span>
                  <span>경과 {status.progress.elapsed_s.toFixed(1)}s</span>
                </div>
              </>
            ) : null}
          </div>
        </MaterialSurface>

        <MaterialSurface className="studio-sidebar">
          <div className="sidebar-section">
            <span className="studio-label">영상 및 구간</span>
            <CompactSelector
              ariaLabel="Video Library"
              value={videoPath}
              disabled={Boolean(status?.busy)}
              className="sidebar-full-selector"
              options={
                videos.length > 0
                  ? videos.map((v) => ({
                      value: v.path,
                      label: v.name,
                      meta: `${v.size_mb} MB`,
                    }))
                  : [{ value: "", label: "영상 없음" }]
              }
              onChange={(val) => {
                setVideoPath(val);
                setStartFrame(0);
                void loadFrame(val, 0, clipLength);
              }}
            />

            <div className="studio-field">
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span className="studio-label">타임라인</span>
                <span style={{ fontSize: "var(--miro-type-meta)", color: "var(--miro-muted)", fontFamily: "var(--miro-editor-family)" }}>
                  {startFrame} / {maxStart}
                </span>
              </div>
              <input
                type="range"
                className="miro-range"
                style={{ ["--range-pct" as string]: `${rangePct}%` }}
                min={0}
                max={maxStart}
                value={startFrame}
                disabled={Boolean(status?.busy) || totalFrames <= 1}
                onChange={(e) => setStartFrame(Number(e.target.value))}
              />
            </div>

            <div className="sidebar-row-2">
              <div className="studio-field">
                <span className="studio-label">시작</span>
                <UtilityInput
                  type="number"
                  min={0}
                  value={startFrame}
                  disabled={Boolean(status?.busy)}
                  onChange={(e) => setStartFrame(Math.max(0, Number(e.target.value)))}
                />
              </div>
              <div className="studio-field">
                <span className="studio-label">길이 (≤120)</span>
                <UtilityInput
                  type="number"
                  min={1}
                  max={120}
                  value={clipLength}
                  disabled={Boolean(status?.busy)}
                  onChange={(e) =>
                    setClipLength(Math.max(1, Math.min(120, Number(e.target.value))))
                  }
                />
              </div>
            </div>

            <CompactAction
              icon={<Crosshair size={13} />}
              label="구간 프레임 불러오기"
              disabled={Boolean(status?.busy) || !videoPath}
              className="sidebar-full-btn"
              onClick={() => void loadFrame(videoPath, startFrame, clipLength)}
            />
          </div>

          <hr className="sidebar-divider" />

          <div className="sidebar-section">
            <span className="studio-label">프롬프트 도구</span>
            <CompactSegmented
              ariaLabel="Prompt Mode"
              value={promptMode}
              options={MODE_OPTIONS}
              onChange={(val) => setPromptMode(val as PromptMode)}
            />
            <div className="sidebar-row-2">
              <CompactAction
                icon={<Undo2 size={13} />}
                label="되돌리기"
                disabled={Boolean(status?.busy) || points.length === 0}
                className="sidebar-full-btn"
                onClick={() => setPoints((prev) => prev.slice(0, -1))}
              />
              <CompactAction
                icon={<RotateCcw size={13} />}
                label="초기화"
                disabled={Boolean(status?.busy)}
                className="sidebar-full-btn"
                onClick={() => {
                  setBox(null);
                  setPoints([]);
                }}
              />
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
              <StatusPill
                label={box ? "Box 지정됨" : "Box 미지정"}
                tone={box ? "success" : "neutral"}
              />
              <StatusPill label={`+${posCount} / -${negCount}`} tone="neutral" />
              {status?.has_mask ? (
                <StatusPill
                  label={`${status.mask_area_px.toLocaleString()} px`}
                  tone="success"
                />
              ) : null}
            </div>
          </div>

          <hr className="sidebar-divider" />

          <div className="sidebar-section" style={{ marginTop: "auto" }}>
            <UtilityAction
              variant="solid"
              icon={<Sparkles size={14} />}
              disabled={Boolean(status?.busy) || !box}
              className="sidebar-full-btn"
              onClick={() => void runPreview()}
            >
              ① 마스크 확인 / 보정 적용
            </UtilityAction>
            <UtilityAction
              variant="solid"
              icon={<Play size={14} />}
              disabled={Boolean(status?.busy) || !status?.has_mask}
              className="sidebar-full-btn"
              onClick={() => void runTrack()}
            >
              ② 확인된 마스크로 추적 시작
            </UtilityAction>
            {status?.busy ? (
              <UtilityAction
                variant="danger"
                icon={<Square size={14} />}
                className="sidebar-full-btn"
                onClick={() => void cancelTask()}
              >
                처리 중단
              </UtilityAction>
            ) : null}
            <CompactAction
              icon={<FolderOpen size={13} />}
              label="결과 폴더 열기"
              className="sidebar-full-btn"
              onClick={() =>
                void rpc("system.open_folder", {
                  path: status?.output_folder || "",
                })
              }
            />
          </div>
        </MaterialSurface>
      </div>
    </div>
  );
}
