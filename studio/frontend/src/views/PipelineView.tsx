import React, { useCallback, useEffect, useState } from "react";
import {
  FolderOpen,
  Play,
  RefreshCw,
  Square,
  Terminal,
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
import { pipelinePreviewUrl, rpc } from "../api";

interface RecordingSession {
  name: string;
  path: string;
  status: string;
  note: string;
  created_at: string;
  cameras: number;
}

interface SegmentationSession {
  name: string;
  path: string;
  status: string;
  source: string;
  source_path?: string;
  frame_count: number;
  written_frames: number;
  device: string;
  processing_seconds: number;
}

interface EdgeSession {
  name: string;
  path: string;
  camera_id: string;
  source: string;
  frames: number;
}

interface PipelineRunItem {
  kind: string;
  name: string;
  path: string;
  is_dir: boolean;
  mtime: string;
}

interface VideoItem {
  path: string;
  label: string;
  name: string;
  category: string;
}

interface PipelineStatus {
  busy: boolean;
  stage: string;
  phase: "idle" | "running" | "done" | "error" | "cancelled";
  message: string;
  command: string;
  logs: string[];
  elapsed_s: number;
  output_dir: string | null;
  has_preview: boolean;
  summary: Record<string, unknown> | null;
  rev: number;
}

type StageGroup = "rig" | "observations" | "pose";

const GROUP_OPTIONS: Array<{ value: StageGroup; label: string }> = [
  { value: "rig", label: "보정 · 동기화" },
  { value: "observations", label: "LSD · 관측" },
  { value: "pose", label: "6DoF 융합 · 평가" },
];

const SUITE_OPTIONS = [
  { value: "selection", label: "selection (개발 선택)" },
  { value: "assignment", label: "assignment (개발 절제)" },
  { value: "primary", label: "primary (독립 주 시험)" },
];

export function PipelineView() {
  const [group, setGroup] = useState<StageGroup>("observations");
  const [recordings, setRecordings] = useState<RecordingSession[]>([]);
  const [segmentations, setSegmentations] = useState<SegmentationSession[]>([]);
  const [edges, setEdges] = useState<EdgeSession[]>([]);
  const [pipelineRuns, setPipelineRuns] = useState<PipelineRunItem[]>([]);
  const [videos, setVideos] = useState<VideoItem[]>([]);
  const [status, setStatus] = useState<PipelineStatus | null>(null);
  const [error, setError] = useState("");

  // Stage A: Rig & Sync inputs
  const [extractSession, setExtractSession] = useState("");
  const [extractSlot, setExtractSlot] = useState("1");
  const [extractFrame, setExtractFrame] = useState("0");
  const [rigConfig, setRigConfig] = useState("outputs/calibration/rig_input.json");
  const [flashConfig, setFlashConfig] = useState("outputs/sync/flash_input.json");
  const [alignConfig, setAlignConfig] = useState("outputs/sync/align_input.json");

  // Stage B: Edges & Observations inputs
  const [selectedVideo, setSelectedVideo] = useState("");
  const [cameraId, setCameraId] = useState("camera_1");
  const [selectedSam, setSelectedSam] = useState("");
  const [selectedLsd, setSelectedLsd] = useState("");
  const [obsFrame, setObsFrame] = useState("11");
  const [obsRig, setObsRig] = useState("");
  const [obsPrior, setObsPrior] = useState("");

  // Stage C: 6DoF Pose & Benchmark inputs
  const [groupConfig, setGroupConfig] = useState("outputs/pose/group.json");
  const [benchSuite, setBenchSuite] = useState("selection");
  const [benchSmoke, setBenchSmoke] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const [sess, vids, st] = await Promise.all([
        rpc<{
          recordings: RecordingSession[];
          segmentations: SegmentationSession[];
          edges: EdgeSession[];
          pipeline_runs: PipelineRunItem[];
        }>("pipeline.sessions"),
        rpc<{ videos: VideoItem[] }>("sam.videos"),
        rpc<PipelineStatus>("pipeline.status"),
      ]);
      setRecordings(sess.recordings);
      setSegmentations(sess.segmentations);
      setEdges(sess.edges || []);
      setPipelineRuns(sess.pipeline_runs || []);
      setVideos(vids.videos || []);
      setStatus(st);

      setExtractSession((prev) => prev || sess.recordings[0]?.path || "");
      setSelectedVideo((prev) => prev || vids.videos[0]?.path || "");
      setSelectedSam((prev) => prev || sess.segmentations[0]?.path || "");
      setSelectedLsd((prev) => prev || (sess.edges || [])[0]?.path || "");
    } catch {
      // ignore transient startup errors
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    if (!status?.busy) return;
    const timer = window.setInterval(async () => {
      try {
        const next = await rpc<PipelineStatus>("pipeline.status");
        setStatus(next);
        if (!next.busy) {
          void refresh();
        }
      } catch {
        // ignore
      }
    }, 600);
    return () => window.clearInterval(timer);
  }, [status?.busy, refresh]);

  const runStage = async (stage: string, params: Record<string, unknown> = {}) => {
    setError("");
    try {
      const next = await rpc<PipelineStatus>("pipeline.run", { stage, params });
      setStatus(next);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const cancelJob = async () => {
    try {
      const next = await rpc<PipelineStatus>("pipeline.cancel");
      setStatus(next);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const busy = Boolean(status?.busy);
  const phaseTone =
    status?.phase === "done"
      ? "success"
      : status?.phase === "error"
        ? "danger"
        : status?.phase === "running"
          ? "warning"
          : "neutral";

  return (
    <div className="studio-main">
      {error ? <div className="studio-error-banner">{error}</div> : null}

      <div className="studio-split">
        {/* Left: Connected Pipeline Runner & Live Preview/Log */}
        <MaterialSurface className="sam-stage-card">
          <div className="sam-stage-head">
            <div style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 0 }}>
              <Terminal size={14} style={{ color: "var(--miro-muted)", flexShrink: 0 }} />
              <strong style={{ fontSize: "var(--miro-type-ui)", whiteSpace: "nowrap" }}>
                엔드투엔드 파이프라인 실행
              </strong>
              <StatusPill
                label={
                  status?.phase === "running"
                    ? `실행 중 · ${status.elapsed_s}s`
                    : status?.phase === "done"
                      ? `완료 · ${status.elapsed_s}s`
                      : status?.phase === "error"
                        ? "오류"
                        : status?.phase === "cancelled"
                          ? "중단됨"
                          : "대기"
                }
                tone={phaseTone}
              />
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <CompactSegmented
                ariaLabel="Pipeline Stage Group"
                value={group}
                options={GROUP_OPTIONS}
                onChange={(v) => setGroup(v as StageGroup)}
              />
              {busy ? (
                <CompactAction
                  icon={<Square size={12} />}
                  label="중단"
                  onClick={() => void cancelJob()}
                />
              ) : null}
              {status?.output_dir ? (
                <CompactAction
                  icon={<FolderOpen size={12} />}
                  label="결과 폴더"
                  onClick={() =>
                    void rpc("system.open_folder", { path: status.output_dir })
                  }
                />
              ) : null}
            </div>
          </div>

          {/* Stage Controls */}
          <div className="pipeline-controls-box">
            {group === "rig" ? (
              <div className="pipeline-stage-grid">
                <div className="pipeline-subpanel">
                  <div className="studio-label">1. 보정 프레임 추출 (rig.extract_frame)</div>
                  <div className="sidebar-full-selector">
                    <CompactSelector
                      ariaLabel="녹화 세션 선택"
                      value={extractSession}
                      options={
                        recordings.length
                          ? recordings.map((r) => ({ value: r.path, label: r.name }))
                          : [{ value: "", label: "녹화 세션 없음" }]
                      }
                      onChange={setExtractSession}
                    />
                  </div>
                  <div className="sidebar-row-2">
                    <UtilityInput
                      value={extractSlot}
                      onChange={(e) => setExtractSlot(e.target.value)}
                      placeholder="슬롯 (1-4)"
                    />
                    <UtilityInput
                      value={extractFrame}
                      onChange={(e) => setExtractFrame(e.target.value)}
                      placeholder="프레임 번호"
                    />
                  </div>
                  <UtilityAction
                    className="sidebar-full-btn"
                    disabled={busy || !extractSession}
                    onClick={() =>
                      void runStage("rig_extract_frame", {
                        session_dir: extractSession,
                        slot: Number(extractSlot) || 1,
                        frame: Number(extractFrame) || 0,
                      })
                    }
                  >
                    <Play size={13} /> PNG 프레임 추출
                  </UtilityAction>
                </div>

                <div className="pipeline-subpanel">
                  <div className="studio-label">2. 내부·외부 보정 (rig.calibrate)</div>
                  <UtilityInput
                    value={rigConfig}
                    onChange={(e) => setRigConfig(e.target.value)}
                    placeholder="rig_input.json 경로"
                  />
                  <UtilityAction
                    className="sidebar-full-btn"
                    disabled={busy || !rigConfig.trim()}
                    onClick={() => void runStage("rig_calibrate", { config: rigConfig })}
                  >
                    <Play size={13} /> 카메라 보정 실행
                  </UtilityAction>
                </div>

                <div className="pipeline-subpanel">
                  <div className="studio-label">3. 점멸 신호 분석 (rig.synchronize flash)</div>
                  <UtilityInput
                    value={flashConfig}
                    onChange={(e) => setFlashConfig(e.target.value)}
                    placeholder="flash_input.json 경로"
                  />
                  <UtilityAction
                    className="sidebar-full-btn"
                    disabled={busy || !flashConfig.trim()}
                    onClick={() => void runStage("rig_sync_flash", { config: flashConfig })}
                  >
                    <Play size={13} /> Flash 분석 실행
                  </UtilityAction>
                </div>

                <div className="pipeline-subpanel">
                  <div className="studio-label">4. 프레임 그룹 정렬 (rig.synchronize align)</div>
                  <UtilityInput
                    value={alignConfig}
                    onChange={(e) => setAlignConfig(e.target.value)}
                    placeholder="align_input.json 경로"
                  />
                  <UtilityAction
                    className="sidebar-full-btn"
                    disabled={busy || !alignConfig.trim()}
                    onClick={() => void runStage("rig_sync_align", { config: alignConfig })}
                  >
                    <Play size={13} /> 타임스탬프 정렬 실행
                  </UtilityAction>
                </div>
              </div>
            ) : group === "observations" ? (
              <div className="pipeline-stage-grid">
                <div className="pipeline-subpanel">
                  <div className="studio-label">1. LSD 선분 후보 추출 (perception.edges)</div>
                  <div className="sidebar-full-selector">
                    <CompactSelector
                      ariaLabel="대상 영상 선택"
                      value={selectedVideo}
                      options={
                        videos.length
                          ? videos.map((v) => ({ value: v.path, label: v.label }))
                          : [{ value: "", label: "영상 없음" }]
                      }
                      onChange={setSelectedVideo}
                    />
                  </div>
                  <UtilityInput
                    value={cameraId}
                    onChange={(e) => setCameraId(e.target.value)}
                    placeholder="camera_id (예: camera_1)"
                  />
                  <UtilityAction
                    className="sidebar-full-btn"
                    disabled={busy || !selectedVideo}
                    onClick={() =>
                      void runStage("edges_lsd", {
                        video: selectedVideo,
                        camera_id: cameraId,
                      })
                    }
                  >
                    <Play size={13} /> LSD 선분 추출 실행
                  </UtilityAction>
                </div>

                <div className="pipeline-subpanel">
                  <div className="studio-label">2. 기하 관측 어댑터 (perception.observations)</div>
                  <div className="sidebar-row-2">
                    <div className="sidebar-full-selector">
                      <CompactSelector
                        ariaLabel="SAM 2.1 세션"
                        value={selectedSam}
                        options={
                          segmentations.length
                            ? segmentations.map((s) => ({ value: s.path, label: s.name }))
                            : [{ value: "", label: "SAM 세션 없음" }]
                        }
                        onChange={setSelectedSam}
                      />
                    </div>
                    <div className="sidebar-full-selector">
                      <CompactSelector
                        ariaLabel="LSD 출력"
                        value={selectedLsd}
                        options={
                          edges.length
                            ? edges.map((e) => ({ value: e.path, label: e.name }))
                            : [{ value: "", label: "LSD 결과 없음" }]
                        }
                        onChange={setSelectedLsd}
                      />
                    </div>
                  </div>
                  <div className="sidebar-row-2">
                    <UtilityInput
                      value={obsFrame}
                      onChange={(e) => setObsFrame(e.target.value)}
                      placeholder="프레임 번호 (예: 11)"
                    />
                    <UtilityInput
                      value={obsRig}
                      onChange={(e) => setObsRig(e.target.value)}
                      placeholder="rig.json (선택)"
                    />
                  </div>
                  <UtilityInput
                    value={obsPrior}
                    onChange={(e) => setObsPrior(e.target.value)}
                    placeholder="prior.json (선택)"
                  />
                  <div className="sidebar-row-2">
                    <UtilityAction
                      disabled={busy || !selectedVideo || !selectedSam || !selectedLsd}
                      onClick={() =>
                        void runStage("observations_extract", {
                          video: selectedVideo,
                          session: selectedSam,
                          lsd: selectedLsd,
                          frame: Number(obsFrame) || 0,
                          camera_id: cameraId,
                          rig: obsRig,
                          prior: obsPrior,
                        })
                      }
                    >
                      <Play size={13} /> 관측 추출
                    </UtilityAction>
                    <UtilityAction
                      disabled={busy}
                      onClick={() => void runStage("observations_demo", {})}
                    >
                      <Play size={13} /> 단일 시점 데모
                    </UtilityAction>
                  </div>
                </div>
              </div>
            ) : (
              <div className="pipeline-stage-grid">
                <div className="pipeline-subpanel">
                  <div className="studio-label">1. 다중 시점 6DoF 자세 융합 (fuse_group)</div>
                  <UtilityInput
                    value={groupConfig}
                    onChange={(e) => setGroupConfig(e.target.value)}
                    placeholder="group.json 경로"
                  />
                  <UtilityAction
                    className="sidebar-full-btn"
                    disabled={busy || !groupConfig.trim()}
                    onClick={() =>
                      void runStage("pose_fuse_group", { config: groupConfig })
                    }
                  >
                    <Play size={13} /> 4시점 6DoF 자세 융합 실행
                  </UtilityAction>
                </div>

                <div className="pipeline-subpanel">
                  <div className="studio-label">2. Phase IV 6DoF 벤치마크 검증</div>
                  <div className="sidebar-full-selector">
                    <CompactSelector
                      ariaLabel="벤치마크 Suite"
                      value={benchSuite}
                      options={SUITE_OPTIONS}
                      onChange={setBenchSuite}
                    />
                  </div>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8 }}>
                    <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: "var(--miro-type-small)", cursor: "pointer" }}>
                      <input
                        type="checkbox"
                        checked={benchSmoke}
                        onChange={(e) => setBenchSmoke(e.target.checked)}
                      />
                      빠른 연결 확인 (--smoke)
                    </label>
                    <UtilityAction
                      disabled={busy}
                      onClick={() =>
                        void runStage("pose_benchmark", {
                          suite: benchSuite,
                          smoke: benchSmoke,
                        })
                      }
                    >
                      <Play size={13} /> 벤치마크 실행
                    </UtilityAction>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Live Output Preview & Terminal Log */}
          <div className="pipeline-output-area">
            {status?.has_preview ? (
              <div className="pipeline-preview-pane">
                <img
                  src={pipelinePreviewUrl(status.rev)}
                  alt="Pipeline output overlay"
                />
              </div>
            ) : null}

            <div className="pipeline-log-pane">
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8 }}>
                <span style={{ fontSize: "var(--miro-type-small)", fontWeight: 600 }}>
                  {status?.message || "대기 중"}
                </span>
                {status?.summary ? (
                  <span style={{ fontSize: "var(--miro-type-meta)", color: "var(--miro-muted)" }}>
                    {Object.entries(status.summary)
                      .filter(([k]) => k !== "output" && k !== "methods")
                      .map(([k, v]) => `${k}: ${Array.isArray(v) ? v.join(",") : String(v)}`)
                      .join(" · ")}
                  </span>
                ) : null}
              </div>
              <pre className="pipeline-log-box">
                {(status?.logs?.length ? status.logs : ["실행 로그가 여기에 표시됩니다."]).join("\n")}
              </pre>
            </div>
          </div>
        </MaterialSurface>

        {/* Right: Sessions & Pipeline Artifacts Sidebar */}
        <MaterialSurface className="studio-sidebar">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <strong style={{ fontSize: "var(--miro-type-ui)" }}>세션 · 산출물</strong>
            <CompactAction
              iconOnly
              icon={<RefreshCw size={13} />}
              label="목록 새로고침"
              onClick={() => void refresh()}
            />
          </div>

          <div className="sidebar-section">
            <div className="studio-label">다중 카메라 녹화 ({recordings.length})</div>
            <div className="session-list" style={{ maxHeight: 140 }}>
              {recordings.length === 0 ? (
                <div style={{ fontSize: "var(--miro-type-small)", color: "var(--miro-muted)" }}>
                  녹화 세션 없음
                </div>
              ) : (
                recordings.map((r) => (
                  <div key={r.name} className="session-row">
                    <div style={{ minWidth: 0 }}>
                      <div style={{ fontWeight: 600, fontSize: "var(--miro-type-small)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                        {r.name}
                      </div>
                      <div style={{ fontSize: "var(--miro-type-meta)", color: "var(--miro-muted)" }}>
                        {r.cameras}대 · {r.status}
                      </div>
                    </div>
                    <CompactAction
                      icon={<FolderOpen size={12} />}
                      label="열기"
                      onClick={() => void rpc("system.open_folder", { path: r.path })}
                    />
                  </div>
                ))
              )}
            </div>
          </div>

          <hr className="sidebar-divider" />

          <div className="sidebar-section">
            <div className="studio-label">SAM 2.1 추적 ({segmentations.length})</div>
            <div className="session-list" style={{ maxHeight: 160 }}>
              {segmentations.length === 0 ? (
                <div style={{ fontSize: "var(--miro-type-small)", color: "var(--miro-muted)" }}>
                  SAM 2.1 결과 없음
                </div>
              ) : (
                segmentations.map((s) => (
                  <div key={s.name} className="session-row">
                    <div style={{ minWidth: 0 }}>
                      <div style={{ fontWeight: 600, fontSize: "var(--miro-type-small)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                        {s.name}
                      </div>
                      <div style={{ fontSize: "var(--miro-type-meta)", color: "var(--miro-muted)" }}>
                        {s.written_frames}/{s.frame_count}f · {s.device.toUpperCase()}
                      </div>
                    </div>
                    <CompactAction
                      icon={<FolderOpen size={12} />}
                      label="열기"
                      onClick={() => void rpc("system.open_folder", { path: s.path })}
                    />
                  </div>
                ))
              )}
            </div>
          </div>

          <hr className="sidebar-divider" />

          <div className="sidebar-section" style={{ flex: 1, minHeight: 0 }}>
            <div className="studio-label">파이프라인 산출물 ({edges.length + pipelineRuns.length})</div>
            <div className="session-list">
              {edges.length === 0 && pipelineRuns.length === 0 ? (
                <div style={{ fontSize: "var(--miro-type-small)", color: "var(--miro-muted)" }}>
                  아직 실행된 파이프라인 산출물이 없습니다.
                </div>
              ) : (
                <>
                  {edges.map((e) => (
                    <div key={e.path} className="session-row">
                      <div style={{ minWidth: 0 }}>
                        <div style={{ fontWeight: 600, fontSize: "var(--miro-type-small)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                          {e.name}
                        </div>
                        <div style={{ fontSize: "var(--miro-type-meta)", color: "var(--miro-muted)" }}>
                          lsd · {e.camera_id} · {e.frames}f
                        </div>
                      </div>
                      <CompactAction
                        icon={<FolderOpen size={12} />}
                        label="열기"
                        onClick={() => void rpc("system.open_folder", { path: e.path })}
                      />
                    </div>
                  ))}
                  {pipelineRuns.map((run) => (
                    <div key={run.path} className="session-row">
                      <div style={{ minWidth: 0 }}>
                        <div style={{ fontWeight: 600, fontSize: "var(--miro-type-small)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                          {run.name}
                        </div>
                        <div style={{ fontSize: "var(--miro-type-meta)", color: "var(--miro-muted)" }}>
                          {run.kind} · {run.mtime}
                        </div>
                      </div>
                      <CompactAction
                        icon={<FolderOpen size={12} />}
                        label="열기"
                        onClick={() => void rpc("system.open_folder", { path: run.path })}
                      />
                    </div>
                  ))}
                </>
              )}
            </div>
          </div>
        </MaterialSurface>
      </div>
    </div>
  );
}
