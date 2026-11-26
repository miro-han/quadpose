import React, { useCallback, useEffect, useState } from "react";
import {
  Camera,
  FolderOpen,
  Plug,
  PlugZap,
  RefreshCw,
  Square,
  Video,
} from "lucide-react";
import {
  CompactAction,
  CompactSelector,
  MaterialSurface,
  StatusPill,
  UtilityAction,
  UtilityInput,
} from "../ui";
import { cameraStreamUrl, rpc } from "../api";

interface CameraDevice {
  key: string;
  index: number;
  name: string;
  backend: number;
}

interface SlotStatus {
  slot: number;
  connected: boolean;
  has_frame?: boolean;
  status: string;
  error: string;
  fps: number;
  size_px: [number, number] | null;
  stalled: boolean;
  written: number;
  dropped: number;
  device_name: string;
}

interface CameraStatusPayload {
  connected: boolean;
  slots: SlotStatus[];
  recording: {
    active: boolean;
    ready: boolean;
    elapsed_s: number;
    folder: string;
    folder_name: string;
    total_written: number;
    total_dropped: number;
    error: string;
    done: boolean;
  };
}

const BACKEND_OPTIONS = [
  { value: "DirectShow", label: "DirectShow" },
  { value: "MSMF", label: "MSMF" },
];

const RES_OPTIONS = [
  { value: "640x480", label: "640 × 480" },
  { value: "1280x720", label: "1280 × 720" },
  { value: "1920x1080", label: "1920 × 1080" },
];

const FPS_OPTIONS = [
  { value: "15", label: "15 FPS" },
  { value: "30", label: "30 FPS" },
];

const CAMERA_ROLES = ["+X 측면", "-Y 정면", "+Z 상단", "대각 보완"];

export function CameraStudioView({
  defaultRecordingsDir,
}: {
  defaultRecordingsDir: string;
}) {
  const [backend, setBackend] = useState("DirectShow");
  const [resolution, setResolution] = useState("1280x720");
  const [fps, setFps] = useState("30");
  const [devices, setDevices] = useState<CameraDevice[]>([]);
  const [slotSelection, setSlotSelection] = useState<Record<number, string>>({
    1: "none",
    2: "none",
    3: "none",
    4: "none",
  });
  const [status, setStatus] = useState<CameraStatusPayload | null>(null);
  const [saveDir, setSaveDir] = useState(defaultRecordingsDir);
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [streamBust, setStreamBust] = useState(0);

  useEffect(() => {
    if (defaultRecordingsDir && !saveDir) {
      setSaveDir(defaultRecordingsDir);
    }
  }, [defaultRecordingsDir, saveDir]);

  const scan = useCallback(
    async (targetBackend = backend) => {
      setBusy(true);
      setError("");
      try {
        const res = await rpc<{ devices: CameraDevice[] }>("camera.scan", {
          backend: targetBackend,
        });
        setDevices(res.devices);
        const preferred = res.devices.filter(
          (d) =>
            !["ativ", "integrated", "built-in", "virtual", "obs"].some((kw) =>
              d.name.toLowerCase().includes(kw),
            ),
        );
        const list = preferred.length > 0 ? preferred : res.devices;
        setSlotSelection({
          1: list[0]?.key ?? "none",
          2: list[1]?.key ?? "none",
          3: list[2]?.key ?? "none",
          4: list[3]?.key ?? "none",
        });
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(false);
      }
    },
    [backend],
  );

  const pollStatus = useCallback(async () => {
    try {
      const s = await rpc<CameraStatusPayload>("camera.status");
      setStatus(s);
    } catch {
      // ignore transient polling errors
    }
  }, []);

  useEffect(() => {
    void scan(backend);
    void pollStatus();
  }, []);

  useEffect(() => {
    const timer = window.setInterval(() => {
      void pollStatus();
    }, 600);
    return () => window.clearInterval(timer);
  }, [pollStatus]);

  const toggleConnection = async () => {
    setBusy(true);
    setError("");
    try {
      if (status?.connected) {
        const s = await rpc<CameraStatusPayload>("camera.disconnect");
        setStatus(s);
      } else {
        const s = await rpc<CameraStatusPayload>("camera.connect", {
          slots: slotSelection,
          resolution,
          fps: Number(fps),
        });
        setStatus(s);
        setStreamBust(Date.now());
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const toggleRecording = async () => {
    setBusy(true);
    setError("");
    try {
      if (status?.recording.active) {
        const s = await rpc<CameraStatusPayload>("camera.record_stop");
        setStatus(s);
      } else {
        const s = await rpc<CameraStatusPayload>("camera.record_start", {
          directory: saveDir,
          fps: Number(fps),
          note,
        });
        setStatus(s);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const openFolder = async () => {
    try {
      await rpc("system.open_folder", {
        path: status?.recording.folder || saveDir,
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const deviceOptions = [
    { value: "none", label: "사용 안 함" },
    ...devices.map((d) => ({ value: d.key, label: d.key })),
  ];

  const isConnected = Boolean(status?.connected);
  const isRecording = Boolean(status?.recording.active);

  return (
    <div className="studio-main">
      {error ? <div className="studio-error-banner">{error}</div> : null}

      <div className="studio-split">
        <div className="camera-grid">
          {[1, 2, 3, 4].map((slot) => {
            const slotInfo = status?.slots.find((s) => s.slot === slot);
            const connected = Boolean(slotInfo?.connected);
            const tone = slotInfo?.error
              ? "danger"
              : slotInfo?.stalled
                ? "warning"
                : connected
                  ? "success"
                  : "neutral";

            return (
              <MaterialSurface key={slot} className="camera-card">
                <div className="camera-card-head">
                  <div className="camera-card-title-group">
                    <Camera size={13} style={{ flexShrink: 0, color: "var(--miro-muted)" }} />
                    <strong style={{ fontSize: "var(--miro-type-small)", whiteSpace: "nowrap" }}>
                      CAM 0{slot}
                    </strong>
                    <span className="camera-role-label">{CAMERA_ROLES[slot - 1]}</span>
                    <CompactSelector
                      ariaLabel={`Camera ${slot} device`}
                      value={slotSelection[slot] ?? "none"}
                      options={deviceOptions}
                      disabled={busy || isConnected}
                      className="camera-inline-selector"
                      onChange={(val) =>
                        setSlotSelection((prev) => ({ ...prev, [slot]: val }))
                      }
                    />
                  </div>
                  <StatusPill
                    label={
                      connected
                        ? `${slotInfo?.status} · ${slotInfo?.fps.toFixed(1)} fps`
                        : "대기"
                    }
                    tone={tone}
                  />
                </div>

                <div className="camera-viewport">
                  {connected && slotInfo?.has_frame ? (
                    <img
                      src={cameraStreamUrl(slot, streamBust)}
                      alt={`Camera ${slot} live preview`}
                    />
                  ) : connected && slotInfo?.error ? (
                    <div className="camera-placeholder error">
                      <b>연결 실패</b>: {slotInfo.error}
                      <br />
                      <span style={{ fontSize: "var(--miro-type-meta)", opacity: 0.85 }}>
                        다른 앱 점유 해제 · MSMF 전환 · 640×480 또는 USB 포트 분산 시도
                      </span>
                    </div>
                  ) : connected ? (
                    <div className="camera-placeholder">
                      카메라 초기화 중… ({slotInfo?.status || "연결 중"})
                    </div>
                  ) : (
                    <div className="camera-placeholder">
                      장치를 선택한 뒤 우측에서 <b>선택 카메라 켜기</b>를 누르세요.
                    </div>
                  )}
                </div>

                <div className="camera-meta-row">
                  <span>
                    {slotInfo?.size_px
                      ? `${slotInfo.size_px[0]}×${slotInfo.size_px[1]}`
                      : "해상도 미확인"}
                  </span>
                  <span>
                    {slotInfo?.error
                      ? "오류"
                      : isRecording
                        ? `저장 ${slotInfo?.written ?? 0} / 누락 ${slotInfo?.dropped ?? 0}`
                        : slotInfo?.device_name || "미연결"}
                  </span>
                </div>
              </MaterialSurface>
            );
          })}
        </div>

        <MaterialSurface className="studio-sidebar">
          <div className="sidebar-section">
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span className="studio-label">장치 및 캡처</span>
              <StatusPill label={`${devices.length}대 감지`} tone={devices.length > 0 ? "success" : "neutral"} />
            </div>
            <CompactAction
              icon={<RefreshCw size={13} />}
              label="카메라 새로 찾기"
              disabled={busy || isConnected}
              className="sidebar-full-btn"
              onClick={() => void scan(backend)}
            />
            <div className="studio-field">
              <span className="studio-label">백엔드</span>
              <CompactSelector
                ariaLabel="Capture Backend"
                value={backend}
                options={BACKEND_OPTIONS}
                disabled={busy || isConnected}
                className="sidebar-full-selector"
                onChange={(val) => {
                  setBackend(val);
                  void scan(val);
                }}
              />
            </div>
            <div className="sidebar-row-2">
              <div className="studio-field">
                <span className="studio-label">해상도</span>
                <CompactSelector
                  ariaLabel="Resolution"
                  value={resolution}
                  options={RES_OPTIONS}
                  disabled={busy || isConnected}
                  className="sidebar-full-selector"
                  onChange={setResolution}
                />
              </div>
              <div className="studio-field">
                <span className="studio-label">프레임</span>
                <CompactSelector
                  ariaLabel="FPS"
                  value={fps}
                  options={FPS_OPTIONS}
                  disabled={busy || isConnected}
                  className="sidebar-full-selector"
                  onChange={setFps}
                />
              </div>
            </div>
          </div>

          <hr className="sidebar-divider" />

          <div className="sidebar-section">
            <span className="studio-label">제어</span>
            <UtilityAction
              variant={isConnected ? "danger" : "solid"}
              icon={isConnected ? <PlugZap size={14} /> : <Plug size={14} />}
              disabled={busy || isRecording}
              className="sidebar-full-btn"
              onClick={() => void toggleConnection()}
            >
              {isConnected ? "카메라 연결 해제" : "선택 카메라 켜기"}
            </UtilityAction>
            <UtilityAction
              variant={isRecording ? "danger" : "solid"}
              icon={isRecording ? <Square size={14} /> : <Video size={14} />}
              disabled={busy || (!isRecording && !status?.recording.ready)}
              className="sidebar-full-btn"
              onClick={() => void toggleRecording()}
            >
              {isRecording ? "전체 녹화 중지" : "전체 녹화 시작"}
            </UtilityAction>
            {isRecording ? (
              <StatusPill
                label={`REC ${status?.recording.elapsed_s.toFixed(1)}s · ${status?.recording.total_written}f`}
                tone="danger"
              />
            ) : status?.recording.done && status.recording.folder_name ? (
              <StatusPill
                label={`저장 완료 · ${status.recording.folder_name}`}
                tone="success"
              />
            ) : null}
          </div>

          <hr className="sidebar-divider" />

          <div className="sidebar-section" style={{ marginTop: "auto" }}>
            <div className="studio-field">
              <span className="studio-label">저장 폴더</span>
              <UtilityInput
                value={saveDir}
                onChange={(e) => setSaveDir(e.target.value)}
                disabled={isRecording}
              />
            </div>
            <div className="studio-field">
              <span className="studio-label">촬영 메모</span>
              <UtilityInput
                value={note}
                placeholder="예: 상단 (-,+,+) 손 가림 테스트"
                onChange={(e) => setNote(e.target.value)}
                disabled={isRecording}
              />
            </div>
            <CompactAction
              icon={<FolderOpen size={13} />}
              label="저장 폴더 열기"
              className="sidebar-full-btn"
              onClick={() => void openFolder()}
            />
          </div>
        </MaterialSurface>
      </div>
    </div>
  );
}
