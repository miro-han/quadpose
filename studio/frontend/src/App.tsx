import React, { useEffect, useMemo, useState } from "react";
import {
  Layers,
  Minus,
  Palette,
  Square,
  X,
} from "lucide-react";
import {
  AppearancePanel,
  CompactAction,
  CompactPill,
  CompactSegmented,
  DesignSystemProvider,
  IconAction,
  MaterialSurface,
  StatusPill,
  UtilityShell,
  WorkspaceShell,
  createNativeShellAdapter,
} from "./ui";
import { rpc } from "./api";
import { CameraStudioView } from "./views/CameraStudioView";
import { SamTrackingView } from "./views/SamTrackingView";
import { PipelineView } from "./views/PipelineView";

type TabId = "cameras" | "tracking" | "pipeline";

const TAB_OPTIONS: Array<{ value: TabId; label: string }> = [
  { value: "cameras", label: "멀티 카메라" },
  { value: "tracking", label: "SAM 2.1 추적" },
  { value: "pipeline", label: "세션 · Pose" },
];

interface SystemInfo {
  root: string;
  default_recordings_dir: string;
  compute: { cuda: boolean; device: string; name: string };
}

function StudioShell() {
  const [tab, setTab] = useState<TabId>("cameras");
  const [appearanceOpen, setAppearanceOpen] = useState(false);
  const [sysInfo, setSysInfo] = useState<SystemInfo | null>(null);
  const nativeShell = useMemo(() => createNativeShellAdapter(), []);

  useEffect(() => {
    void (async () => {
      try {
        const info = await rpc<SystemInfo>("system.info");
        setSysInfo(info);
      } catch {
        // backend may still be starting
      }
    })();
  }, []);

  const upperBar = (
    <header
      className="studio-topbar"
      onMouseDown={(e) => nativeShell.handleTitlebarMouseDown(e)}
    >
      <div className="studio-brand">
        <MaterialSurface
          profile="compact"
          radius={9}
          className="studio-brand-mark"
        >
          T7
        </MaterialSurface>
        <div style={{ minWidth: 0 }}>
          <div className="studio-brand-title">Team 7 · Plate Tracking</div>
          <div className="studio-brand-sub">
            Multi-View Capture · SAM 2.1 · 6DoF Pipeline
          </div>
        </div>
      </div>

      <div className="studio-topbar-nav">
        <CompactSegmented
          ariaLabel="Studio Workspace Navigation"
          value={tab}
          options={TAB_OPTIONS}
          onChange={(val) => setTab(val as TabId)}
        />
      </div>

      <div className="studio-topbar-right">
        <StatusPill
          label={sysInfo?.compute.cuda ? "CUDA GPU" : "CPU"}
          tone={sysInfo?.compute.cuda ? "success" : "neutral"}
        />

        <CompactAction
          iconOnly
          icon={<Palette size={14} />}
          label="화면 설정"
          active={appearanceOpen}
          onClick={() => setAppearanceOpen(true)}
        />

        {nativeShell.isNative ? (
          <CompactPill>
            <IconAction label="Minimize" onClick={() => void nativeShell.minimize()}>
              <Minus size={14} />
            </IconAction>
            <IconAction
              label="Toggle Maximize"
              onClick={() => void nativeShell.toggleMaximize()}
            >
              <Square size={13} />
            </IconAction>
            <IconAction label="Close" onClick={() => void nativeShell.close()}>
              <X size={14} />
            </IconAction>
          </CompactPill>
        ) : null}
      </div>
    </header>
  );

  return (
    <>
      <WorkspaceShell variant="band" upperBar={upperBar}>
        {tab === "cameras" ? (
          <CameraStudioView
            defaultRecordingsDir={sysInfo?.default_recordings_dir || ""}
          />
        ) : tab === "tracking" ? (
          <SamTrackingView />
        ) : (
          <PipelineView />
        )}
      </WorkspaceShell>

      <UtilityShell
        open={appearanceOpen}
        onClose={() => setAppearanceOpen(false)}
        kicker="환경 설정"
        title="화면 및 스타일"
        subtitle="테마, 표면 재질, 광학 굴절과 배율을 실시간으로 조절합니다."
        headerActions={<Layers size={15} style={{ opacity: 0.65 }} />}
      >
        <AppearancePanel />
      </UtilityShell>
    </>
  );
}

export function App() {
  return (
    <DesignSystemProvider
      initialAppearance={{
        theme: "dark",
        surfaceStyle: "liquid",
        opticalEffects: true,
        reduceMotion: false,
        uiScale: 1.0,
        density: "compact",
      }}
      syncDocumentRoot
    >
      <StudioShell />
    </DesignSystemProvider>
  );
}
