import React from "react";
import { Minus, Plus, RotateCcw } from "lucide-react";
import { MaterialStyle } from "../foundation/liquidProfiles";
import { useDesignSystem } from "../foundation/theme";
import { DensityMode, ThemeMode, UI_SCALE_STEP } from "../foundation/tokens";
import { CompactSegmented } from "./CompactControls";
import { OpticalToggle } from "./OpticalToggle";
import { UtilityAction, ZoomIndicator } from "./UtilityControls";
import { UtilityRow, UtilitySection } from "./UtilityShell";

export interface AppearancePanelProps {
  className?: string;
}

const THEME_OPTIONS: Array<{ value: ThemeMode; label: string }> = [
  { value: "light", label: "라이트" },
  { value: "dark", label: "다크" },
  { value: "system", label: "시스템" }
];

const SURFACE_OPTIONS: Array<{ value: MaterialStyle; label: string }> = [
  { value: "flat", label: "플랫" },
  { value: "frosted", label: "반투명" },
  { value: "liquid", label: "리퀴드" }
];

const DENSITY_OPTIONS: Array<{ value: DensityMode; label: string }> = [
  { value: "compact", label: "조밀하게" },
  { value: "comfortable", label: "여유롭게" }
];

/**
 * Reusable Appearance & Material calibration controls (Korean localized).
 */
export function AppearancePanel({ className = "" }: AppearancePanelProps) {
  const {
    theme,
    surfaceStyle,
    opticalEffects,
    reduceMotion,
    uiScale,
    density,
    setTheme,
    setSurfaceStyle,
    setOpticalEffects,
    setReduceMotion,
    stepUiScale,
    resetUiScale,
    setDensity
  } = useDesignSystem();

  return (
    <div
      className={className}
      style={{ display: "flex", flexDirection: "column", gap: "var(--miro-space-4)" }}
    >
      <UtilitySection label="색상 및 재질">
        <UtilityRow
          title="테마"
          description="라이트, 다크 또는 운영체제 기본 테마를 선택합니다."
          control={
            <CompactSegmented
              ariaLabel="테마 선택"
              value={theme}
              options={THEME_OPTIONS}
              onChange={setTheme}
            />
          }
        />
        <UtilityRow
          title="표면 스타일"
          description="불투명 플랫, 블러 반투명, 또는 리퀴드 글래스 스타일을 전환합니다."
          control={
            <CompactSegmented
              ariaLabel="표면 스타일"
              value={surfaceStyle}
              options={SURFACE_OPTIONS}
              onChange={setSurfaceStyle}
            />
          }
        />
        <UtilityRow
          title="광학 경계 굴절"
          description="리퀴드 글래스 테두리에 SVG 광학 굴절 효과를 적용합니다."
          control={
            <OpticalToggle
              label="광학 경계 굴절"
              checked={opticalEffects}
              onChange={setOpticalEffects}
            />
          }
        />
      </UtilitySection>

      <UtilitySection label="화면 밀도 및 배율">
        <UtilityRow
          title="컨텐츠 밀도"
          description="작업 공간의 여백과 컨트롤 간격을 조절합니다."
          control={
            <CompactSegmented
              ariaLabel="컨텐츠 밀도"
              value={density}
              options={DENSITY_OPTIONS}
              onChange={setDensity}
            />
          }
        />
        <UtilityRow
          title="모션 줄이기"
          description="전환 애니메이션을 꺼서 상태 변화를 즉시 반영합니다."
          control={
            <OpticalToggle
              label="모션 줄이기"
              checked={reduceMotion}
              onChange={setReduceMotion}
            />
          }
        />
        <UtilityRow
          title="인터페이스 배율"
          description="전체 글꼴과 UI 크기를 80%~160% 범위로 조절합니다."
          control={
            <div style={{ display: "inline-flex", alignItems: "center", gap: "0.35rem" }}>
              <UtilityAction
                aria-label="축소"
                onClick={() => stepUiScale(-UI_SCALE_STEP)}
              >
                <Minus size={14} aria-hidden="true" />
              </UtilityAction>
              <ZoomIndicator scale={uiScale} />
              <UtilityAction
                aria-label="확대"
                onClick={() => stepUiScale(UI_SCALE_STEP)}
              >
                <Plus size={14} aria-hidden="true" />
              </UtilityAction>
              <UtilityAction
                aria-label="배율 초기화"
                onClick={resetUiScale}
              >
                <RotateCcw size={13} aria-hidden="true" />
              </UtilityAction>
            </div>
          }
        />
      </UtilitySection>
    </div>
  );
}
