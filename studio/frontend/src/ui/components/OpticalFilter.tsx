import React from "react";

export interface OpticalFilterProps {
  id?: string;
  scale?: number;
}

/**
 * Shared SVG refraction filter definition (`#context-edge-refract` by default)
 * used by `.miro-optical-glass` / `.miro-liquid-glass` utility classes.
 */
export function OpticalFilter({
  id = "context-edge-refract",
  scale = 4
}: OpticalFilterProps) {
  return (
    <svg className="miro-optical-svg-defs optical-svg-defs" aria-hidden="true">
      <defs>
        <filter
          id={id}
          x="-5%"
          y="-5%"
          width="110%"
          height="110%"
          colorInterpolationFilters="sRGB"
        >
          <feTurbulence
            type="fractalNoise"
            baseFrequency="0.025 0.04"
            numOctaves={1}
            seed={3}
            result="edgeNoise"
          />
          <feGaussianBlur in="edgeNoise" stdDeviation="2" result="smoothEdge" />
          <feDisplacementMap
            in="SourceGraphic"
            in2="smoothEdge"
            scale={scale}
            xChannelSelector="R"
            yChannelSelector="G"
          />
        </filter>
      </defs>
    </svg>
  );
}
