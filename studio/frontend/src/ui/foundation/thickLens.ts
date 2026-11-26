export type LensProfile = "convex" | "rim";

function clamp01(v: number): number {
  return Math.max(0, Math.min(1, v));
}

function smoothstep(t: number): number {
  const x = clamp01(t);
  return x * x * (3 - 2 * x);
}

/**
 * Computes the signed refraction amplitude along the inward boundary normal
 * using Snell's law on a thick-lens surface profile.
 *
 * Negative return values indicate inward sampling along the inward normal,
 * matching convex edge magnification and rim compression semantics in SIDE.
 */
export function lensAmplitude(
  distanceFromEdge: number,
  bandWidth: number,
  profile: LensProfile = "convex"
): number {
  if (bandWidth <= 0 || distanceFromEdge <= 0 || distanceFromEdge >= bandWidth) {
    return 0;
  }

  const u = clamp01(distanceFromEdge / bandWidth); // 0 at outer edge, 1 at inner calm boundary
  const rimProgress = 1 - u; // 1 at outer edge, 0 at inner calm boundary

  // Surface slope dz/du of a convex or rim meniscus profile
  const slope =
    profile === "rim"
      ? Math.pow(rimProgress, 1.35) * 1.45
      : Math.sin(rimProgress * (Math.PI / 2)) * 1.15;

  const thetaI = Math.atan(slope);
  const nAir = 1.0;
  const nGlass = 1.5;
  const sinThetaT = (nAir / nGlass) * Math.sin(thetaI);
  const thetaT = Math.asin(Math.max(-0.999, Math.min(0.999, sinThetaT)));
  const deflection = Math.tan(thetaI - thetaT);

  // Smooth fade at both the outer 1px anti-aliased silhouette and inner calm boundary
  const outerFade = smoothstep(u / 0.12);
  const innerFade = 1 - smoothstep((u - 0.12) / 0.88);
  const normalized = clamp01(deflection / 0.42);

  return -normalized * outerFade * innerFade;
}
