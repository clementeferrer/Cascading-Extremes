export interface CubeMappingParams {
  a: number;
  b: number;
  offsetScale: number;
}

// Map (R, W) to 3D coordinates within [-1.5, 1.5]^3:
// - W lies on the unit sphere S^{d-1} (L2 norm = 1, can have negative components).
// - R is the L2 magnitude; events have R > u_tau(W) > 1, so all are outside the unit sphere.
// - Log compression: scale = 1 + offsetScale * log(R). R=1 -> on the sphere; R>1 -> outside.
export function mapToCube(w: number[], r: number, _uTau: number, params: CubeMappingParams): [number, number, number] {
  const w1 = w[0] ?? 0;
  const w2 = w[1] ?? 0;
  const w3 = w[2] ?? 0;
  const logR = Math.log(Math.max(r, 1e-8));
  const scale = 1 + params.offsetScale * Math.max(logR, 0);
  const x = clamp(w1 * scale, -1.5, 1.5);
  const y = clamp(w2 * scale, -1.5, 1.5);
  const zc = clamp(w3 * scale, -1.5, 1.5);
  return [x, y, zc];
}

export function clamp(v: number, lo: number, hi: number): number {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}
