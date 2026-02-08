export interface CubeMappingParams {
  a: number;
  b: number;
  offsetScale: number;
}

// Map (W, R, u_tau) to sphere coordinates within [-1.5, 1.5]^3:
// - W lies on the unit sphere S^{d-1} (L2 norm = 1, can have negative components).
// - Exceedance ratio rho = R / u_tau scales the point radially outward from the sphere.
// - rho = 1 -> on the unit sphere; rho > 1 -> outside; rho < 1 -> inside.
export function mapToCube(w: number[], r: number, uTau: number, params: CubeMappingParams): [number, number, number] {
  const w1 = w[0] ?? 0;
  const w2 = w[1] ?? 0;
  const w3 = w[2] ?? 0;
  const rho = r / Math.max(uTau, 1e-8);
  const logRho = Math.log(Math.max(rho, 1e-8));
  const z = 1 / (1 + Math.exp(-params.a * (logRho - params.b)));
  const scale = 1 + (z - 0.5) * params.offsetScale;
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
