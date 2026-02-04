export interface CubeMappingParams {
  a: number;
  b: number;
  offsetScale: number;
}

// Map (W, R, u_tau) to cube coordinates:
// - W lies on the simplex plane x + y + z = 1 within [0,1]^3.
// - Exceedance ratio rho = R / u_tau shifts the point along the simplex normal.
// - rho = 1 -> stays on the plane; rho > 1 -> moves above; rho < 1 -> moves below.
export function mapToCube(w: number[], r: number, uTau: number, params: CubeMappingParams): [number, number, number] {
  const w1 = w[0] ?? 0;
  const w2 = w[1] ?? 0;
  const w3 = w[2] ?? 0;
  const rho = r / Math.max(uTau, 1e-8);
  const logRho = Math.log(Math.max(rho, 1e-8));
  const z = 1 / (1 + Math.exp(-params.a * (logRho - params.b)));
  const s = (z - 0.5) * params.offsetScale;
  const n = 1 / Math.sqrt(3);
  const x = clamp01(w1 + s * n);
  const y = clamp01(w2 + s * n);
  const zc = clamp01(w3 + s * n);
  return [x, y, zc];
}

export function clamp01(v: number): number {
  if (v < 0) return 0;
  if (v > 1) return 1;
  return v;
}
