export interface CubeMappingParams {
  a: number;
  b: number;
  offsetScale: number;
}

// Map (R, W) to 3D position: simply R * W.
// Data is already scaled by log(n/2), so no transform needed.
// Unit sphere at ||x||=1 is the geometric reference.
export function mapToCube(w: number[], r: number, _uTau: number, _params: CubeMappingParams): [number, number, number] {
  const x = (w[0] ?? 0) * r;
  const y = (w[1] ?? 0) * r;
  const z = (w[2] ?? 0) * r;
  return [x, y, z];
}

export function clamp(v: number, lo: number, hi: number): number {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}
