const stops = [
  [68, 1, 84],
  [71, 44, 122],
  [59, 81, 139],
  [44, 113, 142],
  [33, 144, 141],
  [39, 173, 129],
  [92, 200, 99],
  [170, 220, 50],
  [253, 231, 37],
];

export function viridis(t: number): [number, number, number] {
  const clamped = Math.min(1, Math.max(0, t));
  const idx = Math.floor(clamped * (stops.length - 1));
  const next = Math.min(stops.length - 1, idx + 1);
  const local = clamped * (stops.length - 1) - idx;
  const a = stops[idx];
  const b = stops[next];
  const r = a[0] + (b[0] - a[0]) * local;
  const g = a[1] + (b[1] - a[1]) * local;
  const bch = a[2] + (b[2] - a[2]) * local;
  return [r / 255, g / 255, bch / 255];
}
