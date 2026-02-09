import { useMemo, useRef } from "react";
import { BufferAttribute, BufferGeometry, PointsMaterial } from "three";
import { useFrame } from "@react-three/fiber";
import { viridis } from "../utils/color";
import { mapToCube, CubeMappingParams } from "../utils/geometry";

export interface EventPoint {
  t: number;
  w: number[];
  mag: number;
  u_tau: number;
  poc?: number;  // ψ/λ ratio - probability of cascade
}

interface Props {
  events: EventPoint[];
  currentTime: number;
  mapping: CubeMappingParams;
  pointSize: number;
}

export function EventsPoints({ events, currentTime, mapping, pointSize }: Props) {
  const materialRef = useRef<PointsMaterial>(null);

  const { geometry, times } = useMemo(() => {
    const geom = new BufferGeometry();
    if (!events.length) {
      return { geometry: geom, times: [] as number[] };
    }
    const positions = new Float32Array(events.length * 3);
    const times = new Float32Array(events.length);
    const mags = new Float32Array(events.length);
    const colors = new Float32Array(events.length * 3);

    // Use POC (ψ/λ) for color if available, otherwise fall back to magnitude
    const hasCascadeProb = events.some((e) => e.poc !== undefined);

    let colorVals: number[];
    if (hasCascadeProb) {
      // Cascade probability is already in [0, 1] range
      colorVals = events.map((e) => Math.min(1, Math.max(0, e.poc ?? 0)));
    } else {
      // Fall back to normalized magnitude
      const magVals = events.map((e) => e.mag);
      const minMag = Math.min(...magVals);
      const maxMag = Math.max(...magVals);
      const range = Math.max(1e-6, maxMag - minMag);
      colorVals = magVals.map((m) => (m - minMag) / range);
    }

    events.forEach((e, i) => {
      const p = mapToCube(e.w, e.mag, e.u_tau, mapping);
      positions[i * 3 + 0] = p[0];
      positions[i * 3 + 1] = p[1];
      positions[i * 3 + 2] = p[2];
      times[i] = e.t;
      mags[i] = colorVals[i];
      // Color by POC: blue (low) -> yellow/red (high)
      const [r, g, b] = viridis(colorVals[i]);
      colors[i * 3 + 0] = r;
      colors[i * 3 + 1] = g;
      colors[i * 3 + 2] = b;
    });

    geom.setAttribute("position", new BufferAttribute(positions, 3));
    geom.setAttribute("aTime", new BufferAttribute(times, 1));
    geom.setAttribute("aMag", new BufferAttribute(mags, 1));
    geom.setAttribute("color", new BufferAttribute(colors, 3));
    geom.setDrawRange(0, 1);

    return { geometry: geom, times: Array.from(times) };
  }, [events, mapping]);

  useFrame(() => {
    if (!times.length) return;
    let lo = 0;
    let hi = times.length - 1;
    while (lo <= hi) {
      const mid = Math.floor((lo + hi) / 2);
      if (times[mid] <= currentTime) lo = mid + 1;
      else hi = mid - 1;
    }
    const count = Math.max(1, Math.min(times.length, lo));
    geometry.setDrawRange(0, count);
  });

  return (
    <points geometry={geometry} frustumCulled={false}>
      <pointsMaterial
        ref={materialRef}
        vertexColors
        size={pointSize}
        sizeAttenuation
        transparent={false}
        opacity={1.0}
        depthWrite={true}
        depthTest={true}
      />
    </points>
  );
}
