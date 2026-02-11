import { useMemo, useRef } from "react";
import { BufferAttribute, BufferGeometry, PointsMaterial } from "three";
import { useFrame } from "@react-three/fiber";
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

    events.forEach((e, i) => {
      const p = mapToCube(e.w, e.mag, e.u_tau, mapping);
      positions[i * 3 + 0] = p[0];
      positions[i * 3 + 1] = p[1];
      positions[i * 3 + 2] = p[2];
      times[i] = e.t;
    });

    geom.setAttribute("position", new BufferAttribute(positions, 3));
    geom.setAttribute("aTime", new BufferAttribute(times, 1));
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
        color="#ef4444"
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
