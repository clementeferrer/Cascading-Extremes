import { useMemo, useRef } from "react";
import { BufferAttribute, BufferGeometry, PointsMaterial } from "three";
import { useFrame } from "@react-three/fiber";

interface Props {
  positions: [number, number, number][];
  visible: boolean;
  pointSize?: number;
}

const TARGET_OPACITY = 0.28;
const FADE_SPEED = 0.7; // opacity units per second

export function BulkPoints({ positions, visible, pointSize = 0.025 }: Props) {
  const materialRef = useRef<PointsMaterial>(null);
  const opacityRef = useRef(0);

  const geometry = useMemo(() => {
    const geom = new BufferGeometry();
    if (!positions.length) return geom;
    const pos = new Float32Array(positions.length * 3);
    positions.forEach((p, i) => {
      pos[i * 3 + 0] = p[0];
      pos[i * 3 + 1] = p[1];
      pos[i * 3 + 2] = p[2];
    });
    geom.setAttribute("position", new BufferAttribute(pos, 3));
    return geom;
  }, [positions]);

  useFrame((_, delta) => {
    const target = visible ? TARGET_OPACITY : 0;
    const current = opacityRef.current;
    if (Math.abs(current - target) > 0.001) {
      const next = current + Math.sign(target - current) * Math.min(FADE_SPEED * delta, Math.abs(target - current));
      opacityRef.current = next;
      if (materialRef.current) {
        materialRef.current.opacity = next;
      }
    }
  });

  if (!positions.length) return null;

  return (
    <points geometry={geometry} frustumCulled={false}>
      <pointsMaterial
        ref={materialRef}
        color="#64748b"
        size={pointSize}
        sizeAttenuation
        transparent
        opacity={0}
        depthWrite={false}
        depthTest={true}
      />
    </points>
  );
}
