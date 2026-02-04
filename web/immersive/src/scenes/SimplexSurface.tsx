import { useMemo } from "react";
import { BufferGeometry, Float32BufferAttribute } from "three";

export function SimplexSurface() {
  const geometry = useMemo(() => {
    const geom = new BufferGeometry();
    const shift = 1 / 3;
    const vertices = new Float32Array([
      1 - shift, 0 - shift, 0 - shift,
      0 - shift, 1 - shift, 0 - shift,
      0 - shift, 0 - shift, 1 - shift,
    ]);
    geom.setAttribute("position", new Float32BufferAttribute(vertices, 3));
    geom.setIndex([0, 1, 2]);
    geom.computeVertexNormals();
    return geom;
  }, []);

  return (
    <group>
      <mesh geometry={geometry}>
        <meshStandardMaterial color="#f8fafc" transparent opacity={0.35} depthWrite={false} />
      </mesh>
      <lineSegments>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={6}
            array={new Float32Array([
              1 - 1 / 3, 0 - 1 / 3, 0 - 1 / 3, 0 - 1 / 3, 1 - 1 / 3, 0 - 1 / 3,
              0 - 1 / 3, 1 - 1 / 3, 0 - 1 / 3, 0 - 1 / 3, 0 - 1 / 3, 1 - 1 / 3,
              0 - 1 / 3, 0 - 1 / 3, 1 - 1 / 3, 1 - 1 / 3, 0 - 1 / 3, 0 - 1 / 3,
            ])}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial color="#94a3b8" linewidth={1} />
      </lineSegments>
    </group>
  );
}
