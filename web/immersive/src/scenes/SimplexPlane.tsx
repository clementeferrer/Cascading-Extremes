import { useMemo } from "react";
import { BufferGeometry, Float32BufferAttribute, EdgesGeometry, LineBasicMaterial, DoubleSide } from "three";

export function SimplexPlane() {
  const geometry = useMemo(() => {
    const geom = new BufferGeometry();
    const vertices = new Float32Array([
      1, 0, 0,
      0, 1, 0,
      0, 0, 1,
    ]);
    geom.setAttribute("position", new Float32BufferAttribute(vertices, 3));
    geom.setIndex([0, 1, 2]);
    geom.computeVertexNormals();
    return geom;
  }, []);

  const edges = useMemo(() => new EdgesGeometry(geometry), [geometry]);
  const lineMaterial = useMemo(
    () => new LineBasicMaterial({ color: "#93c5fd", transparent: true, opacity: 1.0 }),
    []
  );

  return (
    <group>
      <mesh geometry={geometry}>
        <meshStandardMaterial
          color="#4fd1ff"
          emissive="#4fd1ff"
          emissiveIntensity={0.15}
          transparent
          opacity={0.25}
          depthWrite={false}
          side={DoubleSide}
        />
      </mesh>
      <lineSegments geometry={edges} material={lineMaterial} />
    </group>
  );
}
