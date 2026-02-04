import { useMemo } from "react";
import { BufferGeometry, Float32BufferAttribute } from "three";

export function CubeFrame() {
  const geometry = useMemo(() => {
    const geom = new BufferGeometry();
    const vertices = new Float32Array([
      // bottom square
      0, 0, 0, 1, 0, 0,
      1, 0, 0, 1, 1, 0,
      1, 1, 0, 0, 1, 0,
      0, 1, 0, 0, 0, 0,
      // top square
      0, 0, 1, 1, 0, 1,
      1, 0, 1, 1, 1, 1,
      1, 1, 1, 0, 1, 1,
      0, 1, 1, 0, 0, 1,
      // verticals
      0, 0, 0, 0, 0, 1,
      1, 0, 0, 1, 0, 1,
      1, 1, 0, 1, 1, 1,
      0, 1, 0, 0, 1, 1,
    ]);
    geom.setAttribute("position", new Float32BufferAttribute(vertices, 3));
    return geom;
  }, []);

  return (
    <lineSegments geometry={geometry}>
      <lineBasicMaterial color="#334155" linewidth={1} />
    </lineSegments>
  );
}
