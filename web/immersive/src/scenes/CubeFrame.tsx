import { useMemo } from "react";
import { BufferGeometry, Float32BufferAttribute } from "three";

const LO = -1.5;
const HI = 1.5;

export function CubeFrame() {
  const geometry = useMemo(() => {
    const geom = new BufferGeometry();
    const vertices = new Float32Array([
      // bottom square (z = LO)
      LO, LO, LO, HI, LO, LO,
      HI, LO, LO, HI, HI, LO,
      HI, HI, LO, LO, HI, LO,
      LO, HI, LO, LO, LO, LO,
      // top square (z = HI)
      LO, LO, HI, HI, LO, HI,
      HI, LO, HI, HI, HI, HI,
      HI, HI, HI, LO, HI, HI,
      LO, HI, HI, LO, LO, HI,
      // verticals
      LO, LO, LO, LO, LO, HI,
      HI, LO, LO, HI, LO, HI,
      HI, HI, LO, HI, HI, HI,
      LO, HI, LO, LO, HI, HI,
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
