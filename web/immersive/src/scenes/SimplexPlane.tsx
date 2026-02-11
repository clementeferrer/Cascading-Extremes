import { DoubleSide } from "three";

export function SimplexPlane() {
  return (
    <group>
      {/* Wireframe unit sphere */}
      <mesh>
        <sphereGeometry args={[1, 32, 24]} />
        <meshStandardMaterial
          color="#4fd1ff"
          emissive="#4fd1ff"
          emissiveIntensity={0.15}
          transparent
          opacity={0.08}
          depthWrite={false}
          side={DoubleSide}
          wireframe
        />
      </mesh>
    </group>
  );
}
