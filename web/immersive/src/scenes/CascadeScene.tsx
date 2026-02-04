import { Canvas } from "@react-three/fiber";
import { OrbitControls, Text } from "@react-three/drei";
import { CubeFrame } from "./CubeFrame";
import { SimplexPlane } from "./SimplexPlane";
import { EventsPoints, EventPoint } from "./EventsPoints";
import { CubeMappingParams } from "../utils/geometry";

interface Props {
  events: EventPoint[];
  currentTime: number;
  mapping: CubeMappingParams;
  showSimplex: boolean;
  pointSize: number;
  highlightPoint?: [number, number, number] | null;
  assetLabels?: string[];
}

// Simplex vertices in 3D space (corners of the unit simplex)
const SIMPLEX_VERTICES: [number, number, number][] = [
  [1.08, -0.08, -0.08],   // First asset (e.g., BTC) - offset slightly outward
  [-0.08, 1.08, -0.08],   // Second asset (e.g., ETH)
  [-0.08, -0.08, 1.08],   // Third asset (e.g., BNB)
];

const ASSET_COLORS = ["#38bdf8", "#f97316", "#22c55e"];

function SimplexLabels({ labels }: { labels: string[] }) {
  return (
    <>
      {labels.slice(0, 3).map((label, i) => (
        <Text
          key={i}
          position={SIMPLEX_VERTICES[i]}
          fontSize={0.08}
          color={ASSET_COLORS[i]}
          anchorX="center"
          anchorY="middle"
          outlineWidth={0.004}
          outlineColor="#0b1020"
        >
          {label.replace("-USD", "")}
        </Text>
      ))}
    </>
  );
}

export function CascadeScene({ events, currentTime, mapping, showSimplex, pointSize, highlightPoint, assetLabels }: Props) {
  return (
    <Canvas
      camera={{ position: [2.6, 2.0, 2.8], fov: 42 }}
      style={{ width: "100%", height: "100%" }}
    >
      <color attach="background" args={["#0b1020"]} />
      <fog attach="fog" args={["#0b1020", 3.0, 8.0]} />
      <ambientLight intensity={0.5} />
      <directionalLight position={[2.5, 2.5, 2]} intensity={1.1} />
      <CubeFrame />
      {showSimplex && <SimplexPlane />}
      {assetLabels && assetLabels.length >= 3 && <SimplexLabels labels={assetLabels} />}
      <EventsPoints events={events} currentTime={currentTime} mapping={mapping} pointSize={pointSize} />
      {highlightPoint && (
        <mesh position={highlightPoint}>
          <sphereGeometry args={[0.04, 16, 16]} />
          <meshStandardMaterial color="#e76f51" emissive="#e76f51" emissiveIntensity={0.8} />
        </mesh>
      )}
      <OrbitControls enablePan={false} enableZoom={false} enableRotate={true} target={[0.5, 0.5, 0.5]} />
    </Canvas>
  );
}
