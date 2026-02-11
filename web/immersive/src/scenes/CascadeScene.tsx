import { Canvas } from "@react-three/fiber";
import { Line, OrbitControls, Text } from "@react-three/drei";
import { CubeFrame } from "./CubeFrame";
import { SimplexPlane } from "./SimplexPlane";
import { EventsPoints, EventPoint } from "./EventsPoints";
import { BulkPoints } from "./BulkPoints";
import { CubeMappingParams } from "../utils/geometry";

interface Props {
  events: EventPoint[];
  currentTime: number;
  mapping: CubeMappingParams;
  showSimplex: boolean;
  pointSize: number;
  highlightPositiveOctant: boolean;
  highlightNegativeOctant: boolean;
  highlightPoint?: [number, number, number] | null;
  assetLabels?: string[];
  bulkPoints?: [number, number, number][];
  showBulk?: boolean;
}

// Asset labels placed on each positive axis of the unit sphere
const AXIS_LABEL_POSITIONS: [number, number, number][] = [
  [1.25, 0, 0],    // +X axis
  [0, 1.25, 0],    // +Y axis
  [0, 0, 1.25],    // +Z axis
];

const ASSET_COLORS = ["#38bdf8", "#f97316", "#22c55e"];

function NegativeAxes({ size = 1.2 }: { size?: number }) {
  return (
    <>
      <Line points={[[-size, 0, 0], [0, 0, 0]]} color="#ff4d4d" transparent opacity={0.7} />
      <Line points={[[0, -size, 0], [0, 0, 0]]} color="#5bff5b" transparent opacity={0.7} />
      <Line points={[[0, 0, -size], [0, 0, 0]]} color="#5b8bff" transparent opacity={0.7} />
    </>
  );
}

function AxisLabels({ labels }: { labels: string[] }) {
  return (
    <>
      {labels.slice(0, 3).map((label, i) => (
        <Text
          key={i}
          position={AXIS_LABEL_POSITIONS[i]}
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

export function CascadeScene({
  events,
  currentTime,
  mapping,
  showSimplex,
  pointSize,
  highlightPositiveOctant,
  highlightNegativeOctant,
  highlightPoint,
  assetLabels,
  bulkPoints,
  showBulk,
}: Props) {
  return (
    <Canvas
      camera={{ position: [3.2, 2.4, 3.2], fov: 42 }}
      style={{ width: "100%", height: "100%" }}
    >
      <color attach="background" args={["#0b1020"]} />
      <fog attach="fog" args={["#0b1020", 4.0, 10.0]} />
      <ambientLight intensity={0.5} />
      <directionalLight position={[2.5, 2.5, 2]} intensity={1.1} />
      <CubeFrame />
      <axesHelper args={[1.2]} />
      <NegativeAxes size={1.2} />
      {showSimplex && <SimplexPlane />}
      {assetLabels && assetLabels.length >= 3 && <AxisLabels labels={assetLabels} />}
      {bulkPoints && bulkPoints.length > 0 && <BulkPoints positions={bulkPoints} visible={showBulk ?? false} />}
      <EventsPoints
        events={events}
        currentTime={currentTime}
        mapping={mapping}
        pointSize={pointSize}
        highlightPositiveOctant={highlightPositiveOctant}
        highlightNegativeOctant={highlightNegativeOctant}
      />
      {highlightPoint && (
        <mesh position={highlightPoint}>
          <sphereGeometry args={[0.04, 16, 16]} />
          <meshStandardMaterial color="#e76f51" emissive="#e76f51" emissiveIntensity={0.8} />
        </mesh>
      )}
      <OrbitControls enablePan={false} enableZoom={true} enableRotate={true} target={[0, 0, 0]} />
    </Canvas>
  );
}
