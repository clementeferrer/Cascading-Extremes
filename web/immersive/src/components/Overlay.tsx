import { KPICards } from "./KPICards";
import { Sparklines } from "./Sparklines";
import { Controls } from "./Controls";
import { CompareToggle } from "./CompareToggle";
import { formatTime } from "../utils/time";
import { CascadeSignalPanel } from "./CascadeSignalPanel";
import { ScenarioControls } from "./ScenarioControls";

interface Props {
  maxTime: number;
  currentTime: number;
  displayTime: number;
  displayMaxTime: number;
  lambda?: number;
  psi?: number;
  sparkSeries: { name: string; color: string; data: [number, number][] }[];
  ratioSeries: [number, number][];
  assetProbSeries: [number, number, number][];
  assetLabels: string[];
  timeScale: number;
  scenarioVisible: boolean;
  scenarioEnabled: boolean;
  scenarioAsset: string;
  scenarioAssets: string[];
  scenarioMinMag: number;
  scenarioMagMin: number;
  scenarioMagMax: number;
  scenarioWindowHours: number;
  scenarioCount: number;
  scenarioSeedFound: boolean;
  onScenarioEnabledChange: (v: boolean) => void;
  onScenarioAssetChange: (v: string) => void;
  onScenarioMinMagChange: (v: number) => void;
  onScenarioWindowHoursChange: (v: number) => void;
  runs: { run_id: string; source?: string }[];
  activeRun: string;
  onRunChange: (id: string) => void;
}

export function Overlay({
  maxTime,
  currentTime,
  displayTime,
  displayMaxTime,
  lambda,
  psi,
  sparkSeries,
  ratioSeries,
  assetProbSeries,
  assetLabels,
  timeScale,
  scenarioVisible,
  scenarioEnabled,
  scenarioAsset,
  scenarioAssets,
  scenarioMinMag,
  scenarioMagMin,
  scenarioMagMax,
  scenarioWindowHours,
  scenarioCount,
  scenarioSeedFound,
  onScenarioEnabledChange,
  onScenarioAssetChange,
  onScenarioMinMagChange,
  onScenarioWindowHoursChange,
  runs,
  activeRun,
  onRunChange,
}: Props) {
  const dev = import.meta.env.DEV;
  return (
    <div
      className="pointer-events-none absolute inset-0 flex flex-col"
      style={{ position: "absolute", inset: 0, zIndex: 10, color: "#0f172a" }}
    >
      {dev && (
        <div style={{ position: "absolute", top: 8, left: 8, zIndex: 9999, background: "#ffffff", padding: "4px 6px", fontSize: 11, borderRadius: 6 }}>
          UI OK
        </div>
      )}
      <div className="pointer-events-auto flex items-center justify-between px-8 pt-6">
        <div>
          <div className="text-sm uppercase tracking-[0.3em] text-slate">Live Paper</div>
          <div className="text-3xl font-semibold text-ink font-serif">Cascading Extremes</div>
          <div className="text-sm text-slate">Time {formatTime(displayTime)} / {formatTime(displayMaxTime)}</div>
        </div>
        <CompareToggle runs={runs} activeRun={activeRun} onChange={onRunChange} />
      </div>

      <div className="pointer-events-auto mt-6 px-8">
        <KPICards
          currentTime={displayTime}
          lambda={lambda}
          psi={psi}
        />
      </div>

      <div className="pointer-events-auto mt-4 px-8 flex gap-4 items-start">
        <div className="w-[360px] space-y-4">
          <Sparklines
            series={sparkSeries}
            currentTime={currentTime}
            timeScale={timeScale}
            startDatetimeUtc={null}
          />
          {scenarioVisible && (
            <ScenarioControls
              enabled={scenarioEnabled}
              asset={scenarioAsset}
              assets={scenarioAssets}
              minMag={scenarioMinMag}
              magMin={scenarioMagMin}
              magMax={scenarioMagMax}
              windowHours={scenarioWindowHours}
              count={scenarioCount}
              seedFound={scenarioSeedFound}
              onEnabledChange={onScenarioEnabledChange}
              onAssetChange={onScenarioAssetChange}
              onMinMagChange={onScenarioMinMagChange}
              onWindowHoursChange={onScenarioWindowHoursChange}
            />
          )}
        </div>
        <div className="max-w-xl flex-1">
          <CascadeSignalPanel
            ratioSeries={ratioSeries}
            assetProbSeries={assetProbSeries}
            assetLabels={assetLabels}
            currentTime={currentTime}
            timeScale={timeScale}
          />
        </div>
      </div>

      <div className="pointer-events-auto mt-auto px-8 pb-8 flex justify-between items-end">
        <Controls maxTime={maxTime} />
        <div className="rounded-xl bg-white/90 px-4 py-3 shadow-md text-xs text-slate">
          Direction on simplex, color encodes magnitude R, time via playback.
        </div>
      </div>
    </div>
  );
}
