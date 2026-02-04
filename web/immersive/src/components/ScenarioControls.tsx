interface Props {
  enabled: boolean;
  asset: string;
  assets: string[];
  minMag: number;
  magMin: number;
  magMax: number;
  windowHours: number;
  count: number;
  seedFound: boolean;
  onEnabledChange: (v: boolean) => void;
  onAssetChange: (v: string) => void;
  onMinMagChange: (v: number) => void;
  onWindowHoursChange: (v: number) => void;
}

export function ScenarioControls({
  enabled,
  asset,
  assets,
  minMag,
  magMin,
  magMax,
  windowHours,
  count,
  seedFound,
  onEnabledChange,
  onAssetChange,
  onMinMagChange,
  onWindowHoursChange,
}: Props) {
  return (
    <div className="rounded-xl bg-white/90 p-3 shadow-md space-y-2">
      <div className="text-xs uppercase tracking-wide text-slate">Simulated Cascade Window</div>
      <label className="flex items-center gap-2 text-[11px] text-slate">
        <input
          type="checkbox"
          checked={enabled}
          onChange={(e) => onEnabledChange(e.target.checked)}
        />
        Focus on a seeded window (asset + magnitude + T hours)
      </label>
      <div className="grid grid-cols-2 gap-2">
        <div>
          <div className="text-[11px] text-slate mb-1">Seed asset</div>
          <select
            className="w-full rounded-md border border-slate/20 bg-white px-2 py-1 text-xs"
            value={asset}
            onChange={(e) => onAssetChange(e.target.value)}
            disabled={!enabled}
          >
            {assets.map((a) => (
              <option key={a} value={a}>
                {a}
              </option>
            ))}
          </select>
        </div>
        <div>
          <div className="text-[11px] text-slate mb-1">Window (hours)</div>
          <input
            type="number"
            min={1}
            max={10000}
            value={windowHours}
            onChange={(e) => onWindowHoursChange(Number(e.target.value))}
            className="w-full rounded-md border border-slate/20 bg-white px-2 py-1 text-xs"
            disabled={!enabled}
          />
        </div>
      </div>
      <div>
        <div className="text-[11px] text-slate mb-1">Min magnitude R</div>
        <input
          type="range"
          min={magMin}
          max={magMax}
          step={Math.max((magMax - magMin) / 100, 1e-4)}
          value={minMag}
          onChange={(e) => onMinMagChange(Number(e.target.value))}
          className="w-full accent-ink"
          disabled={!enabled}
        />
        <div className="text-[11px] text-slate">{minMag.toFixed(3)}</div>
      </div>
      <div className="rounded-md bg-ivory/80 px-2 py-1 text-[11px] text-ink">
        {enabled ? (
          seedFound ? (
            <>Estimated extremes in window: <strong>{count}</strong></>
          ) : (
            <>No seed event found for this asset and magnitude.</>
          )
        ) : (
          <>Enable to compute extremes within a fixed horizon.</>
        )}
      </div>
    </div>
  );
}
