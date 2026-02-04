interface Props {
  mode: "real" | "generative";
  viewRunId?: string;
  viewOptions?: { id: string; label: string }[];
  seedAsset: string;
  assetOptions: { id: string; label: string }[];
  minMag: number;
  minMagMin: number;
  minMagMax: number;
  horizon: number;
  generating: boolean;
  onModeChange: (v: "real" | "generative") => void;
  onViewRunChange?: (v: string) => void;
  onSeedAssetChange: (v: string) => void;
  onMinMagChange: (v: number) => void;
  onHorizonChange: (v: number) => void;
  onGenerate: () => void;
}

export function GenerationControls({
  mode,
  viewRunId,
  viewOptions,
  seedAsset,
  assetOptions,
  minMag,
  minMagMin,
  minMagMax,
  horizon,
  generating,
  onModeChange,
  onViewRunChange,
  onSeedAssetChange,
  onMinMagChange,
  onHorizonChange,
  onGenerate,
}: Props) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 backdrop-blur-md p-3 shadow-lg space-y-2">
      <div className="text-xs uppercase tracking-wide text-slate-400">Generative Controls</div>
      <div className="flex gap-2">
        <button
          className={`rounded-full px-3 py-1 text-xs font-semibold ${mode === "real" ? "bg-teal-500/80 text-white" : "bg-white/5 text-slate-200 border border-white/10"}`}
          onClick={() => onModeChange("real")}
        >
          Real
        </button>
        <button
          className={`rounded-full px-3 py-1 text-xs font-semibold ${mode === "generative" ? "bg-teal-500/80 text-white" : "bg-white/5 text-slate-200 border border-white/10"}`}
          onClick={() => onModeChange("generative")}
        >
          Generative
        </button>
      </div>
      {mode === "real" ? (
        <div className="grid grid-cols-1 gap-2 text-[11px]">
          <label className="flex flex-col gap-1 text-slate-300">
            View run
            <select
              className="rounded-md border border-white/10 bg-white/5 px-2 py-1 text-slate-100"
              value={viewRunId}
              onChange={(e) => onViewRunChange?.(e.target.value)}
            >
              {(viewOptions ?? []).map((o) => (
                <option key={o.id} value={o.id}>
                  {o.label}
                </option>
              ))}
            </select>
          </label>
        </div>
      ) : (
        <div className="grid grid-cols-2 gap-2 text-[11px]">
          <label className="flex flex-col gap-1 text-slate-300">
            Asset
            <select
              className="rounded-md border border-white/10 bg-white/5 px-2 py-1 text-slate-100"
              value={seedAsset}
              onChange={(e) => onSeedAssetChange(e.target.value)}
              disabled={mode !== "generative"}
            >
              {assetOptions.map((o) => (
                <option key={o.id} value={o.id}>
                  {o.label}
                </option>
              ))}
            </select>
          </label>
          <label className="flex flex-col gap-1 text-slate-300">
            Radius (min R)
            <input
              type="range"
              min={minMagMin}
              max={minMagMax}
              step={Math.max((minMagMax - minMagMin) / 100, 1e-4)}
              value={minMag}
              onChange={(e) => onMinMagChange(Number(e.target.value))}
              className="accent-teal-300"
              disabled={mode !== "generative"}
            />
            <div className="text-[10px] text-slate-400">{minMag.toFixed(3)}</div>
          </label>
          <label className="flex flex-col gap-1 text-slate-300">
            Horizon (hours)
            <input
              type="number"
              min={1}
              max={1000}
              value={horizon}
              onChange={(e) => onHorizonChange(Number(e.target.value))}
              className="rounded-md border border-white/10 bg-white/5 px-2 py-1 text-slate-100"
              disabled={mode !== "generative"}
            />
          </label>
        </div>
      )}
      <button
        className="w-full rounded-lg bg-teal-500/90 px-3 py-2 text-xs font-semibold text-white disabled:opacity-60"
        onClick={onGenerate}
        disabled={mode !== "generative" || generating}
      >
        {generating ? "Generating..." : "Generate New Cascade"}
      </button>
    </div>
  );
}
