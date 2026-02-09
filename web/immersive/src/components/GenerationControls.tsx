const ASSET_PRESETS: Record<string, { theta: number; phi: number }> = {
  "BTC-USD": { theta: 0, phi: Math.PI / 2 },
  "ETH-USD": { theta: Math.PI / 2, phi: Math.PI / 2 },
  "BNB-USD": { theta: 0, phi: 0 },
};

interface Props {
  mode: "real" | "generative";
  viewRunId?: string;
  viewOptions?: { id: string; label: string }[];
  seedAsset: string;
  assetOptions: { id: string; label: string }[];
  theta: number;
  phi: number;
  magnitude: number;
  horizon: number;
  generating: boolean;
  onModeChange: (v: "real" | "generative") => void;
  onViewRunChange?: (v: string) => void;
  onSeedAssetChange: (v: string) => void;
  onThetaChange: (v: number) => void;
  onPhiChange: (v: number) => void;
  onMagnitudeChange: (v: number) => void;
  onHorizonChange: (v: number) => void;
  onGenerate: () => void;
}

export function GenerationControls({
  mode,
  viewRunId,
  viewOptions,
  seedAsset,
  assetOptions,
  theta,
  phi,
  magnitude,
  horizon,
  generating,
  onModeChange,
  onViewRunChange,
  onSeedAssetChange,
  onThetaChange,
  onPhiChange,
  onMagnitudeChange,
  onHorizonChange,
  onGenerate,
}: Props) {
  const handleAssetChange = (asset: string) => {
    onSeedAssetChange(asset);
    const preset = ASSET_PRESETS[asset];
    if (preset) {
      onThetaChange(preset.theta);
      onPhiChange(preset.phi);
    }
  };

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
              onChange={(e) => handleAssetChange(e.target.value)}
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
          <label className="flex flex-col gap-1 text-slate-300">
            {"\u03B8 (azimuthal)"}
            <input
              type="range"
              min={0}
              max={6.2832}
              step={0.01}
              value={theta}
              onChange={(e) => onThetaChange(Number(e.target.value))}
              className="accent-teal-300"
              disabled={mode !== "generative"}
            />
            <div className="text-[10px] text-slate-400">{theta.toFixed(2)} rad</div>
          </label>
          <label className="flex flex-col gap-1 text-slate-300">
            {"\u03C6 (polar)"}
            <input
              type="range"
              min={0}
              max={3.1416}
              step={0.01}
              value={phi}
              onChange={(e) => onPhiChange(Number(e.target.value))}
              className="accent-teal-300"
              disabled={mode !== "generative"}
            />
            <div className="text-[10px] text-slate-400">{phi.toFixed(2)} rad</div>
          </label>
          <label className="col-span-2 flex flex-col gap-1 text-slate-300">
            R (magnitude)
            <input
              type="range"
              min={0.5}
              max={15}
              step={0.1}
              value={magnitude}
              onChange={(e) => onMagnitudeChange(Number(e.target.value))}
              className="accent-teal-300"
              disabled={mode !== "generative"}
            />
            <div className="text-[10px] text-slate-400">{magnitude.toFixed(1)}</div>
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
