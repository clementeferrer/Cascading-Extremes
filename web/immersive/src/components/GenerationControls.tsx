import { useEffect, useState, type FocusEvent } from "react";

interface Props {
  mode: "real" | "generative";
  viewRunId?: string;
  viewOptions?: { id: string; label: string }[];
  returns: Record<string, number>;
  horizon: number;
  temperature: number;
  generating: boolean;
  generateError: string | null;
  onModeChange: (v: "real" | "generative") => void;
  onViewRunChange?: (v: string) => void;
  onReturnsChange: (asset: string, value: number) => void;
  onHorizonChange: (v: number) => void;
  onTemperatureChange: (v: number) => void;
  onGenerate: () => void;
}

const ASSET_LABELS: Record<string, string> = {
  "BTC-USD": "BTC-USD",
  "ETH-USD": "ETH-USD",
  "BNB-USD": "BNB-USD",
};
const MAX_HORIZON_HOURS = 17468;

export function GenerationControls({
  mode,
  viewRunId,
  viewOptions,
  returns,
  horizon,
  temperature,
  generating,
  generateError,
  onModeChange,
  onViewRunChange,
  onReturnsChange,
  onHorizonChange,
  onTemperatureChange,
  onGenerate,
}: Props) {
  const assets = Object.keys(ASSET_LABELS);
  const [localReturns, setLocalReturns] = useState<Record<string, string>>(() =>
    Object.fromEntries(assets.map((asset) => [asset, String(returns[asset] ?? 0)])) as Record<string, string>
  );
  const [localHorizon, setLocalHorizon] = useState<string>(String(horizon));
  const [localTemperature, setLocalTemperature] = useState<string>(String(temperature));

  useEffect(() => {
    setLocalReturns(
      Object.fromEntries(assets.map((asset) => [asset, String(returns[asset] ?? 0)])) as Record<string, string>
    );
  }, [returns]);

  useEffect(() => {
    setLocalHorizon(String(horizon));
  }, [horizon]);

  useEffect(() => {
    setLocalTemperature(String(temperature));
  }, [temperature]);

  const normalizeDraft = (value: string) => value.replace(",", ".");
  const isNumericDraft = (value: string) => /^-?\d*\.?\d*$/.test(value);
  const toNumber = (value: string): number | null => {
    if (value === "" || value === "-" || value === "." || value === "-.") return null;
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  };
  const onInputFocus = (event: FocusEvent<HTMLInputElement>) => {
    event.currentTarget.select();
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
        <div className="space-y-2 text-[11px]">
          {assets.map((asset) => (
            <label key={asset} className="flex items-center gap-2 text-slate-300">
              <span className="w-[72px] shrink-0 text-slate-200 font-medium">{ASSET_LABELS[asset]}</span>
              <div className="relative flex-1">
                <input
                  type="text"
                  inputMode="decimal"
                  value={localReturns[asset] ?? ""}
                  onFocus={onInputFocus}
                  onChange={(e) => {
                    const next = normalizeDraft(e.target.value);
                    if (!isNumericDraft(next)) return;
                    setLocalReturns((prev) => ({ ...prev, [asset]: next }));
                    const parsed = toNumber(next);
                    if (parsed != null) onReturnsChange(asset, parsed);
                  }}
                  onBlur={() => {
                    const parsed = toNumber(localReturns[asset] ?? "");
                    const committed = parsed ?? (returns[asset] ?? 0);
                    setLocalReturns((prev) => ({ ...prev, [asset]: String(committed) }));
                    onReturnsChange(asset, committed);
                  }}
                  className="w-full rounded-md border border-white/10 bg-white/5 px-2 py-1 pr-6 text-slate-100"
                  disabled={generating}
                />
                <span className="absolute right-2 top-1/2 -translate-y-1/2 text-slate-400 text-[10px]">%</span>
              </div>
            </label>
          ))}
          <label className="flex items-center gap-2 text-slate-300">
            <span className="w-[72px] shrink-0 text-slate-200 font-medium">Horizon</span>
            <div className="relative flex-1">
              <input
                type="text"
                inputMode="numeric"
                value={localHorizon}
                onFocus={onInputFocus}
                onChange={(e) => {
                  const next = normalizeDraft(e.target.value);
                  if (!isNumericDraft(next)) return;
                  setLocalHorizon(next);
                  const parsed = toNumber(next);
                  if (parsed != null) onHorizonChange(Math.max(1, Math.min(MAX_HORIZON_HOURS, Math.round(parsed))));
                }}
                onBlur={() => {
                  const parsed = toNumber(localHorizon);
                  const committed =
                    parsed != null ? Math.max(1, Math.min(MAX_HORIZON_HOURS, Math.round(parsed))) : horizon;
                  setLocalHorizon(String(committed));
                  onHorizonChange(committed);
                }}
                className="w-full rounded-md border border-white/10 bg-white/5 px-2 py-1 pr-6 text-slate-100"
                disabled={generating}
              />
              <span className="absolute right-2 top-1/2 -translate-y-1/2 text-slate-400 text-[10px]">hrs</span>
            </div>
          </label>
          <label className="flex items-center gap-2 text-slate-300">
            <span className="w-[72px] shrink-0 text-slate-200 font-medium">Temperature</span>
            <div className="relative flex-1">
              <input
                type="text"
                inputMode="decimal"
                value={localTemperature}
                onFocus={onInputFocus}
                onChange={(e) => {
                  const next = normalizeDraft(e.target.value);
                  if (!isNumericDraft(next)) return;
                  setLocalTemperature(next);
                  const parsed = toNumber(next);
                  if (parsed != null) onTemperatureChange(Math.max(0.2, Math.min(3.0, parsed)));
                }}
                onBlur={() => {
                  const parsed = toNumber(localTemperature);
                  const committed = parsed != null ? Math.max(0.2, Math.min(3.0, parsed)) : temperature;
                  setLocalTemperature(String(committed));
                  onTemperatureChange(committed);
                }}
                className="w-full rounded-md border border-white/10 bg-white/5 px-2 py-1 text-slate-100"
                disabled={generating}
              />
            </div>
          </label>
          {generateError && (
            <div className="text-[10px] text-red-400 leading-tight">{generateError}</div>
          )}
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
