import { formatTime } from "../utils/time";

interface Props {
  currentTime: number;
  eventCount: number;
  lambda?: number;
  psi?: number;
  assetCounts?: Record<string, number>;
}

export function KPICards({ currentTime, eventCount, lambda, psi, assetCounts }: Props) {
  const countsText =
    assetCounts && Object.keys(assetCounts).length
      ? Object.entries(assetCounts)
          .sort((a, b) => a[0].localeCompare(b[0]))
          .map(([k, v]) => `${k.replace("-USD", "")} ${v}`)
          .join(" · ")
      : "-";
  const ratio = lambda && lambda > 0 ? (psi ?? 0) / lambda : 0;
  return (
    <div className="grid grid-cols-2 gap-2">
      <div className="rounded-2xl border border-white/10 bg-white/5 p-3 shadow-lg h-[84px] col-span-2">
        <div className="text-xs uppercase tracking-wide text-slate-400">Time</div>
        <div className="text-2xl font-semibold text-white tabular-nums leading-none mt-1">{formatTime(currentTime)}</div>
      </div>
      <div className="rounded-2xl border border-white/10 bg-white/5 p-3 shadow-lg h-[84px]">
        <div className="text-xs uppercase tracking-wide text-slate-400">Events</div>
        <div className="text-2xl font-semibold text-white tabular-nums leading-none mt-1">{eventCount}</div>
      </div>
      <div className="rounded-2xl border border-white/10 bg-white/5 p-3 shadow-lg h-[84px]">
        <div className="text-xs uppercase tracking-wide text-slate-400">Cascade Prob</div>
        <div className="text-2xl font-semibold text-white tabular-nums leading-none mt-1">{ratio.toFixed(3)}</div>
      </div>
      <div className="rounded-2xl border border-white/10 bg-white/5 p-3 shadow-lg col-span-2 h-[68px]">
        <div className="text-xs uppercase tracking-wide text-slate-400">Assets</div>
        <div className="text-sm font-semibold text-white truncate leading-none mt-1">{countsText}</div>
      </div>
    </div>
  );
}
