import { formatTime } from "../utils/time";

interface Props {
  currentTime: number;
  lambda?: number;
  psi?: number;
}

export function KPICards({ currentTime, lambda, psi }: Props) {
  const ratio = lambda && lambda > 0 ? (psi ?? 0) / lambda : 0;
  return (
    <div className="grid grid-cols-2 gap-2">
      <div className="rounded-2xl border border-white/10 bg-white/5 p-3 shadow-lg h-[84px] col-span-2">
        <div className="text-xs uppercase tracking-wide text-slate-400">Time</div>
        <div className="text-2xl font-semibold text-white tabular-nums leading-none mt-1">{formatTime(currentTime)}</div>
      </div>
      <div className="rounded-2xl border border-white/10 bg-white/5 p-3 shadow-lg h-[84px] col-span-2">
        <div className="text-xs uppercase tracking-wide text-slate-400">Probability of Propagation</div>
        <div className="text-2xl font-semibold text-white tabular-nums leading-none mt-1">{ratio.toFixed(3)}</div>
      </div>
    </div>
  );
}
