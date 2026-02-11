import { useEffect, useMemo, useState } from "react";

interface Stat {
  label: string;
  value: number;
  suffix?: string;
  precision?: number;
}

interface Props {
  stats: Stat[];
}

function useCountUp(target: number, duration = 1200) {
  const [value, setValue] = useState(0);
  const start = useMemo(() => performance.now(), []);
  useEffect(() => {
    let raf = 0;
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / duration);
      setValue(target * t);
      if (t < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, duration, start]);
  return value;
}

function CounterCard({ label, value, suffix, precision }: Stat) {
  const val = useCountUp(value);
  const resolvedPrecision =
    typeof precision === "number"
      ? precision
      : Number.isInteger(value)
        ? 0
        : 2;
  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
      <div className="text-xs uppercase tracking-[0.3em] text-slate-500">{label}</div>
      <div className="mt-3 text-3xl font-semibold text-white">
        {val.toFixed(resolvedPrecision)}
        {suffix ?? ""}
      </div>
    </div>
  );
}

export function Counters({ stats }: Props) {
  const mdColsClass = stats.length <= 2 ? "md:grid-cols-2" : "md:grid-cols-3";
  const maxWidthClass = stats.length <= 2 ? "max-w-4xl" : "max-w-3xl";

  return (
    <section className="py-16 border-t border-white/5">
      <div className="flex items-center justify-between">
        <h3 className="text-xl font-semibold text-white">By the Numbers</h3>
        <div className="text-xs uppercase tracking-[0.3em] text-slate-500">Credibility</div>
      </div>
      <div className={`mt-8 grid gap-4 ${mdColsClass} ${maxWidthClass} mx-auto`}>
        {stats.map((stat) => (
          <CounterCard key={stat.label} {...stat} />
        ))}
      </div>
    </section>
  );
}
