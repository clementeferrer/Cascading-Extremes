interface TimelineItem {
  id: number;
  t: number;
  asset: string;
  ratio: number;
  context: string;
}

interface Props {
  items: TimelineItem[];
  currentTime: number;
  timeScale: number;
  onHover: (id: number | null) => void;
  onSelect: (t: number) => void;
}

export function VerticalEventTimeline({ items, currentTime, timeScale, onHover, onSelect }: Props) {
  if (!items.length) {
    return (
      <div className="h-full w-full rounded-2xl border border-white/10 bg-white/5 backdrop-blur-md shadow-lg p-4 text-sm text-slate-300">
        No cascade events loaded yet.
      </div>
    );
  }
  return (
    <div className="h-full w-full rounded-2xl border border-white/10 bg-white/5 backdrop-blur-md shadow-lg p-4 overflow-y-auto">
      <div className="text-xs uppercase tracking-wide text-slate-400 mb-3">Event Timeline</div>
      <div className="space-y-3">
        {items.map((item) => {
          const active = item.t <= currentTime + 1e-6;
          return (
            <div
              key={item.id}
              className={`rounded-xl border px-3 py-2 transition hover:shadow-sm hover:border-white/30 ${active ? "border-white/20 bg-white/10" : "border-white/5 bg-white/5"}`}
              onMouseEnter={() => onHover(item.id)}
              onMouseLeave={() => onHover(null)}
              onClick={() => onSelect(item.t)}
              style={{ cursor: "pointer" }}
            >
              <div className="flex items-center justify-between">
                <div className="text-[11px] text-slate-400">t = {(item.t * timeScale).toFixed(2)}h</div>
                <div className="text-[11px] text-slate-400">POC {item.ratio.toFixed(2)}</div>
              </div>
              <div className="mt-1 text-sm font-semibold text-white">{item.asset}</div>
              <div className="text-[11px] text-slate-400">{item.context}</div>
              <div className="mt-2 h-1 w-full rounded-full bg-white/10">
                <div
                  className="h-1 rounded-full bg-teal-400"
                  style={{ width: `${Math.min(100, Math.max(0, item.ratio * 100))}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
