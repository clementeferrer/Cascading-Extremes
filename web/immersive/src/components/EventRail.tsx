import { useLayoutEffect, useMemo, useRef } from "react";

interface RailItem {
  id: number;
  t: number;
  asset: string;
  ratio: number;
}

interface Props {
  items: RailItem[];
  currentTime: number;
  onHover: (id: number | null) => void;
  onSelect: (t: number) => void;
  timeScale: number;
}

const ASSET_COLORS: Record<string, string> = {
  BTC: "#38bdf8",
  ETH: "#f97316",
  BNB: "#22c55e",
};

function normalizeAsset(asset: string) {
  return asset.replace("-USD", "").toUpperCase();
}

export function EventRail({ items, currentTime, onHover, onSelect, timeScale }: Props) {
  const railRef = useRef<HTMLDivElement | null>(null);
  const scrollTopRef = useRef(0);

  const visibleItems = useMemo(() => {
    return items
      .filter((item) => item.t <= currentTime + 1.0e-6)
      .map((item) => {
        const assetKey = normalizeAsset(item.asset);
        const color = ASSET_COLORS[assetKey] ?? "#94a3b8";
        return { ...item, assetKey, color };
      });
  }, [items, currentTime]);

  useLayoutEffect(() => {
    if (!railRef.current) return;
    railRef.current.scrollTop = scrollTopRef.current;
  }, [visibleItems.length]);

  return (
    <div className="h-screen w-full overflow-hidden">
      <div className="relative h-full w-full flex flex-col">
        {/* Header */}
        <div className="shrink-0 px-2 py-2 border-b border-white/10">
          <div className="text-[9px] uppercase tracking-[0.2em] text-slate-500 text-center">
            Cascade Flow
          </div>
        </div>

        {/* Scrollable event list */}
        <div
          ref={railRef}
          className="flex-1 h-full overflow-y-auto overscroll-contain px-1 py-2 pr-4 pb-4 scrollbar-thin"
          style={{ scrollbarGutter: "stable" }}
          onScroll={(e) => {
            scrollTopRef.current = e.currentTarget.scrollTop;
          }}
        >
          <div className="flex flex-col items-center gap-0">
            {visibleItems.map((item, idx) => {
              const isHighCascade = item.ratio > 0.5;
              const displayTime = (item.t * timeScale).toFixed(1);

              return (
                <div key={item.id} className="flex flex-col items-center w-full">
                  {/* Arrow connector from previous event */}
                  {idx > 0 && (
                    <div className="flex flex-col items-center py-1">
                      <div
                        className="w-0.5 h-3 rounded-full"
                        style={{
                          background: `linear-gradient(to bottom, ${visibleItems[idx - 1].color}40, ${item.color}40)`
                        }}
                      />
                      <div
                        className="text-[8px]"
                        style={{ color: item.color }}
                      >
                        ▼
                      </div>
                    </div>
                  )}

                  {/* Event box */}
                  <button
                    className={`
                      w-full max-w-[72px] rounded-lg border px-1.5 py-1.5
                      transition-all duration-200 hover:scale-105
                      ${isHighCascade ? 'border-opacity-60' : 'border-opacity-30'}
                    `}
                    style={{
                      borderColor: item.color,
                      background: `linear-gradient(135deg, ${item.color}15, ${item.color}05)`,
                      boxShadow: isHighCascade ? `0 0 12px ${item.color}30` : 'none',
                    }}
                    onMouseEnter={() => onHover(item.id)}
                    onMouseLeave={() => onHover(null)}
                    onClick={() => onSelect(item.t)}
                  >
                    {/* Asset name */}
                    <div
                      className="text-[10px] font-semibold text-center truncate"
                      style={{ color: item.color }}
                    >
                      {item.assetKey}
                    </div>

                    {/* POP bar */}
                    <div className="mt-1 h-1 w-full rounded-full bg-white/10 overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all duration-300"
                        style={{
                          width: `${Math.min(100, item.ratio * 100)}%`,
                          background: item.color,
                          opacity: 0.7 + item.ratio * 0.3,
                        }}
                      />
                    </div>

                    {/* Probability value */}
                    <div className="mt-0.5 text-[8px] text-slate-400 text-center">
                      POP = {item.ratio.toFixed(2)}
                    </div>

                    {/* Time */}
                    <div className="text-[8px] text-slate-500 text-center">
                      t = {displayTime}h
                    </div>
                  </button>
                </div>
              );
            })}

            {/* Empty state */}
            {visibleItems.length === 0 && (
              <div className="text-[10px] text-slate-500 text-center py-4">
                Press Play to see cascade events
              </div>
            )}
          </div>
        </div>

        {/* Footer with event count */}
        <div className="shrink-0 px-2 pt-2 pb-4 border-t border-white/10">
          <div className="text-[9px] text-slate-500 text-center">
            {visibleItems.length} events
          </div>
        </div>
      </div>
    </div>
  );
}
