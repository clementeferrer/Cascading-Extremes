interface Props {
  showSimplex: boolean;
  onShowSimplexChange: (v: boolean) => void;
  pointSize: "small" | "medium" | "large";
  onPointSizeChange: (v: "small" | "medium" | "large") => void;
  highlightPositiveOctant: boolean;
  highlightNegativeOctant: boolean;
  onHighlightPositiveOctantChange: (v: boolean) => void;
  onHighlightNegativeOctantChange: (v: boolean) => void;
}

export function GeometryControls({
  showSimplex,
  onShowSimplexChange,
  pointSize,
  onPointSizeChange,
  highlightPositiveOctant,
  highlightNegativeOctant,
  onHighlightPositiveOctantChange,
  onHighlightNegativeOctantChange,
}: Props) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 backdrop-blur-md p-3 shadow-lg space-y-2">
      <div className="text-xs uppercase tracking-wide text-slate-400">Geometry</div>
      <label className="flex items-center gap-2 text-[11px] text-slate-300">
        <input type="checkbox" checked={showSimplex} onChange={(e) => onShowSimplexChange(e.target.checked)} />
        Show unit sphere overlay
      </label>
      <label className="flex flex-col gap-1 text-[11px] text-slate-300">
        Point size
        <select
          className="rounded-md border border-white/10 bg-white/5 px-2 py-1 text-slate-100"
          value={pointSize}
          onChange={(e) => onPointSizeChange(e.target.value as "small" | "medium" | "large")}
        >
          <option value="small">Small</option>
          <option value="medium">Medium</option>
          <option value="large">Large</option>
        </select>
      </label>
      <div className="space-y-2 pt-1">
        <div className="text-[11px] tracking-wide text-slate-400">Octants</div>
        <div className="grid grid-cols-2 gap-2">
          <button
            type="button"
            className={`rounded-full border px-2 py-1 text-[10px] transition-colors ${
              highlightPositiveOctant
                ? "border-teal-300/70 bg-teal-400/20 text-teal-100"
                : "border-white/10 bg-white/5 text-slate-300 hover:border-teal-400/30 hover:text-teal-100"
            }`}
            onClick={() => onHighlightPositiveOctantChange(!highlightPositiveOctant)}
          >
            Log Returns (+)
          </button>
          <button
            type="button"
            className={`rounded-full border px-2 py-1 text-[10px] transition-colors ${
              highlightNegativeOctant
                ? "border-rose-300/70 bg-rose-400/20 text-rose-100"
                : "border-white/10 bg-white/5 text-slate-300 hover:border-rose-400/30 hover:text-rose-100"
            }`}
            onClick={() => onHighlightNegativeOctantChange(!highlightNegativeOctant)}
          >
            Log Returns (-)
          </button>
        </div>
      </div>
    </div>
  );
}
