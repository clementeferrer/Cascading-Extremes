interface Props {
  showSimplex: boolean;
  onShowSimplexChange: (v: boolean) => void;
  pointSize: "small" | "medium" | "large";
  onPointSizeChange: (v: "small" | "medium" | "large") => void;
}

export function GeometryControls({
  showSimplex,
  onShowSimplexChange,
  pointSize,
  onPointSizeChange,
}: Props) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 backdrop-blur-md p-3 shadow-lg space-y-2">
      <div className="text-xs uppercase tracking-wide text-slate-400">Geometry</div>
      <label className="flex items-center gap-2 text-[11px] text-slate-300">
        <input type="checkbox" checked={showSimplex} onChange={(e) => onShowSimplexChange(e.target.checked)} />
        Show simplex plane overlay
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
    </div>
  );
}
