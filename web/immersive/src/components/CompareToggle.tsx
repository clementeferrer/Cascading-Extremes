interface Props {
  runs: { run_id: string; source?: string }[];
  activeRun: string;
  onChange: (id: string) => void;
}

export function CompareToggle({ runs, activeRun, onChange }: Props) {
  return (
    <div className="flex gap-2">
      {runs.map((r) => (
        <button
          key={r.run_id}
          className={`rounded-full px-3 py-1 text-xs font-semibold ${
            activeRun === r.run_id ? "bg-ink text-white" : "bg-white text-ink border border-slate/20"
          }`}
          onClick={() => onChange(r.run_id)}
        >
          {r.source ?? r.run_id}
        </button>
      ))}
    </div>
  );
}
