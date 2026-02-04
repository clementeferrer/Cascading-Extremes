interface Step {
  title: string;
  body: string;
}

interface Props {
  steps: Step[];
}

export function Stepper({ steps }: Props) {
  return (
    <section className="py-16 border-t border-white/5">
      <div className="flex items-center justify-between">
        <h3 className="text-xl font-semibold text-white">How It Works</h3>
        <div className="text-xs uppercase tracking-[0.3em] text-slate-500">Process</div>
      </div>
      <div className="mt-8 grid gap-4 md:grid-cols-3">
        {steps.map((step, idx) => (
          <div key={step.title} className="rounded-2xl border border-white/10 bg-white/5 p-4">
            <div className="text-xs uppercase tracking-[0.3em] text-slate-500">{String(idx + 1).padStart(2, "0")}</div>
            <div className="mt-2 text-base font-semibold text-white">{step.title}</div>
            <div className="mt-2 text-sm text-slate-300">{step.body}</div>
          </div>
        ))}
      </div>
    </section>
  );
}
