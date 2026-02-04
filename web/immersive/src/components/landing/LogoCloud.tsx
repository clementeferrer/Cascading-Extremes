interface Props {
  labels: string[];
}

export function LogoCloud({ labels }: Props) {
  return (
    <section className="py-12 border-t border-white/5">
      <div className="text-xs uppercase tracking-[0.3em] text-slate-500">Backed by</div>
      <div className="mt-6 grid grid-cols-2 gap-3 md:grid-cols-4">
        {labels.map((label) => (
          <div
            key={label}
            className="rounded-2xl border border-white/10 bg-white/5 px-4 py-6 text-center text-xs uppercase tracking-[0.2em] text-slate-400"
          >
            {label}
          </div>
        ))}
      </div>
    </section>
  );
}
