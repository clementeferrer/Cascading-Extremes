interface Props {
  headline: string;
  subheadline: string;
  ctaLabel: string;
  ctaHref: string;
}

export function Hero({ headline, subheadline, ctaLabel, ctaHref }: Props) {
  return (
    <section className="pt-12 pb-24">
      <div className="max-w-3xl">
        <h1 className="text-5xl md:text-6xl font-semibold font-serif text-white leading-tight">
          {headline}
        </h1>
        <p className="mt-6 text-lg text-slate-300">{subheadline}</p>
        <div className="mt-8 flex items-center gap-4">
          <a
            href={ctaHref}
            className="rounded-full bg-teal-500/90 px-6 py-3 text-sm font-semibold text-white shadow-lg hover:bg-teal-400 transition"
          >
            {ctaLabel}
          </a>
          <span className="text-xs uppercase tracking-[0.3em] text-slate-500">Live paper demo</span>
        </div>
      </div>
    </section>
  );
}
