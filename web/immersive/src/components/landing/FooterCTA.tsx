interface Props {
  text: string;
  ctaLabel: string;
  ctaHref: string;
}

export function FooterCTA({ text, ctaLabel, ctaHref }: Props) {
  return (
    <section className="py-20 border-t border-white/5">
      <div className="rounded-3xl border border-white/10 bg-white/5 p-10 text-center">
        <h3 className="text-3xl font-semibold font-serif text-white">{text}</h3>
        <div className="mt-6">
          <a
            href={ctaHref}
            className="inline-flex rounded-full bg-teal-500/90 px-6 py-3 text-sm font-semibold text-white shadow-lg hover:bg-teal-400 transition"
          >
            {ctaLabel}
          </a>
        </div>
      </div>
      <div className="mt-8 flex items-center justify-between text-xs text-slate-500">
        <div>Created by Clemente Ferrer</div>
        <div>Authors: De Carvalho, Ferrer &amp; Vallejos</div>
      </div>
    </section>
  );
}
