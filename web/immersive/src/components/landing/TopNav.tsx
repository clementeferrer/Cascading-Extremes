interface NavLink {
  label: string;
  href: string;
}

interface Props {
  links: NavLink[];
  ctaLabel: string;
  ctaHref: string;
}

export function TopNav({ links, ctaLabel, ctaHref }: Props) {
  return (
    <div className="flex items-center justify-between py-6">
      <div className="text-sm uppercase tracking-[0.4em] text-slate-400">Cascading Extremes</div>
      <div className="flex items-center gap-6 text-sm text-slate-300">
        {links.map((link) => (
          <a key={link.href} href={link.href} className="hover:text-white transition">
            {link.label}
          </a>
        ))}
        <a
          href={ctaHref}
          className="rounded-full border border-teal-300/40 px-4 py-2 text-xs font-semibold text-teal-100 hover:border-teal-200 hover:text-white transition"
        >
          {ctaLabel}
        </a>
      </div>
    </div>
  );
}
