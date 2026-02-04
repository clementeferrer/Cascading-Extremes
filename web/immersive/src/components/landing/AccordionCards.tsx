import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

interface Item {
  title: string;
  body: string;
}

interface Props {
  items: Item[];
}

export function AccordionCards({ items }: Props) {
  const [openIdx, setOpenIdx] = useState<number | null>(0);
  return (
    <section className="py-16 border-t border-white/5">
      <div className="flex items-center justify-between">
        <h3 className="text-xl font-semibold text-white">Engine Modules</h3>
        <div className="text-xs uppercase tracking-[0.3em] text-slate-500">Technology</div>
      </div>
      <div className="mt-8 space-y-3">
        {items.map((item, idx) => {
          const open = openIdx === idx;
          return (
            <div key={item.title} className="rounded-2xl border border-white/10 bg-white/5">
              <button
                className="flex w-full items-center justify-between px-5 py-4 text-left text-sm text-slate-200"
                onClick={() => setOpenIdx(open ? null : idx)}
              >
                <span>{item.title}</span>
                <span className="text-slate-400">{open ? "−" : "+"}</span>
              </button>
              <AnimatePresence initial={false}>
                {open && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.3 }}
                    className="overflow-hidden px-5 pb-4 text-sm text-slate-300"
                  >
                    {item.body}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          );
        })}
      </div>
    </section>
  );
}
