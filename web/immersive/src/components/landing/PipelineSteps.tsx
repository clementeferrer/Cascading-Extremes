import { motion } from "framer-motion";

interface Step {
  label: string;
  text: string;
}

interface Props {
  steps: Step[];
}

export function PipelineSteps({ steps }: Props) {
  return (
    <section className="py-16 border-t border-white/5">
      <div className="flex items-center justify-between">
        <h3 className="text-xl font-semibold text-white">Generation Loop</h3>
        <div className="text-xs uppercase tracking-[0.3em] text-slate-500">Pipeline</div>
      </div>
      <div className="mt-8 grid gap-6 md:grid-cols-4">
        {steps.map((step, idx) => (
          <motion.div
            key={step.label}
            className="rounded-2xl border border-white/10 bg-white/5 p-4"
            initial={{ opacity: 0, y: 12 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.3 }}
            transition={{ duration: 0.4, delay: idx * 0.05 }}
          >
            <div className="text-xs uppercase tracking-[0.25em] text-slate-500">{step.label}</div>
            <p className="mt-2 text-sm text-slate-200">{step.text}</p>
          </motion.div>
        ))}
      </div>
    </section>
  );
}
