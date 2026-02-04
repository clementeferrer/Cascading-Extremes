import { motion } from "framer-motion";

interface Props {
  title: string;
  body: string;
}

export function StorySection({ title, body }: Props) {
  return (
    <motion.section
      className="min-h-[50vh] flex flex-col justify-center border-t border-white/5 py-12"
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.4 }}
      transition={{ duration: 0.6, ease: "easeOut" }}
    >
      <div className="max-w-2xl">
        <div className="text-sm uppercase tracking-[0.3em] text-slate-500">Story</div>
        <h2 className="mt-3 text-3xl md:text-4xl font-serif text-white">{title}</h2>
        <p className="mt-4 text-base text-slate-300">{body}</p>
      </div>
    </motion.section>
  );
}
