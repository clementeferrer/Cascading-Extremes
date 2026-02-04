import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";

interface Story {
  title: string;
  body: string;
}

interface Props {
  stories: Story[];
}

export function ScrollStory({ stories }: Props) {
  const [activeIndex, setActiveIndex] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleScroll = () => {
      if (!containerRef.current) return;

      const container = containerRef.current;
      const rect = container.getBoundingClientRect();
      const containerHeight = rect.height;
      const viewportHeight = window.innerHeight;

      // Calculate how far we've scrolled through the container
      const scrollStart = -rect.top;
      const scrollRange = containerHeight - viewportHeight;

      if (scrollRange <= 0) {
        setActiveIndex(0);
        return;
      }

      // Calculate progress through the scroll (0 to 1)
      const progress = Math.max(0, Math.min(1, scrollStart / scrollRange));

      // Map progress to story index
      const newIndex = Math.min(
        stories.length - 1,
        Math.floor(progress * stories.length)
      );

      setActiveIndex(newIndex);
    };

    window.addEventListener("scroll", handleScroll, { passive: true });
    handleScroll(); // Initial check

    return () => window.removeEventListener("scroll", handleScroll);
  }, [stories.length]);

  return (
    <div
      ref={containerRef}
      className="relative border-t border-white/5"
      style={{ height: `${stories.length * 100}vh` }}
    >
      {/* Sticky container that stays in view while scrolling */}
      <div className="sticky top-0 h-screen flex flex-col justify-center">
        <div className="max-w-3xl">
          <div className="text-base uppercase tracking-[0.4em] text-slate-400 mb-4">
            Story
          </div>

          {/* Progress indicators */}
          <div className="flex gap-3 mb-8">
            {stories.map((_, i) => (
              <button
                key={i}
                onClick={() => {
                  if (!containerRef.current) return;
                  const container = containerRef.current;
                  const containerTop = container.offsetTop;
                  const containerHeight = container.offsetHeight;
                  const viewportHeight = window.innerHeight;
                  const scrollRange = containerHeight - viewportHeight;
                  const targetScroll = containerTop + (i / stories.length) * scrollRange;
                  window.scrollTo({ top: targetScroll, behavior: "smooth" });
                }}
                className={`h-1.5 rounded-full transition-all duration-300 ${
                  i === activeIndex
                    ? "w-10 bg-teal-400"
                    : i < activeIndex
                    ? "w-5 bg-teal-400/50"
                    : "w-5 bg-white/20"
                }`}
                aria-label={`Go to story ${i + 1}`}
              />
            ))}
          </div>

          {/* Animated content */}
          <div className="relative min-h-[280px]">
            <AnimatePresence mode="wait">
              <motion.div
                key={activeIndex}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.4, ease: "easeOut" }}
              >
                <h2 className="text-4xl md:text-5xl lg:text-6xl font-serif text-white leading-tight">
                  {stories[activeIndex].title}
                </h2>
                <p className="mt-6 text-lg md:text-xl text-slate-300 leading-relaxed">
                  {stories[activeIndex].body}
                </p>
              </motion.div>
            </AnimatePresence>
          </div>

          {/* Scroll hint */}
          {activeIndex < stories.length - 1 && (
            <motion.div
              className="mt-10 text-sm text-slate-500 flex items-center gap-2"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 1 }}
            >
              <span>Scroll to continue</span>
              <motion.span
                animate={{ y: [0, 4, 0] }}
                transition={{ duration: 1.5, repeat: Infinity }}
              >
                ↓
              </motion.span>
            </motion.div>
          )}
        </div>
      </div>
    </div>
  );
}
