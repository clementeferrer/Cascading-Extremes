import { useTimelineStore } from "../store/useTimelineStore";

interface Props {
  maxTime: number;
  onPlay?: (nextState: boolean) => void;
  disabled?: boolean;
}

export function Controls({ maxTime, onPlay, disabled }: Props) {
  const { currentTime, playing, speed, setPlaying, setCurrentTime, setSpeed } = useTimelineStore();

  const togglePlay = () => {
    if (disabled) return;
    const next = !playing;
    if (onPlay) {
      onPlay(next);
    } else {
      setPlaying(next);
    }
  };

  return (
    <div className={`grid grid-cols-[auto_1fr_auto_auto] items-center gap-2 rounded-2xl border border-white/10 bg-white/5 backdrop-blur-md px-4 py-3 shadow-lg w-full ${disabled ? "opacity-60" : ""}`}>
      <button
        className="rounded-full bg-teal-500/90 px-4 py-2 text-sm font-semibold text-white whitespace-nowrap"
        onClick={togglePlay}
        disabled={disabled}
      >
        {playing ? "Pause" : "Play"}
      </button>
      <input
        type="range"
        min={0}
        max={maxTime}
        step={0.1}
        value={currentTime}
        onChange={(e) => setCurrentTime(Number(e.target.value))}
        className="w-full min-w-0 accent-teal-300"
        disabled={disabled}
      />
      <div className="text-sm text-slate-300 whitespace-nowrap">Speed</div>
      <select
        className="rounded-md border border-white/10 bg-white/5 px-2 py-1 text-sm text-slate-100"
        value={speed}
        onChange={(e) => setSpeed(Number(e.target.value))}
        disabled={disabled}
      >
        <option value={0.25}>0.25x</option>
        <option value={0.5}>0.5x</option>
        <option value={1}>1x</option>
        <option value={2}>2x</option>
        <option value={5}>5x</option>
        <option value={10}>10x</option>
        <option value={20}>20x</option>
      </select>
    </div>
  );
}
