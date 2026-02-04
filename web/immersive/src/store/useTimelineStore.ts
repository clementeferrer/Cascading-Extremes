import { create } from "zustand";

interface TimelineState {
  currentTime: number;
  maxTime: number;
  playing: boolean;
  speed: number;
  setCurrentTime: (t: number | ((prev: number) => number)) => void;
  setMaxTime: (t: number) => void;
  setPlaying: (v: boolean) => void;
  setSpeed: (v: number) => void;
}

export const useTimelineStore = create<TimelineState>((set) => ({
  currentTime: 0,
  maxTime: 0,
  playing: false,
  speed: 1,
  setCurrentTime: (t) =>
    set((state) => ({
      currentTime: typeof t === "function" ? t(state.currentTime) : t,
    })),
  setMaxTime: (t) => set({ maxTime: t }),
  setPlaying: (v) => set({ playing: v }),
  setSpeed: (v) => set({ speed: v }),
}));
