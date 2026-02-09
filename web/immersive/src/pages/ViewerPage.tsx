import { useEffect, useMemo, useState } from "react";
import {
  generateContinue,
  getEvents,
  getMeta,
  getMetrics,
  getRuns,
} from "../api/client";
import { EventRecord, MetricsRecord, RunMeta } from "../api/types";
import { CascadeScene } from "../scenes/CascadeScene";
import { KPICards } from "../components/KPICards";
import { Sparklines } from "../components/Sparklines";
import { Controls } from "../components/Controls";
import { GenerationControls } from "../components/GenerationControls";
import { GeometryControls } from "../components/GeometryControls";
import { EventRail } from "../components/EventRail";
import { useTimelineStore } from "../store/useTimelineStore";
import { CubeMappingParams, mapToCube } from "../utils/geometry";
import { ErrorBoundary } from "../components/ErrorBoundary";

function binarySearch(times: number[], t: number) {
  let lo = 0;
  let hi = times.length - 1;
  while (lo <= hi) {
    const mid = Math.floor((lo + hi) / 2);
    if (times[mid] <= t) lo = mid + 1;
    else hi = mid - 1;
  }
  return Math.max(0, lo - 1);
}

function downsamplePairs(times: number[], values: number[], maxPoints = 200): [number, number][] {
  if (times.length !== values.length) return [];
  if (times.length <= maxPoints) return times.map((t, i) => [t, values[i]]);
  const step = Math.ceil(times.length / maxPoints);
  const out: [number, number][] = [];
  for (let i = 0; i < times.length; i += step) {
    out.push([times[i], values[i]]);
  }
  return out;
}

function stripAssetLabel(label: string) {
  return label.replace("-USD", "");
}

export default function ViewerPage() {
  const [runs, setRuns] = useState<{ run_id: string; source?: string; assets?: string[] }[]>([]);
  const [activeRun, setActiveRun] = useState<string>("");
  const [realRunId, setRealRunId] = useState<string>("");
  const [generativeRunId, setGenerativeRunId] = useState<string>("");
  const [events, setEvents] = useState<EventRecord[]>([]);
  const [metrics, setMetrics] = useState<MetricsRecord[]>([]);
  const [timeScale, setTimeScale] = useState<number>(1);
  const [meta, setMeta] = useState<RunMeta | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);

  const [mode, setMode] = useState<"real" | "generative">("real");
  const [seedRunId, setSeedRunId] = useState<string>("");
  const [seedAsset, setSeedAsset] = useState<string>("BTC-USD");
  const [theta, setTheta] = useState<number>(0);
  const [phi, setPhi] = useState<number>(Math.PI / 2);
  const [magnitude, setMagnitude] = useState<number>(3.0);
  const [horizonHours, setHorizonHours] = useState<number>(240);
  const [generativeHorizon, setGenerativeHorizon] = useState<number | null>(null);
  const [generating, setGenerating] = useState<boolean>(false);
  const [pendingPlayRunId, setPendingPlayRunId] = useState<string | null>(null);
  const [generativeRunReady, setGenerativeRunReady] = useState<boolean>(false);

  const [showSimplex, setShowSimplex] = useState<boolean>(true);
  const [pointSize, setPointSize] = useState<"small" | "medium" | "large">("small");
  const [mapping, setMapping] = useState<CubeMappingParams>({
    a: 3.0,
    b: 0.0,
    offsetScale: 0.3,
  });
  const [hoveredEventId, setHoveredEventId] = useState<number | null>(null);

  const { currentTime, maxTime, playing, speed, setCurrentTime, setMaxTime, setPlaying } = useTimelineStore();

  useEffect(() => {
    getRuns()
      .then((data) => {
        setRuns(data);
        setApiError(null);
        if (!realRunId && data.length > 0) {
          const realRun = data.find((r) => r.source === "real")?.run_id ?? data[0].run_id;
          setRealRunId(realRun);
          setSeedRunId(realRun);
          if (mode === "real") {
            setActiveRun(realRun);
          }
        }
      })
      .catch(() => {
        setApiError("API not reachable. Start the backend at http://localhost:8000.");
      });
  }, []);

  // Mode switch effect - CRITICAL: stops playback and resets state cleanly
  useEffect(() => {
    console.info(`[MODE] switching to: ${mode}`);
    // Stop everything first
    setPlaying(false);
    setCurrentTime(0);
    setMaxTime(0);
    setEvents([]);
    setMetrics([]);
    setMeta(null);
    setPendingPlayRunId(null);

    if (mode === "real") {
      // Clear generative state when switching TO real mode
      setGenerativeHorizon(null);
      setGenerativeRunReady(false);
      if (realRunId) setActiveRun(realRunId);
    } else {
      // Generative mode: DON'T auto-load anything
      // Wait for explicit Play press to generate a new run
      setActiveRun("");
      setGenerativeRunReady(false);
    }
  }, [mode, setPlaying, setCurrentTime, setMaxTime]);

  // Separate effect to load real run when realRunId changes (only in real mode)
  useEffect(() => {
    if (mode === "real" && realRunId) {
      setActiveRun(realRunId);
    }
  }, [realRunId, mode]);

  // Data fetch effect - only fetch when activeRun is set AND mode matches
  useEffect(() => {
    if (!activeRun) return;
    // In generative mode, only fetch if generativeRunReady is true
    if (mode === "generative" && !generativeRunReady) {
      console.info(`[RUN] skipping fetch - generative mode not ready`);
      return;
    }
    console.info(`[RUN] loading: ${activeRun}`);
    (async () => {
      try {
        const [metaResp, ev, mt] = await Promise.all([
          getMeta(activeRun),
          getEvents(activeRun, 0, 200000),
          getMetrics(activeRun, 0, 200000),
        ]);
        console.info(`[DATA] events: ${ev.length}, metrics: ${mt.length}`);
        setMeta(metaResp);
        setApiError(null);
        const isGenerative = metaResp?.source === "generative";
        const t0 = ev.length ? ev[0].t : 0;
        const evShifted = ev.map((e) => ({ ...e, t: e.t - t0 }));
        const mtShifted = mt.map((m) => ({ ...m, t: (m.t ?? 0) - t0 }));
        const maxT = evShifted.length ? evShifted[evShifted.length - 1].t : 0;
        let scale = 1;
        let evScaled = evShifted;
        let mtScaled = mtShifted;
        let maxTimeScaled = maxT;
        if (!isGenerative) {
          const targetDuration = 300;
          scale = maxT > 0 ? maxT / targetDuration : 1;
          evScaled = evShifted.map((e) => ({ ...e, t: e.t / scale }));
          mtScaled = mtShifted.map((m) => ({ ...m, t: (m.t ?? 0) / scale }));
          maxTimeScaled = maxT / scale;
        }
        setTimeScale(scale);
        setEvents(evScaled);
        setMetrics(mtScaled);
        const horizonClamp =
          isGenerative && generativeHorizon != null ? Math.min(maxTimeScaled, generativeHorizon) : maxTimeScaled;
        console.info(`[TIMELINE] maxT=${maxT.toFixed(2)}, scale=${scale.toFixed(4)}, horizon=${horizonClamp.toFixed(2)}, isGenerative=${isGenerative}`);
        setMaxTime(horizonClamp);
        setCurrentTime(0);
        setPlaying(false);
        if (metaResp?.assets?.length && !seedRunId) {
          setSeedRunId(activeRun);
        }
        if (metaResp?.assets?.length) {
          setSeedAsset(metaResp.assets[0]);
        }
        // magnitude range no longer needed — generation uses theta/phi directly
      } catch (err) {
        console.error(err);
        setApiError("Failed to load run data. Ensure exports exist and API is running.");
        setEvents([]);
        setMetrics([]);
        setMaxTime(0);
      }
    })();
  }, [activeRun, seedRunId, generativeHorizon, mode, generativeRunReady, setCurrentTime, setMaxTime, setPlaying]);

  useEffect(() => {
    if (!seedRunId && runs.length) {
      const realRun = runs.find((r) => r.source === "real")?.run_id ?? runs[0].run_id;
      if (realRun) {
        setSeedRunId(realRun);
      }
    }
  }, [runs, seedRunId]);

  useEffect(() => {
    if (seedAsset) return;
    const assets = runs.find((r) => r.run_id === realRunId)?.assets;
    if (assets && assets.length) {
      setSeedAsset(assets[0]);
    }
  }, [runs, realRunId, seedAsset]);

  useEffect(() => {
    console.info(`[viewer] mode -> ${mode}`);
  }, [mode]);

  useEffect(() => {
    console.info(`[viewer] playback -> ${playing ? "running" : "stopped"}`);
  }, [playing]);

  useEffect(() => {
    if (pendingPlayRunId && pendingPlayRunId === activeRun && events.length > 0) {
      setPlaying(true);
      setPendingPlayRunId(null);
    }
  }, [pendingPlayRunId, activeRun, events.length, setPlaying]);

  useEffect(() => {
    let raf = 0;
    let last = performance.now();
    const tick = (now: number) => {
      const dt = (now - last) / 1000;
      last = now;
      setCurrentTime((t) => {
        const next = t + dt * speed;
        if (next >= maxTime) {
          setPlaying(false);
          return maxTime;
        }
        return next;
      });
      raf = requestAnimationFrame(tick);
    };
    if (playing) raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, speed, maxTime, setCurrentTime, setPlaying]);

  const times = useMemo(() => events.map((e) => e.t), [events]);
  const idx = times.length ? binarySearch(times, currentTime) : 0;
  const eventCount = times.length ? idx + 1 : 0;

  const metricIdx = metrics.length ? Math.min(metrics.length - 1, idx) : 0;
  const lambda = metrics.length ? metrics[metricIdx].lambda ?? undefined : undefined;
  const psi = metrics.length ? metrics[metricIdx].psi ?? undefined : undefined;
  const assetCounts = metrics.length ? metrics[metricIdx].per_asset_counts ?? undefined : undefined;

  const displayTime = currentTime * timeScale;

  const metricTimes = metrics.map((m) => m.t ?? 0);
  const lambdaVals = metrics.map((m) => m.lambda ?? 0);
  const psiVals = metrics.map((m) => m.psi ?? 0);
  const muVals = metrics.map((m) => m.mu ?? 0);
  const eventRateVals = metrics.map((m) => m.event_rate ?? 0);
  const ratioVals = metrics.map((m) => {
    const lam = m.lambda ?? 0;
    const psiVal = m.psi ?? 0;
    return lam > 0 ? psiVal / lam : 0;
  });

  // Event points with POC (ψ/λ) for coloring
  const eventPoints = useMemo(
    () => events.map((e, i) => ({
      t: e.t,
      w: e.w,
      mag: e.mag,
      u_tau: e.u_tau,
      poc: ratioVals[i] ?? 0,
    })),
    [events, ratioVals]
  );

  const sparkSeries = [
    { name: "λ(t)", color: "#38bdf8", data: downsamplePairs(metricTimes, lambdaVals, 240) },
    { name: "μ(t)", color: "#22c55e", data: downsamplePairs(metricTimes, muVals, 240) },
    { name: "ψ(t)", color: "#f97316", data: downsamplePairs(metricTimes, psiVals, 240) },
  ];

  const timelineItems = useMemo(() => {
    if (!events.length) return [] as { id: number; t: number; asset: string; ratio: number; context: string }[];
    const contextWindow = 5;
    const items = events.map((e, i) => {
      const ratio = ratioVals[i] ?? 0;
      let context = "Isolated extreme";
      if (e.parent_id != null) {
        context = `Triggered by #${e.parent_id}`;
      } else {
        const start = Math.max(0, i - contextWindow);
        const end = Math.min(events.length, i + contextWindow + 1);
        const neighbors = events.slice(start, end);
        const assets = Array.from(
          new Set(neighbors.filter((n) => n.id !== e.id).map((n) => stripAssetLabel(n.asset)))
        );
        if (assets.length) {
          context = `Approx co-exceedances: ${assets.slice(0, 3).join(", ")}`;
        }
      }
      return {
        id: e.id,
        t: e.t,
        asset: stripAssetLabel(e.asset),
        ratio,
        context,
      };
    });
    return items;
  }, [events, ratioVals]);

  const highlightPoint = useMemo(() => {
    if (hoveredEventId == null) return null;
    const ev = events.find((e) => e.id === hoveredEventId);
    if (!ev) return null;
    return mapToCube(ev.w, ev.mag, ev.u_tau, mapping);
  }, [hoveredEventId, events, mapping]);

  const pointSizeValue = useMemo(() => {
    if (pointSize === "small") return 0.08;
    if (pointSize === "medium") return 0.12;
    return 0.18;
  }, [pointSize]);

  const viewOptions = useMemo(() => {
    return runs
      .filter((r) => r.source === "real")
      .map((r) => ({ id: r.run_id, label: r.run_id }));
  }, [runs]);

  const handleGenerate = async (autoPlay: boolean) => {
    if (generating) return;
    setGenerating(true);
    setPlaying(false);
    setCurrentTime(0);
    setMaxTime(0);
    setEvents([]);
    setMetrics([]);
    try {
      const seed = Math.floor(Math.random() * 2147483647);
      console.info("[GENERATE] START", { seedAsset, theta, phi, magnitude, horizonHours, seed });
      const payload = {
        theta,
        phi,
        magnitude,
        max_time: horizonHours,
        seed,
      };
      const res = await generateContinue(payload);
      const updatedRuns = await getRuns();
      setRuns(updatedRuns);
      setGenerativeRunId(res.run_id);
      setGenerativeHorizon(horizonHours);
      // Mark generative run as ready BEFORE setting activeRun
      // This allows the fetch effect to proceed
      setGenerativeRunReady(true);
      setActiveRun(res.run_id);
      if (autoPlay) {
        setPendingPlayRunId(res.run_id);
      }
      console.info(`[GENERATE] END -> ${res.run_id}`);
    } catch (err) {
      console.error(err);
      console.info("[GENERATE] FAILED");
    } finally {
      setGenerating(false);
    }
  };

  const handlePlay = (nextState: boolean) => {
    console.info(`[PLAYBACK] ${nextState ? "PLAY" : "PAUSE"}`);
    if (!nextState) {
      setPlaying(false);
      return;
    }
    if (mode === "generative") {
      if (generating) return; // Already generating

      if (!generativeRunReady) {
        // No run ready yet -> generate new one
        console.info("[PLAYBACK] No generative run ready, generating...");
        handleGenerate(true);
        return;
      }

      if (currentTime >= maxTime - 0.01) {
        // Run finished -> generate NEW run on explicit play
        console.info("[PLAYBACK] Run finished, generating new...");
        handleGenerate(true);
        return;
      }

      // Run exists, not finished -> just resume playback
      setPlaying(true);
      return;
    }
    // Real mode - just play
    setPlaying(true);
  };

  const playDisabled = mode === "real" ? events.length === 0 || maxTime <= 0 : generating;

  return (
    <div className="h-screen overflow-hidden bg-night text-slate-100 relative">
      <ErrorBoundary>
        <div className="grid h-full w-full grid-cols-1 gap-3 p-3 md:grid-cols-[280px_minmax(0,1fr)_96px] lg:grid-cols-[320px_minmax(0,1fr)_104px]">
          <div
            className="flex h-full flex-col gap-2 overflow-hidden min-h-0 overflow-y-auto overscroll-contain scrollbar-thin pr-4 pb-4"
            style={{ scrollbarGutter: "stable" }}
          >
            <div className="rounded-2xl border border-white/10 bg-white/5 p-4 shadow-lg">
              <div className="text-[10px] uppercase tracking-[0.35em] text-slate-400">Live Paper</div>
              <div className="mt-2 text-2xl font-semibold font-serif text-white">Cascading Extremes</div>
              <div className="mt-2 text-xs text-slate-300">
                Mode: <span className="font-semibold text-white">{mode}</span> · Playback:{" "}
                <span className="font-semibold text-white">{playing ? "Running" : "Stopped"}</span>
              </div>
            </div>
            {apiError && (
              <div className="rounded-2xl border border-red-400/30 bg-red-500/10 p-3 text-xs text-red-100">
                {apiError}
                <div className="mt-2 text-[10px] text-red-200">
                  Start API: <code>python3 -m uvicorn web.api.main:app --reload --reload-dir web/api --port 8000</code>
                </div>
              </div>
            )}
            <KPICards
              currentTime={displayTime}
              eventCount={eventCount}
              lambda={lambda}
              psi={psi}
              assetCounts={assetCounts}
            />
            <Sparklines series={sparkSeries} currentTime={currentTime} />
            <GeometryControls
              showSimplex={showSimplex}
              onShowSimplexChange={setShowSimplex}
              pointSize={pointSize}
              onPointSizeChange={setPointSize}
            />
            <GenerationControls
              mode={mode}
              seedAsset={seedAsset}
              assetOptions={
                (meta?.assets ?? runs.find((r) => r.run_id === realRunId)?.assets ?? []).map((a) => ({
                  id: a,
                  label: a,
                }))
              }
                theta={theta}
                phi={phi}
                magnitude={magnitude}
                horizon={horizonHours}
                generating={generating}
                onModeChange={setMode}
                onSeedAssetChange={setSeedAsset}
                onThetaChange={setTheta}
                onPhiChange={setPhi}
                onMagnitudeChange={setMagnitude}
                onHorizonChange={setHorizonHours}
                onGenerate={() => handleGenerate(true)}
                viewRunId={realRunId}
                viewOptions={viewOptions}
                onViewRunChange={(id) => {
                  setRealRunId(id);
                  if (mode === "real") setActiveRun(id);
                }}
              />
            <div className="mt-auto">
              <Controls maxTime={maxTime} onPlay={handlePlay} disabled={playDisabled} />
              <div className="mt-2 text-[10px] text-slate-500">
                © 2026 | De Carvalho, Ferrer &amp; Vallejos
              </div>
            </div>
          </div>

          <div className="relative h-full overflow-hidden rounded-3xl border border-white/10 bg-night shadow-2xl">
            <CascadeScene
              events={eventPoints}
              currentTime={currentTime}
              mapping={mapping}
              showSimplex={showSimplex}
              pointSize={pointSizeValue}
              highlightPoint={highlightPoint}
              assetLabels={meta?.assets ?? runs.find((r) => r.run_id === realRunId)?.assets}
            />
          </div>

          <div className="h-full w-full flex-none shrink-0">
            <EventRail
              items={timelineItems.map((item) => ({ id: item.id, t: item.t, asset: item.asset, ratio: item.ratio }))}
              currentTime={currentTime}
              timeScale={timeScale}
              onHover={setHoveredEventId}
              onSelect={(t) => setCurrentTime(t)}
            />
          </div>
        </div>
      </ErrorBoundary>
    </div>
  );
}
