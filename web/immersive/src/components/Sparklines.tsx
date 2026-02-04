import ReactECharts from "echarts-for-react";

interface SeriesEntry {
  name: string;
  color: string;
  data: [number, number][];
}

interface Props {
  series: SeriesEntry[];
  currentTime: number;
}

export function Sparklines({ series, currentTime }: Props) {
  const cutSeries = series.map((s) => ({
    ...s,
    data: s.data.filter(([t]) => t <= currentTime),
  }));

  const option = {
    backgroundColor: "transparent",
    grid: { left: 10, right: 10, top: 20, bottom: 10 },
    xAxis: { type: "value", show: false },
    yAxis: { type: "value", show: false },
    legend: {
      data: cutSeries.map((s) => s.name),
      top: 0,
      textStyle: { color: "#cbd5f5", fontSize: 10 },
    },
    tooltip: { trigger: "axis", backgroundColor: "rgba(15,23,42,0.9)", textStyle: { color: "#e2e8f0" } },
    series: cutSeries.map((s) => ({
      name: s.name,
      data: s.data,
      type: "line",
      smooth: true,
      showSymbol: false,
      lineStyle: { color: s.color, width: 1.6 },
    })),
  };

  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 backdrop-blur-md p-3 shadow-lg">
      <div className="text-xs uppercase tracking-wide text-slate-400 mb-1">Signal Overview</div>
      <div className="text-[11px] text-slate-400 mb-2">Lines evolve with the playhead: λ(t), μ(t), ψ(t).</div>
      <ReactECharts option={option} style={{ height: 110 }} />
    </div>
  );
}
