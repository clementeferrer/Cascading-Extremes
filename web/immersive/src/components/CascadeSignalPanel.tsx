import ReactECharts from "echarts-for-react";

interface Props {
  ratioSeries: [number, number][];
  assetProbSeries: [number, number, number][];
  assetLabels: string[];
  currentTime: number;
  timeScale: number;
}

export function CascadeSignalPanel({ ratioSeries, assetProbSeries, assetLabels, currentTime, timeScale }: Props) {
  const ratioCut = ratioSeries.filter(([t]) => t <= currentTime);
  const assetCut = assetProbSeries.filter(([t]) => t <= currentTime);

  if (!ratioSeries.length || !assetLabels.length) {
    return (
      <div className="rounded-xl bg-white/90 p-3 shadow-md text-sm text-slate">
        No cascade signals available for this run yet.
      </div>
    );
  }

  const probOption = {
    grid: { left: 40, right: 20, top: 28, bottom: 30 },
    tooltip: {
      trigger: "axis",
      formatter: (params: any) => {
        const t = params?.[0]?.value?.[0] ?? 0;
        const val = params?.[0]?.value?.[1] ?? 0;
        return `t=${(t * timeScale).toFixed(1)}h<br/>ψ/λ=${Number(val).toFixed(3)}`;
      },
    },
    xAxis: {
      type: "value",
      name: "Time",
      nameTextStyle: { color: "#475569", fontSize: 11 },
      axisLabel: { color: "#64748b", formatter: (v: number) => (v * timeScale).toFixed(0) },
      axisPointer: { show: true, type: "line" },
      splitLine: { lineStyle: { color: "#e2e8f0" } },
    },
    yAxis: {
      type: "value",
      min: 0,
      max: 1,
      name: "Cascade Probability (ψ/λ)",
      nameTextStyle: { color: "#475569", fontSize: 11 },
      axisLabel: { color: "#64748b", formatter: (v: number) => v.toFixed(2) },
      splitLine: { lineStyle: { color: "#eef2f7" } },
    },
    series: [
      {
        name: "ψ/λ",
        type: "line",
        showSymbol: false,
        data: ratioCut,
        lineStyle: { width: 2.2, color: "#e76f51" },
      },
      {
        name: "Now",
        type: "line",
        showSymbol: false,
        data: [
          [currentTime, 0],
          [currentTime, 1],
        ],
        lineStyle: { color: "#0f172a", width: 1, type: "dashed" },
        tooltip: { show: false },
      },
    ],
  };

  const palette = ["#0f172a", "#1d4ed8", "#10b981", "#f97316", "#9333ea"];
  const assetOption = {
    grid: { left: 60, right: 20, top: 28, bottom: 30 },
    tooltip: {
      trigger: "item",
      formatter: (params: any) => {
        const [t, idx, prob] = params.data as [number, number, number];
        const name = assetLabels[idx] ?? "asset";
        return `${name}<br/>t=${(t * timeScale).toFixed(1)}h<br/>ψ/λ=${prob.toFixed(3)}`;
      },
    },
    xAxis: {
      type: "value",
      name: "Time",
      nameTextStyle: { color: "#475569", fontSize: 11 },
      axisLabel: { color: "#64748b", formatter: (v: number) => (v * timeScale).toFixed(0) },
      splitLine: { lineStyle: { color: "#e2e8f0" } },
    },
    yAxis: {
      type: "category",
      data: assetLabels,
      name: "Dominant asset",
      nameTextStyle: { color: "#475569", fontSize: 11 },
      axisLabel: { color: "#64748b" },
      splitLine: { show: false },
    },
    series: [
      {
        name: "Dominant asset",
        type: "scatter",
        data: assetCut.map(([t, prob, idx]) => [t, idx, prob]),
        symbolSize: (val: number[]) => 6 + val[2] * 10,
        itemStyle: {
          color: (params: any) => {
            const idx = params?.data?.[1] ?? 0;
            return palette[idx % palette.length];
          },
        },
        label: {
          show: true,
          formatter: (params: any) => Number(params.data[2]).toFixed(2),
          color: "#0f172a",
          fontSize: 9,
        },
      },
      {
        name: "Now",
        type: "line",
        showSymbol: false,
        data: [
          [currentTime, 0],
          [currentTime, assetLabels.length - 1],
        ],
        lineStyle: { color: "#0f172a", width: 1, type: "dashed" },
        tooltip: { show: false },
      },
    ],
  };

  return (
    <div className="rounded-xl bg-white/90 p-3 shadow-md space-y-4">
      <div>
        <div className="text-xs uppercase tracking-wide text-slate mb-1">Cascade Probability</div>
        <div className="text-[11px] text-slate mb-2">ψ/λ over time. The curve grows with the playhead.</div>
        <ReactECharts option={probOption} style={{ height: 140 }} />
      </div>
      <div>
        <div className="text-xs uppercase tracking-wide text-slate mb-1">Dominant Asset Timeline</div>
        <div className="text-[11px] text-slate mb-2">Each event: asset on the y‑axis, ψ/λ labeled on the point.</div>
        {assetCut.length ? (
          <ReactECharts option={assetOption} style={{ height: 170 }} />
        ) : (
          <div className="text-[11px] text-slate">No events yet in this window.</div>
        )}
      </div>
    </div>
  );
}
