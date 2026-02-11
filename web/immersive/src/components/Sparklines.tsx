import ReactECharts from "echarts-for-react";

interface SeriesEntry {
  name: string;
  color: string;
  data: [number, number][];
}

interface Props {
  series: SeriesEntry[];
  currentTime: number;
  timeScale: number;
  startDatetimeUtc?: string | null;
}

function formatTooltipTime(startDatetimeUtc: string | null | undefined, tScaled: number, timeScale: number) {
  const tHours = tScaled * timeScale;
  if (!startDatetimeUtc) return `t = ${tHours.toFixed(2)}h`;
  const startMs = Date.parse(startDatetimeUtc);
  if (!Number.isFinite(startMs)) return `t = ${tHours.toFixed(2)}h`;
  const dt = new Date(startMs + Math.round(tHours) * 3600_000);
  return `t = ${dt.toISOString().replace("T", " ").slice(0, 16)} UTC`;
}

export function Sparklines({ series, currentTime, timeScale, startDatetimeUtc = null }: Props) {
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
    tooltip: {
      trigger: "axis",
      backgroundColor: "rgba(15,23,42,0.9)",
      textStyle: { color: "#e2e8f0" },
      confine: true,
      position: (point: number[], _params: any, _dom: any, _rect: any, size: any) => {
        const [viewWidth, viewHeight] = size?.viewSize ?? [0, 0];
        const [contentWidth, contentHeight] = size?.contentSize ?? [0, 0];
        const gap = 10;
        let x = (point?.[0] ?? 0) + gap;
        let y = (point?.[1] ?? 0) - contentHeight / 2;
        if (x + contentWidth > viewWidth - 4) x = viewWidth - contentWidth - 4;
        if (x < 4) x = 4;
        if (y + contentHeight > viewHeight - 4) y = viewHeight - contentHeight - 4;
        if (y < 4) y = 4;
        return [x, y];
      },
      formatter: (params: any) => {
        const items = Array.isArray(params) ? params : [params];
        if (!items.length) return "";
        const tScaled = Number(items[0]?.axisValue ?? items[0]?.value?.[0] ?? 0);
        const header = formatTooltipTime(startDatetimeUtc, tScaled, timeScale);
        const rows = items.map((item: any) => {
          const val = Number(item?.value?.[1] ?? item?.value ?? 0);
          return `${item.marker}${item.seriesName}: ${val.toFixed(2)}`;
        });
        return [header, ...rows].join("<br/>");
      },
    },
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
      <ReactECharts option={option} style={{ height: 110 }} />
    </div>
  );
}
