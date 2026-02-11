import { useMemo } from "react";
import ReactECharts from "echarts-for-react";
import { RunReturnsResponse } from "../api/types";

interface Props {
  data: RunReturnsResponse | null;
  currentTime: number;
  loading?: boolean;
  error?: string | null;
}

const ASSET_COLORS: Record<string, string> = {
  BTC: "#38bdf8",
  ETH: "#f97316",
  BNB: "#22c55e",
};

function shortAsset(asset: string) {
  return asset.replace("-USD", "").toUpperCase();
}

function downsamplePairs(data: [number, number][], maxPoints = 2500): [number, number][] {
  if (data.length <= maxPoints) return data;
  const step = Math.ceil(data.length / maxPoints);
  const out: [number, number][] = [];
  for (let i = 0; i < data.length; i += step) {
    out.push(data[i]);
  }
  return out;
}

function formatUtcDate(startIso: string | null | undefined, hourIndex: number, detailed = false): string {
  if (!startIso) return `t = ${Math.round(hourIndex)}h`;
  const startMs = Date.parse(startIso);
  if (!Number.isFinite(startMs)) return `t = ${Math.round(hourIndex)}h`;
  const dt = new Date(startMs + Math.round(hourIndex) * 3600_000);
  if (detailed) {
    return dt.toISOString().replace("T", " ").slice(0, 16) + " UTC";
  }
  return dt.toISOString().slice(5, 16).replace("T", " ");
}

export function ReturnsTracksPanel({ data, currentTime, loading = false, error = null }: Props) {
  const option = useMemo(() => {
    if (!data || !data.assets.length) {
      return null;
    }

    const assets = data.assets;
    const maxX = assets.reduce((acc, asset) => {
      const arr = data.series[asset] ?? [];
      const localMax = arr.length ? arr[arr.length - 1][0] : 0;
      return Math.max(acc, localMax);
    }, 0);
    const playhead = Math.max(0, Math.min(currentTime, maxX));

    const grid = assets.map((_, idx) => ({
      left: 58,
      right: 20,
      top: `${14 + idx * 28}%`,
      height: "20%",
    }));

    const xAxis = assets.map((_, idx) => ({
      type: "value",
      gridIndex: idx,
      min: 0,
      max: maxX,
      axisLine: { show: idx === assets.length - 1, lineStyle: { color: "rgba(148,163,184,0.4)" } },
      axisTick: { show: idx === assets.length - 1 },
      axisLabel: {
        show: idx === assets.length - 1,
        color: "#94a3b8",
        fontSize: 10,
        formatter: (v: number) => formatUtcDate(data.alignment.start_datetime_utc, v),
      },
      splitLine: { show: false },
      axisPointer: {
        show: true,
        lineStyle: { color: "rgba(226,232,240,0.35)", width: 1, type: "dashed" },
      },
      name: idx === assets.length - 1 ? "Time (UTC, aligned to first extreme)" : "",
      nameLocation: "middle",
      nameGap: 30,
      nameTextStyle: { color: "#94a3b8", fontSize: 10 },
    }));

    const yAxis = assets.map((asset, idx) => {
      const key = shortAsset(asset);
      const color = ASSET_COLORS[key] ?? "#cbd5e1";
      return {
        type: "value",
        gridIndex: idx,
        scale: true,
        axisLine: { show: false },
        axisTick: { show: false },
        axisLabel: {
          color: "#94a3b8",
          fontSize: 10,
          formatter: (v: number) => `${v.toFixed(1)}%`,
        },
        splitLine: { show: true, lineStyle: { color: "rgba(148,163,184,0.12)" } },
        name: key,
        nameLocation: "middle",
        nameRotate: 90,
        nameGap: 36,
        nameTextStyle: {
          color,
          fontSize: 11,
          fontWeight: 600,
        },
      };
    });

    const lineSeries = assets.map((asset, idx) => {
      const key = shortAsset(asset);
      const color = ASSET_COLORS[key] ?? "#cbd5e1";
      const raw = (data.series[asset] ?? []) as [number, number][];
      const cut = raw.filter(([t]) => t <= playhead + 1.0e-6);
      return {
        name: `${key} Return`,
        type: "line",
        xAxisIndex: idx,
        yAxisIndex: idx,
        data: downsamplePairs(cut),
        showSymbol: false,
        smooth: 0.12,
        lineStyle: { color, width: 1.5, opacity: 0.95 },
        animation: false,
        markLine: {
          symbol: ["none", "none"],
          silent: true,
          label: { show: false },
          lineStyle: { color: "rgba(226,232,240,0.35)", width: 1, type: "dashed" },
          data: [{ xAxis: playhead }],
        },
      };
    });

    const extremeSeries = assets.map((asset, idx) => {
      const key = shortAsset(asset);
      const extremes = ((data.extreme_points[asset] ?? []) as [number, number][])
        .filter(([t]) => t <= playhead + 1.0e-6);
      return {
        name: `${key} Extreme`,
        type: "scatter",
        xAxisIndex: idx,
        yAxisIndex: idx,
        data: extremes,
        symbolSize: 5,
        animation: false,
        itemStyle: {
          color: "#fbbf24",
          opacity: 0.68,
          borderColor: "rgba(254,243,199,0.38)",
          borderWidth: 1,
          shadowBlur: 0.8,
          shadowColor: "rgba(251,191,36,0.06)",
        },
        z: 8,
      };
    });

    return {
      backgroundColor: "transparent",
      animation: false,
      title: {
        text: "Asset Returns (% log-return)",
        left: 14,
        top: -4,
        textStyle: { color: "#e2e8f0", fontSize: 12, fontWeight: 600 },
      },
      grid,
      xAxis,
      yAxis,
      axisPointer: {
        link: [{ xAxisIndex: "all" }],
      },
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "line" },
        backgroundColor: "rgba(15,23,42,0.96)",
        borderColor: "rgba(148,163,184,0.25)",
        textStyle: { color: "#e2e8f0" },
        formatter: (params: any) => {
          const items = Array.isArray(params) ? params : [params];
          const axisValue = Number(items[0]?.axisValue ?? 0);
          const header = `t = ${formatUtcDate(data.alignment.start_datetime_utc, axisValue, true)}`;
          const rows = items
            .filter((p: any) => p?.seriesType === "line")
            .map((p: any) => `${p.marker}${p.seriesName}: ${Number(p.value?.[1] ?? p.value).toFixed(3)}%`);
          return [header, ...rows].join("<br/>");
        },
      },
      series: [...lineSeries, ...extremeSeries],
    };
  }, [data, currentTime]);

  if (loading) {
    return (
      <div className="h-full rounded-2xl border border-white/10 bg-white/5 p-4 text-xs text-slate-400">
        Loading aligned return tracks...
      </div>
    );
  }

  if (error) {
    return (
      <div className="h-full rounded-2xl border border-amber-400/30 bg-amber-500/10 p-4 text-xs text-amber-200">
        {error}
      </div>
    );
  }

  if (!option) {
    return (
      <div className="h-full rounded-2xl border border-white/10 bg-white/5 p-4 text-xs text-slate-400">
        Return tracks are unavailable for this run.
      </div>
    );
  }

  return (
    <div className="h-full rounded-2xl border border-white/10 bg-white/5 p-2 shadow-lg">
      <ReactECharts option={option} style={{ height: "100%", width: "100%" }} opts={{ renderer: "canvas" }} />
    </div>
  );
}
