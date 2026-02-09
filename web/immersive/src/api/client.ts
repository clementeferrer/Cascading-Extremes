import { EventRecord, MetricsRecord, RunMeta, RunsIndex } from "./types";

const envUrl = import.meta.env.VITE_API_URL as string | undefined;
const API_URL =
  envUrl === "relative"
    ? window.location.origin
    : (envUrl ? envUrl.replace(/\/$/, "") : "http://localhost:8000");

export async function getRuns() {
  const res = await fetch(`${API_URL}/runs`);
  const json = await res.json();
  return RunsIndex.parse(json).runs;
}

export async function getMeta(runId: string): Promise<RunMeta> {
  const res = await fetch(`${API_URL}/runs/${runId}/meta`);
  const json = await res.json();
  return RunMeta.parse(json);
}

export async function getEvents(runId: string, offset = 0, limit = 10000): Promise<EventRecord[]> {
  const res = await fetch(`${API_URL}/runs/${runId}/events?offset=${offset}&limit=${limit}`);
  const json = await res.json();
  return json.events.map((e: unknown) => EventRecord.parse(e));
}

export async function getMetrics(runId: string, offset = 0, limit = 10000): Promise<MetricsRecord[]> {
  const res = await fetch(`${API_URL}/runs/${runId}/metrics?offset=${offset}&limit=${limit}`);
  const json = await res.json();
  return json.metrics.map((m: unknown) => MetricsRecord.parse(m));
}

export async function getSummary(runId: string): Promise<Record<string, unknown>> {
  const res = await fetch(`${API_URL}/runs/${runId}/summary`);
  return res.json();
}

export async function generateContinue(payload: {
  theta: number;
  phi: number;
  magnitude: number;
  max_time: number;
  config?: string;
  seed?: number;
}): Promise<{ run_id: string }> {
  const res = await fetch(`${API_URL}/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    throw new Error(`Generate failed: ${res.status}`);
  }
  return res.json();
}
