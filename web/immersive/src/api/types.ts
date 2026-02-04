import { z } from "zod";

export const RunMeta = z.object({
  run_id: z.string(),
  created_at: z.string(),
  source: z.enum(["simulated", "real", "generative"]),
  assets: z.array(z.string()),
  freq: z.string(),
  threshold: z.object({
    tau: z.number(),
    model: z.string(),
  }),
  model_checkpoint: z.string().nullable().optional(),
  config_hash: z.string(),
});

export const EventRecord = z.object({
  id: z.number(),
  t: z.number(),
  w: z.array(z.number()),
  mag: z.number(),
  u_tau: z.number(),
  asset: z.string(),
  intensity: z.number().nullable().optional(),
  parent_id: z.number().nullable().optional(),
  is_real: z.boolean(),
});

export const MetricsRecord = z.object({
  t: z.number(),
  lambda: z.number().nullable().optional(),
  psi: z.number().nullable().optional(),
  mu: z.number().nullable().optional(),
  mean_mag: z.number().nullable().optional(),
  event_rate: z.number().nullable().optional(),
  per_asset_counts: z.record(z.number()).nullable().optional(),
  direction_density_bin: z.array(z.number()).nullable().optional(),
});

export const RunsIndex = z.object({
  runs: z.array(z.record(z.any())),
});

export type RunMeta = z.infer<typeof RunMeta>;
export type EventRecord = z.infer<typeof EventRecord>;
export type MetricsRecord = z.infer<typeof MetricsRecord>;
