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

const TimeValuePair = z.tuple([z.number(), z.number()]);

export const RunReturnsResponse = z.object({
  run_id: z.string(),
  units: z.literal("log_return_pct"),
  assets: z.array(z.string()),
  series: z.record(z.array(TimeValuePair)),
  extreme_points: z.record(z.array(TimeValuePair)),
  alignment: z.object({
    method: z.string(),
    offset_hours: z.number(),
    candidate_count: z.number(),
    start_datetime_utc: z.string().nullable().optional(),
  }),
  series_mode: z
    .enum(["real_dense", "generative_imputed", "generative_event_only_fallback"])
    .optional(),
  imputation: z
    .object({
      method: z.literal("saits"),
      anchor_method: z.literal("nearest_hour_max_abs"),
      anchor_count: z.number(),
      horizon_hours: z.number(),
      fallback_reason: z.string().optional(),
    })
    .optional(),
  count: z.number().optional(),
});

export type RunMeta = z.infer<typeof RunMeta>;
export type EventRecord = z.infer<typeof EventRecord>;
export type MetricsRecord = z.infer<typeof MetricsRecord>;
export type RunReturnsResponse = z.infer<typeof RunReturnsResponse>;
