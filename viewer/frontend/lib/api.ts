import type { CorefInfo, ExtractOptions, ExtractResponse, ModelEntry, Preset } from "./types";

// Same-origin by default: "/api/*" is proxied to the backend by Next (see
// next.config.mjs), so the browser only talks to the page's own origin -- no
// baked host, no CORS, works however the viewer is reached. Set
// NEXT_PUBLIC_API_BASE to an absolute URL only to point the client directly at
// an external backend (bypasses the proxy).
const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE || "/api";

async function jsonOrThrow(res: Response) {
  if (!res.ok) {
    let detail = `request failed (${res.status})`;
    try {
      const body = await res.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return res.json();
}

export async function extract(
  text: string,
  schema: Record<string, any>,
  options: ExtractOptions,
): Promise<ExtractResponse> {
  const res = await fetch(`${API_BASE}/extract`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, schema, options }),
  });
  return jsonOrThrow(res);
}

export async function importUrl(url: string): Promise<{ text: string }> {
  const res = await fetch(`${API_BASE}/import-url`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
  });
  return jsonOrThrow(res);
}

export async function getPresets(): Promise<Preset[]> {
  const res = await fetch(`${API_BASE}/presets`);
  const body = await jsonOrThrow(res);
  return body.presets ?? [];
}

export async function getModels(): Promise<ModelEntry[]> {
  const res = await fetch(`${API_BASE}/models`);
  const body = await jsonOrThrow(res);
  return body.models ?? [];
}

// What a model ships in its config: the schema it trained on (default_schema) and
// the decode settings it was evaluated with (inference_defaults). Nulls on failure
// by design: the caller falls back to corpus-name matching and the current options.
export type ModelInfo = {
  schema: Record<string, any> | null;
  inferenceDefaults: Partial<ExtractOptions> | null;
  recordDefaults: { record_threshold: number | null; argument_threshold: number | null; boundary: boolean } | null;
  coref: CorefInfo | null;
};

export async function getModelSchema(model: string): Promise<ModelInfo> {
  try {
    const res = await fetch(`${API_BASE}/model-schema?model=${encodeURIComponent(model)}`);
    if (!res.ok) return { schema: null, inferenceDefaults: null, recordDefaults: null, coref: null };
    const body = await res.json();
    return {
      schema: body.schema ?? null,
      inferenceDefaults: body.inference_defaults ?? null,
      recordDefaults: body.record_defaults ?? null,
      coref: body.coref ?? null,
    };
  } catch {
    return { schema: null, inferenceDefaults: null, recordDefaults: null, coref: null };
  }
}

export async function addModel(path: string, label?: string): Promise<ModelEntry[]> {
  const res = await fetch(`${API_BASE}/models`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path, label }),
  });
  const body = await jsonOrThrow(res);
  return body.models ?? [];
}

export async function removeModel(path: string): Promise<ModelEntry[]> {
  const res = await fetch(`${API_BASE}/models?path=${encodeURIComponent(path)}`, {
    method: "DELETE",
  });
  const body = await jsonOrThrow(res);
  return body.models ?? [];
}

// --- EKF tracking -----------------------------------------------------------
// A run takes minutes (N articles x several model calls), so the backend returns
// a job id immediately and we poll it. Polling, not streaming, because the run is
// coarse-grained: one progress tick per article is plenty.

export type EkfRecommendation = {
  window?: string;
  associate?: string;
  gate_threshold?: number;
  max_plausible?: number;
  rollup?: string;
  event_year?: number;
  why: string;
};

export type EkfFeed = {
  path: string;
  articles: number;
  truth: string | null;
  // Documented per-event configuration. Getting these wrong does not degrade a result,
  // it produces a silently empty one -- Turkiye's rollup strips the event type and every
  // stream drops out as nan -- so the panel applies them on selection.
  recommended?: EkfRecommendation | null;
};

export type EkfJob = {
  job_id: string;
  status: "running" | "done" | "error";
  stage: string;
  done: number;
  total: number;
  error: string | null;
  log: string[];
  result: any | null;
};

export async function getEkfFeeds(): Promise<EkfFeed[]> {
  const res = await fetch(`${API_BASE}/ekf-feeds`);
  if (!res.ok) throw new Error(await res.text());
  return (await res.json()).feeds;
}

export async function startEkfTrack(body: Record<string, any>): Promise<EkfJob> {
  const res = await fetch(`${API_BASE}/ekf-track`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getEkfJob(jobId: string): Promise<EkfJob> {
  const res = await fetch(`${API_BASE}/ekf-track/${jobId}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export type PredictionSummary = {
  path: string;
  count: number;
  gold_key: "gold" | "gold_mapped" | null;
  data_file: boolean; // a train/val/test file: its annotations are drawn, there is no prediction
  records: { i: number; chars: number; has_gold: boolean; f1: Record<string, number> }[];
};

export type PredictionRecord = {
  i: number;
  text: string;
  pred: Record<string, any>;
  gold: Record<string, any>;
  has_gold: boolean;
  data_row?: boolean;
  classifications: { task: string; gold: string[]; pred: any; match: boolean }[];
  unlocated: string[];
  f1: Record<string, number>;
  diffs: PredictionDiff[];
  // False when the prediction file has gold clusters but output has no coref_clusters key.
  entity_coref_decoded?: boolean;
};

// One reason a gold or predicted item does not match (backend diffs.py), grouped over occurrences.
export type DiffSide = { label: string; text: string; start: number; end: number; eid?: string | null;
  role?: string; trig?: string[]; idx?: number; tail?: string };
export type PredictionDiff = { reason: string; layer: "entities" | "events" | "relations"; why: string;
  gold: DiffSide | null; pred: DiffSide | null; count: number };

// A tools/infer.py --output file: per-record strict F1 for the pager.
export async function openPredictions(path: string): Promise<PredictionSummary> {
  return jsonOrThrow(await fetch(`${API_BASE}/predictions?path=${encodeURIComponent(path)}`));
}

// One record: prediction and gold in the same result shape, every span flagged `match`.
export async function getPredictionRecord(path: string, i: number): Promise<PredictionRecord> {
  return jsonOrThrow(await fetch(`${API_BASE}/predictions/record?path=${encodeURIComponent(path)}&i=${i}`));
}

export type PredictionFile = { path: string; name: string; mtime: number; size: number; gold_mapped: boolean; spans: boolean };

// infer.py output files under the backend's configured predictions_root, newest first.
export async function listPredictionFiles(): Promise<{ root: string; files: PredictionFile[] }> {
  return jsonOrThrow(await fetch(`${API_BASE}/predictions/files`));
}
