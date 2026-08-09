import type { ExtractOptions, ExtractResponse, ModelEntry, Preset } from "./types";

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

// The schema a model ships in its config (default_schema), or null if it has
// none / can't be read. Null on failure by design: the caller falls back to
// corpus-name matching, so a missing endpoint or bad model id degrades quietly.
export async function getModelSchema(model: string): Promise<Record<string, any> | null> {
  try {
    const res = await fetch(`${API_BASE}/model-schema?model=${encodeURIComponent(model)}`);
    if (!res.ok) return null;
    const body = await res.json();
    return body.schema ?? null;
  } catch {
    return null;
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

export type EkfFeed = { path: string; articles: number; truth: string | null };

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
