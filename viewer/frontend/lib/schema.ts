// Helpers for the co-located default_schema and its `open_vocab` marker.
//
// A model's shipped schema may list task types it was trained on but is
// open-vocabulary for (too many distinct labels to pin down) under `open_vocab`,
// e.g. { events: {...}, open_vocab: ["entities"] }. scaffoldSchema turns each
// such task into an empty, fillable field so the user sees the capability and
// knows to supply specifics; pruneSchema strips those still-empty fields (and the
// marker) before extraction, since the backend rejects empty dimensions.

const EMPTY: Record<string, any> = {
  entities: [],
  relations: [],
  events: {},
  classifications: [],
};

// Expand `open_vocab` task types into empty placeholder fields the user can fill.
// Concrete dimensions are kept as-is; the marker itself is dropped.
export function scaffoldSchema(schema: Record<string, any>): Record<string, any> {
  const { open_vocab, ...rest } = schema || {};
  const out: Record<string, any> = { ...rest };
  for (const task of (open_vocab as string[]) || []) {
    if (!(task in out) && task in EMPTY) out[task] = EMPTY[task];
  }
  return out;
}

function isEmpty(v: any): boolean {
  if (v == null) return true;
  if (Array.isArray(v)) return v.length === 0;
  if (typeof v === "object") return Object.keys(v).length === 0;
  return false;
}

// Drop the `open_vocab` marker and any still-empty dimension so the payload sent
// to /extract is a valid schema (empty dimensions are rejected server-side).
export function pruneSchema(schema: Record<string, any>): Record<string, any> {
  const out: Record<string, any> = {};
  for (const [k, v] of Object.entries(schema || {})) {
    if (k === "open_vocab") continue;
    if (isEmpty(v)) continue;
    out[k] = v;
  }
  return out;
}
