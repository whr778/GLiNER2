import type { Preset } from "./types";

// Map a model path to the corpus preset it was trained on, so selecting e.g.
// `out/fastino/gliner2-base-v1-casie/best` can auto-load the `corpus: casie`
// schema (the exact event types + roles that model expects).
//
// Data-driven, not format-assuming: each corpus preset carries the dataset name
// (`corpus: casie` -> `casie`); we match that token as a trailing path segment
// (`-casie/` or `-casie` at the end), longest token wins so `mendeley-ed` beats
// a stray `ed`.
export function matchCorpusPreset(modelPath: string, presets: Preset[]): Preset | null {
  let best: Preset | null = null;
  let bestLen = 0;
  for (const p of presets) {
    if (p.source !== "corpus") continue;
    const token = p.name.replace(/^corpus:\s*/, "");
    if (!token) continue;
    const esc = token.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const re = new RegExp(`[-/]${esc}(?:/|$)`, "i");
    if (re.test(modelPath) && token.length > bestLen) {
      best = p;
      bestLen = token.length;
    }
  }
  return best;
}
