import type { ExtractionResult, Span } from "./types";

export type Layer = "entities" | "events" | "relations";

export type Mark = {
  start: number;
  end: number;
  label: string; // displayed sublabel: entity type / event type / role / relation
  kind: string; // entity | trigger | argument | head | tail
  confidence?: number;
  eid?: string; // shared id for spans of one event mention / relation instance
  colorKey?: string; // what to color by (event type / relation name); defaults to label
  idx?: number; // 1-based mention index within its event type (for a badge)
};

function asSpan(x: any): Span | null {
  return x && typeof x === "object" && typeof x.start === "number" && typeof x.end === "number"
    ? (x as Span)
    : null;
}

// Reserved top-level keys with a known shape.
export const KNOWN_KEYS = new Set(["entities", "event_extraction", "relation_extraction"]);

export function availableLayers(result: ExtractionResult): Layer[] {
  const layers: Layer[] = [];
  if (result.entities && Object.keys(result.entities).length) layers.push("entities");
  if (result.event_extraction && Object.keys(result.event_extraction).length) layers.push("events");
  if (result.relation_extraction && Object.keys(result.relation_extraction).length) layers.push("relations");
  return layers;
}

export function collectMarks(result: ExtractionResult, layer: Layer): Mark[] {
  const marks: Mark[] = [];
  if (layer === "entities") {
    for (const [label, spans] of Object.entries(result.entities || {})) {
      for (const s of (spans as any[]) || []) {
        const sp = asSpan(s);
        if (sp) marks.push({ start: sp.start!, end: sp.end!, label, kind: "entity", confidence: sp.confidence });
      }
    }
  } else if (layer === "events") {
    let counter = 0;
    for (const [etype, mentions] of Object.entries(result.event_extraction || {})) {
      (mentions as any[])?.forEach((m, mi) => {
        const eid = `evt-${counter++}`;
        for (const t of m.triggers || []) {
          const sp = asSpan(t);
          if (sp) marks.push({ start: sp.start!, end: sp.end!, label: etype, kind: "trigger", confidence: sp.confidence, eid, colorKey: etype, idx: mi + 1 });
        }
        for (const a of m.arguments || []) {
          const sp = asSpan(a.entity);
          if (sp) marks.push({ start: sp.start!, end: sp.end!, label: a.role, kind: "argument", confidence: sp.confidence, eid, colorKey: etype });
        }
      });
    }
  } else if (layer === "relations") {
    let counter = 0;
    for (const [rname, insts] of Object.entries(result.relation_extraction || {})) {
      for (const inst of (insts as any[]) || []) {
        const eid = `rel-${counter++}`;
        const head = Array.isArray(inst) ? inst[0] : inst.head;
        const tail = Array.isArray(inst) ? inst[1] : inst.tail;
        const hs = asSpan(head);
        const ts = asSpan(tail);
        if (hs) marks.push({ start: hs.start!, end: hs.end!, label: rname, kind: "head", confidence: hs.confidence, eid, colorKey: rname });
        if (ts) marks.push({ start: ts.start!, end: ts.end!, label: rname, kind: "tail", confidence: ts.confidence, eid, colorKey: rname });
      }
    }
  }
  return marks;
}

export type Segment = { text: string; mark: Mark | null };

// Non-overlapping segmentation: earliest start wins, then longest; overlaps skipped.
export function buildSegments(text: string, marks: Mark[]): Segment[] {
  const sorted = [...marks].sort((a, b) => a.start - b.start || b.end - b.start - (a.end - a.start));
  const segs: Segment[] = [];
  let cursor = 0;
  for (const m of sorted) {
    if (m.start < cursor || m.start >= m.end || m.end > text.length) continue;
    if (m.start > cursor) segs.push({ text: text.slice(cursor, m.start), mark: null });
    segs.push({ text: text.slice(m.start, m.end), mark: m });
    cursor = m.end;
  }
  if (cursor < text.length) segs.push({ text: text.slice(cursor), mark: null });
  return segs;
}

// Split the "other" (non-reserved) top-level keys into classification vs structure.
function isClassificationValue(v: any): boolean {
  if (typeof v === "string") return true;
  if (v && typeof v === "object" && !Array.isArray(v) && "label" in v) return true;
  if (Array.isArray(v) && v.length > 0) {
    const first = v[0];
    if (typeof first === "string") return true;
    if (first && typeof first === "object" && "label" in first) return true;
  }
  return false;
}

export function splitOtherKeys(result: ExtractionResult): {
  classifications: Record<string, any>;
  structures: Record<string, any>;
} {
  const classifications: Record<string, any> = {};
  const structures: Record<string, any> = {};
  for (const [k, v] of Object.entries(result)) {
    if (KNOWN_KEYS.has(k)) continue;
    if (isClassificationValue(v)) classifications[k] = v;
    else structures[k] = v;
  }
  return { classifications, structures };
}
