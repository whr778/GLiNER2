export type Span = {
  text: string;
  confidence?: number;
  start?: number;
  end?: number;
};

// A span field may be a full span dict or a bare string (when spans are off).
export type SpanLike = Span | string;

export type EventMention = {
  triggers: SpanLike[];
  arguments: { role: string; entity: SpanLike }[];
};

// The flat, task-keyed result dict returned by /extract's `result`.
export type ExtractionResult = Record<string, any>;

export type Preset = {
  name: string;
  schema: Record<string, any>;
  source: string;
};

export type ModelEntry = {
  path: string;
  label: string;
  source?: string; // default | discovered | saved
};

export type ExtractOptions = {
  threshold: number;
  chunk_size: number;
  chunk_overlap: number;
  global_decode: boolean;
  beam_width: number;
  model?: string | null;
  device?: string; // auto | cpu | mps | cuda
  // Boundary models: the record (instance) gate and the argument (field) gate. null =
  // the model's own setting, which by default follows `threshold`.
  record_threshold?: number | null;
  argument_threshold?: number | null;
};

export const DEFAULT_OPTIONS: ExtractOptions = {
  threshold: 0.5,
  chunk_size: 384,
  chunk_overlap: 128,
  global_decode: false,
  beam_width: 8,
  model: null,
  device: "auto",
  record_threshold: null,
  argument_threshold: null,
};

export type ExtractResponse = {
  text: string;
  result: ExtractionResult;
  device?: string;
  // {category: {label sent: the model's trained spelling}} for labels the backend rewrote.
  label_map_applied?: Record<string, Record<string, string>>;
};
