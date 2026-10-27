"use client";

import { useEffect, useRef, useState } from "react";
import { importUrl, listDatasets, type DatasetFile } from "@/lib/api";
import type { CorefInfo, ExtractOptions, ExtractResponse, ModelEntry } from "@/lib/types";

type Props = {
  text: string;
  setText: (t: string) => void;
  options: ExtractOptions;
  setOptions: (o: ExtractOptions) => void;
  queueing: boolean;
  schemaLoading?: boolean; // a model's schema is being fetched; block a new ticket
  onQueue: () => void;
  source: "text" | "jsonl";
  setSource: (s: "text" | "jsonl") => void;
  dataset: string;
  setDataset: (path: string) => void;
  schemaMode: "gold" | "panel";
  setSchemaMode: (m: "gold" | "panel") => void;
  limit: string;
  setLimit: (n: string) => void;
  models: ModelEntry[];
  onManage: () => void;
  resultData: ExtractResponse | null;
  resultSchema?: Record<string, any> | null;
  onLoadResults: (snap: any) => void;
  coref?: CorefInfo | null;
};

function CorefLine({ coref }: { coref: CorefInfo }) {
  const link = coref.record_coref_link
    ? "The trigger link is trained."
    : "This checkpoint has no trigger link.";
  const merge = coref.record_merge_coreferent === "link"
    ? `Trigger merge is link at threshold ${coref.record_coref_link_threshold}. Extra triggers on one event are merged mentions. Arguments are the representative's. Entity mention clusters stay off unless you turn them on.`
    : coref.record_merge_coreferent === "args"
      ? "Trigger merge is args. That joins triggers which share a high-scoring argument. It is not TriggerLink."
      : `Trigger merge is off. Each selected trigger is its own event. ${link} Entity mention clusters are not part of Extract unless you turn them on.`;
  return (
    <div className="hint" style={{ marginBottom: 8 }}>
      {merge} Global decode joins overlapping trigger spans across windows. It is not TriggerLink.
    </div>
  );
}

export default function InputPanel({ text, setText, options, setOptions, queueing, schemaLoading, onQueue, source, setSource, dataset, setDataset, schemaMode, setSchemaMode, limit, setLimit, models, onManage, resultData, resultSchema, onLoadResults, coref }: Props) {
  const [datasetQuery, setDatasetQuery] = useState("");
  const [datasets, setDatasets] = useState<DatasetFile[]>([]);
  const [datasetErr, setDatasetErr] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const loadRef = useRef<HTMLInputElement>(null);
  const [url, setUrl] = useState("");
  const [sourceName, setSourceName] = useState("");
  const [importing, setImporting] = useState(false);
  const [importErr, setImportErr] = useState<string | null>(null);
  const [loadErr, setLoadErr] = useState<string | null>(null);

  useEffect(() => {
    if (source !== "jsonl") return;
    let cancelled = false;
    const timer = setTimeout(() => {
      listDatasets(datasetQuery)
        .then((rows) => { if (!cancelled) { setDatasets(rows); setDatasetErr(null); } })
        .catch((e) => { if (!cancelled) setDatasetErr(e.message); });
    }, 200);
    return () => { cancelled = true; clearTimeout(timer); };
  }, [source, datasetQuery]);

  async function handleFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) {
      setText(await file.text());
      setSourceName(file.name);
    }
    e.target.value = "";
  }

  // Load a previously-saved .results.json back into the viewer (client-side; no
  // model run). Restores the text, schema, options, and rendered results.
  async function handleLoadResults(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    setLoadErr(null);
    try {
      const snap = JSON.parse(await file.text());
      if (typeof snap?.text !== "string" || !snap?.result || typeof snap.result !== "object") {
        setLoadErr("Not a saved results file (expected text + result).");
        return;
      }
      setSourceName(file.name.replace(/\.results\.json$/i, ".txt"));
      onLoadResults(snap);
    } catch {
      setLoadErr("Could not parse that file as JSON.");
    }
  }

  async function handleImport() {
    if (!url.trim()) return;
    setImporting(true);
    setImportErr(null);
    try {
      const res = await importUrl(url.trim());
      setText(res.text);
      setSourceName(deriveUrlName(url.trim()));
    } catch (err: any) {
      setImportErr(err.message || "import failed");
    } finally {
      setImporting(false);
    }
  }

  function stripExt(name: string): string {
    return name.replace(/\.[^./\\]+$/, "");
  }

  // A filename-ish slug for the imported URL, e.g. en.wikipedia.org_wiki_X.txt.
  function deriveUrlName(u: string): string {
    try {
      const parsed = new URL(u);
      const base = (parsed.hostname + parsed.pathname)
        .replace(/[^\w.-]+/g, "_")
        .replace(/_+/g, "_")
        .replace(/^_|_$/g, "");
      if (base) return base.slice(0, 80) + ".txt";
    } catch {
      /* not a URL */
    }
    return "text.txt";
  }

  // Base name (no extension) of the current input source: the uploaded file
  // name, the imported URL, or "text" when pasted from scratch.
  function baseName(): string {
    return sourceName ? stripExt(sourceName) || "text" : "text";
  }

  // Save `contents` via the native save-as picker (Chrome/Edge), or a normal
  // download (Firefox/Safari).
  async function saveToFile(name: string, contents: string, mime: string, ext: string, desc: string) {
    const picker = (window as any).showSaveFilePicker;
    if (picker) {
      try {
        const handle = await picker({
          suggestedName: name,
          types: [{ description: desc, accept: { [mime]: [ext] } }],
        });
        const writable = await handle.createWritable();
        await writable.write(contents);
        await writable.close();
        return;
      } catch (e: any) {
        if (e?.name === "AbortError") return; // user cancelled the dialog
        // any other error: fall through to the download fallback
      }
    }
    const blob = new Blob([contents], { type: `${mime};charset=utf-8` });
    const href = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = href;
    a.download = name;
    a.click();
    URL.revokeObjectURL(href);
  }

  function saveText() {
    if (!text.trim()) return;
    saveToFile(baseName() + ".txt", text, "text/plain", ".txt", "Text");
  }

  function saveResults() {
    if (!resultData) return;
    // Self-contained snapshot: text + schema + options + result, so "Load
    // results" restores the full view (incl. the schema-driven tabs).
    const snapshot = {
      text: resultData.text,
      schema: resultSchema ?? undefined,
      options,
      result: resultData.result,
    };
    saveToFile(
      baseName() + ".results.json",
      JSON.stringify(snapshot, null, 2),
      "application/json",
      ".json",
      "Results JSON",
    );
  }

  const set = <K extends keyof ExtractOptions>(k: K, v: ExtractOptions[K]) =>
    setOptions({ ...options, [k]: v });

  return (
    <div className="card">
      <h2>New job</h2>
      <div className="toggle" style={{ marginBottom: 10 }}>
        <button type="button" className={source === "text" ? "active" : ""} onClick={() => setSource("text")}>Text</button>
        <button type="button" className={source === "jsonl" ? "active" : ""} onClick={() => setSource("jsonl")}>JSONL</button>
      </div>
      {source === "text" ? (
      <textarea
        rows={9}
        placeholder="Paste text to analyze…"
        value={text}
        onChange={(e) => setText(e.target.value)}
      />
      ) : (
      <div className="field">
        <label>Dataset under data/</label>
        <input
          type="text"
          placeholder="casie_off.test"
          value={datasetQuery}
          onChange={(e) => setDatasetQuery(e.target.value)}
        />
        <select style={{ marginTop: 8 }} value={dataset} onChange={(e) => setDataset(e.target.value)}>
          <option value="">{datasets.length ? `${datasets.length} file(s)` : "no match"}</option>
          {dataset && !datasets.some((d) => d.path === dataset) && (
            <option value={dataset}>{dataset}</option>
          )}
          {datasets.map((d) => (
            <option key={d.path} value={d.path}>{d.path} ({(d.bytes / 1024).toFixed(0)} KB)</option>
          ))}
        </select>
        {datasetErr && <div className="error">{datasetErr}</div>}
        <div className="field">
          <label>Schema</label>
          <select value={schemaMode} onChange={(e) => setSchemaMode(e.target.value as "gold" | "panel")}>
            <option value="gold">Each record's own gold labels (evaluation)</option>
            <option value="panel">The schema on this screen</option>
          </select>
        </div>
        <div className="field">
          <label>First N records (blank = the whole file)</label>
          <input type="number" min={1} value={limit} onChange={(e) => setLimit(e.target.value)} />
        </div>
        <div className="hint">
          Writes an infer.py JSONL with spans, then this job opens in Predictions vs gold.
          A long file on MPS keeps memory for every new length. CPU is the safe device here.
          {schemaMode === "gold" ? " The schema below is not sent." : ""}
        </div>
      </div>
      )}

      {source === "text" && <div className="field row">
        <button type="button" onClick={() => fileRef.current?.click()}>Upload .txt</button>
        <button type="button" onClick={saveText} disabled={!text.trim()} title="Save the current text to a file">Save text…</button>
        <button type="button" onClick={saveResults} disabled={!resultData} title="Save the extracted results as <input>.results.json">Save results…</button>
        <button type="button" onClick={() => loadRef.current?.click()} title="Load a saved <input>.results.json back into the viewer">Load results…</button>
        <input ref={fileRef} type="file" accept=".txt,text/plain" hidden onChange={handleFile} />
        <input ref={loadRef} type="file" accept=".json,application/json" hidden onChange={handleLoadResults} />
        <span className="spacer" />
        <span className="hint">{text.length.toLocaleString()} chars</span>
      </div>}
      {loadErr && source === "text" && <div className="error">{loadErr}</div>}

      {source === "text" && <div className="field">
        <label>Import from URL</label>
        <div className="row">
          <input
            type="text"
            placeholder="https://example.com/article"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleImport()}
          />
          <button type="button" onClick={handleImport} disabled={importing || !url.trim()}>
            {importing ? "Importing…" : "Import"}
          </button>
        </div>
        {importErr && <div className="error">{importErr}</div>}
      </div>}

      <details className="field">
        <summary className="hint" style={{ cursor: "pointer" }}>Options</summary>
        <div className="field">
          <label>Threshold: {options.threshold.toFixed(2)}</label>
          <input
            type="range" min={0} max={1} step={0.05}
            value={options.threshold}
            onChange={(e) => set("threshold", parseFloat(e.target.value))}
          />
        </div>
        {(["record_threshold", "argument_threshold"] as const).map((key) => {
          const label = key === "record_threshold" ? "Record (instance) gate" : "Argument gate";
          const value = options[key];
          return (
            <div className="field" key={key}>
              <label>
                <input
                  type="checkbox" style={{ width: "auto", marginRight: 6 }}
                  checked={value != null}
                  onChange={(e) => set(key, e.target.checked ? options.threshold : null)}
                />
                {label}: {value != null ? value.toFixed(2) : "follows threshold"}
              </label>
              {value != null && (
                <input
                  type="range" min={0} max={1} step={0.05}
                  value={value}
                  onChange={(e) => set(key, parseFloat(e.target.value))}
                />
              )}
            </div>
          );
        })}
        <div className="hint">
          Boundary models only. The record gate decides which event/record instances exist; the
          argument gate decides which arguments fill them. Unchecked, both follow the threshold.
        </div>
        <div className="field row">
          <input
            id="gd" type="checkbox"
            style={{ width: "auto" }}
            checked={options.global_decode}
            onChange={(e) => set("global_decode", e.target.checked)}
          />
          <label htmlFor="gd" style={{ margin: 0 }}>
            Global decode (cross-window event assembly for long docs)
          </label>
        </div>
        <div className="field">
          <label>Device</label>
          <select value={options.device ?? "auto"} onChange={(e) => set("device", e.target.value)}>
            <option value="auto">auto (CUDA → MPS → CPU)</option>
            <option value="cpu">cpu</option>
            <option value="mps">mps (Apple Silicon GPU)</option>
            <option value="cuda">cuda (NVIDIA GPU)</option>
          </select>
          <div className="hint">
            For many-label event models, cpu is often much faster than mps (which has
            high per-op overhead). Unavailable devices fall back to auto.
          </div>
        </div>
        {coref?.record_coref_link && (
          <div className="field">
            <label>
              <input
                type="checkbox" style={{ width: "auto", marginRight: 6 }}
                checked={options.merge_coreferent != null}
                onChange={(e) => {
                  setOptions({
                    ...options,
                    merge_coreferent: e.target.checked ? "off" : null,
                    coref_threshold: e.target.checked ? options.coref_threshold : null,
                  });
                }}
              />
              Override trigger merge
            </label>
            {options.merge_coreferent != null && (
              <>
                <select
                  value={options.merge_coreferent}
                  onChange={(e) => set("merge_coreferent", e.target.value as "off" | "link")}
                >
                  <option value="off">off</option>
                  <option value="link">link</option>
                </select>
                {options.merge_coreferent === "link" && (
                  <label>
                    <input
                      type="checkbox" style={{ width: "auto", marginRight: 6 }}
                      checked={options.coref_threshold != null}
                      onChange={(e) => set("coref_threshold", e.target.checked ? (coref.record_coref_link_threshold ?? 0.5) : null)}
                    />
                    Link threshold: {options.coref_threshold != null ? options.coref_threshold.toFixed(2) : "checkpoint"}
                  </label>
                )}
                {options.merge_coreferent === "link" && options.coref_threshold != null && (
                  <input
                    type="range" min={0} max={1} step={0.05}
                    value={options.coref_threshold}
                    onChange={(e) => set("coref_threshold", parseFloat(e.target.value))}
                  />
                )}
                <div className="hint">
                  Off until you check this. Link merges same-type triggers at the threshold; arguments stay the
                  representative&apos;s. The checkpoint value is used when the threshold box is unchecked.
                </div>
              </>
            )}
          </div>
        )}
        {coref?.entity_coref_link && (
          <div className="field">
            <label>
              <input
                type="checkbox" style={{ width: "auto", marginRight: 6 }}
                checked={options.entity_coref_threshold != null}
                onChange={(e) => set("entity_coref_threshold", e.target.checked ? 0.5 : null)}
              />
              Entity mention clusters{options.entity_coref_threshold != null ? `: ${options.entity_coref_threshold.toFixed(2)}` : ""}
            </label>
            {options.entity_coref_threshold != null && (
              <input
                type="range" min={0} max={1} step={0.05}
                value={options.entity_coref_threshold}
                onChange={(e) => set("entity_coref_threshold", parseFloat(e.target.value))}
              />
            )}
            <div className="hint">
              Off until you set a link-logit cut-off. That number is not from the checkpoint. Clusters are
              predicted entity spans plus pronouns. They are not added as event arguments.
            </div>
          </div>
        )}
        <div className="field">
          <label>Model (blank = server default)</label>
          <div className="row">
            <input
              type="text"
              list="model-list"
              placeholder="fastino/gliner2-base-v1 or a local path"
              value={options.model ?? ""}
              onChange={(e) => set("model", e.target.value || null)}
            />
            <datalist id="model-list">
              {models.map((m) => (
                <option key={m.path} value={m.path}>{m.label}</option>
              ))}
            </datalist>
            <button type="button" onClick={onManage} title="Manage models">Manage</button>
          </div>
          <div className="hint">Used models are remembered; local checkpoints are auto-discovered.</div>
        </div>
      </details>

      {coref && <CorefLine coref={coref} />}
      <div className="field row between">
        <span className="hint">Returns a ticket. One model runs at a time.</span>
        <button
          className="primary"
          onClick={onQueue}
          disabled={queueing || schemaLoading || (source === "text" ? !text.trim() : !dataset)}
          title={schemaLoading ? "Loading the selected model's schema…" : undefined}
        >
          {queueing ? "Queuing…" : schemaLoading ? "Loading schema…" : "Queue job"}
        </button>
      </div>
    </div>
  );
}
