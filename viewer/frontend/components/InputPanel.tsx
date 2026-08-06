"use client";

import { useRef, useState } from "react";
import { importUrl } from "@/lib/api";
import type { ExtractOptions, ExtractResponse, ModelEntry } from "@/lib/types";

type Props = {
  text: string;
  setText: (t: string) => void;
  options: ExtractOptions;
  setOptions: (o: ExtractOptions) => void;
  loading: boolean;
  schemaLoading?: boolean; // a model's schema is being fetched; block extraction
  onExtract: () => void;
  models: ModelEntry[];
  onManage: () => void;
  resultData: ExtractResponse | null;
  resultSchema?: Record<string, any> | null;
  onLoadResults: (snap: any) => void;
};

export default function InputPanel({ text, setText, options, setOptions, loading, schemaLoading, onExtract, models, onManage, resultData, resultSchema, onLoadResults }: Props) {
  const fileRef = useRef<HTMLInputElement>(null);
  const loadRef = useRef<HTMLInputElement>(null);
  const [url, setUrl] = useState("");
  const [sourceName, setSourceName] = useState("");
  const [importing, setImporting] = useState(false);
  const [importErr, setImportErr] = useState<string | null>(null);
  const [loadErr, setLoadErr] = useState<string | null>(null);

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
      <h2>Input</h2>
      <textarea
        rows={9}
        placeholder="Paste text to analyze…"
        value={text}
        onChange={(e) => setText(e.target.value)}
      />

      <div className="field row">
        <button type="button" onClick={() => fileRef.current?.click()}>Upload .txt</button>
        <button type="button" onClick={saveText} disabled={!text.trim()} title="Save the current text to a file">Save text…</button>
        <button type="button" onClick={saveResults} disabled={!resultData} title="Save the extracted results as <input>.results.json">Save results…</button>
        <button type="button" onClick={() => loadRef.current?.click()} title="Load a saved <input>.results.json back into the viewer">Load results…</button>
        <input ref={fileRef} type="file" accept=".txt,text/plain" hidden onChange={handleFile} />
        <input ref={loadRef} type="file" accept=".json,application/json" hidden onChange={handleLoadResults} />
        <span className="spacer" />
        <span className="hint">{text.length.toLocaleString()} chars</span>
      </div>
      {loadErr && <div className="error">{loadErr}</div>}

      <div className="field">
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
      </div>

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

      <div className="field row between">
        <span className="hint">Runs the model on the backend.</span>
        <button
          className="primary"
          onClick={onExtract}
          disabled={loading || schemaLoading || !text.trim()}
          title={schemaLoading ? "Loading the selected model's schema…" : undefined}
        >
          {loading ? "Extracting…" : schemaLoading ? "Loading schema…" : "Extract"}
        </button>
      </div>
    </div>
  );
}
