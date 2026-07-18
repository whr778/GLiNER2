"use client";

import { useRef, useState } from "react";
import { importUrl } from "@/lib/api";
import type { ExtractOptions } from "@/lib/types";

type Props = {
  text: string;
  setText: (t: string) => void;
  options: ExtractOptions;
  setOptions: (o: ExtractOptions) => void;
  loading: boolean;
  onExtract: () => void;
};

export default function InputPanel({ text, setText, options, setOptions, loading, onExtract }: Props) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [url, setUrl] = useState("");
  const [importing, setImporting] = useState(false);
  const [importErr, setImportErr] = useState<string | null>(null);

  async function handleFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) setText(await file.text());
    e.target.value = "";
  }

  async function handleImport() {
    if (!url.trim()) return;
    setImporting(true);
    setImportErr(null);
    try {
      const res = await importUrl(url.trim());
      setText(res.text);
    } catch (err: any) {
      setImportErr(err.message || "import failed");
    } finally {
      setImporting(false);
    }
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
        <input ref={fileRef} type="file" accept=".txt,text/plain" hidden onChange={handleFile} />
        <span className="spacer" />
        <span className="hint">{text.length.toLocaleString()} chars</span>
      </div>

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
          <label>Model (blank = server default)</label>
          <input
            type="text"
            placeholder="fastino/gliner2-base-v1 or a local path"
            value={options.model ?? ""}
            onChange={(e) => set("model", e.target.value || null)}
          />
        </div>
      </details>

      <div className="field row between">
        <span className="hint">Runs the model on the backend.</span>
        <button className="primary" onClick={onExtract} disabled={loading || !text.trim()}>
          {loading ? "Extracting…" : "Extract"}
        </button>
      </div>
    </div>
  );
}
