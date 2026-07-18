"use client";

import { useState } from "react";
import type { Preset } from "@/lib/types";

type Props = {
  presets: Preset[];
  schema: Record<string, any>;
  setSchema: (s: Record<string, any>) => void;
};

export default function SchemaPanel({ presets, schema, setSchema }: Props) {
  const [text, setText] = useState(() => JSON.stringify(schema, null, 2));
  const [err, setErr] = useState<string | null>(null);
  const [selected, setSelected] = useState("");

  function applyText(t: string) {
    setText(t);
    try {
      const parsed = JSON.parse(t);
      setSchema(parsed);
      setErr(null);
    } catch (e: any) {
      setErr("Invalid JSON: " + e.message);
    }
  }

  function loadPreset(name: string) {
    setSelected(name);
    const p = presets.find((x) => x.name === name);
    if (p) applyText(JSON.stringify(p.schema, null, 2));
  }

  const builtin = presets.filter((p) => p.source === "builtin");
  const corpus = presets.filter((p) => p.source === "corpus");

  return (
    <div className="card">
      <h2>Schema — what to extract</h2>
      <div className="field">
        <label>Load a preset</label>
        <select value={selected} onChange={(e) => loadPreset(e.target.value)}>
          <option value="">— choose a preset —</option>
          {builtin.length > 0 && (
            <optgroup label="Built-in">
              {builtin.map((p) => (
                <option key={p.name} value={p.name}>{p.name}</option>
              ))}
            </optgroup>
          )}
          {corpus.length > 0 && (
            <optgroup label="From training corpora">
              {corpus.map((p) => (
                <option key={p.name} value={p.name}>{p.name}</option>
              ))}
            </optgroup>
          )}
        </select>
      </div>

      <div className="field">
        <label>Schema JSON</label>
        <textarea
          rows={12}
          className="mono"
          value={text}
          onChange={(e) => applyText(e.target.value)}
          spellCheck={false}
        />
        {err ? (
          <div className="error">{err}</div>
        ) : (
          <div className="hint">
            Keys: <span className="mono">entities, events, relations, classifications, structures</span>
          </div>
        )}
      </div>
    </div>
  );
}
