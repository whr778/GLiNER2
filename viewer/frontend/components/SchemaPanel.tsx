"use client";

import { useEffect, useState } from "react";
import type { Preset } from "@/lib/types";

type Props = {
  presets: Preset[];
  schema: Record<string, any>;
  setSchema: (s: Record<string, any>) => void;
  note?: string | null;
  clearNote?: () => void;
  selectedPreset?: string; // preset to reflect in the dropdown (e.g. matched to the model)
};

export default function SchemaPanel({ presets, schema, setSchema, note, clearNote, selectedPreset }: Props) {
  const [text, setText] = useState(() => JSON.stringify(schema, null, 2));
  const [err, setErr] = useState<string | null>(null);
  const [selected, setSelected] = useState("");

  // Follow the preset the parent picked (e.g. auto-matched to the model).
  useEffect(() => {
    if (selectedPreset) setSelected(selectedPreset);
  }, [selectedPreset]);

  // Reflect schema changes that come from outside the editor (e.g. auto-loaded
  // to match the selected model). Ignore echoes of our own edits.
  useEffect(() => {
    let current: unknown;
    try {
      current = JSON.parse(text);
    } catch {
      current = undefined;
    }
    if (JSON.stringify(current) !== JSON.stringify(schema)) {
      setText(JSON.stringify(schema, null, 2));
      setErr(null);
    }
  }, [schema]); // eslint-disable-line react-hooks/exhaustive-deps

  function applyText(t: string) {
    setText(t);
    clearNote?.();
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
    clearNote?.();
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
        {note && <div className="hint" style={{ marginTop: 6, color: "var(--accent)" }}>{note}</div>}
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
