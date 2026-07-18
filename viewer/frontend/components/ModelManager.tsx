"use client";

import { useState } from "react";
import { addModel, removeModel } from "@/lib/api";
import type { ModelEntry } from "@/lib/types";

type Props = {
  models: ModelEntry[];
  setModels: (m: ModelEntry[]) => void;
  onClose: () => void;
};

export default function ModelManager({ models, setModels, onClose }: Props) {
  const [path, setPath] = useState("");
  const [label, setLabel] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function add() {
    if (!path.trim()) return;
    setBusy(true);
    setErr(null);
    try {
      setModels(await addModel(path.trim(), label.trim() || undefined));
      setPath("");
      setLabel("");
    } catch (e: any) {
      setErr(e.message || "add failed");
    } finally {
      setBusy(false);
    }
  }

  async function remove(p: string) {
    setBusy(true);
    setErr(null);
    try {
      setModels(await removeModel(p));
    } catch (e: any) {
      setErr(e.message || "remove failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <div className="backdrop" onClick={onClose} />
      <div className="modal card">
        <div className="row between">
          <h2>Manage models</h2>
          <button onClick={onClose}>Close</button>
        </div>

        <div className="field">
          <label>Add a model — HF repo id or a local checkpoint path</label>
          <input
            type="text"
            placeholder="fastino/gliner2-base-v1 or /path/to/checkpoint/best"
            value={path}
            onChange={(e) => setPath(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && add()}
          />
          <div className="row" style={{ marginTop: 6 }}>
            <input
              type="text"
              placeholder="label (optional)"
              value={label}
              onChange={(e) => setLabel(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && add()}
            />
            <button className="primary" onClick={add} disabled={busy || !path.trim()}>Add</button>
          </div>
          {err && <div className="error">{err}</div>}
        </div>

        <div className="field">
          <label>Registered models</label>
          <table>
            <tbody>
              {models.map((m) => (
                <tr key={m.path}>
                  <td style={{ width: 92 }}><span className="hint">{m.source}</span></td>
                  <td>
                    <strong>{m.label}</strong>
                    <div className="hint mono" style={{ wordBreak: "break-all" }}>{m.path}</div>
                  </td>
                  <td style={{ width: 90, textAlign: "right" }}>
                    {m.source === "saved" ? (
                      <button onClick={() => remove(m.path)} disabled={busy}>Remove</button>
                    ) : (
                      <span className="hint">built-in</span>
                    )}
                  </td>
                </tr>
              ))}
              {models.length === 0 && (
                <tr><td colSpan={3}><div className="empty">No models yet.</div></td></tr>
              )}
            </tbody>
          </table>
          <div className="hint" style={{ marginTop: 6 }}>
            <strong>default</strong>/<strong>discovered</strong> (local <span className="mono">out/**/best</span>) are always offered;
            only <strong>saved</strong> entries can be removed.
          </div>
        </div>
      </div>
    </>
  );
}
