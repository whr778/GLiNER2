"use client";

import { useEffect, useMemo, useState } from "react";
import AnnotatedText from "./AnnotatedText";
import PredictionsHelp from "./PredictionsHelp";
import { collectMarks, type Layer } from "@/lib/spans";
import {
  getPredictionRecord, listPredictionFiles, openPredictions,
  type DiffSide, type PredictionFile, type PredictionRecord, type PredictionSummary,
} from "@/lib/api";

function sideText(d: DiffSide | null): string {
  if (!d) return "-";
  const lab = d.label + (d.role ? ` / ${d.role}` : "") + (d.idx ? ` #${d.idx}` : "");
  return `"${d.text}"${d.tail ? ` -> "${d.tail}"` : ""}  (${lab})` + (d.trig ? `  under ${d.trig.join(", ") || "-"}` : "");
}

const LAYERS: Layer[] = ["entities", "events", "relations", "coref"];
const LAYER_KEY: Record<Layer, string> = { entities: "entities", events: "event_extraction", relations: "relation_extraction", coref: "coref_clusters" };
const HEADS = ["entity", "event_trigger", "event_argument", "relation"];
const STRICT = new Set(HEADS);
const CLUSTER = ["event_cluster", "event_cluster_argument"];

function f1Text(f1: Record<string, number>): string {
  const parts: string[] = [];
  const strict = HEADS.filter((h) => h in f1).map((h) => `${h} ${f1[h].toFixed(3)}`);
  if (strict.length) parts.push(`strict micro F1 ${strict.join(" · ")}`);
  if ("coref" in f1) parts.push(`mention-pair F1 coref ${f1.coref.toFixed(3)}`);
  const cluster = CLUSTER.filter((h) => h in f1).map((h) => `${h} ${f1[h].toFixed(3)}`);
  if (cluster.length) parts.push(`cluster F1 ${cluster.join(" · ")}`);
  for (const [h, v] of Object.entries(f1)) {
    if (!STRICT.has(h) && h !== "coref" && !CLUSTER.includes(h)) parts.push(`${h} ${v.toFixed(3)}`);
  }
  return parts.join(" · ") || "nothing to score";
}
const PATH_KEY = "predictions.path";

function hasLayer(r: Record<string, any>, l: Layer): boolean {
  return Object.values(r[LAYER_KEY[l]] || {}).some((v: any) => Array.isArray(v) && v.length > 0);
}

// Pages through a tools/infer.py --output file: the prediction drawn as usual, the record's own
// gold as a rim (solid = matches, dashed = overlap that differs -- diffs.py names how -- rim only = missed),
// matched under the scorer's strict keys so the view agrees with the strict F1.
export default function PredictionsPanel({ openPath, reloadKey = "" }: { openPath?: string; reloadKey?: string }) {
  const [path, setPath] = useState(() => {
    try { return localStorage.getItem(PATH_KEY) ?? ""; } catch { return ""; }
  });
  const [summary, setSummary] = useState<PredictionSummary | null>(null);
  const [order, setOrder] = useState("file");
  const [pos, setPos] = useState(0);
  const [rec, setRec] = useState<PredictionRecord | null>(null);
  const [layer, setLayer] = useState<Layer>("entities");
  const [showGold, setShowGold] = useState(true);
  const [diffOnly, setDiffOnly] = useState(false);
  const [pinned, setPinned] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [help, setHelp] = useState(false);

  const [files, setFiles] = useState<{ root: string; files: PredictionFile[] } | null>(null);
  const refreshFiles = () => listPredictionFiles().then(setFiles).catch((e) => setError(e.message));
  useEffect(() => { refreshFiles(); }, []);
  const picked = files?.files.find((f) => f.path === summary?.path);

  async function onOpen(target = path) {
    setError(null);
    try {
      setSummary(await openPredictions(target.trim()));
      setPos(0);
      try { localStorage.setItem(PATH_KEY, target.trim()); } catch { /* per-viewer convenience only */ }
    } catch (e: any) {
      setError(e.message);
      setSummary(null);
    }
  }

  // A job hands its output path in. Reload when the job settles, not on every record.
  useEffect(() => {
    if (!openPath) return;
    let cancelled = false;
    setPath(openPath);
    openPredictions(openPath)
      .then((s) => { if (!cancelled) { setSummary(s); setPos(0); setError(null); } })
      .catch((e) => { if (!cancelled) setError(e.message); });
    return () => { cancelled = true; };
  }, [openPath, reloadKey]);

  // File order, or worst-first on one head (records without that head go last).
  const sequence = useMemo(() => {
    const rs = summary?.records ?? [];
    if (order === "file") return rs.map((r) => r.i);
    const f = (r: (typeof rs)[number]) => r.f1[order] ?? 2;
    return [...rs].sort((a, b) => f(a) - f(b) || a.i - b.i).map((r) => r.i);
  }, [summary, order]);
  const index = sequence[pos];
  const scoreHeads = useMemo(() => {
    const heads = [...HEADS];
    if (summary?.records.some((r) => "coref" in (r.f1 || {}))) heads.push("coref");
    return heads;
  }, [summary]);

  useEffect(() => {
    if (!summary || index === undefined) return;
    let cancelled = false;
    getPredictionRecord(summary.path, index)
      .then((r) => {
        if (cancelled) return;
        setRec(r);
        setPinned(null);
        setLayer((cur) => (hasLayer(r.pred, cur) || hasLayer(r.gold, cur) ? cur
          : LAYERS.find((l) => hasLayer(r.pred, l) || hasLayer(r.gold, l)) ?? "entities"));
      })
      .catch((e) => setError(e.message));
    return () => { cancelled = true; };
  }, [summary, index]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.tagName === "INPUT") return;
      if (e.key === "ArrowRight") setPos((p) => Math.min(p + 1, sequence.length - 1));
      if (e.key === "ArrowLeft") setPos((p) => Math.max(p - 1, 0));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [sequence.length]);

  const marks = useMemo(() => {
    if (!rec) return [];
    const all = [...collectMarks(rec.pred, layer), ...(showGold ? collectMarks(rec.gold, layer, true) : [])];
    return diffOnly ? all.filter((m) => !m.match) : all;
  }, [rec, layer, showGold, diffOnly]);

  return (
    <div>
      <div className="card">
        <div className="row between">
          <h2>Predictions file</h2>
          <button className="secondary" onClick={() => setHelp(true)}>Help</button>
        </div>
        <div className="row" style={{ gap: 8, marginBottom: 8 }}>
          <select style={{ flex: 1 }} value={files?.files.some((f) => f.path === path) ? path : ""}
                  onChange={(e) => { if (e.target.value) { setPath(e.target.value); onOpen(e.target.value); } }}>
            <option value="">
              {files ? `${files.files.length} infer.py output file(s) in ${files.root}` : "loading..."}
            </option>
            {files?.files.map((f) => (
              <option key={f.path} value={f.path}>
                {f.name} -- {new Date(f.mtime * 1000).toLocaleString()} -- {(f.size / 1024).toFixed(0)} KB
                {f.spans ? "" : " -- no spans"}{f.gold_mapped ? " -- gold_mapped" : ""}
              </option>
            ))}
          </select>
          <button className="secondary" onClick={refreshFiles}>Refresh</button>
        </div>
        <div className="row" style={{ gap: 8 }}>
          <input className="mono" style={{ flex: 1 }} value={path} onChange={(e) => setPath(e.target.value)}
                 onKeyDown={(e) => e.key === "Enter" && onOpen()}
                 placeholder="/path/to/preds.jsonl  (tools/infer.py --output)" />
          <button className="primary" onClick={() => onOpen()} disabled={!path.trim()}>Open</button>
        </div>
        {error && <div className="error" style={{ marginTop: 8 }}>{error}</div>}
        {picked && !picked.spans && (
          <div className="error" style={{ marginTop: 8 }}>
            This file&apos;s predictions carry no span offsets (infer.py ran without{" "}
            <span className="mono">--include-spans</span>), so no prediction can be drawn and every gold span
            shows as missed. The F1 numbers and classifications are still right. Re-run infer.py with{" "}
            <span className="mono">--include-spans</span> to see predictions in the text.
          </div>
        )}
        {summary?.gold_key === "gold" && (
          <div className="hint" style={{ marginTop: 8 }}>
            This file has no <span className="mono">gold_mapped</span>: gold is in the corpus spellings, so a label
            spelled differently from the model&apos;s (<span className="mono">location</span> vs{" "}
            <span className="mono">Location</span>) is a miss here, as it is for the strict scorer. Run infer.py with{" "}
            <span className="mono">--labels-file</span> to compare in the model&apos;s spellings.
          </div>
        )}
        {summary && (
          <div className="row" style={{ gap: 8, marginTop: 10, flexWrap: "wrap" }}>
            <button className="secondary" onClick={() => setPos((p) => Math.max(p - 1, 0))} disabled={pos === 0}>Prev</button>
            <input type="number" style={{ width: 80 }} min={1} max={sequence.length} value={pos + 1}
                   onChange={(e) => setPos(Math.min(Math.max(Number(e.target.value) - 1, 0), sequence.length - 1))} />
            <span className="hint">of {summary.count}{order === "file" ? "" : ` (record ${index + 1} in the file)`}</span>
            <button className="secondary" onClick={() => setPos((p) => Math.min(p + 1, sequence.length - 1))}
                    disabled={pos >= sequence.length - 1}>Next</button>
            <select value={order} onChange={(e) => { setOrder(e.target.value); setPos(0); }}>
              <option value="file">file order</option>
              {scoreHeads.map((h) => <option key={h} value={h}>worst {h} F1 first</option>)}
            </select>
            <label className="hint"><input type="checkbox" checked={showGold} onChange={(e) => setShowGold(e.target.checked)} /> show gold</label>
            <label className="hint"><input type="checkbox" checked={diffOnly} onChange={(e) => setDiffOnly(e.target.checked)} /> differences only</label>
          </div>
        )}
      </div>

      {rec && (
        <div className="card annotated">
          <div className="row between">
            <h2>Record {rec.i + 1}</h2>
            <div className="toggle">
              {LAYERS.filter((l) => hasLayer(rec.pred, l) || hasLayer(rec.gold, l)).map((l) => (
                <button key={l} className={l === layer ? "active" : ""} onClick={() => setLayer(l)}>
                  {l === "entities" ? "Entities" : l === "events" ? "Events" : l === "relations" ? "Relations" : "Coref"}
                </button>
              ))}
            </div>
          </div>
          <div className="hint" style={{ marginBottom: 8 }}>
            {rec.has_gold
              ? <>This record: {f1Text(rec.f1)}</>
              : rec.data_row ? "Data file: the record's stored annotations (gold), placed as training places them."
              : "No gold in this record: predictions only."}
          </div>
          {rec.has_gold && (
            <div className="hint" style={{ marginBottom: 8 }}>
              Gold rim: <b>solid</b> = prediction matches gold · <b>dashed</b> = gold and prediction overlap but
              differ (the <b>≠</b> tag says how) · <b>rim only</b> = gold the model missed · fill without rim =
              prediction not in gold. Click an event or relation span to cycle its mentions (pred / gold).
            </div>
          )}
          {("event_cluster" in rec.f1 || "event_cluster_argument" in rec.f1) && (
            <div className="hint" style={{ marginBottom: 8 }}>
              Strict event-argument F1 keys the whole trigger set. Cluster F1 matches a gold event on one shared
              trigger mention, then scores arguments inside that pair. More than one trigger on an event is not,
              by itself, a TriggerLink merge.
            </div>
          )}
          {layer === "coref" && (
            <div className="hint" style={{ marginBottom: 8 }}>
              {rec.entity_coref_decoded === false
                ? "Gold entity-mention clusters are shown. This file has no predicted coref_clusters, so mention-pair F1 is not scored. These are not event arguments."
                : "Coref: one colour per entity cluster (C1, C2, ...; singletons hidden); hover a mention to join its cluster. Gold rim solid = most of the mention's cluster-mates are its mates in gold too; a merge of two entities leaves its smaller side (or both, if even) unmatched. The coref number is mention-pair F1."}
            </div>
          )}
          <AnnotatedText text={rec.text} marks={marks} layer={layer} activeEid={pinned} onSelectEid={setPinned} />
          {rec.unlocated.length > 0 && (
            <div className="hint" style={{ marginTop: 8 }}>
              Gold that cannot be placed in the text (training cannot place it either): {rec.unlocated.join(", ")}
            </div>
          )}
        </div>
      )}

      {rec && rec.has_gold && layer !== "coref" && (
        <div className="card">
          <h2>Differences: {layer}</h2>
          {(() => {
            const rows = rec.diffs.filter((d) => d.layer === layer);
            if (!rows.length) return <div className="hint">None: every {layer} span matches gold.</div>;
            return (
              <table>
                <thead><tr><th>why</th><th>gold</th><th>predicted</th><th>×</th></tr></thead>
                <tbody>
                  {rows.map((d, k) => {
                    const eid = d.gold?.eid ?? d.pred?.eid ?? null;
                    return (
                      <tr key={k} className={eid ? "clickrow" + (eid === pinned ? " active" : "") : undefined}
                          onClick={eid ? () => setPinned(eid === pinned ? null : eid) : undefined}
                          title={d.why}>
                        <td><b>{d.reason}</b></td>
                        <td className="mono">{sideText(d.gold)}</td>
                        <td className="mono">{sideText(d.pred)}</td>
                        <td>{d.count > 1 ? d.count : ""}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            );
          })()}
        </div>
      )}

      {rec && rec.classifications.length > 0 && (
        <div className="card">
          <h2>Classifications</h2>
          <table>
            <thead><tr><th>task</th><th>predicted</th><th>gold</th><th></th></tr></thead>
            <tbody>
              {rec.classifications.map((c) => (
                <tr key={c.task}>
                  <td className="mono">{c.task}</td>
                  <td>{Array.isArray(c.pred) ? c.pred.map((x: any) => x?.label ?? x).join(", ") : String(c.pred ?? "-")}</td>
                  <td>{c.gold.join(", ")}</td>
                  <td>{c.match ? "match" : "differs"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {help && <PredictionsHelp onClose={() => setHelp(false)} />}
    </div>
  );
}
