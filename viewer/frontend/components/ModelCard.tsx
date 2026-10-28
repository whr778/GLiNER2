"use client";

import { Fragment, useEffect, useRef, useState } from "react";
import { getModelCard, type CardPair, type MetricRow, type MetricSection, type ModelCardView } from "@/lib/api";

type Props = {
  model: string;
  onClose: () => void;
};

const SECTION_LABEL: Record<string, string> = {
  "val_metrics.json": "Validation",
  "test_metrics.json": "Test",
  "eval_metrics.json": "Eval",
  "eval_history.jsonl": "Eval history",
};

const SCORE_COLUMNS: { key: keyof MetricRow; label: string }[] = [
  { key: "precision", label: "Strict micro precision" },
  { key: "recall", label: "Strict micro recall" },
  { key: "f1", label: "Strict micro F1" },
  { key: "support", label: "Support" },
  { key: "relaxed_f1", label: "Relaxed micro F1" },
];

function categoryLabel(category: string): string {
  return category.replace(/_/g, " ");
}

function modelTitle(title: string) {
  const at = title.lastIndexOf("@");
  const local = title.startsWith("/") || title.startsWith("~");
  if (!local && at > 0) {
    return (
      <>
        <div className="model-card-name">{title.slice(0, at)}</div>
        <div className="hint mono">{title.slice(at + 1)}</div>
      </>
    );
  }
  return <div className="model-card-name mono">{title}</div>;
}

function DefList({ rows }: { rows: CardPair[] }) {
  return (
    <dl className="kv">
      {rows.map((row) => (
        <Fragment key={row.key}>
          <dt>{row.key}</dt>
          <dd>{row.value}</dd>
        </Fragment>
      ))}
    </dl>
  );
}

function columnsFor(rows: MetricRow[]) {
  return SCORE_COLUMNS.filter((col) => rows.some((row) => row[col.key] != null && row[col.key] !== ""));
}

function MetricTable({ rows }: { rows: MetricRow[] }) {
  const columns = columnsFor(rows);
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Category</th>
            {columns.map((col) => (
              <th key={col.key} className="num">{col.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.category}>
              <td className="cat">{categoryLabel(row.category)}</td>
              {columns.map((col) => (
                <td key={col.key} className="num">{row[col.key] ?? ""}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Section({ section }: { section: MetricSection }) {
  const label = SECTION_LABEL[section.name] ?? section.name;
  return (
    <div className="block">
      <div className="btitle">{label}</div>
      {label !== section.name && <div className="hint mono">{section.name}</div>}
      <div className="hint model-card-kicker">Provenance</div>
      {section.provenance == null && <div className="empty">none</div>}
      {section.provenance != null && section.provenance.length > 0 && (
        <DefList rows={section.provenance} />
      )}
      {section.fields.length > 0 && <DefList rows={section.fields} />}
      {section.rows.length > 0 && <MetricTable rows={section.rows} />}
      {section.full_menu.length > 0 && (
        <>
          <div className="btitle subhead">Full menu</div>
          <MetricTable rows={section.full_menu} />
        </>
      )}
      {section.notes.length > 0 && <DefList rows={section.notes} />}
      {section.per_head && (
        <>
          <div className="btitle subhead">per_head</div>
          {section.per_head.identical_to_global && (
            <div className="hint">identical to the global threshold</div>
          )}
          {section.per_head.thresholds != null && (
            <DefList rows={[{ key: "thresholds", value: section.per_head.thresholds }]} />
          )}
          {section.per_head.rows.length > 0 && <MetricTable rows={section.per_head.rows} />}
        </>
      )}
      {section.coref_merge_off && (
        <>
          <div className="btitle subhead">coref_merge_off</div>
          <MetricTable rows={section.coref_merge_off} />
        </>
      )}
      {section.by_menu.map((menu) => (
        <div key={menu.menu}>
          <div className="btitle subhead">by_menu {menu.menu}</div>
          <MetricTable rows={menu.rows} />
        </div>
      ))}
      {section.by_language && (
        <div className="hint model-card-kicker">
          by_language: {section.by_language} (not expanded)
        </div>
      )}
    </div>
  );
}

export default function ModelCard({ model, onClose }: Props) {
  const [view, setView] = useState<ModelCardView | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const closeRef = useRef(onClose);
  closeRef.current = onClose;

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") closeRef.current();
    };
    window.addEventListener("keydown", onKey);
    let stop = false;
    setView(null);
    setErr(null);
    getModelCard(model)
      .then((card) => { if (!stop) setView(card); })
      .catch((e) => { if (!stop) setErr(e.message || "could not load the model card"); });
    return () => {
      stop = true;
      window.removeEventListener("keydown", onKey);
    };
  }, [model]);

  return (
    <>
      <div className="backdrop" onClick={() => closeRef.current()} />
      <div className="modal card wide model-card">
        <div className="row between model-card-head">
          <div className="model-card-id">
            <h2>Model card</h2>
            {view && modelTitle(view.title)}
          </div>
          <button type="button" onClick={() => closeRef.current()}>Close</button>
        </div>
        {err && <div className="error">{err}</div>}
        {view == null && !err && <div className="hint">Loading metrics…</div>}
        {view && (
          <>
            <div className="block">
              <div className="btitle">Inference defaults</div>
              {view.inference_defaults
                ? <DefList rows={view.inference_defaults} />
                : <div className="empty">none</div>}
            </div>
            {view.train_results && (
              <div className="block">
                <div className="btitle">Train results</div>
                <div className="hint mono">train_results.json</div>
                <DefList rows={view.train_results} />
              </div>
            )}
            {view.sections.map((section) => (
              <Section key={section.name} section={section} />
            ))}
            {view.empty && (
              <div className="block">
                <div className="btitle">Metrics</div>
                <div className="empty">
                  No metrics file (val_metrics.json, test_metrics.json, eval_metrics.json).
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </>
  );
}
