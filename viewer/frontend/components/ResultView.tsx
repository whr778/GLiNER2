"use client";

import { useEffect, useMemo, useState } from "react";
import type { ExtractionResult } from "@/lib/types";
import { fmtConf, labelColor, tagStyle } from "@/lib/colors";
import { availableLayers, collectMarks, splitOtherKeys, type Layer } from "@/lib/spans";
import AnnotatedText from "./AnnotatedText";

function spanText(x: any): string {
  if (x == null) return "";
  if (typeof x === "string") return x;
  return x.text ?? "";
}
function spanConf(x: any): number | undefined {
  return x && typeof x === "object" ? x.confidence : undefined;
}

function Token({ label, value }: { label: string; value: any }) {
  const conf = spanConf(value);
  return (
    <span className="chip" style={tagStyle(label)}>
      {spanText(value) || <span className="hint">∅</span>}
      {conf != null && <span className="conf">{fmtConf(conf)}</span>}
    </span>
  );
}

function partition(obj: Record<string, any[]>) {
  const nonEmpty: [string, any[]][] = [];
  const empty: string[] = [];
  for (const [k, v] of Object.entries(obj)) {
    if (Array.isArray(v) && v.length > 0) nonEmpty.push([k, v]);
    else empty.push(k);
  }
  return { nonEmpty, empty };
}

function EmptyNote({ names }: { names: string[] }) {
  if (!names.length) return null;
  return <div className="hint" style={{ marginTop: 8 }}>No matches: {names.join(", ")}</div>;
}

export default function ResultView({ text, result }: { text: string; result: ExtractionResult }) {
  const layers = useMemo(() => availableLayers(result), [result]);
  const [layer, setLayer] = useState<Layer | null>(layers[0] ?? null);
  useEffect(() => {
    setLayer((cur) => (cur && layers.includes(cur) ? cur : layers[0] ?? null));
  }, [layers]);

  const [expanded, setExpanded] = useState(false);
  useEffect(() => {
    if (!expanded) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setExpanded(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [expanded]);

  const marks = useMemo(() => (layer ? collectMarks(result, layer) : []), [result, layer]);
  const legendLabels = useMemo(
    () => Array.from(new Set(marks.map((m) => m.colorKey ?? m.label))),
    [marks],
  );

  const { classifications, structures } = useMemo(() => splitOtherKeys(result), [result]);

  const entities = result.entities as Record<string, any[]> | undefined;
  const events = result.event_extraction as Record<string, any[]> | undefined;
  const relations = result.relation_extraction as Record<string, any[]> | undefined;

  return (
    <div>
      {/* ---- Annotated text ---- */}
      {expanded && <div className="backdrop" onClick={() => setExpanded(false)} />}
      <div className={"card annotated" + (expanded ? " expanded" : "")}>
        <div className="row between">
          <h2>Annotated text</h2>
          <div className="row" style={{ gap: 8 }}>
            {layers.length > 1 && (
              <div className="toggle">
                {layers.map((l) => (
                  <button key={l} className={l === layer ? "active" : ""} onClick={() => setLayer(l)}>
                    {l === "entities" ? "Entities" : l === "events" ? "Events" : "Relations"}
                  </button>
                ))}
              </div>
            )}
            <button onClick={() => setExpanded((e) => !e)} title={expanded ? "Collapse (Esc)" : "Full size"}>
              {expanded ? "⤡ Collapse" : "⤢ Full size"}
            </button>
          </div>
        </div>
        {legendLabels.length > 0 && (
          <div className="legend">
            {legendLabels.map((l) => (
              <span key={l} className="chip" style={tagStyle(l)}>
                <span className="dot" style={tagStyle(l)} /> {l}
              </span>
            ))}
          </div>
        )}
        {layer && (layer === "events" || layer === "relations") && marks.length > 0 && (
          <div className="hint" style={{ marginBottom: 8 }}>
            Hover a span to trace its{" "}
            {layer === "events" ? "event (trigger → arguments)" : "relation (head → tail)"} connections.
          </div>
        )}
        {layer ? (
          <AnnotatedText text={text} marks={marks} layer={layer} />
        ) : (
          <div className="doc">{text || <span className="hint">(no text)</span>}</div>
        )}
        {layer && marks.length === 0 && <div className="empty">No {layer} spans to highlight.</div>}
      </div>

      {/* ---- Entities ---- */}
      {entities && Object.keys(entities).length > 0 && (
        <div className="card">
          <h2>Entities</h2>
          {(() => {
            const { nonEmpty, empty } = partition(entities);
            return (
              <>
                {nonEmpty.map(([type, spans]) => (
                  <div className="field" key={type}>
                    <div style={{ color: labelColor(type), fontWeight: 650, marginBottom: 4 }}>
                      {type} <span className="hint">({spans.length})</span>
                    </div>
                    <div className="chips">
                      {spans.map((s, i) => <Token key={i} label={type} value={s} />)}
                    </div>
                  </div>
                ))}
                {nonEmpty.length === 0 && <div className="empty">No entities found.</div>}
                <EmptyNote names={empty} />
              </>
            );
          })()}
        </div>
      )}

      {/* ---- Events ---- */}
      {events && Object.keys(events).length > 0 && (
        <div className="card">
          <h2>Events</h2>
          {(() => {
            const { nonEmpty, empty } = partition(events);
            return (
              <>
                {nonEmpty.map(([etype, mentions]) => (
                  <div className="field" key={etype}>
                    <div style={{ color: labelColor(etype), fontWeight: 650 }}>
                      {etype} <span className="hint">({mentions.length})</span>
                    </div>
                    {mentions.map((m: any, mi: number) => (
                      <div className="block" key={mi} style={tagStyle(etype)}>
                        <div className="row" style={{ gap: 6 }}>
                          <span className="role">trigger</span>
                          {(m.triggers || []).map((t: any, ti: number) => (
                            <Token key={ti} label={etype} value={t} />
                          ))}
                          {(!m.triggers || m.triggers.length === 0) && <span className="hint">—</span>}
                        </div>
                        {(m.arguments || []).length > 0 && (
                          <table style={{ marginTop: 8 }}>
                            <tbody>
                              {m.arguments.map((a: any, ai: number) => (
                                <tr key={ai}>
                                  <td style={{ width: 140 }}><span className="role">{a.role}</span></td>
                                  <td><Token label={a.role} value={a.entity} /></td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        )}
                      </div>
                    ))}
                  </div>
                ))}
                {nonEmpty.length === 0 && <div className="empty">No events found.</div>}
                <EmptyNote names={empty} />
              </>
            );
          })()}
        </div>
      )}

      {/* ---- Relations ---- */}
      {relations && Object.keys(relations).length > 0 && (
        <div className="card">
          <h2>Relations</h2>
          {(() => {
            const { nonEmpty, empty } = partition(relations);
            return (
              <>
                {nonEmpty.map(([rname, insts]) => (
                  <div className="field" key={rname}>
                    <div style={{ color: labelColor(rname), fontWeight: 650, marginBottom: 4 }}>{rname}</div>
                    <table>
                      <thead>
                        <tr><th>Head</th><th></th><th>Tail</th></tr>
                      </thead>
                      <tbody>
                        {insts.map((inst: any, i: number) => {
                          const head = Array.isArray(inst) ? inst[0] : inst.head;
                          const tail = Array.isArray(inst) ? inst[1] : inst.tail;
                          return (
                            <tr key={i}>
                              <td><Token label={rname} value={head} /></td>
                              <td className="role">→</td>
                              <td><Token label={rname} value={tail} /></td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                ))}
                {nonEmpty.length === 0 && <div className="empty">No relations found.</div>}
                <EmptyNote names={empty} />
              </>
            );
          })()}
        </div>
      )}

      {/* ---- Classifications ---- */}
      {Object.keys(classifications).length > 0 && (
        <div className="card">
          <h2>Classifications</h2>
          {Object.entries(classifications).map(([task, val]) => {
            const items = Array.isArray(val) ? val : [val];
            return (
              <div className="field" key={task}>
                <div style={{ color: labelColor(task), fontWeight: 650, marginBottom: 4 }}>{task}</div>
                <div className="chips">
                  {items.map((it: any, i: number) => (
                    <Token key={i} label={task} value={typeof it === "string" ? it : it.label} />
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* ---- Structures ---- */}
      {Object.keys(structures).length > 0 && (
        <div className="card">
          <h2>Structures</h2>
          {Object.entries(structures).map(([name, instances]) => (
            <div className="field" key={name}>
              <div style={{ color: labelColor(name), fontWeight: 650, marginBottom: 4 }}>
                {name} <span className="hint">({(instances as any[]).length})</span>
              </div>
              {(instances as any[]).map((inst: any, i: number) => (
                <div className="block" key={i} style={tagStyle(name)}>
                  <table>
                    <tbody>
                      {Object.entries(inst).map(([field, val]) => (
                        <tr key={field}>
                          <td style={{ width: 160 }}><span className="role">{field}</span></td>
                          <td>
                            <div className="chips">
                              {(Array.isArray(val) ? val : [val]).map((v: any, vi: number) => (
                                <Token key={vi} label={field} value={v} />
                              ))}
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ))}
            </div>
          ))}
        </div>
      )}

      {/* ---- Raw JSON ---- */}
      <details className="raw">
        <summary className="hint" style={{ cursor: "pointer" }}>Raw result JSON</summary>
        <pre className="mono">{JSON.stringify(result, null, 2)}</pre>
      </details>
    </div>
  );
}
