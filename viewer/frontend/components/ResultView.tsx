"use client";

import { useEffect, useMemo, useState } from "react";
import type { ExtractionResult } from "@/lib/types";
import { fmtConf, labelColor, tagStyle } from "@/lib/colors";
import { collectMarks, eventTriggerLabel, splitOtherKeys, viewLayers, type Layer } from "@/lib/spans";
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

function hasTriggerAndArguments(mention: any): boolean {
  const triggers = Array.isArray(mention?.triggers) ? mention.triggers : [];
  const args = Array.isArray(mention?.arguments) ? mention.arguments : [];
  return triggers.length > 0 && args.length > 0;
}

export default function ResultView({
  text,
  result,
  schema,
}: {
  text: string;
  result: ExtractionResult;
  schema?: Record<string, any> | null;
}) {
  // Tabs follow the REQUESTED schema (so a requested task with no matches still
  // gets a tab -> "No X spans"); fall back to whatever the result contains.
  const layers = useMemo(() => viewLayers(schema, result), [schema, result]);
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

  const [requireBothTriggerAndArgs, setRequireBothTriggerAndArgs] = useState(false);

  const rawEvents = result.event_extraction as Record<string, any[]> | undefined;
  const events = useMemo(() => {
    if (!rawEvents || !requireBothTriggerAndArgs) return rawEvents;
    const filtered: Record<string, any[]> = {};
    for (const [etype, mentions] of Object.entries(rawEvents)) {
      filtered[etype] = (mentions || []).filter(hasTriggerAndArguments);
    }
    return filtered;
  }, [rawEvents, requireBothTriggerAndArgs]);

  const displayResult = useMemo(
    () => (requireBothTriggerAndArgs && events ? { ...result, event_extraction: events } : result),
    [result, events, requireBothTriggerAndArgs],
  );

  const marks = useMemo(() => (layer ? collectMarks(displayResult, layer) : []), [displayResult, layer]);
  const legendLabels = useMemo(
    () => Array.from(new Set(marks.map((m) => m.colorKey ?? m.label))),
    [marks],
  );

  const { classifications, structures } = useMemo(() => splitOtherKeys(result), [result]);

  const entities = result.entities as Record<string, any[]> | undefined;
  const relations = result.relation_extraction as Record<string, any[]> | undefined;

  // Isolate a single event mention: hovering a mention in the list below (or
  // pinning it with a click) highlights only that trigger -> arguments group in
  // the text, even when several mentions of the same type merge onto one span.
  const [hoverEid, setHoverEid] = useState<string | null>(null);
  const [pinnedEid, setPinnedEid] = useState<string | null>(null);
  useEffect(() => {
    setPinnedEid(null);
    setHoverEid(null);
  }, [requireBothTriggerAndArgs]);
  // Per-mention eid, assigned exactly as collectMarks() does in lib/spans.ts (a
  // running counter over Object.entries(events) -> mentions), so list rows and
  // rendered spans reference the same id.
  const eventEids = useMemo(() => {
    const m = new Map<string, string>();
    let c = 0;
    for (const [etype, mentions] of Object.entries(events || {})) {
      (mentions || []).forEach((_: any, mi: number) => m.set(`${etype}:${mi}`, `evt-${c++}`));
    }
    return m;
  }, [events]);
  const eidOf = (etype: string, mi: number) => eventEids.get(`${etype}:${mi}`) ?? null;
  const activeEid = hoverEid ?? pinnedEid;

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
                    {l === "entities" ? "Entities" : l === "events" ? "Events" : l === "coref" ? "Coref" : "Relations"}
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
        {layer && (layer === "events" || layer === "relations" || layer === "coref") && marks.length > 0 && (
          <div className="hint" style={{ marginBottom: 8 }}>
            {layer === "coref"
              ? "Entity mention clusters. Hover a mention to join its cluster. These are not event arguments."
              : <>Hover a span to trace its {layer === "events" ? "event (trigger → arguments)" : "relation (head → tail)"} connections.</>}
          </div>
        )}
        {layer ? (
          <AnnotatedText
            text={text}
            marks={marks}
            layer={layer}
            activeEid={activeEid}
            onSelectEid={(e) => {
              setPinnedEid(e);
              setHoverEid(null);
            }}
          />
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
      {rawEvents && Object.keys(rawEvents).length > 0 && events && (
        <div className="card">
          <div className="row between" style={{ marginBottom: 8 }}>
            <h2 style={{ margin: 0 }}>Events</h2>
            <label className="hint" style={{ display: "inline-flex", alignItems: "center", gap: 6, cursor: "pointer", margin: 0 }}>
              <input
                type="checkbox"
                style={{ width: "auto" }}
                checked={requireBothTriggerAndArgs}
                onChange={(e) => {
                  setRequireBothTriggerAndArgs(e.target.checked);
                  setPinnedEid(null);
                  setHoverEid(null);
                }}
              />
              Only events with both triggers and arguments
            </label>
          </div>
          {(() => {
            const { nonEmpty, empty } = partition(events);
            return (
              <>
                {nonEmpty.map(([etype, mentions]) => (
                  <div className="field" key={etype}>
                    <div style={{ color: labelColor(etype), fontWeight: 650 }}>
                      {etype} <span className="hint">({mentions.length})</span>
                    </div>
                    {mentions.map((m: any, mi: number) => {
                      const eid = eidOf(etype, mi);
                      const pinned = pinnedEid != null && pinnedEid === eid;
                      return (
                      <div
                        className="block"
                        key={mi}
                        style={{
                          ...tagStyle(etype),
                          cursor: "pointer",
                          ...(pinned ? { outline: "2px solid currentColor", outlineOffset: 2 } : {}),
                        }}
                        onMouseEnter={() => setHoverEid(eid)}
                        onMouseLeave={() => setHoverEid(null)}
                        onClick={() => {
                          if (!eid) return;
                          const unpin = pinnedEid === eid;
                          setPinnedEid(unpin ? null : eid);
                          // Clear the hover highlight on unpin too; otherwise the
                          // still-hovered row keeps activeEid set and it looks like
                          // unpin did nothing until the mouse leaves.
                          if (unpin) setHoverEid(null);
                          setLayer("events");
                        }}
                        title="Hover to trace this event in the text above; click to pin/unpin it"
                      >
                        <div className="row" style={{ gap: 6 }}>
                          <span className="role">{eventTriggerLabel((m.triggers || []).length, mentions.length, mi)}</span>
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
                                  <td style={{ width: 140 }}>
                                    <span className="role">{a.role}</span>
                                    {Number.isInteger(a.coref_cluster) ? <span className="role"> c{a.coref_cluster}</span> : null}
                                  </td>
                                  <td><Token label={a.role} value={a.entity} /></td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        )}
                      </div>
                      );
                    })}
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
