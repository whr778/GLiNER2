"use client";

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { fmtConf, tagStyle } from "@/lib/colors";
import { buildSegments, type Layer, type Mark } from "@/lib/spans";

type Line = { x1: number; y1: number; x2: number; y2: number };
// A nested-endpoint marker: a small circle, plus (when the endpoint is a proper
// sub-range) a thin vertical divider at its boundary.
type Loop = { x: number; top: number; bottom: number; divider: boolean };

// Renders the document with span highlights and, for the events/relations
// layers, draws connection arcs from a mention's trigger (or a relation's head)
// to its arguments/tail when any span of that mention is hovered.
export default function AnnotatedText({
  text,
  marks,
  layer,
  activeEid = null,
  onSelectEid,
}: {
  text: string;
  marks: Mark[];
  layer: Layer;
  // An event/relation mention hovered or pinned in the list below. Folded into
  // the highlight set so a single mention can be isolated out of an overlapping
  // (merged ×N) span.
  activeEid?: string | null;
  // Select/cycle which mention is isolated by clicking a span in the text; called
  // with the next eid, or null to clear. Lets you pick one of several events that
  // overlap on the same span, right from the document.
  onSelectEid?: (eid: string | null) => void;
}) {
  const docRef = useRef<HTMLDivElement>(null);
  const [spanHover, setSpanHover] = useState<Set<string>>(() => new Set());
  const hoverEids = useMemo(() => {
    if (!activeEid) return spanHover;
    const s = new Set(spanHover);
    s.add(activeEid);
    return s;
  }, [spanHover, activeEid]);
  const [lines, setLines] = useState<Line[]>([]);
  const [loops, setLoops] = useState<Loop[]>([]);
  const [size, setSize] = useState({ w: 0, h: 0 });

  const segments = useMemo(() => buildSegments(text, marks), [text, marks]);
  const linkable = layer === "events" || layer === "relations";
  const hovering = hoverEids.size > 0;

  const compute = useCallback(() => {
    const cont = docRef.current;
    if (!cont || !hovering || !linkable) {
      setLines([]);
      setLoops([]);
      return;
    }
    const crect = cont.getBoundingClientRect();
    // Anchor to the span's FIRST line fragment, not its full bounding box: a
    // wrapped span's bounding box spans both lines and its center floats off the
    // visible text, leaving arcs disconnected.
    const rectOf = (el: HTMLElement) => el.getClientRects()[0] ?? el.getBoundingClientRect();
    const center = (el: HTMLElement) => {
      const r = rectOf(el);
      return {
        x: r.left - crect.left + cont.scrollLeft + r.width / 2,
        y: r.top - crect.top + cont.scrollTop + r.height / 2,
      };
    };
    const boxOf = (el: HTMLElement) => {
      const r = rectOf(el);
      return {
        x: r.left - crect.left + cont.scrollLeft,
        top: r.top - crect.top + cont.scrollTop,
        bottom: r.bottom - crect.top + cont.scrollTop,
        w: r.width,
      };
    };
    // A relation whose two endpoints land on the same span (a tail nested in its
    // head): mark the nested sub-range's boundary with a divider, else a circle
    // centered on the span.
    const selfLoop = (el: HTMLElement, eid: string): Loop => {
      const tn = el.firstChild;
      const nestedRaw = el.dataset.nested;
      if (tn && tn.nodeType === 3 && nestedRaw) {
        const len = tn.textContent?.length ?? 0;
        for (const tok of nestedRaw.split(" ")) {
          const [range, eids] = tok.split(":");
          if (!eids || !eids.split(",").includes(eid)) continue;
          const [s, e] = range.split("-").map(Number);
          const rg = document.createRange();
          rg.setStart(tn, Math.min(s, len));
          rg.setEnd(tn, Math.min(e, len));
          const rr = rg.getBoundingClientRect();
          return {
            x: rr.right - crect.left + cont.scrollLeft,
            top: rr.top - crect.top + cont.scrollTop,
            bottom: rr.bottom - crect.top + cont.scrollTop,
            divider: true,
          };
        }
      }
      const b = boxOf(el);
      return { x: b.x + b.w / 2, top: b.top, bottom: b.bottom, divider: false };
    };
    // Map each instance id -> the spans (and their head/tail role) that carry it.
    // A span's data-eids is "rel-0:head rel-3:tail" (one token per role).
    const byEid = new Map<string, { el: HTMLElement; kind: string }[]>();
    for (const el of Array.from(cont.querySelectorAll("mark.tag")) as HTMLElement[]) {
      const raw = el.dataset.eids;
      if (!raw) continue;
      for (const tok of raw.split(" ")) {
        const [eid, kind] = tok.split(":");
        if (!eid) continue;
        let arr = byEid.get(eid);
        if (!arr) byEid.set(eid, (arr = []));
        arr.push({ el, kind });
      }
    }
    const out: Line[] = [];
    const loopsOut: Loop[] = [];
    for (const eid of hoverEids) {
      const els = byEid.get(eid);
      if (!els || els.length < 2) continue; // a relation has both endpoints
      const anchor = els.find((e) => e.kind === "head" || e.kind === "trigger") ?? els[0];
      const targets = els.filter((e) => e.el !== anchor.el);
      if (targets.length === 0) {
        loopsOut.push(selfLoop(anchor.el, eid));
        continue;
      }
      const a = center(anchor.el);
      for (const other of targets) {
        const p = center(other.el);
        out.push({ x1: a.x, y1: a.y, x2: p.x, y2: p.y });
      }
    }
    setLines(out);
    setLoops(loopsOut);
    setSize({ w: cont.scrollWidth, h: cont.scrollHeight });
  }, [hoverEids, hovering, linkable]);

  useLayoutEffect(() => {
    compute();
  }, [compute]);

  useEffect(() => {
    const onResize = () => compute();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [compute]);

  // Recompute arcs when the container itself resizes (e.g. full-size toggle).
  useEffect(() => {
    const cont = docRef.current;
    if (!cont || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => compute());
    ro.observe(cont);
    return () => ro.disconnect();
  }, [compute]);

  return (
    <div className="doc" ref={docRef} data-dim={hovering ? "1" : undefined}>
      {linkable && (lines.length > 0 || loops.length > 0) && (
        <svg className="arcs" width={size.w} height={size.h} aria-hidden="true">
          <defs>
            <marker id="arc-arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
              <path d="M0,0 L6,3 L0,6 Z" fill="var(--accent)" />
            </marker>
          </defs>
          {lines.map((l, i) => {
            const midX = (l.x1 + l.x2) / 2;
            const midY = Math.min(l.y1, l.y2) - 16;
            return (
              <path
                key={i}
                d={`M ${l.x1} ${l.y1} Q ${midX} ${midY} ${l.x2} ${l.y2}`}
                fill="none"
                stroke="var(--accent)"
                strokeWidth={1.6}
                opacity={0.8}
                markerEnd="url(#arc-arrow)"
              />
            );
          })}
          {loops.map((lp, i) => (
            <g key={`loop-${i}`} stroke="var(--accent)" opacity={0.9}>
              {lp.divider && <line x1={lp.x} y1={lp.top} x2={lp.x} y2={lp.bottom} strokeWidth={1.5} />}
              <circle cx={lp.x} cy={lp.top - 5} r={4} fill="none" strokeWidth={1.5} />
            </g>
          ))}
        </svg>
      )}

      {segments.map((s, i) => {
        const mk = s.mark;
        if (!mk) return <span key={i}>{s.text}</span>;
        const p = mk.primary;
        const colorKey = p.colorKey ?? p.label;
        const eidTokens = mk.roles.filter((r) => r.eid).map((r) => `${r.eid}:${r.kind}`).join(" ");
        // Distinct eids on this span, in role order, so clicks cycle deterministically.
        const eidList: string[] = [];
        for (const r of mk.roles) if (r.eid && !eidList.includes(r.eid)) eidList.push(r.eid);
        const eidSet = new Set(eidList);
        const active = [...eidSet].some((e) => hoverEids.has(e));
        const count = eidSet.size;
        const nestedAttr = mk.nested?.map((n) => `${n.s}-${n.e}:${n.eids.join(",")}`).join(" ");
        // 0-based position of the currently-selected mention within this span (-1
        // if the selection is elsewhere / nothing selected).
        const selPos = activeEid ? eidList.indexOf(activeEid) : -1;
        const clickable = linkable && eidList.length > 0 && !!onSelectEid;
        // Cycle: nothing here selected -> first; selected -> next; past the last -> clear.
        const cycle = () => {
          if (!clickable) return;
          const next = selPos < 0 ? eidList[0] : selPos + 1 < eidList.length ? eidList[selPos + 1] : null;
          onSelectEid!(next);
        };
        return (
          <mark
            key={i}
            className={"tag" + (active ? " active" : "")}
            data-eids={eidTokens || undefined}
            data-nested={nestedAttr || undefined}
            data-kind={p.kind}
            style={clickable ? { ...tagStyle(colorKey), cursor: "pointer" } : tagStyle(colorKey)}
            onMouseEnter={() => setSpanHover(eidSet)}
            onMouseLeave={() => setSpanHover(new Set())}
            onClick={clickable ? cycle : undefined}
            title={
              `${p.label} · ${p.kind}` +
              (count > 1 ? ` · ${count} events here — click to view one at a time` : "") +
              (p.confidence != null ? ` · ${fmtConf(p.confidence)}` : "")
            }
          >
            {s.text}
            <span className="taglabel" style={tagStyle(colorKey)}>
              {p.label}
              {p.kind === "trigger" && p.idx ? <sup className="eidx">{p.idx}</sup> : null}
              {count > 1 ? (
                <sup className="eidx">{selPos >= 0 ? `${selPos + 1}/${count}` : `×${count}`}</sup>
              ) : null}
            </span>
          </mark>
        );
      })}
      {segments.length === 0 && <span className="hint">(no text)</span>}
    </div>
  );
}
