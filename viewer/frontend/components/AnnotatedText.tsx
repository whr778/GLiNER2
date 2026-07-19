"use client";

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { fmtConf, tagStyle } from "@/lib/colors";
import { buildSegments, type Layer, type Mark } from "@/lib/spans";

type Line = { x1: number; y1: number; x2: number; y2: number };

// Renders the document with span highlights and, for the events/relations
// layers, draws connection arcs from a mention's trigger (or a relation's head)
// to its arguments/tail when any span of that mention is hovered.
export default function AnnotatedText({ text, marks, layer }: { text: string; marks: Mark[]; layer: Layer }) {
  const docRef = useRef<HTMLDivElement>(null);
  const [hoverEids, setHoverEids] = useState<Set<string>>(() => new Set());
  const [lines, setLines] = useState<Line[]>([]);
  const [size, setSize] = useState({ w: 0, h: 0 });

  const segments = useMemo(() => buildSegments(text, marks), [text, marks]);
  const linkable = layer === "events" || layer === "relations";
  const hovering = hoverEids.size > 0;

  const compute = useCallback(() => {
    const cont = docRef.current;
    if (!cont || !hovering || !linkable) {
      setLines([]);
      return;
    }
    const crect = cont.getBoundingClientRect();
    const center = (el: HTMLElement) => {
      const r = el.getBoundingClientRect();
      return {
        x: r.left - crect.left + cont.scrollLeft + r.width / 2,
        y: r.top - crect.top + cont.scrollTop + r.height / 2,
      };
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
    for (const eid of hoverEids) {
      const els = byEid.get(eid);
      if (!els || els.length < 2) continue; // needs both endpoints (skip self-loops)
      const anchor = els.find((e) => e.kind === "head" || e.kind === "trigger") ?? els[0];
      const a = center(anchor.el);
      for (const other of els) {
        if (other.el === anchor.el) continue;
        const p = center(other.el);
        out.push({ x1: a.x, y1: a.y, x2: p.x, y2: p.y });
      }
    }
    setLines(out);
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
      {linkable && lines.length > 0 && (
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
        </svg>
      )}

      {segments.map((s, i) => {
        const mk = s.mark;
        if (!mk) return <span key={i}>{s.text}</span>;
        const p = mk.primary;
        const colorKey = p.colorKey ?? p.label;
        const eidTokens = mk.roles.filter((r) => r.eid).map((r) => `${r.eid}:${r.kind}`).join(" ");
        const eidSet = new Set(mk.roles.map((r) => r.eid).filter(Boolean) as string[]);
        const active = [...eidSet].some((e) => hoverEids.has(e));
        const count = eidSet.size;
        return (
          <mark
            key={i}
            className={"tag" + (active ? " active" : "")}
            data-eids={eidTokens || undefined}
            data-kind={p.kind}
            style={tagStyle(colorKey)}
            onMouseEnter={() => setHoverEids(eidSet)}
            onMouseLeave={() => setHoverEids(new Set())}
            title={`${p.label} · ${p.kind}${count > 1 ? ` · ${count} links` : ""}${p.confidence != null ? ` · ${fmtConf(p.confidence)}` : ""}`}
          >
            {s.text}
            <span className="taglabel" style={tagStyle(colorKey)}>
              {p.label}
              {p.kind === "trigger" && p.idx ? <sup className="eidx">{p.idx}</sup> : null}
              {count > 1 ? <sup className="eidx">×{count}</sup> : null}
            </span>
          </mark>
        );
      })}
      {segments.length === 0 && <span className="hint">(no text)</span>}
    </div>
  );
}
