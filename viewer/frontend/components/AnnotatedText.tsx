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
  const [hoverEid, setHoverEid] = useState<string | null>(null);
  const [lines, setLines] = useState<Line[]>([]);
  const [size, setSize] = useState({ w: 0, h: 0 });

  const segments = useMemo(() => buildSegments(text, marks), [text, marks]);
  const linkable = layer === "events" || layer === "relations";

  const compute = useCallback(() => {
    const cont = docRef.current;
    if (!cont || !hoverEid || !linkable) {
      setLines([]);
      return;
    }
    const crect = cont.getBoundingClientRect();
    const els = Array.from(cont.querySelectorAll(`[data-eid="${hoverEid}"]`)) as HTMLElement[];
    if (els.length < 2) {
      setLines([]);
      return;
    }
    const center = (el: HTMLElement) => {
      const r = el.getBoundingClientRect();
      return {
        x: r.left - crect.left + cont.scrollLeft + r.width / 2,
        y: r.top - crect.top + cont.scrollTop + r.height / 2,
      };
    };
    const anchorEl = els.find((e) => e.dataset.kind === "trigger" || e.dataset.kind === "head") ?? els[0];
    const a = center(anchorEl);
    const out: Line[] = [];
    for (const el of els) {
      if (el === anchorEl) continue;
      const p = center(el);
      out.push({ x1: a.x, y1: a.y, x2: p.x, y2: p.y });
    }
    setLines(out);
    setSize({ w: cont.scrollWidth, h: cont.scrollHeight });
  }, [hoverEid, linkable]);

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
    <div className="doc" ref={docRef} data-dim={hoverEid ? "1" : undefined}>
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

      {segments.map((s, i) =>
        s.mark ? (
          <mark
            key={i}
            className={"tag" + (hoverEid && s.mark.eid === hoverEid ? " active" : "")}
            data-eid={s.mark.eid}
            data-kind={s.mark.kind}
            style={tagStyle(s.mark.colorKey ?? s.mark.label)}
            onMouseEnter={() => s.mark!.eid && setHoverEid(s.mark!.eid!)}
            onMouseLeave={() => setHoverEid(null)}
            title={`${s.mark.label} · ${s.mark.kind}${s.mark.confidence != null ? ` · ${fmtConf(s.mark.confidence)}` : ""}`}
          >
            {s.text}
            <span className="taglabel" style={tagStyle(s.mark.colorKey ?? s.mark.label)}>
              {s.mark.label}
              {s.mark.kind === "trigger" && s.mark.idx ? <sup className="eidx">{s.mark.idx}</sup> : null}
            </span>
          </mark>
        ) : (
          <span key={i}>{s.text}</span>
        ),
      )}
      {segments.length === 0 && <span className="hint">(no text)</span>}
    </div>
  );
}
