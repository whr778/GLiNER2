"use client";

/**
 * EKF tracking: a news feed in, a tracked casualty timeline out.
 *
 * Runs the whole research line -- classification gate -> DocEE event typing -> casualty
 * structure extraction -> normalization -> EKF -- over a time-ordered JSONL feed.
 *
 * Two things make this a showcase rather than a chart viewer:
 *
 *   COMPARE   Every finding in this line is a comparison (whole-article vs event
 *             envelope, heuristic vs hybrid normalizer, single- vs multi-event casualty
 *             model). Finished runs are kept and can be overlaid on one axis, so a
 *             configuration change is visible instead of remembered.
 *   DRILL     A tracked value is only trustworthy if you can see where it came from.
 *             Observation markers sit on the timeline; clicking one opens the article
 *             behind it -- gate decision, event type, the span the number was bound
 *             from, and the qualifier/source that set its measurement noise.
 *
 * The chart is hand-drawn SVG: several series over a shared grid is ~80 lines, versus a
 * new npm dependency in the Next app.
 */

import { useEffect, useRef, useState } from "react";
import { getEkfFeeds, getEkfJob, getModels, startEkfTrack, type EkfFeed, type EkfJob } from "@/lib/api";
import type { ModelEntry } from "@/lib/types";

const ROLES = ["dead", "injured", "missing"] as const;
type Role = (typeof ROLES)[number];

// Run colours, in selection order. Truth is always green and dashed.
const RUN_COLORS = ["#2563eb", "#db2777", "#f59e0b", "#7c3aed"];
const TRUTH = "#16a34a";
const BASELINE = "#9ca3af";

type Run = { id: string; label: string; result: any };
type Marker = { t: number; value: number; article: number; obs: any };

function markersFor(result: any, role: Role): Marker[] {
  const out: Marker[] = [];
  (result.articles ?? []).forEach((a: any, i: number) => {
    (a.observations ?? []).forEach((o: any) => {
      // `dropped` observations are kept on the article for the per-document trace but
      // never reached the tracker, so they must not be plotted as if they had.
      if (o.role === role && o.mode === result.mode && !o.dropped) {
        out.push({ t: o.t_hours, value: o.value, article: i, obs: o });
      }
    });
  });
  return out;
}

function Chart({ role, runs, showBaseline, showTruth, onPick }: {
  role: Role;
  runs: Run[];
  showBaseline: boolean;
  showTruth: boolean;
  onPick: (runId: string, m: Marker) => void;
}) {
  const W = 620, H = 190, PAD = { l: 48, r: 12, t: 12, b: 28 };
  const base = runs[0]?.result;
  if (!base) return null;
  const grid: number[] = base.grid;
  const truth: (number | null)[] | undefined = showTruth ? base.truth?.[role] : undefined;

  const values: number[] = [];
  runs.forEach((r) => {
    const s = r.result.tracked[r.result.mode][role];
    values.push(...s.ekf, ...(showBaseline ? s.last_value : []));
    markersFor(r.result, role).forEach((m) => values.push(m.value));
  });
  (truth ?? []).forEach((v) => v != null && values.push(v));
  const yMax = Math.max(1, ...values) * 1.12;

  const x0 = grid[0] ?? 0, x1 = grid[grid.length - 1] ?? 1;
  const px = (t: number) => PAD.l + ((t - x0) / Math.max(x1 - x0, 1e-9)) * (W - PAD.l - PAD.r);
  const py = (v: number) => H - PAD.b - (v / yMax) * (H - PAD.t - PAD.b);
  const path = (vals: (number | null)[], xs: number[]) =>
    vals.map((v, i) => (v == null ? null : `${i && vals[i - 1] != null ? "L" : "M"}${px(xs[i])},${py(v)}`))
        .filter(Boolean).join(" ");

  // Deduplicated. yMax has a floor of 1.12 (Math.max(1, ...) * 1.12), so a role whose
  // largest value is <= 1 -- routinely `missing` -- gives [0, round(0.56), round(1.12)]
  // = [0, 1, 1]: two <g> children keyed 1, which React warns about and which also draws
  // two gridlines and two labels on top of each other.
  const ticks = [...new Set([0, 0.5, 1].map((f) => Math.round(yMax * f)))];

  return (
    <div style={{ marginBottom: 16 }}>
      <div className="hint" style={{ marginBottom: 2 }}><strong>{role}</strong></div>
      <svg width={W} height={H} role="img" aria-label={`${role} over time`}>
        {ticks.map((v) => (
          <g key={v}>
            <line x1={PAD.l} y1={py(v)} x2={W - PAD.r} y2={py(v)} stroke="#eef1f4" />
            <text x={4} y={py(v) + 3} fontSize="10" fill="#6b7280">{v}</text>
          </g>
        ))}
        <line x1={PAD.l} y1={H - PAD.b} x2={W - PAD.r} y2={H - PAD.b} stroke="#d1d5db" />
        <text x={PAD.l} y={H - 8} fontSize="10" fill="#6b7280">{x0.toFixed(0)}h</text>
        <text x={W - PAD.r - 32} y={H - 8} fontSize="10" fill="#6b7280">{x1.toFixed(0)}h</text>

        {truth && <path d={path(truth, grid)} fill="none" stroke={TRUTH} strokeWidth={2} strokeDasharray="4 3" />}

        {runs.map((r, ri) => {
          const s = r.result.tracked[r.result.mode][role];
          const color = RUN_COLORS[ri % RUN_COLORS.length];
          return (
            <g key={r.id}>
              {showBaseline && (
                <path d={path(s.last_value, r.result.grid)} fill="none" stroke={BASELINE}
                      strokeWidth={1.2} strokeDasharray="2 2" />
              )}
              <path d={path(s.ekf, r.result.grid)} fill="none" stroke={color} strokeWidth={2} />
              {markersFor(r.result, role).map((m, i) => (
                <circle key={i} cx={px(m.t)} cy={py(m.value)} r={3.5}
                        fill={color} fillOpacity={0.55} stroke={color}
                        style={{ cursor: "pointer" }}
                        onClick={() => onPick(r.id, m)}>
                  <title>{`t=${m.t}h  ${role}=${m.value}  "${m.obs.span}"  (${m.obs.qualifier}, ${m.obs.source})`}</title>
                </circle>
              ))}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

// Where each document STOPPED. The chart only ever shows articles that produced a
// plotted observation, so everything rejected upstream -- the majority on a real feed --
// was invisible: a document dropped by the language gate, one the relevance gate scored
// below threshold, and one that passed both but yielded no figure all looked the same,
// which is to say they looked like nothing at all.
type Stage = { name: string; ok: boolean; detail: string };

function stagesFor(a: any, mode: string): Stage[] {
  const unsupported = a.relevance === "unsupported_language";
  const obs = (a.observations ?? []).filter((o: any) => o.mode === mode);
  const kept = obs.filter((o: any) => !o.dropped);
  const stages: Stage[] = [
    {
      name: "language",
      ok: !unsupported,
      detail: unsupported
        ? `${a.language ?? "?"} not supported`
        : `${a.language ?? "?"}${a.language_confidence != null ? ` ${a.language_confidence.toFixed(2)}` : ""}`,
    },
    {
      name: "gate",
      ok: !!a.relevant,
      detail: unsupported
        ? "not reached"
        : `${a.relevance}${a.relevance_confidence != null ? ` ${a.relevance_confidence.toFixed(2)}` : ""}`,
    },
    {
      name: "event",
      ok: !!a.events?.event_type,
      detail: a.events?.event_type ?? (a.relevant ? "untyped" : "not reached"),
    },
    {
      name: "extract",
      ok: obs.length > 0,
      detail: a.relevant ? `${obs.length} observation(s)` : "not reached",
    },
    {
      name: "tracked",
      ok: kept.length > 0,
      detail:
        obs.length === 0
          ? "nothing to track"
          : `${kept.length} kept${obs.length - kept.length ? `, ${obs.length - kept.length} dropped` : ""}`,
    },
  ];
  return stages;
}

function DocumentFlow({ run, onPick }: { run: Run; onPick: (i: number) => void }) {
  const [onlyStopped, setOnlyStopped] = useState(false);
  const articles = run.result.articles ?? [];
  const rows = articles.map((a: any, i: number) => ({ a, i, stages: stagesFor(a, run.result.mode) }));
  const shown = onlyStopped ? rows.filter((r: any) => r.stages.some((s: Stage) => !s.ok)) : rows;
  const reached = (n: string) => rows.filter((r: any) => r.stages.find((s: Stage) => s.name === n)?.ok).length;

  return (
    <div className="card" style={{ marginTop: 12 }}>
      <div className="row between">
        <h2 style={{ margin: 0 }}>Document flow · {run.label}</h2>
        <label className="hint" style={{ display: "flex", gap: 6, alignItems: "center" }}>
          <input type="checkbox" checked={onlyStopped} onChange={(e) => setOnlyStopped(e.target.checked)} />
          only documents that stopped
        </label>
      </div>

      <div className="hint" style={{ margin: "8px 0" }}>
        {articles.length} documents · language {reached("language")} · gate {reached("gate")} ·
        typed {reached("event")} · extracted {reached("extract")} · tracked {reached("tracked")}
      </div>

      <div style={{ maxHeight: 320, overflow: "auto" }}>
        <table className="flow">
          <thead>
            <tr>
              <th align="right">t</th>
              <th align="left">language</th><th align="left">gate</th>
              <th align="left">event</th><th align="left">extract</th><th align="left">tracked</th>
            </tr>
          </thead>
          <tbody>
            {shown.map(({ a, i, stages }: any) => (
              <tr key={i} onClick={() => onPick(i)} style={{ cursor: "pointer" }}
                  title={String(a.text ?? "").slice(0, 180)}>
                <td align="right" className="mono">{a.t_hours}h</td>
                {stages.map((s: Stage) => (
                  <td key={s.name} className={s.ok ? "flow-ok" : "flow-stop"}>{s.detail}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="hint" style={{ marginTop: 6 }}>
        Click a row to open the document. Green cells passed that stage; grey ones did not.
      </div>
    </div>
  );
}

function ArticleDetail({ run, index, onClose }: { run: Run; index: number; onClose: () => void }) {
  const a = run.result.articles?.[index];
  if (!a) return null;
  const obs = (a.observations ?? []).filter((o: any) => o.mode === run.result.mode);
  return (
    <div className="card" style={{ marginTop: 12 }}>
      <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
        <strong>t = {a.t_hours}h</strong>
        <span className="hint">{run.label}</span>
        <button className="secondary" style={{ marginLeft: "auto" }} onClick={onClose}>close</button>
      </div>

      <div className="hint" style={{ margin: "8px 0" }}>
        gate <span className="mono">{a.relevance}</span>
        {a.relevance_confidence != null && ` (${a.relevance_confidence.toFixed(2)})`}
        {a.events?.event_type && <> · event <span className="mono">{a.events.event_type}</span></>}
        {a.envelopes?.length > 0 && <> · {a.envelopes.length} casualty window(s)</>}
      </div>

      <div style={{ fontSize: 13, lineHeight: 1.5, background: "#f8fafc",
                    padding: 10, borderRadius: 6, maxHeight: 190, overflow: "auto" }}>
        {a.text}
      </div>

      {obs.length > 0 ? (
        <table style={{ marginTop: 10, width: "100%", fontSize: 12 }}>
          <thead>
            <tr><th align="left">role</th><th align="right">value</th>
                <th align="left">bound span</th><th align="left">qualifier</th>
                <th align="left">source</th><th align="right">conf</th></tr>
          </thead>
          <tbody>
            {obs.map((o: any, i: number) => (
              // A dropped observation is shown, struck through, with the reason. It was
              // extracted and then rejected before tracking; hiding it makes an
              // over-ceiling figure indistinguishable from never having been found.
              <tr key={i} className={o.dropped ? "flow-stop" : undefined}
                  title={o.dropped || undefined}>
                <td>{o.role}</td>
                <td align="right">
                  <strong style={o.dropped ? { textDecoration: "line-through" } : undefined}>
                    {o.value}
                  </strong>
                </td>
                <td className="mono">{o.span}</td>
                <td>{o.dropped ? o.dropped : o.qualifier}</td>
                <td>{o.source}</td>
                <td align="right">{(o.confidence ?? 1).toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <div className="hint" style={{ marginTop: 8 }}>No observations from this article.</div>
      )}
    </div>
  );
}


// What each choice actually does, with the numbers that justify it. These are the
// measured claims from tools/ekf_showcase/run_pipeline.py, not restatements of the
// option label -- a hint that only expands the name teaches nothing.
const WINDOW_HINTS: Record<string, string> = {
  long: "Chunks the whole document with overlap and merges. The research default: on Helene it takes `dead` observations 25 → 106.",
  article: "One pass over the whole article. The CLI default, and what every figure before the long path was measured on.",
  event: "Only the event's own 'Casualties and Losses' envelope, so per-reading attributes are judged on per-event text. Needs a stage-1 event model.",
  lead: "Only the article head. Fastest, and blind to tolls that appear further down.",
};
const ASSOCIATE_HINTS: Record<string, string> = {
  record: "The record's OWN location field — the strongest signal, because the same decode step produced the place and the number together.",
  "type+location": "Event type plus the first place name specific enough to identify a place. Generic spans ('the region') are skipped: splitting one event into several streams is worse than pooling it.",
  envelope: "Event type plus the nearest location by character offset. A proximity heuristic — on the Türkiye standfirst both countries sit within 26 characters of both numbers, so it is close to a coin flip.",
  type: "Event type alone. Two earthquakes in one feed collapse into one stream.",
  none: "Pool every observation into one stream. Violates the tracker's single-event assumption: on the multi-event feed this drove normalized RMSE to 102 against 0.313 on clean observations.",
};
const NORMALIZER_HINTS: Record<string, string> = {
  hybrid: "Qualifier from the keyword window, source from the classifier — each taken from whichever measured better (qualifier 0.654 vs 0.395; source 0.605 vs 0.494).",
  heuristic: "Keyword windows for both qualifier and source. No second model pass, so it is the fastest.",
  classify: "Zero-shot classification for both. Better on source (semantic), worse on qualifier (a hedge is a literal lexical cue).",
};

// The real event feeds are all named `feed.jsonl` and differ only by directory
// (datasets/helene2024/_cache/feed.jsonl), so the basename alone renders three
// indistinguishable "feed.jsonl" entries. Name them by their event directory instead.
function feedLabel(path: string): string {
  const parts = path.split("/");
  const base = parts[parts.length - 1].replace(/\.jsonl$/, "");
  if (base !== "feed") return base;
  const event = parts[parts.length - 2] === "_cache" ? parts[parts.length - 3] : parts[parts.length - 2];
  return event || base;
}

export default function EkfPanel() {
  const [feeds, setFeeds] = useState<EkfFeed[]>([]);
  const [feed, setFeed] = useState("");
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [applied, setApplied] = useState<string | null>(null);
  const [rollup, setRollup] = useState<string | null>(null);
  const [eventYear, setEventYear] = useState(0);
  const [casualtyModel, setCasualtyModel] = useState("whr778/gliner2-base-v1-casualty-docee");
  const [eventModel, setEventModel] = useState("");
  const [windowMode, setWindowMode] = useState("article");
  const [normalizer, setNormalizer] = useState("hybrid");
  // Association decides WHICH stream an observation joins. Defaulting to "record" and not
  // "none" because "none" pools every event in the feed into one series -- the failure
  // that association was added to fix (multi-event nRMSE 102 -> 27.975), and what this
  // panel silently did until now.
  const [associate, setAssociate] = useState("record");
  // 0 = off. A per-event plausibility ceiling: anything above the largest credible toll
  // for this event is not a casualty figure. Measured on Helene -- dropping one 94,000
  // (Asheville's population, read as a death toll) is worth 20x on ungated per-place error.
  // 0.9 chosen 2026-08-28 by gate_threshold_sweep.py. The panel had no control for it
  // and the backend default is 0.5, so every viewer run used a different operating point
  // from the research runs it was being compared against.
  const [gateThreshold, setGateThreshold] = useState(0.9);
  const [maxPlausible, setMaxPlausible] = useState(0);
  const [limit, setLimit] = useState(0);
  // Defaults to "cpu" because EkfRequest already defaulted to it: exposing the control
  // must not silently change what an existing run does.
  const [device, setDevice] = useState("cpu");

  const [runs, setRuns] = useState<Run[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [showBaseline, setShowBaseline] = useState(true);
  const [showTruth, setShowTruth] = useState(true);
  const [detail, setDetail] = useState<{ runId: string; index: number } | null>(null);
  const [job, setJob] = useState<EkfJob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<any>(null);

  useEffect(() => {
    getEkfFeeds().then((f) => {
      setFeeds(f);
      if (!f[0]) return;
      setFeed(f[0].path);
      // Apply the first feed's configuration too, not only later changes -- otherwise the
      // default selection is the one case that silently runs misconfigured.
      const rec = f[0].recommended;
      if (!rec) return;
      if (rec.window) setWindowMode(rec.window);
      if (rec.associate) setAssociate(rec.associate);
      if (rec.gate_threshold != null) setGateThreshold(rec.gate_threshold);
      setMaxPlausible(rec.max_plausible ?? 0);
      setRollup(rec.rollup ?? null);
      setEventYear(rec.event_year ?? 0);
      setApplied(rec.why);
    }).catch((e) => setError(String(e)));
    // The model dropdowns are fed by the same registry the main panel uses. A failure
    // here must not blank the panel: the configured default stays selectable because the
    // selects keep an option for a value the registry does not contain.
    getModels().then(setModels).catch(() => {});
    return () => timer.current && clearInterval(timer.current);
  }, []);

  // Selecting a feed applies its documented configuration. Auto-applied rather than
  // offered behind a button because the failure mode is silent: Turkiye with the wrong
  // rollup drops every stream as nan, which looks like "the model found nothing". Every
  // control stays editable afterwards.
  function applyFeed(path: string) {
    setFeed(path);
    const rec = feeds.find((f) => f.path === path)?.recommended;
    if (!rec) { setApplied(null); setRollup(null); setEventYear(0); return; }
    if (rec.window) setWindowMode(rec.window);
    if (rec.associate) setAssociate(rec.associate);
    if (rec.gate_threshold != null) setGateThreshold(rec.gate_threshold);
    setMaxPlausible(rec.max_plausible ?? 0);
    setRollup(rec.rollup ?? null);
    setEventYear(rec.event_year ?? 0);
    setApplied(rec.why);
  }

  function label() {
    // feedLabel, not the basename: the three real feeds are all named feed.jsonl, so
    // overlaying Helene and Turkiye would produce two runs both labelled "feed".
    const parts = [feedLabel(feed), windowMode, normalizer, associate];
    if (eventModel) parts.push("stage1");
    parts.push(casualtyModel.split("/").pop()!);
    return parts.join(" · ");
  }

  async function run() {
    setError(null);
    // Clear the plot for the new run. Leaving the previous run on screen while a new
    // one is in flight reads as "these are your results" when they are the old ones.
    // Only the SELECTION is cleared -- finished runs stay in the Runs list, so a
    // comparison is still one checkbox away once the new run lands.
    setSelected([]);
    setDetail(null);
    try {
      const started = await startEkfTrack({
        feed, casualty_model: casualtyModel, window: windowMode,
        normalizer, associate, limit: Number(limit) || 0,
        max_plausible: Number(maxPlausible) || 0,
        gate_threshold: Number(gateThreshold), device,
        ...(eventYear ? { event_year: eventYear } : {}),
        // null lets the backend resolve a rollup beside the feed; "" disables it, which
        // Turkiye needs and which is NOT the same as omitting the field.
        ...(rollup !== null ? { rollup } : {}),
        ...(eventModel ? { event_model: eventModel } : {}),
      });
      setJob(started);
      const runLabel = label();
      timer.current && clearInterval(timer.current);
      timer.current = setInterval(async () => {
        try {
          const j = await getEkfJob(started.job_id);
          setJob(j);
          if (j.status !== "running") {
            clearInterval(timer.current);
            if (j.status === "done" && j.result) {
              const r: Run = { id: j.job_id, label: runLabel, result: j.result };
              setRuns((prev) => [...prev, r]);
              setSelected((prev) => (prev.length < 2 ? [...prev, r.id] : [prev[1], r.id]));
            }
          }
        } catch (e) { setError(String(e)); clearInterval(timer.current); }
      }, 2000);
    } catch (e) { setError(String(e)); }
  }

  const shown = runs.filter((r) => selected.includes(r.id));
  const detailRun = detail ? runs.find((r) => r.id === detail.runId) : null;

  return (
    <div className="grid">
      <div>
        <div className="card">
          <h2>EKF event tracking</h2>
          <div className="hint" style={{ marginBottom: 10 }}>
            Gate → event typing → casualty extraction → normalize → track. Run a
            configuration, change it, run again: finished runs overlay on one axis.
          </div>

          <label>Feed</label>
          <select value={feed} onChange={(e) => applyFeed(e.target.value)}>
            {feeds.map((f) => (
              <option key={f.path} value={f.path}>
                {feedLabel(f.path)} ({f.articles}{f.truth ? ", truth" : ""})
              </option>
            ))}
          </select>

          {applied && (
            <div className="hint" style={{ marginTop: 6, padding: "6px 8px",
                                           background: "#f0fdf4", borderLeft: "3px solid #16a34a",
                                           borderRadius: 4 }}>
              <strong>Recommended settings applied.</strong> {applied}
            </div>
          )}

          <div className="ekf-models">
            <div className="field">
              <label>Casualty model (stage 2)</label>
              <select value={casualtyModel} onChange={(e) => setCasualtyModel(e.target.value)}>
                {/* The configured model may not be in the registry (a local path, or a
                    checkpoint added since load); keep it selectable rather than silently
                    switching the run to a different model. */}
                {!models.some((m) => m.path === casualtyModel) && (
                  <option value={casualtyModel}>{casualtyModel}</option>
                )}
                {models.map((m) => (
                  <option key={m.path} value={m.path}>{m.label}</option>
                ))}
              </select>
            </div>
            <div className="field">
              <label>Event model (stage 1, optional)</label>
              <select value={eventModel} onChange={(e) => setEventModel(e.target.value)}>
                <option value="">none — skip stage 1</option>
                {!!eventModel && !models.some((m) => m.path === eventModel) && (
                  <option value={eventModel}>{eventModel}</option>
                )}
                {models.map((m) => (
                  <option key={m.path} value={m.path}>{m.label}</option>
                ))}
              </select>
              <div className="hint">
                {eventModel
                  ? "Types each article before extraction, so observations can be keyed by event type. Required by the 'event envelope' window."
                  : "Off: every article goes straight to casualty extraction. Association then falls back to whatever the record itself names."}
              </div>
            </div>
          </div>

          <div className="ekf-controls">
            <div className="field">
              <label>Window</label>
              <select value={windowMode} onChange={(e) => setWindowMode(e.target.value)}>
                <option value="long">whole doc, chunked (research default)</option>
                <option value="article">whole article</option>
                <option value="event">event envelope</option>
                <option value="lead">article lead</option>
              </select>
              <div className="hint">{WINDOW_HINTS[windowMode]}</div>
            </div>
            <div className="field">
              <label>Associate</label>
              <select value={associate} onChange={(e) => setAssociate(e.target.value)}>
                <option value="record">record location</option>
                <option value="type+location">type + location</option>
                <option value="envelope">nearest location</option>
                <option value="type">event type</option>
                <option value="none">none (pool all)</option>
              </select>
              <div className="hint">{ASSOCIATE_HINTS[associate]}</div>
            </div>
            <div className="field">
              <label>Normalizer</label>
              <select value={normalizer} onChange={(e) => setNormalizer(e.target.value)}>
                <option value="hybrid">hybrid</option>
                <option value="heuristic">heuristic</option>
                <option value="classify">classify</option>
              </select>
              <div className="hint">{NORMALIZER_HINTS[normalizer]}</div>
            </div>
            <div className="field">
              <label title="Drop observations above the largest credible toll for this event, before tracking. 0 = off. On Helene a ceiling of 2000 removes a 94,000 that is Asheville's population.">
                Max plausible
              </label>
              <input type="number" min={0} value={maxPlausible}
                     onChange={(e) => setMaxPlausible(Number(e.target.value))} />
            </div>
            <div className="field">
              <label>Device</label>
              <select value={device} onChange={(e) => setDevice(e.target.value)}>
                <option value="cpu">cpu</option>
                <option value="auto">auto (CUDA → MPS → CPU)</option>
                <option value="mps">mps (Apple Silicon GPU)</option>
                <option value="cuda">cuda (NVIDIA GPU)</option>
              </select>
              <div className="hint">
                cpu is the default and often the fastest here: this pipeline runs
                many-label event decode, where mps is 3-4x SLOWER than cpu on per-op
                overhead. mps wins on few-label classification, so the right choice
                depends on the stage. An unavailable device falls back to auto.
              </div>
            </div>
            <div className="field">
              <label>Gate threshold</label>
              <input type="number" min={0} max={1} step={0.01} value={gateThreshold}
                     onChange={(e) => setGateThreshold(Number(e.target.value))} />
              <div className="hint">
                Stage-0 cut. 0.9 was chosen by gate_threshold_sweep.py; the backend default
                is 0.5, so runs below this were at a different operating point from the
                research figures they get compared against.
              </div>
            </div>
            <div className="field">
              <label>Limit</label>
              <input type="number" min={0} value={limit}
                     onChange={(e) => setLimit(Number(e.target.value))} />
            </div>
          </div>

          <button onClick={run} disabled={job?.status === "running" || !feed} style={{ marginTop: 10 }}>
            {job?.status === "running" ? "Running…" : "Run"}
          </button>

          {job?.status === "running" && (
            <div className="hint" style={{ marginTop: 10 }}>
              <span className="mono">{job.stage}</span>
              {job.total > 0 && ` · ${job.done}/${job.total} articles`}
              {job.log?.length > 0 && (
                <pre style={{ fontSize: 11, whiteSpace: "pre-wrap", marginTop: 6 }}>
                  {job.log.join("\n")}
                </pre>
              )}
            </div>
          )}
          {(error || job?.error) && <div className="error">{error || job?.error}</div>}
        </div>

        {runs.length > 0 && (
          <div className="card" style={{ marginTop: 12 }}>
            <h2>Runs</h2>
            <div className="hint" style={{ marginBottom: 8 }}>Select up to two to compare.</div>
            {runs.map((r, i) => {
              const idx = selected.indexOf(r.id);
              return (
                <label key={r.id} style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12 }}>
                  <input type="checkbox" checked={idx >= 0}
                         onChange={() =>
                           setSelected((prev) =>
                             prev.includes(r.id) ? prev.filter((x) => x !== r.id)
                                                 : [...prev, r.id].slice(-2))} />
                  <span style={{ width: 10, height: 10, borderRadius: 5, flex: "0 0 auto",
                                 background: idx >= 0 ? RUN_COLORS[idx % RUN_COLORS.length] : "#d1d5db" }} />
                  <span>{r.label}</span>
                  <span className="hint" style={{ marginLeft: "auto" }}>
                    {r.result.n_observations[r.result.mode]} obs
                  </span>
                </label>
              );
            })}
            <label style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12, marginTop: 8 }}>
              <input type="checkbox" checked={showBaseline}
                     onChange={(e) => setShowBaseline(e.target.checked)} />
              show <span className="mono">last_value</span> baseline
            </label>
            <label style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12, marginTop: 6 }}>
              <input type="checkbox" checked={showTruth}
                     onChange={(e) => setShowTruth(e.target.checked)} />
              overlay <span style={{ color: TRUTH }}>ground truth</span>
            </label>
          </div>
        )}
      </div>

      <div>
        {shown.length > 0 ? (
          <div className="card">
            <div className="hint" style={{ marginBottom: 8 }}>
              {shown[0].result.n_articles} articles · {shown[0].result.n_relevant} passed the gate
              {shown[0].result.truth && " · truth available"} · click a marker to inspect the article
              {shown[0].result.truth_note && (
                <div style={{ marginTop: 4 }}>Ground truth: {shown[0].result.truth_note}.</div>
              )}
            </div>
            <div style={{ display: "flex", gap: 14, fontSize: 12, marginBottom: 10, flexWrap: "wrap" }}>
              {shown.map((r, i) => (
                <span key={r.id} style={{ color: RUN_COLORS[i % RUN_COLORS.length] }}>■ {r.label}</span>
              ))}
              {showBaseline && <span style={{ color: BASELINE }}>■ last_value</span>}
              {showTruth && shown[0].result.truth && <span style={{ color: TRUTH }}>■ ground truth</span>}
            </div>
            {ROLES.map((role) => (
              <Chart key={role} role={role} runs={shown} showBaseline={showBaseline}
                     showTruth={showTruth}
                     onPick={(runId, m) => setDetail({ runId, index: m.article })} />
            ))}
            {detailRun && detail && (
              <ArticleDetail run={detailRun} index={detail.index} onClose={() => setDetail(null)} />
            )}
            {shown[0] && (
              <DocumentFlow run={shown[0]}
                            onPick={(i) => setDetail({ runId: shown[0].id, index: i })} />
            )}
          </div>
        ) : (
          <div className="card">
            <div className="empty">
              {job?.status === "running"
                ? "Tracking… charts appear when the run finishes."
                : "Pick a feed and click Run."}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
