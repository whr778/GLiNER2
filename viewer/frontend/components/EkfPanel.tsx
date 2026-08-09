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
import { getEkfFeeds, getEkfJob, startEkfTrack, type EkfFeed, type EkfJob } from "@/lib/api";

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
      if (o.role === role && o.mode === result.mode) {
        out.push({ t: o.t_hours, value: o.value, article: i, obs: o });
      }
    });
  });
  return out;
}

function Chart({ role, runs, showBaseline, onPick }: {
  role: Role;
  runs: Run[];
  showBaseline: boolean;
  onPick: (runId: string, m: Marker) => void;
}) {
  const W = 620, H = 190, PAD = { l: 48, r: 12, t: 12, b: 28 };
  const base = runs[0]?.result;
  if (!base) return null;
  const grid: number[] = base.grid;
  const truth: (number | null)[] | undefined = base.truth?.[role];

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

  const ticks = [0, 0.5, 1].map((f) => Math.round(yMax * f));

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
              <tr key={i}>
                <td>{o.role}</td>
                <td align="right"><strong>{o.value}</strong></td>
                <td className="mono">{o.span}</td>
                <td>{o.qualifier}</td>
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

export default function EkfPanel() {
  const [feeds, setFeeds] = useState<EkfFeed[]>([]);
  const [feed, setFeed] = useState("");
  const [casualtyModel, setCasualtyModel] = useState("whr778/gliner2-base-v1-casualty");
  const [eventModel, setEventModel] = useState("");
  const [windowMode, setWindowMode] = useState("article");
  const [normalizer, setNormalizer] = useState("hybrid");
  const [limit, setLimit] = useState(0);

  const [runs, setRuns] = useState<Run[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [showBaseline, setShowBaseline] = useState(true);
  const [detail, setDetail] = useState<{ runId: string; index: number } | null>(null);
  const [job, setJob] = useState<EkfJob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<any>(null);

  useEffect(() => {
    getEkfFeeds().then((f) => { setFeeds(f); if (f[0]) setFeed(f[0].path); }).catch((e) => setError(String(e)));
    return () => timer.current && clearInterval(timer.current);
  }, []);

  function label() {
    const parts = [feed.split("/").pop()!.replace(".jsonl", ""), windowMode, normalizer];
    if (eventModel) parts.push("stage1");
    parts.push(casualtyModel.split("/").pop()!);
    return parts.join(" · ");
  }

  async function run() {
    setError(null);
    try {
      const started = await startEkfTrack({
        feed, casualty_model: casualtyModel, window: windowMode,
        normalizer, limit: Number(limit) || 0,
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
          <select value={feed} onChange={(e) => setFeed(e.target.value)}>
            {feeds.map((f) => (
              <option key={f.path} value={f.path}>
                {f.path.split("/").pop()} ({f.articles}{f.truth ? ", truth" : ""})
              </option>
            ))}
          </select>

          <label>Casualty model</label>
          <input value={casualtyModel} onChange={(e) => setCasualtyModel(e.target.value)} />

          <label>Event model (stage 1, optional)</label>
          <input value={eventModel} onChange={(e) => setEventModel(e.target.value)}
                 placeholder="out/fastino/gliner2-large-v1-docee/best" />

          <div style={{ display: "flex", gap: 8 }}>
            <div style={{ flex: 1 }}>
              <label>Window</label>
              <select value={windowMode} onChange={(e) => setWindowMode(e.target.value)}>
                <option value="article">whole article</option>
                <option value="event">event envelope</option>
              </select>
            </div>
            <div style={{ flex: 1 }}>
              <label>Normalizer</label>
              <select value={normalizer} onChange={(e) => setNormalizer(e.target.value)}>
                <option value="hybrid">hybrid</option>
                <option value="heuristic">heuristic</option>
                <option value="classify">classify</option>
              </select>
            </div>
            <div style={{ width: 84 }}>
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
          </div>
        )}
      </div>

      <div>
        {shown.length > 0 ? (
          <div className="card">
            <div className="hint" style={{ marginBottom: 8 }}>
              {shown[0].result.n_articles} articles · {shown[0].result.n_relevant} passed the gate
              {shown[0].result.truth && " · truth available"} · click a marker to inspect the article
            </div>
            <div style={{ display: "flex", gap: 14, fontSize: 12, marginBottom: 10, flexWrap: "wrap" }}>
              {shown.map((r, i) => (
                <span key={r.id} style={{ color: RUN_COLORS[i % RUN_COLORS.length] }}>■ {r.label}</span>
              ))}
              {showBaseline && <span style={{ color: BASELINE }}>■ last_value</span>}
              {shown[0].result.truth && <span style={{ color: TRUTH }}>■ ground truth</span>}
            </div>
            {ROLES.map((role) => (
              <Chart key={role} role={role} runs={shown} showBaseline={showBaseline}
                     onPick={(runId, m) => setDetail({ runId, index: m.article })} />
            ))}
            {detailRun && detail && (
              <ArticleDetail run={detailRun} index={detail.index} onClose={() => setDetail(null)} />
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
