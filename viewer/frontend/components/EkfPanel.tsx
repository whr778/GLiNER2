"use client";

/**
 * EKF tracking: a news feed in, a tracked casualty timeline out.
 *
 * Runs the whole research line in one place -- classification gate -> DocEE event
 * typing -> casualty structure extraction -> normalization -> EKF -- against a
 * time-ordered JSONL feed, and plots the tracked state against the `last_value`
 * baseline and (when the feed ships one) ground truth.
 *
 * The chart is hand-drawn SVG rather than a charting library: three series over a
 * shared time grid needs about sixty lines, and the viewer would otherwise take a
 * new npm dependency for it.
 */

import { useEffect, useRef, useState } from "react";
import { getEkfFeeds, getEkfJob, startEkfTrack, type EkfFeed, type EkfJob } from "@/lib/api";

const ROLES = ["dead", "injured", "missing"] as const;
const COLORS = { ekf: "#2563eb", last_value: "#9ca3af", truth: "#16a34a" };

function Chart({ grid, series, truth, role }: {
  grid: number[];
  series: { ekf: number[]; last_value: number[] };
  truth?: (number | null)[];
  role: string;
}) {
  const W = 560, H = 170, PAD = { l: 44, r: 10, t: 12, b: 26 };
  const xs = grid;
  const all = [...series.ekf, ...series.last_value, ...(truth ?? []).filter((v): v is number => v != null)];
  const yMax = Math.max(1, ...all) * 1.1;
  const x0 = xs[0] ?? 0, x1 = xs[xs.length - 1] ?? 1;
  const px = (t: number) => PAD.l + ((t - x0) / Math.max(x1 - x0, 1e-9)) * (W - PAD.l - PAD.r);
  const py = (v: number) => H - PAD.b - (v / yMax) * (H - PAD.t - PAD.b);
  const path = (vals: (number | null)[]) =>
    vals.map((v, i) => (v == null ? null : `${i && vals[i - 1] != null ? "L" : "M"}${px(xs[i])},${py(v)}`))
        .filter(Boolean).join(" ");

  return (
    <div style={{ marginBottom: 14 }}>
      <div className="hint" style={{ marginBottom: 2 }}>
        <strong>{role}</strong>
      </div>
      <svg width={W} height={H} role="img" aria-label={`${role} over time`}>
        <line x1={PAD.l} y1={H - PAD.b} x2={W - PAD.r} y2={H - PAD.b} stroke="#d1d5db" />
        <line x1={PAD.l} y1={PAD.t} x2={PAD.l} y2={H - PAD.b} stroke="#d1d5db" />
        <text x={4} y={PAD.t + 8} fontSize="10" fill="#6b7280">{Math.round(yMax)}</text>
        <text x={4} y={H - PAD.b} fontSize="10" fill="#6b7280">0</text>
        <text x={PAD.l} y={H - 6} fontSize="10" fill="#6b7280">{x0.toFixed(0)}h</text>
        <text x={W - PAD.r - 30} y={H - 6} fontSize="10" fill="#6b7280">{x1.toFixed(0)}h</text>
        {truth && <path d={path(truth)} fill="none" stroke={COLORS.truth} strokeWidth={2} strokeDasharray="4 3" />}
        <path d={path(series.last_value)} fill="none" stroke={COLORS.last_value} strokeWidth={1.5} />
        <path d={path(series.ekf)} fill="none" stroke={COLORS.ekf} strokeWidth={2} />
      </svg>
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
  const [job, setJob] = useState<EkfJob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<any>(null);

  useEffect(() => {
    getEkfFeeds().then((f) => { setFeeds(f); if (f[0]) setFeed(f[0].path); }).catch((e) => setError(String(e)));
    return () => timer.current && clearInterval(timer.current);
  }, []);

  async function run() {
    setError(null); setJob(null);
    try {
      const started = await startEkfTrack({
        feed, casualty_model: casualtyModel, window: windowMode,
        normalizer, limit: Number(limit) || 0,
        ...(eventModel ? { event_model: eventModel } : {}),
      });
      setJob(started);
      timer.current && clearInterval(timer.current);
      timer.current = setInterval(async () => {
        try {
          const j = await getEkfJob(started.job_id);
          setJob(j);
          if (j.status !== "running") clearInterval(timer.current);
        } catch (e) { setError(String(e)); clearInterval(timer.current); }
      }, 2000);
    } catch (e) { setError(String(e)); }
  }

  const r = job?.status === "done" ? job.result : null;
  const tracked = r ? r.tracked[r.mode] : null;

  return (
    <div className="grid">
      <div>
        <div className="card">
          <h2>EKF event tracking</h2>
          <div className="hint" style={{ marginBottom: 10 }}>
            Gate → event typing → casualty extraction → normalize → track. Point at a
            time-ordered JSONL feed and run.
          </div>

          <label>Feed</label>
          <select value={feed} onChange={(e) => setFeed(e.target.value)}>
            {feeds.map((f) => (
              <option key={f.path} value={f.path}>
                {f.path} ({f.articles} articles{f.truth ? ", has truth" : ""})
              </option>
            ))}
          </select>

          <label>Casualty model</label>
          <input value={casualtyModel} onChange={(e) => setCasualtyModel(e.target.value)} />

          <label>Event model (stage 1, optional)</label>
          <input value={eventModel} onChange={(e) => setEventModel(e.target.value)}
                 placeholder="e.g. out/fastino/gliner2-large-v1-docee/best" />

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
            <div style={{ width: 90 }}>
              <label>Limit</label>
              <input type="number" value={limit} min={0}
                     onChange={(e) => setLimit(Number(e.target.value))} />
            </div>
          </div>

          <button onClick={run} disabled={job?.status === "running" || !feed}
                  style={{ marginTop: 10 }}>
            {job?.status === "running" ? "Running…" : "Run"}
          </button>

          {job && (
            <div className="hint" style={{ marginTop: 10 }}>
              <div>
                <span className="mono">{job.stage}</span>
                {job.total > 0 && ` · ${job.done}/${job.total} articles`}
              </div>
              {job.log?.length > 0 && (
                <pre style={{ fontSize: 11, whiteSpace: "pre-wrap", marginTop: 6 }}>
                  {job.log.join("\n")}
                </pre>
              )}
            </div>
          )}
          {(error || job?.error) && <div className="error">{error || job?.error}</div>}
        </div>
      </div>

      <div>
        {r ? (
          <div className="card">
            <div className="hint" style={{ marginBottom: 10 }}>
              {r.n_articles} articles · {r.n_relevant} passed the gate ·{" "}
              {r.n_observations[r.mode]} observations · mode <span className="mono">{r.mode}</span>
            </div>
            <div style={{ display: "flex", gap: 14, fontSize: 12, marginBottom: 10 }}>
              <span style={{ color: COLORS.ekf }}>■ EKF</span>
              <span style={{ color: COLORS.last_value }}>■ last_value</span>
              {r.truth && <span style={{ color: COLORS.truth }}>■ ground truth</span>}
            </div>
            {ROLES.map((role) => (
              <Chart key={role} role={role} grid={r.grid}
                     series={tracked[role]} truth={r.truth?.[role]} />
            ))}
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
