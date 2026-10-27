"use client";

import type { InferenceJob } from "@/lib/api";

type Props = {
  jobs: InferenceJob[];
  selected: string | null;
  onSelect: (id: string | null) => void;
  onCancel: (id: string) => void;
};

function statusLine(job: InferenceJob): string {
  const n = job.total ? `${job.done}/${job.total}` : "";
  const cap = job.limit ? `first ${job.limit}` : "";
  return [job.status, n, cap].filter(Boolean).join(" · ");
}

export default function JobRail({ jobs, selected, onSelect, onCancel }: Props) {
  const active = jobs.some((j) => j.status === "queued" || j.status === "running");
  return (
    <div className="card jobrail">
      <div className="row between">
        <h2>Jobs</h2>
        <button type="button" className={selected === null ? "" : "secondary"} onClick={() => onSelect(null)}>New</button>
      </div>
      <div className="hint">
        {active
          ? "One model runs at a time. The next ticket waits, so its gates cannot change under the one that is running."
          : "A ticket is a text or a JSONL file. Pick one to see it."}
      </div>
      {jobs.length === 0 && <div className="empty">No jobs yet.</div>}
      {jobs.map((job) => (
        <button
          type="button"
          key={job.id}
          className={job.id === selected ? "job active" : "job"}
          onClick={() => onSelect(job.id)}
        >
          <div>{job.label}</div>
          <div className="meta">{job.kind === "jsonl" ? "JSONL" : "Text"} · {statusLine(job)}</div>
          <div className="meta mono">{job.model}</div>
          {(job.status === "queued" || job.status === "running") && (
            <span
              className="meta"
              style={{ textDecoration: "underline" }}
              onClick={(e) => { e.stopPropagation(); onCancel(job.id); }}
            >
              Cancel
            </span>
          )}
        </button>
      ))}
    </div>
  );
}
