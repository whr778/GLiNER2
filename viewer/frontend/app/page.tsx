"use client";

import { useEffect, useRef, useState } from "react";
import InputPanel from "@/components/InputPanel";
import SchemaPanel from "@/components/SchemaPanel";
import ResultView from "@/components/ResultView";
import ModelManager from "@/components/ModelManager";
import EkfPanel from "@/components/EkfPanel";
import PredictionsPanel from "@/components/PredictionsPanel";
import JobRail from "@/components/JobRail";
import { addModel, cancelJob, createJob, getJob, getModelSchema, getModels, getPresets, listJobs, type InferenceJob } from "@/lib/api";
import { matchCorpusPreset } from "@/lib/models";
import { MODEL_PRESET_VALUE, pruneSchema, scaffoldSchema } from "@/lib/schema";
import {
  DEFAULT_OPTIONS,
  type CorefInfo,
  type ExtractOptions,
  type ExtractResponse,
  type ModelEntry,
  type Preset,
} from "@/lib/types";

const DEFAULT_SCHEMA: Record<string, any> = {
  entities: ["person", "organization", "location", "date"],
};

function JobBody({ ticket, full }: {
  ticket: InferenceJob;
  full: InferenceJob | null;
}) {
  const textResult = full && full.id === ticket.id ? full.result : null;
  const textSchema = full && full.id === ticket.id ? full.schema ?? null : null;
  return (
    <div>
      <div className="card">
        <h2>{ticket.label}</h2>
        <div className="hint">
          {ticket.status}
          {ticket.total ? ` · ${ticket.done}/${ticket.total}` : ""}
          {ticket.kind === "jsonl" ? (ticket.schema_mode === "gold" ? " · gold schema" : " · schema on the form") : ""}
          {ticket.ran_on ? ` · ${ticket.ran_on}` : ""}
        </div>
        {ticket.output_path && <div className="mono hint" style={{ marginTop: 6 }}>{ticket.output_path}</div>}
        {ticket.error && <div className="error">{ticket.error}</div>}
        {ticket.kind === "jsonl" && ticket.status === "running" && ticket.done > 0 && (
          <div className="hint" style={{ marginTop: 6 }}>
            The predictions below opened on the records written so far. They reload when the job finishes.
          </div>
        )}
      </div>
      {ticket.kind === "text" && textResult && (
        <div style={{ marginTop: 14 }}>
          <ResultView text={textResult.text} result={textResult.result} schema={textSchema} />
        </div>
      )}
      {ticket.kind === "text" && ticket.status === "done" && !textResult && (
        <div className="card"><div className="empty">Loading the result…</div></div>
      )}
      {ticket.kind === "jsonl" && ticket.output_path && ticket.done > 0 && (
        <div style={{ marginTop: 14 }}>
          <PredictionsPanel openPath={ticket.output_path} reloadKey={ticket.status} />
        </div>
      )}
    </div>
  );
}

export default function Home() {
  const [text, setText] = useState("");
  const [schema, setSchema] = useState<Record<string, any>>(DEFAULT_SCHEMA);
  const [options, setOptions] = useState<ExtractOptions>(DEFAULT_OPTIONS);
  const [presets, setPresets] = useState<Preset[]>([]);
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [showModels, setShowModels] = useState(false);
  // The viewer had a single view; EKF tracking is a different job (a feed over
  // time, not one text), so it gets its own mode rather than more panels.
  const [mode, setMode] = useState<"extract" | "ekf" | "predictions">("extract");
  const [resp, setResp] = useState<ExtractResponse | null>(null);
  const [usedSchema, setUsedSchema] = useState<Record<string, any> | null>(null);
  const [queueing, setQueueing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [source, setSource] = useState<"text" | "jsonl">("text");
  const [schemaMode, setSchemaMode] = useState<"gold" | "panel">("gold");
  const [dataset, setDataset] = useState("");
  const [limit, setLimit] = useState("");
  const [jobs, setJobs] = useState<InferenceJob[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [full, setFull] = useState<InferenceJob | null>(null);
  const [schemaNote, setSchemaNote] = useState<string | null>(null);
  const [presetName, setPresetName] = useState("");
  const [modelPreset, setModelPreset] = useState<Record<string, any> | null>(null);
  const [coref, setCoref] = useState<CorefInfo | null>(null);
  const [schemaLoading, setSchemaLoading] = useState(false);
  const appliedModel = useRef<string | null>(null);

  useEffect(() => {
    getPresets().then(setPresets).catch(() => {});
    getModels().then(setModels).catch(() => {});
  }, []);

  useEffect(() => {
    let stop = false;
    const tick = () => {
      listJobs().then((rows) => { if (!stop) setJobs(rows); }).catch(() => {});
    };
    tick();
    const id = setInterval(tick, 1500);
    return () => { stop = true; clearInterval(id); };
  }, []);

  const ticket = jobs.find((j) => j.id === selected) ?? null;

  useEffect(() => {
    if (!ticket || ticket.kind !== "text" || ticket.status !== "done") return;
    let stop = false;
    getJob(ticket.id).then((row) => { if (!stop) setFull(row); }).catch(() => {});
    return () => { stop = true; };
  }, [ticket?.id, ticket?.status]);

  // Selecting a model loads the schema it was trained on so its ontology matches
  // out of the box. Prefer the schema the model SHIPS in its config
  // (default_schema); fall back to corpus-name matching for older checkpoints
  // that predate co-located schemas. Either way the user can still pick another
  // preset or edit the JSON below.
  useEffect(() => {
    const m = options.model?.trim();
    if (!m) {
      setCoref(null);
      return;
    }
    if (m === appliedModel.current) return;
    let cancelled = false;
    setSchemaLoading(true); // disable extraction until this model's schema resolves
    (async () => {
      const { schema: shipped, inferenceDefaults, recordDefaults, coref: shippedCoref } = await getModelSchema(m);
      if (cancelled) return;
      setSchemaLoading(false); // network resolved; the schema apply below is synchronous
      setCoref(shippedCoref);
      // Decode the way this model was evaluated; the user can still move the controls.
      // Record/argument gates start at the model's own values (null = follow threshold).
      // Merge and entity-cluster overrides stay off until the user turns them on.
      setOptions((o) => ({
        ...o,
        ...(inferenceDefaults ?? {}),
        record_threshold: recordDefaults?.record_threshold ?? null,
        argument_threshold: recordDefaults?.argument_threshold ?? null,
        merge_coreferent: null,
        coref_threshold: null,
        entity_coref_threshold: null,
      }));
      if (shipped && Object.keys(shipped).length > 0) {
        // Open-vocab task types ship as an `open_vocab` marker; scaffold them into
        // empty fields the user can fill (pruned back out at extract time).
        const ov: string[] = shipped.open_vocab || [];
        const scaffolded = scaffoldSchema(shipped);
        setSchema(scaffolded);
        setModelPreset(scaffolded); // selectable "this model's schema" dropdown entry
        setPresetName(MODEL_PRESET_VALUE); // reflect it as the active selection
        setSchemaNote(
          ov.length
            ? `Loaded the schema shipped with this model. It is open-vocabulary for ${ov.join(", ")} -- add labels to the empty field${ov.length > 1 ? "s" : ""}. You can still pick another below.`
            : "Loaded the schema shipped with this model. You can still pick another below.",
        );
        appliedModel.current = m;
        return;
      }
      if (presets.length === 0) return; // fallback needs presets; wait for them
      setModelPreset(null); // older model: no shipped schema to offer
      const preset = matchCorpusPreset(m, presets);
      if (preset) {
        setSchema(preset.schema);
        setPresetName(preset.name); // reflect it in the "Load a preset" dropdown
        setSchemaNote(`Schema loaded from ${preset.name} to match the selected model.`);
      }
      appliedModel.current = m;
    })();
    return () => {
      cancelled = true;
    };
  }, [options.model, presets]);

  function chooseSource(next: "text" | "jsonl") {
    setSource(next);
    if (next === "jsonl" && (options.device ?? "auto") === "auto") {
      setOptions({ ...options, device: "cpu" });
    }
  }

  async function onQueue() {
    setQueueing(true);
    setError(null);
    try {
      const sent = pruneSchema(schema);
      const n = limit.trim() ? Number(limit) : 0;
      if (limit.trim() && (!Number.isFinite(n) || n < 1)) {
        setError("First N must be a positive number, or blank for the whole file.");
        return;
      }
      const job = await createJob({
        kind: source,
        text: source === "text" ? text : undefined,
        path: source === "jsonl" ? dataset : undefined,
        schema_mode: source === "jsonl" ? schemaMode : "panel",
        schema: sent,
        options,
        limit: n,
      });
      setUsedSchema(sent);
      setSelected(job.id);
      setJobs(await listJobs());
      const m = options.model?.trim();
      if (m && !models.some((x) => x.path === m)) {
        addModel(m).then(setModels).catch(() => {});
      }
    } catch (e: any) {
      setError(e.message || "could not queue the job");
    } finally {
      setQueueing(false);
    }
  }

  async function onCancel(id: string) {
    try {
      await cancelJob(id);
      setJobs(await listJobs());
    } catch (e: any) {
      setError(e.message || "could not cancel");
    }
  }

  // Restore a previously-saved results snapshot (text + schema + result) so the
  // viewer renders it without re-running the model. Older files (just
  // {text, result}) still load; the tabs then fall back to the result's layers.
  function onLoadResults(snap: {
    text: string;
    result: Record<string, any>;
    schema?: Record<string, any>;
    options?: ExtractOptions;
  }) {
    setError(null);
    setText(snap.text);
    if (snap.schema) setSchema(snap.schema);
    if (snap.options) setOptions(snap.options);
    setResp({ text: snap.text, result: snap.result });
    setUsedSchema(snap.schema ?? null);
  }

  return (
    <div className="app">
      <header>
        <h1>GLiNER2 Viewer</h1>
        <span className="sub">entities · relations · events · classifications · structures</span>
        <div style={{ marginLeft: "auto", display: "flex", gap: 6 }}>
          <button className={mode === "extract" ? "" : "secondary"}
                  onClick={() => setMode("extract")}>Jobs</button>
          <button className={mode === "ekf" ? "" : "secondary"}
                  onClick={() => setMode("ekf")}>EKF tracking</button>
          <button className={mode === "predictions" ? "" : "secondary"}
                  onClick={() => setMode("predictions")}>Predictions vs gold</button>
        </div>
      </header>

      {mode === "ekf" ? (
        <EkfPanel />
      ) : mode === "predictions" ? (
        <PredictionsPanel />
      ) : (

      <div className="sessions">
        <JobRail jobs={jobs} selected={selected} onSelect={setSelected} onCancel={onCancel} />
        {ticket ? (
          <JobBody ticket={ticket} full={full} />
        ) : (
          <div className="grid">
            <div>
              <InputPanel
                text={text}
                setText={setText}
                options={options}
                setOptions={setOptions}
                queueing={queueing}
                schemaLoading={schemaLoading}
                onQueue={onQueue}
                source={source}
                setSource={chooseSource}
                dataset={dataset}
                setDataset={setDataset}
                schemaMode={schemaMode}
                setSchemaMode={setSchemaMode}
                limit={limit}
                setLimit={setLimit}
                models={models}
                onManage={() => setShowModels(true)}
                resultData={resp}
                resultSchema={usedSchema}
                onLoadResults={onLoadResults}
                coref={coref}
              />
              <SchemaPanel
                presets={presets}
                schema={schema}
                setSchema={setSchema}
                note={schemaNote}
                clearNote={() => setSchemaNote(null)}
                selectedPreset={presetName}
                modelPreset={modelPreset}
              />
            </div>
            <div>
              {error && (
                <div className="card">
                  <div className="error">{error}</div>
                </div>
              )}
              {resp ? (
                <ResultView text={resp.text} result={resp.result} schema={usedSchema} />
              ) : (
                !error && (
                  <div className="card">
                    <div className="empty">
                      Queue a text or a JSONL file. The ticket shows in the list. A finished JSONL job opens here as predictions against gold.
                    </div>
                  </div>
                )
              )}
            </div>
          </div>
        )}
      </div>

      )}

      {showModels && (
        <ModelManager models={models} setModels={setModels} onClose={() => setShowModels(false)} />
      )}
    </div>
  );
}
