"use client";

import { useEffect, useRef, useState } from "react";
import InputPanel from "@/components/InputPanel";
import SchemaPanel from "@/components/SchemaPanel";
import ResultView from "@/components/ResultView";
import ModelManager from "@/components/ModelManager";
import EkfPanel from "@/components/EkfPanel";
import { addModel, extract, getModelSchema, getModels, getPresets } from "@/lib/api";
import { matchCorpusPreset } from "@/lib/models";
import { MODEL_PRESET_VALUE, pruneSchema, scaffoldSchema } from "@/lib/schema";
import {
  DEFAULT_OPTIONS,
  type ExtractOptions,
  type ExtractResponse,
  type ModelEntry,
  type Preset,
} from "@/lib/types";

const DEFAULT_SCHEMA: Record<string, any> = {
  entities: ["person", "organization", "location", "date"],
};

export default function Home() {
  const [text, setText] = useState("");
  const [schema, setSchema] = useState<Record<string, any>>(DEFAULT_SCHEMA);
  const [options, setOptions] = useState<ExtractOptions>(DEFAULT_OPTIONS);
  const [presets, setPresets] = useState<Preset[]>([]);
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [showModels, setShowModels] = useState(false);
  // The viewer had a single view; EKF tracking is a different job (a feed over
  // time, not one text), so it gets its own mode rather than more panels.
  const [mode, setMode] = useState<"extract" | "ekf">("extract");
  const [resp, setResp] = useState<ExtractResponse | null>(null);
  const [usedSchema, setUsedSchema] = useState<Record<string, any> | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [schemaNote, setSchemaNote] = useState<string | null>(null);
  const [presetName, setPresetName] = useState("");
  const [modelPreset, setModelPreset] = useState<Record<string, any> | null>(null);
  const [schemaLoading, setSchemaLoading] = useState(false);
  const appliedModel = useRef<string | null>(null);

  useEffect(() => {
    getPresets().then(setPresets).catch(() => {});
    getModels().then(setModels).catch(() => {});
  }, []);

  // Selecting a model loads the schema it was trained on so its ontology matches
  // out of the box. Prefer the schema the model SHIPS in its config
  // (default_schema); fall back to corpus-name matching for older checkpoints
  // that predate co-located schemas. Either way the user can still pick another
  // preset or edit the JSON below.
  useEffect(() => {
    const m = options.model?.trim();
    if (!m || m === appliedModel.current) return;
    let cancelled = false;
    setSchemaLoading(true); // disable extraction until this model's schema resolves
    (async () => {
      const shipped = await getModelSchema(m);
      if (cancelled) return;
      setSchemaLoading(false); // network resolved; the schema apply below is synchronous
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

  async function onExtract() {
    setLoading(true);
    setError(null);
    setResp(null); // clear the previous results immediately
    setUsedSchema(null);
    try {
      // Prune empty scaffold fields + the open_vocab marker; send a valid schema.
      const sent = pruneSchema(schema);
      setResp(await extract(text, sent, options));
      setUsedSchema(sent); // snapshot the schema these results actually ran with

      // Remember a newly-used model once it has successfully extracted.
      const m = options.model?.trim();
      if (m && !models.some((x) => x.path === m)) {
        addModel(m).then(setModels).catch(() => {});
      }
    } catch (e: any) {
      setError(e.message || "extraction failed");
      setResp(null);
    } finally {
      setLoading(false);
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
                  onClick={() => setMode("extract")}>Extract</button>
          <button className={mode === "ekf" ? "" : "secondary"}
                  onClick={() => setMode("ekf")}>EKF tracking</button>
        </div>
      </header>

      {mode === "ekf" ? (
        <EkfPanel />
      ) : (

      <div className="grid">
        <div>
          <InputPanel
            text={text}
            setText={setText}
            options={options}
            setOptions={setOptions}
            loading={loading}
            schemaLoading={schemaLoading}
            onExtract={onExtract}
            models={models}
            onManage={() => setShowModels(true)}
            resultData={resp}
            resultSchema={usedSchema}
            onLoadResults={onLoadResults}
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
            <>
              {resp.device && (
                <div className="hint" style={{ marginBottom: 8 }}>
                  Ran on <span className="mono">{resp.device}</span>
                  {resp.device.startsWith("mps") && " (Apple Silicon GPU)"}
                </div>
              )}
              <ResultView text={resp.text} result={resp.result} schema={usedSchema} />
            </>
          ) : loading ? (
            <div className="card">
              <div className="empty">Extracting…</div>
            </div>
          ) : (
            !error && (
              <div className="card">
                <div className="empty">Enter text, pick a schema, and click Extract to see results.</div>
              </div>
            )
          )}
        </div>
      </div>

      )}

      {showModels && (
        <ModelManager models={models} setModels={setModels} onClose={() => setShowModels(false)} />
      )}
    </div>
  );
}
