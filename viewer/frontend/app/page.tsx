"use client";

import { useEffect, useRef, useState } from "react";
import InputPanel from "@/components/InputPanel";
import SchemaPanel from "@/components/SchemaPanel";
import ResultView from "@/components/ResultView";
import ModelManager from "@/components/ModelManager";
import { addModel, extract, getModels, getPresets } from "@/lib/api";
import { matchCorpusPreset } from "@/lib/models";
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
  const [resp, setResp] = useState<ExtractResponse | null>(null);
  const [usedSchema, setUsedSchema] = useState<Record<string, any> | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [schemaNote, setSchemaNote] = useState<string | null>(null);
  const [presetName, setPresetName] = useState("");
  const appliedModel = useRef<string | null>(null);

  useEffect(() => {
    getPresets().then(setPresets).catch(() => {});
    getModels().then(setModels).catch(() => {});
  }, []);

  // Selecting a model loads the schema it was trained on (e.g. the casie model
  // -> the `corpus: casie` events), so its ontology matches out of the box.
  useEffect(() => {
    const m = options.model?.trim();
    if (!m || m === appliedModel.current || presets.length === 0) return;
    const preset = matchCorpusPreset(m, presets);
    if (preset) {
      setSchema(preset.schema);
      setPresetName(preset.name); // reflect it in the "Load a preset" dropdown
      setSchemaNote(`Schema loaded from ${preset.name} to match the selected model.`);
      appliedModel.current = m;
    }
  }, [options.model, presets]);

  async function onExtract() {
    setLoading(true);
    setError(null);
    setResp(null); // clear the previous results immediately
    setUsedSchema(null);
    try {
      setResp(await extract(text, schema, options));
      setUsedSchema(schema); // snapshot the schema these results came from

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

  return (
    <div className="app">
      <header>
        <h1>GLiNER2 Viewer</h1>
        <span className="sub">entities · relations · events · classifications · structures</span>
      </header>

      <div className="grid">
        <div>
          <InputPanel
            text={text}
            setText={setText}
            options={options}
            setOptions={setOptions}
            loading={loading}
            onExtract={onExtract}
            models={models}
            onManage={() => setShowModels(true)}
            resultData={resp}
          />
          <SchemaPanel
            presets={presets}
            schema={schema}
            setSchema={setSchema}
            note={schemaNote}
            clearNote={() => setSchemaNote(null)}
            selectedPreset={presetName}
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

      {showModels && (
        <ModelManager models={models} setModels={setModels} onClose={() => setShowModels(false)} />
      )}
    </div>
  );
}
