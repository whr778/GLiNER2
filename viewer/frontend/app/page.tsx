"use client";

import { useEffect, useState } from "react";
import InputPanel from "@/components/InputPanel";
import SchemaPanel from "@/components/SchemaPanel";
import ResultView from "@/components/ResultView";
import ModelManager from "@/components/ModelManager";
import { addModel, extract, getModels, getPresets } from "@/lib/api";
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
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getPresets().then(setPresets).catch(() => {});
    getModels().then(setModels).catch(() => {});
  }, []);

  async function onExtract() {
    setLoading(true);
    setError(null);
    try {
      setResp(await extract(text, schema, options));
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
          <SchemaPanel presets={presets} schema={schema} setSchema={setSchema} />
        </div>

        <div>
          {error && (
            <div className="card">
              <div className="error">{error}</div>
            </div>
          )}
          {resp ? (
            <ResultView text={resp.text} result={resp.result} />
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
