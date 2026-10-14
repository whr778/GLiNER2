"use client";

import { useEffect, type ReactNode } from "react";
import { tagStyle } from "@/lib/colors";

// A sample span drawn with the SAME classes AnnotatedText uses, so the legend is the real look.
function Sample({ text, label, cls, kind = "entity" }: { text: string; label: string; cls: string; kind?: string }) {
  return (
    <mark className={"tag " + cls} data-kind={kind} style={tagStyle(label)}>
      {text}
      <span className="taglabel" style={tagStyle(label)}>{label}</span>
    </mark>
  );
}

const ROWS: [ReactNode, string, string][] = [
  [<Sample key="s" text="KMTV" label="media outlet" cls="gold" />, "Solid gold rim",
   "The prediction matches gold under the scorer's strict key."],
  [<Sample key="d" text="Papillion" label="Location" cls="gold golddiff" />, "Dashed gold rim",
   "Gold covers this exact text, but no prediction on it matches. For entities the text is the same, so the "
   + "LABEL differs (hover: \"gold here: location\"). For event arguments the role can even be the same: the strict "
   + "argument key includes the mention's trigger set, so an argument attached to a different mention is a miss."],
  [<Sample key="m" text="American Bulldog" label="species" cls="gold goldonly" />, "Rim only, gold label",
   "Gold the model missed: nothing was predicted on this text."],
  [<Sample key="f" text="said" label="Contact.Communicate" cls="" kind="trigger" />, "Fill, no rim",
   "A prediction that is not in gold (a false positive under the strict key)."],
];

// What the Predictions vs gold view draws and why; opened from the panel's Help button.
export default function PredictionsHelp({ onClose }: { onClose: () => void }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <>
      <div className="backdrop" onClick={onClose} />
      <div className="modal card">
        <div className="row between">
          <h2>Predictions vs gold: help</h2>
          <button onClick={onClose}>Close</button>
        </div>

        <h3>What the rims mean</h3>
        <table>
          <tbody>
            {ROWS.map(([sample, name, text]) => (
              <tr key={name}>
                <td style={{ whiteSpace: "nowrap", padding: "6px 10px 6px 0" }}>{sample}</td>
                <td><b>{name}</b><div className="hint">{text}</div></td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="hint">The colored tag on a span is the MODEL&apos;s label; hover a span to see the gold label and its status.</p>

        <h3>How gold is matched</h3>
        <ul className="hint">
          <li>Exactly as the strict scorer (<span className="mono">eval_metrics</span>), so the rims agree with the strict F1:
            entity = (label, text); trigger = (event type, text); argument = (event type, role, text, the mention&apos;s
            triggers); relation = (name, head, tail). Case-sensitive.</li>
          <li>Gold stores text, not positions. It is placed the way training places it: every occurrence of the word run.</li>
          <li>Gold training cannot place (e.g. <span className="mono">Houston</span> inside
            <span className="mono"> Houston-area</span>, one word to the splitter) is listed under the text: the model was never taught it.</li>
        </ul>

        <h3>Notices</h3>
        <ul className="hint">
          <li><b>No gold_mapped:</b> gold is in the corpus spellings, so <span className="mono">location</span> vs
            <span className="mono"> Location</span> is a miss (dashed). Real for the scorer, but not a model error.</li>
          <li><b>No spans:</b> predictions have no positions, so none can be drawn and every gold span reads as missed.
            The F1 numbers and classifications are still right.</li>
        </ul>

        <h3>Controls</h3>
        <ul className="hint">
          <li>Prev / Next or the arrow keys; type a record number to jump.</li>
          <li><b>worst ... F1 first</b> sorts records by one head&apos;s strict F1 (records without that head go last).</li>
          <li><b>show gold</b> toggles the rims; <b>differences only</b> hides everything that matched.</li>
          <li>Click a span on the Events / Relations tab to isolate one mention and its arcs; click again to cycle.</li>
        </ul>

        <h3>Making a file to open</h3>
        <pre className="mono hint" style={{ whiteSpace: "pre-wrap" }}>
{`uv run python tools/infer.py --model <model> --input data/<corpus>.test.jsonl \\
    --gold-schema --include-spans --labels-file <labels file> \\
    --output out/predictions/<name>.jsonl`}
        </pre>
        <p className="hint"><span className="mono">--include-spans</span> gives predictions positions;{" "}
          <span className="mono">--labels-file</span> writes gold_mapped. The dropdown lists files under{" "}
          <span className="mono">predictions_root</span> (viewer/config.yaml).</p>
      </div>
    </>
  );
}
