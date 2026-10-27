// Run: node --test lib/   (Node >= 23 runs TypeScript directly)
import { test } from "node:test";
import assert from "node:assert/strict";
import { buildSegments, codeUnitOffsets, collectMarks, eventTriggerLabel, viewLayers } from "./spans.ts";

const marked = (text: string, result: any, layer: any) =>
  buildSegments(text, collectMarks(result, layer)).filter((s) => s.mark).map((s) => s.text);

test("code-point offsets highlight the right text after emoji", () => {
  // Offsets from the model on this exact text (p5link-junc_w03, 2026-10-07).
  const text = "😀 Emmanuel Macron visited Paris on Monday. 🎉🎉 He later met Olaf Scholz in Berlin.";
  const result = { entities: { person: [{ start: 2, end: 17 }, { start: 59, end: 70 }] } };
  assert.deepEqual(marked(text, result, "entities"), ["Emmanuel Macron", "Olaf Scholz"]);
});

test("ASCII text is unchanged", () => {
  const text = "Emmanuel Macron visited Paris.";
  assert.deepEqual(marked(text, { entities: { person: [{ start: 0, end: 15 }] } }, "entities"), ["Emmanuel Macron"]);
});

test("a span ending at the last code point is kept; past it is dropped", () => {
  const text = "go 🎉Paris";
  assert.deepEqual(marked(text, { entities: { loc: [{ start: 4, end: 9 }, { start: 0, end: 10 }] } }, "entities"), ["Paris"]);
});

test("nested sub-range offsets are local UTF-16, as DOM ranges need", () => {
  const text = "x 🎉Seoul National University";
  const r = { relation_extraction: { located_in: [{ head: { start: 2, end: 28 }, tail: { start: 3, end: 8 } }] } };
  const seg = buildSegments(text, collectMarks(r as any, "relations")).find((s) => s.mark)!;
  const n = seg.mark!.nested![0];
  assert.equal(seg.text.slice(n.s, n.e), "Seoul");
});

test("codeUnitOffsets maps each code point to its UTF-16 index", () => {
  assert.deepEqual(codeUnitOffsets("a😀b"), [0, 1, 3, 4]);
});

test("two triggers of one event share one id", () => {
  const result = { event_extraction: { Attack: [{ triggers: [
    { text: "shot", start: 0, end: 4 },
    { text: "killed", start: 5, end: 11 },
  ], arguments: [] }] } };
  const marks = collectMarks(result, "events");
  assert.equal(marks.length, 2);
  assert.equal(marks[0].eid, "evt-0");
  assert.equal(marks[1].eid, marks[0].eid);
  assert.equal(eventTriggerLabel(2, 1, 0), "2 trigger mentions, one event");
  assert.equal(eventTriggerLabel(1, 2, 1), "trigger #2");
});

test("an empty coref_clusters list still opens the coref layer", () => {
  assert.deepEqual(viewLayers({ entities: ["Person"] }, { coref_clusters: [] }), ["entities", "coref"]);
  assert.deepEqual(viewLayers({ entities: ["Person"] }, {}), ["entities"]);
});

test("gold marks carry gold + match, and gold event ids never collide with predicted ones", () => {
  const span = (text: string, start: number, match?: boolean) => ({ text, start, end: start + text.length, match });
  const pred = { event_extraction: { Attack: [{ triggers: [span("hit", 4, true)], arguments: [] }] } };
  const gold = { event_extraction: { Attack: [{ triggers: [span("hit", 4, true)], arguments: [{ role: "Place", entity: span("Rome", 11, false) }] }] } };
  const p = collectMarks(pred, "events");
  const g = collectMarks(gold, "events", true);
  assert.deepEqual(p.map((m) => [m.gold, m.match, m.eid]), [[false, true, "evt-0"]]);
  assert.deepEqual(g.map((m) => [m.gold, m.match, m.eid]), [[true, true, "g-evt-0"], [true, false, "g-evt-0"]]);
  // A predicted and a gold mark on one range merge into one span; the predicted role is primary.
  const seg = buildSegments("Mob hit in Rome", [...p, ...g]).filter((s) => s.mark);
  assert.equal(seg[0].mark!.roles.length, 2);
  assert.equal(seg[0].mark!.primary.gold, false);
  assert.equal(seg[1].mark!.primary.gold, true);
});
