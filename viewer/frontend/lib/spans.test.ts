// Run: node --test lib/   (Node >= 23 runs TypeScript directly)
import { test } from "node:test";
import assert from "node:assert/strict";
import { buildSegments, codeUnitOffsets, collectMarks } from "./spans.ts";

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
