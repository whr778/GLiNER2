// Run: node --test lib/   (Node >= 23 runs TypeScript directly)
import { test } from "node:test";
import assert from "node:assert/strict";
import { managedModelId } from "./models.ts";

test("a checkpoint without a subfolder is selected by its path", () => {
  assert.equal(managedModelId({ path: "fastino/gliner2-base-v1" }), "fastino/gliner2-base-v1");
  assert.equal(
    managedModelId({ path: "fastino/gliner2-large-v1", id: "fastino/gliner2-large-v1" }),
    "fastino/gliner2-large-v1",
  );
});

test("two epochs of one Hub repo stay separate entries", () => {
  const repo = "whr778/gliner2-eb22d-trace-epoch";
  const epoch1 = managedModelId({ path: repo, id: `${repo}@checkpoint-epoch-1` });
  const epoch2 = managedModelId({ path: repo, id: `${repo}@checkpoint-epoch-2` });
  assert.equal(epoch1, `${repo}@checkpoint-epoch-1`);
  assert.equal(epoch2, `${repo}@checkpoint-epoch-2`);
  assert.notEqual(epoch1, epoch2);
  assert.notEqual(epoch1, repo);
});
