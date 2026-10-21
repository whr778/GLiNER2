"""probe_event_collapse.entity_rows: each gold entity is found on ITS label's query, at its own candidate, with rank and gate."""
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "train"))
from probe_event_collapse import entity_rows  # noqa: E402

TEXT = "Ann met Bob in Paris."
STARTS, ENDS = [0, 4, 8, 12, 15], [3, 7, 11, 14, 21]        # words: Ann met Bob in Paris.


def _cap(offset: int = 0):
    """Two entity queries (person=0, city=1) over three candidates each: (0,1) Ann, (2,3) Bob, (4,5) Paris."""
    idx = torch.tensor([[0, 1], [2, 3], [4, 5]]) + offset
    cand = NS(indices=torch.stack([idx, idx]).unsqueeze(0), valid_mask=torch.ones(1, 2, 3, dtype=torch.bool),
              pair_logits=torch.tensor([[[2.0, 1.0, -3.0], [-1.0, -2.0, 0.5]]]))
    queries = [NS(role_name="person", query_id=0, task_type="entities"),
               NS(role_name="city", query_id=1, task_type="entities")]
    batch = NS(start_mappings=[STARTS], end_mappings=[ENDS], query_layouts=[NS(queries=queries)])
    fwd = {"out": NS(candidates=cand), "core": {"word_offsets": torch.tensor([offset])}}
    pred = {"entities": {"person": [{"text": "Ann"}], "city": []}}
    return pred, batch, fwd


def test_entity_rows_find_each_gold_on_its_label_query():
    rec = {"input": TEXT, "output": {"entities": {"person": ["Ann", "Bob"], "city": ["Paris"], "unknown": ["x"]}}}
    rows = {r["trigger"]: r for r in entity_rows(_cap(), rec, 0.3, 1.0)}
    assert set(rows) == {"Ann", "Bob", "Paris"}
    assert (rows["Ann"]["query_id"], rows["Ann"]["gold_candidate"], rows["Ann"]["gold_logit"], rows["Ann"]["gold_rank"]) == (0, 0, 2.0, 0)
    assert (rows["Bob"]["gold_candidate"], rows["Bob"]["gold_rank"]) == (1, 1)
    assert (rows["Paris"]["query_id"], rows["Paris"]["gold_candidate"], rows["Paris"]["gold_logit"]) == (1, 2, 0.5)
    assert rows["Ann"]["emitted_trigger"] and not rows["Bob"]["emitted_trigger"] and not rows["Paris"]["emitted_type"]
    assert rows["Paris"]["n_above_gate"] == 1 and rows["Ann"]["n_above_gate"] == 2


def test_entity_rows_offset_gold_and_word_offset():
    rec = {"input": TEXT, "output": {"entities": {"city": [{"text": "Paris", "start": 15, "end": 20}]}}}
    (row,) = entity_rows(_cap(offset=7), rec, 0.3, 1.0)
    assert row["reached"] and row["gold_candidate"] == 2 and row["gold_logit"] == 0.5
