"""add_coref_clusters: offsets mapped through normalization, records otherwise byte-identical, no re-run."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "data"))
import add_coref_clusters as A  # noqa: E402

RAW = "美国​海军在南海。美国称演习结束。"


def _docs():
    return [(RAW, [[("美国", 0, 2), ("美国", 9, 11)], [("海军", 3, 5)]])]


def test_offsets_follow_the_normalization_the_disk_text_was_written_under():
    counts = {"kept": 0, "dropped": 0}
    stripped = A.clean_text(RAW)
    out = A.place(stripped, _docs()[0][1], A.clean_text, RAW, counts)
    assert [[(m["text"], m["start"]) for m in c] for c in out] == [[("美国", 0), ("美国", 8)], [("海军", 2)]]
    kept = A.place(RAW, _docs()[0][1], A._old_normalize, RAW, counts)
    assert [m["start"] for m in kept[0]] == [0, 9]
    assert counts == {"kept": 6, "dropped": 0}


def test_a_mention_whose_surface_is_not_at_its_offset_is_dropped():
    counts = {"kept": 0, "dropped": 0}
    A.place(RAW, [[("海军", 0, 2)]], A._old_normalize, RAW, counts)
    assert counts == {"kept": 0, "dropped": 1}


def test_augment_adds_only_the_key_keeps_a_bak_and_refuses_a_second_run(tmp_path):
    rec = {"input": RAW, "output": {"events": [{"event_type": "X", "triggers": ["演习"], "arguments": []}]}}
    other = {"input": "无关文本", "output": {}}
    path = tmp_path / "c.train.jsonl"
    original = "\n".join(json.dumps(r, ensure_ascii=False) for r in (rec, other)) + "\n"
    path.write_text(original, encoding="utf-8")
    counts = A.augment(path, _docs())
    assert (counts["matched"], counts["clusters"], counts["dropped"]) == (1, 2, 0)
    a, b = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(a.pop(A.KEY)) == 2 and a == rec and b == other
    assert path.with_suffix(".jsonl.bak").read_text(encoding="utf-8") == original
    with pytest.raises(SystemExit):
        A.augment(path, _docs())


def test_a_surface_with_edge_spaces_is_trimmed_with_its_offsets():
    raw = "演习。 HM-15中队"
    counts = {"kept": 0, "dropped": 0}
    (c,) = A.place(raw, [[(" HM-15中队", 3, 11)]], A._old_normalize, raw, counts)
    assert c == [{"text": "HM-15中队", "start": 4, "end": 11}] and counts == {"kept": 1, "dropped": 0}


def test_cmnee_converter_offsets_index_the_text_dumps_record_writes():
    import convert_cmnee
    from _split import dumps_record
    row = {"text": RAW, "event_list": [{"event_type": "X", "trigger": {"text": "演习", "offset": [12, 14]},
                                        "arguments": []}],
           "coref_arguments": [[{"text": "美国", "offset": [0, 2]}, {"text": "美国", "offset": [9, 11]}]]}
    written = json.loads(dumps_record(convert_cmnee.convert_row(row)))
    assert all(written["input"][m["start"]:m["end"]] == "美国" for m in written["coref_clusters"][0])
    assert [m["start"] for m in written["coref_clusters"][0]] == [0, 8]
