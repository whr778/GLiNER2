"""`corpora` and `event_files` are concatenated, so a corpus in both was counted twice.

Measured on eb16-eventrecords-tr before the fix: 26 train file entries holding 274,015
records against 17 unique files holding 175,369 -- a 1.56x inflation that silently doubled
the weight of exactly nine corpora (casie, chfinann, cmnee, docee, docfee, duee,
events_biotech, maven, text2json) and of nothing else. No config declares that as intent,
`task_loss_weights` already exists for deliberate weighting, and the standing rule is that
aggregated splits are verified unique.

ONLY EXACT PATH REPEATS ARE DROPPED. `data/cmnee.val.jsonl` and
`data/scaling_joint/cmnee.val.jsonl` are different slices and both must survive -- that is
`event_files` pinning a smaller val, not a duplicate.
"""

import importlib.util
import sys


def _mod():
    sys.path.insert(0, "tools/train")
    spec = importlib.util.spec_from_file_location("tr_dedupe", "tools/train/train.py")
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except SystemExit:
        pass
    return m


def test_exact_repeat_is_dropped_once():
    m = _mod()
    out = m._dedupe_paths(["a.jsonl", "b.jsonl", "a.jsonl"], "train")
    assert out == ["a.jsonl", "b.jsonl"]


def test_order_is_preserved():
    m = _mod()
    out = m._dedupe_paths(["z.jsonl", "a.jsonl", "z.jsonl", "m.jsonl"], "train")
    assert out == ["z.jsonl", "a.jsonl", "m.jsonl"]


def test_different_slices_of_the_same_corpus_both_survive():
    """THE CASE THAT MUST NOT REGRESS: event_files pins a smaller val deliberately."""
    m = _mod()
    paths = ["data/cmnee.val.jsonl", "data/scaling_joint/cmnee.val.jsonl"]
    assert m._dedupe_paths(list(paths), "val") == paths


def test_no_duplicates_is_a_no_op():
    m = _mod()
    paths = ["a.jsonl", "b.jsonl", "c.jsonl"]
    assert m._dedupe_paths(list(paths), "test") == paths


def test_it_reports_what_it_dropped(capsys):
    """Silent deduplication would hide a config error; it must name the files."""
    m = _mod()
    m._dedupe_paths(["data/cmnee.train.jsonl", "data/cmnee.train.jsonl"], "train")
    out = capsys.readouterr().out
    assert "cmnee.train.jsonl" in out
    assert "twice per epoch" in out


def test_the_message_suits_a_non_train_split(capsys):
    m = _mod()
    m._dedupe_paths(["data/x.test.jsonl", "data/x.test.jsonl"], "test")
    out = capsys.readouterr().out
    assert "twice per epoch" not in out, "test records are not trained on"
    assert "twice in test" in out
