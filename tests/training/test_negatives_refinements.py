"""MENU_SPEC s4 negatives refinements: gold-free dimensions, log-uniform K, the whole-record cap."""
import json
import sys
from pathlib import Path

from gliner2.training.negatives import NegativeLabels

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "train"))
import train as T  # noqa: E402

POOLS = {"news": {"annotates": {"entities": True, "events": True},
                  "entities": [f"E{i}" for i in range(40)],
                  "events": {f"T{i}": ["Agent"] for i in range(30)}},
         "part": {"annotates": {"entities": True, "events": True},
                  "entities": [f"E{i}" for i in range(40)], "events": {"T1": ["Agent"]}}}
WITH_GOLD = {"entities": {"E0": ["Ann"]}, "events": [{"event_type": "T0", "triggers": ["x"], "arguments": []}]}
EVENT_FREE = {"entities": {"E0": ["Ann"]}}


def neg(**kw):
    return NegativeLabels(POOLS, {"entities": 20, "events": 20}, seed=1, **kw)


def test_event_free_record_gets_event_absents_only_when_enabled():
    assert not neg().inject(EVENT_FREE, 0, corpus="news").get("absent_events")
    out = neg(gold_free_dims=True).inject(EVENT_FREE, 0, corpus="news")
    assert len(out["absent_events"]) == 20 and set(out["absent_events"]) <= set(POOLS["news"]["events"])
    assert all(v == ["Agent"] for v in out["absent_events"].values())


def test_gold_free_respects_partial_annotation():
    out = neg(gold_free_dims=True, partial={"part": ["events"]}).inject(EVENT_FREE, 0, corpus="part")
    assert not out.get("absent_events")


def test_gold_free_needs_a_known_corpus():
    assert not neg(gold_free_dims=True).inject(EVENT_FREE, 0, corpus=None).get("absent_events")


def test_loguniform_k_varies_within_1_to_k():
    n = neg(k_sampling="loguniform")
    ks = {len(n.inject(WITH_GOLD, i, corpus="news")["entities"]) - 1 for i in range(200)}
    assert min(ks) >= 1 and max(ks) <= 20 and len(ks) > 5


def test_max_per_record_caps_the_total():
    out = neg(max_per_record=12).inject(WITH_GOLD, 0, corpus="news")
    assert len(out["entities"]) - 1 + len(out.get("absent_events") or {}) == 12


def test_unknown_k_sampling_refused():
    import pytest
    with pytest.raises(ValueError):
        neg(k_sampling="uniform")


def test_processor_emits_absent_events_for_a_record_without_gold_events():
    from gliner2.processor import SchemaTransformer
    proc = SchemaTransformer.__new__(SchemaTransformer)
    proc.V_TOKEN = "[V]"
    proc._transform_schema = lambda name, fields, tok: (name, tuple(fields))
    schemas, labels, types = [], [], []
    proc._process_events({"absent_events": {"T1": ["Agent"]}}, schemas, labels, types, None)
    assert types == ["events"] and labels == [[0, []]] and schemas == [("T1", ("trigger", "Agent"))]
    schemas, labels, types = [], [], []
    proc._process_events({}, schemas, labels, types, None)
    assert types == []


def test_read_transformed_tags_the_corpus(tmp_path):
    f = tmp_path / "mycorp.train.jsonl"
    f.write_text(json.dumps({"input": "x", "output": {}}) + "\n")
    assert T.read_transformed([str(f)], {}, set())[0]["_corpus"] == "mycorp"


def test_validation_keeps_the_corpus_tag():
    """DataLoader_Factory.load(validate=True) rebuilt records via to_dict and dropped `_corpus`, so the
    injector saw corpus=None in every validated run and gold-free negatives never fired."""
    from gliner2.training.trainer import DataLoader_Factory
    rec = {"input": "Ann met Bob.", "output": {"entities": {"person": ["Ann"]}}, "_corpus": "news"}
    out = DataLoader_Factory.load(data=[rec], max_samples=-1, shuffle=False, seed=1, validate=True)
    assert out[0]["_corpus"] == "news"


def test_schedule_steps_the_dose_per_epoch_and_holds_the_last_value():
    n = NegativeLabels(POOLS, {"entities": 20}, seed=1, schedule={"entities": [1, 5, 20]})
    doses = []
    for e in range(5):
        n.set_epoch(e)
        doses.append(n.dose("entities"))
    assert doses == [1, 5, 20, 20, 20] and n.dose("events") == 0


def test_set_epoch_reaches_a_persistent_forked_worker():
    """The defect: persistent forked DataLoader workers never saw the parent's set_epoch."""
    import multiprocessing as mp
    from torch.utils.data import DataLoader

    class DS:
        def __init__(self):
            self.negatives = NegativeLabels(POOLS, {"entities": 20}, seed=1, schedule={"entities": [1, 5, 20]})

        def __len__(self):
            return 4

        def __getitem__(self, i):
            out = self.negatives.inject(WITH_GOLD, i, corpus="news")
            return len(out["entities"]) - 1

    if "fork" not in mp.get_all_start_methods():
        return
    ds = DS()
    dl = DataLoader(ds, batch_size=2, num_workers=2, persistent_workers=True, multiprocessing_context="fork",
                    collate_fn=lambda b: b)
    seen = []
    for e in range(3):
        ds.negatives.set_epoch(e)
        seen.append(sorted({k for b in dl for k in b}))
    assert seen == [[1], [5], [20]]
    assert "80 labels injected into 4 records" in ds.negatives.epoch_line()


def test_pickling_for_spawned_workers_keeps_plain_values():
    import pickle
    n = NegativeLabels(POOLS, {"entities": 20}, seed=1, schedule={"entities": [1, 5]})
    n.set_epoch(1)
    m = pickle.loads(pickle.dumps(n))
    assert m.epoch == 1 and m.dose("entities") == 5


def test_training_prompt_does_not_reveal_which_entity_labels_are_present():
    """Absent entity labels are appended after the gold; unshuffled order, or synthetic `entity i`
    names numbered in that order, told the model which labels were present (entity F1 0.000 vs 0.646
    on the same labels in a different order, 2026-10-08)."""
    import random
    from gliner2.processor import SamplingConfig, SchemaTransformer
    proc = SchemaTransformer.__new__(SchemaTransformer)
    proc.is_training, proc.E_TOKEN = True, "[E]"
    proc._transform_schema = lambda name, fields, tok, **kw: list(fields)
    gold = {"entities": {"G0": ["a"], "G1": ["b"]}}
    gold["entities"].update({f"A{i}": [] for i in range(8)})
    random.seed(0)
    first_is_gold, entity1_is_gold = [], []
    for synth in (0.0, 1.0):
        sc = SamplingConfig(synthetic_entity_label_prob=synth, remove_entity_prob=0.0)
        for _ in range(300):
            schema, schemas, labels, types = json.loads(json.dumps(gold)), [], [], []
            proc._process_entities(schema, schemas, labels, types, sc)
            (first_is_gold if not synth else entity1_is_gold).append(bool(labels[0][1][0][0]))
            if synth:
                entity1_is_gold[-1] = bool(schema["entities"]["entity 1"])
    assert 0.1 < sum(first_is_gold) / 300 < 0.35     # base rate 2/10, was 1.00
    assert 0.1 < sum(entity1_is_gold) / 300 < 0.35


def test_abstention_phase_splits_the_config():
    """Main phase: dose 1 into <out>/main; the phase: the configured dose, its epochs, into <out> (best/ ships)."""
    import pytest
    from gliner2.training.trainer import TrainingConfig
    cfg = TrainingConfig(output_dir="out/x", num_epochs=3, negative_labels_per_dim={"entities": 20, "events": 20})
    main, phase = T.abstention_configs(cfg, {"epochs": 1})
    assert (main.output_dir, main.num_epochs, main.negative_labels_per_dim) == ("out/x/main", 3, {"entities": 1, "events": 1})
    assert (phase.output_dir, phase.num_epochs, phase.negative_labels_per_dim) == ("out/x", 1, {"entities": 20, "events": 20})
    assert cfg.negative_labels_per_dim == {"entities": 20, "events": 20} and cfg.output_dir == "out/x"
    with pytest.raises(SystemExit):
        T.abstention_configs(TrainingConfig(output_dir="o", negative_labels_per_dim={"entities": 20},
                                            negative_labels_schedule={"entities": [1, 20]}), {"epochs": 1})
