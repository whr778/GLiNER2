"""A relations dose is proven per epoch: absent relations injected in forked workers reach the epoch line (relneg A/B)."""
import multiprocessing as mp

from gliner2.training.negatives import NegativeLabels

POOLS = {"wiki": {"annotates": {"relations": True}, "relations": ["Founded_By", "Part_Of", "Spouse", "Capital_Of"]}}
REC = {"relations": [{"Founded_By": {"head": "Acme", "tail": "Ann"}}]}


def neg(per_dim):
    return NegativeLabels(POOLS, per_dim, seed=1)


def test_relations_injected_in_a_forked_worker_reach_the_epoch_line():
    n = neg({"relations": 2})
    w = mp.get_context("fork").Process(target=n.inject, args=(REC, 0), kwargs={"corpus": "wiki"})
    w.start()
    w.join()
    assert n.stats["injected"] == 0, "the parent's plain stats never see a worker"
    assert "relations dose 2: 2 absent relations injected" in n.epoch_line()


def test_absent_relations_never_include_gold():
    out = neg({"relations": 3}).inject(REC, 0, corpus="wiki")
    assert sorted(out["absent_relations"]) == ["Capital_Of", "Part_Of", "Spouse"]


def test_no_relations_dose_leaves_the_epoch_line_unchanged():
    n = neg({"entities": 1})
    n.inject(REC, 0, corpus="wiki")
    assert "relations" not in n.epoch_line()


def test_set_epoch_resets_the_relations_count():
    n = neg({"relations": 2})
    n.inject(REC, 0, corpus="wiki")
    n.set_epoch(1)
    assert "relations dose 2: 0 absent relations injected" in n.epoch_line()
