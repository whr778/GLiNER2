"""negative_gold_full_roles: gold event types list their full pool role list, as absent ones do (2026-10-17)."""
from gliner2.training.negatives import NegativeLabels

POOLS = {"cyber": {"annotates": {"entities": True, "events": True}, "entities": ["Org", "Money"],
                   "events": {"Ransom": ["Attacker", "Damage_Amount", "Price"], "Phishing": ["Attacker", "Victim"]}}}
REC = {"entities": {"Org": ["Acme"]},
       "events": [{"event_type": "Ransom", "triggers": ["demanded"], "arguments": [{"role": "Price", "entity": "$5"}]}]}


def neg(**kw):
    return NegativeLabels(POOLS, {"entities": 1, "events": 1}, seed=1, **kw)


def _segments(schema):
    from gliner2.processor import SchemaTransformer
    proc = SchemaTransformer.__new__(SchemaTransformer)
    proc.V_TOKEN = "[V]"
    proc._transform_schema = lambda name, fields, tok: (name, tuple(fields))
    schemas, labels, types = [], [], []
    proc._process_events(schema, schemas, labels, types, None)
    return dict(zip(schemas, labels))


def test_off_is_the_historical_schema():
    assert "event_roles" not in neg().inject(REC, 0, corpus="cyber")


def test_on_attaches_the_full_pool_list_for_gold_types_only():
    n = neg(gold_full_roles=True)
    out = n.inject(REC, 0, corpus="cyber")
    assert out["event_roles"] == {"Ransom": ["Attacker", "Damage_Amount", "Price"]}
    assert n.stats["gold_full_role_types"] == 1 and n.stats["gold_full_roles_added"] == 2


def test_on_respects_partial_annotation_of_events():
    assert "event_roles" not in neg(gold_full_roles=True, partial={"cyber": ["events"]}).inject(REC, 0, corpus="cyber")


def test_processor_lists_unargued_roles_with_empty_targets():
    seg = _segments({**REC, "event_roles": {"Ransom": ["Attacker", "Damage_Amount", "Price"]}})
    assert seg == {("Ransom", ("trigger", "Price", "Attacker", "Damage_Amount")): [1, [[["demanded"], ["$5"], [], []]]]}


def test_processor_without_event_roles_lists_argued_roles_only():
    assert list(_segments(REC)) == [("Ransom", ("trigger", "Price"))]


def test_epoch_line_counts_across_forked_workers():
    """The count rides a shared value: a forked dataloader worker's increments reach the parent's epoch line."""
    import multiprocessing as mp
    n = neg(gold_full_roles=True)
    ctx = mp.get_context("fork")
    w = ctx.Process(target=n.inject, args=(REC, 0), kwargs={"corpus": "cyber"})
    w.start()
    w.join()
    assert n.stats["gold_full_roles_added"] == 0, "the parent's plain stats never see a worker"
    assert "gold full roles: 2 unargued roles listed" in n.epoch_line()
    assert "gold full roles" not in neg().epoch_line()
