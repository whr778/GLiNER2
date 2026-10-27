"""argument_links: every argument gets the id of the entity on its exact span, or a derived one."""
from gliner2.inference.argument_links import link_arguments
from gliner2.training import eval_metrics as M


def _sp(text, start):
    return {"text": text, "start": start, "end": start + len(text)}


def _pred():
    return {"entities": {"Person": [_sp("Banco", 0), _sp("guns", 20)], "WEA": [_sp("guns", 20)]},
            "event_extraction": {"Attack": [{"triggers": [_sp("shot", 10)],
                                             "arguments": [{"role": "Attacker", "entity": _sp("Banco", 0)},
                                                           {"role": "Place", "entity": _sp("Lyon", 30)}]}],
                                 "Die": [{"triggers": [_sp("died", 40)],
                                          "arguments": [{"role": "Place", "entity": _sp("Lyon", 30)}]}]}}


def test_argument_takes_the_id_of_the_entity_on_its_span():
    p = link_arguments(_pred())
    banco = p["entities"]["Person"][0]["id"]
    assert p["event_extraction"]["Attack"][0]["arguments"][0]["entity_id"] == banco


def test_one_id_per_span_across_labels():
    p = link_arguments(_pred())
    assert p["entities"]["Person"][1]["id"] == p["entities"]["WEA"][0]["id"]
    assert p["entities"]["Person"][0]["id"] != p["entities"]["Person"][1]["id"]


def test_argument_without_entity_gets_one_typed_derived_entity():
    p = link_arguments(_pred(), {"Attack": {"Place": ["GPE"]}, "Die": {"Place": ["GPE", "Location"]}})
    (d,) = p["derived_entities"]
    assert (d["text"], d["types"]) == ("Lyon", ["GPE", "Location"])
    assert d["roles"] == [["Attack", "Place"], ["Die", "Place"]]
    assert p["event_extraction"]["Die"][0]["arguments"][0]["entity_id"] == d["id"]
    assert all(d["id"] != sp["id"] for v in p["entities"].values() for sp in v)


def test_a_float_confidence_argument_is_skipped_instead_of_raising():
    p = _pred()
    p["event_extraction"]["Attack"][0]["arguments"].append({"role": "confidence", "entity": 0.81})
    linked = link_arguments(p)
    roles = [a["role"] for a in linked["event_extraction"]["Attack"][0]["arguments"]]
    assert roles == ["Attacker", "Place", "confidence"]
    assert "entity_id" not in linked["event_extraction"]["Attack"][0]["arguments"][-1]
    assert linked["event_extraction"]["Attack"][0]["arguments"][0]["entity_id"]


def test_unmapped_role_derives_untyped():
    assert link_arguments(_pred())["derived_entities"][0]["types"] == []


def test_scorer_sets_unchanged():
    p = _pred()
    before = (M._pred_entity_set(p), M._pred_event_argument_set(p))
    link_arguments(p)
    assert (M._pred_entity_set(p), M._pred_event_argument_set(p)) == before
