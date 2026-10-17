"""menu_poison_check helpers: removing one label, the candidate list, trigger matching."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "train"))
import menu_poison_check as P  # noqa: E402

MENU = {"entities": ["Money", "Person"],
        "events": {"Cyber.Ransom": ["Damage_Amount", "Price", "Victim"], "Cyber.Phishing": ["Damage_Amount", "Victim"]}}


def test_without_a_role_removes_it_from_every_type():
    m = P.without(MENU, "role", "Damage_Amount")
    assert m["events"] == {"Cyber.Ransom": ["Price", "Victim"], "Cyber.Phishing": ["Victim"]}
    assert m["entities"] == MENU["entities"]


def test_without_a_type_and_an_entity():
    assert list(P.without(MENU, "type", "Cyber.Ransom")["events"]) == ["Cyber.Phishing"]
    assert P.without(MENU, "entity", "Money")["entities"] == ["Person"]
    assert P.without({"entities": {"Money": "", "Person": ""}}, "entity", "Money")["entities"] == {"Person": ""}


def test_candidates_roles_once_then_types_then_entities():
    assert P.candidates(MENU, entities=False) == [
        ("role", "Damage_Amount"), ("role", "Price"), ("role", "Victim"),
        ("type", "Cyber.Ransom"), ("type", "Cyber.Phishing")]
    assert P.candidates(MENU, entities=True)[-2:] == [("entity", "Money"), ("entity", "Person")]


def test_triggers_match_on_type_and_casefolded_text():
    gold = {"events": [{"event_type": "Cyber.Ransom", "triggers": ["Paid", "ransom"]},
                       {"event_type": "Cyber.Phishing", "trigger": "phished"}]}
    out = {"event_extraction": {"Cyber.Ransom": [{"triggers": [{"text": "paid", "start": 0, "end": 4}]}],
                                "Cyber.Phishing": [{"triggers": ["ransom"]}]}}
    assert P.pred_triggers(out) & P.gold_triggers(gold) == {("Cyber.Ransom", "paid")}
