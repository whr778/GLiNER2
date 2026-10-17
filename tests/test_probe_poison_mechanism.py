"""probe_poison_mechanism.triggers: best trigger-span probability per requested event type."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "train"))
import probe_poison_mechanism as P


def test_triggers_reads_only_trigger_queries_of_the_requested_types():
    specs = [{"task_type": "events", "task_name": "Cyber.Ransom", "field_name": "trigger"},
             {"task_type": "events", "task_name": "Cyber.Ransom", "field_name": "Victim"},
             {"task_type": "events", "task_name": "Cyber.Phishing", "field_name": "trigger"},
             {"task_type": "entities", "task_name": "entities", "field_name": "trigger"}]
    probs = torch.tensor([[[0.1, 0.7], [0.9, 0.9], [0.2, 0.3], [0.99, 0.99]]])
    fwd = {"core": {"ext_specs": [specs]}, "probs": probs}
    out = P.triggers(fwd, ["Cyber.Ransom"])
    assert list(out) == ["Cyber.Ransom"] and abs(out["Cyber.Ransom"] - 0.7) < 1e-6
