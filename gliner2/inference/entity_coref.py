"""Entity coreference at inference: cluster a document's mentions with the trained mention link.

Mentions are character spans ``{text, start, end}``. They are placed on the model's words and encoded
exactly as training and the readout do (``coref.attach`` -> ``collate_fn_inference`` -> ``_encode_core``
-> mean word states, tools/train/coref_readout.py), so the link scores what it was trained on. Clustering
is BEST ANTECEDENT: left to right, each mention joins the cluster of its highest-scoring earlier
mention when that logit clears the threshold. Joining every pair over the threshold would let one
ambiguous pronoun linked to two people merge them; best antecedent links each mention once.
Mentions the encoder window cannot hold are returned as singletons and counted.
"""
from __future__ import annotations

import re
from collections.abc import Sequence

import torch

from gliner2.models.boundary.mention_link import mention_states
from gliner2.processing import coref

PRONOUNS = {"he", "she", "they", "him", "her", "them", "his", "their", "hers", "theirs", "its", "it", "we",
            "us", "our", "i", "me", "my", "you", "your", "who", "whom", "whose", "himself", "herself",
            "themselves", "itself"}


def candidate_mentions(text: str, entities: dict[str, list]) -> list[dict]:
    """Predicted entity spans plus pronouns (lowercase or Capitalized only), non-overlapping, by offset;
    an entity span wins over a pronoun inside it."""
    spans = sorted({(sp["start"], sp["end"]) for v in (entities or {}).values() for sp in v
                    if isinstance(sp, dict) and "start" in sp}, key=lambda s: (s[0], -s[1]))
    taken: list[tuple] = []
    for s, e in spans:
        if not any(a < e and s < b for a, b in taken):
            taken.append((s, e))
    forms = PRONOUNS | {p.capitalize() for p in PRONOUNS}
    for m in re.finditer(r"(?<!\w)(" + "|".join(sorted(forms, key=len, reverse=True)) + r")(?!\w)", text):
        if not any(a < m.end() and m.start() < b for a, b in taken):
            taken.append((m.start(), m.end()))
    return [{"text": text[s:e], "start": s, "end": e} for s, e in sorted(taken)]


@torch.no_grad()
def link_logits(model, text: str, mentions: Sequence[dict]) -> tuple:
    """([P, P] link logits, indices into ``mentions`` of the P placed ones)."""
    batch = model.processor.collate_fn_inference(
        [(text, coref.attach({}, [[m] for m in mentions]))], architecture="boundary",
        max_len=None, event_records=True, ignore_missing_entities=True)
    placed = batch.coref_mentions[0] if batch.coref_mentions else []
    if len(placed) < 2:
        return torch.zeros(len(placed), len(placed)), [c for _, _, c in placed]
    core = model._encode_core(batch)
    states, mid = mention_states(core["text_states"][0], placed)
    return model.mention_link(states, mid).float().cpu(), [c for _, _, c in placed]


def best_antecedent(logits: torch.Tensor, threshold: float) -> list[int]:
    """Cluster id per row: each mention joins its best earlier mention when that logit >= threshold."""
    label = list(range(logits.shape[0]))
    for j in range(1, logits.shape[0]):
        i = int(logits[j, :j].argmax())
        if float(logits[j, i]) >= threshold:
            label[j] = label[i]
    return label


def resolve(model, text: str, mentions: Sequence[dict], threshold: float) -> list[list[dict]]:
    """Clusters of ``mentions`` (singletons included), in document order."""
    logits, idx = link_logits(model, text, mentions)
    label = {m: m for m in range(len(mentions))}
    for row, lab in zip(idx, best_antecedent(logits, threshold)):
        label[row] = idx[lab]
    groups: dict[int, list[dict]] = {}
    for m in range(len(mentions)):
        groups.setdefault(label[m], []).append(dict(mentions[m]))
    return list(groups.values())
