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


def _iter_event_argument_spans(events: dict[str, list] | None) -> list[tuple[int, int]]:
    if not isinstance(events, dict):
        return []
    found: set[tuple[int, int]] = set()
    for mentions in events.values():
        if not isinstance(mentions, list):
            continue
        for ev in mentions:
            if not isinstance(ev, dict):
                continue
            for arg in ev.get("arguments") or ():
                if not isinstance(arg, dict):
                    continue
                ent = arg.get("entity")
                if (
                    isinstance(ent, dict)
                    and isinstance(ent.get("start"), int)
                    and not isinstance(ent.get("start"), bool)
                    and isinstance(ent.get("end"), int)
                    and not isinstance(ent.get("end"), bool)
                    and ent["start"] < ent["end"]
                ):
                    found.add((ent["start"], ent["end"]))
    return sorted(found, key=lambda s: (s[0], -s[1]))


def candidate_mentions(
    text: str,
    entities: dict[str, list] | list[dict[str, list]] | None = None,
    events: dict[str, list] | None = None,
) -> list[dict]:
    """Predicted entity spans, then non-overlapping event argument spans, then pronouns, by offset.

    An entity span wins over an overlapping event argument span, and both win over a pronoun inside them.
    """
    if isinstance(entities, list):
        entities = entities[0] if entities and isinstance(entities[0], dict) else {}
    spans = sorted(
        {(sp["start"], sp["end"]) for v in (entities or {}).values() if isinstance(v, list)
         for sp in v if isinstance(sp, dict) and isinstance(sp.get("start"), int)
         and not isinstance(sp.get("start"), bool) and isinstance(sp.get("end"), int)
         and not isinstance(sp.get("end"), bool) and sp["start"] < sp["end"]},
        key=lambda s: (s[0], -s[1]),
    )
    taken: list[tuple[int, int]] = []
    for s, e in spans:
        if not any(a < e and s < b for a, b in taken):
            taken.append((s, e))
    for s, e in _iter_event_argument_spans(events):
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


@torch.no_grad()
def resolve_from_states(
    mention_link,
    doc_text_states: torch.Tensor,
    starts: Sequence[int],
    ends: Sequence[int],
    mentions: Sequence[dict],
    threshold: float,
) -> list[list[dict]]:
    """Cluster ``mentions`` directly from already-encoded document word states ``[W, H]``.

    Reuses the forward pass's ``core["text_states"][b, offset : offset + n_words]`` so
    in-engine entity coreference runs without re-tokenizing or re-encoding the document.
    """
    n_words = min(int(doc_text_states.shape[0]), len(starts), len(ends))
    placed = coref.word_mentions([[m] for m in mentions], starts, ends, n_words) if n_words > 0 else []
    label = {m: m for m in range(len(mentions))}
    if len(placed) >= 2 and mention_link is not None:
        states, mid = mention_states(doc_text_states[:n_words], placed)
        logits = mention_link(states, mid).float().cpu()
        idx = [c for _, _, c in placed]
        for row, lab in zip(idx, best_antecedent(logits, threshold)):
            label[row] = idx[lab]
    groups: dict[int, list[dict]] = {}
    for m in range(len(mentions)):
        groups.setdefault(label[m], []).append(dict(mentions[m]))
    return list(groups.values())
