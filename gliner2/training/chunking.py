"""Sliding-window chunking for long training records.

When :class:`gliner2.training.trainer.TrainingConfig.sliding_window` is on, the
trainer expands every record whose tokenized length exceeds the window into a
sequence of overlapping chunks **before** the DataLoader sees the data. Each
chunk inherits a filtered copy of the gold annotations so the model only
trains on supervision that actually appears inside the chunk.

Per-task filtering rules:

* **Entities** — keep mention only in chunks where its surface verbatim
  appears (same rule the rest of the pipeline uses).
* **Entity descriptions** — keep descriptions only for entity types that
  survived the entity filter in this chunk.
* **Classifications** — doc-level labels are inherited by every chunk
  (each chunk becomes its own "given this fragment, the parent doc is X"
  training example).
* **Relations** — emit only when **both** head and tail appear in the
  same chunk; drop split-across-chunks relations.
* **Events** — emit only when the trigger appears in the chunk; per-event
  arguments are independently filtered (so the same event may surface
  with fewer arguments in some chunks).
* **JSON structures** — passed through unchanged; the processor's
  existing verbatim filter handles missing-field cases at collation time.

Window / stride are measured in **subword tokens** from the encoder's
tokenizer. Chunk character boundaries are snapped left/right to whitespace
so word tokens aren't broken across chunks. Records that fit within
``window_size`` subwords are emitted unchanged (no chunking).

With ``stride < window_size``, annotations whose surfaces fall in the
overlap region naturally repeat across multiple adjacent chunks — that's
intentional, it just gives the model slightly more supervision on those
spans.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from tqdm.auto import tqdm


def chunk_text_by_subwords(
    text: str,
    tokenizer,
    window_size: int,
    stride: int,
) -> List[Tuple[int, int, str]]:
    """Split ``text`` into overlapping subword-token chunks.

    Args:
        text: The input string.
        tokenizer: A HuggingFace fast tokenizer (must support
            ``return_offsets_mapping=True``).
        window_size: Window length in subword tokens.
        stride: Step in subword tokens between consecutive chunks.

    Returns:
        A list of ``(char_start, char_end, chunk_text)`` tuples. If the
        whole text fits in ``window_size`` subwords, returns one item with
        the full text and ``(0, len(text), text)`` spans. Boundaries are
        snapped to whitespace so chunks never end mid-word.
    """
    if window_size <= 0 or stride <= 0:
        raise ValueError(
            f"window_size and stride must be positive, got {window_size}/{stride}"
        )

    # verbose=False suppresses the "Token indices sequence length is longer
    # than the specified maximum sequence length for this model" warning.
    # We only use the offset mapping to compute chunk boundaries — the long
    # sequence is never fed through the encoder, so the warning is harmless.
    enc = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        verbose=False,
    )
    offsets = enc["offset_mapping"]
    n = len(offsets)

    if n <= window_size:
        return [(0, len(text), text)]

    chunks: List[Tuple[int, int, str]] = []
    start = 0
    n_text = len(text)
    while True:
        end = min(start + window_size, n)
        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]
        # Snap to whitespace boundaries on both sides so chunks always
        # start and end on a word boundary.
        while char_start > 0 and not text[char_start - 1].isspace():
            char_start -= 1
        while char_end < n_text and not text[char_end].isspace():
            char_end += 1
        chunks.append((char_start, char_end, text[char_start:char_end]))
        if end == n:
            break
        start += stride
    return chunks


# ---------------------------------------------------------------------------
# Per-task annotation filters
# ---------------------------------------------------------------------------

def _filter_entities(
    entities: Dict[str, List[str]],
    chunk_text: str,
) -> Dict[str, List[str]]:
    kept: Dict[str, List[str]] = {}
    for etype, surfaces in (entities or {}).items():
        if not isinstance(etype, str) or not isinstance(surfaces, list):
            continue
        in_chunk = [s for s in surfaces if isinstance(s, str) and s in chunk_text]
        if in_chunk:
            kept[etype] = in_chunk
    return kept


def _filter_relations(
    relations: List[Dict[str, Any]],
    chunk_text: str,
) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    for rel in relations or []:
        if not isinstance(rel, dict) or not rel:
            continue
        # Each rel dict is {name: {head, tail, ...}}.
        all_pass = True
        for _, fields in rel.items():
            if not isinstance(fields, dict):
                all_pass = False
                break
            head = fields.get("head")
            tail = fields.get("tail")
            if not (isinstance(head, str) and head and head in chunk_text):
                all_pass = False
                break
            if not (isinstance(tail, str) and tail and tail in chunk_text):
                all_pass = False
                break
        if all_pass:
            kept.append(rel)
    return kept


def _filter_events(
    events: List[Dict[str, Any]],
    chunk_text: str,
) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    for ev in events or []:
        if not isinstance(ev, dict):
            continue
        trigger = ev.get("trigger", "")
        if not (isinstance(trigger, str) and trigger and trigger in chunk_text):
            continue
        kept_args: List[Dict[str, Any]] = []
        for arg in ev.get("arguments") or []:
            if not isinstance(arg, dict):
                continue
            entity = arg.get("entity")
            if isinstance(entity, str) and entity and entity in chunk_text:
                kept_args.append(arg)
        kept.append({
            "event_type": ev.get("event_type"),
            "trigger": trigger,
            "arguments": kept_args,
        })
    return kept


def _filter_record_output(
    output: Dict[str, Any],
    chunk_text: str,
) -> Dict[str, Any]:
    """Return a new ``output`` dict filtered to annotations present in ``chunk_text``."""
    new_out: Dict[str, Any] = {}

    ent = _filter_entities(output.get("entities") or {}, chunk_text)
    if ent:
        new_out["entities"] = ent

    descs = output.get("entity_descriptions") or {}
    if "entities" in new_out and isinstance(descs, dict):
        kept_descs = {t: d for t, d in descs.items() if t in new_out["entities"]}
        if kept_descs:
            new_out["entity_descriptions"] = kept_descs

    # Classifications: inherited unchanged on every chunk.
    if "classifications" in output:
        new_out["classifications"] = output["classifications"]

    rels = _filter_relations(output.get("relations") or [], chunk_text)
    if rels:
        new_out["relations"] = rels

    evs = _filter_events(output.get("events") or [], chunk_text)
    if evs:
        new_out["events"] = evs

    # JSON structures pass through; the processor verbatim-filters fields.
    if output.get("json_structures"):
        new_out["json_structures"] = output["json_structures"]
    if output.get("json_descriptions"):
        new_out["json_descriptions"] = output["json_descriptions"]

    return new_out


def _record_has_content(out: Dict[str, Any]) -> bool:
    return any(k in out for k in (
        "entities", "relations", "events", "classifications", "json_structures",
    ))


def chunk_records(
    records: List[Dict[str, Any]],
    tokenizer,
    window_size: int,
    stride: int,
    desc: Optional[str] = "chunking",
    show_progress: bool = True,
) -> List[Dict[str, Any]]:
    """Expand ``records`` into overlapping subword-window chunks.

    Records that fit within ``window_size`` subwords are passed through
    unchanged. Chunks with no usable supervision after filtering are
    dropped.

    A tqdm progress bar is shown by default (set ``show_progress=False``
    to silence it); the bar's ``postfix`` reports running counts of
    pass-through records, records actually chunked, and chunks emitted
    so the user can see the windowing pass making progress on long
    corpora.
    """
    expanded: List[Dict[str, Any]] = []
    n_chunked = 0
    n_passthrough = 0
    iterator = tqdm(
        records,
        desc=desc,
        disable=not show_progress,
        unit="rec",
    )
    for record in iterator:
        text = record.get("input")
        if text is None:
            text = record.get("text")
        if not isinstance(text, str) or not text:
            expanded.append(record)
            n_passthrough += 1
            continue
        chunks = chunk_text_by_subwords(text, tokenizer, window_size, stride)
        if len(chunks) == 1 and chunks[0][2] == text:
            expanded.append(record)
            n_passthrough += 1
            continue
        output = record.get("output")
        if output is None:
            # Inference-shape records carry their schema under "schema";
            # pass them through unchunked (training never sees them).
            expanded.append(record)
            n_passthrough += 1
            continue
        n_chunked += 1
        for _, _, chunk_text in chunks:
            new_out = _filter_record_output(output, chunk_text)
            if not _record_has_content(new_out):
                continue
            expanded.append({"input": chunk_text, "output": new_out})
        if show_progress:
            iterator.set_postfix(
                chunked=n_chunked,
                passthrough=n_passthrough,
                out=len(expanded),
            )
    return expanded
