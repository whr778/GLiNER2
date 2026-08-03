"""Prompt construction for synthetic generation.

Builds the system + user messages that ask the model to write ONE document and
annotate it for the requested task types in a single pass -- the same
"real/synthetic text with model annotations" recipe the GLiNER2 paper used.

The model is asked for a flat, easy-to-produce JSON shape (lists of
``{type, text}`` etc.); ``validate.py`` transforms and verbatim-checks it into
the strict GLiNER2 training record. Keeping the two shapes separate makes the
generation prompt simpler and the validation stricter.
"""

from __future__ import annotations

import json
from typing import Dict, List

from schema_spec import (
    CLASSIFICATION_TASKS, ENTITY_TYPES, EVENT_ONTOLOGY, MULTI_LABEL_TASKS,
    RELATION_TYPES, STRUCTURE_TEMPLATES,
)

_VERBATIM = (
    "CRITICAL: every annotated span (entity text, event trigger, argument entity, "
    "relation head/tail, extractive structure value) must be copied VERBATIM as an "
    "exact substring of the document text -- same casing, same punctuation, no "
    "paraphrase. "
    "Reply with ONE strictly valid, parseable JSON object (RFC 8259) and NOTHING "
    "else: no markdown code fences, no comments, no trailing commas. Inside every "
    "string value, escape all double quotes as \\\", backslashes as \\\\, and "
    "newlines as \\n so the entire reply parses with a strict JSON parser. Before "
    "responding, verify the JSON is complete and valid."
)

SYSTEM = (
    "You generate synthetic training data for an information-extraction model. "
    "You write one realistic document, then annotate it. " + _VERBATIM
)

ANNOTATE_SYSTEM = (
    "You annotate documents for an information-extraction model. You are given a "
    "document and must label it -- never rewrite or invent content. " + _VERBATIM
)


def _event_ontology_lines() -> str:
    return "\n".join(
        f"  - {etype}: roles = {', '.join(roles)}"
        for etype, roles in EVENT_ONTOLOGY.items()
    )


def _structure_lines() -> str:
    out = []
    for name, fields in STRUCTURE_TEMPLATES.items():
        parts = []
        for f, choices in fields.items():
            parts.append(f"{f} (one of: {', '.join(choices)})" if choices else f)
        out.append(f"  - {name}: fields = {', '.join(parts)}")
    return "\n".join(out)


def _classification_lines() -> str:
    out = []
    for task, labels in CLASSIFICATION_TASKS.items():
        multi = " (multi-label: pick one or more)" if task in MULTI_LABEL_TASKS else " (single-label)"
        out.append(f"  - {task}{multi}: labels = {', '.join(labels)}")
    return "\n".join(out)


def _task_instructions(tasks: List[str]) -> List[str]:
    """The per-task label sets + output-key instructions (shared by both modes)."""
    sections: List[str] = ["Output JSON keys (include a key only for the tasks listed):"]
    if "entities" in tasks:
        sections += [
            'entities: list of {"type","text"}. Entity types:',
            "  " + ", ".join(ENTITY_TYPES),
        ]
    if "relations" in tasks:
        sections += [
            'relations: list of {"type","head","tail"} where head and tail are '
            "entity surfaces from the text. Relation types:",
            "  " + ", ".join(RELATION_TYPES),
        ]
    if "events" in tasks:
        sections += [
            'events: list of {"event_type","trigger","arguments"} where arguments '
            'is a list of {"role","entity"}. The trigger is the single word/phrase '
            "in the text that most directly evokes the event. Event types and their "
            "allowed roles:",
            _event_ontology_lines(),
        ]
    if "classifications" in tasks:
        sections += [
            'classifications: list of {"task","labels"} where labels is the chosen '
            "true label(s) for that task. Classification tasks:",
            _classification_lines(),
        ]
    if "structures" in tasks:
        sections += [
            'structures: list of {"type","fields"} where fields is an object of '
            "field->value. Include a structure only if the document actually "
            "describes one. Extractive field values must be verbatim substrings; "
            "choice fields must be one of the listed options. Structure templates:",
            _structure_lines(),
        ]
    return sections


def build_user_prompt(domain: str, tasks: List[str], min_words: int, max_words: int) -> str:
    """Assemble the per-record user prompt for one GENERATED document in ``domain``."""
    sections: List[str] = [
        f"Write a {domain} of roughly {min_words}-{max_words} words. Make it "
        "concrete and specific: real-sounding names, organizations, places, dates, "
        "and numbers. Then annotate it. Use ONLY the label sets below.",
        "",
    ]
    sections += _task_instructions(tasks)
    sections += [
        "",
        'Also include a "text" key with the document itself. Omit any annotation '
        "whose span does not appear verbatim in the text. Return only the JSON object.",
    ]
    return "\n".join(sections)


def build_annotate_prompt(text: str, tasks: List[str]) -> str:
    """Assemble the user prompt to annotate EXISTING ``text`` (no text generation)."""
    sections: List[str] = [
        "Annotate the DOCUMENT below for the tasks listed. Do NOT rewrite, "
        "summarize, or invent text -- annotate only what the document actually "
        "states. Use ONLY the label sets below.",
        "",
    ]
    sections += _task_instructions(tasks)
    sections += [
        "",
        "Do NOT include a \"text\" key. Every annotated span must be copied "
        "verbatim from the document. Omit anything not present verbatim, and omit "
        "a task entirely if the document supports none of its labels. Return only "
        "the JSON object.",
        "",
        "DOCUMENT:",
        text,
    ]
    return "\n".join(sections)


def example_shape() -> str:
    """A compact reference of the expected JSON (for the README / dry run)."""
    return json.dumps({
        "text": "<the document>",
        "entities": [{"type": "person", "text": "Jane Doe"}],
        "relations": [{"type": "works_for", "head": "Jane Doe", "tail": "Acme Corp"}],
        "events": [{
            "event_type": "Transaction.TransferOwnership",
            "trigger": "acquired",
            "arguments": [{"role": "Buyer", "entity": "Acme Corp"}],
        }],
        "classifications": [{"task": "topic", "labels": ["business"]}],
        "structures": [{"type": "product", "fields": {"name": "Widget", "condition": "new"}}],
    }, indent=2)
