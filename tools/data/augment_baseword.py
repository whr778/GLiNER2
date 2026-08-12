"""Duplicate a corpus with every surface word -- text AND labels -- reduced to its base form.

Both variants stay in the mix. The point is to stop `killed` / `killing` / `kills` each
learning the same trigger-role association under-powered, which matters most on the small
event corpora (RAMS 7,329 train, CASIE 795, WikiEvents 206). See TODO.md item 12.

**The whole difficulty is that labels must remain verbatim substrings of the text.** Boundary
collation locates each gold surface inside the input; a mention it cannot find is *silently
dropped* (`on_missing_surface="skip"`, counted in `missing_surface_counts()`), so a broken
augmentation does not raise -- it quietly shrinks supervision and reads as "augmentation did
not help".

Two rules keep that from happening, and both are load-bearing:

1. **Lemmatize the token sequence ONCE and re-derive every label from its token range.**
   Never lemmatize text and label strings in separate passes: lemmas are context-sensitive
   (`left` -> `leave` or `left`), so the passes diverge and the label stops matching. Because
   the rebuilt text and the rebuilt labels are joins of the *same* token list, verbatim-ness
   holds by construction rather than by luck.
2. **No lemma means keep the surface word.** Real lemmatizers return nothing for plenty of
   tokens; substituting a guess there is how alignment breaks.
3. **A label must occur exactly as often after lemmatization as before.** Verbatim-ness is
   not enough: lemmatization COLLAPSES surface forms, so a label can start matching
   positions that were never annotated. Measured on RAMS with simplemma -- gold `guns`
   occurs once in the source, and as `gun` it occurs three times in the lemmatized text,
   so collation builds three gold mentions where one was annotated. Across the corpus that
   inflated gold by 1,085 mentions on 31,773 (3.4%), in 718 of 6,680 augmented records.
   These are silently INVENTED positives, the mirror of the silent loss in rule 1, and no
   `missing_surface_counts()` check can see them.

A record is augmented only when EVERY label remaps cleanly onto token boundaries; otherwise
only the original is emitted, and the skip is counted. Partial augmentation would be the
silent-supervision-loss failure this file exists to avoid.

    uv run python tools/data/augment_baseword.py --in data/rams.train.jsonl \
        --out data/rams_aug.train.jsonl --backend simplemma --lang en
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from _split import dumps_record
from gliner2.processor import WhitespaceTokenSplitter

_SPLIT = WhitespaceTokenSplitter()

WORD = re.compile(r"\S+")
CORE = re.compile(r"^(\W*)(.*?)(\W*)$", re.DOTALL)


def tokens_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in WORD.finditer(text)]


def mock_lemma(word: str, lang: str) -> str:
    """Deterministic stand-in used to test ALIGNMENT, not lemma quality.

    Alignment is where the bugs are, and it can be exercised without pulling in a
    lemmatizer dependency -- which matters when a long job is already running out of
    the same virtualenv.
    """
    low = word.lower()
    for suffix, keep in (("ies", 3), ("ing", 3), ("ed", 2), ("s", 1)):
        if len(low) > keep + 2 and low.endswith(suffix):
            return low[: -keep]
    return low


def simplemma_lemma(word: str, lang: str) -> str:
    import simplemma

    return simplemma.lemmatize(word, lang=lang)


def lemmatize_token(token: str, lemma: Callable[[str, str], str], lang: str) -> str:
    """Lemmatize a token's core, preserving attached punctuation.

    Falls back to the surface form whenever the backend yields nothing -- the case that
    breaks naive implementations.
    """
    head, core, tail = CORE.match(token).groups()
    if not core:
        return token
    try:
        out = lemma(core, lang) or core
    except Exception:
        out = core
    return f"{head}{out}{tail}"


def remap(value: str, text: str, spans: List[Tuple[str, int, int]],
          lemmas: List[str], starts: List[int]) -> Optional[str]:
    """The lemmatized form of `value`, or None if it does not align to token boundaries."""
    at = text.find(value)
    if at < 0:
        return None
    covered = [i for i, (_, s, e) in enumerate(spans) if s >= at and e <= at + len(value)]
    if not covered:
        return None
    first, last = covered[0], covered[-1]
    if spans[first][1] != at or spans[last][2] != at + len(value):
        return None      # partial token: reattaching would not be verbatim
    return " ".join(lemmas[first:last + 1])


def token_count(text: str, needle: str) -> int:
    """Occurrences of `needle` in `text` under the tokenization COLLATION uses.

    It must be that tokenization and not a simpler one. `WhitespaceTokenSplitter` is a
    regex tokenizer: it splits trailing punctuation into its own token and lower-cases.
    Counting with `str.split()` instead misses matches the collator makes -- measured, gold
    `they` counted once by whitespace and twice by the collator, because the second
    occurrence was `they,`. Guarding on the wrong tokenization passes records that then
    gain phantom gold.
    """
    tokens = [t for t, _, _ in _SPLIT(text)]
    want = [t for t, _, _ in _SPLIT(needle)]
    if not want:
        return 0
    n = len(want)
    return sum(1 for i in range(len(tokens) - n + 1) if tokens[i:i + n] == want)


def walk_labels(output: dict):
    """Yield (container, key, value) for every surface string the schema declares."""
    ents = output.get("entities")
    for block in ([ents] if isinstance(ents, dict) else ents or []):
        if isinstance(block, dict):
            for _, values in block.items():
                for i, v in enumerate(values if isinstance(values, list) else [values]):
                    if isinstance(v, str) and v.strip():
                        yield values, i, v
    for block in output.get("json_structures") or []:
        for _, fields in (block or {}).items():
            for name, v in list(fields.items()):
                if isinstance(v, str) and v.strip():
                    yield fields, name, v
                elif isinstance(v, list):
                    for i, x in enumerate(v):
                        if isinstance(x, str) and x.strip():
                            yield v, i, x
    for ev in output.get("events") or []:
        for i, t in enumerate(ev.get("triggers") or []):
            if isinstance(t, str) and t.strip():
                yield ev["triggers"], i, t
        for arg in ev.get("arguments") or []:
            if isinstance(arg.get("entity"), str) and arg["entity"].strip():
                yield arg, "entity", arg["entity"]


def augment(record: dict, lemma: Callable[[str, str], str], lang: str) -> Optional[dict]:
    """A base-word copy of `record`, or None when any label fails to remap."""
    text = record.get("input", "")
    spans = tokens_with_offsets(text)
    if not spans:
        return None
    lemmas = [lemmatize_token(tok, lemma, lang) for tok, _, _ in spans]
    starts = [s for _, s, _ in spans]

    lemma_text = " ".join(lemmas)
    out = json.loads(json.dumps(record["output"]))
    for container, key, value in walk_labels(out):
        mapped = remap(value, text, spans, lemmas, starts)
        if mapped is None:
            return None
        if token_count(lemma_text, mapped) != token_count(text, value):
            return None      # lemmatization merged forms: gold would gain phantom positions
        container[key] = mapped
    return {"input": lemma_text, "output": out}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", choices=("mock", "simplemma"), default="mock",
                    help="mock is a deterministic stand-in for testing ALIGNMENT without a "
                         "lemmatizer dependency; simplemma is the real one.")
    ap.add_argument("--lang", default="en",
                    help="Lemmatization is a no-op for Chinese and a different operation for "
                         "agglutinative languages -- opt in per corpus, not across data/.")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    lemma = simplemma_lemma if args.backend == "simplemma" else mock_lemma
    n = kept = skipped = 0
    with Path(args.src).open(encoding="utf-8") as fh, \
            Path(args.out).open("w", encoding="utf-8") as out:
        for line in fh:
            if args.limit and n >= args.limit:
                break
            record = json.loads(line)
            n += 1
            out.write(dumps_record(record) + "\n")     # original always survives
            extra = augment(record, lemma, args.lang)
            if extra is None:
                skipped += 1
                continue
            out.write(dumps_record(extra) + "\n")
            kept += 1

    print(f"{n} records read -> {n + kept} written ({kept} augmented, {skipped} not)")
    print(f"augmentation rate {kept / max(n, 1):.1%}"
          + ("  <- labels are not landing on token boundaries, or lemmatization merged a "
             "label with other text (both are refusals, not losses)"
             if kept / max(n, 1) < 0.75 else ""))


if __name__ == "__main__":
    main()
