"""Trace absent-event negatives exactly as training injects them, and test whether "absent" is true.

Reproduces ExtractorDataset.__getitem__ -> NegativeLabels.inject on a config's real training records
(real build_pools, seed, partial_annotation), then flags an injected absent event type whose
known trigger surface occurs in the document: an UPPER bound, and a TIGHTENED rate counting only
unambiguous surfaces (>=80% of the surface's trigger uses are that type, >=5 uses). Both are
detector rates, not contradiction rates: read the printed sentences, or adjudicate a random
sample with tools/data/adjudicate_absent_events.py.

    uv run python tools/data/trace_absent_event_negatives.py [--config <yaml>]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/train"))
import train as T  # noqa: E402
import yaml  # noqa: E402
from gliner2.training.negatives import NegativeLabels  # noqa: E402

CJK = re.compile(r"[㐀-鿿]")


def load(config: str):
    """(records by event corpus, injector, trigger lexicon) for a training config."""
    cfg = yaml.safe_load(open(config))
    fns = T._category_fns(T.load_labels_cfg(cfg, config))
    spec = importlib.util.spec_from_file_location("bnp", ROOT / "tools/data/build_negative_pools.py")
    bnp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bnp)
    pools = bnp.build_pools(cfg, Path(config))
    inj = NegativeLabels(pools, cfg["training"]["negative_labels_per_dim"], seed=42,
                         partial=cfg["data"].get("partial_annotation"))
    paths = bnp._corpus_train_paths(cfg)
    recs, lex = {}, defaultdict(Counter)
    for n, p in paths.items():
        if not (pools.get(n) or {}).get("annotates", {}).get("events"):
            continue
        recs[n] = [T.transform_record(json.loads(l), fns) for l in open(p, encoding="utf-8") if l.strip()]
        for r in recs[n]:
            for e in r["output"].get("events") or []:
                for t in e.get("triggers") or []:
                    lex[e["event_type"]][t.lower()] += 1
    return recs, inj, lex


def present(text: str, surf: str) -> bool:
    if CJK.search(surf):
        return len(surf) >= 2 and surf in text
    return len(surf) >= 3 and re.search(rf"(?<!\w){re.escape(surf)}(?!\w)", text.lower()) is not None


def unambiguous_fn(lex):
    by_surface = defaultdict(Counter)
    for etype, c in lex.items():
        for s, k in c.items():
            by_surface[s][etype] += k

    def unambiguous(etype: str, s: str) -> bool:
        tot = sum(by_surface[s].values())
        return tot >= 5 and by_surface[s][etype] / tot >= 0.8
    return unambiguous


def injections(corpus: str, recs, inj, n_docs: int = 1500, seed: int = 0):
    """Yield (record, absent event type) exactly as training would inject them."""
    sample = random.Random(seed).sample(recs[corpus], min(n_docs, len(recs[corpus])))
    for i, r in enumerate(sample):
        for etype in inj.inject(r["output"], i, corpus).get("absent_events") or {}:
            yield r, etype


def flags(r, etype, lex, unambiguous):
    """(any known trigger in text, any UNAMBIGUOUS known trigger in text) for one injection."""
    gold = {t.lower() for e in r["output"].get("events") or [] for t in e.get("triggers") or []}
    loose = [s for s, k in lex[etype].most_common() if k >= 3 and s not in gold and present(r["input"], s)]
    tight = [s for s in lex[etype] if s not in gold and unambiguous(etype, s) and present(r["input"], s)]
    return loose, tight


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="tools/train/config/base/eb18-balanced.yaml")
    args = ap.parse_args()
    recs, inj, lex = load(args.config)
    unambiguous = unambiguous_fn(lex)
    print(f"[trace] corpora with event pools: {list(recs)}")
    for n in recs:
        c, ex = Counter(), []
        for r, etype in injections(n, recs, inj):
            c["injected"] += 1
            loose, tight = flags(r, etype, lex, unambiguous)
            c["loose"] += bool(loose)
            c["tight"] += bool(tight)
            if tight and len(ex) < 3:
                ex.append((etype, tight[0]))
        k = max(c["injected"], 1)
        print(f"[trace] {n:24s} injected {c['injected']:5d} | known trigger in text {100 * c['loose'] / k:5.1f}% "
              f"(upper bound) | unambiguous {100 * c['tight'] / k:5.1f}%  e.g. {ex}")


if __name__ == "__main__":
    main()
