"""The processor must pickle, or macOS cannot run DataLoader workers at all.

`SchemaTransformer._tokenize_cached` is a `functools.lru_cache` wrapper, and those can
never be pickled -- the unpickler cannot prove the restored object is the same cache. It
was the ONLY unpicklable attribute on the class; the fast tokenizer itself pickles fine.

The cost of that one attribute was not a slow loader. `trainer.py` forced
``num_workers = 0`` on darwin because spawn must pickle the collator, so a local run and a
GPU run executed STRUCTURALLY DIFFERENT data paths -- one process here, forked workers
there. Any defect requiring worker semantics was invisible on the operator's machine by
construction, which is how an eb17 crash that reproduces on the box within 6,269 steps
survived five local traces over ~600,000 samples.

These fail if the __getstate__/__setstate__ pair is removed.
"""

from __future__ import annotations

import functools
import pickle

import pytest

from gliner2.processor import SchemaTransformer


@pytest.fixture(scope="module")
def processor():
    from gliner2 import AutoExtractor
    try:
        return AutoExtractor.from_pretrained(
            "out/maven-tier2-eventrecords/best", map_location="cpu").processor
    except Exception as exc:  # no local checkpoint in this environment
        pytest.skip(f"no checkpoint available: {type(exc).__name__}")


def test_processor_pickles(processor):
    assert isinstance(pickle.loads(pickle.dumps(processor)), SchemaTransformer)


def test_the_lru_cache_is_dropped_not_serialised(processor):
    """It is the one attribute that cannot be pickled; it must not be in the state."""
    assert isinstance(processor._tokenize_cached, functools._lru_cache_wrapper)
    assert "_tokenize_cached" not in processor.__getstate__()


def test_the_cache_is_rebuilt_on_unpickle(processor):
    back = pickle.loads(pickle.dumps(processor))
    assert isinstance(back._tokenize_cached, functools._lru_cache_wrapper)


def test_tokenisation_is_byte_exact_after_a_roundtrip(processor):
    """Dropping the cache is only safe because tokenize() is deterministic per string."""
    back = pickle.loads(pickle.dumps(processor))
    for text in ("Hurricane Helene killed 33 people in Asheville",
                 "兰州肉苁蓉为列当科肉苁蓉属下的一个种",
                 "Kahramanmaraş merkezli depremlerde"):
        assert processor._tokenize_text(text) == back._tokenize_text(text)


def test_a_dataloader_with_workers_can_be_built(processor):
    """The point of the exercise: workers require a picklable collator."""
    from gliner2.training.trainer import ExtractorCollator
    coll = ExtractorCollator(processor, is_training=True, architecture="boundary")
    assert pickle.loads(pickle.dumps(coll)) is not None
