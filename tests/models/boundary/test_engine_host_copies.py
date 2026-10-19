"""Engine decode reads device tensors through one host copy (upstream #188 port, TODO #32)."""
import pytest
import torch

from gliner2.inference.schema import AttributeGroup
from tests.models.boundary.test_record_head_pipeline import _build_tiny_records_model

TEXTS = [
    "Alice bought apples in Paris",
    "Bob sold oranges to Carol",
    "The camera is excellent but the battery is weak",
]


def _accelerator():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None


def test_batched_classification_matches_one_sample_at_a_time():
    model = _build_tiny_records_model()
    schema = (
        model.create_schema()
        .entities(["person", "fruit"])
        .classification("sentiment", ["positive", "negative", "neutral"])
        .classification("topic", ["trade", "travel", "tech"], multi_label=True, cls_threshold=0.0)
    )
    # Batching moves floats ~1e-7 through padding (entity scores too), so confidences match to 1e-5.
    batched = model.batch_extract(TEXTS, schema, batch_size=len(TEXTS), include_confidence=True)
    single = [model.batch_extract([t], schema, batch_size=1, include_confidence=True)[0] for t in TEXTS]

    def classifications(r):
        topic = [(t["label"], t["confidence"]) for t in r["topic"]]
        return [(r["sentiment"]["label"], r["sentiment"]["confidence"])] + topic

    for got, want in zip(batched, single):
        assert [label for label, _ in classifications(got)] == [label for label, _ in classifications(want)]
        assert [c for _, c in classifications(got)] == pytest.approx([c for _, c in classifications(want)], abs=1e-5)
    assert all("sentiment" in r and len(r["topic"]) == 3 for r in batched)
    assert len({str(r["sentiment"]) for r in batched}) > 1


@pytest.mark.skipif(_accelerator() is None, reason="needs CUDA or MPS")
def test_entity_attributes_decode_on_an_accelerator():
    model = _build_tiny_records_model().to(_accelerator())
    schema = model.create_schema().entities(["product", "person"]).entity_attributes({
        "sentiment": AttributeGroup(["positive", "negative"], applies_to=["product"]),
    })
    results = model.batch_extract(TEXTS, schema, batch_size=len(TEXTS), threshold=0.0)

    products = [item for r in results for item in r["entities"]["product"]]
    assert products and all("sentiment" in item for item in products)
