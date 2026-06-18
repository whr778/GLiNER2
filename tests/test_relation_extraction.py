"""
Test relation extraction with include_confidence and include_spans parameters.

This test demonstrates the new API features for relation extraction.
"""

import json

import pytest

from gliner2 import GLiNER2


def test_relation_extraction():
    """Test relation extraction with various output formats."""
    
    print("=" * 80)
    print("RELATION EXTRACTION TESTS")
    print("=" * 80)
    
    print("\nLoading model: fastino/gliner2-base-v1...")
    model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
    print("Model loaded successfully!\n")
    
    text = "Apple CEO Tim Cook announced the iPhone 15 launch in Cupertino on September 12."
    relation_types = ["CEO_of", "located_in", "announced_on"]
    
    print(f"\nTest Text: {text}")
    print(f"Relation Types: {relation_types}")
    print("\n" + "-" * 80)
    
    # Test 1: Basic extraction (default - tuples)
    print("\n1. BASIC EXTRACTION (tuple format)")
    print("-" * 40)
    result = model.extract_relations(text, relation_types)
    print(json.dumps(result, indent=2))
    
    # Test 2: With confidence scores
    print("\n2. WITH CONFIDENCE SCORES")
    print("-" * 40)
    result = model.extract_relations(text, relation_types, include_confidence=True)
    print(json.dumps(result, indent=2))
    
    # Test 3: With span positions
    print("\n3. WITH SPAN POSITIONS")
    print("-" * 40)
    result = model.extract_relations(text, relation_types, include_spans=True)
    print(json.dumps(result, indent=2))
    
    # Test 4: With both confidence and spans
    print("\n4. WITH CONFIDENCE AND SPAN POSITIONS")
    print("-" * 40)
    result = model.extract_relations(text, relation_types, include_confidence=True, include_spans=True)
    print(json.dumps(result, indent=2))
    
    # Test 5: Verify character positions
    print("\n5. VERIFY CHARACTER POSITIONS")
    print("-" * 40)
    for rel_type, relations in result["relation_extraction"].items():
        print(f"\n{rel_type}:")
        for rel in relations:
            head_text = text[rel["head"]["start"]:rel["head"]["end"]]
            tail_text = text[rel["tail"]["start"]:rel["tail"]["end"]]
            print(f"  Head: '{rel['head']['text']}' at [{rel['head']['start']}:{rel['head']['end']}]")
            print(f"    Verification: '{head_text}' - Match: {head_text == rel['head']['text']}")
            print(f"  Tail: '{rel['tail']['text']}' at [{rel['tail']['start']}:{rel['tail']['end']}]")
            print(f"    Verification: '{tail_text}' - Match: {tail_text == rel['tail']['text']}")
    
    print("\n" + "=" * 80)
    print("Relation extraction tests completed!")
    print("=" * 80)


def test_batch_relation_extraction():
    """Test batch relation extraction with various output formats."""
    
    print("\n" + "=" * 80)
    print("BATCH RELATION EXTRACTION TESTS")
    print("=" * 80)
    
    print("\nLoading model: fastino/gliner2-base-v1...")
    model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
    print("Model loaded successfully!\n")
    
    texts = [
        "Apple CEO Tim Cook works in Cupertino.",
        "Google CEO Sundar Pichai leads the company in Mountain View.",
        "Microsoft was founded by Bill Gates."
    ]
    relation_types = ["CEO_of", "works_in", "founded_by"]
    
    print(f"\nNumber of texts: {len(texts)}")
    print(f"Relation Types: {relation_types}")
    
    # Test with full metadata
    print("\n1. BATCH WITH CONFIDENCE AND SPANS")
    print("-" * 40)
    results = model.batch_extract_relations(
        texts, relation_types, batch_size=2,
        include_confidence=True, include_spans=True
    )
    
    for i, (text, result) in enumerate(zip(texts, results)):
        print(f"\nText {i+1}: {text}")
        print(f"Result: {json.dumps(result, indent=2)}")
    
    print("\n" + "=" * 80)
    print("Batch relation extraction tests completed!")
    print("=" * 80)


@pytest.mark.slow
def test_relation_scoring_via_compute_metrics():
    """Regression for the tuple-format scoring bug: the metrics scorer must read
    the engine's real relation output, and the model's own predictions fed back
    as gold must score > 0 through compute_metrics (it was a silent 0.0)."""
    from gliner2.training.metrics import compute_metrics, _pred_relation_set

    model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
    text = "Apple CEO Tim Cook works in Cupertino."
    relation_types = ["CEO_of", "works_in"]

    pred = model.extract_relations(text, relation_types)  # default (tuple) output shape
    block = pred.get("relation_extraction", {})
    if not any(block.values()):
        pytest.skip("model extracted no relations for this text; cannot test scoring")

    # The scorer must not silently drop the engine's tuple output.
    assert _pred_relation_set(pred), "model emitted relations but the scorer parsed none (format bug)"

    # Self-consistency: feed the model's own predictions back as gold; a correct
    # scorer must then score > 0 (would be 0.0 if the output shape isn't parsed).
    gold_relations = []
    for name, items in block.items():
        for it in items:
            head, tail = (it[0], it[1]) if isinstance(it, (list, tuple)) else (it.get("head"), it.get("tail"))
            if isinstance(head, dict):
                head = head.get("text")
            if isinstance(tail, dict):
                tail = tail.get("text")
            if head and tail:
                gold_relations.append({name: {"head": head, "tail": tail}})
    gold_output = {"relations": gold_relations}

    class _DS:
        def __len__(self):
            return 1

        def __getitem__(self, i):
            return text, gold_output

    metrics = compute_metrics(model, _DS(), batch_size=1, threshold=0.5)
    assert metrics.get("eval_relation_strict_micro_f1", 0.0) > 0.0


if __name__ == "__main__":
    test_relation_extraction()
    test_batch_relation_extraction()

