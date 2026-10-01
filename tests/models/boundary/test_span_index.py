"""_span_index: one host transfer, same mapping as the per-element version it replaced."""

import torch

from gliner2.models.boundary.records import _span_index


def test_maps_each_span_to_its_row():
    spans = torch.tensor([[3, 5], [0, 1], [7, 7]])
    assert _span_index(spans) == {(3, 5): 0, (0, 1): 1, (7, 7): 2}


def test_duplicate_span_keeps_last_row():
    spans = torch.tensor([[3, 5], [0, 1], [3, 5]])
    assert _span_index(spans) == {(3, 5): 2, (0, 1): 1}


def test_keys_are_python_ints_and_empty_is_empty():
    (key,) = _span_index(torch.tensor([[2, 4]]))
    assert all(type(x) is int for x in key)
    assert _span_index(torch.zeros((0, 2), dtype=torch.long)) == {}
