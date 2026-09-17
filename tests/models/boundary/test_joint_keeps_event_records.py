"""Joint decode must NOT silently drop event groups that the record head owns.

`decode_mode: joint` took an early return and never ran `_decode_records`, on the reasoning
that records come out of the beam via role edges. That holds for structures. It does NOT hold
for events under `event_records: true`, where events ARE records: measured on 60 cmnee
documents, the beam's role-edge path dropped 35 triggers, added ZERO, and was net -23 correct
items against greedy.

The decode-arms null that blessed joint mode was measured with `event_records` OFF, so the
interaction could not appear there -- which is why a passing aggregate comparison did not catch
this.

This pins the contract both ways: event groups come back, and the beam still owns everything
else.
"""

import inspect

from gliner2.models.boundary import engine


def test_joint_branch_decodes_event_owned_record_groups():
    src = inspect.getsource(engine.BoundaryExtractor._extract_from_batch)
    joint_branch = src.split("if joint:", 1)[1].split("return sample", 1)[0]

    assert "_decode_records" in joint_branch, (
        "the joint branch must run the record head for event-owned groups; without it, "
        "event_records + joint silently loses triggers"
    )
    assert "event_owners" in joint_branch, (
        "the reclaim must be scoped by event_owners -- reclaiming ALL record groups would "
        "double-emit the structures the beam legitimately owns"
    )
    assert "_record_instances_to_events" in joint_branch, (
        "record instances must be converted to event shape, as the non-joint path does"
    )


def test_the_joint_branch_still_runs_the_beam():
    """The fix must not turn joint into greedy: the beam still owns entities and relations."""
    src = inspect.getsource(engine.BoundaryExtractor._extract_from_batch)
    joint_branch = src.split("if joint:", 1)[1].split("return sample", 1)[0]

    assert "_decode_joint" in joint_branch
    # and the beam runs FIRST, so the record head's event output overwrites it rather than
    # being overwritten by it
    assert joint_branch.index("_decode_joint") < joint_branch.index("_decode_records")
