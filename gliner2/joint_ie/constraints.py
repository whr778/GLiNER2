"""Post-decoding constraints for joint entity and relation extraction.

The implementation deliberately uses structural (duck-typed) accessors so it
can operate on dictionaries, dataclasses, or decoder result objects without
importing the schema or compiler modules.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence


def _get(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _label(value: Any) -> Any:
    return _get(value, "label", "type", "name", "entity_type", "relation", "relation_type")


def _endpoint(relation: Any, side: str) -> Any:
    return _get(relation, side, f"{side}_entity")


def _identity(value: Any) -> Any:
    """Return a stable entity identity, preferring offsets over surface text."""
    if value is None:
        return None
    identifier = _get(value, "id", "entity_id", "candidate_id", "index")
    if identifier is not None:
        return identifier
    start, end = _get(value, "start"), _get(value, "end")
    if start is not None and end is not None:
        return (start, end, _label(value))
    text = _get(value, "text", "value")
    if text is not None:
        return (text, _label(value))
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _relation_key(value: Any) -> tuple[Any, Any, Any]:
    return (_label(value), _identity(_endpoint(value, "head")), _identity(_endpoint(value, "tail")))


def _matches(relation: Any, relation_type: Optional[str]) -> bool:
    return relation_type is None or _label(relation) == relation_type


class Constraint(ABC):
    """Base API for incremental, decoder-independent constraints.

    ``allows`` checks a candidate against already accepted output. ``apply``
    performs deterministic greedy filtering. Constraints that describe a
    required companion edge override ``validate`` for whole-result checks.
    """

    def allows(self, candidate: Any, relations: Sequence[Any] = (), entities: Sequence[Any] = ()) -> bool:
        return True

    def allow_node(self, candidate: Any, nodes: Sequence[Any] = (), edges: Sequence[Any] = ()) -> bool:
        return True

    def allow_edge(self, candidate: Any, nodes: Sequence[Any] = (), edges: Sequence[Any] = ()) -> bool:
        return self.allows(candidate, edges, nodes)

    def penalty_edge(self, candidate: Any, nodes: Sequence[Any] = (), edges: Sequence[Any] = ()) -> float:
        return 0.0

    def __call__(self, candidate: Any, relations: Sequence[Any] = (), entities: Sequence[Any] = ()) -> bool:
        return self.allows(candidate, relations, entities)

    def apply(self, relations: Iterable[Any], entities: Sequence[Any] = ()) -> list[Any]:
        accepted: list[Any] = []
        for candidate in relations:
            if self.allows(candidate, accepted, entities):
                accepted.append(candidate)
        return accepted

    def validate(self, relations: Sequence[Any], entities: Sequence[Any] = ()) -> bool:
        accepted: list[Any] = []
        for candidate in relations:
            if not self.allows(candidate, accepted, entities):
                return False
            accepted.append(candidate)
        return True

    def to_dict(self) -> dict[str, Any]:
        data = {"type": type(self).__name__}
        data.update(self.__dict__)
        return data


@dataclass(frozen=True)
class TypedEndpoints(Constraint):
    relation: Optional[str] = None
    head_types: tuple[str, ...] = ()
    tail_types: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "head_types", tuple(self.head_types))
        object.__setattr__(self, "tail_types", tuple(self.tail_types))

    def allows(self, candidate: Any, relations: Sequence[Any] = (), entities: Sequence[Any] = ()) -> bool:
        if not _matches(candidate, self.relation):
            return True
        head, tail = _label(_endpoint(candidate, "head")), _label(_endpoint(candidate, "tail"))
        return (not self.head_types or head in self.head_types) and (not self.tail_types or tail in self.tail_types)


def _unqualify(value: Any) -> str:
    """Strip a `qualified_query_type` prefix: ``"7::Location"`` -> ``"Location"``.

    A MIRROR of `gliner2.models.base.display_query_type`, kept local because this module
    deliberately imports no schema or model code. `tests/joint_ie/test_typed_role.py`
    asserts the two agree on every shape -- three separate bugs on 2026-09-19 were one
    feature living at two call sites that silently drifted, so the parity is tested, not
    assumed.
    """
    text = str(value)
    head, sep, name = text.partition("::")
    return name if sep and head.isdigit() else text


def _span_of(value: Any) -> Optional[tuple[int, int]]:
    start, end = _get(value, "start"), _get(value, "end")
    if start is None or end is None:
        if isinstance(value, (tuple, list)) and len(value) == 3:
            return (int(value[1]), int(value[2]))
        return None
    return (int(start), int(end))


# THE REFUSAL COUNTER. Option 2's step-3 gate: zero refusals means the constraint did not
# apply, which this programme has shipped three times. Module-level because TypedRole is
# frozen and the beam builds it per sample; read it AFTER decoding, from a point where the
# refusal has already happened.
_TYPED_ROLE_REFUSALS: "defaultdict[tuple, int]" = defaultdict(int)


# Counted SEPARATELY from beam refusals on purpose. A beam refusal happens inside
# `_decode_joint`, whose event output the record head then overwrites, so it changes nothing;
# a DROP removes an argument from the emitted sample. Summing them into one number says "the
# constraint fired 45 times" when 5 of those touched the output -- the same conflation as
# counting firings instead of hits.
_TYPED_ROLE_DROPS: "defaultdict[tuple, int]" = defaultdict(int)


def typed_role_refusals() -> dict:
    """``{(event_type, role): refusals}`` inside the BEAM, since the last reset."""
    return dict(_TYPED_ROLE_REFUSALS)


def typed_role_drops() -> dict:
    """``{(event_type, role): drops}`` from the EMITTED sample, since the last reset."""
    return dict(_TYPED_ROLE_DROPS)


def reset_typed_role_refusals() -> None:
    _TYPED_ROLE_REFUSALS.clear()
    _TYPED_ROLE_DROPS.clear()


@dataclass(frozen=True)
class TypedRole(Constraint):
    """An event role edge may only land on a span the model types compatibly.

    Applied PER (event_type, role), never globally: the sweep measured that `Location`
    and `Date` discriminate while `Subject` does not, so a global constraint spends its
    precision where type identity carries no information.

    The role edge's tail is keyed by the ROLE (``(role, start, end)``), not by an entity
    type, so the filler's type cannot be read off the edge. It is read from the accepted
    NODES at the same span -- excluding the role-keyed node itself, which would otherwise
    "type" every filler as its own role and make the constraint vacuous.

    ``require_typed`` decides the no-evidence case. Default FALSE: a span no entity query
    proposed carries no type to judge, and refusing it would make this a recall filter of
    the kind option 3 already measured negative. Set True only to test that arm
    deliberately.
    """

    event_type: Optional[str] = None
    role: Optional[str] = None
    allowed_types: tuple[str, ...] = ()
    require_typed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "allowed_types", tuple(self.allowed_types))

    def _applies_to(self, candidate: Any) -> bool:
        """Does this edge carry the (event_type, role) this constraint governs?"""
        rel = _label(candidate)
        if rel is None:
            return False
        text = str(rel)
        task, sep, rest = text.partition("::")
        role = _unqualify(rest if sep else text)
        if self.event_type is not None and task != self.event_type:
            return False
        if self.role is not None and role != self.role:
            return False
        return True

    def allows(self, candidate: Any, relations: Sequence[Any] = (), entities: Sequence[Any] = ()) -> bool:
        if not self.allowed_types or not self._applies_to(candidate):
            return True
        tail = _endpoint(candidate, "tail")
        span = _span_of(tail)
        if span is None:
            return True
        own_role = _unqualify(_label(tail)) if _label(tail) is not None else None
        types = {
            _unqualify(_label(node)) for node in entities
            if _span_of(node) == span and _label(node) is not None
        }
        types.discard(own_role)
        if not types:
            return not self.require_typed          # no typed evidence either way
        ok = bool(types & set(self.allowed_types))
        if not ok:
            _TYPED_ROLE_REFUSALS[(self.event_type, self.role)] += 1
        return ok


@dataclass(frozen=True)
class NoSelfLoops(Constraint):
    relation: Optional[str] = None

    def allows(self, candidate: Any, relations: Sequence[Any] = (), entities: Sequence[Any] = ()) -> bool:
        return not _matches(candidate, self.relation) or _identity(_endpoint(candidate, "head")) != _identity(_endpoint(candidate, "tail"))


@dataclass(frozen=True)
class UniqueRelationPair(Constraint):
    relation: Optional[str] = None
    directed: bool = True

    def allows(self, candidate: Any, relations: Sequence[Any] = (), entities: Sequence[Any] = ()) -> bool:
        if not _matches(candidate, self.relation):
            return True
        head, tail = _identity(_endpoint(candidate, "head")), _identity(_endpoint(candidate, "tail"))
        for existing in relations:
            if _label(existing) != _label(candidate):
                continue
            old_head, old_tail = _identity(_endpoint(existing, "head")), _identity(_endpoint(existing, "tail"))
            if (head, tail) == (old_head, old_tail) or (not self.directed and (head, tail) == (old_tail, old_head)):
                return False
        return True


@dataclass(frozen=True)
class UniqueRelationSlot(Constraint):
    relation: Optional[str] = None
    slot: str = "head"

    def __post_init__(self) -> None:
        if self.slot not in {"head", "tail", "slot"}:
            raise ValueError("slot must be 'head', 'tail', or 'slot'")

    def allows(self, candidate: Any, relations: Sequence[Any] = (), entities: Sequence[Any] = ()) -> bool:
        if not _matches(candidate, self.relation):
            return True
        value = _identity(_endpoint(candidate, self.slot))
        return all(_label(old) != _label(candidate) or _identity(_endpoint(old, self.slot)) != value for old in relations)


@dataclass(frozen=True)
class EntityOverlapPolicy(Constraint):
    policy: str = "disallow"

    def __post_init__(self) -> None:
        if self.policy not in {"allow", "disallow", "nested"}:
            raise ValueError("policy must be 'allow', 'disallow', or 'nested'")

    def allows(self, candidate: Any, relations: Sequence[Any] = (), entities: Sequence[Any] = ()) -> bool:
        return True

    def allow_node(self, candidate: Any, nodes: Sequence[Any] = (), edges: Sequence[Any] = ()) -> bool:
        if self.policy == "allow":
            return True
        start, end = _get(candidate, "start"), _get(candidate, "end")
        if start is None or end is None:
            return True
        for old in nodes:
            if old is candidate:
                continue
            old_start, old_end = _get(old, "start"), _get(old, "end")
            if old_start is None or old_end is None or end <= old_start or old_end <= start:
                continue
            nested = ((start >= old_start and end <= old_end) or
                      (old_start >= start and old_end <= end))
            if self.policy == "disallow" or not nested:
                return False
        return True

    def apply_entities(self, entities: Iterable[Any]) -> list[Any]:
        if self.policy == "allow":
            return list(entities)
        accepted: list[Any] = []
        for candidate in entities:
            start, end = _get(candidate, "start"), _get(candidate, "end")
            if start is None or end is None:
                accepted.append(candidate)
                continue
            valid = True
            for old in accepted:
                old_start, old_end = _get(old, "start"), _get(old, "end")
                if old_start is None or old_end is None or end <= old_start or old_end <= start:
                    continue
                nested = (start >= old_start and end <= old_end) or (old_start >= start and old_end <= end)
                if self.policy == "disallow" or not nested:
                    valid = False
                    break
            if valid:
                accepted.append(candidate)
        return accepted


@dataclass(frozen=True)
class MaxRelationsPerHead(Constraint):
    limit: int
    relation: Optional[str] = None

    def __post_init__(self) -> None:
        if self.limit < 0:
            raise ValueError("limit must be non-negative")

    def allows(self, candidate: Any, relations: Sequence[Any] = (), entities: Sequence[Any] = ()) -> bool:
        if not _matches(candidate, self.relation):
            return True
        key = _identity(_endpoint(candidate, "head"))
        return sum(_matches(old, self.relation) and _identity(_endpoint(old, "head")) == key for old in relations) < self.limit


@dataclass(frozen=True)
class MaxRelationsPerTail(Constraint):
    limit: int
    relation: Optional[str] = None

    def __post_init__(self) -> None:
        if self.limit < 0:
            raise ValueError("limit must be non-negative")

    def allows(self, candidate: Any, relations: Sequence[Any] = (), entities: Sequence[Any] = ()) -> bool:
        if not _matches(candidate, self.relation):
            return True
        key = _identity(_endpoint(candidate, "tail"))
        return sum(_matches(old, self.relation) and _identity(_endpoint(old, "tail")) == key for old in relations) < self.limit


@dataclass(frozen=True)
class SymmetricRelation(Constraint):
    relation: str

    def allows(self, candidate: Any, relations: Sequence[Any] = (), entities: Sequence[Any] = ()) -> bool:
        return True

    def validate(self, relations: Sequence[Any], entities: Sequence[Any] = ()) -> bool:
        keys = {_relation_key(item) for item in relations}
        return all(_label(item) != self.relation or (self.relation, _identity(_endpoint(item, "tail")), _identity(_endpoint(item, "head"))) in keys for item in relations)


@dataclass(frozen=True)
class InverseRelation(Constraint):
    relation: str
    inverse: str

    def allows(self, candidate: Any, relations: Sequence[Any] = (), entities: Sequence[Any] = ()) -> bool:
        return True

    def validate(self, relations: Sequence[Any], entities: Sequence[Any] = ()) -> bool:
        keys = {_relation_key(item) for item in relations}
        for item in relations:
            label = _label(item)
            if label == self.relation and (self.inverse, _identity(_endpoint(item, "tail")), _identity(_endpoint(item, "head"))) not in keys:
                return False
            if label == self.inverse and (self.relation, _identity(_endpoint(item, "tail")), _identity(_endpoint(item, "head"))) not in keys:
                return False
        return True


@dataclass(frozen=True)
class AcyclicRelation(Constraint):
    relation: str

    def allows(self, candidate: Any, relations: Sequence[Any] = (), entities: Sequence[Any] = ()) -> bool:
        if not _matches(candidate, self.relation):
            return True
        head, tail = _identity(_endpoint(candidate, "head")), _identity(_endpoint(candidate, "tail"))
        if head == tail:
            return False
        graph: dict[Any, list[Any]] = defaultdict(list)
        for old in relations:
            if _matches(old, self.relation):
                graph[_identity(_endpoint(old, "head"))].append(_identity(_endpoint(old, "tail")))
        stack, seen = [tail], set()
        while stack:
            node = stack.pop()
            if node == head:
                return False
            if node not in seen:
                seen.add(node)
                stack.extend(graph[node])
        return True


_CONSTRAINT_TYPES = {cls.__name__: cls for cls in (
    TypedEndpoints, NoSelfLoops, UniqueRelationPair, UniqueRelationSlot,
    EntityOverlapPolicy, MaxRelationsPerHead, MaxRelationsPerTail,
    SymmetricRelation, InverseRelation, AcyclicRelation,
)}

def constraint_from_dict(data: Mapping[str, Any]) -> Constraint:
    values = dict(data)
    kind = values.pop("type", None)
    try:
        cls = _CONSTRAINT_TYPES[kind]
    except KeyError as exc:
        raise ValueError(f"unknown constraint type {kind!r}") from exc
    return cls(**values)
