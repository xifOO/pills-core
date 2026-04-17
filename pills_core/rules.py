from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Generic, TypeVar

from pills_core.exceptions import UnknownDomainTagError
from pills_core.explain import Explanation

ValueT = TypeVar("ValueT")
ContextT = TypeVar("ContextT")


_DOMAIN_TAG_FIELDS: frozenset[str] = frozenset(
    {"is_ratio", "is_monetary", "is_rate", "is_score", "is_count"}
)


@dataclass(frozen=True, slots=True)
class Decision(Generic[ValueT]):
    value: ValueT
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def to_explanation(self, name: str) -> Explanation:
        return Explanation(name=name, value=self.value, reasons=list(self.reasons))


@dataclass(frozen=True, slots=True)
class DomainTags:
    is_ratio: bool = False
    is_monetary: bool = False
    is_rate: bool = False
    is_score: bool = False
    is_count: bool = False

    @classmethod
    def from_tag_name(cls, tag_name: str) -> "DomainTags":
        if tag_name not in _DOMAIN_TAG_FIELDS:
            raise UnknownDomainTagError(tag_name)
        return cls(**{tag_name: True})

    @classmethod
    def from_group_name(cls, group_name: str) -> "DomainTags":
        return cls.from_tag_name(f"is_{group_name}")

    def merge(self, other: "DomainTags") -> "DomainTags":
        return DomainTags(
            is_ratio=self.is_ratio or other.is_ratio,
            is_monetary=self.is_monetary or other.is_monetary,
            is_rate=self.is_rate or other.is_rate,
            is_score=self.is_score or other.is_score,
            is_count=self.is_count or other.is_count,
        )

    def any_set(self) -> bool:
        return any(
            (
                self.is_ratio,
                self.is_monetary,
                self.is_rate,
                self.is_score,
                self.is_count,
            )
        )


@dataclass(frozen=True, slots=True)
class DomainRule:
    name: str
    keywords: tuple[str, ...]
    tags: DomainTags

    def matches(self, column_name: str) -> bool:
        normalized = column_name.lower()
        return any(keyword in normalized for keyword in self.keywords)


@dataclass(frozen=True, slots=True)
class DomainPolicy:
    rules: tuple[DomainRule, ...]

    def resolve(self, column_name: str) -> DomainTags:
        tags = DomainTags()
        for rule in self.rules:
            if rule.matches(column_name):
                tags = tags.merge(rule.tags)
        return tags


@dataclass(frozen=True, slots=True)
class MatchRule(Generic[ContextT, ValueT]):
    name: str
    value: ValueT
    predicate: Callable[[ContextT], bool]
    reason_builder: Callable[[ContextT], tuple[str, ...]]

    def evaluate(self, context: ContextT) -> Decision[ValueT] | None:
        if not self.predicate(context):
            return None
        return Decision(value=self.value, reasons=self.reason_builder(context))


@dataclass(frozen=True, slots=True)
class MatchPolicy(Generic[ContextT, ValueT]):
    rules: tuple[MatchRule[ContextT, ValueT], ...]
    fallback_value: ValueT
    fallback_reasons: tuple[str, ...]

    def resolve(self, context: ContextT) -> Decision[ValueT]:
        for rule in self.rules:
            decision = rule.evaluate(context)
            if decision is not None:
                return decision
        return Decision(value=self.fallback_value, reasons=self.fallback_reasons)
