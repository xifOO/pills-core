from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Generic, TypeVar

ValueT = TypeVar("ValueT")
ContextT = TypeVar("ContextT")


@dataclass(frozen=True, slots=True)
class Decision(Generic[ValueT]):
    value: ValueT
    reasons: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class DomainTags:
    is_ratio: bool = False
    is_monetary: bool = False
    is_rate: bool = False
    is_score: bool = False

    def merge(self, other: "DomainTags") -> "DomainTags":
        return DomainTags(
            is_ratio=self.is_ratio or other.is_ratio,
            is_monetary=self.is_monetary or other.is_monetary,
            is_rate=self.is_rate or other.is_rate,
            is_score=self.is_score or other.is_score,
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


@dataclass(frozen=True, slots=True)
class DomainPolicy:
    rules: tuple[DomainRule, ...]

    def resolve(self, column_name: str) -> DomainTags:
        tags = DomainTags()
        for rule in self.rules:
            if rule.matches(column_name):
                tags = tags.merge(rule.tags)
        return tags
