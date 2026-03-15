import pytest

from pills_core._enums import TransformPhase
from pills_core.strategies.numeric._registry import (
    build_imputation_registry,
    build_outliers_registry,
    build_scaling_registry,
)
from pills_core.strategies.resolver import resolve_phase_order

IMP = TransformPhase.IMPUTATION
OUT = TransformPhase.OUTLIER
SCL = TransformPhase.SCALING


def assert_before(order, a, b):
    assert order.index(a) < order.index(b), f"{a} should appear before {b} in {order}"


imputation_registry = build_imputation_registry()
outlier_registry = build_outliers_registry()
scaling_registry = build_scaling_registry()

IMPUTATIONS = imputation_registry.strategies
OUTLIERS = outlier_registry.strategies
SCALERS = scaling_registry.strategies


class TestReturnContract:
    @pytest.mark.parametrize("imp", IMPUTATIONS)
    @pytest.mark.parametrize("out", OUTLIERS)
    @pytest.mark.parametrize("scl", SCALERS)
    def test_returns_all_three_phases(self, imp, out, scl):
        order = resolve_phase_order(imp, out, scl)

        assert len(order) == 3
        assert set(order) == {IMP, OUT, SCL}


class TestInvariantDependencies:
    @pytest.mark.parametrize("imp", IMPUTATIONS)
    @pytest.mark.parametrize("out", OUTLIERS)
    @pytest.mark.parametrize("scl", SCALERS)
    def test_imputation_always_before_scaling(self, imp, out, scl):
        order = resolve_phase_order(imp, out, scl)

        assert_before(order, IMP, SCL)


class TestImputationOutlierDependency:
    @pytest.mark.parametrize("imp", IMPUTATIONS)
    @pytest.mark.parametrize("out", OUTLIERS)
    def test_imputation_that_requires_clean_data_runs_after_outliers(self, imp, out):
        order = resolve_phase_order(imp, out, SCALERS[0])

        if imp.requires_outliers_removed:
            assert_before(order, OUT, IMP)
        else:
            assert_before(order, IMP, OUT)


class TestScalingSensitivity:
    @pytest.mark.parametrize("scl", SCALERS)
    def test_outlier_removal_precedes_sensitive_scalers(self, scl):
        order = resolve_phase_order(IMPUTATIONS[0], OUTLIERS[0], scl)

        if scl.requires_outliers_removed:
            assert_before(order, OUT, SCL)


class TestGraphValidity:
    @pytest.mark.parametrize("imp", IMPUTATIONS)
    @pytest.mark.parametrize("out", OUTLIERS)
    @pytest.mark.parametrize("scl", SCALERS)
    def test_result_respects_all_declared_dependencies(self, imp, out, scl):
        order = resolve_phase_order(imp, out, scl)

        edges = []

        edges.append((IMP, SCL))

        if imp.requires_outliers_removed:
            edges.append((OUT, IMP))
        else:
            edges.append((IMP, OUT))

        if scl.requires_outliers_removed:
            edges.append((OUT, SCL))

        for a, b in edges:
            assert_before(order, a, b)


class TestDeterminism:
    @pytest.mark.parametrize("imp", IMPUTATIONS)
    @pytest.mark.parametrize("out", OUTLIERS)
    @pytest.mark.parametrize("scl", SCALERS)
    def test_resolver_is_deterministic(self, imp, out, scl):

        results = set()

        for _ in range(30):
            order = resolve_phase_order(imp, out, scl)
            results.add(tuple(order))

        assert len(results) == 1


class TestRegistryIntegrity:
    def test_strategy_names_are_unique(self):

        registries = [
            imputation_registry,
            outlier_registry,
            scaling_registry,
        ]

        for registry in registries:
            names = [s.name for s in registry.strategies]
            assert len(names) == len(set(names)), "duplicate strategy names detected"
