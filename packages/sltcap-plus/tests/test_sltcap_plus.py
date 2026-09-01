from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import sltcap_plus as slt


def make_args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "mode": "sltcap",
        "concentration": 150.0,
        "num_cations": 1,
        "num_anions": 1,
        "charge_cation": 1,
        "charge_anion": -1,
        "solute_charge": -5,
        "solvent_volume": None,
        "num_water_molecules": 5550,
        "solver": "numerical",
        "output_format": None,
        "verbose": False,
        "quiet": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def args_for_c_v(c_v: float, **overrides: object) -> argparse.Namespace:
    concentration = float(overrides.pop("concentration", 150.0))
    volume = c_v / (concentration * slt.MILLIMOLAR_TO_MOLAR * slt.AVOGADRO)
    return make_args(
        concentration=concentration,
        solvent_volume=volume,
        num_water_molecules=None,
        **overrides,
    )


def args_for_rho(
    rho: float,
    *,
    charge_cation: int,
    charge_anion: int,
    num_cations: int,
    num_anions: int,
    solver: str,
) -> argparse.Namespace:
    if rho == 0:
        return args_for_c_v(
            10.0,
            charge_cation=charge_cation,
            charge_anion=charge_anion,
            num_cations=num_cations,
            num_anions=num_anions,
            solute_charge=0,
            solver=solver,
        )
    solute_charge = 1 if rho < 0 else -1
    c_v = -solute_charge / (rho * num_cations * charge_cation)
    return args_for_c_v(
        c_v,
        charge_cation=charge_cation,
        charge_anion=charge_anion,
        num_cations=num_cations,
        num_anions=num_anions,
        solute_charge=solute_charge,
        solver=solver,
    )


def test_bulk_formula_units_from_volume() -> None:
    # One cubic nanometre at 1 M contains 0.602214076 formula units.
    assert slt.get_num_bulk_salt_molecules(1e-24, None, 1000.0) == pytest.approx(0.602214076)


def test_bulk_formula_units_from_water_count() -> None:
    assert slt.get_num_bulk_salt_molecules(None, 5550, 100.0) == pytest.approx(10.0)


def test_unrepresentably_large_bulk_system_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite, positive"):
        slt.get_num_bulk_salt_molecules(np.finfo(float).max, None, np.finfo(float).max)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"num_cations": 0}, "Number of cations"),
        ({"num_anions": -1}, "Number of anions"),
        ({"charge_cation": 0}, "Charge of cation"),
        ({"charge_anion": 1}, "Charge of anion"),
        ({"num_cations": 1, "num_anions": 1, "charge_cation": 2, "charge_anion": -1}, "formula unit"),
        ({"concentration": 0.0}, "Concentration"),
        ({"concentration": float("nan")}, "Concentration"),
        ({"concentration": float("inf")}, "Concentration"),
        ({"num_water_molecules": 0}, "water molecules"),
        ({"num_water_molecules": None, "solvent_volume": None}, "Either solvent volume"),
        ({"num_water_molecules": 100, "solvent_volume": 1e-24}, "not both"),
        ({"num_water_molecules": None, "solvent_volume": 0.0}, "Solvent volume"),
        ({"num_water_molecules": None, "solvent_volume": float("nan")}, "Solvent volume"),
        ({"num_water_molecules": None, "solvent_volume": float("inf")}, "Solvent volume"),
    ],
)
def test_invalid_arguments_are_rejected(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        slt.check_arguments(make_args(**overrides))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_cations", 1.5),
        ("num_anions", True),
        ("charge_cation", 1.0),
        ("charge_anion", -1.0),
        ("solute_charge", 0.5),
        ("num_water_molecules", 10.0),
    ],
)
def test_discrete_arguments_must_be_integers(field: str, value: object) -> None:
    with pytest.raises(ValueError, match="integer"):
        slt.check_arguments(make_args(**{field: value}))


@pytest.mark.parametrize("field", ["solute_charge", "charge_cation", "num_water_molecules"])
def test_unrepresentably_large_integers_are_rejected(field: str) -> None:
    with pytest.raises(ValueError, match="too large"):
        slt.check_arguments(make_args(**{field: 10**1000}))


@pytest.mark.parametrize(
    ("salt", "expected_class"),
    [
        ((1, -1), slt.SLTCAP_MONOVALENT),
        ((2, -1), slt.SLTCAP_DIVALENT_CASE_1),
        ((1, -2), slt.SLTCAP_DIVALENT_CASE_2),
        ((3, -2), slt.SLTCAP_NUMERICAL),
    ],
)
def test_analytic_estimator_dispatch(
    salt: tuple[int, int], expected_class: type[slt.SLTCAP_BASE]
) -> None:
    assert slt.get_SLTCAP_estimator_class(slt.get_charge_ratio(*salt)) is expected_class


@pytest.mark.parametrize("solver", ["numerical", "auto", "analytic"])
@pytest.mark.parametrize(
    ("charge_cation", "charge_anion", "num_cations", "num_anions"),
    [(1, -1, 1, 1), (2, -1, 1, 2), (1, -2, 2, 1)],
)
def test_neutral_solute_returns_bulk_counts(
    solver: str,
    charge_cation: int,
    charge_anion: int,
    num_cations: int,
    num_anions: int,
) -> None:
    result = slt.estimate(
        args_for_c_v(
            12.5,
            charge_cation=charge_cation,
            charge_anion=charge_anion,
            num_cations=num_cations,
            num_anions=num_anions,
            solute_charge=0,
            solver=solver,
        )
    )
    assert result.total_cations == pytest.approx(num_cations * 12.5)
    assert result.total_anions == pytest.approx(num_anions * 12.5)
    assert result.residual_charge == pytest.approx(0.0, abs=1e-12)


def test_equal_valence_closed_form_matches_manuscript_equation() -> None:
    args = args_for_c_v(20.0, solute_charge=-8, solver="analytic")
    result = slt.estimate(args)
    rho = 8.0 / 20.0
    s = math.exp(math.asinh(rho / 2.0))
    assert result.total_cations == pytest.approx(20.0 * s)
    assert result.total_anions == pytest.approx(20.0 / s)
    assert result.residual_charge == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    "rho",
    [
        -100.0,
        -10.0,
        -slt.RHO_CRIT_DIVALENT,
        -1.0,
        -1e-8,
        0.0,
        1e-8,
        1.0,
        slt.RHO_CRIT_DIVALENT,
        10.0,
        100.0,
    ],
)
@pytest.mark.parametrize(
    ("charge_cation", "charge_anion", "num_cations", "num_anions"),
    [(1, -1, 1, 1), (2, -1, 1, 2), (1, -2, 2, 1)],
)
def test_analytic_and_numerical_solvers_agree(
    rho: float,
    charge_cation: int,
    charge_anion: int,
    num_cations: int,
    num_anions: int,
) -> None:
    analytic = slt.estimate(
        args_for_rho(
            rho,
            charge_cation=charge_cation,
            charge_anion=charge_anion,
            num_cations=num_cations,
            num_anions=num_anions,
            solver="analytic",
        )
    )
    numerical = slt.estimate(
        args_for_rho(
            rho,
            charge_cation=charge_cation,
            charge_anion=charge_anion,
            num_cations=num_cations,
            num_anions=num_anions,
            solver="numerical",
        )
    )
    assert analytic.total_cations == pytest.approx(numerical.total_cations, rel=2e-12, abs=1e-12)
    assert analytic.total_anions == pytest.approx(numerical.total_anions, rel=2e-12, abs=1e-12)


@pytest.mark.parametrize(
    ("charge_cation", "charge_anion", "num_cations", "num_anions"),
    [(3, -2, 2, 3), (4, -3, 3, 4), (5, -2, 2, 5), (6, -1, 1, 6)],
)
@pytest.mark.parametrize("solute_charge", [-37, -1, 0, 1, 41])
def test_arbitrary_valence_numerical_solution_satisfies_defining_equations(
    charge_cation: int,
    charge_anion: int,
    num_cations: int,
    num_anions: int,
    solute_charge: int,
) -> None:
    c_v = 23.75
    args = args_for_c_v(
        c_v,
        charge_cation=charge_cation,
        charge_anion=charge_anion,
        num_cations=num_cations,
        num_anions=num_anions,
        solute_charge=solute_charge,
    )
    result = slt.estimate(args)

    cation_factor = result.total_cations / (num_cations * c_v)
    anion_factor = result.total_anions / (num_anions * c_v)
    t_from_cations = cation_factor ** (1.0 / charge_cation)
    t_from_anions = anion_factor ** (1.0 / charge_anion)

    assert result.total_cations > 0
    assert result.total_anions > 0
    assert t_from_cations == pytest.approx(t_from_anions, rel=2e-13)
    assert result.residual_charge == pytest.approx(0.0, abs=2e-11)


def test_numerical_solver_is_accurate_near_neutral_limit() -> None:
    result = slt.estimate(
        args_for_rho(
            1e-8,
            charge_cation=1,
            charge_anion=-1,
            num_cations=1,
            num_anions=1,
            solver="numerical",
        )
    )
    assert result.residual_charge == 0.0


def test_analytic_solver_rejects_unsupported_ratio() -> None:
    with pytest.raises(ValueError, match="No analytical estimator"):
        slt.estimate(
            args_for_c_v(
                10.0,
                charge_cation=3,
                charge_anion=-2,
                num_cations=2,
                num_anions=3,
                solver="analytic",
            )
        )


def test_auto_solver_uses_numerical_for_general_ratio() -> None:
    result = slt.estimate(
        args_for_c_v(
            10.0,
            charge_cation=3,
            charge_anion=-2,
            num_cations=2,
            num_anions=3,
            solver="auto",
        )
    )
    assert result.estimator == "SLTCAP_NUMERICAL"
    assert result.fallback_to_numerical is False


def test_auto_solver_falls_back_if_closed_form_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(_: slt.SLTCAP_MONOVALENT) -> None:
        raise RuntimeError("synthetic closed-form failure")

    monkeypatch.setattr(slt.SLTCAP_MONOVALENT, "run_implementation", fail)
    result = slt.estimate(args_for_c_v(10.0, solver="auto"))
    assert result.estimator == "SLTCAP_NUMERICAL"
    assert result.fallback_to_numerical is True


@pytest.mark.parametrize(
    ("solute_charge", "expected_cations", "expected_anions"),
    [(-6, 16.0, 10.0), (6, 10.0, 16.0), (0, 10.0, 10.0)],
)
def test_add_then_neutralize_counts(
    solute_charge: int, expected_cations: float, expected_anions: float
) -> None:
    result = slt.estimate(args_for_c_v(10.0, mode="an", solute_charge=solute_charge))
    assert result.total_cations == pytest.approx(expected_cations)
    assert result.total_anions == pytest.approx(expected_anions)
    assert result.residual_charge == pytest.approx(0.0)
    assert result.solver == "not_applicable"


def test_unsupported_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported mode"):
        slt.estimate(make_args(mode="other"))


@pytest.mark.parametrize(
    ("qp", "qn", "solute_charge", "expected"),
    [(1, -1, 7, True), (2, -1, 7, True), (2, -2, 7, False), (6, -4, 8, True), (6, -4, 7, False)],
)
def test_exact_integer_neutrality_criterion(
    qp: int, qn: int, solute_charge: int, expected: bool
) -> None:
    assert slt.exact_integer_neutrality_possible(qp, qn, solute_charge) is expected


def test_non_integer_values_never_claim_exact_integer_neutrality() -> None:
    assert slt.exact_integer_neutrality_possible(1.0, -1, 0) is False  # type: ignore[arg-type]


def test_private_numeric_guards_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="positive"):
        slt._ceil_div(1, 0)
    with pytest.raises(RuntimeError, match="non-positive or non-finite"):
        slt._positive_finite(0.0, "test")
    assert slt._positive_finite(complex(2.0, 1e-12), "test") == 2.0
    with pytest.raises(RuntimeError, match="imaginary component"):
        slt._positive_finite(complex(2.0, 1e-3), "test")


@pytest.mark.parametrize(
    ("rho", "charge_cation", "charge_anion", "num_cations", "num_anions"),
    [
        (slt.RHO_CRIT_DIVALENT * (1.0 + 1e-10), 2, -1, 1, 2),
        (-slt.RHO_CRIT_DIVALENT * (1.0 + 1e-10), 1, -2, 2, 1),
    ],
)
def test_complex_continuation_matches_numerical_just_beyond_discriminant(
    rho: float,
    charge_cation: int,
    charge_anion: int,
    num_cations: int,
    num_anions: int,
) -> None:
    common = {
        "charge_cation": charge_cation,
        "charge_anion": charge_anion,
        "num_cations": num_cations,
        "num_anions": num_anions,
    }
    analytic = slt.estimate(args_for_rho(rho, solver="analytic", **common))
    numerical = slt.estimate(args_for_rho(rho, solver="numerical", **common))
    assert analytic.total_cations == pytest.approx(numerical.total_cations, rel=2e-12)
    assert analytic.total_anions == pytest.approx(numerical.total_anions, rel=2e-12)


@pytest.mark.parametrize(
    "rho",
    [-100.0, -10.0, -1.0, -1e-8, 0.0, 1e-8, 1.0, 10.0, 100.0],
)
def test_compact_divalent_root_satisfies_defining_cubic(rho: float) -> None:
    s = slt._divalent_cubic_root_s(rho)
    assert s > 0.0
    assert s**3 - rho * s - 1.0 == pytest.approx(0.0, abs=2e-11 * max(1.0, abs(rho)))


def test_shared_analytical_x_back_substitution() -> None:
    class FixedX(slt.SLTCAP_ANALYTICAL_X_BASE):
        def get_x(self) -> float:
            return 4.0

    estimator = FixedX(
        args_for_c_v(
            10.0,
            charge_cation=2,
            charge_anion=-1,
            num_cations=1,
            num_anions=2,
        )
    )
    assert estimator.as_floats() == pytest.approx((40.0, 10.0))
    with pytest.raises(NotImplementedError):
        slt.SLTCAP_ANALYTICAL_X_BASE(args_for_c_v(10.0)).get_x()


def test_rounding_reports_impossible_exact_neutrality() -> None:
    result = slt.nearest_neutral_integer_pair(10.4, 7.6, 2, -2, 1)
    assert result == (10, 8, False, "independent_rounding_exact_neutrality_impossible")


@pytest.mark.parametrize(
    ("n_p_float", "n_n_float", "qp", "qn", "solute_charge"),
    [
        (10.4, 13.4, 1, -1, 3),
        (10.4, 23.8, 2, -1, 3),
        (10.9, 20.8, 2, -1, -1),
        (0.1, 0.1, 3, -2, -7),
        (30.7, 12.2, 4, -3, -2),
        (1e6 + 0.2, 2e6 + 0.1, 2, -1, 0),
    ],
)
def test_neutral_rounding_is_globally_nearest_nonnegative_pair(
    n_p_float: float,
    n_n_float: float,
    qp: int,
    qn: int,
    solute_charge: int,
) -> None:
    rounded_p, rounded_n, possible, method = slt.nearest_neutral_integer_pair(
        n_p_float, n_n_float, qp, qn, solute_charge
    )
    assert possible is True
    assert method == "nearest_exact_neutral_pair"
    assert rounded_p >= 0 and rounded_n >= 0
    assert qp * rounded_p + qn * rounded_n + solute_charge == 0

    # Every neutral pair is on a one-dimensional lattice. Checking a generous
    # interval around the returned pair independently verifies the minimizer.
    returned_score = (rounded_p - n_p_float) ** 2 + (rounded_n - n_n_float) ** 2
    step_p = abs(qn) // math.gcd(qp, abs(qn))
    step_n = qp // math.gcd(qp, abs(qn))
    for offset in range(-100, 101):
        candidate_p = rounded_p + offset * step_p
        candidate_n = rounded_n + offset * step_n
        if candidate_p < 0 or candidate_n < 0:
            continue
        candidate_score = (candidate_p - n_p_float) ** 2 + (candidate_n - n_n_float) ** 2
        assert returned_score <= candidate_score + 1e-12


def test_result_is_immutable() -> None:
    result = slt.estimate(args_for_c_v(10.0))
    with pytest.raises(FrozenInstanceError):
        result.total_cations = 1.0  # type: ignore[misc]


def test_text_result_contains_all_result_fields() -> None:
    result = slt.estimate(args_for_c_v(10.0))
    text = slt.format_text_result(result)
    for field in result.__dataclass_fields__:
        assert f"{field}=" in text


def test_emit_result_supports_text_and_json(capsys: pytest.CaptureFixture[str]) -> None:
    result = slt.estimate(args_for_c_v(10.0))
    slt.emit_result(result, "text")
    assert "rounded_cations=" in capsys.readouterr().out
    slt.emit_result(result, "json")
    assert json.loads(capsys.readouterr().out)["mode"] == "sltcap"
    with pytest.raises(ValueError, match="Unsupported output format"):
        slt.emit_result(result, "xml")


def test_main_emits_requested_format(capsys: pytest.CaptureFixture[str]) -> None:
    result = slt.main(args_for_c_v(10.0, output_format="json"))
    assert json.loads(capsys.readouterr().out)["rounded_cations"] == result.rounded_cations


def test_estimator_accessors_cache_and_round_results() -> None:
    estimator = slt.SLTCAP_NUMERICAL(args_for_c_v(10.0))
    first = estimator.as_floats()
    assert estimator.as_floats() == first
    rounded = estimator.as_integers()
    assert estimator.qp * rounded[0] + estimator.qn * rounded[1] + estimator.solute_charge == 0


def test_estimator_base_and_validation_failure_paths(caplog: pytest.LogCaptureFixture) -> None:
    args = args_for_c_v(10.0)
    with pytest.raises(NotImplementedError):
        slt.SLTCAP_BASE(args).run_implementation()

    class Missing(slt.SLTCAP_BASE):
        def run_implementation(self) -> None:
            pass

    class NonFinite(slt.SLTCAP_BASE):
        def run_implementation(self) -> None:
            self._estimated_total_num_cations = float("inf")
            self._estimated_total_num_anions = 1.0

    class Negative(slt.SLTCAP_BASE):
        def run_implementation(self) -> None:
            self._estimated_total_num_cations = -1.0
            self._estimated_total_num_anions = 1.0

    class TinyNegative(slt.SLTCAP_BASE):
        def run_implementation(self) -> None:
            self._estimated_total_num_cations = -slt.NEGATIVE_ION_COUNT_TOL / 2
            self._estimated_total_num_anions = 1.0

    class Charged(slt.SLTCAP_BASE):
        def run_implementation(self) -> None:
            self._estimated_total_num_cations = 1.0
            self._estimated_total_num_anions = 1.0

    with pytest.raises(RuntimeError, match="did not produce"):
        Missing(args).run()
    with pytest.raises(RuntimeError, match="non-finite"):
        NonFinite(args).run()
    with pytest.raises(RuntimeError, match="negative"):
        Negative(args).run()
    assert TinyNegative(args).as_floats()[0] == 0.0
    with caplog.at_level(logging.WARNING):
        Charged(args).run()
    assert "Residual charge" in caplog.text


def test_fallback_can_be_disabled() -> None:
    class Failing(slt.SLTCAP_BASE):
        def run_implementation(self) -> None:
            raise RuntimeError("failure")

    with pytest.raises(RuntimeError, match="failure"):
        slt.run_estimator_with_optional_fallback(args_for_c_v(10.0), Failing, allow_fallback=False)


def test_unsupported_solver_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported solver"):
        slt.estimate(args_for_c_v(10.0, solver="other"))


def test_cli_function_success_and_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert slt.cli(["sltcap", "-Q", "0", "-W", "5550", "--quiet"]) == 0
    assert "rounded_residual_charge=0" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc_info:
        slt.cli(["sltcap", "-Q", "0", "-W", "5550", "-c", "nan"])
    assert exc_info.value.code == 2
    assert "Concentration must be finite" in capsys.readouterr().err
