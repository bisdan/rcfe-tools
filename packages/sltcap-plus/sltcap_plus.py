#!/usr/bin/env python3
"""Estimate ion counts for neutralizing a charged solute with binary salts.

This script implements the add-then-neutralize heuristic and SLTCAP/SLTCAP+
estimators for binary salts.  By default, SLTCAP mode uses the robust numerical
root solver for arbitrary valence ratios.  Analytical closed-form branches for
1:1, 2:1, and 1:2 charge ratios are retained for validation and can be selected
with ``--solver auto`` or ``--solver analytic``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Optional

import numpy as np
from numpy import arccosh, arcsinh, cosh, exp, sinh, sqrt
from scipy import constants
from scipy.optimize import root_scalar

logger = logging.getLogger(__name__)

__version__ = "0.1.0"

# Physical constants and unit conversion factors.
AVOGADRO = constants.Avogadro
MILLIMOLAR_TO_MOLAR = 1e-3
WATER_MOLARITY_M = 55.5

# Numerical tolerances.
ABS_RESIDUAL_CHARGE_TOL = 1e-6
REL_RESIDUAL_CHARGE_TOL = 1e-12
NEGATIVE_ION_COUNT_TOL = 1e-12
CLOSED_FORM_IMAG_TOL = 1e-10

# Critical rho at which s^3 - rho*s - 1 changes from one to three real roots.
# The compact two-branch implementation does not branch here, but exposing the
# value is useful for validation at the discriminant boundary.
RHO_CRIT_DIVALENT = (27.0 / 4.0) ** (1.0 / 3.0)


@dataclass(frozen=True)
class EstimationResult:
    mode: str
    solver: str
    estimator: str
    fallback_to_numerical: bool
    total_cations: float
    total_anions: float
    rounded_cations: int
    rounded_anions: int
    residual_charge: float
    rounded_residual_charge: float
    exact_integer_neutrality_possible: bool
    rounding_method: str


def check_arguments(args: argparse.Namespace) -> None:
    integer_arguments = {
        "Number of anions": args.num_anions,
        "Number of cations": args.num_cations,
        "Charge of cation": args.charge_cation,
        "Charge of anion": args.charge_anion,
        "Solute charge": args.solute_charge,
    }
    if args.num_water_molecules is not None:
        integer_arguments["Number of water molecules"] = args.num_water_molecules
    for name, value in integer_arguments.items():
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{name} must be an integer.")
        try:
            value_as_float = float(value)
        except OverflowError as exc:
            raise ValueError(f"{name} is too large for floating-point calculation.") from exc
        if not math.isfinite(value_as_float):
            raise ValueError(f"{name} is too large for floating-point calculation.")

    if args.num_anions <= 0:
        raise ValueError("Number of anions must be greater than 0.")
    if args.num_cations <= 0:
        raise ValueError("Number of cations must be greater than 0.")
    if args.charge_cation <= 0:
        raise ValueError("Charge of cation must be greater than 0.")
    if args.charge_anion >= 0:
        raise ValueError("Charge of anion must be less than 0.")
    if args.num_cations * args.charge_cation + args.num_anions * args.charge_anion != 0:
        raise ValueError("Total charge of the salt formula unit must be zero.")
    if not math.isfinite(args.concentration) or args.concentration <= 0:
        raise ValueError("Concentration must be finite and greater than 0.")
    if args.solvent_volume is not None and (
        not math.isfinite(args.solvent_volume) or args.solvent_volume <= 0
    ):
        raise ValueError("Solvent volume must be finite and greater than 0.")
    if args.num_water_molecules is not None and args.num_water_molecules <= 0:
        raise ValueError("Number of water molecules must be greater than 0.")
    if args.solvent_volume is not None and args.num_water_molecules is not None:
        raise ValueError("Either solvent volume or number of water molecules must be provided, but not both.")
    if args.solvent_volume is None and args.num_water_molecules is None:
        raise ValueError("Either solvent volume or number of water molecules must be provided.")


def get_num_bulk_salt_molecules(
    solvent_volume: Optional[float],
    num_water_molecules: Optional[int],
    concentration: float,
) -> float:
    """Return the number of bulk salt formula units at the target concentration."""
    try:
        if solvent_volume is not None:
            result = float(solvent_volume) * float(concentration) * MILLIMOLAR_TO_MOLAR * AVOGADRO
        else:
            assert num_water_molecules is not None
            result = float(concentration) * MILLIMOLAR_TO_MOLAR * (num_water_molecules / WATER_MOLARITY_M)
    except OverflowError as exc:
        raise ValueError("The solvent definition produces an unrepresentably large system.") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError("The solvent definition must produce a finite, positive number of salt formula units.")
    return result


def get_charge_ratio(charge_cation: int, charge_anion: int) -> float:
    return charge_cation / abs(charge_anion)


def compute_residual_charge(
    qp: float,
    qn: float,
    total_cations: float,
    total_anions: float,
    solute_charge: float,
) -> float:
    return qp * total_cations + qn * total_anions + solute_charge


def residual_charge_tolerance(
    qp: float,
    qn: float,
    total_cations: float,
    total_anions: float,
    solute_charge: float,
) -> float:
    scale = max(abs(qp * total_cations), abs(qn * total_anions), abs(solute_charge), 1.0)
    return max(ABS_RESIDUAL_CHARGE_TOL, REL_RESIDUAL_CHARGE_TOL * scale)


def _is_int_like(value: float) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def exact_integer_neutrality_possible(qp: int, qn: int, solute_charge: int) -> bool:
    """Return whether integer ion counts can exactly neutralize the solute."""
    if not (_is_int_like(qp) and _is_int_like(qn) and _is_int_like(solute_charge)):
        return False
    qp_i = int(qp)
    qn_i = int(qn)
    q_solute_i = int(solute_charge)
    return (-q_solute_i) % math.gcd(qp_i, abs(qn_i)) == 0


def _extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """Return (g, x, y) such that a*x + b*y = g = gcd(a, b)."""
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t
    return old_r, old_s, old_t


def _ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("Denominator must be positive.")
    return -((-numerator) // denominator)


def nearest_neutral_integer_pair(
    n_p_float: float,
    n_n_float: float,
    qp: int,
    qn: int,
    solute_charge: int,
) -> tuple[int, int, bool, str]:
    """Round to the nearest non-negative integer pair, preserving neutrality if possible.

    The neutrality condition is qp*N+ + qn*N- + Q = 0.  If exact integer
    neutralization is impossible because gcd(qp, |qn|) does not divide -Q,
    the function falls back to independent rounding and reports this explicitly.
    """
    rounded_independent = (max(0, int(round(n_p_float))), max(0, int(round(n_n_float))))

    if not exact_integer_neutrality_possible(qp, qn, solute_charge):
        return (*rounded_independent, False, "independent_rounding_exact_neutrality_impossible")

    a = int(qp)
    b = abs(int(qn))
    c = -int(solute_charge)
    g, x, y = _extended_gcd(a, b)
    multiplier = c // g

    # a*x0 - b*y0 = c, because a*x + b*y = g from _extended_gcd.
    n_p0 = x * multiplier
    n_n0 = -y * multiplier

    step_p = b // g
    step_n = a // g

    # Non-negativity constraints for n_p0 + step_p*k and n_n0 + step_n*k.
    k_min = max(_ceil_div(-n_p0, step_p), _ceil_div(-n_n0, step_n))

    # Continuous minimizer of squared Euclidean distance to the float-valued estimate.
    k_star = (
        step_p * (n_p_float - n_p0) + step_n * (n_n_float - n_n0)
    ) / (step_p**2 + step_n**2)

    candidate_ks = {k_min}
    k_center = int(round(k_star))
    for delta in range(-8, 9):
        candidate_ks.add(max(k_min, k_center + delta))

    best_pair: Optional[tuple[int, int]] = None
    best_score = float("inf")
    for k in candidate_ks:
        n_p = n_p0 + step_p * k
        n_n = n_n0 + step_n * k
        if n_p < 0 or n_n < 0:
            continue
        score = (n_p - n_p_float) ** 2 + (n_n - n_n_float) ** 2
        if score < best_score:
            best_score = score
            best_pair = (n_p, n_n)

    if best_pair is None:
        # This should not happen for positive charges, but keep a safe fallback.
        return (*rounded_independent, True, "independent_rounding_no_nonnegative_neutral_pair_found")

    return (*best_pair, True, "nearest_exact_neutral_pair")


def _positive_finite(value: complex | float, context: str) -> float:
    value = np.asarray(value).item()
    if isinstance(value, complex):
        if abs(value.imag) > CLOSED_FORM_IMAG_TOL * max(1.0, abs(value.real)):
            raise RuntimeError(f"{context} produced a non-negligible imaginary component: {value!r}.")
        value = value.real
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise RuntimeError(f"{context} produced a non-positive or non-finite value: {value!r}.")
    return value


def _divalent_cubic_root_s(rho: float) -> float:
    """Return the positive physical root of s**3 - rho*s - 1 = 0.

    This is the compact two-branch form used in the supplementary derivation.
    For positive rho, the acosh argument may be below one; casting to complex
    activates the principal complex continuation, which is equivalent to the
    usual trigonometric branch but keeps the implementation to two cases.
    """
    if rho == 0.0:
        return 1.0
    if rho < 0.0:
        asinh_arg = (3.0 / (2.0 * rho)) * sqrt(-3.0 / rho)
        s = -2.0 * sqrt(-rho / 3.0) * sinh((1.0 / 3.0) * arcsinh(asinh_arg))
    else:
        acosh_arg = (3.0 / (2.0 * rho)) * sqrt(3.0 / rho)
        s = 2.0 * sqrt(rho / 3.0) * cosh(
            (1.0 / 3.0) * arccosh(complex(acosh_arg, 0.0))
        )
    return _positive_finite(s, "divalent cubic root")


class SLTCAP_BASE:
    """Base estimator for total ion counts (float-valued)."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.np = args.num_cations
        self.qp = args.charge_cation
        self.nn = args.num_anions
        self.qn = args.charge_anion
        self.solute_charge = args.solute_charge
        self.cV = get_num_bulk_salt_molecules(args.solvent_volume, args.num_water_molecules, args.concentration)
        rho_conversion_factor = self.cV * self.np * self.qp
        if rho_conversion_factor <= 0:
            raise ValueError("Invalid parameters: cV * np * qp must be greater than 0.")
        self.rho = -(args.solute_charge / rho_conversion_factor)
        if not np.isfinite(self.rho):
            raise ValueError("Computed rho is not finite. Please verify concentration, solvent definition, and charges.")
        self._estimated_total_num_cations: Optional[float] = None
        self._estimated_total_num_anions: Optional[float] = None

    def run_implementation(self) -> None:
        raise NotImplementedError

    def _validate_estimates(self) -> None:
        if self._estimated_total_num_cations is None or self._estimated_total_num_anions is None:
            raise RuntimeError("Estimator did not produce cation/anion totals.")

        self._estimated_total_num_cations = float(self._estimated_total_num_cations)
        self._estimated_total_num_anions = float(self._estimated_total_num_anions)

        if not np.isfinite(self._estimated_total_num_cations) or not np.isfinite(self._estimated_total_num_anions):
            raise RuntimeError("Estimator produced non-finite cation/anion totals.")
        if self._estimated_total_num_cations < -NEGATIVE_ION_COUNT_TOL or self._estimated_total_num_anions < -NEGATIVE_ION_COUNT_TOL:
            raise RuntimeError("Estimator produced negative ion counts.")

        # Clamp tiny negative numerical noise to zero.
        self._estimated_total_num_cations = max(0.0, self._estimated_total_num_cations)
        self._estimated_total_num_anions = max(0.0, self._estimated_total_num_anions)

    def run(self) -> None:
        class_name = self.__class__.__name__
        logger.info(
            "Running %s with np=%s qp=%s nn=%s qn=%s cV=%s rho=%s",
            class_name,
            self.np,
            self.qp,
            self.nn,
            self.qn,
            self.cV,
            self.rho,
        )
        self.run_implementation()
        self._validate_estimates()
        logger.info(
            "Estimated totals: cations=%s anions=%s",
            self._estimated_total_num_cations,
            self._estimated_total_num_anions,
        )
        residual_charge = self.residual_charge()
        tol = residual_charge_tolerance(
            self.qp,
            self.qn,
            self._estimated_total_num_cations,
            self._estimated_total_num_anions,
            self.solute_charge,
        )
        if abs(residual_charge) > tol:
            logger.warning(
                "Residual charge=%s exceeds tolerance=%s. Parameters or estimator may be unstable.",
                residual_charge,
                tol,
            )

    def as_floats(self) -> tuple[float, float]:
        if self._estimated_total_num_cations is None or self._estimated_total_num_anions is None:
            self.run()
        assert self._estimated_total_num_cations is not None
        assert self._estimated_total_num_anions is not None
        return self._estimated_total_num_cations, self._estimated_total_num_anions

    def as_integers(self) -> tuple[int, int]:
        n_p, n_n = self.as_floats()
        rounded_p, rounded_n, _, _ = nearest_neutral_integer_pair(n_p, n_n, self.qp, self.qn, self.solute_charge)
        return rounded_p, rounded_n

    def residual_charge(self) -> float:
        n_p, n_n = self.as_floats()
        return compute_residual_charge(self.qp, self.qn, n_p, n_n, self.solute_charge)


class SLTCAP_AN(SLTCAP_BASE):
    """Add-then-neutralize heuristic.

    Start from bulk-equivalent counts (np*cV, nn*cV), then add only the
    counterion species needed to neutralize the solute charge.
    """

    def run_implementation(self) -> None:
        n_p = self.np * self.cV
        n_n = self.nn * self.cV
        if self.solute_charge > 0:
            n_n -= self.solute_charge / self.qn
        else:
            n_p -= self.solute_charge / self.qp
        self._estimated_total_num_cations = n_p
        self._estimated_total_num_anions = n_n


class SLTCAP_ANALYTICAL_X_BASE(SLTCAP_BASE):
    """Analytical SLTCAP estimator that first computes x = t**q+.

    Once x is known, the shared back-substitution is
    N+ = nu+*cV*x and N- = nu-*cV*x**(q-/q+).
    """

    def get_x(self) -> float:
        raise NotImplementedError

    def run_implementation(self) -> None:
        x = _positive_finite(self.get_x(), f"{self.__class__.__name__}.get_x")
        self._estimated_total_num_cations = self.np * self.cV * x
        self._estimated_total_num_anions = self.nn * self.cV * (x ** (self.qn / self.qp))


class SLTCAP_MONOVALENT(SLTCAP_ANALYTICAL_X_BASE):
    """Closed-form SLTCAP for q+ : q- = +v : -v."""

    def get_x(self) -> float:
        # x = t**q+ = exp(asinh(rho/2)).
        return float(exp(arcsinh(self.rho / 2.0)))


class SLTCAP_DIVALENT_CASE_1(SLTCAP_ANALYTICAL_X_BASE):
    """Closed-form SLTCAP for charge ratio +2:-1.

    With s = sqrt(t**q+), the general SLTCAP+ equation becomes
    s**3 - rho*s - 1 = 0; therefore x = t**q+ = s(rho)**2.
    """

    def get_x(self) -> float:
        s = _divalent_cubic_root_s(self.rho)
        return s**2


class SLTCAP_DIVALENT_CASE_2(SLTCAP_ANALYTICAL_X_BASE):
    """Closed-form SLTCAP for charge ratio +1:-2.

    This is the reciprocal transform of the +2:-1 case: x = t**q+ = 1/s(-rho).
    """

    def get_x(self) -> float:
        s = _divalent_cubic_root_s(-self.rho)
        return 1.0 / s


class SLTCAP_NUMERICAL(SLTCAP_BASE):
    """Numerical solver for arbitrary valence ratios.

    Solves x**qp - x**qn - rho = 0 in log-space (x = exp(y)) with brentq.
    Here x = t = exp(-beta*phi) if q values are used directly as exponents.
    """

    def _log_equation(self, y: float) -> float:
        y = np.float64(y)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            # expm1 avoids cancellation between two values close to one when
            # |rho| is small (the neutral-solute limit).
            return np.expm1(self.qp * y) - np.expm1(self.qn * y) - self.rho

    def _get_log_bracket(self) -> tuple[float, float]:
        log_fmax = np.log(np.finfo(np.float64).max)
        lower = np.nextafter(log_fmax / self.qn, np.inf)
        upper = np.nextafter(log_fmax / self.qp, 0.0)
        if not (np.isfinite(lower) and np.isfinite(upper) and lower < upper):
            raise RuntimeError("Failed to build a finite numerical bracket.")
        return lower, upper

    def run_implementation(self) -> None:
        lower, upper = self._get_log_bracket()
        f_lower = self._log_equation(lower)
        f_upper = self._log_equation(upper)
        if not (f_lower <= 0 and f_upper >= 0):
            raise RuntimeError(
                "Could not bracket root for numerical SLTCAP estimator "
                f"(f(lower)={f_lower}, f(upper)={f_upper})."
            )

        result = root_scalar(
            self._log_equation,
            bracket=[lower, upper],
            method="brentq",
            xtol=np.nextafter(0.0, 1.0),
            rtol=4.0 * np.finfo(np.float64).eps,
            maxiter=2000,
        )
        if not result.converged:
            raise RuntimeError("Root finding did not converge.")

        root = np.exp(result.root)
        val_p = root**self.qp
        val_n = root**self.qn
        self._estimated_total_num_cations = self.np * self.cV * val_p
        self._estimated_total_num_anions = self.nn * self.cV * val_n


CHARGE_RATIO_2_ESTIMATOR = {
    1.0: SLTCAP_MONOVALENT,
    2.0: SLTCAP_DIVALENT_CASE_1,
    0.5: SLTCAP_DIVALENT_CASE_2,
}


def get_SLTCAP_estimator_class(charge_ratio: float) -> type[SLTCAP_BASE]:
    for known_ratio, estimator_class in CHARGE_RATIO_2_ESTIMATOR.items():
        if math.isclose(charge_ratio, known_ratio, rel_tol=0.0, abs_tol=1e-15):
            return estimator_class
    return SLTCAP_NUMERICAL


def run_estimator_with_optional_fallback(
    args: argparse.Namespace,
    estimator_class: type[SLTCAP_BASE],
    allow_fallback: bool = True,
) -> tuple[SLTCAP_BASE, float, float, bool]:
    try:
        estimator = estimator_class(args)
        n_p, n_n = estimator.as_floats()
        return estimator, n_p, n_n, False
    except (FloatingPointError, OverflowError, RuntimeError, TypeError, ValueError) as exc:
        if estimator_class is SLTCAP_NUMERICAL or not allow_fallback:
            raise
        logger.warning(
            "Estimator %s failed (%s). Falling back to %s.",
            estimator_class.__name__,
            exc,
            SLTCAP_NUMERICAL.__name__,
        )
        estimator = SLTCAP_NUMERICAL(args)
        n_p, n_n = estimator.as_floats()
        return estimator, n_p, n_n, True


def _select_sltcap_estimator(args: argparse.Namespace) -> tuple[type[SLTCAP_BASE], bool, str]:
    solver = getattr(args, "solver", "numerical")
    charge_ratio = get_charge_ratio(args.charge_cation, args.charge_anion)
    if solver == "numerical":
        return SLTCAP_NUMERICAL, False, solver
    if solver == "auto":
        return get_SLTCAP_estimator_class(charge_ratio), True, solver
    if solver == "analytic":
        estimator_class = get_SLTCAP_estimator_class(charge_ratio)
        if estimator_class is SLTCAP_NUMERICAL:
            raise ValueError(
                "No analytical estimator is implemented for charge ratio "
                f"{charge_ratio:g}. Use --solver numerical or --solver auto."
            )
        return estimator_class, False, solver
    raise ValueError(f"Unsupported solver: {solver}")


def estimate(args: argparse.Namespace) -> EstimationResult:
    check_arguments(args)

    solver = getattr(args, "solver", "numerical")

    if not exact_integer_neutrality_possible(args.charge_cation, args.charge_anion, args.solute_charge):
        logger.warning(
            "Exact integer neutralization is impossible because gcd(qp, |qn|) does not divide -Q. "
            "Rounded counts will retain a non-zero residual charge."
        )

    if args.mode == "an":
        estimator = SLTCAP_AN(args)
        n_p, n_n = estimator.as_floats()
        fallback_to_numerical = False
        solver = "not_applicable"
    elif args.mode == "sltcap":
        estimator_class, allow_fallback, solver = _select_sltcap_estimator(args)
        estimator, n_p, n_n, fallback_to_numerical = run_estimator_with_optional_fallback(
            args,
            estimator_class,
            allow_fallback=allow_fallback,
        )
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    residual = compute_residual_charge(args.charge_cation, args.charge_anion, n_p, n_n, args.solute_charge)
    rounded_p, rounded_n, exact_possible, rounding_method = nearest_neutral_integer_pair(
        n_p,
        n_n,
        args.charge_cation,
        args.charge_anion,
        args.solute_charge,
    )
    rounded_residual = compute_residual_charge(
        args.charge_cation,
        args.charge_anion,
        rounded_p,
        rounded_n,
        args.solute_charge,
    )

    return EstimationResult(
        mode=args.mode,
        solver=solver,
        estimator=estimator.__class__.__name__,
        fallback_to_numerical=fallback_to_numerical,
        total_cations=float(n_p),
        total_anions=float(n_n),
        rounded_cations=int(rounded_p),
        rounded_anions=int(rounded_n),
        residual_charge=float(residual),
        rounded_residual_charge=float(rounded_residual),
        exact_integer_neutrality_possible=bool(exact_possible),
        rounding_method=rounding_method,
    )


def format_text_result(result: EstimationResult) -> str:
    lines = [
        f"mode={result.mode}",
        f"solver={result.solver}",
        f"estimator={result.estimator}",
        f"fallback_to_numerical={result.fallback_to_numerical}",
        f"total_cations={result.total_cations:.12g}",
        f"total_anions={result.total_anions:.12g}",
        f"rounded_cations={result.rounded_cations}",
        f"rounded_anions={result.rounded_anions}",
        f"residual_charge={result.residual_charge:.12g}",
        f"rounded_residual_charge={result.rounded_residual_charge:.12g}",
        f"exact_integer_neutrality_possible={result.exact_integer_neutrality_possible}",
        f"rounding_method={result.rounding_method}",
    ]
    return "\n".join(lines)


def emit_result(result: EstimationResult, output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(asdict(result), sort_keys=True))
        return
    if output_format == "text":
        print(format_text_result(result))
        return
    raise ValueError(f"Unsupported output format: {output_format}")


def main(args: argparse.Namespace) -> EstimationResult:
    result = estimate(args)
    output_format = getattr(args, "output_format", None)
    if output_format is not None:
        emit_result(result, output_format)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calculate the optimal number of salt ions for a given concentration, solute charge, and solvent definition."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("mode", choices=["an", "sltcap"], help="Script mode.")
    parser.add_argument("-c", "--concentration", type=float, default=150.0, help="Salt formula-unit concentration in mM.")
    parser.add_argument("-np", "--num-cations", type=int, default=1, help="Number of cations in the binary salt formula unit.")
    parser.add_argument("-nn", "--num-anions", type=int, default=1, help="Number of anions in the binary salt formula unit.")
    parser.add_argument("-qp", "--charge-cation", type=int, default=1, help="Cation charge in elementary-charge units.")
    parser.add_argument("-qn", "--charge-anion", type=int, default=-1, help="Anion charge in elementary-charge units.")
    parser.add_argument("-Q", "--solute-charge", type=int, required=True, help="Total solute charge in elementary-charge units.")
    parser.add_argument("-V", "--solvent-volume", type=float, default=None, help="Solvent volume in liters.")
    parser.add_argument("-W", "--num-water-molecules", type=int, default=None, help="Number of water molecules.")
    parser.add_argument(
        "--solver",
        choices=["numerical", "auto", "analytic"],
        default="numerical",
        help=(
            "SLTCAP solver backend. 'numerical' is the robust default for arbitrary valence ratios; "
            "'auto' uses implemented analytical forms when available and falls back to numerical; "
            "'analytic' requires an implemented analytical form."
        ),
    )
    parser.add_argument(
        "--output-format",
        choices=["text", "json"],
        default="text",
        help="Result output format.",
    )
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging.")
    verbosity.add_argument("-q", "--quiet", action="store_true", help="Reduce logging verbosity.")
    return parser


def cli(argv: Optional[list[str]] = None) -> int:
    """Run the command-line interface and return a process exit status."""
    parser = build_parser()
    cli_args = parser.parse_args(argv)
    logging_level = logging.INFO
    if cli_args.quiet:
        logging_level = logging.WARNING
    elif cli_args.verbose:
        logging_level = logging.DEBUG
    logging.basicConfig(level=logging_level, format="%(asctime)s - %(levelname)s - %(message)s")
    try:
        main(cli_args)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(cli())
