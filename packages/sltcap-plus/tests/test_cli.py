from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "sltcap-plus.py"


def run_cli(*arguments: str, module: bool = False) -> subprocess.CompletedProcess[str]:
    command = [sys.executable]
    command += ["-m", "sltcap_plus"] if module else [str(SCRIPT)]
    return subprocess.run(
        [*command, *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize("module", [False, True])
def test_help_works_for_compatibility_script_and_module(module: bool) -> None:
    completed = run_cli("--help", module=module)
    assert completed.returncode == 0
    assert "SLTCAP solver backend" in completed.stdout
    assert completed.stderr == ""


@pytest.mark.parametrize("module", [False, True])
def test_version_works_for_compatibility_script_and_module(module: bool) -> None:
    completed = run_cli("--version", module=module)
    assert completed.returncode == 0
    assert completed.stdout.strip().endswith("0.1.0")
    assert completed.stderr == ""


def test_json_output_is_machine_readable() -> None:
    completed = run_cli(
        "sltcap",
        "-Q",
        "-10",
        "-W",
        "5550",
        "--solver",
        "analytic",
        "--output-format",
        "json",
        "--quiet",
    )
    assert completed.returncode == 0
    result = json.loads(completed.stdout)
    assert result["estimator"] == "SLTCAP_MONOVALENT"
    assert result["rounded_residual_charge"] == 0.0
    assert completed.stderr == ""


def test_cli_supports_mgcl2() -> None:
    completed = run_cli(
        "sltcap",
        "-Q",
        "-10",
        "-W",
        "5550",
        "-c",
        "100",
        "-np",
        "1",
        "-nn",
        "2",
        "-qp",
        "2",
        "-qn",
        "-1",
        "--solver",
        "analytic",
        "--quiet",
    )
    assert completed.returncode == 0
    assert "estimator=SLTCAP_DIVALENT_CASE_1" in completed.stdout
    assert "rounded_residual_charge=0" in completed.stdout


def test_invalid_input_is_a_clean_cli_error() -> None:
    completed = run_cli("sltcap", "-Q", "1", "-W", "100", "-c", "nan")
    assert completed.returncode == 2
    assert "Concentration must be finite" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_conflicting_verbosity_flags_are_rejected() -> None:
    completed = run_cli("sltcap", "-Q", "1", "-W", "100", "--quiet", "--verbose")
    assert completed.returncode == 2
    assert "not allowed with argument" in completed.stderr


def test_exactly_one_solvent_definition_is_required() -> None:
    neither = run_cli("sltcap", "-Q", "1")
    both = run_cli("sltcap", "-Q", "1", "-W", "100", "-V", "1e-24")
    assert neither.returncode == 2
    assert both.returncode == 2
    assert "Either solvent volume" in neither.stderr
    assert "but not both" in both.stderr
