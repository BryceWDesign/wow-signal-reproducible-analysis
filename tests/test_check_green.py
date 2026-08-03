from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _REPOSITORY_ROOT / "check_green.py"


def _load_check_green() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "repository_check_green",
        _SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise AssertionError(
            "unable to load check_green.py"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def check_green() -> ModuleType:
    return _load_check_green()


def test_default_gate_order_is_deterministic(
    tmp_path: Path,
    check_green: ModuleType,
) -> None:
    steps = check_green.build_check_steps(
        _REPOSITORY_ROOT,
        python_executable="python-test",
        build_output_directory=tmp_path,
    )

    assert tuple(
        step.step_id for step in steps
    ) == (
        "ruff-check",
        "ruff-format",
        "mypy",
        "pytest",
        "package-build",
        "repository-contract",
    )
    assert steps[0].command == (
        "python-test",
        "-m",
        "ruff",
        "check",
        ".",
    )
    assert steps[-1].command[:4] == (
        "python-test",
        "-m",
        "wow_signal_analysis",
        "verify",
    )
    assert (
        str(_REPOSITORY_ROOT.resolve())
        in steps[-1].command
    )


def test_build_step_can_be_skipped(
    tmp_path: Path,
    check_green: ModuleType,
) -> None:
    steps = check_green.build_check_steps(
        _REPOSITORY_ROOT,
        build_output_directory=tmp_path,
        include_build=False,
    )

    assert "package-build" not in {
        step.step_id for step in steps
    }
    assert len(steps) == 5


def test_successful_gate_runs_every_step(
    monkeypatch: pytest.MonkeyPatch,
    check_green: ModuleType,
) -> None:
    commands: list[tuple[str, ...]] = []

    def fake_run(
        command: tuple[str, ...],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        commands.append(command)
        assert (
            kwargs["cwd"]
            == _REPOSITORY_ROOT.resolve()
        )
        assert kwargs["check"] is False

        environment = kwargs["env"]
        assert (
            str(_REPOSITORY_ROOT / "src")
            in environment["PYTHONPATH"]
        )
        return subprocess.CompletedProcess(
            command,
            0,
        )

    monkeypatch.setattr(
        check_green.subprocess,
        "run",
        fake_run,
    )

    report = check_green.run_green_checks(
        _REPOSITORY_ROOT,
        python_executable="python-test",
    )

    assert report.passed
    assert len(report.results) == 6
    assert len(commands) == 6
    assert report.failed_results == ()


def test_gate_stops_at_first_failure_by_default(
    monkeypatch: pytest.MonkeyPatch,
    check_green: ModuleType,
) -> None:
    return_codes = iter(
        (0, 7, 0, 0, 0, 0)
    )

    def fake_run(
        command: tuple[str, ...],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(
            command,
            next(return_codes),
        )

    monkeypatch.setattr(
        check_green.subprocess,
        "run",
        fake_run,
    )

    report = check_green.run_green_checks(
        _REPOSITORY_ROOT,
        include_build=True,
    )

    assert not report.passed
    assert len(report.results) == 2
    assert report.planned_step_count == 6
    assert tuple(
        result.step.step_id
        for result in report.failed_results
    ) == ("ruff-format",)


def test_continue_on_failure_runs_the_complete_gate(
    monkeypatch: pytest.MonkeyPatch,
    check_green: ModuleType,
) -> None:
    return_codes = iter(
        (1, 0, 2, 0, 0)
    )

    def fake_run(
        command: tuple[str, ...],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(
            command,
            next(return_codes),
        )

    monkeypatch.setattr(
        check_green.subprocess,
        "run",
        fake_run,
    )

    report = check_green.run_green_checks(
        _REPOSITORY_ROOT,
        include_build=False,
        continue_on_failure=True,
    )

    assert not report.passed
    assert len(report.results) == 5
    assert tuple(
        result.step.step_id
        for result in report.failed_results
    ) == (
        "ruff-check",
        "mypy",
    )


def test_invalid_repository_root_fails_closed(
    tmp_path: Path,
    check_green: ModuleType,
) -> None:
    with pytest.raises(
        check_green.GreenCheckError,
        match="existing directory",
    ):
        check_green.build_check_steps(
            tmp_path / "missing",
            build_output_directory=tmp_path,
        )


def test_main_returns_green_and_red_statuses(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    check_green: ModuleType,
) -> None:
    successful_step = check_green.CheckStep(
        step_id="test",
        label="Test",
        command=("python", "-V"),
    )
    successful_report = (
        check_green.GreenCheckReport(
            planned_step_count=1,
            results=(
                check_green.CheckResult(
                    step=successful_step,
                    return_code=0,
                ),
            ),
        )
    )
    monkeypatch.setattr(
        check_green,
        "run_green_checks",
        lambda *args, **kwargs: successful_report,
    )

    assert (
        check_green.main(
            ["--root", str(_REPOSITORY_ROOT)]
        )
        == 0
    )
    assert (
        "QUALITY GATE: GREEN (1/1)"
        in capsys.readouterr().out
    )

    failed_report = check_green.GreenCheckReport(
        planned_step_count=1,
        results=(
            check_green.CheckResult(
                step=successful_step,
                return_code=3,
            ),
        ),
    )
    monkeypatch.setattr(
        check_green,
        "run_green_checks",
        lambda *args, **kwargs: failed_report,
    )

    assert (
        check_green.main(
            ["--root", str(_REPOSITORY_ROOT)]
        )
        == 1
    )
    assert (
        "QUALITY GATE: RED (test"
        in capsys.readouterr().err
    )
