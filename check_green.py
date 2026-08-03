"""Run the complete local quality gate without requiring a shell-specific script."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Final

_PROJECT_PACKAGE: Final = "wow_signal_analysis"


class GreenCheckError(ValueError):
    """Raised when the quality-gate configuration is invalid."""


@dataclass(frozen=True, slots=True)
class CheckStep:
    """One ordered subprocess invocation in the repository quality gate."""

    step_id: str
    label: str
    command: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.step_id or any(
            character.isspace() for character in self.step_id
        ):
            raise GreenCheckError(
                "step_id must be non-empty and contain no whitespace"
            )
        if not self.label.strip():
            raise GreenCheckError("step label must be non-empty")
        if not self.command or any(
            not argument for argument in self.command
        ):
            raise GreenCheckError(
                "step command must contain only non-empty arguments"
            )


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Return status for one completed quality-gate step."""

    step: CheckStep
    return_code: int

    def __post_init__(self) -> None:
        if not isinstance(self.return_code, int) or isinstance(
            self.return_code,
            bool,
        ):
            raise GreenCheckError("return_code must be an integer")

    @property
    def passed(self) -> bool:
        """Return whether the step exited successfully."""

        return self.return_code == 0


@dataclass(frozen=True, slots=True)
class GreenCheckReport:
    """Ordered results from one quality-gate execution."""

    planned_step_count: int
    results: tuple[CheckResult, ...]

    def __post_init__(self) -> None:
        if self.planned_step_count <= 0:
            raise GreenCheckError(
                "planned_step_count must be positive"
            )
        if len(self.results) > self.planned_step_count:
            raise GreenCheckError(
                "results cannot exceed planned_step_count"
            )

        step_ids = tuple(
            result.step.step_id for result in self.results
        )
        if len(set(step_ids)) != len(step_ids):
            raise GreenCheckError(
                "completed step IDs must be unique"
            )

    @property
    def passed(self) -> bool:
        """Return whether every planned step completed successfully."""

        return (
            len(self.results) == self.planned_step_count
            and all(result.passed for result in self.results)
        )

    @property
    def failed_results(self) -> tuple[CheckResult, ...]:
        """Return completed steps that exited unsuccessfully."""

        return tuple(
            result
            for result in self.results
            if not result.passed
        )


def build_check_steps(
    repository_root: Path,
    *,
    python_executable: str = sys.executable,
    build_output_directory: Path,
    include_build: bool = True,
) -> tuple[CheckStep, ...]:
    """Construct the deterministic quality-gate command sequence."""

    root = repository_root.resolve()
    if not root.is_dir():
        raise GreenCheckError(
            f"repository_root must be an existing directory: {root}"
        )
    if not python_executable.strip():
        raise GreenCheckError(
            "python_executable must be non-empty"
        )

    steps = [
        CheckStep(
            step_id="ruff-check",
            label="Ruff lint",
            command=(
                python_executable,
                "-m",
                "ruff",
                "check",
                ".",
            ),
        ),
        CheckStep(
            step_id="ruff-format",
            label="Ruff formatting",
            command=(
                python_executable,
                "-m",
                "ruff",
                "format",
                "--check",
                ".",
            ),
        ),
        CheckStep(
            step_id="mypy",
            label="Mypy strict typing",
            command=(
                python_executable,
                "-m",
                "mypy",
            ),
        ),
        CheckStep(
            step_id="pytest",
            label="Pytest suite",
            command=(
                python_executable,
                "-m",
                "pytest",
            ),
        ),
    ]

    if include_build:
        steps.append(
            CheckStep(
                step_id="package-build",
                label="Wheel and source-distribution build",
                command=(
                    python_executable,
                    "-m",
                    "build",
                    "--outdir",
                    str(build_output_directory.resolve()),
                ),
            )
        )

    steps.extend(
        (
            CheckStep(
                step_id="repository-contract",
                label="Canonical repository contract",
                command=(
                    python_executable,
                    "-m",
                    _PROJECT_PACKAGE,
                    "verify",
                    "--root",
                    str(root),
                    "--json",
                ),
            ),
            CheckStep(
                step_id="release-reproduction",
                label="Isolated release reproduction",
                command=(
                    python_executable,
                    "-m",
                    f"{_PROJECT_PACKAGE}.release_verification",
                    "--root",
                    str(root),
                    "--json",
                ),
            ),
        )
    )

    return tuple(steps)


def run_green_checks(
    repository_root: Path,
    *,
    python_executable: str = sys.executable,
    include_build: bool = True,
    continue_on_failure: bool = False,
) -> GreenCheckReport:
    """Execute every configured quality gate and return its exact status."""

    root = repository_root.resolve()
    environment = os.environ.copy()
    source_directory = str(root / "src")
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_directory
        if not existing_pythonpath
        else source_directory
        + os.pathsep
        + existing_pythonpath
    )

    with TemporaryDirectory(
        prefix="wow-signal-build-"
    ) as temporary_directory:
        steps = build_check_steps(
            root,
            python_executable=python_executable,
            build_output_directory=Path(
                temporary_directory
            ),
            include_build=include_build,
        )
        results: list[CheckResult] = []

        for index, step in enumerate(steps, start=1):
            print(
                f"==> [{index}/{len(steps)}] {step.label}",
                flush=True,
            )
            print(
                f"$ {_display_command(step.command)}",
                flush=True,
            )

            completed = subprocess.run(
                step.command,
                cwd=root,
                env=environment,
                check=False,
            )
            result = CheckResult(
                step=step,
                return_code=completed.returncode,
            )
            results.append(result)

            status = (
                "PASS"
                if result.passed
                else f"FAIL ({result.return_code})"
            )
            print(
                f"{status}: {step.step_id}",
                flush=True,
            )

            if (
                not result.passed
                and not continue_on_failure
            ):
                break

    return GreenCheckReport(
        planned_step_count=len(steps),
        results=tuple(results),
    )


def main(argv: list[str] | None = None) -> int:
    """Run the local quality gate and return a process-compatible status."""

    parser = argparse.ArgumentParser(
        description=(
            "Run lint, formatting, typing, tests, package build, "
            "canonical verification, and isolated release reproduction."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help=(
            "Repository root. Defaults to the directory containing "
            "this script."
        ),
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used for every subprocess.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help=(
            "Skip wheel and source-distribution construction."
        ),
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Run remaining checks after a failed step.",
    )
    arguments = parser.parse_args(argv)

    try:
        report = run_green_checks(
            arguments.root,
            python_executable=arguments.python,
            include_build=not arguments.skip_build,
            continue_on_failure=(
                arguments.continue_on_failure
            ),
        )
    except (GreenCheckError, OSError) as error:
        print(
            f"QUALITY GATE ERROR: {error}",
            file=sys.stderr,
        )
        return 2

    if report.passed:
        print(
            "QUALITY GATE: GREEN "
            f"({len(report.results)}/"
            f"{report.planned_step_count})"
        )
        return 0

    failed = ", ".join(
        result.step.step_id
        for result in report.failed_results
    )
    if not failed:
        failed = "incomplete execution"

    print(
        f"QUALITY GATE: RED ({failed}; "
        f"completed {len(report.results)}/"
        f"{report.planned_step_count})",
        file=sys.stderr,
    )
    return 1


def _display_command(
    command: tuple[str, ...],
) -> str:
    return " ".join(
        (
            f'"{argument}"'
            if any(
                character.isspace()
                for character in argument
            )
            else argument
        )
        for argument in command
    )


if __name__ == "__main__":
    raise SystemExit(main())
