from __future__ import annotations

import tomllib
from pathlib import Path

from wow_signal_analysis import DISPLAY_NAME, __version__

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_CITATION_PATH = _REPOSITORY_ROOT / "CITATION.cff"
_CONTRIBUTING_PATH = _REPOSITORY_ROOT / "CONTRIBUTING.md"
_PYPROJECT_PATH = _REPOSITORY_ROOT / "pyproject.toml"


def _top_level_scalar(document: str, key: str) -> str:
    prefix = f"{key}:"
    matches = tuple(
        line[len(prefix) :].strip() for line in document.splitlines() if line.startswith(prefix)
    )
    if len(matches) != 1:
        raise AssertionError(f"expected one top-level {key!r} field, found {len(matches)}")

    value = matches[0]
    if len(value) >= 2 and value[0] == value[-1] == '"':
        return value[1:-1]
    return value


def test_citation_metadata_matches_package_identity() -> None:
    citation = _CITATION_PATH.read_text(encoding="utf-8")
    project = tomllib.loads(_PYPROJECT_PATH.read_text(encoding="utf-8"))["project"]

    assert _top_level_scalar(citation, "cff-version") == "1.2.0"
    assert _top_level_scalar(citation, "type") == "software"
    assert _top_level_scalar(citation, "title") == DISPLAY_NAME
    assert _top_level_scalar(citation, "version") == __version__
    assert _top_level_scalar(citation, "license") == project["license"]
    assert project["version"] == __version__


def test_citation_metadata_names_the_repository_author() -> None:
    citation = _CITATION_PATH.read_text(encoding="utf-8")

    assert "  - family-names: Lovell\n" in citation
    assert "    given-names: Bryce\n" in citation
    assert "Please cite this software" in citation


def test_citation_metadata_preserves_research_scope() -> None:
    citation = _CITATION_PATH.read_text(encoding="utf-8")

    required_terms = (
        "evidence-bound",
        "beam-transit modeling",
        "hydrogen-line",
        "International Morse",
        "permutation controls",
        "uncertainty analysis",
        "claim boundaries",
    )
    for term in required_terms:
        assert term in citation


def test_contribution_policy_declares_all_evidence_classes() -> None:
    policy = _CONTRIBUTING_PATH.read_text(encoding="utf-8")

    for classification in (
        "**Observed**",
        "**Derived**",
        "**Compatibility**",
        "**Interpretive**",
        "**Speculative**",
    ):
        assert classification in policy


def test_contribution_policy_requires_provenance_and_complete_validation() -> None:
    policy = " ".join(_CONTRIBUTING_PATH.read_text(encoding="utf-8").split())

    required_requirements = (
        "Update the matching manifest",
        "Recalculate the exact SHA-256 digest",
        "Add or update tamper-detection tests",
        "python check_green.py",
        "Do not report the repository as green",
        "Never substitute a visual inspection",
        "Do not hand-edit files under `artifacts/generated/`",
    )
    for requirement in required_requirements:
        assert requirement in policy


def test_contribution_policy_preserves_interpretive_limits() -> None:
    policy = " ".join(_CONTRIBUTING_PATH.read_text(encoding="utf-8").split())

    assert "must never be presented as recovered transmitter intent" in policy
    assert "must not be described as transmitted text" in policy
    assert "must not be generalized into source-origin or intent claims" in policy
