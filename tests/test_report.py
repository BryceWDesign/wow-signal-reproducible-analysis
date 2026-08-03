from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

from wow_signal_analysis.analysis_snapshot import (
    ANALYSIS_SNAPSHOT_ID,
    AnalysisSnapshot,
    SnapshotConfig,
    build_analysis_snapshot,
)
from wow_signal_analysis.beam_model import GaussianSearchConfig
from wow_signal_analysis.report import (
    ANALYSIS_REPORT_ID,
    ANALYSIS_REPORT_TITLE,
    RenderedAnalysisReport,
    ReportRenderError,
    build_analysis_report,
    render_analysis_report,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_FAST_CONFIG = SnapshotConfig(
    gaussian_search=GaussianSearchConfig(
        grid_points=21,
        refinement_rounds=3,
    )
)


@pytest.fixture(scope="module")
def snapshot() -> AnalysisSnapshot:
    return build_analysis_snapshot(
        _REPOSITORY_ROOT,
        config=_FAST_CONFIG,
    )


@pytest.fixture(scope="module")
def report(
    snapshot: AnalysisSnapshot,
) -> RenderedAnalysisReport:
    return build_analysis_report(snapshot)


def test_report_identity_and_content_digest_are_deterministic(
    snapshot: AnalysisSnapshot,
    report: RenderedAnalysisReport,
) -> None:
    second = build_analysis_report(snapshot)

    assert report == second
    assert report.report_id == ANALYSIS_REPORT_ID
    assert report.analysis_id == ANALYSIS_SNAPSHOT_ID
    assert report.markdown == render_analysis_report(snapshot)
    assert report.content == report.markdown.encode("utf-8")
    assert report.byte_count == len(report.content)
    assert report.sha256_hex == sha256(report.content).hexdigest()
    assert len(report.sha256_hex) == 64


def test_report_has_canonical_title_and_stable_line_endings(
    report: RenderedAnalysisReport,
) -> None:
    assert report.markdown.startswith(
        f"# {ANALYSIS_REPORT_TITLE}\n"
    )
    assert report.markdown.endswith("\n")
    assert not report.markdown.endswith("\n\n")
    assert "\r" not in report.markdown


def test_report_contains_the_complete_five_layer_summary(
    report: RenderedAnalysisReport,
) -> None:
    assert "1. **PRESENCE**" in report.markdown
    assert "2. **TRANSIT**" in report.markdown
    assert "3. **BEACON**" in report.markdown
    assert "4. **HYDROGEN**" in report.markdown
    assert "5. **QUESTION**" in report.markdown

    assert "`6EQUJ5`" in report.markdown
    assert "`+++--`" in report.markdown
    assert "`compatible-not-proven`" in report.markdown
    assert "`..--..` (`?`)" in report.markdown
    assert "`--..--` (`,`)" in report.markdown


def test_report_preserves_exact_morse_null_fractions(
    report: RenderedAnalysisReport,
) -> None:
    assert "96/720 (2/15)" in report.markdown
    assert "192/20160 (1/105)" in report.markdown
    assert (
        "does not establish intentional Morse transmission"
        in report.markdown
    )


def test_report_contains_all_major_analysis_sections(
    report: RenderedAnalysisReport,
) -> None:
    expected_sections = (
        "## Reproducibility scope",
        "## Five-layer evidence summary",
        "## Canonical observation",
        "## Gaussian beam-transit fit",
        "## Leave-one-out shape-model comparison",
        "## Printer-bin corner sensitivity",
        "## Exhaustive threshold and Morse correspondence",
        "## Neutral-hydrogen frequency context",
        "## Evidence-bound claim ledger",
        "## Hypothesis and falsification matrix",
        "## Interpretive limits",
    )

    for section in expected_sections:
        assert section in report.markdown


def test_report_keeps_speculative_conclusions_unestablished(
    report: RenderedAnalysisReport,
) -> None:
    assert "`speculative-intentional-message`" in report.markdown
    assert "`speculative-extraterrestrial-technology`" in report.markdown
    assert "Artificial-carrier compatibility does not establish" in report.markdown
    assert "extraterrestrial technology" in report.markdown
    assert "do not claim a decoded alien message" in report.markdown
    assert "do not establish an intentional message" in report.markdown


def test_report_records_canonical_claim_and_hypothesis_counts(
    report: RenderedAnalysisReport,
) -> None:
    assert "| observed | 1 |" in report.markdown
    assert "| derived | 7 |" in report.markdown
    assert "| compatibility | 1 |" in report.markdown
    assert "| interpretive | 1 |" in report.markdown
    assert "| speculative | 2 |" in report.markdown

    assert "| supported-as-model | 1 |" in report.markdown
    assert "| compatible-not-proven | 1 |" in report.markdown
    assert "| not-discriminated | 1 |" in report.markdown
    assert "| not-established | 2 |" in report.markdown


def test_report_builder_requires_a_real_snapshot() -> None:
    with pytest.raises(
        ReportRenderError,
        match="must be an AnalysisSnapshot",
    ):
        build_analysis_report(object())  # type: ignore[arg-type]


def test_rendered_report_rejects_identity_and_format_drift(
    report: RenderedAnalysisReport,
) -> None:
    with pytest.raises(ReportRenderError, match="report_id must be"):
        replace(
            report,
            report_id="different-report",
        )

    with pytest.raises(ReportRenderError, match="analysis_id must be"):
        replace(
            report,
            analysis_id="different-analysis",
        )

    with pytest.raises(ReportRenderError, match="canonical report title"):
        replace(
            report,
            markdown="# Different title\n",
        )

    with pytest.raises(ReportRenderError, match="end with exactly one"):
        replace(
            report,
            markdown=report.markdown.rstrip("\n"),
        )

    with pytest.raises(ReportRenderError, match="blank trailing line"):
        replace(
            report,
            markdown=report.markdown + "\n",
        )

    with pytest.raises(ReportRenderError, match="LF line endings"):
        replace(
            report,
            markdown=report.markdown.replace("\n", "\r\n"),
        )
