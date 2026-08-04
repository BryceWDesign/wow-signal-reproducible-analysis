from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from wow_signal_analysis.analysis_snapshot import (
    AnalysisSnapshot,
    SnapshotConfig,
    build_analysis_snapshot,
)
from wow_signal_analysis.beam_model import GaussianSearchConfig
from wow_signal_analysis.visualization import (
    ANALYSIS_FIGURE_SET_ID,
    BEAM_FIT_FIGURE_ID,
    MODEL_COMPARISON_FIGURE_ID,
    AnalysisFigureSet,
    SvgFigure,
    VisualizationError,
    build_analysis_figures,
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
def figures(
    snapshot: AnalysisSnapshot,
) -> AnalysisFigureSet:
    return build_analysis_figures(snapshot)


def test_figure_set_identity_and_order_are_deterministic(
    snapshot: AnalysisSnapshot,
    figures: AnalysisFigureSet,
) -> None:
    second = build_analysis_figures(snapshot)

    assert figures == second
    assert figures.figure_set_id == ANALYSIS_FIGURE_SET_ID
    assert figures.analysis_id == snapshot.analysis_id
    assert tuple(figure.figure_id for figure in figures.figures) == (
        BEAM_FIT_FIGURE_ID,
        MODEL_COMPARISON_FIGURE_ID,
    )


def test_figures_are_self_contained_accessible_svg(
    figures: AnalysisFigureSet,
) -> None:
    for figure in figures.figures:
        assert figure.svg.startswith('<svg xmlns="http://www.w3.org/2000/svg"')
        assert figure.svg.endswith("</svg>\n")
        assert "\r" not in figure.svg
        assert "<script" not in figure.svg.lower()
        assert " href=" not in figure.svg.lower()
        assert "xlink:href" not in figure.svg.lower()
        assert f'<title id="{figure.figure_id}-title">' in figure.svg
        assert f'<desc id="{figure.figure_id}-description">' in figure.svg
        assert figure.content == figure.svg.encode("utf-8")
        assert figure.byte_count == len(figure.content)


def test_beam_figure_preserves_observations_and_fit_context(
    figures: AnalysisFigureSet,
) -> None:
    figure = figures.beam_fit

    assert "Sequence 6EQUJ5" in figure.svg
    assert "R²=" in figure.svg
    assert figure.svg.count('<circle class="observed-point"') == 7
    assert "Sample 0: elapsed 0 seconds, observed SNR 6.5" in figure.svg
    assert "Sample 5: elapsed 60 seconds, observed SNR 5.5" in figure.svg
    assert "Gaussian fit" in figure.svg


def test_model_comparison_figure_preserves_ranking_and_errors(
    figures: AnalysisFigureSet,
) -> None:
    figure = figures.model_comparison

    for model in (
        "gaussian-transit",
        "quadratic",
        "constant",
        "affine",
    ):
        assert model in figure.svg

    assert figure.svg.count('<rect class="model-bar') == 4
    assert "2.013" in figure.svg
    assert "6.596" in figure.svg
    assert "11.250" in figure.svg
    assert "15.953" in figure.svg
    assert "it does not identify the emitter" in figure.svg


def test_figure_lookup_fails_closed(
    figures: AnalysisFigureSet,
) -> None:
    assert figures.figure_by_id(BEAM_FIT_FIGURE_ID) is figures.beam_fit
    assert figures.figure_by_id(MODEL_COMPARISON_FIGURE_ID) is figures.model_comparison

    with pytest.raises(
        VisualizationError,
        match="found 0",
    ):
        figures.figure_by_id("missing-figure")


def test_figure_metadata_rejects_identity_and_unsafe_content(
    figures: AnalysisFigureSet,
) -> None:
    with pytest.raises(
        VisualizationError,
        match="analysis_id must be",
    ):
        replace(
            figures.beam_fit,
            analysis_id="different-analysis",
        )

    with pytest.raises(
        VisualizationError,
        match="executable scripts",
    ):
        SvgFigure(
            figure_id=BEAM_FIT_FIGURE_ID,
            analysis_id=figures.analysis_id,
            title=figures.beam_fit.title,
            description=figures.beam_fit.description,
            svg=figures.beam_fit.svg.replace(
                "</svg>",
                "<script>unsafe()</script></svg>",
            ),
        )

    with pytest.raises(
        VisualizationError,
        match="external resources",
    ):
        SvgFigure(
            figure_id=BEAM_FIT_FIGURE_ID,
            analysis_id=figures.analysis_id,
            title=figures.beam_fit.title,
            description=figures.beam_fit.description,
            svg=figures.beam_fit.svg.replace(
                "<style>",
                ('<image href="https://example.invalid/figure.svg" /><style>'),
            ),
        )


def test_figure_set_rejects_component_drift(
    figures: AnalysisFigureSet,
) -> None:
    with pytest.raises(
        VisualizationError,
        match="figure_set_id must be",
    ):
        replace(
            figures,
            figure_set_id="different-figure-set",
        )

    with pytest.raises(
        VisualizationError,
        match="beam_fit figure_id",
    ):
        replace(
            figures,
            beam_fit=figures.model_comparison,
        )
