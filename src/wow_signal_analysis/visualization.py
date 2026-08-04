"""Dependency-free deterministic SVG figures for the canonical analysis."""

from __future__ import annotations

import html
import math
import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Final

from wow_signal_analysis.analysis_snapshot import (
    ANALYSIS_SNAPSHOT_ID,
    AnalysisSnapshot,
)

ANALYSIS_FIGURE_SET_ID: Final = "wow-signal-analysis-figures-v1"
BEAM_FIT_FIGURE_ID: Final = "wow-signal-beam-fit-v1"
MODEL_COMPARISON_FIGURE_ID: Final = "wow-signal-model-comparison-v1"

_SVG_WIDTH: Final = 960
_SVG_HEIGHT: Final = 540
_IDENTIFIER_PATTERN: Final = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class VisualizationError(ValueError):
    """Raised when an analysis figure cannot be rendered safely."""


@dataclass(frozen=True, slots=True)
class SvgFigure:
    """One accessible deterministic SVG figure."""

    figure_id: str
    analysis_id: str
    title: str
    description: str
    svg: str

    def __post_init__(self) -> None:
        if not _IDENTIFIER_PATTERN.fullmatch(self.figure_id):
            raise VisualizationError("figure_id must be a lowercase hyphen-delimited identifier")
        if self.analysis_id != ANALYSIS_SNAPSHOT_ID:
            raise VisualizationError(f"analysis_id must be {ANALYSIS_SNAPSHOT_ID!r}")
        if not self.title.strip() or not self.description.strip():
            raise VisualizationError("title and description must be non-empty")
        if not self.svg.startswith("<svg "):
            raise VisualizationError("svg must begin with an SVG root element")
        if self.svg.endswith("</svg>\n\n"):
            raise VisualizationError("svg must not contain a blank trailing line")
        if not self.svg.endswith("</svg>\n"):
            raise VisualizationError("svg must end with exactly one line terminator")
        if "\r" in self.svg:
            raise VisualizationError("svg must use LF line endings")
        if "<script" in self.svg.lower():
            raise VisualizationError("svg must not contain executable scripts")
        if " href=" in self.svg.lower() or "xlink:href" in self.svg.lower():
            raise VisualizationError("svg must not reference external resources")

        title_id = f"{self.figure_id}-title"
        description_id = f"{self.figure_id}-description"
        if f'aria-labelledby="{title_id} {description_id}"' not in self.svg:
            raise VisualizationError("svg must bind its accessible title and description")
        if f'<title id="{title_id}">{html.escape(self.title)}</title>' not in self.svg:
            raise VisualizationError("svg title does not match figure metadata")
        if f'<desc id="{description_id}">{html.escape(self.description)}</desc>' not in self.svg:
            raise VisualizationError("svg description does not match figure metadata")

    @property
    def content(self) -> bytes:
        """Return UTF-8 encoded SVG bytes."""

        return self.svg.encode("utf-8")

    @property
    def byte_count(self) -> int:
        """Return the exact UTF-8 byte count."""

        return len(self.content)


@dataclass(frozen=True, slots=True)
class AnalysisFigureSet:
    """Canonical figure set for one analysis snapshot."""

    figure_set_id: str
    analysis_id: str
    beam_fit: SvgFigure
    model_comparison: SvgFigure

    def __post_init__(self) -> None:
        if self.figure_set_id != ANALYSIS_FIGURE_SET_ID:
            raise VisualizationError(f"figure_set_id must be {ANALYSIS_FIGURE_SET_ID!r}")
        if self.analysis_id != ANALYSIS_SNAPSHOT_ID:
            raise VisualizationError(f"analysis_id must be {ANALYSIS_SNAPSHOT_ID!r}")
        if self.beam_fit.figure_id != BEAM_FIT_FIGURE_ID:
            raise VisualizationError(f"beam_fit figure_id must be {BEAM_FIT_FIGURE_ID!r}")
        if self.model_comparison.figure_id != MODEL_COMPARISON_FIGURE_ID:
            raise VisualizationError(
                f"model_comparison figure_id must be {MODEL_COMPARISON_FIGURE_ID!r}"
            )
        if any(figure.analysis_id != self.analysis_id for figure in self.figures):
            raise VisualizationError("all figures must belong to the figure set analysis_id")
        if len({figure.figure_id for figure in self.figures}) != len(self.figures):
            raise VisualizationError("figure IDs must be unique")

    @property
    def figures(self) -> tuple[SvgFigure, ...]:
        """Return figures in deterministic presentation order."""

        return (self.beam_fit, self.model_comparison)

    def figure_by_id(self, figure_id: str) -> SvgFigure:
        """Return one unique figure by its canonical identifier."""

        matches = tuple(figure for figure in self.figures if figure.figure_id == figure_id)
        if len(matches) != 1:
            raise VisualizationError(f"expected one figure for {figure_id!r}, found {len(matches)}")
        return matches[0]


def build_analysis_figures(
    snapshot: AnalysisSnapshot,
) -> AnalysisFigureSet:
    """Render the canonical beam-fit and model-comparison figures."""

    if not isinstance(snapshot, AnalysisSnapshot):
        raise VisualizationError("snapshot must be an AnalysisSnapshot")

    return AnalysisFigureSet(
        figure_set_id=ANALYSIS_FIGURE_SET_ID,
        analysis_id=snapshot.analysis_id,
        beam_fit=_render_beam_fit(snapshot),
        model_comparison=_render_model_comparison(snapshot),
    )


def _render_beam_fit(snapshot: AnalysisSnapshot) -> SvgFigure:
    title = "Wow! signal midpoint samples and Gaussian beam-transit fit"
    description = (
        "Six midpoint signal-to-noise observations from 6EQUJ5 are shown with "
        "the deterministic zero-baseline Gaussian fit."
    )
    figure_id = BEAM_FIT_FIGURE_ID
    plot = _PlotArea(
        left=88.0,
        top=62.0,
        right=924.0,
        bottom=454.0,
    )

    samples = snapshot.gaussian_fit.samples
    times = tuple(float(sample.elapsed_seconds) for sample in samples)
    observed = tuple(float(sample.observed_snr) for sample in samples)
    predicted = tuple(sample.predicted_snr for sample in samples)

    x_min = min(times)
    x_max = max(times)
    y_max = _rounded_axis_max(max(*observed, *predicted))

    lines = _svg_header(figure_id, title, description)
    lines.extend(_style_lines())
    lines.extend(
        _plot_background(
            plot,
            heading="Observed midpoint SNR and fitted beam response",
            note=(
                f"Sequence {snapshot.dataset.printer_sequence}; "
                f"R²={snapshot.gaussian_fit.coefficient_of_determination:.4f}"
            ),
        )
    )
    lines.extend(
        _axis_lines(
            plot,
            x_ticks=tuple(
                (value, _decimal_label(sample.elapsed_seconds))
                for value, sample in zip(
                    times,
                    samples,
                    strict=True,
                )
            ),
            y_ticks=_linear_ticks(0.0, y_max, 5),
            x_min=x_min,
            x_max=x_max,
            y_min=0.0,
            y_max=y_max,
            x_label="Elapsed time (seconds)",
            y_label="Signal-to-noise ratio",
        )
    )

    predicted_points = " ".join(
        f"{_x_position(value, x_min, x_max, plot):.2f},"
        f"{_y_position(prediction, 0.0, y_max, plot):.2f}"
        for value, prediction in zip(
            times,
            predicted,
            strict=True,
        )
    )
    lines.append(f'<polyline class="fit-line" points="{predicted_points}" />')

    for sample, time, value in zip(
        samples,
        times,
        observed,
        strict=True,
    ):
        x = _x_position(time, x_min, x_max, plot)
        y = _y_position(value, 0.0, y_max, plot)
        label = (
            f"Sample {sample.sample_index}: elapsed "
            f"{sample.elapsed_seconds} seconds, "
            f"observed SNR {sample.observed_snr}"
        )
        lines.extend(
            (
                (f'<circle class="observed-point" cx="{x:.2f}" cy="{y:.2f}" r="6">'),
                f"  <title>{html.escape(label)}</title>",
                "</circle>",
            )
        )

    lines.extend(
        (
            '<g class="legend" aria-label="Legend">',
            ('  <circle class="observed-point" cx="706" cy="89" r="5" />'),
            '  <text x="719" y="94">Observed midpoint</text>',
            ('  <line class="fit-line" x1="706" y1="111" x2="739" y2="111" />'),
            '  <text x="748" y="116">Gaussian fit</text>',
            "</g>",
            "</svg>",
        )
    )

    return SvgFigure(
        figure_id=figure_id,
        analysis_id=snapshot.analysis_id,
        title=title,
        description=description,
        svg="\n".join(lines) + "\n",
    )


def _render_model_comparison(
    snapshot: AnalysisSnapshot,
) -> SvgFigure:
    title = "Leave-one-out prediction error by candidate shape model"
    description = (
        "Held-out root mean squared prediction error is compared for constant, "
        "affine, quadratic, and Gaussian-transit models; lower is better."
    )
    figure_id = MODEL_COMPARISON_FIGURE_ID
    plot = _PlotArea(
        left=96.0,
        top=62.0,
        right=924.0,
        bottom=430.0,
    )
    results = snapshot.model_comparison.ranked_by_prediction_error
    values = tuple(result.root_mean_squared_prediction_error for result in results)
    y_max = _rounded_axis_max(max(values))

    lines = _svg_header(figure_id, title, description)
    lines.extend(_style_lines())
    lines.extend(
        _plot_background(
            plot,
            heading="Leave-one-out root mean squared prediction error",
            note=("Lower held-out error indicates better prediction within this model set"),
        )
    )
    lines.extend(
        _axis_lines(
            plot,
            x_ticks=(),
            y_ticks=_linear_ticks(0.0, y_max, 5),
            x_min=0.0,
            x_max=float(len(results)),
            y_min=0.0,
            y_max=y_max,
            x_label="Candidate model",
            y_label="Held-out RMSE",
        )
    )

    slot_width = plot.width / len(results)
    bar_width = slot_width * 0.58
    for index, result in enumerate(results):
        value = result.root_mean_squared_prediction_error
        x = plot.left + index * slot_width + (slot_width - bar_width) / 2.0
        y = _y_position(value, 0.0, y_max, plot)
        height = plot.bottom - y
        model_label = result.model.value

        lines.extend(
            (
                (
                    f'<rect class="model-bar rank-{index + 1}" '
                    f'x="{x:.2f}" y="{y:.2f}" '
                    f'width="{bar_width:.2f}" height="{height:.2f}">'
                ),
                (f"  <title>{html.escape(model_label)}: held-out RMSE {value:.6f}</title>"),
                "</rect>",
                (
                    f'<text class="bar-value" '
                    f'x="{x + bar_width / 2.0:.2f}" '
                    f'y="{max(plot.top + 18.0, y - 9.0):.2f}">'
                    f"{value:.3f}</text>"
                ),
                (
                    f'<text class="category-label" '
                    f'x="{x + bar_width / 2.0:.2f}" '
                    f'y="{plot.bottom + 28.0:.2f}">'
                    f"{html.escape(model_label)}</text>"
                ),
            )
        )

    lines.extend(
        (
            (
                f'<text class="annotation" '
                f'x="{plot.left:.2f}" y="492">'
                "Ranking is descriptive of four predeclared models and "
                "six samples; it does not identify the emitter.</text>"
            ),
            "</svg>",
        )
    )

    return SvgFigure(
        figure_id=figure_id,
        analysis_id=snapshot.analysis_id,
        title=title,
        description=description,
        svg="\n".join(lines) + "\n",
    )


@dataclass(frozen=True, slots=True)
class _PlotArea:
    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value)
            for value in (
                self.left,
                self.top,
                self.right,
                self.bottom,
            )
        ):
            raise VisualizationError("plot bounds must be finite")
        if self.left >= self.right or self.top >= self.bottom:
            raise VisualizationError("plot bounds must define a positive area")

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top


def _svg_header(
    figure_id: str,
    title: str,
    description: str,
) -> list[str]:
    title_id = f"{figure_id}-title"
    description_id = f"{figure_id}-description"

    return [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{_SVG_WIDTH}" height="{_SVG_HEIGHT}" '
            f'viewBox="0 0 {_SVG_WIDTH} {_SVG_HEIGHT}" '
            f'role="img" '
            f'aria-labelledby="{title_id} {description_id}">'
        ),
        f'<title id="{title_id}">{html.escape(title)}</title>',
        (f'<desc id="{description_id}">{html.escape(description)}</desc>'),
    ]


def _style_lines() -> tuple[str, ...]:
    return (
        "<style>",
        "  .background { fill: #ffffff; }",
        ("  .plot-background { fill: #f7f7f7; stroke: #1f2933; stroke-width: 1; }"),
        "  .grid-line { stroke: #cbd2d9; stroke-width: 1; }",
        "  .axis-line { stroke: #1f2933; stroke-width: 1.5; }",
        ("  .fit-line { fill: none; stroke: #1f4e79; stroke-width: 3; }"),
        ("  .observed-point { fill: #ffffff; stroke: #b42318; stroke-width: 3; }"),
        ("  .model-bar { fill: #4d6f8f; stroke: #1f2933; stroke-width: 1; }"),
        "  .rank-1 { fill: #2f6b4f; }",
        ("  text { fill: #1f2933; font-family: Arial, Helvetica, sans-serif; }"),
        "  .heading { font-size: 20px; font-weight: 700; }",
        "  .note { font-size: 13px; }",
        "  .tick-label { font-size: 12px; }",
        "  .axis-label { font-size: 14px; font-weight: 700; }",
        "  .legend { font-size: 13px; }",
        ("  .bar-value { font-size: 13px; font-weight: 700; text-anchor: middle; }"),
        ("  .category-label { font-size: 12px; text-anchor: middle; }"),
        "  .annotation { font-size: 12px; }",
        "</style>",
    )


def _plot_background(
    plot: _PlotArea,
    *,
    heading: str,
    note: str,
) -> tuple[str, ...]:
    return (
        ('<rect class="background" x="0" y="0" width="960" height="540" />'),
        (f'<text class="heading" x="{plot.left:.2f}" y="30">{html.escape(heading)}</text>'),
        (f'<text class="note" x="{plot.left:.2f}" y="49">{html.escape(note)}</text>'),
        (
            f'<rect class="plot-background" '
            f'x="{plot.left:.2f}" y="{plot.top:.2f}" '
            f'width="{plot.width:.2f}" '
            f'height="{plot.height:.2f}" />'
        ),
    )


def _axis_lines(
    plot: _PlotArea,
    *,
    x_ticks: tuple[tuple[float, str], ...],
    y_ticks: tuple[tuple[float, str], ...],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    x_label: str,
    y_label: str,
) -> list[str]:
    lines: list[str] = []

    for value, label in y_ticks:
        y = _y_position(
            value,
            y_min,
            y_max,
            plot,
        )
        lines.extend(
            (
                (
                    f'<line class="grid-line" '
                    f'x1="{plot.left:.2f}" y1="{y:.2f}" '
                    f'x2="{plot.right:.2f}" y2="{y:.2f}" />'
                ),
                (
                    f'<text class="tick-label" '
                    f'x="{plot.left - 12.0:.2f}" '
                    f'y="{y + 4.0:.2f}" text-anchor="end">'
                    f"{html.escape(label)}</text>"
                ),
            )
        )

    for value, label in x_ticks:
        x = _x_position(
            value,
            x_min,
            x_max,
            plot,
        )
        lines.extend(
            (
                (
                    f'<line class="grid-line" '
                    f'x1="{x:.2f}" y1="{plot.top:.2f}" '
                    f'x2="{x:.2f}" y2="{plot.bottom:.2f}" />'
                ),
                (
                    f'<text class="tick-label" x="{x:.2f}" '
                    f'y="{plot.bottom + 22.0:.2f}" '
                    f'text-anchor="middle">'
                    f"{html.escape(label)}</text>"
                ),
            )
        )

    lines.extend(
        (
            (
                f'<line class="axis-line" '
                f'x1="{plot.left:.2f}" y1="{plot.bottom:.2f}" '
                f'x2="{plot.right:.2f}" y2="{plot.bottom:.2f}" />'
            ),
            (
                f'<line class="axis-line" '
                f'x1="{plot.left:.2f}" y1="{plot.top:.2f}" '
                f'x2="{plot.left:.2f}" y2="{plot.bottom:.2f}" />'
            ),
            (
                f'<text class="axis-label" '
                f'x="{(plot.left + plot.right) / 2.0:.2f}" '
                f'y="{plot.bottom + 57.0:.2f}" '
                f'text-anchor="middle">'
                f"{html.escape(x_label)}</text>"
            ),
            (
                f'<text class="axis-label" '
                f'transform="translate(25 '
                f"{(plot.top + plot.bottom) / 2.0:.2f}) "
                f'rotate(-90)" text-anchor="middle">'
                f"{html.escape(y_label)}</text>"
            ),
        )
    )

    return lines


def _x_position(
    value: float,
    minimum: float,
    maximum: float,
    plot: _PlotArea,
) -> float:
    if not all(
        math.isfinite(item)
        for item in (
            value,
            minimum,
            maximum,
        )
    ):
        raise VisualizationError("x-axis values must be finite")
    if minimum >= maximum:
        raise VisualizationError("x-axis minimum must be below maximum")

    return plot.left + ((value - minimum) / (maximum - minimum)) * plot.width


def _y_position(
    value: float,
    minimum: float,
    maximum: float,
    plot: _PlotArea,
) -> float:
    if not all(
        math.isfinite(item)
        for item in (
            value,
            minimum,
            maximum,
        )
    ):
        raise VisualizationError("y-axis values must be finite")
    if minimum >= maximum:
        raise VisualizationError("y-axis minimum must be below maximum")

    return plot.bottom - ((value - minimum) / (maximum - minimum)) * plot.height


def _linear_ticks(
    minimum: float,
    maximum: float,
    interval_count: int,
) -> tuple[tuple[float, str], ...]:
    if interval_count <= 0:
        raise VisualizationError("interval_count must be positive")

    step = (maximum - minimum) / interval_count
    return tuple(
        (
            minimum + index * step,
            f"{minimum + index * step:.1f}",
        )
        for index in range(interval_count + 1)
    )


def _rounded_axis_max(value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise VisualizationError("axis maximum source value must be positive and finite")

    magnitude = 10.0 ** math.floor(math.log10(value))
    normalized = value / magnitude

    if normalized <= 1.0:
        rounded = 1.0
    elif normalized <= 2.0:
        rounded = 2.0
    elif normalized <= 5.0:
        rounded = 5.0
    else:
        rounded = 10.0

    return rounded * magnitude


def _decimal_label(value: Decimal) -> str:
    if not value.is_finite():
        raise VisualizationError("decimal labels must be finite")

    return str(value)
