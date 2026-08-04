"""Deterministic human-readable reporting for the canonical analysis snapshot."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from hashlib import sha256
from typing import Final

from wow_signal_analysis.analysis_snapshot import (
    ANALYSIS_SNAPSHOT_ID,
    AnalysisSnapshot,
)
from wow_signal_analysis.claim_ledger import (
    ClaimClassification,
    ClaimVerdict,
)
from wow_signal_analysis.hypothesis_matrix import HypothesisStatus
from wow_signal_analysis.morse_correspondence import ThresholdMorseComparison
from wow_signal_analysis.quantization import FitMetric

ANALYSIS_REPORT_ID: Final = "wow-signal-evidence-report-v1"
ANALYSIS_REPORT_TITLE: Final = "Reproducible, Evidence-Bound Analysis of the 1977 Wow! Signal"


class ReportRenderError(ValueError):
    """Raised when an analysis report cannot be rendered safely."""


@dataclass(frozen=True, slots=True)
class RenderedAnalysisReport:
    """One deterministic Markdown report and its content identity."""

    report_id: str
    analysis_id: str
    markdown: str

    def __post_init__(self) -> None:
        if self.report_id != ANALYSIS_REPORT_ID:
            raise ReportRenderError(f"report_id must be {ANALYSIS_REPORT_ID!r}")
        if self.analysis_id != ANALYSIS_SNAPSHOT_ID:
            raise ReportRenderError(f"analysis_id must be {ANALYSIS_SNAPSHOT_ID!r}")
        if not isinstance(self.markdown, str) or not self.markdown:
            raise ReportRenderError("markdown must be non-empty")
        if "\r" in self.markdown:
            raise ReportRenderError("markdown must use LF line endings")
        if not self.markdown.startswith(f"# {ANALYSIS_REPORT_TITLE}\n"):
            raise ReportRenderError("markdown must begin with the canonical report title")
        if not self.markdown.endswith("\n"):
            raise ReportRenderError("markdown must end with exactly one line terminator")
        if self.markdown.endswith("\n\n"):
            raise ReportRenderError("markdown must not end with a blank trailing line")

    @property
    def content(self) -> bytes:
        """Return the UTF-8 report bytes."""

        return self.markdown.encode("utf-8")

    @property
    def byte_count(self) -> int:
        """Return the exact UTF-8 byte count."""

        return len(self.content)

    @property
    def sha256_hex(self) -> str:
        """Return the lowercase SHA-256 digest of the Markdown bytes."""

        return sha256(self.content).hexdigest()


def build_analysis_report(
    snapshot: AnalysisSnapshot,
) -> RenderedAnalysisReport:
    """Render one complete evidence-bound Markdown report."""

    if not isinstance(snapshot, AnalysisSnapshot):
        raise ReportRenderError("snapshot must be an AnalysisSnapshot")

    markdown = _render_markdown(snapshot)

    return RenderedAnalysisReport(
        report_id=ANALYSIS_REPORT_ID,
        analysis_id=snapshot.analysis_id,
        markdown=markdown,
    )


def render_analysis_report(snapshot: AnalysisSnapshot) -> str:
    """Return only the deterministic Markdown representation."""

    return build_analysis_report(snapshot).markdown


def _render_markdown(snapshot: AnalysisSnapshot) -> str:
    sections = (
        _render_header(snapshot),
        _render_reproducibility_scope(snapshot),
        _render_five_layer_summary(snapshot),
        _render_observation_table(snapshot),
        _render_beam_fit(snapshot),
        _render_model_comparison(snapshot),
        _render_quantization_sensitivity(snapshot),
        _render_symbolic_correspondence(snapshot),
        _render_frequency_context(snapshot),
        _render_claim_ledger(snapshot),
        _render_hypothesis_matrix(snapshot),
        _render_interpretive_limits(),
    )

    return "\n\n".join(section.rstrip("\n") for section in sections) + "\n"


def _render_header(snapshot: AnalysisSnapshot) -> str:
    return "\n".join(
        (
            f"# {ANALYSIS_REPORT_TITLE}",
            "",
            f"**Report ID:** `{ANALYSIS_REPORT_ID}`  ",
            f"**Analysis ID:** `{snapshot.analysis_id}`  ",
            f"**Snapshot schema:** `{snapshot.schema_version}`",
            "",
            (
                "> This report distinguishes direct observations, reproducible "
                "transformations, compatible hypotheses, and unsupported "
                "conclusions. These results do not claim a decoded alien message."
            ),
        )
    )


def _render_reproducibility_scope(
    snapshot: AnalysisSnapshot,
) -> str:
    component_rows = tuple(
        (
            component.component_id,
            str(component.artifact_path),
            str(component.record_count),
        )
        for component in snapshot.repository.components
    )

    return "\n".join(
        (
            "## Reproducibility scope",
            "",
            (
                f"The repository contract verifies "
                f"**{snapshot.repository.verified_component_count}** canonical "
                f"components containing "
                f"**{snapshot.repository.total_record_count}** records."
            ),
            "",
            _markdown_table(
                ("Component", "Canonical artifact", "Records"),
                component_rows,
            ),
        )
    )


def _render_five_layer_summary(
    snapshot: AnalysisSnapshot,
) -> str:
    question = snapshot.permutation_null.summary_for_glyph("?")
    comma = snapshot.permutation_null.summary_for_glyph(",")
    historical_offset = snapshot.frequency_context.offsets[0]
    best_model = snapshot.model_comparison.best_model

    return "\n".join(
        (
            "## Five-layer evidence summary",
            "",
            (
                f"1. **PRESENCE** — The retained printer sequence is "
                f"`{snapshot.dataset.printer_sequence}`. Its characters encode "
                f"receiver-strength bins, not transmitted letters."
            ),
            (
                f"2. **TRANSIT** — The midpoint sequence has trend signature "
                f"`{snapshot.profile.trend_signature}`, a unique interior peak, "
                f"and a Gaussian fit with "
                f"`R² = {_format_float(snapshot.gaussian_fit.coefficient_of_determination)}`."
            ),
            (
                "3. **BEACON** — A stable artificial carrier is compatible with "
                "the coarse envelope, but the claim ledger classifies that "
                "possibility as `compatible-not-proven`."
            ),
            (
                f"4. **HYDROGEN** — The historical frequency estimate is "
                f"{_format_decimal(historical_offset.absolute_offset_khz)} kHz "
                f"from the declared H I rest frequency. Proximity alone does "
                f"not identify a source."
            ),
            (
                f"5. **QUESTION** — One threshold and polarity produce `..--..` "
                f"(`?`), while the opposite polarity produces `--..--` (`,`). "
                f"Each occurs in {_format_fraction(question.sequence_fraction)} "
                f"and {_format_fraction(comma.sequence_fraction)} of unique "
                f"temporal permutations, respectively."
            ),
            "",
            (
                f"The lowest leave-one-out prediction error belongs to "
                f"`{best_model.model.value}`. This ranking applies only to the "
                f"four predeclared models and six retained samples."
            ),
        )
    )


def _render_observation_table(
    snapshot: AnalysisSnapshot,
) -> str:
    rows = tuple(
        (
            str(sample.sample_index),
            _format_decimal(sample.elapsed_seconds),
            sample.intensity.symbol,
            _format_decimal(sample.intensity.lower_snr),
            _format_decimal(sample.intensity.upper_snr),
            _format_decimal(sample.intensity.midpoint_snr),
        )
        for sample in snapshot.dataset.samples
    )

    mirror_rows = tuple(
        (
            f"{comparison.left_index}:{comparison.right_index}",
            _format_decimal(comparison.left_snr),
            _format_decimal(comparison.right_snr),
            _format_decimal(comparison.signed_difference),
        )
        for comparison in snapshot.profile.mirror_comparisons
    )

    return "\n".join(
        (
            "## Canonical observation",
            "",
            _markdown_table(
                (
                    "Index",
                    "Elapsed seconds",
                    "Printer symbol",
                    "SNR lower",
                    "SNR upper",
                    "SNR midpoint",
                ),
                rows,
            ),
            "",
            (
                f"**Peak:** index `{snapshot.profile.peak_index}` at midpoint "
                f"SNR `{_format_decimal(snapshot.profile.peak_value)}`  "
            ),
            (
                f"**Strict single peak:** "
                f"`{_boolean_text(snapshot.profile.is_strict_single_peak)}`  "
            ),
            (f"**Exact palindrome:** `{_boolean_text(snapshot.profile.is_exact_palindrome)}`"),
            "",
            "### Mirrored sample comparisons",
            "",
            _markdown_table(
                (
                    "Index pair",
                    "Left midpoint",
                    "Right midpoint",
                    "Signed difference",
                ),
                mirror_rows,
            ),
        )
    )


def _render_beam_fit(snapshot: AnalysisSnapshot) -> str:
    fit = snapshot.gaussian_fit

    parameter_rows = (
        ("Amplitude SNR", _format_float(fit.amplitude_snr)),
        ("Center seconds", _format_float(fit.center_seconds)),
        ("Sigma seconds", _format_float(fit.sigma_seconds)),
        ("FWHM seconds", _format_float(fit.fwhm_seconds)),
        ("Sum squared error", _format_float(fit.sum_squared_error)),
        (
            "Root mean squared error",
            _format_float(fit.root_mean_squared_error),
        ),
        (
            "Coefficient of determination",
            _format_float(fit.coefficient_of_determination),
        ),
    )

    sample_rows = tuple(
        (
            str(sample.sample_index),
            _format_decimal(sample.elapsed_seconds),
            _format_decimal(sample.observed_snr),
            _format_float(sample.predicted_snr),
            _format_float(sample.residual_snr),
        )
        for sample in fit.samples
    )

    return "\n".join(
        (
            "## Gaussian beam-transit fit",
            "",
            (
                "The fitted model is a zero-baseline Gaussian response. "
                "Fit quality establishes shape compatibility only."
            ),
            "",
            _markdown_table(("Parameter", "Value"), parameter_rows),
            "",
            "### Per-sample fit",
            "",
            _markdown_table(
                (
                    "Index",
                    "Elapsed seconds",
                    "Observed SNR",
                    "Predicted SNR",
                    "Residual SNR",
                ),
                sample_rows,
            ),
        )
    )


def _render_model_comparison(
    snapshot: AnalysisSnapshot,
) -> str:
    rows = tuple(
        (
            str(rank),
            result.model.value,
            str(result.parameter_count),
            _format_float(result.root_mean_squared_prediction_error),
            _format_float(result.mean_absolute_prediction_error),
            _format_float(result.prediction_sum_squares),
        )
        for rank, result in enumerate(
            snapshot.model_comparison.ranked_by_prediction_error,
            start=1,
        )
    )

    return "\n".join(
        (
            "## Leave-one-out shape-model comparison",
            "",
            (
                "Models are ranked by held-out root mean squared prediction "
                "error. Lower error does not identify the physical emitter."
            ),
            "",
            _markdown_table(
                (
                    "Rank",
                    "Model",
                    "Parameters",
                    "Held-out RMSE",
                    "Held-out MAE",
                    "PRESS",
                ),
                rows,
            ),
        )
    )


def _render_quantization_sensitivity(
    snapshot: AnalysisSnapshot,
) -> str:
    rows = tuple(
        (
            metric.value,
            _format_float(snapshot.quantization.envelope(metric).minimum),
            _format_float(snapshot.quantization.envelope(metric).maximum),
            _format_float(snapshot.quantization.envelope(metric).span),
            snapshot.quantization.envelope(metric).minimum_corner_pattern,
            snapshot.quantization.envelope(metric).maximum_corner_pattern,
        )
        for metric in FitMetric
    )

    return "\n".join(
        (
            "## Printer-bin corner sensitivity",
            "",
            (
                f"All **{snapshot.quantization.evaluated_corner_count}** "
                f"lower-bound/upper-supremum interval corners were fitted."
            ),
            "",
            (
                "These ranges are deterministic corner-sensitivity envelopes. "
                "They are not confidence intervals."
            ),
            "",
            _markdown_table(
                (
                    "Metric",
                    "Minimum",
                    "Maximum",
                    "Span",
                    "Minimum corner",
                    "Maximum corner",
                ),
                rows,
            ),
        )
    )


def _render_symbolic_correspondence(
    snapshot: AnalysisSnapshot,
) -> str:
    rows: list[tuple[str, ...]] = []

    for glyph in snapshot.config.selected_morse_glyphs:
        symbol = snapshot.morse_registry.symbol_for_glyph(glyph)
        comparisons = snapshot.morse_correspondence.comparisons_for_glyph(glyph)
        null_summary = snapshot.permutation_null.summary_for_glyph(glyph)

        if comparisons:

            def _comparison_text(
                comparison: ThresholdMorseComparison,
            ) -> str:
                interval = _interval_text(
                    comparison.lower_bound_inclusive,
                    comparison.upper_bound_exclusive,
                )
                return (
                    f"cut {comparison.cut_index}, "
                    f"{comparison.direction.value}, "
                    f"{comparison.polarity.value}, "
                    f"{interval}"
                )

            observed = "; ".join(_comparison_text(comparison) for comparison in comparisons)
            pattern = comparisons[0].morse_pattern
        else:
            observed = "No observed correspondence"
            pattern = symbol.pattern

        rows.append(
            (
                glyph,
                symbol.label,
                pattern,
                observed,
                (
                    f"{null_summary.matched_sequence_count}/"
                    f"{null_summary.total_sequence_count} "
                    f"({_format_fraction(null_summary.sequence_fraction)})"
                ),
                (
                    f"{null_summary.matched_comparison_count}/"
                    f"{null_summary.total_comparison_count} "
                    f"({_format_fraction(null_summary.comparison_fraction)})"
                ),
            )
        )

    return "\n".join(
        (
            "## Exhaustive threshold and Morse correspondence",
            "",
            (
                f"The analysis evaluates "
                f"**{len(snapshot.morse_correspondence.threshold_cases)}** "
                f"threshold partitions and "
                f"**{len(snapshot.morse_correspondence.comparisons)}** "
                f"threshold/direction/polarity combinations."
            ),
            "",
            _markdown_table(
                (
                    "Glyph",
                    "Registry label",
                    "Morse pattern",
                    "Observed analysis paths",
                    "Null sequences",
                    "Null comparisons",
                ),
                tuple(rows),
            ),
            "",
            (
                "The question-mark correspondence is reproducible, but the "
                "opposite polarity produces a comma. The exhaustive result "
                "does not establish intentional Morse transmission."
            ),
        )
    )


def _render_frequency_context(
    snapshot: AnalysisSnapshot,
) -> str:
    rows = tuple(
        (
            estimate.estimate_id,
            estimate.status.value,
            _format_decimal(estimate.frequency_mhz),
            _format_decimal(estimate.uncertainty_mhz),
            _format_decimal(snapshot.frequency_context.offset_for(estimate.estimate_id).delta_mhz),
            _format_decimal(
                snapshot.frequency_context.offset_for(estimate.estimate_id).absolute_offset_khz
            ),
            _boolean_text(
                snapshot.frequency_context.offset_for(
                    estimate.estimate_id
                ).uncertainty_interval_contains_rest
            ),
        )
        for estimate in snapshot.frequency_context.estimates
    )

    return "\n".join(
        (
            "## Neutral-hydrogen frequency context",
            "",
            (
                f"**Declared H I rest frequency:** "
                f"`{_format_decimal(snapshot.frequency_context.rest_line.rest_frequency_mhz)} MHz`"
            ),
            "",
            _markdown_table(
                (
                    "Estimate",
                    "Status",
                    "Frequency MHz",
                    "Uncertainty MHz",
                    "Delta MHz",
                    "Absolute offset kHz",
                    "Interval contains rest",
                ),
                rows,
            ),
            "",
            (
                "The estimates remain separate because they derive from "
                "different analyses. Frequency context does not establish "
                "artificial or extraterrestrial origin."
            ),
        )
    )


def _render_claim_ledger(
    snapshot: AnalysisSnapshot,
) -> str:
    count_rows = tuple(
        (
            classification.value,
            str(len(snapshot.claim_ledger.claims_by_classification(classification))),
        )
        for classification in ClaimClassification
    )

    claim_rows = tuple(
        (
            claim.claim_id,
            claim.classification.value,
            claim.verdict.value,
            claim.statement,
        )
        for claim in snapshot.claim_ledger.topological_claims
    )

    not_established = snapshot.claim_ledger.claims_by_verdict(ClaimVerdict.NOT_ESTABLISHED)

    return "\n".join(
        (
            "## Evidence-bound claim ledger",
            "",
            _markdown_table(
                ("Classification", "Claim count"),
                count_rows,
            ),
            "",
            _markdown_table(
                (
                    "Claim ID",
                    "Classification",
                    "Verdict",
                    "Statement",
                ),
                claim_rows,
            ),
            "",
            "**Claims explicitly not established:**",
            "",
            *(f"- `{claim.claim_id}` — {claim.statement}" for claim in not_established),
        )
    )


def _render_hypothesis_matrix(
    snapshot: AnalysisSnapshot,
) -> str:
    status_rows = tuple(
        (
            status.value,
            str(len(snapshot.hypothesis_matrix.hypotheses_by_status(status))),
        )
        for status in HypothesisStatus
    )

    hypothesis_rows = tuple(
        (
            hypothesis.record.hypothesis_id,
            hypothesis.record.scope.value,
            hypothesis.record.status.value,
            hypothesis.record.proposition,
            hypothesis.record.limitations[0],
        )
        for hypothesis in snapshot.hypothesis_matrix.hypotheses
    )

    return "\n".join(
        (
            "## Hypothesis and falsification matrix",
            "",
            _markdown_table(
                ("Status", "Hypothesis count"),
                status_rows,
            ),
            "",
            _markdown_table(
                (
                    "Hypothesis ID",
                    "Scope",
                    "Status",
                    "Proposition",
                    "Primary limitation",
                ),
                hypothesis_rows,
            ),
        )
    )


def _render_interpretive_limits() -> str:
    return "\n".join(
        (
            "## Interpretive limits",
            "",
            ("- A good beam-shape fit does not identify the emitter, its location, or its origin."),
            (
                "- Receiver-strength bins do not preserve transmitter-scale "
                "dot, dash, character, or word timing."
            ),
            (
                "- Analyst-selected thresholds and polarities are transformations, "
                "not recovered transmitter metadata."
            ),
            ("- Hydrogen-line context does not establish artificial origin."),
            ("- Artificial-carrier compatibility does not establish extraterrestrial technology."),
            (
                "- The surviving measurements do not establish an intentional "
                "message or decoded plaintext."
            ),
        )
    )


def _markdown_table(
    headers: tuple[str, ...],
    rows: tuple[tuple[str, ...], ...],
) -> str:
    if not headers:
        raise ReportRenderError("Markdown table headers must not be empty")
    if any(not header.strip() for header in headers):
        raise ReportRenderError("Markdown table headers must be non-empty")
    if any(len(row) != len(headers) for row in rows):
        raise ReportRenderError("Markdown table rows must match the header width")

    header_line = "| " + " | ".join(_escape_cell(header) for header in headers) + " |"
    separator_line = "| " + " | ".join("---" for _ in headers) + " |"
    body_lines = tuple("| " + " | ".join(_escape_cell(cell) for cell in row) + " |" for row in rows)

    return "\n".join((header_line, separator_line, *body_lines))


def _escape_cell(value: str) -> str:
    if not isinstance(value, str):
        raise ReportRenderError("Markdown table values must be strings")

    return " ".join(value.split()).replace("|", "\\|")


def _format_float(value: float) -> str:
    if not math.isfinite(value):
        raise ReportRenderError("report float values must be finite")

    normalized = 0.0 if abs(value) < 0.5e-12 else value
    return f"{normalized:.6f}"


def _format_decimal(value: Decimal) -> str:
    if not value.is_finite():
        raise ReportRenderError("report decimal values must be finite")

    return str(value)


def _format_fraction(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _boolean_text(value: bool) -> str:
    return "yes" if value else "no"


def _interval_text(
    lower_bound_inclusive: Decimal | None,
    upper_bound_exclusive: Decimal | None,
) -> str:
    lower = "-inf" if lower_bound_inclusive is None else _format_decimal(lower_bound_inclusive)
    upper = "+inf" if upper_bound_exclusive is None else _format_decimal(upper_bound_exclusive)
    left_bracket = "(" if lower_bound_inclusive is None else "["

    return f"{left_bracket}{lower}, {upper})"
