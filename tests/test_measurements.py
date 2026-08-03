from decimal import Decimal

import pytest

from wow_signal_analysis.measurements import (
    WOW_INTEGRATION_SECONDS,
    WOW_PRINTER_SEQUENCE,
    WOW_SAMPLE_CADENCE_SECONDS,
    IntensityCode,
    SignalSample,
    canonical_wow_samples,
    decode_printer_sequence,
    decode_printer_symbol,
)


def test_canonical_constants_preserve_observation_timing() -> None:
    assert WOW_PRINTER_SEQUENCE == "6EQUJ5"
    assert WOW_INTEGRATION_SECONDS == Decimal("10")
    assert WOW_SAMPLE_CADENCE_SECONDS == Decimal("12")


def test_canonical_sequence_decodes_to_documented_intervals() -> None:
    samples = canonical_wow_samples()

    assert [sample.intensity.truncated_snr for sample in samples] == [6, 14, 26, 30, 19, 5]
    assert [sample.intensity.lower_snr for sample in samples] == [
        Decimal("6"),
        Decimal("14"),
        Decimal("26"),
        Decimal("30"),
        Decimal("19"),
        Decimal("5"),
    ]
    assert [sample.intensity.upper_snr for sample in samples] == [
        Decimal("7"),
        Decimal("15"),
        Decimal("27"),
        Decimal("31"),
        Decimal("20"),
        Decimal("6"),
    ]
    assert [sample.intensity.midpoint_snr for sample in samples] == [
        Decimal("6.5"),
        Decimal("14.5"),
        Decimal("26.5"),
        Decimal("30.5"),
        Decimal("19.5"),
        Decimal("5.5"),
    ]


def test_sequence_assigns_zero_based_indices_and_twelve_second_cadence() -> None:
    samples = canonical_wow_samples()

    assert [sample.sample_index for sample in samples] == [0, 1, 2, 3, 4, 5]
    assert [sample.elapsed_seconds for sample in samples] == [
        Decimal("0"),
        Decimal("12"),
        Decimal("24"),
        Decimal("36"),
        Decimal("48"),
        Decimal("60"),
    ]


def test_decoder_supports_blank_digits_and_letters() -> None:
    assert decode_printer_symbol(" ").truncated_snr == 0
    assert decode_printer_symbol("1").truncated_snr == 1
    assert decode_printer_symbol("9").truncated_snr == 9
    assert decode_printer_symbol("A").truncated_snr == 10
    assert decode_printer_symbol("Z").truncated_snr == 35


@pytest.mark.parametrize("symbol", ["", "0", "a", "!", "AA"])
def test_decoder_rejects_symbols_not_emitted_by_the_printer_scheme(symbol: str) -> None:
    with pytest.raises(ValueError):
        decode_printer_symbol(symbol)


def test_intensity_code_rejects_an_inconsistent_manual_value() -> None:
    with pytest.raises(ValueError, match="does not match"):
        IntensityCode(symbol="E", truncated_snr=15)


def test_signal_sample_rejects_negative_index() -> None:
    with pytest.raises(ValueError, match="sample_index"):
        SignalSample(
            sample_index=-1,
            elapsed_seconds=Decimal("0"),
            intensity=decode_printer_symbol("6"),
        )


def test_signal_sample_rejects_negative_elapsed_time() -> None:
    with pytest.raises(ValueError, match="elapsed_seconds"):
        SignalSample(
            sample_index=0,
            elapsed_seconds=Decimal("-1"),
            intensity=decode_printer_symbol("6"),
        )


def test_sequence_requires_content_and_positive_cadence() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        decode_printer_sequence("")

    with pytest.raises(ValueError, match="must be positive"):
        decode_printer_sequence("6", sample_cadence_seconds=Decimal("0"))
