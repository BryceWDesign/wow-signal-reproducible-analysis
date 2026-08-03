"""Public package identity and canonical measurement API."""

from typing import Final

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

DISPLAY_NAME: Final = "Reproducible Analysis of the Wow! Signal"
PROJECT_SLUG: Final = "wow-signal-reproducible-analysis"
__version__: Final = "0.1.0"

__all__ = [
    "DISPLAY_NAME",
    "PROJECT_SLUG",
    "WOW_INTEGRATION_SECONDS",
    "WOW_PRINTER_SEQUENCE",
    "WOW_SAMPLE_CADENCE_SECONDS",
    "IntensityCode",
    "SignalSample",
    "__version__",
    "canonical_wow_samples",
    "decode_printer_sequence",
    "decode_printer_symbol",
]
