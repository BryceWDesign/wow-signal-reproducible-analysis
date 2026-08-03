# Contributing

Contributions are welcome when they preserve the repository's evidence boundaries,
provenance controls, deterministic behavior, and reproducibility requirements.

## Evidence rules

Claims must remain classified as:

- **Observed** — directly supported by verified evidence.
- **Derived** — reproducible from verified inputs.
- **Compatibility** — consistent with evidence but not proven.
- **Interpretive** — a labeled summary that adds no evidentiary strength.
- **Speculative** — not established by the surviving evidence.

Do not present a model fit, threshold pattern, Morse correspondence, or frequency
association as proof of transmitter intent, decoded language, artificial origin, or
extraterrestrial origin.

The `6EQUJ5` characters are receiver-strength codes, not transmitted letters.

## Development

Use Python 3.11, 3.12, or 3.13.

Run:
```
    python -m pip install --upgrade pip
    python -m pip install -e ".[dev]"
    python check_green.py
```
Do not report the repository as green unless the complete quality gate passes.

## Data and provenance

Changes under `data/raw/` or `data/reference/` must:

- Update the matching manifest under `data/provenance/`.
- Recalculate the SHA-256 digest from the final bytes.
- Update record counts when needed.
- Preserve separate sources and estimates.
- Include tamper-detection tests.

Never silently replace a historical value with a newer estimate.

## Analysis changes

Analysis code must:

- Be deterministic for identical inputs and configuration.
- Reject malformed or non-finite inputs.
- Use `Decimal` or `Fraction` when exactness matters.
- Label confidence intervals, sensitivity envelopes, and exact counts correctly.
- Include normal, boundary, and failure-path tests.
- Avoid hidden state and unnecessary runtime dependencies.

Thresholds, directions, and polarities must be enumerated rather than selectively
reported.

## Generated artifacts

Generate and verify the artifact bundle with:
```
    python -m wow_signal_analysis generate --root .
    python -m wow_signal_analysis generate --root . --check
    python -m wow_signal_analysis audit --root .
```
Do not hand-edit files under `artifacts/generated/`.

## Required validation

Before submitting a contribution, run:
```
    python check_green.py
```
The gate checks Ruff, formatting, mypy, pytest, package builds, the repository
contract, and isolated release reproduction.

When a check cannot be run, state exactly what was and was not executed. A partial
test run or visual inspection is not a cumulative green result.

## Pull requests

State:

- What the change addresses.
- Which evidence class is affected.
- Which files and manifests changed.
- Which commands were executed.
- Any remaining uncertainty.
- Whether generated artifacts were reproduced.

## Citation

Use `CITATION.cff` when citing this software. External historical, standards, and
scientific sources must still be cited separately.
