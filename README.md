# Reproducible Analysis of the Wow! Signal

A reproducible, evidence-bound forensic reconstruction of the 1977 Wow! signal.

This repository examines what the retained `6EQUJ5` printer sequence can support through
verified source data, deterministic analysis, uncertainty controls, explicit claim
classification, and independently auditable generated artifacts.

It does **not** claim that `6EQUJ5` contains decoded alien plaintext.

## Purpose

The project separates five related but scientifically distinct questions:

1. **PRESENCE** — What was retained in the historical printer record?
2. **TRANSIT** — Is the six-sample intensity envelope compatible with a beam transit?
3. **BEACON** — Is a stable artificial carrier compatible with the measurements?
4. **HYDROGEN** — How do declared frequency estimates relate to the neutral-hydrogen line?
5. **QUESTION** — Can threshold transformations reproduce Morse punctuation patterns?

The analyses are designed to show both what can be reproduced and what remains unproven.

## Evidence boundaries

Every formal claim belongs to one of five classes:

| Class | Meaning |
| --- | --- |
| Observed | Directly represented by a verified source artifact |
| Derived | Reproducible from verified inputs and declared algorithms |
| Compatibility | Consistent with the measurements but not identified as true |
| Interpretive | A labeled summary that adds no evidentiary strength |
| Speculative | Not established by the surviving evidence |

The repository deliberately preserves these distinctions.

A model fit is not a source identification. A symbolic correspondence is not recovered
transmitter intent. Frequency proximity is not proof of artificial or extraterrestrial origin.

## What the repository analyzes

### Canonical printer sequence

The retained sequence is:

```
6EQUJ5
```

The characters are receiver-strength bin codes. They are not assumed to be transmitted letters.

The canonical loader verifies the source CSV and its provenance manifest before analysis.

### Sequence profile

The midpoint reconstruction measures:

- Adjacent rises and falls
- Interior peak behavior
- Mirrored sample differences
- Exact and approximate symmetry
- Deterministic trend signatures

### Gaussian beam-transit model

The project fits a zero-baseline Gaussian response to the six midpoint samples using a bounded,
deterministic grid-refinement procedure.

The fit reports:

- Amplitude
- Center time
- Sigma
- Full width at half maximum
- Sum squared error
- Root mean squared error
- Coefficient of determination
- Per-sample predictions and residuals

A close Gaussian fit establishes shape compatibility only. It does not identify the emitter,
its location, or its origin.

### Printer-bin sensitivity

Each printer character represents an interval rather than an exact signal-to-noise value.

The analysis therefore evaluates all 64 lower-bound and upper-supremum corners for the six
intervals and reports metric envelopes across those fits.

These are deterministic sensitivity envelopes, not statistical confidence intervals.

### Leave-one-out model comparison

Four predeclared candidate models are compared using leave-one-out prediction error:

- Constant
- Affine
- Quadratic
- Gaussian transit

The comparison ranks only those models against the six retained samples under the declared
scoring rule. It does not determine whether the source was natural or artificial.

### Exhaustive threshold analysis

Every threshold partition of the six midpoint values is evaluated under:

- Forward sequence order
- Reverse sequence order
- Above-threshold-as-dot polarity
- Above-threshold-as-dash polarity

This produces all 28 threshold, direction, and polarity combinations rather than selectively
reporting a preferred transformation.

### International Morse comparison

The transformed patterns are compared with a provenance-verified International Morse registry.

The analysis preserves both notable punctuation mappings:

```
..--..  question mark
--..--  comma
```

The opposite-polarity result is retained rather than discarded.

These correspondences are reproducible transformations selected by the analysis. They are not
transmitter metadata and do not establish intentional Morse communication.

### Exact permutation control

The repository evaluates all 720 unique temporal permutations of the six distinct midpoint
values.

For every permutation it repeats the complete threshold, direction, and polarity analysis. This
provides exact null counts and fractions for selected Morse glyphs without random sampling.

### Neutral-hydrogen frequency context

The project stores the declared neutral-hydrogen rest frequency and multiple Wow! signal
frequency estimates as separate provenance-bound records.

Historical and later analytical estimates are not silently merged or substituted for one
another.

The analysis reports signed and absolute offsets from the rest frequency while preserving each
estimate's status and uncertainty.

Frequency proximity alone does not establish a beacon, artificial origin, or extraterrestrial
technology.

### Claim ledger

The machine-readable claim ledger binds each claim to:

- Evidence identifiers
- Classification
- Verdict
- Dependencies
- Limitations
- Reproduction paths

Among the explicit boundaries preserved by the ledger:

- The five-layer summary is interpretive, not decoded plaintext.
- A stable artificial carrier remains compatible but unproven.
- Intentional Morse punctuation is not established.
- Extraterrestrial technology is not established.

### Hypothesis matrix

The hypothesis matrix distinguishes model support from source conclusions.

It evaluates:

- Gaussian transit-shaped envelope
- Stable artificial carrier or beacon
- Natural versus artificial origin
- Intentional Morse question mark
- Extraterrestrial technology

Each hypothesis has a declared scope, status, supporting claims, conflicting claims, and
limitations.

## Reproducibility architecture

The repository uses only the Python standard library at runtime.

Scientific and evidentiary controls include:

- Verified SHA-256 source manifests
- Exact decimal interval handling
- Exact rational null fractions
- Deterministic bounded model fitting
- Exhaustive threshold enumeration
- Exact permutation enumeration
- Machine-readable claim classifications
- Machine-readable hypothesis statuses
- Deterministic JSON serialization
- Dependency-free accessible SVG generation
- Detached SHA-256 files
- A content-addressed artifact manifest
- Independent artifact auditing
- Double-build release reproduction

## Requirements

- Python 3.11, 3.12, or 3.13
- No third-party runtime dependencies
- Development tools installed through the `dev` optional dependency group

## Installation

Clone or download the repository, then run from its root:

```
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The package also installs the command:

```
wow-signal-analysis
```

Every command can alternatively be run with:

```
python -m wow_signal_analysis
```

## Verify the canonical repository

Verify all canonical source artifacts, hashes, record counts, and cross-bindings:

```
python -m wow_signal_analysis verify --root .
```

Deterministic JSON output:

```
python -m wow_signal_analysis verify --root . --json
```

The repository contract covers:

- Canonical observation dataset
- International Morse registry
- Frequency context
- Claim ledger
- Hypothesis matrix

## Generate the complete analysis

```
python -m wow_signal_analysis generate --root .
```

Generation builds the canonical snapshot, human-readable report, accessible figures, detached
checksums, and content-addressed manifest.

To refuse replacement of existing generated files:

```
python -m wow_signal_analysis generate --root . --no-overwrite
```

To explicitly select Gaussian search controls:

```
python -m wow_signal_analysis generate \
  --root . \
  --grid-points 21 \
  --refinement-rounds 3
```

On Windows PowerShell, the same command can be entered on one line:

```
python -m wow_signal_analysis generate --root . --grid-points 21 --refinement-rounds 3
```

Selected Morse glyphs can be supplied more than once:

```
python -m wow_signal_analysis generate --root . --glyph "?" --glyph ","
```

## Verify reproducibility against generated artifacts

Rebuild the analysis in memory and compare it byte-for-byte with the generated files:

```
python -m wow_signal_analysis generate --root . --check
```

This mode verifies that the committed or locally generated files match the current verified
inputs, implementation, and configuration.

## Independently audit generated artifacts

The independent audit reads the generated manifest rather than trusting the in-memory generation
bundle:

```
python -m wow_signal_analysis audit --root .
```

The audit verifies:

- Exact manifest schema
- Canonical artifact paths and order
- Media types
- File byte counts
- SHA-256 digests
- Detached checksum contents
- Manifest checksum
- Path safety
- Symbolic-link rejection
- Strict generated-directory inventory

Deterministic JSON output:

```
python -m wow_signal_analysis audit --root . --json
```

Extra regular files are rejected by default. They can be allowed without relaxing canonical
artifact verification:

```
python -m wow_signal_analysis audit --root . --allow-extra-files
```

## Run isolated release reproduction

```
python -m wow_signal_analysis.release_verification --root .
```

The release verifier:

1. Verifies the canonical repository contract.
2. Builds the complete analysis twice.
3. Compares every generated path, media type, and byte sequence.
4. Writes one build into an empty temporary directory.
5. Re-reads and verifies every written file.
6. Independently audits the generated manifest and payload.
7. Removes the temporary output after completion.

JSON output is available with:

```
python -m wow_signal_analysis.release_verification --root . --json
```

## Generated artifacts

The generator produces ten files under `artifacts/generated/`:

| Artifact | Purpose |
| --- | --- |
| `analysis_snapshot.json` | Complete machine-readable canonical analysis |
| `analysis_snapshot.sha256` | Detached snapshot checksum |
| `analysis_report.md` | Human-readable evidence-bound report |
| `analysis_report.sha256` | Detached report checksum |
| `beam_fit.svg` | Accessible midpoint and Gaussian-fit figure |
| `beam_fit.sha256` | Detached beam-figure checksum |
| `model_comparison.svg` | Accessible held-out model comparison |
| `model_comparison.sha256` | Detached comparison-figure checksum |
| `artifact_manifest.json` | Content-addressed payload inventory |
| `artifact_manifest.sha256` | Detached manifest checksum |

Do not hand-edit generated artifacts. Regenerate the complete bundle instead.

## Complete quality gate

Run the same cumulative gate used by continuous integration:

```
python check_green.py
```

The gate executes, in order:

1. Ruff lint validation
2. Ruff formatting validation
3. Strict mypy validation
4. Complete pytest suite
5. Wheel and source-distribution construction
6. Canonical repository-contract verification
7. Isolated double-build release reproduction

A green status is valid only when every planned step completes successfully.

To continue after failures and collect additional diagnostics:

```
python check_green.py --continue-on-failure
```

To omit only the package-build step during local diagnostics:

```
python check_green.py --skip-build
```

Skipping a check does not establish a complete green release result.

## Continuous integration

The GitHub Actions workflow runs the cumulative quality gate on:

- Ubuntu with Python 3.11
- Ubuntu with Python 3.12
- Ubuntu with Python 3.13
- Windows with Python 3.13

The workflow uses read-only repository permissions and executes the same `check_green.py` entry
point available to local contributors.

## Repository structure

```
.
├── .github/
│   └── workflows/
│       └── ci.yml
├── artifacts/
│   └── generated/
├── data/
│   ├── provenance/
│   ├── raw/
│   └── reference/
├── src/
│   └── wow_signal_analysis/
├── tests/
├── check_green.py
├── CITATION.cff
├── CONTRIBUTING.md
├── LICENSE
├── pyproject.toml
└── README.md
```

Important modules include:

| Module | Responsibility |
| --- | --- |
| `measurements.py` | Printer-code decoding and canonical measurements |
| `dataset.py` | Verified observation loading |
| `provenance.py` | Source-manifest and digest verification |
| `profile.py` | Deterministic sequence profiling |
| `thresholds.py` | Exhaustive threshold partitions |
| `morse.py` | Verified Morse registry |
| `morse_correspondence.py` | Direction and polarity correspondence analysis |
| `null_model.py` | Exact permutation controls |
| `frequency_context.py` | Hydrogen-line references and frequency offsets |
| `beam_model.py` | Deterministic Gaussian transit fitting |
| `quantization.py` | Printer-bin corner sensitivity |
| `model_comparison.py` | Leave-one-out candidate-model comparison |
| `claim_ledger.py` | Evidence-bound claims and verdicts |
| `hypothesis_matrix.py` | Hypothesis scope, status, and limitations |
| `repository_contract.py` | Cumulative canonical source verification |
| `analysis_snapshot.py` | Complete deterministic analysis assembly |
| `report.py` | Human-readable Markdown reporting |
| `visualization.py` | Dependency-free accessible SVG figures |
| `artifacts.py` | Artifact bundle, checksums, and manifest |
| `artifact_audit.py` | Independent generated-file auditing |
| `release_verification.py` | Isolated double-build release gate |
| `cli.py` | Cross-platform command-line interface |

## Interpretation guide

### Supported

The repository can support statements such as:

- The retained printer sequence is `6EQUJ5`.
- The symbols encode receiver-strength intervals.
- The midpoint envelope has a pronounced interior peak.
- A Gaussian transit model closely matches the coarse six-sample shape.
- The Gaussian model has the lowest held-out error among the four declared models.
- Specific threshold transformations reproduce registered Morse punctuation.
- Opposite polarity produces an alternative punctuation mapping.
- Exact permutation controls quantify how often selected mappings occur.
- Declared frequency estimates can be compared reproducibly with the H I rest frequency.

### Compatible but not proven

The retained envelope can be described as compatible with a stable source crossing a telescope
beam.

That compatibility does not establish whether the source was:

- Natural
- Human-made interference
- An intentional terrestrial transmission
- An extraterrestrial beacon

### Not established

The surviving measurements do not establish:

- An intentional Morse message
- A decoded question
- Decoded plaintext
- Artificial origin
- Extraterrestrial origin
- Extraterrestrial technology
- The identity or location of the emitter

## Limitations

The primary retained sequence contains only six integrated intensity bins.

The repository does not possess a complete voltage time series, phase information, transmitter
timing, repeated detections, or enough independent measurements to identify the source.

Thresholds, direction choices, and polarities are analysis operations applied after observation.
They are not known transmitter settings.

The permutation analysis controls temporal ordering within the retained values. It does not model
every possible astronomical or radio-frequency process.

The candidate-model comparison is intentionally small and declared in advance. A model that ranks
best within this set is not necessarily the true physical model.

## Contributing

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) before proposing changes.

Contributions must preserve:

- Verified provenance
- Deterministic output
- Evidence classifications
- Explicit uncertainty
- Fail-closed validation
- Exact claim limitations
- Complete quality-gate reporting

Do not report a change as green when only a partial test or visual inspection was performed.

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff).

Citing this software does not replace citation of the historical, standards, or scientific sources
represented by the provenance manifests.

## License

Licensed under the Apache License 2.0. See [`LICENSE`](LICENSE).

## Core conclusion

`6EQUJ5` remains an extraordinary historical observation.

Its coarse shape can be reconstructed, modeled, transformed, controlled, and audited
reproducibly. Those operations clarify the evidence, but they do not convert a six-character
receiver-strength record into proven alien language.

The repository's central rule is therefore:

> Reproduce the result. Preserve the alternatives. Do not claim more than the evidence supports.
