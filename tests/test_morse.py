from __future__ import annotations

import json
from pathlib import Path

import pytest

from wow_signal_analysis.morse import (
    MORSE_MANIFEST_PATH,
    MORSE_REFERENCE_PATH,
    MORSE_STANDARD_ID,
    MorseCategory,
    MorseError,
    MorseRegistry,
    MorseSymbol,
    load_morse_registry,
    load_verified_morse_registry,
)
from wow_signal_analysis.provenance import ProvenanceError

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_REFERENCE_PATH = _REPOSITORY_ROOT / MORSE_REFERENCE_PATH


def test_verified_registry_matches_the_in_force_itu_reference() -> None:
    registry = load_verified_morse_registry(_REPOSITORY_ROOT)

    assert registry.schema_version == 1
    assert registry.standard_id == MORSE_STANDARD_ID
    assert registry.source_url == (
        "https://www.itu.int/rec/R-REC-M.1677-1-200910-I/en"
    )
    assert len(registry.symbols) == 51


def test_registry_category_counts_are_complete_for_its_declared_scope() -> None:
    registry = load_morse_registry(_REFERENCE_PATH)

    assert sum(
        symbol.category is MorseCategory.LETTER
        for symbol in registry.symbols
    ) == 27
    assert sum(
        symbol.category is MorseCategory.FIGURE
        for symbol in registry.symbols
    ) == 10
    assert sum(
        symbol.category is MorseCategory.PUNCTUATION
        for symbol in registry.symbols
    ) == 14


def test_question_mark_and_comma_assignments_are_not_interchanged() -> None:
    registry = load_morse_registry(_REFERENCE_PATH)

    question = registry.symbol_for_glyph("?")
    comma = registry.symbol_for_glyph(",")

    assert question.pattern == "..--.."
    assert question.label == "question mark or request for repetition"
    assert comma.pattern == "--..--"


def test_pattern_lookup_preserves_standard_defined_ambiguity() -> None:
    registry = load_morse_registry(_REFERENCE_PATH)

    matches = registry.symbols_for_pattern("-..-")

    assert tuple(symbol.glyph for symbol in matches) == ("X", "×")
    assert registry.has_pattern("-..-")
    assert not registry.has_pattern("........")


def test_all_registry_glyphs_are_unique() -> None:
    registry = load_morse_registry(_REFERENCE_PATH)

    glyphs = tuple(symbol.glyph for symbol in registry.symbols)
    assert len(glyphs) == len(set(glyphs))


def test_unknown_glyph_and_invalid_patterns_fail_closed() -> None:
    registry = load_morse_registry(_REFERENCE_PATH)

    with pytest.raises(MorseError, match="found 0"):
        registry.symbol_for_glyph("$")
    with pytest.raises(MorseError, match="only '.' and '-'"):
        registry.symbols_for_pattern("001100")
    with pytest.raises(MorseError, match="eight elements"):
        registry.has_pattern(".........")


def test_symbol_validation_rejects_invalid_manual_records() -> None:
    with pytest.raises(MorseError, match="exactly one"):
        MorseSymbol(
            glyph="AB",
            category=MorseCategory.LETTER,
            pattern=".-",
            section="1.1.1",
            label="invalid",
        )

    with pytest.raises(MorseError, match="section"):
        MorseSymbol(
            glyph="A",
            category=MorseCategory.LETTER,
            pattern=".-",
            section="2.1",
            label="invalid",
        )


def test_registry_rejects_duplicate_glyphs() -> None:
    symbol = MorseSymbol(
        glyph="A",
        category=MorseCategory.LETTER,
        pattern=".-",
        section="1.1.1",
        label="letter A",
    )

    with pytest.raises(MorseError, match="glyphs must be unique"):
        MorseRegistry(
            schema_version=1,
            standard_id=MORSE_STANDARD_ID,
            title="duplicate test",
            source_url="https://example.test/morse",
            scope_note="validation test",
            symbols=(symbol, symbol),
        )


def test_loader_rejects_invalid_json_and_unknown_categories(
    tmp_path: Path,
) -> None:
    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{", encoding="utf-8")
    with pytest.raises(MorseError, match="invalid Morse registry JSON"):
        load_morse_registry(invalid_json)

    payload = json.loads(_REFERENCE_PATH.read_text(encoding="utf-8"))
    payload["symbols"][0]["category"] = "unknown"
    unknown_category = tmp_path / "unknown-category.json"
    unknown_category.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    with pytest.raises(MorseError, match="unsupported Morse category"):
        load_morse_registry(unknown_category)


def test_verified_loader_detects_reference_tampering(tmp_path: Path) -> None:
    manifest_target = tmp_path / MORSE_MANIFEST_PATH
    reference_target = tmp_path / MORSE_REFERENCE_PATH
    manifest_target.parent.mkdir(parents=True)
    reference_target.parent.mkdir(parents=True)
    manifest_target.write_bytes(
        (_REPOSITORY_ROOT / MORSE_MANIFEST_PATH).read_bytes()
    )
    reference_target.write_text(
        _REFERENCE_PATH.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ProvenanceError, match="hash-mismatch"):
        load_verified_morse_registry(tmp_path)
