"""Verified International Morse code reference data and strict lookup utilities."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Final, cast
from urllib.parse import urlparse

from wow_signal_analysis.provenance import (
    ProvenanceError,
    load_source_manifest,
    require_verified_artifacts,
)

MORSE_STANDARD_ID: Final = "ITU-R M.1677-1"
MORSE_REFERENCE_PATH: Final = PurePosixPath("data/reference/itu_m1677_1_morse_symbols.json")
MORSE_MANIFEST_PATH: Final = PurePosixPath("data/provenance/morse_source_manifest.json")

_VALID_PATTERN_CHARACTERS: Final = frozenset({".", "-"})


class MorseError(ValueError):
    """Raised when Morse reference data or lookup input is invalid."""


class MorseCategory(StrEnum):
    """ITU table category for a printable Morse symbol."""

    LETTER = "letter"
    FIGURE = "figure"
    PUNCTUATION = "punctuation"


@dataclass(frozen=True, slots=True)
class MorseSymbol:
    """One printable symbol and its International Morse code pattern."""

    glyph: str
    category: MorseCategory
    pattern: str
    section: str
    label: str

    def __post_init__(self) -> None:
        if len(self.glyph) != 1 or self.glyph.isspace():
            raise MorseError("glyph must contain exactly one non-whitespace character")
        _validate_pattern(self.pattern)
        if not self.section.startswith("1.1."):
            raise MorseError("section must identify an ITU table under section 1.1")
        if not self.label.strip():
            raise MorseError("label must be non-empty")


@dataclass(frozen=True, slots=True)
class MorseRegistry:
    """Immutable, validated registry of printable International Morse symbols."""

    schema_version: int
    standard_id: str
    title: str
    source_url: str
    scope_note: str
    symbols: tuple[MorseSymbol, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise MorseError("unsupported Morse registry schema_version")
        if not self.standard_id.strip() or not self.title.strip() or not self.scope_note.strip():
            raise MorseError("standard_id, title, and scope_note must be non-empty")
        parsed_url = urlparse(self.source_url)
        if parsed_url.scheme != "https" or not parsed_url.netloc:
            raise MorseError("source_url must be an absolute HTTPS URL")
        if not self.symbols:
            raise MorseError("Morse registry must contain at least one symbol")

        glyphs = tuple(symbol.glyph for symbol in self.symbols)
        if len(set(glyphs)) != len(glyphs):
            raise MorseError("Morse registry glyphs must be unique")

    def symbol_for_glyph(self, glyph: str) -> MorseSymbol:
        """Return the unique registry entry for a printable glyph."""

        matches = tuple(symbol for symbol in self.symbols if symbol.glyph == glyph)
        if len(matches) != 1:
            raise MorseError(
                f"expected exactly one Morse symbol for glyph {glyph!r}, found {len(matches)}"
            )
        return matches[0]

    def symbols_for_pattern(self, pattern: str) -> tuple[MorseSymbol, ...]:
        """Return every printable symbol assigned to a valid Morse pattern."""

        _validate_pattern(pattern)
        return tuple(symbol for symbol in self.symbols if symbol.pattern == pattern)

    def has_pattern(self, pattern: str) -> bool:
        """Return whether a valid pattern is assigned to any printable symbol."""

        return bool(self.symbols_for_pattern(pattern))


def load_morse_registry(path: Path) -> MorseRegistry:
    """Load and validate a normalized Morse reference JSON document."""

    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise MorseError(f"unable to read Morse registry: {path}") from error
    except json.JSONDecodeError as error:
        raise MorseError(f"invalid Morse registry JSON: {path}") from error

    root = _require_mapping(payload, "registry")
    symbol_values = _required_list(root, "symbols")
    symbols = tuple(
        _symbol_from_mapping(_require_mapping(item, "symbol")) for item in symbol_values
    )
    return MorseRegistry(
        schema_version=_required_int(root, "schema_version"),
        standard_id=_required_text(root, "standard_id"),
        title=_required_text(root, "title"),
        source_url=_required_text(root, "source_url"),
        scope_note=_required_text(root, "scope_note"),
        symbols=symbols,
    )


def load_verified_morse_registry(
    repository_root: Path,
    *,
    manifest_path: PurePosixPath = MORSE_MANIFEST_PATH,
    registry_path: PurePosixPath = MORSE_REFERENCE_PATH,
) -> MorseRegistry:
    """Verify the reference artifact before loading the canonical registry."""

    root = repository_root.resolve()
    manifest = load_source_manifest(root / manifest_path)
    require_verified_artifacts(manifest, root)

    matches = tuple(
        artifact for artifact in manifest.artifacts if artifact.path == str(registry_path)
    )
    if len(matches) != 1:
        raise ProvenanceError(
            f"expected one manifest artifact for {registry_path}, found {len(matches)}"
        )

    registry = load_morse_registry(root / registry_path)
    artifact = matches[0]
    if len(registry.symbols) != artifact.record_count:
        raise MorseError(
            f"manifest record_count is {artifact.record_count}, "
            f"but the Morse registry contains {len(registry.symbols)} symbols"
        )
    if registry.standard_id != MORSE_STANDARD_ID:
        raise MorseError(
            f"expected standard_id {MORSE_STANDARD_ID!r}, received {registry.standard_id!r}"
        )
    return registry


def _symbol_from_mapping(value: Mapping[str, object]) -> MorseSymbol:
    category_text = _required_text(value, "category")
    try:
        category = MorseCategory(category_text)
    except ValueError as error:
        raise MorseError(f"unsupported Morse category: {category_text!r}") from error

    return MorseSymbol(
        glyph=_required_text(value, "glyph"),
        category=category,
        pattern=_required_text(value, "pattern"),
        section=_required_text(value, "section"),
        label=_required_text(value, "label"),
    )


def _validate_pattern(pattern: str) -> None:
    if not pattern or any(character not in _VALID_PATTERN_CHARACTERS for character in pattern):
        raise MorseError("Morse pattern must contain only '.' and '-' characters")
    if len(pattern) > 8:
        raise MorseError("Morse pattern must not exceed eight elements")


def _require_mapping(value: object, field_name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise MorseError(f"{field_name} must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise MorseError(f"{field_name} keys must be strings")
    return cast(dict[str, object], value)


def _required_list(value: Mapping[str, object], field_name: str) -> list[object]:
    item = value.get(field_name)
    if not isinstance(item, list):
        raise MorseError(f"{field_name} must be a JSON array")
    return cast(list[object], item)


def _required_text(value: Mapping[str, object], field_name: str) -> str:
    item = value.get(field_name)
    if not isinstance(item, str) or not item.strip():
        raise MorseError(f"{field_name} must be a non-empty string")
    return item


def _required_int(value: Mapping[str, object], field_name: str) -> int:
    item = value.get(field_name)
    if not isinstance(item, int) or isinstance(item, bool):
        raise MorseError(f"{field_name} must be an integer")
    return item
