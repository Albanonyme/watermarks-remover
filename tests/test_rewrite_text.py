"""Tests for Layer B rewrite_text hook (offline / print-prompt)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import pytest  # noqa: E402

from rewrite_text import (  # noqa: E402
    DEFAULT_MAX_CHUNK_CHARS,
    build_prompt,
    chunk_text,
    detect_language,
    rewrite,
)


def test_build_prompt_paraphrase_contains_text():
    p = build_prompt("paraphrase", "Hello world facts 42.", lang="French", original_lang="English")
    assert "Hello world facts 42." in p
    assert "Rewrite" in p or "rewrite" in p.lower()


def test_print_prompt_backend():
    out, info = rewrite(
        "Sample prose about water marks.",
        backend="print-prompt",
        model=None,
        base_url=None,
        api_key=None,
        strength="paraphrase",
        lang="French",
        original_lang="English",
        timeout=5.0,
        layer_a_after=True,
    )
    assert info["mode"] == "print-prompt"
    assert "Sample prose" in out
    assert info["backend"] == "print-prompt"


def test_structural_and_backtranslate_prompts():
    for strength in ("structural", "backtranslate"):
        p = build_prompt(strength, "ABC 123", lang="German", original_lang="English")
        assert "ABC 123" in p


# --- chunking -------------------------------------------------------------

FR = (
    "Le poulpe a construit son intelligence dans un couloir evolutif separe du notre. "
    "Notre dernier ancetre commun vivait il y a six cents millions d'annees. "
)


@pytest.mark.parametrize("max_chars", [1, 7, 50, 200, 1000, 100000])
def test_chunk_text_round_trips_exactly(max_chars):
    """Chunking must never lose or duplicate a character."""
    text = (FR * 20) + "\n\nUn paragraphe final.\n"
    assert "".join(chunk_text(text, max_chars)) == text


@pytest.mark.parametrize("max_chars", [50, 200, 1000])
def test_chunk_text_respects_limit(max_chars):
    text = (FR * 20) + "\n\nUn paragraphe final.\n"
    assert all(len(c) <= max_chars for c in chunk_text(text, max_chars))


def test_chunk_text_short_input_is_single_chunk():
    assert chunk_text("court", 4000) == ["court"]


def test_chunk_text_zero_disables_chunking():
    text = FR * 50
    assert chunk_text(text, 0) == [text]


def test_chunk_text_prefers_paragraph_breaks():
    text = "a" * 100 + "\n\n" + "b" * 100 + "\n\n" + "c" * 100
    chunks = chunk_text(text, 150)
    assert "".join(chunks) == text
    # each chunk should end at a paragraph boundary rather than mid-run
    assert all(set(c.strip()) <= {"a", "b", "c"} for c in chunks)


def test_rewrite_chunks_long_input():
    text = FR * 60
    out, info = rewrite(
        text,
        backend="print-prompt",
        model=None,
        base_url=None,
        api_key=None,
        strength="paraphrase",
        lang="English",
        original_lang="French",
        timeout=5.0,
        layer_a_after=False,
        max_chunk_chars=1000,
    )
    assert info["chunks"] > 1
    assert "===== chunk 1/" in out
    assert f"===== chunk {info['chunks']}/" in out


def test_rewrite_single_chunk_has_no_banner():
    out, info = rewrite(
        "Texte court.",
        backend="print-prompt",
        model=None,
        base_url=None,
        api_key=None,
        strength="paraphrase",
        lang="English",
        original_lang="French",
        timeout=5.0,
        layer_a_after=False,
    )
    assert info["chunks"] == 1
    assert "===== chunk" not in out


# --- source language ------------------------------------------------------


def test_detect_language_french():
    assert detect_language(FR * 5) == "French"


def test_detect_language_english():
    text = (
        "The octopus is not like the brain of a dog, which we can recognise from our own "
        "anatomy, and that is what makes it so hard to think about. "
    ) * 5
    assert detect_language(text) == "English"


def test_detect_language_returns_none_when_too_short():
    assert detect_language("Bonjour.") is None


def test_paraphrase_prompt_pins_output_language():
    """Regression: English instructions on French input made models reply in English."""
    p = build_prompt("paraphrase", FR, lang="English", original_lang="French")
    assert "in French" in p


def test_structural_prompt_pins_output_language():
    p = build_prompt("structural", FR, lang="English", original_lang="French")
    assert "in French" in p


def test_backtranslate_rejects_degenerate_pivot():
    """Pivoting through the source language is a no-op, not a rewrite."""
    with pytest.raises(ValueError, match="pivot"):
        build_prompt("backtranslate", FR, lang="French", original_lang="French")


def test_backtranslate_valid_pivot_directions():
    p = build_prompt("backtranslate", FR, lang="English", original_lang="French")
    assert "to English" in p and "back to French" in p
    assert "final French text" in p


def test_default_chunk_budget_is_positive():
    assert DEFAULT_MAX_CHUNK_CHARS > 0
