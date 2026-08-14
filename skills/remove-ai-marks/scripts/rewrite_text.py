#!/usr/bin/env python3
"""Layer B optional rewrite hook for statistical (token-sampling) watermarks.

Backends:
  print-prompt       — emit prompt only (default; CI-safe, no model)
  ollama             — POST to Ollama /api/chat
  openai-compatible  — POST to OpenAI-style /v1/chat/completions

Env (optional):
  WATERMARKS_REWRITE_BACKEND
  WATERMARKS_REWRITE_BASE_URL
  WATERMARKS_REWRITE_MODEL
  WATERMARKS_REWRITE_API_KEY
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import cleaned_path, eprint, read_text_input, write_text_output  # noqa: E402
from text_unicode import clean_text  # noqa: E402

# Default per-request budget. Layer B sends whole documents to a model; without a
# cap a long input silently overruns a small local model's context and comes back
# truncated, which looks like a successful rewrite.
DEFAULT_MAX_CHUNK_CHARS = 4000

PROMPTS = {
    "paraphrase": (
        "Rewrite the following text so that every sentence uses different wording and "
        "structure while preserving all facts, numbers, names, and technical identifiers. "
        "Do not add or remove claims. Output only the rewritten text.\n\n---\n{TEXT}"
    ),
    "backtranslate_out": (
        "Translate the following text to {LANG}. Output only the translation.\n\n---\n{TEXT}"
    ),
    "backtranslate_back": (
        "Translate the following text to {ORIGINAL_LANG}. Preserve meaning; use natural "
        "phrasing. Output only the translation.\n\n---\n{TEXT}"
    ),
    "structural_outline": (
        "Extract a bullet outline of all claims and structure from the text "
        "(no full sentences). Output only the outline.\n\n---\n{TEXT}"
    ),
    "structural_write": (
        "Write a complete document from this outline in a clear professional style. "
        "Do not omit any bullet. Output only the document.\n\n---\n{TEXT}"
    ),
}


# Function-word signatures for source-language detection. Stdlib-only and
# deliberately coarse: the goal is to catch "the caller left --original-lang at
# its English default while feeding non-English text", not to classify precisely.
_LANG_STOPWORDS: dict[str, frozenset[str]] = {
    "English": frozenset(
        "the of and to in is that it for as was with on are this be by not but from have has"
        " they their which you we can more than were".split()
    ),
    "French": frozenset(
        "le la les des une un et de du dans que qui est pour sur pas plus ce cette au aux par"
        " se sont avec ils elle nous vous comme mais ou son ses leur tout".split()
    ),
    "Spanish": frozenset(
        "el la los las un una de que en y es por para con no se su al lo como más pero sus le"
        " ya este sí porque esta entre".split()
    ),
    "German": frozenset(
        "der die das und den ist von zu mit sich des auf für nicht ein eine als auch es an werden"
        " aus er hat dass sie nach bei um".split()
    ),
    "Italian": frozenset(
        "il lo la i gli le di che e un una del della per con non si sono come più anche nella"
        " alla dei sul questo ma da".split()
    ),
    "Portuguese": frozenset(
        "de que não uma dos das para com por como mais mas ao mesmo mesma mesmos pelo pela mas mais"
        " são mas mas foi mas seu sua".split()
    ),
    "Dutch": frozenset(
        "de het een en van is dat op te zijn met voor niet aan er maar om ook als door over"
        " naar dan wordt worden".split()
    ),
}

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def detect_language(text: str, *, min_words: int = 25, margin: float = 1.25) -> str | None:
    """Best-effort source-language guess, or None when the signal is too weak.

    Returns None rather than guessing on short or ambiguous input, so callers can
    fall back to an explicit flag instead of acting on a coin flip.
    """
    words = [w.lower() for w in _WORD_RE.findall(text)]
    if len(words) < min_words:
        return None
    scores = {
        lang: sum(1 for w in words if w in stops) / len(words)
        for lang, stops in _LANG_STOPWORDS.items()
    }
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best, best_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    if best_score < 0.04:
        return None
    if runner_up > 0 and best_score < runner_up * margin:
        return None
    return best


_PARA_SEP_RE = re.compile(r"\n[ \t]*\n")
_SENT_SEP_RE = re.compile(r"(?<=[.!?…])[ \t\n]+")


def _split_keeping_separators(text: str, sep_re: re.Pattern[str]) -> list[str]:
    """Split so that ''.join(result) == text (separators stay with the left part)."""
    parts: list[str] = []
    pos = 0
    for m in sep_re.finditer(text):
        if m.end() > pos:
            parts.append(text[pos : m.end()])
            pos = m.end()
    if pos < len(text):
        parts.append(text[pos:])
    return parts or ([text] if text else [])


def _hard_split(unit: str, max_chars: int) -> list[str]:
    return [unit[i : i + max_chars] for i in range(0, len(unit), max_chars)] or [unit]


def chunk_text(text: str, max_chars: int) -> list[str]:
    """Split text into chunks of at most max_chars, preferring paragraph breaks.

    Falls back to sentence breaks for oversized paragraphs, then to a hard cut for
    a single oversized sentence. Round-trips exactly: ''.join(chunks) == text.
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]

    units: list[str] = []
    for para in _split_keeping_separators(text, _PARA_SEP_RE):
        if len(para) <= max_chars:
            units.append(para)
            continue
        for sent in _split_keeping_separators(para, _SENT_SEP_RE):
            units.extend([sent] if len(sent) <= max_chars else _hard_split(sent, max_chars))

    chunks: list[str] = []
    current = ""
    for unit in units:
        if current and len(current) + len(unit) > max_chars:
            chunks.append(current)
            current = unit
        else:
            current += unit
    if current:
        chunks.append(current)
    return chunks or [text]


def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return v


def _warn_remote(base_url: str) -> None:
    host = urlparse(base_url).hostname or ""
    if host not in ("localhost", "127.0.0.1", "::1"):
        eprint(
            f"warning: rewrite base URL host is '{host}' (not localhost); "
            "content will leave this machine"
        )


def build_prompt(strength: str, text: str, *, lang: str, original_lang: str) -> str:
    # Every prompt pins the output language. The instructions are English, so a
    # model given non-English input will otherwise often answer in English —
    # silently translating the document instead of rewriting it.
    if strength == "paraphrase":
        return PROMPTS["paraphrase"].format(TEXT=text).replace(
            "Output only the rewritten text.",
            f"Write the rewritten text in {original_lang}. Output only the rewritten text.",
        )
    if strength == "backtranslate":
        if lang.strip().lower() == original_lang.strip().lower():
            raise ValueError(
                f"backtranslate pivot language ({lang}) equals the source language "
                f"({original_lang}); pick a different --lang"
            )
        # single combined instruction for print-prompt / one-shot backends
        return (
            f"Translate the text to {lang}, then translate that result back to "
            f"{original_lang}. Preserve all facts, numbers, and names. "
            f"Output only the final {original_lang} text.\n\n---\n{text}"
        )
    if strength == "structural":
        return (
            "First extract a bullet outline of all claims (no full sentences). "
            "Then write a complete document from that outline in a clear professional style "
            "without omitting any bullet. "
            f"Write the final document in {original_lang}. "
            "Output only the final document.\n\n---\n"
            f"{text}"
        )
    raise ValueError(f"unknown strength: {strength}")


def _http_json(url: str, payload: dict, headers: dict[str, str], timeout: float) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def call_ollama(base_url: str, model: str, prompt: str, timeout: float) -> str:
    url = base_url.rstrip("/") + "/api/chat"
    data = _http_json(
        url,
        {
            "model": model,
            "stream": False,
            "messages": [{"role": "user", "content": prompt}],
        },
        {},
        timeout,
    )
    msg = data.get("message") or {}
    content = msg.get("content")
    if not content:
        raise RuntimeError(f"ollama empty response: {data!r}"[:500])
    return str(content).strip()


def call_openai_compatible(
    base_url: str, model: str, prompt: str, api_key: str | None, timeout: float
) -> str:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = _http_json(
        url,
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
        },
        headers,
        timeout,
    )
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"openai-compatible empty choices: {data!r}"[:500])
    content = (choices[0].get("message") or {}).get("content")
    if not content:
        raise RuntimeError(f"openai-compatible empty content: {data!r}"[:500])
    return str(content).strip()


def rewrite(
    text: str,
    *,
    backend: str,
    model: str | None,
    base_url: str | None,
    api_key: str | None,
    strength: str,
    lang: str,
    original_lang: str,
    timeout: float,
    layer_a_after: bool,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> tuple[str, dict]:
    chunks = chunk_text(text, max_chunk_chars)
    prompts = [
        build_prompt(strength, c, lang=lang, original_lang=original_lang) for c in chunks
    ]
    info: dict = {
        "backend": backend,
        "strength": strength,
        "model": model,
        "base_url": base_url,
        "original_lang": original_lang,
        "chunks": len(chunks),
        "max_chunk_chars": max_chunk_chars,
        "prompt_chars": sum(len(p) for p in prompts),
        "input_chars": len(text),
    }

    if backend == "print-prompt":
        info["mode"] = "print-prompt"
        if len(prompts) == 1:
            return prompts[0], info
        # One prompt per chunk: run them in order and concatenate the replies.
        banner = (
            f"# {len(prompts)} chunks — send each prompt separately, in order, "
            "then join the replies with a blank line.\n"
        )
        body = "\n\n".join(
            f"===== chunk {i}/{len(prompts)} =====\n{p}" for i, p in enumerate(prompts, 1)
        )
        return banner + body, info

    if not model:
        raise SystemExit("error: --model required for ollama/openai-compatible backends")
    if not base_url:
        raise SystemExit("error: --base-url required for ollama/openai-compatible backends")

    _warn_remote(base_url)

    outputs: list[str] = []
    for i, prompt in enumerate(prompts, 1):
        if len(prompts) > 1:
            eprint(f"rewriting chunk {i}/{len(prompts)} ({len(chunks[i - 1])} chars)")
        if backend == "ollama":
            outputs.append(call_ollama(base_url, model, prompt, timeout))
        elif backend == "openai-compatible":
            outputs.append(call_openai_compatible(base_url, model, prompt, api_key, timeout))
        else:
            raise SystemExit(f"unknown backend: {backend}")
    out = "\n\n".join(outputs)

    if layer_a_after:
        out, stats = clean_text(out)
        info["layer_a_after"] = stats

    info["output_chars"] = len(out)
    info["mode"] = "rewritten"
    info["note"] = (
        "Layer B is best-effort against statistical token-sampling watermarks; "
        "cannot certify removal against a vendor detector."
    )
    return out, info


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", nargs="?", default="-", help="Input text file, or - for stdin")
    p.add_argument("-o", "--output", help="Output path (default: stdout or *.rewritten.*)")
    p.add_argument(
        "--backend",
        choices=("print-prompt", "ollama", "openai-compatible"),
        default=_env("WATERMARKS_REWRITE_BACKEND", "print-prompt"),
    )
    p.add_argument("--model", default=_env("WATERMARKS_REWRITE_MODEL"))
    p.add_argument(
        "--base-url",
        default=_env("WATERMARKS_REWRITE_BASE_URL", "http://127.0.0.1:11434"),
    )
    p.add_argument("--api-key", default=_env("WATERMARKS_REWRITE_API_KEY"))
    p.add_argument(
        "--strength",
        choices=("paraphrase", "backtranslate", "structural"),
        default="paraphrase",
    )
    p.add_argument("--lang", default="French", help="Pivot language for backtranslate")
    p.add_argument(
        "--original-lang",
        default="auto",
        help="Language to write the output in ('auto' detects it from the input)",
    )
    p.add_argument(
        "--max-chunk-chars",
        type=int,
        default=DEFAULT_MAX_CHUNK_CHARS,
        help=f"Split input into chunks of at most N chars (0 disables; default {DEFAULT_MAX_CHUNK_CHARS})",
    )
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument(
        "--no-layer-a-after",
        action="store_true",
        help="Skip Layer A scrub on model output",
    )
    p.add_argument("--json-stats", action="store_true", help="Stats JSON on stderr")
    args = p.parse_args()

    text = read_text_input(args.path)

    detected = detect_language(text)
    original_lang = args.original_lang
    if original_lang == "auto":
        if detected:
            original_lang = detected
            eprint(f"detected source language: {detected} (override with --original-lang)")
        else:
            original_lang = "English"
            eprint(
                "warning: could not detect source language (short or ambiguous input); "
                "assuming English — set --original-lang if that is wrong"
            )
    elif detected and detected.lower() != original_lang.lower():
        eprint(
            f"warning: --original-lang is {original_lang} but the input looks like {detected}; "
            "the output would be written in the wrong language"
        )

    pivot = args.lang
    if args.strength == "backtranslate" and pivot.strip().lower() == original_lang.strip().lower():
        # Round-tripping through the source language is a no-op; pick another pivot.
        pivot = "English" if original_lang.strip().lower() != "english" else "French"
        eprint(
            f"warning: --lang pivot equals the source language ({original_lang}); "
            f"using {pivot} as the pivot instead"
        )

    try:
        result, info = rewrite(
            text,
            backend=args.backend,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            strength=args.strength,
            lang=pivot,
            original_lang=original_lang,
            timeout=args.timeout,
            layer_a_after=not args.no_layer_a_after,
            max_chunk_chars=args.max_chunk_chars,
        )
    except ValueError as e:
        eprint(f"error: {e}")
        return 2
    except (urllib.error.URLError, TimeoutError, RuntimeError) as e:
        eprint(f"rewrite failed: {e}")
        return 1

    out = args.output
    if out is None and args.path not in (None, "-") and args.backend != "print-prompt":
        out = str(cleaned_path(Path(args.path), suffix=".rewritten"))
    elif out is None and args.backend == "print-prompt":
        out = "-"

    write_text_output(result, out)
    if args.json_stats:
        eprint(json.dumps(info, indent=2, ensure_ascii=False))
    else:
        eprint(
            f"backend={info['backend']} strength={info['strength']} "
            f"lang={info['original_lang']} chunks={info['chunks']} "
            f"mode={info.get('mode')} chars {info['input_chars']}->{info.get('output_chars', len(result))}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
