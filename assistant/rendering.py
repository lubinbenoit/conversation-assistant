from __future__ import annotations

import re

from pylatexenc.latex2text import LatexNodes2Text


_CONVERTER = LatexNodes2Text()
_FENCE = re.compile(r"^[ \t]*(`{3,}|~{3,})")
_INLINE_CODE = re.compile(r"(`+)(.+?)\1", re.DOTALL)
_MATH = re.compile(
    r"(?<!\\)\$\$.+?(?<!\\)\$\$"
    r"|(?<!\\)\\\[.+?(?<!\\)\\\]"
    r"|(?<!\\)\\\(.+?(?<!\\)\\\)"
    # A leading digit is treated as currency, per the UI's documented rule.
    r"|(?<![\\$])\$(?![$\d])[^\n]+?(?<!\\)\$(?!\$)",
    re.DOTALL,
)


def _escape_markdown(text: str) -> str:
    for character in "\\`*_{}[]<>#":
        text = text.replace(character, f"\\{character}")
    return text


def _convert_math(match: re.Match[str]) -> str:
    source = match.group(0)
    try:
        converted = _CONVERTER.latex_to_text(source).strip()
    except Exception:
        return source
    if not converted:
        return source
    converted = _escape_markdown(converted)
    if source.startswith(("$$", r"\[")):
        return f"\n\n{converted}\n\n"
    return converted


def _convert_prose(text: str) -> str:
    parts = _INLINE_CODE.split(text)
    # re.split() includes the delimiter and body capture groups. Every group of
    # three after the first is: backticks, code contents, then following prose.
    rendered = [parts[0]]
    for index in range(1, len(parts), 3):
        marker = parts[index]
        code = parts[index + 1]
        rendered.append(f"{marker}{code}{marker}")
        rendered.append(parts[index + 2])

    for index in range(0, len(rendered), 2):
        rendered[index] = _MATH.sub(_convert_math, rendered[index])
    return "".join(rendered)


def render_assistant_markdown(text: str) -> str:
    """Convert LaTeX math in Markdown prose to terminal-friendly Unicode."""
    rendered: list[str] = []
    prose: list[str] = []
    fence_character: str | None = None
    fence_length = 0

    def flush_prose() -> None:
        if prose:
            rendered.append(_convert_prose("".join(prose)))
            prose.clear()

    for line in text.splitlines(keepends=True):
        match = _FENCE.match(line)
        if fence_character is None:
            if match is None:
                prose.append(line)
                continue
            flush_prose()
            marker = match.group(1)
            fence_character = marker[0]
            fence_length = len(marker)
            rendered.append(line)
            continue

        rendered.append(line)
        if match is not None:
            marker = match.group(1)
            if (
                marker[0] == fence_character
                and len(marker) >= fence_length
                and not line[match.end():].strip()
            ):
                fence_character = None
                fence_length = 0

    flush_prose()
    return "".join(rendered)
