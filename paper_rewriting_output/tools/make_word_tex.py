#!/usr/bin/env python3
"""Create a Pandoc-compatible TeX view without changing the canonical source."""

from __future__ import annotations

import sys
from pathlib import Path


TABLE_REPLACEMENTS = (
    (r"\begin{tabularx}{\linewidth}{p{0.20\linewidth}YY}", r"\begin{tabular}{lll}"),
    (r"\begin{tabularx}{\linewidth}{Ycc}", r"\begin{tabular}{lcc}"),
    (r"\begin{tabularx}{\linewidth}{lY}", r"\begin{tabular}{ll}"),
    (
        r"\begin{tabularx}{\linewidth}{p{0.10\linewidth}p{0.15\linewidth}Yccc}",
        r"\begin{tabular}{lllccc}",
    ),
    (
        r"\begin{tabularx}{\linewidth}{p{0.20\linewidth}p{0.20\linewidth}p{0.20\linewidth}Y}",
        r"\begin{tabular}{llll}",
    ),
    (r"\begin{tabularx}{\linewidth}{Ycccc}", r"\begin{tabular}{lcccc}"),
)


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: make_word_tex.py <main.tex> <word.tex>")
    source = Path(sys.argv[1]).read_text(encoding="utf-8")
    converted = source.replace(
        r"\newcolumntype{Y}{>{\raggedright\arraybackslash}X}", ""
    )
    for old, new in TABLE_REPLACEMENTS:
        if old not in converted:
            raise SystemExit(f"expected table signature not found: {old}")
        converted = converted.replace(old, new, 1)
    converted = converted.replace(r"\end{tabularx}", r"\end{tabular}")
    if r"\begin{tabularx}" in converted or r"\end{tabularx}" in converted:
        raise SystemExit("unconverted tabularx environment remains")
    Path(sys.argv[2]).write_text(converted, encoding="utf-8")


if __name__ == "__main__":
    main()
