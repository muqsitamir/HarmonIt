#!/bin/bash
# Build the ISBI 2027 manuscript. Requires pdflatex and bibtex (e.g. TinyTeX at ~/Library/TinyTeX).
set -euo pipefail
cd "$(dirname "$0")"
TEX_BIN=$(ls -d ~/Library/TinyTeX/bin/*/ 2>/dev/null | head -1 || true)
[ -n "$TEX_BIN" ] && export PATH="$TEX_BIN:$PATH"
pdflatex -interaction=nonstopmode main.tex > /dev/null
bibtex main > /dev/null
pdflatex -interaction=nonstopmode main.tex > /dev/null
pdflatex -interaction=nonstopmode main.tex | grep -E "^!|Output written"
