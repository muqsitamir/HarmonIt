#!/bin/bash
# Build the arXiv upload, dist/isbi2027_arxiv.zip. arXiv compiles the LaTeX source itself with
# pdflatex and does not run BibTeX, so the package carries main.bbl and only the files main.tex uses.
# Works in a temporary copy, so tracked files (main.pdf) are left untouched.
set -euo pipefail
cd "$(dirname "$0")"
TEX_BIN=$(ls -d ~/Library/TinyTeX/bin/*/ 2>/dev/null | head -1 || true)
[ -n "$TEX_BIN" ] && export PATH="$TEX_BIN:$PATH"
SOURCES="main.tex spconf.sty tables/table_main.tex figures/fig_probe_verdicts.pdf figures/fig_qualitative_wide.pdf"
OUT=$(cd ../.. && pwd)/dist/isbi2027_arxiv
rm -rf "$OUT" "$OUT.zip"
mkdir -p "$OUT"
rsync -R $SOURCES "$OUT/"
WORK=$(mktemp -d)
rsync -R $SOURCES refs.bib IEEEbib.bst "$WORK/"
(cd "$WORK" && pdflatex -interaction=nonstopmode main.tex > /dev/null && bibtex main > /dev/null)
cp "$WORK/main.bbl" "$OUT/"
rm -rf "$WORK"
(cd "$OUT" && zip -qr "$OUT.zip" .)
# Compile a copy the way arXiv does (no BibTeX), so a missing file shows up before uploading.
TEST=$(mktemp -d)
cp -R "$OUT/." "$TEST/"
(cd "$TEST" && pdflatex -interaction=nonstopmode main.tex > /dev/null &&
  pdflatex -interaction=nonstopmode main.tex | grep -E "^!|Output written")
grep -E "undefined|Missing|not found" "$TEST/main.log" || true
rm -rf "$TEST"
echo "$OUT.zip"
