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
# The IEEE submission has no PDF links or bookmarks (IEEE Xplore rule); the arXiv copy keeps them.
perl -0pi -e 's/% Plain URLs: IEEE Xplore PDFs may not contain links or bookmarks\.\n\\usepackage\{url\}\n/% arXiv version: clickable citations, cross-references and URLs.\n\\usepackage{hyperref}\n\\hypersetup{hidelinks}\n/' "$OUT/main.tex"
grep -q 'usepackage{hyperref}' "$OUT/main.tex" || { echo "make_arxiv.sh: could not enable hyperref in main.tex" >&2; exit 1; }
WORK=$(mktemp -d)
rsync -R refs.bib IEEEbib.bst "$WORK/"
(cd "$OUT" && rsync -R $SOURCES "$WORK/")
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
