#!/bin/sh
# Teach this clone to strip notebook outputs on commit. Run once per clone:
#
#     sh tools/install_nb_filter.sh
#
# It writes to .git/config only -- nothing here is committed, and no file on disk is changed.
set -e
cd "$(git rev-parse --show-toplevel)"

# Prefer the project env, fall back to whatever python is on PATH.
PY="C:/Users/alireza/miniconda3/envs/BCI_analysis/python.exe"
[ -x "$PY" ] || PY="$(command -v python3 || command -v python)"
[ -n "$PY" ] || { echo "no python found; set PY by hand in this script" >&2; exit 1; }

git config filter.nbstrip.clean "'$PY' -I tools/nb_strip_outputs.py"
git config filter.nbstrip.smudge cat
git config filter.nbstrip.required true   # fail loudly rather than quietly commit the figures

echo "installed:"
git config --get-regexp '^filter\.nbstrip\.'
echo
echo "Now run:  git add --renormalize '*.ipynb'"
echo "to restage the already-committed notebooks without their outputs."
