"""Keep the notebook generator scripts in sync with the notebooks.

The notebooks in this repo are *generated* by the `build_*.py` scripts next to this file.
That means edits can flow in two directions, and they must not be mixed up:

    python builders/sync_builders.py     notebook -> builder   (capture manual Jupyter edits)
    python builders/build_nb.py          builder  -> notebook  (regenerate; DROPS outputs)

**Always run `sync_builders.py` before touching or re-running a builder.** Otherwise any
edit made directly in Jupyter is silently destroyed by the next rebuild.

What this script does, per (builder, notebook) pair:
  1. backs up the existing builder to `<name>.py.bak` (gitignored),
  2. regenerates the builder *from* the notebook -- one `md(...)` / `code(...)` call per
     cell, preserving cell ids, choosing a string delimiter that cannot break on nested
     quotes or trailing backslashes, and inserting `# ---- heading ----` markers so the
     file stays navigable,
  3. verifies by rebuilding to a temp file and comparing every cell, printing
     `EXACT MATCH` or a per-cell diff.

It never modifies the notebooks.
"""
from pathlib import Path
import difflib
import io
import json
import os
import runpy
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

PAIRS = [
    ("build_nb.py", "neural_analysis.ipynb"),
    ("build_explore_nb.py", "neural_exploration_remapping.ipynb"),
    ("build_cursor_nb.py", "cursor_shape_analysis.ipynb"),
    ("build_state_nb.py", "cursor_state_segmentation.ipynb"),
    ("build_nc_nb.py", "neural_cursor_analysis.ipynb"),
    ("build_fig_nb.py", "figure_panels.ipynb"),
    ("build_dynamics_nb.py", "Neural_Dynamics_Analysis.ipynb"),
    ("build_recon_nb.py", "cursor_reconstruction.ipynb"),
    # NOTE: wrapup_notebooks/*.ipynb are deliberately NOT listed here. They are
    # hand-edited notebooks with no builder, so nothing can regenerate and clobber
    # a manual edit. Edit the .ipynb directly.
]

HEADER = '''"""AUTO-GENERATED from {nbname} by builders/sync_builders.py.

Run this file to regenerate the notebook (this DROPS all outputs -- re-execute afterwards).
If you edited the notebook in Jupyter, run builders/sync_builders.py FIRST so those edits
are captured here instead of being overwritten.

Set NB_OUT to write somewhere else (used by sync_builders.py to verify round-trips).
"""
import os
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def _add(kind, src, cid=None):
    c = nbf.v4.new_markdown_cell(src) if kind == "markdown" else nbf.v4.new_code_cell(src)
    if cid:
        c["id"] = cid
    cells.append(c)


def md(src, cid=None):
    _add("markdown", src, cid)


def code(src, cid=None):
    _add("code", src, cid)

'''

FOOTER = '''
nb["cells"] = cells
nb["metadata"] = {metadata!r}

out = os.environ.get("NB_OUT") or str(Path(__file__).resolve().parent.parent / {nbname!r})
with open(out, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote", out, "cells:", len(cells))
'''


def emit_string(s):
    """Python source for a literal equal to `s`, preferring readable raw triple-quotes."""
    if not s.endswith("\\"):
        if '"""' not in s and not s.endswith('"'):
            return 'r"""' + s + '"""'
        if "'''" not in s and not s.endswith("'"):
            return "r'''" + s + "'''"
    return repr(s)  # always correct, just less pretty


def build_builder_source(nb_path, nbname):
    nb = json.loads(Path(nb_path).read_text(encoding="utf-8"))
    parts = [HEADER.format(nbname=nbname)]
    for c in nb["cells"]:
        src = "".join(c["source"])
        kind = c["cell_type"]
        cid = c.get("id")
        if kind == "markdown":  # navigable marker before each heading
            first = next((l for l in src.splitlines() if l.strip()), "")
            if first.lstrip().startswith("#"):
                label = first.lstrip("# ").strip()[:70]
                parts.append("\n# " + "-" * 16 + " " + label + " " + "-" * 16 + "\n")
        call = "md" if kind == "markdown" else "code"
        arg = emit_string(src)
        parts.append(f"{call}({arg}" + (f", cid={cid!r})" if cid else ")") + "\n")
    parts.append(FOOTER.format(metadata=nb.get("metadata", {}), nbname=nbname))
    return "".join(parts)


def cells_of(path):
    nb = json.loads(Path(path).read_text(encoding="utf-8"))
    return [(c["cell_type"], "".join(c["source"])) for c in nb["cells"]]


def main():
    all_ok = True
    for builder, nbname in PAIRS:
        real = REPO / nbname
        bpath = HERE / builder
        if not real.exists():
            print(f"{nbname:38s} SKIPPED (notebook not found)")
            continue

        if bpath.exists():  # keep a backup (*.bak is gitignored)
            bpath.with_suffix(".py.bak").write_text(bpath.read_text(encoding="utf-8"), encoding="utf-8")
        bpath.write_text(build_builder_source(real, nbname), encoding="utf-8")

        # verify: rebuild to a temp file and compare cell-by-cell.
        # nbname may contain a subfolder (wrapup_notebooks/...), so flatten it -- the
        # temp file has to land directly in builders/, which exists.
        tmp = HERE / f"_verify_{Path(nbname).name}"
        env_prev = os.environ.get("NB_OUT")
        os.environ["NB_OUT"] = str(tmp)
        try:
            runpy.run_path(str(bpath), run_name="__main__")
        finally:
            if env_prev is None:
                os.environ.pop("NB_OUT", None)
            else:
                os.environ["NB_OUT"] = env_prev

        A, B = cells_of(tmp), cells_of(real)
        same = A == B
        all_ok &= same
        print(f"{nbname:38s} builder={len(A):3d} cells  notebook={len(B):3d} cells  ->  "
              + ("EXACT MATCH" if same else "*** MISMATCH ***"))
        if not same:
            for i in range(max(len(A), len(B))):
                a = A[i] if i < len(A) else ("<none>", "")
                b = B[i] if i < len(B) else ("<none>", "")
                if a != b:
                    print(f"   cell {i} differs")
                    for l in list(difflib.unified_diff(a[1].splitlines(), b[1].splitlines(),
                                                       lineterm="", n=0))[:12]:
                        print("     " + l)
        tmp.unlink(missing_ok=True)

    print("\nALL BUILDERS IN SYNC" if all_ok else "\nSOME BUILDERS STILL DIFFER")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
