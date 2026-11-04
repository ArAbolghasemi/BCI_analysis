"""Git clean filter: strip a notebook's OUTPUTS on the way into git, code and prose intact.

Wired up by `tools/install_nb_filter.sh`. Git feeds the working copy in on stdin and stores
what comes out on stdout, so the file on disk -- the one open in Jupyter, with every figure
still in it -- is never touched. What git stores is the same notebook with the pictures
taken out.

It is deliberately conservative:

  * only `outputs`, `execution_count` and a short list of volatile metadata keys are removed;
  * the result is still a valid nbformat notebook, so `git show HEAD:some.ipynb` opens;
  * it is idempotent, so running it on an already-stripped notebook changes nothing;
  * **anything it cannot parse is passed through unchanged**. A filter that mangles a
    notebook it did not understand would be worse than one that lets some figures through,
    so the failure mode is "committed with outputs", never "committed corrupted".

Run it by hand to see what a commit would store:  python -I tools/nb_strip_outputs.py < nb.ipynb
"""

import json
import sys

# Per-cell metadata that changes every time a cell runs and means nothing to a reader.
CELL_META_DROP = ("execution", "collapsed", "scrolled", "ExecuteTime")

# Notebook-level metadata that can be megabytes on its own.
NB_META_DROP = ("widgets", "signature")


def strip(nb):
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        meta = cell.get("metadata")
        if isinstance(meta, dict):
            for key in CELL_META_DROP:
                meta.pop(key, None)
    meta = nb.get("metadata")
    if isinstance(meta, dict):
        for key in NB_META_DROP:
            meta.pop(key, None)
    return nb


def main():
    raw = sys.stdin.buffer.read()          # binary both ways: text mode would rewrite newlines
    try:
        nb = json.loads(raw.decode("utf-8"))
        if not isinstance(nb, dict) or "cells" not in nb:
            raise ValueError("not a notebook")
        out = json.dumps(strip(nb), indent=1, sort_keys=True, ensure_ascii=False)
        out = "\n".join(line.rstrip() for line in out.split("\n")) + "\n"
    except Exception:                      # unparseable -> hand git exactly what it gave us
        sys.stdout.buffer.write(raw)
        return
    sys.stdout.buffer.write(out.encode("utf-8"))


if __name__ == "__main__":
    main()
