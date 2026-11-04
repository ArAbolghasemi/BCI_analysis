# builders/

The analysis notebooks in this repo are **generated** from the scripts in this folder:

| builder | notebook |
|---|---|
| `build_nb.py` | `../neural_analysis.ipynb` |
| `build_explore_nb.py` | `../neural_exploration_remapping.ipynb` |
| `build_cursor_nb.py` | `../cursor_shape_analysis.ipynb` |
| `build_state_nb.py` | `../cursor_state_segmentation.ipynb` |
| `build_nc_nb.py` | `../neural_cursor_analysis.ipynb` |
| `build_fig_nb.py` | `../figure_panels.ipynb` |
| `build_dynamics_nb.py` | `../Neural_Dynamics_Analysis.ipynb` |

`first_pass_analysis.ipynb` and `behavior_analysis.ipynb` are **not** generated — leave them alone.

## The one rule

Edits flow in two directions, and mixing them up loses work:

```bash
# notebook -> builder   (capture edits you made by hand in Jupyter)
python builders/sync_builders.py

# builder -> notebook   (regenerate; this DROPS all outputs)
python builders/build_nb.py
```

**Always run `sync_builders.py` before editing or re-running a builder.** A rebuild
overwrites the notebook wholesale, so any manual Jupyter edit that has not been synced
back is destroyed.

## Typical workflow

1. You edit `neural_analysis.ipynb` in Jupyter.
2. `python builders/sync_builders.py` → the builder now reproduces your notebook exactly
   (it prints `EXACT MATCH` per notebook, or a per-cell diff if something is off).
3. Someone edits the builder for a larger change.
4. `python builders/build_nb.py` → notebook regenerated **without outputs**.
5. Re-execute the notebook to restore figures.

## Notes

* `sync_builders.py` never modifies the notebooks; it only rewrites the builders, keeping a
  `*.py.bak` of the previous version (`*.bak` is gitignored).
* Cell ids are preserved across sync/rebuild, so git diffs stay small.
* Each cell is emitted with a string delimiter chosen so nested quotes / trailing
  backslashes cannot break the file (`r"""` → `r'''` → `repr()` fallback).
* Set `NB_OUT` to make a builder write somewhere else — used internally to verify
  round-trips without touching the real notebook:

  ```bash
  NB_OUT=/tmp/check.ipynb python builders/build_nb.py
  ```

* Heavy results are cached under `output/<mouse>/...`; cache filenames embed the analysis
  parameters, so changing a parameter forces a recompute rather than silently reusing
  stale numbers.

## Figure control

`cursor_shape_analysis.ipynb` and `cursor_state_segmentation.ipynb` route every figure through a
`show(fig, name, level)` helper. Figures are **always saved** to `output/<analysis>/`; `FIG_LEVEL`
in the setup cell only controls which are displayed inline (`"summary"` by default, `"full"` for
everything). Per-animal repeats collapse to the primary animal unless `FIG_LEVEL="full"`.
