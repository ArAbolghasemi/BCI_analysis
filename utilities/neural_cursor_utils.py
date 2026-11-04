"""
neural_cursor_utils.py
======================

Reusable *tools* for asking what the neural population is doing during the different
**cursor-shape clusters** found in ``cursor_shape_analysis.ipynb``.

Same philosophy as the other ``*_utils`` modules: primitives only. No figure code and no
per-analysis orchestration -- the analysis flow and every plot stay visible in the notebook.

Two questions this module supports
----------------------------------
1. **How does neural dimensionality (participation ratio) evolve along a trial**, separately
   for each cursor-shape cluster?
2. **What does the average neural trajectory of each cluster look like in PC space?**

The fairness problem, and how it is handled
-------------------------------------------
The cursor-shape clusters differ enormously in trial duration -- in ``mouse_0403L`` the median
correct-trial duration is ~6.5 s for one cluster and ~0.7 s for another, a **9x** range. That
makes a naive comparison of participation ratio (PR) between clusters meaningless, because PR
is sensitive to at least four things that all covary with duration:

* **Sample count.** PR is bounded above by ``n_samples - 1`` and grows with it. A cluster with
  longer trials contributes more frames.
* **Trial count.** More trials pooled means more across-trial variance enters the covariance.
* **Real-time window width.** Neural state drifts, so a window spanning 2 s explores more
  directions than one spanning 0.2 s -- even at identical sample counts.
* **Temporal autocorrelation.** Calcium signals are smooth, so frames drawn from a short window
  are more redundant than frames drawn from a long one.

The first two are removed exactly, by construction, in :func:`matched_pr_curve`: every estimate
uses the **same number of trials** and the **same number of frames per trial**, drawn at random,
bootstrapped. The last two cannot both be removed at once, because holding the window's
*fraction of the trial* fixed and holding its *duration in seconds* fixed are mutually exclusive
when trials differ in length. So this module supports both and the notebook reports both:

* ``mode="relative"`` -- window width is a fixed **fraction of each trial** (the natural way to
  talk about "stage of the trial"; window seconds then differ between clusters).
* ``mode="absolute"`` -- window width is a fixed **number of seconds** (equalises drift and
  autocorrelation; the window then covers a different fraction of a short vs a long trial).

Agreement between the two is the evidence that a difference is not a duration artefact. They are
*not* interchangeable, and :func:`matched_pr_curve` always reports the realised window duration
so the residual confound stays visible.

Absolute PR values are biased downward by the matched sample count and are **only comparable
within one setting of** ``(n_trials, n_per_trial, mode, width)``. Compare shapes and relative
ordering, never raw magnitudes across settings.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

import neural_analysis_utils as na


# ===========================================================================
# 1 - cursor-shape clusters and period labels
# ===========================================================================
#: Periods analysed. Washout is deliberately excluded -- it is always last in the session and is
#: confounded with time-on-task and motivation (see the cursor-shape and state notebooks).
PERIODS = ["stable", "remap_intuitive", "remap_new"]


def load_cursor_shape_cache(out_dir, tag: str):
    """Load the ``(windows, trials)`` cache written by ``cursor_shape_analysis.ipynb``.

    Returns ``(M, trials_df)`` where ``M`` is the time-warped cursor matrix and ``trials_df``
    carries one row per trial with ``mouse/day/phase/block/start_frame/end_frame/outcome/...``."""
    from pathlib import Path
    out_dir = Path(out_dir)
    M = np.load(out_dir / f"M_{tag}.npy")
    T = pd.read_csv(out_dir / f"trials_{tag}.csv")
    T["day"] = T["day"].astype(str)
    if len(M) != len(T):
        raise ValueError(f"cache is inconsistent: M has {len(M)} rows, trials has {len(T)}")
    return M, T


def shape_clusters(M: np.ndarray, trials_df: pd.DataFrame, *, mouse: str, outcome: str = "correct",
                   n_clusters: int = 4, normalize: str = "none", var_target: float = 0.95):
    """Re-derive the Section-7 cursor-shape clusters for one animal.

    Recomputed rather than imported, because the pipeline is fully deterministic -- PCA (full SVD)
    followed by Ward linkage involves no randomness -- so this reproduces the notebook's labels
    exactly, and the notebook asserts that it does. Returns ``(labels, sub_df, pca_result)`` where
    ``sub_df`` is the matching slice of ``trials_df`` with its original index preserved."""
    import cursor_analysis_utils as ca
    idx = trials_df.index[(trials_df.mouse == mouse) & (trials_df.outcome == outcome)].to_numpy()
    if len(idx) == 0:
        raise ValueError(f"no {outcome} trials for {mouse}")
    X = ca.normalize_rows(M[idx], normalize)
    pca = ca.pca_reduce(X, var_target)
    labels = ca.hierarchical_labels(pca.scores, n_clusters)
    sub = trials_df.loc[idx].copy()
    sub["cluster"] = labels
    return labels, sub, pca


def assign_periods(trials_df: pd.DataFrame, *, intuitive_max_min: float = 10.0) -> pd.Series:
    """Label each trial ``stable`` / ``remap_intuitive`` / ``remap_new``, or NaN to discard.

    * ``stable``          -- any trial from a stable-control day.
    * ``remap_intuitive`` -- the pre-rotation intuitive-axis block of a remapping day, further
      restricted to the first ``intuitive_max_min`` minutes of the session.
    * ``remap_new``       -- the post-rotation new-axis block of a remapping day.
    * washout and anything else -> NaN, so it is dropped."""
    out = pd.Series(np.nan, index=trials_df.index, dtype=object)
    stable = trials_df.phase == "stable"
    out[stable] = "stable"
    remap = trials_df.phase == "remap"
    out[remap & (trials_df.block == "intuitive_pre")
        & (trials_df.t_in_session_min <= intuitive_max_min)] = "remap_intuitive"
    out[remap & (trials_df.block == "new")] = "remap_new"
    return out


# ===========================================================================
# 2 - per-trial neural segments
# ===========================================================================
def extract_trial_neural(r, trials_sub: pd.DataFrame, *, bin_frames: int = 1,
                         min_frames: int = 1, zscore: bool = True,
                         dtype=np.float32, progress=None) -> dict:
    """Per-trial neural matrices, keyed by the row index of ``trials_sub``.

    Each value is ``(samples, neurons)`` covering ``[start_frame, end_frame)``.

    ``bin_frames=1`` (default) keeps the native 30 Hz, which is what the main analysis needs: the
    shortest cursor-shape cluster has ~0.7 s trials, so at the project's usual 6 Hz a
    one-third-of-a-trial window would be barely two bins wide.

    ``bin_frames=5`` gives the project-standard 6 Hz and is used for the cross-experiment control.
    Binning is applied to the **whole session before z-scoring**, matching
    ``neural_analysis_utils.bin_and_zscore`` -- the order matters, because z-scoring is a
    per-neuron rescaling and binning changes each neuron's variance by a different amount, which
    would reweight neurons in the covariance and hence change the participation ratio.

    Trials yielding fewer than ``min_frames`` samples after binning are omitted, so callers should
    compare ``len(result)`` against ``len(trials_sub)``. Neurons are z-scored over the whole
    session (not just trial frames), so units are comparable across days and trial windows keep
    their true relative amplitude."""
    segs = {}
    b = max(1, int(bin_frames))
    days = list(trials_sub.groupby("day"))
    it = progress(days) if progress is not None else days
    for day, g in it:
        X = na.load_neural(r, day)                          # (neurons, frames)
        if b > 1:
            X = na.bin_session(X, b)                        # bin BEFORE z-scoring
        X = X.T.astype(dtype)                               # (samples, neurons)
        if zscore:
            mu = X.mean(axis=0)
            sd = X.std(axis=0)
            sd[sd < 1e-6] = 1.0
            X = (X - mu) / sd
        n = len(X)
        for i, row in g.iterrows():
            lo, hi = int(row.start_frame) // b, int(row.end_frame) // b
            if hi - lo >= min_frames and 0 <= lo and hi <= n:
                segs[i] = np.ascontiguousarray(X[lo:hi])
        del X
    return segs


def segment_frame_counts(segs: dict, ids: Sequence[int]) -> np.ndarray:
    return np.array([segs[i].shape[0] for i in ids if i in segs])


# ===========================================================================
# 3 - relative / absolute windows along a trial
# ===========================================================================
def window_bounds(n_frames: int, n_positions: int, position: int, *, mode: str = "relative",
                  width_frac: float = 1 / 3, width_frames: int = 12) -> tuple:
    """``(lo, hi)`` frame bounds of one window inside a trial of ``n_frames``.

    Window centres are spread evenly so that the first window starts at the trial start and the
    last ends at the trial end -- i.e. positions span the whole trial and **overlap** whenever the
    width exceeds the spacing. ``position`` is 0-based in ``[0, n_positions)``."""
    if mode == "relative":
        w = int(round(width_frac * n_frames))
    elif mode == "absolute":
        w = int(width_frames)
    else:
        raise ValueError(f"mode must be 'relative' or 'absolute', got {mode!r}")
    w = int(np.clip(w, 2, n_frames))
    if n_positions == 1:
        centres = np.array([n_frames / 2.0])
    else:
        centres = np.linspace(w / 2.0, n_frames - w / 2.0, n_positions)
    lo = int(round(centres[position] - w / 2.0))
    lo = int(np.clip(lo, 0, max(n_frames - w, 0)))
    return lo, lo + w


def relative_positions(n_positions: int) -> np.ndarray:
    """Window-centre positions on a 0..1 axis (0 = trial start, 1 = trial end)."""
    return np.linspace(0.0, 1.0, n_positions)


# ===========================================================================
# 4 - matched participation-ratio curves
# ===========================================================================
@contextmanager
def limited_blas(n_threads: int = 1):
    """Temporarily cap BLAS/OpenMP threads.

    The matrices here are small (a few hundred samples x ~470 neurons) but are decomposed many
    thousands of times. On this machine OpenBLAS defaults to 24 threads, and the thread-pool
    overhead then dominates the arithmetic: a single participation-ratio evaluation measured
    2.6 ms single-threaded versus 51 ms at 24 threads, and the multi-threaded timing swung
    between 11 and 107 ms run to run. Results are bit-identical either way -- this only removes
    contention. Falls through silently if ``threadpoolctl`` is unavailable."""
    try:
        import threadpoolctl
    except Exception:
        yield
        return
    with threadpoolctl.threadpool_limits(n_threads):
        yield


def participation_ratio(X: np.ndarray) -> float:
    """``(sum lambda)^2 / sum lambda^2`` of the covariance of a ``(samples, neurons)`` matrix.

    Identical in value to ``neural_analysis_utils.pca_spectrum(X).participation_ratio`` but
    evaluated on whichever of the covariance (``neurons x neurons``) or the Gram matrix
    (``samples x samples``) is smaller. The two share every non-zero eigenvalue, and zero
    eigenvalues contribute nothing to either sum, so the ratio is exact -- it is only faster.
    This matters here because the matched sample budget is deliberately small (a few hundred
    rows) while the population is ~470 neurons, and the curves need thousands of bootstraps."""
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    if n < 2:
        return float("nan")
    Xc = X - X.mean(axis=0)
    G = (Xc @ Xc.T) if n < d else (Xc.T @ Xc)
    vals = np.linalg.eigvalsh(G)
    vals = np.clip(vals, 0.0, None)
    s1 = vals.sum()
    s2 = (vals ** 2).sum()
    return float(s1 * s1 / s2) if s2 > 0 else float("nan")



@dataclass
class PRCurveConfig:
    """Everything that fixes the fairness of a PR comparison (see the module docstring)."""
    n_positions: int = 9
    mode: str = "relative"
    width_frac: float = 1 / 3
    width_sec: float = 0.4
    n_trials: int = 70          # matched trial count per (period, cluster, position)
    n_per_trial: int = 4        # matched frames drawn per trial
    n_boot: int = 25
    sr: float = 30.0
    seed: int = 0
    blas_threads: int = 1       # see limited_blas(); 1 is ~20x faster for these matrix sizes

    @property
    def width_frames(self) -> int:
        return max(2, int(round(self.width_sec * self.sr)))

    @property
    def n_samples(self) -> int:
        return self.n_trials * self.n_per_trial


def matched_pr_curve(segs: dict, ids: Sequence[int], cfg: PRCurveConfig) -> pd.DataFrame:
    """PR along the trial at a **fixed sample budget**, for one group of trials.

    For every window position: keep the trials whose window holds at least ``n_per_trial`` frames,
    draw ``n_trials`` of them at random, draw ``n_per_trial`` frames from each, stack into a
    ``(n_trials * n_per_trial, neurons)`` matrix and take its participation ratio. Repeat
    ``n_boot`` times.

    Because the trial count and the frames-per-trial are identical at every position, in every
    cluster and every period, sample-count and trial-count biases cancel. Returns one row per
    position with ``pr`` (mean over bootstraps), ``pr_sd``, ``n_available`` and ``win_sec`` --
    the realised mean window duration, which is what still differs between clusters in
    ``mode="relative"``."""
    ids = [i for i in ids if i in segs]
    rows = []
    with limited_blas(cfg.blas_threads):
      for p in range(cfg.n_positions):
          # Record only the window bounds here; the frames themselves are sliced lazily inside the
          # bootstrap, so a group of 2000 trials does not pay to copy 2000 windows when each draw
          # uses 70 of them.
          specs, widths = [], []
          for i in ids:
              nf = segs[i].shape[0]
              lo, hi = window_bounds(nf, cfg.n_positions, p, mode=cfg.mode,
                                     width_frac=cfg.width_frac, width_frames=cfg.width_frames)
              widths.append(hi - lo)
              if hi - lo >= cfg.n_per_trial:
                  specs.append((i, lo, hi - lo))
          win_sec = float(np.mean(widths) / cfg.sr) if widths else np.nan
          if len(specs) < cfg.n_trials:
              rows.append(dict(position=p, pr=np.nan, pr_sd=np.nan, n_available=len(specs),
                               win_sec=win_sec))
              continue
          prs = []
          rng = np.random.RandomState(cfg.seed + 1009 * p)
          buf = np.empty((cfg.n_trials * cfg.n_per_trial, segs[specs[0][0]].shape[1]), dtype=np.float64)
          for _ in range(cfg.n_boot):
              for row, j in enumerate(rng.choice(len(specs), cfg.n_trials, replace=False)):
                  i, lo, w = specs[j]
                  sel = lo + rng.choice(w, cfg.n_per_trial, replace=False)
                  buf[row * cfg.n_per_trial:(row + 1) * cfg.n_per_trial] = segs[i][sel]
              prs.append(participation_ratio(buf))
          rows.append(dict(position=p, pr=float(np.mean(prs)),
                           pr_sd=float(np.std(prs, ddof=1)) if cfg.n_boot > 1 else 0.0,
                           n_available=len(specs), win_sec=win_sec))
    out = pd.DataFrame(rows)
    out["rel_position"] = relative_positions(cfg.n_positions)
    return out


def per_trial_pr(segs: dict, ids: Sequence[int], *, n_positions: int = 3,
                 width_frames: int = 20, mode: str = "absolute", width_frac: float = 1 / 3,
                 sr: float = 30.0) -> pd.DataFrame:
    """One participation ratio **per trial per stage**, at a matched sample count.

    Unlike :func:`matched_pr_curve`, which pools trials to estimate a single PR per group, this
    returns one value per trial so the *distribution* across trials can be plotted (swarm/box).

    With the default ``mode="absolute"`` each stage window is exactly ``width_frames`` frames, so
    every value in the table is computed from an identical number of samples and the comparison
    between stages, periods and days is fair by construction. Trials shorter than ``width_frames``
    are omitted -- callers should report how many, because that cut is duration-biased.

    **Read the values with care.** A single trial supplies only ``width_frames`` samples for a
    ~470-neuron population, so the covariance is rank-deficient and PR is hard-capped at
    ``width_frames - 1``. Measured on this dataset the median sits at roughly 75-80 % of that cap
    with an interquartile range of only ~10 % of it: single-trial PR is therefore mostly a measure
    of *how evenly variance is spread over the few directions a short window can resolve*, not of
    the population's full dimensionality. Differences between conditions are meaningful because the
    cap is identical everywhere; the absolute numbers are not comparable with pooled PR estimates
    computed from hundreds of samples.

    Returns one row per (trial, stage) with ``pr``, the trial's ``duration_sec``, and the realised
    window bounds."""
    rows = []
    with limited_blas(1):
        for i in ids:
            S = segs.get(i)
            if S is None or S.shape[0] < max(width_frames, 2):
                continue
            nf = S.shape[0]
            for p in range(n_positions):
                lo, hi = window_bounds(nf, n_positions, p, mode=mode,
                                       width_frac=width_frac, width_frames=width_frames)
                if hi - lo < 2:
                    continue
                rows.append(dict(trial=i, stage=p, pr=participation_ratio(S[lo:hi].astype(np.float64)),
                                 win_frames=hi - lo, n_frames=nf, duration_sec=nf / sr))
    return pd.DataFrame(rows)


def pr_curves_by_group(segs: dict, groups: dict, cfg: PRCurveConfig) -> pd.DataFrame:
    """Run :func:`matched_pr_curve` for every ``{(period, cluster): ids}`` entry of ``groups``."""
    frames = []
    for key, ids in groups.items():
        d = matched_pr_curve(segs, ids, cfg)
        period, cluster = key if isinstance(key, tuple) else (key, None)
        d.insert(0, "period", period)
        d.insert(1, "cluster", cluster)
        d["n_group"] = len([i for i in ids if i in segs])
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def min_group_size(groups: dict, segs: dict) -> int:
    """Smallest group size -- the ceiling on a fair ``n_trials``."""
    return int(min(len([i for i in ids if i in segs]) for ids in groups.values()))


# ===========================================================================
# 5 - average neural trajectories in PC space
# ===========================================================================
def fit_pca_balanced(segs: dict, groups: dict, *, k: int = 6, n_per_group: Optional[int] = None,
                     seed: int = 0):
    """Fit one PC basis on frames sampled **equally from every group**.

    A shared basis is what makes trajectories comparable between clusters and between periods; the
    balancing stops the largest group (usually the longest-trial cluster) from defining the axes
    on its own. Returns a :class:`neural_analysis_utils.Subspace`."""
    rng = np.random.RandomState(seed)
    pools = {}
    for key, ids in groups.items():
        mats = [segs[i] for i in ids if i in segs]
        pools[key] = np.vstack(mats) if mats else np.empty((0, 0))
    sizes = [P.shape[0] for P in pools.values() if P.size]
    if not sizes:
        raise ValueError("no frames available to fit PCA")
    n_each = int(min(sizes) if n_per_group is None else min(min(sizes), n_per_group))
    chunks = [P[rng.choice(P.shape[0], n_each, replace=False)] for P in pools.values() if P.size]
    return na.fit_subspace(np.vstack(chunks).astype(np.float64), k)


def project(segs_or_mat, subspace) -> np.ndarray:
    """Project a ``(frames, neurons)`` matrix onto a fitted subspace."""
    X = np.asarray(segs_or_mat, dtype=np.float64)
    return (X - subspace.mean) @ subspace.components.T


def mean_trajectory(segs: dict, ids: Sequence[int], subspace, *, n_warp: int = 40,
                    n_trials: Optional[int] = None, seed: int = 0):
    """Trial-averaged neural trajectory in PC space, over normalised trial time.

    Every trial is projected onto ``subspace`` and then **linearly time-warped** to ``n_warp``
    points spanning trial start to trial end, which is the same warping the cursor-shape analysis
    uses, so "position along the trial" means the same thing in both. Returns
    ``(mean, sem, n_used)`` with ``mean``/``sem`` shaped ``(n_warp, k)``.

    *Assumption:* warping equates a fast and a slow trial that differ only in speed. Since the
    clusters differ systematically in duration, read these as trajectories through *relative*
    trial time, not real time."""
    ids = [i for i in ids if i in segs]
    if n_trials is not None and len(ids) > n_trials:
        ids = list(np.random.RandomState(seed).choice(ids, n_trials, replace=False))
    trajs = []
    for i in ids:
        Y = project(segs[i], subspace)
        if Y.shape[0] >= 2:
            trajs.append(na.resample_trajectory(Y, n_warp))
    if not trajs:
        k = subspace.components.shape[0]
        return np.full((n_warp, k), np.nan), np.full((n_warp, k), np.nan), 0
    A = np.stack(trajs, axis=0)                       # (trials, n_warp, k)
    mean = A.mean(axis=0)
    sem = A.std(axis=0, ddof=1) / np.sqrt(A.shape[0]) if A.shape[0] > 1 else np.zeros_like(mean)
    return mean, sem, A.shape[0]


def trajectory_table(segs: dict, groups: dict, subspace, *, n_warp: int = 40,
                     n_trials: Optional[int] = None, seed: int = 0) -> dict:
    """:func:`mean_trajectory` for every group, as ``{key: (mean, sem, n)}``."""
    return {key: mean_trajectory(segs, ids, subspace, n_warp=n_warp, n_trials=n_trials, seed=seed)
            for key, ids in groups.items()}
