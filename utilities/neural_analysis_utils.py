"""
neural_analysis_utils.py
========================

Reusable *tools* for neural population analysis of the Hakim-lab BMI packets
(richfile datasets in ``C:\\Data\\datasets\\bmi_packets``).

Design philosophy (per project preference)
-------------------------------------------
This module contains only reusable, well-scoped computational tools -- data
access, segment extraction, dimensionality, trajectory geometry, subspace
alignment, sliding windows. It deliberately contains **no figure code and no
per-analysis orchestration**: the analysis flow and all plotting live in the
notebook so they stay visible and easy to edit during exploration.

Data facts (verified empirically, mouse_0403L; see also first_pass_analysis)
----------------------------------------------------------------------------
* ``r["data_neural_aligned"][day].load()`` -> ``(n_neurons, n_frames)`` float32,
  non-negative firing-rate-like values (NOT z-scored), 30 Hz. n_neurons is
  constant across days (aligned/tracked), so cross-day subspaces are comparable.
* Trials logger columns include ``frameNum_trialStart``, ``frameNum_trialEnd``,
  ``success_outcome`` (1=hit, 0=miss), ``factor_to_use`` (2=intuitive axis,
  3=new/rotated axis on remapping days), ``blockNum`` and the trialType flags.
* A "true BMI trial" has ``trialType_cursorOn == trialType_feedbackLinked ==
  trialType_rewardOn == 1``.
* Miss trials last ~20 s (the max allowed trial duration in this dataset).
* Reward is delivered ~13 frames (~0.43 s) AFTER ``frameNum_trialEnd``, so the
  window ``[trialStart, trialEnd]`` for a hit is free of reward consumption.
* Active rewards: ``reward_times_typed[(1, 1, 1, 1)]`` (frames, 30 Hz).

Note: decoder / factor *weights* (``weights_day0_idxFactors``) are considered
unreliable for now and are intentionally NOT used here. ``factor_to_use`` is
used only as a categorical trial label (which axis), never as a weight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.ndimage import gaussian_filter1d
from scipy.linalg import subspace_angles
from sklearn.decomposition import FactorAnalysis
from sklearn.model_selection import KFold


# ---------------------------------------------------------------------------
# richfile access
# ---------------------------------------------------------------------------
def register_richfile_types(rfs) -> None:
    """Register numpy / pandas / scipy-sparse loaders on a ``richfile`` module.

    Mirrors the canonical registration in ``first_pass_analysis.ipynb``. Safe to
    call repeatedly (each type is only registered once).
    """
    have = {p["type_name"] for p in rfs.functions._TYPE_LOOKUP}

    def _reg(name, load, save, cls, lib, suf):
        if name not in have:
            rfs.functions.register_type(
                type_name=name, function_load=load, function_save=save,
                object_class=cls, library=lib, suffix=suf,
            )

    _reg("pandas_dataframe", lambda path: pd.read_csv(path),
         lambda path, obj: obj.to_csv(path), pd.DataFrame, "pandas", "csv")
    _reg("pandas_series", lambda path: pd.read_csv(path),
         lambda path, obj: obj.to_csv(path), pd.Series, "pandas", "csv")
    _reg("scipy_sparse", lambda path: sparse.load_npz(path),
         lambda path, obj: sparse.save_npz(path, obj), sparse.csr_matrix, "scipy", "npz")
    _reg("numpy_array", lambda path: np.load(path, allow_pickle=True),
         lambda path, obj: np.save(path, obj), np.ndarray, "numpy", "npy")
    _reg("scipy_sparse_array", lambda path: sparse.load_npz(path).toarray(),
         lambda path, obj: sparse.save_npz(path, sparse.csr_matrix(obj)), np.ndarray, "scipy", "npz")


def open_mouse(rfs, data_root, mouse_name: str):
    """Open a mouse ``.richfile`` and return the RichFile handle."""
    import os
    return rfs.RichFile(os.path.join(str(data_root), f"{mouse_name}.richfile"))


def get_day_list(r) -> list[str]:
    """Ordered list of recording-day keys (as stored under data_neural_aligned)."""
    return list(r["data_neural_aligned"].keys())


def load_neural(r, day: str) -> np.ndarray:
    """Load one day of aligned neural data as ``(n_neurons, n_frames)`` float64."""
    return np.asarray(r["data_neural_aligned"][day].load(), dtype=np.float64)


# trials logger column order used when the logger is stored as a bare array
_TRIAL_COLS = [
    "trialNum_trialStart", "time_now_trialStart", "frameNum_trialStart",
    "trialType_cursorOn", "trialType_feedbackLinked", "trialType_rewardOn",
    "blockNum", "factor_to_use", "trialNum_trialEnd", "time_now_trialEnd",
    "frameNum_trialEnd", "success_outcome",
]


def get_trials_df(r, day: str) -> pd.DataFrame:
    """Return the trials logger for ``day`` as a DataFrame.

    Handles both storage forms seen in this dataset (a saved DataFrame under
    ``extras/logger_dfs`` or a bare array under ``extras/loggers``).
    """
    if "logger_dfs" in r["extras"].keys():
        return r["extras"]["logger_dfs"][day]["trials"].load()
    arr = r["extras"]["loggers"][day]["trials"].load()
    return pd.DataFrame(arr, columns=_TRIAL_COLS)


def get_active_reward_frames(r, day: str) -> np.ndarray:
    """Active-reward frame indices (cursor+feedback+reward+lick all on)."""
    rtt = r["reward_times_typed"][day].load()
    key = (1, 1, 1, 1)
    if key not in rtt:
        return np.array([], dtype=float)
    return np.asarray(rtt[key], dtype=float).ravel()


# ---------------------------------------------------------------------------
# trial table
# ---------------------------------------------------------------------------
@dataclass
class Trials:
    """Tidied true-BMI-trial table for one day (frames at 30 Hz)."""
    start: np.ndarray          # frameNum_trialStart
    end: np.ndarray            # frameNum_trialEnd (== reward-achievement moment for hits)
    success: np.ndarray        # 1 = hit, 0 = miss
    factor: np.ndarray         # factor_to_use (2 = intuitive axis, 3 = new axis)
    block: np.ndarray          # blockNum

    def __len__(self):
        return len(self.start)

    def mask(self, *, success: Optional[int] = None, factor: Optional[float] = None) -> np.ndarray:
        m = np.ones(len(self), dtype=bool)
        if success is not None:
            m &= (self.success == success)
        if factor is not None:
            m &= (self.factor == factor)
        return m


def get_true_bmi_trials(trials_df: pd.DataFrame) -> Trials:
    """Filter to true BMI trials and return a :class:`Trials` with float arrays."""
    df = trials_df
    is_true = (
        (df["trialType_feedbackLinked"] == 1)
        & (df["trialType_cursorOn"] == 1)
        & (df["trialType_rewardOn"] == 1)
    )
    t = df[is_true]
    return Trials(
        start=t["frameNum_trialStart"].to_numpy(dtype=float),
        end=t["frameNum_trialEnd"].to_numpy(dtype=float),
        success=t["success_outcome"].to_numpy(dtype=float),
        factor=t["factor_to_use"].to_numpy(dtype=float),
        block=t["blockNum"].to_numpy(dtype=float),
    )


def remapping_start_frame(trials: Trials) -> Optional[float]:
    """Frame at which the axis rotates (first trial whose factor differs from the
    day's first factor). Returns None if the day has a single factor."""
    fac = trials.factor[np.isfinite(trials.factor)]
    if fac.size == 0 or np.unique(fac).size <= 1:
        return None
    base = trials.factor[0]
    changed = np.where((trials.factor != base) & np.isfinite(trials.factor))[0]
    if changed.size == 0:
        return None
    # the new-axis block starts right after the last old-axis trial ends
    i = changed[0]
    return float(trials.end[i - 1]) if i > 0 else float(trials.start[i])


# ---------------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------------
def preprocess_session(
    neural: np.ndarray,
    *,
    smooth_sigma_frames: float = 5.0,
    zscore: bool = True,
    eps: float = 1e-9,
) -> np.ndarray:
    """Smooth (Gaussian over time) then per-neuron z-score a ``(N, T)`` session.

    Smoothing is applied on the continuous session *before* cutting into
    segments, so segment edges use real neighbouring samples. Raw 30 Hz data in
    this dataset is noise-dominated (participation ratio ~170); modest smoothing
    concentrates variance into an interpretable number of dimensions. Set
    ``smooth_sigma_frames=0`` to disable.
    """
    x = np.asarray(neural, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if smooth_sigma_frames and smooth_sigma_frames > 0:
        x = gaussian_filter1d(x, sigma=float(smooth_sigma_frames), axis=1, mode="nearest")
    if zscore:
        mu = x.mean(axis=1, keepdims=True)
        sd = x.std(axis=1, keepdims=True)
        sd[sd < eps] = 1.0
        x = (x - mu) / sd
    return x


# ---------------------------------------------------------------------------
# condition -> frame intervals
# ---------------------------------------------------------------------------
# Each "condition" is expressed as a list of half-open frame intervals
# ``(start, end)``. This keeps the *definition* of each condition explicit and
# separate from the matrix/trajectory building, so conditions are easy to tweak.

def _clip_interval(s: float, e: float, T: int) -> Optional[tuple[int, int]]:
    s = int(round(s)); e = int(round(e))
    s = max(s, 0); e = min(e, T)
    return (s, e) if e > s else None


def correct_trial_intervals(trials: Trials, T: int, *, factor: Optional[float] = None) -> list[tuple[int, int]]:
    """Hits: ``[trialStart, trialEnd]`` (variable length; free of reward consumption).

    Optionally restrict to one axis via ``factor`` (2=intuitive, 3=new)."""
    m = trials.mask(success=1, factor=factor)
    out = []
    for s, e in zip(trials.start[m], trials.end[m]):
        iv = _clip_interval(s, e, T)
        if iv:
            out.append(iv)
    return out


def incorrect_trial_intervals(trials: Trials, T: int, *, window_sec: float = 20.0,
                              sr: int = 30, factor: Optional[float] = None) -> list[tuple[int, int]]:
    """Misses: fixed window ``[trialStart, trialStart + window_sec]`` (20 s = max
    trial duration in this dataset)."""
    m = trials.mask(success=0, factor=factor)
    w = int(round(window_sec * sr))
    out = []
    for s in trials.start[m]:
        iv = _clip_interval(s, s + w, T)
        if iv:
            out.append(iv)
    return out


def reward_consumption_intervals(reward_frames: Sequence[float], T: int, *,
                                 dur_sec: float = 7.0, sr: int = 30) -> list[tuple[int, int]]:
    """Reward consumption: ``[rewardFrame, rewardFrame + dur_sec]`` per active reward."""
    w = int(round(dur_sec * sr))
    out = []
    for rf in np.asarray(reward_frames, dtype=float).ravel():
        iv = _clip_interval(rf, rf + w, T)
        if iv:
            out.append(iv)
    return out


def baseline_intervals(trials: Trials, T: int, *, exclude_sec: float = 10.0,
                       sr: int = 30, min_len_sec: float = 1.0) -> list[tuple[int, int]]:
    """Baseline: contiguous stretches with no trial start within +/- ``exclude_sec``.

    Returns intervals at least ``min_len_sec`` long."""
    keep = np.ones(T, dtype=bool)
    half = int(round(exclude_sec * sr))
    for s in trials.start[np.isfinite(trials.start)]:
        s = int(round(s))
        keep[max(0, s - half): min(T, s + half + 1)] = False
    # contiguous True runs
    out = []
    min_len = int(round(min_len_sec * sr))
    i = 0
    while i < T:
        if keep[i]:
            j = i
            while j < T and keep[j]:
                j += 1
            if j - i >= min_len:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


def intervals_within(intervals: Sequence[tuple[int, int]], w0: int, w1: int,
                     *, min_len: int = 1) -> list[tuple[int, int]]:
    """Clip a list of intervals to the window ``[w0, w1)`` (for sliding windows)."""
    out = []
    for s, e in intervals:
        s2, e2 = max(s, w0), min(e, w1)
        if e2 - s2 >= min_len:
            out.append((s2, e2))
    return out


# ---------------------------------------------------------------------------
# intervals -> data matrices / trajectories
# ---------------------------------------------------------------------------
def intervals_to_matrix(neural: np.ndarray, intervals: Sequence[tuple[int, int]]) -> np.ndarray:
    """Concatenate frames from ``intervals`` into a ``(n_samples, n_neurons)`` matrix.

    This is the ``N x TK -> (TK) x N`` design matrix for PCA / FA: trial-time
    points are observations, neurons are features. Trial identity/length is
    discarded (fine for covariance/likelihood estimation)."""
    if len(intervals) == 0:
        return np.empty((0, neural.shape[0]))
    return np.concatenate([neural[:, s:e] for s, e in intervals], axis=1).T


def intervals_to_trajectories(neural: np.ndarray,
                              intervals: Sequence[tuple[int, int]]) -> list[np.ndarray]:
    """One ``(t, n_neurons)`` trajectory per interval (keeps trial identity)."""
    return [neural[:, s:e].T for s, e in intervals]


def n_samples(intervals: Sequence[tuple[int, int]]) -> int:
    return int(sum(e - s for s, e in intervals))


# ---------------------------------------------------------------------------
# dimensionality
# ---------------------------------------------------------------------------
@dataclass
class PCASpectrum:
    eigenvalues: np.ndarray        # explained_variance_ (descending)
    ratio: np.ndarray              # explained_variance_ratio_
    cumulative: np.ndarray         # cumulative ratio
    participation_ratio: float     # (sum l)^2 / sum l^2
    n_samples: int
    n_features: int


def _covariance_eigvals(X: np.ndarray) -> np.ndarray:
    """Descending eigenvalues of the (feature x feature) covariance of ``X``.

    Equivalent to PCA's ``explained_variance_`` but computed via the covariance
    matrix + ``eigvalsh`` -- far cheaper than a full SVD when we need the *whole*
    spectrum (as the participation ratio does)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    n = Xc.shape[0]
    if n < 2:
        return np.zeros(X.shape[1])
    cov = (Xc.T @ Xc) / (n - 1)
    ev = np.linalg.eigvalsh(cov)          # ascending
    return np.clip(ev[::-1], 0.0, None)   # descending, non-negative


def pca_spectrum(X: np.ndarray, *, max_components: Optional[int] = None) -> PCASpectrum:
    """Full PCA spectrum + participation ratio of a ``(samples, features)`` matrix."""
    ev = _covariance_eigvals(X)
    total = ev.sum()
    ratio = ev / total if total > 0 else np.zeros_like(ev)
    pr = float((total ** 2) / np.sum(ev ** 2)) if np.sum(ev ** 2) > 0 else np.nan
    return PCASpectrum(ev, ratio, np.cumsum(ratio), pr, X.shape[0], X.shape[1])


def dim_at_variance(spectrum: PCASpectrum, frac: float = 0.8) -> int:
    """Number of PCs needed to reach ``frac`` cumulative explained variance."""
    return int(np.searchsorted(spectrum.cumulative, frac) + 1)


def subsample_rows(X: np.ndarray, n: int, *, seed: int = 0) -> np.ndarray:
    """Randomly pick ``n`` rows (time points) of ``X`` without replacement."""
    if X.shape[0] <= n:
        return X
    rng = np.random.RandomState(seed)
    return X[rng.choice(X.shape[0], n, replace=False)]


def participation_ratio_matched(
    X: np.ndarray, n_samples: int, *, n_boot: int = 10, seed: int = 0,
) -> tuple[float, float]:
    """Participation ratio at a *fixed* sample count (mean +/- sd over bootstraps).

    Dimensionality estimates depend on the number of samples, so to compare
    conditions / days / windows fairly (the proposal's "matched trial counts")
    every group should be evaluated at the same ``n_samples``. Returns
    ``(nan, nan)`` if ``X`` has fewer than ``n_samples`` rows."""
    if X.shape[0] < n_samples:
        return np.nan, np.nan
    if X.shape[0] == n_samples:
        return pca_spectrum(X).participation_ratio, 0.0
    prs = []
    for b in range(n_boot):
        rng = np.random.RandomState(seed + b)
        Xb = X[rng.choice(X.shape[0], n_samples, replace=False)]
        prs.append(pca_spectrum(Xb).participation_ratio)
    return float(np.mean(prs)), float(np.std(prs, ddof=1) if n_boot > 1 else 0.0)


def cv_fa_dimensionality(
    X: np.ndarray,
    dims: Sequence[int] = (2, 3, 5, 8, 12, 18, 25, 35, 50),
    *,
    n_folds: int = 4,
    sample_cap: Optional[int] = 8000,
    max_iter: int = 500,
    seed: int = 0,
    n_jobs: int = 1,
) -> dict:
    """Cross-validated Factor-Analysis dimensionality (the proposal's method).

    For each candidate dimensionality, run ``n_folds``-fold CV: fit FA on the
    training folds (~75 %) and evaluate held-out average log-likelihood on the
    remaining fold (~25 %). Average across folds. The effective dimensionality is
    the ``dims`` value that maximises held-out log-likelihood.

    ``sample_cap`` randomly subsamples time points before CV (FA on hundreds of
    features is the bottleneck); set to None to use all samples. Returns a dict
    with ``dims``, ``mean_ll``, ``sem_ll``, ``best_dim`` and ``best_is_edge``
    (True if the optimum sits at the largest tested dim -> raise the grid).
    """
    dims = [int(d) for d in dims]
    rng = np.random.RandomState(seed)
    Xs = X
    if sample_cap is not None and X.shape[0] > sample_cap:
        Xs = X[rng.choice(X.shape[0], sample_cap, replace=False)]

    if Xs.shape[0] < n_folds + 1:
        nan = np.full(len(dims), np.nan)
        return dict(dims=np.array(dims), mean_ll=nan, sem_ll=nan,
                    best_dim=np.nan, best_is_edge=False, n_samples=Xs.shape[0])

    # only test dims that FA can fit (n_components < n_features)
    dims = [d for d in dims if d < Xs.shape[1]]
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = list(kf.split(Xs))

    def _score(d, tr, te):
        fa = FactorAnalysis(n_components=d, max_iter=max_iter)
        fa.fit(Xs[tr])
        return fa.score(Xs[te])

    jobs = [(d, tr, te) for d in dims for (tr, te) in splits]
    if n_jobs != 1:
        # threading backend: FA fits are BLAS-heavy (release the GIL) and this
        # avoids loky's process-pool resource-tracker noise on Windows.
        from joblib import Parallel, delayed
        flat = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(_score)(d, tr, te) for d, tr, te in jobs)
    else:
        flat = [_score(d, tr, te) for d, tr, te in jobs]

    flat = np.array(flat).reshape(len(dims), n_folds)
    mean_ll = flat.mean(axis=1)
    sem_ll = flat.std(axis=1, ddof=1) / np.sqrt(n_folds)
    best_i = int(np.argmax(mean_ll))
    return dict(
        dims=np.array(dims), mean_ll=mean_ll, sem_ll=sem_ll,
        best_dim=int(dims[best_i]), best_is_edge=(best_i == len(dims) - 1),
        n_samples=Xs.shape[0],
    )


# ---------------------------------------------------------------------------
# trajectory geometry (correct trials)
# ---------------------------------------------------------------------------
@dataclass
class Subspace:
    """A fixed k-dimensional PCA subspace to project trajectories into."""
    components: np.ndarray   # (k, n_neurons) orthonormal rows
    mean: np.ndarray         # (n_neurons,)
    explained_ratio: np.ndarray

    @property
    def k(self) -> int:
        return self.components.shape[0]

    @property
    def basis(self) -> np.ndarray:
        """(n_neurons, k) orthonormal basis (columns) for subspace-angle math."""
        return self.components.T

    def project(self, traj_or_matrix: np.ndarray) -> np.ndarray:
        """Project a ``(t, n_neurons)`` trajectory/matrix to ``(t, k)``."""
        return (traj_or_matrix - self.mean) @ self.components.T


def fit_subspace(X: np.ndarray, k: int) -> Subspace:
    """Fit a k-PC subspace on a ``(samples, neurons)`` matrix (via covariance eigh)."""
    mean = X.mean(axis=0)
    Xc = X - mean
    n = Xc.shape[0]
    cov = (Xc.T @ Xc) / max(n - 1, 1)
    vals, vecs = np.linalg.eigh(cov)          # ascending
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    total = np.clip(vals, 0.0, None).sum()
    k = min(k, vecs.shape[1])
    components = vecs[:, :k].T                 # (k, n_neurons), orthonormal rows
    ratio = np.clip(vals[:k], 0.0, None) / total if total > 0 else np.zeros(k)
    return Subspace(components, mean, ratio)


def trajectory_length(traj: np.ndarray) -> float:
    """Path length: sum of Euclidean steps along a ``(t, k)`` trajectory."""
    if traj.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)))


def resample_trajectory(traj: np.ndarray, n_points: int) -> np.ndarray:
    """Linearly resample a ``(t, k)`` trajectory to ``n_points`` over normalized time."""
    t = traj.shape[0]
    if t == n_points:
        return traj
    src = np.linspace(0.0, 1.0, t)
    dst = np.linspace(0.0, 1.0, n_points)
    return np.column_stack([np.interp(dst, src, traj[:, j]) for j in range(traj.shape[1])])


@dataclass
class TrajectoryStats:
    n_trials: int
    mean_length: float
    length_sd: float
    var_raw: float             # mean over normalized-time of cross-trial variance (sum over dims)
    var_per_length: float      # var_raw / mean_length**2  (length-deconfounded)
    var_per_meanpath: float    # var_raw / variance of the mean trajectory (length-deconfounded)


def trajectory_stats(trajs: Sequence[np.ndarray], *, n_resample: int = 50) -> TrajectoryStats:
    """Trajectory length + trial-to-trial variability, de-confounded from length.

    ``var_raw`` time-normalizes every trajectory to ``n_resample`` points, takes
    the cross-trial variance at each normalized-time point (summed over dims) and
    averages over normalized time. Because it is computed at matched
    normalized-time points, it measures how much trajectories *differ from each
    other* rather than how long each one is -- but its absolute scale still grows
    with trajectory size, so two length-deconfounded versions are also returned:

    * ``var_per_length``   = var_raw / mean_length**2
    * ``var_per_meanpath`` = var_raw / Var(mean trajectory)   (unitless shape spread)
    """
    trajs = [t for t in trajs if t.shape[0] >= 2]
    n = len(trajs)
    if n == 0:
        return TrajectoryStats(0, np.nan, np.nan, np.nan, np.nan, np.nan)
    lengths = np.array([trajectory_length(t) for t in trajs])
    mean_len = float(np.mean(lengths))

    if n >= 2:
        R = np.stack([resample_trajectory(t, n_resample) for t in trajs], axis=0)  # (n, P, k)
        mean_traj = R.mean(axis=0)                                                 # (P, k)
        # cross-trial variance at each normalized-time point, summed over dims
        var_t = np.sum(R.var(axis=0, ddof=1), axis=1)                              # (P,)
        var_raw = float(np.mean(var_t))
        meanpath_var = float(np.sum(mean_traj.var(axis=0)))
    else:
        var_raw, meanpath_var = np.nan, np.nan

    return TrajectoryStats(
        n_trials=n,
        mean_length=mean_len,
        length_sd=float(np.std(lengths, ddof=1)) if n >= 2 else np.nan,
        var_raw=var_raw,
        var_per_length=var_raw / mean_len ** 2 if mean_len > 0 else np.nan,
        var_per_meanpath=var_raw / meanpath_var if meanpath_var and meanpath_var > 0 else np.nan,
    )


# ---------------------------------------------------------------------------
# subspace alignment (principal angles)
# ---------------------------------------------------------------------------
def principal_angles_deg(A: Subspace | np.ndarray, B: Subspace | np.ndarray) -> np.ndarray:
    """Principal angles (degrees, ascending) between two subspaces.

    Accepts :class:`Subspace` objects or ``(n_neurons, k)`` basis matrices."""
    Ab = A.basis if isinstance(A, Subspace) else np.asarray(A)
    Bb = B.basis if isinstance(B, Subspace) else np.asarray(B)
    return np.degrees(subspace_angles(Ab, Bb))


def alignment_index(A: Subspace | np.ndarray, B: Subspace | np.ndarray) -> float:
    """Scalar subspace alignment in [0, 1]: mean of cos^2(principal angles).

    1 = subspaces coincide, 0 = orthogonal. (Mean over min(kA, kB) angles.)"""
    ang = np.radians(principal_angles_deg(A, B))
    return float(np.mean(np.cos(ang) ** 2))


def variance_capture(source: Subspace, target_matrix: np.ndarray) -> float:
    """Fraction of ``target_matrix`` variance captured by ``source`` subspace.

    Projects the target data onto ``source`` and returns explained-variance
    fraction. Asymmetric: "how much of what the animal did on axis B lives in the
    subspace it used on axis A". Useful for the one-vs-two-trajectory question."""
    Xc = target_matrix - target_matrix.mean(axis=0, keepdims=True)
    total = float(np.sum(Xc.var(axis=0)))
    if total <= 0:
        return np.nan
    proj = Xc @ source.components.T @ source.components
    return float(np.sum(proj.var(axis=0)) / total)


# ---------------------------------------------------------------------------
# sliding windows
# ---------------------------------------------------------------------------
@dataclass
class Window:
    start: int          # frame
    end: int            # frame
    center_sec: float   # center time in seconds


def sliding_windows(T: int, *, win_sec: float = 150.0, step_sec: float = 30.0,
                    sr: int = 30) -> list[Window]:
    """Sliding windows over ``[0, T)`` (defaults: 2.5 min window, 30 s step)."""
    w = int(round(win_sec * sr))
    step = int(round(step_sec * sr))
    out = []
    s = 0
    while s + w <= T:
        out.append(Window(s, s + w, (s + w / 2) / sr))
        s += step
    if not out and T > 0:  # session shorter than one window
        out.append(Window(0, T, (T / 2) / sr))
    return out


def reward_rate_in_windows(reward_frames: Sequence[float], windows: Sequence[Window],
                           *, sr: int = 30) -> np.ndarray:
    """Active rewards per minute within each window."""
    rf = np.asarray(reward_frames, dtype=float).ravel()
    out = np.zeros(len(windows))
    for i, w in enumerate(windows):
        count = np.sum((rf >= w.start) & (rf < w.end))
        out[i] = count / ((w.end - w.start) / sr / 60.0)
    return out


def smooth_series(y: np.ndarray, window: int = 5) -> np.ndarray:
    """Moving-average smooth of a 1-D series, ignoring NaNs (edge-safe)."""
    y = np.asarray(y, dtype=float)
    if window <= 1:
        return y
    kernel = np.ones(window)
    valid = np.isfinite(y).astype(float)
    yf = np.where(np.isfinite(y), y, 0.0)
    num = np.convolve(yf, kernel, mode="same")
    den = np.convolve(valid, kernel, mode="same")
    out = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
    return out
