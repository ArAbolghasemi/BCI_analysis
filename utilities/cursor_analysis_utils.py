"""
cursor_analysis_utils.py
========================

Reusable *tools* for analysing the shape of the BCI cursor on single trials.

Design philosophy (same as ``neural_analysis_utils.py``): this module holds only
reusable, well-scoped primitives -- signal access, trial-window extraction,
time-warping, shape metrics, and the clustering/embedding building blocks. It
contains **no figure code and no per-analysis orchestration**; the analysis flow
and all plotting live in the notebook so they stay visible and editable.

Cursor signals (verified on all four mice)
------------------------------------------
The decoder logger holds several correlated-but-different cursor signals, at
**one row per imaging frame (30 Hz)**, with ``len(decoder_df) == n_neural_frames``:

* ``freqToOutput``   -- the commanded auditory tone in **Hz** (~1000-15108, i.e.
  3.9 octaves). This is the closest available proxy for what the animal actually
  hears, so it is the default here.
* ``cursor_output``  -- the decoder scalar after clipping/flooring; drives the tone.
* ``cursor_brain`` / ``cursor_brain_raw`` -- the raw decoder scalar (these two are
  effectively identical).
* ``voltage_cursorCurrentPos``, ``decoder_angles``, ``avgVec_angle`` -- other logged values.

``r["data_ws"][day]["cursor"]`` is a *different* signal: the analog WaveSurfer
readback of the audio-control line. It lags ``freqToOutput`` by ~1 frame (33 ms),
is ~5x more coarsely quantised and carries measurement noise, so it is offered
here as a selectable source but is not the default.

Because pitch perception is approximately logarithmic -- and the task maps the
cursor exponentially onto frequency so equal cursor steps feel like equal pitch
steps -- ``transform="log2"`` (octaves) is the recommended representation for
shape analysis. See :func:`load_cursor_signal`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# decoder logger access (handles both storage formats)
# ---------------------------------------------------------------------------
#: column order when the decoder logger is stored as a bare array (no ``logger_dfs``)
LOGGER_DECODER_COLS = [
    "cursor_brain", "cursor_brain_raw", "cursor_output", "freqToOutput",
    "voltage_cursorCurrentPos", "decoder_angles", "avgVec_angle",
]

#: signals that can be used as "the cursor" (``data_ws_cursor`` comes from WaveSurfer)
CURSOR_SOURCES = [
    "freqToOutput", "cursor_output", "cursor_brain", "cursor_brain_raw",
    "voltage_cursorCurrentPos", "data_ws_cursor",
]


def get_decoder_df(r, day: str) -> pd.DataFrame:
    """Decoder logger for ``day`` as a DataFrame, for either storage format.

    Some animals have ``extras/logger_dfs`` (a saved DataFrame, which carries an
    extra leading ``Unnamed: 0`` index column) and some have ``extras/loggers``
    (a bare array needing :data:`LOGGER_DECODER_COLS`). Always index by column
    **name**, never position, because the two formats differ in width."""
    if "logger_dfs" in r["extras"].keys():
        return r["extras"]["logger_dfs"][day]["decoder"].load()
    arr = r["extras"]["loggers"][day]["decoder"].load()
    return pd.DataFrame(arr, columns=LOGGER_DECODER_COLS)


def get_timeseries_df(r, day: str) -> pd.DataFrame:
    """Task ``timeSeries`` logger for ``day``, for either storage format.

    Same two-format problem as :func:`get_decoder_df`: animals with ``extras/logger_dfs`` have a
    saved DataFrame, the others have a bare array whose column names live in
    ``expParams/<day>/loggerNames/timeSeries``. Always index by name."""
    if "logger_dfs" in r["extras"].keys():
        return r["extras"]["logger_dfs"][day]["timeSeries"].load()
    arr = np.asarray(r["extras"]["loggers"][day]["timeSeries"].load(), dtype=float)
    names = list(r["extras"]["expParams"][day]["loggerNames"]["timeSeries"].load())
    return pd.DataFrame(arr, columns=names)


def load_reward_override(r, day: str) -> np.ndarray:
    """0/1 trace marking frames where the speaker is playing the fixed **reward tone**.

    During reward delivery the task stops rendering the cursor as pitch and clamps the output to a
    constant tone (8777.2 Hz = 13.100 octaves in every animal measured, for ~0.97 s). The flag is
    ``frequencyOverride``, which is frame-for-frame identical to ``CE_rewardDelivery``.

    This matters whenever ``freqToOutput`` is used as a behavioural signal: the tail of every
    correct trial, and essentially the whole post-reward flank, is this constant and carries no
    information about what the animal was doing. The *decoder* signals (``cursor_brain``,
    ``cursor_output``) are **not** overridden and keep reporting through the reward."""
    ts = get_timeseries_df(r, day)
    return np.nan_to_num(pd.to_numeric(ts["frequencyOverride"], errors="coerce").to_numpy(float))


def reward_threshold_by_factor(r, day: str) -> dict:
    """``{decoder factor: reward threshold}`` for one day.

    **Remapping days use a different threshold per decoder.** Measured in ``mouse_0403L``:
    factor 2 (intuitive axis) -> 1.9, factor 3 (new axis) -> 0.9, i.e. the rotated decoder is held
    to roughly half the bar. The quiescence thresholds differ in the same way
    (``cursor_thresh_quiescence_cursorDecoder`` 0.19 vs 0.09).

    This matters because it is easy to mistake the lower threshold for a different *scale*: with
    each factor's own threshold, ~99 % of the task's own threshold-crossing frames clear the bar on
    **both** decoders, so the logged ``cursor_brain`` is on one consistent scale. Compare a trial's
    cursor against its own factor's threshold, never against another factor's."""
    ts = get_timeseries_df(r, day)
    thr = pd.to_numeric(ts["cursor_threshold_reward"], errors="coerce").to_numpy(dtype=float)
    fac = pd.to_numeric(ts["cursor_factor_to_use"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(thr) & np.isfinite(fac)
    out = {}
    for f in np.unique(fac[ok]):
        v = thr[ok & (fac == f)]
        out[float(f)] = float(np.median(v))
    return out


def reward_threshold(r, day: str) -> float:
    """The single reward threshold for a day -- **raises if the day uses more than one**.

    Refuses to guess on purpose: an earlier version of this helper silently returned the first of
    several values, which on a remapping day is the *new-axis* threshold (0.9) rather than the
    intuitive one (1.9), and that quietly turned a threshold difference into an apparent scale
    difference. Use :func:`reward_threshold_by_factor` on days with more than one decoder."""
    by = reward_threshold_by_factor(r, day)
    vals = sorted(set(round(v, 6) for v in by.values()))
    if len(vals) != 1:
        raise ValueError(f"{day} uses {len(vals)} reward thresholds {by} -- "
                         "use reward_threshold_by_factor() instead")
    return float(vals[0])


def interpolate_short_gaps(x: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly fill runs of NaN no longer than ``max_gap`` samples; leave longer runs NaN.

    The logged cursor is ~92-98 % finite: most missing samples are isolated
    single frames (dropouts) but each session also has long NaN stretches at its
    edges, where the decoder was not running. Short gaps are safe to interpolate;
    long ones must stay NaN so the trials overlapping them can be rejected."""
    x = np.asarray(x, dtype=float).copy()
    bad = ~np.isfinite(x)
    if not bad.any() or bad.all():
        return x
    idx = np.arange(len(x))
    good = ~bad
    filled = np.interp(idx, idx[good], x[good])
    # re-open any NaN run longer than max_gap
    i = 0
    while i < len(x):
        if bad[i]:
            j = i
            while j < len(x) and bad[j]:
                j += 1
            if (j - i) > max_gap:
                filled[i:j] = np.nan
            i = j
        else:
            i += 1
    # never extrapolate past the first/last finite sample
    first, last = idx[good][0], idx[good][-1]
    filled[:first] = np.nan
    filled[last + 1:] = np.nan
    return filled


def load_cursor_signal(r, day: str, *, column: str = "freqToOutput",
                       transform: Optional[str] = "log2",
                       max_gap_sec: float = 0.5, sr: float = 30.0) -> np.ndarray:
    """One day's cursor signal at 30 Hz, gap-filled and optionally transformed.

    ``column`` is any of :data:`CURSOR_SOURCES`. ``transform`` may be
    ``None`` (raw units) or ``"log2"`` (octaves -- appropriate for
    ``freqToOutput``/``cursor_output`` because pitch is perceived logarithmically;
    non-positive samples become NaN)."""
    if column == "data_ws_cursor":
        raw = np.asarray(r["data_ws"][day].load()["cursor"], dtype=float).ravel()
    else:
        df = get_decoder_df(r, day)
        if column not in df.columns:
            raise KeyError(f"{column!r} not in decoder logger; have {list(df.columns)}")
        raw = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    x = interpolate_short_gaps(raw, int(round(max_gap_sec * sr)))
    if transform == "log2":
        with np.errstate(invalid="ignore", divide="ignore"):
            x = np.where(x > 0, np.log2(np.where(x > 0, x, np.nan)), np.nan)
    elif transform not in (None, "none"):
        raise ValueError(f"unknown transform {transform!r}")
    return x


# ---------------------------------------------------------------------------
# trial windows: [start - pre, effective_end + post], trial portion time-warped
# ---------------------------------------------------------------------------
@dataclass
class WindowSpec:
    """Geometry of a single-trial cursor window.

    The window is ``pre_sec`` before trial start, the trial itself, then
    ``post_sec`` after the trial's *effective end* (reward time for a correct
    trial, ``start + 20 s`` timeout for an incorrect one).

    Only the **trial portion** is time-warped, to ``n_trial_bins``; the two
    flanks keep their real-time sampling. This keeps trials of very different
    duration comparable column-by-column while leaving the fixed-duration
    context around them undistorted."""
    pre_sec: float = 2.0
    post_sec: float = 2.0
    n_trial_bins: int = 60
    sr: float = 30.0

    @property
    def pre_n(self) -> int:
        return int(round(self.pre_sec * self.sr))

    @property
    def post_n(self) -> int:
        return int(round(self.post_sec * self.sr))

    @property
    def n_cols(self) -> int:
        return self.pre_n + self.n_trial_bins + self.post_n

    @property
    def trial_slice(self) -> slice:
        """Columns of the warped window holding the trial itself."""
        return slice(self.pre_n, self.pre_n + self.n_trial_bins)

    def axis(self) -> np.ndarray:
        """Plot axis: seconds for the flanks, 0..1 normalised inside the trial.

        Returns a monotone axis where the trial occupies [0, 1], the pre-flank is
        negative (in units of the trial) and the post-flank is > 1. Use
        :meth:`tick_labels` for a human-readable version."""
        pre = np.linspace(-self.pre_n / self.n_trial_bins, 0, self.pre_n, endpoint=False)
        mid = np.linspace(0, 1, self.n_trial_bins, endpoint=False)
        post = np.linspace(1, 1 + self.post_n / self.n_trial_bins, self.post_n, endpoint=False)
        return np.concatenate([pre, mid, post])


def resample_1d(y: np.ndarray, n: int) -> np.ndarray:
    """Linear resample a 1-D trace to ``n`` points over normalised time."""
    y = np.asarray(y, dtype=float)
    if len(y) == n:
        return y.copy()
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(y)), y)


def extract_trial_windows(cursor: np.ndarray, starts: Sequence[float], ends: Sequence[float],
                          spec: WindowSpec, *, max_nan_frac: float = 0.10
                          ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build the time-warped trial x normalised-time cursor matrix.

    Returns ``(M, kept, info)``:

    * ``M``    -- ``(n_kept, spec.n_cols)`` warped windows,
    * ``kept`` -- integer indices (into ``starts``) of the trials that survived,
    * ``info`` -- per-kept-trial book-keeping (duration, NaN fraction, reject reason).

    A trial is rejected when its window runs off either end of the recording, when
    the trial portion is shorter than 2 samples, or when more than ``max_nan_frac``
    of the window is NaN after short-gap interpolation."""
    cursor = np.asarray(cursor, dtype=float)
    T = len(cursor)
    rows, kept, recs = [], [], []
    for i, (s, e) in enumerate(zip(starts, ends)):
        rec = dict(trial=i, duration_sec=np.nan, nan_frac=np.nan, reject="")
        if not (np.isfinite(s) and np.isfinite(e) and e > s):
            rec["reject"] = "bad_times"; recs.append(rec); continue
        s_i, e_i = int(round(s)), int(round(e))
        lo, hi = s_i - spec.pre_n, e_i + spec.post_n
        rec["duration_sec"] = (e_i - s_i) / spec.sr
        if lo < 0 or hi > T:
            rec["reject"] = "window_off_recording"; recs.append(rec); continue
        mid = cursor[s_i:e_i]
        if len(mid) < 2:
            rec["reject"] = "trial_too_short"; recs.append(rec); continue
        win = cursor[lo:hi]
        nan_frac = float(np.mean(~np.isfinite(win)))
        rec["nan_frac"] = nan_frac
        if nan_frac > max_nan_frac:
            rec["reject"] = "too_many_nan"; recs.append(rec); continue
        pre = cursor[lo:s_i]
        post = cursor[e_i:hi]
        # interpolate any residual NaN inside each piece so warping is well defined
        row = np.concatenate([_fill_all(pre), resample_1d(_fill_all(mid), spec.n_trial_bins), _fill_all(post)])
        if not np.isfinite(row).all():
            rec["reject"] = "nan_after_fill"; recs.append(rec); continue
        rows.append(row); kept.append(i); recs.append(rec)
    M = np.array(rows) if rows else np.empty((0, spec.n_cols))
    return M, np.array(kept, dtype=int), pd.DataFrame(recs)


def _fill_all(x: np.ndarray) -> np.ndarray:
    """Interpolate every NaN in a short piece (edges held constant)."""
    x = np.asarray(x, dtype=float)
    good = np.isfinite(x)
    if good.all() or not good.any():
        return x
    idx = np.arange(len(x))
    return np.interp(idx, idx[good], x[good])


# ---------------------------------------------------------------------------
# shape metrics
# ---------------------------------------------------------------------------
def planar_curvature(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Curvature of the planar curve ``(t, y)``: ``|y''| / (1 + y'^2)^{3/2}``.

    Curvature is **not** scale invariant, so callers should pass a standardised
    shape (see :func:`shape_metrics`, which scales time to [0, 1] and z-scores the
    cursor) -- otherwise the value depends on the units of the axes."""
    t = np.asarray(t, float); y = np.asarray(y, float)
    if len(y) < 3:
        return np.array([np.nan])
    d1 = np.gradient(y, t)
    d2 = np.gradient(d1, t)
    return np.abs(d2) / np.power(1.0 + d1 ** 2, 1.5)


def count_local_maxima(y: np.ndarray, prominence_frac: float = 0.10) -> int:
    """Number of local maxima with prominence >= ``prominence_frac`` of the range."""
    y = np.asarray(y, float)
    rng = np.nanmax(y) - np.nanmin(y)
    if not np.isfinite(rng) or rng <= 0:
        return 0
    try:
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(y, prominence=prominence_frac * rng)
        return int(len(peaks))
    except Exception:
        return int(np.sum((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:])))


def shape_metrics(row: np.ndarray, spec: WindowSpec, *, duration_sec: float = np.nan) -> dict:
    """Scalar descriptors of one warped trial window.

    Metrics split into three families (all documented in the notebook):

    **Amplitude / level** (in the cursor's own units, e.g. octaves if ``log2``)
      ``pre_level`` (mean of the pre-trial flank), ``peak``, ``trough``, ``range``,
      ``mean_level``, ``end_level`` (value at trial end), ``net_change``
      (end minus pre-trial level), ``post_drop`` (end minus mean of post flank).

    **Excursion / path**
      ``path_length`` -- summed |diff| across the whole warped window;
      ``path_length_trial`` -- the same restricted to the trial portion;
      ``path_length_raw_per_s`` -- trial path length divided by the real trial
      duration, i.e. how fast the cursor actually moved;
      ``tortuosity`` -- trial path length divided by |net displacement| (1 = a
      straight monotone ramp, large = wandering).

    **Shape** (computed after standardising: time -> [0, 1], cursor -> z-score, so
    these are invariant to offset and gain)
      ``mean_curvature`` / ``max_curvature`` (see :func:`planar_curvature`),
      ``roughness`` (mean |second difference|), ``monotonicity`` (Spearman rho of
      cursor vs time within the trial: +1 = a clean ramp up), ``n_peaks``,
      ``time_to_peak`` (normalised position of the maximum within the trial,
      0 = trial start, 1 = trial end)."""
    row = np.asarray(row, float)
    ts = spec.trial_slice
    trial = row[ts]
    pre = row[:spec.pre_n]
    post = row[spec.pre_n + spec.n_trial_bins:]

    out = {}
    out["pre_level"] = float(np.mean(pre)) if len(pre) else np.nan
    out["peak"] = float(np.max(trial))
    out["trough"] = float(np.min(trial))
    out["range"] = out["peak"] - out["trough"]
    out["mean_level"] = float(np.mean(trial))
    out["end_level"] = float(trial[-1])
    out["net_change"] = out["end_level"] - out["pre_level"]
    out["post_drop"] = out["end_level"] - (float(np.mean(post)) if len(post) else np.nan)

    out["path_length"] = float(np.sum(np.abs(np.diff(row))))
    pl_trial = float(np.sum(np.abs(np.diff(trial))))
    out["path_length_trial"] = pl_trial
    out["path_length_raw_per_s"] = pl_trial / duration_sec if duration_sec and np.isfinite(duration_sec) and duration_sec > 0 else np.nan
    net = abs(trial[-1] - trial[0])
    out["tortuosity"] = pl_trial / net if net > 1e-12 else np.nan

    # ---- standardised shape descriptors ----
    sd = np.std(trial)
    z = (trial - np.mean(trial)) / sd if sd > 1e-12 else np.zeros_like(trial)
    tau = np.linspace(0.0, 1.0, len(trial))
    kappa = planar_curvature(tau, z)
    out["mean_curvature"] = float(np.nanmean(kappa))
    out["max_curvature"] = float(np.nanmax(kappa))
    out["roughness"] = float(np.mean(np.abs(np.diff(z, n=2)))) if len(z) > 2 else np.nan
    if len(trial) > 2 and sd > 1e-12:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(tau, trial)
        out["monotonicity"] = float(rho)
    else:
        out["monotonicity"] = np.nan
    out["n_peaks"] = count_local_maxima(trial)
    out["time_to_peak"] = float(np.argmax(trial) / max(len(trial) - 1, 1))
    return out


SHAPE_METRIC_NAMES = [
    "pre_level", "peak", "trough", "range", "mean_level", "end_level", "net_change", "post_drop",
    "path_length", "path_length_trial", "path_length_raw_per_s", "tortuosity",
    "mean_curvature", "max_curvature", "roughness", "monotonicity", "n_peaks", "time_to_peak",
]

#: (family, plain-language meaning, how it is computed, units) for every shape metric.
#: ``U`` = the cursor's own unit (octaves when ``transform="log2"``, otherwise raw Hz).
METRIC_DESCRIPTIONS = {
    "pre_level":   ("level", "Cursor level in the 2 s before the trial starts, i.e. the baseline pitch the animal is sitting at.",
                    "mean of the pre-trial flank (first `pre_n` columns).", "U"),
    "peak":        ("level", "Highest cursor value reached during the trial.",
                    "max over the warped trial columns.", "U"),
    "trough":      ("level", "Lowest cursor value during the trial.",
                    "min over the warped trial columns.", "U"),
    "range":       ("level", "Total spread covered within the trial (modulation depth).",
                    "peak - trough.", "U"),
    "mean_level":  ("level", "Average cursor level during the trial.",
                    "mean over the warped trial columns.", "U"),
    "end_level":   ("level", "Cursor value at the moment the trial ends (reward for hits, timeout for misses).",
                    "last warped trial column.", "U"),
    "net_change":  ("level", "How much higher the cursor ended than it started, relative to baseline.",
                    "end_level - pre_level.", "U"),
    "post_drop":   ("level", "How far the cursor falls back in the 2 s after the trial ends.",
                    "end_level - mean(post-trial flank).", "U"),
    "path_length": ("path", "Total up-and-down distance travelled across the whole window, including both flanks.",
                    "sum of |differences| between consecutive columns of the full window.", "U"),
    "path_length_trial": ("path", "Total distance travelled during the trial only; large means a lot of movement regardless of direction.",
                    "sum of |differences| across the warped trial columns.", "U"),
    "path_length_raw_per_s": ("path", "How fast the cursor actually moved, in real time. This is the speed information that time-warping removes.",
                    "path_length_trial divided by the trial's real duration in seconds.", "U/s"),
    "tortuosity":  ("path", "How wandering versus direct the trajectory was. 1 = a straight monotone ramp; large = lots of back-and-forth for little net progress.",
                    "path_length_trial / |end - start| within the trial.", "unitless"),
    "mean_curvature": ("shape", "Average bendiness of the trace: how much it keeps changing direction rather than running straight.",
                    "mean of |y''|/(1+y'^2)^{3/2} after standardising (time -> [0,1], cursor -> z-score).", "unitless"),
    "max_curvature":  ("shape", "The single sharpest bend in the trace.",
                    "max of the same standardised curvature.", "unitless"),
    "roughness":   ("shape", "High-frequency jitter: how jagged the trace is from sample to sample.",
                    "mean |2nd difference| of the z-scored trial trace.", "unitless"),
    "monotonicity": ("shape", "How ramp-like the trial is. +1 = the cursor rises steadily throughout, 0 = no consistent direction, -1 = steadily falls.",
                    "Spearman rho between the cursor and time within the trial.", "-1..+1"),
    "n_peaks":     ("shape", "Number of separate upward attempts/bumps in the trial.",
                    "count of local maxima with prominence >= 10 % of the trial's range.", "count"),
    "time_to_peak": ("shape", "When in the trial the maximum occurred. 0 = at trial start, 1 = at trial end.",
                    "argmax position within the trial, normalised to [0, 1].", "0..1"),
}


def metric_description_table(names=None):
    """The metric documentation as a DataFrame (family / meaning / computation / units)."""
    names = names or SHAPE_METRIC_NAMES
    return pd.DataFrame(
        [dict(metric=n, family=METRIC_DESCRIPTIONS[n][0], meaning=METRIC_DESCRIPTIONS[n][1],
              computed_as=METRIC_DESCRIPTIONS[n][2], units=METRIC_DESCRIPTIONS[n][3])
         for n in names if n in METRIC_DESCRIPTIONS]
    ).set_index("metric")


def shape_metrics_table(M: np.ndarray, spec: WindowSpec,
                        durations: Optional[Sequence[float]] = None) -> pd.DataFrame:
    """Apply :func:`shape_metrics` to every row of a warped matrix."""
    if durations is None:
        durations = [np.nan] * len(M)
    return pd.DataFrame([shape_metrics(r, spec, duration_sec=d) for r, d in zip(M, durations)])


# ---------------------------------------------------------------------------
# representation for clustering
# ---------------------------------------------------------------------------
def normalize_rows(M: np.ndarray, mode: str = "none") -> np.ndarray:
    """Per-trial normalisation of the warped matrix before PCA/clustering.

    * ``"none"``     -- leave as-is. Clusters can then differ in overall level and
      gain as well as shape (with ``log2`` data, level differences are pitch shifts).
    * ``"zscore"``   -- z-score each trial: isolates **shape only**, discarding how
      high or how modulated the cursor was.
    * ``"minmax"``   -- scale each trial to [0, 1]: keeps shape and relative timing,
      discards absolute level and gain.
    * ``"center"``   -- subtract each trial's own pre-trial level: removes the
      starting pitch but keeps modulation depth."""
    M = np.asarray(M, float)
    if mode in (None, "none"):
        return M.copy()
    if mode == "zscore":
        mu = M.mean(1, keepdims=True); sd = M.std(1, keepdims=True); sd[sd < 1e-12] = 1.0
        return (M - mu) / sd
    if mode == "minmax":
        lo = M.min(1, keepdims=True); hi = M.max(1, keepdims=True)
        rng = np.where((hi - lo) < 1e-12, 1.0, hi - lo)
        return (M - lo) / rng
    if mode == "center":
        return M - M.mean(1, keepdims=True)
    raise ValueError(f"unknown mode {mode!r}")


@dataclass
class PCAResult:
    scores: np.ndarray          # (n_trials, n_kept_pcs)
    components: np.ndarray      # (n_kept_pcs, n_cols)
    explained_ratio: np.ndarray # full spectrum
    mean: np.ndarray
    n_kept: int
    var_target: float


def pca_reduce(M: np.ndarray, var_target: float = 0.95, max_pcs: Optional[int] = None) -> PCAResult:
    """PCA on the trial x time matrix, keeping enough PCs to reach ``var_target``."""
    from sklearn.decomposition import PCA
    n = min(M.shape) if max_pcs is None else min(max_pcs, *M.shape)
    p = PCA(n_components=n).fit(M)
    ratio = p.explained_variance_ratio_
    k = int(np.searchsorted(np.cumsum(ratio), var_target) + 1)
    k = max(1, min(k, n))
    return PCAResult(p.transform(M)[:, :k], p.components_[:k], ratio, p.mean_, k, var_target)


def hierarchical_labels(scores: np.ndarray, k: int, method: str = "ward", metric: str = "euclidean") -> np.ndarray:
    """Agglomerative cluster labels (1..k) from a linkage on PCA scores."""
    from scipy.cluster.hierarchy import linkage, fcluster
    Z = linkage(scores, method=method, metric=metric)
    return fcluster(Z, t=k, criterion="maxclust")


def linkage_matrix(scores: np.ndarray, method: str = "ward", metric: str = "euclidean"):
    from scipy.cluster.hierarchy import linkage
    return linkage(scores, method=method, metric=metric)


def silhouette_scan(scores: np.ndarray, k_values: Sequence[int], method: str = "ward",
                    sample_size: Optional[int] = 4000, seed: int = 0) -> pd.DataFrame:
    """Mean silhouette for a range of cluster counts (higher = better separated)."""
    from sklearn.metrics import silhouette_score
    rows = []
    for k in k_values:
        if k < 2 or k >= len(scores):
            continue
        lab = hierarchical_labels(scores, k, method=method)
        try:
            s = silhouette_score(scores, lab, sample_size=min(sample_size or len(scores), len(scores)),
                                 random_state=seed)
        except Exception:
            s = np.nan
        rows.append(dict(k=k, silhouette=s, n_clusters=len(np.unique(lab))))
    return pd.DataFrame(rows)


def embed_tsne(scores: np.ndarray, *, perplexity: float = 30.0, seed: int = 0) -> np.ndarray:
    from sklearn.manifold import TSNE
    perp = float(min(perplexity, max(5.0, (len(scores) - 1) / 3.0)))
    return TSNE(n_components=2, perplexity=perp, init="pca", random_state=seed).fit_transform(scores)


def embed_umap(scores: np.ndarray, *, n_neighbors: int = 25, min_dist: float = 0.1, seed: int = 0) -> np.ndarray:
    import umap
    nn = int(min(n_neighbors, max(2, len(scores) - 1)))
    return umap.UMAP(n_components=2, n_neighbors=nn, min_dist=min_dist, random_state=seed).fit_transform(scores)


# ---------------------------------------------------------------------------
# small helpers used by the notebook
# ---------------------------------------------------------------------------
def within_day_tercile(values: Sequence[float], labels=("early", "middle", "late")) -> np.ndarray:
    """Split an ordered per-trial quantity (e.g. trial start time) into 3 equal-count bins."""
    v = np.asarray(values, float)
    out = np.array([labels[1]] * len(v), dtype=object)
    ok = np.isfinite(v)
    if ok.sum() >= 3:
        q = np.quantile(v[ok], [1 / 3, 2 / 3])
        out[ok & (v <= q[0])] = labels[0]
        out[ok & (v > q[0]) & (v <= q[1])] = labels[1]
        out[ok & (v > q[1])] = labels[2]
    return out


def cluster_composition(df: pd.DataFrame, cluster_col: str, group_col: str,
                        normalize: str = "index") -> pd.DataFrame:
    """Cross-tab of cluster membership against any grouping (day, phase, tercile...).

    ``normalize="index"`` gives, for each group, the fraction of its trials in each
    cluster (rows sum to 1)."""
    ct = pd.crosstab(df[group_col], df[cluster_col])
    if normalize == "index":
        ct = ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0)
    elif normalize == "columns":
        ct = ct.div(ct.sum(axis=0).replace(0, np.nan), axis=1)
    return ct
