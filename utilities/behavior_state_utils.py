"""
behavior_state_utils.py
=======================

Reusable *tools* for segmenting the BCI cursor time series into discrete behavioural
states ("actively controlling", "trying slowly", "resting / given up"), and for relating
those states to the other behavioural signals recorded alongside (treadmill locomotion,
licking, imaging motion).

Same philosophy as the other ``*_utils`` modules: primitives only. No figure code and no
per-analysis orchestration -- the analysis flow and every plot stay visible in the notebook.

Why a sticky AR-HMM
-------------------
The states we are after differ in **how the cursor moves**, not only in where it sits:

* *active control* -- large, fast, structured excursions toward the reward tone,
* *intermediate*   -- slower, smaller, less directed movement,
* *rest / give up* -- the cursor decays to its floor and stays there.

A plain Gaussian HMM whose emissions depend only on the instantaneous value cannot tell
"sitting at mid pitch, climbing hard" from "sitting at mid pitch, drifting", and its
geometric dwell-time prior makes it flicker between states many times a second. An
**autoregressive HMM** gives every state its own linear dynamics

    y_t = W_k @ [y_{t-1}, ..., y_{t-L}, 1] + e_t,     e_t ~ N(0, S_k)

so a state is defined by its *dynamics* (AR coefficients), its *set point* (the bias
column) and its *vigour* (the innovation covariance ``S_k``). That is the same modelling
choice MoSeq makes for pose dynamics.

A **sticky** prior (extra Dirichlet mass on the self-transition, see
:func:`make_transition_prior`) is what turns this into bouts of seconds-to-minutes rather
than fragmented sub-second motifs.

What is deliberately *not* borrowed from MoSeq: the HDP / non-parametric state count and
the syllable-grammar machinery. Those target sub-second motifs in high-dimensional pose
data; here the observable is a 1-D control signal and the goal is a handful of
interpretable engagement states, so the state count is fixed and chosen by held-out
likelihood (:func:`cv_select_states`).

No HMM package is installed in this environment (``ssm``, ``hmmlearn``, ``dynamax`` and
``jax`` are all absent, and ``ssm`` needs compilation), so the model is implemented here
directly. It is validated against synthetic data with known ground truth in the notebook.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular

try:  # numba only makes the message-passing loops fast; the code runs without it
    from numba import njit as _njit

    def njit(f):
        return _njit(cache=True)(f)

    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    def njit(f):
        return f

    HAVE_NUMBA = False


# ===========================================================================
# 1 - behavioural covariates recorded alongside the cursor
# ===========================================================================
#: Analog channels in ``r["data_ws"][day]``, all sampled one row per imaging frame (30 Hz).
WS_CHANNELS = ["sync_pulse", "laser_pickoff", "treadmill", "lickometer",
               "rewards", "cursor", "y_galvo"]

#: Column order of the ``motionCorrection`` logger when it is stored as a bare array.
LOGGER_MC_COLS = ["xShift", "yShift", "MC_correlation", "current_position_z", "deltaMoved",
                  "z_correlation_1", "z_correlation_2", "z_correlation_3",
                  "z_correlation_4", "z_correlation_5"]

#: Treadmill -> speed transforms offered by :func:`load_treadmill_speed`.
TREADMILL_MODES = ["abs_dev", "rolling_sd", "abs_diff"]


def _fill_nan(x: np.ndarray) -> np.ndarray:
    """Linear-interpolate NaNs (edges held constant); an all-NaN input becomes zeros."""
    x = np.asarray(x, dtype=float).copy()
    good = np.isfinite(x)
    if good.all():
        return x
    if not good.any():
        return np.zeros_like(x)
    idx = np.arange(len(x))
    return np.interp(idx, idx[good], x[good])


def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    """Centred boxcar mean, NaN-safe, with shrinking windows at the edges."""
    x = np.asarray(x, dtype=float)
    if win <= 1:
        return x.copy()
    good = np.isfinite(x).astype(float)
    filled = np.where(np.isfinite(x), x, 0.0)
    k = np.ones(int(win))
    num = np.convolve(filled, k, mode="same")
    den = np.convolve(good, k, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def load_ws_channel(r, day: str, channel: str) -> np.ndarray:
    """One raw WaveSurfer channel for a day (30 Hz, one sample per imaging frame)."""
    ws = r["data_ws"][day].load()
    if channel not in ws:
        raise KeyError(f"{channel!r} not in data_ws; have {list(ws.keys())}")
    return np.asarray(ws[channel], dtype=float).ravel()


def load_treadmill_speed(r, day: str, *, mode: str = "abs_dev", smooth_sec: float = 0.5,
                         sr: float = 30.0, normalize: bool = True) -> np.ndarray:
    """Locomotion speed proxy from the analog treadmill channel.

    The channel is a rotary-encoder voltage that rests at a fixed value (~3825 in every
    animal measured) and deviates when the wheel turns, so speed is the **deviation from
    that resting level**, smoothed. ``mode`` selects the transform:

    * ``"abs_dev"``    -- ``|raw - median|`` (default): the most direct reading of speed.
    * ``"rolling_sd"`` -- local standard deviation over ``smooth_sec``.
    * ``"abs_diff"``   -- ``|diff(raw)|``, i.e. change rather than level.

    All three were validated against an independent movement proxy (frame-to-frame
    motion-correction shifts, :func:`load_mc_motion`) and track it at Spearman rho ~0.65-0.75;
    they are monotonically related to one another, so the choice changes units rather than
    conclusions.

    ``normalize=True`` makes sessions comparable by dividing by the **90th percentile** rather than by the standard deviation or
    the MAD: the signal is near zero most of the time, so both of those are tiny and would
    inflate the result by orders of magnitude. After normalising, 1.0 means "a briskly moving
    wheel" for that session."""
    if mode not in TREADMILL_MODES:
        raise ValueError(f"mode must be one of {TREADMILL_MODES}")
    raw = _fill_nan(load_ws_channel(r, day, "treadmill"))
    win = max(1, int(round(smooth_sec * sr)))
    if mode == "abs_dev":
        v = np.abs(raw - np.median(raw))
    elif mode == "abs_diff":
        v = np.abs(np.diff(raw, prepend=raw[0]))
    else:
        m1 = moving_average(raw, win)
        m2 = moving_average(raw ** 2, win)
        v = np.sqrt(np.maximum(m2 - m1 ** 2, 0.0))
    v = moving_average(v, win)
    if normalize:
        scale = float(np.nanpercentile(v, 90))
        v = v / scale if scale > 1e-9 else v
    return v


def load_lick_rate(r, day: str, *, smooth_sec: float = 1.0, sr: float = 30.0,
                   n_frames: Optional[int] = None) -> np.ndarray:
    """Lick rate in licks/s from the pre-extracted ``lick_times`` (which are **frame indices**)."""
    lt = np.asarray(r["lick_times"][day].load(), dtype=float).ravel()
    if n_frames is None:
        n_frames = len(load_ws_channel(r, day, "cursor"))
    counts = np.zeros(int(n_frames))
    idx = np.round(lt[np.isfinite(lt)]).astype(int)
    idx = idx[(idx >= 0) & (idx < n_frames)]
    np.add.at(counts, idx, 1.0)
    return moving_average(counts, max(1, int(round(smooth_sec * sr)))) * sr


def load_mc_motion(r, day: str, *, smooth_sec: float = 0.5, sr: float = 30.0) -> np.ndarray:
    """Frame-to-frame imaging motion ``|dx| + |dy|`` from the motion-correction logger.

    An *independent* movement proxy: when the animal moves its body the field of view
    shifts. Used to validate the treadmill transform, and as a second movement regressor
    that does not share the treadmill's measurement noise."""
    if "logger_dfs" in r["extras"].keys():
        mc = r["extras"]["logger_dfs"][day]["motionCorrection"].load()
        xs, ys = mc["xShift"].to_numpy(dtype=float), mc["yShift"].to_numpy(dtype=float)
    else:
        arr = np.asarray(r["extras"]["loggers"][day]["motionCorrection"].load(), dtype=float)
        xs, ys = arr[:, 0], arr[:, 1]
    xs, ys = _fill_nan(xs), _fill_nan(ys)
    mag = np.abs(np.diff(xs, prepend=xs[0])) + np.abs(np.diff(ys, prepend=ys[0]))
    return moving_average(mag, max(1, int(round(smooth_sec * sr))))


def bin_signal(x: np.ndarray, bin_size: int, how: str = "mean") -> np.ndarray:
    """Non-overlapping temporal binning (the project's convention; never smoothing).

    Smoothing would inject correlation between neighbouring samples; binning keeps them
    independent. Trailing samples that do not fill a whole bin are dropped."""
    x = np.asarray(x, dtype=float)
    if bin_size <= 1:
        return x.copy()
    n = (len(x) // bin_size) * bin_size
    b = x[:n].reshape(-1, bin_size)
    if how == "mean":
        return np.nanmean(b, axis=1)
    if how == "median":
        return np.nanmedian(b, axis=1)
    if how == "max":
        return np.nanmax(b, axis=1)
    if how == "sum":
        return np.nansum(b, axis=1)
    raise ValueError(f"unknown how={how!r}")


def contiguous_finite_segments(x: np.ndarray, min_len: int = 1) -> list:
    """``[(start, stop), ...]`` half-open spans over which ``x`` is finite everywhere.

    The AR-HMM needs gap-free sequences (an AR model cannot step across a hole), so each
    day is split into its finite runs and each run is fitted as its own sequence."""
    x = np.asarray(x)
    good = np.isfinite(x) if x.ndim == 1 else np.isfinite(x).all(axis=1)
    out, i, n = [], 0, len(good)
    while i < n:
        if good[i]:
            j = i
            while j < n and good[j]:
                j += 1
            if (j - i) >= min_len:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


# ===========================================================================
# 2 - sticky autoregressive HMM
# ===========================================================================
@njit
def _forward(log_pi, log_P, logB):
    T, K = logB.shape
    alpha = np.empty((T, K))
    tmp = np.empty(K)
    for k in range(K):
        alpha[0, k] = log_pi[k] + logB[0, k]
    for t in range(1, T):
        for k in range(K):
            m = -1.0e300
            for j in range(K):
                v = alpha[t - 1, j] + log_P[j, k]
                tmp[j] = v
                if v > m:
                    m = v
            s = 0.0
            for j in range(K):
                s += np.exp(tmp[j] - m)
            alpha[t, k] = m + np.log(s) + logB[t, k]
    return alpha


@njit
def _backward(log_P, logB):
    T, K = logB.shape
    beta = np.zeros((T, K))
    tmp = np.empty(K)
    for t in range(T - 2, -1, -1):
        for j in range(K):
            m = -1.0e300
            for k in range(K):
                v = log_P[j, k] + logB[t + 1, k] + beta[t + 1, k]
                tmp[k] = v
                if v > m:
                    m = v
            s = 0.0
            for k in range(K):
                s += np.exp(tmp[k] - m)
            beta[t, j] = m + np.log(s)
    return beta


@njit
def _xi_sum(alpha, beta, log_P, logB, ll):
    """Time-summed pairwise posteriors: the expected transition-count statistic."""
    T, K = logB.shape
    N = np.zeros((K, K))
    for t in range(T - 1):
        for j in range(K):
            aj = alpha[t, j]
            for k in range(K):
                N[j, k] += np.exp(aj + log_P[j, k] + logB[t + 1, k] + beta[t + 1, k] - ll)
    return N


@njit
def _viterbi(log_pi, log_P, logB):
    T, K = logB.shape
    delta = np.empty((T, K))
    psi = np.zeros((T, K), dtype=np.int64)
    for k in range(K):
        delta[0, k] = log_pi[k] + logB[0, k]
    for t in range(1, T):
        for k in range(K):
            best = -1.0e300
            arg = 0
            for j in range(K):
                v = delta[t - 1, j] + log_P[j, k]
                if v > best:
                    best = v
                    arg = j
            delta[t, k] = best + logB[t, k]
            psi[t, k] = arg
    path = np.zeros(T, dtype=np.int64)
    best = -1.0e300
    arg = 0
    for k in range(K):
        if delta[T - 1, k] > best:
            best = delta[T - 1, k]
            arg = k
    path[T - 1] = arg
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path


def _logsumexp(v: np.ndarray) -> float:
    m = float(np.max(v))
    return m + float(np.log(np.sum(np.exp(v - m))))


def _as_TD(y) -> np.ndarray:
    """Coerce a sequence to ``(T, D)``, tolerating a transposed ``(D, T)`` input."""
    y = np.atleast_2d(np.asarray(y, dtype=float))
    if y.shape[0] < y.shape[1] and y.shape[0] <= 4:
        y = y.T
    return y


def make_design(y: np.ndarray, order: int):
    """Split a sequence into AR regressors and targets.

    Returns ``(U, Y)`` where ``Y = y[order:]`` are the targets and each row of ``U`` is
    ``[y_{t-1}, ..., y_{t-order}, 1]`` -- the lagged values plus a constant, so the bias
    column carries each state's set point."""
    y = _as_TD(y)
    T, D = y.shape
    if T <= order:
        return np.empty((0, D * order + 1)), np.empty((0, D))
    cols = [y[order - l - 1: T - l - 1] for l in range(order)]
    U = np.concatenate(cols + [np.ones((T - order, 1))], axis=1)
    return U, y[order:]


def make_transition_prior(n_states: int, dwell_target: float, strength: float) -> np.ndarray:
    """Dirichlet pseudo-counts encoding a target mean dwell time (the "sticky" prior).

    ``dwell_target`` is in **samples of the fitting time base**; ``strength`` is the total
    pseudo-count mass per row, which only bites when it is a non-trivial fraction of the
    number of samples. A geometric dwell of ``d`` samples corresponds to a self-transition
    probability of ``1 - 1/d``."""
    p_self = 1.0 - 1.0 / max(float(dwell_target), 1.0 + 1e-9)
    off = (1.0 - p_self) / max(n_states - 1, 1)
    prior = np.full((n_states, n_states), off) * strength
    np.fill_diagonal(prior, p_self * strength)
    return prior


@dataclass
class ARHMMFit:
    """Fitted parameters and diagnostics of a :class:`StickyARHMM`."""
    log_pi: np.ndarray
    log_P: np.ndarray
    W: np.ndarray            # (K, D, D*order + 1) -- AR coefficients plus a bias column
    Sigma: np.ndarray        # (K, D, D)           -- innovation covariance
    loglik: float
    ll_trace: list = field(default_factory=list)
    n_iter: int = 0
    converged: bool = False


class StickyARHMM:
    """Sticky autoregressive HMM fitted by EM (Baum-Welch).

    Parameters
    ----------
    n_states : number of discrete states K.
    ar_order : number of autoregressive lags L (L=2 lets a state express both a decay rate
        and an oscillatory / overshooting tendency). **L=0 reduces the model to a plain
        sticky Gaussian HMM** (each state is then just a mean and a covariance), which is the
        right choice when the observations are already smooth summary features rather than a
        raw fast time series.
    dwell_target_samples : target mean dwell of the sticky prior, in samples of the fitting
        time base. Match it to the timescale of the behaviour you are segmenting.
    prior_strength : weight of that prior, **always as a fraction of the total number of
        samples** (0 = no stickiness, 0.5 = prior worth half the data). It is never read as
        absolute pseudo-counts, so raising it always increases stickiness.
    ridge : ridge added to the AR normal equations and to the innovation covariance, to keep
        both well conditioned.
    """

    def __init__(self, n_states: int = 3, ar_order: int = 2, *, dwell_target_samples: float = 60.0,
                 prior_strength: float = 0.02, ridge: float = 1e-6, seed: int = 0):
        self.K = int(n_states)
        self.L = int(ar_order)
        self.dwell_target = float(dwell_target_samples)
        self.prior_strength = float(prior_strength)
        self.ridge = float(ridge)
        self.seed = int(seed)
        self.fit_: Optional[ARHMMFit] = None

    # ---------------- internals ----------------
    def _log_emissions(self, U, Y, W, Sigma):
        T, D = Y.shape
        logB = np.empty((T, self.K))
        for k in range(self.K):
            R = Y - U @ W[k].T
            S = Sigma[k]
            try:
                cf = np.linalg.cholesky(S)
            except np.linalg.LinAlgError:
                cf = np.linalg.cholesky(S + np.eye(D) * (self.ridge * 10 + 1e-9))
            sol = solve_triangular(cf, R.T, lower=True)
            quad = np.sum(sol ** 2, axis=0)
            logdet = 2.0 * float(np.sum(np.log(np.diag(cf))))
            logB[:, k] = -0.5 * (D * np.log(2 * np.pi) + logdet + quad)
        return logB

    def _mstep(self, seqs, gammas):
        D = seqs[0][1].shape[1]
        P = seqs[0][0].shape[1]
        Sxx = np.zeros((self.K, P, P))
        Sxy = np.zeros((self.K, P, D))
        Syy = np.zeros((self.K, D, D))
        Nk = np.zeros(self.K)
        for (U, Y), g in zip(seqs, gammas):
            for k in range(self.K):
                w = g[:, k]
                Uw = U * w[:, None]
                Sxx[k] += Uw.T @ U
                Sxy[k] += Uw.T @ Y
                Syy[k] += (Y * w[:, None]).T @ Y
                Nk[k] += float(w.sum())
        W = np.zeros((self.K, D, P))
        Sigma = np.zeros((self.K, D, D))
        for k in range(self.K):
            A = Sxx[k] + self.ridge * np.eye(P) * max(Nk[k], 1.0)
            W[k] = np.linalg.solve(A, Sxy[k]).T
            # residual second moment: Syy - W @ Sxy  (exact at the normal-equation optimum)
            resid = Syy[k] - W[k] @ Sxy[k]
            S = resid / max(Nk[k], 1e-9)
            S = 0.5 * (S + S.T) + np.eye(D) * self.ridge
            ev = np.linalg.eigvalsh(S)
            if ev.min() <= 0:
                S = S + np.eye(D) * (abs(float(ev.min())) + self.ridge)
            Sigma[k] = S
        return W, Sigma, Nk

    def _init_responsibilities(self, seqs, rng, restart):
        Yall = np.concatenate([Y for _, Y in seqs], axis=0)
        Uall = np.concatenate([U for U, _ in seqs], axis=0)
        D = Yall.shape[1]
        if restart == 0:
            from sklearn.cluster import KMeans
            jump = np.abs(Yall - Uall[:, :D])            # local vigour
            F = np.hstack([Yall, jump])
            F = (F - F.mean(0)) / (F.std(0) + 1e-9)
            lab = KMeans(self.K, n_init=4, random_state=self.seed).fit_predict(F)
            g = np.full((len(Yall), self.K), 1e-3)
            g[np.arange(len(Yall)), lab] = 1.0
        else:
            g = rng.dirichlet(np.ones(self.K) * 1.5, size=len(Yall))
        g = g / g.sum(1, keepdims=True)
        out, i = [], 0
        for _, Y in seqs:
            out.append(g[i:i + len(Y)])
            i += len(Y)
        return out

    # ---------------- public API ----------------
    def fit(self, sequences: Sequence[np.ndarray], *, n_iter: int = 100, tol: float = 1e-5,
            n_restarts: int = 2, verbose: bool = False) -> "StickyARHMM":
        """Fit by EM on a list of gap-free sequences, each ``(T_i, D)``."""
        seqs = [make_design(y, self.L) for y in sequences]
        seqs = [(U, Y) for U, Y in seqs if len(Y) > self.L + 2]
        if not seqs:
            raise ValueError("no sequence is long enough for the requested ar_order")
        n_tot = sum(len(Y) for _, Y in seqs)
        strength = self.prior_strength * n_tot        # always a fraction of the samples
        prior = make_transition_prior(self.K, self.dwell_target, strength)

        best = None
        for restart in range(max(1, n_restarts)):
            rng = np.random.default_rng(self.seed + 1000 * restart)
            gammas = self._init_responsibilities(seqs, rng, restart)
            W, Sigma, _ = self._mstep(seqs, gammas)
            log_pi = np.log(np.full(self.K, 1.0 / self.K))
            log_P = np.log(prior / prior.sum(1, keepdims=True))
            ll_prev, trace, converged = -np.inf, [], False
            for it in range(n_iter):
                total_ll = 0.0
                gammas = []
                Nsum = np.zeros((self.K, self.K))
                pi_acc = np.zeros(self.K)
                for U, Y in seqs:
                    logB = self._log_emissions(U, Y, W, Sigma)
                    alpha = _forward(log_pi, log_P, logB)
                    beta = _backward(log_P, logB)
                    ll = _logsumexp(alpha[-1])
                    total_ll += ll
                    g = np.exp(alpha + beta - ll)
                    g = g / np.maximum(g.sum(1, keepdims=True), 1e-300)
                    gammas.append(g)
                    Nsum += _xi_sum(alpha, beta, log_P, logB, ll)
                    pi_acc += g[0]
                trace.append(total_ll)
                W, Sigma, Nk = self._mstep(seqs, gammas)
                Pn = Nsum + prior
                log_P = np.log(Pn / Pn.sum(1, keepdims=True))
                pi_acc = pi_acc + 1e-3
                log_pi = np.log(pi_acc / pi_acc.sum())
                if it > 0 and abs(total_ll - ll_prev) < tol * max(1.0, abs(ll_prev)):
                    converged = True
                    ll_prev = total_ll
                    break
                ll_prev = total_ll
            if verbose:
                print(f"    restart {restart}: ll={ll_prev:.1f} iters={len(trace)} converged={converged}")
            if best is None or ll_prev > best.loglik:
                best = ARHMMFit(log_pi, log_P, W, Sigma, ll_prev, trace, len(trace), converged)
        self.fit_ = best
        return self

    def loglik(self, sequences: Sequence[np.ndarray]) -> float:
        """Total log-likelihood of held-out sequences under the fitted model."""
        f = self._require_fit()
        tot = 0.0
        for y in sequences:
            U, Y = make_design(y, self.L)
            if len(Y) == 0:
                continue
            logB = self._log_emissions(U, Y, f.W, f.Sigma)
            tot += _logsumexp(_forward(f.log_pi, f.log_P, logB)[-1])
        return tot

    def loglik_per_sample(self, sequences: Sequence[np.ndarray]) -> float:
        n = sum(max(len(_as_TD(y)) - self.L, 0) for y in sequences)
        return self.loglik(sequences) / max(n, 1)

    def decode(self, y: np.ndarray, pad: bool = True) -> np.ndarray:
        """Most likely state path (Viterbi).

        With ``pad=True`` the result has the same length as ``y``; the first ``ar_order``
        samples have no AR history, so they are given the first decoded label."""
        f = self._require_fit()
        U, Y = make_design(y, self.L)
        if len(Y) == 0:
            return np.full(len(_as_TD(y)), -1, dtype=int)
        logB = self._log_emissions(U, Y, f.W, f.Sigma)
        path = _viterbi(f.log_pi, f.log_P, logB)
        return np.concatenate([np.full(self.L, path[0]), path]) if pad else path

    def posterior(self, y: np.ndarray) -> np.ndarray:
        """Per-sample state posterior (rows sum to 1), aligned like :meth:`decode`."""
        f = self._require_fit()
        U, Y = make_design(y, self.L)
        logB = self._log_emissions(U, Y, f.W, f.Sigma)
        alpha, beta = _forward(f.log_pi, f.log_P, logB), _backward(f.log_P, logB)
        ll = _logsumexp(alpha[-1])
        g = np.exp(alpha + beta - ll)
        g = g / np.maximum(g.sum(1, keepdims=True), 1e-300)
        return np.vstack([np.repeat(g[:1], self.L, axis=0), g])

    def n_parameters(self) -> int:
        f = self._require_fit()
        K, D, P = self.K, f.W.shape[1], f.W.shape[2]
        return K * D * P + K * D * (D + 1) // 2 + K * (K - 1) + (K - 1)

    def transition_matrix(self) -> np.ndarray:
        return np.exp(self._require_fit().log_P)

    def expected_dwell(self) -> np.ndarray:
        """Mean dwell implied by the transition matrix, in samples: ``1 / (1 - P_kk)``."""
        p = np.diag(self.transition_matrix())
        return 1.0 / np.maximum(1.0 - p, 1e-12)

    def _require_fit(self) -> ARHMMFit:
        if self.fit_ is None:
            raise RuntimeError("model has not been fitted")
        return self.fit_

    # ---------------- simulation, for validating the implementation ----------------
    def simulate(self, T: int, W: np.ndarray, Sigma: np.ndarray, P: np.ndarray, seed: int = 0):
        """Draw ``(y, z)`` from a known AR-HMM -- used to test recovery against ground truth."""
        rng = np.random.default_rng(seed)
        K, D, _ = W.shape
        y = np.zeros((T, D))
        z = np.zeros(T, dtype=int)
        chol = [np.linalg.cholesky(Sigma[k]) for k in range(K)]
        z[0] = rng.integers(K)
        for t in range(1, T):
            z[t] = rng.choice(K, p=P[z[t - 1]])
            if t < self.L:
                y[t] = rng.normal(size=D)
                continue
            u = np.concatenate([y[t - l - 1] for l in range(self.L)] + [np.ones(1)])
            y[t] = W[z[t]] @ u + chol[z[t]] @ rng.normal(size=D)
        return y, z


# ===========================================================================
# 3 - model selection, ordering and characterisation
# ===========================================================================
def cv_select_states(sequences: Sequence[np.ndarray], k_values: Sequence[int], *,
                     ar_order: int = 2, n_folds: int = 4, dwell_target_samples: float = 60.0,
                     prior_strength: float = 0.02, n_restarts: int = 1, n_iter: int = 60,
                     seed: int = 0, verbose: bool = False) -> pd.DataFrame:
    """Held-out log-likelihood per sample for each K, holding out **whole sequences**.

    Sequences (days) are the unit of held-out data, so the score measures generalisation to
    a new session rather than to interpolated time points inside a session already fitted."""
    seqs = list(sequences)
    n = len(seqs)
    folds = [np.arange(i, n, n_folds) for i in range(min(n_folds, n))]
    rows = []
    for K in k_values:
        tr_ll, te_ll = [], []
        for f_i, te_idx in enumerate(folds):
            te_set = set(te_idx.tolist())
            te = [seqs[i] for i in te_idx]
            tr = [seqs[i] for i in range(n) if i not in te_set]
            if not tr or not te:
                continue
            m = StickyARHMM(K, ar_order, dwell_target_samples=dwell_target_samples,
                            prior_strength=prior_strength, seed=seed + f_i)
            m.fit(tr, n_iter=n_iter, n_restarts=n_restarts)
            tr_ll.append(m.loglik_per_sample(tr))
            te_ll.append(m.loglik_per_sample(te))
        if te_ll:
            rows.append(dict(K=K, train_ll=float(np.mean(tr_ll)), test_ll=float(np.mean(te_ll)),
                             test_ll_sem=float(np.std(te_ll) / max(np.sqrt(len(te_ll)), 1)),
                             n_folds=len(te_ll)))
            if verbose:
                print(f"  K={K}: train {rows[-1]['train_ll']:.4f}  test {rows[-1]['test_ll']:.4f}")
    return pd.DataFrame(rows)


def order_states_by(model: StickyARHMM, sequences: Sequence[np.ndarray],
                    labels: Sequence[np.ndarray], key: str = "speed") -> np.ndarray:
    """Permutation putting states in a canonical order, so names are stable across refits.

    ``key="speed"`` orders by the mean absolute sample-to-sample change of the observation
    within each state (quietest first), matching the intuition "state 0 = rest ...
    state K-1 = most active". ``key="level"`` orders by mean value, ``key="innovation"`` by
    the fitted innovation standard deviation."""
    K = model.K
    if key == "innovation":
        f = model._require_fit()
        score = np.array([np.sqrt(np.trace(f.Sigma[k])) for k in range(K)])
    else:
        num, den = np.zeros(K), np.zeros(K)
        for y, lab in zip(sequences, labels):
            y2 = _as_TD(y)
            v = (np.abs(np.diff(y2, axis=0, prepend=y2[:1])).sum(1) if key == "speed"
                 else y2.sum(1))
            lab = np.asarray(lab)
            for k in range(K):
                m = lab[:len(v)] == k
                num[k] += float(np.nansum(v[:len(m)][m]))
                den[k] += float(m.sum())
        score = num / np.maximum(den, 1)
    return np.argsort(score)


def apply_state_order(labels: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Relabel a state path with the permutation returned by :func:`order_states_by`."""
    inv = np.empty(len(order), dtype=int)
    inv[np.asarray(order)] = np.arange(len(order))
    out = np.asarray(labels).copy()
    ok = out >= 0
    out[ok] = inv[out[ok]]
    return out


def dwell_segments(labels: np.ndarray) -> pd.DataFrame:
    """Contiguous runs of a state path, as ``(state, start, stop, length)``."""
    lab = np.asarray(labels)
    if len(lab) == 0:
        return pd.DataFrame(columns=["state", "start", "stop", "length"])
    change = np.flatnonzero(np.diff(lab) != 0) + 1
    starts = np.concatenate([[0], change])
    stops = np.concatenate([change, [len(lab)]])
    return pd.DataFrame(dict(state=lab[starts], start=starts, stop=stops, length=stops - starts))


def state_occupancy(labels: np.ndarray, n_states: int) -> np.ndarray:
    """Fraction of time spent in each state (ignores ``-1`` padding)."""
    lab = np.asarray(labels)
    lab = lab[lab >= 0]
    if len(lab) == 0:
        return np.full(n_states, np.nan)
    return np.bincount(lab, minlength=n_states)[:n_states] / len(lab)


def match_states(true_labels: np.ndarray, pred_labels: np.ndarray, n_states: int):
    """Best label matching between two state paths (Hungarian), and the resulting accuracy.

    Used to score recovery against synthetic ground truth, where state *identity* is
    arbitrary but the partition is what matters."""
    from scipy.optimize import linear_sum_assignment
    C = np.zeros((n_states, n_states))
    for a in range(n_states):
        for b in range(n_states):
            C[a, b] = -np.sum((true_labels == a) & (pred_labels == b))
    r, c = linear_sum_assignment(C)
    mapping = np.zeros(n_states, dtype=int)
    mapping[c] = r
    return mapping, float(np.mean(true_labels == mapping[pred_labels]))


def state_covariate_table(labels: np.ndarray, covariates: dict, n_states: int) -> pd.DataFrame:
    """Mean of each covariate within each state -- the core state-characterisation table."""
    lab = np.asarray(labels)
    rows = []
    for k in range(n_states):
        m = lab == k
        rec = dict(state=k, n_samples=int(m.sum()), occupancy=float(m.mean()))
        for name, v in covariates.items():
            v = np.asarray(v, dtype=float)
            n = min(len(v), len(m))
            mm, vv = m[:n], v[:n]
            rec[name] = float(np.nanmean(vv[mm])) if mm.any() else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)
