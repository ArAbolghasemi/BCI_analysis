"""Shared data layer for the wrap-up notebooks.

Everything here is lifted **verbatim** from the validated cells of
``02_learning_and_remapping.ipynb`` -- the decoder chain, the tone law, the session loader and
the other-axis simulation -- so that a second notebook can use them without a second copy of
the maths drifting away from the first. Nothing in this module is new analysis.

Why a module at all: these are the only genuinely reusable primitives (loading, the rig's own
transformations, the fitted readout). Every analysis step and every line of plotting stays
visible in the notebook that owns it.

The rig's chain, verified against the logger:

    M = X . F                 scores on all K factors
    CS_k = M_k / ||M||        a *direction cosine* in K-dim factor-score space
    D_k = |CS_k|^p . M_k      p = 2, so D_k = M_k^3 / ||M||^2
    f   = F0 . 2^(SLOPE . D_k / theta_k)

and the logger stores ``CS`` (``decoder_angles``) and ``D`` (``cursor_brain_raw``), so

    M_live = D / CS^2         and      ||M|| = D / CS^3

``||M||`` is the norm of the whole factor-score vector and is therefore **shared by every
factor** -- that is what makes the other-axis simulation possible.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import lfilter

# ------------------------------------------------------------------ the tone law
# log2 f = log2(F0) + SLOPE * D/theta, R^2 = 1.000000 on 66/66 single-threshold mouse-days.
TONE_F0, TONE_SLOPE = 2962.635, 1.5669
HZ_BASE = TONE_F0                      # the tone at cursor 0
HZ_REWARD = 8777.2                     # the tone at cursor = theta (the reward clamp)
HZ_FLOOR, HZ_CEIL = 1000.0, 15107.6    # where the tone saturates, on every animal


def d_to_hz(d, thr):
    """A decoder value ``D`` as the tone the speaker would play, in Hz."""
    return np.clip(TONE_F0 * 2.0 ** (TONE_SLOPE * np.asarray(d, float) / thr),
                   HZ_FLOOR, HZ_CEIL)


REPS = {
    "hz":    dict(label="cursor (Hz)", scale="log", lim=(950.0, 16500.0),
                  ticks=[1000, 2963, 8777, 15108],
                  ticklabels=["1000", "2963", "8777", "15108"]),
    "oct":   dict(label="cursor (octaves re 1 kHz)", scale="linear", lim=(-0.10, 4.00),
                  ticks=[0.0, 1.5669, 3.1338, 3.9173],
                  ticklabels=["0", "1.57", "3.13", "3.92"]),
    "brain": dict(label="cursor ($D$)", scale="linear", lim=(-2.6, 4.2),
                  ticks=None, ticklabels=None),
    "theta": dict(label=r"cursor ($D/\theta$)", scale="linear", lim=(-1.4, 2.2),
                  ticks=[-1, 0, 1, 2], ticklabels=["-1", "0", "1", "2"]),
}

# ------------------------------------------------------------------ module state
_S = SimpleNamespace(na=None, ca=None, rfs=None, R={}, DAYS={}, TAB={},
                     data_root=None, sr=30.0, bin=5, smooth_bin=2.0, smooth_sec=None,
                     rep=None, sim_cache=None, day0_override={}, exp_log=None)
_SESS: dict = {}
_SIM: dict = {}

# the other-axis simulation, exactly as fitted and validated in notebook 02
SIM_TAU, SIM_RIDGE, SIM_CS_MIN = 4.0, 0.1, 0.05


def init(repo_root, data_root, mice, *, rep="hz", sr=30.0, bin=5, smooth_bin=2.0,
         sim_cache=None, day0_override=None, exp_log=None):
    """Open the packets and fix the conventions. Returns the module's state namespace.

    ``sim_cache`` is where the fitted other-axis readouts live. Point it at notebook 02's
    cache folder to reuse the files it already wrote instead of refitting 15 sessions.
    """
    repo_root, data_root = Path(repo_root), Path(data_root)
    sys.path.insert(0, str(repo_root / "utilities"))
    import neural_analysis_utils as na
    import cursor_analysis_utils as ca
    import richfile as rfs
    na.register_richfile_types(rfs)
    _S.na, _S.ca, _S.rfs = na, ca, rfs
    _S.data_root, _S.sr, _S.bin, _S.smooth_bin = data_root, float(sr), int(bin), float(smooth_bin)
    _S.smooth_sec = _S.smooth_bin * _S.bin / _S.sr
    _S.day0_override = dict(day0_override or {})
    _S.sim_cache = Path(sim_cache) if sim_cache is not None else None
    if _S.sim_cache is not None:
        _S.sim_cache.mkdir(parents=True, exist_ok=True)
    _S.exp_log = exp_log
    _S.R = {m: na.open_mouse(rfs, data_root, m) for m in mice}
    _S.DAYS = {m: na.get_day_list(_S.R[m]) for m in mice}
    set_rep(rep)
    _S.TAB = {m: day_table(m) for m in mice}
    return _S


def set_rep(name):
    """Choose how "the cursor" is drawn everywhere. Clears the per-session cache."""
    if name not in REPS:
        raise ValueError(f"rep must be one of {sorted(REPS)}, got {name!r}")
    _S.rep = dict(REPS[name], name=name)
    return _S.rep


def rep():
    return _S.rep


def state():
    return _S


# ------------------------------------------------------------------ representation helpers
def d_to_rep(d, thr):
    """A decoder value ``D`` (with its own block's threshold) in the active representation."""
    nm = _S.rep["name"]
    if nm == "brain":
        return np.asarray(d, float)
    if nm == "theta":
        return np.asarray(d, float) / np.asarray(thr, float)
    hz = d_to_hz(d, thr)
    return hz if nm == "hz" else np.log2(hz / 1000.0)


def rep_axis(ax, ylim=None, ticks=True, label=None, fontsize=12):
    """Scale, limits, ticks and y label for an axis drawn in the active representation."""
    R_ = _S.rep
    ax.set_yscale(R_["scale"])
    ax.set_ylim(*(R_["lim"] if ylim is None else ylim))
    if ticks and R_["ticks"] is not None:
        ax.set_yticks(R_["ticks"])
        ax.set_yticklabels(R_["ticklabels"])
        ax.minorticks_off()
    if label is not False:
        ax.set_ylabel(R_["label"] if label is None else label, fontsize=fontsize)
    return ax


def rep_frac(f, ylim=None):
    """A height given as a fraction of the cursor axis, honouring a log scale."""
    lo, hi = (_S.rep["lim"] if ylim is None else ylim)
    if _S.rep["scale"] == "log":
        return float(np.exp(np.log(lo) + f * (np.log(hi) - np.log(lo))))
    return float(lo + f * (hi - lo))


def reward_level_value():
    """The reward reference as a single number, where the representation has one."""
    nm = _S.rep["name"]
    if nm == "hz":
        return HZ_REWARD
    if nm == "oct":
        return float(np.log2(HZ_REWARD / 1000.0))
    if nm == "theta":
        return 1.0
    return np.nan            # "brain": the threshold differs per block, so there is no one line


def base_level_value():
    """Where cursor zero sits in the active representation."""
    nm = _S.rep["name"]
    if nm == "hz":
        return HZ_BASE
    if nm == "oct":
        return float(np.log2(HZ_BASE / 1000.0))
    return 0.0


def smooth_nan(x, sigma):
    """Gaussian smooth that neither spreads a NaN nor fills one (normalised convolution)."""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    if sigma <= 0:
        return np.where(ok, x, np.nan)
    num = gaussian_filter1d(np.where(ok, x, 0.0), sigma, mode="nearest")
    den = gaussian_filter1d(ok.astype(float), sigma, mode="nearest")
    out = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 1e-6)
    out[~ok] = np.nan
    return out


# ------------------------------------------------------------------ the session loader
def session(mouse, day_i):
    """The per-frame behaviour of one session: the cursor, its trials, the block structure."""
    key = (mouse, int(day_i), _S.rep["name"])
    if key in _SESS:
        return _SESS[key]
    na, ca = _S.na, _S.ca
    r, day = _S.R[mouse], _S.DAYS[mouse][int(day_i)]
    dec, ts = ca.get_decoder_df(r, day), ca.get_timeseries_df(r, day)
    nf = min(len(dec), len(ts))                      # a few sessions have a truncated logger
    f = np.asarray(dec["freqToOutput"], float)[:nf]
    D = np.asarray(dec["cursor_brain"], float)[:nf]
    ovr = np.asarray(ts["frequencyOverride"], float)[:nf] > 0.5
    inx = np.asarray(ts["CE_experimentRunning"], float)[:nf] > 0.5
    thr = np.asarray(ts["cursor_threshold_reward"], float)[:nf]
    hz_raw = np.where(inx & (f > 0), f, np.nan)      # Hz as played, reward clamp included
    nm = _S.rep["name"]
    if nm == "hz":
        raw, rew_level, base_level = hz_raw, np.full(nf, HZ_REWARD), HZ_BASE
    elif nm == "oct":
        raw = np.log2(hz_raw / 1000.0)
        rew_level = np.full(nf, np.log2(HZ_REWARD / 1000.0))
        base_level = float(np.log2(HZ_BASE / 1000.0))
    elif nm == "brain":
        raw = np.where(inx, D, np.nan)
        rew_level, base_level = thr, 0.0             # a step at the block boundary
    else:                                            # "theta"
        raw = np.where(inx, D / thr, np.nan)
        rew_level, base_level = np.ones(nf), 0.0
    tr = na.get_true_bmi_trials(na.get_trials_df(r, day))
    sm = _S.smooth_sec * _S.sr
    S = SimpleNamespace(
        mouse=mouse, i=int(day_i), day=day, nf=nf, rep=nm,
        cur=smooth_nan(raw, sm), cur_raw=raw,
        rew_level=rew_level, base_level=base_level,
        hz=smooth_nan(hz_raw, sm), hz_raw=hz_raw,
        ovr=ovr, in_exp=inx,
        quiet=np.asarray(ts["CS_quiescence"], float)[:nf] > 0.5,
        trials=tr, rew=na.get_active_reward_frames(r, day),
        remap=na.remapping_start_frame(tr), blocks=na.segment_factor_blocks(tr),
        thr=float(np.nanmedian(thr)),
        thr_all=np.unique(np.round(thr[np.isfinite(thr)], 3)),
        thr_by_f=ca.reward_threshold_by_factor(r, day))
    _SESS[key] = S
    return S


def day_zero_index(mouse):
    """Index of the session that counts as day 0 for this animal."""
    d0 = _S.day0_override.get(mouse) or _S.R[mouse]["date_day0"].load()
    d0 = str(np.asarray(d0).ravel()[0]) if not isinstance(d0, str) else d0
    return _S.DAYS[mouse].index(d0)


def day_table(mouse):
    """One row per session: trial counts, hit rate, time-to-target, block structure."""
    na, ca = _S.na, _S.ca
    r, days = _S.R[mouse], _S.DAYS[mouse]
    z = day_zero_index(mouse)
    rows = []
    for i, d in enumerate(days):
        tr = na.get_true_bmi_trials(na.get_trials_df(r, d))
        rec = dict(i=i, day=d, rel=i - z, n=len(tr))
        if len(tr):
            hit = tr.success == 1
            dur = (tr.end - tr.start) / _S.sr
            fac = tr.factor[np.isfinite(tr.factor)]
            rec.update(n_hit=int(hit.sum()), n_miss=int((tr.success == 0).sum()),
                       hit_rate=float(hit.mean()),
                       t_hit=float(np.nanmedian(dur[hit])) if hit.any() else np.nan,
                       factors=",".join(f"{v:g}" for v in np.unique(fac)),
                       is_remap=bool(np.unique(fac).size > 1),
                       remap_f=na.remapping_start_frame(tr),
                       n_rew=len(na.get_active_reward_frames(r, d)))
        rows.append(rec)
    t = pd.DataFrame(rows)
    t["is_remap"] = t["is_remap"].fillna(False).astype(bool)
    return t


def rel_day(mouse, i):
    """Session number counted from that animal's day 0 -- the number the figures label."""
    return int(_S.TAB[mouse].set_index("i").loc[int(i), "rel"])


def remap_indices(mouse):
    t = _S.TAB[mouse]
    return t.loc[t["is_remap"], "i"].to_numpy()


def stable_indices(mouse):
    """Sessions with trials, before the rotation phase (day 0 itself has no trials)."""
    t = _S.TAB[mouse]
    return t.loc[(t["n"] > 0) & (~t["is_remap"]), "i"].to_numpy()


def block_end(S, k):
    """End frame of block k, falling back to the session length for the last block."""
    e = S.blocks[k].end_frame
    return float(e) if np.isfinite(e) else float(S.nf)


# ------------------------------------------------------------------ the other-axis simulation
def _causal_exp(X, tau):
    """One-sided exponential filter along time -- the cursor-reconstruction preprocessing."""
    a = float(np.exp(-1.0 / tau))
    return lfilter(np.array([1.0 - a], np.float32), np.array([1.0, -a], np.float32), X, axis=1)


def _ridge(A, y, lam, chunk=20000):
    """Ridge in float64 normal equations, accumulated in chunks to stay in memory."""
    n, p = A.shape
    cm = np.zeros(p)
    for s in range(0, n, chunk):
        cm += A[s:s + chunk].astype(np.float64).sum(0)
    cm /= n
    ym = float(np.mean(y))
    G = np.zeros((p, p))
    h = np.zeros(p)
    for s in range(0, n, chunk):
        Ac = A[s:s + chunk].astype(np.float64) - cm
        G += Ac.T @ Ac
        h += Ac.T @ (y[s:s + chunk] - ym)
    b = np.linalg.solve(G + lam * n * np.eye(p), h)
    return b, ym - cm @ b


def _fill_gaps(y):
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    if not ok.any():
        return y
    t = np.arange(len(y))
    out = y.copy()
    out[~ok] = np.interp(t[~ok], t[ok], y[ok])
    return out


def simulate_other_axis(mouse, day_i, recompute=False):
    """What the cursor would have read on the decoder axis that was **not** live.

    Three of the four quantities come straight out of the logger (``CS``, ``D`` and therefore
    ``||M||``); only the other factor's linear readout ``M_other`` is estimated, by ridge
    regression from the population onto the rig's own ``M``, fitted **inside the block where
    that factor really was the cursor**.

    Held out on the washout (the intuitive axis live again), r(M) = 0.92-0.96, r(D) =
    0.73-0.88, r(tone) = 0.72-0.85 over the 15 remapping sessions of mouse_0403L.

    On a session with only one factor there is no other axis, so ``D_other`` is that same
    factor re-read through its own fitted model -- a reconstruction, not a second solution.
    """
    key = (mouse, int(day_i))
    if key in _SIM:
        return _SIM[key]
    na = _S.na
    S = session(mouse, day_i)
    fp = None
    if _S.sim_cache is not None:
        tag = f"sim_{mouse}_{S.day}_tau{SIM_TAU:g}_lam{SIM_RIDGE:g}_Dfill"
        fp = _S.sim_cache / f"{tag}.npz"
        if fp.exists() and not recompute:
            z = np.load(fp, allow_pickle=True)
            out = SimpleNamespace(**{k: z[k] for k in z.files})
            out.val = out.val.item()
            _SIM[key] = out
            return out
    dec = _S.ca.get_decoder_df(_S.R[mouse], S.day)
    D = np.asarray(dec["cursor_brain_raw"], float)[:S.nf]
    CS = np.asarray(dec["decoder_angles"], float)[:S.nf]
    good = np.isfinite(D) & np.isfinite(CS) & (np.abs(CS) > SIM_CS_MIN)
    M_live = np.where(good, D / CS ** 2, np.nan)
    normM = _fill_gaps(np.where(good, D / CS ** 3, np.nan))

    X = na.load_neural(_S.R[mouse], S.day).astype(np.float32)[:, :S.nf]
    X -= X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd[sd == 0] = 1
    X /= sd
    X = _causal_exp(X, SIM_TAU)
    XT = np.ascontiguousarray(X.T)
    del X

    fac = np.full(S.nf, np.nan)
    for b in S.blocks:
        e = b.end_frame if np.isfinite(b.end_frame) else S.nf
        fac[int(b.start_frame):int(min(e, S.nf))] = b.factor
    present = sorted({float(b.factor) for b in S.blocks})
    other = {f: [g for g in present if g != f][0] if len(present) > 1 else f for f in present}

    # A block runs from its FIRST true BMI trial to its LAST, so the rig's ~30 s changeover
    # between two blocks carries no factor label at all. Carry every unlabelled frame BACKWARD
    # from the next block that has one (tail forward from the last), so the simulation is
    # defined on every in-session frame. `fac` stays strict and is what the FITS use: a
    # changeover frame must never join a factor's training set.
    fac_f = fac.copy()
    _k = np.where(np.isfinite(fac))[0]
    if len(_k):
        fac_f = fac[_k[np.clip(np.searchsorted(_k, np.arange(S.nf), "left"), 0, len(_k) - 1)]]

    base = S.in_exp & good & np.isfinite(M_live)
    Mhat = {}
    for f in present:
        tr_m = base & (fac == f)
        beta, c = _ridge(XT[tr_m], M_live[tr_m], SIM_RIDGE)
        Mhat[f] = XT @ beta + c

    fac_other = np.array([other.get(v, np.nan) if np.isfinite(v) else np.nan for v in fac_f])
    th_live = np.array([S.thr_by_f.get(v, np.nan) if np.isfinite(v) else np.nan
                        for v in fac_f])
    th_other = np.array([S.thr_by_f.get(v, np.nan) if np.isfinite(v) else np.nan
                         for v in fac_other])
    M_o = np.full(S.nf, np.nan)
    M_l = np.full(S.nf, np.nan)
    for f in present:
        M_l[fac_f == f] = Mhat[f][fac_f == f]
        M_o[fac_other == f] = Mhat[f][fac_other == f]
    D_other = M_o ** 3 / normM ** 2
    D_live = M_l ** 3 / normM ** 2
    D_other[~S.in_exp] = np.nan
    D_live[~S.in_exp] = np.nan

    # held-out check: fit the intuitive axis on block 0, simulate it through the washout
    val = {}
    if len(S.blocks) >= 3 and S.blocks[0].factor == S.blocks[2].factor:
        f0 = float(S.blocks[0].factor)
        th = S.thr_by_f[f0]
        m_fit = base & (fac == f0) & (np.arange(S.nf) < block_end(S, 0))
        m_te = base & (fac == f0) & (np.arange(S.nf) >= S.blocks[2].start_frame) & ~S.ovr
        if m_fit.sum() > 500 and m_te.sum() > 500:
            beta, c = _ridge(XT[m_fit], M_live[m_fit], SIM_RIDGE)
            Mp = XT[m_te] @ beta + c
            Dp = Mp ** 3 / normM[m_te] ** 2
            hp = d_to_hz(Dp, th)
            val = dict(n=int(m_te.sum()),
                       r_M=float(np.corrcoef(Mp, M_live[m_te])[0, 1]),
                       r_D=float(np.corrcoef(Dp, D[m_te])[0, 1]),
                       r_hz=float(np.corrcoef(hp, S.hz_raw[m_te])[0, 1]),
                       lo_sim=float(np.mean(Dp < 0)), lo_real=float(np.mean(D[m_te] < 0)),
                       frac_sim=float(np.mean(hp >= HZ_REWARD)),
                       frac_real=float(np.mean(S.hz_raw[m_te] >= HZ_REWARD)))
    out = SimpleNamespace(D_other=D_other.astype(np.float32),
                          D_live=D_live.astype(np.float32),
                          th_other=th_other.astype(np.float32),
                          th_live=th_live.astype(np.float32),
                          M_other=M_o.astype(np.float32),
                          M_live=M_live.astype(np.float32),
                          normM=normM.astype(np.float32),
                          fac=fac_f.astype(np.float32),
                          fac_block=fac.astype(np.float32),
                          fac_other=fac_other.astype(np.float32),
                          Mhat=np.array([Mhat[f] for f in present], np.float32),
                          present=np.array(present, float),
                          val=val)
    if fp is not None:
        np.savez_compressed(fp, **vars(out))
    _SIM[key] = out
    return out


def sim_cursor(sim, which="other", smooth=True):
    """The simulated cursor in the ACTIVE representation. ``which`` is "other" or "live"."""
    D = np.asarray(getattr(sim, f"D_{which}"), float)
    th = np.asarray(getattr(sim, f"th_{which}"), float)
    y = d_to_rep(D, th)
    return smooth_nan(y, _S.smooth_sec * _S.sr) if smooth else y


def has_other_axis(mouse, day_i):
    """True only where a *second* decoder axis was really live in that session."""
    return len({float(b.factor) for b in session(mouse, day_i).blocks}) > 1
