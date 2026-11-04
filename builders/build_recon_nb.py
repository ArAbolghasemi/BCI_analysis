"""AUTO-GENERATED from cursor_reconstruction.ipynb by builders/sync_builders.py.

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


# ---------------- Cursor reconstruction — can we rebuild the BMI cursor from `spks` + th ----------------
md(r"""# Cursor reconstruction — can we rebuild the BMI cursor from `spks` + the factor vectors?

**Question.** For every day, how closely does a cursor recomputed from the packet's neural data and
the stored decoder weights match the cursor the rig actually produced and logged (`cursor_brain`)?

**Why this is not trivial.** The online cursor was computed from **dF/F**, read out from day-0 ROI
footprints non-rigidly warped into that day's field of view. The packet instead stores suite2p
**`spks`** (OASIS-deconvolved, *not* dF/F) for a ROICaT-tracked neuron set, and only a *subset* of
the decoder's neurons survived that tracking. The reconstruction is therefore necessarily an
approximation, and the point of this notebook is to measure how good an approximation it is, day by
day.

## What the rig did (BWAIN, `core_functions/userFunction_BMIv11_withZ.m`)

```
dFoF_i = (F_i - B_i) / B_i                 B = rolling 30th percentile, 10 min window
M_k    = sum_i dFoF_i * factor_space[i,k]  factor_space columns are unit-norm
CS_k   = M_k / norm(M)                     cosine similarity in the K-dim factor-score space
D_k    = abs(CS_k)^p * M_k                 p = angle_power = 2
cursor_brain = D_[factor_to_use]           factor_to_use is MATLAB 1-indexed
```

## What we do here

The identical algebra, with `spks` in place of dF/F and the packet's (truncated)
`weights_day0_idxFactors` in place of `factor_space`:

```
M_k = sum_i x_i * W[i,k]   ->  then the same CS, the same gate, the same factor selection
```

Two preprocessing steps are applied to `x` before the projection, each justified in section 4a:

1. **per-neuron z-score** over the session — approximately undoes the brightness gain that
   separates `spks` (raw-fluorescence units) from dF/F (divided by baseline `B_i`);
2. **a short causal exponential filter** — `spks` is a deconvolved impulse train while dF/F is its
   calcium-filtered version, so the filter puts the two on comparable timescales.

## Two comparison targets

| target | what it tests |
|---|---|
| `cursor_brain` | the full pipeline, **including** the non-linear angle gate |
| `M` recovered as `cursor_brain_raw / abs(decoder_angles)^p` | the **purely linear** readout, gate removed |

The second target exists because the rig also logged `decoder_angles` (= `CS` for the active
factor), so the linear magnitude can be divided back out exactly. Comparing against it separates
"did we get the axis right" from "did we get the gate right".

*This notebook computes and displays; it deliberately does not draw conclusions.*""", cid='recon-00')
code(r"""%matplotlib inline
import sys, hashlib, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import lfilter

REPO = Path.cwd()
sys.path.insert(0, str(REPO / "utilities"))
import neural_analysis_utils as na      # packet access, trials
import cursor_analysis_utils as ca      # decoder-logger access
import richfile as rfs
na.register_richfile_types(rfs)

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **k): return x

# ============================ CONFIG ============================
SR          = 30                                   # Hz, native imaging / logger rate
DATA_ROOT   = Path(r"C:\Data\datasets\bmi_packets\bmi_packets")
OUTPUT_ROOT = REPO / "output" / "cursor_reconstruction"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
MICE        = ["mouse_0403L", "mouse_0322R", "mouse_0403R", "mouse_0916N"]
COLS        = dict(zip(MICE, ["crimson", "seagreen", "steelblue", "darkorange"]))

ANGLE_POWER_FALLBACK = 2.0   # only if expParams is unreadable; the real value is read per day
TAU_FRAMES           = 5     # causal exponential filter, ~0.17 s (chosen by the scan in section 4a)
CS_FLOOR             = 0.05  # |decoder_angles| below this makes the recovered M numerically unusable
EXAMPLE_MOUSE, EXAMPLE_DAY = "mouse_0403L", "20230709"

print("config loaded")""", cid='recon-01')

# ---------------- 1 · Helpers — reading what the rig used ----------------
md(r"""## 1 · Helpers — reading what the rig used

Data loading comes from `na` / `ca`; everything specific to this analysis is defined here so the
method stays visible.

`expParams[day].baselineStuff` is the rig's own state for that session: `factor_space` (the exact
online weight matrix, in **day-0 ROI** indexing), `cursors` (angle power, reward threshold, and
which factor each cursor reads) and `block_sequence` (which cursor each block used). On remapping
days `cursors` is a *list* of cursor structs and the block sequence switches between them
mid-session.""", cid='recon-02')
code(r"""def baseline_stuff(r, day):
    # Rig state for one session, or None on days with no BMI. (Day 0 is a spontaneous session,
    # where baselineStuff is stored as a bare json leaf rather than a dict.)
    try:
        bs = r["extras"]["expParams"][day]["baselineStuff"]
        return bs if hasattr(bs, "keys") and "factor_space" in bs.keys() else None
    except Exception:
        return None


def online_factor_space(r, day):
    # The exact weight matrix the rig used that session, in day-0 ROI indexing. None if no BMI.
    bs = baseline_stuff(r, day)
    return None if bs is None else np.asarray(bs["factor_space"].load(), float)


def online_params(r, day):
    # angle_power, and the MATLAB factor index behind each cursor, as the rig was configured.
    bs = baseline_stuff(r, day)
    if bs is None:
        return dict(angle_power=ANGLE_POWER_FALLBACK, cursor_factors=[], thresholds=[])
    cu = bs["cursors"].load()
    cu = cu if isinstance(cu, list) else [cu]
    g = lambda d, k: float(np.asarray(d[k]).ravel()[0])
    return dict(angle_power=g(cu[0], "angle_power"),
                cursor_factors=[g(c, "factor_to_use") for c in cu],
                thresholds=[g(c, "threshold_reward") for c in cu])


def frame_factor_col(trials, T):
    # Per-frame python column index of the axis actually driving the cursor.
    # The trials logger records factor_to_use per trial, MATLAB 1-indexed, so subtract 1.
    # Each trial's axis is held forward until the next trial starts (the cursor keeps running
    # through the ITI on the block's axis); frames before the first trial take the first value.
    col = np.full(T, np.nan)
    for s, f in zip(np.asarray(trials.start, float), np.asarray(trials.factor, float)):
        if np.isfinite(f) and np.isfinite(s):
            col[max(int(s), 0):] = f - 1.0
    if np.isfinite(col).any():
        col[~np.isfinite(col)] = col[np.isfinite(col)][0]
    else:
        col[:] = 0.0
    return col.astype(int)


def load_day(r, day):
    # spks (N,T), the decoder logger and the trials, all truncated to a common length.
    # The truncation matters: on three days of mouse_0403L the logger stops at 108,000 frames
    # while the neural data runs to 144,000. Alignment holds from frame 0, so clipping is correct.
    X  = na.load_neural(r, day)
    df = ca.get_decoder_df(r, day)
    T  = min(X.shape[1], len(df))
    tr = na.get_true_bmi_trials(na.get_trials_df(r, day))
    return X[:, :T], df.iloc[:T].reset_index(drop=True), tr, T


print("packet helpers defined")""", cid='recon-03')

# ---------------- 2 · Helpers — the reconstruction itself ----------------
md(r"""## 2 · Helpers — the reconstruction itself

`reconstruct` is a literal transcription of the four MATLAB lines quoted in the header.
`recovered_M` inverts the angle gate on the *logged* signals to expose the rig's own linear
magnitude.""", cid='recon-04')
code(r"""def zscore_rows(X):
    # Per-neuron z-score over the whole session (justified in section 4a).
    sd = X.std(axis=1, keepdims=True)
    sd[sd < 1e-9] = 1.0
    return (X - X.mean(axis=1, keepdims=True)) / sd


def calcium_filter(X, tau_frames):
    # Causal single-pole exponential, y[t] = a*y[t-1] + (1-a)*x[t], applied along time.
    # Approximates the calcium impulse response that deconvolution removed.
    if tau_frames is None or tau_frames <= 0:
        return X
    a = float(np.exp(-1.0 / tau_frames))
    return lfilter([1.0 - a], [1.0, -a], X, axis=-1)


def reconstruct(X, W, p):
    # The rig's algebra. X: (N,T) neural, W: (N,K) weights. Returns M, CS, D, each (K,T).
    M  = W.T @ X                                 # M_k = sum_i x_i * W[i,k]
    nn = np.linalg.norm(M, axis=0)
    nn[nn < 1e-12] = 1e-12
    CS = M / nn                                  # cosine similarity across the K factors
    D  = (np.abs(CS) ** p) * M                   # the angle gate
    return M, CS, D


def pick_factor(A, col):
    # Select one row of a (K,T) array per frame, using the per-frame factor column.
    col = np.clip(col, 0, A.shape[0] - 1)
    return A[col, np.arange(A.shape[1])]


def recovered_M(df, p):
    # Invert the angle gate on the LOGGED signals to recover the rig's own linear magnitude:
    #   cursor_brain_raw = |CS|^p * M   and   decoder_angles = CS
    #   =>  M = cursor_brain_raw / |CS|^p
    # Frames with |CS| near zero blow up, so they are masked out rather than clipped.
    cbr = np.asarray(df["cursor_brain_raw"], float)
    cs  = np.asarray(df["decoder_angles"], float)
    M   = cbr / np.maximum(np.abs(cs) ** p, 1e-12)
    usable = np.isfinite(M) & (np.abs(cs) > CS_FLOOR)
    return M, usable


def pearson(a, b, mask=None):
    # Correlation over the frames where both signals are finite.
    m = np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        m = m & mask
    if m.sum() < 100 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


print("reconstruction helpers defined")""", cid='recon-05')

# ---------------- 3 · Integrity check — is the stored weight matrix the one the rig used ----------------
md(r"""## 3 · Integrity check — is the stored weight matrix the one the rig used?

The packet holds **one** weight matrix per mouse. Everything downstream assumes that matrix is the
decoder the rig was running on every day. That assumption has to be tested before any cursor is
compared, because if the rig was re-fit mid-experiment the packet can only be right for part of the
dataset.

Two checks:

**3a — decoder epochs.** Hash each day's online `factor_space` and group days by it. One group means
one decoder for the whole experiment; more than one means the decoder changed.

**3b — index chain.** Confirm the packet's weight rows are an exact subset of the online rows, and
measure what UCID tracking removed. Online column norms are 1 by construction, so whatever they are
in the packet is the fraction of that axis still present.""", cid='recon-06')
code(r"""def factor_space_id(fsp):
    # Exact identity of a weight matrix: shape plus a hash of its bytes.
    return (fsp.shape, hashlib.md5(np.ascontiguousarray(fsp, dtype=np.float64).tobytes()).hexdigest()[:10])


epoch_rows, day_epoch = [], {}      # day_epoch[(mouse, day)] -> epoch index, 0 = the packet's
for mouse in MICE:
    r = na.open_mouse(rfs, DATA_ROOT, mouse)
    days = [d for d in na.get_day_list(r) if baseline_stuff(r, d) is not None]
    ids = {d: factor_space_id(online_factor_space(r, d)) for d in days}

    W = np.asarray(r["weights_day0_idxFactors"].load(), float)
    Wk = W[~np.isnan(W).any(axis=1)]

    order, seen = [], {}
    for d in days:
        if ids[d] not in seen:
            seen[ids[d]] = len(order)
            order.append(ids[d])
    for eid in order:
        eday = [d for d in days if ids[d] == eid]
        fsp = online_factor_space(r, eday[0])
        rowset = {tuple(np.round(v, 12)) for v in fsp}
        n_match = sum(tuple(np.round(v, 12)) in rowset for v in Wk)
        epoch_rows.append(dict(
            mouse=mouse, epoch=seen[eid], n_days=len(eday),
            first_day=eday[0], last_day=eday[-1],
            online_shape=str(fsp.shape), n_factors=fsp.shape[1],
            packet_rows_matched=f"{n_match}/{len(Wk)}",
            matches_packet=bool(n_match == len(Wk) and len(Wk) > 0),
            coverage_pct=round(100 * len(Wk) / fsp.shape[0], 1) if n_match == len(Wk) else np.nan))
    ok_epochs = {e["epoch"] for e in epoch_rows if e["mouse"] == mouse and e["matches_packet"]}
    for d in days:
        day_epoch[(mouse, d)] = seen[ids[d]] in ok_epochs

epochs = pd.DataFrame(epoch_rows)
print("One row per distinct online decoder. matches_packet = every packet weight row is present")
print("in that epoch's factor_space, i.e. the packet's matrix IS that decoder.\n")
display(epochs)""", cid='recon-07')
code(r"""rows = []
for mouse in MICE:
    r = na.open_mouse(rfs, DATA_ROOT, mouse)
    W = np.asarray(r["weights_day0_idxFactors"].load(), float)
    keep = ~np.isnan(W).any(axis=1)
    e = epochs[(epochs.mouse == mouse) & epochs.matches_packet]
    valid = sum(day_epoch[(mouse, d)] for d in na.get_day_list(r) if (mouse, d) in day_epoch)
    total = sum(1 for d in na.get_day_list(r) if (mouse, d) in day_epoch)
    rows.append(dict(mouse=mouse, packet_shape=str(W.shape),
                     rows_with_weight=int(keep.sum()),
                     coverage_pct=float(e.coverage_pct.iloc[0]) if len(e) else np.nan,
                     bmi_days=total, days_packet_decoder_is_correct=valid,
                     col_norms=np.round(np.linalg.norm(np.nan_to_num(W), axis=0), 3),
                     col_norm_sq=np.round(np.linalg.norm(np.nan_to_num(W), axis=0) ** 2, 3)))
display(pd.DataFrame(rows))
print("col_norm_sq is the fraction of each axis's squared length still present after UCID tracking.")""", cid='recon-08')

# ---------------- 4 · One day in detail ----------------
md(r"""## 4 · One day in detail

Work a single stable day end to end, then (4a) justify the two preprocessing choices by scanning
them against the data rather than asserting them.""", cid='recon-09')
code(r"""r_ex = na.open_mouse(rfs, DATA_ROOT, EXAMPLE_MOUSE)
W_ex = np.nan_to_num(np.asarray(r_ex["weights_day0_idxFactors"].load(), float), nan=0.0)
X, df, tr, T = load_day(r_ex, EXAMPLE_DAY)
par = online_params(r_ex, EXAMPLE_DAY)
col = frame_factor_col(tr, T)
p   = par["angle_power"]

cb         = np.asarray(df["cursor_brain"], float)
M_log, okM = recovered_M(df, p)

Xp = calcium_filter(zscore_rows(X), TAU_FRAMES)
M, CS, D = reconstruct(Xp, W_ex, p)
d_hat = pick_factor(D, col)
m_hat = pick_factor(M, col)

print(f"{EXAMPLE_MOUSE} {EXAMPLE_DAY}:  N={X.shape[0]} neurons, T={T} frames ({T/SR/60:.0f} min)")
print(f"  angle_power p = {p};  cursor->factor map (MATLAB) = {par['cursor_factors']};  "
      f"reward thresholds = {par['thresholds']}")
print(f"  per-frame python factor column: {dict(zip(*np.unique(col, return_counts=True)))}")
print(f"  cursor_brain finite on {100*np.isfinite(cb).mean():.1f}% of frames;  "
      f"recovered M usable on {100*okM.mean():.1f}%")
print()
print(f"  r(reconstructed gated D , logged cursor_brain) = {pearson(d_hat, cb):.4f}")
print(f"  r(reconstructed linear M , recovered linear M) = {pearson(m_hat, M_log, okM):.4f}")""", cid='recon-10')
code(r"""fig, ax = plt.subplots(2, 2, figsize=(14, 6.4), gridspec_kw=dict(width_ratios=[2.4, 1]))
z = lambda v: (v - np.nanmean(v)) / np.nanstd(v)
s0, s1 = int(12 * 60 * SR), int(13 * 60 * SR)      # a 60 s window, 12 min into the session
t = np.arange(s0, s1) / SR

ax[0, 0].plot(t, z(cb)[s0:s1], lw=1.4, color="k", label="logged cursor_brain")
ax[0, 0].plot(t, z(d_hat)[s0:s1], lw=1.2, color="crimson", alpha=.85, label="reconstructed (gated D)")
ax[0, 0].set_ylabel("cursor (z)")
ax[0, 0].legend(fontsize=8, loc="upper right")
ax[0, 0].set_title(f"{EXAMPLE_MOUSE}  {EXAMPLE_DAY}  —  60 s excerpt", fontsize=10)

M_masked = np.where(okM, M_log, np.nan)
ax[1, 0].plot(t, z(M_masked)[s0:s1], lw=1.4, color="k", label="recovered linear M (logged)")
ax[1, 0].plot(t, z(m_hat)[s0:s1], lw=1.2, color="steelblue", alpha=.85, label="reconstructed M")
ax[1, 0].set_xlabel("time (s)")
ax[1, 0].set_ylabel("magnitude (z)")
ax[1, 0].legend(fontsize=8, loc="upper right")

for a, (xx, yy, lab) in zip(ax[:, 1],
        [(cb, d_hat, "gated D  vs  cursor_brain"),
         (M_masked, m_hat, "linear M  vs  recovered M")]):
    m = np.isfinite(xx) & np.isfinite(yy)
    a.hexbin(xx[m], z(yy)[m], gridsize=55, bins="log", cmap="Greys", mincnt=1)
    a.set_title(f"{lab}\nr = {pearson(xx, yy):.3f}", fontsize=9)
    a.set_xlabel("logged")
    a.set_ylabel("reconstructed (z)")

plt.tight_layout()
plt.savefig(OUTPUT_ROOT / "example_day.png", dpi=130)
plt.show()""", cid='recon-11')

# ---------------- 4a · Do the two preprocessing steps actually help? ----------------
md(r"""### 4a · Do the two preprocessing steps actually help?

`spks` differs from the dF/F the decoder read in two ways, each with a candidate correction:

* **gain** — `spks` is in raw-fluorescence units, so neuron *i*'s amplitude scales with its baseline
  brightness `B_i`, and dF/F divides by exactly that `B_i`. Per-neuron z-scoring should therefore
  approximately reproduce the dF/F normalisation. Below, **z-scored** vs **raw spks** input.
* **timescale** — deconvolution strips the calcium transient, so `spks` is far spikier than dF/F. A
  short causal exponential puts it back; `tau = 0` means no filter.

Neither correction is assumed — the scan picks them. The curve is flat across `tau` ≈ 3–6, and
`TAU_FRAMES` in the config sits inside that plateau (the dotted line); the difference across it
is under 0.003 in r, so the exact value is not load-bearing.""", cid='recon-12')
code(r"""scan_days = [d for d in na.get_day_list(r_ex) if baseline_stuff(r_ex, d) is not None][4:9]
cache = {}
for d in scan_days:
    Xd, dfd, trd, Td = load_day(r_ex, d)
    p_d = online_params(r_ex, d)["angle_power"]
    Mld, okd = recovered_M(dfd, p_d)
    cache[d] = dict(raw=Xd, z=zscore_rows(Xd), col=frame_factor_col(trd, Td), p=p_d,
                    cb=np.asarray(dfd["cursor_brain"], float), Ml=Mld, ok=okd)

grid = []
for tau in [0, 2, 3, 4, 5, 6, 8, 12, 20]:
    for inp in ["z-scored", "raw spks"]:
        rD, rM = [], []
        for d in scan_days:
            c = cache[d]
            Xin = calcium_filter(c["z"] if inp == "z-scored" else c["raw"], tau)
            Mi, _, Di = reconstruct(Xin, W_ex, c["p"])
            rD.append(pearson(pick_factor(Di, c["col"]), c["cb"]))
            rM.append(pearson(pick_factor(Mi, c["col"]), c["Ml"], c["ok"]))
        grid.append(dict(tau_frames=tau, input=inp,
                         r_gated_D=np.nanmean(rD), r_linear_M=np.nanmean(rM)))
scan = pd.DataFrame(grid)

fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
for inp, c in [("z-scored", "crimson"), ("raw spks", "grey")]:
    s = scan[scan["input"] == inp]
    ax[0].plot(s.tau_frames, s.r_gated_D, "o-", color=c, label=inp)
    ax[1].plot(s.tau_frames, s.r_linear_M, "o-", color=c, label=inp)
for a, ttl in zip(ax, ["gated D  vs  cursor_brain", "linear M  vs  recovered M"]):
    a.axvline(TAU_FRAMES, color="k", ls=":", lw=1)
    a.set_xlabel("exponential filter tau (frames; 0 = none)")
    a.set_ylabel("mean r")
    a.set_title(ttl, fontsize=10)
    a.grid(alpha=.3)
    a.legend(fontsize=8)
plt.suptitle(f"{EXAMPLE_MOUSE} — preprocessing scan over {len(scan_days)} days "
             f"(dotted = TAU_FRAMES in use)", fontsize=10)
plt.tight_layout()
plt.savefig(OUTPUT_ROOT / "preprocessing_scan.png", dpi=130)
plt.show()
display(scan.pivot_table(index="tau_frames", columns="input",
                         values=["r_gated_D", "r_linear_M"]).round(4))""", cid='recon-13')

# ---------------- 5 · Every day, every mouse ----------------
md(r"""## 5 · Every day, every mouse

Three reconstructions per day, so the contribution of each modelling choice stays visible:

| variant | meaning |
|---|---|
| `r_D_perframe` | gated cursor, axis taken **per frame** from the trials logger — the honest reconstruction |
| `r_D_fixed`    | gated cursor, axis fixed to the day's **modal** factor — wrong on remapping days, kept to show by how much |
| `r_M`          | linear magnitude against the recovered linear magnitude — the gate removed |

`packet_decoder_ok` carries the section-3a verdict: **False** means the rig was running a decoder
the packet does not contain, so the reconstruction on that day is not expected to work at all.
`frac_cursor_finite` drops days where the logged cursor is itself mostly missing (day 0 is a
spontaneous session with no BMI running).""", cid='recon-14')
code(r"""CACHE = OUTPUT_ROOT / "reconstruction_per_day.csv"
RECOMPUTE = True       # set False after the first run to reuse the cache

if CACHE.exists() and not RECOMPUTE:
    per_day = pd.read_csv(CACHE, dtype={"day": str})
    print(f"loaded cache: {CACHE}  ({len(per_day)} rows)   -- set RECOMPUTE=True to rebuild")
else:
    recs = []
    for mouse in MICE:
        r = na.open_mouse(rfs, DATA_ROOT, mouse)
        W = np.nan_to_num(np.asarray(r["weights_day0_idxFactors"].load(), float), nan=0.0)
        norms = np.linalg.norm(W, axis=0)
        days = na.get_day_list(r)
        for day in tqdm(days, desc=mouse, leave=False):
            if baseline_stuff(r, day) is None:
                continue
            Xd, dfd, trd, Td = load_day(r, day)
            p_d  = online_params(r, day)["angle_power"]
            cold = frame_factor_col(trd, Td)
            cbd  = np.asarray(dfd["cursor_brain"], float)
            Mld, okd = recovered_M(dfd, p_d)

            Xin = calcium_filter(zscore_rows(Xd), TAU_FRAMES)
            Mi, _, Di = reconstruct(Xin, W, p_d)
            modal = int(np.bincount(cold).argmax())

            recs.append(dict(
                mouse=mouse, day=day, day_idx=days.index(day),
                packet_decoder_ok=bool(day_epoch.get((mouse, day), False)),
                n_neurons=Xd.shape[0], n_frames=Td,
                n_factors_used=int(len(np.unique(cold))), modal_factor=modal,
                is_remap=bool(len(np.unique(cold)) > 1),
                frac_cursor_finite=float(np.isfinite(cbd).mean()),
                frac_M_usable=float(okd.mean()),
                r_D_perframe=pearson(pick_factor(Di, cold), cbd),
                r_D_fixed=pearson(Di[modal], cbd),
                r_M=pearson(pick_factor(Mi, cold), Mld, okd),
                retained_norm_sq=float(norms[min(modal, len(norms) - 1)] ** 2)))
    per_day = pd.DataFrame(recs)
    per_day.to_csv(CACHE, index=False)
    print(f"computed and cached -> {CACHE}")

per_day = per_day[per_day.frac_cursor_finite > 0.5].reset_index(drop=True)
print("\nSplit by whether the packet actually contains the decoder the rig was running:")
display(per_day.groupby(["mouse", "packet_decoder_ok"])[["r_D_perframe", "r_M"]]
        .agg(["median", "min", "max", "count"]).round(3))""", cid='recon-15')
code(r"""fig, ax = plt.subplots(1, 2, figsize=(13, 3.8), sharey=True)
for m, g in per_day.groupby("mouse"):
    g = g.sort_values("day_idx")
    for a, key in zip(ax, ["r_D_perframe", "r_M"]):
        ok, bad = g[g.packet_decoder_ok], g[~g.packet_decoder_ok]
        a.plot(g.day_idx, g[key], "-", lw=1.0, color=COLS[m], alpha=.45)
        a.plot(ok.day_idx, ok[key], "o", ms=5, color=COLS[m], label=m)
        a.plot(bad.day_idx, bad[key], "x", ms=6, color=COLS[m], mew=1.6)
        rmd = g[g.is_remap]
        a.plot(rmd.day_idx, rmd[key], "o", ms=9, mfc="none", mec=COLS[m], mew=1.3)
for a, ttl in zip(ax, ["gated D  vs  logged cursor_brain", "linear M  vs  recovered linear M"]):
    a.set_xlabel("day index within mouse")
    a.set_title(ttl, fontsize=10)
    a.grid(alpha=.3)
    a.set_ylim(0, 1)
ax[0].set_ylabel("Pearson r")
ax[0].legend(fontsize=8, loc="lower left")
plt.suptitle("Fidelity per day.  dot = packet holds the running decoder,  x = it does not,  "
             "open ring = remapping day", fontsize=10)
plt.tight_layout()
plt.savefig(OUTPUT_ROOT / "fidelity_per_day.png", dpi=130)
plt.show()""", cid='recon-16')

# ---------------- 6 · Remapping days need the per-frame axis ----------------
md(r"""## 6 · Remapping days need the per-frame axis

On remapping sessions the rig switches cursor mid-session: `block_sequence` selects a different
entry of `cursors`, which reads a different factor. Treating `cursor_brain` as one axis across such
a session mixes two different readouts. The comparison below is the direct cost of getting this
wrong.""", cid='recon-17')
code(r"""rmap = per_day[per_day.is_remap & per_day.packet_decoder_ok].copy()
if len(rmap) == 0:
    print("no remapping days with a matching packet decoder")
else:
    rmap["delta"] = rmap.r_D_perframe - rmap.r_D_fixed
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.6))
    idx = np.arange(len(rmap))
    ax[0].bar(idx - .2, rmap.r_D_fixed, .4, label="fixed modal axis",
              color="lightgrey", edgecolor="k")
    ax[0].bar(idx + .2, rmap.r_D_perframe, .4, label="per-frame axis", color="crimson")
    ax[0].set_xticks(idx)
    ax[0].set_xticklabels(rmap.day, rotation=90, fontsize=6)
    ax[0].set_ylabel("r vs cursor_brain")
    ax[0].legend(fontsize=8)
    ax[0].set_title("Remapping days: cost of assuming a single axis", fontsize=10)

    ax[1].hist(rmap.delta, bins=14, color="crimson", alpha=.8, edgecolor="k")
    ax[1].axvline(0, color="k", lw=1)
    ax[1].set_xlabel("r(per-frame) - r(fixed)")
    ax[1].set_ylabel("days")
    ax[1].set_title(f"median gain = {rmap.delta.median():+.3f}", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_ROOT / "remap_axis.png", dpi=130)
    plt.show()
    display(rmap.groupby("mouse")[["r_D_fixed", "r_D_perframe", "delta"]].median().round(3))""", cid='recon-18')

# ---------------- 7 · What limits the fidelity ----------------
md(r"""## 7 · What limits the fidelity

Two candidate limits, on different scales:

* **wrong decoder** — a per-*day* property (section 3a), so it can be tested within a mouse;
* **axis truncation** — `coverage_pct` and `retained_norm_sq` are fixed per mouse, so truncation can
  only be compared *across* mice, n = 4. Treat that panel as descriptive.""", cid='recon-19')
code(r"""fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))

ok  = per_day[per_day.packet_decoder_ok]
bad = per_day[~per_day.packet_decoder_ok]
ax[0].boxplot([ok.r_D_perframe.dropna(), bad.r_D_perframe.dropna()],
              labels=[f"packet decoder\ncorrect (n={len(ok)})",
                      f"rig used a different\ndecoder (n={len(bad)})"], widths=.55)
for i, g in enumerate([ok, bad], start=1):
    ax[0].scatter(np.random.normal(i, .05, len(g)), g.r_D_perframe,
                  s=16, c=[COLS[m] for m in g.mouse], alpha=.75, zorder=3)
ax[0].set_ylabel("r  (gated D vs cursor_brain)")
ax[0].set_title("Does the packet hold the running decoder?", fontsize=10)

summ = (per_day[per_day.packet_decoder_ok].groupby("mouse")
        .agg(r=("r_D_perframe", "median"), rM=("r_M", "median"),
             nrm=("retained_norm_sq", "median"), nn=("n_neurons", "max")))
cov = {row.mouse: row.coverage_pct for row in
       epochs[epochs.matches_packet].itertuples()}
for m, row in summ.iterrows():
    ax[1].scatter(cov.get(m, np.nan), row.r, s=90, color=COLS[m], label=m)
    ax[1].annotate(m.replace("mouse_", ""), (cov.get(m, np.nan), row.r),
                   textcoords="offset points", xytext=(6, -3), fontsize=7)
ax[1].set_xlabel("% of decoder neurons retained in the packet")
ax[1].set_ylabel("median r  (correct-decoder days)")
ax[1].set_title("Truncation, across mice (n=4)", fontsize=10)

for m, g in per_day[per_day.packet_decoder_ok].groupby("mouse"):
    ax[2].scatter(g.frac_M_usable, g.r_M, s=24, color=COLS[m], label=m, alpha=.8)
ax[2].set_xlabel("fraction of frames where recovered M is usable")
ax[2].set_ylabel("r  (linear M vs recovered M)")
ax[2].set_title("Gate-free comparison", fontsize=10)
ax[2].legend(fontsize=7)
for a in ax:
    a.grid(alpha=.3)
    a.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(OUTPUT_ROOT / "fidelity_limits.png", dpi=130)
plt.show()

print("per-mouse summary, correct-decoder days only")
display(per_day[per_day.packet_decoder_ok].groupby("mouse").agg(
    days=("day", "count"), neurons=("n_neurons", "max"),
    retained_norm_sq=("retained_norm_sq", "median"),
    r_gated_D=("r_D_perframe", "median"), r_linear_M=("r_M", "median")).round(3))""", cid='recon-20')

# ---------------- 8 · Caveats ----------------
md(r"""## 8 · Caveats

1. **`spks` is not the signal the decoder read.** The rig used its own online dF/F with a rolling
   30th-percentile baseline. The z-score and the exponential filter are approximations of that
   relationship, chosen empirically in 4a, not derived.
2. **Neuron identity differs.** Online, "the same neuron" meant the same day-0 pixel mask warped
   into today's field of view (`imregdemons`); in the packet it means the same ROICaT UCID. These
   are independent correspondence rules and will disagree on some neurons on some days.
3. **The axes are truncated and no longer orthonormal.** Column norms in section 3b are below 1,
   and unequally so, which distorts `CS` — a ratio *across* factors.
4. **`decoder_angles` near zero.** The recovered linear `M` divides by `|CS|^p`, so frames with
   `|CS| < CS_FLOOR` are discarded rather than clipped; `frac_M_usable` reports how many survive.
5. **Logger truncation.** Three days of mouse_0403L have a logger shorter than the neural data;
   `load_day` clips both to the common length, which is correct only because the alignment starts
   at frame 0.
6. **Correlation is scale-free.** A high `r` says the *shape* of the reconstructed cursor matches.
   It says nothing about absolute units, and the reconstruction is not on the rig's scale, so the
   reward thresholds cannot be applied to it directly.
7. **`day_epoch` depends on section 3a having run.** Re-running section 5 in a fresh kernel without
   section 3 will silently mark every day as `packet_decoder_ok = False`.""", cid='recon-21')

nb["cells"] = cells
nb["metadata"] = {'kernelspec': {'display_name': 'BCI_analysis', 'language': 'python', 'name': 'python3'}, 'language_info': {'codemirror_mode': {'name': 'ipython', 'version': 3}, 'file_extension': '.py', 'mimetype': 'text/x-python', 'name': 'python', 'nbconvert_exporter': 'python', 'pygments_lexer': 'ipython3', 'version': '3.11.15'}, 'widgets': {'application/vnd.jupyter.widget-state+json': {'state': {'01473061340e44579225b3bf95409a98': {'model_module': '@jupyter-widgets/controls', 'model_module_version': '2.0.0', 'model_name': 'HTMLModel', 'state': {'_dom_classes': [], '_model_module': '@jupyter-widgets/controls', '_model_module_version': '2.0.0', '_model_name': 'HTMLModel', '_view_count': None, '_view_module': '@jupyter-widgets/controls', '_view_module_version': '2.0.0', '_view_name': 'HTMLView', 'description': '', 'description_allow_html': False, 'layout': 'IPY_MODEL_c44c420a781044e09c7a7da6310ebf3c', 'placeholder': '\u200b', 'style': 'IPY_MODEL_2586261739b847758e141361ab3383e9', 'tabbable': None, 'tooltip': None, 'value': 'mouse_0403R:\u2007100%'}}, '0155efb81ac04a578aa2f458422e2119': {'model_module': '@jupyter-widgets/controls', 'model_module_version': '2.0.0', 'model_name': 'HTMLModel', 'state': {'_dom_classes': [], '_model_module': '@jupyter-widgets/controls', '_model_module_version': '2.0.0', '_model_name': 'HTMLModel', '_view_count': None, '_view_module': '@jupyter-widgets/controls', '_view_module_version': '2.0.0', '_view_name': 'HTMLView', 'description': '', 'description_allow_html': False, 'layout': 'IPY_MODEL_cf590bb2ba9c471d9fb5254e7132104d', 'placeholder': '\u200b', 'style': 'IPY_MODEL_ee8a9abefa7942899d8a0218db0a6cc3', 'tabbable': None, 'tooltip': None, 'value': '\u20074/4\u2007[00:04&lt;00:00,\u2007\u20071.16s/it]'}}, '02962dce5fce40aca54eacea486abd4b': {'model_module': '@jupyter-widgets/base', 'model_module_version': '2.0.0', 'model_name': 'LayoutModel', 'state': {'_model_module': '@jupyter-widgets/base', '_model_module_version': '2.0.0', '_model_name': 'LayoutModel', '_view_count': None, '_view_module': '@jupyter-widgets/base', '_view_module_version': '2.0.0', '_view_name': 'LayoutView', 'align_content': None, 'align_items': None, 'align_self': None, 'border_bottom': None, 'border_left': None, 'border_right': None, 'border_top': None, 'bottom': None, 'display': None, 'flex': None, 'flex_flow': None, 'grid_area': None, 'grid_auto_columns': None, 'grid_auto_flow': None, 'grid_auto_rows': None, 'grid_column': None, 'grid_gap': None, 'grid_row': None, 'grid_template_areas': None, 'grid_template_columns': None, 'grid_template_rows': None, 'height': None, 'justify_content': None, 'justify_items': None, 'left': None, 'margin': None, 'max_height': None, 'max_width': None, 'min_height': None, 'min_width': None, 'object_fit': None, 'object_position': None, 'order': None, 'overflow': None, 'padding': None, 'right': None, 'top': None, 'visibility': None, 'width': None}}, '08336208e35b4d60bc076e80cce9dfe8': {'model_module': '@jupyter-widgets/base', 'model_module_version': '2.0.0', 'model_name': 'LayoutModel', 'state': {'_model_module': '@jupyter-widgets/base', '_model_module_version': '2.0.0', '_model_name': 'LayoutModel', '_view_count': None, '_view_module': '@jupyter-widgets/base', '_view_module_version': '2.0.0', '_view_name': 'LayoutView', 'align_content': None, 'align_items': None, 'align_self': None, 'border_bottom': None, 'border_left': None, 'border_right': None, 'border_top': None, 'bottom': None, 'display': None, 'flex': None, 'flex_flow': None, 'grid_area': None, 'grid_auto_columns': None, 'grid_auto_flow': None, 'grid_auto_rows': None, 'grid_column': None, 'grid_gap': None, 'grid_row': None, 'grid_template_areas': None, 'grid_template_columns': None, 'grid_template_rows': None, 'height': None, 'justify_content': None, 'justify_items': None, 'left': None, 'margin': None, 'max_height': None, 'max_width': None, 'min_height': None, 'min_width': None, 'object_fit': None, 'object_position': None, 'order': None, 'overflow': None, 'padding': None, 'right': None, 'top': None, 'visibility': None, 'width': None}}, '18b9a27b3e20494b99c9ac3bbc1d4df7': {'model_module': '@jupyter-widgets/controls', 'model_module_version': '2.0.0', 'model_name': 'HTMLModel', 'state': {'_dom_classes': [], '_model_module': '@jupyter-widgets/controls', '_model_module_version': '2.0.0', '_model_name': 'HTMLModel', '_view_count': None, '_view_module': '@jupyter-widgets/controls', '_view_module_version': '2.0.0', '_view_name': 'HTMLView', 'description': '', 'description_allow_html': False, 'layout': 'IPY_MODEL_5ea2f3b0a44d47c693a5f55a2cfcfac2', 'placeholder': '\u200b', 'style': 'IPY_MODEL_77f59f5dd5b445fdbf417811ca6bf2d4', 'tabbable': None, 'tooltip': None, 'value': 'mouse_0916N:\u2007100%'}}, '2586261739b847758e141361ab3383e9': {'model_module': '@jupyter-widgets/controls', 'model_module_version': '2.0.0', 'model_name': 'HTMLStyleModel', 'state': {'_model_module': '@jupyter-widgets/controls', '_model_module_version': '2.0.0', '_model_name': 'HTMLStyleModel', '_view_count': None, '_view_module': '@jupyter-widgets/base', '_view_module_version': '2.0.0', '_view_name': 'StyleView', 'background': None, 'description_width': '', 'font_size': None, 'text_color': None}}, '3121df2c0b2a42bdb1d0ff88714d036a': {'model_module': '@jupyter-widgets/controls', 'model_module_version': '2.0.0', 'model_name': 'ProgressStyleModel', 'state': {'_model_module': '@jupyter-widgets/controls', '_model_module_version': '2.0.0', '_model_name': 'ProgressStyleModel', '_view_count': None, '_view_module': '@jupyter-widgets/base', '_view_module_version': '2.0.0', '_view_name': 'StyleView', 'bar_color': None, 'description_width': ''}}, '3ba0342e68d64879914c4802edce22ba': {'model_module': '@jupyter-widgets/base', 'model_module_version': '2.0.0', 'model_name': 'LayoutModel', 'state': {'_model_module': '@jupyter-widgets/base', '_model_module_version': '2.0.0', '_model_name': 'LayoutModel', '_view_count': None, '_view_module': '@jupyter-widgets/base', '_view_module_version': '2.0.0', '_view_name': 'LayoutView', 'align_content': None, 'align_items': None, 'align_self': None, 'border_bottom': None, 'border_left': None, 'border_right': None, 'border_top': None, 'bottom': None, 'display': None, 'flex': None, 'flex_flow': None, 'grid_area': None, 'grid_auto_columns': None, 'grid_auto_flow': None, 'grid_auto_rows': None, 'grid_column': None, 'grid_gap': None, 'grid_row': None, 'grid_template_areas': None, 'grid_template_columns': None, 'grid_template_rows': None, 'height': None, 'justify_content': None, 'justify_items': None, 'left': None, 'margin': None, 'max_height': None, 'max_width': None, 'min_height': None, 'min_width': None, 'object_fit': None, 'object_position': None, 'order': None, 'overflow': None, 'padding': None, 'right': None, 'top': None, 'visibility': 'hidden', 'width': None}}, '41171d3b4dda4de6a75d83a071e49bf4': {'model_module': '@jupyter-widgets/base', 'model_module_version': '2.0.0', 'model_name': 'LayoutModel', 'state': {'_model_module': '@jupyter-widgets/base', '_model_module_version': '2.0.0', '_model_name': 'LayoutModel', '_view_count': None, '_view_module': '@jupyter-widgets/base', '_view_module_version': '2.0.0', '_view_name': 'LayoutView', 'align_content': None, 'align_items': None, 'align_self': None, 'border_bottom': None, 'border_left': None, 'border_right': None, 'border_top': None, 'bottom': None, 'display': None, 'flex': None, 'flex_flow': None, 'grid_area': None, 'grid_auto_columns': None, 'grid_auto_flow': None, 'grid_auto_rows': None, 'grid_column': None, 'grid_gap': None, 'grid_row': None, 'grid_template_areas': None, 'grid_template_columns': None, 'grid_template_rows': None, 'height': None, 'justify_content': None, 'justify_items': None, 'left': None, 'margin': None, 'max_height': None, 'max_width': None, 'min_height': None, 'min_width': None, 'object_fit': None, 'object_position': None, 'order': None, 'overflow': None, 'padding': None, 'right': None, 'top': None, 'visibility': 'hidden', 'width': None}}, '47d10e9fcaa7480fbbb1d1a2196752c5': {'model_module': '@jupyter-widgets/controls', 'model_module_version': '2.0.0', 'model_name': 'ProgressStyleModel', 'state': {'_model_module': '@jupyter-widgets/controls', '_model_module_version': '2.0.0', '_model_name': 'ProgressStyleModel', '_view_count': None, '_view_module': '@jupyter-widgets/base', '_view_module_version': '2.0.0', '_view_name': 'StyleView', 'bar_color': None, 'description_width': ''}}, '4d1de65d73f143eaa3291ebd361d2ba9': {'model_module': '@jupyter-widgets/controls', 'model_module_version': '2.0.0', 'model_name': 'HBoxModel', 'state': {'_dom_classes': [], '_model_module': '@jupyter-widgets/controls', '_model_module_version': '2.0.0', '_model_name': 'HBoxModel', '_view_count': None, '_view_module': '@jupyter-widgets/controls', '_view_module_version': '2.0.0', '_view_name': 'HBoxView', 'box_style': '', 'children': ['IPY_MODEL_01473061340e44579225b3bf95409a98', 'IPY_MODEL_964f5ec2ea23440cb217aca88b4673c6', 'IPY_MODEL_594b5d795b2942a7963877dfc652f237'], 'layout': 'IPY_MODEL_616be13145014c478e6e7aa031ec342f', 'tabbable': None, 'tooltip': None}}, '5088089135174637bdd15b6de2400d5c': {'model_module': '@jupyter-widgets/controls', 'model_module_version': '2.0.0', 'model_name': 'HBoxModel', 'state': {'_dom_classes': [], '_model_module': '@jupyter-widgets/controls', '_model_module_version': '2.0.0', '_model_name': 'HBoxModel', '_view_count': None, '_view_module': '@jupyter-widgets/controls', '_view_module_version': '2.0.0', '_view_name': 'HBoxView', 'box_style': '', 'children': ['IPY_MODEL_81f2894d693346349cdb9fa53c0e59b2', 'IPY_MODEL_723d6836d4b04f099bf83e5a226ba087', 'IPY_MODEL_757975239ea847588e4dfaf91d04c5ef'], 'layout': 'IPY_MODEL_3ba0342e68d64879914c4802edce22ba', 'tabbable': None, 'tooltip': None}}, '53a34a65669345cd877414f467372cc6': {'model_module': '@jupyter-widgets/base', 'model_module_version': '2.0.0', 'model_name': 'LayoutModel', 'state': {'_model_module': '@jupyter-widgets/base', '_model_module_version': '2.0.0', '_model_name': 'LayoutModel', '_view_count': None, '_view_module': '@jupyter-widgets/base', '_view_module_version': '2.0.0', '_view_name': 'LayoutView', 'align_content': None, 'align_items': None, 'align_self': None, 'border_bottom': None, 'border_left': None, 'border_right': None, 'border_top': None, 'bottom': None, 'display': None, 'flex': None, 'flex_flow': None, 'grid_area': None, 'grid_auto_columns': None, 'grid_auto_flow': None, 'grid_auto_rows': None, 'grid_column': None, 'grid_gap': None, 'grid_row': None, 'grid_template_areas': None, 'grid_template_columns': None, 'grid_template_rows': None, 'height': None, 'justify_content': None, 'justify_items': None, 'left': None, 'margin': None, 'max_height': None, 'max_width': None, 'min_height': None, 'min_width': None, 'object_fit': None, 'object_position': None, 'order': None, 'overflow': None, 'padding': None, 'right': None, 'top': None, 'visibility': None, 'width': None}}, '594b5d795b2942a7963877dfc652f237': {'model_module': '@jupyter-widgets/controls', 'model_module_version': '2.0.0', 'model_name': 'HTMLModel', 'state': {'_dom_classes': [], '_model_module': '@jupyter-widgets/controls', '_model_module_version': '2.0.0', '_model_name': 'HTMLModel', '_view_count': None, '_view_module': '@jupyter-widgets/controls', '_view_module_version': '2.0.0', '_view_name': 'HTMLView', 'description': '', 'description_allow_html': False, 'layout': 'IPY_MODEL_53a34a65669345cd877414f467372cc6', 'placeholder': '\u200b', 'style': 'IPY_MODEL_a6b9b0035ca54ef18477c879604704a2', 'tabbable': None, 'tooltip': None, 'value': '\u20076/6\u2007[00:06&lt;00:00,\u2007\u20071.35s/it]'}}, '5ea2f3b0a44d47c693a5f55a2cfcfac2': {'model_module': '@jupyter-widgets/base', 'model_module_version': '2.0.0', 'model_name': 'LayoutModel', 'state': {'_model_module': '@jupyter-widgets/base', '_model_module_version': '2.0.0', '_model_name': 'LayoutModel', '_view_count': None, '_view_module': '@jupyter-widgets/base', '_view_module_version': '2.0.0', '_view_name': 'LayoutView', 'align_content': None, 'align_items': None, 'align_self': None, 'border_bottom': None, 'border_left': None, 'border_right': None, 'border_top': None, 'bottom': None, 'display': None, 'flex': None, 'flex_flow': None, 'grid_area': None, 'grid_auto_columns': None, 'grid_auto_flow': None, 'grid_auto_rows': None, 'grid_column': None, 'grid_gap': None, 'grid_row': None, 'grid_template_areas': None, 'grid_template_columns': None, 'grid_template_rows': None, 'height': None, 'justify_content': None, 'justify_items': None, 'left': None, 'margin': None, 'max_height': None, 'max_width': None, 'min_height': None, 'min_width': None, 'object_fit': None, 'object_position': None, 'order': None, 'overflow': None, 'padding': None, 'right': None, 'top': None, 'visibility': None, 'width': None}}, '5fd53944fb2e423daa3d704ec885ce6a': {'model_module': '@jupyter-widgets/controls', 'model_module_version': '2.0.0', 'model_name': 'HTMLStyleModel', 'state': {'_model_module': '@jupyter-widgets/controls', '_model_module_version': '2.0.0', '_model_name': 'HTMLStyleModel', '_view_count': None, '_view_module': '@jupyter-widgets/base', '_view_module_version': '2.0.0', '_view_name': 'StyleView', 'background': None, 'description_width': '', 'font_size': None, 'text_color': None}}, '60b2f81e787b46e2ae97db3776a6965f': {'model_module': '@jupyter-widgets/base', 'model_module_version': '2.0.0', 'model_name': 'LayoutModel', 'state': {'_model_module': '@jupyter-widgets/base', '_model_module_version': '2.0.0', '_model_name': 'LayoutModel', '_view_count': None, '_view_module': '@jupyter-widgets/base', '_view_module_version': '2.0.0', '_view_name': 'LayoutView', 'align_content': None, 'align_items': None, 'align_self': None, 'border_bottom': None, 'border_left': None, 'border_right': None, 'border_top': None, 'bottom': None, 'display': None, 'flex': None, 'flex_flow': None, 'grid_area': None, 'grid_auto_columns': None, 'grid_auto_flow': None, 'grid_auto_rows': None, 'grid_column': None, 'grid_gap': None, 'grid_row': None, 'grid_template_areas': None, 'grid_template_columns': None, 'grid_template_rows': None, 'height': None, 'justify_content': None, 'justify_items': None, 'left': None, 'margin': None, 'max_height': None, 'max_width': None, 'min_height': None, 'min_width': None, 'object_fit': None, 'object_position': None, 'order': None, 'overflow': None, 'padding': None, 'right': None, 'top': None, 'visibility': 'hidden', 'width': None}}, '616be13145014c478e6e7aa031ec342f': {'model_module': '@jupyter-widgets/base', 'model_module_version': '2.0.0', 'model_name': 'LayoutModel', 'state': {'_model_module': '@jupyter-widgets/base', '_model_module_version': '2.0.0', '_model_name': 'LayoutModel', '_view_count': None, '_view_module': '@jupyter-widgets/base', '_view_module_version': '2.0.0', '_view_name': 'LayoutView', 'align_content': None, 'align_items': None, 'align_self': None, 'border_bottom': None, 'border_left': None, 'border_right': None, 'border_top': None, 'bottom': None, 'display': None, 'flex': None, 'flex_flow': None, 'grid_area': None, 'grid_auto_columns': None, 'grid_auto_flow': None, 'grid_auto_rows': None, 'grid_column': None, 'grid_gap': None, 'grid_row': None, 'grid_template_areas': None, 'grid_template_columns': None, 'grid_template_rows': None, 'height': None, 'justify_content': None, 'justify_items': None, 'left': None, 'margin': None, 'max_height': None, 'max_width': None, 'min_height': None, 'min_width': None, 'object_fit': None, 'object_position': None, 'order': None, 'overflow': None, 'padding': None, 'right': None, 'top': None, 'visibility': 'hidden', 'width': None}}, '723d6836d4b04f099bf83e5a226ba087': {'model_module': '@jupyter-widgets/controls', 'model_module_version': '2.0.0', 'model_name': 'FloatProgressModel', 'state': {'_dom_classes': [], '_model_module': '@jupyter-widgets/controls', '_model_module_version': '2.0.0', '_model_name': 'FloatProgressModel', '_view_count': None, '_view_module': '@jupyter-widgets/controls', '_view_module_version': '2.0.0', '_view_name': 'ProgressView', 'bar_style': '', 'description': '', 'description_allow_html': False, 'layout': 'IPY_MODEL_f2e0ffcc19b04c0286c56f8d8fcbf30f', 'max': 35.0, 'min': 0.0, 'orientation': 'horizontal', 'style': 'IPY_MODEL_3121df2c0b2a42bdb1d0ff88714d036a', 'tabbable': None, 'tooltip': None, 'value': 35.0}}, '757975239ea847588e4dfaf91d04c5ef': {'model_module': '@jupyter-widgets/controls', 'model_module_version': '2.0.0', 'model_name': 'HTMLModel', 'state': {'_dom_classes': [], '_model_module': '@jupyter-widgets/controls', '_model_module_version': '2.0.0', '_model_name': 'HTMLModel', '_view_count': None, '_view_module': '@jupyter-widgets/controls', '_view_module_version': '2.0.0', '_view_name': 'HTMLView', 'description': '', 'description_allow_html': False, 'layout': 'IPY_MODEL_08336208e35b4d60bc076e80cce9dfe8', 'placeholder': '\u200b', 'style': 'IPY_MODEL_a18d867aba3644568fd0427282cb84ee', 'tabbable': None, 'tooltip': None, 'value': '\u200733/35\u2007[00:15&lt;00:01,\u2007\u20071.74it/s]'}}, '77f59f5dd5b445fdbf417811ca6bf2d4': {'model_module': '@jupyter-widgets/controls', 'model_module_version': '2.0.0', 'model_name': 'HTMLStyleModel', 'state': {'_model_module': '@jupyter-widgets/controls', '_model_module_version': '2.0.0', '_model_name': 'HTMLStyleModel', '_view_count': None, '_view_module': '@jupyter-widgets/base', '_view_module_version': '2.0.0', '_view_name': 'StyleView', 'background': None, 'description_width': '', 'font_size': None, 'text_color': None}}, '78143a9588f84487a487d6a1bbe362b3': {'model_module': '@jupyter-widgets/controls', 'model_module_version': '2.0.0', 'model_name': 'HBoxModel', 'state': {'_dom_classes': [], '_model_module': '@jupyter-widgets/controls', '_model_module_version': '2.0.0', '_model_name': 'HBoxModel', '_view_count': None, '_view_module': '@jupyter-widgets/controls', '_view_module_version': '2.0.0', '_view_name': 'HBoxView', 'box_style': '', 'children': ['IPY_MODEL_ba845c766cef417f8aefe6b46f9c2cec', 'IPY_MODEL_b62b9f13c25843428eda96a5863f7a89', 'IPY_MODEL_b7abcb09299345988bab43bc7d67ed79'], 'layout': 'IPY_MODEL_60b2f81e787b46e2ae97db3776a6965f', 'tabbable': None, 'tooltip': None}}, '81f2894d693346349cdb9fa53c0e59b2': {'model_module': '@jupyter-widgets/controls', 'model_module_version': '2.0.0', 'model_name': 'HTMLModel', 'state': {'_dom_classes': [], '_model_module': '@jupyter-widgets/controls', '_model_module_version': '2.0.0', '_model_name': 'HTMLModel', '_view_count': None, '_view_module': '@jupyter-widgets/controls', '_view_module_version': '2.0.0', '_view_name': 'HTMLView', 'description': '', 'description_allow_html': False, 'layout': 'IPY_MODEL_ae3e53a5c76f4e17b5a039e2098e0ae1', 'placeholder': '\u200b', 'style': 'IPY_MODEL_dbcb5015f44e445fb18ac6eadcbb0a44', 'tabbable': None, 'tooltip': None, 'value': 'mouse_0322R:\u2007\u200794%'}}, '8a5eba1eb91c40b7a7b61ca9bdb6c838': {'model_module': '@jupyter-widgets/controls', 'model_module_version': '2.0.0', 'model_name': 'FloatProgressModel', 'state': {'_dom_classes': [], '_model_module': '@jupyter-widgets/controls', '_model_module_version': '2.0.0', '_model_name': 'FloatProgressModel', '_view_count': None, '_view_module': '@jupyter-widgets/controls', '_view_module_version': '2.0.0', '_view_name': 'ProgressView', 'bar_style': '', 'description': '', 'description_allow_html': False, 'layout': 'IPY_MODEL_b80bf537b58045febef29983c7e1ee39', 'max': 4.0, 'min': 0.0, 'orientation': 'horizontal', 'style': 'IPY_MODEL_47d10e9fcaa7480fbbb1d1a2196752c5', 'tabbable': None, 'tooltip': None, 'value': 4.0}}, '90b42eac7dac426a915845d8e30f0d29': {'model_module': '@jupyter-widgets/controls', 'model_module_version': '2.0.0', 'model_name': 'ProgressStyleModel', 'state': {'_model_module': '@jupyter-widgets/controls', '_model_module_version': '2.0.0', '_model_name': 'ProgressStyleModel', '_view_count': None, '_view_module': '@jupyter-widgets/base', '_view_module_version': '2.0.0', '_view_name': 'StyleView', 'bar_color': None, 'description_width': ''}}, '964f5ec2ea23440cb217aca88b4673c6': {'model_module': '@jupyter-widgets/controls', 'model_module_version': '2.0.0', 'model_name': 'FloatProgressModel', 'state': {'_dom_classes': [], '_model_module': '@jupyter-widgets/controls', '_model_module_version': '2.0.0', '_model_name': 'FloatProgressModel', '_view_count': None, '_view_module': '@jupyter-widgets/controls', '_view_module_version': '2.0.0', '_view_name': 'ProgressView', 'bar_style': '', 'description': '', 'description_allow_html': False, 'layout': 'IPY_MODEL_eb8b2507ac184c7394f48fdbd5242252', 'max': 6.0, 'min': 0.0, 'orientation': 'horizontal', 'style': 'IPY_MODEL_90b42eac7dac426a915845d8e30f0d29', 'tabbable': None, 'tooltip': None, 'value': 6.0}}, 'a18d867aba3644568fd0427282cb84ee': {'model_module': '@jupyter-widgets/controls', 'model_module_version': '2.0.0', 'model_name': 'HTMLStyleModel', 'state': {'_model_module': '@jupyter-widgets/controls', '_model_module_version': '2.0.0', '_model_name': 'HTMLStyleModel', '_view_count': None, '_view_module': '@jupyter-widgets/base', '_view_module_version': '2.0.0', '_view_name': 'StyleView', 'background': None, 'description_width': '', 'font_size': None, 'text_color': None}}, 'a6b9b0035ca54ef18477c879604704a2': {'model_module': '@jupyter-widgets/controls', 'model_module_version': '2.0.0', 'model_name': 'HTMLStyleModel', 'state': {'_model_module': '@jupyter-widgets/controls', '_model_module_version': '2.0.0', '_model_name': 'HTMLStyleModel', '_view_count': None, '_view_module': '@jupyter-widgets/base', '_view_module_version': '2.0.0', '_view_name': 'StyleView', 'background': None, 'description_width': '', 'font_size': None, 'text_color': None}}, 'ad118e2bc95149fabc8da8207cb6e78d': {'model_module': '@jupyter-widgets/controls', 'model_module_version': '2.0.0', 'model_name': 'ProgressStyleModel', 'state': {'_model_module': '@jupyter-widgets/controls', '_model_module_version': '2.0.0', '_model_name': 'ProgressStyleModel', '_view_count': None, '_view_module': '@jupyter-widgets/base', '_view_module_version': '2.0.0', '_view_name': 'StyleView', 'bar_color': None, 'description_width': ''}}, 'ae3e53a5c76f4e17b5a039e2098e0ae1': {'model_module': '@jupyter-widgets/base', 'model_module_version': '2.0.0', 'model_name': 'LayoutModel', 'state': {'_model_module': '@jupyter-widgets/base', '_model_module_version': '2.0.0', '_model_name': 'LayoutModel', '_view_count': None, '_view_module': '@jupyter-widgets/base', '_view_module_version': '2.0.0', '_view_name': 'LayoutView', 'align_content': None, 'align_items': None, 'align_self': None, 'border_bottom': None, 'border_left': None, 'border_right': None, 'border_top': None, 'bottom': None, 'display': None, 'flex': None, 'flex_flow': None, 'grid_area': None, 'grid_auto_columns': None, 'grid_auto_flow': None, 'grid_auto_rows': None, 'grid_column': None, 'grid_gap': None, 'grid_row': None, 'grid_template_areas': None, 'grid_template_columns': None, 'grid_template_rows': None, 'height': None, 'justify_content': None, 'justify_items': None, 'left': None, 'margin': None, 'max_height': None, 'max_width': None, 'min_height': None, 'min_width': None, 'object_fit': None, 'object_position': None, 'order': None, 'overflow': None, 'padding': None, 'right': None, 'top': None, 'visibility': None, 'width': None}}, 'b62b9f13c25843428eda96a5863f7a89': {'model_module': '@jupyter-widgets/controls', 'model_module_version': '2.0.0', 'model_name': 'FloatProgressModel', 'state': {'_dom_classes': [], '_model_module': '@jupyter-widgets/controls', '_model_module_version': '2.0.0', '_model_name': 'FloatProgressModel', '_view_count': None, '_view_module': '@jupyter-widgets/controls', '_view_module_version': '2.0.0', '_view_name': 'ProgressView', 'bar_style': '', 'description': '', 'description_allow_html': False, 'layout': 'IPY_MODEL_eaddf1f8df4340dca02428b484230ada', 'max': 36.0, 'min': 0.0, 'orientation': 'horizontal', 'style': 'IPY_MODEL_ad118e2bc95149fabc8da8207cb6e78d', 'tabbable': None, 'tooltip': None, 'value': 36.0}}, 'b7abcb09299345988bab43bc7d67ed79': {'model_module': '@jupyter-widgets/controls', 'model_module_version': '2.0.0', 'model_name': 'HTMLModel', 'state': {'_dom_classes': [], '_model_module': '@jupyter-widgets/controls', '_model_module_version': '2.0.0', '_model_name': 'HTMLModel', '_view_count': None, '_view_module': '@jupyter-widgets/controls', '_view_module_version': '2.0.0', '_view_name': 'HTMLView', 'description': '', 'description_allow_html': False, 'layout': 'IPY_MODEL_02962dce5fce40aca54eacea486abd4b', 'placeholder': '\u200b', 'style': 'IPY_MODEL_5fd53944fb2e423daa3d704ec885ce6a', 'tabbable': None, 'tooltip': None, 'value': '\u200736/36\u2007[00:57&lt;00:00,\u2007\u20072.01s/it]'}}, 'b80bf537b58045febef29983c7e1ee39': {'model_module': '@jupyter-widgets/base', 'model_module_version': '2.0.0', 'model_name': 'LayoutModel', 'state': {'_model_module': '@jupyter-widgets/base', '_model_module_version': '2.0.0', '_model_name': 'LayoutModel', '_view_count': None, '_view_module': '@jupyter-widgets/base', '_view_module_version': '2.0.0', '_view_name': 'LayoutView', 'align_content': None, 'align_items': None, 'align_self': None, 'border_bottom': None, 'border_left': None, 'border_right': None, 'border_top': None, 'bottom': None, 'display': None, 'flex': None, 'flex_flow': None, 'grid_area': None, 'grid_auto_columns': None, 'grid_auto_flow': None, 'grid_auto_rows': None, 'grid_column': None, 'grid_gap': None, 'grid_row': None, 'grid_template_areas': None, 'grid_template_columns': None, 'grid_template_rows': None, 'height': None, 'justify_content': None, 'justify_items': None, 'left': None, 'margin': None, 'max_height': None, 'max_width': None, 'min_height': None, 'min_width': None, 'object_fit': None, 'object_position': None, 'order': None, 'overflow': None, 'padding': None, 'right': None, 'top': None, 'visibility': None, 'width': None}}, 'ba845c766cef417f8aefe6b46f9c2cec': {'model_module': '@jupyter-widgets/controls', 'model_module_version': '2.0.0', 'model_name': 'HTMLModel', 'state': {'_dom_classes': [], '_model_module': '@jupyter-widgets/controls', '_model_module_version': '2.0.0', '_model_name': 'HTMLModel', '_view_count': None, '_view_module': '@jupyter-widgets/controls', '_view_module_version': '2.0.0', '_view_name': 'HTMLView', 'description': '', 'description_allow_html': False, 'layout': 'IPY_MODEL_da67fd5d4a9d48bea2b4ec20f220060f', 'placeholder': '\u200b', 'style': 'IPY_MODEL_c6df30f7e7c6420cbd52e04ee3d05f42', 'tabbable': None, 'tooltip': None, 'value': 'mouse_0403L:\u2007100%'}}, 'c44c420a781044e09c7a7da6310ebf3c': {'model_module': '@jupyter-widgets/base', 'model_module_version': '2.0.0', 'model_name': 'LayoutModel', 'state': {'_model_module': '@jupyter-widgets/base', '_model_module_version': '2.0.0', '_model_name': 'LayoutModel', '_view_count': None, '_view_module': '@jupyter-widgets/base', '_view_module_version': '2.0.0', '_view_name': 'LayoutView', 'align_content': None, 'align_items': None, 'align_self': None, 'border_bottom': None, 'border_left': None, 'border_right': None, 'border_top': None, 'bottom': None, 'display': None, 'flex': None, 'flex_flow': None, 'grid_area': None, 'grid_auto_columns': None, 'grid_auto_flow': None, 'grid_auto_rows': None, 'grid_column': None, 'grid_gap': None, 'grid_row': None, 'grid_template_areas': None, 'grid_template_columns': None, 'grid_template_rows': None, 'height': None, 'justify_content': None, 'justify_items': None, 'left': None, 'margin': None, 'max_height': None, 'max_width': None, 'min_height': None, 'min_width': None, 'object_fit': None, 'object_position': None, 'order': None, 'overflow': None, 'padding': None, 'right': None, 'top': None, 'visibility': None, 'width': None}}, 'c6df30f7e7c6420cbd52e04ee3d05f42': {'model_module': '@jupyter-widgets/controls', 'model_module_version': '2.0.0', 'model_name': 'HTMLStyleModel', 'state': {'_model_module': '@jupyter-widgets/controls', '_model_module_version': '2.0.0', '_model_name': 'HTMLStyleModel', '_view_count': None, '_view_module': '@jupyter-widgets/base', '_view_module_version': '2.0.0', '_view_name': 'StyleView', 'background': None, 'description_width': '', 'font_size': None, 'text_color': None}}, 'cf590bb2ba9c471d9fb5254e7132104d': {'model_module': '@jupyter-widgets/base', 'model_module_version': '2.0.0', 'model_name': 'LayoutModel', 'state': {'_model_module': '@jupyter-widgets/base', '_model_module_version': '2.0.0', '_model_name': 'LayoutModel', '_view_count': None, '_view_module': '@jupyter-widgets/base', '_view_module_version': '2.0.0', '_view_name': 'LayoutView', 'align_content': None, 'align_items': None, 'align_self': None, 'border_bottom': None, 'border_left': None, 'border_right': None, 'border_top': None, 'bottom': None, 'display': None, 'flex': None, 'flex_flow': None, 'grid_area': None, 'grid_auto_columns': None, 'grid_auto_flow': None, 'grid_auto_rows': None, 'grid_column': None, 'grid_gap': None, 'grid_row': None, 'grid_template_areas': None, 'grid_template_columns': None, 'grid_template_rows': None, 'height': None, 'justify_content': None, 'justify_items': None, 'left': None, 'margin': None, 'max_height': None, 'max_width': None, 'min_height': None, 'min_width': None, 'object_fit': None, 'object_position': None, 'order': None, 'overflow': None, 'padding': None, 'right': None, 'top': None, 'visibility': None, 'width': None}}, 'da67fd5d4a9d48bea2b4ec20f220060f': {'model_module': '@jupyter-widgets/base', 'model_module_version': '2.0.0', 'model_name': 'LayoutModel', 'state': {'_model_module': '@jupyter-widgets/base', '_model_module_version': '2.0.0', '_model_name': 'LayoutModel', '_view_count': None, '_view_module': '@jupyter-widgets/base', '_view_module_version': '2.0.0', '_view_name': 'LayoutView', 'align_content': None, 'align_items': None, 'align_self': None, 'border_bottom': None, 'border_left': None, 'border_right': None, 'border_top': None, 'bottom': None, 'display': None, 'flex': None, 'flex_flow': None, 'grid_area': None, 'grid_auto_columns': None, 'grid_auto_flow': None, 'grid_auto_rows': None, 'grid_column': None, 'grid_gap': None, 'grid_row': None, 'grid_template_areas': None, 'grid_template_columns': None, 'grid_template_rows': None, 'height': None, 'justify_content': None, 'justify_items': None, 'left': None, 'margin': None, 'max_height': None, 'max_width': None, 'min_height': None, 'min_width': None, 'object_fit': None, 'object_position': None, 'order': None, 'overflow': None, 'padding': None, 'right': None, 'top': None, 'visibility': None, 'width': None}}, 'dbcb5015f44e445fb18ac6eadcbb0a44': {'model_module': '@jupyter-widgets/controls', 'model_module_version': '2.0.0', 'model_name': 'HTMLStyleModel', 'state': {'_model_module': '@jupyter-widgets/controls', '_model_module_version': '2.0.0', '_model_name': 'HTMLStyleModel', '_view_count': None, '_view_module': '@jupyter-widgets/base', '_view_module_version': '2.0.0', '_view_name': 'StyleView', 'background': None, 'description_width': '', 'font_size': None, 'text_color': None}}, 'eaddf1f8df4340dca02428b484230ada': {'model_module': '@jupyter-widgets/base', 'model_module_version': '2.0.0', 'model_name': 'LayoutModel', 'state': {'_model_module': '@jupyter-widgets/base', '_model_module_version': '2.0.0', '_model_name': 'LayoutModel', '_view_count': None, '_view_module': '@jupyter-widgets/base', '_view_module_version': '2.0.0', '_view_name': 'LayoutView', 'align_content': None, 'align_items': None, 'align_self': None, 'border_bottom': None, 'border_left': None, 'border_right': None, 'border_top': None, 'bottom': None, 'display': None, 'flex': None, 'flex_flow': None, 'grid_area': None, 'grid_auto_columns': None, 'grid_auto_flow': None, 'grid_auto_rows': None, 'grid_column': None, 'grid_gap': None, 'grid_row': None, 'grid_template_areas': None, 'grid_template_columns': None, 'grid_template_rows': None, 'height': None, 'justify_content': None, 'justify_items': None, 'left': None, 'margin': None, 'max_height': None, 'max_width': None, 'min_height': None, 'min_width': None, 'object_fit': None, 'object_position': None, 'order': None, 'overflow': None, 'padding': None, 'right': None, 'top': None, 'visibility': None, 'width': None}}, 'eb8b2507ac184c7394f48fdbd5242252': {'model_module': '@jupyter-widgets/base', 'model_module_version': '2.0.0', 'model_name': 'LayoutModel', 'state': {'_model_module': '@jupyter-widgets/base', '_model_module_version': '2.0.0', '_model_name': 'LayoutModel', '_view_count': None, '_view_module': '@jupyter-widgets/base', '_view_module_version': '2.0.0', '_view_name': 'LayoutView', 'align_content': None, 'align_items': None, 'align_self': None, 'border_bottom': None, 'border_left': None, 'border_right': None, 'border_top': None, 'bottom': None, 'display': None, 'flex': None, 'flex_flow': None, 'grid_area': None, 'grid_auto_columns': None, 'grid_auto_flow': None, 'grid_auto_rows': None, 'grid_column': None, 'grid_gap': None, 'grid_row': None, 'grid_template_areas': None, 'grid_template_columns': None, 'grid_template_rows': None, 'height': None, 'justify_content': None, 'justify_items': None, 'left': None, 'margin': None, 'max_height': None, 'max_width': None, 'min_height': None, 'min_width': None, 'object_fit': None, 'object_position': None, 'order': None, 'overflow': None, 'padding': None, 'right': None, 'top': None, 'visibility': None, 'width': None}}, 'ee8a9abefa7942899d8a0218db0a6cc3': {'model_module': '@jupyter-widgets/controls', 'model_module_version': '2.0.0', 'model_name': 'HTMLStyleModel', 'state': {'_model_module': '@jupyter-widgets/controls', '_model_module_version': '2.0.0', '_model_name': 'HTMLStyleModel', '_view_count': None, '_view_module': '@jupyter-widgets/base', '_view_module_version': '2.0.0', '_view_name': 'StyleView', 'background': None, 'description_width': '', 'font_size': None, 'text_color': None}}, 'f2e0ffcc19b04c0286c56f8d8fcbf30f': {'model_module': '@jupyter-widgets/base', 'model_module_version': '2.0.0', 'model_name': 'LayoutModel', 'state': {'_model_module': '@jupyter-widgets/base', '_model_module_version': '2.0.0', '_model_name': 'LayoutModel', '_view_count': None, '_view_module': '@jupyter-widgets/base', '_view_module_version': '2.0.0', '_view_name': 'LayoutView', 'align_content': None, 'align_items': None, 'align_self': None, 'border_bottom': None, 'border_left': None, 'border_right': None, 'border_top': None, 'bottom': None, 'display': None, 'flex': None, 'flex_flow': None, 'grid_area': None, 'grid_auto_columns': None, 'grid_auto_flow': None, 'grid_auto_rows': None, 'grid_column': None, 'grid_gap': None, 'grid_row': None, 'grid_template_areas': None, 'grid_template_columns': None, 'grid_template_rows': None, 'height': None, 'justify_content': None, 'justify_items': None, 'left': None, 'margin': None, 'max_height': None, 'max_width': None, 'min_height': None, 'min_width': None, 'object_fit': None, 'object_position': None, 'order': None, 'overflow': None, 'padding': None, 'right': None, 'top': None, 'visibility': None, 'width': None}}, 'f992171351d44ed4b9d274894e966de8': {'model_module': '@jupyter-widgets/controls', 'model_module_version': '2.0.0', 'model_name': 'HBoxModel', 'state': {'_dom_classes': [], '_model_module': '@jupyter-widgets/controls', '_model_module_version': '2.0.0', '_model_name': 'HBoxModel', '_view_count': None, '_view_module': '@jupyter-widgets/controls', '_view_module_version': '2.0.0', '_view_name': 'HBoxView', 'box_style': '', 'children': ['IPY_MODEL_18b9a27b3e20494b99c9ac3bbc1d4df7', 'IPY_MODEL_8a5eba1eb91c40b7a7b61ca9bdb6c838', 'IPY_MODEL_0155efb81ac04a578aa2f458422e2119'], 'layout': 'IPY_MODEL_41171d3b4dda4de6a75d83a071e49bf4', 'tabbable': None, 'tooltip': None}}}, 'version_major': 2, 'version_minor': 0}}}

out = os.environ.get("NB_OUT") or str(Path(__file__).resolve().parent.parent / 'cursor_reconstruction.ipynb')
with open(out, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote", out, "cells:", len(cells))
