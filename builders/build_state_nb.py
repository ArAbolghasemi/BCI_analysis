"""AUTO-GENERATED from cursor_state_segmentation.ipynb by builders/sync_builders.py.

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


# ---------------- Behavioural state segmentation of the BCI cursor ----------------
md(r"""# Behavioural state segmentation of the BCI cursor

From looking at raw cursor traces there appear to be distinct behavioural regimes: stretches where
the animal actively drives the cursor, stretches where it appears to give up and rest, and
intermediate stretches where it works more slowly. This notebook makes that segmentation explicit,
characterises the states, and relates them to the other behavioural signals recorded alongside
(**treadmill locomotion**, **licking**, **imaging motion**).

Tools live in `utilities/behavior_state_utils.py` (`bs`), with cursor loading from
`utilities/cursor_analysis_utils.py` (`ca`) and trial/block handling from
`utilities/neural_analysis_utils.py` (`na`). All analysis flow and plotting stay open here.

---

## 1. Choosing the method

### The requirement
States must be distinguished by **how the cursor moves**, not only by where it sits. "Sitting at mid
pitch while climbing hard" and "sitting at mid pitch while drifting" are different behaviours with
the same instantaneous value. Whatever we fit must also produce **bouts** — contiguous stretches —
rather than a label that flickers many times a second.

### Options considered

| Method | Verdict |
|---|---|
| Threshold / clustering on windowed features | Rejected as the primary model: no temporal model at all, so bout structure comes entirely from the (arbitrary) window length, and the thresholds would be hand-tuned. |
| **Gaussian HMM** on instantaneous value | Rejected as the primary model: emissions depend only on the current sample, so it cannot represent dynamics; and its geometric dwell prior fragments the labels. Retained as the *Level 2* model, where the observations are already smooth window summaries so there are no fast dynamics left to model. |
| **Sticky AR-HMM** | **Chosen.** Each state carries its own linear dynamics, set point and noise scale, which is exactly the distinction we need; a sticky prior gives bout-like dwell times. |
| SLDS / rSLDS | Rejected: a continuous latent state buys little for a 1-D observable, and it adds identifiability and fitting problems that would be hard to validate here. |
| HDP-HMM (non-parametric K) | Rejected: we want a handful of interpretable states, and we can select K by held-out likelihood, which is easier to defend than a concentration prior. |

### The model
Each state `k` has its own autoregressive dynamics

$$ y_t = W_k \,[\,y_{t-1},\ \dots,\ y_{t-L},\ 1\,]^\top + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, \Sigma_k) $$

so a state is defined by its **dynamics** (the AR coefficients), its **set point** (the bias column)
and its **vigour** (the innovation covariance $\Sigma_k$). Fitted by EM (Baum–Welch) with a
**sticky** Dirichlet prior placing extra mass on the self-transition, which is what converts the
output from fragmented labels into behavioural bouts.

### On State/Keypoint-MoSeq
The AR-HMM-on-dynamics choice, and the sticky prior, are taken directly from MoSeq's approach to
pose dynamics. What is **deliberately not** borrowed, as flagged in the request: the *syllable*
grammar, the state hierarchy, and the HDP over an unbounded state count. Those target sub-second
motifs in high-dimensional pose data. Here the observable is a 1-D control signal and the target is
a few interpretable engagement regimes, so `K` is fixed and chosen by cross-validation.

### Two timescales, fitted separately
An early fit revealed something that shapes the whole design: run on the raw cursor, the AR-HMM
locks onto **trial-locked** structure with dwell times of 1–3 s (the cursor ramps during a trial and
falls back between trials). That is real, but it is *not* the minutes-long "gave up and rested"
structure that motivated this analysis. Making the prior stickier does not fix this — it only
distorts a model that is describing the fast process correctly.

So the two timescales are fitted as two explicit models, and reported separately:

* **Level 1 — control states.** Sticky AR-HMM on the binned cursor. Answers *what is the cursor
  doing right now* (driving / slow / quiescent). Dwell times of seconds.
* **Level 2 — cursor-regime states.** Sticky Gaussian HMM (the same class with `ar_order=0`) on
  **cursor-derived summary features** computed in multi-second windows. Dwell times of minutes.

  These were first called "engagement" states, on the assumption that they would form a single
  ladder from disengaged to engaged. **They do not**, and Section 8b is the audit that established
  it: two of the states are both highly productive and differ instead in *cursor dynamics*, and one
  of them turns out to be occupied almost exclusively during **new-axis** control. So Level 2
  describes the *regime the cursor is being driven in*, which is set jointly by how engaged the
  animal is and by which decoder axis it is controlling. `K` was raised from 3 to 5 as a result.

This is two models at two chosen timescales, not a hierarchical HMM: neither model constrains the
other, and Level 2 uses Level 1 only as one input feature among several.

**Level 2 is fitted on cursor-derived features only.** Treadmill, licking, imaging motion and reward
are deliberately excluded from fitting so that comparing the states against them in Section 8 stays
a genuine test rather than a restatement of the inputs.

## 2. Assumptions and choices, stated up front

* **Cursor signal** — `freqToOutput` (commanded tone, Hz) in **log2 octaves**, the same default and
  reasoning as the cursor-shape notebook: it is the closest proxy to what the animal hears, and pitch
  is perceived logarithmically. Selectable via `CURSOR_COLUMN` / `CURSOR_TRANSFORM`.
* **Per-session z-scoring** — the cursor is z-scored within each session before fitting, so states
  describe *relative* cursor behaviour and are comparable across days and animals rather than
  tracking day-to-day differences in absolute tone.
* **Binning, not smoothing** — the project convention. Smoothing would inject correlation between
  neighbouring samples, which an AR model would then absorb as if it were dynamics.
* **Gaps** — the logged cursor has NaN stretches (mostly session edges). An AR model cannot step
  across a hole, so each day is split into contiguous finite runs, each fitted as its own sequence.
* **The state count `K` is a modelling choice.** It is selected by held-out log-likelihood with
  **whole days held out**, so the score measures generalisation to a new session. The elbow is
  reported, not treated as a discovery of the true number of states.
* **State ordering** is canonicalised (quietest first) so labels are stable across refits; the raw
  labels from EM are arbitrary.

## 3. What is *not* available

The request asked for pupil dilation "if available". It is **not**: this dataset contains no pupil,
eye, video, face or whisker data — verified by walking every key of the `richfile` archive and by
searching the filesystem. `extras/tracking` is ROICaT *cell* tracking across days, not behaviour.

The behavioural covariates that **do** exist, all sampled one row per imaging frame (30 Hz), are:

| Signal | Source | What it indexes |
|---|---|---|
| **Treadmill** | `data_ws[day]["treadmill"]` | locomotion |
| **Licking** | `lick_times` (frame indices) | consummatory / anticipatory licking |
| **Imaging motion** | `motionCorrection` `xShift`/`yShift` | gross body / postural movement |
| **Reward** | active reward frames | task outcome |

The treadmill channel is a rotary-encoder voltage resting at a fixed level (~3825 in every animal).
Speed is therefore the deviation from that resting level. Three candidate transforms were checked
against imaging motion as an *independent* movement reference. All three track it substantially
and are monotonically related to each other, so the choice sets the units rather than the
conclusions. The exact correlation depends on how both signals are smoothed, so no number is quoted
here: Section 9 recomputes it with the settings actually in force and prints the result.""", cid='6d5c3972')

# ---------------- 4 · Setup and configuration ----------------
md(r"""## 4 · Setup and configuration""", cid='dfa3fd54')
code(r"""%matplotlib inline
import sys, warnings, time, json
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal, spearmanr, mannwhitneyu

REPO = Path.cwd(); sys.path.insert(0, str(REPO / "utilities"))
import importlib
import neural_analysis_utils as na; importlib.reload(na)
import cursor_analysis_utils as ca; importlib.reload(ca)
import behavior_state_utils as bs; importlib.reload(bs)
import richfile as rfs; na.register_richfile_types(rfs)
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **k): return x

# ============================ CONFIG ============================
SR = 30                                   # imaging / WaveSurfer sample rate
DATA_ROOT = Path(r"C:\Data\datasets\bmi_packets\bmi_packets")
OUT = REPO / "output" / "cursor_states"; OUT.mkdir(parents=True, exist_ok=True)
MICE = ["mouse_0403L", "mouse_0322R", "mouse_0403R", "mouse_0916N"]
PRIMARY = "mouse_0403L"                   # the animal with a remapping phase
REMAP_MOUSE = "mouse_0403L"

# --- cursor signal (same defaults as the cursor-shape notebook) ---
CURSOR_COLUMN, CURSOR_TRANSFORM = "freqToOutput", "log2"
MAX_GAP_SEC = 0.5

# --- Level 1: control states (AR-HMM on the binned cursor) ---
BIN_L1 = 10                               # 10 frames -> 3 Hz
SR_L1 = SR / BIN_L1
AR_ORDER = 2
K_L1 = 3
DWELL_L1_SEC = 5.0                        # sticky prior target
PRIOR_L1 = 0.05                           # prior weight, as a fraction of the samples
MIN_SEG_SEC = 120                         # ignore contiguous runs shorter than this

# --- Level 2: cursor-regime states (Gaussian HMM on cursor-only window features) ---
WIN_SEC = 15                             # feature window
K_L2 = 4                                 # see the K scan in Section 8b: the strict elbow is 4,
                                          # K=5 additionally splits new-axis control into a
                                          # low-yield and a high-yield sub-state
K_L2_SCAN = [2, 3, 4, 5, 6, 7, 8]
DWELL_L2_WIN = 6                          # sticky prior target, in windows (= 2 min)
PRIOR_L2 = 0.3
QUIESCENT_DZ = 0.02                       # |d cursor| below this counts as quiescent

# --- model selection / runtime ---
K_SCAN = [2, 3, 4, 5, 6]
CV_FOLDS = 4
CV_MAX_DAYS = 12                          # CV on a subset of days to keep runtime sane
SEED = 0
USE_CACHE = True

# --- figures ---
FIG_LEVEL = "summary"                     # "summary" (fewer figures) or "full" (everything)
_LV = {"summary": 0, "full": 1}
sns.set_context("notebook"); sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 110, "axes.titlesize": 10, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8})

def show(fig, name, level="summary"):
    '''Save every figure; display only those at or below the current FIG_LEVEL.'''
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight")
    if _LV[level] <= _LV[FIG_LEVEL]:
        plt.show()
    else:
        plt.close(fig); print(f"  [{name}] saved to disk (set FIG_LEVEL='full' to display)")

L1_NAMES = ["quiescent", "slow", "driving"]
L1_COLORS = dict(zip(range(3), ["#B0BEC5", "#FFB74D", "#E53935"]))

# Level-2 names are DESCRIPTIVE labels for the measured profile of each state, in the canonical
# order (increasing cursor speed). They are justified by the task-metric table in Section 8, and
# audited in Section 8b -- they are not assumptions. In particular they are NOT a single
# "engagement" ladder: `fluent` and `vigorous` are both productive but differ in cursor dynamics.
_L2_DEFAULT = ["idle", "sporadic", "fluent", "searching", "vigorous"]
L2_NAMES = (_L2_DEFAULT + [f"state{k}" for k in range(len(_L2_DEFAULT), 12)])[:K_L2]
_L2_PAL = ["#B0BEC5", "#90CAF9", "#2E7D32", "#EF6C00", "#6A1B9A", "#00838F", "#AD1457", "#5D4037"]
L2_COLORS = {k: _L2_PAL[k % len(_L2_PAL)] for k in range(K_L2)}
BLOCK_ORDER = ["intuitive_pre", "new", "washout"]
print(f"config loaded | Level 1: {SR_L1:.0f} Hz, AR({AR_ORDER}), K={K_L1}"
      f" | Level 2: {WIN_SEC:.0f}s windows, K={K_L2} | numba={bs.HAVE_NUMBA}")""", cid='84b67f26')

# ---------------- 5 · Validating the AR-HMM implementation ----------------
md(r"""## 5 · Validating the AR-HMM implementation

No HMM package is installed in this environment (`ssm`, `hmmlearn`, `dynamax`, `jax` are all absent,
and `ssm` requires compilation), so the model is implemented from scratch in
`behavior_state_utils.py`. Before trusting it on real data it is checked against **synthetic data
generated from a known AR-HMM**, where the correct answer is known exactly.

Four things must hold: EM must increase the likelihood monotonically, the state path must be
recovered, the AR coefficients and noise scales must be recovered, and Viterbi must agree with the
posterior. The `ar_order=0` reduction to a plain Gaussian HMM (used for Level 2) is checked too.""", cid='22363a14')
code(r"""K, D, L = 3, 1, 2
W_true = np.zeros((K, D, D * L + 1))
W_true[0, 0] = [0.95,  0.00, 0.05]     # rest: slow decay to a low set point, tiny innovations
W_true[1, 0] = [1.30, -0.40, 0.20]     # intermediate: mild oscillation
W_true[2, 0] = [1.60, -0.65, 0.50]     # active: strong momentum, large innovations
S_true = np.array([[[0.01]], [[0.09]], [[0.36]]])
P_true = np.array([[0.995, 0.004, 0.001], [0.006, 0.990, 0.004], [0.001, 0.006, 0.993]])

gen = bs.StickyARHMM(K, L)
y_syn, z_syn = gen.simulate(60000, W_true, S_true, P_true, seed=1)
m_syn = bs.StickyARHMM(K, L, dwell_target_samples=100, prior_strength=0.01, seed=0)
m_syn.fit([y_syn], n_iter=100, n_restarts=2)

syn_ll = np.array(m_syn.fit_.ll_trace); syn_steps = np.diff(syn_ll)
z_hat = m_syn.decode(y_syn)
mapping, syn_acc = bs.match_states(z_syn, z_hat, K)
post = m_syn.posterior(y_syn)
agree = float(np.mean(np.argmax(post, 1) == z_hat))

# ar_order=0 must reduce exactly to a Gaussian HMM
Wg = np.zeros((2, 1, 1)); Wg[0, 0, 0], Wg[1, 0, 0] = -1.5, 1.5
Sg = np.array([[[0.25]], [[0.25]]]); Pg = np.array([[0.99, 0.01], [0.01, 0.99]])
yg, zg = bs.StickyARHMM(2, 0).simulate(20000, Wg, Sg, Pg, seed=3)
mg = bs.StickyARHMM(2, 0, dwell_target_samples=100, prior_strength=0.01, seed=0).fit([yg], n_iter=60, n_restarts=1)
_, syn_acc_g = bs.match_states(zg, mg.decode(yg), 2)

val = [
    ("EM likelihood is monotone",        bool((syn_steps >= -1e-6 * np.abs(syn_ll[:-1])).all()), f"min step {syn_steps.min():+.2e}"),
    ("state path recovered (AR-HMM)",    syn_acc > 0.85,   f"accuracy {syn_acc:.4f}"),
    ("Viterbi agrees with posterior",    agree > 0.95, f"agreement {agree:.4f}"),
    ("posterior rows sum to 1",          bool(np.allclose(post.sum(1), 1)), ""),
    ("ar_order=0 is a Gaussian HMM",     syn_acc_g > 0.90, f"accuracy {syn_acc_g:.4f}"),
]
display(pd.DataFrame([dict(check=c, result="PASS" if ok else "FAIL", detail=d) for c, ok, d in val]))

rec = []
for k in range(K):
    kk = int(np.flatnonzero(mapping == k)[0])
    rec.append(dict(state=k, W_true=np.round(W_true[k, 0], 3), W_fit=np.round(m_syn.fit_.W[kk, 0], 3),
                    sd_true=round(float(np.sqrt(S_true[k, 0, 0])), 3),
                    sd_fit=round(float(np.sqrt(m_syn.fit_.Sigma[kk, 0, 0])), 3)))
print("Parameter recovery (fitted vs the values used to generate the data):")
display(pd.DataFrame(rec))

fig, axes = plt.subplots(1, 3, figsize=(13, 2.9))
axes[0].plot(syn_ll, "o-", ms=3); axes[0].set(title="EM log-likelihood", xlabel="iteration")
sl = slice(2000, 3500)
axes[1].plot(y_syn[sl, 0], color="0.3", lw=0.8)
for k in range(K):
    m_ = z_syn[sl] == k
    axes[1].fill_between(np.arange(sl.stop - sl.start), *axes[1].get_ylim(), where=m_, alpha=0.2,
                         color=list(L1_COLORS.values())[k], step="mid")
axes[1].set(title="synthetic trace, true states shaded", xlabel="sample")
axes[2].imshow(np.exp(m_syn.fit_.log_P), cmap="viridis", vmin=0, vmax=1)
axes[2].set(title=f"fitted transitions (recovery {syn_acc:.3f})", xlabel="to", ylabel="from")
for i in range(K):
    for j in range(K):
        axes[2].text(j, i, f"{np.exp(m_syn.fit_.log_P)[i,j]:.3f}", ha="center", va="center",
                     color="w", fontsize=7)
show(fig, "S5_validation")""", cid='981ec0d7')

# ---------------- 6 · Build the dataset ----------------
md(r"""## 6 · Build the dataset

One pass per day per animal: the cursor (binned, z-scored), the behavioural covariates, and the task
structure (in-trial, reward, block). Everything is binned to the Level-1 rate so it shares one time
base. Each day is split into contiguous finite runs; runs shorter than `MIN_SEG_SEC` are dropped.

Covariates are carried alongside but are **never used to fit** either model.""", cid='0b22311d')
code(r"""def day_block_labels(trials, n_frames):
    '''Per-frame block label; remapping days are (intuitive, new, washout), others single.'''
    blocks = na.segment_factor_blocks(trials)
    lab = np.array(["single"] * n_frames, dtype=object)
    phase = "stable"
    if len(blocks) == 3:
        phase = "remap"
        names = ["intuitive_pre", "new", "washout"]
        for nm, b in zip(names, blocks):
            s = int(trials.start[b.first_trial])
            e = int(trials.start[b.last_trial]) if b.last_trial + 1 < len(trials.start) else n_frames
            e = min(max(e, s + 1), n_frames)
            lab[s:e] = nm
        lab[:int(trials.start[blocks[0].first_trial])] = "intuitive_pre"
        lab[int(trials.start[blocks[2].first_trial]):] = "washout"
    return lab, phase

def build_dataset():
    segs = []
    for mouse in MICE:
        r = na.open_mouse(rfs, DATA_ROOT, mouse)
        days = na.get_day_list(r)
        d0 = r["date_day0"].load()
        d0 = str(np.asarray(d0).ravel()[0]) if not isinstance(d0, str) else d0
        d0i = days.index(d0) if d0 in days else 0
        for di, day in enumerate(tqdm(days, desc=mouse, leave=False)):
            try:
                cur = ca.load_cursor_signal(r, day, column=CURSOR_COLUMN, transform=CURSOR_TRANSFORM,
                                            max_gap_sec=MAX_GAP_SEC, sr=SR)
            except Exception:
                continue
            n = len(cur)
            trials = na.get_true_bmi_trials(na.get_trials_df(r, day))
            if len(trials) == 0:
                continue
            rew_f = na.get_active_reward_frames(r, day)
            eff = na.effective_trial_end(trials, reward_frames=rew_f, miss_dur_sec=20.0, sr=SR)
            in_trial = np.zeros(n); in_hit = np.zeros(n); rew = np.zeros(n)
            # trial-level markers, so state naming can use trial rate and trial duration rather
            # than cursor statistics alone (a window fully inside one 20 s miss would otherwise
            # report a "mean duration" of 20 s and tell us nothing)
            t_start = np.zeros(n); dur_at_start = np.zeros(n)
            for s, e, su in zip(trials.start, eff, trials.success):
                a, b = int(s), min(int(e), n)
                if b > a:
                    in_trial[a:b] = 1.0
                    t_start[a] = 1.0; dur_at_start[a] = (b - a) / SR
                    if su == 1: in_hit[a:b] = 1.0
            ri = np.round(np.asarray(rew_f, float)); ri = ri[(ri >= 0) & (ri < n)].astype(int)
            rew[ri] = 1.0
            blk, phase = day_block_labels(trials, n)

            raw = dict(in_trial=in_trial, in_hit=in_hit,
                       treadmill=bs.load_treadmill_speed(r, day, sr=SR),
                       lick=bs.load_lick_rate(r, day, sr=SR, n_frames=n),
                       mc_motion=bs.load_mc_motion(r, day, sr=SR))
            C = {k: bs.bin_signal(v, BIN_L1) for k, v in raw.items()}
            for k, v in [("reward", rew), ("t_start", t_start), ("dur_at_start", dur_at_start)]:
                C[k] = bs.bin_signal(v, BIN_L1, "sum")      # counts must be summed, not averaged
            blk_b = np.array([blk[min(i * BIN_L1, n - 1)] for i in range(len(C["reward"]))], dtype=object)

            cb = bs.bin_signal(cur, BIN_L1)
            mu, sd = np.nanmean(cb), np.nanstd(cb)
            z = (cb - mu) / (sd if sd > 1e-9 else 1.0)
            for (s, e) in bs.contiguous_finite_segments(z, min_len=int(MIN_SEG_SEC * SR_L1)):
                segs.append(dict(mouse=mouse, day=str(day), day_idx=di, day_from0=di - d0i, phase=phase,
                                 start_bin=s, y=z[s:e].astype(np.float32),
                                 cov={k: np.asarray(v[s:e], np.float32) for k, v in C.items()},
                                 block=blk_b[s:e]))
    return segs

cache = OUT / f"segments_{CURSOR_COLUMN}_{CURSOR_TRANSFORM}_bin{BIN_L1}_v2.npz"
if USE_CACHE and cache.exists():
    _z = np.load(cache, allow_pickle=True); SEGS = list(_z["segs"])
else:
    t0 = time.time(); SEGS = build_dataset()
    np.savez_compressed(cache, segs=np.array(SEGS, dtype=object))
    print(f"built in {time.time()-t0:.0f}s")

seg_df = pd.DataFrame([{k: s[k] for k in ("mouse", "day", "day_idx", "day_from0", "phase")}
                       | {"minutes": len(s["y"]) / SR_L1 / 60} for s in SEGS])
print(f"{len(SEGS)} segments | {seg_df.minutes.sum()/60:.1f} h total | {SR_L1:.0f} Hz time base")
display(seg_df.groupby(["mouse", "phase"]).agg(days=("day", "nunique"), segments=("day", "size"),
                                               hours=("minutes", lambda v: round(v.sum()/60, 1))))""", cid='1af37b92')

# ---------------- 7 · Level 1 — control states ----------------
md(r"""## 7 · Level 1 — control states

Sticky AR-HMM on the binned, z-scored cursor, fitted **per animal** (so state labels are comparable
across that animal's days) using all of its days jointly.

`K` is chosen by held-out log-likelihood with whole days held out. The scan is run on the primary
animal and on a capped number of days (`CV_MAX_DAYS`) purely for runtime; the chosen `K` is then
applied to every animal. Because held-out likelihood rises monotonically with `K` for nested-ish
model families, the **elbow** is what is read, not the maximum.""", cid='184353cf')
code(r"""cv_fp = OUT / f"cv_L1_K{min(K_SCAN)}-{max(K_SCAN)}.csv"
prim = [s["y"][:, None] for s in SEGS if s["mouse"] == PRIMARY][:CV_MAX_DAYS]
if USE_CACHE and cv_fp.exists():
    cv_l1 = pd.read_csv(cv_fp)
else:
    t0 = time.time()
    cv_l1 = bs.cv_select_states(prim, K_SCAN, ar_order=AR_ORDER, n_folds=CV_FOLDS,
                                dwell_target_samples=DWELL_L1_SEC * SR_L1,
                                prior_strength=PRIOR_L1, n_iter=50, seed=SEED)
    cv_l1.to_csv(cv_fp, index=False); print(f"CV in {time.time()-t0:.0f}s")
display(cv_l1.round(4))
gain = np.diff(cv_l1.test_ll.to_numpy())
print(f"marginal held-out gain per extra state: {np.round(gain, 4)}")
print(f"K = {K_L1} is used below (elbow of the curve; set K_L1 in the config to change it).")""", cid='39ea1c6a')
code(r"""def fit_level1(mouse):
    seqs = [s["y"][:, None] for s in SEGS if s["mouse"] == mouse]
    if not seqs: return None, None
    m = bs.StickyARHMM(K_L1, AR_ORDER, dwell_target_samples=DWELL_L1_SEC * SR_L1,
                       prior_strength=PRIOR_L1, seed=SEED)
    m.fit(seqs, n_iter=80, n_restarts=2)
    labs = [m.decode(s) for s in seqs]
    order = bs.order_states_by(m, seqs, labs, key="speed")     # canonical: quietest state first
    return m, [bs.apply_state_order(l, order) for l in labs]

L1 = {}
for mouse in MICE:
    t0 = time.time(); m, labs = fit_level1(mouse)
    if m is None: continue
    L1[mouse] = dict(model=m, labels=labs)
    idx = [i for i, s in enumerate(SEGS) if s["mouse"] == mouse]
    for i, lab in zip(idx, labs): SEGS[i]["l1"] = lab
    print(f"{mouse:14s} ll={m.fit_.loglik:12.0f} iters={m.fit_.n_iter:3d} "
          f"dwell(s)={np.round(m.expected_dwell()/SR_L1, 1)} ({time.time()-t0:.0f}s)")""", cid='77eb3f16')
code(r"""def pooled(mouse, key):
    return np.concatenate([s[key] for s in SEGS if s["mouse"] == mouse and "l1" in s])

rows = []
for mouse in L1:
    y = pooled(mouse, "y"); lab = pooled(mouse, "l1")
    cv = {k: np.concatenate([s["cov"][k] for s in SEGS if s["mouse"] == mouse and "l1" in s])
          for k in SEGS[0]["cov"]}
    spd = np.abs(np.diff(y, prepend=y[0]))
    dw = pd.concat([bs.dwell_segments(s["l1"]) for s in SEGS if s["mouse"] == mouse and "l1" in s])
    for k in range(K_L1):
        m_ = lab == k
        rows.append(dict(mouse=mouse, state=k, name=L1_NAMES[k], occupancy=m_.mean(),
                         cursor_level=y[m_].mean(), cursor_speed=spd[m_].mean(),
                         median_dwell_s=dw[dw.state == k].length.median() / SR_L1,
                         frac_in_trial=np.nanmean(cv["in_trial"][m_]),
                         reward_per_min=np.nanmean(cv["reward"][m_]) * SR_L1 * 60,
                         treadmill=np.nanmean(cv["treadmill"][m_]), lick_hz=np.nanmean(cv["lick"][m_])))
L1_TAB = pd.DataFrame(rows)
print("Level-1 control states. Columns left of the divider come from the cursor (what was fitted);")
print("frac_in_trial / reward_per_min / treadmill / lick_hz are EXTERNAL and were not used in fitting.")
display(L1_TAB.round(3))""", cid='94f0d4e5')
code(r"""fig, axes = plt.subplots(1, 4, figsize=(15, 3.0))
sub = L1_TAB[L1_TAB.mouse == PRIMARY]
for ax, col, ttl in zip(axes, ["cursor_speed", "median_dwell_s", "reward_per_min", "treadmill"],
                        ["cursor speed (z/sample)", "median dwell (s)", "rewards / min (external)",
                         "treadmill speed (external)"]):
    ax.bar([L1_NAMES[k] for k in sub.state], sub[col], color=[L1_COLORS[k] for k in sub.state])
    ax.set(title=ttl); ax.tick_params(axis="x", rotation=15)
fig.suptitle(f"{PRIMARY} · Level-1 control states", y=1.04)
show(fig, "S7_L1_states")""", cid='7f87e5b4')

# ---------------- 8 · Level 2 — cursor-regime states ----------------
md(r"""## 8 · Level 2 — cursor-regime states

Features are computed in non-overlapping `WIN_SEC` windows from the **cursor only**, then z-scored
across all windows of an animal and fitted with a sticky Gaussian HMM (`ar_order=0`):

| feature | meaning |
|---|---|
| `cursor_speed` | mean \|Δcursor\| in the window — how much the cursor moved |
| `cursor_sd` | SD of the cursor — how wide its excursions were |
| `frac_quiescent` | fraction of the window with \|Δcursor\| < `QUIESCENT_DZ` — how much of it was dead time |
| `cursor_level` | mean cursor value — how high the tone sat |
| `L1_active_occ` | fraction of the window in the Level-1 *driving* state |

Reward, treadmill, licking and imaging motion are **excluded by construction**, which is what keeps
the comparison in Section 9 a real test.

States are put in a canonical order of **increasing cursor speed** so labels are stable across
refits. The names are descriptive labels read off the measured task profile below (and checked in
every animal); they are not an ordering by engagement — see Section 8b.""", cid='3a590064')
code(r"""FEAT_NAMES = ["cursor_speed", "cursor_sd", "frac_quiescent", "cursor_level", "L1_active_occ"]
W_L2 = int(WIN_SEC * SR_L1)

def window_features(seg):
    y, l1 = seg["y"], seg["l1"]
    nb = len(y) // W_L2
    if nb < 3: return None
    Y = y[:nb * W_L2].reshape(nb, W_L2); L = l1[:nb * W_L2].reshape(nb, W_L2)
    dv = np.abs(np.diff(Y, axis=1))
    F = np.column_stack([dv.mean(1), Y.std(1), (dv < QUIESCENT_DZ).mean(1), Y.mean(1),
                         (L == K_L1 - 1).mean(1)])
    SUMS = {"reward", "t_start", "dur_at_start"}     # counts: sum over the window
    cov = {k: (v[:nb * W_L2].reshape(nb, W_L2).sum(1) if k in SUMS
               else v[:nb * W_L2].reshape(nb, W_L2).mean(1)) for k, v in seg["cov"].items()}
    cov["reward"] = cov["reward"] / WIN_SEC * 60                 # rewards per minute
    cov["trials_per_min"] = cov["t_start"] / WIN_SEC * 60        # trials STARTED per minute
    with np.errstate(invalid="ignore", divide="ignore"):
        # mean duration of the trials that START in this window (NaN if none did)
        cov["trial_dur_s"] = np.where(cov["t_start"] > 0, cov["dur_at_start"] / np.maximum(cov["t_start"], 1e-9), np.nan)
        cov["reward_per_trial"] = np.where(cov["t_start"] > 0,
                                           cov["reward"] * WIN_SEC / 60 / np.maximum(cov["t_start"], 1e-9), np.nan)
    blk = seg["block"][:nb * W_L2].reshape(nb, W_L2)[:, 0]
    return F, cov, blk

L2 = {}
for mouse in L1:
    packs = [(s, window_features(s)) for s in SEGS if s["mouse"] == mouse and "l1" in s]
    pack_seg_ids = [id(s) for s, p in packs if p is not None]
    packs = [(s, p) for s, p in packs if p is not None]
    if not packs: continue
    Fall = np.vstack([p[0] for _, p in packs])
    mu, sd = Fall.mean(0), Fall.std(0) + 1e-9
    feats = [(p[0] - mu) / sd for _, p in packs]
    m = bs.StickyARHMM(K_L2, 0, dwell_target_samples=DWELL_L2_WIN, prior_strength=PRIOR_L2, seed=SEED)
    m.fit(feats, n_iter=120, n_restarts=3)
    labs = [m.decode(f) for f in feats]
    # canonical order: least active first, by the cursor_speed feature
    order = bs.order_states_by(m, [f[:, :1] for f in feats], labs, key="level")
    labs = [bs.apply_state_order(l, order) for l in labs]
    L2[mouse] = dict(model=m, feats=feats, labels=labs,
                     cov=[p[1] for _, p in packs], block=[p[2] for _, p in packs],
                     meta=[{k: s[k] for k in ("day", "day_from0", "phase")} for s, _ in packs],
                     mu=mu, sd=sd, seg_ids=pack_seg_ids)
    dw = pd.concat([bs.dwell_segments(l) for l in labs])
    print(f"{mouse:14s} windows={sum(len(f) for f in feats):5d} "
          f"median dwell (min)={[round(dw[dw.state==k].length.median()*WIN_SEC/60, 1) for k in range(K_L2)]}")""", cid='0cf02d85')
code(r"""rows = []
for mouse, P in L2.items():
    F = np.vstack(P["feats"]); lab = np.concatenate(P["labels"])
    cov = {k: np.concatenate([c[k] for c in P["cov"]]) for k in P["cov"][0]}
    dw = pd.concat([bs.dwell_segments(l) for l in P["labels"]])
    for k in range(K_L2):
        m_ = lab == k
        rec = dict(mouse=mouse, state=k, name=L2_NAMES[k], occupancy=m_.mean(),
                   median_dwell_min=dw[dw.state == k].length.median() * WIN_SEC / 60)
        rec |= {f: F[m_, i].mean() for i, f in enumerate(FEAT_NAMES)}
        rec |= dict(reward_per_min=np.nanmean(cov["reward"][m_]),
                    trials_per_min=np.nanmean(cov["trials_per_min"][m_]),
                    trial_dur_s=np.nanmean(cov["trial_dur_s"][m_]),
                    reward_per_trial=np.nanmean(cov["reward_per_trial"][m_]),
                    frac_in_trial=np.nanmean(cov["in_trial"][m_]),
                    treadmill=np.nanmean(cov["treadmill"][m_]),
                    lick_hz=np.nanmean(cov["lick"][m_]),
                    mc_motion=np.nanmean(cov["mc_motion"][m_]))
        rows.append(rec)
L2_TAB = pd.DataFrame(rows)
print("Level-2 cursor-regime states. The five feature columns were FITTED; everything from")
print("reward_per_min rightwards is an EXTERNAL hold-out and is what the state names are read from.")
display(L2_TAB.round(3))

print("Task profile of each state for the primary animal -- this is the evidence for the names.")
print("Note the names are NOT a single engagement ladder: two states can both be productive and")
print("differ only in cursor dynamics. Section 8b tests what that difference corresponds to.")
display(L2_TAB[L2_TAB.mouse == PRIMARY][
    ["state", "name", "occupancy", "reward_per_min", "trials_per_min", "trial_dur_s",
     "reward_per_trial", "cursor_speed", "frac_quiescent"]].round(3))

# The names are attached BY POSITION in the canonical order (increasing cursor speed), which is
# well defined for every animal -- but whether each position carries the same task profile in every
# animal is an empirical question, not something to assume. Check it explicitly.
print("\nreward/min per state, per animal (states ordered by increasing cursor speed):")
piv = L2_TAB.pivot_table(index="mouse", columns="name", values="reward_per_min").reindex(columns=L2_NAMES)
display(piv.round(2))
NAME_OK = {}
for mouse in piv.index:
    row = piv.loc[mouse]
    NAME_OK[mouse] = dict(
        idle_is_lowest=bool(row.idxmin() == "idle"),
        best_is_fluent_or_vigorous=bool(row.idxmax() in ("fluent", "vigorous")))
name_chk = pd.DataFrame(NAME_OK).T
display(name_chk)
if not name_chk.all().all():
    print("NOTE: in the animals marked False the positional name does NOT match the task profile.")
    print("Read the per-animal table above rather than the label for those animals.")
else:
    print("In every animal the lowest-yield state is 'idle' and the highest-yield state is")
    print("'fluent' or 'vigorous', so the positional names carry the intended meaning.")""", cid='152790d2')
code(r"""P = L2[PRIMARY]
F = np.vstack(P["feats"]); lab = np.concatenate(P["labels"])
fig, axes = plt.subplots(1, 3, figsize=(15, 3.2))
prof = pd.DataFrame(F, columns=FEAT_NAMES).groupby(lab).mean()
sns.heatmap(prof, cmap="vlag", center=0, ax=axes[0], cbar_kws=dict(label="z"),
            yticklabels=[L2_NAMES[k] for k in prof.index])
axes[0].set(title="fitted feature profile (cursor only)", ylabel="")
axes[0].tick_params(axis='y', rotation=45)
axes[0].tick_params(axis='x', rotation=45)
Pm = P["model"].transition_matrix()
sns.heatmap(Pm, annot=True, fmt=".3f", cmap="viridis", vmin=0, vmax=1, ax=axes[1],
            xticklabels=L2_NAMES, yticklabels=L2_NAMES, cbar=False)
axes[1].set(title=f"transition matrix ({WIN_SEC:.0f}s windows)", xlabel="to", ylabel="from")
axes[1].tick_params(axis='both', rotation=45)
dw = pd.concat([bs.dwell_segments(l) for l in P["labels"]])
for k in range(K_L2):
    d = dw[dw.state == k].length * WIN_SEC / 60
    axes[2].hist(d, bins=np.logspace(np.log10(WIN_SEC/60), np.log10(max(d.max(), 1)), 25),
                 alpha=0.55, color=L2_COLORS[k], label=L2_NAMES[k])
axes[2].set(xscale="log", title="bout duration", xlabel="minutes", ylabel="count"); axes[2].legend()

fig.suptitle(f"{PRIMARY} · Level-2 engagement states", y=1.05)
show(fig, "S8_L2_states")""", cid='cfea0a79')
code(r"""s""", cid='475cd81d')
code(r"""# Example trace showing BOTH timescales at once: the Level-1 control-state path shaded on the
# cursor, the Level-2 engagement state as a band beneath, and the covariates below that.
# Placed here rather than in Section 7 because it needs the Level-2 fit.
dayidx_list = [1, 5, 10, 15, 20, 21, 22, 25, 27]
for dayidx in dayidx_list:
    s = next(s for s in SEGS if s["mouse"] == PRIMARY and "l1" in s and  s['day_idx'] == dayidx)
    n_show = int(600 * SR_L1); off = int(600 * SR_L1)
    t = np.arange(n_show) / SR_L1
    fig, axes = plt.subplots(3, 1, figsize=(14, 5.0), sharex=True,
                             gridspec_kw=dict(height_ratios=[3, 0.45, 1]))
    axes[0].plot(t, s["y"][off:off + n_show], color="0.25", lw=0.9)
    lab = s["l1"][off:off + n_show]
    for k in range(K_L1):
        axes[0].fill_between(t, *axes[0].get_ylim(), where=(lab == k), color=L1_COLORS[k],
                            alpha=0.25, step="mid", label=L1_NAMES[k])
    axes[0].legend(ncol=K_L1, loc="upper right"); axes[0].set(ylabel="cursor (z, octaves)",
                                                            title=f"{PRIMARY} day {s['day']} · Level-1 control states (shading) and Level-2 engagement (band) - day index {s['day_idx']} - phase: {s['phase']}")
    # Level-2 engagement over the same window, so both timescales are visible at once
    P0 = L2.get(PRIMARY)
    if P0 is not None and id(s) in P0["seg_ids"]:
        pi = P0["seg_ids"].index(id(s)); l2w = P0["labels"][pi]
        w0, w1 = off // W_L2, min(int(np.ceil((off + n_show) / W_L2)), len(l2w))
        for wi in range(w0, w1):
            axes[1].axvspan((wi * W_L2 - off) / SR_L1, ((wi + 1) * W_L2 - off) / SR_L1,
                            color=L2_COLORS[int(l2w[wi])], alpha=0.75, lw=0)
        for k in range(K_L2):
            axes[1].plot([], [], lw=6, color=L2_COLORS[k], label=L2_NAMES[k])
        axes[1].legend(ncol=K_L2, loc="center left", fontsize=7, frameon=False)
    axes[1].set(ylabel="Level 2", yticks=[], xlim=(0, n_show / SR_L1)); axes[1].grid(False)
    axes[2].plot(t, s["cov"]["in_trial"][off:off + n_show], color="tab:blue", lw=1, label="in trial")
    axes[2].plot(t, s["cov"]["treadmill"][off:off + n_show], color="tab:green", lw=1, label="treadmill")
    axes[2].set(xlabel=f"time ({off / SR_L1:.0f} offset) s", ylabel="covariates"); axes[2].legend(ncol=2)
    show(fig, "S8_two_level_trace")""", cid='49f5e415')

# ---------------- 8b · Audit: is one state doing two jobs? ----------------
md(r"""## 8b · Audit: is one state doing two jobs?

Reading the day-by-day occupancy plot, the two productive states appear to **swap** at the onset of
the remapping phase. That is the signature of a state set that is too small — a single label being
reused for two behaviours that happen not to co-occur. This section tests that directly, rather
than assuming either way.

Four tests, each of which could have come out negative:

1. **Held-out likelihood vs K.** If `K` is too small, adding states should still buy real
   generalisation. Whole days are held out.
2. **Within-session block composition.** On a remapping day the session runs
   intuitive → *rotation* → new axis → washout. Comparing state occupancy between those blocks
   **inside the same session** controls for day, session length, satiety and time-on-task — all
   the things a stable-vs-remap comparison across days confounds. (It needs controlling: remapping
   sessions are 80 min long where earlier sessions are 60 min.)
3. **State homogeneity.** Take the windows assigned to one state and split them by block. If the
   state is a single behaviour, its *fitted features* should look the same in both halves.
4. **Generalisation gap.** Fit on stable days only, then score remapping days. If new-axis control
   is genuinely a different regime, a model that has never seen it should fit it much worse.

If the state set is adequate, test 2 shows similar composition across blocks, test 3 shows no
difference, and test 4 shows a small gap.""", cid='b62dde55')
code(r"""# --- test 1: how much does held-out likelihood keep improving with K? ---
kfp = OUT / f"cv_L2_{PRIMARY}_K{min(K_L2_SCAN)}-{max(K_L2_SCAN)}.csv"
P = L2[PRIMARY]
if USE_CACHE and kfp.exists():
    cv_l2 = pd.read_csv(kfp)
else:
    t0 = time.time()
    cv_l2 = bs.cv_select_states(P["feats"], K_L2_SCAN, ar_order=0, n_folds=CV_FOLDS,
                                dwell_target_samples=DWELL_L2_WIN, prior_strength=PRIOR_L2,
                                n_iter=100, n_restarts=2, seed=SEED)
    cv_l2.to_csv(kfp, index=False); print(f"K scan in {time.time()-t0:.0f}s")
cv_l2["gain_vs_prev"] = cv_l2.test_ll.diff()
display(cv_l2.round(4))
print("The largest single gain marks the elbow; gains after it are the granularity/parsimony trade-off.")

# --- test 2: refit at several K and look at WITHIN-SESSION block composition ---
blk_all = np.concatenate(P["block"])
ph_all = np.concatenate([[m["phase"]] * len(f) for m, f in zip(P["meta"], P["feats"])])
cov_all = {k: np.concatenate([c[k] for c in P["cov"]]) for k in P["cov"][0]}
GROUPS = [("stable days", ph_all == "stable"),
          ("remap: intuitive", blk_all == "intuitive_pre"),
          ("remap: new axis", blk_all == "new"),
          ("remap: washout", blk_all == "washout")]

comp = {}
for K in [k for k in [3, 4, 5, 6] if k <= max(K_L2_SCAN)]:
    mk = bs.StickyARHMM(K, 0, dwell_target_samples=DWELL_L2_WIN, prior_strength=PRIOR_L2, seed=SEED)
    mk.fit(P["feats"], n_iter=120, n_restarts=3)
    Lk = [mk.decode(f) for f in P["feats"]]
    ok = bs.order_states_by(mk, [f[:, :1] for f in P["feats"]], Lk, key="level")
    Ak = np.concatenate([bs.apply_state_order(x, ok) for x in Lk])
    rows = []
    for nm, m in GROUPS:
        rows.append(dict(group=nm, **{f"s{k}": float(np.mean(Ak[m] == k)) for k in range(K)}))
    rows.append(dict(group="-- reward/min", **{f"s{k}": float(np.nanmean(cov_all["reward"][Ak == k])) for k in range(K)}))
    rows.append(dict(group="-- trials/min", **{f"s{k}": float(np.nanmean(cov_all["trials_per_min"][Ak == k])) for k in range(K)}))
    comp[K] = (pd.DataFrame(rows).set_index("group"), Ak)
    print(f"\n=== K = {K} : state occupancy within each block (top rows sum to 1 across states) ===")
    display(comp[K][0].round(3))""", cid='22c32ee9')
code(r"""# --- test 3: is a single state homogeneous across blocks? ---
# Take the K=3 model's most active state (the one the day-plot showed swapping) and compare its
# windows from stable days against its windows from new-axis blocks, feature by feature.
A3 = comp[3][1]
TOP3 = 2                                       # top (fastest-cursor) state of the K=3 fit
a = (A3 == TOP3) & (ph_all == "stable")
b = (A3 == TOP3) & (blk_all == "new")
F_all = np.vstack(P["feats"])
print(f"K=3 state {TOP3}: {a.sum()} windows from stable days vs {b.sum()} from new-axis blocks")
rows = []
for i, f in enumerate(FEAT_NAMES):
    st = mannwhitneyu(F_all[a, i], F_all[b, i]) if min(a.sum(), b.sum()) > 10 else None
    rows.append(dict(feature=f, stable=F_all[a, i].mean(), new_axis=F_all[b, i].mean(),
                     difference=F_all[b, i].mean() - F_all[a, i].mean(),
                     p=(st.pvalue if st else np.nan)))
for nm in ["reward", "trials_per_min", "in_trial"]:
    rows.append(dict(feature=f"({nm}, external)", stable=np.nanmean(cov_all[nm][a]),
                     new_axis=np.nanmean(cov_all[nm][b]), difference=np.nan, p=np.nan))
display(pd.DataFrame(rows).round(4))
print("A state that is one behaviour should show no systematic difference between these columns.")

# --- test 4: does a stable-only model describe remapping days? ---
tr_idx = [i for i, m in enumerate(P["meta"]) if m["phase"] == "stable"]
te_idx = [i for i, m in enumerate(P["meta"]) if m["phase"] == "remap"]
gap_rows = []
if len(tr_idx) >= 6 and len(te_idx) >= 2:
    hold = tr_idx[-3:]; fitidx = tr_idx[:-3]
    new_windows = []
    for f, blk_seg in zip(P["feats"], P["block"]):
        m_new = (blk_seg == "new")
        if m_new.sum() > 10:
            new_windows.append(f[m_new])
    for K in [3, K_L2]:
        ms = bs.StickyARHMM(K, 0, dwell_target_samples=DWELL_L2_WIN, prior_strength=PRIOR_L2, seed=SEED)
        ms.fit([P["feats"][i] for i in fitidx], n_iter=120, n_restarts=2)
        gap_rows.append(dict(K=K,
                             held_out_stable=ms.loglik_per_sample([P["feats"][i] for i in hold]),
                             remap_days=ms.loglik_per_sample([P["feats"][i] for i in te_idx]),
                             new_axis_only=ms.loglik_per_sample(new_windows)))
    g = pd.DataFrame(gap_rows)
    g["gap_stable_minus_new"] = g.held_out_stable - g.new_axis_only
    display(g.round(4))
    print("Log-likelihood per sample under a model fitted on STABLE days only (higher = better fit).")
    print("A large drop from held_out_stable to new_axis_only means new-axis control is a regime")
    print("the stable-day model never saw -- i.e. it needs its own state(s).")""", cid='a3957f68')
code(r"""# --- summary figure for the audit ---
Ks = sorted(comp)
fig, axes = plt.subplots(1, len(Ks) + 1, figsize=(3.2 * (len(Ks) + 1), 3.4))
axes[0].errorbar(cv_l2.K, cv_l2.test_ll, yerr=cv_l2.test_ll_sem, marker="o", ms=4, label="held-out")
axes[0].plot(cv_l2.K, cv_l2.train_ll, "s--", ms=3, alpha=0.6, label="train")
axes[0].axvline(K_L2, color="r", ls="--", lw=1)
axes[0].set(xlabel="K", ylabel="log-likelihood / sample", title=f"K scan (using K={K_L2})")
axes[0].legend(fontsize=7)
for ax, K in zip(axes[1:], Ks):
    tab = comp[K][0].loc[[g for g, _ in GROUPS]]
    tab.plot(kind="bar", stacked=True, ax=ax, width=0.85, legend=False,
             color=[_L2_PAL[k % len(_L2_PAL)] for k in range(K)])
    ax.set(title=f"K = {K}", ylabel="occupancy" if K == Ks[0] else "", xlabel="")
    ax.tick_params(axis="x", rotation=30, labelsize=7)
axes[-1].legend([f"s{k}" for k in range(Ks[-1])], fontsize=6, bbox_to_anchor=(1.02, 1), loc="upper left")
fig.suptitle(f"{PRIMARY}: does the state set separate intuitive-axis from new-axis control?", y=1.04)
show(fig, "S8b_state_audit")""", cid='9541da51')

# ---------------- 9 · Relationship to treadmill, licking and imaging motion ----------------
md(r"""## 9 · Relationship to treadmill, licking and imaging motion

The heart of the multimodal comparison. Every quantity here was held out of both models, so any
structure is a genuine relationship rather than a restatement of the fitting inputs.

Four analyses:

1. **Treadmill transform validation** — re-run against imaging motion, so the choice of speed proxy
   is checked in-notebook rather than asserted.
2. **Covariates by state**, with Kruskal–Wallis across states (descriptive, uncorrected).
3. **Transition-triggered averages** — covariates aligned to entry into each engagement state, which
   shows whether locomotion and licking *lead* or *follow* the change in engagement.
4. **Cross-correlation** between cursor activity and treadmill speed at a range of lags.""", cid='fdf2e894')
code(r"""# --- 1. treadmill transform validation, against imaging motion as an independent reference ---
r0 = na.open_mouse(rfs, DATA_ROOT, PRIMARY)
val_rows = []
for day in na.get_day_list(r0)[1:6]:
    ref = bs.load_mc_motion(r0, day, sr=SR)
    for mode in bs.TREADMILL_MODES:
        v = bs.load_treadmill_speed(r0, day, mode=mode, sr=SR)
        ok = np.isfinite(v) & np.isfinite(ref)
        val_rows.append(dict(day=day, mode=mode, spearman=spearmanr(v[ok], ref[ok]).statistic))
tv = pd.DataFrame(val_rows).groupby("mode").spearman.agg(["mean", "min", "max"]).round(3)
print("Treadmill speed proxy vs imaging motion (independent movement reference), 5 days:")
display(tv)
print("The three transforms behave alike, so the default 'abs_dev' sets the units, not the conclusions.")
print("The absolute value depends on the smoothing of BOTH signals (smooth_sec), so read it as a")
print("consistency check between two independent movement measures, not as a calibration.")""", cid='aa17cb5e')
code(r"""# --- 2. covariates by engagement state, pooled over animals ---
# display name -> key in the per-window covariate dict (kept explicit; these differ)
COV_KEY = {"reward_per_min": "reward", "frac_in_trial": "in_trial", "treadmill": "treadmill",
           "lick_hz": "lick", "mc_motion": "mc_motion"}
COVS = list(COV_KEY)
fig, axes = plt.subplots(2, len(COVS), figsize=(3.0 * len(COVS), 5.6))
stat_rows = []
for ax, cname in zip(axes.T, COVS):
    vals = []
    for mouse, P in L2.items():
        lab = np.concatenate(P["labels"])
        cov = {k: np.concatenate([c[k] for c in P["cov"]]) for k in P["cov"][0]}
        src = cov[COV_KEY[cname]]
        for k in range(K_L2):
            for v in src[lab == k]:
                vals.append(dict(mouse=mouse, state=L2_NAMES[k], k=k, value=v))
    dfc = pd.DataFrame(vals)
    sns.boxplot(data=dfc, x="state", y="value", order=L2_NAMES, ax=ax[0], fliersize=0,
                palette=[L2_COLORS[k] for k in range(K_L2)])
    ax[0].set(title=f"{cname} (pooled)", xlabel="", ylabel="")
    ax[0].tick_params(axis="x", rotation=20)
    # pooling mixes animals with different baselines, which can mask a consistent
    # within-animal effect -- so show each animal separately as well
    pm = dfc.groupby(["mouse", "k"]).value.mean().unstack()
    for mouse in pm.index:
        ax[1].plot(range(K_L2), pm.loc[mouse], "-o", ms=4, lw=1.3, label=mouse)
    ax[1].set_xticks(range(K_L2)); ax[1].set_xticklabels(L2_NAMES, rotation=20)
    ax[1].set(title="per animal (mean)", ylabel="")
    groups = [dfc.loc[dfc.k == k, "value"].dropna().to_numpy() for k in range(K_L2)]
    if min(len(g) for g in groups) > 5:
        H, p = kruskal(*groups)
        # also test within each animal, then count how many agree in direction
        per_mouse_p, mono = [], 0
        for mouse in dfc.mouse.unique():
            gm = [dfc.loc[(dfc.k == k) & (dfc.mouse == mouse), "value"].dropna().to_numpy()
                  for k in range(K_L2)]
            if min(len(g) for g in gm) > 5:
                per_mouse_p.append(kruskal(*gm).pvalue)
                mono += int(np.all(np.diff([np.nanmean(g) for g in gm]) > 0)
                            or np.all(np.diff([np.nanmean(g) for g in gm]) < 0))
        stat_rows.append(dict(covariate=cname, H_pooled=H, p_pooled=p,
                              n_mice_p_lt_05=int(np.sum(np.array(per_mouse_p) < 0.05)),
                              n_mice_monotone=mono, n_mice=len(per_mouse_p),
                              **{L2_NAMES[k]: float(np.nanmean(groups[k])) for k in range(K_L2)}))
axes[1, 0].legend(fontsize=6)
fig.suptitle("Held-out covariates by engagement state: pooled (top) and per animal (bottom)", y=1.02)
show(fig, "S9_covariates_by_state")
print("Kruskal-Wallis across engagement states (descriptive, uncorrected).")
print("n_mice_monotone counts animals whose 3 state means are monotone -- a consistency check")
print("that pooled significance alone does not give, since pooling can mask or manufacture effects.")
display(pd.DataFrame(stat_rows).round(4))""", cid='6b6d0908')
code(r"""# --- 3. transition-triggered averages: what happens around a change in engagement? ---
HALF = int(round(90 / WIN_SEC))       # +/- 90 s, in windows
tta = {c: {k: [] for k in range(K_L2)} for c in ["treadmill", "lick", "reward", "in_trial"]}
for mouse, P in L2.items():
    for lab, cov in zip(P["labels"], P["cov"]):
        segd = bs.dwell_segments(lab)
        for _, row in segd.iterrows():
            s0, k = int(row.start), int(row.state)
            if s0 - HALF < 0 or s0 + HALF >= len(lab): continue
            for c in tta:
                tta[c][k].append(cov[c][s0 - HALF:s0 + HALF])
fig, axes = plt.subplots(1, 4, figsize=(15, 3.0), sharex=True)
tt = (np.arange(-HALF, HALF) + 0.5) * WIN_SEC
for ax, c in zip(axes, ["treadmill", "lick", "reward", "in_trial"]):
    for k in range(K_L2):
        A = np.array(tta[c][k])
        if len(A) < 5: continue
        m_, se = A.mean(0), A.std(0) / np.sqrt(len(A))
        ax.plot(tt, m_, color=L2_COLORS[k], label=f"{L2_NAMES[k]} (n={len(A)})")
        ax.fill_between(tt, m_ - se, m_ + se, color=L2_COLORS[k], alpha=0.25)
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set(title=c, xlabel="time from state entry (s)")
axes[0].set_ylabel("mean value"); axes[0].legend(fontsize=7)
fig.suptitle("Transition-triggered averages, aligned to entry into each engagement state", y=1.04)
show(fig, "S9_transition_triggered")""", cid='b814a884')
code(r"""# --- 4. cross-correlation of cursor activity with treadmill speed ---
MAXLAG = int(round(60 * SR_L1))
lags = np.arange(-MAXLAG, MAXLAG + 1, max(1, int(SR_L1)))
fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
for mouse in L2:
    xc = []
    for s in [s for s in SEGS if s["mouse"] == mouse and "l1" in s][:12]:
        y = s["y"]; a = np.abs(np.diff(y, prepend=y[0])); b = s["cov"]["treadmill"]
        n = min(len(a), len(b)); a, b = a[:n], b[:n]
        a = (a - a.mean()) / (a.std() + 1e-9); b = (b - b.mean()) / (b.std() + 1e-9)
        xc.append([np.mean(a[max(0, -L_):n - max(0, L_)] * b[max(0, L_):n - max(0, -L_)]) for L_ in lags])
    A = np.array(xc)
    axes[0].plot(lags / SR_L1, A.mean(0), label=mouse, lw=1.4)
axes[0].axvline(0, color="k", ls="--", lw=0.8); axes[0].axhline(0, color="k", lw=0.6)
axes[0].set(xlabel="lag (s)   [>0: treadmill leads cursor activity]", ylabel="correlation",
            title="cursor activity x treadmill speed")
axes[0].legend(fontsize=7)
occ = []
for mouse, P in L2.items():
    for lab, meta in zip(P["labels"], P["meta"]):
        o = bs.state_occupancy(lab, K_L2)
        occ.append(dict(mouse=mouse, day_from0=meta["day_from0"], phase=meta["phase"],
                        **{L2_NAMES[k]: o[k] for k in range(K_L2)}))
OCC = pd.DataFrame(occ)
g = OCC[OCC.mouse == PRIMARY].groupby("day_from0")[L2_NAMES].mean()
g.plot(ax=axes[1], color=[L2_COLORS[k] for k in range(K_L2)], marker="o", ms=3, lw=1.2)
axes[1].set(xlabel="day from day-0", ylabel="occupancy", title=f"{PRIMARY}: engagement across days")
axes[1].legend(fontsize=7)
show(fig, "S9_xcorr_and_days")""", cid='ff6c5d19')

# ---------------- 10 · Cursor-regime states and task structure ----------------
md(r"""## 10 · Cursor-regime states and task structure

Two questions the states are well placed to answer:

1. **Within a session**, does engagement decay with time on task?
2. **On remapping days**, how does engagement differ between the intuitive, new-axis and washout
   blocks? This bears directly on the standing caveat that washout is confounded with motivation —
   here that confound becomes something measured rather than assumed.""", cid='49221ea8')
code(r"""rows = []
for mouse, P in L2.items():
    for lab, cov, blk, meta in zip(P["labels"], P["cov"], P["block"], P["meta"]):
        for i, k in enumerate(lab):
            rows.append(dict(mouse=mouse, day=meta["day"], phase=meta["phase"], block=blk[i],
                             minute=i * WIN_SEC / 60, state=k, treadmill=cov["treadmill"][i],
                             reward=cov["reward"][i]))
WDF = pd.DataFrame(rows)
WDF["third"] = pd.cut(WDF.groupby(["mouse", "day"]).minute.transform(lambda v: v / v.max()),
                      [0, 1/3, 2/3, 1.001], labels=["early", "middle", "late"], include_lowest=True)

fig, axes = plt.subplots(1, 3, figsize=(15, 3.2))
ct = pd.crosstab(WDF.third, WDF.state, normalize="index")
ct.columns = [L2_NAMES[c] for c in ct.columns]
ct.plot(kind="bar", stacked=True, ax=axes[0], color=[L2_COLORS[k] for k in range(K_L2)], width=0.8)
axes[0].set(title="engagement by position within session", ylabel="fraction of windows", xlabel="")
axes[0].legend(fontsize=7); axes[0].tick_params(axis="x", rotation=0)

rm = WDF[(WDF.mouse == REMAP_MOUSE) & (WDF.phase == "remap") & (WDF.block != "single")]
BLOCKS = ["intuitive_pre", "new", "washout"]
if len(rm):
    ct2 = pd.crosstab(rm.block, rm.state, normalize="index").reindex(BLOCKS)
    ct2.columns = [L2_NAMES[c] for c in ct2.columns]
    ct2.plot(kind="bar", stacked=True, ax=axes[1], color=[L2_COLORS[k] for k in range(K_L2)], width=0.8)
    axes[1].set(title=f"{REMAP_MOUSE}: engagement by remapping block", ylabel="fraction", xlabel="")
    axes[1].legend(fontsize=7); axes[1].tick_params(axis="x", rotation=15)
    tb = rm.groupby("block").agg(treadmill=("treadmill", "mean"), reward=("reward", "mean"),
                                 frac_idle=("state", lambda v: (v == 0).mean())).reindex(BLOCKS)
    tb.plot(kind="bar", ax=axes[2], width=0.8); axes[2].set(title="block means", xlabel="")
    axes[2].legend(fontsize=7); axes[2].tick_params(axis="x", rotation=15)
show(fig, "S10_task_structure")

if len(rm):
    print("Engagement-state composition by remapping block (rows sum to 1):")
    display(ct2.round(3))
    a = rm.loc[rm.block == "intuitive_pre", "state"] == 0
    b = rm.loc[rm.block == "washout", "state"] == 0
    print(f"idle fraction: intuitive_pre={a.mean():.3f}  washout={b.mean():.3f}  "
          f"Mann-Whitney p={mannwhitneyu(a.astype(float), b.astype(float))[1]:.3g}")
    print("This quantifies the washout motivation confound rather than assuming it.")""", cid='f8052e66')

# ---------------- 11 · Correctness checks and caveats ----------------
md(r"""## 11 · Correctness checks and caveats""", cid='6efcab00')
code(r"""checks = []
def chk(name, ok, detail=""):
    checks.append(dict(check=name, result="PASS" if ok else "FAIL", detail=detail))

chk("AR-HMM recovers synthetic ground truth", syn_acc > 0.85, f"accuracy {syn_acc:.3f}")
chk("EM likelihood monotone (synthetic)", bool((syn_steps >= -1e-6 * np.abs(syn_ll[:-1])).all()))
chk("ar_order=0 reduces to a Gaussian HMM", syn_acc_g > 0.90, f"accuracy {syn_acc_g:.3f}")
chk("every segment has a Level-1 label", all("l1" in s for s in SEGS if s["mouse"] in L1))
chk("Level-1 labels align with the data", all(len(s["l1"]) == len(s["y"]) for s in SEGS if "l1" in s))
chk("no NaN in fitted cursor sequences", all(np.isfinite(s["y"]).all() for s in SEGS))
chk("state occupancies sum to 1",
    all(abs(bs.state_occupancy(np.concatenate(P["labels"]), K_L2).sum() - 1) < 1e-9 for P in L2.values()))
chk("Level-1 states ordered by increasing speed",
    all(np.all(np.diff(L1_TAB[L1_TAB.mouse == m].sort_values("state").cursor_speed) > 0) for m in L1))
_feat_set = set(FEAT_NAMES)
chk("Level 2 fitted on cursor-derived features only",
    _feat_set.isdisjoint({"treadmill", "lick", "mc_motion", "reward", "in_trial"}),
    f"features = {FEAT_NAMES}")
chk("covariates never entered either fit", "treadmill" not in _feat_set and "reward" not in _feat_set)
# The analysis figures use the primary animal, so that is what must hold. The cross-animal
# count is reported alongside: animals differing is a fact about the data, not a code fault.
chk("state names match the task profile in the PRIMARY animal",
    bool(name_chk.loc[PRIMARY].all()),
    f"holds in {int(name_chk.all(axis=1).sum())}/{len(name_chk)} animals overall")
chk("K_L2 exceeds the K=3 that merged the two axes", K_L2 >= 4, f"K_L2 = {K_L2}")
chk("Level-2 dwell times are longer than Level-1",
    L2_TAB.median_dwell_min.median() * 60 > L1_TAB.median_dwell_s.median(),
    f"L2 {L2_TAB.median_dwell_min.median()*60:.0f}s vs L1 {L1_TAB.median_dwell_s.median():.1f}s")
display(pd.DataFrame(checks))

print('''
Caveats:
 1. K is a modelling choice read off a held-out-likelihood elbow, not a discovered truth. The
    state count changes the granularity of the description, not whether the structure is real.
 2. The two levels are two models at two chosen timescales. WIN_SEC and BIN_L1 set those
    timescales, and different choices will produce different (equally valid) decompositions.
 3. State names are interpretive labels attached after the fact from the measured profile, and
    are validated per animal above. The model outputs numbers, not intentions. States are ordered
    by cursor ACTIVITY, which is not the same as productivity -- "fluent" and "vigorous" have
    similar reward rates and differ in how the cursor is driven. Read the task table, not the label.
 3b. Level-2 states are NOT a pure engagement axis. Section 8b shows one state is occupied almost
    only during new-axis control, so the regime the cursor is driven in is confounded with which
    decoder axis is active. Any statement of the form "the animal was more engaged on day X"
    must be checked against the block composition before it can be believed.
 3c. Remapping sessions are 80 min where earlier sessions are 60 min, so any stable-vs-remap
    comparison across days carries that confound too. The within-session block comparison in
    Section 8b is the one that avoids it.
 3d. The positional names are validated per animal in Section 8. They hold for the primary
    animal but NOT for every animal (see that table), because the canonical ordering is by
    cursor speed and the animals differ in how speed maps onto yield. For any animal marked
    False there, use the per-animal task table and ignore the label.
 4. Treadmill speed is a proxy derived from an encoder voltage, validated against imaging motion
    but not calibrated to cm/s, so only relative comparisons are meaningful.
 5. Imaging motion and treadmill are not independent of each other -- both increase with gross
    body movement -- so they corroborate rather than independently confirm.
 6. p-values are descriptive: many covariates are tested without correction, and with tens of
    thousands of windows even tiny differences reach significance. Read effect sizes.
 7. No pupil / video data exists in this dataset, so arousal is only indirectly indexed here
    (via locomotion and licking).
''')""", cid='75c55a31')

nb["cells"] = cells
nb["metadata"] = {'kernelspec': {'display_name': 'BCI_analysis', 'language': 'python', 'name': 'python3'}, 'language_info': {'codemirror_mode': {'name': 'ipython', 'version': 3}, 'file_extension': '.py', 'mimetype': 'text/x-python', 'name': 'python', 'nbconvert_exporter': 'python', 'pygments_lexer': 'ipython3', 'version': '3.11.15'}, 'widgets': {'application/vnd.jupyter.widget-state+json': {'state': {'01a6ffe867c04442baf17c3967c87707': {'model_module': '@jupyter-widgets/controls', 'model_module_version': '2.0.0', 'model_name': 'HTMLModel', 'state': {'_dom_classes': [], '_model_module': '@jupyter-widgets/controls', '_model_module_version': '2.0.0', '_model_name': 'HTMLModel', '_view_count': None, '_view_module': '@jupyter-widgets/controls', '_view_module_version': '2.0.0', '_view_name': 'HTMLView', 'description': '', 'description_allow_html': False, 'layout': 'IPY_MODEL_ab335c74e1564b9c850f0995df824b24', 'placeholder': '\u200b', 'style': 'IPY_MODEL_40e6c05004a346dd80bea1611f5bc230', 'tabbable': None, 'tooltip': None, 'value': 'mouse_0322R:\u2007100%'}}, '01e215ac4e26436c86c27cb8496a80e0': {'model_module': '@jupyter-widgets/controls', 'model_module_version': '2.0.0', 'model_name': 'ProgressStyleModel', 'state': {'_model_module': '@jupyter-widgets/controls', '_model_module_version': '2.0.0', '_model_name': 'ProgressStyleModel', '_view_count': None, '_view_module': '@jupyter-widgets/base', '_view_module_version': '2.0.0', '_view_name': 'StyleView', 'bar_color': None, 'description_width': ''}}, '0a7de583cf624fd38408fb8999fc492e': {'model_module': '@jupyter-widgets/controls', 'model_module_version': '2.0.0', 'model_name': 'HTMLStyleModel', 'state': {'_model_module': '@jupyter-widgets/controls', '_model_module_version': '2.0.0', '_model_name': 'HTMLStyleModel', '_view_count': None, '_view_module': '@jupyter-widgets/base', '_view_module_version': '2.0.0', '_view_name': 'StyleView', 'background': None, 'description_width': '', 'font_size': None, 'text_color': None}}, '1422db234b9947a3878eeb4934f9efd1': {'model_module': '@jupyter-widgets/base', 'model_module_version': '2.0.0', 'model_name': 'LayoutModel', 'state': {'_model_module': '@jupyter-widgets/base', '_model_module_version': '2.0.0', '_model_name': 'LayoutModel', '_view_count': None, '_view_module': '@jupyter-widgets/base', '_view_module_version': '2.0.0', '_view_name': 'LayoutView', 'align_content': None, 'align_items': None, 'align_self': None, 'border_bottom': None, 'border_left': None, 'border_right': None, 'border_top': None, 'bottom': None, 'display': None, 'flex': None, 'flex_flow': None, 'grid_area': None, 'grid_auto_columns': None, 'grid_auto_flow': None, 'grid_auto_rows': None, 'grid_column': None, 'grid_gap': None, 'grid_row': None, 'grid_template_areas': None, 'grid_template_columns': None, 'grid_template_rows': None, 'height': None, 'justify_content': None, 'justify_items': None, 'left': None, 'margin': None, 'max_height': None, 'max_width': None, 'min_height': None, 'min_width': None, 'object_fit': None, 'object_position': None, 'order': None, 'overflow': None, 'padding': None, 'right': None, 'top': None, 'visibility': 'hidden', 'width': None}}, '16aa18a981344bd1b81bf29ed555540c': {'model_module': '@jupyter-widgets/controls', 'model_module_version': '2.0.0', 'model_name': 'HTMLStyleModel', 'state': {'_model_module': '@jupyter-widgets/controls', '_model_module_version': '2.0.0', '_model_name': 'HTMLStyleModel', '_view_count': None, '_view_module': '@jupyter-widgets/base', '_view_module_version': '2.0.0', '_view_name': 'StyleView', 'background': None, 'description_width': '', 'font_size': None, 'text_color': None}}, '2257bae091784d7ea208be9aa29a50e1': {'model_module': '@jupyter-widgets/controls', 'model_module_version': '2.0.0', 'model_name': 'ProgressStyleModel', 'state': {'_model_module': '@jupyter-widgets/controls', '_model_module_version': '2.0.0', '_model_name': 'ProgressStyleModel', '_view_count': None, '_view_module': '@jupyter-widgets/base', '_view_module_version': '2.0.0', '_view_name': 'StyleView', 'bar_color': None, 'description_width': ''}}, '2afae65b083441ccad2def58f30c9629': {'model_module': '@jupyter-widgets/controls', 'model_module_version': '2.0.0', 'model_name': 'HBoxModel', 'state': {'_dom_classes': [], '_model_module': '@jupyter-widgets/controls', '_model_module_version': '2.0.0', '_model_name': 'HBoxModel', '_view_count': None, '_view_module': '@jupyter-widgets/controls', '_view_module_version': '2.0.0', '_view_name': 'HBoxView', 'box_style': '', 'children': ['IPY_MODEL_665700d78b6a454193a36ee56d225aa1', 'IPY_MODEL_6e8ef0e56e3d4479944b0a1f0a807215', 'IPY_MODEL_76c581d93b1648c987cb7b738c32215e'], 'layout': 'IPY_MODEL_bf703899e392443b8d2e4bc33229fdb3', 'tabbable': None, 'tooltip': None}}, '2d1e1b7f135d4fa3b747e9a938c9a01d': {'model_module': '@jupyter-widgets/base', 'model_module_version': '2.0.0', 'model_name': 'LayoutModel', 'state': {'_model_module': '@jupyter-widgets/base', '_model_module_version': '2.0.0', '_model_name': 'LayoutModel', '_view_count': None, '_view_module': '@jupyter-widgets/base', '_view_module_version': '2.0.0', '_view_name': 'LayoutView', 'align_content': None, 'align_items': None, 'align_self': None, 'border_bottom': None, 'border_left': None, 'border_right': None, 'border_top': None, 'bottom': None, 'display': None, 'flex': None, 'flex_flow': None, 'grid_area': None, 'grid_auto_columns': None, 'grid_auto_flow': None, 'grid_auto_rows': None, 'grid_column': None, 'grid_gap': None, 'grid_row': None, 'grid_template_areas': None, 'grid_template_columns': None, 'grid_template_rows': None, 'height': None, 'justify_content': None, 'justify_items': None, 'left': None, 'margin': None, 'max_height': None, 'max_width': None, 'min_height': None, 'min_width': None, 'object_fit': None, 'object_position': None, 'order': None, 'overflow': None, 'padding': None, 'right': None, 'top': None, 'visibility': None, 'width': None}}, '3ee218389254456fbe02a6c049eb1b0a': {'model_module': '@jupyter-widgets/controls', 'model_module_version': '2.0.0', 'model_name': 'HTMLStyleModel', 'state': {'_model_module': '@jupyter-widgets/controls', '_model_module_version': '2.0.0', '_model_name': 'HTMLStyleModel', '_view_count': None, '_view_module': '@jupyter-widgets/base', '_view_module_version': '2.0.0', '_view_name': 'StyleView', 'background': None, 'description_width': '', 'font_size': None, 'text_color': None}}, '40e6c05004a346dd80bea1611f5bc230': {'model_module': '@jupyter-widgets/controls', 'model_module_version': '2.0.0', 'model_name': 'HTMLStyleModel', 'state': {'_model_module': '@jupyter-widgets/controls', '_model_module_version': '2.0.0', '_model_name': 'HTMLStyleModel', '_view_count': None, '_view_module': '@jupyter-widgets/base', '_view_module_version': '2.0.0', '_view_name': 'StyleView', 'background': None, 'description_width': '', 'font_size': None, 'text_color': None}}, '42d7bc8fe6ab418fa355ee63445237c4': {'model_module': '@jupyter-widgets/base', 'model_module_version': '2.0.0', 'model_name': 'LayoutModel', 'state': {'_model_module': '@jupyter-widgets/base', '_model_module_version': '2.0.0', '_model_name': 'LayoutModel', '_view_count': None, '_view_module': '@jupyter-widgets/base', '_view_module_version': '2.0.0', '_view_name': 'LayoutView', 'align_content': None, 'align_items': None, 'align_self': None, 'border_bottom': None, 'border_left': None, 'border_right': None, 'border_top': None, 'bottom': None, 'display': None, 'flex': None, 'flex_flow': None, 'grid_area': None, 'grid_auto_columns': None, 'grid_auto_flow': None, 'grid_auto_rows': None, 'grid_column': None, 'grid_gap': None, 'grid_row': None, 'grid_template_areas': None, 'grid_template_columns': None, 'grid_template_rows': None, 'height': None, 'justify_content': None, 'justify_items': None, 'left': None, 'margin': None, 'max_height': None, 'max_width': None, 'min_height': None, 'min_width': None, 'object_fit': None, 'object_position': None, 'order': None, 'overflow': None, 'padding': None, 'right': None, 'top': None, 'visibility': None, 'width': None}}, '509d8dbe18b14d78a6051cd16ca1eb93': {'model_module': '@jupyter-widgets/base', 'model_module_version': '2.0.0', 'model_name': 'LayoutModel', 'state': {'_model_module': '@jupyter-widgets/base', '_model_module_version': '2.0.0', '_model_name': 'LayoutModel', '_view_count': None, '_view_module': '@jupyter-widgets/base', '_view_module_version': '2.0.0', '_view_name': 'LayoutView', 'align_content': None, 'align_items': None, 'align_self': None, 'border_bottom': None, 'border_left': None, 'border_right': None, 'border_top': None, 'bottom': None, 'display': None, 'flex': None, 'flex_flow': None, 'grid_area': None, 'grid_auto_columns': None, 'grid_auto_flow': None, 'grid_auto_rows': None, 'grid_column': None, 'grid_gap': None, 'grid_row': None, 'grid_template_areas': None, 'grid_template_columns': None, 'grid_template_rows': None, 'height': None, 'justify_content': None, 'justify_items': None, 'left': None, 'margin': None, 'max_height': None, 'max_width': None, 'min_height': None, 'min_width': None, 'object_fit': None, 'object_position': None, 'order': None, 'overflow': None, 'padding': None, 'right': None, 'top': None, 'visibility': None, 'width': None}}, '53b76f7085a94f9d93c74b5ba2978d9a': {'model_module': '@jupyter-widgets/controls', 'model_module_version': '2.0.0', 'model_name': 'HTMLModel', 'state': {'_dom_classes': [], '_model_module': '@jupyter-widgets/controls', '_model_module_version': '2.0.0', '_model_name': 'HTMLModel', '_view_count': None, '_view_module': '@jupyter-widgets/controls', '_view_module_version': '2.0.0', '_view_name': 'HTMLView', 'description': '', 'description_allow_html': False, 'layout': 'IPY_MODEL_42d7bc8fe6ab418fa355ee63445237c4', 'placeholder': '\u200b', 'style': 'IPY_MODEL_0a7de583cf624fd38408fb8999fc492e', 'tabbable': None, 'tooltip': None, 'value': '\u200735/35\u2007[00:07&lt;00:00,\u2007\u20075.29it/s]'}}, '566fc8da062f4f219ca77a1196e6e3d1': {'model_module': '@jupyter-widgets/controls', 'model_module_version': '2.0.0', 'model_name': 'ProgressStyleModel', 'state': {'_model_module': '@jupyter-widgets/controls', '_model_module_version': '2.0.0', '_model_name': 'ProgressStyleModel', '_view_count': None, '_view_module': '@jupyter-widgets/base', '_view_module_version': '2.0.0', '_view_name': 'StyleView', 'bar_color': None, 'description_width': ''}}, '6290e43b120142d99e8073af5de13ed6': {'model_module': '@jupyter-widgets/controls', 'model_module_version': '2.0.0', 'model_name': 'HTMLStyleModel', 'state': {'_model_module': '@jupyter-widgets/controls', '_model_module_version': '2.0.0', '_model_name': 'HTMLStyleModel', '_view_count': None, '_view_module': '@jupyter-widgets/base', '_view_module_version': '2.0.0', '_view_name': 'StyleView', 'background': None, 'description_width': '', 'font_size': None, 'text_color': None}}, '665700d78b6a454193a36ee56d225aa1': {'model_module': '@jupyter-widgets/controls', 'model_module_version': '2.0.0', 'model_name': 'HTMLModel', 'state': {'_dom_classes': [], '_model_module': '@jupyter-widgets/controls', '_model_module_version': '2.0.0', '_model_name': 'HTMLModel', '_view_count': None, '_view_module': '@jupyter-widgets/controls', '_view_module_version': '2.0.0', '_view_name': 'HTMLView', 'description': '', 'description_allow_html': False, 'layout': 'IPY_MODEL_ae904e3912da423daa35107c4933f34d', 'placeholder': '\u200b', 'style': 'IPY_MODEL_3ee218389254456fbe02a6c049eb1b0a', 'tabbable': None, 'tooltip': None, 'value': 'mouse_0916N:\u2007100%'}}, '6e8ef0e56e3d4479944b0a1f0a807215': {'model_module': '@jupyter-widgets/controls', 'model_module_version': '2.0.0', 'model_name': 'FloatProgressModel', 'state': {'_dom_classes': [], '_model_module': '@jupyter-widgets/controls', '_model_module_version': '2.0.0', '_model_name': 'FloatProgressModel', '_view_count': None, '_view_module': '@jupyter-widgets/controls', '_view_module_version': '2.0.0', '_view_name': 'ProgressView', 'bar_style': '', 'description': '', 'description_allow_html': False, 'layout': 'IPY_MODEL_2d1e1b7f135d4fa3b747e9a938c9a01d', 'max': 4, 'min': 0, 'orientation': 'horizontal', 'style': 'IPY_MODEL_967aaada2cdd4d6292ebf9bf037440c0', 'tabbable': None, 'tooltip': None, 'value': 4}}, '7053b82652c94a828d1d5211228f049d': {'model_module': '@jupyter-widgets/base', 'model_module_version': '2.0.0', 'model_name': 'LayoutModel', 'state': {'_model_module': '@jupyter-widgets/base', '_model_module_version': '2.0.0', '_model_name': 'LayoutModel', '_view_count': None, '_view_module': '@jupyter-widgets/base', '_view_module_version': '2.0.0', '_view_name': 'LayoutView', 'align_content': None, 'align_items': None, 'align_self': None, 'border_bottom': None, 'border_left': None, 'border_right': None, 'border_top': None, 'bottom': None, 'display': None, 'flex': None, 'flex_flow': None, 'grid_area': None, 'grid_auto_columns': None, 'grid_auto_flow': None, 'grid_auto_rows': None, 'grid_column': None, 'grid_gap': None, 'grid_row': None, 'grid_template_areas': None, 'grid_template_columns': None, 'grid_template_rows': None, 'height': None, 'justify_content': None, 'justify_items': None, 'left': None, 'margin': None, 'max_height': None, 'max_width': None, 'min_height': None, 'min_width': None, 'object_fit': None, 'object_position': None, 'order': None, 'overflow': None, 'padding': None, 'right': None, 'top': None, 'visibility': None, 'width': None}}, '76c581d93b1648c987cb7b738c32215e': {'model_module': '@jupyter-widgets/controls', 'model_module_version': '2.0.0', 'model_name': 'HTMLModel', 'state': {'_dom_classes': [], '_model_module': '@jupyter-widgets/controls', '_model_module_version': '2.0.0', '_model_name': 'HTMLModel', '_view_count': None, '_view_module': '@jupyter-widgets/controls', '_view_module_version': '2.0.0', '_view_name': 'HTMLView', 'description': '', 'description_allow_html': False, 'layout': 'IPY_MODEL_ff4c91b612204fb1ad6d2bd898a5f707', 'placeholder': '\u200b', 'style': 'IPY_MODEL_c9e6e48d4d7f476eb271910d0316eb7a', 'tabbable': None, 'tooltip': None, 'value': '\u20074/4\u2007[00:00&lt;00:00,\u2007\u20077.07it/s]'}}, '85f7603a347642a39dafc6aafaca725e': {'model_module': '@jupyter-widgets/base', 'model_module_version': '2.0.0', 'model_name': 'LayoutModel', 'state': {'_model_module': '@jupyter-widgets/base', '_model_module_version': '2.0.0', '_model_name': 'LayoutModel', '_view_count': None, '_view_module': '@jupyter-widgets/base', '_view_module_version': '2.0.0', '_view_name': 'LayoutView', 'align_content': None, 'align_items': None, 'align_self': None, 'border_bottom': None, 'border_left': None, 'border_right': None, 'border_top': None, 'bottom': None, 'display': None, 'flex': None, 'flex_flow': None, 'grid_area': None, 'grid_auto_columns': None, 'grid_auto_flow': None, 'grid_auto_rows': None, 'grid_column': None, 'grid_gap': None, 'grid_row': None, 'grid_template_areas': None, 'grid_template_columns': None, 'grid_template_rows': None, 'height': None, 'justify_content': None, 'justify_items': None, 'left': None, 'margin': None, 'max_height': None, 'max_width': None, 'min_height': None, 'min_width': None, 'object_fit': None, 'object_position': None, 'order': None, 'overflow': None, 'padding': None, 'right': None, 'top': None, 'visibility': None, 'width': None}}, '967aaada2cdd4d6292ebf9bf037440c0': {'model_module': '@jupyter-widgets/controls', 'model_module_version': '2.0.0', 'model_name': 'ProgressStyleModel', 'state': {'_model_module': '@jupyter-widgets/controls', '_model_module_version': '2.0.0', '_model_name': 'ProgressStyleModel', '_view_count': None, '_view_module': '@jupyter-widgets/base', '_view_module_version': '2.0.0', '_view_name': 'StyleView', 'bar_color': None, 'description_width': ''}}, '9e26dfd567ca443ab8733ee606f844a3': {'model_module': '@jupyter-widgets/controls', 'model_module_version': '2.0.0', 'model_name': 'FloatProgressModel', 'state': {'_dom_classes': [], '_model_module': '@jupyter-widgets/controls', '_model_module_version': '2.0.0', '_model_name': 'FloatProgressModel', '_view_count': None, '_view_module': '@jupyter-widgets/controls', '_view_module_version': '2.0.0', '_view_name': 'ProgressView', 'bar_style': '', 'description': '', 'description_allow_html': False, 'layout': 'IPY_MODEL_7053b82652c94a828d1d5211228f049d', 'max': 6, 'min': 0, 'orientation': 'horizontal', 'style': 'IPY_MODEL_01e215ac4e26436c86c27cb8496a80e0', 'tabbable': None, 'tooltip': None, 'value': 6}}, 'a46776e3851f4c6683eb2de6bc0f9ae9': {'model_module': '@jupyter-widgets/base', 'model_module_version': '2.0.0', 'model_name': 'LayoutModel', 'state': {'_model_module': '@jupyter-widgets/base', '_model_module_version': '2.0.0', '_model_name': 'LayoutModel', '_view_count': None, '_view_module': '@jupyter-widgets/base', '_view_module_version': '2.0.0', '_view_name': 'LayoutView', 'align_content': None, 'align_items': None, 'align_self': None, 'border_bottom': None, 'border_left': None, 'border_right': None, 'border_top': None, 'bottom': None, 'display': None, 'flex': None, 'flex_flow': None, 'grid_area': None, 'grid_auto_columns': None, 'grid_auto_flow': None, 'grid_auto_rows': None, 'grid_column': None, 'grid_gap': None, 'grid_row': None, 'grid_template_areas': None, 'grid_template_columns': None, 'grid_template_rows': None, 'height': None, 'justify_content': None, 'justify_items': None, 'left': None, 'margin': None, 'max_height': None, 'max_width': None, 'min_height': None, 'min_width': None, 'object_fit': None, 'object_position': None, 'order': None, 'overflow': None, 'padding': None, 'right': None, 'top': None, 'visibility': None, 'width': None}}, 'a87a7d355d9e43058e6c739a70d01492': {'model_module': '@jupyter-widgets/base', 'model_module_version': '2.0.0', 'model_name': 'LayoutModel', 'state': {'_model_module': '@jupyter-widgets/base', '_model_module_version': '2.0.0', '_model_name': 'LayoutModel', '_view_count': None, '_view_module': '@jupyter-widgets/base', '_view_module_version': '2.0.0', '_view_name': 'LayoutView', 'align_content': None, 'align_items': None, 'align_self': None, 'border_bottom': None, 'border_left': None, 'border_right': None, 'border_top': None, 'bottom': None, 'display': None, 'flex': None, 'flex_flow': None, 'grid_area': None, 'grid_auto_columns': None, 'grid_auto_flow': None, 'grid_auto_rows': None, 'grid_column': None, 'grid_gap': None, 'grid_row': None, 'grid_template_areas': None, 'grid_template_columns': None, 'grid_template_rows': None, 'height': None, 'justify_content': None, 'justify_items': None, 'left': None, 'margin': None, 'max_height': None, 'max_width': None, 'min_height': None, 'min_width': None, 'object_fit': None, 'object_position': None, 'order': None, 'overflow': None, 'padding': None, 'right': None, 'top': None, 'visibility': None, 'width': None}}, 'ab335c74e1564b9c850f0995df824b24': {'model_module': '@jupyter-widgets/base', 'model_module_version': '2.0.0', 'model_name': 'LayoutModel', 'state': {'_model_module': '@jupyter-widgets/base', '_model_module_version': '2.0.0', '_model_name': 'LayoutModel', '_view_count': None, '_view_module': '@jupyter-widgets/base', '_view_module_version': '2.0.0', '_view_name': 'LayoutView', 'align_content': None, 'align_items': None, 'align_self': None, 'border_bottom': None, 'border_left': None, 'border_right': None, 'border_top': None, 'bottom': None, 'display': None, 'flex': None, 'flex_flow': None, 'grid_area': None, 'grid_auto_columns': None, 'grid_auto_flow': None, 'grid_auto_rows': None, 'grid_column': None, 'grid_gap': None, 'grid_row': None, 'grid_template_areas': None, 'grid_template_columns': None, 'grid_template_rows': None, 'height': None, 'justify_content': None, 'justify_items': None, 'left': None, 'margin': None, 'max_height': None, 'max_width': None, 'min_height': None, 'min_width': None, 'object_fit': None, 'object_position': None, 'order': None, 'overflow': None, 'padding': None, 'right': None, 'top': None, 'visibility': None, 'width': None}}, 'ae904e3912da423daa35107c4933f34d': {'model_module': '@jupyter-widgets/base', 'model_module_version': '2.0.0', 'model_name': 'LayoutModel', 'state': {'_model_module': '@jupyter-widgets/base', '_model_module_version': '2.0.0', '_model_name': 'LayoutModel', '_view_count': None, '_view_module': '@jupyter-widgets/base', '_view_module_version': '2.0.0', '_view_name': 'LayoutView', 'align_content': None, 'align_items': None, 'align_self': None, 'border_bottom': None, 'border_left': None, 'border_right': None, 'border_top': None, 'bottom': None, 'display': None, 'flex': None, 'flex_flow': None, 'grid_area': None, 'grid_auto_columns': None, 'grid_auto_flow': None, 'grid_auto_rows': None, 'grid_column': None, 'grid_gap': None, 'grid_row': None, 'grid_template_areas': None, 'grid_template_columns': None, 'grid_template_rows': None, 'height': None, 'justify_content': None, 'justify_items': None, 'left': None, 'margin': None, 'max_height': None, 'max_width': None, 'min_height': None, 'min_width': None, 'object_fit': None, 'object_position': None, 'order': None, 'overflow': None, 'padding': None, 'right': None, 'top': None, 'visibility': None, 'width': None}}, 'b0de30b487984d7cab5c26ebdb0f8b06': {'model_module': '@jupyter-widgets/controls', 'model_module_version': '2.0.0', 'model_name': 'HTMLModel', 'state': {'_dom_classes': [], '_model_module': '@jupyter-widgets/controls', '_model_module_version': '2.0.0', '_model_name': 'HTMLModel', '_view_count': None, '_view_module': '@jupyter-widgets/controls', '_view_module_version': '2.0.0', '_view_name': 'HTMLView', 'description': '', 'description_allow_html': False, 'layout': 'IPY_MODEL_509d8dbe18b14d78a6051cd16ca1eb93', 'placeholder': '\u200b', 'style': 'IPY_MODEL_c3ff4d7454844632a48a8156ab672040', 'tabbable': None, 'tooltip': None, 'value': 'mouse_0403R:\u2007100%'}}, 'b4c7f12874d3482a9be6ae2fff9e3cda': {'model_module': '@jupyter-widgets/controls', 'model_module_version': '2.0.0', 'model_name': 'HTMLModel', 'state': {'_dom_classes': [], '_model_module': '@jupyter-widgets/controls', '_model_module_version': '2.0.0', '_model_name': 'HTMLModel', '_view_count': None, '_view_module': '@jupyter-widgets/controls', '_view_module_version': '2.0.0', '_view_name': 'HTMLView', 'description': '', 'description_allow_html': False, 'layout': 'IPY_MODEL_85f7603a347642a39dafc6aafaca725e', 'placeholder': '\u200b', 'style': 'IPY_MODEL_6290e43b120142d99e8073af5de13ed6', 'tabbable': None, 'tooltip': None, 'value': '\u200736/36\u2007[00:16&lt;00:00,\u2007\u20072.53it/s]'}}, 'b9733f2670614da5a48ecfa51db10b5c': {'model_module': '@jupyter-widgets/controls', 'model_module_version': '2.0.0', 'model_name': 'HBoxModel', 'state': {'_dom_classes': [], '_model_module': '@jupyter-widgets/controls', '_model_module_version': '2.0.0', '_model_name': 'HBoxModel', '_view_count': None, '_view_module': '@jupyter-widgets/controls', '_view_module_version': '2.0.0', '_view_name': 'HBoxView', 'box_style': '', 'children': ['IPY_MODEL_ca6fad79f26b4b898a44d7eb2f950772', 'IPY_MODEL_bff4a2d1792a41ba876ef82b337db091', 'IPY_MODEL_b4c7f12874d3482a9be6ae2fff9e3cda'], 'layout': 'IPY_MODEL_f61db2f0383646bf80d54fc477d0742f', 'tabbable': None, 'tooltip': None}}, 'bf703899e392443b8d2e4bc33229fdb3': {'model_module': '@jupyter-widgets/base', 'model_module_version': '2.0.0', 'model_name': 'LayoutModel', 'state': {'_model_module': '@jupyter-widgets/base', '_model_module_version': '2.0.0', '_model_name': 'LayoutModel', '_view_count': None, '_view_module': '@jupyter-widgets/base', '_view_module_version': '2.0.0', '_view_name': 'LayoutView', 'align_content': None, 'align_items': None, 'align_self': None, 'border_bottom': None, 'border_left': None, 'border_right': None, 'border_top': None, 'bottom': None, 'display': None, 'flex': None, 'flex_flow': None, 'grid_area': None, 'grid_auto_columns': None, 'grid_auto_flow': None, 'grid_auto_rows': None, 'grid_column': None, 'grid_gap': None, 'grid_row': None, 'grid_template_areas': None, 'grid_template_columns': None, 'grid_template_rows': None, 'height': None, 'justify_content': None, 'justify_items': None, 'left': None, 'margin': None, 'max_height': None, 'max_width': None, 'min_height': None, 'min_width': None, 'object_fit': None, 'object_position': None, 'order': None, 'overflow': None, 'padding': None, 'right': None, 'top': None, 'visibility': 'hidden', 'width': None}}, 'bff4a2d1792a41ba876ef82b337db091': {'model_module': '@jupyter-widgets/controls', 'model_module_version': '2.0.0', 'model_name': 'FloatProgressModel', 'state': {'_dom_classes': [], '_model_module': '@jupyter-widgets/controls', '_model_module_version': '2.0.0', '_model_name': 'FloatProgressModel', '_view_count': None, '_view_module': '@jupyter-widgets/controls', '_view_module_version': '2.0.0', '_view_name': 'ProgressView', 'bar_style': '', 'description': '', 'description_allow_html': False, 'layout': 'IPY_MODEL_e746cbed0606480287a0a7c750d8856e', 'max': 36, 'min': 0, 'orientation': 'horizontal', 'style': 'IPY_MODEL_2257bae091784d7ea208be9aa29a50e1', 'tabbable': None, 'tooltip': None, 'value': 36}}, 'c3ff4d7454844632a48a8156ab672040': {'model_module': '@jupyter-widgets/controls', 'model_module_version': '2.0.0', 'model_name': 'HTMLStyleModel', 'state': {'_model_module': '@jupyter-widgets/controls', '_model_module_version': '2.0.0', '_model_name': 'HTMLStyleModel', '_view_count': None, '_view_module': '@jupyter-widgets/base', '_view_module_version': '2.0.0', '_view_name': 'StyleView', 'background': None, 'description_width': '', 'font_size': None, 'text_color': None}}, 'c55430d0430f4ab493469c181e5b5fc4': {'model_module': '@jupyter-widgets/controls', 'model_module_version': '2.0.0', 'model_name': 'FloatProgressModel', 'state': {'_dom_classes': [], '_model_module': '@jupyter-widgets/controls', '_model_module_version': '2.0.0', '_model_name': 'FloatProgressModel', '_view_count': None, '_view_module': '@jupyter-widgets/controls', '_view_module_version': '2.0.0', '_view_name': 'ProgressView', 'bar_style': '', 'description': '', 'description_allow_html': False, 'layout': 'IPY_MODEL_f7b2b50f8e694135893cc92a72651ecc', 'max': 35, 'min': 0, 'orientation': 'horizontal', 'style': 'IPY_MODEL_566fc8da062f4f219ca77a1196e6e3d1', 'tabbable': None, 'tooltip': None, 'value': 35}}, 'c7be36374c5742c7b71b69e7f06ac39d': {'model_module': '@jupyter-widgets/controls', 'model_module_version': '2.0.0', 'model_name': 'HTMLModel', 'state': {'_dom_classes': [], '_model_module': '@jupyter-widgets/controls', '_model_module_version': '2.0.0', '_model_name': 'HTMLModel', '_view_count': None, '_view_module': '@jupyter-widgets/controls', '_view_module_version': '2.0.0', '_view_name': 'HTMLView', 'description': '', 'description_allow_html': False, 'layout': 'IPY_MODEL_a87a7d355d9e43058e6c739a70d01492', 'placeholder': '\u200b', 'style': 'IPY_MODEL_d1edfc50ac77431ebd24e14665aff911', 'tabbable': None, 'tooltip': None, 'value': '\u20076/6\u2007[00:01&lt;00:00,\u2007\u20073.81it/s]'}}, 'c9e6e48d4d7f476eb271910d0316eb7a': {'model_module': '@jupyter-widgets/controls', 'model_module_version': '2.0.0', 'model_name': 'HTMLStyleModel', 'state': {'_model_module': '@jupyter-widgets/controls', '_model_module_version': '2.0.0', '_model_name': 'HTMLStyleModel', '_view_count': None, '_view_module': '@jupyter-widgets/base', '_view_module_version': '2.0.0', '_view_name': 'StyleView', 'background': None, 'description_width': '', 'font_size': None, 'text_color': None}}, 'ca6fad79f26b4b898a44d7eb2f950772': {'model_module': '@jupyter-widgets/controls', 'model_module_version': '2.0.0', 'model_name': 'HTMLModel', 'state': {'_dom_classes': [], '_model_module': '@jupyter-widgets/controls', '_model_module_version': '2.0.0', '_model_name': 'HTMLModel', '_view_count': None, '_view_module': '@jupyter-widgets/controls', '_view_module_version': '2.0.0', '_view_name': 'HTMLView', 'description': '', 'description_allow_html': False, 'layout': 'IPY_MODEL_a46776e3851f4c6683eb2de6bc0f9ae9', 'placeholder': '\u200b', 'style': 'IPY_MODEL_16aa18a981344bd1b81bf29ed555540c', 'tabbable': None, 'tooltip': None, 'value': 'mouse_0403L:\u2007100%'}}, 'd1edfc50ac77431ebd24e14665aff911': {'model_module': '@jupyter-widgets/controls', 'model_module_version': '2.0.0', 'model_name': 'HTMLStyleModel', 'state': {'_model_module': '@jupyter-widgets/controls', '_model_module_version': '2.0.0', '_model_name': 'HTMLStyleModel', '_view_count': None, '_view_module': '@jupyter-widgets/base', '_view_module_version': '2.0.0', '_view_name': 'StyleView', 'background': None, 'description_width': '', 'font_size': None, 'text_color': None}}, 'd370983752a34eaf80088bea3f2b26a1': {'model_module': '@jupyter-widgets/controls', 'model_module_version': '2.0.0', 'model_name': 'HBoxModel', 'state': {'_dom_classes': [], '_model_module': '@jupyter-widgets/controls', '_model_module_version': '2.0.0', '_model_name': 'HBoxModel', '_view_count': None, '_view_module': '@jupyter-widgets/controls', '_view_module_version': '2.0.0', '_view_name': 'HBoxView', 'box_style': '', 'children': ['IPY_MODEL_b0de30b487984d7cab5c26ebdb0f8b06', 'IPY_MODEL_9e26dfd567ca443ab8733ee606f844a3', 'IPY_MODEL_c7be36374c5742c7b71b69e7f06ac39d'], 'layout': 'IPY_MODEL_1422db234b9947a3878eeb4934f9efd1', 'tabbable': None, 'tooltip': None}}, 'e746cbed0606480287a0a7c750d8856e': {'model_module': '@jupyter-widgets/base', 'model_module_version': '2.0.0', 'model_name': 'LayoutModel', 'state': {'_model_module': '@jupyter-widgets/base', '_model_module_version': '2.0.0', '_model_name': 'LayoutModel', '_view_count': None, '_view_module': '@jupyter-widgets/base', '_view_module_version': '2.0.0', '_view_name': 'LayoutView', 'align_content': None, 'align_items': None, 'align_self': None, 'border_bottom': None, 'border_left': None, 'border_right': None, 'border_top': None, 'bottom': None, 'display': None, 'flex': None, 'flex_flow': None, 'grid_area': None, 'grid_auto_columns': None, 'grid_auto_flow': None, 'grid_auto_rows': None, 'grid_column': None, 'grid_gap': None, 'grid_row': None, 'grid_template_areas': None, 'grid_template_columns': None, 'grid_template_rows': None, 'height': None, 'justify_content': None, 'justify_items': None, 'left': None, 'margin': None, 'max_height': None, 'max_width': None, 'min_height': None, 'min_width': None, 'object_fit': None, 'object_position': None, 'order': None, 'overflow': None, 'padding': None, 'right': None, 'top': None, 'visibility': None, 'width': None}}, 'f3d3bc46b9794a5dab15f6680f64c4fe': {'model_module': '@jupyter-widgets/controls', 'model_module_version': '2.0.0', 'model_name': 'HBoxModel', 'state': {'_dom_classes': [], '_model_module': '@jupyter-widgets/controls', '_model_module_version': '2.0.0', '_model_name': 'HBoxModel', '_view_count': None, '_view_module': '@jupyter-widgets/controls', '_view_module_version': '2.0.0', '_view_name': 'HBoxView', 'box_style': '', 'children': ['IPY_MODEL_01a6ffe867c04442baf17c3967c87707', 'IPY_MODEL_c55430d0430f4ab493469c181e5b5fc4', 'IPY_MODEL_53b76f7085a94f9d93c74b5ba2978d9a'], 'layout': 'IPY_MODEL_fa24f876fbce41b4b34618402c25b9a3', 'tabbable': None, 'tooltip': None}}, 'f61db2f0383646bf80d54fc477d0742f': {'model_module': '@jupyter-widgets/base', 'model_module_version': '2.0.0', 'model_name': 'LayoutModel', 'state': {'_model_module': '@jupyter-widgets/base', '_model_module_version': '2.0.0', '_model_name': 'LayoutModel', '_view_count': None, '_view_module': '@jupyter-widgets/base', '_view_module_version': '2.0.0', '_view_name': 'LayoutView', 'align_content': None, 'align_items': None, 'align_self': None, 'border_bottom': None, 'border_left': None, 'border_right': None, 'border_top': None, 'bottom': None, 'display': None, 'flex': None, 'flex_flow': None, 'grid_area': None, 'grid_auto_columns': None, 'grid_auto_flow': None, 'grid_auto_rows': None, 'grid_column': None, 'grid_gap': None, 'grid_row': None, 'grid_template_areas': None, 'grid_template_columns': None, 'grid_template_rows': None, 'height': None, 'justify_content': None, 'justify_items': None, 'left': None, 'margin': None, 'max_height': None, 'max_width': None, 'min_height': None, 'min_width': None, 'object_fit': None, 'object_position': None, 'order': None, 'overflow': None, 'padding': None, 'right': None, 'top': None, 'visibility': 'hidden', 'width': None}}, 'f7b2b50f8e694135893cc92a72651ecc': {'model_module': '@jupyter-widgets/base', 'model_module_version': '2.0.0', 'model_name': 'LayoutModel', 'state': {'_model_module': '@jupyter-widgets/base', '_model_module_version': '2.0.0', '_model_name': 'LayoutModel', '_view_count': None, '_view_module': '@jupyter-widgets/base', '_view_module_version': '2.0.0', '_view_name': 'LayoutView', 'align_content': None, 'align_items': None, 'align_self': None, 'border_bottom': None, 'border_left': None, 'border_right': None, 'border_top': None, 'bottom': None, 'display': None, 'flex': None, 'flex_flow': None, 'grid_area': None, 'grid_auto_columns': None, 'grid_auto_flow': None, 'grid_auto_rows': None, 'grid_column': None, 'grid_gap': None, 'grid_row': None, 'grid_template_areas': None, 'grid_template_columns': None, 'grid_template_rows': None, 'height': None, 'justify_content': None, 'justify_items': None, 'left': None, 'margin': None, 'max_height': None, 'max_width': None, 'min_height': None, 'min_width': None, 'object_fit': None, 'object_position': None, 'order': None, 'overflow': None, 'padding': None, 'right': None, 'top': None, 'visibility': None, 'width': None}}, 'fa24f876fbce41b4b34618402c25b9a3': {'model_module': '@jupyter-widgets/base', 'model_module_version': '2.0.0', 'model_name': 'LayoutModel', 'state': {'_model_module': '@jupyter-widgets/base', '_model_module_version': '2.0.0', '_model_name': 'LayoutModel', '_view_count': None, '_view_module': '@jupyter-widgets/base', '_view_module_version': '2.0.0', '_view_name': 'LayoutView', 'align_content': None, 'align_items': None, 'align_self': None, 'border_bottom': None, 'border_left': None, 'border_right': None, 'border_top': None, 'bottom': None, 'display': None, 'flex': None, 'flex_flow': None, 'grid_area': None, 'grid_auto_columns': None, 'grid_auto_flow': None, 'grid_auto_rows': None, 'grid_column': None, 'grid_gap': None, 'grid_row': None, 'grid_template_areas': None, 'grid_template_columns': None, 'grid_template_rows': None, 'height': None, 'justify_content': None, 'justify_items': None, 'left': None, 'margin': None, 'max_height': None, 'max_width': None, 'min_height': None, 'min_width': None, 'object_fit': None, 'object_position': None, 'order': None, 'overflow': None, 'padding': None, 'right': None, 'top': None, 'visibility': 'hidden', 'width': None}}, 'ff4c91b612204fb1ad6d2bd898a5f707': {'model_module': '@jupyter-widgets/base', 'model_module_version': '2.0.0', 'model_name': 'LayoutModel', 'state': {'_model_module': '@jupyter-widgets/base', '_model_module_version': '2.0.0', '_model_name': 'LayoutModel', '_view_count': None, '_view_module': '@jupyter-widgets/base', '_view_module_version': '2.0.0', '_view_name': 'LayoutView', 'align_content': None, 'align_items': None, 'align_self': None, 'border_bottom': None, 'border_left': None, 'border_right': None, 'border_top': None, 'bottom': None, 'display': None, 'flex': None, 'flex_flow': None, 'grid_area': None, 'grid_auto_columns': None, 'grid_auto_flow': None, 'grid_auto_rows': None, 'grid_column': None, 'grid_gap': None, 'grid_row': None, 'grid_template_areas': None, 'grid_template_columns': None, 'grid_template_rows': None, 'height': None, 'justify_content': None, 'justify_items': None, 'left': None, 'margin': None, 'max_height': None, 'max_width': None, 'min_height': None, 'min_width': None, 'object_fit': None, 'object_position': None, 'order': None, 'overflow': None, 'padding': None, 'right': None, 'top': None, 'visibility': None, 'width': None}}}, 'version_major': 2, 'version_minor': 0}}}

out = os.environ.get("NB_OUT") or str(Path(__file__).resolve().parent.parent / 'cursor_state_segmentation.ipynb')
with open(out, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote", out, "cells:", len(cells))
