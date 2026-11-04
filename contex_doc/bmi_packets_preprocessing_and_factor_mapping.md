# What the packet's neural data actually is, and how the decoder axes map onto it

Answers two questions:

1. **What preprocessing happens from raw calcium signal to `data_neural_aligned`?**
2. **How do the factors in `weights_day0_idxFactors` map into that neural space, and is a given
   factor the same direction across days?**

Companion to [`bmi_packets_pipeline.md`](bmi_packets_pipeline.md) (where each file comes from) and
[`3_define_decoder_SPCA_SINGLEDAY_explained.md`](3_define_decoder_SPCA_SINGLEDAY_explained.md)
(how the factors were built). Numbers marked **[verified]** were measured directly from
`C:\Data\datasets\bmi_packets`.

---

# 1. Preprocessing: raw → `data_neural_aligned`

## Short answer

**`data_neural_aligned[date]` is suite2p's deconvolved `spks`, with rows re-indexed by ROICaT UCID.
Nothing else is done to it.** It is *not* dF/F. Between suite2p and the packet there is no
neuropil subtraction, no ΔF/F normalisation, no z-scoring, no detrending, no smoothing, no binning,
no resampling, and no trace-quality filtering.

## The chain

```
movie ──suite2p──► F, Fneu ──suite2p dcnv (OASIS)──► spks.npy ──np.load──► rows permuted by UCID ──► packet
```

Everything that shapes the signal happens **inside suite2p**:

```
Fc  = F - neucoeff * Fneu                 # neucoeff lives in ops.npy
F0  = baseline filter of Fc               # ops['baseline'], 'win_baseline', 'sig_baseline'
spks = OASIS(Fc - F0; tau = ops['tau'])   # non-negative, L1-sparse deconvolution
```

⚠ `ops.npy` is **not** in the packet, so these parameters are not auditable from the packet alone.

Downstream of suite2p, in `prep_BMI_data_for_Sam.ipynb`:

```python
def load_data_neural(path):
    return np.load(path, mmap_mode='r')          # cell 33 — no transformation

data_neural_aligned, idx_inverse_ucids = roicat.util.match_arrays_with_ucids(...)   # cell 55
```

`match_arrays_with_ucids` allocates a NaN array of `max_ucid` rows and copies in rows with UCID ≥ 0.
**It only permutes and selects rows.**

A z-scoring helper *is* defined (cell 89) —

```python
def preprocess_neural_data(data):
    out = torch.as_tensor(data, dtype=torch.float32)
    return (out - torch.nanmean(out, -1, keepdims=True)) / bnpm.torch_helpers.nanstd(out, -1, keepdims=True)
data_neural_preprocessed = {date: preprocess_neural_data(d) for date, d in ...}   # cell 90
```

— but `data_neural_preprocessed` is **never saved**. Only `data_neural_aligned` goes into the packet.

## Evidence that it is `spks` and not dF/F

1. **The stored provenance path** **[verified]** —
   `extras/paths/paths_neural[date]` = `.../face_rhythm/{mouse}/spks/{date}/spks.npy`.
   (The commented-out cell 13 shows an earlier version that used `val_dFoF__idx_SI4x.npy`; the
   active cell 14 globs `spks\.npy`.)
2. **The value distribution** (0403L, 20230703) **[verified]** — strictly non-negative, **61.7 % of
   samples exactly 0**, max ≈ 2667, median of non-zero values ≈ 37, zero NaNs. dF/F would be signed
   and continuous; this is OASIS output in raw-fluorescence units.
3. **It is not CASCADE.** `NBAP/pipeline_res2p_BMI/CASCADE/run_cascade.py` saves
   `spike_prob.npy`, a different filename, and CASCADE emits NaN edges — the packet has none.

## Neuron selection that *was* applied

Four filters, all in `prep_BMI_data_for_Sam.ipynb` cells 36–40, all acting on UCIDs:

1. ROICaT drawn classifier `preds[-1]` must pass **on every session**
2. `sample_silhouette > 0.0`
3. `cluster_silhouette > -0.2`
4. `discard_UCIDs_with_fewer_matches(n_sesh_thresh='all')` — present in **all** sessions

→ 0403L: **468** neurons on all 36 days **[verified]**.

### ⚠ `iscell_tqm` is loaded, stored, and never applied

The trace-quality mask is in `extras/tracking/iscell_tqm[date]` but is absent from the filtering
chain. Measured on 0403L **[verified]**:

- **267 / 468 (57.1 %)** of packet neurons fail trace quality on at least one day
- **9.7 %** of all neuron-days are tqm failures
- one neuron fails on **all 36 days** and is still in the packet

If you want trace quality enforced, apply it yourself — map UCID → that day's ROI index with
`extras/tracking/labels_bySession_modified[date]`, then index `iscell_tqm[date]`.

## Temporal structure

**One row = one ScanImage frame, on the same clock as everything else.** No resampling anywhere.
`data_neural_aligned`, `data_ws` and the logger tables all carry one sample per frame
**[verified]**: 108,000 / 108,000 / 108,000 on 0403L day 0.

Exception **[verified]**: 0403L on **20230728, 20230730, 20230731** the logger has 108,000 rows
against 144,000 neural frames. Alignment holds from frame 0; the logger is truncated and the last
36,000 frames have none.

## Practical consequences

- **Units are arbitrary fluorescence**, per-ROI, not comparable to dF/F or to spikes/s.
- **Highly non-Gaussian** — a ~62 % zero-inflated, heavy-tailed distribution. Means, variances, PCA
  and correlation are all dominated by the sparse large events. Consider `log1p`, a square root, or
  binning before anything that assumes roughly Gaussian data.
- **Deconvolution is a nonlinear, per-neuron operation.** Two neurons with identical dF/F but
  different inferred `tau` or baseline get different `spks`. Cross-neuron weighted sums (which is
  what the decoder axes are) are therefore not the same object they were in dF/F.
- **Day-to-day scale is stable but not constant** **[verified]**, 0403L across 36 days:
  mean 22.0–32.4, std 55.7–83.8, zero-fraction 0.586–0.624 — roughly ±20 %. Good enough for most
  comparisons, but worth normalising for anything about magnitude.

---

# 2. How the factors map into packet neural space

## The index chain

```
factors_final             (K, n_iscell_day0)   rows unit-norm, zero-mean over neurons, orthogonal
   └ .T  ────────────────► weights_day0        (n_iscell_day0, K)       0403L: (1132, 5)
       └ scatter by iscell_day0 ─────────────► weights_day0_idxDay0     (n_ROIs_day0,  K)  (3164, 5)
           └ [ucids_inverse[date_day0]] ─────► weights_day0_idxFactors  (n_ucid,       K)  ( 468, 5)
```

I reproduced the last step **bit-for-bit** from the packet's own `weights_day0_idxDay0` and
`labels_bySession_modified[date_day0]` **[verified]**. `ucids_inverse` has no `-1` entries, because
`n_sesh_thresh='all'` guarantees every retained UCID exists on day 0.

`weights_day0` is confirmed to carry the decoder notebook's geometry **[verified]**: column L2 norms
exactly 1.0, column means ~0 (each factor ⟂ the all-ones vector), and
**max |off-diagonal Gram| ≈ 10⁻¹⁵** — exactly orthogonal.

## Is a factor the same direction across days?

### Formally, yes — by construction

There is **one** weight matrix for the whole packet. Nothing is re-fit per day. Row *i* of every
day's neural matrix is UCID *i* **[verified]**. So `weights_day0_idxFactors[:, k]` is literally the
same linear functional of the same 468 slots on all 36 days. **Any apparent across-day change in a
factor's activity is a change in the neurons, not in the readout.**

### In practice there are four caveats, all measurable

#### (a) The axis in packet space is a *truncated* version of the real axis

Only the rows with a day-0 weight count: 0403L has **90 / 468 NaN rows**, and the surviving 378
cover only **33.4 %** of the 1132 day-0 decoder neurons. Worse, each factor loses a *different*
amount. Fraction of each factor's squared norm retained **[verified]**:

| mouse | K | decoder neurons | tracked | with weight | coverage | squared norm retained per factor | max off-orthogonal |
|---|---|---|---|---|---|---|---|
| **0403L** | 5 | 1132 | 468 | 378 | 33.4 % | 0.538, 0.618, 0.495, **0.279**, 0.797 | 8.9° |
| **0403R** | 3 | 510 | 589 | 289 | 56.7 % | 0.749, 0.717, 0.828 | 3.6° |
| **0322R** | 5 | 238 | 111 | **52** | **21.8 %** | 0.757, 0.675, 0.724, 0.761, **0.108** | **26.0°** |
| **0916N** | 6 | 497 | 768 | 420 | **84.5 %** | 0.957, 0.943, 0.882, 0.920, 0.965, 0.794 | 3.6° |

Read this as: for 0403L factor 3 (0-indexed), **72 % of the axis lives in neurons that are not in
the packet at all**. For 0322R factor 4, **89 %** is missing and only 52 neurons carry any weight —
factor-space analyses of 0322R are on very thin ice. 0916N is the clean case.

#### (b) The axes are no longer orthonormal in packet space

Subsetting to the weighted rows destroys both properties. For 0403L the Gram matrix of the 378
weighted rows **[verified]**:

```
[[ 0.538  0.089 -0.004  0.050  0.057]
 [ 0.089  0.618  0.012 -0.010 -0.039]
 [-0.004  0.012  0.495 -0.007 -0.072]
 [ 0.050 -0.010 -0.007  0.279 -0.026]
 [ 0.057 -0.039 -0.072 -0.026  0.797]]
```

Diagonals are 0.28–0.80 (not 1), and the largest pairwise cosine is 0.154 — ~9° from orthogonal
(0322R: 26°). **If you need an orthonormal basis, re-orthonormalise explicitly, and be aware that
doing so changes the axes.** Comparing raw projection magnitudes *between* factors is already
confounded by the unequal diagonals.

#### (c) The signal is different from the one the axes were fit on

The factors were fit on **pre-orthogonalised dF/F**; the packet holds **deconvolved spks**. Add
that the online cursor used a *third* signal — BWAIN's own online dF/F with a 10-min rolling 30th
percentile baseline, computed from day-0 footprints warped into today's FOV — and you have three
different neural representations of the same session.

#### (d) Online neuron identity ≠ packet neuron identity

| | online (BWAIN) | packet |
|---|---|---|
| "same neuron" means | same **day-0 pixel mask**, non-rigidly warped into today's FOV via `imregdemons` | same **ROICaT UCID** cluster of that day's own suite2p ROIs |
| re-derived | every session, by `AlignSpatialFootprints2_day0.m` | once, by the tracking run |
| uses ROICaT | **no** | yes |

These are independent correspondence rules that will disagree for some neurons on some days, and
nothing in the packet records the disagreement.

### Measured fidelity: how well does the packet reproduce the real cursor?

Direct test — project the packet's spks onto `weights_day0_idxFactors`, apply the same angle gate
the rig uses (`D_k = |CS_k|² · M_k`, with `CS_k = M_k/‖M‖` over the K factors), smooth with a
15-frame boxcar, and correlate against the logged `cursor_brain`. mouse_0403L **[verified]**:

| days | factor selection | r vs `cursor_brain` |
|---|---|---|
| 20230705–20230724 (single-cursor) | fixed, python idx 1 | **0.69 – 0.85**, median ≈ 0.80 |
| 20230725–20230808 (remapping) | fixed, python idx 1 | 0.13 – 0.37 ← **wrong** |
| 20230725–20230808 (remapping) | per-frame, `cursor_factor_to_use − 1` | **0.53 – 0.73** |

So the packet recovers the real cursor **well but not exactly** — r ≈ 0.8, not ≈ 1. The residual is
(a)+(b)+(c)+(d) above. The angle-gated `D` consistently beats the raw projection `M` (e.g. 0.803 vs
0.775 on 20230706), confirming the gating form is right.

No other factor correlates with the cursor on single-cursor days (|r| ≤ 0.32 for all of 0, 2, 3, 4),
so the identification is unambiguous.

### Factor indexing — the one thing most likely to bite you

`cursor_factor_to_use` in the logger is **MATLAB 1-indexed**. The Python column is
**`cursor_factor_to_use − 1`**.

Verified for mouse_0403L **[verified]**:

- 20230703–20230724: `cursor_factor_to_use = 2` → **python column 1**, threshold 1.9 (1.5 for the
  first two days)
- 20230725–20230808: `{2, 3}` **within the same session**, selected per block by
  `sm.cursor = baselineStuff.cursors(sm.blocks.cursor_to_use)` — threshold **1.9 for factor 2**,
  **0.9 for factor 3**

So on remapping days `cursor_brain` is **not one axis**: it switches between python columns 1 and 2
block by block. Any analysis that treats `cursor_brain` as a single signal across a remap session is
mixing two different axes. Use the per-frame `cursor_factor_to_use` column, and compare each cursor
to **its own factor's threshold**.

`cursor_angle_power = 2.0` and `cursor_win_smooth_cursor = 1` (i.e. no smoothing) throughout.

---

## Recipe: reconstruct the decoder signals from the packet

```python
X = np.asarray(r['data_neural_aligned'][day].load(), dtype=np.float64)   # (n_ucid, T) spks
W = np.asarray(r['weights_day0_idxFactors'].load())                      # (n_ucid, K), has NaN rows
ok = ~np.isnan(W).any(1)                                                 # 378/468 for 0403L

M  = W[ok].T @ X[ok]                                   # (K, T) projection magnitudes
CS = M / np.linalg.norm(M, axis=0, keepdims=True)      # cosine sim in factor space
D  = (np.abs(CS) ** 2.0) * M                           # angle-gated cursor signals

ts  = r['extras']['logger_dfs'][day]['timeSeries'].load()      # or ['loggers'] for 0322R/0916N
col = ts['cursor_factor_to_use'].to_numpy() - 1                # MATLAB 1-indexed -> python
cursor = D[np.nan_to_num(col, nan=1).astype(int), np.arange(D.shape[1])]
```

Caveats to carry with the output: `D` is in spks units (not dF/F, not comparable to
`cursor_threshold_reward`); the axes are not orthonormal (§2b); the online version NaNs out
violating neurons per frame (`F_baseline < 100` or `dFoF > 15`) and uses `nansum`, which has no
equivalent here; and `len(ts)` may be shorter than `X.shape[1]` on 0403L 20230728/30/31.

## Checklist when interpreting factor-space results

- [ ] Convert `cursor_factor_to_use` with **−1** before indexing columns.
- [ ] On remap days, select the factor **per frame**, and threshold each against its own value.
- [ ] Report coverage and retained norm for the mouse and factor you are using (table in §2a) —
      conclusions about 0322R factor 4 (10.8 % retained, 52 neurons) carry far less weight than
      0916N (79–97 %).
- [ ] Decide explicitly whether you want the axes re-orthonormalised; don't assume they are.
- [ ] Don't compare raw projection magnitudes between factors (unequal Gram diagonals) or across
      days (spks scale drifts ~±20 %, and 0403L's factor-1 projection SD falls ~35 % over 36 days)
      without normalising.
- [ ] Apply `extras/tracking/iscell_tqm` yourself if trace quality matters — it was never applied.
- [ ] Remember the packet's neural data is deconvolved, non-negative, ~62 % zeros — transform before
      Gaussian-assuming methods.

---

# 3. Addendum: is a factor the *same direction* across days in my z-scored analysis space?

Section 2 answered this formally ("yes, one weight matrix, nothing re-fit"). That is true but
incomplete, because `neural_analysis.ipynb`'s `preprocess_day` applies a **per-day** transform:

```python
X = na.bin_and_zscore(neural, BIN_SIZE_FRAMES, zscore=True)   # divides each neuron by its WITHIN-DAY std
```

A readout `f` defined on raw data becomes `σ_d ⊙ f` in z-scored coordinates, which *is* day-dependent.
So the question is real. Everything below is measured on mouse_0403L, 20 stable days
(`factor_to_use=2` → python column 1), ground truth = logged `cursor_brain` **[verified]**.

## 3.1 The spks-vs-dF/F mismatch is self-cancelling — do not reverse it

Scan of `readout = f1 ⊙ σ_d^α` applied to the z-scored `X`, correlated with the logged cursor:

| α | −1.0 | −0.5 | −0.25 | **0.0** | +0.25 | +0.5 | +1.0 |
|---|---|---|---|---|---|---|---|
| r | 0.761 | 0.790 | 0.802 | **0.809** | 0.815 | 0.809 | 0.758 |

`α = 0` — i.e. **apply `f1` as-is to the z-scored data** — is within 0.006 of the optimum, and clearly
beats `α = +1` (0.758), the value the naive change-of-variables argument predicts.

**Why.** `spks` is in raw-fluorescence units, so `σ_i^spks ∝ B_i` (neuron *i*'s baseline brightness).
dF/F divides by exactly that `B_i`. Per-day z-scoring therefore performs, to good approximation, the
same per-neuron normalisation the online decoder operated in. **No inverse transform is required.**
(Per-neuron log-σ profiles correlate r = 0.858 between days, so σ is dominated by stable brightness.)

## 3.2 Measured: the direction does not change across days

Readout fit independently on each day from the logged cursor, **inside the 4-PC plane** (4 free
parameters — well-conditioned):

| quantity | value |
|---|---|
| between-day angle (4-D) | median **8.6°**, max 18.2° |
| **within-day split-half floor** | **7.4°** |
| PC1–PC2 arrow angle, 20 days | median 5.5°, full range **12°** |
| PC1–PC3 arrow angle, 20 days | median 37.6°, full range 18° |

Between-day ≈ the within-day noise floor. **The direction is the same across days.**

⚠ The same fit in the **full 468-dim** space gives 54° between days — but its within-day floor is
**44°**, so that number is estimation noise, not drift. Cross-day prediction r = 0.859 against a
0.901 within-day held-out ceiling. Do not quote high-dim readout angles without a split-half floor.

## 3.3 Plotting the axis in a PC figure

```python
f1 = np.nan_to_num(r["weights_day0_idxFactors"].load(), nan=0.0)[:, 1]   # python col 1 = cursor axis
a  = common.components @ (f1 / np.linalg.norm(f1))      # NOT common.project() — an arrow has no offset
```

mouse_0403L: `a = [0.892, −0.290, 0.346, −0.012]` (unit). **One arrow for all days.**

Only **10.6 %** of `f1`'s squared length lies in the 4-PC plane — but that shadow predicts the cursor
at r = 0.804 vs 0.809 for the full axis (**99 %** of the explained variance). The arrow is
behaviourally faithful. Per panel, though:

| panel | in-plane length | draw it? |
|---|---|---|
| PC1–PC2 | 0.305 | yes |
| PC1–PC3 | 0.311 | yes |
| PC1–PC4 | 0.290 | yes |
| PC2–PC3 | 0.147 | **no** |
| PC2–PC4 | 0.094 | **no** |
| PC3–PC4 | 0.113 | **no** |

The factor is essentially PC1. In panels without PC1 the in-plane component is small and its angle is
unstable across days (114–174° swing under the σ-corrected version) — an arrow there invites
over-reading.

**The stored arrow carries a constant bias.** Empirical arrow: PC1–PC2 = +5.5°. Stored `f1` arrow:
−18.0°. Offset ≈ 28° (21–35° across days) — a *bias*, not a drift, from the 33 % neuron truncation
(§2a) plus residual spks/dF/F plus online-vs-ROICaT identity (§2d). For an **accurate** axis, fit it
from the logged cursor; for the decoder's **nominal** axis, use `f1` and label it as such.

**If you ever re-fit PCA per day**, the arrow's PC coordinates will change — but that is a change of
basis, not a change of direction. Only a single across-day `common` subspace makes per-day arrows
comparable.

## 3.4 A real slow rotation sits on top of the noise

Across the 20 stable days, monotone and significant (Spearman vs day index):

| measure | rho | p |
|---|---|---|
| PC1–PC2 angle | +0.58 | 0.007 |
| PC1–PC3 angle | −0.72 | 3.4e−4 |
| PC1 component | +0.66 | 0.0015 |

The readout axis rotates ~10–18° toward PC1 over three weeks. Per-day this is below the 7.4° floor,
but the monotone trend is not. Relevant to the intuitive-axis consolidation claim.

## 3.5 Other mice

0916N (84.5 % coverage) is the clean case. 0403R (56.7 %) is usable but has only 4 stable days.
**0322R has 52 weighted neurons / 21.8 % coverage and 26° off-orthogonal axes — do not draw factor
arrows for it.**
