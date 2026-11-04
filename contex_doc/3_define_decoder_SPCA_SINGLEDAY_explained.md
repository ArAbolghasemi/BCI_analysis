# `3_define_decoder_SPCA_SINGLEDAY.ipynb` — what it does

Rich Hakim's **decoder-definition** notebook. It runs **once per mouse, on the day-0 session
(before any BMI training)** and produces the fixed neuron→cursor weight matrix that the online
MATLAB rig then uses for every subsequent day. Everything you analyse downstream is a consequence
of the choices made here.

As-run values below are from the committed outputs: **mouse_0403L, 20230703**, FOV 512×1024,
108,000 frames (≈60 min at 30 Hz), GPU.

Source for `bmi_idap.decoder` / `bmi_idap.helpers` is not in this repo — it is on the
`pc-decoder` branch of `RichieHakim/BMI_IDAP`, which is where the math below is read from.

---

## 1. Inputs

| Input | Role |
|---|---|
| suite2p `plane0` (`F`, `Fneu`, `stat`, `ops`) | raw traces + ROI masks |
| ROICaT classification `.pkl` (`preds[-1]`) | is-cell, from the hand-drawn/NN classifier |
| `trace_quality.pkl` | second is-cell mask (`iscell_tqm`) **and** the dF/F parameters |
| z-stack TIFF (ScanImage) | axial sensitivity curves — **computed but never applied** (see §6) |

Neuron selection is the **AND** of both masks:
`iscell = roicat_classification['preds'][-1] * trace_quality['iscell_tqm']`.
Everything after this point lives in that reduced neuron set.

---

## 2. dF/F

`ca2p_preprocessing.make_dFoF` on `F[iscell]`, `Fneu[iscell]`, i.e.
`(F − c − α·Fneu − F₀)/F₀` with:

- `neuropil_fraction` α, `percentile_baseline`, `channelOffset_correction` c — all **inherited from
  `trace_quality.pkl`**, not set here (the z-stack cell hard-codes α = 0.7, the usual value)
- `F₀` = **rolling percentile over a 10-min window** (`10*60*Fs` frames), stride 10 frames

No timepoints excluded: `idx_time_bad = np.arange(0,0)` → `num excluded time points: 0`.

---

## 3. Pre-orthogonalisation (nuisance removal)

Builds a regressor matrix `sigs` and OLS-projects it out of `dFoF`:

1. the **mean population trace**, split into 5 FIR bands with boundaries
   `[0, 1/300, 1/20, 1/2, 5, ∞] Hz` (10,001-tap Kaiser, zero-phase `filtfilt`)
2. the "mean neuropil trace", **identically band-split** — but see §6, this is a duplicate
3. a linear ramp −1→1
4. a constant (becomes all-zero after the subsequent mean-centring — inert)
5. 4 M-spline basis functions (order 3) spanning the session

Because the 5 bands **tile the whole spectrum**, this removes the mean population trace entirely,
but with a *separate coefficient per frequency band per neuron* — a crude frequency-dependent
global-signal regression, more flexible than regressing out one trace. The ramp + splines absorb
slow session drift.

Output is re-baselined per neuron (`− percentile_baseline` of the residual) → **`dFoF_orth`**.

> Run value: **`EVR_total = 0.0713`** — 7.1 % of total dF/F variance removed.

---

## 4. Factors: SPCA → ridge → orthogonalise → normalise → flip → order

### 4a. Sparse PCA
`sklearn.decomposition.SparsePCA(n_components=6, ridge_alpha=0.01, method='lars')` on
`dFoF_orth` (mean-centred), warm-started from the GPU SVD. `alpha` (the L1 penalty) is swept and
chosen so the **mean number of non-zero neurons per component lands near a goal of 100**:

```
alpha:  7 → 192.8 nonzero    alpha: 9 → 156.2    alpha: 12 → 117.0   ← chosen
```

Final fit at `alpha=12`, `tol=1e-6`. Non-zero neurons per component: **`[320, 150, 60, 26, 9, 8]`**.
Components with **< 30** non-zero neurons are dropped → components 3, 4, 5 excluded,
**`n_factors_included = 3`**.

### 4b. The chain that turns SPCA components into decoder weights

| step | operation | effect |
|---|---|---|
| `factors_original` | the 3 surviving SPCA rows | sparse |
| `factors_regressed` | **Ridge(neural → scores, λ = 1e7)** | **re-densifies**: ridge has no L1, so every neuron gets a weight |
| `factors_orth` | sequential Gram–Schmidt on rows (`orthogonalize_matrix`) | factor **weight vectors** mutually orthogonal |
| `factors_orth_ms` | subtract each row's mean **across neurons** | each factor ⟂ the all-ones vector |
| `..._norm` | unit L2 norm per row | sets cursor units |
| `..._flipped` | × `sign(skew(score))` | each factor oriented so transients go **up** |
| `factors_final` | reordered by EVR of `dFoF_centered`, descending | factor 0 = most variance explained |

Two of these steps matter a lot for interpretation:

- **The ridge step destroys the sparsity.** SPCA is used to *find the directions*; the weights that
  actually ship are dense. With λ = 1e7 the solution sits deep in the shrinkage regime where the
  direction → `Xᵀy`, i.e. roughly the **neuron↔score covariance ("encoding") pattern** of each SPCA
  score, rather than a decoding weight. Unit-normalisation afterwards means only the direction
  survives. (Worth confirming on your data: compare λ to `eigvalsh(XᵀX)`.)
- **Mean-subtraction across neurons** makes the cursor **blind to the population mean**. The animal
  cannot earn reward by raising everything. This is the main reason the pipeline gets away with not
  applying the §3 orthogonalisation online (§6, point 2).

> Orthogonality is in *weight space*. The resulting **score time series are still correlated**,
> because neurons are correlated — the notebook checks this with `corrcoef(scores_final)`.

---

## 5. The decoder itself (`Decoder_angle_magnitude`, p = 2.0)

With `F` = `factors_final.T` (n_neurons × 3), unit-norm columns, applied to **raw `dFoF`**:

```
M_k(t)  = X(t) · f_k                      magnitude: plain projection onto factor k
CS_k(t) = M_k(t) / ‖M(t)‖₂                cosine sim in the 3-D FACTOR-SCORE space
D_k(t)  = |CS_k(t)|^p · M_k(t)            the cursor signal
```

For p = 2 this collapses to a clean closed form:

```
D_k = M_k³ / (M_0² + M_1² + M_2²)
```

Properties worth holding onto:

- **It is not a linear readout.** `D` is cubic-over-quadratic in the projections, so linear subspace
  tools (PCA, regression onto the cursor) will not map one-to-one onto it.
- **Angular gating.** `D_k/M_k = |CS_k|^p ∈ [0,1]`. Pure factor-k state → gate 1. An equal
  three-way mixture → gate 1/3. To hit threshold with a 50/50 two-factor mix you need **twice**
  the projection you'd need from a pure state.
- **Scale-equivariant.** Doubling the population amplitude doubles `D`. The power penalises
  *mixing*, not amplitude.
- **CS is defined among the 3 included factors only** — it measures exclusivity relative to the
  other two decoder axes, not relative to the whole neural space.
- `sign(D) = sign(M)`.
- Units of `D` are dF/F-projection units (`f_k` is unit-norm), which is what the threshold 6.0 is in.

`p` is picked from the **EVR-vs-power curve** (cells 113–116): sweep `power` over `[0, 0.1…100]`
and ask how much `dFoF` variance the decoder outputs explain, custom factors vs PCA factors.
Higher power = sharper angular selectivity = less variance explained.

### Simulated experiment
`simple_cursor_simulation` replays day-0 spontaneous activity through the decoder:

- cursor = `D[idx_factor = 1]`, `thresh_reward = 6.0`
- reward after the cursor is above threshold for `duration_threshold_hold = 3` **frames** (≈100 ms)
- next trial needs quiescence — `M ≤ 10th-percentile(M₁)` **and** `D ≤ 0.6` — for 3 frames
- `win_smooth_cursor = 1` → kernel `[0,1]`, i.e. **no smoothing** in this run

> Run value: **1.233 rewards/min**. This is the knob the alpha / power / threshold choices were
> tuned against — the goal is a decoder that is reachable but not trivially reachable from
> spontaneous activity.

---

## 6. Things to keep in mind when interpreting results

**Design consequences**

1. **The decoder is frozen from one 60-min pre-training session.** Every later day's cursor is this
   fixed matrix applied to that day's neurons. Any across-day change you measure is a change in the
   *neural activity relative to a fixed readout*, never a change in the readout.
2. **Online, the decoder is applied to raw `dFoF`, not `dFoF_orth`.** The nuisance regression of §3
   shaped *where the factors point*; it is not part of the online path. The protection against
   global drift comes entirely from the factors being zero-mean across neurons.
3. **Your "neural axes" are dense, not sparse,** despite the SPCA name (§4b).
4. **The cursor is nonlinear.** `D = M³/‖M‖²`. Do not treat cursor value as a linear projection.
   The animal can raise the cursor by *suppressing the other two factors* without changing its own
   projection at all — a real behavioural strategy that a linear analysis will miss.
5. **`idx_factor = 1` here (0-indexed) = factor 2 in the MATLAB-side 1-indexed convention** — and
   the `.mat` is explicitly written 1-indexed for MATLAB. Your logged `cursor_factor_to_use` values
   (2 and 3) are consistent with intuitive axis = Python index 1 and rotated axis = Python index 2,
   both drawn from the same 3 factors. **Verify against the experiment's own params before relying
   on it** — this notebook only simulates, it does not set the online config.
6. **The simulation's timing is close to the task's timing, not far from it.** ⚠ *Corrected
   2026-11-02 — this point previously claimed "the real task uses far longer holds", which is
   wrong.* Measured from `expParams.params.trial` and the `timeSeries` flags on 0403L, 0403R and
   0916N: `duration_threshold = 3.0` and `duration_quiescenceHold = 3.0` are **frames**, the same
   ~100 ms the simulation uses. `CS_threshold` runs have a median length of 2–3 frames and the
   threshold onset precedes the reward onset by a median of 2–5 frames. What the simulation does
   *not* share with the task is the trial structure around those holds (`duration_trial = 20 s`,
   `duration_ITI = 3 s`, `duration_timeout = 4 s`), so 1.233 rewards/min remains a tuning
   diagnostic rather than a prediction of behaviour.
7. **Only 3 of 6 components survived** the 30-neuron floor. The decoder space is 3-D, and 2 of the
   3 are used as cursor axes — so the "intuitive" and "new" axes are near-neighbours in a very
   low-dimensional, mutually orthogonal, same-day-defined basis.

**Bugs / inconsistencies in the notebook (flagged, not fixed)**

- **Cell 59**: `Fneu_centered = dFoF − dFoF.mean(1)` — uses `dFoF`, not `Fneu`. The "mean neuropil"
  regressors are therefore **exact duplicates** of the mean-population regressors, so **neuropil was
  never independently orthogonalised out**. OLS absorbs the rank deficiency silently, and neuropil
  is already subtracted at the dF/F step, so the effect is bounded — but the 7.1 % EVR figure comes
  from 10 unique regressors, not 16.
- **Cells 106 & 115** index `factors_final[:-1]`, written for a disabled cell (102) that would have
  appended an all-ones row. That cell is `raw`, so `factors_final` has exactly 3 rows and these two
  cells **silently use only the first 2 factors**. The EVR-vs-power curve therefore compares
  **2 custom factors against 3 PCs** — not like-for-like. **The decoder itself (cell 120) uses all
  3 rows and is unaffected.**
- **Cell 104**: `results['factors']['scores_orth_ms_norm_raw_ordered']` is assigned
  `factors_orth_ms_norm_flipped_ordered` — a *factors* array under a *scores* key. Don't trust that
  key if you load the pickle.
- **Cell 23**: `results['data']['n_rois']` is the **pre-`iscell`** ROI count, so it does **not**
  match `factors.shape[1]`. Use the footprint/factor shapes instead.
- **Cell 65**: the `np.ones` regressor becomes all-zero after `sigs -= sigs.mean(1)`.
- **Z-stack section is dead code.** `component_z_plus/minus` are computed and stored but the
  z-displacement exclusion is commented out and `idx_time_bad` is empty — **no axial-motion
  rejection was applied** to the data the factors were fitted on.
- **Cell 82** displays the alpha = 7 sweep result, not the chosen alpha = 12 (display only).
- Scores are computed as plain dot products, **not** `spca.transform()` (which applies its own
  ridge, `ridge_alpha=0.01`). Correct for a linear decoder, but the two differ — and cells 82/88
  plot the `transform()` version.
- `scores_final` uses raw `dFoF` while `scores_orth_*` use `dFoF_orth`; `order_factors_by_EVR` ranks
  on `dFoF_centered`. Three different matrices in the same section — check which one you mean.

---

## 7. Outputs

| File | Contents |
|---|---|
| `weights_day0.mat` / `.npy` | `factors` = `factors_final` (3 × n_iscell), spatial footprints + neuropil masks (scipy sparse **and** 1-indexed COO for MATLAB), `FOV_height/width`, `refImg`, `params` |
| `decoder_definition.pkl` | `{'results', 'params'}` — every intermediate above: all 8 factor variants, all score variants, SPCA objects for every swept alpha, z-stack curves, manifold-alignment EVR curves, the full simulation trace |
| `params_day0_2.json` | parameters only |

The `.mat` is what the online MATLAB rig loads.
