# How `bmi_packets` is generated

End-to-end provenance of the `*.richfile` datasets in `C:\Data\datasets\bmi_packets`.
Every claim below is traced to a specific file; where a fact was verified by reading the packet
itself rather than the code, it is marked **[verified]**.

Repos involved (all `github.com/RichieHakim`):

| repo | role |
|---|---|
| **BWAIN** | MATLAB rig software — runs the online BMI, writes the logger |
| **suite2p** (fork) | offline ROI extraction + OASIS deconvolution |
| **ROICaT** | ROI classification (is-cell) and across-day tracking (UCIDs) |
| **NBAP** | "Neural-Behavior Analysis Pipeline" — cluster jobs for dF/F, trace quality, WaveSurfer→ScanImage alignment |
| **BMI_IDAP** (`pc-decoder` branch) | the decoder object + cursor simulation used in decoder definition |
| **basic_neural_processing_modules** (`bnpm`) | `make_dFoF`, `trace_quality_metrics`, `import_s2p`, `similarity`, … |
| **richfile** | the on-disk container format |

---

## Stage map

```
ScanImage 2p movie ─┬─► BWAIN online BMI ──► logger.mat + expParams.mat ──► logger_aligned.mat   (5)
                    │        (rig, MATLAB)        │
                    │                             └─► cursor_brain, freqToOutput, trials …
                    ├─► WaveSurfer 7 analog ch ──► ws_idxSI.pkl                                  (4)
                    └─► suite2p ──► F, Fneu, spks, stat, ops                                     (1)
                                      │
                                      ├─► ROICaT classify + track ──► preds, UCIDs, silhouettes  (2)
                                      ├─► make_dFoF_and_tqm.py ──► dFoF.npy, tqm.pkl,
                                      │                             iscell_tqm.npy               (3)
                                      └─► [DAY 0 ONLY] decoder notebook ──► weights_day0.mat     (6)
                                                                                │
                                                       AlignSpatialFootprints2_day0.m ──► baselineStuff.mat (7)
                                                                                │              (feeds BWAIN)
                     prep_BMI_data_for_Sam.ipynb  ◄────────────────────────────┘
                                 │
                                 └──► {mouse}.richfile          (8)
```

---

## 1. suite2p

Standard suite2p run per session → `F.npy`, `Fneu.npy`, `spks.npy`, `stat.npy`, `ops.npy`,
`iscell.npy` under `.../scanimage_data/exp/jobNum_0/suite2p/plane0`.
Loaded via `bnpm.ca2p_preprocessing.import_s2p`.

`spks.npy` is suite2p's **OASIS deconvolution** output. Its parameters (`neucoeff`, `baseline`,
`win_baseline`, `sig_baseline`, `tau`) live in that session's `ops.npy`, which is **not** shipped in
the packet — you cannot audit them from the packet alone.

This file is the one that becomes the packet's neural data. See
[the preprocessing doc](bmi_packets_preprocessing_and_factor_mapping.md) for the evidence.

## 2. ROICaT — classification and tracking

Two separate `.pkl` results, both per-mouse (all sessions), paths **[verified]** from
`extras/paths`:

- `{mouse}.ROICaT.classification_drawn.results.pkl` → `preds[i]` = is-cell boolean per session,
  from a hand-drawn/NN classifier.
- `stat_and_ops.ROICaT.tracking.results.pkl` → `clusters.labels_bySession[i]` = **UCID** per ROI
  (`-1` = unmatched), plus `quality_metrics.sample_silhouette` and `.cluster_silhouette`.

A UCID is a cluster of ROIs across sessions that ROICaT believes is one neuron. This is the only
across-day neuron correspondence used in the packet.

## 3. dF/F and trace-quality metrics (cluster job)

`NBAP/behavioral_drift_analysis/5a_make_dFoF_and_tqm/make_dFoF_and_tqm.py`, dispatched per
session by `master_controller_make_dFoF_and_tqm.ipynb`, rsynced down by `MC_fileTransfers.ipynb`
to `{mouse}/dFoF_and_tqm/{date}/jobNum_0/`.

```python
dFoF, dF, F_neuSub, F_baseline_roll = bnpm.ca2p_preprocessing.make_dFoF(
    F, Fneu=Fneu, roll_centered=True, roll_stride=1,
    roll_interpolation='linear', multicore_pref=True, **params['dFoF_params'])
tqm, iscell_tqm = bnpm.ca2p_preprocessing.trace_quality_metrics(...)
np.save('dFoF.npy', dFoF); pickle_save(tqm, 'tqm.pkl'); np.save('iscell_tqm.npy', iscell_tqm)
```

`make_dFoF` (bnpm) is exactly:

```
F_neuSub   = (F + offset) - neuropil_fraction * (Fneu + offset)
F_baseline = rolling_percentile(F_neuSub, ptile=percentile_baseline,
                                window=rolling_percentile_window/roll_stride, center=roll_centered)
dFoF       = (F_neuSub - F_baseline) / F_baseline
```

`trace_quality_metrics` computes 8 metrics and `iscell_tqm = good_ROIs` is the AND of all 8 being
inside their `(min, max)` thresholds (NaN ⇒ excluded):

`var_ratio__Fneu_over_F`, `EV__F_by_Fneu`, `base_FneuSub`, `base_F`, `nsr_autoregressive`,
`noise_derivMAD`, `max_dFoF`, `baseline_var`.

**Only `iscell_tqm.npy` reaches the packet.** `dFoF.npy` and `tqm.pkl` do not.

## 4. WaveSurfer → ScanImage frame clock

`NBAP/behavioral_drift_analysis/4a4_align_tca_and_ws_to_SI/align_tca_and_ws_to_SI.py`.

The 7 analog channels are named positionally from the raw h5:

```python
names_ws = ['sync_pulse','laser_pickoff','treadmill','lickometer','rewards','cursor','y_galvo']
ws = {name: trace for name, trace in zip(names_ws, ws_raw[last_sweep]['analogScans'])}
```

ScanImage frame onsets are found from the galvo flyback:

```python
peaks = scipy.signal.find_peaks(x=-np.diff(ws['y_galvo'], n=1), height=200)[0] + 1
val_idxWs__idx_SI = peaks
```

Every channel is then linearly interpolated (`scipy.interpolate.interp1d`) onto those frame
indices and saved as **`ws_idxSI.pkl`** → becomes `data_ws` in the packet, one sample per imaging
frame. Note `data_ws['cursor']` is the **analog voltage** of the cursor, a separate record from the
logger's `cursor_brain`.

## 5. Logger → movie alignment

`BWAIN/notebooks/Align_logger_to_movie.m`. BWAIN logs a per-frame `image_hash`
(`logger.timeSeries` col 27). Offline, the hash sequence is recomputed from the saved movie
(`compute_hashSequence.m`) and matched (`compare_hashSequences.m`) to produce `ia_2to1`:

```matlab
logger_aligned.timeSeries       = indexWithNaNs(logger.timeSeries, ia_2to1);
logger_aligned.timers           = indexWithNaNs(logger.timers, ia_2to1);
logger_aligned.decoder          = indexWithNaNs(logger.decoder, ia_2to1);
logger_aligned.motionCorrection = indexWithNaNs(logger.motionCorrection, ia_2to1);
logger_aligned.trials           = logger.trials;             % rows kept
logger_aligned.trials(:,3) = indexWithNaNs(ia_2to1, logger.trials(:,3));   % frame cols remapped
logger_aligned.trials(:,9) = indexWithNaNs(ia_2to1, logger.trials(:,9));
```

So **logger rows are movie frames**, and `NaN` means "no logger entry for this frame" — this is one
source of the NaNs in `cursor_brain`; the other is the pre-BMI baseline block, where BWAIN writes
`NaN` explicitly (`params.mode == 'baseline'`).

## 6. Decoder definition — day 0 only

`3_define_decoder_SPCA_SINGLEDAY.ipynb` → `weights_day0.mat` / `.npy`, `decoder_definition.pkl`,
`iscell_day0.npy`. Documented separately in
[`3_define_decoder_SPCA_SINGLEDAY_explained.md`](3_define_decoder_SPCA_SINGLEDAY_explained.md).

⚠ **The committed copy of that notebook is not the run that produced the shipped weights.** It
writes to `.../BMI/day0_stuff/mouse_0403L/` and ends with **3** factors; the packet loads
`.../face_rhythm/mouse_0403L/day0_analysis/weights_day0.npy`, which has **5** **[verified]**. Treat
the notebook as the *method*, and the packet's `extras/tracking/weights_day0` as the *artefact*.

Per-mouse factor counts **[verified]**: 0403L = 5, 0403R = 3, 0322R = 5, 0916N = 6.

## 7. Deployment to the rig

`BWAIN/notebooks/AlignSpatialFootprints2_day0.m`, re-run **at the start of every session**:

```matlab
load([directory_weights '\weights_day0.mat']);          % factors, footprints, refImg
if size(factors,2) > size(factors,1); factors = factors.'; end       % -> (n_neurons, K)
factor_space = factors ./ vecnorm(factors);                          % re-unit-norm columns
...
[D_field, movingReg] = imregdemons(tmp_im_moving, tmp_im_fixed, 500, 'PyramidLevels', 4, ...);
spatial_footprints_warped(ii,:,:) = imwarp(squeeze(spatial_footprints_dense(ii,:,:)), D_field);
Fneu_masks_warped(ii,:,:)         = imwarp(squeeze(Fneu_masks_dense(ii,:,:)), D_field);
```

Day-0 ROI masks are **non-rigidly warped** (diffeomorphic demons) into today's FOV and stored in
`baselineStuff.mat` alongside `factor_space`. **BWAIN never uses ROICaT** — online neuron identity
is "the same warped pixels", a completely different correspondence rule from the packet's UCIDs.

The online loop (`BWAIN/core_functions/userFunction_BMIv11_withZ.m`) then does, per frame:

```matlab
vals_neurons = nansum(F_lam,1) - nanmean(Fneu_lam,1) * params.dFoF.frac_neuropil;   % 0.7
F_baseline   = roll_ptile.step(vals_neurons + additive_offset);   % online rolling 30th pctile,
                                                                  % 10 min window, updated every 10 frames
dFoF         = (vals_neurons_offset - F_baseline) ./ F_baseline;
dFoF(F_baseline < 100 | dFoF > 15) = NaN;                         % online trace-quality gate
decoder_magnitudes = nansum(dFoF' .* baselineStuff.factor_space);                  % M
decoder_angles     = decoder_magnitudes / norm(decoder_magnitudes);                % CS
cursor_positions   = (abs(decoder_angles).^sm.cursor.angle_power) .* decoder_magnitudes;  % D
cursor_brain_raw   = cursor_positions(sm.cursor.factor_to_use);
cursor_brain       = cursor_brain_raw;                            % no smoothing in this version
```

This is identical in form to `bmi_idap.decoder.Decoder_angle_magnitude`, and
`sm.cursor = baselineStuff.cursors(sm.blocks.cursor_to_use)` means **`factor_to_use` can change
between blocks inside one session** — which is exactly what happens on remapping days.

## 8. Packaging — `prep_BMI_data_for_Sam.ipynb`

This is the generator. Its final cell's save dict is *exactly* the packet's top-level key set
**[verified]**, so the identification is certain (the notebook writes to a folder called
`gershman_packets`; the local copy is `bmi_packets`).

Inputs it globs (paths **[verified]** from `extras/paths`):

| variable | glob |
|---|---|
| `paths_neural` | `{mouse}/spks/{date}/spks.npy` |
| `paths_tqm` | `{mouse}/dFoF_and_tqm/{date}/jobNum_0/iscell_tqm.npy` |
| `paths_ws` | `{mouse}/ws_and_tca__idx_SI/{date}/ws_idxSI.pkl` |
| `paths_loggers` | `/media/rich/bigHDD/logger_alignment_data/{mouse}/{date}/analysis_data/logger_aligned.mat` |
| `paths_expParams` | `{mouse}/loggers/{date}/expParams.mat` |
| classification | `{mouse}/{mouse}.ROICaT.classification_drawn.results.pkl` |
| tracking | `{mouse}/stat_and_ops.ROICaT.tracking.results.pkl` |

`dates = sorted(set.intersection(*[set(p) for p in ...]))` — **a date missing from any one source is
silently dropped.** (0403L: `20230702` is in `paths_neural` but not in the packet's 36 days.)

### UCID filtering chain (cells 36–40) — this defines the neuron set

```python
labels = roicat.util.mask_UCIDs_with_iscell(labels_bySession, iscell_roicatClassification)
labels = roicat.util.mask_UCIDs_with_iscell(labels, [s > 0.0 for s in sampleSilhouette_bySession])
labels = roicat.util.mask_UCIDs_by_label(labels, cluster_labels_unique[cluster_silhouette > -0.2])
labels = roicat.util.discard_UCIDs_with_fewer_matches(labels, n_sesh_thresh='all')
labels = roicat.util.squeeze_UCID_labels(labels)
```

Each step sets failing UCIDs to `-1`; `n_sesh_thresh='all'` keeps **only UCIDs present in every
session**, and `squeeze` renumbers survivors to `0..N-1`. Consequence: every day's neural matrix has
the same row count and **row *i* is UCID *i* on every day** **[verified]** — e.g. 0403L is
(468, T) on all 36 days, with no all-NaN and no all-zero rows.

Note `iscells_tqm` is loaded and stored but **never used in this chain**.

### Neural alignment (cell 55)

```python
data_neural_aligned, idx_inverse_ucids = roicat.util.match_arrays_with_ucids(
    [np.array(data_neural[k], dtype=np.float32) for k in sorted(data_neural.keys())],
    ucids=list(labels_bySession_modified.values()), return_indices=True, squeeze=True)
```

This **only permutes and selects rows** — it allocates a NaN array of `max_ucid` rows and copies in
rows whose UCID is ≥ 0. No arithmetic is applied to the traces.

### Decoder-weight remapping (cells 49, 51)

```python
weights_day0 = np.load(path_weights_decoder, allow_pickle=True)[()]['factors'].T   # (n_iscell_day0, K)
weights_day0_idxDay0 = np.full((iscell_day0.shape[0], K), np.nan)
weights_day0_idxDay0[iscell_day0] = weights_day0                                    # (n_ROIs_day0, K)
weights_day0_idxFactors = weights_day0_idxDay0[np.array(ucids_inverse[date_day0], dtype=np.int64)]
```

(`session_type == 'multi_day'`, cell 50, is the 0322R branch — it recovers `iscell_day0` from
`iscell_classifier_matching_tqm` UCIDs instead of a saved `iscell_day0.npy`.)

I reproduced `weights_day0_idxFactors` from `weights_day0_idxDay0` and
`labels_bySession_modified[date_day0]` **bit-for-bit** **[verified]**.

### Behaviour events (cells 66–75)

`reward_times` are **peaks in the analog `data_ws['rewards']` channel**, not logger events:

```python
trace = trace/15000; trace -= savgol_filter(trace, window_length=int(30*0.9), polyorder=2)
peaks = find_peaks(np.diff(trace, prepend=0), height=trace.max()*0.4, distance=max(30*0.05,1))[0]
```

⚠ `lick_times` is computed with **`find_reward_peaks`, not `find_lickometer_peaks`** (cell 67) —
`find_lickometer_peaks` is defined and never called. Treat `lick_times` as suspect.

`reward_times_typed` (cell 74) keys trials by the 4-bit tuple
`(trialType_cursorOn, trialType_feedbackLinked, trialType_rewardOn, success_outcome)`. For each
trial it matches `frameNum_trialEnd - 1` (the `-1` converts MATLAB 1-indexing) to the nearest analog
reward peak within `thresh_diff=30` frames; unmatched trials fall back to
`frameNum_trialEnd - 1 + offset_frames` (2). Trial types with `rewardOn=0` **or** `success=0` always
use the raw end frame.

### Final write (cell 105)

```python
r = rf.demo.RichFile_data(path=f'.../gershman_packets/{mouse}')
r.save(obj={'data_neural_aligned', 'weights_day0_idxFactors', 'lick_times',
            'reward_times_typed', 'data_ws', 'date_day0', 'extras': {...}})
```

---

## What is actually in a packet

| key | content | shape / notes |
|---|---|---|
| `data_neural_aligned[date]` | suite2p `spks`, UCID-aligned | `(n_ucid, n_frames)` float32 |
| `weights_day0_idxFactors` | day-0 decoder weights in UCID rows | `(n_ucid, K)` float64, NaN rows |
| `data_ws[date]` | 7 analog channels on the SI frame clock | each `(n_frames,)` |
| `reward_times_typed[date][bits4]` | reward/end frames per trial type | frame indices |
| `lick_times[date]` | ⚠ computed with the reward detector | frame indices |
| `date_day0` | decoder-definition date | str |
| `extras/paths` | every source path | provenance |
| `extras/tracking` | `weights_day0`, `weights_day0_idxDay0`, `labels_bySession(_modified)`, `iscell_roicatClassification`, `iscell_tqm`, `sampleSilhouette_bySession`, `idx_inverse_ucids` | |
| `extras/logger_dfs[date]` **or** `extras/loggers[date]` | `timeSeries`, `trials`, `timers`, `decoder`, `motionCorrection` | two storage formats — see below |
| `extras/expParams[date]` | incl. `loggerNames` | needed for the bare-array format |
| `extras/reward_times[date]`, `extras/dates` | | |

**Two logger formats [verified]:** 0403L and 0403R store `extras/logger_dfs` (pandas DataFrames,
with a spurious `Unnamed: 0` column); 0322R and 0916N store `extras/loggers` (bare arrays; column
names come from `extras/expParams[date]['loggerNames']`). Always index columns by name.

**Frame-count audit across all four mice [verified]** — neural vs logger vs `data_ws` lengths agree
everywhere **except mouse_0403L on 20230728, 20230730, 20230731**, where neural and `data_ws` have
144,000 frames but the logger has only 108,000. Alignment still holds from frame 0 (the
cursor-reconstruction correlation on those days matches neighbouring days), so the logger is simply
truncated — most likely `params.timing.duration_session` was set for a 60-min session while the
movie ran 80 min. **The last 36,000 frames of those three days have no logger.**

## Day/session facts (mouse_0403L) **[verified]**

- 36 days, `date_day0 = 20230703`; 108,000 frames (60 min @ 30 Hz) for 20230703–20230724, then
  144,000 frames (80 min) from 20230725 on.
- `cursor_factor_to_use` = **2** on 20230703–20230724; **{2, 3}** from 20230725 (remapping —
  both axes appear within a session, selected per block).
- `cursor_threshold_reward` = 1.5 (20230703–05), 1.9 (20230706–20230724), then {0.9, 1.9} —
  0.9 belongs to factor 3, 1.9 to factor 2.
- `cursor_angle_power` = 2.0 and `cursor_win_smooth_cursor` = 1 throughout.
