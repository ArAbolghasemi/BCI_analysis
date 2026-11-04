"""
dynamics_utils.py
=================

Reusable *tools* for inferring and characterising the **low-dimensional neural
dynamics** (flow field / vector field, fixed points, attractors, basins) of the
Hakim-lab BMI packets, following the method of Luo, Kim et al. (2025, Nature
646:1156) "Transitions in dynamical regime and neural mode during perceptual
decisions" and its FINDR engine (Kim et al., ICML 2025).

Design philosophy (per project preference)
------------------------------------------
Only reusable, well-scoped computational tools live here: the model, its
training loop, field extraction and field comparison, plus synthetic
ground-truth systems for validation. **No figure code and no per-analysis
orchestration** -- the analysis flow and all plotting stay visible in
``Neural_Dynamics_Analysis.ipynb``.


The model
---------
A sequential variational autoencoder over a latent stochastic differential
equation, i.e. FINDR adapted to this dataset.

*Generative model.* Latent ``z_t`` of dimension ``d`` evolves as an
Euler-discretised SDE

    z_{t+1} = z_t + dt * F(z_t, u_t, e) + sqrt(dt) * sigma_z * eps_t         (1)

where ``u_t`` are (optional) *exogenous* task-event inputs, ``e`` are (optional)
learned embeddings of the partition the segment belongs to (day / block /
condition), and ``eps_t ~ N(0, I)``. ``F`` is the object of interest: the
**velocity vector field / flow field**.

``F`` is parameterised as a **gated neural ODE** (Kim, Can & Krishnamurthy,
ICML 2023 -- the parameterisation FINDR uses):

    F(z, u, e) = G(z, u, e) (.) [ -z + f(z, u, e) ] / tau                    (2)

with ``G`` a sigmoid-output MLP (so the *effective time constant* ``tau/G`` is
state dependent -- this is what lets the field express widely separated
timescales, hence approximate line/continuous attractors) and ``f`` a
soft-bounded MLP. The ``-z`` leak makes initialisation stable rather than
explosive.

*Observation model.* Each neuron's (transformed) activity is

    y_{t,n} ~ N( softplus(w_n . z_t + b_{t,n} + c_n),  s_n^2 )               (3)

``b_{t,n}`` is a **precomputed, fixed nuisance baseline** (slow session drift,
task-event kernels, locomotion/licking -- see the nuisance section) so that the
latent is not spent on decision-irrelevant signal; ``c_n`` is a learned
intercept and ``s_n`` a learned per-neuron noise SD.

Gaussian, *not* Poisson: measured on this dataset, per-neuron var/mean is
63--98 (Poisson would be 1) and the values are continuous, not counts. A
variance-stabilising transform (``sqrt`` by default) is applied first; see
:func:`transform_activity`.

*Inference.* A bidirectional GRU reads the whole segment and emits a posterior
drift ``g_t``; the latent path is sampled with the same diffusion as the prior,
so the KL reduces to the Euler/Girsanov form. The loss is FINDR's

    L = L_recon + c_dyn * sum_t (dt/2) || (g_t - F(z_t)) / sigma_z ||^2      (4)

(the second term is exactly eq. 12 of the paper up to the ``dt/2`` scaling;
``c_dyn`` is their ``c``, default 0.1), plus a KL on the initial state.


What is and is not identifiable (read before interpreting)
----------------------------------------------------------
* ``z`` is only defined up to an invertible linear map (``W`` can absorb it).
  :class:`LatentTransform` / :func:`canonical_transform` implement the paper's
  fix: SVD of ``W`` so that the readout is an isometry, hence **angles and
  distances in the canonical latent space are meaningful in neural space**. A
  residual rotation/reflection ambiguity remains, so compare fields either in a
  *shared* canonical frame (same ``W``) or via rotation-invariant summaries.
* An **input** term is only separable from autonomous dynamics if it varies
  independently of ``z``. In this closed-loop BMI the auditory tone is a readout
  of the population state, so it must NOT be passed as ``u``: doing so relabels
  autonomous dynamics as "input dynamics". Use ``u`` only for genuinely
  exogenous events (trial onset, reward delivery, timeout).
* Naive estimation of a field by regressing ``dz`` on ``z`` from noisy
  latents is biased toward contraction and **can hallucinate attractors**
  (errors-in-variables). :func:`naive_flow_field` implements that estimator
  precisely so the notebook can demonstrate the failure; it is a control, not a
  method.
"""

from __future__ import annotations

import contextlib
import gc
import math
import time
from dataclasses import dataclass, field as dc_field
from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from scipy import optimize


# ---------------------------------------------------------------------------
# device / memory / seeding
# ---------------------------------------------------------------------------
def get_device(prefer_gpu: bool = True) -> torch.device:
    """The device to fit on (``cuda`` when available)."""
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_report(device: Optional[torch.device] = None) -> str:
    """One-line description of the compute device, for notebook headers."""
    device = device or get_device()
    if device.type != "cuda":
        return f"device={device.type} (no GPU: fits will be slow)"
    i = device.index or 0
    p = torch.cuda.get_device_properties(i)
    return (f"device=cuda:{i} {p.name} | {p.total_memory / 1e9:.1f} GB | "
            f"sm_{p.major}{p.minor} | torch {torch.__version__}")


def free(*objs) -> None:
    """Drop references, run the collector and release cached GPU blocks.

    Call after each fit / field extraction in a per-day loop: PyTorch keeps a
    caching allocator, so without this a long loop over days holds every
    intermediate allocation until the process exits.
    """
    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


@contextlib.contextmanager
def gpu_session(label: str = "", *, device: Optional[torch.device] = None,
                verbose: bool = True, reset_peak: bool = True):
    """Context manager that reports peak GPU memory and always cleans up.

    ``with dy.gpu_session("day 20230715") as dev:`` -- on exit the caching
    allocator is emptied even if the body raised, and the peak allocation is
    printed so a long per-day loop cannot silently creep toward OOM.
    """
    device = device or get_device()
    is_cuda = device.type == "cuda"
    if is_cuda and reset_peak:
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    try:
        yield device
    finally:
        peak = torch.cuda.max_memory_allocated(device) / 1e9 if is_cuda else 0.0
        free()
        if verbose:
            tag = f"[{label}] " if label else ""
            msg = f"{tag}{time.time() - t0:.1f} s"
            if is_cuda:
                msg += f" | peak GPU {peak:.2f} GB"
            print(msg)


def set_seed(seed: int) -> None:
    """Seed python/numpy/torch RNGs (call before every fit you want repeatable)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def enable_fast_matmul(tf32: bool = True) -> None:
    """Turn on TF32 matmuls and cuDNN autotuning.

    Preferred over bf16 autocast for this model: TF32 keeps fp32 dynamic range
    (which the difference-of-drifts loss needs) while still using the tensor
    cores."""
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass
class PreprocConfig:
    """Preprocessing choices, all empirically motivated on this dataset.

    ``bin_frames=5`` (30 Hz -> 6 Hz, dt = 1/6 s) because the cross-validated
    predictability of ``dz`` from ``z`` peaks there: 0.12-0.15 at 33 ms rising
    to 0.15-0.31 at 167 ms, while the white-noise fraction of the latents falls
    from 0.19 to 0.10-0.16. ``transform="sqrt"`` because raw activity has
    skew 4.2 / kurtosis 30 and var-vs-mean slope 1.5, versus skew 1.5 /
    kurtosis 4.9 / slope 0.92 after sqrt.
    """
    bin_frames: int = 5
    sr_raw: float = 30.0
    transform: str = "sqrt"            # "sqrt" | "log1p" | "none"
    log1p_scale: float = 20.0          # divisor used by the log1p transform
    # "std", not "robust": MAD-based scaling is pathological on sparse data. At
    # 30 Hz this signal is 59 % zeros, so the MAD of many neurons is near zero and
    # robust scaling amplifies exactly the noisiest units -- measured white-noise
    # share of the top latents 0.745 (robust) vs 0.203 (std) at 33 ms bins. At the
    # working 167 ms bin (12 % zeros) the two agree (0.102 vs 0.109), but the
    # pathology silently corrupts any comparison across bin sizes.
    scale_mode: str = "std"            # per-neuron scaling: "std" | "robust" | "none"
    max_segment_bins: int = 150        # long baseline stretches are cut into chunks

    @property
    def dt(self) -> float:
        """Model time step in seconds."""
        return self.bin_frames / self.sr_raw

    @property
    def sr(self) -> float:
        """Effective sampling rate in Hz."""
        return self.sr_raw / self.bin_frames

    @property
    def tag(self) -> str:
        return f"b{self.bin_frames}_{self.transform}_{self.scale_mode}"


@dataclass
class ModelConfig:
    """Architecture of the latent-SDE model (see module docstring)."""
    d: int = 3                         # latent dimensionality
    n_neurons: int = 0                 # set from the data
    dt: float = 1.0 / 6.0
    # flow network F: gated neural ODE
    flow_hidden: int = 64
    flow_depth: int = 2                # number of Linear layers in f
    gate_hidden: int = 32
    gate_depth: int = 2                # number of Linear layers in G
    tau_init: float = 0.5              # seconds
    learn_tau: bool = True
    flow_bound: float = 4.0            # soft bound on f's output (keeps sims finite)
    # encoder
    enc_proj: int = 64                 # neurons -> projection before the GRU
    enc_hidden: int = 64               # GRU hidden size (per direction)
    enc_drift_hidden: int = 64
    # inputs / partition conditioning
    u_dim: int = 0                     # exogenous event channels (0 = autonomous)
    cond_sizes: dict = dc_field(default_factory=dict)   # e.g. {"day": 20, "block": 3}
    cond_dim: int = 4                  # embedding width per conditioning factor
    # latent noise & initial state
    sigma_z_init: float = 0.2          # per sqrt(second)
    learn_sigma_z: bool = True
    sigma0_prior: float = 1.0          # prior SD of the initial state
    # observation model
    obs: str = "gauss"                 # "gauss" (this dataset) | "poisson"
    softplus_readout: bool = True

    @property
    def tag(self) -> str:
        c = "-".join(sorted(self.cond_sizes)) if self.cond_sizes else "none"
        return (f"d{self.d}_f{self.flow_hidden}x{self.flow_depth}"
                f"_g{self.gate_hidden}x{self.gate_depth}_e{self.enc_hidden}_c{c}_u{self.u_dim}")


@dataclass
class TrainConfig:
    """Optimisation settings."""
    lr: float = 3e-3
    weight_decay: float = 1e-5
    epochs: int = 1500
    # Large batches on purpose: the latent recursion is one sequential kernel
    # launch per time step, so training is launch-bound, not FLOP-bound.
    # Measured: 656 ms/epoch at batch 64 vs 143 ms/epoch at batch 320.
    batch_size: int = 256
    # FINDR uses 0.1; on this data that is too weak. c_dyn trades "explain the
    # neurons" against "be a dynamical system", and it changes the inferred latent,
    # not just the field. Measured on 6 pooled stable days, 3 seeds:
    #   c_dyn  field R2 / ceiling   across-seed cosine   increment autocorr   attractors
    #   0.1     0.081 / 0.187        0.846               -0.006 (random walk)   1
    #   2.0     0.174 / 0.248        0.893               +0.311                 1
    #   8.0     0.543 / 0.616        0.968               +0.671 (smooth)        0
    # 2.0 is the default: the field captures 70 % of the achievable motion, is
    # reproducible, and trajectories converge, for ~0.6 % worse reconstruction.
    # IMPORTANT: whether attractors exist at all is NOT robust across this range,
    # so sweep it before claiming any attractor result.
    c_dyn: float = 2.0                 # FINDR's c (weight on the dynamics/KL term)
    # Deliberately small: the initial-state KL is a nuisance regulariser (it only
    # constrains where segments start), but at weight 1 it prunes latent
    # dimensions. Measured on a synthetic point attractor: at kl0_weight=1 the
    # second latent dimension collapsed (canonical correlation with the true
    # latent 0.075, versus 0.999 for plain PCA on the same data).
    kl0_weight: float = 0.05
    warmup_epochs: int = 50            # linear ramp of c_dyn and kl0 (avoids posterior collapse)
    grad_clip: float = 5.0
    patience: int = 250                # early stopping on held-out loss
    min_epochs: int = 200
    val_frac: float = 0.2
    # bf16 autocast is OFF by default on purpose: the dynamics term is a
    # *difference* of two network outputs, and bf16's ~3 significant digits
    # swamp it once the posterior drift approaches the field. TF32 matmuls
    # (see enable_fast_matmul) give most of the speed at far better precision.
    amp: bool = False
    seed: int = 0
    eval_every: int = 5
    verbose: bool = True
    balance_conditions: bool = True    # equal expected time per condition per epoch
    pca_init: bool = True              # initialise the readout from PCA (see init_readout_from_pca)

    @property
    def tag(self) -> str:
        return f"lr{self.lr:g}_c{self.c_dyn:g}_ep{self.epochs}_s{self.seed}"


# ---------------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------------
def transform_activity(X: np.ndarray, cfg: PreprocConfig) -> np.ndarray:
    """Variance-stabilising transform of a ``(N, T)`` activity matrix.

    The raw signal is continuous, zero-inflated and heavy tailed with a strongly
    level-dependent variance, which breaks both Poisson and constant-variance
    Gaussian assumptions; ``sqrt`` brings skew from ~4.2 to ~1.5 and the
    log-log var-vs-mean slope from ~1.5 to ~0.9."""
    x = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    x = np.maximum(x, 0.0)
    if cfg.transform == "sqrt":
        return np.sqrt(x)
    if cfg.transform == "log1p":
        return np.log1p(x / cfg.log1p_scale)
    if cfg.transform in ("none", None):
        return x
    raise ValueError(f"unknown transform {cfg.transform!r}")


def bin_matrix(X: np.ndarray, bin_frames: int) -> np.ndarray:
    """Non-overlapping temporal mean: ``(N, T)`` -> ``(N, T // bin_frames)``."""
    b = int(bin_frames)
    if b <= 1:
        return np.asarray(X, dtype=np.float64)
    N, T = X.shape
    T2 = (T // b) * b
    return X[:, :T2].reshape(N, T2 // b, b).mean(axis=2)


def neuron_scales(Xb: np.ndarray, cfg: PreprocConfig,
                  ref_cols: Optional[np.ndarray] = None) -> np.ndarray:
    """Per-neuron scale factors used to put all neurons on a comparable footing.

    Scaling (not centring -- the readout is non-negative) is computed on
    ``ref_cols`` when given. Pass the *baseline* timepoints as the reference so
    that cross-day differences in imaging gain are removed without letting a
    condition contrast rescale itself away."""
    R = Xb if ref_cols is None else Xb[:, ref_cols]
    if cfg.scale_mode == "none":
        return np.ones(Xb.shape[0])
    if cfg.scale_mode == "std":
        s = R.std(axis=1)
    else:  # robust -- see the warning on PreprocConfig.scale_mode before using
        med = np.median(R, axis=1, keepdims=True)
        s = 1.4826 * np.median(np.abs(R - med), axis=1)
        # a near-zero MAD means a mostly-zero neuron, not a quiet one: fall back to
        # the SD rather than dividing by ~0 and amplifying pure noise
        weak = s < 0.05 * np.maximum(R.std(axis=1), 1e-12)
        if weak.any():
            s[weak] = R[weak].std(axis=1)
    s = np.asarray(s, dtype=np.float64)
    s[s < 1e-6] = 1.0
    return s


def preprocess_for_dynamics(X: np.ndarray, cfg: PreprocConfig, *,
                            ref_cols: Optional[np.ndarray] = None,
                            scales: Optional[np.ndarray] = None):
    """``(N, T_frames)`` raw -> ``(N, T_bins)`` transformed, binned, scaled.

    Returns ``(Xb, scales)``. Reuse ``scales`` across days/partitions that must
    share a latent space."""
    Xt = transform_activity(X, cfg)
    Xb = bin_matrix(Xt, cfg.bin_frames)
    if scales is None:
        scales = neuron_scales(Xb, cfg, ref_cols=ref_cols)
    return Xb / scales[:, None], scales


# ---------------------------------------------------------------------------
# segments
# ---------------------------------------------------------------------------
@dataclass
class SegmentSet:
    """A set of variable-length activity segments, ready for the model.

    ``Y`` is ``(B, Tmax, N)`` (time-major per segment), ``mask`` is ``(B, Tmax)``
    with 1 for valid samples. ``labels`` holds one integer code per conditioning
    factor per segment (e.g. ``{"day": ..., "condition": ...}``), ``meta`` the
    human-readable table (list of dicts) in the same order.
    """
    Y: np.ndarray
    mask: np.ndarray
    labels: dict
    meta: list
    B: Optional[np.ndarray] = None      # nuisance baseline, same shape as Y
    U: Optional[np.ndarray] = None      # (B, Tmax, u_dim) exogenous inputs

    def __len__(self) -> int:
        return self.Y.shape[0]

    @property
    def n_neurons(self) -> int:
        return self.Y.shape[2]

    @property
    def n_valid(self) -> int:
        return int(self.mask.sum())

    def subset(self, idx: Sequence[int]) -> "SegmentSet":
        idx = np.asarray(idx, dtype=int)
        return SegmentSet(
            Y=self.Y[idx], mask=self.mask[idx],
            labels={k: v[idx] for k, v in self.labels.items()},
            meta=[self.meta[i] for i in idx],
            B=None if self.B is None else self.B[idx],
            U=None if self.U is None else self.U[idx],
        )

    def summary(self) -> str:
        parts = [f"{len(self)} segments", f"{self.n_valid} valid samples"]
        for k, v in self.labels.items():
            parts.append(f"{k}:{len(np.unique(v))} levels")
        return ", ".join(parts)


def build_segment_set(Xb: np.ndarray,
                      intervals_by_condition: dict,
                      *, max_bins: int = 150, min_bins: int = 8,
                      baseline: Optional[np.ndarray] = None,
                      inputs: Optional[np.ndarray] = None,
                      extra_labels: Optional[dict] = None) -> SegmentSet:
    """Cut a session into padded segments from ``{condition: [(start, end), ...]}``.

    Intervals are in *bin* units. Segments longer than ``max_bins`` are split
    into consecutive chunks (long baseline stretches and 20 s timeout trials
    would otherwise dominate both the padding and the loss); segments shorter
    than ``min_bins`` are dropped. ``extra_labels`` maps a factor name to either
    a scalar (applied to every segment) or a ``{condition: value}`` dict.
    """
    N, T = Xb.shape
    segs, meta = [], []
    for cname, ivs in intervals_by_condition.items():
        for k, (s, e) in enumerate(ivs):
            s, e = int(max(s, 0)), int(min(e, T))
            if e - s < min_bins:
                continue
            for c0 in range(s, e, max_bins):
                c1 = min(c0 + max_bins, e)
                if c1 - c0 < min_bins:
                    continue
                segs.append((c0, c1))
                meta.append(dict(condition=cname, start=c0, end=c1,
                                 n_bins=c1 - c0, source_index=k))
    if not segs:
        raise ValueError("no segments survived the length filters")

    Tmax = max(e - s for s, e in segs)
    Bn = len(segs)
    Y = np.zeros((Bn, Tmax, N), dtype=np.float32)
    mask = np.zeros((Bn, Tmax), dtype=np.float32)
    Base = np.zeros_like(Y) if baseline is not None else None
    U = (np.zeros((Bn, Tmax, inputs.shape[0]), dtype=np.float32)
         if inputs is not None else None)
    for i, (s, e) in enumerate(segs):
        n = e - s
        Y[i, :n] = Xb[:, s:e].T
        mask[i, :n] = 1.0
        if Base is not None:
            Base[i, :n] = baseline[:, s:e].T
        if U is not None:
            U[i, :n] = inputs[:, s:e].T

    conds = np.array([m["condition"] for m in meta])
    levels = sorted(set(conds))
    labels = {"condition": np.array([levels.index(c) for c in conds], dtype=np.int64)}
    for i, m in enumerate(meta):
        m["condition_code"] = int(labels["condition"][i])
    if extra_labels:
        for name, val in extra_labels.items():
            if isinstance(val, dict):
                labels[name] = np.array([int(val[m["condition"]]) for m in meta], dtype=np.int64)
            else:
                labels[name] = np.full(Bn, int(val), dtype=np.int64)
            for i, m in enumerate(meta):
                m[name] = int(labels[name][i])
    return SegmentSet(Y=Y, mask=mask, labels=labels, meta=meta, B=Base, U=U)


def concat_segment_sets(sets: Sequence[SegmentSet]) -> SegmentSet:
    """Concatenate segment sets (e.g. one per day), padding to a common length."""
    sets = [s for s in sets if len(s) > 0]
    if not sets:
        raise ValueError("nothing to concatenate")
    Tmax = max(s.Y.shape[1] for s in sets)
    N = sets[0].n_neurons
    keys = set(sets[0].labels)
    for s in sets:
        if set(s.labels) != keys:
            raise ValueError("segment sets have different label factors")

    def pad(a, width):
        if a is None:
            return None
        out = np.zeros((a.shape[0], width) + a.shape[2:], dtype=a.dtype)
        out[:, :a.shape[1]] = a
        return out

    Y = np.concatenate([pad(s.Y, Tmax) for s in sets], axis=0)
    mask = np.concatenate([pad(s.mask, Tmax) for s in sets], axis=0)
    B = (np.concatenate([pad(s.B, Tmax) for s in sets], axis=0)
         if all(s.B is not None for s in sets) else None)
    U = (np.concatenate([pad(s.U, Tmax) for s in sets], axis=0)
         if all(s.U is not None for s in sets) else None)
    labels = {k: np.concatenate([s.labels[k] for s in sets]) for k in keys}
    meta = [m for s in sets for m in s.meta]

    # Re-encode the condition codes over the UNION of levels. Each SegmentSet numbers
    # its conditions from its own sorted level list, so a set that happens to lack a
    # level shifts the codes of every later one, and concatenating those would put two
    # different conditions under one embedding index. (Harmless on the current data --
    # incorrect_early and incorrect_late always appear together, and the surviving
    # {baseline, correct} keep codes 0 and 1 -- but it breaks the moment a level is
    # missing on its own, e.g. a day whose misses are all shorter than the median hit.)
    if meta and all("condition" in m for m in meta):
        code = {c: i for i, c in enumerate(sorted({m["condition"] for m in meta}))}
        labels["condition"] = np.array([code[m["condition"]] for m in meta],
                                       dtype=np.int64)
        for m in meta:
            m["condition_code"] = code[m["condition"]]
    assert Y.shape[2] == N
    return SegmentSet(Y=Y, mask=mask, labels=labels, meta=meta, B=B, U=U)


def split_train_val(segset: SegmentSet, *, val_frac: float = 0.2, seed: int = 0,
                    stratify: Optional[str] = "condition"):
    """Random split over *segments* (not timepoints), optionally stratified."""
    rng = np.random.default_rng(seed)
    n = len(segset)
    idx = np.arange(n)
    val = []
    if stratify and stratify in segset.labels:
        for lev in np.unique(segset.labels[stratify]):
            sub = idx[segset.labels[stratify] == lev]
            rng.shuffle(sub)
            k = max(1, int(round(val_frac * sub.size))) if sub.size > 2 else 0
            val.append(sub[:k])
        val = np.concatenate(val) if val else np.array([], dtype=int)
    else:
        rng.shuffle(idx)
        val = idx[:max(1, int(round(val_frac * n)))]
    tr = np.setdiff1d(idx, val)
    return tr, val


def time_balanced_weights(segset: SegmentSet, factor: str = "condition") -> np.ndarray:
    """Per-segment **loss** weights equalising total time per level of ``factor``.

    Needed because a 20 s timeout contributes ~6x the timepoints of a 3 s hit, so
    an unweighted fit lets failure trials dominate which region of state space
    the field is estimated on.

    These are loss weights, not sampling weights, on purpose: reweighting the
    *sampling* distribution would also distort which segments within a condition
    get seen (weighted draws favour whichever segments carry the weight), whereas
    scaling each segment's contribution to the objective by ``1 / total_time`` of
    its condition balances conditions exactly and leaves the within-condition
    distribution alone. Weights are scaled to average 1 so the loss stays on a
    comparable scale to an unweighted fit."""
    n = len(segset)
    if factor not in segset.labels:
        return np.ones(n)
    lab = segset.labels[factor]
    dur = segset.mask.sum(axis=1)
    w = np.ones(n, dtype=np.float64)
    for lev in np.unique(lab):
        m = lab == lev
        tot = dur[m].sum()
        if tot > 0:
            w[m] = 1.0 / tot
    wsum = (w * dur).sum()
    if wsum > 0:
        w *= dur.sum() / wsum          # average weight ~1 in time-weighted terms
    return w


# ---------------------------------------------------------------------------
# condition intervals (bridge from neural_analysis_utils' frame intervals)
# ---------------------------------------------------------------------------
def intervals_to_bins(intervals: Sequence[tuple], bin_frames: int,
                      T_bins: Optional[int] = None) -> list:
    """Convert frame intervals (as built by ``neural_analysis_utils``) to bin units."""
    out = []
    for s, e in intervals:
        s2 = int(math.floor(s / bin_frames))
        e2 = int(math.ceil(e / bin_frames))
        if T_bins is not None:
            s2, e2 = max(s2, 0), min(e2, T_bins)
        if e2 > s2:
            out.append((s2, e2))
    return out


def split_incorrect_intervals(intervals: Sequence[tuple], match_bins: int):
    """Split miss trials into a duration-matched head and the remaining tail.

    A 20 s timeout is not "a hit that failed": it contains a give-up /
    disengagement period, and it contributes ~6x the timepoints of a ~3 s hit.
    Comparing hits with the *whole* miss therefore confounds outcome with both
    duration and disengagement. ``early`` is the first ``match_bins`` of each
    miss (set it to the median hit duration); ``late`` is what follows."""
    early, late = [], []
    for s, e in intervals:
        cut = min(int(s + match_bins), e)
        if cut > s:
            early.append((s, cut))
        if e > cut:
            late.append((cut, e))
    return early, late


def median_duration_bins(intervals: Sequence[tuple]) -> int:
    """Median interval length in bins (used to match miss trials to hits)."""
    if not intervals:
        return 0
    return int(np.median([e - s for s, e in intervals]))


# ---------------------------------------------------------------------------
# nuisance baseline model
# ---------------------------------------------------------------------------
# Rationale, and why the default is deliberately minimal:
#
# The latent should not be spent on activity that has an obvious non-dynamical
# cause. The clearest such cause is slow session drift (bleaching, z-drift,
# arousal): its timescale is far longer than a trial, so it is separable from
# trial-level dynamics, and leaving it in demonstrably hurts -- removing a slow
# trend *raised* the cross-validated predictability of dz from z (0.21 -> 0.41)
# on this data.
#
# Everything else is left OUT by default. Licking is time-locked to reward,
# locomotion is coupled to task engagement, and trial-onset kernels absorb
# exactly the average trial-locked trajectory that the flow field is supposed to
# describe. Regressing those out would risk deleting the signal. They are
# available as explicit components so the notebook can (a) test whether they
# differ across the partitions being compared and (b) refit with them included
# as a robustness check.
def raised_cosine_basis(n_basis: int, length: int, *, log_stretch: float = 0.0) -> np.ndarray:
    """``(length, n_basis)`` smooth bumps tiling ``[0, length)``.

    ``log_stretch > 0`` warps the bumps to be denser near time 0, the standard
    choice for event kernels whose response is fastest right after the event."""
    t = np.arange(length, dtype=float)
    if log_stretch > 0:
        warp = np.log(t + log_stretch) - np.log(log_stretch)
        warp = warp / max(warp[-1], 1e-9)
    else:
        warp = t / max(t[-1], 1e-9)
    centers = np.linspace(0, 1, n_basis)
    width = (centers[1] - centers[0]) if n_basis > 1 else 1.0
    B = np.zeros((length, n_basis))
    for k, c in enumerate(centers):
        x = (warp - c) / max(width, 1e-9)
        inside = np.abs(x) < 1
        B[inside, k] = 0.5 * (1 + np.cos(np.pi * x[inside]))
    return B


def session_drift_basis(T: int, n_basis: int = 8) -> np.ndarray:
    """``(T, n_basis)`` smooth basis over session time, for slow drift.

    Keep ``n_basis`` small enough that each bump spans many minutes (8 bumps over
    a 60 min session ~ 15 min each). This is a deliberate limit, not laziness:
    only *session-scale* trends are safely attributable to nuisance (bleaching,
    z-drift, arousal). Fluctuations on a ~60 s scale must be left in, because slow
    neural dynamics live there -- a line attractor or slow manifold produces
    exactly that kind of slow wandering, so high-passing it away would delete the
    phenomenon this whole analysis is trying to detect. (High-pass filtering the
    latents at 60 s does raise the apparent predictability of ``dz`` from ``z``
    from 0.21 to 0.41, which is precisely why that number must not be used to
    justify aggressive detrending.)"""
    return raised_cosine_basis(n_basis, T)


def n_drift_for_session(T: int, sr: float, *, minutes_per_basis: float = 15.0) -> int:
    """Number of drift basis functions giving roughly one bump per N minutes."""
    return int(max(2, round((T / sr / 60.0) / max(minutes_per_basis, 1e-6))))


def event_design(event_bins: Sequence[float], T: int, *, n_basis: int = 4,
                 pre_bins: int = 0, post_bins: int = 18,
                 log_stretch: float = 2.0) -> np.ndarray:
    """``(T, n_basis)`` event-locked predictors: impulses convolved with a basis."""
    length = int(pre_bins + post_bins)
    if length < 2 or n_basis < 1:
        return np.zeros((T, 0))
    B = raised_cosine_basis(n_basis, length, log_stretch=log_stretch)
    imp = np.zeros(T)
    for f in np.asarray(event_bins, dtype=float).ravel():
        i = int(round(f))
        if 0 <= i < T:
            imp[i] = 1.0
    out = np.zeros((T, n_basis))
    for k in range(n_basis):
        conv = np.convolve(imp, B[:, k], mode="full")[:T + pre_bins]
        out[:, k] = conv[pre_bins:pre_bins + T] if pre_bins else conv[:T]
    return out


def covariate_design(x: np.ndarray, *, n_lags: int = 3, lag_step: int = 2,
                     zscore: bool = True) -> np.ndarray:
    """``(T, n_lags)`` lagged copies of a behavioural covariate (e.g. treadmill)."""
    x = np.nan_to_num(np.asarray(x, dtype=float).ravel(), nan=0.0)
    if zscore:
        sd = x.std()
        x = (x - x.mean()) / (sd if sd > 1e-9 else 1.0)
    T = x.size
    out = np.zeros((T, n_lags))
    for k in range(n_lags):
        sh = k * lag_step
        out[sh:, k] = x[:T - sh] if sh else x
    return out


@dataclass
class NuisanceModel:
    """Per-neuron nuisance baseline, as a pre-softplus offset ``b`` of shape ``(N, T)``."""
    b: np.ndarray
    weights: np.ndarray            # (N, P)
    names: list
    groups: dict                   # group name -> column indices
    r2: np.ndarray                 # per-neuron variance explained
    group_r2: dict                 # group name -> per-neuron unique variance explained

    def summary(self) -> str:
        s = [f"nuisance: {self.b.shape[0]} neurons x {len(self.names)} predictors",
             f"median R2 = {np.median(self.r2):.3f}"]
        for g, v in self.group_r2.items():
            s.append(f"{g}: {np.median(v):.4f}")
        return " | ".join(s)


def build_nuisance_design(T: int, *, components: Sequence[str] = ("drift",),
                          n_drift: int = 8,
                          trial_start_bins: Optional[Sequence[float]] = None,
                          reward_bins: Optional[Sequence[float]] = None,
                          covariates: Optional[dict] = None,
                          n_event_basis: int = 4, event_post_bins: int = 18,
                          n_cov_lags: int = 3):
    """Assemble the nuisance design matrix. Returns ``(D (T, P), names, groups)``.

    ``components`` selects families: ``"drift"`` (default), ``"events"``
    (trial-onset and reward kernels), ``"covariates"`` (whatever is passed in
    ``covariates``, e.g. treadmill / lickometer / motion metrics). See the note
    above on why only ``drift`` is on by default."""
    cols, names, groups = [], [], {}

    def add(block, prefix, group):
        if block.shape[1] == 0:
            return
        i0 = sum(c.shape[1] for c in cols)
        cols.append(block)
        names.extend([f"{prefix}{k}" for k in range(block.shape[1])])
        groups.setdefault(group, []).extend(range(i0, i0 + block.shape[1]))

    add(np.ones((T, 1)), "const", "const")
    if "drift" in components:
        add(session_drift_basis(T, n_drift), "drift", "drift")
    if "events" in components:
        if trial_start_bins is not None:
            add(event_design(trial_start_bins, T, n_basis=n_event_basis,
                             post_bins=event_post_bins), "trialon", "events")
        if reward_bins is not None:
            add(event_design(reward_bins, T, n_basis=n_event_basis,
                             post_bins=event_post_bins), "reward", "events")
    if "covariates" in components and covariates:
        for cname, sig in covariates.items():
            add(covariate_design(sig, n_lags=n_cov_lags), f"{cname}_lag", "covariates")
    D = np.concatenate(cols, axis=1) if cols else np.zeros((T, 0))
    return D, names, groups


def fit_nuisance_baseline(Xb: np.ndarray, D: np.ndarray, *, names: Optional[list] = None,
                          groups: Optional[dict] = None, ridge: float = 1e-3,
                          iters: int = 400, lr: float = 0.05, device=None,
                          verbose: bool = False) -> NuisanceModel:
    """Fit ``y_n(t) ~ softplus(D w_n)`` for every neuron at once, on the GPU.

    The softplus link matches the model's readout, so the fitted values can be
    handed straight to :class:`FlowVAE` as the pre-softplus offset ``b``. Groups
    are refit with each family held out to report its *unique* explained
    variance -- that is what tells you whether e.g. locomotion is even worth
    including.

    ``ridge`` is small by default: the penalty is a mean over weights and the
    loss a mean over samples, so ``ridge=1`` halves the achievable fit (measured
    median R2 0.006 versus 0.012 converged on a stable day)."""
    device = device or get_device()
    N, T = Xb.shape
    P = D.shape[1]
    names = names or [f"p{i}" for i in range(P)]
    groups = groups or {}
    Y = torch.tensor(np.asarray(Xb, dtype=np.float32).T, device=device)   # (T, N)
    Dt = torch.tensor(np.asarray(D, dtype=np.float32), device=device)     # (T, P)

    def _fit(Dsub):
        w = torch.zeros((Dsub.shape[1], N), device=device, requires_grad=True)
        with torch.no_grad():
            w[0] = torch.log(torch.expm1(Y.mean(0).clamp_min(1e-3)))
        opt = torch.optim.Adam([w], lr=lr)
        for it in range(iters):
            opt.zero_grad(set_to_none=True)
            pred = Fnn.softplus(Dsub @ w)
            loss = ((Y - pred) ** 2).mean() + ridge * (w[1:] ** 2).mean()
            loss.backward()
            opt.step()
            if verbose and it % 100 == 0:
                print(f"    nuisance it {it}: {float(loss):.5f}")
        with torch.no_grad():
            pred = Fnn.softplus(Dsub @ w)
            ss_res = ((Y - pred) ** 2).sum(0)
            ss_tot = ((Y - Y.mean(0)) ** 2).sum(0).clamp_min(1e-9)
            r2 = (1 - ss_res / ss_tot).cpu().numpy()
        return w.detach(), r2

    w_full, r2_full = _fit(Dt)
    group_r2 = {}
    for g, idx in groups.items():
        if g == "const":
            continue
        keep = [i for i in range(P) if i not in set(idx)]
        if not keep:
            continue
        _, r2_wo = _fit(Dt[:, keep])
        group_r2[g] = r2_full - r2_wo

    b = (Dt @ w_full).T.cpu().numpy()          # (N, T) pre-softplus offset
    W = w_full.T.cpu().numpy()
    free(Y, Dt, w_full)
    return NuisanceModel(b=b, weights=W, names=names, groups=groups,
                         r2=r2_full, group_r2=group_r2)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    """Plain feedforward net; ``depth`` counts Linear layers (L in the gnODE paper)."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int, depth: int,
                 hidden_act: Callable[[], nn.Module] = nn.ReLU):
        super().__init__()
        layers, dim = [], in_dim
        for _ in range(max(depth - 1, 0)):
            layers += [nn.Linear(dim, hidden), hidden_act()]
            dim = hidden
        layers += [nn.Linear(dim, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GatedFlow(nn.Module):
    """The flow field, as a gated neural ODE (Kim, Can & Krishnamurthy 2023).

        F(z, u, e) = G(z, u, e) (.) [ -z + f(z, u, e) ] / tau

    ``G`` is sigmoid-output, so ``tau / G`` is a state-dependent time constant --
    the mechanism that lets one field hold both fast and very slow directions
    (approximate line / continuous attractors). ``f`` is soft-bounded by
    ``flow_bound * tanh(f / flow_bound)``, which is ~identity near the origin but
    keeps long simulations finite. The ``-z`` leak keeps initialisation stable.

    ``e`` is the concatenation of partition embeddings; with ``cond_sizes={}``
    the field is a single shared function of ``z`` (the primary, autonomous case).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.d + cfg.u_dim + cfg.cond_dim * len(cfg.cond_sizes)
        self.f_net = MLP(in_dim, cfg.d, cfg.flow_hidden, cfg.flow_depth)
        self.g_net = MLP(in_dim, cfg.d, cfg.gate_hidden, cfg.gate_depth)
        self.log_tau = nn.Parameter(torch.tensor(math.log(cfg.tau_init)),
                                    requires_grad=cfg.learn_tau)
        self.register_buffer("bound", torch.tensor(float(cfg.flow_bound)))
        # start with a modest gate so the initial field is slow and stable
        with torch.no_grad():
            self.g_net.net[-1].bias.fill_(-1.0)

    def forward(self, z, u=None, cond=None):
        parts = [z]
        if u is not None:
            parts.append(u)
        if cond is not None:
            parts.append(cond)
        x = torch.cat(parts, dim=-1) if len(parts) > 1 else z
        f = self.bound * torch.tanh(self.f_net(x) / self.bound)
        g = torch.sigmoid(self.g_net(x))
        return g * (f - z) / torch.exp(self.log_tau)


class Encoder(nn.Module):
    """Bidirectional-GRU recognition network (FINDR's ``G``).

    Reads the whole segment (so it can average out observation noise, which on
    this data carries a large white component) and emits the initial-state
    posterior plus a per-step posterior drift."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Linear(cfg.n_neurons, cfg.enc_proj)
        self.gru = nn.GRU(cfg.enc_proj + cfg.u_dim, cfg.enc_hidden,
                          batch_first=True, bidirectional=True)
        h2 = 2 * cfg.enc_hidden
        self.to_z0 = nn.Linear(h2, 2 * cfg.d)
        self.drift = MLP(cfg.d + h2 + cfg.u_dim + cfg.cond_dim * len(cfg.cond_sizes),
                         cfg.d, cfg.enc_drift_hidden, 2)

    def encode(self, Y, U=None):
        x = torch.tanh(self.proj(Y))
        if U is not None:
            x = torch.cat([x, U], dim=-1)
        H, _ = self.gru(x)
        z0 = self.to_z0(H[:, 0])
        mu0, logvar0 = z0[..., :self.cfg.d], z0[..., self.cfg.d:].clamp(-8, 4)
        return H, mu0, logvar0

    def step_drift(self, z, h, u=None, cond=None):
        parts = [z, h]
        if u is not None:
            parts.append(u)
        if cond is not None:
            parts.append(cond)
        return self.drift(torch.cat(parts, dim=-1))


class FlowVAE(nn.Module):
    """Latent-SDE VAE whose drift is the flow field of interest.

    ``forward`` returns a dict of loss terms (already normalised per valid
    timepoint) plus the sampled latent path, so the notebook can inspect any
    piece of the objective."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.flow = GatedFlow(cfg)
        self.encoder = Encoder(cfg)
        self.W = nn.Parameter(torch.randn(cfg.n_neurons, cfg.d) / math.sqrt(cfg.d))
        self.c_out = nn.Parameter(torch.zeros(cfg.n_neurons))
        self.log_s = nn.Parameter(torch.zeros(cfg.n_neurons))
        self.log_sigma_z = nn.Parameter(
            torch.full((cfg.d,), math.log(cfg.sigma_z_init)),
            requires_grad=cfg.learn_sigma_z)
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(int(n), cfg.cond_dim) for name, n in cfg.cond_sizes.items()})
        for emb in self.embeddings.values():
            nn.init.normal_(emb.weight, std=0.01)

    # -- conditioning -------------------------------------------------------
    def cond_vector(self, labels: Optional[dict], batch: int, device) -> Optional[torch.Tensor]:
        """Concatenated embeddings for a batch (``None`` when unconditioned).

        A scalar or length-1 label is broadcast to ``batch`` rows, which is what
        field extraction needs: ``field_fn(model, labels={"condition": [2]})``
        evaluates one partition's field on a whole grid of latent points."""
        if not self.embeddings:
            return None
        parts = []
        for name in sorted(self.embeddings):
            if labels is not None and name in labels:
                idx = labels[name]
                if not torch.is_tensor(idx):
                    idx = torch.as_tensor(np.atleast_1d(np.asarray(idx)), device=device)
                idx = idx.reshape(-1).to(device=device, dtype=torch.long)
                if idx.numel() == 1 and batch > 1:
                    idx = idx.expand(batch)
                elif idx.numel() != batch:
                    raise ValueError(f"label {name!r} has {idx.numel()} entries "
                                     f"for a batch of {batch}")
            else:
                idx = torch.zeros(batch, dtype=torch.long, device=device)
            parts.append(self.embeddings[name](idx))
        return torch.cat(parts, dim=-1)

    # -- observation model --------------------------------------------------
    def readout(self, z, base=None):
        x = z @ self.W.T + self.c_out
        if base is not None:
            x = x + base
        return Fnn.softplus(x) if self.cfg.softplus_readout else x

    def obs_nll(self, y, mu):
        if self.cfg.obs == "poisson":
            rate = mu.clamp_min(1e-6)
            return rate - y * torch.log(rate)
        s = self.log_s.clamp(-6, 4).exp()
        return 0.5 * (((y - mu) / s) ** 2) + self.log_s.clamp(-6, 4) + 0.5 * math.log(2 * math.pi)

    # -- the objective ------------------------------------------------------
    def forward(self, Y, mask, *, base=None, U=None, labels=None,
                sample: bool = True, c_dyn: float = 0.1, kl0_weight: float = 1.0,
                seg_weight=None):
        cfg = self.cfg
        Bn, T, _ = Y.shape
        device = Y.device
        # mask stays a strict 0/1 validity indicator; per-segment loss weights are
        # kept separate so they cannot leak into the transition mask or the
        # normalisers (see time_balanced_weights)
        wt = (torch.ones((Bn, 1), device=device) if seg_weight is None
              else seg_weight.reshape(Bn, 1).float())
        wmask = mask * wt
        cond = self.cond_vector(labels, Bn, device)
        H, mu0, logvar0 = self.encoder.encode(Y, U)
        H = H.float()
        sigma_z = self.log_sigma_z.clamp(-8, 2).exp().float()
        sqdt = math.sqrt(cfg.dt)

        mu0, logvar0 = mu0.float(), logvar0.float()
        z = mu0 + (torch.randn_like(mu0) * (0.5 * logvar0).exp() if sample else 0.0)
        dyn = torch.zeros((), device=device, dtype=torch.float32)
        zs = [z]
        # only the latent recursion has to be sequential; the readout is done in one
        # batched matmul afterwards (a per-step readout over ~500 neurons dominated
        # the step time otherwise)
        for t in range(T - 1):
            u_t = None if U is None else U[:, t]
            # cast both drifts to fp32 *before* differencing: the dynamics term is a
            # difference of two network outputs, so in bf16 it would be swamped by
            # rounding once the posterior drift approaches the field.
            g = self.encoder.step_drift(z, H[:, t], u_t, cond).float()
            f = self.flow(z, u_t, cond).float()
            # Euler/Girsanov KL between posterior and prior SDE
            step_kl = 0.5 * cfg.dt * (((g - f) / sigma_z) ** 2).sum(-1)
            dyn = dyn + (step_kl * wmask[:, t] * mask[:, t + 1]).sum().float()
            noise = torch.randn_like(z) * sigma_z * sqdt if sample else 0.0
            z = z + cfg.dt * g + noise
            zs.append(z)

        Z = torch.stack(zs, dim=1)                        # (B, T, d)
        mu = self.readout(Z, base).float()
        recon = (self.obs_nll(Y.float(), mu) * wmask.unsqueeze(-1)).sum().float()

        kl0 = 0.5 * ((mu0 ** 2 / cfg.sigma0_prior ** 2 + logvar0.exp() / cfg.sigma0_prior ** 2
                      - logvar0 - 1 + 2 * math.log(cfg.sigma0_prior)).sum(-1) * wt.squeeze(-1)).sum()
        n_valid = wmask.sum().clamp_min(1.0)
        out = {
            "recon": recon / n_valid,
            "dyn": dyn / n_valid,
            "kl0": kl0 / n_valid,
            "z": Z,
            "mu": mu,
        }
        out["loss"] = out["recon"] + c_dyn * out["dyn"] + kl0_weight * out["kl0"]
        return out

    # -- the field ----------------------------------------------------------
    @torch.inference_mode()
    def drift_np(self, Z: np.ndarray, *, labels: Optional[dict] = None,
                 u: Optional[np.ndarray] = None, device=None,
                 chunk: int = 200_000) -> np.ndarray:
        """Evaluate ``F`` at arbitrary latent points. ``Z`` is ``(M, d)`` -> ``(M, d)``."""
        device = device or next(self.parameters()).device
        Z = np.atleast_2d(np.asarray(Z, dtype=np.float32))
        out = np.empty_like(Z)
        for i in range(0, Z.shape[0], chunk):
            zt = torch.as_tensor(Z[i:i + chunk], device=device)
            cond = self.cond_vector(labels, zt.shape[0], device)
            ut = None
            if u is not None:
                ut = torch.as_tensor(np.broadcast_to(np.asarray(u, dtype=np.float32),
                                                     (zt.shape[0], len(np.ravel(u)))).copy(),
                                     device=device)
            out[i:i + chunk] = self.flow(zt, ut, cond).float().cpu().numpy()
        return out

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def _to_device_batch(segset: SegmentSet, idx: np.ndarray, device, u_dim: int):
    """Move one batch to the device, trimmed to the batch's own longest segment.

    Trimming matters: segments range from ~8 to ~150 bins, and the latent
    recursion costs one sequential step per padded time point, so batching short
    trials against the global maximum would spend most of the time on padding."""
    Tb = int(segset.mask[idx].sum(axis=1).max())
    Y = torch.as_tensor(segset.Y[idx][:, :Tb], device=device)
    mask = torch.as_tensor(segset.mask[idx][:, :Tb], device=device)
    base = (torch.as_tensor(segset.B[idx][:, :Tb], device=device)
            if segset.B is not None else None)
    U = (torch.as_tensor(segset.U[idx][:, :Tb, :u_dim], device=device)
         if (segset.U is not None and u_dim > 0) else None)
    labels = {k: torch.as_tensor(v[idx], device=device) for k, v in segset.labels.items()}
    return Y, mask, base, U, labels


def init_readout_from_pca(model: FlowVAE, segset: SegmentSet, *,
                          idx: Optional[np.ndarray] = None) -> None:
    """Initialise the readout ``W`` (and intercept) from PCA of the activity.

    Starting the readout at the data's principal subspace, scaled so the implied
    latent has unit variance, stops the model from pruning a latent dimension
    before it has learnt to use it -- and costs nothing, since PCA of the
    observed activity is exactly the subspace the latent should span."""
    idx = np.arange(len(segset)) if idx is None else np.asarray(idx, dtype=int)
    rows = [segset.Y[i, segset.mask[i] > 0] for i in idx]
    X = np.concatenate(rows, axis=0).astype(np.float64)
    if segset.B is not None:
        Bres = np.concatenate([segset.B[i, segset.mask[i] > 0] for i in idx], axis=0)
        X = X - np.log1p(np.exp(np.clip(Bres, -30, 30)))       # residual after nuisance
    mu = X.mean(0)
    Xc = X - mu
    d = model.cfg.d
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    n = max(Xc.shape[0], 2)
    score_sd = np.maximum(S[:d] / math.sqrt(n - 1), 1e-6)
    W0 = (Vt[:d] * score_sd[:, None]).T                        # (N, d), unit-variance latent
    with torch.no_grad():
        model.W.copy_(torch.as_tensor(W0, dtype=model.W.dtype, device=model.W.device))
        if segset.B is None:
            inv = np.log(np.expm1(np.clip(mu, 1e-3, None)))
            model.c_out.copy_(torch.as_tensor(inv, dtype=model.c_out.dtype,
                                              device=model.c_out.device))
        resid = Xc - Xc @ (Vt[:d].T @ Vt[:d])
        model.log_s.fill_(float(np.log(max(resid.std(), 1e-3))))


@dataclass
class FitResult:
    """A fitted model plus its optimisation history and held-out scores."""
    model: FlowVAE
    history: dict
    train_idx: np.ndarray
    val_idx: np.ndarray
    best_epoch: int
    best_val: float
    config: ModelConfig
    train_config: TrainConfig

    @property
    def summary(self) -> str:
        return (f"d={self.config.d} params={self.model.n_params()} "
                f"best_epoch={self.best_epoch} val_loss={self.best_val:.4f}")


def fit_flow_model(segset: SegmentSet, mcfg: ModelConfig, tcfg: TrainConfig, *,
                   device=None, train_idx=None, val_idx=None) -> FitResult:
    """Fit the latent-SDE model to a :class:`SegmentSet`.

    The whole segment tensor is moved to the GPU once (a day at 6 Hz is only
    tens of MB), so there is no dataloader in the inner loop. Early stopping
    uses the held-out *segments* total loss."""
    device = device or get_device()
    enable_fast_matmul()
    set_seed(tcfg.seed)
    mcfg.n_neurons = segset.n_neurons
    model = FlowVAE(mcfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)

    if train_idx is None or val_idx is None:
        train_idx, val_idx = split_train_val(segset, val_frac=tcfg.val_frac, seed=tcfg.seed)
    if tcfg.pca_init:
        init_readout_from_pca(model, segset, idx=train_idx)
    seg_w = (time_balanced_weights(segset, "condition") if tcfg.balance_conditions
             else np.ones(len(segset)))

    amp = bool(tcfg.amp and device.type == "cuda")
    hist = {"epoch": [], "train": [], "val": [], "recon": [], "dyn": []}
    best_val, best_epoch, best_state, bad = np.inf, -1, None, 0
    rng = np.random.default_rng(tcfg.seed)
    nb = max(1, int(math.ceil(train_idx.size / tcfg.batch_size)))

    for ep in range(tcfg.epochs):
        ramp = min(1.0, (ep + 1) / max(tcfg.warmup_epochs, 1))
        c_dyn, kl0 = tcfg.c_dyn * ramp, tcfg.kl0_weight * ramp
        model.train()
        tot = totn = 0.0
        perm = rng.permutation(train_idx)
        for bi in range(nb):
            pick = perm[bi * tcfg.batch_size:(bi + 1) * tcfg.batch_size]
            if pick.size == 0:
                continue
            Y, mask, base, U, labels = _to_device_batch(segset, pick, device, mcfg.u_dim)
            sw = torch.as_tensor(seg_w[pick], device=device, dtype=torch.float32)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                out = model(Y, mask, base=base, U=U, labels=labels, seg_weight=sw,
                            c_dyn=c_dyn, kl0_weight=kl0)
                loss = out["loss"]
            opt.zero_grad(set_to_none=True)
            loss.float().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            opt.step()
            tot += float(loss.detach()) * pick.size
            totn += pick.size
            del Y, mask, base, U, labels, out, loss, sw

        if ep % tcfg.eval_every == 0 or ep == tcfg.epochs - 1:
            val = evaluate(model, segset, val_idx, device=device, c_dyn=tcfg.c_dyn,
                           kl0_weight=tcfg.kl0_weight, amp=amp, seg_weights=seg_w)
            hist["epoch"].append(ep)
            hist["train"].append(tot / max(totn, 1))
            hist["val"].append(val["loss"])
            hist["recon"].append(val["recon"])
            hist["dyn"].append(val["dyn"])
            if val["loss"] < best_val - 1e-5:
                best_val, best_epoch, bad = val["loss"], ep, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += tcfg.eval_every
            if tcfg.verbose and (ep % (10 * tcfg.eval_every) == 0):
                print(f"  ep {ep:4d}  train {tot / max(totn, 1):9.4f}  "
                      f"val {val['loss']:9.4f}  (recon {val['recon']:.4f} dyn {val['dyn']:.4f})")
            if ep >= tcfg.min_epochs and bad >= tcfg.patience:
                if tcfg.verbose:
                    print(f"  early stop at epoch {ep} (best {best_epoch}, val {best_val:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    free(best_state)
    return FitResult(model=model, history=hist, train_idx=train_idx, val_idx=val_idx,
                     best_epoch=best_epoch, best_val=best_val,
                     config=mcfg, train_config=tcfg)


@torch.no_grad()
def evaluate(model: FlowVAE, segset: SegmentSet, idx: np.ndarray, *, device=None,
             c_dyn: float = 0.1, kl0_weight: float = 1.0, amp: bool = False,
             n_samples: int = 1, batch_size: int = 256,
             seg_weights: Optional[np.ndarray] = None) -> dict:
    """Mean loss terms on the given segments (per valid timepoint, summed over neurons).

    Pass the same ``seg_weights`` used in training so that held-out scores are
    comparable to training scores and are not dominated by whichever condition
    happens to contribute the most time."""
    device = device or next(model.parameters()).device
    model.eval()
    idx = np.asarray(idx, dtype=int)
    if idx.size == 0:
        return {"loss": np.nan, "recon": np.nan, "dyn": np.nan, "kl0": np.nan}
    acc = {"loss": 0.0, "recon": 0.0, "dyn": 0.0, "kl0": 0.0}
    wsum = 0.0
    for i in range(0, idx.size, batch_size):
        pick = idx[i:i + batch_size]
        Y, mask, base, U, labels = _to_device_batch(segset, pick, device, model.cfg.u_dim)
        sw = (torch.as_tensor(seg_weights[pick], device=device, dtype=torch.float32)
              if seg_weights is not None else None)
        wt = float(mask.sum() if sw is None else (mask * sw.reshape(-1, 1)).sum())
        for _ in range(n_samples):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                out = model(Y, mask, base=base, U=U, labels=labels, seg_weight=sw,
                            c_dyn=c_dyn, kl0_weight=kl0_weight)
            for k in acc:
                acc[k] += float(out[k].float()) * wt / n_samples
        wsum += wt
        del Y, mask, base, U, labels, out, sw
    return {k: v / max(wsum, 1e-9) for k, v in acc.items()}


@torch.no_grad()
def infer_latents(model: FlowVAE, segset: SegmentSet, idx: Optional[np.ndarray] = None, *,
                  device=None, sample: bool = False, batch_size: int = 128):
    """Posterior latent paths for segments: list of ``(T_i, d)`` arrays."""
    device = device or next(model.parameters()).device
    idx = np.arange(len(segset)) if idx is None else np.asarray(idx, dtype=int)
    model.eval()
    out = []
    for i in range(0, idx.size, batch_size):
        pick = idx[i:i + batch_size]
        Y, mask, base, U, labels = _to_device_batch(segset, pick, device, model.cfg.u_dim)
        res = model(Y, mask, base=base, U=U, labels=labels, sample=sample)
        Z = res["z"].float().cpu().numpy()
        m = mask.cpu().numpy()
        for j in range(pick.size):
            out.append(Z[j, m[j] > 0])
        del Y, mask, base, U, labels, res
    return out


@torch.no_grad()
def predict_activity(model: FlowVAE, segset: SegmentSet, idx: Optional[np.ndarray] = None, *,
                     device=None, batch_size: int = 128):
    """Predicted mean activity per segment: list of ``(T_i, N)`` arrays."""
    device = device or next(model.parameters()).device
    idx = np.arange(len(segset)) if idx is None else np.asarray(idx, dtype=int)
    model.eval()
    out = []
    for i in range(0, idx.size, batch_size):
        pick = idx[i:i + batch_size]
        Y, mask, base, U, labels = _to_device_batch(segset, pick, device, model.cfg.u_dim)
        res = model(Y, mask, base=base, U=U, labels=labels, sample=False)
        mu = res["mu"].float().cpu().numpy()
        m = mask.cpu().numpy()
        for j in range(pick.size):
            out.append(mu[j, m[j] > 0])
        del Y, mask, base, U, labels, res
    return out


# ---------------------------------------------------------------------------
# canonical latent coordinates
# ---------------------------------------------------------------------------
class LatentTransform:
    """Invertible linear change of latent coordinates ``zc = A z + shift``.

    The model's ``z`` is only defined up to an invertible linear map. Following
    the paper we take ``A = S V^T`` from ``W = U S V^T``, which makes the readout
    a semi-orthogonal isometry, so **distances and angles in ``zc`` correspond to
    those in neural space**. The field in the new coordinates is
    ``Fc(zc) = A F(A^{-1} (zc - shift))``, evaluated exactly (no refitting).
    """

    def __init__(self, A: np.ndarray, shift: Optional[np.ndarray] = None):
        self.A = np.asarray(A, dtype=np.float64)
        self.Ainv = np.linalg.inv(self.A)
        self.shift = np.zeros(self.A.shape[0]) if shift is None else np.asarray(shift, float)

    def forward(self, Z: np.ndarray) -> np.ndarray:
        return np.atleast_2d(Z) @ self.A.T + self.shift

    def inverse(self, Zc: np.ndarray) -> np.ndarray:
        return (np.atleast_2d(Zc) - self.shift) @ self.Ainv.T

    def velocity(self, V: np.ndarray) -> np.ndarray:
        """Push a velocity (not a point) into canonical coordinates."""
        return np.atleast_2d(V) @ self.A.T

    def compose(self, R: np.ndarray) -> "LatentTransform":
        """Apply a further linear map ``R`` (e.g. a rotation) after this one."""
        R = np.asarray(R, dtype=np.float64)
        return LatentTransform(R @ self.A, R @ self.shift)


def canonical_transform(model: FlowVAE) -> LatentTransform:
    """The paper's isometric latent transform, from the SVD of the readout ``W``."""
    W = model.W.detach().float().cpu().numpy().astype(np.float64)
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    return LatentTransform(np.diag(S) @ Vt)


def normalize_scale(transform: LatentTransform, Zc: np.ndarray, *,
                    mode: str = "pc1_span") -> LatentTransform:
    """Rescale a latent frame by a single global factor, so distances are O(1).

    A PCA-initialised readout leaves the canonical latent with a scale set by the
    activity (sd ~14 on real data), which silently breaks every distance-based
    default in this module (``merge_dist``, ``capture_dist``, ``settle_tol``).
    Scaling is *isotropic*, so all angles, speed ratios and shape statistics are
    unchanged.

    ``mode="pc1_span"`` follows the paper (latent along PC1 spans about [-1, 1]);
    ``mode="rms"`` sets the pooled radius to 1."""
    Zc = np.atleast_2d(Zc)
    if mode == "rms":
        s = float(np.sqrt((Zc ** 2).sum(axis=1).mean()))
    else:
        s = float(np.abs(Zc[:, 0]).max())
    s = s if s > 1e-9 else 1.0
    return transform.compose(np.eye(transform.A.shape[0]) / s)


def pca_align_transform(Zc: np.ndarray, *, scale_to_unit: bool = True) -> np.ndarray:
    """Rotation aligning canonical axes to the PCs of a set of latent states.

    The paper defines its ``z1``/``z2`` axes by PCA on simulated trajectories and
    scales so PC1 spans ~[-1, 1]. Returns a matrix to pass to
    :meth:`LatentTransform.compose`."""
    Zc = np.asarray(Zc, dtype=np.float64)
    Zc = Zc - Zc.mean(0)
    _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
    R = Vt
    if scale_to_unit:
        span = np.abs(Zc @ R.T).max(axis=0)
        span[span < 1e-9] = 1.0
        R = np.diag(1.0 / span) @ R
    return R


def canonical_readout(model: FlowVAE):
    """``(U, transform)``: the orthonormal readout basis and the matching latent map.

    ``U`` (``N x d``, orthonormal columns) lives in *neuron* space, which is what
    makes two independently fitted models comparable: the animals' neurons are
    tracked across days, so two models' readouts can be aligned even though their
    latent frames are arbitrary."""
    W = model.W.detach().float().cpu().numpy().astype(np.float64)
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    return U, LatentTransform(np.diag(S) @ Vt)


def align_models(model_a: FlowVAE, model_b: FlowVAE):
    """Map model B's canonical latent frame onto model A's, via the shared neurons.

    Returns ``(transform_b, info)``. ``transform_b`` takes B's *canonical* latents
    (and velocities) into A's canonical frame, so the two fields can be evaluated
    on one grid and compared directly.

    ``info`` separates the two ways two fits can differ, which must not be
    conflated:

    * ``subspace_mismatch`` / ``principal_angles_deg`` -- the models describe
      *different neural subspaces*. Compare fields only to the extent this is
      small; large angles mean the comparison is between different coordinates,
      not different dynamics.
    * whatever :func:`field_distance` then reports -- different *flow within the
      same subspace*, which is the dynamical claim.
    """
    Ua, _ = canonical_readout(model_a)
    Ub, _ = canonical_readout(model_b)
    # orthogonal Procrustes: Ub @ Q ~= Ua  =>  Ub z = Ua (Q^T z)
    M, s, Nt = np.linalg.svd(Ub.T @ Ua)
    Q = M @ Nt
    resid = np.linalg.norm(Ub @ Q - Ua) / max(np.linalg.norm(Ua), 1e-12)
    try:
        from scipy.linalg import subspace_angles
        ang = np.degrees(subspace_angles(Ua, Ub))
    except Exception:
        ang = np.array([np.nan])
    info = dict(subspace_mismatch=float(resid),
                principal_angles_deg=np.sort(ang),
                procrustes_singular_values=s)
    return LatentTransform(Q.T), info


def reproducible_subspace(models: Sequence[FlowVAE], *,
                          angle_threshold_deg: float = 30.0) -> dict:
    """How many latent dimensions are actually *identified*, across refits?

    Fits the same data with different seeds and compares the readout subspaces.
    A dimension that changes orientation from seed to seed carries no dynamical
    claim, however well it fits.

    This is a **sharper test than occupancy anisotropy** and it is not implied by
    it. On a real day with ``d=3``, occupancy passed the anisotropy guardrail
    (``sd_ratio`` 0.39, third dimension at 39 % of the first's sd) yet two seeds'
    principal angles were 1.6 deg, 7.0 deg and **66 deg** -- the first two
    dimensions were reproducible and the third was arbitrary. Report
    ``n_reproducible`` and confine field claims to that many dimensions."""
    models = list(models)
    if len(models) < 2:
        return dict(n_reproducible=np.nan, angles=np.array([]), pairs=0)
    Us = [canonical_readout(m)[0] for m in models]
    from scipy.linalg import subspace_angles
    per_dim = []
    for i in range(len(Us)):
        for j in range(i + 1, len(Us)):
            per_dim.append(np.degrees(np.sort(subspace_angles(Us[i], Us[j]))))
    A = np.stack(per_dim)                    # (pairs, d) sorted ascending
    worst = A.max(axis=0)
    return dict(n_reproducible=int((worst <= angle_threshold_deg).sum()),
                angles_mean=A.mean(axis=0), angles_worst=worst,
                pairs=A.shape[0], threshold_deg=angle_threshold_deg)


def field_fn(model: FlowVAE, *, transform: Optional[LatentTransform] = None,
             labels: Optional[dict] = None, u: Optional[np.ndarray] = None) -> Callable:
    """Return ``f(Zc) -> Vc``: the flow field as a plain numpy function.

    This is the single object every downstream analysis consumes, so a field can
    come from a fitted model, a different partition of the same model (via
    ``labels``), or a synthetic system, interchangeably."""
    if transform is None:
        return lambda Z: model.drift_np(Z, labels=labels, u=u)

    def f(Zc):
        Z = transform.inverse(Zc)
        V = model.drift_np(Z, labels=labels, u=u)
        return transform.velocity(V)
    return f


# ---------------------------------------------------------------------------
# grids, sample zone, simulation
# ---------------------------------------------------------------------------
def make_grid(limits: Sequence[tuple], n: int | Sequence[int] = 60):
    """Regular grid over a box. Returns ``(points (M, d), shape, axes)``."""
    d = len(limits)
    ns = [n] * d if isinstance(n, int) else list(n)
    axes = [np.linspace(lo, hi, k) for (lo, hi), k in zip(limits, ns)]
    mesh = np.meshgrid(*axes, indexing="ij")
    pts = np.stack([m.ravel() for m in mesh], axis=-1)
    return pts, tuple(ns), axes


def limits_from_points(Z: np.ndarray, *, pad: float = 0.15,
                       quantile: float = 0.005) -> list:
    """Bounding box of a point cloud with a margin, robust to outliers."""
    Z = np.atleast_2d(Z)
    lo = np.quantile(Z, quantile, axis=0)
    hi = np.quantile(Z, 1 - quantile, axis=0)
    span = np.maximum(hi - lo, 1e-6)
    return [(float(l - pad * s), float(h + pad * s)) for l, h, s in zip(lo, hi, span)]


def speed(V: np.ndarray) -> np.ndarray:
    """Euclidean norm of each velocity vector."""
    return np.linalg.norm(np.atleast_2d(V), axis=-1)


def simulate_field(field: Callable, z0: np.ndarray, n_steps: int, dt: float, *,
                   sigma: Optional[np.ndarray] = None, rng=None,
                   clip: Optional[float] = 50.0) -> np.ndarray:
    """Euler-Maruyama rollout of a field. ``z0`` is ``(B, d)`` -> ``(B, n_steps+1, d)``."""
    rng = np.random.default_rng() if rng is None else rng
    z = np.atleast_2d(np.asarray(z0, dtype=np.float64)).copy()
    out = np.empty((z.shape[0], n_steps + 1, z.shape[1]))
    out[:, 0] = z
    sq = math.sqrt(dt)
    for t in range(n_steps):
        v = field(z)
        z = z + dt * v
        if sigma is not None:
            z = z + sq * np.asarray(sigma) * rng.standard_normal(z.shape)
        if clip is not None:
            np.clip(z, -clip, clip, out=z)
        out[:, t + 1] = z
    return out


def model_sigma_z(model: FlowVAE, transform: Optional[LatentTransform] = None) -> np.ndarray:
    """Latent diffusion SD, optionally expressed in canonical coordinates.

    In canonical coordinates the diffusion is generally no longer diagonal; the
    returned value is the per-axis SD of ``A diag(sigma) eps``, which is what a
    per-axis simulation needs."""
    s = model.log_sigma_z.detach().float().cpu().numpy().clip(-8, 2)
    sig = np.exp(s)
    if transform is None:
        return sig
    C = transform.A @ np.diag(sig ** 2) @ transform.A.T
    return np.sqrt(np.clip(np.diag(C), 1e-12, None))


def sample_zone(field: Callable, z0: np.ndarray, *, dt: float, n_steps: int,
                limits: Sequence[tuple], grid_n: int = 60,
                sigma: Optional[np.ndarray] = None, n_sim: int = 5000,
                min_count: int = 50, rng=None):
    """The paper's "sample zone": the part of state space the model actually visits.

    Simulates ``n_sim`` trajectories from the generative model and keeps grid
    cells visited by at least ``min_count`` of them (their criterion: 50 of
    5000). Returns ``(mask (grid_shape) bool, counts, axes)``."""
    rng = np.random.default_rng(0) if rng is None else rng
    z0 = np.atleast_2d(np.asarray(z0, dtype=np.float64))
    pick = rng.integers(0, z0.shape[0], size=n_sim)
    traj = simulate_field(field, z0[pick], n_steps, dt, sigma=sigma, rng=rng)
    _, shape, axes = make_grid(limits, grid_n)
    d = traj.shape[-1]
    idx = []
    ok = np.ones(traj.shape[0] * traj.shape[1], dtype=bool)
    flat = traj.reshape(-1, d)
    for k in range(d):
        e = axes[k]
        step = e[1] - e[0]
        i = np.floor((flat[:, k] - e[0]) / step + 0.5).astype(int)
        ok &= (i >= 0) & (i < shape[k])
        idx.append(np.clip(i, 0, shape[k] - 1))
    lin = np.ravel_multi_index([i[ok] for i in idx], shape)
    # count distinct trajectories per cell, not visits
    traj_id = np.repeat(np.arange(traj.shape[0]), traj.shape[1])[ok]
    counts = np.zeros(int(np.prod(shape)), dtype=np.int32)
    uniq = np.unique(np.stack([lin, traj_id]), axis=1)
    np.add.at(counts, uniq[0], 1)
    counts = counts.reshape(shape)
    return counts >= min_count, counts, axes


def occupancy(Z: np.ndarray, limits: Sequence[tuple], grid_n: int = 60, *,
              weights: Optional[np.ndarray] = None, normalize: bool = True):
    """Histogram of visited latent states (per unit time when ``normalize``).

    Always compare conditions using *density per unit time*: a 20 s timeout
    contributes far more samples than a 3 s hit."""
    Z = np.atleast_2d(Z)
    _, shape, axes = make_grid(limits, grid_n)
    edges = []
    for e in axes:
        step = e[1] - e[0]
        edges.append(np.concatenate([e - step / 2, [e[-1] + step / 2]]))
    H, _ = np.histogramdd(Z, bins=edges, weights=weights)
    if normalize and H.sum() > 0:
        H = H / H.sum()
    return H, axes


# ---------------------------------------------------------------------------
# fixed points, stability, basins
# ---------------------------------------------------------------------------
@dataclass
class FixedPoint:
    """A **slow point** of the flow field, with its local linearisation.

    Deliberately not called a root: a *fitted* field essentially never has an
    exact zero, so what is reported is a local minimum of ``||F||`` together with
    its residual speed. ``rel_speed`` (residual over the field's median speed) is
    the number that says whether to read it as a fixed point at all: values of a
    few percent behave like one, values near 1 do not.
    """
    z: np.ndarray
    speed: float
    eigenvalues: np.ndarray
    jacobian: np.ndarray
    kind: str                     # "stable" | "unstable" | "saddle"
    is_spiral: bool
    slowest_tau: float            # seconds; time constant of the slowest mode
    n_unstable: int
    non_normality: float
    n_marginal: int = 0          # near-zero eigenvalues => continuous attractor
    tau_exceeds_window: bool = False
    rel_speed: float = np.nan     # speed / median field speed
    in_sample_zone: bool = True
    basin_frac: float = np.nan    # filled in by find_attractors_by_simulation
    endpoint_rel_speed: float = np.nan   # mean relative speed where trajectories ended up

    @property
    def label(self) -> str:
        s = self.kind + (" spiral" if self.is_spiral else "")
        return (f"{s} @ ({', '.join(f'{v:.2f}' for v in self.z)})"
                f" [rel_speed={self.rel_speed:.3f}]")


def numeric_jacobian(field: Callable, z: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Central-difference Jacobian of a field at one point."""
    z = np.asarray(z, dtype=np.float64).ravel()
    d = z.size
    J = np.empty((d, d))
    for k in range(d):
        dz = np.zeros(d)
        dz[k] = eps
        J[:, k] = (field(z + dz).ravel() - field(z - dz).ravel()) / (2 * eps)
    return J


def classify_jacobian(J: np.ndarray, *, marginal_tol: float = 0.05) -> dict:
    """Stability class, timescales, marginal directions and non-normality.

    ``n_marginal`` counts eigenvalues whose real part is within ``marginal_tol``
    of zero (in units of the largest |Re|). It is the signature that separates a
    **continuous** attractor from a point attractor: 0 marginal directions is an
    isolated fixed point, 1 is a line attractor, and so on. It must be *reported*
    rather than filtered out -- a near-zero eigenvalue is exactly what a
    continuous attractor has."""
    ev = np.linalg.eigvals(J)
    re = ev.real
    n_unstable = int((re > 1e-6).sum())
    if n_unstable == 0:
        kind = "stable"
    elif n_unstable == J.shape[0]:
        kind = "unstable"
    else:
        kind = "saddle"
    slow = re[np.argmin(np.abs(re))]
    tau = float(1.0 / abs(slow)) if abs(slow) > 1e-9 else np.inf
    scale = max(float(np.max(np.abs(re))), 1e-12)
    n_marginal = int((np.abs(re) / scale < marginal_tol).sum())
    nn = float(np.linalg.norm(J @ J.T - J.T @ J, "fro") /
               max(np.linalg.norm(J, "fro") ** 2, 1e-12))
    return dict(eigenvalues=ev, kind=kind, is_spiral=bool(np.any(np.abs(ev.imag) > 1e-6)),
                slowest_tau=tau, n_unstable=n_unstable, n_marginal=n_marginal,
                non_normality=nn)


def find_slow_points(field: Callable, limits: Sequence[tuple], *,
                     grid_n: int = 40, merge_dist: float = 0.1,
                     max_starts: int = 200, rel_speed_max: Optional[float] = None,
                     zone_mask: Optional[np.ndarray] = None,
                     zone_axes: Optional[Sequence[np.ndarray]] = None,
                     seed_points: Optional[np.ndarray] = None,
                     points: Optional[np.ndarray] = None,
                     support_states: Optional[np.ndarray] = None,
                     support_rad: Optional[float] = None) -> list:
    """Locate and classify the slow points of a field (candidate fixed points).

    Strategy: evaluate ``||F||`` on a grid, take true local minima, refine each by
    minimising ``||F||^2``, deduplicate, then classify by the Jacobian. This is
    the standard treatment for a *fitted* field, and it is what the earlier
    root-only search got wrong: on a synthetic bistable system whose field the
    model had recovered to cosine 0.955, requiring ``||F||`` below a few percent
    of the median speed found the saddle but **missed both attractors**, because
    a fitted network's minima are slow, not exactly zero.

    Every returned point carries ``rel_speed`` (residual over median field
    speed). Filter with ``rel_speed_max`` if you want only near-exact roots, but
    prefer reporting the value: a chain of slow points along a line is what an
    approximate *continuous* attractor looks like, and :func:`slow_region`'s
    ``elongation`` distinguishes that from a point attractor.

    Cross-check attractors with :func:`find_attractors_by_simulation`, which
    finds them by integration and so does not depend on this search at all.

    Pass ``points`` (e.g. the observed latent states) to search **without a
    grid**. Required above ~3 latent dimensions: a 25-point grid in 5-D is 9.8
    million cells, of which real data visited 1403 (0.014 %), so a grid search
    spends all its effort on empty space. Searching from visited states instead
    keeps the search where the data actually is.
    """
    from scipy import ndimage
    if points is not None:
        starts = np.atleast_2d(np.asarray(points, dtype=float))
        med_speed = float(np.median(speed(field(starts))))
        axes = None
        if starts.shape[0] > max_starts:
            order = np.argsort(speed(field(starts)))[:max_starts]
            starts = starts[order]
    else:
        pts, shape, axes = make_grid(limits, grid_n)
        sp = speed(field(pts))
        med_speed = float(np.median(sp))
        spg = sp.reshape(shape)
        # true local minima on the grid
        fp = ndimage.minimum_filter(spg, size=3, mode="nearest")
        cand = (spg <= fp) & np.isfinite(spg)
        if zone_mask is not None and zone_mask.shape == cand.shape:
            cand &= zone_mask
        starts = pts.reshape(shape + (pts.shape[-1],))[cand]
        if seed_points is not None and len(seed_points):
            starts = np.vstack([starts, np.atleast_2d(seed_points)])
        if starts.shape[0] > max_starts:
            order = np.argsort(speed(field(starts)))[:max_starts]
            starts = starts[order]

    lo = np.array([l for l, _ in limits], dtype=float)
    hi = np.array([h for _, h in limits], dtype=float)
    found = []
    for s in starts:
        try:
            sol = optimize.least_squares(lambda z: field(z).ravel(), s, bounds=(lo, hi),
                                         xtol=1e-12, ftol=1e-12, gtol=1e-12)
            z = sol.x if np.all(np.isfinite(sol.x)) else s
        except Exception:
            z = s
        if any(np.linalg.norm(z - f) < merge_dist for f in found):
            continue
        found.append(np.asarray(z, dtype=float))

    out = []
    for z in found:
        J = numeric_jacobian(field, z)
        info = classify_jacobian(J)
        s_here = float(np.linalg.norm(field(z).ravel()))
        rel = s_here / max(med_speed, 1e-12)
        if rel_speed_max is not None and rel > rel_speed_max:
            continue
        inside = True
        if support_states is not None:
            rad = (support_rad if support_rad is not None
                   else support_radius(support_states))
            inside = bool(support_distance(z[None, :], support_states)[0] <= rad)
        elif zone_mask is not None and zone_axes is not None:
            inside = bool(_lookup_mask(zone_mask, zone_axes, z))
        out.append(FixedPoint(z=z, speed=s_here, rel_speed=rel, jacobian=J,
                              in_sample_zone=inside, **info))
    out.sort(key=lambda p: (p.kind != "stable", p.rel_speed))
    return out


def find_attractors_by_simulation(field: Callable, limits: Sequence[tuple], *,
                                  dt: float = 1 / 6, n_steps: int = 3000,
                                  grid_n: int = 25, merge_dist: float = 0.15,
                                  settle_tol: float = 0.05,
                                  points: Optional[np.ndarray] = None,
                                  max_starts: int = 4000, rng=None,
                                  support_states: Optional[np.ndarray] = None,
                                  support_rad: Optional[float] = None,
                                  max_tau_sec: Optional[float] = None):
    """Find attractors by integrating the field and clustering where it ends up.

    Independent of any root search, so it is the honest cross-check on attractor
    *count*. Also reports each attractor's basin fraction, and flags trajectories
    that never settle (``n_unsettled``) -- a large value means a limit cycle or an
    escaping field rather than point attractors.

    ``n_steps`` must cover several times the field's slowest time constant, which
    for these fits reaches tens of seconds; the default integrates for
    ``n_steps * dt`` = 500 s at 6 Hz.

    Pass ``points`` (e.g. observed latent states) instead of relying on a grid
    when the latent has more than ~3 dimensions; ``basin_frac`` is then the
    fraction of *observed states* that flow to each attractor, which is the more
    meaningful quantity anyway.

    Pass ``support_states`` to mark which attractors lie inside the data (see
    :func:`support_distance`) and ``max_tau_sec`` (e.g. the session or segment
    length) to flag ones whose slowest time constant is longer than anything the
    recording could constrain. Both matter: an unfiltered real fit reported
    "attractors" with basin 0.00 and tau ~1e8 s sitting in saturated periphery.

    Returns ``(attractors, info)`` with attractors as :class:`FixedPoint` (their
    ``basin_frac`` filled in)."""
    rng = np.random.default_rng(0) if rng is None else rng
    if points is not None:
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        if pts.shape[0] > max_starts:
            pts = pts[rng.choice(pts.shape[0], max_starts, replace=False)]
    else:
        pts, _, _ = make_grid(limits, grid_n)
    traj = simulate_field(field, pts, n_steps, dt, sigma=None)
    ends = traj[:, -1]
    med_speed = float(np.median(speed(field(pts))))
    # Endpoint speed relative to the field's median, NOT a binary "settled" test.
    # Two earlier criteria both misled: a displacement threshold called everything
    # unsettled on a field whose slowest time constants were tens of seconds, and a
    # hard speed threshold then reported "no attractors" on a real fit where
    # trajectories demonstrably converged to a slow *region* (relative speed ~0.1)
    # rather than to an exact zero. The honest output is the distribution: a cluster
    # at relative speed 0.001 is a fixed point, one at 0.1 is a slow manifold, and
    # the difference matters here.
    end_rel = speed(field(ends)) / max(med_speed, 1e-12)
    settled = end_rel < settle_tol
    radius0 = np.linalg.norm(pts - pts.mean(0), axis=-1)
    radius1 = np.linalg.norm(ends - pts.mean(0), axis=-1)

    centers, members = [], []
    for i, e in enumerate(ends):
        if not np.all(np.isfinite(e)):
            continue
        hit = None
        for k, c in enumerate(centers):
            if np.linalg.norm(e - c) < merge_dist:
                hit = k
                break
        if hit is None:
            centers.append(e.copy())
            members.append([i])
        else:
            members[hit].append(i)
            centers[hit] = np.mean(ends[members[hit]], axis=0)

    out = []
    n_tot = max(pts.shape[0], 1)
    rad = None
    if support_states is not None:
        rad = support_rad if support_rad is not None else support_radius(support_states)
    for c, m in zip(centers, members):
        J = numeric_jacobian(field, c)
        cls = classify_jacobian(J)
        s_here = float(np.linalg.norm(field(c).ravel()))
        inside = True
        if support_states is not None:
            inside = bool(support_distance(np.asarray(c)[None, :], support_states)[0] <= rad)
        # A time constant longer than the recording is reported, NOT used to reject:
        # a continuous attractor has a zero eigenvalue by construction, so filtering
        # on tau threw away exactly the structure of most interest (it rejected all
        # 31 slow points of a synthetic line attractor). Data support decides
        # instead; `tau_exceeds_window` just flags that the timescale itself is not
        # constrained by a recording this long.
        long_tau = bool(max_tau_sec is not None and cls["slowest_tau"] > max_tau_sec)
        p = FixedPoint(z=np.asarray(c, float), speed=s_here,
                       rel_speed=s_here / max(med_speed, 1e-12), jacobian=J,
                       basin_frac=len(m) / n_tot, in_sample_zone=inside,
                       tau_exceeds_window=long_tau, **cls)
        p.endpoint_rel_speed = float(np.mean(end_rel[m]))
        out.append(p)
    out.sort(key=lambda p: -p.basin_frac)
    info = dict(n_unsettled=int((~settled).sum()), n_start=int(n_tot),
                frac_unsettled=float((~settled).mean()),
                endpoint_rel_speed_median=float(np.median(end_rel)),
                endpoint_rel_speed_q90=float(np.quantile(end_rel, 0.9)),
                radius_growth=float(np.median(radius1) / max(np.median(radius0), 1e-9)),
                support_radius=rad,
                n_in_support=int(sum(p.in_sample_zone for p in out)))
    return out, info


def find_fixed_points(field: Callable, limits: Sequence[tuple], *,
                      rel_speed_max: float = 0.05, **kw) -> list:
    """Slow points restricted to near-exact roots (thin wrapper on
    :func:`find_slow_points`; prefer that function and report ``rel_speed``)."""
    kw.pop("n_seeds", None)
    kw.pop("tol", None)
    return find_slow_points(field, limits, rel_speed_max=rel_speed_max, **kw)


def support_distance(Z_query: np.ndarray, Z_states: np.ndarray) -> np.ndarray:
    """Distance from each query point to the nearest observed latent state.

    The dimension-agnostic replacement for a grid mask, and the thing to filter
    slow points and attractors by. On a real fit this mattered: the search
    returned "attractors" with basins of 0.00 and slowest time constants of
    ~1e8 s, which are not slow attractors but flat, unvisited periphery where the
    flow network's soft output bound has saturated."""
    from scipy.spatial import cKDTree
    tree = cKDTree(np.atleast_2d(np.asarray(Z_states, dtype=float)))
    d, _ = tree.query(np.atleast_2d(np.asarray(Z_query, dtype=float)), k=1)
    return np.asarray(d, dtype=float)


def support_radius(Z_states: np.ndarray, *, quantile: float = 0.95, k: int = 20,
                   sample: int = 5000, seed: int = 0,
                   min_frac_of_scale: float = 0.02) -> float:
    """A ``radius`` for :func:`support_distance` that does not shrink with data size.

    Uses the distance to the ``k``-th nearest state (not the 1st) and floors the
    result at ``min_frac_of_scale`` of the occupancy scale. The naive
    nearest-neighbour version was density dependent: on densely sampled data the
    radius collapsed toward zero and *every* attractor was then judged to be
    outside the data (a synthetic line attractor had all 31 of its slow points
    rejected), while on sparse data it was far too permissive."""
    from scipy.spatial import cKDTree
    Z = np.atleast_2d(np.asarray(Z_states, dtype=float))
    if Z.shape[0] > sample:
        rng = np.random.default_rng(seed)
        Z = Z[rng.choice(Z.shape[0], sample, replace=False)]
    kk = int(min(max(k, 1), max(Z.shape[0] - 1, 1)))
    tree = cKDTree(Z)
    d, _ = tree.query(Z, k=kk + 1)
    rad = float(np.quantile(d[:, -1], quantile))
    scale = float(np.sqrt(np.sum(Z.var(axis=0))))
    return max(rad, min_frac_of_scale * scale)


def lookup_mask(mask: np.ndarray, axes: Sequence[np.ndarray], z: np.ndarray) -> bool:
    """Is point ``z`` inside a boolean grid mask (e.g. a sample zone)?"""
    return _lookup_mask(mask, axes, z)


def _lookup_mask(mask: np.ndarray, axes: Sequence[np.ndarray], z: np.ndarray) -> bool:
    idx = []
    for k, e in enumerate(axes):
        step = e[1] - e[0]
        i = int(round((z[k] - e[0]) / step))
        if i < 0 or i >= mask.shape[k]:
            return False
        idx.append(i)
    return bool(mask[tuple(idx)])


def basin_map(field: Callable, limits: Sequence[tuple], fixed_points: Sequence[FixedPoint], *,
              grid_n: int = 60, dt: float = 1 / 6, n_steps: int = 600,
              capture_dist: float = 0.2):
    """Which attractor each starting point flows to (deterministic integration).

    Returns ``(labels (grid_shape) int, axes)`` with ``-1`` for points that reach
    no attractor within ``n_steps`` (i.e. still travelling, or escaped)."""
    attractors = [p for p in fixed_points if p.kind == "stable"]
    pts, shape, axes = make_grid(limits, grid_n)
    if not attractors:
        return np.full(shape, -1, dtype=int), axes
    traj_end = simulate_field(field, pts, n_steps, dt, sigma=None)[:, -1]
    A = np.stack([p.z for p in attractors])
    dist = np.linalg.norm(traj_end[:, None, :] - A[None, :, :], axis=-1)
    lab = np.argmin(dist, axis=1)
    lab[dist.min(axis=1) > capture_dist] = -1
    return lab.reshape(shape), axes


def divergence_curl(field: Callable, limits: Sequence[tuple], *, grid_n: int = 40,
                    eps: float = 1e-3):
    """Divergence (contraction/expansion) and, in 2-D, the scalar curl (rotation)."""
    pts, shape, axes = make_grid(limits, grid_n)
    d = pts.shape[1]
    div = np.zeros(pts.shape[0])
    curl = np.zeros(pts.shape[0]) if d == 2 else None
    for k in range(d):
        dz = np.zeros(d)
        dz[k] = eps
        vp, vm = field(pts + dz), field(pts - dz)
        div += (vp[:, k] - vm[:, k]) / (2 * eps)
        if d == 2:
            other = 1 - k
            sign = 1.0 if k == 0 else -1.0
            curl += sign * (vp[:, other] - vm[:, other]) / (2 * eps)
    return (div.reshape(shape),
            None if curl is None else curl.reshape(shape), axes)


def slow_region(field: Callable, limits: Sequence[tuple], *, grid_n: int = 60,
                speed_quantile: float = 0.05, zone_mask: Optional[np.ndarray] = None,
                points: Optional[np.ndarray] = None):
    """Geometry of the slowest part of the field -- the "slow manifold" test.

    Returns area fraction, elongation (PCA aspect ratio of the slow set; high =
    line-attractor-like, ~1 = point-attractor-like) and the set's centroid.

    Pass ``points`` (observed latent states) to evaluate without a grid, which is
    necessary above ~3 latent dimensions.

    Careful: ``elongation`` means different things in the two modes and they are
    not comparable. From a grid it is the shape of the slow set *in space* (5.8
    for a synthetic bistable system); from observed points it is the shape of the
    slow set *among visited states* (38.4 for the same system, because the states
    themselves lie along a line). Only ever compare like with like."""
    if points is not None:
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        shape, axes = (pts.shape[0],), None
    else:
        pts, shape, axes = make_grid(limits, grid_n)
    sp = speed(field(pts))
    keep = np.ones(sp.shape, dtype=bool)
    if zone_mask is not None and points is None:
        keep = zone_mask.ravel()
    if keep.sum() < 10:
        return dict(area_frac=np.nan, elongation=np.nan, centroid=np.full(pts.shape[1], np.nan),
                    speed_threshold=np.nan)
    thr = np.quantile(sp[keep], speed_quantile)
    sel = keep & (sp <= thr)
    P = pts[sel]
    if P.shape[0] < 3:
        return dict(area_frac=float(sel.sum() / max(keep.sum(), 1)), elongation=np.nan,
                    centroid=P.mean(0) if P.size else np.full(pts.shape[1], np.nan),
                    speed_threshold=float(thr))
    C = np.cov((P - P.mean(0)).T)
    w = np.linalg.eigvalsh(np.atleast_2d(C))[::-1]
    elong = float(np.sqrt(max(w[0], 1e-12) / max(w[-1], 1e-12)))
    return dict(area_frac=float(sel.sum() / max(keep.sum(), 1)), elongation=elong,
                centroid=P.mean(0), speed_threshold=float(thr))


# ---------------------------------------------------------------------------
# comparing two fields
# ---------------------------------------------------------------------------
def field_distance(field_a: Callable, field_b: Callable,
                   limits: Optional[Sequence[tuple]] = None, *,
                   grid_n: int = 40, mask: Optional[np.ndarray] = None,
                   weights: Optional[np.ndarray] = None,
                   points: Optional[np.ndarray] = None) -> dict:
    """How different are two flow fields over a common region?

    ``cosine`` -- magnitude-weighted mean cosine of the angle between the two
    fields (1 = same direction everywhere). ``nrmse`` -- RMS vector difference
    over RMS mean magnitude (0 = identical). ``log_speed_ratio`` -- mean
    log-ratio of speeds. Always compare against the *split-half* distance of a
    single partition: that is the noise floor.

    Pass ``points`` (observed latent states) instead of ``limits`` to compare the
    fields exactly where the data lives -- required above ~3 latent dimensions,
    and preferable even in 2-D because it weights the comparison by occupancy."""
    if points is not None:
        pts = np.atleast_2d(np.asarray(points, dtype=float))
    else:
        pts, _, _ = make_grid(limits, grid_n)
    va, vb = field_a(pts), field_b(pts)
    sel = (np.ones(pts.shape[0], dtype=bool)
           if (mask is None or points is not None) else mask.ravel().astype(bool))
    if sel.sum() < 5:
        return dict(cosine=np.nan, nrmse=np.nan, log_speed_ratio=np.nan, n_cells=int(sel.sum()))
    va, vb = va[sel], vb[sel]
    w = np.ones(va.shape[0]) if weights is None else weights.ravel()[sel]
    sa, sb = speed(va), speed(vb)
    good = (sa > 1e-12) & (sb > 1e-12)
    cos = np.sum(va * vb, axis=1) / np.maximum(sa * sb, 1e-12)
    wm = w * (sa + sb) / 2
    cosine = float(np.sum(cos[good] * wm[good]) / max(np.sum(wm[good]), 1e-12))
    nrmse = float(np.sqrt(np.average(np.sum((va - vb) ** 2, axis=1), weights=w)) /
                  max(np.sqrt(np.average(np.sum(((va + vb) / 2) ** 2, axis=1), weights=w)), 1e-12))
    lsr = float(np.average(np.log(np.maximum(sa[good], 1e-12) / np.maximum(sb[good], 1e-12)),
                           weights=w[good])) if good.any() else np.nan
    return dict(cosine=cosine, nrmse=nrmse, log_speed_ratio=lsr, n_cells=int(sel.sum()))


def ensemble_field(fields: Sequence[Callable]) -> Callable:
    """Average several fields (e.g. one per random seed) into one.

    Worth doing by default at the data volumes available here. The validation
    sweep showed the fitted field is high-variance in the small-data regime: with
    ~400-1700 s of data the attractor count from a single fit flipped between 1
    and 2 across seeds (correct answer 2), and only at ~6700 s did all seeds
    agree. Averaging fields reduces that variance, and the spread across the
    ensemble (:func:`field_spread`) is the honest uncertainty to report next to
    any attractor claim."""
    fields = list(fields)

    def f(Z):
        acc = None
        for fn in fields:
            v = np.atleast_2d(fn(Z))
            acc = v if acc is None else acc + v
        return acc / max(len(fields), 1)
    return f


def field_spread(fields: Sequence[Callable], limits: Optional[Sequence[tuple]] = None, *,
                 grid_n: int = 25, points: Optional[np.ndarray] = None) -> dict:
    """Disagreement among several fields -- the noise floor for field comparisons.

    Any claim that two conditions have different dynamics must clear this: it is
    the difference you get from refitting the *same* data with a different seed
    (or from a split half)."""
    fields = list(fields)
    if len(fields) < 2:
        return dict(mean_pairwise_cosine=np.nan, mean_pairwise_nrmse=np.nan, n_pairs=0)
    cos, nrm = [], []
    for i in range(len(fields)):
        for j in range(i + 1, len(fields)):
            d = field_distance(fields[i], fields[j], limits, grid_n=grid_n, points=points)
            cos.append(d["cosine"])
            nrm.append(d["nrmse"])
    return dict(mean_pairwise_cosine=float(np.nanmean(cos)),
                sd_pairwise_cosine=float(np.nanstd(cos)),
                mean_pairwise_nrmse=float(np.nanmean(nrm)),
                sd_pairwise_nrmse=float(np.nanstd(nrm)),
                n_pairs=len(cos))


def field_agreement_map(fields: Sequence[Callable], points: np.ndarray) -> dict:
    """Per-location agreement among several fields (e.g. one per seed).

    The spatial version of :func:`field_spread`. Use it to *show* which parts of a
    fitted field are trustworthy: agreement is not uniform, and a field can be
    well determined where the data is dense while being arbitrary elsewhere.
    Returns per-point mean pairwise cosine and the mean speed, so a figure can
    mask or shade the unreliable regions rather than presenting the whole field
    as equally solid."""
    fields = list(fields)
    P = np.atleast_2d(np.asarray(points, dtype=float))
    Vs = [np.atleast_2d(f(P)) for f in fields]
    sp = np.stack([speed(v) for v in Vs])
    n = len(fields)
    if n < 2:
        return dict(cosine=np.full(P.shape[0], np.nan), speed=sp.mean(0))
    cos_acc = np.zeros(P.shape[0])
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            num = np.sum(Vs[i] * Vs[j], axis=1)
            den = np.maximum(speed(Vs[i]) * speed(Vs[j]), 1e-12)
            cos_acc += num / den
            pairs += 1
    return dict(cosine=cos_acc / max(pairs, 1), speed=sp.mean(0),
                speed_cv=sp.std(0) / np.maximum(sp.mean(0), 1e-12))


def white_noise_fraction(X: np.ndarray, lags: Sequence[int] = (1, 2, 3, 4)) -> np.ndarray:
    """Per-row share of variance that is temporally white (observation noise).

    Extrapolates the autocorrelation at ``lags`` back to lag 0; the shortfall
    from 1 is the white component. Use it to convert a reconstruction R2 into a
    *fraction of explainable variance*: on this dataset ~80 % of each neuron's
    variance at 30 Hz is white, so a per-neuron R2 of 0.16 is most of what is
    achievable, not a poor fit."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    Xc = X - X.mean(axis=1, keepdims=True)
    sd = Xc.std(axis=1)
    sd[sd < 1e-12] = 1.0
    Xn = Xc / sd[:, None]
    T = Xn.shape[1]
    acs = np.stack([(Xn[:, :-L] * Xn[:, L:]).sum(axis=1) / (T - L) for L in lags], axis=1)
    L = np.asarray(lags, dtype=float)
    A = np.vstack([L, np.ones_like(L)]).T
    coef, *_ = np.linalg.lstsq(A, acs.T, rcond=None)
    a0 = coef[1]                                     # intercept = extrapolated AC(0)
    return np.clip(1.0 - np.clip(a0, 0.0, 1.0), 0.0, 1.0)


def field_difference_map(field_a: Callable, field_b: Callable, limits: Sequence[tuple], *,
                         grid_n: int = 40):
    """Per-cell vector difference ``a - b`` and its magnitude, for Δ-field plots."""
    pts, shape, axes = make_grid(limits, grid_n)
    dv = field_a(pts) - field_b(pts)
    return dv.reshape(shape + (pts.shape[1],)), speed(dv).reshape(shape), axes


# ---------------------------------------------------------------------------
# naive (biased) field estimator -- a control, not a method
# ---------------------------------------------------------------------------
def pca_latents(segset: SegmentSet, d: int, *, idx: Optional[np.ndarray] = None):
    """Per-segment PCA latents of the *observed* activity: list of ``(T_i, d)``.

    This is the ordinary way people obtain "the low-dimensional trajectory", and
    it is the correct input to :func:`naive_flow_field` when demonstrating the
    errors-in-variables failure: unlike the model's encoder, PCA does no temporal
    denoising, so its latents carry the full white observation-noise component
    (measured at 10-19 % of latent variance in this dataset).

    Returns ``(latents, basis, mean)``."""
    idx = np.arange(len(segset)) if idx is None else np.asarray(idx, dtype=int)
    rows = [segset.Y[i, segset.mask[i] > 0] for i in idx]
    X = np.concatenate(rows, axis=0)
    mu = X.mean(0)
    Xc = X - mu
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    basis = Vt[:d]
    return [(r - mu) @ basis.T for r in rows], basis, mu


def naive_flow_field(Z_segments: Sequence[np.ndarray], dt: float, *, kind: str = "mlp",
                     hidden: int = 64, ridge: float = 1e-3, epochs: int = 400,
                     seed: int = 0, device=None) -> Callable:
    """Fit ``dz/dt ~ f(z)`` pointwise from observed latents (the standard shortcut).

    Provided so the notebook can *demonstrate the failure mode*: because ``z`` is
    measured with noise, this estimator is an errors-in-variables regression,
    biased toward contraction, and readily invents stable fixed points that the
    generating system does not have. Measured on this dataset, increments have
    lag-1 autocorrelation ~-0.3 (pure white noise gives -0.5), so the bias is
    large here."""
    X = np.concatenate([z[:-1] for z in Z_segments if len(z) > 1], axis=0)
    Yd = np.concatenate([np.diff(z, axis=0) / dt for z in Z_segments if len(z) > 1], axis=0)
    if kind == "ridge":
        Xa = np.hstack([X, np.ones((X.shape[0], 1))])
        A = np.linalg.solve(Xa.T @ Xa + ridge * np.eye(Xa.shape[1]), Xa.T @ Yd)
        return lambda Z: np.hstack([np.atleast_2d(Z),
                                    np.ones((np.atleast_2d(Z).shape[0], 1))]) @ A
    device = device or get_device()
    set_seed(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    xt = torch.tensor((X - mu) / sd, dtype=torch.float32, device=device)
    yt = torch.tensor(Yd, dtype=torch.float32, device=device)
    net = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.GELU(),
                        nn.Linear(hidden, hidden), nn.GELU(),
                        nn.Linear(hidden, X.shape[1])).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = ((net(xt) - yt) ** 2).mean()
        loss.backward()
        opt.step()
    net.eval()

    @torch.no_grad()
    def f(Z):
        z = torch.tensor((np.atleast_2d(Z) - mu) / sd, dtype=torch.float32, device=device)
        return net(z).cpu().numpy().astype(np.float64)
    return f


# ---------------------------------------------------------------------------
# synthetic ground-truth systems (validation)
# ---------------------------------------------------------------------------
def sys_point_attractor(tau: float = 1.0, center=(0.0, 0.0)) -> Callable:
    """Single stable fixed point."""
    c = np.asarray(center, dtype=float)

    def f(Z):
        Z = np.atleast_2d(Z)
        return (c - Z) / tau
    return f


def sys_line_attractor(tau: float = 0.5, axis: int = 0) -> Callable:
    """Continuous attractor: zero flow along ``axis``, decay in the others."""
    def f(Z):
        Z = np.atleast_2d(Z)
        V = -Z / tau
        V[:, axis] = 0.0
        return V
    return f


def sys_nonnormal_line(theta_deg: float = 60.0, tau: float = 0.5) -> Callable:
    """Line attractor whose input direction is not aligned to it (their Fig. 1h).

    ``M = Q diag(0, -1) Q^{-1}`` with ``Q = [[1, sin t], [0, cos t]]`` -- exactly the
    cFINDR parameterisation, so a fitted field can be checked against it."""
    t = math.radians(theta_deg)
    Q = np.array([[1.0, math.sin(t)], [0.0, math.cos(t)]])
    M = Q @ np.diag([0.0, -1.0 / tau]) @ np.linalg.inv(Q)

    def f(Z):
        return np.atleast_2d(Z) @ M.T
    return f


def sys_bistable(sep: float = 1.0, tau: float = 0.5, cross_tau: float = 0.8) -> Callable:
    """Two point attractors at ``(+-sep, 0)`` with a saddle between them.

    ``cross_tau`` (off-axis decay) is deliberately not much faster than ``tau``:
    with a very fast off-axis decay the simulated trajectories collapse onto the
    ``z2 = 0`` line within a couple of steps, and then *no* method can recover the
    off-axis field because there is no data there."""
    def f(Z):
        Z = np.atleast_2d(Z)
        V = np.empty_like(Z)
        V[:, 0] = (Z[:, 0] * (sep ** 2 - Z[:, 0] ** 2)) / (tau * sep ** 2)
        V[:, 1:] = -Z[:, 1:] / cross_tau
        return V
    return f


def sys_ddm_line_attractor(bound: float = 1.0, tau: float = 0.3) -> Callable:
    """Their Fig. 1g: flat (line-attractor) flow inside ``|z1| < bound``, pulled to
    point attractors at the ends outside it."""
    def f(Z):
        Z = np.atleast_2d(Z)
        V = np.zeros_like(Z)
        over = np.abs(Z[:, 0]) - bound
        V[:, 0] = -np.sign(Z[:, 0]) * np.maximum(over, 0.0) / tau
        V[:, 1:] = -Z[:, 1:] / tau
        return V
    return f


def sys_two_attractors(angle_deg: float = 70.0, radius: float = 1.0,
                       tau: float = 0.4, cross_tau: float = 0.3) -> Callable:
    """Two attractors separated by an *angle*, not lying on a common line.

    The shape the two-solution (intuitive vs new decoder axis) hypothesis
    predicts, and the case a 1-D projection cannot distinguish from one blob."""
    a = math.radians(angle_deg) / 2
    c1 = radius * np.array([math.cos(a), math.sin(a)])
    c2 = radius * np.array([math.cos(a), -math.sin(a)])

    def f(Z):
        Z = np.atleast_2d(Z)
        d1 = np.linalg.norm(Z - c1, axis=1, keepdims=True)
        d2 = np.linalg.norm(Z - c2, axis=1, keepdims=True)
        w1 = np.exp(-d1 ** 2 / (2 * cross_tau ** 2))
        w2 = np.exp(-d2 ** 2 / (2 * cross_tau ** 2))
        tot = w1 + w2 + 1e-9
        target = (w1 * c1 + w2 * c2) / tot
        return (target - Z) / tau
    return f


def sys_unstable_origin(tau: float = 0.5, rot: float = 0.5,
                        radius: float = 1.0) -> Callable:
    """Unstable spiral at the origin with a stable limit cycle at ``radius``.

    FINDR reports an *unstable* origin, but a purely expanding linear field is
    not a usable test case: simulated trajectories diverge exponentially and pile
    up against the integrator's clip, so the "ground truth" data is garbage (a
    first attempt with a pure expanding spiral scored cosine 0.21 for the model
    and 0.12 for the naive estimator -- both meaningless). Bounding the expansion
    with a limit cycle keeps the origin unstable while leaving the trajectories
    in a finite region, and additionally tests whether a *cyclic* attractor is
    correctly reported as having no stable fixed point."""
    def f(Z):
        Z = np.atleast_2d(Z)
        r = np.linalg.norm(Z, axis=1, keepdims=True)
        radial = (radius ** 2 - r ** 2) * Z / (tau * radius ** 2)
        rotational = rot * np.stack([-Z[:, 1], Z[:, 0]], axis=1)
        return radial + rotational
    return f


SYNTHETIC_SYSTEMS = {
    "point_attractor": sys_point_attractor,
    "line_attractor": sys_line_attractor,
    "nonnormal_line": sys_nonnormal_line,
    "bistable": sys_bistable,
    "ddm_line_attractor": sys_ddm_line_attractor,
    "two_attractors": sys_two_attractors,
    "unstable_origin": sys_unstable_origin,
}


def synthesize_dataset(system: Callable, *, d: int = 2, n_segments: int = 200,
                       n_neurons: int = 200,
                       dt: float = 1 / 6, seg_bins=(20, 90), sigma_z: float = 0.15,
                       obs_noise: float = 0.5, z0_scale: float = 0.4,
                       z0_mode: str = "gauss", baseline: float = 1.0, seed: int = 0,
                       W: Optional[np.ndarray] = None):
    """Generate a :class:`SegmentSet` from a known field, matched to this dataset.

    ``obs_noise`` is the observation SD *relative to the SD of the noiseless
    readout across time*, so it maps directly onto the measured signal-to-noise
    of the real recordings. Variable segment lengths are drawn uniformly from
    ``seg_bins`` on purpose: variable trial duration must be a non-issue for the
    method, and this is where that gets verified.

    Returns ``(segset, truth)`` where ``truth`` holds the latent paths, ``W`` and
    the generating field."""
    rng = np.random.default_rng(seed)
    if W is not None:
        d = W.shape[1]
    lens = rng.integers(seg_bins[0], seg_bins[1] + 1, size=n_segments)
    if z0_mode == "uniform":
        # spread starting states over a box: probes the field the way the paper's
        # 5,000 simulated trajectories do, instead of only along one axis
        z0 = z0_scale * (2 * rng.random((n_segments, d)) - 1)
    else:
        z0 = z0_scale * rng.standard_normal((n_segments, d))
    paths = []
    for i in range(n_segments):
        p = simulate_field(system, z0[i:i + 1], int(lens[i]) - 1, dt,
                           sigma=np.full(d, sigma_z), rng=rng)[0]
        paths.append(p)
    if W is None:
        W = rng.standard_normal((n_neurons, d)) / math.sqrt(d)
    allz = np.concatenate(paths, axis=0)
    clean_sd = float(np.std(allz @ W.T)) + 1e-9
    noise_sd = obs_noise * clean_sd

    Tmax = int(lens.max())
    Y = np.zeros((n_segments, Tmax, n_neurons), dtype=np.float32)
    mask = np.zeros((n_segments, Tmax), dtype=np.float32)
    for i, p in enumerate(paths):
        mu = np.log1p(np.exp(p @ W.T + baseline))          # softplus readout
        Y[i, :len(p)] = mu + noise_sd * rng.standard_normal(mu.shape)
        mask[i, :len(p)] = 1.0
    np.maximum(Y, 0.0, out=Y)
    meta = [dict(condition="synthetic", n_bins=int(l), start=0, end=int(l)) for l in lens]
    segset = SegmentSet(Y=Y, mask=mask,
                        labels={"condition": np.zeros(n_segments, dtype=np.int64)},
                        meta=meta)
    truth = dict(paths=paths, W=W, field=system, sigma_z=sigma_z, dt=dt,
                 noise_sd=noise_sd, clean_sd=clean_sd)
    return segset, truth


def fit_linear_alignment(Z_fit: np.ndarray, Z_true: np.ndarray) -> LatentTransform:
    """Least-squares *general linear* map taking fitted latents onto true latents.

    Beware: when the latents barely explore one direction this map is
    ill-conditioned and inflates velocities along the degenerate axis (condition
    numbers in the hundreds were observed on synthetic data whose second latent
    dimension had sd 0.09). Prefer :func:`fit_orthogonal_alignment` for scoring."""
    A_ls, *_ = np.linalg.lstsq(np.atleast_2d(Z_fit), np.atleast_2d(Z_true), rcond=None)
    return LatentTransform(A_ls.T)


def fit_orthogonal_alignment(Z_fit: np.ndarray, Z_true: np.ndarray, *,
                             allow_scale: bool = True,
                             center: bool = True) -> LatentTransform:
    """Procrustes alignment (rotation/reflection + optional global scale).

    This is the *correct* freedom to allow when scoring a fitted field against a
    known one: after :func:`canonical_transform` the readout is an isometry, so
    the latent frame is identified only up to an orthogonal map. Unlike a general
    linear fit this cannot manufacture or destroy structure along a
    weakly-explored direction."""
    A = np.atleast_2d(np.asarray(Z_fit, dtype=np.float64))
    B = np.atleast_2d(np.asarray(Z_true, dtype=np.float64))
    mu_a = A.mean(0) if center else np.zeros(A.shape[1])
    mu_b = B.mean(0) if center else np.zeros(B.shape[1])
    A0, B0 = A - mu_a, B - mu_b
    U, S, Vt = np.linalg.svd(A0.T @ B0)
    R = (U @ Vt).T                                  # maps fitted -> true
    s = (S.sum() / max((A0 ** 2).sum(), 1e-12)) if allow_scale else 1.0
    M = s * R
    return LatentTransform(M, shift=mu_b - M @ mu_a)


def occupancy_anisotropy(Z: np.ndarray) -> dict:
    """How evenly do the latent states explore each direction?

    **Check this before interpreting field structure.** Where the data does not
    go, the field is extrapolation, and a fitted network will invent structure
    there. Measured on synthetic non-normal line-attractor data, the trajectories
    collapsed onto a line within a couple of time constants and the off-line
    direction was then unrecoverable (canonical correlation 0.09 for the model
    and 0.03 for PCA) -- the field along the unexplored direction simply is not
    identifiable, by any method.

    ``sd_ratio`` near 1 means all directions are explored; a small value means
    the last direction is nearly unused."""
    Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
    sd = np.sqrt(np.linalg.eigvalsh(np.cov((Z - Z.mean(0)).T))[::-1])
    sd = np.maximum(sd, 0.0)
    return dict(sd=sd, sd_ratio=float(sd[-1] / max(sd[0], 1e-12)),
                participation=float(sd.sum() ** 2 / max((sd ** 2).sum(), 1e-12)))


def latent_subspace_corr(Z_a: np.ndarray, Z_b: np.ndarray) -> np.ndarray:
    """Canonical correlations between two sets of latent trajectories.

    Scale- and rotation-invariant, so it answers "does this latent span the same
    subspace as that one?" -- the first thing to check about any fitted latent
    (a value near 0 on the last dimension means that dimension was never really
    used)."""
    A = np.atleast_2d(np.asarray(Z_a, dtype=np.float64))
    B = np.atleast_2d(np.asarray(Z_b, dtype=np.float64))
    Qa, _ = np.linalg.qr(A - A.mean(0))
    Qb, _ = np.linalg.qr(B - B.mean(0))
    return np.linalg.svd(Qa.T @ Qb, compute_uv=False)


def occupancy_mask(Z: np.ndarray, limits: Sequence[tuple], grid_n: int = 40, *,
                   min_count: int = 5, dilate: int = 1) -> np.ndarray:
    """Grid cells actually visited by a set of latent states.

    Every field comparison must be restricted to a region the data visits:
    outside it the field is pure extrapolation, and a fitted network will happily
    place fixed points there. This is the data-driven counterpart of the paper's
    simulation-based :func:`sample_zone`."""
    _, shape, axes = make_grid(limits, grid_n)
    idx, ok = [], np.ones(np.atleast_2d(Z).shape[0], dtype=bool)
    Z = np.atleast_2d(Z)
    for k, e in enumerate(axes):
        step = e[1] - e[0]
        i = np.floor((Z[:, k] - e[0]) / step + 0.5).astype(int)
        ok &= (i >= 0) & (i < shape[k])
        idx.append(np.clip(i, 0, shape[k] - 1))
    counts = np.zeros(shape, dtype=np.int64)
    np.add.at(counts, tuple(i[ok] for i in idx), 1)
    m = counts >= min_count
    if dilate:
        from scipy import ndimage
        m = ndimage.binary_dilation(m, iterations=int(dilate))
    return m


# ---------------------------------------------------------------------------
# caching
# ---------------------------------------------------------------------------
def save_fit(path, res: FitResult) -> None:
    """Persist a fitted model (state dict + configs, not the data).

    **The caller owns the cache key, and it must include the preprocessing.** Only
    the model and training configs are stored, so a cached fit says nothing about
    which data it saw. This bit once: changing ``PreprocConfig.scale_mode``
    silently reused models fit on differently scaled activity, and the reported
    losses and field statistics were meaningless until the caches were discarded.
    Put ``PreprocConfig.tag`` in the filename."""
    import pickle
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.cpu() for k, v in res.model.state_dict().items()},
                "mcfg": res.config, "tcfg": res.train_config,
                "history": res.history, "train_idx": res.train_idx, "val_idx": res.val_idx,
                "best_epoch": res.best_epoch, "best_val": res.best_val}, p)


def load_fit(path, *, device=None) -> FitResult:
    """Reload a fit saved by :func:`save_fit`."""
    device = device or get_device()
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = FlowVAE(blob["mcfg"]).to(device)
    model.load_state_dict({k: v.to(device) for k, v in blob["state_dict"].items()})
    model.eval()
    return FitResult(model=model, history=blob["history"], train_idx=blob["train_idx"],
                     val_idx=blob["val_idx"], best_epoch=blob["best_epoch"],
                     best_val=blob["best_val"], config=blob["mcfg"],
                     train_config=blob["tcfg"])
