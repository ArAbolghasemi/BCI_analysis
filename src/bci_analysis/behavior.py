from __future__ import annotations

import datetime as _datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
from scipy.ndimage import gaussian_filter1d


EXPERIMENT_LOG_COLUMNS = [
    "date",
    "laser_power",
    "obj",
    "goniometer_x_set",
    "goniometer_y_set",
    "goniometer_x_mesured",
    "goniometer_y_mesured",
    "mouse_orginal_weight",
    "mouse_weight_per_day",
    "mouse_weight_persantage",
    "meanlm intensity",
    "num_of_rewards",
    "Decoder factor",
    "Trial type",
    "block type",
    "rig temperature",
    "note",
    "Laser power percent used",
    "incl. sensor mode",
]


@dataclass
class RewardMetrics:
    day_list: list[str]
    all_reward_rate: np.ndarray
    active_reward_rate: np.ndarray
    all_reward_times_min_list: list[np.ndarray]
    active_reward_times_min_list: list[np.ndarray]
    exp_length_min: np.ndarray
    reward_rate_timecourses: list[np.ndarray]
    reward_rate_time_min: np.ndarray


@dataclass
class TrialMetrics:
    num_hits: np.ndarray
    num_misses: np.ndarray
    hit_rate: np.ndarray
    remapping_or_not: np.ndarray
    remapping_start_frame: np.ndarray
    trials_outcome_list: list[np.ndarray]
    hit_trials_list: list[np.ndarray]
    miss_trials_list: list[np.ndarray]
    trial_length_frames_list: list[np.ndarray]
    trial_start_frames_list: list[np.ndarray]
    trial_end_frames_list: list[np.ndarray]
    trial_factor_used_list: list[np.ndarray]


@dataclass
class PhaseInfo:
    day_zero: str
    day_zero_idx: int
    first_remapping_day: str | None
    first_remapping_day_idx: int | None
    end_remapping_phase: str | None
    end_remapping_phase_idx: int | None
    stable_control_days_idx: list[int]
    remapping_days_idx: list[int]


@dataclass
class BehaviorAnalysis:
    mouse_name: str
    sampling_rate: float
    day_list: list[str]
    day_zero: str
    output_dir: Path
    reward: RewardMetrics
    trial: TrialMetrics
    phase: PhaseInfo
    mouse_weight_percentage: np.ndarray
    smoothed_reward_rate_stack: np.ndarray
    reward_rate_time_min: np.ndarray


def register_richfile_types(rfs: Any) -> None:
    """Register common richfile loaders used by these BMI packets."""
    registered = [prop["type_name"] for prop in rfs.functions._TYPE_LOOKUP]

    if "pandas_dataframe" not in registered:
        rfs.functions.register_type(
            type_name="pandas_dataframe",
            function_load=lambda path: pd.read_csv(path),
            function_save=lambda path, obj: obj.to_csv(path),
            object_class=pd.DataFrame,
            library="pandas",
            suffix="csv",
        )

    if "pandas_series" not in registered:
        rfs.functions.register_type(
            type_name="pandas_series",
            function_load=lambda path: pd.read_csv(path).squeeze("columns"),
            function_save=lambda path, obj: obj.to_csv(path),
            object_class=pd.Series,
            library="pandas",
            suffix="csv",
        )

    if "scipy_sparse" not in registered:
        rfs.functions.register_type(
            type_name="scipy_sparse",
            function_load=lambda path: sparse.load_npz(path),
            function_save=lambda path, obj: sparse.save_npz(path, obj),
            object_class=sparse.csr_matrix,
            library="scipy",
            suffix="npz",
        )

    if "numpy_array" not in registered:
        rfs.functions.register_type(
            type_name="numpy_array",
            function_load=lambda path: np.load(path, allow_pickle=True),
            function_save=lambda path, obj: np.save(path, obj),
            object_class=np.ndarray,
            library="numpy",
            suffix="npy",
        )

    if "scipy_sparse_array" not in registered:
        rfs.functions.register_type(
            type_name="scipy_sparse_array",
            function_load=lambda path: sparse.load_npz(path).toarray(),
            function_save=lambda path, obj: sparse.save_npz(path, sparse.csr_matrix(obj)),
            object_class=np.ndarray,
            library="scipy",
            suffix="npz",
        )


def find_experiment_log_path(data_root: str | Path, project_root: str | Path) -> Path:
    candidates = [
        Path(data_root) / "RH + GH experiment log.xlsx",
        Path(project_root) / "RH + GH experiment log.xlsx",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find RH + GH experiment log.xlsx in data_root or project_root."
    )


def load_experiment_tracking(experiment_log_path: str | Path) -> dict[str, pd.DataFrame]:
    xls = pd.ExcelFile(experiment_log_path)
    tracking = {
        sheet_name: pd.read_excel(experiment_log_path, sheet_name=sheet_name)
        for sheet_name in xls.sheet_names
    }
    return {
        sheet_name: df.dropna(how="all").dropna(axis=1, how="all")
        for sheet_name, df in tracking.items()
    }


def normalize_experiment_metadata(exp_meta: pd.DataFrame) -> pd.DataFrame:
    exp_meta = exp_meta.dropna(how="all").dropna(axis=1, how="all").copy()
    if len(exp_meta.columns) < len(EXPERIMENT_LOG_COLUMNS):
        raise ValueError(
            f"Experiment sheet has {len(exp_meta.columns)} columns, expected at least "
            f"{len(EXPERIMENT_LOG_COLUMNS)}."
        )

    exp_meta = exp_meta.iloc[:, : len(EXPERIMENT_LOG_COLUMNS)].copy()
    exp_meta.columns = EXPERIMENT_LOG_COLUMNS

    exp_meta["date_converted"] = [
        x.strftime("%Y%m%d")
        if isinstance(x, (pd.Timestamp, _datetime.datetime, _datetime.date))
        else None
        for x in exp_meta["date"]
    ]
    exp_meta = exp_meta[
        ["date_converted"] + [col for col in exp_meta.columns if col != "date_converted"]
    ]
    return exp_meta


def get_mouse_weight_percentage(
    experiment_tracking: dict[str, pd.DataFrame],
    mouse_name: str,
    day_list: list[str],
) -> np.ndarray:
    weight_percentage = np.full(len(day_list), np.nan, dtype=float)
    if mouse_name not in experiment_tracking:
        return weight_percentage

    exp_meta = normalize_experiment_metadata(experiment_tracking[mouse_name])
    for day_idx, day_name in enumerate(day_list):
        match = exp_meta.loc[exp_meta["date_converted"] == day_name]
        if len(match) > 0:
            weight_percentage[day_idx] = match["mouse_weight_persantage"].values[0]
    return weight_percentage


def open_mouse_richfile(rfs: Any, data_root: str | Path, mouse_name: str) -> Any:
    richfile_path = Path(data_root) / f"{mouse_name}.richfile"
    if not richfile_path.exists():
        raise FileNotFoundError(richfile_path)
    return rfs.RichFile(str(richfile_path))


def get_day_list(r: Any) -> list[str]:
    return list(r["data_neural_aligned"].keys())


def detect_threshold_crossings(signal: np.ndarray, threshold_fraction: float = 0.5) -> np.ndarray:
    signal = np.asarray(signal).ravel()
    if signal.size == 0 or np.all(np.isnan(signal)):
        return np.array([], dtype=int)

    threshold = threshold_fraction * np.nanmax(signal)
    samples = np.flatnonzero(signal > threshold)
    if samples.size == 0:
        return samples.astype(int)
    return samples[np.insert(np.diff(samples) > 1, 0, True)].astype(int)


def get_reward_times_from_typed(
    reward_times_typed: Any,
    active_reward_key: tuple[int, int, int, int] = (1, 1, 1, 1),
) -> np.ndarray:
    if active_reward_key not in reward_times_typed:
        raise KeyError(f"Missing active reward key {active_reward_key}.")
    return np.asarray(reward_times_typed[active_reward_key], dtype=float).ravel()


def sliding_reward_rate(
    reward_times_sec: np.ndarray,
    session_length_sec: float,
    window_min: float = 2.5,
    step_sec: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    window_sec = float(window_min) * 60
    if window_sec <= 0:
        raise ValueError("window_min must be positive.")

    max_start_sec = max(float(session_length_sec) - window_sec, 0.0)
    starts_sec = np.arange(0, np.floor(max_start_sec) + 1, step_sec)
    if starts_sec.size == 0:
        starts_sec = np.array([0.0])

    rates = np.zeros(len(starts_sec), dtype=float)
    reward_times_sec = np.asarray(reward_times_sec, dtype=float)
    for i, start_sec in enumerate(starts_sec):
        end_sec = start_sec + window_sec
        rates[i] = np.sum(
            (reward_times_sec >= start_sec) & (reward_times_sec < end_sec)
        ) / window_min

    time_min = (starts_sec + window_sec / 2) / 60
    return rates, time_min


def stack_variable_length(arrays: list[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(x, dtype=float).ravel() for x in arrays]
    if len(arrays) == 0:
        return np.empty((0, 0), dtype=float)
    max_len = max(len(x) for x in arrays)
    stacked = np.full((len(arrays), max_len), np.nan, dtype=float)
    for i, x in enumerate(arrays):
        stacked[i, : len(x)] = x
    return stacked


def moving_average_same(x: np.ndarray, window_size: int = 20) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    if window_size <= 1 or len(x) == 0:
        return x.copy()
    kernel = np.ones(int(window_size), dtype=float) / int(window_size)
    return np.convolve(x, kernel, mode="same")


def compute_reward_metrics(
    r: Any,
    day_list: list[str],
    sampling_rate: float = 30,
    sliding_window_min: float = 2.5,
    active_reward_key: tuple[int, int, int, int] = (1, 1, 1, 1),
) -> RewardMetrics:
    all_reward_rate = np.full(len(day_list), np.nan, dtype=float)
    active_reward_rate = np.full(len(day_list), np.nan, dtype=float)
    all_reward_times_min_list: list[np.ndarray] = []
    active_reward_times_min_list: list[np.ndarray] = []
    exp_length_min = np.full(len(day_list), np.nan, dtype=float)
    reward_rate_timecourses: list[np.ndarray] = []
    reward_rate_time_min_list: list[np.ndarray] = []

    for day_idx, day_name in enumerate(day_list):
        reward_signal = np.asarray(r["data_ws"][day_name]["rewards"].load()).ravel()
        reward_times_typed = r["reward_times_typed"][day_name].load()

        session_length_sec = len(reward_signal) / sampling_rate
        exp_length_min[day_idx] = session_length_sec / 60

        all_reward_samples = detect_threshold_crossings(reward_signal)
        all_reward_times_min = all_reward_samples / sampling_rate / 60
        all_reward_times_min_list.append(all_reward_times_min)

        active_reward_samples = get_reward_times_from_typed(
            reward_times_typed,
            active_reward_key=active_reward_key,
        )
        active_reward_times_sec = active_reward_samples / sampling_rate
        active_reward_times_min = active_reward_times_sec / 60
        active_reward_times_min_list.append(active_reward_times_min)

        if exp_length_min[day_idx] > 0:
            all_reward_rate[day_idx] = len(all_reward_samples) / exp_length_min[day_idx]
            active_reward_rate[day_idx] = len(active_reward_samples) / exp_length_min[day_idx]

        rate, time_min = sliding_reward_rate(
            active_reward_times_sec,
            session_length_sec=session_length_sec,
            window_min=sliding_window_min,
            step_sec=1.0,
        )
        reward_rate_timecourses.append(rate)
        reward_rate_time_min_list.append(time_min)

    longest_time = (
        reward_rate_time_min_list[
            int(np.argmax([len(x) for x in reward_rate_time_min_list]))
        ]
        if reward_rate_time_min_list
        else np.array([], dtype=float)
    )

    return RewardMetrics(
        day_list=day_list,
        all_reward_rate=all_reward_rate,
        active_reward_rate=active_reward_rate,
        all_reward_times_min_list=all_reward_times_min_list,
        active_reward_times_min_list=active_reward_times_min_list,
        exp_length_min=exp_length_min,
        reward_rate_timecourses=reward_rate_timecourses,
        reward_rate_time_min=longest_time,
    )


def true_trial_mask(trials_df: pd.DataFrame) -> pd.Series:
    return (
        (trials_df["trialType_feedbackLinked"] == 1)
        & (trials_df["trialType_cursorOn"] == 1)
        & (trials_df["trialType_rewardOn"] == 1)
    )


def compute_trial_metrics(
    r: Any,
    day_list: list[str],
) -> TrialMetrics:
    num_hits = np.full(len(day_list), np.nan, dtype=float)
    num_misses = np.full(len(day_list), np.nan, dtype=float)
    hit_rate = np.full(len(day_list), np.nan, dtype=float)
    remapping_or_not = np.zeros(len(day_list), dtype=bool)
    remapping_start_frame = np.full(len(day_list), np.nan, dtype=float)
    trials_outcome_list: list[np.ndarray] = []
    hit_trials_list: list[np.ndarray] = []
    miss_trials_list: list[np.ndarray] = []
    trial_length_frames_list: list[np.ndarray] = []
    trial_start_frames_list: list[np.ndarray] = []
    trial_end_frames_list: list[np.ndarray] = []
    trial_factor_used_list: list[np.ndarray] = []

    for day_idx, day_name in enumerate(day_list):
        trials_df = r["extras"]["logger_dfs"][day_name]["trials"].load()
        mask = true_trial_mask(trials_df)

        if np.sum(mask) == 0:
            trials_outcome_list.append(np.array([]))
            hit_trials_list.append(np.array([]))
            miss_trials_list.append(np.array([]))
            trial_length_frames_list.append(np.array([]))
            trial_start_frames_list.append(np.array([]))
            trial_end_frames_list.append(np.array([]))
            trial_factor_used_list.append(np.array([]))
            continue

        filtered = trials_df.loc[mask]
        outcome = filtered["success_outcome"].to_numpy()
        start_frames = filtered["frameNum_trialStart"].to_numpy()
        end_frames = filtered["frameNum_trialEnd"].to_numpy()
        factors = filtered["factor_to_use"].to_numpy()

        hit_idx = np.flatnonzero(outcome == 1)
        miss_idx = np.flatnonzero(outcome == 0)
        num_hits[day_idx] = len(hit_idx)
        num_misses[day_idx] = len(miss_idx)
        total_trials = num_hits[day_idx] + num_misses[day_idx]
        if total_trials > 0:
            hit_rate[day_idx] = num_hits[day_idx] / total_trials

        trials_outcome_list.append(outcome)
        hit_trials_list.append(hit_idx)
        miss_trials_list.append(miss_idx)
        trial_length_frames_list.append(end_frames - start_frames)
        trial_start_frames_list.append(start_frames)
        trial_end_frames_list.append(end_frames)
        trial_factor_used_list.append(factors)

        if len(factors) > 0 and len(np.unique(factors)) > 1:
            remapping_or_not[day_idx] = True
            base_factor = factors[0]
            first_new_factor_idx = np.flatnonzero(factors != base_factor)[0]
            if first_new_factor_idx > 0:
                remapping_start_frame[day_idx] = end_frames[first_new_factor_idx - 1]
            else:
                remapping_start_frame[day_idx] = start_frames[first_new_factor_idx]

    return TrialMetrics(
        num_hits=num_hits,
        num_misses=num_misses,
        hit_rate=hit_rate,
        remapping_or_not=remapping_or_not,
        remapping_start_frame=remapping_start_frame,
        trials_outcome_list=trials_outcome_list,
        hit_trials_list=hit_trials_list,
        miss_trials_list=miss_trials_list,
        trial_length_frames_list=trial_length_frames_list,
        trial_start_frames_list=trial_start_frames_list,
        trial_end_frames_list=trial_end_frames_list,
        trial_factor_used_list=trial_factor_used_list,
    )


def derive_phase_info(
    day_list: list[str],
    day_zero: str,
    remapping_or_not: np.ndarray,
) -> PhaseInfo:
    day_zero_idx = day_list.index(day_zero)
    remapping_idx = np.flatnonzero(remapping_or_not)

    if remapping_idx.size > 0:
        first_idx = int(remapping_idx[0])
        end_idx = int(remapping_idx[-1])
        first_day = day_list[first_idx]
        end_day = day_list[end_idx]
        stable_idx = list(range(day_zero_idx, first_idx))
        remap_idx = list(range(first_idx, end_idx + 1))
    else:
        first_idx = None
        end_idx = None
        first_day = None
        end_day = None
        stable_idx = list(range(day_zero_idx, len(day_list)))
        remap_idx = []

    return PhaseInfo(
        day_zero=day_zero,
        day_zero_idx=day_zero_idx,
        first_remapping_day=first_day,
        first_remapping_day_idx=first_idx,
        end_remapping_phase=end_day,
        end_remapping_phase_idx=end_idx,
        stable_control_days_idx=stable_idx,
        remapping_days_idx=remap_idx,
    )


def make_smoothed_reward_rate_stack(
    reward_rate_timecourses: list[np.ndarray],
    smoothing_window_size: int = 20,
) -> np.ndarray:
    smoothed = [
        moving_average_same(x, window_size=smoothing_window_size)
        for x in reward_rate_timecourses
    ]
    return stack_variable_length(smoothed)


def build_behavior_analysis(
    r: Any,
    mouse_name: str,
    output_root: str | Path,
    experiment_tracking: dict[str, pd.DataFrame] | None = None,
    sampling_rate: float = 30,
    sliding_window_min: float = 2.5,
    smoothing_window_size: int = 20,
    active_reward_key: tuple[int, int, int, int] = (1, 1, 1, 1),
) -> BehaviorAnalysis:
    day_list = get_day_list(r)
    day_zero = r["date_day0"].load()
    output_dir = Path(output_root) / mouse_name
    output_dir.mkdir(parents=True, exist_ok=True)

    reward = compute_reward_metrics(
        r,
        day_list=day_list,
        sampling_rate=sampling_rate,
        sliding_window_min=sliding_window_min,
        active_reward_key=active_reward_key,
    )
    trial = compute_trial_metrics(r, day_list=day_list)
    phase = derive_phase_info(
        day_list=day_list,
        day_zero=day_zero,
        remapping_or_not=trial.remapping_or_not,
    )
    weight = (
        get_mouse_weight_percentage(experiment_tracking, mouse_name, day_list)
        if experiment_tracking is not None
        else np.full(len(day_list), np.nan, dtype=float)
    )
    smoothed_stack = make_smoothed_reward_rate_stack(
        reward.reward_rate_timecourses,
        smoothing_window_size=smoothing_window_size,
    )

    return BehaviorAnalysis(
        mouse_name=mouse_name,
        sampling_rate=sampling_rate,
        day_list=day_list,
        day_zero=day_zero,
        output_dir=output_dir,
        reward=reward,
        trial=trial,
        phase=phase,
        mouse_weight_percentage=weight,
        smoothed_reward_rate_stack=smoothed_stack,
        reward_rate_time_min=reward.reward_rate_time_min,
    )


def average_success_trial_length_sec(analysis: BehaviorAnalysis) -> tuple[np.ndarray, np.ndarray]:
    mean_sec = np.full(len(analysis.day_list), np.nan, dtype=float)
    sem_sec = np.full(len(analysis.day_list), np.nan, dtype=float)

    for day_idx in range(len(analysis.day_list)):
        lengths = analysis.trial.trial_length_frames_list[day_idx]
        outcome = analysis.trial.trials_outcome_list[day_idx]
        success_lengths = lengths[outcome == 1] / analysis.sampling_rate
        if len(success_lengths) == 0:
            continue
        mean_sec[day_idx] = np.nanmean(success_lengths)
        sem_sec[day_idx] = np.nanstd(success_lengths) / np.sqrt(len(success_lengths))
    return mean_sec, sem_sec


def make_daily_summary_table(analysis: BehaviorAnalysis) -> pd.DataFrame:
    mean_trial_length, sem_trial_length = average_success_trial_length_sec(analysis)
    day_from_zero = np.arange(len(analysis.day_list)) - analysis.phase.day_zero_idx
    remapping_start_min = analysis.trial.remapping_start_frame / analysis.sampling_rate / 60

    phase_label = np.full(len(analysis.day_list), "pre_day_zero", dtype=object)
    phase_label[analysis.phase.stable_control_days_idx] = "stable_control"
    phase_label[analysis.phase.remapping_days_idx] = "remapping"
    if analysis.phase.day_zero_idx < len(phase_label):
        phase_label[analysis.phase.day_zero_idx] = "day_zero"

    return pd.DataFrame(
        {
            "day_name": analysis.day_list,
            "day_from_day_zero": day_from_zero,
            "phase": phase_label,
            "exp_length_min": analysis.reward.exp_length_min,
            "all_reward_rate_per_min": analysis.reward.all_reward_rate,
            "active_reward_rate_per_min": analysis.reward.active_reward_rate,
            "num_hits": analysis.trial.num_hits,
            "num_misses": analysis.trial.num_misses,
            "hit_rate": analysis.trial.hit_rate,
            "avg_success_trial_length_sec": mean_trial_length,
            "sem_success_trial_length_sec": sem_trial_length,
            "mouse_weight_percentage": analysis.mouse_weight_percentage,
            "remapping_start_min": remapping_start_min,
        }
    )


def add_phase_vlines(ax: plt.Axes, phase: PhaseInfo, include_labels: bool = True) -> None:
    ax.axvline(
        x=phase.day_zero_idx,
        color="tab:blue",
        linestyle="--",
        linewidth=1,
        label="Day zero" if include_labels else None,
    )
    if phase.first_remapping_day_idx is not None:
        ax.axvline(
            x=phase.first_remapping_day_idx,
            color="tab:red",
            linestyle="--",
            linewidth=1,
            label="Start rotation" if include_labels else None,
        )
    if phase.end_remapping_phase_idx is not None:
        ax.axvline(
            x=phase.end_remapping_phase_idx,
            color="tab:green",
            linestyle="--",
            linewidth=1,
            label="End rotation" if include_labels else None,
        )


def add_phase_hlines(ax: plt.Axes, phase: PhaseInfo, include_labels: bool = True) -> None:
    ax.axhline(
        y=phase.day_zero_idx - 0.5,
        color="tab:blue",
        linestyle="--",
        linewidth=1,
        label="Day zero" if include_labels else None,
    )
    if phase.first_remapping_day_idx is not None:
        ax.axhline(
            y=phase.first_remapping_day_idx - 0.5,
            color="tab:red",
            linestyle="--",
            linewidth=1,
            label="Start rotation" if include_labels else None,
        )
    if phase.end_remapping_phase_idx is not None:
        ax.axhline(
            y=phase.end_remapping_phase_idx + 0.5,
            color="tab:green",
            linestyle="--",
            linewidth=1,
            label="End rotation" if include_labels else None,
        )


def save_figure(
    fig: plt.Figure,
    output_dir: str | Path,
    filename: str,
    dpi: int = 300,
    show: bool = True,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    if show:
        plt.show()
    return path


def plot_reward_rates_over_days(analysis: BehaviorAnalysis, show: bool = True) -> Path:
    fig, ax = plt.subplots(figsize=(7, 2.5))
    x = np.arange(len(analysis.day_list))
    ax.bar(x, analysis.reward.all_reward_rate, label="Total reward rate", color="0.65")
    ax.bar(x, analysis.reward.active_reward_rate, label="Active reward rate", color="black")
    add_phase_vlines(ax, analysis.phase)
    ax.set_xlabel("Day")
    ax.set_ylabel("Reward rate (rewards/min)")
    ax.set_title("Active vs total reward rate over days")
    ax.legend(loc=[1.05, 0.3])
    return save_figure(
        fig,
        analysis.output_dir,
        f"{analysis.mouse_name}_reward_rate_over_days.png",
        show=show,
    )


def plot_hit_rate_over_days(analysis: BehaviorAnalysis, show: bool = True) -> Path:
    fig, ax = plt.subplots(figsize=(7, 2.5))
    x = np.arange(len(analysis.day_list))
    ax.bar(x, analysis.trial.hit_rate, label="Hit rate", color="black")
    add_phase_vlines(ax, analysis.phase)
    ax.set_xlabel("Day")
    ax.set_ylabel("Hit rate")
    ax.set_ylim(0, 1)
    ax.set_title("Hit rate over days")
    ax.legend(loc=[1.05, 0.3])
    return save_figure(
        fig,
        analysis.output_dir,
        f"{analysis.mouse_name}_hit_rate_over_days.png",
        show=show,
    )


def plot_weight_and_trial_length(analysis: BehaviorAnalysis, show: bool = True) -> Path:
    mean_sec, sem_sec = average_success_trial_length_sec(analysis)
    x = np.arange(len(analysis.day_list))

    fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    ax[0].plot(x, analysis.mouse_weight_percentage, color="tab:blue", marker="o")
    ax[0].set_ylabel("Mouse weight (%)")
    ax[0].set_title("Mouse weight percentage over days")

    ax[1].bar(x, mean_sec, label="Average trial length", color="0.65")
    ax[1].errorbar(x, mean_sec, yerr=sem_sec, fmt=".", color="black", label="SEM")
    ax[1].set_xlabel("Day")
    ax[1].set_ylabel("Average trial length (s)")
    ax[1].set_title("Successful trial length over days")

    for i, axis in enumerate(ax):
        add_phase_vlines(axis, analysis.phase, include_labels=i == 1)
    ax[1].legend(loc=[1.05, 0.3])

    return save_figure(
        fig,
        analysis.output_dir,
        f"{analysis.mouse_name}_weight_and_trial_length_over_days.png",
        show=show,
    )


def plot_time_to_k_rewards(
    analysis: BehaviorAnalysis,
    k_rewards: int = 20,
    show: bool = True,
) -> Path:
    time_to_k = np.array(
        [
            reward_times[k_rewards - 1] if len(reward_times) >= k_rewards else np.nan
            for reward_times in analysis.reward.active_reward_times_min_list
        ],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(7, 2.5))
    x = np.arange(len(analysis.day_list))
    ax.bar(x, time_to_k, label=f"Time to {k_rewards} rewards", color="black")
    add_phase_vlines(ax, analysis.phase)
    ax.set_xlabel("Day")
    ax.set_ylabel(f"Time to {k_rewards} rewards (min)")
    ax.set_title(f"Time to {k_rewards} rewards over days")
    ax.legend(loc=[1.05, 0.3])
    return save_figure(
        fig,
        analysis.output_dir,
        f"{analysis.mouse_name}_time_to_{k_rewards}_rewards_over_days.png",
        show=show,
    )


def shift_with_nan(x: np.ndarray, shift_bins: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.full_like(x, np.nan, dtype=float)
    if shift_bins > 0:
        y[shift_bins:] = x[:-shift_bins]
    elif shift_bins < 0:
        y[:shift_bins] = x[-shift_bins:]
    else:
        y[:] = x
    return y


def align_reward_rate_to_remapping(
    analysis: BehaviorAnalysis,
    rotation_to_new_time_min: float = 10,
) -> np.ndarray:
    aligned = np.full_like(analysis.smoothed_reward_rate_stack, np.nan, dtype=float)
    for day_idx in analysis.phase.remapping_days_idx:
        remapping_start_min = (
            analysis.trial.remapping_start_frame[day_idx] / analysis.sampling_rate / 60
        )
        if not np.isfinite(remapping_start_min):
            continue
        shift_min = rotation_to_new_time_min - remapping_start_min
        shift_bins = int(round(shift_min * 60))
        aligned[day_idx] = shift_with_nan(
            analysis.smoothed_reward_rate_stack[day_idx],
            shift_bins,
        )
    return aligned


def plot_reward_rate_progression(
    analysis: BehaviorAnalysis,
    first_k_days: int | None = None,
    rotation_to_new_time_min: float = 10,
    rotation_to_old_time_min: float = 70,
    show: bool = True,
) -> Path:
    fig, ax = plt.subplots(2, 1, figsize=(7, 5))
    time_min = analysis.reward_rate_time_min

    stable_days = analysis.phase.stable_control_days_idx
    if first_k_days is not None:
        stable_days = stable_days[:first_k_days]
    if len(stable_days) > 0:
        average = np.nanmean(analysis.smoothed_reward_rate_stack[stable_days], axis=0)
        color_palette = sns.color_palette(
            "GnBu" if first_k_days is not None else "Reds",
            max(len(stable_days), 1),
        )
        for color_idx, day_idx in enumerate(stable_days):
            ax[0].plot(
                time_min,
                analysis.smoothed_reward_rate_stack[day_idx],
                label=f"Day {day_idx - analysis.phase.day_zero_idx}",
                alpha=0.5,
                linewidth=1.5,
                color=color_palette[color_idx],
            )
        ax[0].plot(time_min, average, label="Average", color="black", linewidth=2.5, linestyle="--")
    ax[0].set_ylabel("Reward rate (rewards/min)")
    ax[0].set_title("Stable control phase")
    ax[0].grid(color="lightgray", linestyle="--", linewidth=0.5)

    remap_days = analysis.phase.remapping_days_idx
    if first_k_days is not None:
        remap_days = remap_days[:first_k_days]
    if len(remap_days) > 0:
        aligned = align_reward_rate_to_remapping(
            analysis,
            rotation_to_new_time_min=rotation_to_new_time_min,
        )
        average = np.nanmean(aligned[remap_days], axis=0)
        color_palette = sns.color_palette(
            "GnBu" if first_k_days is not None else "Reds",
            max(len(remap_days), 1),
        )
        for color_idx, day_idx in enumerate(remap_days):
            ax[1].plot(
                time_min,
                aligned[day_idx],
                label=f"Day {day_idx - analysis.phase.first_remapping_day_idx}",
                alpha=0.5,
                linewidth=1.5,
                color=color_palette[color_idx],
            )
        ax[1].plot(time_min, average, label="Average", color="black", linewidth=2.5, linestyle="--")
        ax[1].axvline(rotation_to_new_time_min, color="tab:blue", linestyle="--", label="Rotation to new")
        ax[1].axvline(rotation_to_old_time_min, color="tab:red", linestyle="--", label="Rotation to old")
    ax[1].set_xlabel("Time in session (min)")
    ax[1].set_ylabel("Reward rate (rewards/min)")
    ax[1].set_title("Remapping phase")
    ax[1].grid(color="lightgray", linestyle="--", linewidth=0.5)

    title = "Reward rate progression"
    suffix = ""
    if first_k_days is not None:
        title = f"Reward rate progression for first {first_k_days} days in each phase"
        suffix = f"_first_{first_k_days}_days"
    fig.suptitle(title)
    return save_figure(
        fig,
        analysis.output_dir,
        f"{analysis.mouse_name}_reward_rate_progression{suffix}.png",
        show=show,
    )


def row_colors_by_phase(analysis: BehaviorAnalysis) -> list[str]:
    colors = []
    for day_idx in range(len(analysis.day_list)):
        if day_idx in analysis.phase.stable_control_days_idx:
            colors.append("tab:blue")
        elif day_idx in analysis.phase.remapping_days_idx:
            colors.append("tab:red")
        else:
            colors.append("0.45")
    return colors


def plot_reward_times_raster(
    analysis: BehaviorAnalysis,
    xlim_min: float | None = None,
    suffix: str = "",
    show: bool = True,
) -> Path:
    fig_height = max(4, 0.1 * len(analysis.day_list))
    fig, ax = plt.subplots(figsize=(9, fig_height))
    colors = row_colors_by_phase(analysis)

    for day_idx, reward_times_min in enumerate(analysis.reward.active_reward_times_min_list):
        if xlim_min is not None:
            reward_times_min = reward_times_min[reward_times_min <= xlim_min]
        y = np.full_like(reward_times_min, day_idx, dtype=float)
        ax.scatter(
            reward_times_min,
            y,
            s=5 if xlim_min is not None else 4,
            color=colors[day_idx],
            alpha=1 if xlim_min is not None else 0.75,
            linewidths=0,
        )

    add_phase_hlines(ax, analysis.phase)
    tick_step = 2
    yticks = np.arange(0, len(analysis.day_list), tick_step)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{i - analysis.phase.day_zero_idx:d}" for i in yticks])
    ax.set_ylim(-0.75, len(analysis.day_list) - 0.25)
    ax.set_xlim(0, xlim_min if xlim_min is not None else np.nanmax(analysis.reward.exp_length_min))
    ax.invert_yaxis()
    ax.set_xlabel("Time in session (min)")
    ax.set_ylabel("Day from day zero")
    ax.set_title(f"{analysis.mouse_name} reward times raster")
    ax.grid(axis="x", color="lightgray", linestyle="--", linewidth=0.5)
    ax.legend(loc="upper right")

    return save_figure(
        fig,
        analysis.output_dir,
        f"{analysis.mouse_name}_reward_times_raster{suffix}.png",
        show=show,
    )


def successful_trial_lengths_sec(
    analysis: BehaviorAnalysis,
    day_idx: int,
) -> np.ndarray:
    outcome = analysis.trial.trials_outcome_list[day_idx]
    lengths = analysis.trial.trial_length_frames_list[day_idx]
    return lengths[outcome == 1] / analysis.sampling_rate


def plot_reward_time_distributions(
    analysis: BehaviorAnalysis,
    n_first_days: int = 6,
    random_seed: int = 0,
    show: bool = True,
) -> dict[str, Path]:
    rng = np.random.default_rng(random_seed)
    paths: dict[str, Path] = {}

    fig, ax = plt.subplots(2, 1, figsize=(6, 5))
    for day_diff_idx in range(n_first_days):
        day_idx = analysis.phase.day_zero_idx + day_diff_idx + 1
        if day_idx >= len(analysis.day_list):
            continue
        lengths_sec = successful_trial_lengths_sec(analysis, day_idx)
        if len(lengths_sec) == 0:
            continue
        ax[0].hist(
            lengths_sec,
            bins=np.arange(0, 25, 1),
            density=True,
            label=f"Day {day_diff_idx + 1}",
            histtype="step",
            linewidth=2,
            alpha=0.7,
            color=f"C{day_diff_idx}",
        )
        if len(lengths_sec) > 1:
            sns.kdeplot(
                lengths_sec,
                ax=ax[1],
                color=f"C{day_diff_idx}",
                linewidth=1.5,
                clip=(0, None),
                alpha=0.7,
            )
    ax[1].set_xlabel("Successful trial length (s)")
    ax[0].set_ylabel("Density")
    ax[0].legend()
    paths["first_days_distribution"] = save_figure(
        fig,
        analysis.output_dir,
        f"{analysis.mouse_name}_reward_times_distribuation_first_{n_first_days}_days.png",
        show=show,
    )

    fig, ax = plt.subplots(figsize=(7, 2.5))
    for day_diff_idx in range(n_first_days):
        day_idx = analysis.phase.day_zero_idx + day_diff_idx + 1
        if day_idx >= len(analysis.day_list):
            continue
        lengths_sec = successful_trial_lengths_sec(analysis, day_idx)
        if len(lengths_sec) == 0:
            continue
        sample_size = max(1, int(len(lengths_sec) * 0.4))
        sample_idx = rng.choice(len(lengths_sec), size=sample_size, replace=False)
        sns.stripplot(
            x=np.full(sample_size, day_diff_idx + 1),
            y=lengths_sec[sample_idx],
            ax=ax,
            color=f"C{day_diff_idx}",
            alpha=1,
            size=2,
            jitter=0.2,
        )
        sns.boxplot(
            x=np.full(len(lengths_sec), day_diff_idx + 1),
            y=lengths_sec,
            ax=ax,
            color="gray",
            fliersize=0,
            width=0.25,
            boxprops=dict(alpha=0.3),
        )
    ax.set_xlabel("Day")
    ax.set_ylabel("Successful trial length (s)")
    ax.set_xticks(range(1, n_first_days + 1))
    ax.set_xticklabels([f"Day {i + 1}" for i in range(n_first_days)])
    paths["first_days_strip_box"] = save_figure(
        fig,
        analysis.output_dir,
        f"{analysis.mouse_name}_reward_times_violin_first_{n_first_days}_days.png",
        show=show,
    )

    stable_days_after_day_zero = [
        idx
        for idx in analysis.phase.stable_control_days_idx
        if idx > analysis.phase.day_zero_idx
    ]
    fig, ax = plt.subplots(figsize=(12, 2.5))
    for plot_idx, day_idx in enumerate(stable_days_after_day_zero):
        lengths_sec = successful_trial_lengths_sec(analysis, day_idx)
        if len(lengths_sec) == 0:
            continue
        sns.stripplot(
            x=np.full(len(lengths_sec), plot_idx + 1),
            y=lengths_sec,
            ax=ax,
            color=f"C{plot_idx}",
            alpha=0.5,
            size=2,
            jitter=0.2,
        )
        sns.boxplot(
            x=np.full(len(lengths_sec), plot_idx + 1),
            y=lengths_sec,
            ax=ax,
            color="gray",
            fliersize=0,
            width=0.25,
            boxprops=dict(alpha=0.4),
        )
    ax.set_xlabel("Day")
    ax.set_ylabel("Successful trial length (s)")
    ax.set_xticks(range(1, len(stable_days_after_day_zero) + 1))
    ax.set_xticklabels(
        [f"Day {i + 1}" for i in range(len(stable_days_after_day_zero))],
        rotation=45,
    )
    paths["stable_control_strip_box"] = save_figure(
        fig,
        analysis.output_dir,
        f"{analysis.mouse_name}_reward_times_violin_stable_control_days.png",
        show=show,
    )

    return paths


def rolling_zscore_neural_data(
    neural_data: np.ndarray,
    window_sec: float = 60,
    sampling_rate: float = 30,
    min_periods: int | None = None,
    center: bool = False,
    eps: float = 1e-8,
) -> np.ndarray:
    """Rolling z-score each neuron over time. Expects neural_data as neurons x time."""
    neural_data = np.asarray(neural_data, dtype=float)
    window_samples = int(round(window_sec * sampling_rate))
    if window_samples < 2:
        raise ValueError("window_sec is too small; rolling window must contain at least 2 samples.")
    if min_periods is None:
        min_periods = window_samples

    neural_df = pd.DataFrame(neural_data.T)
    rolling_mean = (
        neural_df.rolling(window_samples, min_periods=min_periods, center=center)
        .mean()
        .to_numpy()
        .T
    )
    rolling_std = (
        neural_df.rolling(window_samples, min_periods=min_periods, center=center)
        .std(ddof=0)
        .to_numpy()
        .T
    )

    neural_z = (neural_data - rolling_mean) / rolling_std
    neural_z[rolling_std < eps] = 0
    return neural_z


def get_rolling_zscore_neural_data(
    r: Any,
    day_name: str,
    window_sec: float = 60,
    sampling_rate: float = 30,
    min_periods: int | None = None,
    center: bool = False,
) -> np.ndarray:
    neural_data = r["data_neural_aligned"][day_name].load()
    return rolling_zscore_neural_data(
        neural_data,
        window_sec=window_sec,
        sampling_rate=sampling_rate,
        min_periods=min_periods,
        center=center,
    )


def get_cursor_from_neural_data_simple_projection(
    r: Any,
    day_name: str,
) -> np.ndarray:
    neural_data = r["data_neural_aligned"][day_name].load()
    decoder_ws_mat = r["weights_day0_idxFactors"].load()
    factor_to_use = r["extras"]["expParams"][day_name]["baselineStuff"]["factor_to_use"].load()
    decoder_ws_vec = decoder_ws_mat[:, int(factor_to_use) - 1]
    decoder_ws_vec = np.nan_to_num(decoder_ws_vec, nan=0.0)
    return neural_data.T @ decoder_ws_vec


def get_cursor_from_neural_data_decoder_formula(
    r: Any,
    day_name: str,
    p: float = 2,
    sampling_rate: float = 30,
    zscore_window_sec: float | None = 120,
) -> np.ndarray:
    neural_data = r["data_neural_aligned"][day_name].load()
    decoder_ws_mat = r["weights_day0_idxFactors"].load()
    factor_to_use = r["extras"]["expParams"][day_name]["baselineStuff"]["factor_to_use"].load()

    decoder_ws_vec = decoder_ws_mat[:, int(factor_to_use) - 1]
    decoder_ws_vec = np.nan_to_num(decoder_ws_vec, nan=0.0)
    neural_data = np.nan_to_num(neural_data, nan=0.0)
    if zscore_window_sec is not None:
        neural_data = rolling_zscore_neural_data(
            neural_data,
            window_sec=zscore_window_sec,
            sampling_rate=sampling_rate,
        )

    decoder_norm = np.linalg.norm(decoder_ws_vec)
    if decoder_norm == 0:
        return np.zeros(neural_data.shape[1])

    dot_with_decoder = neural_data.T @ decoder_ws_vec
    neural_norm = np.linalg.norm(neural_data, axis=0)
    cosine_similarity = np.divide(
        dot_with_decoder,
        neural_norm * decoder_norm,
        out=np.zeros_like(dot_with_decoder, dtype=float),
        where=neural_norm > 0,
    )
    projection_vec = (dot_with_decoder / decoder_norm**2)[None, :] * decoder_ws_vec[:, None]
    projection_norm = np.linalg.norm(projection_vec, axis=0)
    return (np.abs(cosine_similarity) ** p) * projection_norm


def get_auditory_tone_from_cursor(
    cursor: np.ndarray,
    f_min: float = 2000,
    f_max: float = 18000,
) -> np.ndarray:
    return f_min * (f_max / f_min) ** cursor


# Backward-compatible aliases for the exploratory notebook spelling.
get_coursor_from_neural_data_simple_projection = get_cursor_from_neural_data_simple_projection
get_coursor_from_neural_data_decoder_formula = get_cursor_from_neural_data_decoder_formula
get_coursor_projection_from_neural_data = get_cursor_from_neural_data_decoder_formula
get_audirory_tone_from_cursor = get_auditory_tone_from_cursor


def plot_cursor_diagnostic(
    analysis: BehaviorAnalysis,
    r: Any,
    day_idx: int | None = None,
    start_time_min: float = 9,
    duration_min: float = 3,
    cursor_smoothing_sigma: float = 5,
    p: float = 2,
    show: bool = True,
) -> Path | None:
    if day_idx is None:
        day_idx = analysis.phase.first_remapping_day_idx
    if day_idx is None:
        return None

    day_name = analysis.day_list[day_idx]
    cursor = np.asarray(r["data_ws"][day_name]["cursor"].load()).squeeze()
    time_min = np.arange(len(cursor)) / analysis.sampling_rate / 60
    active_reward_times = analysis.reward.active_reward_times_min_list[day_idx]
    trial_start_frames = analysis.trial.trial_start_frames_list[day_idx]

    neural_cursor_value = get_cursor_from_neural_data_decoder_formula(
        r,
        day_name,
        p=p,
        sampling_rate=analysis.sampling_rate,
    )
    neural_tone = get_auditory_tone_from_cursor(neural_cursor_value)

    decoder_logger = r["extras"]["logger_dfs"][day_name]["decoder"].load()
    freq_to_output = decoder_logger["freqToOutput"].to_numpy()

    start_sample = int(start_time_min * analysis.sampling_rate * 60)
    n_samples = int(duration_min * analysis.sampling_rate * 60)
    stop_sample = min(start_sample + n_samples, len(cursor))
    samples = slice(start_sample, stop_sample)

    cursor_smooth = gaussian_filter1d(cursor.astype(float), sigma=cursor_smoothing_sigma)
    neural_tone_smooth = gaussian_filter1d(neural_tone.astype(float), sigma=cursor_smoothing_sigma)
    freq_to_output_smooth = gaussian_filter1d(freq_to_output.astype(float), sigma=cursor_smoothing_sigma)

    fig, ax = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    ax[0].plot(time_min[samples], cursor_smooth[samples], label="cursor")
    ax[0].plot(time_min[samples], cursor[samples], label="cursor raw", alpha=0.3, color="gray")
    ax[1].plot(time_min[samples], neural_tone_smooth[samples], label="decoded tone", color="orange")
    ax[1].plot(time_min[samples], neural_tone[samples], label="decoded tone raw", alpha=0.3, color="gray")
    ax[2].plot(time_min[samples], freq_to_output_smooth[samples], label="freqToOutput", color="tab:blue")
    ax[2].plot(time_min[samples], freq_to_output[samples], label="freqToOutput raw", alpha=0.3, color="gray")

    for reward_time in active_reward_times:
        if start_time_min <= reward_time <= start_time_min + duration_min:
            for axis in ax:
                axis.axvline(reward_time, color="black", linestyle="--", alpha=0.5)

    for trial_start_frame in trial_start_frames:
        trial_start_min = trial_start_frame / analysis.sampling_rate / 60
        if start_time_min <= trial_start_min <= start_time_min + duration_min:
            for axis in ax:
                axis.axvline(trial_start_min, color="green", linestyle="--", alpha=0.5)

    ax[0].set_title("Cursor and reward signals from WaveSurfer")
    ax[1].set_title("Neural decoder formula converted to tone")
    ax[2].set_title("Logged freqToOutput")
    ax[2].set_xlabel("Time in session (min)")
    for axis in ax:
        axis.legend(loc="upper right")

    return save_figure(
        fig,
        analysis.output_dir,
        f"{analysis.mouse_name}_cursor_neural_projection_freqToOutput_day_{day_idx}.png",
        show=show,
    )


def plot_cursor_aligned_to_remapping(
    analysis: BehaviorAnalysis,
    r: Any,
    k_days: int = 9,
    start_of_remapping_min: float = 10,
    duration_to_plot_min: float = 7,
    start_time_diff_min: float = 2,
    sigma_cursor: float = 5,
    show: bool = True,
) -> Path | None:
    if analysis.phase.first_remapping_day_idx is None:
        return None

    fig, ax = plt.subplots(figsize=(12, 9))
    y_lap = 0.0
    start_day = analysis.phase.first_remapping_day_idx
    end_day = min(start_day + k_days, len(analysis.day_list))

    for day_idx in range(start_day, end_day):
        cursor = np.asarray(r["data_ws"][analysis.day_list[day_idx]]["cursor"].load()).squeeze()
        cursor_smooth = gaussian_filter1d(cursor.astype(float), sigma=sigma_cursor)

        remap_frame = analysis.trial.remapping_start_frame[day_idx]
        if not np.isfinite(remap_frame):
            continue
        remap_frame = int(remap_frame)

        aligned_time_min = (
            np.arange(len(cursor)) - remap_frame
        ) / analysis.sampling_rate / 60 + start_of_remapping_min

        start_sample = int(remap_frame - start_time_diff_min * analysis.sampling_rate * 60)
        stop_sample = int(start_sample + duration_to_plot_min * analysis.sampling_rate * 60)
        start_sample = max(start_sample, 0)
        stop_sample = min(stop_sample, len(cursor_smooth))

        cursor_segment = cursor_smooth[start_sample:stop_sample]
        time_segment = aligned_time_min[start_sample:stop_sample]
        if len(cursor_segment) == 0:
            continue
        segment_max = np.nanmax(cursor_segment)

        ax.plot(
            time_segment,
            cursor_segment + y_lap,
            label=f"Day {day_idx - start_day}",
        )

        for reward_time_min in analysis.reward.active_reward_times_min_list[day_idx]:
            reward_frame = int(reward_time_min * analysis.sampling_rate * 60)
            if start_sample <= reward_frame <= stop_sample:
                reward_time_aligned = (
                    reward_frame - remap_frame
                ) / analysis.sampling_rate / 60 + start_of_remapping_min
                ax.plot(
                    reward_time_aligned,
                    y_lap + segment_max,
                    "k.",
                    alpha=0.5,
                    markersize=8,
                )
                ax.plot(
                    [reward_time_aligned, reward_time_aligned],
                    [y_lap + segment_max / 2, y_lap + segment_max],
                    "--",
                    alpha=0.5,
                    color="gray",
                )

        y_lap += segment_max * 2 / 3 + 100

    ax.axvline(
        start_of_remapping_min,
        color="red",
        linestyle="--",
        alpha=0.8,
        label="Remapping start",
    )
    ax.set_title("Cursor and reward signals from WaveSurfer aligned to remapping")
    ax.set_xlabel("Aligned time (min)")
    ax.set_ylabel("Cursor signal (a.u.)")
    ax.set_yticks([])
    ax.legend()

    return save_figure(
        fig,
        analysis.output_dir,
        f"{analysis.mouse_name}_cursor_aligned_to_remapping.png",
        show=show,
    )


def run_standard_behavior_plots(
    analysis: BehaviorAnalysis,
    r: Any | None = None,
    first_k_days: int = 5,
    n_first_days: int = 6,
    k_rewards: int = 20,
    include_cursor_diagnostics: bool = True,
    show: bool = True,
) -> dict[str, Path | None]:
    paths: dict[str, Path | None] = {}
    paths["reward_rate_over_days"] = plot_reward_rates_over_days(analysis, show=show)
    paths["hit_rate_over_days"] = plot_hit_rate_over_days(analysis, show=show)
    paths["weight_and_trial_length"] = plot_weight_and_trial_length(analysis, show=show)
    paths[f"time_to_{k_rewards}_rewards"] = plot_time_to_k_rewards(
        analysis,
        k_rewards=k_rewards,
        show=show,
    )
    paths["reward_rate_progression"] = plot_reward_rate_progression(analysis, show=show)
    paths[f"reward_rate_progression_first_{first_k_days}"] = plot_reward_rate_progression(
        analysis,
        first_k_days=first_k_days,
        show=show,
    )
    paths["reward_times_raster"] = plot_reward_times_raster(analysis, show=show)
    paths["reward_times_raster_first_15_min"] = plot_reward_times_raster(
        analysis,
        xlim_min=15,
        suffix="_first_15_min",
        show=show,
    )
    paths.update(
        plot_reward_time_distributions(
            analysis,
            n_first_days=n_first_days,
            show=show,
        )
    )

    if include_cursor_diagnostics and r is not None:
        paths["cursor_diagnostic"] = plot_cursor_diagnostic(analysis, r, show=show)
        paths["cursor_aligned_to_remapping"] = plot_cursor_aligned_to_remapping(
            analysis,
            r,
            show=show,
        )
    return paths
