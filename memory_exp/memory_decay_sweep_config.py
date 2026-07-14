from __future__ import annotations

"""Sweep shared memory parameters in the Fast and Slow bar-chart models.

Config-style version for the urgency/deadline model files:
  - fast_1dpTask.py
  - slow_1dpTask.py

Edit the parameter blocks at the end of this file, then run:

    python memory_decay_sweep_config.py
"""

import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

MODELS = REPO_ROOT / "models"
VAL = REPO_ROOT / "baseline_validation"

import fast_1dpTask as fast_mod
import slow_1dpTask as slow_mod
from validate_fast_slow_baseline_rng import TrialResult, build_tasks, choose_bar_pairs


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Model builders / runners
# -----------------------------------------------------------------------------


def make_fast_planner(common_params: dict, fast_only_params: dict):
    return fast_mod.FastMemoryDecayTenthsPlanner(
        n_ticks=common_params["n_ticks"],
        height_step=common_params["height_step"],
        policy_len=common_params["policy_len"],
        gamma=common_params["gamma"],
        use_states_info_gain=True,
        action_selection="deterministic",
        non_report_cost=common_params["non_report_cost"],
        mem_retention=common_params["mem_retention"],
        mem_forget=common_params["mem_forget"],
        mem_sigma=common_params["mem_sigma"],
        pair_obs_sigma=fast_only_params["pair_obs_sigma"],
        segment_sigma=common_params["segment_sigma"],
        course_tick_anchor_bias=common_params["course_tick_anchor_bias"],
        segment_tick_anchor_bias=common_params["segment_tick_anchor_bias"],
        info_gain_target=common_params["info_gain_target"],
        report_action_instant=common_params["report_action_instant"],
        policy_eval_mode=common_params["policy_eval_mode"],
        use_log_prefs=common_params["use_log_prefs"],
        urgency_slope=fast_only_params.get("urgency_slope", 0.0),
    )



def make_slow_planner(common_params: dict, slow_only_params: dict):
    return slow_mod.SlowMemoryDecayTenthsPlanner(
        n_ticks=common_params["n_ticks"],
        height_step=common_params["height_step"],
        policy_len=common_params["policy_len"],
        gamma=common_params["gamma"],
        use_states_info_gain=True,
        action_selection="deterministic",
        non_report_cost=common_params["non_report_cost"],
        mem_retention=common_params["mem_retention"],
        mem_forget=common_params["mem_forget"],
        mem_sigma=common_params["mem_sigma"],
        segment_sigma=common_params["segment_sigma"],
        course_tick_anchor_bias=common_params["course_tick_anchor_bias"],
        bar_obs_sigma=slow_only_params["bar_obs_sigma"],
        segment_tick_anchor_bias=common_params["segment_tick_anchor_bias"],
        info_gain_target=common_params["info_gain_target"],
        report_action_instant=common_params["report_action_instant"],
        policy_eval_mode=common_params["policy_eval_mode"],
        use_log_prefs=common_params["use_log_prefs"],
        urgency_slope=slow_only_params.get("urgency_slope", 0.0),
    )



def run_one_fast(planner, bar_heights, seed, common_params, fast_only_params):
    np.random.seed(seed)
    env = fast_mod.DecimalFastAvgEnv(
        bar_heights=bar_heights,
        n_ticks=common_params["n_ticks"],
        height_step=common_params["height_step"],
        pair_obs_sigma=fast_only_params["pair_obs_sigma"],
        segment_sigma=common_params["segment_sigma"],
        course_tick_anchor_bias=common_params["course_tick_anchor_bias"],
        segment_tick_anchor_bias=common_params["segment_tick_anchor_bias"],
    )
    planner.reset_beliefs()
    meta = planner.meta()
    force_report = bool(common_params["force_report_at_deadline"])
    forced_rule = common_params["forced_report_rule"]

    for t in range(fast_only_params["T"]):
        if force_report and t == fast_only_params["T"] - 1:
            action = planner.forced_report_action(rule=forced_rule)
        else:
            q_pi, _ = planner.infer_policies()
            action = planner.sample_action(q_pi)
        obs = env.step(action)
        planner.update_beliefs(action, obs)
        if action >= meta.report_start:
            report_idx = action - meta.report_start
            reported = float(meta.avg_grid[report_idx])
            return TrialResult(
                model="fast",
                bar0=float(bar_heights[0]),
                bar1=float(bar_heights[1]),
                true_avg=float(env.true_avg),
                reported_avg=reported,
                correct=bool(obs[3] == 2),
                steps=t + 1,
                seed=int(seed),
            )

    return TrialResult(
        model="fast",
        bar0=float(bar_heights[0]),
        bar1=float(bar_heights[1]),
        true_avg=float(env.true_avg),
        reported_avg=None,
        correct=False,
        steps=int(fast_only_params["T"]),
        seed=int(seed),
    )



def run_one_slow(planner, bar_heights, seed, common_params, slow_only_params):
    np.random.seed(seed)
    env = slow_mod.DecimalSegmentEnv(
        bar_heights=bar_heights,
        n_ticks=common_params["n_ticks"],
        height_step=common_params["height_step"],
        segment_sigma=common_params["segment_sigma"],
        course_tick_anchor_bias=common_params["course_tick_anchor_bias"],
        bar_obs_sigma=slow_only_params["bar_obs_sigma"],
        segment_tick_anchor_bias=common_params["segment_tick_anchor_bias"],
        info_gain_target=common_params["info_gain_target"],
        report_action_instant=common_params["report_action_instant"],
    )
    planner.reset_beliefs()
    meta = planner.meta()
    force_report = bool(common_params["force_report_at_deadline"])
    forced_rule = common_params["forced_report_rule"]

    for t in range(slow_only_params["T"]):
        if force_report and t == slow_only_params["T"] - 1:
            action = planner.forced_report_action(rule=forced_rule)
        else:
            q_pi, _ = planner.infer_policies()
            action = planner.sample_action(q_pi)
        obs = env.step(action)
        planner.update_beliefs(action, obs)
        if action >= meta.report_start:
            report_idx = action - meta.report_start
            reported = float(planner.report_grid[report_idx])
            return TrialResult(
                model="slow",
                bar0=float(bar_heights[0]),
                bar1=float(bar_heights[1]),
                true_avg=float(env.true_avg),
                reported_avg=reported,
                correct=bool(obs[3] == 2),
                steps=t + 1,
                seed=int(seed),
            )

    return TrialResult(
        model="slow",
        bar0=float(bar_heights[0]),
        bar1=float(bar_heights[1]),
        true_avg=float(env.true_avg),
        reported_avg=None,
        correct=False,
        steps=int(slow_only_params["T"]),
        seed=int(seed),
    )


# -----------------------------------------------------------------------------
# Parallel worker state
# -----------------------------------------------------------------------------


_FAST_PLANNER = None
_SLOW_PLANNER = None
_COMMON_PARAMS = None
_FAST_ONLY_PARAMS = None
_SLOW_ONLY_PARAMS = None



def _init_worker(common_params: dict, fast_only_params: dict, slow_only_params: dict):
    global _FAST_PLANNER, _SLOW_PLANNER, _COMMON_PARAMS, _FAST_ONLY_PARAMS, _SLOW_ONLY_PARAMS
    _COMMON_PARAMS = common_params
    _FAST_ONLY_PARAMS = fast_only_params
    _SLOW_ONLY_PARAMS = slow_only_params
    _FAST_PLANNER = make_fast_planner(common_params, fast_only_params)
    _SLOW_PLANNER = make_slow_planner(common_params, slow_only_params)



def _run_trial(task):
    bars, trial_seed = task
    bars = tuple(map(float, bars))
    fast_res = run_one_fast(_FAST_PLANNER, bars, seed=trial_seed, common_params=_COMMON_PARAMS, fast_only_params=_FAST_ONLY_PARAMS)
    slow_res = run_one_slow(_SLOW_PLANNER, bars, seed=trial_seed, common_params=_COMMON_PARAMS, slow_only_params=_SLOW_ONLY_PARAMS)
    return fast_res, slow_res


# -----------------------------------------------------------------------------
# Analysis helpers
# -----------------------------------------------------------------------------


def results_to_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["reported_avg"] = pd.to_numeric(df["reported_avg"], errors="coerce")
    df["true_avg"] = pd.to_numeric(df["true_avg"], errors="coerce")
    df["bar0"] = pd.to_numeric(df["bar0"], errors="coerce")
    df["bar1"] = pd.to_numeric(df["bar1"], errors="coerce")
    df["steps"] = pd.to_numeric(df["steps"], errors="coerce")
    df["correct"] = df["correct"].astype(bool)
    df["reported"] = df["reported_avg"].notna()
    df["abs_error"] = (df["reported_avg"] - df["true_avg"]).abs()
    df["bar0_on_tick"] = np.isclose(df["bar0"], np.round(df["bar0"]), atol=1e-8)
    df["bar1_on_tick"] = np.isclose(df["bar1"], np.round(df["bar1"]), atol=1e-8)
    df["pair_category"] = np.select(
        [
            df["bar0_on_tick"] & df["bar1_on_tick"],
            df["bar0_on_tick"] ^ df["bar1_on_tick"],
        ],
        ["on_on", "on_off"],
        default="off_off",
    )
    return df



def mean_ci(values: Iterable[float]) -> tuple[float, float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(arr))
    if arr.size == 1:
        return mean, mean, mean
    se = float(np.std(arr, ddof=1) / math.sqrt(arr.size))
    lo = max(0.0, mean - 1.96 * se)
    hi = min(1.0 if np.all((arr >= 0) & (arr <= 1)) else np.inf, mean + 1.96 * se)
    return mean, lo, hi



def summarize(df: pd.DataFrame, sweep_param: str) -> pd.DataFrame:
    rows = []
    for keys, sub in df.groupby(["model", sweep_param], sort=True):
        model, sweep_value = keys
        acc, acc_lo, acc_hi = mean_ci(sub["correct"].astype(float))
        steps, steps_lo, steps_hi = mean_ci(sub["steps"].astype(float))
        mae, mae_lo, mae_hi = mean_ci(sub.loc[sub["reported"], "abs_error"].astype(float))
        rows.append(
            {
                "model": model,
                sweep_param: float(sweep_value),
                "n_trials": int(len(sub)),
                "accuracy": acc,
                "accuracy_lo": acc_lo,
                "accuracy_hi": acc_hi,
                "mean_steps": steps,
                "mean_steps_lo": steps_lo,
                "mean_steps_hi": steps_hi,
                "mae": mae,
                "mae_lo": mae_lo,
                "mae_hi": mae_hi,
            }
        )
    return pd.DataFrame(rows).sort_values([sweep_param, "model"]).reset_index(drop=True)



def summarize_by_category(df: pd.DataFrame, sweep_param: str) -> pd.DataFrame:
    rows = []
    for keys, sub in df.groupby(["model", sweep_param, "pair_category"], sort=True):
        model, sweep_value, pair_category = keys
        acc, acc_lo, acc_hi = mean_ci(sub["correct"].astype(float))
        rows.append(
            {
                "model": model,
                sweep_param: float(sweep_value),
                "pair_category": pair_category,
                "n_trials": int(len(sub)),
                "accuracy": acc,
                "accuracy_lo": acc_lo,
                "accuracy_hi": acc_hi,
            }
        )
    return pd.DataFrame(rows).sort_values([sweep_param, "pair_category", "model"]).reset_index(drop=True)



def x_values_for_plot(df: pd.DataFrame, sweep_param: str) -> pd.Series:
    if sweep_param == "mem_retention":
        return 1.0 - df[sweep_param]
    return df[sweep_param]



def x_label_for_plot(sweep_param: str) -> str:
    if sweep_param == "mem_retention":
        return "Memory decay rate (1 - mem_retention)"
    if sweep_param == "mem_forget":
        return "Memory forgetting probability (mem_forget)"
    if sweep_param == "mem_sigma":
        return "Memory diffusion width (mem_sigma)"
    return sweep_param


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def save_figure(fig: plt.Figure, basepath: Path) -> None:
    ensure_parent(basepath)
    fig.savefig(basepath.with_suffix('.png'), dpi=600, bbox_inches='tight')
    fig.savefig(basepath.with_suffix('.pdf'), bbox_inches='tight')



def plot_main(summary_df: pd.DataFrame, sweep_param: str, outbase: Path, title: str | None = None) -> None:
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(3.35, 4.5),
        gridspec_kw={"height_ratios": [1.0, 0.9]},
        constrained_layout=True,
    )

    xlabel = x_label_for_plot(sweep_param)
    for model in ["fast", "slow"]:
        sub = summary_df[summary_df["model"] == model].copy()
        x = x_values_for_plot(sub, sweep_param)
        ax1.errorbar(
            x,
            sub["accuracy"],
            yerr=np.vstack([sub["accuracy"] - sub["accuracy_lo"], sub["accuracy_hi"] - sub["accuracy"]]),
            fmt="o-",
            capsize=2.5,
            linewidth=1.2,
            markersize=3.8,
            label=model.capitalize(),
        )
        ax2.errorbar(
            x,
            sub["mean_steps"],
            yerr=np.vstack([sub["mean_steps"] - sub["mean_steps_lo"], sub["mean_steps_hi"] - sub["mean_steps"]]),
            fmt="o-",
            capsize=2.5,
            linewidth=1.2,
            markersize=3.8,
            label=model.capitalize(),
        )

    ax1.set_ylabel("Accuracy", fontsize=8)
    ax1.set_ylim(0.0, 1.02)
    ax1.set_title("A) Accuracy vs. memory parameter", loc="left", fontsize=9, pad=2)
    ax1.grid(alpha=0.25, linewidth=0.5)
    ax1.tick_params(axis="both", labelsize=7)
    ax1.yaxis.set_major_locator(MaxNLocator(5))
    ax1.legend(frameon=False, fontsize=7, ncol=2, loc="lower left")

    ax2.set_xlabel(xlabel, fontsize=8)
    ax2.set_ylabel("Mean steps", fontsize=8)
    ax2.set_title("B) Mean steps vs. memory parameter", loc="left", fontsize=9, pad=2)
    ax2.grid(alpha=0.25, linewidth=0.5)
    ax2.tick_params(axis="both", labelsize=7)
    ax2.yaxis.set_major_locator(MaxNLocator(5))

    if title:
        fig.suptitle(title, fontsize=10)

    save_figure(fig, outbase)
    plt.close(fig)



def plot_category_accuracy(cat_df: pd.DataFrame, sweep_param: str, outbase: Path, title: str | None = None) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(3.35, 6.0), constrained_layout=True)
    cat_order = ["on_on", "on_off", "off_off"]
    cat_labels = {"on_on": "on/on", "on_off": "on/off", "off_off": "off/off"}
    xlabel = x_label_for_plot(sweep_param)

    for ax, cat in zip(axes, cat_order):
        for model in ["fast", "slow"]:
            sub = cat_df[(cat_df["pair_category"] == cat) & (cat_df["model"] == model)].copy()
            if sub.empty:
                continue
            x = x_values_for_plot(sub, sweep_param)
            ax.errorbar(
                x,
                sub["accuracy"],
                yerr=np.vstack([sub["accuracy"] - sub["accuracy_lo"], sub["accuracy_hi"] - sub["accuracy"]]),
                fmt="o-",
                capsize=2.0,
                linewidth=1.1,
                markersize=3.4,
                label=model.capitalize(),
            )
        ax.set_ylim(0.0, 1.02)
        ax.set_ylabel("Accuracy", fontsize=8)
        ax.set_title(f"{cat_labels[cat]}", loc="left", fontsize=9, pad=2)
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.tick_params(axis="both", labelsize=7)
        ax.yaxis.set_major_locator(MaxNLocator(5))

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(frameon=False, fontsize=7, ncol=2, loc="lower left")
    axes[-1].set_xlabel(xlabel, fontsize=8)

    if title:
        fig.suptitle(title, fontsize=10)

    save_figure(fig, outbase)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Sweep runner
# -----------------------------------------------------------------------------


def run_sweep(
    common_params: dict,
    fast_only_params: dict,
    slow_only_params: dict,
    sweep_param: str,
    sweep_values: list[float],
    tasks: list[tuple[tuple[float, float], int]],
    parallel: bool,
    max_workers: int | None,
    chunksize: int,
) -> list[dict]:
    all_rows: list[dict] = []
    total_settings = len(sweep_values)

    for i, sweep_value in enumerate(sweep_values, start=1):
        common_i = dict(common_params)
        common_i[sweep_param] = float(sweep_value)

        print(f"[{i}/{total_settings}] {sweep_param} = {sweep_value:.6g}")
        setting_rows: list[dict] = []

        if parallel:
            workers = max_workers
            if workers is None:
                workers = max(1, (os.cpu_count() or 1) - 1)
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=ctx,
                initializer=_init_worker,
                initargs=(common_i, fast_only_params, slow_only_params),
            ) as ex:
                total = len(tasks)
                for j, (fast_res, slow_res) in enumerate(ex.map(_run_trial, tasks, chunksize=chunksize), start=1):
                    for res in (fast_res, slow_res):
                        row = asdict(res)
                        row[sweep_param] = float(sweep_value)
                        if sweep_param == "mem_retention":
                            row["decay_rate"] = 1.0 - float(sweep_value)
                        setting_rows.append(row)
                    if j % 25 == 0 or j == total:
                        print(f"    completed {j}/{total} trials")
        else:
            fast_planner = make_fast_planner(common_i, fast_only_params)
            slow_planner = make_slow_planner(common_i, slow_only_params)
            total = len(tasks)
            for j, (bars, trial_seed) in enumerate(tasks, start=1):
                for res in (
                    run_one_fast(fast_planner, bars, seed=trial_seed, common_params=common_i, fast_only_params=fast_only_params),
                    run_one_slow(slow_planner, bars, seed=trial_seed, common_params=common_i, slow_only_params=slow_only_params),
                ):
                    row = asdict(res)
                    row[sweep_param] = float(sweep_value)
                    if sweep_param == "mem_retention":
                        row["decay_rate"] = 1.0 - float(sweep_value)
                    setting_rows.append(row)
                if j % 25 == 0 or j == total:
                    print(f"    completed {j}/{total} trials")

        all_rows.extend(setting_rows)

        setting_df = results_to_df(setting_rows)
        by_model = setting_df.groupby("model", as_index=False).agg(
            accuracy=("correct", "mean"),
            mean_steps=("steps", "mean"),
            mae=("abs_error", "mean"),
            n_trials=("correct", "size"),
        )
        with pd.option_context("display.max_columns", None, "display.width", 160):
            print(by_model.to_string(index=False))
        print()

    return all_rows


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main(
    common_params: dict,
    fast_only_params: dict,
    slow_only_params: dict,
    sweep_config: dict,
    target_props: dict,
    seeds=range(3),
    exhaustive: bool = False,
    random_n_pairs: int | None = 15,
    random_seed: int = 7,
    outdir: str | Path = "memory_decay_sweep_out_urgency",
    title: str | None = None,
    parallel: bool = True,
    max_workers: int | None = None,
    chunksize: int = 16,
    seed_stride: int | None = None,
) -> None:
    sweep_param = sweep_config["sweep_param"]
    sweep_values = list(sweep_config["sweep_values"])
    if sweep_param not in {"mem_retention", "mem_forget", "mem_sigma"}:
        raise ValueError("sweep_param must be one of: mem_retention, mem_forget, mem_sigma")

    outdir = Path(outdir)
    grid = slow_mod.make_height_grid(common_params["n_ticks"], common_params["height_step"])
    pairs, sampling_meta = choose_bar_pairs(
        grid,
        target_props=target_props,
        exhaustive=bool(exhaustive),
        random_n_pairs=None if exhaustive else random_n_pairs,
        random_seed=random_seed,
    )
    seeds = list(seeds)
    tasks = build_tasks(pairs, seeds=seeds, seed_stride=seed_stride)

    print(f"Sweep parameter : {sweep_param}")
    print(f"Sweep values    : {sweep_values}")
    print(f"N bar pairs     : {len(pairs)}")
    print(f"Base seeds      : {seeds}")
    print(f"T_fast / T_slow : {fast_only_params['T']} / {slow_only_params['T']}")
    print(f"Urgency slopes  : fast={fast_only_params.get('urgency_slope', 0.0)}, slow={slow_only_params.get('urgency_slope', 0.0)}")
    print(f"Force deadline  : {bool(common_params['force_report_at_deadline'])} ({common_params['forced_report_rule']})")
    if "sampled_pair_counts" in sampling_meta:
        print(f"Pair coverage   : {sampling_meta['sampled_pair_counts']}")
    print()

    rows = run_sweep(
        common_params=common_params,
        fast_only_params=fast_only_params,
        slow_only_params=slow_only_params,
        sweep_param=sweep_param,
        sweep_values=sweep_values,
        tasks=tasks,
        parallel=bool(parallel),
        max_workers=max_workers,
        chunksize=chunksize,
    )

    df = results_to_df(rows)
    summary_df = summarize(df, sweep_param)
    category_df = summarize_by_category(df, sweep_param)

    outdir.mkdir(parents=True, exist_ok=True)
    results_json = outdir / "memory_sweep_results.json"
    summary_csv = outdir / "memory_sweep_summary.csv"
    category_csv = outdir / "memory_sweep_by_category.csv"
    copied_meta_json = outdir / "memory_sweep_metadata.json"

    payload = {
        "metadata": {
            "common": common_params,
            "fast_only": fast_only_params,
            "slow_only": slow_only_params,
            "sweep": sweep_config,
            "sampling": sampling_meta,
            "base_seeds": seeds,
            "seed_stride": seed_stride,
            "parallel": bool(parallel),
            "max_workers": max_workers,
            "chunksize": chunksize,
        },
        "results": rows,
    }
    results_json.write_text(json.dumps(payload, indent=2))
    summary_df.to_csv(summary_csv, index=False)
    category_df.to_csv(category_csv, index=False)
    copied_meta_json.write_text(json.dumps(payload["metadata"], indent=2))

    plot_main(summary_df, sweep_param, outdir / "memory_sweep_main", title=title)
    plot_category_accuracy(category_df, sweep_param, outdir / "memory_sweep_category_accuracy", title=title)

    print("Saved:")
    for path in [
        results_json,
        summary_csv,
        category_csv,
        copied_meta_json,
        outdir / "memory_sweep_main.png",
        outdir / "memory_sweep_main.pdf",
        outdir / "memory_sweep_category_accuracy.png",
        outdir / "memory_sweep_category_accuracy.pdf",
    ]:
        print(f"  {path}")


if __name__ == "__main__":
    COMMON = dict(
        n_ticks=4,
        height_step=0.1,
        policy_len=4,
        gamma=8.0,
        policy_eval_mode="rollout",
        info_gain_target="report",
        report_action_instant=True,
        # 
        mem_retention=0.97,
        mem_forget=0.005,
        mem_sigma=5.0,
        # ,
        non_report_cost=0.35,
        segment_sigma=0.045,
        course_tick_anchor_bias=0.0,
        segment_tick_anchor_bias=0.0,
        # 
        use_log_prefs=False,
        force_report_at_deadline=True,
        forced_report_rule="map",
    )

    FAST_ONLY = dict(
        # T=10,
        T=7,
        pair_obs_sigma=0.22,
        urgency_slope=0.05,
    )

    SLOW_ONLY = dict(
        # T=16,
        T=10,
        bar_obs_sigma=0.00,
        urgency_slope=0.05,
    )

    SWEEP = dict(
        sweep_param="mem_retention",
        # sweep_values=[0.97, 0.94, 0.91, 0.88, 0.85],
        sweep_values=[0.99, .95, .9, .85, .8, .75, .7, .65, .6, .55, .5, .45, .4, .35, .3, .25, .2, .15, 0.1, 0.05],
    )
    # SWEEP = dict(
    #     sweep_param="mem_sigma",
    #     sweep_values=[0.0, 0.25, 0.5, 1.0, 2.0, 5.0],
    # )

    TARGET_PROPORTIONS = {  # Not used in exhaustive mode
        "on_on": 0.10,
        "on_off": 0.30,
        "off_off": 0.60,
    }

    main(
        COMMON,
        FAST_ONLY,
        SLOW_ONLY,
        SWEEP,
        TARGET_PROPORTIONS,
        # seeds=range(3),
        seeds=range(10),
        exhaustive=False,
        random_n_pairs=10,
        # random_n_pairs=3,
        random_seed=7,
        outdir="memory_decay_sweep_out_deadlines_bigSweep",
        title=None,
        parallel=True,
        max_workers=None,
        chunksize=16,
        seed_stride=None,
    )
