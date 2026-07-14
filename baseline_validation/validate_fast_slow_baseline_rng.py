from __future__ import annotations

"""Validate baseline accuracy of the Fast and Slow bar-chart models.

This script runs both uploaded models over many bar-height pairs and reports
exact-match accuracy and mean absolute error (MAE) of the reported average.

It uses the model classes directly rather than parsing printed output from
`run_sim()`. The baseline defaults match the shared settings you identified
for the paper, with model-specific parameters for the fast and slow scripts.
"""

from dataclasses import asdict, dataclass
from itertools import product, combinations_with_replacement
from pathlib import Path
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

MODELS = REPO_ROOT / "models"

import fast_1dpTask as fast_mod
import slow_1dpTask as slow_mod


@dataclass
class TrialResult:
    model: str
    bar0: float
    bar1: float
    true_avg: float
    reported_avg: float | None
    correct: bool
    steps: int
    seed: int


def is_on_tick(h: float) -> bool:
    return bool(np.isclose(h, round(h), atol=1e-8))


def _integer_quotas(n: int, proportions: dict[str, float]) -> dict[str, int]:
    """
    Convert target proportions into integer counts that sum exactly to n
    using largest-remainder rounding.
    """
    raw = {k: n * p for k, p in proportions.items()}
    q = {k: int(np.floor(v)) for k, v in raw.items()}
    remainder = n - sum(q.values())

    order = sorted(
        proportions.keys(),
        key=lambda k: raw[k] - q[k],
        reverse=True,
    )
    for k in order[:remainder]:
        q[k] += 1
    return q


def choose_bar_pairs(
    grid: np.ndarray,
    target_props: dict,
    exhaustive: bool = False,
    sample_step: int = 5,
    random_n_pairs: int | None = None,
    random_seed: int = 0,
) -> tuple[list[tuple[float, float]], dict]:
    if exhaustive:
        pairs = [tuple(map(float, p)) for p in product(grid, grid)]
        return pairs, {
            "sampling_mode": "exhaustive",
            "n_pairs": len(pairs),
            "heights_sampled": [float(x) for x in grid],
        }

    if random_n_pairs is not None:
        rng = np.random.default_rng(random_seed)
        on_tick = [float(h) for h in grid if is_on_tick(float(h))]
        off_tick = [float(h) for h in grid if not is_on_tick(float(h))]

        # Unordered pools: (a,b) and (b,a) are treated as the same pair
        categories = {
            "on_on": list(combinations_with_replacement(on_tick, 2)),
            "on_off": [(a, b) for a in on_tick for b in off_tick],
            "off_off": list(combinations_with_replacement(off_tick, 2)),
        }

        quotas = _integer_quotas(random_n_pairs, target_props)

        sampled = []
        for key in ["on_on", "on_off", "off_off"]:
            pool = categories[key]
            k = min(quotas[key], len(pool))
            idx = rng.choice(len(pool), size=k, replace=False)
            sampled.extend(pool[i] for i in np.atleast_1d(idx))

        rng.shuffle(sampled)

        return sampled, {
            "sampling_mode": "natural_unordered_stratified_random",
            "random_seed": random_seed,
            "random_n_pairs": random_n_pairs,
            "n_pairs": len(sampled),
            "target_pair_proportions": target_props,
            "target_pair_counts": quotas,
            "n_on_tick_heights": len(on_tick),
            "n_off_tick_heights": len(off_tick),
            "sampled_pair_counts": {
                "on_on": int(sum(is_on_tick(a) and is_on_tick(b) for a, b in sampled)),
                "on_off": int(sum(is_on_tick(a) ^ is_on_tick(b) for a, b in sampled)),
                "off_off": int(sum((not is_on_tick(a)) and (not is_on_tick(b)) for a, b in sampled)),
            },
        }

    raise ValueError("Set either exhaustive=True or provide random_n_pairs.")


def build_tasks(
    pairs: list[tuple[float, float]],
    seeds: list[int],
    seed_stride: int | None = None,
) -> list[tuple[tuple[float, float], int]]:
    """
    Build tasks with a unique trial seed for every (pair, base_seed).

    trial_seed = base_seed * seed_stride + pair_idx

    Choose seed_stride > number of pairs to avoid collisions.
    """
    if seed_stride is None:
        seed_stride = max(100000, len(pairs) + 1)

    if seed_stride <= len(pairs):
        raise ValueError(
            f"seed_stride={seed_stride} must be greater than len(pairs)={len(pairs)}"
        )

    tasks = []
    for base_seed in seeds:
        for pair_idx, bars in enumerate(pairs):
            trial_seed = base_seed * seed_stride + pair_idx
            tasks.append((tuple(map(float, bars)), trial_seed))
    return tasks


def make_fast_planner(params: dict):
    return fast_mod.FastMemoryDecayTenthsPlanner(
        n_ticks=params["n_ticks"],
        height_step=params["height_step"],
        policy_len=params["policy_len"],
        gamma=params["gamma"],
        use_states_info_gain=True,
        action_selection="deterministic",
        non_report_cost=params["non_report_cost"],
        mem_retention=params["mem_retention"],
        mem_forget=params["mem_forget"],
        mem_sigma=params["mem_sigma"],
        pair_obs_sigma=params["pair_obs_sigma"],
        segment_sigma=params["segment_sigma"],
        course_tick_anchor_bias=params["course_tick_anchor_bias"],
        segment_tick_anchor_bias=params["segment_tick_anchor_bias"],
        urgency_slope=params.get("urgency_slope", 0.0),
    )


def make_slow_planner(params: dict):
    return slow_mod.SlowMemoryDecayTenthsPlanner(
        n_ticks=params["n_ticks"],
        height_step=params["height_step"],
        policy_len=params["policy_len"],
        gamma=params["gamma"],
        use_states_info_gain=True,
        action_selection="deterministic",
        non_report_cost=params["non_report_cost"],
        mem_retention=params["mem_retention"],
        mem_forget=params["mem_forget"],
        mem_sigma=params["mem_sigma"],
        segment_sigma=params["segment_sigma"],
        course_tick_anchor_bias=params["course_tick_anchor_bias"],
        segment_tick_anchor_bias=params["segment_tick_anchor_bias"],
        urgency_slope=params.get("urgency_slope", 0.0),
    )


def run_one_fast(planner, bar_heights, seed, params):
    np.random.seed(seed)
    env = fast_mod.DecimalFastAvgEnv(
        bar_heights=bar_heights,
        n_ticks=params["n_ticks"],
        height_step=params["height_step"],
        pair_obs_sigma=params["pair_obs_sigma"],
        segment_sigma=params["segment_sigma"],
        course_tick_anchor_bias=params["course_tick_anchor_bias"],
        segment_tick_anchor_bias=params["segment_tick_anchor_bias"],
    )
    planner.reset_beliefs()
    meta = planner.meta()
    obs = (0, 0, 0, 0)

    force_report_at_deadline = params.get("force_report_at_deadline", True)
    forced_report_rule = params.get("forced_report_rule", "map")

    for t in range(params["T"]):
        if force_report_at_deadline and t == params["T"] - 1:
            action = planner.forced_report_action(rule=forced_report_rule)
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
                seed=seed,
            )

    return TrialResult(
        "fast",
        float(bar_heights[0]),
        float(bar_heights[1]),
        float(env.true_avg),
        None,
        False,
        params["T"],
        seed,
    )


def run_one_slow(planner, bar_heights: tuple[float, float], seed: int, params: dict) -> TrialResult:
    np.random.seed(seed)
    env = slow_mod.DecimalSegmentEnv(
        bar_heights=bar_heights,
        n_ticks=params["n_ticks"],
        height_step=params["height_step"],
        segment_sigma=params["segment_sigma"],
        course_tick_anchor_bias=params["course_tick_anchor_bias"],
        segment_tick_anchor_bias=params["segment_tick_anchor_bias"],
    )
    planner.reset_beliefs()
    meta = planner.meta()
    obs = (0, 0, 0, 0)

    force_report_at_deadline = params.get("force_report_at_deadline", True)
    forced_report_rule = params.get("forced_report_rule", "map")

    for t in range(params["T"]):
        if force_report_at_deadline and t == params["T"] - 1:
            action = planner.forced_report_action(rule=forced_report_rule)
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
                seed=seed,
            )

    return TrialResult(
        "slow",
        float(bar_heights[0]),
        float(bar_heights[1]),
        float(env.true_avg),
        None,
        False,
        params["T"],
        seed,
    )


def summarize(results: list[TrialResult]) -> None:
    by_model = {}
    for r in results:
        by_model.setdefault(r.model, []).append(r)

    print("\n=== Validation summary ===")
    for model, rows in by_model.items():
        n = len(rows)
        n_report = sum(r.reported_avg is not None for r in rows)
        n_correct = sum(r.correct for r in rows)
        maes = [abs(r.reported_avg - r.true_avg) for r in rows if r.reported_avg is not None]
        mean_steps = float(np.mean([r.steps for r in rows]))
        print(
            f"{model:>4s} | trials={n:4d} | report_rate={n_report/n:6.3f} | "
            f"accuracy={n_correct/n:6.3f} | MAE={np.mean(maes):6.3f} | mean_steps={mean_steps:5.2f}"
        )

        failures = [r for r in rows if not r.correct][:5]
        if failures:
            print("      example failures:")
            for r in failures:
                rep = "None" if r.reported_avg is None else f"{r.reported_avg:.2f}"
                print(
                    f"        bars=({r.bar0:.1f}, {r.bar1:.1f}) "
                    f"true={r.true_avg:.2f} reported={rep} seed={r.seed}"
                )


def save_results(results: list[TrialResult], metadata: dict, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    payload = {
        "metadata": metadata,
        "results": [asdict(r) for r in results],
    }
    output_path.write_text(json.dumps(payload, indent=2))
    return output_path


# -------------------------
# Parallel helpers
# -------------------------

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os

_FAST_PLANNER = None
_SLOW_PLANNER = None
_FAST_PARAMS = None
_SLOW_PARAMS = None


def _init_worker(fast_params: dict, slow_params: dict):
    """Build one fast and one slow planner per worker process."""
    global _FAST_PLANNER, _SLOW_PLANNER, _FAST_PARAMS, _SLOW_PARAMS
    _FAST_PARAMS = fast_params
    _SLOW_PARAMS = slow_params
    _FAST_PLANNER = make_fast_planner(fast_params)
    _SLOW_PLANNER = make_slow_planner(slow_params)


def _run_trial(task):
    """Run both models for one (bars, trial_seed) trial."""
    bars, trial_seed = task
    bars = tuple(map(float, bars))

    fast_result = run_one_fast(_FAST_PLANNER, bars, seed=trial_seed, params=_FAST_PARAMS)
    slow_result = run_one_slow(_SLOW_PLANNER, bars, seed=trial_seed, params=_SLOW_PARAMS)

    return fast_result, slow_result


def main(
    common_params,
    fast_only_params,
    slow_only_params,
    target_props,
    seeds=range(3),
    exhaustive: bool = False,
    sample_step: int = 5,
    random_n_pairs: int | None = 40,
    random_seed: int = 0,
    output_path: str = "validation_results.json",
    parallel: bool = True,
    max_workers: int | None = None,
    chunksize: int = 16,
    seed_stride: int | None = None,
) -> None:
    grid = slow_mod.make_height_grid(common_params["n_ticks"], common_params["height_step"])
    pairs, sampling_meta = choose_bar_pairs(
        grid,
        target_props,
        exhaustive=exhaustive,
        sample_step=sample_step,
        random_n_pairs=None if exhaustive else random_n_pairs,
        random_seed=random_seed,
    )

    fast_params = {**common_params, **fast_only_params}
    slow_params = {**common_params, **slow_only_params}

    seeds = list(seeds)
    if seed_stride is None:
        seed_stride = max(100000, len(pairs) + 1)

    tasks = build_tasks(pairs, seeds, seed_stride=seed_stride)

    print(f"Testing {len(pairs)} bar pairs across {len(seeds)} base seeds...")
    print(f"Sampling mode: {sampling_meta['sampling_mode']}")
    print(f"Trial seed stride: {seed_stride}")
    if "sampled_pair_counts" in sampling_meta:
        print(f"Pair coverage: {sampling_meta['sampled_pair_counts']}")

    results = []

    if parallel:
        if max_workers is None:
            max_workers = max(1, (os.cpu_count() or 1) - 1)

        ctx = mp.get_context("spawn")  # safest on macOS
        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(fast_params, slow_params),
        ) as ex:
            total = len(tasks)
            for i, (fast_res, slow_res) in enumerate(
                ex.map(_run_trial, tasks, chunksize=chunksize), 1
            ):
                results.append(fast_res)
                results.append(slow_res)

                if i % 25 == 0 or i == total:
                    print(f"Completed {i}/{total} trials")
    else:
        fast_planner = make_fast_planner(fast_params)
        slow_planner = make_slow_planner(slow_params)

        total = len(tasks)
        for i, (bars, trial_seed) in enumerate(tasks, 1):
            results.append(run_one_fast(fast_planner, bars, seed=trial_seed, params=fast_params))
            results.append(run_one_slow(slow_planner, bars, seed=trial_seed, params=slow_params))

            if i % 25 == 0 or i == total:
                print(f"Completed {i}/{total} trials")

    summarize(results)
    saved_to = save_results(
        results,
        metadata={
            "common": common_params,
            "fast_only": fast_only_params,
            "slow_only": slow_only_params,
            "base_seeds": seeds,
            "seed_stride": seed_stride,
            "exhaustive": exhaustive,
            "sample_step": sample_step,
            "random_n_pairs": random_n_pairs,
            "random_seed": random_seed,
            "parallel": parallel,
            "max_workers": max_workers,
            "chunksize": chunksize,
            **sampling_meta,
        },
        output_path=HERE / output_path,
    )
    print(f"Saved results to {saved_to}")


if __name__ == "__main__":
    # Default: stratified random sample spanning on-tick and off-tick bar heights.
    # Set exhaustive=True to test every possible bar-height pair on the 0.1 grid.
    COMMON = dict(
        n_ticks=4,
        height_step=0.1,
        policy_len=4,
        gamma=8.0,
        policy_eval_mode="rollout",
        info_gain_target="report",
        report_action_instant=True,
        mem_retention=0.97,
        mem_forget=0.005,
        mem_sigma=0.25,
        non_report_cost=0.35,
        segment_sigma=0.045,
        course_tick_anchor_bias=0.0,
        segment_tick_anchor_bias=0.0,
    )

    FAST_ONLY = dict(
        T=12,
        pair_obs_sigma=0.22,
    )

    SLOW_ONLY = dict(
        T=26,
        bar_obs_sigma=0.00,
    )

    TARGET_PROPORTIONS = {  # Not used in exhaustive mode
        "on_on": 0.10,
        "on_off": 0.30,
        "off_off": 0.60,
    }

    main(
        COMMON,
        FAST_ONLY,
        SLOW_ONLY,
        TARGET_PROPORTIONS,
        seeds=range(10),
        exhaustive=True,
        random_n_pairs=3,
        random_seed=7,
        parallel=True,
        max_workers=None,
        chunksize=16,
        seed_stride=None,
    )