import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


# ------------------------------------------------------------------
# Import model files robustly
# ------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_SEARCH_DIRS = [
    _THIS_DIR,
    Path.cwd(),
    (_THIS_DIR / "../models").resolve(),
]
for _p in _SEARCH_DIRS:
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import fast_1dpTask as fast_mod
import slow_1dpTask as slow_mod


# ------------------------------------------------------------------
# Style
# ------------------------------------------------------------------
IEEE_COL_W = 3.35
TITLE_FS = 8
SUBTITLE_FS = 7
PANEL_TITLE_FS = 7
LABEL_FS = 7
TICK_FS = 6
ANNOT_FS = 6


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _fmt_num(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.1f}"


def _short_action_name_fast(action: int, meta) -> str:
    if action == 0:
        return "PAIR"
    if action == 1:
        return "AXIS"
    if action == 2:
        return "SEG"
    return f"R:{meta.report_names[action - meta.report_start]}"


def _short_action_name_slow(action: int, meta) -> str:
    if action == 0:
        return "B0"
    if action == 1:
        return "B1"
    if action == 2:
        return "AXIS"
    if action == 3:
        return "SEG"
    return f"R:{meta.report_names[action - meta.report_start]}"


def _action_labels(trace: dict) -> list[str]:
    if trace["model"] == "fast":
        return ["start"] + [_short_action_name_fast(a, trace["meta"]) for a in trace["actions"]]
    return ["start"] + [_short_action_name_slow(a, trace["meta"]) for a in trace["actions"]]


def _report_summary_text(trace: dict) -> str:
    reported_value = trace.get("reported_value")
    if reported_value is None:
        return f"true={trace['env'].true_report:.1f} | reported=None | no report"
    status = "correct" if bool(trace.get("correct")) else "incorrect"
    return f"true={trace['env'].true_report:.1f} | reported={reported_value:.1f} | {status}"


def _style_xticklabels(ax, labels: list[str], rotation: int = 90) -> None:
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=rotation, ha="center", fontsize=TICK_FS)
    ax.tick_params(axis="x", pad=1, length=2)


def _sparse_yticks(labels: list[str], max_ticks: int = 8):
    n = len(labels)
    if n <= max_ticks:
        idx = np.arange(n)
    else:
        idx = np.unique(np.linspace(0, n - 1, max_ticks).round().astype(int))
    return idx, [labels[i] for i in idx]


def _plot_heatmap(ax, data: np.ndarray, ylabels: list[str], title: str, cmap: str = "Greys",
                  vmax: float = 1.0, gamma: float = 0.5):
    norm = colors.PowerNorm(gamma=gamma, vmin=0.0, vmax=vmax)
    im = ax.imshow(
        data,
        aspect="auto",
        origin="upper",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )
    yt, yl = _sparse_yticks(ylabels, max_ticks=8)
    ax.set_yticks(yt)
    ax.set_yticklabels(yl, fontsize=TICK_FS)
    if title is not None:
        ax.set_title(title, fontsize=PANEL_TITLE_FS, pad=2)
    ax.set_xlabel("Time", fontsize=LABEL_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS, length=2, pad=1)
    return im


# ------------------------------------------------------------------
# Trace collection
# ------------------------------------------------------------------
def collect_fast_trace(
    bar_heights=(1.2, 3.7),
    seed=1,
    T=12,
    n_ticks=4,
    height_step=0.1,
    policy_len=4,
    gamma=8.0,
    mem_retention=0.97,
    mem_forget=0.005,
    mem_sigma=0.25,
    non_report_cost=0.35,
    pair_obs_sigma=0.22,
    segment_sigma=0.045,
    course_tick_anchor_bias=0.0,
    segment_tick_anchor_bias=0.0,
    policy_eval_mode="rollout",
):
    np.random.seed(seed)

    env = fast_mod.DecimalFastAvgEnv(
        bar_heights=bar_heights,
        n_ticks=n_ticks,
        height_step=height_step,
        pair_obs_sigma=pair_obs_sigma,
        segment_sigma=segment_sigma,
        course_tick_anchor_bias=course_tick_anchor_bias,
        segment_tick_anchor_bias=segment_tick_anchor_bias,
    )

    planner = fast_mod.FastMemoryDecayTenthsPlanner(
        n_ticks=n_ticks,
        height_step=height_step,
        policy_len=policy_len,
        gamma=gamma,
        use_states_info_gain=True,
        action_selection="deterministic",
        non_report_cost=non_report_cost,
        mem_retention=mem_retention,
        mem_forget=mem_forget,
        mem_sigma=mem_sigma,
        pair_obs_sigma=pair_obs_sigma,
        segment_sigma=segment_sigma,
        course_tick_anchor_bias=course_tick_anchor_bias,
        segment_tick_anchor_bias=segment_tick_anchor_bias,
        info_gain_target="report",
        report_action_instant=True,
        policy_eval_mode=policy_eval_mode,
    )

    meta = planner.meta()

    reg_hist, mem_hist, avg_hist = [], [], []
    action_hist, obs_hist = [], []
    reported_value = None
    correct = None

    def snapshot():
        q_reg, q_mem = planner.q_marginals()
        q_avg = planner.q_report_from(planner.q)
        reg_hist.append(q_reg.copy())
        mem_hist.append(q_mem.copy())
        avg_hist.append(q_avg.copy())

    snapshot()
    for _ in range(T):
        q_pi, scores = planner.infer_policies()
        action = planner.sample_action(q_pi)

        if action >= meta.report_start:
            # decision-time snapshot: this is what the model believed when it committed
            action_hist.append(action)

            obs = env.step(action)
            obs_hist.append(obs)

            snapshot()  # pre-feedback belief, labeled by the report action

            reported_idx = action - meta.report_start
            reported_value = float(meta.avg_grid[reported_idx])
            correct = bool(obs[3] == 2)

            # still update internally if you want post-feedback diagnostics later
            planner.update_beliefs(action, obs)
            break

        obs = env.step(action)
        planner.update_beliefs(action, obs)

        action_hist.append(action)
        obs_hist.append(obs)
        snapshot()

    return {
        "model": "fast",
        "meta": meta,
        "env": env,
        "planner": planner,
        "reg": np.stack(reg_hist, axis=1),
        "mem": np.stack(mem_hist, axis=1),
        "avg": np.stack(avg_hist, axis=1),
        "report_grid": planner.avg_grid.copy(),
        "actions": action_hist,
        "obs": obs_hist,
        "reported_value": reported_value,
        "correct": correct,
    }



def collect_slow_trace(
    bar_heights=(1.2, 3.7),
    seed=1,
    T=26,
    n_ticks=4,
    height_step=0.1,
    policy_len=4,
    gamma=8.0,
    mem_retention=0.97,
    mem_forget=0.005,
    mem_sigma=0.25,
    non_report_cost=0.35,
    bar_obs_sigma=0.08,
    segment_sigma=0.045,
    course_tick_anchor_bias=0.0,
    segment_tick_anchor_bias=0.0,
    policy_eval_mode="rollout",
):
    np.random.seed(seed)

    env = slow_mod.DecimalSegmentEnv(
        bar_heights=bar_heights,
        n_ticks=n_ticks,
        height_step=height_step,
        bar_obs_sigma=bar_obs_sigma,
        segment_sigma=segment_sigma,
        course_tick_anchor_bias=course_tick_anchor_bias,
        segment_tick_anchor_bias=segment_tick_anchor_bias,
        info_gain_target="report",
        report_action_instant=True,
    )

    planner = slow_mod.SlowMemoryDecayTenthsPlanner(
        n_ticks=n_ticks,
        height_step=height_step,
        policy_len=policy_len,
        gamma=gamma,
        use_states_info_gain=True,
        action_selection="deterministic",
        non_report_cost=non_report_cost,
        mem_retention=mem_retention,
        mem_forget=mem_forget,
        mem_sigma=mem_sigma,
        bar_obs_sigma=bar_obs_sigma,
        segment_sigma=segment_sigma,
        course_tick_anchor_bias=course_tick_anchor_bias,
        segment_tick_anchor_bias=segment_tick_anchor_bias,
        info_gain_target="report",
        report_action_instant=True,
        policy_eval_mode=policy_eval_mode,
    )

    meta = planner.meta()

    reg_hist, mem0_hist, mem1_hist, avg_hist = [], [], [], []
    action_hist, obs_hist = [], []
    reported_value = None
    correct = None

    def snapshot():
        q_reg, q_mem0, q_mem1 = planner.q_marginals()
        q_avg = planner.q_avg()
        reg_hist.append(q_reg.copy())
        mem0_hist.append(q_mem0.copy())
        mem1_hist.append(q_mem1.copy())
        avg_hist.append(q_avg.copy())

    snapshot()
    for _ in range(T):
        q_pi, scores = planner.infer_policies()
        action = planner.sample_action(q_pi)

        if action >= meta.report_start:
            action_hist.append(action)

            obs = env.step(action)
            obs_hist.append(obs)

            snapshot()  # pre-feedback belief at decision time

            reported_idx = action - meta.report_start
            reported_value = float(planner.report_grid[reported_idx])
            correct = bool(obs[3] == 2)

            planner.update_beliefs(action, obs)
            break

        obs = env.step(action)
        planner.update_beliefs(action, obs)

        action_hist.append(action)
        obs_hist.append(obs)
        snapshot()

    return {
        "model": "slow",
        "meta": meta,
        "env": env,
        "planner": planner,
        "reg": np.stack(reg_hist, axis=1),
        "mem0": np.stack(mem0_hist, axis=1),
        "mem1": np.stack(mem1_hist, axis=1),
        "avg": np.stack(avg_hist, axis=1),
        "report_grid": planner.report_grid.copy(),
        "actions": action_hist,
        "obs": obs_hist,
        "reported_value": reported_value,
        "correct": correct,
    }


# ------------------------------------------------------------------
# Standard dense diagnostics
# ------------------------------------------------------------------
def plot_fast_trace(trace: dict, figsize=(IEEE_COL_W, 4.0), gamma: float = 0.5):
    meta = trace["meta"]
    fig, axes = plt.subplots(
        3, 1,
        figsize=figsize,
        constrained_layout=False,
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.2, 1.45]},
    )

    _plot_heatmap(axes[0], trace["reg"], meta.register_names, "Register", gamma=gamma)
    _plot_heatmap(axes[1], trace["mem"], meta.mem_names, "Avg memory", gamma=gamma)
    _plot_heatmap(axes[2], trace["avg"], meta.report_names, "Report posterior", gamma=gamma)

    _style_xticklabels(axes[-1], _action_labels(trace), rotation=90)
    fig.subplots_adjust(top=0.86, bottom=0.15, left=0.22, right=0.98, hspace=0.65)
    fig.text(0.5, 0.985, "FAST posterior beliefs", ha="center", va="top", fontsize=TITLE_FS)
    fig.text(0.5, 0.955, _report_summary_text(trace), ha="center", va="top", fontsize=SUBTITLE_FS)
    return fig, axes



def plot_slow_trace(trace: dict, include_memory: bool = True, figsize=None, gamma: float = 0.5):
    meta = trace["meta"]

    if figsize is None:
        figsize = (IEEE_COL_W, 5.2 if include_memory else 3.0)

    if include_memory:
        fig, axes = plt.subplots(
            4, 1,
            figsize=figsize,
            constrained_layout=False,
            sharex=True,
            gridspec_kw={"height_ratios": [1.0, 1.25, 1.25, 1.45]},
        )
        _plot_heatmap(axes[0], trace["reg"], meta.register_names, "Register", gamma=gamma)
        _plot_heatmap(axes[1], trace["mem0"], meta.mem_names, "Bar 0 memory", gamma=gamma)
        _plot_heatmap(axes[2], trace["mem1"], meta.mem_names, "Bar 1 memory", gamma=gamma)
        _plot_heatmap(axes[3], trace["avg"], meta.report_names, "Report posterior", gamma=gamma)
        fig.subplots_adjust(top=0.88, bottom=0.13, left=0.22, right=0.98, hspace=0.75)
    else:
        fig, axes = plt.subplots(
            2, 1,
            figsize=figsize,
            constrained_layout=False,
            sharex=True,
            gridspec_kw={"height_ratios": [1.0, 1.45]},
        )
        _plot_heatmap(axes[0], trace["reg"], meta.register_names, "Register", gamma=gamma)
        _plot_heatmap(axes[1], trace["avg"], meta.report_names, "Report posterior", gamma=gamma)
        fig.subplots_adjust(top=0.84, bottom=0.17, left=0.22, right=0.98, hspace=0.6)

    _style_xticklabels(axes[-1], _action_labels(trace), rotation=90)
    fig.text(0.5, 0.985, "SLOW posterior beliefs", ha="center", va="top", fontsize=TITLE_FS)
    fig.text(0.5, 0.955, _report_summary_text(trace), ha="center", va="top", fontsize=SUBTITLE_FS)
    return fig, axes



def plot_fast_vs_slow(fast_trace: dict, slow_trace: dict, figsize=(IEEE_COL_W, 5.0), gamma: float = 0.5):
    fig, axes = plt.subplots(
        4, 1,
        figsize=figsize,
        constrained_layout=False,
        sharex=False,
        gridspec_kw={"height_ratios": [1.0, 1.35, 1.0, 1.35]},
    )

    _plot_heatmap(axes[0], fast_trace["reg"], fast_trace["meta"].register_names, "FAST register", gamma=gamma)
    _plot_heatmap(axes[1], fast_trace["avg"], fast_trace["meta"].report_names, "FAST report", gamma=gamma)
    _plot_heatmap(axes[2], slow_trace["reg"], slow_trace["meta"].register_names, "SLOW register", gamma=gamma)
    _plot_heatmap(axes[3], slow_trace["avg"], slow_trace["meta"].report_names, "SLOW report", gamma=gamma)

    _style_xticklabels(axes[1], _action_labels(fast_trace), rotation=90)
    _style_xticklabels(axes[3], _action_labels(slow_trace), rotation=90)

    fig.subplots_adjust(top=0.88, bottom=0.13, left=0.22, right=0.98, hspace=0.75)
    fig.text(0.5, 0.985, "FAST vs SLOW posterior beliefs", ha="center", va="top", fontsize=TITLE_FS)
    fig.text(
        0.5, 0.955,
        f"FAST: {_report_summary_text(fast_trace)}    SLOW: {_report_summary_text(slow_trace)}",
        ha="center", va="top", fontsize=SUBTITLE_FS,
    )
    return fig, axes


# ------------------------------------------------------------------
# Focused failure-case figure
# ------------------------------------------------------------------
def _report_window_indices(trace: dict, pad: float = 0.2) -> np.ndarray:
    grid = np.asarray(trace["report_grid"], dtype=float)
    true_val = float(trace["env"].true_report)
    reported_val = trace.get("reported_value")
    if reported_val is None:
        lo = hi = true_val
    else:
        lo = min(true_val, float(reported_val))
        hi = max(true_val, float(reported_val))
    lo -= pad
    hi += pad
    idx = np.where((grid >= lo - 1e-9) & (grid <= hi + 1e-9))[0]
    if len(idx) < 5:
        center_idx = int(np.argmin(np.abs(grid - true_val)))
        lo_i = max(0, center_idx - 2)
        hi_i = min(len(grid), center_idx + 3)
        idx = np.arange(lo_i, hi_i)
    return idx



def _posterior_mean(trace: dict) -> np.ndarray:
    grid = np.asarray(trace["report_grid"], dtype=float)
    probs = np.asarray(trace["avg"], dtype=float)
    return (grid[:, None] * probs).sum(axis=0)



def _posterior_map(trace: dict) -> np.ndarray:
    grid = np.asarray(trace["report_grid"], dtype=float)
    probs = np.asarray(trace["avg"], dtype=float)
    return grid[np.argmax(probs, axis=0)]



def _plot_stimulus_panel(ax, fast_trace: dict, slow_trace: dict) -> None:
    bars = np.asarray(fast_trace["env"].bar_heights, dtype=float)
    true_report = float(fast_trace["env"].true_report)
    fast_ans = fast_trace.get("reported_value")
    slow_ans = slow_trace.get("reported_value")

    x = np.array([0.0, 1.0])
    ax.bar(x, bars, width=0.55, color="0.75", edgecolor="0.15", linewidth=0.8)
    ax.axhline(true_report, color="tab:green", linestyle="--", linewidth=1.0)

    for xi, yi in zip(x, bars):
        ax.text(xi, yi + 0.07, _fmt_num(float(yi)), ha="center", va="bottom", fontsize=ANNOT_FS)

    ax.text(
        0.98, 0.76,
        f"target = {_fmt_num(true_report)}",
        color="tab:green",
        transform=ax.transAxes,
        ha="right", va="center", fontsize=ANNOT_FS,
    )

    if fast_ans is not None and slow_ans is not None:
        ax.text(
            0.02, 0.95,
            f"Fast {_fmt_num(float(fast_ans))} {'✗' if not fast_trace['correct'] else '✓'}   |   Slow {_fmt_num(float(slow_ans))} {'✓' if slow_trace['correct'] else '✗'}",
            transform=ax.transAxes,
            ha="left", va="top", fontsize=ANNOT_FS,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="0.7", linewidth=0.6),
        )

    ax.set_title("Stimulus", fontsize=PANEL_TITLE_FS, pad=2)
    ax.set_xlim(-0.6, 2.0)
    ax.set_ylim(0.0, max(3.0, bars.max() + 0.45))
    ax.set_ylabel("Bar height", fontsize=LABEL_FS)
    ax.set_xticks(x)
    ax.set_xticklabels(["Bar 0", "Bar 1"], fontsize=TICK_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS, length=2, pad=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)



def _plot_local_report_panel(ax, trace: dict, model_name: str, pad: float = 0.2, gamma: float = 0.45) -> None:
    idx = _report_window_indices(trace, pad=pad)
    local_grid = np.asarray(trace["report_grid"], dtype=float)[idx]
    local_data = np.asarray(trace["avg"], dtype=float)[idx, :]
    n_time = local_data.shape[1]
    step = float(np.median(np.diff(local_grid))) if len(local_grid) > 1 else 0.1

    norm = colors.PowerNorm(gamma=gamma, vmin=0.0, vmax=1.0)
    ax.imshow(
        local_data,
        origin="lower",
        aspect="auto",
        cmap="Greys",
        norm=norm,
        interpolation="nearest",
        extent=[-0.5, n_time - 0.5, local_grid[0] - step / 2, local_grid[-1] + step / 2],
    )

    true_val = float(trace["env"].true_report)
    local_probs = np.asarray(trace["avg"], dtype=float)[idx, :]
    denom = local_probs.sum(axis=0)
    local_mean = np.full(n_time, np.nan, dtype=float)
    mask = denom > 1e-12
    local_mean[mask] = (local_grid[:, None] * local_probs).sum(axis=0)[mask] / denom[mask]
    if len(local_mean) > 0:
        local_mean[0] = np.nan

    ax.axhline(true_val, color="tab:green", linestyle="--", linewidth=1.0)
    ax.plot(np.arange(n_time), local_mean, color="tab:blue", linewidth=1.2)

    if trace.get("reported_value") is not None:
        ax.plot(
            n_time - 1,
            float(trace["reported_value"]),
            marker="o",
            markersize=4.5,
            markerfacecolor="white",
            markeredgewidth=1.0,
            markeredgecolor="tab:blue",
        )

    status = "correct" if bool(trace.get("correct")) else "incorrect"
    edge = "tab:green" if bool(trace.get("correct")) else "tab:red"
    ax.text(
        0.98, 0.95,
        f"{model_name}: {_fmt_num(float(trace['reported_value']))} | {status}",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=ANNOT_FS,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=edge, linewidth=0.8),
    )

    ax.set_title(f"{model_name} report posterior", fontsize=PANEL_TITLE_FS, pad=2)
    ax.set_ylabel("Report value", fontsize=LABEL_FS)
    ax.set_yticks(local_grid)
    ax.set_yticklabels([_fmt_num(v) for v in local_grid], fontsize=TICK_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS, length=2, pad=1)
    _style_xticklabels(ax, _action_labels(trace), rotation=90)
    ax.set_xlabel("Time", fontsize=LABEL_FS)

def _abbr_obs_name(name: str) -> str:
    """
    Compact display labels for the top observation axis.
    """
    if name in ("Null", "", None):
        return "·"

    mapping = {
        "OnTick": "tick",
        "LowerHalf": "low",
        "Midpoint": "mid",
        "UpperHalf": "high",
        "Correct": "✓",
        "Incorrect": "✗",
    }
    if name in mapping:
        return mapping[name]

    # compact common autogenerated names a bit
    name = str(name)
    name = name.replace("Tick", "T")
    name = name.replace("Lower", "L")
    name = name.replace("Upper", "U")
    name = name.replace("Interval", "I")
    name = name.replace("Segment", "S")
    return name


def _obs_labels(trace: dict) -> list[str]:
    """
    One compact observation label per displayed time column.
    First column is the prior 'start' column, so it gets a blank label.
    """
    meta = trace["meta"]
    labels = [""]

    for action, obs in zip(trace["actions"], trace["obs"]):
        rel_obs, axis_obs, seg_obs, fb_obs = map(int, obs)

        if action >= meta.report_start:
            name = meta.fb_names[fb_obs]
        elif seg_obs > 0:
            name = meta.seg_names[seg_obs]
        elif axis_obs > 0:
            name = meta.axis_names[axis_obs]
        elif rel_obs > 0:
            name = meta.rel_names[rel_obs]
        else:
            name = "Null"

        labels.append(_abbr_obs_name(name))

    return labels


def _add_obs_top_axis(ax, trace: dict, axis_label: str = "Obs.") -> None:
    """
    Add a duplicate x-axis on top showing observation labels.
    """
    top = ax.twiny()
    top.set_xlim(ax.get_xlim())

    obs_labels = _obs_labels(trace)
    top.set_xticks(np.arange(len(obs_labels)))
    top.set_xticklabels(obs_labels, rotation=0, ha="center", fontsize=TICK_FS)
    top.tick_params(axis="x", pad=1, length=2, labelsize=TICK_FS)

    top.set_xlabel(axis_label, fontsize=LABEL_FS, labelpad=4)

    # Keep the top axis visually light
    top.spines["bottom"].set_visible(False)
    top.spines["right"].set_visible(False)
    top.spines["left"].set_visible(False)

def plot_failure_case_fast_vs_slow(
    fast_trace: dict,
    slow_trace: dict,
    figsize=(IEEE_COL_W, 3.8),
    pad: float = 0.2,
    gamma: float = 0.45,
    stimulus_pane=False,
):
    if stimulus_pane:
        nrows = 3
        fast_idx = 1
        slow_idx = 2
        gridspec = {"height_ratios": [1.0, 1.55, 1.55]}
    else:
        nrows = 2
        fast_idx = 0
        slow_idx = 1
        gridspec = {"height_ratios": [1.55, 1.55]}

    fig, axes = plt.subplots(
        nrows, 1,
        figsize=figsize,
        constrained_layout=False,
        gridspec_kw=gridspec,
    )

    if stimulus_pane:
        _plot_stimulus_panel(axes[0], fast_trace, slow_trace)

    _plot_local_report_panel(axes[fast_idx], fast_trace, "Fast", pad=pad, gamma=gamma)
    _plot_local_report_panel(axes[slow_idx], slow_trace, "Slow", pad=pad, gamma=gamma)
    axes[fast_idx].set_title(None)
    axes[slow_idx].set_title(None)
    axes[fast_idx].set_ylabel('Fast report posterior')
    axes[slow_idx].set_ylabel('Slow (effective) report posterior')

    # add top observation traces
    _add_obs_top_axis(axes[fast_idx], fast_trace, axis_label="Obs.")
    _add_obs_top_axis(axes[slow_idx], slow_trace, axis_label=None)

    # a little more room now that each panel has a top axis
    fig.subplots_adjust(
        top=0.82, 
        bottom=0.11, 
        # left=0.22, 
        # right=0.98, 
        hspace=0.62
        )

    # fig.text(0.5, 0.985, "Heuristic failure case", ha="center", va="top", fontsize=TITLE_FS)
    # fig.text(
    #     0.5, 0.962,
    #     "Fast commits one bin low (2.1)\nSlow refines to the correct report (2.2)",
    #     ha="center", va="top", fontsize=SUBTITLE_FS,
    # )

    axes[fast_idx].set_xlabel(None)
    axes[slow_idx].set_xlabel("Action")

    return fig, axes

# def plot_failure_case_fast_vs_slow(
#     fast_trace: dict,
#     slow_trace: dict,
#     figsize=(IEEE_COL_W, 3.8),
#     pad: float = 0.2,
#     gamma: float = 0.45,
#     stimulus_pane = False,
# ):
#     if stimulus_pane:
#         nrows = 3
#         fast_idx = 1
#         slow_idx = 2
#         gridspec = {"height_ratios": [1.0, 1.55, 1.55]}
#     else: 
#         nrows = 2
#         fast_idx = 0
#         slow_idx = 1
#         gridspec = {"height_ratios": [1.55, 1.55]}

#     fig, axes = plt.subplots(
#         nrows, 1,
#         figsize=figsize,
#         constrained_layout=False,
#         gridspec_kw=gridspec,
#     )

#     if stimulus_pane:
#         _plot_stimulus_panel(axes[0], fast_trace, slow_trace)
#     _plot_local_report_panel(axes[fast_idx], fast_trace, "Fast", pad=pad, gamma=gamma)
#     _plot_local_report_panel(axes[slow_idx], slow_trace, "Slow", pad=pad, gamma=gamma)

#     # fig.subplots_adjust(top=0.90, bottom=0.11, left=0.22, right=0.98, hspace=0.72)
#     fig.subplots_adjust(top=0.8, hspace=0.4)
#     fig.text(0.5, 0.985, "Heuristic failure case", ha="center", va="top", fontsize=TITLE_FS)
#     fig.text(
#         0.5, 0.955,
#         "Bar 0 height = 2.4\n"\
#         "Bar 1 height = 1.9\n"\
#         "Fast commits one bin low (2.1) \n Slow refines to the correct report (2.2)",
#         ha="center", va="top", fontsize=SUBTITLE_FS,
#     )
#     axes[fast_idx].set_xlabel('')

#     return fig, axes


# ------------------------------------------------------------------
# Saving and demo
# ------------------------------------------------------------------
def save_figure(fig: plt.Figure, basepath: Path, dpi: int = 600) -> None:
    fig.savefig(basepath.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(basepath.with_suffix(".pdf"), bbox_inches="tight")


if __name__ == "__main__":
    COMMON = dict(
        bar_heights=(2.4, 1.9),
        seed=2,
        n_ticks=4,
        height_step=0.1,
        policy_len=4,
        gamma=8.0,
        policy_eval_mode="rollout",
        mem_retention=0.97,
        mem_forget=0.005,
        mem_sigma=0.5,
        non_report_cost=0.35,
        segment_sigma=0.045,
        course_tick_anchor_bias=0.0,
        segment_tick_anchor_bias=0.0,
    )

    FAST_ONLY = dict(T=7, pair_obs_sigma=0.22)
    SLOW_ONLY = dict(T=10, bar_obs_sigma=0.0)

    fast_trace = collect_fast_trace(**{**COMMON, **FAST_ONLY})
    slow_trace = collect_slow_trace(**{**COMMON, **SLOW_ONLY})

    # fig, _ = plot_fast_trace(fast_trace, gamma=0.5)
    # save_figure(fig, Path("fast_trace"))
    # plt.close(fig)

    # fig, _ = plot_slow_trace(slow_trace, include_memory=True, gamma=0.5)
    # save_figure(fig, Path("slow_trace"))
    # plt.close(fig)

    # fig, _ = plot_fast_vs_slow(fast_trace, slow_trace, gamma=0.5)
    # save_figure(fig, Path("fast_vs_slow_trace"))
    # plt.close(fig)

    fig, _ = plot_failure_case_fast_vs_slow(fast_trace, slow_trace, gamma=0.45)
    save_figure(fig, Path("failure_case_fast_vs_slow"))
    plt.close(fig)
