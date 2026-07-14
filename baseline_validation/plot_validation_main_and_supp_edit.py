from __future__ import annotations

"""Plot main-text and supplement figures from validation_results.json.

This script is designed for the JSON produced by validate_fast_slow_baseline_rng.py.
It reads the saved trial-level results, makes one compact main-text figure and one
larger supplement figure, and exports a shortlist of candidate Fast>Slow failure
cases for separate cognitive-trace plots.

Outputs
-------
- main_text_validation.(png|pdf)
- supplement_validation.(png|pdf)
- summary_by_model.csv
- summary_by_category.csv
- candidate_failure_cases.csv

Usage
-----
python plot_validation_main_and_supp.py validation_results.json --outdir figs
"""

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


SLOW_COLOR = '#9A3C6A'
FAST_COLOR = '#4F8A3A'

# -----------------------------------------------------------------------------
# IO + preprocessing
# -----------------------------------------------------------------------------


def load_results(path: str | Path) -> tuple[pd.DataFrame, dict]:
    path = Path(path)
    payload = json.loads(path.read_text())
    meta = payload.get("metadata", {})
    rows = payload.get("results", [])
    if not rows:
        raise ValueError(f"No results found in {path}")

    df = pd.DataFrame(rows)
    required = {"model", "bar0", "bar1", "true_avg", "reported_avg", "correct", "steps", "seed"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in results JSON: {sorted(missing)}")

    df["reported_avg"] = pd.to_numeric(df["reported_avg"], errors="coerce")
    df["true_avg"] = pd.to_numeric(df["true_avg"], errors="coerce")
    df["bar0"] = pd.to_numeric(df["bar0"], errors="coerce")
    df["bar1"] = pd.to_numeric(df["bar1"], errors="coerce")
    df["steps"] = pd.to_numeric(df["steps"], errors="coerce")
    df["correct"] = df["correct"].astype(bool)
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce").astype(int)
    df["abs_error"] = (df["reported_avg"] - df["true_avg"]).abs()
    df["signed_error"] = df["reported_avg"] - df["true_avg"]
    df["reported"] = df["reported_avg"].notna()

    # Reconstruct the same on-tick vs off-tick split used in the validation script.
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

    # Also make a permutation-invariant pair label for candidate-case ranking.
    pair_lo = np.minimum(df["bar0"], df["bar1"])
    pair_hi = np.maximum(df["bar0"], df["bar1"])
    df["pair_key"] = [f"({a:.1f}, {b:.1f})" for a, b in zip(pair_lo, pair_hi)]
    df["pair_lo"] = pair_lo
    df["pair_hi"] = pair_hi

    return df, meta


# -----------------------------------------------------------------------------
# Statistics helpers
# -----------------------------------------------------------------------------


def bootstrap_mean_ci(values: Iterable[float], n_boot: int = 4000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(arr))
    if arr.size == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot = arr[idx].mean(axis=1)
    lo = float(np.quantile(boot, alpha / 2.0))
    hi = float(np.quantile(boot, 1.0 - alpha / 2.0))
    return mean, lo, hi


# -----------------------------------------------------------------------------
# Summary tables
# -----------------------------------------------------------------------------


def summarize_by_model(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in ["fast", "slow"]:
        sub = df[df["model"] == model].copy()
        acc, acc_lo, acc_hi = bootstrap_mean_ci(sub["correct"].astype(float), seed=11)
        mae, mae_lo, mae_hi = bootstrap_mean_ci(sub.loc[sub["reported"], "abs_error"], seed=12)
        report_rate, rep_lo, rep_hi = bootstrap_mean_ci(sub["reported"].astype(float), seed=13)
        steps, steps_lo, steps_hi = bootstrap_mean_ci(sub["steps"], seed=14)
        rows.append(
            {
                "model": model,
                "n_trials": len(sub),
                "accuracy": acc,
                "accuracy_lo": acc_lo,
                "accuracy_hi": acc_hi,
                "mae": mae,
                "mae_lo": mae_lo,
                "mae_hi": mae_hi,
                "report_rate": report_rate,
                "report_rate_lo": rep_lo,
                "report_rate_hi": rep_hi,
                "mean_steps": steps,
                "mean_steps_lo": steps_lo,
                "mean_steps_hi": steps_hi,
            }
        )
    return pd.DataFrame(rows)


def summarize_by_category(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    order = ["on_on", "on_off", "off_off"]
    for model in ["fast", "slow"]:
        for cat in order:
            sub = df[(df["model"] == model) & (df["pair_category"] == cat)].copy()
            if len(sub) == 0:
                continue
            acc, acc_lo, acc_hi = bootstrap_mean_ci(sub["correct"].astype(float), seed=21)
            mae, mae_lo, mae_hi = bootstrap_mean_ci(sub.loc[sub["reported"], "abs_error"], seed=22)
            report_rate, rep_lo, rep_hi = bootstrap_mean_ci(sub["reported"].astype(float), seed=23)
            steps, steps_lo, steps_hi = bootstrap_mean_ci(sub["steps"], seed=24)
            rows.append(
                {
                    "model": model,
                    "pair_category": cat,
                    "n_trials": len(sub),
                    "accuracy": acc,
                    "accuracy_lo": acc_lo,
                    "accuracy_hi": acc_hi,
                    "mae": mae,
                    "mae_lo": mae_lo,
                    "mae_hi": mae_hi,
                    "report_rate": report_rate,
                    "report_rate_lo": rep_lo,
                    "report_rate_hi": rep_hi,
                    "mean_steps": steps,
                    "mean_steps_lo": steps_lo,
                    "mean_steps_hi": steps_hi,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["pair_category"] = pd.Categorical(out["pair_category"], ["on_on", "on_off", "off_off"], ordered=True)
        out = out.sort_values(["pair_category", "model"]).reset_index(drop=True)
    return out


def summarize_by_pair(df: pd.DataFrame) -> pd.DataFrame:
    grp = (
        df.groupby(["model", "bar0", "bar1", "pair_key", "pair_lo", "pair_hi", "pair_category"], as_index=False)
        .agg(
            n_trials=("correct", "size"),
            failure_rate=("correct", lambda s: 1.0 - float(np.mean(s.astype(float)))),
            accuracy=("correct", "mean"),
            mae=("abs_error", "mean"),
            signed_error=("signed_error", "mean"),
            report_rate=("reported", "mean"),
            mean_steps=("steps", "mean"),
            true_avg=("true_avg", "mean"),
            reported_avg=("reported_avg", "mean"),
        )
        .copy()
    )
    return grp


def candidate_failure_cases(df: pd.DataFrame) -> pd.DataFrame:
    pair = (
        df.groupby(["model", "pair_key", "pair_lo", "pair_hi", "pair_category"], as_index=False)
        .agg(
            n_trials=("correct", "size"),
            failure_rate=("correct", lambda s: 1.0 - float(np.mean(s.astype(float)))),
            accuracy=("correct", "mean"),
            mae=("abs_error", "mean"),
            mean_steps=("steps", "mean"),
            true_avg=("true_avg", "mean"),
            reported_avg=("reported_avg", "mean"),
        )
    )

    wide = pair.pivot_table(
        index=["pair_key", "pair_lo", "pair_hi", "pair_category"],
        columns="model",
        values=["n_trials", "failure_rate", "accuracy", "mae", "mean_steps", "true_avg", "reported_avg"],
    )
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()

    for col in ["failure_rate_fast", "failure_rate_slow", "mae_fast", "mae_slow", "accuracy_fast", "accuracy_slow"]:
        if col not in wide:
            wide[col] = np.nan

    wide["failure_rate_diff_fast_minus_slow"] = wide["failure_rate_fast"] - wide["failure_rate_slow"]
    wide["mae_diff_fast_minus_slow"] = wide["mae_fast"] - wide["mae_slow"]

    wide = wide.sort_values(
        ["failure_rate_diff_fast_minus_slow", "mae_diff_fast_minus_slow", "failure_rate_fast"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return wide


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------


def save_figure(fig: plt.Figure, basepath: Path, dpi: int = 600) -> None:
    fig.savefig(basepath.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(basepath.with_suffix(".pdf"), bbox_inches="tight")


def draw_point_ci(ax, x, mean, lo, hi, label=None):
    ax.errorbar(
        x,
        mean,
        yerr=np.array([[mean - lo], [hi - mean]]),
        fmt="o",
        capsize=4,
        label=label,
    )


def make_matrix(pair_df: pd.DataFrame, value_col: str, heights: np.ndarray) -> np.ndarray:
    heights = np.asarray(sorted(np.unique(heights)))
    idx = {float(h): i for i, h in enumerate(heights)}
    M = np.full((len(heights), len(heights)), np.nan)
    for _, row in pair_df.iterrows():
        i = idx[float(row["bar0"])]
        j = idx[float(row["bar1"])]
        M[i, j] = float(row[value_col])
    return M


def plot_main_text_figure(df: pd.DataFrame, outpath: Path, title: str | None = None) -> None:
    by_model = summarize_by_model(df)
    by_cat = summarize_by_category(df)
    by_pair = summarize_by_pair(df)
    heights = np.sort(np.unique(np.r_[df["bar0"].values, df["bar1"].values]))

    fast_pair = by_pair[by_pair["model"] == "fast"].copy()
    slow_pair = by_pair[by_pair["model"] == "slow"].copy()
    merged = fast_pair.merge(slow_pair, on=["bar0", "bar1"], suffixes=("_fast", "_slow"))
    merged["failure_rate_diff"] = merged["failure_rate_fast"] - merged["failure_rate_slow"]
    diff_matrix = make_matrix(
        merged.rename(columns={"failure_rate_diff": "value"}),
        "value",
        heights,
    )

    # ---------- single-column layout ----------
    # ~3.35 in is a good IEEE-ish single-column width
    fig, (ax_acc, ax_heat) = plt.subplots(
        2,
        1,
        figsize=(3.35, 4.3),
        gridspec_kw={"height_ratios": [0.75, 1.45]},
        constrained_layout=True,
    )

    # -------------------------
    # Panel A: compact accuracy plot
    # -------------------------
    order = ["fast", "slow"]
    x_labels = ["all", "on/on", "on/off", "off/off"]
    x = np.arange(len(x_labels))
    cat_order = ["on_on", "on_off", "off_off"]

    for model in order:
        if model == "fast":
            _color = FAST_COLOR
        else:
            _color = SLOW_COLOR

        row_overall = by_model[by_model["model"] == model].iloc[0]
        sub = (
            by_cat[by_cat["model"] == model]
            .set_index("pair_category")
            .reindex(cat_order)
        )

        means = np.r_[row_overall["accuracy"], sub["accuracy"].to_numpy()]
        los = np.r_[row_overall["accuracy_lo"], sub["accuracy_lo"].to_numpy()]
        his = np.r_[row_overall["accuracy_hi"], sub["accuracy_hi"].to_numpy()]

        ax_acc.errorbar(
            x,
            means,
            yerr=np.vstack([means - los, his - means]),
            fmt="o-",
            capsize=2.5,
            linewidth=1.2,
            markersize=3.8,
            label=model.capitalize(),
            color=_color
        )

    ax_acc.set_xticks(x, x_labels)
    ax_acc.set_ylim(0.875, 1.01)
    ax_acc.set_ylabel("Accuracy", fontsize=8)
    ax_acc.set_title("A)  Validation accuracy", loc="left", fontsize=9, pad=2)
    ax_acc.yaxis.set_major_locator(MaxNLocator(5))
    ax_acc.grid(alpha=0.25, linewidth=0.5)
    ax_acc.tick_params(axis="both", labelsize=7)
    ax_acc.legend(
        frameon=False,
        fontsize=7,
        ncol=2,
        loc="lower left",
        handlelength=1.6,
        columnspacing=1.0,
        borderaxespad=0.2,
    )

    # -------------------------
    # Panel B: failure-rate difference heatmap
    # -------------------------
    vmax = float(np.nanmax(np.abs(diff_matrix))) if np.isfinite(diff_matrix).any() else 1.0
    if vmax == 0:
        vmax = 1.0

    cmap = LinearSegmentedColormap.from_list(
        "blue_white_red",
        # ["#2166ac", "#ffffff", "#b2182b"],
        [SLOW_COLOR, "#ffffff", FAST_COLOR],
        N=256,
    )
    # cmap = 
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    im = ax_heat.imshow(
        diff_matrix,
        origin="lower",
        aspect="equal",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )

    # show only a subset of tick labels so the axis stays readable in one column
    max_ticks = 7
    step = max(1, int(np.ceil(len(heights) / max_ticks)))
    tick_idx = np.arange(0, len(heights), step)
    if tick_idx[-1] != len(heights) - 1:
        tick_idx = np.r_[tick_idx, len(heights) - 1]

    tick_labels = [f"{heights[i]:.1f}" for i in tick_idx]

    ax_heat.set_xticks(tick_idx)
    ax_heat.set_xticklabels(tick_labels, rotation=0)
    ax_heat.set_yticks(tick_idx)
    ax_heat.set_yticklabels(tick_labels)

    ax_heat.set_xlabel("bar1", fontsize=8)
    ax_heat.set_ylabel("bar0", fontsize=8)
    ax_heat.set_title("B)  Failure-rate difference (Fast - Slow)", loc="left", fontsize=9, pad=2)
    ax_heat.tick_params(axis="both", labelsize=7, length=2)

    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.02)
    cbar.set_label("Diff. in failure rate", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    if title:
        fig.suptitle(title, fontsize=10)

    save_figure(fig, outpath)
    plt.close(fig)


def plot_supplement_figure(df: pd.DataFrame, outpath: Path, title: str | None = None) -> None:
    by_pair = summarize_by_pair(df)
    heights = np.sort(np.unique(np.r_[df["bar0"].values, df["bar1"].values]))

    fast_pair = by_pair[by_pair["model"] == "fast"].copy()
    slow_pair = by_pair[by_pair["model"] == "slow"].copy()
    fast_fail = make_matrix(fast_pair, "failure_rate", heights)
    slow_fail = make_matrix(slow_pair, "failure_rate", heights)
    fast_mae = make_matrix(fast_pair, "mae", heights)
    slow_mae = make_matrix(slow_pair, "mae", heights)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.ravel()

    # Failure heatmaps
    im1 = ax1.imshow(fast_fail, origin="lower", aspect="auto", vmin=0.0, vmax=np.nanmax([fast_fail, slow_fail]))
    ax1.set_title("A. Fast failure rate")
    im2 = ax2.imshow(slow_fail, origin="lower", aspect="auto", vmin=0.0, vmax=np.nanmax([fast_fail, slow_fail]))
    ax2.set_title("B. Slow failure rate")

    # MAE heatmaps
    mae_vmax = float(np.nanmax([fast_mae, slow_mae])) if np.isfinite(np.nanmax([fast_mae, slow_mae])) else 1.0
    im3 = ax3.imshow(fast_mae, origin="lower", aspect="auto", vmin=0.0, vmax=mae_vmax)
    ax3.set_title("C. Fast MAE")
    im4 = ax4.imshow(slow_mae, origin="lower", aspect="auto", vmin=0.0, vmax=mae_vmax)
    ax4.set_title("D. Slow MAE")

    # Step distributions
    step_bins = np.arange(df["steps"].min() - 0.5, df["steps"].max() + 1.5, 1.0)
    for model in ["fast", "slow"]:
        sub = df[df["model"] == model]
        ax5.hist(sub["steps"], bins=step_bins, alpha=0.6, density=True, label=model.capitalize())
    ax5.set_xlabel("Steps to report")
    ax5.set_ylabel("Density")
    ax5.set_title("E. Step-count distribution")
    ax5.legend(frameon=False)
    ax5.grid(alpha=0.25)

    # Signed-error by category
    cat_order = ["on_on", "on_off", "off_off"]
    base = np.arange(len(cat_order))
    offsets = {"fast": -0.15, "slow": 0.15}
    width = 0.25
    for model in ["fast", "slow"]:
        vals = []
        for cat in cat_order:
            sub = df[(df["model"] == model) & (df["pair_category"] == cat) & (df["reported"])]
            vals.append(float(sub["signed_error"].mean()) if len(sub) else np.nan)
        ax6.bar(base + offsets[model], vals, width=width, label=model.capitalize())
    ax6.axhline(0.0, linewidth=1.0)
    ax6.set_xticks(base, ["on/on", "on/off", "off/off"])
    ax6.set_ylabel("Mean signed error")
    ax6.set_title("F. Mean signed error by class")
    ax6.legend(frameon=False)
    ax6.grid(alpha=0.25)

    # Shared axis labels for heatmaps
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xticks(np.arange(len(heights)))
        ax.set_xticklabels([f"{h:.1f}" for h in heights], rotation=90)
        ax.set_yticks(np.arange(len(heights)))
        ax.set_yticklabels([f"{h:.1f}" for h in heights])
        ax.set_xlabel("Bar 1")
        ax.set_ylabel("Bar 0")

    cbar1 = fig.colorbar(im2, ax=[ax1, ax2], fraction=0.022, pad=0.02)
    cbar1.set_label("Failure rate")
    cbar2 = fig.colorbar(im4, ax=[ax3, ax4], fraction=0.022, pad=0.02)
    cbar2.set_label("MAE")

    if title:
        fig.suptitle(title, fontsize=12)

    save_figure(fig, outpath)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_json", type=Path, help="Path to validation_results.json")
    parser.add_argument("--outdir", type=Path, default=Path("validation_figs"), help="Directory for output figures/tables")
    parser.add_argument("--title", type=str, default=None, help="Optional figure suptitle")
    args = parser.parse_args()

    df, meta = load_results(args.results_json)
    args.outdir.mkdir(parents=True, exist_ok=True)

    by_model = summarize_by_model(df)
    by_category = summarize_by_category(df)
    candidates = candidate_failure_cases(df)

    by_model.to_csv(args.outdir / "summary_by_model.csv", index=False)
    by_category.to_csv(args.outdir / "summary_by_category.csv", index=False)
    candidates.to_csv(args.outdir / "candidate_failure_cases.csv", index=False)

    plot_main_text_figure(df, args.outdir / "main_text_validation", title=args.title)
    plot_supplement_figure(df, args.outdir / "supplement_validation", title=args.title)

    print("Saved:")
    print(f"  {args.outdir / 'main_text_validation.png'}")
    print(f"  {args.outdir / 'main_text_validation.pdf'}")
    print(f"  {args.outdir / 'supplement_validation.png'}")
    print(f"  {args.outdir / 'supplement_validation.pdf'}")
    print(f"  {args.outdir / 'summary_by_model.csv'}")
    print(f"  {args.outdir / 'summary_by_category.csv'}")
    print(f"  {args.outdir / 'candidate_failure_cases.csv'}")

    print("\nTop candidate Fast>Slow failure cases:")
    cols = [
        "pair_key",
        "pair_category",
        "failure_rate_fast",
        "failure_rate_slow",
        "failure_rate_diff_fast_minus_slow",
        "mae_fast",
        "mae_slow",
    ]
    show = candidates.loc[:, [c for c in cols if c in candidates.columns]].head(10)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(show.to_string(index=False))

    if meta:
        with open(args.outdir / "copied_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
