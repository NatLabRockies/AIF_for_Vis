from __future__ import annotations

"""Make box-and-whisker sweep plots for Fast/Slow model experiments.

This is intended for the tick-bias and memory-decay sweeps you have been
running for the 1-decimal-place Fast and Slow models. The script is designed to
be forgiving about input format:

- a single JSON file with a top-level {"results": [...], "metadata": {...}}
- a CSV file
- a directory containing multiple JSON/CSV sweep outputs
- multiple files/directories passed at once

The script tries to infer the sweep parameter from the data, but you can and
should override it explicitly with --x-col for reliable behavior.

Typical usage
-------------
Tick-bias sweep
    python plot_sweep_boxplots.py tick_bias_results.json \
        --x-col course_tick_anchor_bias --outdir figs_tick_box

Memory-decay sweep
    python plot_sweep_boxplots.py memory_decay_results.json \
        --x-col mem_sigma --outdir figs_mem_box

Directory of one-file-per-sweep-value results
    python plot_sweep_boxplots.py ./sweep_outputs \
        --x-col mem_sigma --outdir figs_mem_box

What is plotted
---------------
By default, the script first aggregates trial-level results *within each seed*,
then makes side-by-side Fast/Slow boxplots over those seed-level summaries at
all sweep values.

Panel 1: seed-level accuracy
Panel 2: seed-level mean absolute error (MAE)
Panel 3: seed-level mean number of steps (optional; enabled by default)

Outliers are shown using matplotlib's standard flier markers.
"""

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


TRIAL_LIKE_COLUMNS = {
    "model",
    "seed",
    "trial_seed",
    "base_seed",
    "bar0",
    "bar1",
    "true_avg",
    "reported_avg",
    "correct",
    "steps",
    "abs_error",
    "signed_error",
    "source_file",
    "filename",
    "path",
}

PREFERRED_SWEEP_COLUMNS = [
    "course_tick_anchor_bias",
    "segment_tick_anchor_bias",
    "mem_sigma",
    "mem_retention",
    "mem_forget",
    "gamma",
    "non_report_cost",
    "sweep_value",
    "param_value",
    "value",
]

MODEL_ORDER = ["fast", "slow"]
MODEL_LABELS = {"fast": "Fast", "slow": "Slow"}
MODEL_COLORS = {"fast": "0.25", "slow": "0.65"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+", help="JSON/CSV files or directories of sweep outputs")
    p.add_argument("--x-col", default=None, help="Column to use on the x-axis, e.g. course_tick_anchor_bias or mem_sigma")
    p.add_argument("--group-col", default="seed", help="Column defining repeated runs for the boxplots; default: seed")
    p.add_argument(
        "--metrics",
        nargs="+",
        default=["accuracy", "mae", "steps"],
        choices=["accuracy", "mae", "steps"],
        help="Panels to plot",
    )
    p.add_argument("--title", default=None, help="Optional figure title")
    p.add_argument("--xlabel", default=None, help="Optional x-axis label override")
    p.add_argument("--outdir", default="figs_boxplots", help="Output directory")
    p.add_argument("--basename", default=None, help="Output file base name; default derived from x-col")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--no-showfliers", action="store_true", help="Hide outliers")
    p.add_argument(
        "--row-level",
        action="store_true",
        help="Plot one box per row instead of aggregating within group-col first",
    )
    return p.parse_args()


def iter_input_files(items: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    for item in items:
        p = Path(item).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        if p.is_dir():
            out.extend(sorted(q for q in p.rglob("*") if q.suffix.lower() in {".json", ".csv"}))
        elif p.suffix.lower() in {".json", ".csv"}:
            out.append(p)
    if not out:
        raise ValueError("No JSON or CSV files found in the provided inputs")
    return out


def _flatten_scalars(obj, prefix: str = "") -> dict[str, object]:
    flat: dict[str, object] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_prefix = f"{prefix}.{k}" if prefix else str(k)
            flat.update(_flatten_scalars(v, new_prefix))
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        flat[prefix] = obj
    return flat


def load_one_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
        df["source_file"] = path.name
        return df

    payload = json.loads(path.read_text())

    if isinstance(payload, dict) and "results" in payload:
        rows = payload.get("results", [])
        df = pd.DataFrame(rows)
        meta = payload.get("metadata", {})
        flat_meta = _flatten_scalars(meta)
        top_scalar_keys = {
            k: v
            for k, v in payload.items()
            if k not in {"results", "metadata"} and isinstance(v, (str, int, float, bool))
        }
        for k, v in {**flat_meta, **top_scalar_keys}.items():
            if k not in df.columns:
                df[k] = v
        df["source_file"] = path.name
        return df

    if isinstance(payload, list):
        df = pd.DataFrame(payload)
        df["source_file"] = path.name
        return df

    if isinstance(payload, dict):
        df = pd.DataFrame([payload])
        df["source_file"] = path.name
        return df

    raise ValueError(f"Could not parse supported rows from {path}")


def coerce_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    rename_map = {}
    if "trial_seed" in df.columns and "seed" not in df.columns:
        rename_map["trial_seed"] = "seed"
    if "base_seed" in df.columns and "seed" not in df.columns:
        rename_map["base_seed"] = "seed"
    if rename_map:
        df = df.rename(columns=rename_map)

    numeric_hint_cols = [
        "seed",
        "bar0",
        "bar1",
        "true_avg",
        "reported_avg",
        "steps",
        "abs_error",
        "signed_error",
    ] + PREFERRED_SWEEP_COLUMNS

    for col in df.columns:
        if col in numeric_hint_cols or col.startswith(("common.", "fast_only.", "slow_only.")):
            df[col] = pd.to_numeric(df[col], errors="ignore")

    if "model" in df.columns:
        df["model"] = df["model"].astype(str).str.strip().str.lower()

    if "correct" in df.columns:
        if pd.api.types.is_bool_dtype(df["correct"]):
            pass
        else:
            normalized = df["correct"].astype(str).str.strip().str.lower()
            df["correct"] = normalized.map(
                {"true": True, "false": False, "1": True, "0": False}
            ).fillna(df["correct"])
            if not pd.api.types.is_bool_dtype(df["correct"]):
                df["correct"] = pd.to_numeric(df["correct"], errors="coerce").astype(float) > 0.5

    if "abs_error" not in df.columns and {"reported_avg", "true_avg"}.issubset(df.columns):
        df["reported_avg"] = pd.to_numeric(df["reported_avg"], errors="coerce")
        df["true_avg"] = pd.to_numeric(df["true_avg"], errors="coerce")
        df["abs_error"] = (df["reported_avg"] - df["true_avg"]).abs()

    return df


def choose_x_col(df: pd.DataFrame, requested: str | None) -> str:
    if requested is not None:
        if requested not in df.columns:
            raise ValueError(
                f"Requested --x-col '{requested}' not found. Available columns:\n"
                + ", ".join(df.columns)
            )
        return requested

    for col in PREFERRED_SWEEP_COLUMNS:
        if col in df.columns and df[col].dropna().nunique() > 1:
            return col

    numeric_candidates: list[str] = []
    for col in df.columns:
        if col in TRIAL_LIKE_COLUMNS:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        nunique = series.dropna().nunique()
        if nunique > 1:
            numeric_candidates.append(col)

    if len(numeric_candidates) == 1:
        return numeric_candidates[0]
    if len(numeric_candidates) > 1:
        raise ValueError(
            "Could not uniquely infer the sweep column. Please pass --x-col. "
            f"Candidate varying numeric columns: {numeric_candidates}"
        )

    raise ValueError("Could not infer a varying sweep column. Please pass --x-col explicitly.")


def prepare_boxplot_frame(df: pd.DataFrame, x_col: str, group_col: str, row_level: bool) -> pd.DataFrame:
    df = df.copy()

    if "model" not in df.columns:
        raise ValueError("Input data must contain a 'model' column")

    metrics_present = []
    if "correct" in df.columns:
        metrics_present.append("accuracy")
    if "abs_error" in df.columns:
        metrics_present.append("mae")
    if "steps" in df.columns:
        metrics_present.append("steps")

    if not metrics_present:
        for needed in ("accuracy", "mae", "steps"):
            if needed in df.columns:
                metrics_present.append(needed)

    if not metrics_present:
        raise ValueError(
            "Need trial-level columns like 'correct'/'abs_error'/'steps' or pre-aggregated "
            "columns like 'accuracy'/'mae'/'steps'."
        )

    if row_level or group_col not in df.columns:
        out = df[[c for c in df.columns if c in {x_col, "model", group_col, "accuracy", "mae", "steps", "correct", "abs_error"}]].copy()
        if "accuracy" not in out.columns and "correct" in out.columns:
            out["accuracy"] = out["correct"].astype(float)
        if "mae" not in out.columns and "abs_error" in out.columns:
            out["mae"] = out["abs_error"].astype(float)
        if group_col not in out.columns:
            out[group_col] = np.arange(len(out))
        return out

    agg_map = {}
    if "correct" in df.columns:
        agg_map["accuracy"] = ("correct", lambda s: float(np.mean(s.astype(float))))
    elif "accuracy" in df.columns:
        agg_map["accuracy"] = ("accuracy", "mean")

    if "abs_error" in df.columns:
        agg_map["mae"] = ("abs_error", "mean")
    elif "mae" in df.columns:
        agg_map["mae"] = ("mae", "mean")

    if "steps" in df.columns:
        agg_map["steps"] = ("steps", "mean")

    grouped = (
        df.groupby([x_col, "model", group_col], dropna=False)
        .agg(**agg_map)
        .reset_index()
        .sort_values([x_col, "model", group_col])
    )
    return grouped


def _nice_label(v) -> str:
    if pd.isna(v):
        return "NA"
    if isinstance(v, (float, np.floating)):
        return f"{float(v):g}"
    return str(v)


def _metric_label(metric: str) -> str:
    return {
        "accuracy": "Accuracy",
        "mae": "Mean absolute error",
        "steps": "Mean steps",
    }[metric]


def _make_boxplots(ax, data: pd.DataFrame, x_col: str, metric: str, showfliers: bool) -> None:
    x_vals = sorted(data[x_col].dropna().unique())
    if not x_vals:
        raise ValueError(f"No values available for x column '{x_col}'")

    x_index = {x: i for i, x in enumerate(x_vals)}
    centers = np.arange(len(x_vals), dtype=float)
    offsets = {"fast": -0.18, "slow": 0.18}
    width = 0.30

    for model in MODEL_ORDER:
        sub = data[data["model"] == model]
        if sub.empty or metric not in sub.columns:
            continue

        pos_list = []
        val_list = []
        median_x = []
        median_y = []

        for x in x_vals:
            vals = pd.to_numeric(sub.loc[sub[x_col] == x, metric], errors="coerce").dropna().to_numpy()
            if vals.size == 0:
                continue
            pos = centers[x_index[x]] + offsets[model]
            pos_list.append(pos)
            val_list.append(vals)
            median_x.append(pos)
            median_y.append(float(np.median(vals)))

        if not val_list:
            continue

        ax.boxplot(
            val_list,
            positions=pos_list,
            widths=width,
            patch_artist=True,
            showfliers=showfliers,
            boxprops=dict(facecolor=MODEL_COLORS[model], edgecolor="black", linewidth=1.0, alpha=0.55),
            whiskerprops=dict(color="black", linewidth=1.0),
            capprops=dict(color="black", linewidth=1.0),
            medianprops=dict(color="black", linewidth=1.4),
            flierprops=dict(marker="o", markersize=4, markerfacecolor=MODEL_COLORS[model], markeredgecolor="black", alpha=0.85),
        )
        ax.plot(median_x, median_y, linestyle="-", linewidth=1.0, color="black", alpha=0.7)

    ax.set_xticks(centers)
    ax.set_xticklabels([_nice_label(x) for x in x_vals])
    ax.set_ylabel(_metric_label(metric))
    ax.grid(axis="y", alpha=0.25)

    if metric == "accuracy":
        ax.set_ylim(-0.02, 1.02)


def plot_boxplot_figure(
    plot_df: pd.DataFrame,
    x_col: str,
    metrics: list[str],
    outdir: Path,
    basename: str,
    title: str | None,
    xlabel: str | None,
    dpi: int,
    showfliers: bool,
) -> tuple[Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)

    metrics = [m for m in metrics if m in plot_df.columns]
    if not metrics:
        raise ValueError("None of the requested metrics were available after aggregation")

    n_x = plot_df[x_col].dropna().nunique()
    fig_w = max(6.8, 0.58 * n_x + 2.8)
    fig_h = 2.5 * len(metrics) + (0.45 if title else 0.0)

    fig, axes = plt.subplots(len(metrics), 1, figsize=(fig_w, fig_h), sharex=True, constrained_layout=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        _make_boxplots(ax, plot_df, x_col, metric, showfliers=showfliers)

    axes[-1].set_xlabel(xlabel or x_col)

    legend_handles = [
        Patch(facecolor=MODEL_COLORS[m], edgecolor="black", alpha=0.55, label=MODEL_LABELS[m])
        for m in MODEL_ORDER
        if m in set(plot_df["model"].unique())
    ]
    if legend_handles:
        axes[0].legend(handles=legend_handles, loc="best", frameon=True)

    if title:
        fig.suptitle(title)

    png_path = outdir / f"{basename}.png"
    pdf_path = outdir / f"{basename}.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    args = parse_args()
    files = iter_input_files(args.inputs)

    dfs = [coerce_columns(load_one_file(path)) for path in files]
    combined = pd.concat(dfs, ignore_index=True, sort=False)

    x_col = choose_x_col(combined, args.x_col)
    plot_df = prepare_boxplot_frame(combined, x_col=x_col, group_col=args.group_col, row_level=args.row_level)

    basename = args.basename or f"boxplot_sweep_{x_col}"
    outdir = Path(args.outdir)

    png_path, pdf_path = plot_boxplot_figure(
        plot_df=plot_df,
        x_col=x_col,
        metrics=args.metrics,
        outdir=outdir,
        basename=basename,
        title=args.title,
        xlabel=args.xlabel,
        dpi=args.dpi,
        showfliers=not args.no_showfliers,
    )

    summary_path = outdir / f"{basename}_seed_level_summary.csv"
    plot_df.to_csv(summary_path, index=False)

    print(f"Loaded {len(files)} input file(s)")
    print(f"Using sweep column: {x_col}")
    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")
    print(f"Saved summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
