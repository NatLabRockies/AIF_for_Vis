# Active Inference for simulating human interpretation of data visualization 

Code and precomputed results accompanying:

**Harrison J. Goldwyn, Graham Johnson, Christopher Ibarra, Lace Padilla, and Kenny Gruchalla**  
*Beyond the Post Hoc User Study: Modeling Visual Decision-Making with Active Inference*

## Overview

This repository contains the Active Inference models and analysis scripts used to study a simple visualization-interpretation task: estimating the average value of two bars in a bar chart.

The work is a proof of concept for translating hypothesized cognitive strategies into executable, inspectable process models. We implement two idealized strategies inspired by dual-process accounts of visualization-aided decision making:

- **Fast model:** a compressed, heuristic strategy that estimates the visual midpoint of the two bars and maintains a single belief over their average.
- **Slow model:** a sequential, analytic strategy that estimates the two bar heights separately and maintains them in working memory before computing an average.

Both models use a common Active-Inference-inspired framework for sequential perception, belief updating, action selection, and reporting. Their different internal representations produce distinct predicted vulnerabilities:

- the **Fast model** is more susceptible to **tick-salience bias**;
- the **Slow model** is more susceptible to **working-memory decay**.

The repository includes the model implementations, scripts used for the experiments reported in the paper, precomputed trial-level results, and plotting scripts.

> **Scope:** These are task-specific proof-of-concept cognitive process models. They use symbolic observations and hand-designed policy libraries rather than operating directly on chart pixels. They should not be interpreted as general-purpose chart-reading systems or as empirically validated models of human cognition.

## Repository structure

```text
.
├── models/
│   ├── fast_1dpTask.py
│   └── slow_1dpTask.py
│
├── baseline_validation/
│   ├── validate_fast_slow_baseline_rng.py
│   ├── plot_validation_main_and_supp_edit.py
│   ├── validation_results.json
│   └── validation_figs_properRNG/
│
├── memory_exp/
│   ├── memory_decay_sweep_config.py
│   └── memory_decay_sweep_out_deadlines_bigSweep/
│
├── tick_bias/
│   ├── tick_bias_sweep_config.py
│   ├── plot_sweep_boxplots.py
│   └── seg_tick_bias_sweep_dense/
│
└── plot_heuristic_failure/
    ├── plot_belief_timeseries.py
    └── precomputed figure outputs
```

### `models/`

Contains the two core model implementations.

#### `fast_1dpTask.py`

Implements the Fast, heuristic strategy. The model compresses the task into a single latent memory for the estimated bar-pair average and follows a staged perceptual sequence before reporting.

#### `slow_1dpTask.py`

Implements the Slow, analytic strategy. The model maintains separate memories for the two bar heights and sequentially gathers and integrates evidence before reporting their average.

### `baseline_validation/`

Runs the two models on bar-height pairs under the baseline parameterization and summarizes their accuracy, error structure, and pair-specific failure patterns.

The included `validation_results.json` contains the precomputed trial-level results used for the paper. The included `validation_figs_properRNG/` directory contains the corresponding figures and summary tables.

### `memory_exp/`

Contains the working-memory-decay experiment. The paper configuration sweeps memory retention while holding the remaining shared and model-specific parameters fixed.

The included `memory_decay_sweep_out_deadlines_bigSweep/` directory contains the precomputed trial-level results, metadata, summary tables, and figures.

### `tick_bias/`

Contains the tick-salience-bias experiment. The paper configuration varies the bias applied to the segment-refinement observation and compares the resulting Fast and Slow model accuracies.

The included `seg_tick_bias_sweep_dense/` directory contains the precomputed trial-level results, metadata, summary tables, and figures. `plot_sweep_boxplots.py` is a standalone helper for generating seed-level boxplots from compatible sweep outputs.

### `plot_heuristic_failure/`

Generates posterior-belief traces for the illustrative failure case with bar heights `(2.4, 1.9)`, in which the Fast model commits to an incorrect report while the Slow model reaches the correct one.

## Requirements

The code requires **Python 3.10 or newer** and the following packages:

```text
numpy
pandas
matplotlib
```

A minimal environment can be created with:

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib
```

The current model implementations are self-contained and do not require a separate Active Inference package.

## Quick start: regenerate figures from included results

The repository includes precomputed simulation outputs, so most figures can be regenerated without rerunning the full simulations.

From the repository root:

```bash
python baseline_validation/plot_validation_main_and_supp_edit.py \
    baseline_validation/validation_results.json \
    --outdir baseline_validation/validation_figs_reproduced
```

This writes:

```text
main_text_validation.png
main_text_validation.pdf
supplement_validation.png
supplement_validation.pdf
summary_by_model.csv
summary_by_category.csv
candidate_failure_cases.csv
```

The tick-bias results can also be replotted without rerunning the sweep:

```bash
python tick_bias/tick_bias_sweep_config.py \
    --mode load \
    --load-from tick_bias/seg_tick_bias_sweep_dense \
    --outdir tick_bias/seg_tick_bias_replotted
```

## Reproducing the simulations

The model imports are resolved relative to each script's location, so the scripts do not depend on machine-specific absolute paths. The commands below are written to keep newly generated outputs inside the corresponding experiment directory.

### 1. Baseline validation

The baseline experiment evaluates both models over every ordered pair of one-decimal bar heights from `0.0` to `3.0`, using 10 random seeds per pair.

From the repository root:

```bash
python baseline_validation/validate_fast_slow_baseline_rng.py
```

The configured run evaluates:

- 31 possible bar heights;
- 961 ordered bar-height pairs;
- 10 seeds per pair;
- both Fast and Slow models.

Results are written to:

```text
baseline_validation/validation_results.json
```

The default parameterization is defined in the `if __name__ == "__main__":` block at the bottom of `validate_fast_slow_baseline_rng.py` and is also stored in the output metadata.

To regenerate the baseline figures after rerunning the simulations:

```bash
python baseline_validation/plot_validation_main_and_supp_edit.py \
    baseline_validation/validation_results.json \
    --outdir baseline_validation/validation_figs_reproduced
```

### 2. Illustrative Fast-versus-Slow failure trace

To reproduce the cognitive-trace figure for the example with bar heights `(2.4, 1.9)`:

```bash
cd plot_heuristic_failure
python plot_belief_timeseries.py
cd ..
```

The default configuration uses seed `2` and writes:

```text
plot_heuristic_failure/failure_case_fast_vs_slow.png
plot_heuristic_failure/failure_case_fast_vs_slow.pdf
```

Additional plotting functions for detailed Fast, Slow, and combined posterior traces are included in the same script.

### 3. Working-memory-decay experiment

The paper configuration is defined in the parameter blocks at the bottom of:

```text
memory_exp/memory_decay_sweep_config.py
```

The reported sweep varies:

```text
mem_retention = 0.99, 0.95, 0.90, ..., 0.05
```

using 10 sampled bar pairs and 10 random seeds per sweep value.

To rerun the configured experiment:

```bash
cd memory_exp
python memory_decay_sweep_config.py
cd ..
```

By default, results are written to:

```text
memory_exp/memory_decay_sweep_out_deadlines_bigSweep/
```

The output directory contains trial-level results, metadata, summary tables, and figures. To preserve the included precomputed results, change the `outdir` value in the configuration block before rerunning.

### 4. Tick-salience-bias experiment

The paper configuration is defined at the bottom of:

```text
tick_bias/tick_bias_sweep_config.py
```

The reported sweep varies:

```text
segment_tick_anchor_bias = 0.00, 0.05, 0.10, ..., 0.95
```

using 10 sampled bar pairs and 10 random seeds per sweep value.

To rerun the sweep while preserving the included precomputed results:

```bash
python tick_bias/tick_bias_sweep_config.py \
    --mode run \
    --outdir tick_bias/seg_tick_bias_sweep_reproduced
```

To replot an existing sweep without rerunning simulations:

```bash
python tick_bias/tick_bias_sweep_config.py \
    --mode load \
    --load-from tick_bias/seg_tick_bias_sweep_dense \
    --outdir tick_bias/seg_tick_bias_replotted
```

Additional plotting options are available with:

```bash
python tick_bias/tick_bias_sweep_config.py --help
```

## Reproducibility notes

The experiment scripts record trial-level results together with the relevant model parameters, sampling settings, and random seeds.

The baseline experiment uses exhaustive coverage of the one-decimal bar-height grid. The memory-decay and tick-bias sweeps use the same stratified sample of 10 unordered bar pairs with target proportions:

- 10% both bars on ticks;
- 30% one bar on a tick and one off tick;
- 60% both bars off tick.

Each sampled pair is evaluated across 10 base seeds at every sweep value.

The principal model parameters examined in the paper include:

- `mem_retention`: fraction of working-memory probability mass retained between updates;
- `mem_forget`: probability mass shifted toward an unset memory state;
- `mem_sigma`: width of diffusion across neighboring memory states;
- `segment_tick_anchor_bias`: strength of tick-directed bias in the segment-refinement cue;
- `course_tick_anchor_bias`: bias applied to the initial coarse perceptual cue.

The exact parameterizations used for each experiment are defined directly in the corresponding experiment scripts and are stored with the precomputed results as JSON metadata.

## Scientific interpretation

The Fast and Slow agents are not intended to establish that human observers necessarily implement these exact algorithms. Instead, they demonstrate how hypothesized cognitive strategies can be made computationally explicit.

The resulting simulations generate inspectable predictions at multiple levels, including:

- task accuracy;
- pair-specific failure patterns;
- sensitivity to working-memory decay;
- sensitivity to tick-salience bias;
- action sequences; and
- posterior belief trajectories.

These outputs can, in principle, be compared with behavioral, response-time, and eye-tracking data to test, refine, or falsify the mechanisms encoded by a model.

## Citation

Please cite the associated paper when using this code or its results:

> Harrison J. Goldwyn, Graham Johnson, Christopher Ibarra, Lace Padilla, and Kenny Gruchalla.  
> **Beyond the Post Hoc User Study: Modeling Visual Decision-Making with Active Inference.**

The final publication citation and DOI will be added here when available.
