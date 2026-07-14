from __future__ import annotations

"""
Known-action Active Inference fast model with decaying average memory
and one-decimal-place bar values.

Purpose
-------
This is the fast / Type-1 comparison model for the slow memory-decay model.
It uses the same general known-action, multi-report structure and the same
memory decay law, but compresses the task into a *single decaying average
memory* rather than two separate bar memories.

Hidden state
------------
  0) register_state : UNSET / PAIR / PAIR_ANCHORED
  1) mem_avg        : UNSET / average grid

The environment still contains the true bar heights, but report actions are
scored against the *memory contents*, not directly against truth. The
perceptual sequence is:

  LOOK_PAIR    -> coarse relative-position cue about the average
  LOOK_AXIS    -> local axis anchor for the average and anchoring of register
  LOOK_SEGMENT -> noisy within-interval cue for the average on a 0.1 grid
  REPORT_r     -> explicit report value

Between actions, the average-memory factor decays through the same style of
retention + diffusion + forgetting kernel used by the slow model.

This model is intended for direct comparison against the slow model:
- same one-decimal report criterion
- same memory-decay parameters
- fewer memory-dependent steps
- one scalar memory trace instead of two
"""

import time
from dataclasses import dataclass
import numpy as np

DTYPE = np.float64
EPS = 1e-16

LOOK_PAIR = 0
LOOK_AXIS = 1
LOOK_SEGMENT = 2

SEL_UNSET = 0
SEL_PAIR = 1
SEL_PAIR_ANCHORED = 2


# ===============================================================
# Helpers
# ===============================================================


def fmt_num(v: float) -> str:
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return s if s else "0"


def softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=DTYPE)
    x = x - np.max(x)
    y = np.exp(x)
    return y / y.sum()


def entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=DTYPE)
    mask = p > 0.0
    return float(-(p[mask] * np.log(p[mask] + EPS)).sum())


def normalize(p: np.ndarray) -> np.ndarray:
    total = float(p.sum())
    if total <= 0.0:
        raise ValueError("Cannot normalize array with non-positive mass")
    return p / total


def make_height_grid(n_ticks: int, height_step: float) -> np.ndarray:
    max_h = n_ticks - 1
    n = int(round(max_h / height_step)) + 1
    grid = np.round(np.arange(n, dtype=DTYPE) * height_step, 10)
    if abs(grid[-1] - max_h) > 1e-8:
        raise ValueError("height_step must land exactly on the top tick")
    return grid


def make_average_grid(height_grid: np.ndarray) -> np.ndarray:
    avg_grid = np.unique(np.round((height_grid[:, None] + height_grid[None, :]) / 2.0, 10))
    return avg_grid.astype(DTYPE)


def round_half_up_to_step(v: float, step: float = 0.1) -> float:
    q = float(v) / step
    return float(np.round(np.floor(q + 0.5 + 1e-12) * step, 10))


def make_report_grid(n_ticks: int, report_step: float = 0.1) -> np.ndarray:
    max_h = n_ticks - 1
    n = int(round(max_h / report_step)) + 1
    grid = np.round(np.arange(n, dtype=DTYPE) * report_step, 10)
    if abs(grid[-1] - max_h) > 1e-8:
        raise ValueError("report_step must land exactly on the top tick")
    return grid


def value_to_index(v: float, grid: np.ndarray, tol: float = 1e-8) -> int:
    idx = int(np.argmin(np.abs(grid - v)))
    if abs(float(grid[idx]) - float(v)) > tol:
        raise ValueError(f"Value {v} is not on the allowed grid {grid}")
    return idx


def classify_relative_position(h: float, n_ticks: int) -> int:
    lower = int(np.floor(h + 1e-10))
    frac = float(h) - lower
    if np.isclose(frac, 0.0):
        return 1  # OnTick
    if np.isclose(frac, 0.5):
        return 3  # Midpoint
    if frac < 0.5:
        return 2  # LowerHalf
    return 4      # UpperHalf


def classify_axis_anchor(h: float, n_ticks: int) -> tuple[str, int]:
    lower = int(np.floor(h + 1e-10))
    frac = float(h) - lower
    if np.isclose(frac, 0.0):
        return ("tick", lower)
    if lower < 0 or lower >= n_ticks - 1:
        raise ValueError(f"Height {h} does not lie in a valid interval")
    return ("lower", lower)


def classify_avg_segment_index(h: float) -> int:
    """Return the one-decimal-place segment index 0..9 within the anchored interval."""
    lower = int(np.floor(h + 1e-10))
    frac = float(h) - lower
    seg_value = round_half_up_to_step(frac, 0.1)
    if np.isclose(seg_value, 1.0):
        seg = 0
    else:
        seg = int(np.floor(seg_value * 10.0 + 1e-10))
    return max(0, min(9, seg))


def avg_signature_stage1(avg: float, n_ticks: int) -> tuple[int, int, int]:
    rel = classify_relative_position(avg, n_ticks)
    kind, k = classify_axis_anchor(avg, n_ticks)
    axis_obs = 1 + k if kind == "tick" else 1 + n_ticks + k
    seg = 1 + classify_avg_segment_index(avg)  # 1..10 because seg_obs has Null at 0
    return rel, axis_obs, seg


def validate_avg_grid_stage1(avg_grid: np.ndarray, n_ticks: int) -> None:
    """Ensure (rel, axis, segment) uniquely identifies each average on the grid."""
    seen = {}
    for a in avg_grid:
        sig = avg_signature_stage1(float(a), n_ticks)
        if sig in seen:
            raise ValueError(
                "This fast Stage-1 model is not identifiable for the chosen grids. "
                f"Averages {seen[sig]} and {float(a)} share signature {sig}."
            )
        seen[sig] = float(a)


def discrete_gaussian(center: int, n: int, sigma: float) -> np.ndarray:
    xs = np.arange(n, dtype=DTYPE)
    p = np.exp(-0.5 * ((xs - float(center)) / max(float(sigma), 1e-6)) ** 2)
    return p / p.sum()


def nearest_tick_value(v: float, n_ticks: int) -> float:
    """Nearest integer tick using half-up rounding, clipped to the valid axis range."""
    return float(np.clip(np.floor(float(v) + 0.5), 0, n_ticks - 1))


def delta_on_grid(value: float, grid: np.ndarray) -> np.ndarray:
    p = np.zeros(len(grid), dtype=DTYPE)
    p[value_to_index(float(value), grid)] = 1.0
    return p


def gaussian_on_grid(center_value: float, grid: np.ndarray, sigma: float) -> np.ndarray:
    center_value = float(center_value)
    grid = grid.astype(DTYPE)
    sigma = float(sigma)

    # Deterministic snap when sigma is zero (or effectively zero)
    if sigma <= 0.0:
        idx = int(np.argmin(np.abs(grid - center_value)))
        p = np.zeros(len(grid), dtype=DTYPE)
        p[idx] = 1.0
        return p

    # Stable Gaussian on grid
    z = -0.5 * ((grid - center_value) / sigma) ** 2
    z = z - np.max(z)
    p = np.exp(z)

    total = float(p.sum())
    if (not np.isfinite(total)) or total <= 0.0:
        idx = int(np.argmin(np.abs(grid - center_value)))
        p = np.zeros(len(grid), dtype=DTYPE)
        p[idx] = 1.0
        return p

    return p / total

def noisy_rel_obs_distribution(
    value: float,
    grid: np.ndarray,
    n_ticks: int,
    percept_sigma: float,
    bias: float = 0.0,
    tick_sigma: float = 0.08,  # unused, kept for compatibility
) -> np.ndarray:
    bias = float(np.clip(bias, 0.0, 1.0))
    tick_value = nearest_tick_value(float(value), n_ticks)
    shifted_value = (1.0 - bias) * float(value) + bias * float(tick_value)

    p_val = gaussian_on_grid(shifted_value, grid, percept_sigma)

    p_obs = np.zeros(5, dtype=DTYPE)
    for idx, grid_val in enumerate(grid):
        p_obs[classify_relative_position(float(grid_val), n_ticks)] += p_val[idx]

    return normalize(p_obs)


def biased_value_distribution(value: float, grid: np.ndarray, n_ticks: int, bias: float, tick_sigma: float) -> np.ndarray:
    bias = float(np.clip(bias, 0.0, 1.0))
    p_true = delta_on_grid(float(value), grid)
    if bias <= 0.0:
        return p_true
    tick_value = nearest_tick_value(float(value), n_ticks)
    p_tick = gaussian_on_grid(tick_value, grid, tick_sigma)
    return normalize((1.0 - bias) * p_true + bias * p_tick)


def biased_rel_obs_distribution(value: float, grid: np.ndarray, n_ticks: int, bias: float, tick_sigma: float) -> np.ndarray:
    p_val = biased_value_distribution(float(value), grid, n_ticks, bias, tick_sigma)
    p_obs = np.zeros(5, dtype=DTYPE)
    for idx, grid_val in enumerate(grid):
        p_obs[classify_relative_position(float(grid_val), n_ticks)] += p_val[idx]
    return normalize(p_obs)

def biased_avg_segment_obs_distribution(
    value: float,
    grid: np.ndarray,
    n_ticks: int,
    bias: float,
    percept_sigma: float,
) -> np.ndarray:
    """
    Segment cue with mean shifted toward nearest tick, but width unchanged.
    Returns 11 probs: [Null, seg0, ..., seg9].
    """
    bias = float(np.clip(bias, 0.0, 1.0))
    tick_value = nearest_tick_value(float(value), n_ticks)
    shifted_value = (1.0 - bias) * float(value) + bias * float(tick_value)

    # same-width uncertainty around shifted value
    p_val = gaussian_on_grid(shifted_value, grid, percept_sigma)

    p_obs = np.zeros(11, dtype=DTYPE)
    for idx, grid_val in enumerate(grid):
        p_obs[1 + classify_avg_segment_index(float(grid_val))] += p_val[idx]

    return normalize(p_obs)


def pretty_top_k(q: np.ndarray, labels, k: int = 5):
    idxs = np.argsort(q)[::-1][:k]
    return [(labels[i], float(q[i])) for i in idxs]


# ===============================================================
# Policy library
# ===============================================================


def build_policy_library(n_reports: int, horizon: int = 4):
    if horizon != 4:
        raise ValueError("This implementation currently hard-codes horizon=4")

    report_start = 3

    def REPORT(r: int) -> int:
        return report_start + r

    policies = []

    def add(seq):
        seq = np.asarray(seq, dtype=int)
        if seq.shape != (horizon,):
            raise ValueError(f"Bad policy shape {seq.shape}")
        policies.append(seq)

    # acquisition / refinement templates
    add([LOOK_PAIR, LOOK_AXIS, LOOK_SEGMENT, LOOK_AXIS])
    add([LOOK_PAIR, LOOK_AXIS, LOOK_SEGMENT, LOOK_PAIR])
    add([LOOK_AXIS, LOOK_SEGMENT, LOOK_PAIR, LOOK_AXIS])
    add([LOOK_SEGMENT, LOOK_AXIS, LOOK_SEGMENT, LOOK_PAIR])

    # direct report templates
    for r in range(n_reports):
        add([LOOK_PAIR, LOOK_AXIS, LOOK_SEGMENT, REPORT(r)])
        add([LOOK_PAIR, LOOK_AXIS, REPORT(r), REPORT(r)])
        add([LOOK_AXIS, LOOK_SEGMENT, REPORT(r), REPORT(r)])
        add([LOOK_SEGMENT, REPORT(r), REPORT(r), REPORT(r)])
        add([REPORT(r), REPORT(r), REPORT(r), REPORT(r)])

    return policies


# ===============================================================
# Environment
# ===============================================================


class DecimalFastAvgEnv:
    def __init__(
        self,
        bar_heights=(1.2, 3.7),
        n_ticks: int = 5,
        height_step: float = 0.1,
        pair_obs_sigma: float = 0.22,
        segment_sigma: float = 0.60,
        course_tick_anchor_bias: float = 0.0,
        # tick_anchor_sigma: float = 0.08,
        segment_tick_anchor_bias: float | None = None,
    ):
        self.n_ticks = int(n_ticks)
        self.height_step = float(height_step)
        self.pair_obs_sigma = float(pair_obs_sigma)
        self.segment_sigma = float(segment_sigma)
        self.course_tick_anchor_bias = float(np.clip(course_tick_anchor_bias, 0.0, 1.0))
        # self.tick_anchor_sigma = float(tick_anchor_sigma)
        if segment_tick_anchor_bias is None:
            # Shared initial tick snap; leave refinement unbiased by default so the
            # fast model's bias comes from early commitment rather than extra bias later.
            segment_tick_anchor_bias = 0.0
        self.segment_tick_anchor_bias = float(np.clip(segment_tick_anchor_bias, 0.0, 1.0))
        self.height_grid = make_height_grid(self.n_ticks, self.height_step)
        self.report_step = 0.1
        self.avg_grid = make_report_grid(self.n_ticks, self.report_step)
        self.report_grid = self.avg_grid.copy()
        validate_avg_grid_stage1(self.avg_grid, self.n_ticks)

        self.bar_heights = [float(v) for v in bar_heights]
        if len(self.bar_heights) != 2:
            raise ValueError("This proof of concept expects exactly two bars")
        self.true_avg = float(np.mean(self.bar_heights))
        self.true_report = round_half_up_to_step(self.true_avg, self.report_step)

        # coarse percept can remain continuous / off-grid
        # self.perceived_avg_pair = self.true_avg
        # ^ Seems to be cause of bug in tick_bias increasing accuracy?
        self.perceived_avg_pair = self.true_report

        # task-refinement cues should match the one-decimal hidden state space
        self.perceived_avg_task = self.true_report

        self.true_avg_idx = value_to_index(self.true_report, self.avg_grid)
        self.register_state = SEL_UNSET

    def step(self, action: int):
        n_reports = len(self.avg_grid)
        report_start = 3
        if action < 0 or action >= report_start + n_reports:
            raise ValueError(f"Action {action} out of range")

        rel_obs = 0
        axis_obs = 0
        seg_obs = 0
        fb_obs = 0

        if action == LOOK_PAIR:
            self.register_state = SEL_PAIR
            p_rel = noisy_rel_obs_distribution(
                self.perceived_avg_pair,
                self.avg_grid,
                self.n_ticks,
                self.pair_obs_sigma,
                self.course_tick_anchor_bias,
                # self.tick_anchor_sigma,
            )
            rel_obs = int(np.random.choice(np.arange(5), p=p_rel))

        elif action == LOOK_AXIS:
            kind, k = classify_axis_anchor(self.perceived_avg_task, self.n_ticks)
            if self.register_state == SEL_PAIR:
                axis_obs = 1 + k if kind == "tick" else 1 + self.n_ticks + k
                self.register_state = SEL_PAIR_ANCHORED
            elif self.register_state == SEL_PAIR_ANCHORED:
                axis_obs = 1 + k if kind == "tick" else 1 + self.n_ticks + k

        elif action == LOOK_SEGMENT:
            if self.register_state == SEL_PAIR_ANCHORED:
                p_seg = biased_avg_segment_obs_distribution(
                    self.perceived_avg_task,
                    self.avg_grid,
                    self.n_ticks,
                    self.segment_tick_anchor_bias,
                    self.segment_sigma,   # keep segment width fixed
                )[1:]

                seg_obs = 1 + int(np.random.choice(np.arange(10), p=p_seg))

        else:
            r = action - report_start
            fb_obs = 2 if r == self.true_avg_idx else 1

        return (rel_obs, axis_obs, seg_obs, fb_obs)


# ===============================================================
# Planner metadata
# ===============================================================


@dataclass
class PlannerMeta:
    height_grid: np.ndarray
    avg_grid: np.ndarray
    rel_names: list[str]
    axis_names: list[str]
    seg_names: list[str]
    fb_names: list[str]
    register_names: list[str]
    mem_names: list[str]
    report_names: list[str]
    policies: list[np.ndarray]
    report_start: int


# ===============================================================
# Planner
# ===============================================================


class FastMemoryDecayTenthsPlanner:
    def __init__(
        self,
        n_ticks: int = 5,
        height_step: float = 0.1,
        policy_len: int = 4,
        gamma: float = 8.0,
        use_states_info_gain: bool = True,
        action_selection: str = "deterministic",
        non_report_cost: float = 0.28,
        # pair_action_cost: float = 0.42,
        # pair_revisit_penalty: float = 0.25,
        mem_retention: float = 0.97,
        mem_forget: float = 0.005,
        mem_sigma: float = 0.25,
        pair_obs_sigma: float = 0.22,
        segment_sigma: float = 0.60,
        course_tick_anchor_bias: float = 0.0,
        # tick_anchor_sigma: float = 0.08,
        segment_tick_anchor_bias: float | None = None,
        info_gain_target: str = "report",
        report_action_instant: bool = True,
        policy_eval_mode: str = "rollout",
        use_log_prefs: bool = False,
        pref_incorrect: float = -80.0,
        urgency_slope: float = 0.0,
    ):
        self.n_ticks = int(n_ticks)
        self.height_step = float(height_step)
        self.policy_len = int(policy_len)
        self.gamma = float(gamma)
        self.use_states_info_gain = bool(use_states_info_gain)
        self.action_selection = action_selection
        self.non_report_cost = float(non_report_cost)
        # self.pair_action_cost = float(pair_action_cost)
        # self.pair_revisit_penalty = float(pair_revisit_penalty)
        self.mem_retention = float(mem_retention)
        self.mem_forget = float(mem_forget)
        self.mem_sigma = float(mem_sigma)
        self.pair_obs_sigma = float(pair_obs_sigma)
        self.segment_sigma = float(segment_sigma)
        self.course_tick_anchor_bias = float(np.clip(course_tick_anchor_bias, 0.0, 1.0))
        # self.tick_anchor_sigma = float(tick_anchor_sigma)
        if segment_tick_anchor_bias is None:
            # Shared initial tick snap; refinement remains unbiased by default.
            segment_tick_anchor_bias = 0.0
        self.segment_tick_anchor_bias = float(np.clip(segment_tick_anchor_bias, 0.0, 1.0))
        if info_gain_target not in ("state", "report"):
            raise ValueError("info_gain_target must be 'state' or 'report'")
        self.info_gain_target = str(info_gain_target)
        self.report_action_instant = bool(report_action_instant)
        if policy_eval_mode not in ("rollout", "branching"):
            raise ValueError("policy_eval_mode must be 'rollout' or 'branching'")
        self.policy_eval_mode = str(policy_eval_mode)
        self.use_log_prefs = bool(use_log_prefs)

        self.pref_incorrect = pref_incorrect
        self.urgency_slope = float(urgency_slope)

        if self.mem_retention < 0 or self.mem_forget < 0 or self.mem_retention + self.mem_forget > 1.0:
            raise ValueError("Require 0 <= retention, forget and retention+forget <= 1")

        self.height_grid = make_height_grid(self.n_ticks, self.height_step)
        self.report_step = 0.1
        self.avg_grid = make_report_grid(self.n_ticks, self.report_step)
        self.report_grid = self.avg_grid.copy()
        validate_avg_grid_stage1(self.avg_grid, self.n_ticks)

        self.n_avg = len(self.avg_grid)
        self.n_mem = 1 + self.n_avg  # UNSET + one-decimal avg values
        self.n_reg = 3
        self.n_reports = self.n_avg
        self.report_start = 3
        self.n_actions = self.report_start + self.n_reports

        self.rel_names = ["Null", "OnTick", "LowerHalf", "Midpoint", "UpperHalf"]
        self.axis_names = ["Null"] + [f"Tick{k}" for k in range(self.n_ticks)] + [f"LowerTick{k}" for k in range(self.n_ticks - 1)]
        self.seg_names = ["Null"] + [f"Seg{k}" for k in range(10)]
        self.fb_names = ["Null", "Incorrect", "Correct"]
        self.register_names = ["UNSET", "PAIR", "PAIR_ANCHORED"]
        self.mem_names = ["UNSET"] + [fmt_num(v) for v in self.avg_grid]
        self.report_names = [fmt_num(v) for v in self.avg_grid]
        self.policies = build_policy_library(self.n_reports, horizon=self.policy_len)

        self._build_decay_kernel()
        self._build_report_projection()
        self._build_action_likelihoods()
        self.reset_beliefs()

    def _build_decay_kernel(self):
        D = np.zeros((self.n_mem, self.n_mem), dtype=DTYPE)
        D[0, 0] = 1.0
        value_states = np.arange(self.n_avg, dtype=DTYPE)
        for prev_mem in range(1, self.n_mem):
            prev_idx = prev_mem - 1
            p = np.exp(-0.5 * ((value_states - prev_idx) / max(self.mem_sigma, 1e-6)) ** 2)
            p = p / p.sum()
            diffuse_mass = max(0.0, 1.0 - self.mem_retention - self.mem_forget)
            D[1:, prev_mem] += diffuse_mass * p
            D[prev_mem, prev_mem] += self.mem_retention
            D[0, prev_mem] += self.mem_forget
        if not np.allclose(D.sum(axis=0), 1.0):
            raise ValueError("Decay kernel columns are not normalized")
        self.mem_decay = D

        # Write kernel: only open an UNSET memory; otherwise preserve current content.
        self.write_dist = np.zeros(self.n_mem, dtype=DTYPE)
        self.write_dist[1:] = 1.0 / self.n_avg
        self.mem_write = np.zeros((self.n_mem, self.n_mem), dtype=DTYPE)
        self.mem_write[:, 0] = self.write_dist
        for m in range(1, self.n_mem):
            self.mem_write[m, m] = 1.0


    def _build_report_projection(self):
        """
        Deterministic map from the fast hidden state onto the task-relevant
        one-decimal report distribution.

        If memory is UNSET, treat the state as maximally uninformative over the
        report grid. Otherwise mem_avg already lives on the report grid.
        """
        n_flat = self.n_reg * self.n_mem
        proj = np.zeros((self.n_reports, n_flat), dtype=DTYPE)
        uniform = np.ones(self.n_reports, dtype=DTYPE) / self.n_reports
        idx = 0
        for reg in range(self.n_reg):
            for m in range(self.n_mem):
                if m == 0:
                    proj[:, idx] = uniform
                else:
                    proj[m - 1, idx] = 1.0
                idx += 1
        self.report_projection = proj

    def q_report_from(self, q: np.ndarray) -> np.ndarray:
        return normalize(self.report_projection @ q.reshape(-1))

    def _expected_report_entropy_after_action(self, q_pred: np.ndarray, action: int) -> float:
        table = self.obs_tables[action]["table"]
        q_flat = q_pred.reshape(-1)
        H_post_exp = 0.0
        for o_idx in range(table.shape[0]):
            weighted = table[o_idx] * q_flat
            po = float(weighted.sum())
            if po <= 0.0:
                continue
            q_report_post = normalize(self.report_projection @ (weighted / po))
            H_post_exp += po * entropy(q_report_post)
        return H_post_exp

    def _build_action_likelihoods(self):
        shape = (self.n_reg, self.n_mem)
        shape_flat = int(np.prod(shape))
        self.obs_tables = {}

        # LOOK_PAIR depends on mem_avg only; now includes ordinary coarse perceptual uncertainty
        # plus optional tick-anchoring bias.
        table = np.zeros((5, *shape), dtype=DTYPE)
        for reg in range(self.n_reg):
            for m in range(self.n_mem):
                if m == 0:
                    table[0, reg, m] = 1.0
                else:
                    p_rel = noisy_rel_obs_distribution(
                        float(self.avg_grid[m - 1]),
                        self.avg_grid,
                        self.n_ticks,
                        self.pair_obs_sigma,
                        # self.course_tick_anchor_bias,
                        0,
                        # self.tick_anchor_sigma,
                    )
                    table[:, reg, m] = p_rel
        self.obs_tables[LOOK_PAIR] = {"kind": "rel", "table": table.reshape(5, shape_flat), "pref": None}

        # LOOK_AXIS depends on anchored/selected register and memory value
        n_axis = 1 + self.n_ticks + (self.n_ticks - 1)
        table = np.zeros((n_axis, *shape), dtype=DTYPE)
        for reg in range(self.n_reg):
            for m in range(self.n_mem):
                if reg in (SEL_PAIR, SEL_PAIR_ANCHORED) and m > 0:
                    kind, k = classify_axis_anchor(float(self.avg_grid[m - 1]), self.n_ticks)
                    axis = 1 + k if kind == "tick" else 1 + self.n_ticks + k
                    table[axis, reg, m] = 1.0
                else:
                    table[0, reg, m] = 1.0
        self.obs_tables[LOOK_AXIS] = {"kind": "axis", "table": table.reshape(n_axis, shape_flat), "pref": None}

        # LOOK_SEGMENT informative only in anchored state
        table = np.zeros((11, *shape), dtype=DTYPE)
        for reg in range(self.n_reg):
            for m in range(self.n_mem):
                if reg == SEL_PAIR_ANCHORED and m > 0:
                    value = float(self.avg_grid[m - 1])
                    table[:, reg, m] = biased_avg_segment_obs_distribution(
                        value,
                        self.avg_grid,
                        self.n_ticks,
                        # self.segment_tick_anchor_bias,
                        0,  # agent doesnt know its
                        self.segment_sigma,   # same width as unbiased segment cue
                    )
                else:
                    table[0, reg, m] = 1.0
        self.obs_tables[LOOK_SEGMENT] = {"kind": "seg", "table": table.reshape(11, shape_flat), "pref": None}

        # REPORT_r depends on memory contents, not truth
        pref = np.array([0.0, self.pref_incorrect, 12.0], dtype=DTYPE)
        for r in range(self.n_reports):
            table = np.zeros((3, *shape), dtype=DTYPE)
            for reg in range(self.n_reg):
                for m in range(self.n_mem):
                    if m == 0:
                        table[1, reg, m] = 1.0
                    else:
                        table[2 if (r == (m - 1)) else 1, reg, m] = 1.0
            self.obs_tables[self.report_start + r] = {"kind": "fb", "table": table.reshape(3, shape_flat), "pref": pref}

    def reset_beliefs(self):
        self.q = np.zeros((self.n_reg, self.n_mem), dtype=DTYPE)
        self.q[SEL_UNSET, 0] = 1.0
        self.last_q_pi = None
        self.last_scores = None
        self.last_best_policy_idx = 0
        self.trial_step = 0

    def transition(self, q: np.ndarray, action: int) -> np.ndarray:
        q = np.asarray(q, dtype=DTYPE)
        out = np.zeros_like(q)

        if action == LOOK_PAIR:
            prev_mem = q.sum(axis=0)
            next_mem = normalize(self.mem_write @ prev_mem)
            out[SEL_PAIR, :] = next_mem
            return out

        if action == LOOK_AXIS:
            decayed = np.zeros_like(q)
            for reg in range(self.n_reg):
                decayed[reg] = self.mem_decay @ q[reg]
            out[SEL_UNSET] += decayed[SEL_UNSET]
            out[SEL_PAIR_ANCHORED] += decayed[SEL_PAIR] + decayed[SEL_PAIR_ANCHORED]
            return normalize(out)

        # REPORT is modeled as an instantaneous commitment based on the
        # current memory contents, so planning a report does not incur an
        # extra decay step before feedback.
        if action >= self.report_start and self.report_action_instant:
            return q.copy()

        # LOOK_SEGMENT and (optionally) REPORT: register holds, memory decays.
        for reg in range(self.n_reg):
            out[reg] = self.mem_decay @ q[reg]
        return normalize(out)

    def expected_obs(self, q: np.ndarray, action: int) -> np.ndarray:
        table = self.obs_tables[action]["table"]
        q_flat = q.reshape(-1)
        qo = table @ q_flat
        return normalize(qo)

    def posterior_given_obs(self, q_pred: np.ndarray, action: int, obs_idx: int) -> np.ndarray:
        table = self.obs_tables[action]["table"]
        q_flat = q_pred.reshape(-1)
        post = table[int(obs_idx)] * q_flat
        total = float(post.sum())
        if total <= 0.0:
            raise ValueError(f"Observation {obs_idx} has zero likelihood under action {action}")
        return (post / total).reshape(q_pred.shape)

    def update_beliefs(self, action: int, obs: tuple[int, int, int, int]) -> np.ndarray:
        q_pred = self.transition(self.q, action)
        if action == LOOK_PAIR:
            obs_idx = obs[0]
        elif action == LOOK_AXIS:
            obs_idx = obs[1]
        elif action == LOOK_SEGMENT:
            obs_idx = obs[2]
        else:
            obs_idx = obs[3]
        self.q = self.posterior_given_obs(q_pred, action, int(obs_idx))
        self.trial_step += 1
        return self.q

    def _epistemic_value_from(self, q_pred: np.ndarray, action: int, qo: np.ndarray | None = None) -> float:
        if not self.use_states_info_gain:
            return 0.0

        if qo is None:
            qo = self.expected_obs(q_pred, action)

        if self.info_gain_target == "state":
            H_prior = entropy(q_pred.reshape(-1))
            H_post_exp = 0.0
            table = self.obs_tables[action]["table"]
            q_flat = q_pred.reshape(-1)
            for o_idx in range(table.shape[0]):
                po = float((table[o_idx] * q_flat).sum())
                if po <= 0.0:
                    continue
                post = (table[o_idx] * q_flat) / po
                H_post_exp += po * entropy(post)
        else:
            H_prior = entropy(self.q_report_from(q_pred))
            H_post_exp = self._expected_report_entropy_after_action(q_pred, action)

        return float(H_prior - H_post_exp)

    def action_cost(self, action: int, time_offset: int = 0) -> float:
        if action >= self.report_start:
            return 0.0
        elapsed = self.trial_step + int(time_offset)
        return self.non_report_cost + self.urgency_slope * elapsed

    def forced_report_action(self, rule: str = "map") -> int:
        q_rep = self.q_report_from(self.q)
        if rule == "mean":
            mean_value = float(np.sum(q_rep * self.report_grid))
            report_idx = int(np.argmin(np.abs(self.report_grid - mean_value)))
        elif rule == "map":
            report_idx = int(np.argmax(q_rep))
        else:
            raise ValueError("rule must be 'map' or 'mean'")
        return self.report_start + report_idx

    def score_action(self, q_before: np.ndarray, q_pred: np.ndarray, action: int) -> float:
        qo = self.expected_obs(q_pred, action)
        utility = 0.0
        pref = self.obs_tables[action]["pref"]
        if pref is not None:
            utility = float(qo @ pref)

        info_gain = self._epistemic_value_from(q_pred, action, qo=qo)
        cost = self.action_cost(action)
        return utility + info_gain - cost

    def immediate_value(
        self,
        q: np.ndarray,
        action: int,
        use_log_prefs: bool | None = None,
        time_offset: int = 0,
    ):
        """
        One-step value from posterior q under action a.

        Returns
        -------
        q_pred : predicted belief after transition, before observation
        qo     : predictive observation distribution
        val    : immediate value = extrinsic + epistemic - cost
        """
        if use_log_prefs is None:
            use_log_prefs = self.use_log_prefs

        q_pred = self.transition(q, action)
        qo = self.expected_obs(q_pred, action)

        pref = self.obs_tables[action]["pref"]
        extrinsic = 0.0
        if pref is not None:
            if use_log_prefs:
                p_pref = softmax(pref)
                extrinsic = float(qo @ np.log(p_pref + EPS))
            else:
                extrinsic = float(qo @ pref)

        epistemic = self._epistemic_value_from(q_pred, action, qo=qo)
        cost = self.action_cost(action, time_offset=time_offset)

        return q_pred, qo, extrinsic + epistemic - cost

    def evaluate_policy(self, policy: np.ndarray, use_log_prefs: bool | None = None) -> float:
        q_roll = self.q.copy()
        score = 0.0
        for t, action in enumerate(policy):
            action = int(action)
            q_pred, qo, immediate = self.immediate_value(
                q_roll, action, use_log_prefs=use_log_prefs, time_offset=t
            )
            score += immediate
            if action >= self.report_start:
                break
            q_roll = q_pred
        return float(score)

    def evaluate_policy_recursive(
        self,
        q: np.ndarray,
        policy: np.ndarray,
        depth: int = 0,
        use_log_prefs: bool | None = None,
    ) -> float:
        """Recursive expected policy value with posterior branching over observations."""
        if depth >= len(policy):
            return 0.0

        action = int(policy[depth])
        q_pred, qo, immediate = self.immediate_value(
            q, action, use_log_prefs=use_log_prefs, time_offset=depth
        )

        if action >= self.report_start:
            return float(immediate)

        future = 0.0
        for o_idx, po in enumerate(qo):
            if po <= EPS:
                continue
            q_post = self.posterior_given_obs(q_pred, action, o_idx)
            future += float(po) * self.evaluate_policy_recursive(
                q_post, policy, depth + 1, use_log_prefs=use_log_prefs
            )

        return float(immediate + future)

    def infer_policies(self, policy_eval_mode: str | None = None, use_log_prefs: bool | None = None):
        if policy_eval_mode is None:
            policy_eval_mode = self.policy_eval_mode
        if use_log_prefs is None:
            use_log_prefs = self.use_log_prefs

        if policy_eval_mode == "branching":
            scores = np.array(
                [
                    self.evaluate_policy_recursive(self.q, pol, depth=0, use_log_prefs=use_log_prefs)
                    for pol in self.policies
                ],
                dtype=DTYPE,
            )
        elif policy_eval_mode == "rollout":
            scores = np.array(
                [self.evaluate_policy(pol, use_log_prefs=use_log_prefs) for pol in self.policies],
                dtype=DTYPE,
            )
        else:
            raise ValueError("policy_eval_mode must be 'rollout' or 'branching'")

        q_pi = softmax(self.gamma * scores)
        self.last_scores = scores
        self.last_q_pi = q_pi
        self.last_best_policy_idx = int(np.argmax(scores))
        return q_pi, scores

    def sample_action(self, q_pi: np.ndarray | None = None) -> int:
        if q_pi is None:
            if self.last_q_pi is None:
                raise ValueError("Call infer_policies() before sample_action()")
            q_pi = self.last_q_pi
        if self.action_selection == "stochastic":
            first_actions = np.array([pol[0] for pol in self.policies], dtype=int)
            marginals = np.zeros(self.n_actions, dtype=DTYPE)
            for p_idx, a in enumerate(first_actions):
                marginals[a] += q_pi[p_idx]
            return int(np.random.choice(np.arange(self.n_actions), p=marginals))
        return int(self.policies[self.last_best_policy_idx][0])

    def q_marginals(self):
        q_reg = self.q.sum(axis=1)
        q_mem = self.q.sum(axis=0)
        return q_reg, q_mem

    def meta(self) -> PlannerMeta:
        return PlannerMeta(
            height_grid=self.height_grid,
            avg_grid=self.avg_grid,
            rel_names=self.rel_names,
            axis_names=self.axis_names,
            seg_names=self.seg_names,
            fb_names=self.fb_names,
            register_names=self.register_names,
            mem_names=self.mem_names,
            report_names=self.report_names,
            policies=self.policies,
            report_start=self.report_start,
        )


# ===============================================================
# Simulation helpers
# ===============================================================


def action_to_name(action: int, meta: PlannerMeta) -> str:
    if action == LOOK_PAIR:
        return "LOOK_PAIR"
    if action == LOOK_AXIS:
        return "LOOK_AXIS"
    if action == LOOK_SEGMENT:
        return "LOOK_SEGMENT"
    return f"REPORT {meta.report_names[action - meta.report_start]}"


def run_sim(
    bar_heights=(1.2, 3.7),
    n_ticks: int = 5,
    height_step: float = 0.1,
    T: int = 12,
    policy_len: int = 4,
    gamma: float = 8.0,
    seed: int = 1,
    mem_retention: float = 0.97,
    mem_forget: float = 0.005,
    mem_sigma: float = 0.25,
    non_report_cost: float = 0.28,
    # pair_action_cost: float = 0.42,
    # pair_revisit_penalty: float = 0.25,
    # pair_obs_sigma: float = 0.22,
    pair_obs_sigma: float = 0.0,
    segment_sigma: float = 0.60,
    course_tick_anchor_bias: float = 0.0,
    # tick_anchor_sigma: float = 0.08,
    segment_tick_anchor_bias: float | None = None,
    info_gain_target: str = "report",
    report_action_instant: bool = True,
    policy_eval_mode: str = "rollout",
    pref_incorrect: float = -80.0,
    use_log_prefs: bool = False,
    urgency_slope: float = 0.0,
    force_report_at_deadline: bool = True,
    forced_report_rule: str = "map",
    verbose=True,
    action_selection='deterministic'
):
    np.random.seed(seed)

    env = DecimalFastAvgEnv(
        bar_heights=bar_heights,
        n_ticks=n_ticks,
        height_step=height_step,
        pair_obs_sigma=pair_obs_sigma,
        segment_sigma=segment_sigma,
        course_tick_anchor_bias=course_tick_anchor_bias,
        # tick_anchor_sigma=tick_anchor_sigma,
        segment_tick_anchor_bias=segment_tick_anchor_bias,
    )
    planner = FastMemoryDecayTenthsPlanner(
        n_ticks=n_ticks,
        height_step=height_step,
        policy_len=policy_len,
        gamma=gamma,
        use_states_info_gain=True,
        action_selection=action_selection,
        non_report_cost=non_report_cost,
        # pair_action_cost=pair_action_cost,
        # pair_revisit_penalty=pair_revisit_penalty,
        mem_retention=mem_retention,
        mem_forget=mem_forget,
        mem_sigma=mem_sigma,
        pair_obs_sigma=pair_obs_sigma,
        segment_sigma=segment_sigma,
        course_tick_anchor_bias=course_tick_anchor_bias,
        # tick_anchor_sigma=tick_anchor_sigma,
        segment_tick_anchor_bias=segment_tick_anchor_bias,
        info_gain_target=info_gain_target,
        report_action_instant=report_action_instant,
        policy_eval_mode=policy_eval_mode,
        use_log_prefs=use_log_prefs,
        pref_incorrect=pref_incorrect,
        urgency_slope=urgency_slope,
    )
    meta = planner.meta()
    obs = (0, 0, 0, 0)

    if verbose:
        print("\n--- Known-action Active Inference (FAST DIRECT-AVERAGE MEMORY + ONE-DECIMAL CODE) ---\n")
        print(f"Allowed heights : {[fmt_num(v) for v in meta.height_grid]}")
        print(f"Allowed averages: {meta.report_names}")
        print(f"True bars: {list(env.bar_heights)} | True avg: {fmt_num(env.true_avg)} | Rounded target: {fmt_num(env.true_report)}")
        print(f"Start env.register_state: {meta.register_names[env.register_state]}")
        print(f"Policies in library: {len(meta.policies)}")
        print(
            f"mem_retention={mem_retention} | mem_forget={mem_forget} | mem_sigma={mem_sigma} | "
            f"segment_sigma={segment_sigma} | non_report_cost={non_report_cost} | urgency_slope={urgency_slope} | "
            # f"pair_action_cost={pair_action_cost} | "
            # f"pair_revisit_penalty={pair_revisit_penalty} | "
            f"course_tick_anchor_bias={course_tick_anchor_bias} | "
            # f"tick_anchor_sigma={tick_anchor_sigma} | "
            f"segment_tick_anchor_bias={segment_tick_anchor_bias if segment_tick_anchor_bias is not None else course_tick_anchor_bias} | "
            f"info_gain_target={info_gain_target} | report_action_instant={report_action_instant} | "
            f"policy_eval_mode={policy_eval_mode} | use_log_prefs={use_log_prefs}"
        )

    for t in range(T):
        q_reg, q_mem = planner.q_marginals()
        if verbose:
            print(f"\nt={t}  obs={obs}")
            print(f"  env.register_state : {meta.register_names[env.register_state]}")
            print(f"  q(register_state)  : {np.round(q_reg, 3)}  MAP={meta.register_names[int(np.argmax(q_reg))]}")
            print(f"  mem_avg beliefs    : {pretty_top_k(q_mem, meta.mem_names, k=8)}")

        t0 = time.perf_counter()
        if force_report_at_deadline and t == T - 1:
            action = planner.forced_report_action(rule=forced_report_rule)
            best_policy_idx = planner.last_best_policy_idx
            dt = (time.perf_counter() - t0) * 1000.0
            q_pi, scores = None, None
        else:
            q_pi, scores = planner.infer_policies()
            action = planner.sample_action(q_pi)
            best_policy_idx = planner.last_best_policy_idx
            dt = (time.perf_counter() - t0) * 1000.0
        if verbose:
            print(f"  planning ms        : {dt:.2f}")

        obs = env.step(action)
        planner.update_beliefs(action, obs)

        action_name = action_to_name(action, meta)
        if verbose:
            if force_report_at_deadline and t == T - 1:
                print("  best policy        : FORCED_REPORT_AT_DEADLINE")
            else:
                print(f"  best policy        : {[action_to_name(a, meta) for a in meta.policies[best_policy_idx]]}")
            print(f"  action             : {action_name}")
            print(f"  rel_obs            : {meta.rel_names[obs[0]]}")
            print(f"  axis_obs           : {meta.axis_names[obs[1]]}")
            print(f"  seg_obs            : {meta.seg_names[obs[2]]}")
            print(f"  fb_obs             : {meta.fb_names[obs[3]]}")

        if action >= meta.report_start:
            print("  feedback =>", "CORRECT" if obs[3] == 2 else "INCORRECT")
            print(f"  action             : {action_name}")
            break


if __name__ == "__main__":
    # for i in range(30):
    run_sim(
        bar_heights=(2.4, 1.9),
        # bar_heights=(3.0, 2.9),
        seed=2,
        # seed=0,
        # 
        T=7,
        # 
        n_ticks = 4,
        height_step = 0.1,
        policy_len = 4,
        gamma = 8.0,
        policy_eval_mode = "rollout",
        # policy_eval_mode = "branching",
        info_gain_target = "report",
        report_action_instant = True,
        # 
        pair_obs_sigma = 0, 
        # 
        mem_retention=0.97,
        # mem_retention=0.3,
        # 
        mem_forget=0.005,
        mem_sigma=0.5,
        # mem_sigma=2.0,
        # 
        non_report_cost=0.35,
        # 
        segment_sigma = 0.045,
        # segment_sigma = 0.2,
        # 
        course_tick_anchor_bias = 0.00,
        # tick_anchor_sigma = 0.02,
        # course_tick_anchor_bias = 0.0,
        # tick_anchor_sigma = 0.02,
        segment_tick_anchor_bias = 0.0,
        # pref_incorrect = -50
        # verbose=False
        # action_selection='stochastic'
        action_selection='deterministic'
    )
