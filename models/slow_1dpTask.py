from __future__ import annotations

"""
Known-action Active Inference slow model with decaying working memory
and one-decimal-place bar values.

Purpose
-------
This extends the working slow memory-decay model to one-decimal-place bar values
while keeping the same core architecture:

  0) register_state : UNSET / BAR0 / BAR1 / BAR0_ANCHORED / BAR1_ANCHORED
  1) mem_bar0       : UNSET / value grid
  2) mem_bar1       : UNSET / value grid

The environment still contains the true bar heights, but report actions are
scored against the *memory contents*, not the truth. The perceptual sequence is:

  LOOK_BARi   -> coarse relative-position cue
  LOOK_AXIS   -> local axis anchor and anchoring of register state
  LOOK_SEGMENT-> noisy tenth-segment cue within the anchored interval
  REPORT_r    -> explicit report value

Between actions, memory factors decay through a diffusion + forgetting kernel.
This is intended as the slow / Type-2 model for comparison against a faster
model with the same decay law but fewer memory-dependent steps.
"""

import time
from dataclasses import dataclass
import numpy as np

DTYPE = np.float64
EPS = 1e-16

LOOK_BAR0 = 0
LOOK_BAR1 = 1
LOOK_AXIS = 2
LOOK_SEGMENT = 3

SEL_UNSET = 0
SEL_BAR0 = 1
SEL_BAR1 = 2
SEL_BAR0_ANCHORED = 3
SEL_BAR1_ANCHORED = 4


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


def classify_segment_index(h: float) -> int:
    """Return the one-decimal-place segment index 0..9 within the anchored interval."""
    lower = int(np.floor(h + 1e-10))
    frac = float(h) - lower
    seg = int(round(frac * 10.0))
    # integer ticks map to segment 0 in their anchored interval
    if seg == 10:
        seg = 0
    return max(0, min(9, seg))


def height_signature_stage1(h: float, n_ticks: int) -> tuple[int, int, int]:
    rel = classify_relative_position(h, n_ticks)
    kind, k = classify_axis_anchor(h, n_ticks)
    axis_obs = 1 + k if kind == "tick" else 1 + n_ticks + k
    seg = 1 + classify_segment_index(h)  # 1..10 because seg_obs has Null at 0
    return rel, axis_obs, seg


def validate_height_grid_stage1(height_grid: np.ndarray, n_ticks: int) -> None:
    """Ensure (rel, axis, segment) uniquely identifies each height on the grid."""
    seen = {}
    for h in height_grid:
        sig = height_signature_stage1(float(h), n_ticks)
        if sig in seen:
            raise ValueError(
                "This Stage-1 slow model is not identifiable for the chosen height_step. "
                f"Heights {seen[sig]} and {float(h)} share signature {sig}. "
                "Use a coarser grid or add a finer refinement observation."
            )
        seen[sig] = float(h)


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

def biased_segment_obs_distribution(
    value: float,
    grid: np.ndarray,
    n_ticks: int,
    bias: float,
    percept_sigma: float,
) -> np.ndarray:
    bias = float(np.clip(bias, 0.0, 1.0))
    tick_value = nearest_tick_value(float(value), n_ticks)
    shifted_value = (1.0 - bias) * float(value) + bias * float(tick_value)

    p_val = gaussian_on_grid(shifted_value, grid, percept_sigma)

    p_obs = np.zeros(11, dtype=DTYPE)
    for idx, grid_val in enumerate(grid):
        p_obs[1 + classify_segment_index(float(grid_val))] += p_val[idx]
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

    report_start = 4

    def REPORT(r: int) -> int:
        return report_start + r

    policies = []

    def add(seq):
        seq = np.asarray(seq, dtype=int)
        if seq.shape != (horizon,):
            raise ValueError(f"Bad policy shape {seq.shape}")
        policies.append(seq)

    # exploratory / acquisition templates that favor the intended subroutine:
    # LOOK_BAR -> LOOK_AXIS -> LOOK_SEGMENT before switching targets.
    add([LOOK_BAR0, LOOK_AXIS, LOOK_SEGMENT, LOOK_BAR1])
    add([LOOK_BAR1, LOOK_AXIS, LOOK_SEGMENT, LOOK_BAR0])
    add([LOOK_AXIS, LOOK_SEGMENT, LOOK_BAR0, LOOK_AXIS])
    add([LOOK_AXIS, LOOK_SEGMENT, LOOK_BAR1, LOOK_AXIS])
    add([LOOK_SEGMENT, LOOK_BAR0, LOOK_AXIS, LOOK_SEGMENT])
    add([LOOK_SEGMENT, LOOK_BAR1, LOOK_AXIS, LOOK_SEGMENT])

    # direct refine->report templates
    for r in range(n_reports):
        add([LOOK_BAR0, LOOK_AXIS, LOOK_SEGMENT, REPORT(r)])
        add([LOOK_BAR1, LOOK_AXIS, LOOK_SEGMENT, REPORT(r)])
        add([LOOK_AXIS, LOOK_SEGMENT, REPORT(r), REPORT(r)])
        add([LOOK_SEGMENT, REPORT(r), REPORT(r), REPORT(r)])
        add([REPORT(r), REPORT(r), REPORT(r), REPORT(r)])

    return policies


# ===============================================================
# Environment
# ===============================================================


class DecimalSegmentEnv:
    def __init__(
        self,
        bar_heights=(1.2, 3.7),
        n_ticks: int = 5,
        height_step: float = 0.1,
        segment_sigma: float = 0.45,
        course_tick_anchor_bias: float = 0.0,
        bar_obs_sigma: float = 0.08,
        segment_tick_anchor_bias: float | None = None,
        info_gain_target: str = "report",
        report_action_instant: bool = True,
    ):
        self.n_ticks = int(n_ticks)
        self.height_step = float(height_step)
        self.segment_sigma = float(segment_sigma)
        self.course_tick_anchor_bias = float(np.clip(course_tick_anchor_bias, 0.0, 1.0))
        self.bar_obs_sigma = float(bar_obs_sigma)
        if segment_tick_anchor_bias is None:
            # Shared initial tick snap, but refinement is unbiased by default so
            # the slow model can recover the true value with additional effort.
            segment_tick_anchor_bias = 0.0
        self.segment_tick_anchor_bias = float(np.clip(segment_tick_anchor_bias, 0.0, 1.0))
        self.height_grid = make_height_grid(self.n_ticks, self.height_step)
        validate_height_grid_stage1(self.height_grid, self.n_ticks)
        self.avg_grid = make_average_grid(self.height_grid)
        self.report_step = 0.1
        self.report_grid = make_report_grid(self.n_ticks, self.report_step)

        self.bar_heights = [float(v) for v in bar_heights]
        if len(self.bar_heights) != 2:
            raise ValueError("This proof of concept expects exactly two bars")
        self.bar_idx = [value_to_index(h, self.height_grid) for h in self.bar_heights]

        self.true_avg = float(np.mean(self.bar_heights))
        self.true_report = round_half_up_to_step(self.true_avg, self.report_step)
        self.true_avg_idx = value_to_index(self.true_report, self.report_grid)
        self.register_state = SEL_UNSET

    def step(self, action: int):
        n_reports = len(self.report_grid)
        report_start = 4
        if action < 0 or action >= report_start + n_reports:
            raise ValueError(f"Action {action} out of range")

        rel_obs = 0
        axis_obs = 0
        seg_obs = 0
        fb_obs = 0

        if action == LOOK_BAR0:
            self.register_state = SEL_BAR0
            p_rel = noisy_rel_obs_distribution(
                self.bar_heights[0],
                self.height_grid,
                self.n_ticks,
                self.bar_obs_sigma,
                self.course_tick_anchor_bias,
            )
            rel_obs = int(np.random.choice(np.arange(5), p=p_rel))

        elif action == LOOK_BAR1:
            self.register_state = SEL_BAR1
            p_rel = noisy_rel_obs_distribution(
                self.bar_heights[1],
                self.height_grid,
                self.n_ticks,
                self.bar_obs_sigma,
                self.course_tick_anchor_bias,
            )
            rel_obs = int(np.random.choice(np.arange(5), p=p_rel))

        elif action == LOOK_AXIS:
            if self.register_state == SEL_BAR0:
                kind, k = classify_axis_anchor(self.bar_heights[0], self.n_ticks)
                axis_obs = 1 + k if kind == "tick" else 1 + self.n_ticks + k
                self.register_state = SEL_BAR0_ANCHORED
            elif self.register_state == SEL_BAR1:
                kind, k = classify_axis_anchor(self.bar_heights[1], self.n_ticks)
                axis_obs = 1 + k if kind == "tick" else 1 + self.n_ticks + k
                self.register_state = SEL_BAR1_ANCHORED
            elif self.register_state == SEL_BAR0_ANCHORED:
                kind, k = classify_axis_anchor(self.bar_heights[0], self.n_ticks)
                axis_obs = 1 + k if kind == "tick" else 1 + self.n_ticks + k
            elif self.register_state == SEL_BAR1_ANCHORED:
                kind, k = classify_axis_anchor(self.bar_heights[1], self.n_ticks)
                axis_obs = 1 + k if kind == "tick" else 1 + self.n_ticks + k

        elif action == LOOK_SEGMENT:
            if self.register_state == SEL_BAR0_ANCHORED:
                if self.segment_tick_anchor_bias > 0.0:
                    p_seg = biased_segment_obs_distribution(
                        self.bar_heights[0], self.height_grid, self.n_ticks,
                        self.segment_tick_anchor_bias, self.bar_obs_sigma
                    )[1:]
                else:
                    true_seg = classify_segment_index(self.bar_heights[0])
                    p_seg = discrete_gaussian(true_seg, 10, self.segment_sigma)
                seg_obs = 1 + int(np.random.choice(np.arange(10), p=p_seg))
            elif self.register_state == SEL_BAR1_ANCHORED:
                if self.segment_tick_anchor_bias > 0.0:
                    p_seg = biased_segment_obs_distribution(
                        self.bar_heights[1], self.height_grid, self.n_ticks,
                        self.segment_tick_anchor_bias, self.bar_obs_sigma
                    )[1:]
                else:
                    true_seg = classify_segment_index(self.bar_heights[1])
                    p_seg = discrete_gaussian(true_seg, 10, self.segment_sigma)
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


class SlowMemoryDecayTenthsPlanner:
    def __init__(
        self,
        n_ticks: int = 5,
        height_step: float = 0.1,
        policy_len: int = 4,
        gamma: float = 8.0,
        use_states_info_gain: bool = True,
        action_selection: str = "deterministic",
        non_report_cost: float = 0.35,
        mem_retention: float = 0.97,
        mem_forget: float = 0.005,
        mem_sigma: float = 0.25,
        segment_sigma: float = 0.45,
        course_tick_anchor_bias: float = 0.0,
        bar_obs_sigma: float = 0.08,
        segment_tick_anchor_bias: float | None = None,
        info_gain_target: str = "report",
        report_action_instant: bool = True,
        policy_eval_mode: str = "rollout",   # "rollout" or "branching"
        use_log_prefs: bool = False,
        urgency_slope: float = 0.0,
    ):
        self.n_ticks = int(n_ticks)
        self.height_step = float(height_step)
        self.policy_len = int(policy_len)
        self.gamma = float(gamma)
        self.use_states_info_gain = bool(use_states_info_gain)
        self.action_selection = action_selection
        self.non_report_cost = float(non_report_cost)
        self.mem_retention = float(mem_retention)
        self.mem_forget = float(mem_forget)
        self.mem_sigma = float(mem_sigma)
        self.segment_sigma = float(segment_sigma)
        self.course_tick_anchor_bias = float(np.clip(course_tick_anchor_bias, 0.0, 1.0))
        self.bar_obs_sigma = float(bar_obs_sigma)
        if segment_tick_anchor_bias is None:
            # Shared initial tick snap, but analytic refinement is unbiased by default.
            segment_tick_anchor_bias = 0.0
        self.segment_tick_anchor_bias = float(np.clip(segment_tick_anchor_bias, 0.0, 1.0))
        if info_gain_target not in ("state", "report"):
            raise ValueError("info_gain_target must be 'state' or 'report'")
        self.info_gain_target = str(info_gain_target)
        self.report_action_instant = bool(report_action_instant)
        self.policy_eval_mode = policy_eval_mode
        self.use_log_prefs = use_log_prefs
        self.urgency_slope = float(urgency_slope)

        if self.mem_retention < 0 or self.mem_forget < 0 or self.mem_retention + self.mem_forget > 1.0:
            raise ValueError("Require 0 <= retention, forget and retention+forget <= 1")

        self.height_grid = make_height_grid(self.n_ticks, self.height_step)
        validate_height_grid_stage1(self.height_grid, self.n_ticks)
        self.avg_grid = make_average_grid(self.height_grid)
        self.report_step = 0.1
        self.report_grid = make_report_grid(self.n_ticks, self.report_step)

        self.n_h = len(self.height_grid)
        self.n_mem = 1 + self.n_h
        self.n_reg = 5
        self.n_reports = len(self.report_grid)
        self.report_start = 4
        self.n_actions = self.report_start + self.n_reports

        self.rel_names = ["Null", "OnTick", "LowerHalf", "Midpoint", "UpperHalf"]
        self.axis_names = ["Null"] + [f"Tick{k}" for k in range(self.n_ticks)] + [f"LowerTick{k}" for k in range(self.n_ticks - 1)]
        self.seg_names = ["Null"] + [f"Seg{k}" for k in range(10)]
        self.fb_names = ["Null", "Incorrect", "Correct"]
        self.register_names = ["UNSET", "BAR0", "BAR1", "BAR0_ANCHORED", "BAR1_ANCHORED"]
        self.mem_names = ["UNSET"] + [fmt_num(v) for v in self.height_grid]
        self.report_names = [fmt_num(v) for v in self.report_grid]
        self.policies = build_policy_library(self.n_reports, horizon=self.policy_len)

        self._build_decay_kernel()
        self._build_report_projection()
        self._build_action_likelihoods()
        self.reset_beliefs()

    def _build_decay_kernel(self):
        D = np.zeros((self.n_mem, self.n_mem), dtype=DTYPE)
        D[0, 0] = 1.0
        value_states = np.arange(self.n_h, dtype=DTYPE)
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

        # Write kernel: only open an UNSET memory; otherwise preserve its current content.
        self.write_dist = np.zeros(self.n_mem, dtype=DTYPE)
        self.write_dist[1:] = 1.0 / self.n_h
        self.mem_write = np.zeros((self.n_mem, self.n_mem), dtype=DTYPE)
        self.mem_write[:, 0] = self.write_dist
        for m in range(1, self.n_mem):
            self.mem_write[m, m] = 1.0

    def _build_report_projection(self):
        """
        Deterministic map from the full hidden-state tensor onto the
        task-relevant one-decimal report distribution.

        Hidden states with an UNSET bar memory are treated as maximally
        uninformative over the report grid.
        """
        n_flat = self.n_reg * self.n_mem * self.n_mem
        proj = np.zeros((self.n_reports, n_flat), dtype=DTYPE)
        uniform = np.ones(self.n_reports, dtype=DTYPE) / self.n_reports
        idx = 0
        for reg in range(self.n_reg):
            for m0 in range(self.n_mem):
                for m1 in range(self.n_mem):
                    if m0 == 0 or m1 == 0:
                        proj[:, idx] = uniform
                    else:
                        avg = 0.5 * (float(self.height_grid[m0 - 1]) + float(self.height_grid[m1 - 1]))
                        report_value = round_half_up_to_step(avg, self.report_step)
                        report_idx = value_to_index(report_value, self.report_grid)
                        proj[report_idx, idx] = 1.0
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
        shape = (self.n_reg, self.n_mem, self.n_mem)
        shape_flat = int(np.prod(shape))
        self.obs_tables = {}

        # LOOK_BAR0 depends on mem0 value only; UNSET -> Null
        table = np.zeros((5, *shape), dtype=DTYPE)
        for reg in range(self.n_reg):
            for m0 in range(self.n_mem):
                if m0 == 0:
                    table[0, reg, m0, :] = 1.0
                else:
                    p_rel = noisy_rel_obs_distribution(
                        float(self.height_grid[m0 - 1]),
                        self.height_grid,
                        self.n_ticks,
                        self.bar_obs_sigma,
                        0.0,   # agent does not know it is biased
                    )
                    table[:, reg, m0, :] = p_rel[:, None]
        self.obs_tables[LOOK_BAR0] = {"kind": "rel", "table": table.reshape(5, shape_flat), "pref": None}

        # LOOK_BAR1 depends on mem1 value only; UNSET -> Null
        table = np.zeros((5, *shape), dtype=DTYPE)
        for reg in range(self.n_reg):
            for m1 in range(self.n_mem):
                if m1 == 0:
                    table[0, reg, :, m1] = 1.0
                else:
                    p_rel = noisy_rel_obs_distribution(
                        float(self.height_grid[m1 - 1]),
                        self.height_grid,
                        self.n_ticks,
                        self.bar_obs_sigma,
                        0.0,   # agent does not know it is biased
                    )
                    table[:, reg, :, m1] = p_rel[:, None]
        self.obs_tables[LOOK_BAR1] = {"kind": "rel", "table": table.reshape(5, shape_flat), "pref": None}

        # LOOK_AXIS depends on anchored/selected register and corresponding memory
        n_axis = 1 + self.n_ticks + (self.n_ticks - 1)
        table = np.zeros((n_axis, *shape), dtype=DTYPE)
        for reg in range(self.n_reg):
            for m0 in range(self.n_mem):
                for m1 in range(self.n_mem):
                    if reg in (SEL_BAR0, SEL_BAR0_ANCHORED) and m0 > 0:
                        kind, k = classify_axis_anchor(float(self.height_grid[m0 - 1]), self.n_ticks)
                        axis = 1 + k if kind == "tick" else 1 + self.n_ticks + k
                        table[axis, reg, m0, m1] = 1.0
                    elif reg in (SEL_BAR1, SEL_BAR1_ANCHORED) and m1 > 0:
                        kind, k = classify_axis_anchor(float(self.height_grid[m1 - 1]), self.n_ticks)
                        axis = 1 + k if kind == "tick" else 1 + self.n_ticks + k
                        table[axis, reg, m0, m1] = 1.0
                    else:
                        table[0, reg, m0, m1] = 1.0
        self.obs_tables[LOOK_AXIS] = {"kind": "axis", "table": table.reshape(n_axis, shape_flat), "pref": None}

        # LOOK_SEGMENT informative only in anchored states for the corresponding memory
        table = np.zeros((11, *shape), dtype=DTYPE)
        for reg in range(self.n_reg):
            for m0 in range(self.n_mem):
                for m1 in range(self.n_mem):
                    if reg == SEL_BAR0_ANCHORED and m0 > 0:
                        if self.segment_tick_anchor_bias > 0.0:
                            p_seg = biased_segment_obs_distribution(
                                float(self.height_grid[m0 - 1]), 
                                self.height_grid, 
                                self.n_ticks,
                                # self.segment_tick_anchor_bias, 
                                0,  # agent isnt aware of bias
                                self.bar_obs_sigma
                            )
                            table[:, reg, m0, m1] = p_seg
                        else:
                            seg = classify_segment_index(float(self.height_grid[m0 - 1]))
                            p = discrete_gaussian(seg, 10, self.segment_sigma)
                            table[1:, reg, m0, m1] = p
                    elif reg == SEL_BAR1_ANCHORED and m1 > 0:
                        if self.segment_tick_anchor_bias > 0.0:
                            p_seg = biased_segment_obs_distribution(
                                float(self.height_grid[m1 - 1]), 
                                self.height_grid, 
                                self.n_ticks,
                                # self.segment_tick_anchor_bias, 
                                0,  # agent isnt aware of bias
                                self.bar_obs_sigma
                            )
                            table[:, reg, m0, m1] = p_seg
                        else:
                            seg = classify_segment_index(float(self.height_grid[m1 - 1]))
                            p = discrete_gaussian(seg, 10, self.segment_sigma)
                            table[1:, reg, m0, m1] = p
                    else:
                        table[0, reg, m0, m1] = 1.0
        self.obs_tables[LOOK_SEGMENT] = {"kind": "seg", "table": table.reshape(11, shape_flat), "pref": None}

        # REPORT_r depends on memory contents, not true bars
        pref = np.array([0.0, -80.0, 12.0], dtype=DTYPE)
        for r in range(self.n_reports):
            table = np.zeros((3, *shape), dtype=DTYPE)
            for reg in range(self.n_reg):
                for m0 in range(self.n_mem):
                    for m1 in range(self.n_mem):
                        if m0 == 0 or m1 == 0:
                            table[1, reg, m0, m1] = 1.0
                        else:
                            avg = 0.5 * (float(self.height_grid[m0 - 1]) + float(self.height_grid[m1 - 1]))
                            report_value = round_half_up_to_step(avg, self.report_step)
                            report_idx = value_to_index(report_value, self.report_grid)
                            table[2 if r == report_idx else 1, reg, m0, m1] = 1.0
            self.obs_tables[self.report_start + r] = {"kind": "fb", "table": table.reshape(3, shape_flat), "pref": pref}

    def reset_beliefs(self):
        self.q = np.zeros((self.n_reg, self.n_mem, self.n_mem), dtype=DTYPE)
        self.q[SEL_UNSET, 0, 0] = 1.0
        self.trial_step = 0
        self.last_q_pi = None
        self.last_scores = None
        self.last_best_policy_idx = 0

    def transition(self, q: np.ndarray, action: int) -> np.ndarray:
        q = np.asarray(q, dtype=DTYPE)
        out = np.zeros_like(q)

        if action == LOOK_BAR0:
            prev_m0 = q.sum(axis=(0, 2))
            next_m0 = normalize(self.mem_write @ prev_m0)
            prev_m1 = q.sum(axis=(0, 1))
            decayed_m1 = normalize(self.mem_decay @ prev_m1)
            out[SEL_BAR0, :, :] = np.outer(next_m0, decayed_m1)
            return out

        if action == LOOK_BAR1:
            prev_m0 = q.sum(axis=(0, 2))
            decayed_m0 = normalize(self.mem_decay @ prev_m0)
            prev_m1 = q.sum(axis=(0, 1))
            next_m1 = normalize(self.mem_write @ prev_m1)
            out[SEL_BAR1, :, :] = np.outer(decayed_m0, next_m1)
            return out

        if action == LOOK_AXIS:
            # Both memories decay. Register transitions to anchored if currently selecting a bar.
            decayed = np.zeros_like(q)
            for reg in range(self.n_reg):
                decayed[reg] = self.mem_decay @ q[reg] @ self.mem_decay.T
            out[SEL_UNSET] += decayed[SEL_UNSET]
            out[SEL_BAR0_ANCHORED] += decayed[SEL_BAR0] + decayed[SEL_BAR0_ANCHORED]
            out[SEL_BAR1_ANCHORED] += decayed[SEL_BAR1] + decayed[SEL_BAR1_ANCHORED]
            return normalize(out)

        # REPORT is modeled as an instantaneous commitment based on the
        # current memory contents, so planning a report does not incur an
        # extra decay step before feedback.
        if action >= self.report_start and self.report_action_instant:
            return q.copy()

        # LOOK_SEGMENT and (optionally) REPORT actions: register holds, both memories decay.
        for reg in range(self.n_reg):
            out[reg] = self.mem_decay @ q[reg] @ self.mem_decay.T
        return normalize(out)

    def action_cost(self, action: int, time_offset: int = 0) -> float:
        """Time-dependent non-report cost with linear urgency."""
        if action >= self.report_start:
            return 0.0
        elapsed = self.trial_step + int(time_offset)
        return self.non_report_cost + self.urgency_slope * elapsed

    def forced_report_action(self, rule: str = "map") -> int:
        """Choose a report action directly from the current posterior over reports."""
        q_rep = self.q_avg()
        if rule == "mean":
            mean_value = float(np.sum(q_rep * self.report_grid))
            report_idx = int(np.argmin(np.abs(self.report_grid - mean_value)))
        elif rule == "map":
            report_idx = int(np.argmax(q_rep))
        else:
            raise ValueError("rule must be 'map' or 'mean'")
        return self.report_start + report_idx

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
        if action in (LOOK_BAR0, LOOK_BAR1):
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

    # def score_action(self, q_pred: np.ndarray, action: int) -> float:
    #     qo = self.expected_obs(q_pred, action)
    #     utility = 0.0
    #     pref = self.obs_tables[action]["pref"]
    #     if pref is not None:
    #         utility = float(qo @ pref)

    #     info_gain = 0.0
    #     if self.use_states_info_gain:
    #         if self.info_gain_target == "state":
    #             H_prior = entropy(q_pred.reshape(-1))
    #             H_post_exp = 0.0
    #             table = self.obs_tables[action]["table"]
    #             q_flat = q_pred.reshape(-1)
    #             for o_idx in range(table.shape[0]):
    #                 po = float((table[o_idx] * q_flat).sum())
    #                 if po <= 0.0:
    #                     continue
    #                 post = (table[o_idx] * q_flat) / po
    #                 H_post_exp += po * entropy(post)
    #         else:
    #             H_prior = entropy(self.q_report_from(q_pred))
    #             H_post_exp = self._expected_report_entropy_after_action(q_pred, action)
    #         info_gain = H_prior - H_post_exp

    #     cost = 0.0 if action >= self.report_start else self.non_report_cost
    #     return utility + info_gain - cost
    
    def immediate_value(
        self,
        q: np.ndarray,
        action: int,
        use_log_prefs: bool = False,
        time_offset: int = 0,
    ):
        """
        One-step policy value from posterior q under action.

        Returns
        -------
        q_pred : predicted belief after transition, before observation
        qo     : predictive observation distribution
        val    : immediate value = extrinsic + epistemic - cost
        """
        q_pred = self.transition(q, action)
        qo = self.expected_obs(q_pred, action)

        # -----------------------------------------------------------
        # Extrinsic term
        # -----------------------------------------------------------
        pref = self.obs_tables[action]["pref"]
        extrinsic = 0.0
        if pref is not None:
            if use_log_prefs:
                # Only use this if pref is intentionally treated as logits over outcomes
                p_pref = softmax(pref)
                extrinsic = float(qo @ np.log(p_pref + EPS))
            else:
                # Backward-compatible with the working 1dp scripts
                extrinsic = float(qo @ pref)

        # -----------------------------------------------------------
        # Epistemic term
        # -----------------------------------------------------------
        epistemic = 0.0
        if self.use_states_info_gain:
            table = self.obs_tables[action]["table"]

            if self.info_gain_target == "state":
                H_prior = entropy(q_pred.reshape(-1))
                H_post_exp = 0.0
                for o_idx, po in enumerate(qo):
                    if po <= EPS:
                        continue
                    q_post = self.posterior_given_obs(q_pred, action, o_idx)
                    H_post_exp += float(po) * entropy(q_post.reshape(-1))
            else:
                # Preserve the task-aligned 1dp fix:
                # compute epistemic value over the rounded report distribution
                H_prior = entropy(self.q_report_from(q_pred))
                H_post_exp = 0.0
                for o_idx, po in enumerate(qo):
                    if po <= EPS:
                        continue
                    q_post = self.posterior_given_obs(q_pred, action, o_idx)
                    H_post_exp += float(po) * entropy(self.q_report_from(q_post))

            epistemic = H_prior - H_post_exp

        # -----------------------------------------------------------
        # Action cost
        # -----------------------------------------------------------
        cost = self.action_cost(action, time_offset=time_offset)

        return q_pred, qo, extrinsic + epistemic - cost

    def evaluate_policy(self, policy: np.ndarray, use_log_prefs: bool | None = None) -> float:
        """
        Fast open-loop rollout evaluation.

        This still scores both:
        - extrinsic value
        - epistemic value

        but it does so without branching over future observations.
        It rolls forward through predicted states only.
        """
        if use_log_prefs is None:
            use_log_prefs = self.use_log_prefs

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

            # rollout mode stays open-loop:
            # continue from the predictive state, not an observation-conditioned posterior
            q_roll = q_pred

        return float(score)

    def evaluate_policy_recursive(
        self,
        q: np.ndarray,
        policy: np.ndarray,
        depth: int = 0,
        use_log_prefs: bool = False,
    ) -> float:
        """
        Recursive expected policy value with posterior branching over observations.
        """
        if depth >= len(policy):
            return 0.0

        action = int(policy[depth])
        q_pred, qo, immediate = self.immediate_value(
            q, action, use_log_prefs=use_log_prefs, time_offset=depth
        )

        # Terminal after report
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
                dtype=DTYPE
            )
        elif policy_eval_mode == "rollout":
            scores = np.array(
                [self.evaluate_policy(pol, use_log_prefs=use_log_prefs) for pol in self.policies],
                dtype=DTYPE,
            )

        # scores are interpreted as higher-is-better (i.e. -G)
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
        q_reg = self.q.sum(axis=(1, 2))
        q_mem0 = self.q.sum(axis=(0, 2))
        q_mem1 = self.q.sum(axis=(0, 1))
        return q_reg, q_mem0, q_mem1

    def q_avg(self) -> np.ndarray:
        return self.q_report_from(self.q)

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
    if action == LOOK_BAR0:
        return "LOOK_BAR0"
    if action == LOOK_BAR1:
        return "LOOK_BAR1"
    if action == LOOK_AXIS:
        return "LOOK_AXIS"
    if action == LOOK_SEGMENT:
        return "LOOK_SEGMENT"
    return f"REPORT {meta.report_names[action - meta.report_start]}"


def run_sim(
    bar_heights=(1.2, 3.7),
    n_ticks: int = 5,
    height_step: float = 0.1,
    T: int = 16,
    policy_len: int = 4,
    gamma: float = 8.0,
    seed: int = 1,
    mem_retention: float = 0.97,
    mem_forget: float = 0.005,
    mem_sigma: float = 0.25,
    non_report_cost: float = 0.35,
    segment_sigma: float = 0.45,
    course_tick_anchor_bias: float = 0.0,
    bar_obs_sigma: float = 0.08,
    segment_tick_anchor_bias: float = 0.0,
    policy_eval_mode: str = 'rollout',
    urgency_slope: float = 0.0,
    force_report_at_deadline: bool = True,
    forced_report_rule: str = 'map',
    verbose=True,
    action_selection="deterministic"
):
    np.random.seed(seed)

    env = DecimalSegmentEnv(
        bar_heights=bar_heights,
        n_ticks=n_ticks,
        height_step=height_step,
        segment_sigma=segment_sigma,
        course_tick_anchor_bias=course_tick_anchor_bias,
        bar_obs_sigma=bar_obs_sigma,
        segment_tick_anchor_bias=segment_tick_anchor_bias,
        info_gain_target="report",
        report_action_instant=True,
    )
    planner = SlowMemoryDecayTenthsPlanner(
        n_ticks=n_ticks,
        height_step=height_step,
        policy_len=policy_len,
        gamma=gamma,
        use_states_info_gain=True,
        action_selection=action_selection,
        non_report_cost=non_report_cost,
        mem_retention=mem_retention,
        mem_forget=mem_forget,
        mem_sigma=mem_sigma,
        segment_sigma=segment_sigma,
        course_tick_anchor_bias=course_tick_anchor_bias,
        bar_obs_sigma=bar_obs_sigma,
        segment_tick_anchor_bias=segment_tick_anchor_bias,
        policy_eval_mode=policy_eval_mode, 
        urgency_slope=urgency_slope,
    )
    meta = planner.meta()
    obs = (0, 0, 0, 0)

    if verbose:
        print("\n--- Known-action Active Inference (SLOW MEMORY-DECAY + TENTH SEGMENTS) ---\n")
        print(f"Allowed heights : {meta.mem_names[1:]}")
        print(f"Allowed averages: {meta.report_names}")
        print(f"True bars: {list(env.bar_heights)} | True avg: {fmt_num(env.true_avg)} | Rounded target: {fmt_num(env.true_report)}")
        print(f"Start env.register_state: {meta.register_names[env.register_state]}")
        print(f"Policies in library: {len(meta.policies)}")
        print(
            f"mem_retention={mem_retention} | mem_forget={mem_forget} | mem_sigma={mem_sigma} | "
            f"segment_sigma={segment_sigma} | non_report_cost={non_report_cost} | "
            f"course_tick_anchor_bias={course_tick_anchor_bias} | bar_obs_sigma={bar_obs_sigma} | "
            f"segment_tick_anchor_bias={segment_tick_anchor_bias} | urgency_slope={urgency_slope} | "
            f"force_report_at_deadline={force_report_at_deadline}"
        )

    for t in range(T):
        q_reg, q_mem0, q_mem1 = planner.q_marginals()
        q_avg = planner.q_avg()

        if verbose:
            print(f"\nt={t}  obs={obs}")
            print(f"  env.register_state : {meta.register_names[env.register_state]}")
            print(f"  q(register_state)  : {np.round(q_reg, 3)}  MAP={meta.register_names[int(np.argmax(q_reg))]}")
            print(f"  mem_bar0 beliefs   : {pretty_top_k(q_mem0, meta.mem_names, k=6)}")
            print(f"  mem_bar1 beliefs   : {pretty_top_k(q_mem1, meta.mem_names, k=6)}")
            print(f"  avg beliefs        : {pretty_top_k(q_avg, meta.report_names, k=6)}")

        t0 = time.perf_counter()
        if force_report_at_deadline and t == T - 1:
            action = planner.forced_report_action(rule=forced_report_rule)
            best_policy_idx = None
            dt = 0.0
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
            if best_policy_idx is None:
                print("  best policy        : ['FORCED_REPORT_AT_DEADLINE']")
            else:
                print(f"  best policy        : {[action_to_name(a, meta) for a in meta.policies[best_policy_idx]]}")
            print(f"  action             : {action_name}")
            print(f"  rel_obs            : {meta.rel_names[obs[0]]}")
            print(f"  axis_obs           : {meta.axis_names[obs[1]]}")
            print(f"  seg_obs            : {meta.seg_names[obs[2]]}")
            print(f"  fb_obs             : {meta.fb_names[obs[3]]}")

        if action >= meta.report_start:
            print("  feedback =>", "CORRECT" if obs[3] == 2 else "INCORRECT")
            if not verbose:
                print(f"  action             : {action_name}")
            break


if __name__ == "__main__":
    # for t in range(7, 16):    
    #     print(f"T = {t}")
    run_sim(
        bar_heights=(2.4, 1.9),
        # bar_heights=(1.1, 2.5),
        # bar_heights=(2.0, 2.2),
        seed=3,
        # T=t,
        T=10,
        # 
        n_ticks=4,
        height_step=0.1,
        policy_len=4,
        gamma=8.0,
        policy_eval_mode='rollout',
        # info_gain_target = "report",
        # report_action_instant = True,
        # 
        bar_obs_sigma=0.00,
        # 
        mem_retention=0.97,
        # mem_retention=0.90,
        mem_forget=0.005,
        mem_sigma=0.25,
        # mem_sigma=5.0,
        # 
        non_report_cost=0.35,
        # 
        segment_sigma=0.045,
        # 
        course_tick_anchor_bias=0.00,
        segment_tick_anchor_bias=0.,
        # policy_eval_mode='branching'
        # verbose=False
        # action_selection='stochastic'
    )
