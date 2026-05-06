"""
social_nav_task.py
==================
MoLMoSpaces task definition for socially-compliant robot navigation.

Extends the base navigation task with social metrics:
    - goal_reached      : did the robot reach the goal?
    - social_intrusions : how many times did the robot enter someone's personal space?
    - path_length       : total distance travelled (metres)
    - time_steps        : steps taken

Usage in experiments/social_nav/run_pipeline.py — the task owns the success
criteria and metrics; the MPPI controller in mppi_nav.py owns the control loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SocialNavConfig:
    """Hyperparameters for one social-nav episode."""

    # --- Goal ---
    goal_xy: tuple[float, float] = (0.0, 0.0)
    goal_radius: float = 0.20          # metres; reaching this = success

    # --- Social metrics ---
    personal_space_radius: float = 0.80  # metres; intrusion threshold per person
    max_steps: int = 500

    # --- Starting pose ---
    start_xy: tuple[float, float] = (-1.4, 4.6)
    start_yaw_deg: float = 0.0


# ---------------------------------------------------------------------------
# Episode metrics
# ---------------------------------------------------------------------------

@dataclass
class SocialNavMetrics:
    goal_reached: bool = False
    steps: int = 0
    path_length: float = 0.0           # metres
    social_intrusions: int = 0         # cumulative intrusion events
    min_human_distance: float = math.inf   # closest approach to any human
    final_distance_to_goal: float = math.inf

    def as_dict(self) -> dict:
        return {
            "goal_reached":           self.goal_reached,
            "steps":                  self.steps,
            "path_length":            round(self.path_length, 3),
            "social_intrusions":      self.social_intrusions,
            "min_human_distance":     round(self.min_human_distance, 3),
            "final_dist_to_goal":     round(self.final_distance_to_goal, 3),
        }


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

class SocialNavTask:
    """
    Stateful social navigation task for one episode.

    The task tracks position history and social metrics; the MPPI controller
    provides velocity commands each step.

    Minimal interface so that either the standalone run_social_nav.py or
    a full MoLMoSpaces task framework can drive it.
    """

    def __init__(
        self,
        config: SocialNavConfig,
        human_positions: list[tuple[float, float]],
    ) -> None:
        self.cfg = config
        self.human_positions = human_positions
        self._metrics = SocialNavMetrics()
        self._prev_xy: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Episode control
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._metrics = SocialNavMetrics()
        self._prev_xy = np.array(self.cfg.start_xy, dtype=np.float32)

    def step(self, robot_xy: np.ndarray) -> None:
        """
        Record one navigation step.

        Parameters
        ----------
        robot_xy : shape (2,) float, current robot position in world frame.
        """
        xy = np.asarray(robot_xy, dtype=np.float32)
        self._metrics.steps += 1

        # Path length
        if self._prev_xy is not None:
            self._metrics.path_length += float(np.linalg.norm(xy - self._prev_xy))
        self._prev_xy = xy.copy()

        # Social intrusions
        for hx, hy in self.human_positions:
            d = float(np.hypot(xy[0] - hx, xy[1] - hy))
            if d < self._metrics.min_human_distance:
                self._metrics.min_human_distance = d
            if d < self.cfg.personal_space_radius:
                self._metrics.social_intrusions += 1

        # Goal check
        dist = float(np.linalg.norm(np.array(self.cfg.goal_xy, dtype=np.float32) - xy))
        self._metrics.final_distance_to_goal = dist
        if dist <= self.cfg.goal_radius:
            self._metrics.goal_reached = True

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def goal_xy(self) -> np.ndarray:
        return np.array(self.cfg.goal_xy, dtype=np.float32)

    @property
    def is_done(self) -> bool:
        return (
            self._metrics.goal_reached
            or self._metrics.steps >= self.cfg.max_steps
        )

    @property
    def metrics(self) -> SocialNavMetrics:
        return self._metrics

    def print_summary(self) -> None:
        m = self._metrics
        status = "REACHED" if m.goal_reached else "FAILED"
        print(
            f"\n[SocialNavTask] {status}  "
            f"steps={m.steps}  dist_to_goal={m.final_distance_to_goal:.3f}m  "
            f"path={m.path_length:.2f}m  "
            f"intrusions={m.social_intrusions}  "
            f"min_human_dist={m.min_human_distance:.3f}m"
        )
