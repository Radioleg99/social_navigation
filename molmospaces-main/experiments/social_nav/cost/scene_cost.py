"""
Scene Cost (global, static)
----------------------------
Two types of hard constraints (infinite cost) + soft navigation objectives:

  1. Wall / obstacle collision  — robot enters an occupied cell
  2. Human body collision       — robot enters a person's physical body radius

Both are treated as infinite cost: MPPI trajectories that hit either get
effectively zero weight, so the planner never chooses them.

Soft terms: goal distance, clearance gradient, control effort, smoothness.
"""

from __future__ import annotations

import math

import numpy as np
import torch

# Effectively infinite for float32 MPPI (exp(-1e6/λ) ≈ 0 for any λ ≤ 100)
_INF_COST = 1e6


class SceneCost:
    """
    Precomputes map tensors once, evaluates per-trajectory cost each MPPI step.

    running(state, action, t) → per-sample cost  (N,)
    terminal(states, actions) → per-sample terminal cost (N,)
    """

    def __init__(
        self,
        scene_map,
        grid_free: np.ndarray,
        distance_transform: np.ndarray,
        grid_spacing: float,
        downscale: int,
        # --- hard constraints ---
        human_positions: list[tuple[float, float]] | None = None,
        human_radius: float = 0.30,          # physical body radius (m)
        human_yaws_deg: list[float] | None = None,
        human_states: list[str] | None = None,
        walking_soft_weight: float = 8.0,
        walking_front_sigma: float = 0.85,
        walking_back_sigma: float = 0.25,
        walking_side_sigma: float = 0.35,
        # --- soft weights ---
        goal_stage_weight: float       = 4.0,
        goal_terminal_weight: float    = 20.0,
        heading_terminal_weight: float = 0.5,
        clearance_threshold: float     = 0.28,
        clearance_weight: float        = 25.0,
        control_weight: float          = 0.15,
        smoothness_weight: float       = 1.5,
        device: str | torch.device     = "cpu",
        dtype: torch.dtype             = torch.float32,
    ) -> None:
        self.grid_spacing          = float(grid_spacing)
        self.downscale             = int(downscale)
        self.goal_stage_weight     = float(goal_stage_weight)
        self.goal_terminal_weight  = float(goal_terminal_weight)
        self.heading_terminal_weight = float(heading_terminal_weight)
        self.clearance_threshold   = float(clearance_threshold)
        self.clearance_weight      = float(clearance_weight)
        self.control_weight        = float(control_weight)
        self.smoothness_weight     = float(smoothness_weight)
        self.human_radius          = float(human_radius)
        self.walking_soft_weight   = float(walking_soft_weight)
        self.walking_front_sigma   = float(walking_front_sigma)
        self.walking_back_sigma    = float(walking_back_sigma)
        self.walking_side_sigma    = float(walking_side_sigma)
        self.device = torch.device(device)
        self.dtype  = dtype

        self.world_to_map_t = torch.as_tensor(
            scene_map.world_to_map, dtype=dtype, device=self.device
        )
        self.grid_free_t = torch.as_tensor(grid_free, dtype=torch.bool, device=self.device)
        self.distance_transform_t = torch.as_tensor(
            distance_transform, dtype=dtype, device=self.device
        )

        # Human positions tensor: (M, 2) — static for the episode
        if human_positions:
            self._human_pos_t = torch.tensor(
                human_positions, dtype=dtype, device=self.device
            )  # (M, 2)
        else:
            self._human_pos_t = None
        self._human_fx_t: torch.Tensor | None = None
        self._human_fy_t: torch.Tensor | None = None
        self._walking_mask_t: torch.Tensor | None = None
        self._update_human_motion_tensors(human_yaws_deg, human_states)

        # Updated each command() call
        self._goal_xy_t     = torch.zeros(2, dtype=dtype, device=self.device)
        self._prev_action_t = torch.zeros(2, dtype=dtype, device=self.device)

    def set_goal(self, goal_xy: np.ndarray | torch.Tensor) -> None:
        self._goal_xy_t = torch.as_tensor(goal_xy, dtype=self.dtype, device=self.device)

    def set_prev_action(self, action: np.ndarray | torch.Tensor) -> None:
        self._prev_action_t = torch.as_tensor(action, dtype=self.dtype, device=self.device)

    def update_humans(
        self,
        human_positions: list[tuple[float, float]],
        human_yaws_deg: list[float] | None = None,
        human_states: list[str] | None = None,
    ) -> None:
        """Update human positions (call when humans move)."""
        if human_positions:
            self._human_pos_t = torch.tensor(
                human_positions, dtype=self.dtype, device=self.device
            )
        else:
            self._human_pos_t = None
        self._update_human_motion_tensors(human_yaws_deg, human_states)

    # ------------------------------------------------------------------
    # MPPI cost interface
    # ------------------------------------------------------------------

    def running(self, state: torch.Tensor, action: torch.Tensor, _t: int) -> torch.Tensor:
        """Per-step running cost. state: (N, state_dim), action: (N, 2)."""
        pos = state[:, :2]

        # --- Hard constraint 1: wall / obstacle collision → infinite ---
        clearance, wall_collision = self._clearance_and_collision(pos)
        cost = _INF_COST * wall_collision.to(self.dtype)

        # --- Hard constraint 2: human body collision → infinite ---
        if self._human_pos_t is not None:
            human_collision = self._human_collision(pos)
            cost = cost + _INF_COST * human_collision.to(self.dtype)
            cost = cost + self._walking_directional_cost(pos)

        # --- Soft: clearance gradient (reward keeping distance from walls) ---
        clearance_violation = torch.clamp(self.clearance_threshold - clearance, min=0.0)
        cost = cost + self.clearance_weight * torch.square(clearance_violation)

        # --- Soft: goal distance ---
        goal_dist = torch.linalg.norm(pos - self._goal_xy_t.unsqueeze(0), dim=-1)
        cost = cost + self.goal_stage_weight * goal_dist

        # --- Soft: control effort + smoothness ---
        prev_action = state[:, 5:7] if state.shape[-1] >= 7 else self._prev_action_t.expand_as(action)
        control_delta = action - prev_action
        cost = cost + self.control_weight    * (torch.square(action[:, 0]) + 0.25 * torch.square(action[:, 1]))
        cost = cost + self.smoothness_weight * torch.sum(torch.square(control_delta), dim=-1)

        return cost

    def terminal(self, states: torch.Tensor, _actions: torch.Tensor) -> torch.Tensor:
        """Terminal cost. states: (N, H, state_dim) or (1, N, H, state_dim)."""
        final = states[0, :, -1, :] if states.dim() == 4 else states[:, -1, :]
        goal_vec   = self._goal_xy_t.unsqueeze(0) - final[:, :2]
        goal_dist  = torch.linalg.norm(goal_vec, dim=-1)
        desired_heading = torch.atan2(goal_vec[:, 1], goal_vec[:, 0])
        heading_error   = torch.abs(_wrap_pi(desired_heading - final[:, 2]))
        return self.goal_terminal_weight * goal_dist + self.heading_terminal_weight * heading_error

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _xy_to_grid(self, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """World xy (N,2) → downscaled grid (row, col, in_bounds)."""
        flat = xy.reshape(-1, 2)
        xyz  = torch.zeros((flat.shape[0], 3), dtype=self.dtype, device=self.device)
        xyz[:, :2] = flat
        hom  = torch.cat((xyz, torch.ones((flat.shape[0], 1), dtype=self.dtype, device=self.device)), dim=-1)
        px   = torch.round(hom @ self.world_to_map_t.T)
        grid = torch.floor(px / self.downscale).to(torch.long)
        row, col = grid[:, 0], grid[:, 1]
        H, W = self.grid_free_t.shape
        in_bounds = (row >= 0) & (row < H) & (col >= 0) & (col < W)
        return row, col, in_bounds

    def _clearance_and_collision(self, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        row, col, in_bounds = self._xy_to_grid(xy)
        N = xy.reshape(-1, 2).shape[0]
        clearance = torch.zeros(N, dtype=self.dtype, device=self.device)
        collision  = ~in_bounds
        if torch.any(in_bounds):
            free = torch.zeros_like(in_bounds)
            free[in_bounds] = self.grid_free_t[row[in_bounds], col[in_bounds]]
            collision = collision | ~free
            valid = in_bounds & free
            if torch.any(valid):
                clearance[valid] = self.distance_transform_t[row[valid], col[valid]] * self.grid_spacing
        return clearance.reshape(xy.shape[:-1]), collision.reshape(xy.shape[:-1])

    def _human_collision(self, xy: torch.Tensor) -> torch.Tensor:
        """Returns bool tensor (N,): True if any human is within human_radius."""
        # xy: (N, 2),  _human_pos_t: (M, 2)
        # dist: (N, M)
        diff = xy.unsqueeze(1) - self._human_pos_t.unsqueeze(0)   # (N, M, 2)
        dist = torch.linalg.norm(diff, dim=-1)                     # (N, M)
        return torch.any(dist < self.human_radius, dim=-1)         # (N,)

    def _update_human_motion_tensors(
        self,
        human_yaws_deg: list[float] | None,
        human_states: list[str] | None,
    ) -> None:
        if self._human_pos_t is None:
            self._human_fx_t = None
            self._human_fy_t = None
            self._walking_mask_t = None
            return

        n = int(self._human_pos_t.shape[0])
        yaws = list(human_yaws_deg or [0.0] * n)[:n]
        if len(yaws) < n:
            yaws.extend([0.0] * (n - len(yaws)))
        yaw_t = torch.tensor(
            [math.radians(float(y)) for y in yaws],
            dtype=self.dtype,
            device=self.device,
        )
        self._human_fx_t = torch.cos(yaw_t)
        self._human_fy_t = torch.sin(yaw_t)

        states = list(human_states or ["idle"] * n)[:n]
        if len(states) < n:
            states.extend(["idle"] * (n - len(states)))
        self._walking_mask_t = torch.tensor(
            [str(s).lower() == "walking" for s in states],
            dtype=torch.bool,
            device=self.device,
        )

    def _walking_directional_cost(self, xy: torch.Tensor) -> torch.Tensor:
        """Soft local cost for walking humans: front > rear, not a social relation."""
        if (
            self._human_pos_t is None
            or self._human_fx_t is None
            or self._human_fy_t is None
            or self._walking_mask_t is None
            or not torch.any(self._walking_mask_t)
            or self.walking_soft_weight <= 0.0
        ):
            return torch.zeros(xy.shape[0], dtype=self.dtype, device=self.device)

        diff = xy.unsqueeze(1) - self._human_pos_t.unsqueeze(0)
        dx, dy = diff[..., 0], diff[..., 1]
        along = dx * self._human_fx_t + dy * self._human_fy_t
        perp = -dx * self._human_fy_t + dy * self._human_fx_t
        sigma_along = torch.where(
            along >= 0.0,
            torch.tensor(self.walking_front_sigma, dtype=self.dtype, device=self.device),
            torch.tensor(self.walking_back_sigma, dtype=self.dtype, device=self.device),
        )
        field = torch.exp(
            -(along ** 2) / (2 * sigma_along ** 2)
            -(perp ** 2) / (2 * self.walking_side_sigma ** 2)
        )
        field = torch.where(self._walking_mask_t.unsqueeze(0), field, torch.zeros_like(field))
        return self.walking_soft_weight * field.max(dim=-1).values

    def clearance_of(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Numpy convenience wrapper for info/debug."""
        xy_t = torch.as_tensor(xy, dtype=self.dtype, device=self.device)
        c, col = self._clearance_and_collision(xy_t)
        return c.cpu().numpy(), col.cpu().numpy()


def _wrap_pi(x: torch.Tensor) -> torch.Tensor:
    return (x + math.pi) % (2 * math.pi) - math.pi
