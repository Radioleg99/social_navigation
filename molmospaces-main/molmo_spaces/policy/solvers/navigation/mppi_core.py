from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from pytorch_mppi import MPPI as TorchMPPI


def wrap_to_pi(angle: np.ndarray | float) -> np.ndarray | float:
    return np.arctan2(np.sin(angle), np.cos(angle))


def _wrap_to_pi_torch(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def make_downscaled_grid(scene_map, downscale: int) -> np.ndarray:
    grid = scene_map.occupancy.copy().astype(bool)
    pad_rows = (-grid.shape[0]) % downscale
    pad_cols = (-grid.shape[1]) % downscale
    padded = np.zeros((grid.shape[0] + pad_rows, grid.shape[1] + pad_cols), dtype=bool)
    padded[: grid.shape[0], : grid.shape[1]] = grid
    return (
        padded.reshape(
            padded.shape[0] // downscale,
            downscale,
            padded.shape[1] // downscale,
            downscale,
        )
        .min(axis=1)
        .min(axis=-1)
    )


def xy_to_grid(scene_map, xy: np.ndarray, downscale: int) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float32)
    flat = xy.reshape(-1, 2)
    xyz = np.zeros((flat.shape[0], 3), dtype=np.float32)
    xyz[:, :2] = flat
    px = scene_map.pos_m_to_px(xyz)
    grid = np.floor(px / downscale).astype(np.int32)
    return grid.reshape(*xy.shape[:-1], 2)


def grid_to_xy(scene_map, cell_rc: tuple[int, int], downscale: int) -> np.ndarray:
    px = np.array(
        [[cell_rc[0] * downscale + downscale / 2.0, cell_rc[1] * downscale + downscale / 2.0]],
        dtype=np.float32,
    )
    return scene_map.pos_px_to_m(px)[0, :2].astype(np.float32)


def resolve_free_xy(
    scene_map,
    grid_free: np.ndarray,
    xy: np.ndarray,
    downscale: int,
    max_search: int = 40,
) -> np.ndarray | None:
    cell = xy_to_grid(scene_map, np.asarray(xy, dtype=np.float32), downscale)
    row, col = int(cell[0]), int(cell[1])
    if 0 <= row < grid_free.shape[0] and 0 <= col < grid_free.shape[1] and grid_free[row, col]:
        return np.asarray(xy, dtype=np.float32)

    for search_range in range(1, max_search + 1):
        row_lo = max(0, row - search_range)
        row_hi = min(grid_free.shape[0] - 1, row + search_range)
        col_lo = max(0, col - search_range)
        col_hi = min(grid_free.shape[1] - 1, col + search_range)
        for cand_row in range(row_lo, row_hi + 1):
            for cand_col in range(col_lo, col_hi + 1):
                if abs(cand_row - row) != search_range and abs(cand_col - col) != search_range:
                    continue
                if grid_free[cand_row, cand_col]:
                    return grid_to_xy(scene_map, (cand_row, cand_col), downscale)
    return None


@dataclass
class MPPIConfig:
    horizon: int
    num_samples: int
    num_iters: int
    dt: float
    temperature: float
    noise_alpha: float
    noise_v: float
    noise_w: float
    v_min: float
    v_max: float
    w_max: float
    cruise_speed: float
    goal_stage_weight: float
    goal_terminal_weight: float
    heading_terminal_weight: float
    collision_penalty: float
    clearance_threshold: float
    clearance_weight: float
    control_weight: float
    smoothness_weight: float
    social_weight: float = 0.0


class _CorrelatedMPPI(TorchMPPI):
    def __init__(self, *args, noise_alpha: float = 0.0, **kwargs) -> None:
        self.noise_alpha = float(noise_alpha)
        super().__init__(*args, **kwargs)

    def _sample_noise(self, shape):
        if self.noise_alpha <= 0.0 or len(shape) not in (1, 2):
            return super()._sample_noise(shape)

        horizon = int(shape[-1])
        if horizon <= 0:
            return super()._sample_noise(shape)

        innovation_scale = math.sqrt(max(1.0 - self.noise_alpha**2, 1e-6))
        noise_shape = (*shape, self.nu)
        z = torch.randn(*noise_shape, device=self.d, dtype=self.dtype)
        if self._diagonal_sigma:
            base = z * self._noise_sigma_sqrt_diag
        else:
            base = z @ self._noise_sigma_chol.T

        noise = torch.empty_like(base)
        mean = self.noise_mu
        noise[..., 0, :] = base[..., 0, :] + mean
        for t in range(1, horizon):
            innovation = base[..., t, :] * innovation_scale
            noise[..., t, :] = (
                self.noise_alpha * noise[..., t - 1, :]
                + innovation
                + (1.0 - self.noise_alpha) * mean
            )
        return noise


class MPPIController:
    def __init__(
        self,
        scene_map,
        grid_free: np.ndarray,
        distance_transform: np.ndarray,
        grid_spacing: float,
        downscale: int,
        config: MPPIConfig,
        rng: np.random.Generator,
        social_costmap: np.ndarray | None = None,
    ) -> None:
        self.scene_map = scene_map
        self.grid_free = grid_free
        self.distance_transform = distance_transform
        self.grid_spacing = float(grid_spacing)
        self.downscale = int(downscale)
        self.cfg = config
        self.rng = rng

        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.world_to_map_t = torch.as_tensor(scene_map.world_to_map, dtype=self.dtype, device=self.device)
        self.grid_free_t = torch.as_tensor(grid_free, dtype=torch.bool, device=self.device)
        self.distance_transform_t = torch.as_tensor(
            distance_transform, dtype=self.dtype, device=self.device
        )
        # social costmap：shape 同 grid_free，值域 [0,1]，0=无代价
        if social_costmap is not None:
            self.social_costmap_t = torch.as_tensor(
                social_costmap.astype(np.float32), dtype=self.dtype, device=self.device
            )
        else:
            self.social_costmap_t = None
        self._goal_xy_t = torch.zeros(2, dtype=self.dtype, device=self.device)
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.prev_action_t = torch.zeros(2, dtype=self.dtype, device=self.device)
        self._initial_action_t = torch.tensor(
            [float(config.cruise_speed), 0.0], dtype=self.dtype, device=self.device
        )
        self._controller = self._build_controller()

    def reset(self) -> None:
        self.prev_action.fill(0.0)
        self.prev_action_t.zero_()
        self._goal_xy_t.zero_()
        self._controller.U = self._initial_control_sequence()

    def command(self, state: np.ndarray, goal_xy: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        self._goal_xy_t = torch.as_tensor(goal_xy, dtype=self.dtype, device=self.device)
        state_t = torch.as_tensor(self._augment_state(state), dtype=self.dtype, device=self.device)
        torch.manual_seed(int(self.rng.integers(0, 2**31 - 1)))

        with torch.no_grad():
            action_t = None
            for iter_idx in range(max(int(self.cfg.num_iters), 1)):
                action_t = self._controller.command(
                    state_t,
                    shift_nominal_trajectory=(iter_idx == 0),
                )

        action = action_t.detach().cpu().numpy().astype(np.float32)
        info = self._build_info()
        self.prev_action = action.copy()
        self.prev_action_t = torch.as_tensor(action, dtype=self.dtype, device=self.device)
        return action, info

    def _build_controller(self) -> _CorrelatedMPPI:
        noise_sigma = torch.diag(
            torch.tensor(
                [
                    max(float(self.cfg.noise_v) ** 2, 1e-8),
                    max(float(self.cfg.noise_w) ** 2, 1e-8),
                ],
                dtype=self.dtype,
                device=self.device,
            )
        )
        u_min = torch.tensor(
            [float(self.cfg.v_min), -float(self.cfg.w_max)],
            dtype=self.dtype,
            device=self.device,
        )
        u_max = torch.tensor(
            [float(self.cfg.v_max), float(self.cfg.w_max)],
            dtype=self.dtype,
            device=self.device,
        )
        return _CorrelatedMPPI(
            dynamics=self._dynamics_torch,
            running_cost=self._running_cost_torch,
            terminal_state_cost=self._terminal_cost_torch,
            nx=3,
            noise_sigma=noise_sigma,
            num_samples=int(self.cfg.num_samples),
            horizon=int(self.cfg.horizon),
            device=self.device,
            lambda_=max(float(self.cfg.temperature), 1e-6),
            u_min=u_min,
            u_max=u_max,
            u_init=self._initial_action_t.clone(),
            U_init=self._initial_control_sequence(),
            step_dependent_dynamics=True,
            noise_alpha=float(self.cfg.noise_alpha),
        )

    def _initial_control_sequence(self) -> torch.Tensor:
        return self._initial_action_t.repeat(int(self.cfg.horizon), 1).clone()

    def _augment_state(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32)
        return np.concatenate((state, self.prev_action, self.prev_action), dtype=np.float32)

    def _dynamics_torch(self, state: torch.Tensor, action: torch.Tensor, _t: int) -> torch.Tensor:
        if state.shape[-1] < 5:
            last_action = self.prev_action_t.expand(state.shape[0], -1)
        else:
            last_action = state[:, 3:5]

        theta = state[:, 2]
        next_state = torch.empty((state.shape[0], 7), dtype=self.dtype, device=self.device)
        next_state[:, 0] = state[:, 0] + self.cfg.dt * action[:, 0] * torch.cos(theta)
        next_state[:, 1] = state[:, 1] + self.cfg.dt * action[:, 0] * torch.sin(theta)
        next_state[:, 2] = _wrap_to_pi_torch(theta + self.cfg.dt * action[:, 1])
        next_state[:, 3:5] = action
        next_state[:, 5:7] = last_action
        return next_state

    def _running_cost_torch(self, state: torch.Tensor, action: torch.Tensor, _t: int) -> torch.Tensor:
        pos = state[:, :2]
        goal_dist = torch.linalg.norm(pos - self._goal_xy_t.unsqueeze(0), dim=-1)
        clearance, collision = self._clearance_and_collision_torch(pos)
        clearance_violation = torch.clamp(float(self.cfg.clearance_threshold) - clearance, min=0.0)

        if state.shape[-1] >= 7:
            prev_action = state[:, 5:7]
        else:
            prev_action = self.prev_action_t.expand_as(action)
        control_delta = action - prev_action

        cost = float(self.cfg.goal_stage_weight) * goal_dist
        cost = cost + float(self.cfg.clearance_weight) * torch.square(clearance_violation)
        cost = cost + float(self.cfg.collision_penalty) * collision.to(self.dtype)
        cost = cost + float(self.cfg.control_weight) * (
            torch.square(action[:, 0]) + 0.25 * torch.square(action[:, 1])
        )
        cost = cost + float(self.cfg.smoothness_weight) * torch.sum(torch.square(control_delta), dim=-1)

        # social cost：从 LLM 生成的 costmap 查表
        if self.social_costmap_t is not None and float(self.cfg.social_weight) > 0.0:
            social = self._lookup_social_cost_torch(pos)
            cost = cost + float(self.cfg.social_weight) * social

        return cost

    def _lookup_social_cost_torch(self, xy: torch.Tensor) -> torch.Tensor:
        """在 social_costmap_t 上查表，越界格子返回 0。"""
        flat_xy = xy.reshape(-1, 2)
        xyz = torch.zeros((flat_xy.shape[0], 3), dtype=self.dtype, device=self.device)
        xyz[:, :2] = flat_xy
        hom = torch.cat(
            (xyz, torch.ones((flat_xy.shape[0], 1), dtype=self.dtype, device=self.device)), dim=-1
        )
        px = torch.round(hom @ self.world_to_map_t.T)
        grid = torch.floor(px / self.downscale).to(torch.long)
        row = grid[:, 0]
        col = grid[:, 1]
        in_bounds = (
            (row >= 0)
            & (row < self.social_costmap_t.shape[0])
            & (col >= 0)
            & (col < self.social_costmap_t.shape[1])
        )
        social = torch.zeros(flat_xy.shape[0], dtype=self.dtype, device=self.device)
        if torch.any(in_bounds):
            social[in_bounds] = self.social_costmap_t[row[in_bounds], col[in_bounds]]
        return social.reshape(xy.shape[:-1])

    def _terminal_cost_torch(self, states: torch.Tensor, _actions: torch.Tensor) -> torch.Tensor:
        if states.dim() == 4:
            final_state = states[0, :, -1, :]
        else:
            final_state = states[:, -1, :]

        final_goal_vec = self._goal_xy_t.unsqueeze(0) - final_state[:, :2]
        final_goal_dist = torch.linalg.norm(final_goal_vec, dim=-1)
        desired_heading = torch.atan2(final_goal_vec[:, 1], final_goal_vec[:, 0])
        heading_error = torch.abs(_wrap_to_pi_torch(desired_heading - final_state[:, 2]))
        return (
            float(self.cfg.goal_terminal_weight) * final_goal_dist
            + float(self.cfg.heading_terminal_weight) * heading_error
        )

    def _clearance_and_collision_torch(
        self, xy: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat_xy = xy.reshape(-1, 2)
        xyz = torch.zeros((flat_xy.shape[0], 3), dtype=self.dtype, device=self.device)
        xyz[:, :2] = flat_xy
        hom = torch.cat(
            (xyz, torch.ones((flat_xy.shape[0], 1), dtype=self.dtype, device=self.device)), dim=-1
        )
        px = torch.round(hom @ self.world_to_map_t.T)
        grid = torch.floor(px / self.downscale).to(torch.long)

        row = grid[:, 0]
        col = grid[:, 1]
        in_bounds = (
            (row >= 0)
            & (row < self.grid_free_t.shape[0])
            & (col >= 0)
            & (col < self.grid_free_t.shape[1])
        )

        clearance = torch.zeros(flat_xy.shape[0], dtype=self.dtype, device=self.device)
        collision = ~in_bounds
        if torch.any(in_bounds):
            free = torch.zeros_like(in_bounds)
            free[in_bounds] = self.grid_free_t[row[in_bounds], col[in_bounds]]
            collision = collision | ~free
            valid = in_bounds & free
            if torch.any(valid):
                clearance[valid] = (
                    self.distance_transform_t[row[valid], col[valid]] * float(self.grid_spacing)
                )

        return clearance.reshape(xy.shape[:-1]), collision.reshape(xy.shape[:-1])

    def _build_info(self) -> dict[str, float]:
        best_cost = math.inf
        mean_cost = math.inf
        best_clearance = 0.0
        pred_final_goal_dist = math.inf

        if self._controller.cost_total is not None:
            costs = self._controller.cost_total.detach().cpu().numpy()
            if costs.size > 0:
                best_idx = int(np.argmin(costs))
                best_cost = float(costs[best_idx])
                mean_cost = float(np.mean(costs))
                if self._controller.states is not None:
                    best_states = self._controller.states[0, best_idx]
                    best_xy = best_states[:, :2]
                    final_xy = best_xy[-1]
                    pred_final_goal_dist = float(
                        torch.linalg.norm(final_xy - self._goal_xy_t).detach().cpu().item()
                    )
                    clearance, collision = self._clearance_and_collision_torch(best_xy)
                    safe_clearance = clearance[~collision]
                    if safe_clearance.numel() > 0:
                        best_clearance = float(torch.min(safe_clearance).detach().cpu().item())
                    else:
                        best_clearance = 0.0

        action_seq = self._controller.get_action_sequence()
        action = action_seq[0].detach().cpu().numpy().astype(np.float32)
        return {
            "best_cost": best_cost,
            "mean_cost": mean_cost,
            "best_clearance": best_clearance,
            "pred_final_goal_dist": pred_final_goal_dist,
            "action_v": float(action[0]),
            "action_w_deg": math.degrees(float(action[1])),
        }


class MultiModalMPPIController:
    """
    并行运行 N 个 MPPIController 实例，各自带不同的角速度初始偏置（mode）。
    每一步选 best_cost 最低的实例的 action 执行。

    解决标准 MPPI 在多模态代价地形（如绕行障碍/人群时左绕和右绕代价相近）
    下加权平均相消、导致机器人原地振荡的问题。

    Args:
        w_biases_deg: 各模式的角速度偏置（度/秒）。
            默认 [0, +w_max*0.35, -w_max*0.35]，即直走、左偏、右偏三个模式。
        mppi_kwargs:  与 MPPIController 完全相同的参数。
    """

    def __init__(
        self,
        w_biases_deg: list[float] | None = None,
        **mppi_kwargs,
    ) -> None:
        cfg: MPPIConfig = mppi_kwargs["config"]

        if w_biases_deg is None:
            w_max_deg = math.degrees(cfg.w_max)
            w_biases_deg = [0.0, w_max_deg * 0.35, -w_max_deg * 0.35]

        self.controllers: list[MPPIController] = []
        for w_deg in w_biases_deg:
            ctrl = MPPIController(**mppi_kwargs)
            # 设置该模式的初始 action 偏置
            w_rad = math.radians(w_deg)
            ctrl._initial_action_t = torch.tensor(
                [float(cfg.cruise_speed), w_rad],
                dtype=ctrl.dtype, device=ctrl.device,
            )
            # 用新偏置重新初始化名义轨迹
            ctrl._controller.U = ctrl._initial_control_sequence()
            self.controllers.append(ctrl)

        self._n_modes = len(w_biases_deg)
        self._active_mode: int = 0   # 上一步选中的模式（供调试用）

    def reset(self) -> None:
        for ctrl in self.controllers:
            ctrl.reset()

    def command(
        self, state: np.ndarray, goal_xy: np.ndarray
    ) -> tuple[np.ndarray, dict]:
        """
        并行运行所有模式，选 best_cost 最低的 action 执行。
        同步所有控制器的 prev_action，确保平滑代价计算一致。
        """
        results = [ctrl.command(state, goal_xy) for ctrl in self.controllers]

        best_idx = min(
            range(self._n_modes),
            key=lambda i: results[i][1].get("best_cost", math.inf),
        )
        best_action, best_info = results[best_idx]
        self._active_mode = best_idx
        best_info["active_mode"] = best_idx

        # 所有控制器同步到实际执行的 action，保证下一步平滑代价正确
        for ctrl in self.controllers:
            ctrl.prev_action = best_action.copy()
            ctrl.prev_action_t = torch.as_tensor(
                best_action, dtype=ctrl.dtype, device=ctrl.device
            )

        return best_action, best_info
