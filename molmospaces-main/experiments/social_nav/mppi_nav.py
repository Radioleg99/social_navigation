"""
MPPI 社交导航控制器
-------------------
组合 SceneCost（障碍/goal/控制）和 SocialCost（社交 costmap 查表），
传给 MPPIController 作为外部 cost 函数。

Cost 模块：
  experiments/social_nav/cost/scene_cost.py  — 物理/几何层
  experiments/social_nav/cost/social_cost.py — 社交层
  experiments/social_nav/cost/llm_costmap.py — 社交 costmap 生成（LLM/rule）
"""

from __future__ import annotations

import math

import numpy as np

from experiments.social_nav.cost.llm_costmap import SocialEntityParams
from experiments.social_nav.cost.scene_cost import SceneCost
from experiments.social_nav.cost.social_cost import SocialCost
from molmo_spaces.policy.solvers.navigation.mppi_core import MPPIConfig, MPPIController


class MPPINav:
    """
    差速底盘的 MPPI 导航控制器。

    用法：
        nav = MPPINav(scene_map, grid_free, distance_transform, grid_spacing, downscale, ...)
        v, w, info = nav.step(robot_xy_theta, goal_xy)
        nav.reset()
        nav.update_social_costmap(new_costmap)   # LLM 刷新时调用
    """

    def __init__(
        self,
        scene_map,
        grid_free: np.ndarray,
        distance_transform: np.ndarray,
        grid_spacing: float,
        downscale: int,
        # ---------- MPPI 超参数 ----------
        horizon: int = 24,
        num_samples: int = 512,
        num_iters: int = 4,
        dt: float = 0.10,
        temperature: float = 1.0,
        noise_alpha: float = 0.8,
        noise_v: float = 0.10,
        noise_w_deg: float = 30.0,
        v_min: float = -0.05,
        v_max: float = 0.45,
        w_max_deg: float = 120.0,
        cruise_speed: float = 0.22,
        # ---------- Scene cost 权重 ----------
        goal_stage_weight: float      = 4.0,
        goal_terminal_weight: float   = 20.0,
        heading_terminal_weight: float = 0.5,
        clearance_threshold: float    = 0.28,
        clearance_weight: float       = 25.0,
        control_weight: float         = 0.15,
        smoothness_weight: float      = 1.5,
        # ---------- Social cost ----------
        social_params: list[SocialEntityParams] | None = None,
        social_weight: float                           = 30.0,
        social_back_scale: float                       = 1.0,
        # ---------- Human hard constraint ----------
        human_positions: list[tuple[float, float]] | None = None,
        human_radius: float = 0.30,
        human_yaws_deg: list[float] | None = None,
        human_states: list[str] | None = None,
        seed: int = 0,
    ) -> None:
        cfg = MPPIConfig(
            horizon=horizon,
            num_samples=num_samples,
            num_iters=num_iters,
            dt=dt,
            temperature=temperature,
            noise_alpha=noise_alpha,
            noise_v=noise_v,
            noise_w=math.radians(noise_w_deg),
            v_min=v_min,
            v_max=v_max,
            w_max=math.radians(w_max_deg),
            cruise_speed=cruise_speed,
        )

        scene_cost = SceneCost(
            scene_map=scene_map,
            grid_free=grid_free,
            distance_transform=distance_transform,
            grid_spacing=grid_spacing,
            downscale=downscale,
            human_positions=human_positions,
            human_radius=human_radius,
            human_yaws_deg=human_yaws_deg,
            human_states=human_states,
            goal_stage_weight=goal_stage_weight,
            goal_terminal_weight=goal_terminal_weight,
            heading_terminal_weight=heading_terminal_weight,
            clearance_threshold=clearance_threshold,
            clearance_weight=clearance_weight,
            control_weight=control_weight,
            smoothness_weight=smoothness_weight,
        )
        self._scene_cost = scene_cost

        social_cost = SocialCost(
            params=social_params,
            weight=social_weight,
            back_scale=social_back_scale,
        )

        self._ctrl = MPPIController(
            scene_map=scene_map,
            grid_free=grid_free,
            grid_spacing=grid_spacing,
            downscale=downscale,
            config=cfg,
            rng=np.random.default_rng(seed),
            scene_cost=scene_cost,
            social_cost=social_cost,
        )
        self._social_cost = social_cost

    def reset(self) -> None:
        self._ctrl.reset()

    def update_social_params(self, params: list[SocialEntityParams]) -> None:
        """LLM 刷新后调用，热更新 social params，无需重建控制器。"""
        self._social_cost.update_params(params)

    def update_humans(
        self,
        human_positions: list[tuple[float, float]],
        human_yaws_deg: list[float] | None = None,
        human_states: list[str] | None = None,
    ) -> None:
        """人物位置变化时调用，更新 physical collision 和 walking soft cost。"""
        self._scene_cost.update_humans(human_positions, human_yaws_deg, human_states)

    def step(
        self,
        robot_state: np.ndarray,  # [x, y, theta_rad]
        goal_xy: np.ndarray,      # [x, y]
    ) -> tuple[float, float, dict]:
        action, info = self._ctrl.command(robot_state, goal_xy)
        return float(action[0]), float(action[1]), info
