"""
MPPI 社交导航控制器
-------------------
把底层的 MPPIController（在 molmo_spaces/policy/solvers/navigation/mppi_core.py）
包一层，提供更直接的接口：给机器人当前状态 + 目标点，返回 (线速度, 角速度)。

如果以后想加 A*、社交力模型、RL 策略等，参考这个文件的接口写一个新文件就行，
run_social_nav.py 里替换一行 import 即可。
"""

from __future__ import annotations

import math

import numpy as np

from molmo_spaces.policy.solvers.navigation.mppi_core import MPPIConfig, MPPIController


class MPPINav:
    """
    差速底盘的 MPPI 导航控制器。

    用法：
        nav = MPPINav(scene_map, grid_free, distance_transform, grid_spacing, downscale, cfg)
        v, w = nav.step(robot_xy_theta, goal_xy)   # 每步调用一次
        nav.reset()                                  # 新 episode 开始时调用
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
        # ---------- 代价函数权重 ----------
        goal_stage_weight: float = 4.0,       # 每步的目标距离代价
        goal_terminal_weight: float = 20.0,   # 终端目标距离代价
        heading_terminal_weight: float = 0.5, # 终端朝向误差代价
        collision_penalty: float = 500.0,     # 碰撞惩罚
        clearance_threshold: float = 0.28,    # 期望的安全间距（米）
        clearance_weight: float = 25.0,       # 低于安全间距时的代价
        control_weight: float = 0.15,         # 控制量大小的代价（能量）
        smoothness_weight: float = 1.5,       # 控制变化率的代价（平滑）
        # ---------- Social costmap ----------
        social_costmap: np.ndarray | None = None,  # shape=grid_free.shape，值域[0,1]
        social_weight: float = 30.0,               # social cost 的缩放权重
        seed: int = 0,
    ) -> None:
        # 把所有超参数打包进 MPPIConfig（底层 dataclass）
        cfg = MPPIConfig(
            horizon=horizon,
            num_samples=num_samples,
            num_iters=num_iters,
            dt=dt,
            temperature=temperature,
            noise_alpha=noise_alpha,
            noise_v=noise_v,
            noise_w=math.radians(noise_w_deg),   # 内部用弧度
            v_min=v_min,
            v_max=v_max,
            w_max=math.radians(w_max_deg),
            cruise_speed=cruise_speed,
            goal_stage_weight=goal_stage_weight,
            goal_terminal_weight=goal_terminal_weight,
            heading_terminal_weight=heading_terminal_weight,
            collision_penalty=collision_penalty,
            clearance_threshold=clearance_threshold,
            clearance_weight=clearance_weight,
            control_weight=control_weight,
            smoothness_weight=smoothness_weight,
            social_weight=social_weight,
        )

        # 构建底层控制器
        self._ctrl = MPPIController(
            scene_map=scene_map,
            grid_free=grid_free,
            distance_transform=distance_transform,
            grid_spacing=grid_spacing,
            downscale=downscale,
            config=cfg,
            rng=np.random.default_rng(seed),
            social_costmap=social_costmap,
        )

    def reset(self) -> None:
        """新 episode 开始时调用，清空内部状态。"""
        self._ctrl.reset()

    def step(
        self,
        robot_state: np.ndarray,  # [x, y, theta_rad]，机器人当前底盘位姿
        goal_xy: np.ndarray,      # [x, y]，目标位置
    ) -> tuple[float, float, dict]:
        """
        计算下一步控制指令。

        返回：
            v      : 线速度（m/s）
            w      : 角速度（rad/s）
            info   : 调试信息字典（代价、间距等）
        """
        action, info = self._ctrl.command(robot_state, goal_xy)
        v = float(action[0])
        w = float(action[1])
        return v, w, info
