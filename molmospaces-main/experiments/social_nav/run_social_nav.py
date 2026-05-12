"""
社交导航主脚本
==============
一个文件跑完一个 episode：解析参数 → 加载场景 → 构建地图 → 运行导航 → 打印结果。

用法示例：
    ./.venv/bin/mjpython experiments/social_nav/run_social_nav.py \
        --scene-xml assets/scenes/ithor/FloorPlan203_physics.xml \
        --layout-json assets/layouts/FloorPlan203_object_positions.json \
        --start-pose=-1.4,4.6,0 \
        --goal-xy=-0.6,4.6

加 --no-viewer 可以无界面运行（headless）。
"""

from __future__ import annotations

# 必须在任何 matplotlib 导入之前设置，防止 macOS Cocoa 后端被初始化。
# scene_maps.py 在模块级别 import matplotlib.pyplot，如果此时后端已是 Cocoa，
# 后续 launch_passive 会报 "another MuJoCo viewer is already open"。
import os
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import threading

# 把仓库根目录加入路径，这样不管从哪里运行都能找到模块
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.social_nav.cost.llm_costmap import build_entity_params, synthesize_costmap
from experiments.social_nav.mppi_nav import MPPINav
from molmo_spaces.policy.solvers.navigation.mppi_core import (
    make_downscaled_grid,
    resolve_free_xy,
    wrap_to_pi,
)
from molmo_spaces.utils import distance_transform_utils as dtutils
from molmo_spaces.utils import scene_maps
from scripts import manual_capture_rgbd as scene_loader


# =============================================================================
# 1. 命令行参数
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="社交导航 episode 运行脚本")

    # --- 场景 ---
    p.add_argument("--scene-xml", type=Path, required=True,
                   help="场景 XML 文件路径，如 assets/scenes/ithor/FloorPlan203_physics.xml")
    p.add_argument("--layout-json", type=Path, default=None,
                   help="布局 JSON 文件（人物、机器人初始位置等），不传则自动推断")
    p.add_argument("--layout-runtime", choices=["auto", "on", "off"], default="auto",
                   help="是否用 layout JSON 组装场景（auto=有 JSON 就用）")
    p.add_argument("--robot-type", choices=["franka", "rby1", "rby1m", "navbot"], default="navbot",
                   help="机器人类型，社交导航一般用 navbot")

    # --- 导航目标 ---
    p.add_argument("--start-pose", type=_parse_xyz, default=(-1.4, 4.6, 0.0),
                   help="起始位姿 x,y,yaw_deg")
    p.add_argument("--goal-xy", type=_parse_xy, required=True,
                   help="目标位置 x,y（米，MuJoCo 世界坐标）")
    p.add_argument("--goal-radius", type=float, default=0.20,
                   help="到达半径（米），机器人进入此范围视为成功")
    p.add_argument("--max-steps", type=int, default=500,
                   help="最大控制步数，超过视为失败")

    # --- 地图参数 ---
    p.add_argument("--agent-radius", type=float, default=0.22,
                   help="地图膨胀半径（米），用于碰撞代价图")

    # --- MPPI 超参数 ---
    p.add_argument("--horizon", type=int, default=24, help="MPPI 预测步数")
    p.add_argument("--num-samples", type=int, default=512, help="MPPI 采样轨迹数")
    p.add_argument("--num-iters", type=int, default=4, help="每步 MPPI 迭代次数")
    p.add_argument("--dt", type=float, default=0.10, help="控制积分步长（秒）")
    p.add_argument("--temperature", type=float, default=1.0, help="MPPI 温度参数 λ")
    p.add_argument("--noise-alpha", type=float, default=0.8, help="噪声时间相关系数")
    p.add_argument("--noise-v", type=float, default=0.10, help="线速度噪声标准差（m/s）")
    p.add_argument("--noise-w-deg", type=float, default=30.0, help="角速度噪声标准差（deg/s）")
    p.add_argument("--v-min", type=float, default=-0.05, help="最小线速度（m/s）")
    p.add_argument("--v-max", type=float, default=0.45, help="最大线速度（m/s）")
    p.add_argument("--w-max-deg", type=float, default=120.0, help="最大角速度（deg/s）")
    p.add_argument("--cruise-speed", type=float, default=0.22, help="巡航速度偏置（m/s）")

    # --- 代价函数权重 ---
    p.add_argument("--goal-stage-weight", type=float, default=4.0, help="每步目标距离代价权重")
    p.add_argument("--goal-terminal-weight", type=float, default=20.0, help="终端目标距离代价权重")
    p.add_argument("--heading-terminal-weight", type=float, default=0.5, help="终端朝向误差权重")
    p.add_argument("--human-radius", type=float, default=0.30, help="人体碰撞半径（米），进入此范围代价无穷大")
    p.add_argument("--clearance-threshold", type=float, default=0.28, help="期望安全间距（米）")
    p.add_argument("--clearance-weight", type=float, default=25.0, help="安全间距违反代价权重")
    p.add_argument("--control-weight", type=float, default=0.15, help="控制量大小代价（节能）")
    p.add_argument("--smoothness-weight", type=float, default=1.5, help="控制平滑代价权重")

    # --- Social costmap ---
    p.add_argument("--social-method", choices=["none", "rule", "llm", "nn"], default="none",
                   help="社交代价图生成方式：none=不使用，rule=解析式规则，llm=LLM生成，nn=神经场")
    p.add_argument("--llm-model", type=str, default="doubao-pro-32k",
                   help="调用的 LLM 模型名，如 claude-opus-4-6 / gpt-4o")
    p.add_argument("--nn-checkpoint", type=str, default="checkpoints/scf2.pt",
                   help="SocialCostField 检查点路径（--social-method nn 时使用）")
    p.add_argument("--social-weight", type=float, default=1.0,
                   help="social costmap 在 MPPI 代价函数中的权重；默认较低，主要社交路线选择交给 A*")
    p.add_argument("--social-back-scale", type=float, default=1.0,
                   help="人后方社交代价 sigma 倍数；调大可抑制从人背后窄缝钻过")
    p.add_argument("--llm-update-interval", type=int, default=0,
                   help="每隔 N 步异步刷新 LLM costmap（0=禁用，仅 --social-method llm 有效）")
    p.add_argument("--scene-graph", type=Path, default=None,
                   help="HumanSSG scene_graph.json 路径；若指定则用 build_entity_params（LLM/rule 生成社交参数）")

    # --- A* / MPPI coupling ---
    p.add_argument("--astar-social-weight", type=float, default=30.0,
                   help="social costmap 在 A* 全局路径规划中的权重；路线选择主要由它决定")
    p.add_argument("--astar-human-block-radius", type=float, default=0.65,
                   help="A* 中把每个人周围多少米视为墙一样的不可侵犯圆；0=禁用")
    p.add_argument("--astar-num-candidates", type=int, default=1,
                   help="生成多少条 A* 候选路线再按整条路径代价选择；1=普通单路径 A*")
    p.add_argument("--astar-diversity-penalty", type=float, default=8.0,
                   help="重复 A* 时对已找到路径附近加的软惩罚，用于产生替代绕行路线")
    p.add_argument("--astar-candidate-clearance-weight", type=float, default=8.0,
                   help="候选路径评分中的窄通道/低 clearance 惩罚权重")
    p.add_argument("--astar-replan-interval", type=int, default=10,
                   help="每隔 N 步从当前位置重跑 A*（0=只在 costmap 更新时重规划）")
    p.add_argument("--mppi-target-lookahead", type=float, default=1.20,
                   help="MPPI 追踪 A* 路径前方多少米的局部目标，越大越不贴离散 waypoint")
    p.add_argument("--waypoint-radius", type=float, default=0.35,
                   help="A* waypoint 进度推进半径（米）")
    p.add_argument("--astar-smoothing", choices=["none", "shortcut"], default="shortcut",
                   help="A* 输出路径后处理：none=原始格子路径，shortcut=可通行/低社交代价直线简化")
    p.add_argument("--astar-shortcut-social-threshold", type=float, default=0.45,
                   help="shortcut 平滑时允许穿过的最大 social costmap 单格值")

    # --- 可视化 ---
    p.add_argument("--viewer-camera", choices=["follower", "robot"], default="follower",
                   help="viewer 显示的摄像头视角")
    p.add_argument("--robot-camera-name", type=str, default="robot_0/head_camera")
    p.add_argument("--follower-camera-name", type=str, default="robot_0/camera_follower")
    p.add_argument("--show-ui", action="store_true", default=False,
                   help="显示 MuJoCo viewer 的左右 UI 面板")
    p.add_argument("--no-viewer", action="store_true", default=False,
                   help="无界面运行（headless，适合服务器）")

    # --- 鸟瞰图 ---
    p.add_argument("--save-topdown", type=Path, default=None,
                   help="保存鸟瞰图 PNG 路径，如 outputs/episode.png")
    p.add_argument("--save-gif", type=Path, default=None,
                   help="保存轨迹 GIF 路径（每步捕帧，需要 Pillow）")
    p.add_argument("--show-topdown", action="store_true", default=False,
                   help="实时显示鸟瞰图 cv2 窗口（每 20 步刷新）")

    # --- 调试 ---
    p.add_argument("--log-every", type=int, default=10, help="每 N 步打印一次状态")
    p.add_argument("--wall-sleep", type=float, default=0.02, help="每步 viewer 刷新的等待时间（秒）")
    p.add_argument("--seed", type=int, default=0, help="随机种子")

    return p.parse_args()


def _parse_xy(s: str) -> tuple[float, float]:
    """解析 'x,y' 格式的字符串。"""
    vals = [float(v.strip()) for v in s.split(",")]
    if len(vals) != 2:
        raise argparse.ArgumentTypeError("格式应为 x,y，例如 -0.6,4.6")
    return vals[0], vals[1]


def _parse_xyz(s: str) -> tuple[float, float, float]:
    """解析 'x,y,yaw_deg' 格式的字符串。"""
    vals = [float(v.strip()) for v in s.split(",")]
    if len(vals) != 3:
        raise argparse.ArgumentTypeError("格式应为 x,y,yaw_deg，例如 -1.4,4.6,0")
    return vals[0], vals[1], vals[2]


# =============================================================================
# 2. 场景加载
# =============================================================================

def load_scene(args: argparse.Namespace):
    """
    加载 MuJoCo 场景，返回 (model, data) 以及各关节在 qpos 中的地址。

    场景加载逻辑：
    - 如果有 layout JSON 且 layout-runtime 不是 off，则用 basic_robot_human_scene 组装
      （自动插入机器人和人物）
    - 否则直接加载原始 XML
    """
    scene_xml = args.scene_xml
    if not scene_xml.is_file():
        raise FileNotFoundError(f"找不到场景文件: {scene_xml}")

    # 如果没有指定 layout JSON，尝试自动推断（同目录下同名的 JSON）
    layout_json = args.layout_json
    if layout_json is None:
        layout_json = scene_loader.infer_default_layout_json(scene_xml)
        if layout_json is not None:
            print(f"[场景] 自动找到 layout JSON: {layout_json}")

    # 用 manual_capture_rgbd 里的函数组装场景（它同时处理了机器人和人物的插入）
    model, data, source = scene_loader.load_scene_with_optional_layout_runtime(
        scene_xml, layout_json, args.layout_runtime,
        robot_type_override=args.robot_type,
    )
    print(f"[场景] 加载完成，来源: {source}")

    # 找机器人底盘关节在 qpos 里的位置
    base_x_adr = scene_loader.find_joint_qposadr(model, "robot_0/base_x")
    base_y_adr = scene_loader.find_joint_qposadr(model, "robot_0/base_y")
    base_theta_adr = scene_loader.find_joint_qposadr(model, "robot_0/base_theta")
    if None in (base_x_adr, base_y_adr, base_theta_adr):
        raise RuntimeError(
            "机器人模型里找不到 robot_0/base_x / base_y / base_theta 关节，"
            "请确认 --robot-type 正确（social nav 应使用 navbot/rby1/rby1m）"
        )

    return model, data, int(base_x_adr), int(base_y_adr), int(base_theta_adr)


# =============================================================================
# 3. 占用地图 + 距离变换
# =============================================================================

def _make_grid_from_occ(occ: np.ndarray, downscale: int) -> np.ndarray:
    """Downscale boolean occupancy (True=free) via min-pooling (conservative)."""
    pad_rows = (-occ.shape[0]) % downscale
    pad_cols = (-occ.shape[1]) % downscale
    padded = np.zeros((occ.shape[0] + pad_rows, occ.shape[1] + pad_cols), dtype=bool)
    padded[: occ.shape[0], : occ.shape[1]] = occ
    return (
        padded.reshape(padded.shape[0] // downscale, downscale, padded.shape[1] // downscale, downscale)
        .min(axis=1).min(axis=-1)
    )


def build_map(scene_xml: Path, agent_radius: float, downscale: int = 5, px_per_m: int = 200):
    """
    从场景 XML 构建 2D 占用地图和距离变换。

    返回：
        scene_map        : 原始地图对象（含坐标转换方法）
        grid_free        : 下采样后的可通行格子（bool 矩阵，MPPI 碰撞代价用）
        grid_free_astar  : 下采样后的可通行格子（较小膨胀，A* 连通性用）
        distance_transform: 每个格子到最近障碍物的距离（米）
        grid_spacing     : 每个格子的实际边长（米）
    """
    import cv2
    scene_maps._delete_blacklisted_bodies = lambda spec: 0

    xml_str = str(scene_xml)
    if "ithor" in xml_str:
        scene_map = scene_maps.iTHORMap.from_mj_model_path(
            model_path=xml_str, agent_radius=agent_radius, px_per_m=px_per_m
        )
    elif "procthor" in xml_str or "holodeck" in xml_str:
        scene_map = scene_maps.ProcTHORMap.from_mj_model_path(
            model_path=xml_str, agent_radius=agent_radius, px_per_m=px_per_m
        )
    else:
        raise ValueError(f"无法从路径判断场景类型: {scene_xml}")

    # MPPI 碰撞网格（含完整 agent_radius 膨胀）
    grid_free = make_downscaled_grid(scene_map, downscale)

    # A* 路径规划网格：将自由空间反向扩张，减少膨胀量，保留走廊连通性
    # 目标膨胀半径 0.08m → 恢复 (agent_radius - 0.08) 的膨胀
    _astar_target_radius = 0.08  # meters
    _recover_px = max(0, int((agent_radius - _astar_target_radius) * px_per_m))
    if _recover_px > 0:
        _k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * _recover_px + 1, 2 * _recover_px + 1))
        _occ_astar = cv2.dilate(scene_map.occupancy.astype(np.uint8), _k, iterations=1).astype(bool)
        grid_free_astar = _make_grid_from_occ(_occ_astar, downscale)
    else:
        grid_free_astar = grid_free

    print(f"[map] A* grid: {grid_free_astar.sum()} free cells (MPPI: {grid_free.sum()})")

    # 距离变换（基于 MPPI 网格）
    grid_spacing = downscale / px_per_m
    distance_transform = dtutils.make_distance_transform(grid_free, grid_spacing)

    return scene_map, grid_free, grid_free_astar, distance_transform, grid_spacing


# =============================================================================
# 4. Episode 主循环
# =============================================================================

def run_episode(args: argparse.Namespace) -> dict:
    """
    运行一个完整的导航 episode。

    流程：
        1. 加载场景
        2. 构建地图
        3. 把起始点/目标点对齐到可通行格子
        4. 初始化机器人位姿
        5. 构建 MPPI 控制器
        6. 打开 viewer（可选）
        7. 主循环：读状态 → 判断是否到达 → MPPI 求解动作 → 积分更新位姿
        8. 打印结果

    返回结果字典：{ reached_goal, steps, final_dist }
    """

    # --- 加载场景 ---
    model, data, x_adr, y_adr, th_adr = load_scene(args)

    # --- 构建地图 ---
    DOWNSCALE = 5
    PX_PER_M = 200
    scene_map, grid_free, grid_free_astar, dist_transform, grid_spacing = build_map(
        args.scene_xml, agent_radius=args.agent_radius,
        downscale=DOWNSCALE, px_per_m=PX_PER_M,
    )

    # --- 从 scene_map 推导世界坐标范围（用于 social costmap）---
    _occ = scene_map.occupancy
    _H, _W = _occ.shape
    _corners_px = np.array([[0, 0], [_H - 1, _W - 1]], dtype=float)
    _corners_m  = scene_map.pos_px_to_m(_corners_px)     # (2, 3) xyz
    _scene_x_range = (float(min(_corners_m[:, 0])), float(max(_corners_m[:, 0])))
    _scene_y_range = (float(min(_corners_m[:, 1])), float(max(_corners_m[:, 1])))

    # --- 把起点/终点对齐到可通行格子 ---
    # 如果指定位置恰好在障碍物上，自动吸附到最近的空闲格子
    start_xy = np.array([args.start_pose[0], args.start_pose[1]], dtype=np.float32)
    goal_xy = np.array([args.goal_xy[0], args.goal_xy[1]], dtype=np.float32)

    start_xy = _snap_to_free(scene_map, grid_free, start_xy, DOWNSCALE, "起点")
    goal_xy = _snap_to_free(scene_map, grid_free, goal_xy, DOWNSCALE, "目标")

    # --- 初始化机器人位姿 ---
    data.qpos[x_adr] = float(start_xy[0])
    data.qpos[y_adr] = float(start_xy[1])
    data.qpos[th_adr] = math.radians(float(args.start_pose[2]))
    mujoco.mj_forward(model, data)

    # --- 生成 social params（可选）---
    social_params  = None
    _llm_log_init  = ""
    _scene_for_refresh = None   # 用于 LLM 刷新时复用 SceneDescription
    if args.social_method in ("rule", "llm"):
        if args.scene_graph is not None:
            import sys as _sys
            _repo = str(Path(__file__).resolve().parents[3])
            if _repo not in _sys.path:
                _sys.path.insert(0, _repo)
            from pipeline.scene_bridge import scene_graph_to_scene_description
            _scene_for_refresh = scene_graph_to_scene_description(args.scene_graph)
            print(f"[scene_graph] 加载: {len(_scene_for_refresh.humans)} 人, "
                  f"{len(_scene_for_refresh.obstacles)} 障碍")
            print(f"[social] method={args.social_method}  model={args.llm_model}")
            social_params, _llm_log_init = build_entity_params(
                _scene_for_refresh,
                method=args.social_method,
                llm_model=args.llm_model,
                verbose=True,
                robot_pos=tuple(start_xy),
                robot_goal=tuple(goal_xy),
            )
        else:
            raise ValueError(f"--social-method {args.social_method} 需要指定 --scene-graph")

    if _scene_for_refresh is not None and args.astar_human_block_radius > 0.0:
        grid_free_astar = _block_human_disks_on_grid(
            scene_map,
            grid_free_astar,
            DOWNSCALE,
            grid_spacing,
            [h.pos for h in _scene_for_refresh.humans],
            radius_m=float(args.astar_human_block_radius),
        )

    # --- 初始化 MPPI 控制器 ---
    nav = MPPINav(
        scene_map=scene_map,
        grid_free=grid_free,
        distance_transform=dist_transform,
        grid_spacing=grid_spacing,
        downscale=DOWNSCALE,
        horizon=args.horizon,
        num_samples=args.num_samples,
        num_iters=args.num_iters,
        dt=args.dt,
        temperature=args.temperature,
        noise_alpha=args.noise_alpha,
        noise_v=args.noise_v,
        noise_w_deg=args.noise_w_deg,
        v_min=args.v_min,
        v_max=args.v_max,
        w_max_deg=args.w_max_deg,
        cruise_speed=args.cruise_speed,
        goal_stage_weight=args.goal_stage_weight,
        goal_terminal_weight=args.goal_terminal_weight,
        heading_terminal_weight=args.heading_terminal_weight,
        clearance_threshold=args.clearance_threshold,
        clearance_weight=args.clearance_weight,
        control_weight=args.control_weight,
        smoothness_weight=args.smoothness_weight,
        social_params=social_params,
        social_weight=args.social_weight,
        social_back_scale=args.social_back_scale,
        human_positions=[h.pos for h in _scene_for_refresh.humans] if _scene_for_refresh else None,
        human_radius=args.human_radius,
        seed=args.seed,
    )
    nav.reset()

    # --- A* 全局路径规划（引导 MPPI waypoint 追踪）---
    _WP_RADIUS = max(float(args.waypoint_radius), 1e-3)
    _waypoints: np.ndarray | None = None
    _wp_idx = 0
    _social_cm_for_astar = (
        synthesize_costmap(social_params, grid_free_astar.shape,
                           x_range=_scene_x_range, y_range=_scene_y_range,
                           distance_transform=dist_transform,
                           clearance_cap=0.5, clearance_weight=0.3,
                           back_scale=args.social_back_scale)
        if social_params is not None else None
    )
    try:
        _astar_result = _plan_global_path(
            scene_map, grid_free_astar, DOWNSCALE, grid_spacing,
            start_xy, goal_xy, _social_cm_for_astar,
            distance_transform=dist_transform,
            args=args,
        )
        if _astar_result is not None and len(_astar_result) >= 2:
            _waypoints = _postprocess_astar_path(
                _astar_result, scene_map, grid_free_astar, DOWNSCALE, grid_spacing,
                _social_cm_for_astar, args,
            )
            print(f"[A*] 规划完成: {len(_astar_result)} raw → {len(_waypoints)} tracking waypoints")
    except Exception as _ae:
        print(f"[警告] A* 规划失败，退化为直接导航目标: {_ae}")

    # --- 鸟瞰图初始化 ---
    viz = None
    save_topdown = getattr(args, "save_topdown", None)
    save_gif     = getattr(args, "save_gif",     None)
    if save_topdown is not None or save_gif is not None or args.show_topdown:
        try:
            from experiments.social_nav.topdown_viz import TopdownViz
            viz = TopdownViz(scene_map, grid_free, grid_spacing, DOWNSCALE)
            viz.set_start_goal(start_xy, goal_xy)
            if _scene_for_refresh is not None:
                viz.set_humans(_scene_for_refresh.humans)
            if _social_cm_for_astar is not None:
                viz.set_social_costmap(_social_cm_for_astar)
            if _llm_log_init:
                viz.set_llm_log(_llm_log_init)
            if _waypoints is not None:
                viz.set_astar_path(_waypoints)
                print(f"[topdown_viz] A* path: {len(_waypoints)} waypoints")
        except Exception as _e:
            print(f"[警告] 鸟瞰图初始化失败: {_e}")
            viz = None

    # --- 解析摄像头 ID（用于 viewer 视角切换）---
    robot_cam_id = _find_camera(model, args.robot_camera_name)
    follower_cam_id = _find_camera(model, args.follower_camera_name)
    if args.viewer_camera == "robot" and robot_cam_id is None:
        raise RuntimeError(f"找不到机器人摄像头: {args.robot_camera_name}")

    # ==========================================================================
    # 主循环
    # ==========================================================================
    step = 0
    reached_goal = False
    quit_flag = False   # 按 Q 退出

    # --- 异步 LLM params 刷新（仅 --social-method llm 且 --llm-update-interval > 0）---
    _pending_params: list[list | None] = [None]
    _pending_llm_log: list[str | None] = [None]
    _llm_trigger = threading.Event()
    _llm_stop = threading.Event()

    def _llm_refresh_worker():
        while not _llm_stop.is_set():
            triggered = _llm_trigger.wait(timeout=1.0)
            if not triggered:
                continue
            _llm_trigger.clear()
            try:
                if _scene_for_refresh is not None:
                    new_params, _new_llm_log = build_entity_params(
                        _scene_for_refresh,
                        method="llm",
                        llm_model=args.llm_model,
                        verbose=False,
                        robot_pos=tuple(np.array([float(data.qpos[x_adr]), float(data.qpos[y_adr])], dtype=np.float32)),
                        robot_goal=tuple(goal_xy),
                    )
                    _pending_params[0]  = new_params
                    _pending_llm_log[0] = _new_llm_log
            except Exception as exc:
                print(f"[LLM刷新] 失败: {exc}")

    _use_llm_refresh = (args.social_method == "llm" and args.llm_update_interval > 0)
    if _use_llm_refresh:
        _t = threading.Thread(target=_llm_refresh_worker, daemon=True)
        _t.start()

    # 启动鸟瞰图子进程（独立 event loop，不与 MuJoCo viewer 冲突）
    _topdown_proc = None
    if args.show_topdown:
        import subprocess as _subprocess
        _live_png = str(Path("outputs/_topdown_live.png").resolve())
        _viewer_script = (
            "import cv2, time, os, sys\n"
            f"path = {_live_png!r}\n"
            "while True:\n"
            "    if os.path.exists(path):\n"
            "        img = cv2.imread(path)\n"
            "        if img is not None:\n"
            "            cv2.imshow('Topdown View', img)\n"
            "    k = cv2.waitKey(200)\n"
            "    if k == ord('q'):\n"
            "        break\n"
        )
        _topdown_proc = _subprocess.Popen(
            [sys.executable, "-c", _viewer_script],
        )

    def _key_cb(keycode: int) -> None:
        nonlocal quit_flag
        if keycode in (ord("q"), ord("Q")):
            quit_flag = True

    # 打开 viewer（只开一次，含按键回调；headless 时跳过）
    viewer_cm = _open_viewer(model, data, args, key_callback=_key_cb) if not args.no_viewer else None

    with (viewer_cm if viewer_cm is not None else _null_ctx()) as viewer:
        while step < args.max_steps and not quit_flag:
            with (viewer.lock() if viewer is not None else _null_ctx()):

                # 读当前位姿
                bx = float(data.qpos[x_adr])
                by = float(data.qpos[y_adr])
                th = float(data.qpos[th_adr])
                cur_xy = np.array([bx, by], dtype=np.float32)

                # 判断是否到达目标
                dist = float(np.linalg.norm(goal_xy - cur_xy))
                if dist <= args.goal_radius:
                    reached_goal = True
                    mujoco.mj_forward(model, data)
                    _set_viewer_camera(viewer, robot_cam_id, follower_cam_id, args.viewer_camera)
                    if viewer is not None:
                        viewer.sync()
                    break

                # 触发异步 LLM costmap 刷新
                if _use_llm_refresh and step > 0 and step % args.llm_update_interval == 0:
                    if not _llm_trigger.is_set():  # 上一次还没处理完就跳过
                        _llm_trigger.set()
                        print(f"[步骤 {step}] 触发 LLM costmap 刷新...")

                # 应用新 params（若后台线程已准备好）
                if _pending_params[0] is not None:
                    new_params = _pending_params[0]
                    _pending_params[0] = None
                    nav.update_social_params(new_params)
                    # 重算 social costmap 并 A* 重规划
                    _social_cm_for_astar = synthesize_costmap(
                        new_params, grid_free_astar.shape,
                        x_range=_scene_x_range, y_range=_scene_y_range,
                        distance_transform=dist_transform,
                        clearance_cap=0.5, clearance_weight=0.3,
                        back_scale=args.social_back_scale,
                    )
                    try:
                        _new_wp = _plan_global_path(
                            scene_map, grid_free_astar, DOWNSCALE, grid_spacing,
                            cur_xy, goal_xy, _social_cm_for_astar,
                            distance_transform=dist_transform,
                            args=args,
                        )
                        if _new_wp is not None and len(_new_wp) >= 2:
                            _waypoints = _postprocess_astar_path(
                                _new_wp, scene_map, grid_free_astar, DOWNSCALE, grid_spacing,
                                _social_cm_for_astar, args,
                            )
                            _wp_idx = 0
                    except Exception:
                        pass
                    if viz is not None:
                        viz.set_social_costmap(_social_cm_for_astar)
                        if _waypoints is not None:
                            viz.set_astar_path(_waypoints)
                    print(f"[步骤 {step}] ✓ LLM social params 已更新，A* 重规划")
                if _pending_llm_log[0] is not None and viz is not None:
                    viz.set_llm_log(_pending_llm_log[0])
                    _pending_llm_log[0] = None

                # 静态/规则 costmap 下也允许 A* 从当前位置周期性重规划。
                # 这样全局引导不会固定在 episode 初始路径上，鸟瞰图能显示路径随当前位置更新。
                if (
                    args.astar_replan_interval > 0
                    and step > 0
                    and step % args.astar_replan_interval == 0
                ):
                    try:
                        _new_wp = _plan_global_path(
                            scene_map, grid_free_astar, DOWNSCALE, grid_spacing,
                            cur_xy, goal_xy, _social_cm_for_astar,
                            distance_transform=dist_transform,
                            args=args,
                        )
                        if _new_wp is not None and len(_new_wp) >= 2:
                            _waypoints = _postprocess_astar_path(
                                _new_wp, scene_map, grid_free_astar, DOWNSCALE, grid_spacing,
                                _social_cm_for_astar, args,
                            )
                            _wp_idx = 0
                            if viz is not None:
                                viz.set_astar_path(_waypoints)
                    except Exception as _ae:
                        print(f"[步骤 {step}] A* 周期重规划失败: {_ae}")

                # 推进 waypoint 索引（滑动窗口：到达当前 wp 就切下一个）
                if _waypoints is not None:
                    while _wp_idx < len(_waypoints) - 1:
                        if np.linalg.norm(cur_xy - _waypoints[_wp_idx]) < _WP_RADIUS:
                            _wp_idx += 1
                        else:
                            break
                _mppi_target = (
                    _path_lookahead_target(_waypoints, _wp_idx, cur_xy, args.mppi_target_lookahead)
                    if _waypoints is not None else goal_xy
                )

                # MPPI 求解动作
                robot_state = np.array([bx, by, th], dtype=np.float32)
                v, w, info = nav.step(robot_state, _mppi_target)

                # 差速底盘积分（简单欧拉法）
                bx += args.dt * v * math.cos(th)
                by += args.dt * v * math.sin(th)
                th = wrap_to_pi(th + args.dt * w)

                # 写回 MuJoCo
                data.qpos[x_adr] = bx
                data.qpos[y_adr] = by
                data.qpos[th_adr] = th
                mujoco.mj_forward(model, data)

                # 更新摄像头视角
                _set_viewer_camera(viewer, robot_cam_id, follower_cam_id, args.viewer_camera)

                # 更新鸟瞰图
                if viz is not None:
                    capture = save_gif is not None and step % 3 == 0
                    viz.update(np.array([bx, by]), capture_frame=capture)
                    _rxy = info.get("rollout_xy")
                    _rc  = info.get("rollout_costs")
                    if _rxy is not None and _rc is not None:
                        viz.update_mppi_rollouts(_rxy, _rc)

                # 打印状态
                if args.log_every > 0 and step % args.log_every == 0:
                    print(
                        f"[步骤 {step:4d}]  距目标={dist:.3f}m  "
                        f"wp={_wp_idx if _waypoints is not None else -1}  "
                        f"v={v:.3f}m/s  w={math.degrees(w):.1f}°/s  "
                        f"最优代价={info['best_cost']:.2f}  "
                        f"最小间距={info['best_clearance']:.3f}m"
                    )

            if viewer is not None:
                viewer.sync()
            # 实时鸟瞰图：每 20 步写一张 PNG，子进程负责显示
            if args.show_topdown and viz is not None and step % 20 == 0:
                _live_png = Path("outputs/_topdown_live.png")
                _live_png.parent.mkdir(parents=True, exist_ok=True)
                viz.save(str(_live_png), verbose=False)
            time.sleep(max(0.0, args.wall_sleep))
            step += 1

    # 关闭鸟瞰图子进程
    if _topdown_proc is not None:
        _topdown_proc.terminate()

    # 停止后台 LLM 刷新线程
    if _use_llm_refresh:
        _llm_stop.set()
        _llm_trigger.set()  # 解除 wait() 阻塞让线程自然退出

    # --- 打印最终结果 ---
    final_xy = np.array([float(data.qpos[x_adr]), float(data.qpos[y_adr])], dtype=np.float32)
    final_dist = float(np.linalg.norm(goal_xy - final_xy))
    status = "✓ 到达目标" if reached_goal else "✗ 未到达"
    print(
        f"\n[结果] {status}  步数={step}  "
        f"最终位置={tuple(np.round(final_xy, 3))}  "
        f"目标位置={tuple(np.round(goal_xy, 3))}  "
        f"剩余距离={final_dist:.3f}m"
    )
    # --- 导出鸟瞰图 ---
    if viz is not None:
        # 最终轨迹图
        if save_topdown is not None:
            viz.save(save_topdown)
        # GIF（如果有帧）
        if save_gif is not None:
            viz.save_gif(save_gif, fps=10)
        viz.close()

    import sys as _sys
    _sys.exit(0 if reached_goal else 1)


# =============================================================================
# 5. 辅助函数
# =============================================================================

def _block_human_disks_on_grid(
    scene_map,
    grid_free: np.ndarray,
    downscale: int,
    grid_spacing: float,
    human_positions: list[tuple[float, float]],
    radius_m: float,
) -> np.ndarray:
    """Treat each human's minimum intrusion circle as non-traversable for A*."""
    if not human_positions or radius_m <= 0.0:
        return grid_free

    blocked = grid_free.copy()
    H, W = blocked.shape
    radius_cells = max(int(math.ceil(radius_m / max(grid_spacing, 1e-6))), 1)
    before = int(blocked.sum())

    for pos in human_positions:
        hr, hc = _world_to_grid_rc(scene_map, np.asarray(pos, dtype=np.float32), downscale, blocked.shape)
        r0, r1 = max(0, hr - radius_cells), min(H, hr + radius_cells + 1)
        c0, c1 = max(0, hc - radius_cells), min(W, hc + radius_cells + 1)
        for r in range(r0, r1):
            for c in range(c0, c1):
                if math.hypot(r - hr, c - hc) * grid_spacing <= radius_m:
                    blocked[r, c] = False

    after = int(blocked.sum())
    print(
        f"[A*] human hard exclusion: radius={radius_m:.2f}m, "
        f"blocked={before - after} cells, free={after}"
    )
    return blocked


def _plan_global_path(
    scene_map,
    grid_free: np.ndarray,
    downscale: int,
    grid_spacing: float,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    social_costmap: np.ndarray | None,
    distance_transform: np.ndarray | None,
    args: argparse.Namespace,
) -> np.ndarray | None:
    """Generate multiple A* candidates and choose the best path-level route."""
    k = max(int(getattr(args, "astar_num_candidates", 1)), 1)
    social_w = float(getattr(args, "astar_social_weight", 30.0))
    diversity_w = float(getattr(args, "astar_diversity_penalty", 8.0))
    clearance_w = float(getattr(args, "astar_candidate_clearance_weight", 8.0))
    clearance_cap = 0.5

    penalty = np.zeros(grid_free.shape, dtype=np.float32)
    candidates: list[tuple[float, np.ndarray]] = []
    for cand_idx in range(k):
        path = _astar_on_grid(
            scene_map, grid_free, downscale, grid_spacing,
            start_xy, goal_xy, social_costmap,
            social_w=social_w,
            distance_transform=distance_transform,
            clearance_w=3.0,
            clearance_cap=clearance_cap,
            extra_costmap=penalty if cand_idx > 0 else None,
        )
        if path is None or len(path) < 2:
            break

        score = _score_astar_candidate(
            path, scene_map, grid_free, downscale, grid_spacing,
            social_costmap, distance_transform,
            social_w=social_w,
            clearance_w=clearance_w,
            clearance_cap=clearance_cap,
        )
        candidates.append((score, path))
        penalty = np.maximum(
            penalty,
            _path_penalty_grid(path, scene_map, grid_free.shape, downscale, radius_cells=3)
            * diversity_w,
        )

    if not candidates:
        return None

    best_idx, (best_score, best_path) = min(enumerate(candidates), key=lambda item: item[1][0])
    print(
        "[A*] candidates: "
        + ", ".join(f"{i}:n={len(p)} score={s:.1f}" for i, (s, p) in enumerate(candidates))
        + f"  -> choose {best_idx}"
    )
    return best_path


def _astar_on_grid(
    scene_map, grid_free: np.ndarray, downscale: int, grid_spacing: float,
    start_xy: np.ndarray, goal_xy: np.ndarray,
    social_costmap: np.ndarray | None = None,
    social_w: float = 12.0,
    distance_transform: np.ndarray | None = None,
    clearance_w: float = 3.0,
    clearance_cap: float = 0.5,
    extra_costmap: np.ndarray | None = None,
) -> np.ndarray | None:
    """
    A* on the downscaled occupancy grid using scene_map coordinate system.
    Returns (N,2) world-frame waypoints, or None if no path found.

    Combined edge cost = geometric distance
                       × (1 + social_w × social_costmap[cell]
                            + clearance_w × clearance_cost[cell])

    clearance_cost = 1 - clip(distance_transform / clearance_cap, 0, 1)
    즉 벽에 가까울수록 clearance_cost→1, 열린 공간→0
    """
    import heapq

    H, W = grid_free.shape

    def _world_to_rc(xy):
        xyz = np.array([[xy[0], xy[1], 0.0]], dtype=np.float32)
        px  = scene_map.pos_m_to_px(xyz)[0]   # (row, col)
        r   = int(np.clip(px[0] / downscale, 0, H - 1))
        c   = int(np.clip(px[1] / downscale, 0, W - 1))
        return r, c

    def _rc_to_world(r, c):
        px = np.array([[(r + 0.5) * downscale, (c + 0.5) * downscale]], dtype=np.float32)
        return scene_map.pos_px_to_m(px)[0, :2]

    src = _world_to_rc(start_xy)
    dst = _world_to_rc(goal_xy)
    if not grid_free[src] or not grid_free[dst]:
        return None

    # Social cost overlay (0~1 → scaled by social_w)
    # costmap 已经在 synthesize_costmap 里叠加了 clearance，直接用
    ext_social = (social_w * social_costmap).astype(np.float64) \
                 if social_costmap is not None else None
    ext_clear = None
    if distance_transform is not None and clearance_w > 0:
        dt = distance_transform.astype(np.float64)
        if dt.shape == grid_free.shape:
            ext_clear = clearance_w * (1.0 - np.clip(dt * grid_spacing / clearance_cap, 0.0, 1.0))
    ext_extra = extra_costmap.astype(np.float64) if extra_costmap is not None else None

    dirs  = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    dcost = [1.,1.,1.,1., math.sqrt(2),math.sqrt(2),math.sqrt(2),math.sqrt(2)]

    def h(r, c): return math.hypot(r - dst[0], c - dst[1])

    dist_g = np.full((H, W), np.inf)
    dist_g[src] = 0.0
    prev: dict = {}
    heap = [(h(*src), src)]

    while heap:
        f, (r, c) = heapq.heappop(heap)
        if (r, c) == dst:
            break
        if f > dist_g[r, c] + h(r, c) + 1e-9:
            continue
        for (dr, dc), mc in zip(dirs, dcost):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W and grid_free[nr, nc]):
                continue
            overlay = float(ext_social[nr, nc]) if ext_social is not None else 0.0
            if ext_clear is not None:
                overlay += float(ext_clear[nr, nc])
            if ext_extra is not None:
                overlay += float(ext_extra[nr, nc])
            cost = mc * (1.0 + overlay)
            g = dist_g[r, c] + cost
            if g < dist_g[nr, nc]:
                dist_g[nr, nc] = g
                prev[(nr, nc)] = (r, c)
                heapq.heappush(heap, (g + h(nr, nc), (nr, nc)))

    if dst not in prev and src != dst:
        print(f"[A*] 未找到路径: src={src} dst={dst} (grid可能不连通)")
        return None

    path, cur = [], dst
    while cur in prev:
        path.append(cur)
        cur = prev[cur]
    path.append(src)
    path.reverse()
    return np.array([_rc_to_world(r, c) for r, c in path], dtype=np.float32)


def _score_astar_candidate(
    path: np.ndarray,
    scene_map,
    grid_free: np.ndarray,
    downscale: int,
    grid_spacing: float,
    social_costmap: np.ndarray | None,
    distance_transform: np.ndarray | None,
    social_w: float,
    clearance_w: float,
    clearance_cap: float,
) -> float:
    if len(path) < 2:
        return math.inf

    length = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    social_sum = 0.0
    clearance_penalty = 0.0
    min_clearance = math.inf
    collision_penalty = 0.0
    samples = 0

    for a, b in zip(path[:-1], path[1:]):
        seg_len = float(np.linalg.norm(b - a))
        n = max(int(math.ceil(seg_len / max(grid_spacing * 0.5, 0.02))), 1)
        for alpha in np.linspace(0.0, 1.0, n + 1):
            r, c = _world_to_grid_rc(scene_map, a + alpha * (b - a), downscale, grid_free.shape)
            if not grid_free[r, c]:
                collision_penalty += 1000.0
                samples += 1
                continue
            if social_costmap is not None:
                social_sum += float(social_costmap[r, c])
            if distance_transform is not None and distance_transform.shape == grid_free.shape:
                clearance_m = float(distance_transform[r, c]) * grid_spacing
                min_clearance = min(min_clearance, clearance_m)
                clearance_penalty += max(0.0, clearance_cap - clearance_m) / clearance_cap
            samples += 1

    if samples <= 0:
        return math.inf
    social_avg = social_sum / samples
    clearance_avg = clearance_penalty / samples
    min_clearance_term = 0.0 if math.isinf(min_clearance) else max(0.0, 0.25 - min_clearance) * 20.0
    return (
        length
        + social_w * social_avg * length
        + clearance_w * clearance_avg * length
        + min_clearance_term
        + collision_penalty
    )


def _path_penalty_grid(
    path: np.ndarray,
    scene_map,
    grid_shape: tuple[int, int],
    downscale: int,
    radius_cells: int = 3,
) -> np.ndarray:
    penalty = np.zeros(grid_shape, dtype=np.float32)
    H, W = grid_shape
    for xy in path:
        r, c = _world_to_grid_rc(scene_map, xy, downscale, grid_shape)
        r0, r1 = max(0, r - radius_cells), min(H, r + radius_cells + 1)
        c0, c1 = max(0, c - radius_cells), min(W, c + radius_cells + 1)
        for rr in range(r0, r1):
            for cc in range(c0, c1):
                d = math.hypot(rr - r, cc - c)
                if d <= radius_cells:
                    penalty[rr, cc] = max(penalty[rr, cc], 1.0 - d / max(radius_cells, 1))
    return penalty


def _world_to_grid_rc(
    scene_map,
    xy: np.ndarray,
    downscale: int,
    grid_shape: tuple[int, int],
) -> tuple[int, int]:
    H, W = grid_shape
    xyz = np.array([[xy[0], xy[1], 0.0]], dtype=np.float32)
    px = scene_map.pos_m_to_px(xyz)[0]
    r = int(np.clip(px[0] / downscale, 0, H - 1))
    c = int(np.clip(px[1] / downscale, 0, W - 1))
    return r, c


def _postprocess_astar_path(
    waypoints: np.ndarray,
    scene_map,
    grid_free: np.ndarray,
    downscale: int,
    grid_spacing: float,
    social_costmap: np.ndarray | None,
    args: argparse.Namespace,
) -> np.ndarray:
    """Convert raw grid A* output into a cleaner tracking path for MPPI."""
    waypoints = np.asarray(waypoints, dtype=np.float32)
    if len(waypoints) < 3 or getattr(args, "astar_smoothing", "shortcut") == "none":
        return waypoints
    return _shortcut_path(
        waypoints,
        scene_map,
        grid_free,
        downscale,
        grid_spacing,
        social_costmap,
        max_social=float(getattr(args, "astar_shortcut_social_threshold", 0.45)),
    )


def _shortcut_path(
    waypoints: np.ndarray,
    scene_map,
    grid_free: np.ndarray,
    downscale: int,
    grid_spacing: float,
    social_costmap: np.ndarray | None,
    max_social: float,
) -> np.ndarray:
    """Greedy line-of-sight simplification constrained by occupancy and social cost."""
    if len(waypoints) < 3:
        return waypoints

    out = [waypoints[0]]
    i = 0
    while i < len(waypoints) - 1:
        j = len(waypoints) - 1
        while j > i + 1:
            if _segment_allowed(
                waypoints[i], waypoints[j], scene_map, grid_free, downscale,
                grid_spacing, social_costmap, max_social,
            ):
                break
            j -= 1
        out.append(waypoints[j])
        i = j
    return np.asarray(out, dtype=np.float32)


def _segment_allowed(
    a: np.ndarray,
    b: np.ndarray,
    scene_map,
    grid_free: np.ndarray,
    downscale: int,
    grid_spacing: float,
    social_costmap: np.ndarray | None,
    max_social: float,
) -> bool:
    dist = float(np.linalg.norm(b - a))
    if dist <= 1e-6:
        return True

    step = max(float(grid_spacing) * 0.5, 0.02)
    n = max(int(math.ceil(dist / step)), 2)
    H, W = grid_free.shape
    for alpha in np.linspace(0.0, 1.0, n + 1):
        xy = a + alpha * (b - a)
        xyz = np.array([[xy[0], xy[1], 0.0]], dtype=np.float32)
        px = scene_map.pos_m_to_px(xyz)[0]
        r = int(np.clip(px[0] / downscale, 0, H - 1))
        c = int(np.clip(px[1] / downscale, 0, W - 1))
        if not grid_free[r, c]:
            return False
        if social_costmap is not None and float(social_costmap[r, c]) > max_social:
            return False
    return True


def _path_lookahead_target(
    waypoints: np.ndarray,
    start_idx: int,
    cur_xy: np.ndarray,
    lookahead_m: float,
) -> np.ndarray:
    """Pick a target ahead on the A* polyline instead of tracking every grid waypoint."""
    if waypoints is None or len(waypoints) == 0:
        return cur_xy
    idx = int(np.clip(start_idx, 0, len(waypoints) - 1))
    if idx >= len(waypoints) - 1 or lookahead_m <= 1e-6:
        return waypoints[idx]

    remaining = float(lookahead_m)
    prev = np.asarray(cur_xy, dtype=np.float32)
    for j in range(idx, len(waypoints)):
        nxt = waypoints[j]
        seg = float(np.linalg.norm(nxt - prev))
        if seg >= remaining:
            if seg <= 1e-6:
                return nxt
            alpha = remaining / seg
            return (prev + alpha * (nxt - prev)).astype(np.float32)
        remaining -= seg
        prev = nxt
    return waypoints[-1]


def _snap_to_free(scene_map, grid_free, xy: np.ndarray, downscale: int, label: str) -> np.ndarray:
    """把坐标吸附到最近的可通行格子，如果已经在空闲区域则原样返回。"""
    snapped = resolve_free_xy(scene_map, grid_free, xy, downscale)
    if snapped is None:
        raise RuntimeError(f"{label} {tuple(xy)} 无法找到附近的可通行位置，请检查坐标")
    if not np.allclose(snapped, xy, atol=1e-3):
        print(f"[警告] {label} 吸附: {tuple(np.round(xy, 3))} → {tuple(np.round(snapped, 3))}")
    return snapped


def _find_camera(model: mujoco.MjModel, name: str) -> int | None:
    """查找摄像头 ID，找不到返回 None（不报错）。"""
    try:
        return scene_loader.resolve_camera_id(model, name)
    except Exception:
        return None


def _set_viewer_camera(viewer, robot_cam_id, follower_cam_id, mode: str) -> None:
    """切换 viewer 摄像头视角。"""
    if viewer is None:
        return
    cam_id = robot_cam_id
    if mode == "follower" and follower_cam_id is not None:
        cam_id = follower_cam_id
    if cam_id is not None:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = int(cam_id)


def _kill_other_mjpython() -> bool:
    """杀掉除自身以外的所有 mjpython 进程，返回是否有进程被杀掉。"""
    import signal
    import subprocess
    my_pid = os.getpid()
    try:
        result = subprocess.run(
            ["pgrep", "-f", "mjpython"],
            capture_output=True, text=True
        )
        pids = [int(p) for p in result.stdout.split() if p.strip()]
        others = [p for p in pids if p != my_pid]
        for pid in others:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        return len(others) > 0
    except Exception:
        return False


def _open_viewer(model, data, args, key_callback=None):
    """打开 MuJoCo viewer，返回 context manager。

    macOS 上 mjpython 同时只允许一个 viewer。若发现冲突，自动杀掉残留进程后重试一次。
    """
    import gc
    import time
    import mujoco.viewer as mj_viewer

    gc.collect()

    def _try_launch():
        return mj_viewer.launch_passive(
            model, data,
            key_callback=key_callback,
            show_left_ui=args.show_ui,
            show_right_ui=args.show_ui,
        )

    try:
        return _try_launch()
    except RuntimeError as e:
        if "already open" in str(e):
            print("[viewer] 检测到 viewer 冲突，正在清理残留进程后重试...")
            killed = _kill_other_mjpython()
            if killed:
                time.sleep(0.5)
            gc.collect()
            try:
                return _try_launch()
            except RuntimeError as e2:
                raise RuntimeError(
                    "MuJoCo viewer 无法打开（连续两次失败）。\n"
                    "请手动运行 'pkill -f mjpython' 后重试，\n"
                    "或加 --no-viewer 改用 headless 模式。"
                ) from e2
        if sys.platform == "darwin" and "mjpython" in str(e):
            raise RuntimeError(
                "macOS 下需要用 ./.venv/bin/mjpython 运行本脚本。\n"
                "如果想跳过界面，加上 --no-viewer 参数。"
            ) from e
        raise


class _null_ctx:
    """空 context manager，用于 headless 模式下统一 with 语法。"""
    def __enter__(self): return None
    def __exit__(self, *_): return False
    def lock(self): return self
    def sync(self): pass


# =============================================================================
# 入口
# =============================================================================

if __name__ == "__main__":
    args = parse_args()
    run_episode(args)
