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

import argparse
import math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

# 把仓库根目录加入路径，这样不管从哪里运行都能找到模块
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.social_nav.llm_costmap import build_social_costmap
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
    p.add_argument("--collision-penalty", type=float, default=500.0, help="碰撞惩罚值")
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
    p.add_argument("--social-weight", type=float, default=30.0,
                   help="social costmap 在 MPPI 代价函数中的权重")

    # --- 可视化 ---
    p.add_argument("--viewer-camera", choices=["follower", "robot"], default="follower",
                   help="viewer 显示的摄像头视角")
    p.add_argument("--robot-camera-name", type=str, default="robot_0/head_camera")
    p.add_argument("--follower-camera-name", type=str, default="robot_0/camera_follower")
    p.add_argument("--show-ui", action="store_true", default=False,
                   help="显示 MuJoCo viewer 的左右 UI 面板")
    p.add_argument("--no-viewer", action="store_true", default=False,
                   help="无界面运行（headless，适合服务器）")

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

def build_map(scene_xml: Path, agent_radius: float, downscale: int = 5, px_per_m: int = 200):
    """
    从场景 XML 构建 2D 占用地图和距离变换。

    返回：
        scene_map        : 原始地图对象（含坐标转换方法）
        grid_free        : 下采样后的可通行格子（bool 矩阵）
        distance_transform: 每个格子到最近障碍物的距离（米）
        grid_spacing     : 每个格子的实际边长（米）
    """
    # 地图类型根据场景路径自动判断（iTHOR / ProcTHOR / Holodeck）
    # manager.py 原来有一段猴子补丁（_delete_blacklisted_bodies = lambda...），
    # 目的是绕过 task_sampler 的资源锁，这里保留同样处理
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

    # 下采样：把高分辨率占用图缩小，减少 MPPI 的计算量
    grid_free = make_downscaled_grid(scene_map, downscale)

    # 距离变换：grid_free 每个格子 → 到最近障碍物的欧氏距离（米）
    grid_spacing = downscale / px_per_m   # 每格对应的实际距离
    distance_transform = dtutils.make_distance_transform(grid_free, grid_spacing)

    return scene_map, grid_free, distance_transform, grid_spacing


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
    scene_map, grid_free, dist_transform, grid_spacing = build_map(
        args.scene_xml, agent_radius=args.agent_radius,
        downscale=DOWNSCALE, px_per_m=PX_PER_M,
    )

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

    # --- 生成 social costmap（可选）---
    social_costmap = None
    if args.social_method in ("rule", "llm"):
        if args.layout_json is None:
            raise ValueError(f"--social-method {args.social_method} 需要指定 --layout-json")
        social_costmap = build_social_costmap(
            layout_json_path=args.layout_json,
            scene_map=scene_map,
            grid_shape=grid_free.shape,
            downscale=DOWNSCALE,
            method=args.social_method,
            llm_model=args.llm_model,
        )
    elif args.social_method == "nn":
        if args.layout_json is None:
            raise ValueError("--social-method nn 需要指定 --layout-json")
        from experiments.social_nav.llm_costmap import build_nn_costmap
        social_costmap = build_nn_costmap(
            layout_json_path=args.layout_json,
            scene_map=scene_map,
            grid_shape=grid_free.shape,
            checkpoint_path=args.nn_checkpoint,
            robot_pos=tuple(start_xy),
            robot_goal=tuple(goal_xy),
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
        collision_penalty=args.collision_penalty,
        clearance_threshold=args.clearance_threshold,
        clearance_weight=args.clearance_weight,
        control_weight=args.control_weight,
        smoothness_weight=args.smoothness_weight,
        social_costmap=social_costmap,
        social_weight=args.social_weight,
        seed=args.seed,
    )
    nav.reset()

    # --- 解析摄像头 ID（用于 viewer 视角切换）---
    robot_cam_id = _find_camera(model, args.robot_camera_name)
    follower_cam_id = _find_camera(model, args.follower_camera_name)
    if args.viewer_camera == "robot" and robot_cam_id is None:
        raise RuntimeError(f"找不到机器人摄像头: {args.robot_camera_name}")

    # --- 打开 viewer（或 headless 模式）---
    viewer_cm = _open_viewer(model, data, args) if not args.no_viewer else None

    # ==========================================================================
    # 主循环
    # ==========================================================================
    step = 0
    reached_goal = False
    quit_flag = False   # 按 Q 退出

    def _key_cb(keycode: int) -> None:
        nonlocal quit_flag
        if keycode in (ord("q"), ord("Q")):
            quit_flag = True

    # 重新开 viewer 并传入按键回调（headless 时跳过）
    if not args.no_viewer:
        viewer_cm = _open_viewer(model, data, args, key_callback=_key_cb)

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
                    if args.no_viewer:
                        break
                    time.sleep(max(0.0, args.wall_sleep))
                    continue

                # MPPI 求解动作
                robot_state = np.array([bx, by, th], dtype=np.float32)
                v, w, info = nav.step(robot_state, goal_xy)

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

                # 打印状态
                if args.log_every > 0 and step % args.log_every == 0:
                    print(
                        f"[步骤 {step:4d}]  距目标={dist:.3f}m  "
                        f"v={v:.3f}m/s  w={math.degrees(w):.1f}°/s  "
                        f"最优代价={info['best_cost']:.2f}  "
                        f"最小间距={info['best_clearance']:.3f}m"
                    )

            if viewer is not None:
                viewer.sync()
            time.sleep(max(0.0, args.wall_sleep))
            step += 1

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
    return {"reached_goal": reached_goal, "steps": step, "final_dist": final_dist}


# =============================================================================
# 5. 辅助函数
# =============================================================================

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


def _open_viewer(model, data, args, key_callback=None):
    """打开 MuJoCo viewer，返回 context manager。"""
    import mujoco.viewer as mj_viewer
    try:
        return mj_viewer.launch_passive(
            model, data,
            key_callback=key_callback,
            show_left_ui=args.show_ui,
            show_right_ui=args.show_ui,
        )
    except RuntimeError as e:
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
