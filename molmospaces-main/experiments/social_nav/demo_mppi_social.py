"""
Standalone MPPI Social Navigation Demo
=======================================
架构：A* 全局规划（在 social costmap 上）→ Multi-modal MPPI 局部跟踪执行。
不需要 MuJoCo。

用法：
    # 有社交代价（A* 绕开人群）
    ./.venv/bin/python3 experiments/social_nav/demo_mppi_social.py

    # 对比：同时画无社交 vs 有社交两条路径
    ./.venv/bin/python3 experiments/social_nav/demo_mppi_social.py --compare
"""

from __future__ import annotations

import heapq
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt, zoom

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.social_nav.llm_costmap import (
    COARSE_GRID_SIZE,
    OBSTACLE_CATEGORIES,
    SCENE_X_MAX,
    SCENE_X_MIN,
    SCENE_Y_MAX,
    SCENE_Y_MIN,
    build_param_costmap,
    extract_scene,
)
from molmo_spaces.policy.solvers.navigation.mppi_core import (
    MPPIConfig,
    MultiModalMPPIController,
    make_downscaled_grid,
)

# ---------------------------------------------------------------------------
# 地图参数
# ---------------------------------------------------------------------------
PX_PER_M     = 8       # 8 px/m → 64×64 occupancy
DOWNSCALE    = 4       # 16×16 grid_free，与 coarse costmap 对齐
GRID_SPACING = DOWNSCALE / PX_PER_M   # 0.5 m/cell
OBS_RADIUS_M = 0.4


# ---------------------------------------------------------------------------
# Mock SceneMap
# ---------------------------------------------------------------------------

class _MockSceneMap:
    def __init__(self, occupancy: np.ndarray, world_to_map: np.ndarray) -> None:
        self.occupancy    = occupancy
        self.world_to_map = world_to_map


def build_mock_scene_map(layout_json_path: str | Path) -> _MockSceneMap:
    with open(layout_json_path) as f:
        data = json.load(f)

    H = round((SCENE_Y_MAX - SCENE_Y_MIN) * PX_PER_M)
    W = round((SCENE_X_MAX - SCENE_X_MIN) * PX_PER_M)

    world_to_map = np.array([
        [0.0,      -PX_PER_M, 0.0,  SCENE_Y_MAX * PX_PER_M],
        [PX_PER_M,  0.0,      0.0, -SCENE_X_MIN * PX_PER_M],
    ], dtype=np.float64)

    occ = np.ones((H, W), dtype=bool)
    occ[:1, :] = occ[-1:, :] = occ[:, :1] = occ[:, -1:] = False

    r_px = max(1, round(OBS_RADIUS_M * PX_PER_M))
    for obj in data.get("objects", []):
        if obj["category"] not in OBSTACLE_CATEGORIES:
            continue
        ox, oy = float(obj["position_xyz"][0]), float(obj["position_xyz"][1])
        orow = round((SCENE_Y_MAX - oy) * PX_PER_M)
        ocol = round((ox - SCENE_X_MIN) * PX_PER_M)
        for dr in range(-r_px, r_px + 1):
            for dc in range(-r_px, r_px + 1):
                if dr * dr + dc * dc <= r_px * r_px:
                    nr, nc = orow + dr, ocol + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        occ[nr, nc] = False

    return _MockSceneMap(occ, world_to_map)


# ---------------------------------------------------------------------------
# 坐标转换
# ---------------------------------------------------------------------------

def rc_to_world(row: int, col: int) -> tuple[float, float]:
    cell_m = (SCENE_X_MAX - SCENE_X_MIN) / COARSE_GRID_SIZE
    return (
        SCENE_X_MIN + (col + 0.5) * cell_m,
        SCENE_Y_MAX - (row + 0.5) * cell_m,
    )


# ---------------------------------------------------------------------------
# A* 全局路径规划
# ---------------------------------------------------------------------------

# 8 方向移动：上下左右 + 对角线
_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1),
         (-1, -1), (-1, 1), (1, -1), (1, 1)]
_COSTS = [1.0, 1.0, 1.0, 1.0,
          math.sqrt(2), math.sqrt(2), math.sqrt(2), math.sqrt(2)]


def plan_astar(
    grid_free:      np.ndarray,
    social_costmap: np.ndarray | None,
    start_rc:       tuple[int, int],
    goal_rc:        tuple[int, int],
    social_w:       float = 0.0,
) -> list[tuple[int, int]]:
    """
    在 grid_free 上跑 A*，格子代价 = 移动代价 + social_w * social_costmap。
    返回 [(row, col), ...] 路径（含起点和终点）。
    """
    H, W = grid_free.shape

    # 每个格子的额外代价（仅自由格有意义）
    extra: np.ndarray = np.zeros((H, W), dtype=np.float32)
    if social_costmap is not None and social_w > 0:
        extra = social_w * social_costmap.astype(np.float32)

    def heuristic(r: int, c: int) -> float:
        return math.hypot(r - goal_rc[0], c - goal_rc[1])

    dist = np.full((H, W), np.inf, dtype=np.float64)
    prev: dict[tuple, tuple] = {}
    dist[start_rc] = 0.0
    heap: list[tuple[float, tuple[int, int]]] = [(heuristic(*start_rc), start_rc)]

    while heap:
        f, (r, c) = heapq.heappop(heap)
        if (r, c) == goal_rc:
            break
        if f > dist[r, c] + heuristic(r, c) + 1e-9:
            continue
        for (dr, dc), move_cost in zip(_DIRS, _COSTS):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W and grid_free[nr, nc]):
                continue
            new_g = dist[r, c] + move_cost * (1.0 + extra[nr, nc])
            if new_g < dist[nr, nc]:
                dist[nr, nc] = new_g
                prev[(nr, nc)] = (r, c)
                heapq.heappush(heap, (new_g + heuristic(nr, nc), (nr, nc)))

    # 回溯路径
    if goal_rc not in prev and start_rc != goal_rc:
        raise RuntimeError(f"A* 找不到路径: {start_rc} → {goal_rc}")

    path: list[tuple[int, int]] = []
    cur = goal_rc
    while cur in prev:
        path.append(cur)
        cur = prev[cur]
    path.append(start_rc)
    path.reverse()
    return path


def path_to_world(path: list[tuple[int, int]]) -> np.ndarray:
    """将格子路径转为世界坐标数组 (N, 2)。"""
    return np.array([rc_to_world(r, c) for r, c in path], dtype=np.float32)


# ---------------------------------------------------------------------------
# 主仿真
# ---------------------------------------------------------------------------

def run_episode(
    layout_json:    str,
    scene_map:      _MockSceneMap,
    grid_free:      np.ndarray,
    dist_transform: np.ndarray,
    social_costmap: np.ndarray | None,
    start_rc:       tuple[int, int],
    goal_rc:        tuple[int, int],
    social_weight:  float,
    max_steps:      int,
    goal_radius:    float,
    waypoint_radius: float,
    horizon:        int   = 32,
    num_samples:    int   = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """
    单次仿真：A* 规划 → Multi-modal MPPI 跟踪。
    返回 (trajectory, astar_path_world)。
    """
    # A* 全局路径
    astar_social_w = 8.0 if (social_costmap is not None and social_weight > 0) else 0.0
    path_rc = plan_astar(grid_free, social_costmap, start_rc, goal_rc,
                         social_w=astar_social_w)
    waypoints = path_to_world(path_rc)   # (N, 2) 世界坐标
    print(f"[A*] 路径长度: {len(path_rc)} 个格子, social_w={astar_social_w}")

    # MPPI 控制器
    cfg = MPPIConfig(
        horizon=horizon, num_samples=num_samples, num_iters=3,
        dt=0.1, temperature=1.0, noise_alpha=0.8,
        noise_v=0.10, noise_w=math.radians(30),
        v_min=-0.05, v_max=0.45, w_max=math.radians(120),
        cruise_speed=0.22,
        goal_stage_weight=4.0,
        goal_terminal_weight=20.0,
        heading_terminal_weight=0.5,
        collision_penalty=500.0,
        clearance_threshold=0.28,
        clearance_weight=25.0,
        control_weight=0.15,
        smoothness_weight=1.5,
        social_weight=social_weight,
    )
    ctrl = MultiModalMPPIController(
        scene_map=scene_map,
        grid_free=grid_free,
        distance_transform=dist_transform,
        grid_spacing=GRID_SPACING,
        downscale=DOWNSCALE,
        config=cfg,
        rng=np.random.default_rng(42),
        social_costmap=social_costmap,
    )

    # 仿真循环：跟踪 waypoints
    start_xy = np.array(rc_to_world(*start_rc), dtype=np.float32)
    goal_xy  = np.array(rc_to_world(*goal_rc),  dtype=np.float32)
    state    = np.array([start_xy[0], start_xy[1], 0.0], dtype=np.float32)

    wp_idx   = 0          # 当前目标 waypoint 索引
    trajectory = [state[:2].copy()]

    for step in range(max_steps):
        # 跳过已经到达的 waypoints
        while wp_idx < len(waypoints) - 1:
            if np.linalg.norm(state[:2] - waypoints[wp_idx]) < waypoint_radius:
                wp_idx += 1
            else:
                break

        current_target = waypoints[wp_idx]

        # 最终目标到达判断
        if np.linalg.norm(state[:2] - goal_xy) < goal_radius:
            print(f"  ✓ 到达终点！步数={step}")
            break

        action, info = ctrl.command(state, current_target)
        v, w = float(action[0]), float(action[1])
        state[0] += cfg.dt * v * math.cos(state[2])
        state[1] += cfg.dt * v * math.sin(state[2])
        state[2]  = math.atan2(
            math.sin(state[2] + cfg.dt * w),
            math.cos(state[2] + cfg.dt * w),
        )
        trajectory.append(state[:2].copy())

        if step % 100 == 0:
            print(f"  step {step:4d} | pos=({state[0]:.2f},{state[1]:.2f}) "
                  f"| wp={wp_idx}/{len(waypoints)-1} "
                  f"| dist_goal={np.linalg.norm(state[:2]-goal_xy):.2f}m")
    else:
        print(f"  ✗ 达到最大步数，最终距离={np.linalg.norm(state[:2]-goal_xy):.2f}m")

    return np.array(trajectory), waypoints


def run_demo(
    layout_json:    str   = "assets/layouts/FloorPlan203_object_positions.json",
    start_rc:       tuple = (11, 13),
    goal_rc:        tuple = (2, 2),
    social_weight:  float = 8.0,
    max_steps:      int   = 600,
    goal_radius:    float = 0.4,
    waypoint_radius: float = 0.6,
    compare:        bool  = False,
    horizon:        int   = 32,
    num_samples:    int   = 512,
    save_fig:       str   = "experiments/social_nav/demo_output.png",
) -> None:
    scene      = extract_scene(layout_json)
    scene_map  = build_mock_scene_map(layout_json)
    grid_free  = make_downscaled_grid(scene_map, DOWNSCALE)
    dist_tf    = distance_transform_edt(grid_free).astype(np.float32)
    social_cm  = build_param_costmap(scene, grid_shape=grid_free.shape, method="rule", verbose=True)

    print(f"[demo] {len(scene.humans)} 人, {len(scene.groups)} 个对话群组")
    print(f"[demo] grid_free {grid_free.shape}, spacing={GRID_SPACING}m/cell")

    if compare:
        # --- 对比：无社交 vs 有社交 ---
        print("\n=== 无社交代价（baseline）===")
        traj_base, path_base = run_episode(
            layout_json, scene_map, grid_free, dist_tf,
            social_costmap=None,
            start_rc=start_rc, goal_rc=goal_rc,
            social_weight=0.0,
            max_steps=max_steps, goal_radius=goal_radius,
            waypoint_radius=waypoint_radius,
            horizon=horizon, num_samples=num_samples,
        )
        print("\n=== 有社交代价 ===")
        traj_social, path_social = run_episode(
            layout_json, scene_map, grid_free, dist_tf,
            social_costmap=social_cm,
            start_rc=start_rc, goal_rc=goal_rc,
            social_weight=social_weight,
            max_steps=max_steps, goal_radius=goal_radius,
            waypoint_radius=waypoint_radius,
            horizon=horizon, num_samples=num_samples,
        )
        _plot_compare(scene, grid_free, social_cm,
                      traj_base, path_base,
                      traj_social, path_social,
                      start_rc, goal_rc, save_fig)
    else:
        print("\n=== 有社交代价 ===")
        traj, path = run_episode(
            layout_json, scene_map, grid_free, dist_tf,
            social_costmap=social_cm,
            start_rc=start_rc, goal_rc=goal_rc,
            social_weight=social_weight,
            max_steps=max_steps, goal_radius=goal_radius,
            waypoint_radius=waypoint_radius,
            horizon=horizon, num_samples=num_samples,
        )
        _plot_single(scene, grid_free, social_cm, traj, path, start_rc, goal_rc, save_fig)


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

_EXTENT = [SCENE_X_MIN, SCENE_X_MAX, SCENE_Y_MIN, SCENE_Y_MAX]


def _draw_base(ax, grid_free, social_costmap=None):
    """画地图底图（占据格 + 可选社交热力图）。"""
    if social_costmap is not None:
        im = ax.imshow(social_costmap, cmap="YlOrRd", vmin=0, vmax=1,
                       extent=_EXTENT, origin="upper", alpha=0.55)
        plt.colorbar(im, ax=ax, shrink=0.6, label="social cost")
    ax.imshow(grid_free.astype(float), cmap="gray", vmin=0, vmax=1,
              extent=_EXTENT, origin="upper", alpha=0.25)
    ax.set_xlim(SCENE_X_MIN, SCENE_X_MAX)
    ax.set_ylim(SCENE_Y_MIN, SCENE_Y_MAX)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.2)


def _draw_humans(ax, scene):
    """画人物位置、朝向箭头、个人空间圆、对话连线。"""
    colors = ["tab:orange", "tab:cyan", "tab:green"]
    for i, h in enumerate(scene.humans):
        c = colors[i % len(colors)]
        ax.plot(*h.pos, "o", color=c, ms=11, zorder=7,
                markeredgecolor="k", markeredgewidth=1.2)
        yaw = math.radians(h.yaw_deg)
        ax.annotate("", xy=(h.pos[0] + 0.45 * math.cos(yaw),
                             h.pos[1] + 0.45 * math.sin(yaw)),
                    xytext=h.pos,
                    arrowprops=dict(arrowstyle="->", color=c, lw=1.8), zorder=8)
        ax.text(h.pos[0] + 0.12, h.pos[1] + 0.18,
                f"P{i}({h.state})", fontsize=7.5, zorder=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.65))
        ax.add_patch(plt.Circle(h.pos, 1.0, color=c, fill=False,
                                linestyle="--", lw=1, alpha=0.45))
    for grp in scene.groups:
        if len(grp) >= 2:
            p0, p1 = scene.humans[grp[0]].pos, scene.humans[grp[1]].pos
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], "r-", lw=2, alpha=0.5, zorder=6)


def _draw_start_goal(ax, start_rc, goal_rc):
    sx, sy = rc_to_world(*start_rc)
    gx, gy = rc_to_world(*goal_rc)
    ax.plot(sx, sy, "go", ms=12, zorder=9, label="start")
    ax.plot(gx, gy, "r*", ms=16, zorder=9, label="goal")


def _plot_single(scene, grid_free, social_cm, traj, astar_path,
                 start_rc, goal_rc, save_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    _draw_base(ax, grid_free, social_cm)
    _draw_humans(ax, scene)
    ax.plot(astar_path[:, 0], astar_path[:, 1], "b--", lw=1.5,
            alpha=0.6, label="A* global path")
    ax.plot(traj[:, 0], traj[:, 1], "b-", lw=2.2, label="MPPI trajectory", zorder=6)
    _draw_start_goal(ax, start_rc, goal_rc)
    ax.set_title("Social Navigation: A* + Multi-modal MPPI")
    ax.legend(fontsize=8, loc="lower right")
    _save(fig, save_path)


def _plot_compare(scene, grid_free, social_cm,
                  traj_base, path_base,
                  traj_social, path_social,
                  start_rc, goal_rc, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # 左：无社交
    ax = axes[0]
    _draw_base(ax, grid_free, social_costmap=None)
    _draw_humans(ax, scene)
    ax.plot(path_base[:, 0], path_base[:, 1], "g--", lw=1.5, alpha=0.6,
            label="A* path (no social)")
    ax.plot(traj_base[:, 0], traj_base[:, 1], "g-", lw=2.2,
            label="MPPI traj (no social)", zorder=6)
    _draw_start_goal(ax, start_rc, goal_rc)
    ax.set_title("Baseline (no social cost)")
    ax.legend(fontsize=8, loc="lower right")

    # 右：有社交
    ax = axes[1]
    _draw_base(ax, grid_free, social_cm)
    _draw_humans(ax, scene)
    ax.plot(path_social[:, 0], path_social[:, 1], "b--", lw=1.5, alpha=0.6,
            label="A* path (social)")
    ax.plot(traj_social[:, 0], traj_social[:, 1], "b-", lw=2.2,
            label="MPPI traj (social)", zorder=6)
    _draw_start_goal(ax, start_rc, goal_rc)
    ax.set_title("Social Navigation (with social cost)")
    ax.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    _save(fig, save_path)


def _save(fig, path: str):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[demo] 保存 → {out}")
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--layout-json", default="assets/layouts/FloorPlan203_object_positions.json")
    p.add_argument("--start",  default="11,13")
    p.add_argument("--goal",   default="2,2")
    p.add_argument("--social-weight", type=float, default=8.0)
    p.add_argument("--max-steps",     type=int,   default=600)
    p.add_argument("--compare", action="store_true",
                   help="同时跑无社交和有社交，对比画图")
    p.add_argument("--horizon",     type=int, default=32)
    p.add_argument("--num-samples", type=int, default=512)
    p.add_argument("--save-fig", default="experiments/social_nav/demo_output.png")
    args = p.parse_args()

    run_demo(
        layout_json   = args.layout_json,
        start_rc      = tuple(int(v) for v in args.start.split(",")),
        goal_rc       = tuple(int(v) for v in args.goal.split(",")),
        social_weight = args.social_weight,
        max_steps     = args.max_steps,
        compare       = args.compare,
        horizon       = args.horizon,
        num_samples   = args.num_samples,
        save_fig      = args.save_fig,
    )
