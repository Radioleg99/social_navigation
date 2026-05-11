"""
Stage1 static social path planning (no MPPI, no dynamic replanning).

Goal:
- In static scenes, find a global path that minimizes geometric + social cost.
- Keep only essential components for Stage1 validation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

REPO_ROOT = Path(__file__).resolve().parents[2]
GRAD_ROOT = REPO_ROOT.parent
for _p in (str(REPO_ROOT), str(GRAD_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.social_nav.cost.llm_costmap import build_entity_params, synthesize_costmap
from experiments.social_nav.run_social_nav import _astar_on_grid, _world_to_grid_rc, build_map
from molmo_spaces.policy.solvers.navigation.mppi_core import resolve_free_xy
from pipeline.scene_bridge import scene_graph_to_scene_description
from scripts import manual_capture_rgbd as scene_loader


def _parse_xy(s: str) -> tuple[float, float]:
    vals = [float(v.strip()) for v in s.split(",")]
    if len(vals) != 2:
        raise argparse.ArgumentTypeError("Expected x,y")
    return vals[0], vals[1]


def _parse_xyz(s: str) -> tuple[float, float, float]:
    vals = [float(v.strip()) for v in s.split(",")]
    if len(vals) != 3:
        raise argparse.ArgumentTypeError("Expected x,y,yaw_deg")
    return vals[0], vals[1], vals[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage1 static social A* planner")
    p.add_argument("--scene-xml", type=Path, required=True)
    p.add_argument("--layout-json", type=Path, default=None)
    p.add_argument("--layout-runtime", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--robot-type", choices=["franka", "rby1", "rby1m", "navbot"], default="navbot")

    p.add_argument("--start-pose", type=_parse_xyz, required=True)
    p.add_argument("--goal-xy", type=_parse_xy, required=True)

    p.add_argument("--agent-radius", type=float, default=0.22)
    p.add_argument("--social-method", choices=["none", "rule", "llm"], default="rule")
    p.add_argument("--llm-model", type=str, default="moonshot-v1-8k")
    p.add_argument("--scene-graph", type=Path, default=None)

    p.add_argument("--astar-social-weight", type=float, default=30.0)
    p.add_argument("--astar-human-block-radius", type=float, default=0.65)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-topdown", type=Path, default=None, help="Save static topdown visualization PNG")
    p.add_argument("--show-topdown", action="store_true", help="Show static topdown visualization window")
    p.add_argument("--out-json", type=Path, default=None, help="Optional: save full numeric result JSON")
    return p.parse_args()


def _load_scene(args: argparse.Namespace):
    model, data, source = scene_loader.load_scene_with_optional_layout_runtime(
        args.scene_xml, args.layout_json, args.layout_runtime, robot_type_override=args.robot_type
    )
    print(f"[scene] loaded from: {source}")
    return model, data


def _snap_to_free(scene_map, grid_free, xy: np.ndarray, downscale: int, label: str) -> np.ndarray:
    snapped = resolve_free_xy(scene_map, grid_free, xy, downscale)
    if snapped is None:
        raise RuntimeError(f"{label} {tuple(xy)} cannot be projected to free space")
    if not np.allclose(snapped, xy, atol=1e-3):
        print(f"[snap] {label}: {tuple(np.round(xy,3))} -> {tuple(np.round(snapped,3))}")
    return snapped


def _block_humans_on_grid(
    scene_map,
    grid_free: np.ndarray,
    downscale: int,
    grid_spacing: float,
    human_positions: list[tuple[float, float]],
    radius_m: float,
) -> np.ndarray:
    if not human_positions or radius_m <= 0.0:
        return grid_free
    out = grid_free.copy()
    H, W = out.shape
    radius_cells = max(int(math.ceil(radius_m / max(grid_spacing, 1e-6))), 1)
    for pos in human_positions:
        hr, hc = _world_to_grid_rc(scene_map, np.asarray(pos, dtype=np.float32), downscale, out.shape)
        r0, r1 = max(0, hr - radius_cells), min(H, hr + radius_cells + 1)
        c0, c1 = max(0, hc - radius_cells), min(W, hc + radius_cells + 1)
        for r in range(r0, r1):
            for c in range(c0, c1):
                if math.hypot(r - hr, c - hc) * grid_spacing <= radius_m:
                    out[r, c] = False
    return out


def _sample_cost_along_path(
    path_xy: np.ndarray,
    scene_map,
    grid_shape: tuple[int, int],
    downscale: int,
    social_costmap: np.ndarray | None,
    step_m: float = 0.05,
) -> tuple[float, float]:
    if social_costmap is None or len(path_xy) < 2:
        length = float(np.sum(np.linalg.norm(np.diff(path_xy, axis=0), axis=1))) if len(path_xy) > 1 else 0.0
        return length, 0.0
    social_sum = 0.0
    length = 0.0
    for a, b in zip(path_xy[:-1], path_xy[1:]):
        seg = b - a
        seg_len = float(np.linalg.norm(seg))
        length += seg_len
        n = max(int(math.ceil(seg_len / max(step_m, 1e-3))), 1)
        for alpha in np.linspace(0.0, 1.0, n + 1):
            xy = a + alpha * seg
            r, c = _world_to_grid_rc(scene_map, xy, downscale, grid_shape)
            social_sum += float(social_costmap[r, c])
    return length, social_sum


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    _ = _load_scene(args)  # keeps compatibility with layout-runtime checks

    DOWNSCALE = 5
    PX_PER_M = 200
    scene_map, grid_free, grid_free_astar, dist_transform, grid_spacing = build_map(
        args.scene_xml, agent_radius=args.agent_radius, downscale=DOWNSCALE, px_per_m=PX_PER_M
    )

    start_xy = np.array([args.start_pose[0], args.start_pose[1]], dtype=np.float32)
    goal_xy = np.array([args.goal_xy[0], args.goal_xy[1]], dtype=np.float32)
    start_xy = _snap_to_free(scene_map, grid_free, start_xy, DOWNSCALE, "start")
    goal_xy = _snap_to_free(scene_map, grid_free, goal_xy, DOWNSCALE, "goal")

    social_costmap = None
    human_positions: list[tuple[float, float]] = []
    llm_log = ""
    if args.social_method in ("rule", "llm"):
        if args.scene_graph is None:
            raise ValueError("--social-method rule/llm requires --scene-graph")
        scene = scene_graph_to_scene_description(args.scene_graph)
        human_positions = [h.pos for h in scene.humans]
        params, llm_log = build_entity_params(
            scene,
            method=args.social_method,
            llm_model=args.llm_model,
            verbose=True,
            robot_pos=tuple(start_xy),
            robot_goal=tuple(goal_xy),
        )
        occ = scene_map.occupancy
        corners_px = np.array([[0, 0], [occ.shape[0] - 1, occ.shape[1] - 1]], dtype=float)
        corners_m = scene_map.pos_px_to_m(corners_px)
        x_range = (float(min(corners_m[:, 0])), float(max(corners_m[:, 0])))
        y_range = (float(min(corners_m[:, 1])), float(max(corners_m[:, 1])))
        social_costmap = synthesize_costmap(
            params,
            grid_free_astar.shape,
            x_range=x_range,
            y_range=y_range,
            distance_transform=dist_transform,
            clearance_cap=0.5,
            clearance_weight=0.3,
        )

    grid_free_astar = _block_humans_on_grid(
        scene_map,
        grid_free_astar,
        DOWNSCALE,
        grid_spacing,
        human_positions,
        float(args.astar_human_block_radius),
    )

    path = _astar_on_grid(
        scene_map=scene_map,
        grid_free=grid_free_astar,
        downscale=DOWNSCALE,
        grid_spacing=grid_spacing,
        start_xy=start_xy,
        goal_xy=goal_xy,
        social_costmap=social_costmap,
        social_w=float(args.astar_social_weight),
        distance_transform=dist_transform,
        clearance_w=3.0,
        clearance_cap=0.5,
    )
    if path is None or len(path) < 2:
        raise RuntimeError("No valid static social path found")

    path_length, social_sum = _sample_cost_along_path(
        path, scene_map, grid_free_astar.shape, DOWNSCALE, social_costmap
    )
    straight = np.vstack([start_xy, goal_xy]).astype(np.float32)
    straight_len, straight_social_sum = _sample_cost_along_path(
        straight, scene_map, grid_free_astar.shape, DOWNSCALE, social_costmap
    )

    result = {
        "scene_xml": str(args.scene_xml),
        "layout_json": str(args.layout_json) if args.layout_json else None,
        "scene_graph": str(args.scene_graph) if args.scene_graph else None,
        "social_method": args.social_method,
        "start_xy": [float(start_xy[0]), float(start_xy[1])],
        "goal_xy": [float(goal_xy[0]), float(goal_xy[1])],
        "num_waypoints": int(len(path)),
        "path_waypoints": [[float(p[0]), float(p[1])] for p in path],
        "path_length_m": float(path_length),
        "path_social_sum": float(social_sum),
        "straight_length_m": float(straight_len),
        "straight_social_sum": float(straight_social_sum),
        "social_improvement_vs_straight": float(straight_social_sum - social_sum),
        "llm_log": llm_log,
    }

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, indent=2))
    print(f"[stage1-static] path waypoints: {len(path)}")
    print(
        f"[stage1-static] path_len={path_length:.3f}m social_sum={social_sum:.3f} | "
        f"straight_social={straight_social_sum:.3f}"
    )
    if llm_log:
        print("\n[reasoning-log]")
        print(llm_log)

    try:
        from experiments.social_nav.topdown_viz import TopdownViz

        viz = TopdownViz(scene_map, grid_free, grid_spacing, DOWNSCALE)
        viz.set_start_goal(start_xy, goal_xy)
        if human_positions:
            scene = scene_graph_to_scene_description(args.scene_graph) if args.scene_graph is not None else None
            if scene is not None:
                viz.set_humans(scene.humans)
        if social_costmap is not None:
            viz.set_social_costmap(social_costmap)
        viz.set_astar_path(path)
        if llm_log:
            viz.set_llm_log(llm_log)
        viz.update(start_xy, capture_frame=False)
        if args.save_topdown is not None:
            viz.save(args.save_topdown)
            print(f"[stage1-static] saved topdown -> {args.save_topdown}")
        if args.show_topdown:
            import cv2

            tmp_img = Path("outputs/_stage1_static_preview.png")
            tmp_img.parent.mkdir(parents=True, exist_ok=True)
            viz.save(tmp_img, verbose=False)
            img = cv2.imread(str(tmp_img))
            if img is None:
                raise RuntimeError(f"failed to load preview image: {tmp_img}")
            cv2.imshow("Stage1 Static Social Path", img)
            print("[stage1-static] press any key to close topdown window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        viz.close()
    except Exception as e:
        print(f"[stage1-static] visualization skipped: {e}")
    if args.out_json is not None:
        print(f"[stage1-static] saved result -> {args.out_json}")


if __name__ == "__main__":
    main()
