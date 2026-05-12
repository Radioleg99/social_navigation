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
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import label as label_components

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[2]
GRAD_ROOT = REPO_ROOT.parent
for _p in (str(REPO_ROOT), str(GRAD_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.social_nav.cost.llm_costmap import build_entity_params, synthesize_costmap
from experiments.social_nav.run_social_nav import (
    _plan_global_path,
    _postprocess_astar_path,
    _world_to_grid_rc,
    build_map,
)
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


def _parse_bounds(s: str) -> tuple[float, float, float, float]:
    vals = [float(v.strip()) for v in s.split(",")]
    if len(vals) != 4:
        raise argparse.ArgumentTypeError("Expected x_min,x_max,y_min,y_max")
    if vals[1] <= vals[0] or vals[3] <= vals[2]:
        raise argparse.ArgumentTypeError("Bounds must satisfy x_max>x_min and y_max>y_min")
    return vals[0], vals[1], vals[2], vals[3]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage1 static social A* planner")
    p.add_argument("--scene-xml", type=Path, default=None,
                   help="MuJoCo scene XML. Not needed when --birdview-image is provided")
    p.add_argument("--birdview-image", type=Path, default=None,
                   help="Optional occupancy/bird-view image for pure 2D Stage1 planning")
    p.add_argument("--map-bounds", type=_parse_bounds, default=None,
                   help="World bounds for --birdview-image as x_min,x_max,y_min,y_max")
    p.add_argument("--birdview-free-threshold", type=float, default=0.5,
                   help="Grayscale threshold for free space in --birdview-image")
    p.add_argument("--birdview-free-is-dark", action="store_true", default=False,
                   help="Treat dark pixels as free instead of light pixels")
    p.add_argument("--birdview-downscale", type=int, default=5,
                   help="Conservative downscale factor for image occupancy")
    p.add_argument("--layout-json", type=Path, default=None)
    p.add_argument("--layout-runtime", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--validate-layout-runtime", action="store_true", default=False,
                   help="Load the assembled MuJoCo scene once to validate layout runtime wiring")
    p.add_argument("--robot-type", choices=["franka", "rby1", "rby1m", "navbot"], default="navbot")

    p.add_argument("--start-pose", type=_parse_xyz, required=True)
    p.add_argument("--goal-xy", type=_parse_xy, required=True)

    p.add_argument("--agent-radius", type=float, default=0.22)
    p.add_argument("--social-method", choices=["none", "rule", "llm"], default="rule")
    p.add_argument("--llm-model", type=str, default="moonshot-v1-8k")
    p.add_argument("--scene-graph", type=Path, default=None)

    p.add_argument("--astar-social-weight", type=float, default=30.0)
    p.add_argument("--astar-human-block-radius", type=float, default=0.65)
    p.add_argument("--astar-num-candidates", type=int, default=3,
                   help="Generate multiple diverse A* candidates and choose the best path")
    p.add_argument("--astar-diversity-penalty", type=float, default=8.0,
                   help="Soft penalty around previously found paths when generating candidates")
    p.add_argument("--astar-candidate-clearance-weight", type=float, default=8.0,
                   help="Candidate scoring weight for low-clearance/narrow regions")
    p.add_argument("--astar-smoothing", choices=["none", "shortcut"], default="shortcut",
                   help="Post-process raw A* path for a cleaner static route")
    p.add_argument("--astar-shortcut-social-threshold", type=float, default=0.45,
                   help="Maximum social cost allowed on shortcut-smoothed segments")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-topdown", type=Path, default=None, help="Save static topdown visualization PNG")
    p.add_argument("--show-topdown", action="store_true", default=False,
                   help="Open a pure 2D bird-view window after planning")
    p.add_argument("--out-json", type=Path, default=None, help="Optional: save full numeric result JSON")
    return p.parse_args()


def _load_scene(args: argparse.Namespace):
    if args.scene_xml is None:
        raise ValueError("--scene-xml is required for layout runtime validation")
    model, data, source = scene_loader.load_scene_with_optional_layout_runtime(
        args.scene_xml, args.layout_json, args.layout_runtime, robot_type_override=args.robot_type
    )
    print(f"[scene] loaded from: {source}")
    return model, data


class _BirdviewSceneMap:
    """Minimal scene_map adapter for image-based Stage1 planning."""

    def __init__(self, occupancy: np.ndarray, bounds: tuple[float, float, float, float]) -> None:
        self.occupancy = occupancy.astype(bool)
        self.x_min, self.x_max, self.y_min, self.y_max = bounds
        self.height, self.width = self.occupancy.shape
        self.x_px_per_m = self.width / (self.x_max - self.x_min)
        self.y_px_per_m = self.height / (self.y_max - self.y_min)
        self.world_to_map = np.array(
            [
                [0.0, -self.y_px_per_m, 0.0, self.y_max * self.y_px_per_m],
                [self.x_px_per_m, 0.0, 0.0, -self.x_min * self.x_px_per_m],
            ],
            dtype=np.float64,
        )

    def pos_m_to_px(self, xyz: np.ndarray) -> np.ndarray:
        arr = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
        row = (self.y_max - arr[:, 1]) * self.y_px_per_m
        col = (arr[:, 0] - self.x_min) * self.x_px_per_m
        return np.stack([row, col], axis=-1).astype(np.float32)

    def pos_px_to_m(self, px: np.ndarray) -> np.ndarray:
        arr = np.asarray(px, dtype=np.float32).reshape(-1, 2)
        x = arr[:, 1] / self.x_px_per_m + self.x_min
        y = self.y_max - arr[:, 0] / self.y_px_per_m
        z = np.zeros_like(x)
        return np.stack([x, y, z], axis=-1).astype(np.float32)


def _make_grid_from_occ(occ: np.ndarray, downscale: int) -> np.ndarray:
    pad_rows = (-occ.shape[0]) % downscale
    pad_cols = (-occ.shape[1]) % downscale
    padded = np.zeros((occ.shape[0] + pad_rows, occ.shape[1] + pad_cols), dtype=bool)
    padded[: occ.shape[0], : occ.shape[1]] = occ
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


def _build_map_from_birdview(args: argparse.Namespace):
    if args.map_bounds is None:
        raise ValueError("--map-bounds is required with --birdview-image")
    if args.birdview_image is None or not args.birdview_image.is_file():
        raise FileNotFoundError(f"Bird-view image not found: {args.birdview_image}")

    import cv2

    img = cv2.imread(str(args.birdview_image), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read bird-view image: {args.birdview_image}")

    gray = img.astype(np.float32) / 255.0
    threshold = float(args.birdview_free_threshold)
    occ = gray <= threshold if args.birdview_free_is_dark else gray >= threshold
    scene_map = _BirdviewSceneMap(occ, args.map_bounds)

    downscale = max(int(args.birdview_downscale), 1)
    grid_free = _make_grid_from_occ(scene_map.occupancy, downscale)
    grid_spacing_x = (scene_map.x_max - scene_map.x_min) / scene_map.width * downscale
    grid_spacing_y = (scene_map.y_max - scene_map.y_min) / scene_map.height * downscale
    grid_spacing = float(max(grid_spacing_x, grid_spacing_y))
    dist_transform = distance_transform_edt(grid_free).astype(np.float32)
    print(
        f"[map] birdview={args.birdview_image} grid={grid_free.shape} "
        f"free={int(grid_free.sum())} spacing={grid_spacing:.3f}m"
    )
    return scene_map, grid_free, grid_free.copy(), dist_transform, grid_spacing, downscale


def _build_stage1_map(args: argparse.Namespace):
    if args.birdview_image is not None:
        return _build_map_from_birdview(args)
    if args.scene_xml is None:
        raise ValueError("Either --scene-xml or --birdview-image must be provided")
    downscale = 5
    px_per_m = 200
    scene_map, grid_free, grid_free_astar, dist_transform, grid_spacing = build_map(
        args.scene_xml, agent_radius=args.agent_radius, downscale=downscale, px_per_m=px_per_m
    )
    return scene_map, grid_free, grid_free_astar, dist_transform, grid_spacing, downscale


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


def _ensure_same_component(
    scene_map,
    grid_free: np.ndarray,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    downscale: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep start/goal on the same reachable 2D component for static A*."""
    labels, _num = label_components(grid_free)
    sr, sc = _world_to_grid_rc(scene_map, start_xy, downscale, grid_free.shape)
    gr, gc = _world_to_grid_rc(scene_map, goal_xy, downscale, grid_free.shape)

    src_label = int(labels[sr, sc]) if grid_free[sr, sc] else 0
    if src_label == 0:
        free = np.argwhere(grid_free)
        if free.size == 0:
            raise RuntimeError("No free cells remain after hard constraints")
        nearest_idx = int(np.argmin(np.sum((free - np.array([sr, sc])) ** 2, axis=1)))
        sr, sc = map(int, free[nearest_idx])
        src_label = int(labels[sr, sc])
        start_xy = scene_map.pos_px_to_m(
            np.array([[(sr + 0.5) * downscale, (sc + 0.5) * downscale]], dtype=np.float32)
        )[0, :2]
        print(f"[snap] start -> same-component free cell {tuple(np.round(start_xy, 3))}")

    if int(labels[gr, gc]) == src_label and grid_free[gr, gc]:
        return start_xy.astype(np.float32), goal_xy.astype(np.float32)

    component = np.argwhere(labels == src_label)
    if component.size == 0:
        raise RuntimeError("Start component is empty after hard constraints")
    nearest_idx = int(np.argmin(np.sum((component - np.array([gr, gc])) ** 2, axis=1)))
    nr, nc = map(int, component[nearest_idx])
    snapped_goal = scene_map.pos_px_to_m(
        np.array([[(nr + 0.5) * downscale, (nc + 0.5) * downscale]], dtype=np.float32)
    )[0, :2].astype(np.float32)
    print(
        f"[snap] goal: requested component unreachable; "
        f"{tuple(np.round(goal_xy, 3))} -> {tuple(np.round(snapped_goal, 3))}"
    )
    return start_xy.astype(np.float32), snapped_goal


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


def _sample_clearance_along_path(
    path_xy: np.ndarray,
    scene_map,
    grid_shape: tuple[int, int],
    downscale: int,
    distance_transform: np.ndarray,
    grid_spacing: float,
    step_m: float = 0.05,
) -> tuple[float, float]:
    if len(path_xy) < 2:
        return 0.0, 0.0
    values: list[float] = []
    for a, b in zip(path_xy[:-1], path_xy[1:]):
        seg = b - a
        seg_len = float(np.linalg.norm(seg))
        n = max(int(math.ceil(seg_len / max(step_m, 1e-3))), 1)
        for alpha in np.linspace(0.0, 1.0, n + 1):
            r, c = _world_to_grid_rc(scene_map, a + alpha * seg, downscale, grid_shape)
            values.append(float(distance_transform[r, c]) * grid_spacing)
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=np.float32)
    return float(np.min(arr)), float(np.mean(arr))


def run_stage1(args: argparse.Namespace) -> dict:
    np.random.seed(args.seed)
    if getattr(args, "validate_layout_runtime", False):
        _ = _load_scene(args)

    scene_map, grid_free, grid_free_astar, dist_transform, grid_spacing, DOWNSCALE = _build_stage1_map(args)

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
    start_xy, goal_xy = _ensure_same_component(
        scene_map, grid_free_astar, start_xy, goal_xy, DOWNSCALE
    )

    shortest_args = argparse.Namespace(**vars(args))
    shortest_args.astar_num_candidates = 1
    shortest_args.astar_social_weight = 0.0
    shortest_args.astar_candidate_clearance_weight = 0.0
    shortest_raw_path = _plan_global_path(
        scene_map=scene_map,
        grid_free=grid_free_astar,
        downscale=DOWNSCALE,
        grid_spacing=grid_spacing,
        start_xy=start_xy,
        goal_xy=goal_xy,
        social_costmap=None,
        distance_transform=None,
        args=shortest_args,
    )
    if shortest_raw_path is None or len(shortest_raw_path) < 2:
        raise RuntimeError("No valid shortest-path A* baseline found")
    shortest_path = _postprocess_astar_path(
        shortest_raw_path,
        scene_map,
        grid_free_astar,
        DOWNSCALE,
        grid_spacing,
        None,
        shortest_args,
    )

    raw_path = _plan_global_path(
        scene_map=scene_map,
        grid_free=grid_free_astar,
        downscale=DOWNSCALE,
        grid_spacing=grid_spacing,
        start_xy=start_xy,
        goal_xy=goal_xy,
        social_costmap=social_costmap,
        distance_transform=dist_transform,
        args=args,
    )
    if raw_path is None or len(raw_path) < 2:
        raise RuntimeError("No valid static social path found")
    path = _postprocess_astar_path(
        raw_path,
        scene_map,
        grid_free_astar,
        DOWNSCALE,
        grid_spacing,
        social_costmap,
        args,
    )

    path_length, social_sum = _sample_cost_along_path(
        path, scene_map, grid_free_astar.shape, DOWNSCALE, social_costmap
    )
    raw_path_length, raw_social_sum = _sample_cost_along_path(
        raw_path, scene_map, grid_free_astar.shape, DOWNSCALE, social_costmap
    )
    shortest_path_length, shortest_social_sum = _sample_cost_along_path(
        shortest_path, scene_map, grid_free_astar.shape, DOWNSCALE, social_costmap
    )
    min_clearance, mean_clearance = _sample_clearance_along_path(
        path, scene_map, grid_free_astar.shape, DOWNSCALE, dist_transform, grid_spacing
    )
    straight = np.vstack([start_xy, goal_xy]).astype(np.float32)
    straight_len, straight_social_sum = _sample_cost_along_path(
        straight, scene_map, grid_free_astar.shape, DOWNSCALE, social_costmap
    )

    result = {
        "scene_xml": str(args.scene_xml) if args.scene_xml else None,
        "birdview_image": str(args.birdview_image) if args.birdview_image else None,
        "map_bounds": list(args.map_bounds) if args.map_bounds else None,
        "layout_json": str(args.layout_json) if args.layout_json else None,
        "scene_graph": str(args.scene_graph) if args.scene_graph else None,
        "social_method": args.social_method,
        "start_xy": [float(start_xy[0]), float(start_xy[1])],
        "goal_xy": [float(goal_xy[0]), float(goal_xy[1])],
        "num_waypoints": int(len(path)),
        "num_raw_waypoints": int(len(raw_path)),
        "num_shortest_waypoints": int(len(shortest_path)),
        "path_waypoints": [[float(p[0]), float(p[1])] for p in path],
        "raw_path_waypoints": [[float(p[0]), float(p[1])] for p in raw_path],
        "shortest_path_waypoints": [[float(p[0]), float(p[1])] for p in shortest_path],
        "path_length_m": float(path_length),
        "path_social_sum": float(social_sum),
        "raw_path_length_m": float(raw_path_length),
        "raw_path_social_sum": float(raw_social_sum),
        "shortest_path_length_m": float(shortest_path_length),
        "shortest_path_social_sum": float(shortest_social_sum),
        "social_delta_vs_shortest": float(shortest_social_sum - social_sum),
        "length_ratio_vs_shortest": float(path_length / max(shortest_path_length, 1e-6)),
        "path_min_clearance_m": float(min_clearance),
        "path_mean_clearance_m": float(mean_clearance),
        "straight_length_m": float(straight_len),
        "straight_social_sum": float(straight_social_sum),
        "social_improvement_vs_straight": float(straight_social_sum - social_sum),
        "llm_log": llm_log,
    }

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, indent=2))
    print(f"[stage1-static] path waypoints: {len(raw_path)} raw -> {len(path)} final")
    print(
        f"[stage1-static] path_len={path_length:.3f}m social_sum={social_sum:.3f} | "
        f"shortest_len={shortest_path_length:.3f}m shortest_social={shortest_social_sum:.3f} | "
        f"straight_social={straight_social_sum:.3f} | "
        f"clearance_min={min_clearance:.3f}m clearance_mean={mean_clearance:.3f}m"
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
        if getattr(args, "show_topdown", False):
            import cv2

            preview_path = args.save_topdown or Path("outputs/_stage1_topdown_preview.png")
            viz.save(preview_path, verbose=False)
            img = cv2.imread(str(preview_path))
            if img is None:
                raise RuntimeError(f"failed to load topdown preview: {preview_path}")
            cv2.imshow("Stage1 Bird-View Costmap", img)
            print("[stage1-static] showing pure 2D bird-view; press any key to close")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        viz.close()
    except Exception as e:
        print(f"[stage1-static] visualization skipped: {e}")
    if args.out_json is not None:
        print(f"[stage1-static] saved result -> {args.out_json}")
    return result


def main() -> None:
    args = parse_args()
    run_stage1(args)


if __name__ == "__main__":
    main()
