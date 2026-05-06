#!/usr/bin/env python3
"""Minimal iTHOR point-navigation demo with A* path planning for navbot."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import mujoco
import networkx as nx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import manual_capture_rgbd as manual


def parse_start_pose(text: str) -> tuple[float, float, float]:
    vals = [float(x.strip()) for x in text.split(",")]
    if len(vals) != 3:
        raise argparse.ArgumentTypeError("Expected --start-pose as x,y,yaw_deg")
    return vals[0], vals[1], vals[2]


def parse_goal_xy(text: str) -> tuple[float, float]:
    vals = [float(x.strip()) for x in text.split(",")]
    if len(vals) != 2:
        raise argparse.ArgumentTypeError("Expected --goal-xy as x,y")
    return vals[0], vals[1]


def wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def resolve_camera_id_optional(model: mujoco.MjModel, camera_name: str) -> int | None:
    try:
        return manual.resolve_camera_id(model, camera_name)
    except Exception:
        return None


def make_scene_map(scene_xml: Path, agent_radius: float, px_per_m: int = 200):
    from molmo_spaces.utils import scene_maps

    # Avoid importing task_sampler/resource-manager lock machinery from scene map blacklist handling.
    scene_maps._delete_blacklisted_bodies = lambda spec: 0
    ProcTHORMap = scene_maps.ProcTHORMap
    iTHORMap = scene_maps.iTHORMap

    scene_xml_str = str(scene_xml)
    if "ithor" in scene_xml_str:
        return iTHORMap.from_mj_model_path(
            model_path=scene_xml_str,
            agent_radius=agent_radius,
            px_per_m=px_per_m,
        )
    if "procthor" in scene_xml_str or "holodeck" in scene_xml_str:
        return ProcTHORMap.from_mj_model_path(
            model_path=scene_xml_str,
            agent_radius=agent_radius,
            px_per_m=px_per_m,
        )
    raise ValueError(f"Unknown scene type from path: {scene_xml}")


def discretize_location(scene_map, location: np.ndarray, downscale: int) -> np.ndarray:
    return np.floor(scene_map.pos_m_to_px(location) / downscale).astype(np.int32)


def make_downscaled_grid(scene_map, downscale: int) -> np.ndarray:
    grid = scene_map.occupancy.copy().astype(bool)
    padded = np.zeros(
        (
            grid.shape[0] + (downscale - grid.shape[0] % downscale),
            grid.shape[1] + (downscale - grid.shape[1] % downscale),
        ),
        dtype=bool,
    )
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


def resolve_plannable_location(
    graph: nx.Graph,
    scene_map,
    position_m: np.ndarray,
    downscale: int,
    max_search: int = 40,
) -> tuple[int, int] | None:
    discrete = discretize_location(scene_map, position_m, downscale)
    key = tuple(int(v) for v in discrete)
    if key in graph:
        return key
    for search_range in range(1, max_search + 1):
        for dr in range(-search_range, search_range + 1):
            for dc in range(-search_range, search_range + 1):
                if dr != search_range and dc != search_range:
                    continue
                candidate = (key[0] + dr, key[1] + dc)
                if candidate in graph:
                    return candidate
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-xml", type=Path, required=True, help="Path to *_physics.xml scene file")
    parser.add_argument("--layout-json", type=Path, default=None, help="Optional layout json")
    parser.add_argument(
        "--layout-runtime",
        choices=["auto", "on", "off"],
        default="auto",
        help="Compose scene with layout runtime (robot/humans)",
    )
    parser.add_argument(
        "--robot-type",
        choices=["franka", "rby1", "rby1m", "navbot"],
        default="navbot",
        help="Robot type override for layout runtime composition",
    )
    parser.add_argument(
        "--start-pose",
        type=parse_start_pose,
        default=(-1.4, 4.6, 0.0),
        help="Initial robot base pose x,y,yaw_deg",
    )
    parser.add_argument("--goal-xy", type=parse_goal_xy, required=True, help="Goal position x,y")
    parser.add_argument("--goal-radius", type=float, default=0.20, help="Goal success radius in meters")
    parser.add_argument(
        "--waypoint-radius",
        type=float,
        default=0.18,
        help="Waypoint switching radius in meters",
    )
    parser.add_argument("--move-step", type=float, default=0.05, help="Meters per control tick")
    parser.add_argument("--yaw-step-deg", type=float, default=5.0, help="Degrees per control tick")
    parser.add_argument(
        "--turn-in-place-threshold-deg",
        type=float,
        default=12.0,
        help="Rotate in place when heading error exceeds this threshold",
    )
    parser.add_argument("--max-steps", type=int, default=1200, help="Maximum control ticks")
    parser.add_argument(
        "--planner-agent-radius",
        type=float,
        default=0.22,
        help="A* occupancy dilation radius in meters",
    )
    parser.add_argument(
        "--viewer-camera",
        choices=["follower", "robot"],
        default="follower",
        help="Viewer camera to display while navigating",
    )
    parser.add_argument("--robot-camera-name", type=str, default="robot_0/head_camera")
    parser.add_argument("--follower-camera-name", type=str, default="robot_0/camera_follower")
    parser.add_argument("--show-ui", action="store_true", default=False)
    parser.add_argument("--no-viewer", action="store_true", default=False, help="Run headless")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.scene_xml.is_file():
        raise FileNotFoundError(f"scene xml not found: {args.scene_xml}")

    layout_json = args.layout_json
    if layout_json is None:
        layout_json = manual.infer_default_layout_json(args.scene_xml)
        if layout_json is not None:
            print(f"[INFO] Auto-detected layout json: {layout_json}")

    model, data, model_source = manual.load_scene_with_optional_layout_runtime(
        args.scene_xml,
        layout_json,
        args.layout_runtime,
        robot_type_override=args.robot_type,
    )
    print(f"[INFO] model_source={model_source}")

    base_x_adr = manual.find_joint_qposadr(model, "robot_0/base_x")
    base_y_adr = manual.find_joint_qposadr(model, "robot_0/base_y")
    base_theta_adr = manual.find_joint_qposadr(model, "robot_0/base_theta")
    if base_x_adr is None or base_y_adr is None or base_theta_adr is None:
        raise RuntimeError("Robot model does not expose robot_0/base_x, base_y, base_theta.")

    data.qpos[base_x_adr] = float(args.start_pose[0])
    data.qpos[base_y_adr] = float(args.start_pose[1])
    data.qpos[base_theta_adr] = math.radians(float(args.start_pose[2]))
    mujoco.mj_forward(model, data)

    from molmo_spaces.utils import distance_transform_utils as dtutils

    downscale = 5
    px_per_m = 200
    grid_spacing = downscale / px_per_m
    scene_map = make_scene_map(args.scene_xml, agent_radius=float(args.planner_agent_radius), px_per_m=px_per_m)
    downscaled_grid = make_downscaled_grid(scene_map, downscale)
    distance_transform = dtutils.make_distance_transform(downscaled_grid, grid_spacing)
    graph = dtutils.make_grid_graph(downscaled_grid, distance_transform, weight_exp=2)

    start_xy = np.array([float(args.start_pose[0]), float(args.start_pose[1]), 0.0], dtype=np.float32)
    goal_xy = np.array([float(args.goal_xy[0]), float(args.goal_xy[1]), 0.0], dtype=np.float32)
    discrete_start = resolve_plannable_location(graph, scene_map, start_xy, downscale)
    discrete_goal = resolve_plannable_location(graph, scene_map, goal_xy, downscale)
    if discrete_start is None:
        raise RuntimeError(f"Start pose is not plannable: {args.start_pose}")
    if discrete_goal is None:
        raise RuntimeError(f"Goal is not plannable: {args.goal_xy}")

    try:
        discrete_waypoints, _, _ = dtutils.make_discrete_path(
            graph,
            discrete_start[0],
            discrete_start[1],
            discrete_goal[0],
            discrete_goal[1],
            distance_transform,
            3,
            grid_spacing,
            0.6,
        )
        pixel_waypoints = np.array(discrete_waypoints) * downscale
        waypoints = scene_map.pos_px_to_m(pixel_waypoints)[:, :2]
    except nx.NetworkXUnfeasible as exc:
        raise RuntimeError(
            f"A* failed to produce a path from {args.start_pose[:2]} to {args.goal_xy}"
        ) from exc

    if len(waypoints) == 0:
        raise RuntimeError(f"A* failed to produce a path from {args.start_pose[:2]} to {args.goal_xy}")

    waypoints = np.asarray(waypoints, dtype=np.float32)
    print(
        "[INFO] Planned path: "
        f"start={tuple(np.round(start_xy[:2], 3))} goal={tuple(np.round(goal_xy[:2], 3))} "
        f"waypoints={len(waypoints)}"
    )

    robot_cam_id = resolve_camera_id_optional(model, args.robot_camera_name)
    follower_cam_id = resolve_camera_id_optional(model, args.follower_camera_name)
    if args.viewer_camera == "robot" and robot_cam_id is None:
        raise RuntimeError(f"Robot camera not found: {args.robot_camera_name}")
    if args.viewer_camera == "follower" and follower_cam_id is None:
        print(f"[WARN] Follower camera not found: {args.follower_camera_name}; falling back to robot camera.")
        if robot_cam_id is None:
            raise RuntimeError(f"Robot camera not found: {args.robot_camera_name}")

    import mujoco.viewer as mj_viewer

    current_waypoint_idx = 0
    step_count = 0
    reached_goal = False
    quit_requested = False

    def key_callback(keycode: int) -> None:
        nonlocal quit_requested
        if keycode in (ord("q"), ord("Q")):
            quit_requested = True

    if args.no_viewer:
        viewer_context = None
    else:
        viewer_context = mj_viewer.launch_passive(
            model,
            data,
            key_callback=key_callback,
            show_left_ui=bool(args.show_ui),
            show_right_ui=bool(args.show_ui),
        )

    class _NullViewer:
        def is_running(self) -> bool:
            return True

        def sync(self) -> None:
            return None

        class _NullLock:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        def lock(self):
            return self._NullLock()

        cam = type("Cam", (), {"type": None, "fixedcamid": -1})()

    context = viewer_context if viewer_context is not None else _NullViewer()

    manager = context if viewer_context is not None else None
    if manager is None:
        viewers = [context]
    else:
        viewers = [manager]

    for viewer in viewers:
        if viewer_context is not None:
            enter = viewer.__enter__
            exit_ = viewer.__exit__
            viewer = enter()
        else:
            exit_ = None

        while viewer.is_running() and not quit_requested and step_count < int(args.max_steps):
            with viewer.lock():
                bx = float(data.qpos[base_x_adr])
                by = float(data.qpos[base_y_adr])
                th = float(data.qpos[base_theta_adr])
                cur_xy = np.array([bx, by], dtype=np.float32)

                if np.linalg.norm(goal_xy[:2] - cur_xy) <= float(args.goal_radius):
                    reached_goal = True
                    mujoco.mj_forward(model, data)
                    if args.viewer_camera == "robot" and robot_cam_id is not None:
                        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                        viewer.cam.fixedcamid = int(robot_cam_id)
                    else:
                        active_cam_id = follower_cam_id if follower_cam_id is not None else robot_cam_id
                        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                        viewer.cam.fixedcamid = int(active_cam_id)
                    if args.no_viewer:
                        viewer.sync()
                        break
                    viewer.sync()
                    time.sleep(0.01)
                    continue

                while current_waypoint_idx < len(waypoints) - 1:
                    waypoint = waypoints[current_waypoint_idx]
                    if np.linalg.norm(waypoint - cur_xy) > float(args.waypoint_radius):
                        break
                    current_waypoint_idx += 1

                waypoint = waypoints[current_waypoint_idx]
                delta = waypoint - cur_xy
                desired_yaw = math.atan2(float(delta[1]), float(delta[0]))
                yaw_err = wrap_to_pi(desired_yaw - th)

                if abs(yaw_err) > math.radians(float(args.turn_in_place_threshold_deg)):
                    th += math.copysign(
                        min(abs(yaw_err), math.radians(float(args.yaw_step_deg))),
                        yaw_err,
                    )
                else:
                    step_size = min(float(args.move_step), float(np.linalg.norm(delta)))
                    bx += step_size * math.cos(th)
                    by += step_size * math.sin(th)

                data.qpos[base_x_adr] = bx
                data.qpos[base_y_adr] = by
                data.qpos[base_theta_adr] = th
                mujoco.mj_forward(model, data)

                active_cam_id = robot_cam_id
                if args.viewer_camera == "follower" and follower_cam_id is not None:
                    active_cam_id = follower_cam_id
                if active_cam_id is not None:
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    viewer.cam.fixedcamid = int(active_cam_id)

            viewer.sync()
            step_count += 1
            time.sleep(0.01)
        if exit_ is not None:
            exit_(None, None, None)

    final_xy = np.array(
        [float(data.qpos[base_x_adr]), float(data.qpos[base_y_adr])],
        dtype=np.float32,
    )
    final_dist = float(np.linalg.norm(goal_xy[:2] - final_xy))
    print(
        "[RESULT] "
        f"reached_goal={reached_goal} steps={step_count} final_xy={tuple(np.round(final_xy, 3))} "
        f"goal_xy={tuple(np.round(goal_xy[:2], 3))} final_dist={final_dist:.3f}"
    )


if __name__ == "__main__":
    main()
