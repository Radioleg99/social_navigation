#!/usr/bin/env python3
"""Manual robot-first-person RGB/Depth/Pose capture for ConceptGraph.

By default, captures are taken from the robot camera that already exists
in the composed scene (e.g. ``robot_0/head_camera``).

Hotkeys:
  Up/Down   : move forward/backward
  Left/Right: yaw left/right
  P         : capture one frame (color/depth/pose)
  H         : print help
  Q         : quit
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from collections import deque
from pathlib import Path
from threading import Lock

import imageio.v2 as imageio
import mujoco
import numpy as np

# Allow direct script execution without requiring PYTHONPATH=. from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from molmo_spaces.renderer.opengl_rendering import MjOpenGLRenderer
from scripts.manual_capture_paths import DEFAULT_CAPTURE_OUTPUT_ROOT

KEY_LEFT = 263
KEY_RIGHT = 262
KEY_UP = 265
KEY_DOWN = 264


def normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return v.copy()
    return v / n


def build_camera_basis(yaw_deg: float) -> tuple[np.ndarray, np.ndarray]:
    yaw = math.radians(yaw_deg)
    forward = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float32)
    forward = normalize(forward)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = normalize(np.cross(forward, world_up))
    up = normalize(np.cross(right, forward))
    return forward, up


def cam2world_gl_from_basis(pos: np.ndarray, forward: np.ndarray, up: np.ndarray) -> np.ndarray:
    right = normalize(np.cross(forward, up))
    up_orth = normalize(np.cross(right, forward))
    T = np.eye(4, dtype=np.float32)
    T[:3, 0] = right
    # OpenGL camera basis in world: +x right, +y up, +z backward.
    T[:3, 1] = up_orth
    T[:3, 2] = -forward
    T[:3, 3] = pos
    return T


def cam_pose_gl_from_mj_camera(data: mujoco.MjData, cam_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pos = np.asarray(data.cam_xpos[cam_id], dtype=np.float32).copy()
    R = np.asarray(data.cam_xmat[cam_id], dtype=np.float32).reshape(3, 3)
    right = normalize(R[:, 0])
    up = normalize(R[:, 1])
    backward = normalize(R[:, 2])
    forward = -backward

    T = np.eye(4, dtype=np.float32)
    T[:3, 0] = right
    T[:3, 1] = up
    T[:3, 2] = backward
    T[:3, 3] = pos
    return pos, forward, up, T


def planar_forward_from_robot_camera(
    data: mujoco.MjData,
    cam_id: int,
    fallback_yaw_rad: float,
) -> np.ndarray:
    _, forward, _, _ = cam_pose_gl_from_mj_camera(data, cam_id)
    planar = np.array([float(forward[0]), float(forward[1])], dtype=np.float32)
    norm = float(np.linalg.norm(planar))
    if norm < 1e-8:
        return np.array(
            [math.cos(fallback_yaw_rad), math.sin(fallback_yaw_rad)],
            dtype=np.float32,
        )
    return planar / norm


def gl_to_cv_pose(T_gl: np.ndarray) -> np.ndarray:
    T_corr = np.eye(4, dtype=np.float32)
    T_corr[1, 1] = -1.0
    T_corr[2, 2] = -1.0
    return T_gl @ T_corr


def compute_intrinsics(vfov_deg: float, height: int, width: int) -> np.ndarray:
    f = float(height) / (2.0 * math.tan(math.radians(vfov_deg) / 2.0))
    return np.array(
        [[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def infer_default_layout_json(scene_xml: Path) -> Path | None:
    name = scene_xml.stem
    if name.endswith("_physics"):
        name = name[: -len("_physics")]
    candidate = REPO_ROOT / "assets" / "layouts" / f"{name}_object_positions.json"
    if candidate.is_file():
        return candidate
    return None


def load_layout_payload_safe(layout_json: Path | None) -> dict[str, object] | None:
    if layout_json is None or not layout_json.is_file():
        return None
    try:
        payload = json.loads(layout_json.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def infer_primary_human_state_count(layout_payload: dict[str, object] | None) -> int:
    if not isinstance(layout_payload, dict):
        return 0
    runtime = layout_payload.get("human_runtime")
    if not isinstance(runtime, dict):
        return 0
    if runtime.get("enabled", True) is False:
        return 0

    poses = runtime.get("poses")
    if isinstance(poses, list):
        return len(poses)

    state_xmls = runtime.get("human_state_xmls")
    if not isinstance(state_xmls, list):
        state_xmls = runtime.get("state_xmls")
    if isinstance(state_xmls, list):
        return len(state_xmls)
    return 0


def enforce_single_primary_human_state_visibility(
    model: mujoco.MjModel,
    layout_payload: dict[str, object] | None,
    requested_state_idx: int,
) -> tuple[int, int] | None:
    num_states = infer_primary_human_state_count(layout_payload)
    if num_states <= 1:
        return None

    if requested_state_idx < 0:
        raise ValueError("--primary-human-state-index must be >= 0")
    active_idx = min(requested_state_idx, num_states - 1)

    for i in range(num_states):
        prefix = f"human_state_{i}/"
        for gid in range(model.ngeom):
            gname = model.geom(gid).name
            if not gname.startswith(prefix):
                continue
            if i != active_idx:
                model.geom_rgba[gid, 3] = 0.0
    return active_idx, num_states


def hide_robot_geometries(model: mujoco.MjModel) -> int:
    hidden = 0
    for gid in range(model.ngeom):
        gname = model.geom(gid).name
        if gname.startswith("robot_0/") or gname.startswith("robot_"):
            model.geom_rgba[gid, 3] = 0.0
            hidden += 1
    return hidden


def load_basic_scene_module():
    module_path = Path(__file__).with_name("basic_robot_human_scene.py")
    spec = importlib.util.spec_from_file_location("basic_robot_human_scene", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_default_build_args_for_layout(scene_xml: Path, layout_json: Path | None) -> argparse.Namespace:
    return argparse.Namespace(
        scene_source="ithor",
        scene_split="train",
        scene_index=1,
        scene_xml=scene_xml,
        layout_json=layout_json,
        robot_type="franka",
        robot_pos=[0.0, -0.15, 0.0],
        robot_base=[1.5, 0.0, 0.0],
        human_pos=[1.5, 0.0, 0.0],
        human_yaw_deg=180.0,
        human_roll_deg=90.0,
        human_pitch_deg=0.0,
        human_yaw_offset_deg=0.0,
        human_z_offset=0.0,
        human_collider_type="none",
        human_collider_size=[0.28, 1.65, 0.0],
        human_static=False,
        human_uid=None,
        human_xml=None,
        human_state_xmls=None,
        human_state_times=None,
        human_state_loop=False,
        human_state_hold_sec=2.0,
        human_trajectory_file=None,
        trajectory_loop=False,
        save_xml=None,
        no_viewer=True,
        focus_human=False,
        camera_distance=2.2,
        camera_azimuth=150.0,
        camera_elevation=-15.0,
        tune_human=False,
        tune_step_xy=0.05,
        tune_step_z=0.02,
        tune_step_yaw_deg=5.0,
        extra_humans=[],
    )


def load_scene_raw(scene_xml: Path) -> tuple[mujoco.MjModel, mujoco.MjData, str]:
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, "raw_scene_xml"


def load_scene_with_optional_layout_runtime(
    scene_xml: Path,
    layout_json: Path | None,
    layout_runtime: str,
    robot_type_override: str | None = None,
) -> tuple[mujoco.MjModel, mujoco.MjData, str]:
    if layout_runtime == "off":
        return load_scene_raw(scene_xml)

    has_layout = layout_json is not None and layout_json.is_file()
    if layout_runtime == "on" and not has_layout:
        raise FileNotFoundError("--layout-runtime on requires a valid --layout-json file.")

    should_try_layout = (layout_runtime == "on") or (layout_runtime == "auto" and has_layout)
    if not should_try_layout:
        return load_scene_raw(scene_xml)

    try:
        bhs = load_basic_scene_module()
        build_args = make_default_build_args_for_layout(scene_xml, layout_json)
        if robot_type_override is not None:
            build_args.robot_type = robot_type_override
        payload = bhs.load_layout_json(layout_json)
        bhs.apply_robot_runtime_from_layout(build_args, payload)
        bhs.apply_human_runtime_from_layout(build_args, payload)
        model, data, _, _ = bhs.build_scene(build_args)
        mujoco.mj_forward(model, data)
        return model, data, "layout_runtime"
    except Exception as e:
        if layout_runtime == "on":
            raise RuntimeError(
                "Failed to build scene with layout runtime while --layout-runtime on was requested."
            ) from e
        print(f"[WARN] Layout runtime composition failed ({e}); fallback to raw scene xml.")
        return load_scene_raw(scene_xml)


def find_joint_qposadr(model: mujoco.MjModel, joint_name: str) -> int | None:
    try:
        joint = model.joint(joint_name)
    except Exception:
        return None
    return int(joint.qposadr[0])


def resolve_camera_id(model: mujoco.MjModel, camera_name: str) -> int:
    for cid in range(model.ncam):
        if model.cam(cid).name == camera_name:
            return cid
    available = [model.cam(i).name for i in range(model.ncam)]
    raise ValueError(f"Camera '{camera_name}' not found. Available cameras: {available}")


def render_frame(
    renderer: MjOpenGLRenderer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    fov_deg: float,
    depth: bool,
    *,
    fixed_cam_id: int | None = None,
    pos: np.ndarray | None = None,
    forward: np.ndarray | None = None,
    up: np.ndarray | None = None,
) -> np.ndarray:
    prev_fovy = float(model.vis.global_.fovy)
    model.vis.global_.fovy = float(fov_deg)

    cam = mujoco.MjvCamera()
    if fixed_cam_id is not None:
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = int(fixed_cam_id)
    else:
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    renderer.update(data, cam)

    if fixed_cam_id is None:
        if pos is None or forward is None or up is None:
            raise ValueError("Free-camera rendering requires pos/forward/up.")
        for gl_cam in renderer.scene.camera:
            gl_cam.pos = pos.astype(np.float32)
            gl_cam.forward = forward.astype(np.float32)
            gl_cam.up = up.astype(np.float32)

    if depth:
        renderer.enable_depth_rendering()
        frame = renderer.render()
        renderer.disable_depth_rendering()
    else:
        frame = renderer.render()

    model.vis.global_.fovy = prev_fovy
    return frame


def write_dataconfig(path: Path, height: int, width: int, K: np.ndarray, depth_scale: float) -> None:
    text = (
        "dataset_name: 'ai2thor'\n"
        "camera_params:\n"
        f"  image_height: {height}\n"
        f"  image_width: {width}\n"
        f"  fx: {float(K[0, 0]):.6f}\n"
        f"  fy: {float(K[1, 1]):.6f}\n"
        f"  cx: {float(K[0, 2]):.6f}\n"
        f"  cy: {float(K[1, 2]):.6f}\n"
        f"  png_depth_scale: {float(depth_scale):.1f}\n"
    )
    path.write_text(text, encoding="utf-8")


def parse_start_pose(text: str) -> tuple[float, float, float]:
    vals = [float(x.strip()) for x in text.split(",")]
    if len(vals) != 3:
        raise argparse.ArgumentTypeError("Expected --start-pose as x,y,yaw_deg")
    return vals[0], vals[1], vals[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene-xml", type=Path, required=True, help="Path to *_physics.xml scene file")
    p.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_CAPTURE_OUTPUT_ROOT,
        help="Dataset root",
    )
    p.add_argument("--scene-id", type=str, default="ManualCapture/frames", help="Relative scene path")
    p.add_argument(
        "--layout-json",
        type=Path,
        default=None,
        help="Optional layout json. If omitted, auto-detect from scene name under assets/layouts.",
    )
    p.add_argument(
        "--layout-runtime",
        choices=["auto", "on", "off"],
        default="auto",
        help="Compose scene with layout runtime (robot/humans). auto=use layout when available.",
    )
    p.add_argument(
        "--primary-human-state-index",
        type=int,
        default=0,
        help="For multi-state primary human, keep only this mesh(state) visible.",
    )
    p.add_argument(
        "--hide-robot",
        dest="hide_robot",
        action="store_true",
        default=False,
        help="Hide robot geometries in rendered images.",
    )
    p.add_argument(
        "--no-hide-robot",
        dest="hide_robot",
        action="store_false",
        help="Do not hide robot geometries (default).",
    )
    p.add_argument(
        "--camera-source",
        choices=["robot", "free"],
        default="robot",
        help="Camera source. robot=use existing robot camera in scene; free=manual standalone camera.",
    )
    p.add_argument(
        "--robot-camera-name",
        type=str,
        default="robot_0/head_camera",
        help="Camera name when --camera-source robot.",
    )
    p.add_argument(
        "--robot-type",
        choices=["franka", "rby1", "rby1m", "navbot"],
        default=None,
        help=(
            "Optional robot type override when composing with --layout-runtime on/auto. "
            "Useful for swapping FloorPlan layouts from rby1 to the smaller navbot."
        ),
    )

    p.add_argument("--image-width", type=int, default=640)
    p.add_argument("--image-height", type=int, default=480)
    p.add_argument("--vfov-deg", type=float, default=90.0)

    p.add_argument("--start-pose", type=parse_start_pose, default=(0.0, 0.0, 0.0))
    p.add_argument("--camera-height", type=float, default=1.45)
    p.add_argument("--move-step", type=float, default=0.15, help="Meters per key press")
    p.add_argument("--yaw-step-deg", type=float, default=8.0, help="Degrees per key press")
    p.add_argument(
        "--show-ui",
        action="store_true",
        default=False,
        help="Show MuJoCo left/right UI panels (default off to reduce key conflicts).",
    )
    p.add_argument(
        "--simulate",
        action="store_true",
        default=False,
        help="Run continuous physics stepping while viewer is open.",
    )
    p.add_argument("--max-depth", type=float, default=15.0)
    p.add_argument("--min-depth", type=float, default=0.05)
    p.add_argument("--depth-scale", type=float, default=1000.0)
    p.add_argument("--pose-convention", choices=["cv", "gl"], default="cv")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.scene_xml.is_file():
        raise FileNotFoundError(f"scene xml not found: {args.scene_xml}")

    scene_root = args.output_root / args.scene_id
    color_dir = scene_root / "color"
    depth_dir = scene_root / "depth"
    pose_dir = scene_root / "pose"

    if scene_root.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {scene_root}. Use --overwrite to overwrite files in-place.")

    color_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    pose_dir.mkdir(parents=True, exist_ok=True)

    layout_json = args.layout_json
    if layout_json is None:
        layout_json = infer_default_layout_json(args.scene_xml)
        if layout_json is not None:
            print(f"[INFO] Auto-detected layout json: {layout_json}")

    model, data, model_source = load_scene_with_optional_layout_runtime(
        args.scene_xml, layout_json, args.layout_runtime, robot_type_override=args.robot_type
    )
    layout_payload = load_layout_payload_safe(layout_json)
    chosen_state = enforce_single_primary_human_state_visibility(
        model,
        layout_payload,
        args.primary_human_state_index,
    )
    if chosen_state is not None:
        active_idx, total_states = chosen_state
        print(
            "[INFO] Multi-state primary human detected: "
            f"states={total_states}, using state_index={active_idx}"
        )
    hidden_robot_geoms = 0
    if args.hide_robot:
        hidden_robot_geoms = hide_robot_geometries(model)
        if hidden_robot_geoms > 0:
            print(f"[INFO] Robot hidden in render: geoms={hidden_robot_geoms}")
    print(f"[INFO] model_source={model_source}")

    base_x_adr = find_joint_qposadr(model, "robot_0/base_x")
    base_y_adr = find_joint_qposadr(model, "robot_0/base_y")
    base_theta_adr = find_joint_qposadr(model, "robot_0/base_theta")

    robot_cam_id = None
    if args.camera_source == "robot":
        robot_cam_id = resolve_camera_id(model, args.robot_camera_name)
        if base_x_adr is not None and base_y_adr is not None and base_theta_adr is not None:
            data.qpos[base_x_adr] = float(args.start_pose[0])
            data.qpos[base_y_adr] = float(args.start_pose[1])
            data.qpos[base_theta_adr] = math.radians(float(args.start_pose[2]))
            mujoco.mj_forward(model, data)
        print(f"[INFO] Using robot camera: {args.robot_camera_name} (id={robot_cam_id})")
    else:
        print("[INFO] Using free camera mode.")

    try:
        renderer = MjOpenGLRenderer(model=model, width=int(args.image_width), height=int(args.image_height))
    except Exception as e:
        lower = str(e).lower()
        if "coregraphics" in lower or "cgl" in lower:
            raise RuntimeError(
                "Failed to create OpenGL context on macOS. Run with mjpython: "
                "./.venv/bin/mjpython scripts/manual_capture_rgbd.py ..."
            ) from e
        raise

    K = compute_intrinsics(args.vfov_deg, args.image_height, args.image_width)
    np.savetxt(scene_root / "intrinsics.txt", K)
    write_dataconfig(scene_root / "dataconfig.yaml", args.image_height, args.image_width, K, args.depth_scale)

    x, y, yaw_deg = float(args.start_pose[0]), float(args.start_pose[1]), float(args.start_pose[2])
    frame_count = 0
    quit_flag = False
    captures: list[dict[str, float]] = []
    pending_actions: deque[str] = deque()
    pending_actions_lock = Lock()

    def enqueue_action(action: str) -> None:
        with pending_actions_lock:
            pending_actions.append(action)

    def drain_actions() -> list[str]:
        with pending_actions_lock:
            actions = list(pending_actions)
            pending_actions.clear()
        return actions

    def print_help() -> None:
        print("[MANUAL CAPTURE] Hotkeys:")
        print("  Up/Down             : move forward/backward")
        print("  Left/Right          : yaw left/right")
        print("  P         : capture current frame")
        print("  H         : print help")
        print("  Q         : quit")
        print("  Tip: click viewer window first.")

    def print_pose() -> None:
        if (
            args.camera_source == "robot"
            and base_x_adr is not None
            and base_y_adr is not None
            and base_theta_adr is not None
        ):
            bx = float(data.qpos[base_x_adr])
            by = float(data.qpos[base_y_adr])
            byaw = math.degrees(float(data.qpos[base_theta_adr]))
            print(f"[POSE] robot_base x={bx:.3f} y={by:.3f} yaw_deg={byaw:.2f}")
        else:
            print(f"[POSE] x={x:.3f} y={y:.3f} yaw_deg={yaw_deg:.2f} z={args.camera_height:.3f}")

    def capture_one() -> None:
        nonlocal frame_count
        if args.camera_source == "robot":
            assert robot_cam_id is not None
            pos, forward, up, T_gl = cam_pose_gl_from_mj_camera(data, robot_cam_id)
            rgb = render_frame(
                renderer,
                model,
                data,
                float(args.vfov_deg),
                depth=False,
                fixed_cam_id=robot_cam_id,
            )
            depth_m = render_frame(
                renderer,
                model,
                data,
                float(args.vfov_deg),
                depth=True,
                fixed_cam_id=robot_cam_id,
            )
        else:
            pos = np.array([x, y, float(args.camera_height)], dtype=np.float32)
            forward, up = build_camera_basis(yaw_deg)
            T_gl = cam2world_gl_from_basis(pos, forward, up)
            rgb = render_frame(
                renderer,
                model,
                data,
                float(args.vfov_deg),
                depth=False,
                pos=pos,
                forward=forward,
                up=up,
            )
            depth_m = render_frame(
                renderer,
                model,
                data,
                float(args.vfov_deg),
                depth=True,
                pos=pos,
                forward=forward,
                up=up,
            )

        depth_m = np.asarray(depth_m, dtype=np.float32)
        depth_m[~np.isfinite(depth_m)] = 0.0
        depth_m[(depth_m < args.min_depth) | (depth_m > args.max_depth)] = 0.0
        depth_png = np.round(depth_m * float(args.depth_scale)).astype(np.uint16)

        T_save = gl_to_cv_pose(T_gl) if args.pose_convention == "cv" else T_gl

        stem = f"{frame_count:06d}"
        imageio.imwrite(color_dir / f"{stem}.png", np.asarray(rgb, dtype=np.uint8))
        imageio.imwrite(depth_dir / f"{stem}.png", depth_png)
        np.savetxt(pose_dir / f"{stem}.txt", T_save)
        if (
            args.camera_source == "robot"
            and base_x_adr is not None
            and base_y_adr is not None
            and base_theta_adr is not None
        ):
            bx = float(data.qpos[base_x_adr])
            by = float(data.qpos[base_y_adr])
            byaw = math.degrees(float(data.qpos[base_theta_adr]))
            captures.append(
                {
                    "idx": frame_count,
                    "robot_base_x": bx,
                    "robot_base_y": by,
                    "robot_base_yaw_deg": byaw,
                    "camera_name": args.robot_camera_name,
                }
            )
            print(
                f"[CAPTURE] saved frame {stem} at robot_base x={bx:.3f}, y={by:.3f}, yaw={byaw:.2f}"
            )
        else:
            captures.append(
                {
                    "idx": frame_count,
                    "x": x,
                    "y": y,
                    "yaw_deg": yaw_deg,
                }
            )
            print(f"[CAPTURE] saved frame {stem} at x={x:.3f}, y={y:.3f}, yaw={yaw_deg:.2f}")
        frame_count += 1

    def key_callback(keycode: int) -> None:
        nonlocal quit_flag
        if keycode in (ord("p"), ord("P")):
            enqueue_action("capture")
            return
        if keycode in (ord("h"), ord("H")):
            enqueue_action("help")
            return
        if keycode in (ord("q"), ord("Q")):
            quit_flag = True
            enqueue_action("quit")
            return

        if keycode == KEY_UP:
            enqueue_action("forward")
        elif keycode == KEY_DOWN:
            enqueue_action("backward")
        elif keycode == KEY_LEFT:
            enqueue_action("turn_left")
        elif keycode == KEY_RIGHT:
            enqueue_action("turn_right")

    import mujoco.viewer as mj_viewer

    print_help()
    print_pose()
    print(f"[INFO] output scene folder: {scene_root}")
    print(f"[INFO] viewer_ui={'on' if args.show_ui else 'off'}")
    print(f"[INFO] simulate={'on' if args.simulate else 'off'}")
    print("[INFO] Close viewer window or press Q when done.")

    with mj_viewer.launch_passive(
        model,
        data,
        key_callback=key_callback,
        show_left_ui=bool(args.show_ui),
        show_right_ui=bool(args.show_ui),
    ) as viewer:
        wall_start = time.perf_counter()
        next_sync = wall_start
        while viewer.is_running() and not quit_flag:
            actions = drain_actions()
            num_capture_requests = 0
            should_print_help = False

            with viewer.lock():
                if (
                    args.camera_source == "robot"
                    and base_x_adr is not None
                    and base_y_adr is not None
                    and base_theta_adr is not None
                ):
                    bx = float(data.qpos[base_x_adr])
                    by = float(data.qpos[base_y_adr])
                    th = float(data.qpos[base_theta_adr])
                    moved_robot = False
                    for action in actions:
                        if action == "forward":
                            if robot_cam_id is not None:
                                planar_fwd = planar_forward_from_robot_camera(data, robot_cam_id, th)
                                bx += float(args.move_step) * float(planar_fwd[0])
                                by += float(args.move_step) * float(planar_fwd[1])
                            else:
                                bx += float(args.move_step) * math.cos(th)
                                by += float(args.move_step) * math.sin(th)
                            moved_robot = True
                        elif action == "backward":
                            if robot_cam_id is not None:
                                planar_fwd = planar_forward_from_robot_camera(data, robot_cam_id, th)
                                bx -= float(args.move_step) * float(planar_fwd[0])
                                by -= float(args.move_step) * float(planar_fwd[1])
                            else:
                                bx -= float(args.move_step) * math.cos(th)
                                by -= float(args.move_step) * math.sin(th)
                            moved_robot = True
                        elif action == "turn_left":
                            th += math.radians(float(args.yaw_step_deg))
                            moved_robot = True
                        elif action == "turn_right":
                            th -= math.radians(float(args.yaw_step_deg))
                            moved_robot = True
                        elif action == "capture":
                            num_capture_requests += 1
                        elif action == "help":
                            should_print_help = True
                        elif action == "quit":
                            quit_flag = True
                    if moved_robot:
                        data.qpos[base_x_adr] = bx
                        data.qpos[base_y_adr] = by
                        data.qpos[base_theta_adr] = th
                else:
                    for action in actions:
                        if action == "forward":
                            th = math.radians(yaw_deg)
                            x += float(args.move_step) * math.cos(th)
                            y += float(args.move_step) * math.sin(th)
                        elif action == "backward":
                            th = math.radians(yaw_deg)
                            x -= float(args.move_step) * math.cos(th)
                            y -= float(args.move_step) * math.sin(th)
                        elif action == "turn_left":
                            yaw_deg += float(args.yaw_step_deg)
                        elif action == "turn_right":
                            yaw_deg -= float(args.yaw_step_deg)
                        elif action == "capture":
                            num_capture_requests += 1
                        elif action == "help":
                            should_print_help = True
                        elif action == "quit":
                            quit_flag = True

                if args.simulate:
                    mujoco.mj_step(model, data)
                else:
                    mujoco.mj_forward(model, data)

                if args.camera_source == "robot" and robot_cam_id is not None:
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    viewer.cam.fixedcamid = int(robot_cam_id)
                else:
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                    viewer.cam.fixedcamid = -1
                    viewer.cam.lookat[:] = [x, y, args.camera_height]
                    viewer.cam.distance = 0.01
                    viewer.cam.azimuth = float(yaw_deg) - 90.0
                    viewer.cam.elevation = 0.0

                for _ in range(num_capture_requests):
                    capture_one()

            if should_print_help:
                print_help()

            viewer.sync()
            if args.simulate:
                next_sync += model.opt.timestep
                sleep_time = next_sync - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)
            else:
                time.sleep(0.01)

    renderer.close()

    meta = {
        "scene_xml": str(args.scene_xml),
        "layout_json": str(layout_json) if layout_json is not None else None,
        "layout_runtime": args.layout_runtime,
        "model_source": model_source,
        "hide_robot": bool(args.hide_robot),
        "hidden_robot_geoms": int(hidden_robot_geoms),
        "primary_human_state_index": int(args.primary_human_state_index),
        "effective_primary_human_state": int(chosen_state[0]) if chosen_state is not None else None,
        "camera_source": args.camera_source,
        "robot_camera_name": args.robot_camera_name if args.camera_source == "robot" else None,
        "scene_id": args.scene_id,
        "output_root": str(args.output_root),
        "image_width": int(args.image_width),
        "image_height": int(args.image_height),
        "vfov_deg": float(args.vfov_deg),
        "camera_height": float(args.camera_height),
        "pitch_deg": 0.0,
        "depth_scale": float(args.depth_scale),
        "pose_convention": args.pose_convention,
        "move_step": float(args.move_step),
        "yaw_step_deg": float(args.yaw_step_deg),
        "show_ui": bool(args.show_ui),
        "simulate": bool(args.simulate),
        "capture_count": int(frame_count),
        "captures": captures,
    }
    (scene_root / "capture_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n[OK] Manual capture complete.")
    print(f"[OK] Captured frames: {frame_count}")
    print(f"[OK] Scene folder: {scene_root}")
    print(f"[OK] Data config : {scene_root / 'dataconfig.yaml'}")


if __name__ == "__main__":
    main()
