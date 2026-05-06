#!/usr/bin/env python3
"""Export reusable RGB/Depth/Pose sequences from MuJoCo scenes for ConceptGraph.

Outputs folder layout compatible with conceptgraph Ai2thorDataset:
  <output_root>/<scene_id>/
    color/000000.png
    depth/000000.png   # uint16 depth (mm by default)
    pose/000000.txt    # 4x4 cam2world matrix
    intrinsics.txt
    dataconfig.yaml
    capture_meta.json

Scene source modes:
- raw scene xml only
- scene composed with layout runtime (robot/humans) via basic_robot_human_scene.py

Usage example:
  ./.venv/bin/mjpython scripts/export_conceptgraph_rgbd.py \
    --scene-xml assets/scenes/ithor/FloorPlan203_physics.xml \
    --layout-json assets/layouts/FloorPlan203_object_positions.json \
    --layout-runtime auto \
    --trajectory-mode orbit \
    --orbit-radius 1.4 \
    --output-root /Users/ljj/project/graduation/molmospaces-main/data_3dgs \
    --scene-id FloorPlan203/frames \
    --n-views 36
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

# Allow direct script execution without requiring PYTHONPATH=. from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from molmo_spaces.renderer.opengl_rendering import MjOpenGLRenderer
from scripts.output_root_config import DEFAULT_OUTPUT_ROOT


def parse_vec2(text: str) -> tuple[float, float]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected format x,y")
    return float(parts[0]), float(parts[1])


def normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return v.copy()
    return v / n


def infer_center_from_layout(layout_json: Path | None, model: mujoco.MjModel) -> tuple[np.ndarray, str]:
    if layout_json is not None and layout_json.is_file():
        try:
            payload = json.loads(layout_json.read_text(encoding="utf-8"))
        except Exception:
            payload = {}

        if isinstance(payload, dict):
            rr = payload.get("robot_runtime", {})
            if isinstance(rr, dict):
                rb = rr.get("robot_base")
                if isinstance(rb, list) and len(rb) >= 2:
                    return (
                        np.array([float(rb[0]), float(rb[1])], dtype=np.float32),
                        "robot_runtime.robot_base",
                    )

            hr = payload.get("human_runtime", {})
            if isinstance(hr, dict):
                hp = hr.get("human_pos")
                if isinstance(hp, list) and len(hp) >= 2:
                    return (
                        np.array([float(hp[0]), float(hp[1])], dtype=np.float32),
                        "human_runtime.human_pos",
                    )

            objs = payload.get("objects")
            if isinstance(objs, list):
                pts: list[list[float]] = []
                for obj in objs:
                    if not isinstance(obj, dict):
                        continue
                    p = obj.get("position_xyz")
                    if not (isinstance(p, list) and len(p) >= 2):
                        continue
                    x, y = float(p[0]), float(p[1])
                    # Skip trivial origin placeholders used by many static meshes.
                    if abs(x) < 1e-6 and abs(y) < 1e-6:
                        continue
                    pts.append([x, y])
                if pts:
                    arr = np.asarray(pts, dtype=np.float32)
                    return np.median(arr, axis=0), "layout.objects median"

    # Fallback: estimate center from model geom world positions after forward().
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    if data.geom_xpos.shape[0] > 0:
        xy = data.geom_xpos[:, :2]
        return np.median(xy, axis=0).astype(np.float32), "model.geom_xpos median"

    return np.array([0.0, 0.0], dtype=np.float32), "default(0,0)"


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

    # Primary multi-state human is attached as parallel meshes:
    #   human_state_0/, human_state_1/, ...
    # During dataset export we force a single visible state to avoid overlapped humans.
    state_geom_ids: list[list[int]] = []
    for i in range(num_states):
        prefix = f"human_state_{i}/"
        ids: list[int] = []
        for gid in range(model.ngeom):
            gname = model.geom(gid).name
            if gname.startswith(prefix):
                ids.append(gid)
        state_geom_ids.append(ids)

    for i, ids in enumerate(state_geom_ids):
        for gid in ids:
            if i != active_idx:
                model.geom_rgba[gid, 3] = 0.0

    return active_idx, num_states


def build_camera_basis_from_yaw_pitch(yaw_deg: float, pitch_deg: float) -> tuple[np.ndarray, np.ndarray]:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)

    forward = np.array(
        [
            math.cos(yaw) * math.cos(pitch),
            math.sin(yaw) * math.cos(pitch),
            math.sin(pitch),
        ],
        dtype=np.float32,
    )
    forward = normalize(forward)

    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    right = normalize(right)
    up = normalize(np.cross(right, forward))

    return forward, up


def compute_camera_pose_for_view(
    view_idx: int,
    n_views: int,
    center_xy: np.ndarray,
    camera_height: float,
    start_yaw_deg: float,
    sweep_deg: float,
    pitch_deg: float,
    trajectory_mode: str,
    orbit_radius: float,
    look_at_center: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if n_views <= 1:
        phase_offset = 0.0
    elif abs(float(sweep_deg) - 360.0) < 1e-6:
        # Keep backward-compatible sampling for a full ring:
        # use n_views slots on [0, 360) without duplicating the start angle.
        phase_offset = float(sweep_deg) * view_idx / float(n_views)
    else:
        # For partial sweeps, include both ends.
        phase_offset = float(sweep_deg) * view_idx / float(n_views - 1)
    phase_deg = start_yaw_deg + phase_offset

    if trajectory_mode == "rotate":
        pos = np.array([center_xy[0], center_xy[1], camera_height], dtype=np.float32)
        yaw_deg = phase_deg
    elif trajectory_mode == "orbit":
        theta = math.radians(phase_deg)
        pos = np.array(
            [
                center_xy[0] + orbit_radius * math.cos(theta),
                center_xy[1] + orbit_radius * math.sin(theta),
                camera_height,
            ],
            dtype=np.float32,
        )
        if look_at_center:
            to_center = center_xy - pos[:2]
            yaw_deg = math.degrees(math.atan2(float(to_center[1]), float(to_center[0])))
        else:
            # Keep camera facing along global phase direction.
            yaw_deg = phase_deg
    else:
        raise ValueError(f"Unsupported trajectory_mode: {trajectory_mode}")

    forward, up = build_camera_basis_from_yaw_pitch(yaw_deg, pitch_deg)
    return pos, forward, up, yaw_deg


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


def render_frame(
    renderer: MjOpenGLRenderer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    pos: np.ndarray,
    forward: np.ndarray,
    up: np.ndarray,
    fov_deg: float,
    depth: bool,
) -> np.ndarray:
    prev_fovy = float(model.vis.global_.fovy)
    model.vis.global_.fovy = float(fov_deg)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    renderer.update(data, cam)

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
) -> tuple[mujoco.MjModel, mujoco.MjData, str]:
    if layout_runtime == "off":
        return load_scene_raw(scene_xml)

    has_layout = layout_json is not None and layout_json.is_file()
    if layout_runtime == "on" and not has_layout:
        raise FileNotFoundError(
            "--layout-runtime on requires a valid --layout-json file."
        )

    should_try_layout = (layout_runtime == "on") or (layout_runtime == "auto" and has_layout)
    if not should_try_layout:
        return load_scene_raw(scene_xml)

    try:
        bhs = load_basic_scene_module()
        build_args = make_default_build_args_for_layout(scene_xml, layout_json)
        payload = bhs.load_layout_json(layout_json)
        bhs.apply_robot_runtime_from_layout(build_args, payload)
        bhs.apply_human_runtime_from_layout(build_args, payload)
        model, data, _, _ = bhs.build_scene(build_args)
        mujoco.mj_forward(model, data)
        print("[INFO] Scene composed with layout runtime (robot/humans included).")
        return model, data, "layout_runtime"
    except Exception as e:
        if layout_runtime == "on":
            raise RuntimeError(
                "Failed to build scene with layout runtime while --layout-runtime on was requested."
            ) from e
        print(f"[WARN] Layout runtime composition failed ({e}); fallback to raw scene xml.")
        return load_scene_raw(scene_xml)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene-xml", type=Path, required=True, help="Path to *_physics.xml scene file")
    p.add_argument(
        "--layout-json",
        type=Path,
        default=None,
        help="Optional layout json for center inference and runtime composition",
    )
    p.add_argument(
        "--layout-runtime",
        choices=["auto", "on", "off"],
        default="auto",
        help="Whether to compose scene with layout runtime (robot/humans). auto=use when layout exists.",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Dataset root",
    )
    p.add_argument("--scene-id", type=str, default="FloorPlan203/frames", help="Relative scene path")
    p.add_argument(
        "--primary-human-state-index",
        type=int,
        default=0,
        help=(
            "When layout uses multi-state primary human (poses/human_state_xmls), "
            "select which single mesh(state) stays visible during export. "
            "Default 0 = first state."
        ),
    )

    p.add_argument("--image-width", type=int, default=640)
    p.add_argument("--image-height", type=int, default=480)
    p.add_argument("--vfov-deg", type=float, default=90.0, help="Vertical FOV in degrees")

    p.add_argument("--n-views", type=int, default=24, help="Number of frames to capture")
    p.add_argument("--camera-height", type=float, default=1.45, help="Camera z in world")
    p.add_argument(
        "--center-xy",
        type=parse_vec2,
        default=None,
        help="Manual center override as x,y (default: infer from layout/model)",
    )
    p.add_argument("--start-yaw-deg", type=float, default=0.0)
    p.add_argument(
        "--sweep-deg",
        type=float,
        default=360.0,
        help="Yaw sweep angle across all views. Use <360 for non-360 capture.",
    )
    p.add_argument("--pitch-deg", type=float, default=-12.0, help="Negative looks downward")
    p.add_argument(
        "--trajectory-mode",
        choices=["orbit", "rotate"],
        default="orbit",
        help="Camera trajectory: orbit translates around center; rotate stays in place.",
    )
    p.add_argument(
        "--orbit-radius",
        type=float,
        default=1.2,
        help="Orbit radius in meters (used when --trajectory-mode orbit).",
    )
    p.add_argument(
        "--look-at-center",
        dest="look_at_center",
        action="store_true",
        default=True,
        help="When orbiting, keep camera yaw pointed to scene center (default on).",
    )
    p.add_argument(
        "--no-look-at-center",
        dest="look_at_center",
        action="store_false",
        help="When orbiting, do not force yaw to center.",
    )

    p.add_argument("--max-depth", type=float, default=15.0, help="Depth cutoff in meters")
    p.add_argument("--min-depth", type=float, default=0.05)
    p.add_argument("--depth-scale", type=float, default=1000.0, help="Meters->PNG scale")
    p.add_argument(
        "--pose-convention",
        choices=["cv", "gl"],
        default="cv",
        help="Saved pose convention. Use 'cv' for ConceptGraph pinhole backprojection.",
    )
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
        # Keep safe by default: refuse accidental overwrite.
        raise FileExistsError(
            f"Output already exists: {scene_root}. Use --overwrite to overwrite files in-place."
        )

    color_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    pose_dir.mkdir(parents=True, exist_ok=True)

    model, data, model_source = load_scene_with_optional_layout_runtime(
        args.scene_xml, args.layout_json, args.layout_runtime
    )
    layout_payload = load_layout_payload_safe(args.layout_json)
    chosen_state = enforce_single_primary_human_state_visibility(
        model,
        layout_payload,
        args.primary_human_state_index,
    )
    if chosen_state is not None:
        active_idx, total_states = chosen_state
        print(
            "[INFO] Multi-state primary human detected: "
            f"states={total_states}, exporting only state_index={active_idx}"
        )

    if args.center_xy is not None:
        center_xy = np.array([args.center_xy[0], args.center_xy[1]], dtype=np.float32)
        center_src = "manual --center-xy"
    else:
        center_xy, center_src = infer_center_from_layout(args.layout_json, model)

    print(f"[INFO] model_source={model_source}")
    print(f"[INFO] output_root={args.output_root}")
    print(f"[INFO] center_xy={center_xy.tolist()} (source={center_src})")
    print(
        f"[INFO] trajectory_mode={args.trajectory_mode} "
        f"(orbit_radius={args.orbit_radius:.3f}, look_at_center={args.look_at_center})"
    )

    if args.trajectory_mode == "orbit" and args.orbit_radius <= 0.0:
        raise ValueError("--orbit-radius must be > 0 when --trajectory-mode orbit is used.")

    try:
        renderer = MjOpenGLRenderer(model=model, width=int(args.image_width), height=int(args.image_height))
    except Exception as e:
        lower_msg = str(e).lower()
        if "coregraphics" in lower_msg or "cgl" in lower_msg:
            raise RuntimeError(
                "Failed to create OpenGL context on macOS. "
                "Run this script with mjpython instead of python, e.g.: "
                "./.venv/bin/mjpython scripts/export_conceptgraph_rgbd.py ..."
            ) from e
        raise

    K = compute_intrinsics(args.vfov_deg, args.image_height, args.image_width)
    np.savetxt(scene_root / "intrinsics.txt", K)

    trajectory_positions: list[np.ndarray] = []
    for i in range(args.n_views):
        pos, forward, up, yaw = compute_camera_pose_for_view(
            view_idx=i,
            n_views=args.n_views,
            center_xy=center_xy,
            camera_height=args.camera_height,
            start_yaw_deg=args.start_yaw_deg,
            sweep_deg=args.sweep_deg,
            pitch_deg=args.pitch_deg,
            trajectory_mode=args.trajectory_mode,
            orbit_radius=args.orbit_radius,
            look_at_center=args.look_at_center,
        )
        trajectory_positions.append(pos.copy())

        rgb = render_frame(renderer, model, data, pos, forward, up, args.vfov_deg, depth=False)
        depth_m = render_frame(renderer, model, data, pos, forward, up, args.vfov_deg, depth=True)

        depth_m = np.asarray(depth_m, dtype=np.float32)
        depth_m[~np.isfinite(depth_m)] = 0.0
        depth_m[(depth_m < args.min_depth) | (depth_m > args.max_depth)] = 0.0
        depth_png = np.round(depth_m * args.depth_scale).astype(np.uint16)

        T_gl = cam2world_gl_from_basis(pos, forward, up)
        T_save = gl_to_cv_pose(T_gl) if args.pose_convention == "cv" else T_gl

        stem = f"{i:06d}"
        imageio.imwrite(color_dir / f"{stem}.png", np.asarray(rgb, dtype=np.uint8))
        imageio.imwrite(depth_dir / f"{stem}.png", depth_png)
        np.savetxt(pose_dir / f"{stem}.txt", T_save)

        if (i + 1) % max(1, args.n_views // 6) == 0 or i == args.n_views - 1:
            print(f"[INFO] captured {i + 1}/{args.n_views}")

    renderer.close()

    if trajectory_positions:
        traj = np.asarray(trajectory_positions, dtype=np.float32)
        xy = traj[:, :2]
        xy_std = xy.std(axis=0)
        xy_span = xy.max(axis=0) - xy.min(axis=0)
        print(f"[INFO] trajectory_xy_std={xy_std.tolist()}")
        print(f"[INFO] trajectory_xy_span={xy_span.tolist()}")
    else:
        xy_std = np.zeros((2,), dtype=np.float32)
        xy_span = np.zeros((2,), dtype=np.float32)

    write_dataconfig(scene_root / "dataconfig.yaml", args.image_height, args.image_width, K, args.depth_scale)

    meta = {
        "scene_xml": str(args.scene_xml),
        "layout_json": str(args.layout_json) if args.layout_json is not None else None,
        "scene_id": args.scene_id,
        "layout_runtime": args.layout_runtime,
        "model_source": model_source,
        "center_xy": [float(center_xy[0]), float(center_xy[1])],
        "center_source": center_src,
        "image_width": int(args.image_width),
        "image_height": int(args.image_height),
        "vfov_deg": float(args.vfov_deg),
        "n_views": int(args.n_views),
        "camera_height": float(args.camera_height),
        "start_yaw_deg": float(args.start_yaw_deg),
        "sweep_deg": float(args.sweep_deg),
        "pitch_deg": float(args.pitch_deg),
        "trajectory_mode": args.trajectory_mode,
        "orbit_radius": float(args.orbit_radius),
        "look_at_center": bool(args.look_at_center),
        "trajectory_xy_std": [float(xy_std[0]), float(xy_std[1])],
        "trajectory_xy_span": [float(xy_span[0]), float(xy_span[1])],
        "pose_convention": args.pose_convention,
        "depth_scale": float(args.depth_scale),
        "min_depth": float(args.min_depth),
        "max_depth": float(args.max_depth),
        "primary_human_state_index": int(args.primary_human_state_index),
        "effective_primary_human_state": int(chosen_state[0]) if chosen_state is not None else None,
    }
    (scene_root / "capture_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n[OK] Export complete.")
    print(f"[OK] Scene folder: {scene_root}")
    print(f"[OK] Data config : {scene_root / 'dataconfig.yaml'}")
    print("\n[Next] ConceptGraph command:")
    print(
        "conda run -n conceptgraph python slam/rerun_realtime_mapping.py "
        f"dataset_root={args.output_root} "
        f"dataset_config={scene_root / 'dataconfig.yaml'} "
        f"scene_id={args.scene_id} "
        "start=0 end=-1 stride=2 "
        "use_rerun=true save_video=false vis_render=false "
        "spatial_sim_type=iou merge_interval=-1 run_merge_final_frame=false"
    )


if __name__ == "__main__":
    main()
