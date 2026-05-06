#!/usr/bin/env python3
"""SMPL-motion-driven humanoid in a MolmoSpaces scene (walk + head motion)."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any


def parse_vec3(text: str) -> list[float]:
    values = [float(x.strip()) for x in text.split(",")]
    if len(values) != 3:
        raise argparse.ArgumentTypeError(f"Expected 3 comma-separated numbers, got: {text}")
    return values


def axis_angle_to_quat_wxyz(aa: list[float] | tuple[float, float, float] | Any) -> list[float]:
    x, y, z = float(aa[0]), float(aa[1]), float(aa[2])
    theta = math.sqrt(x * x + y * y + z * z)
    if theta < 1e-12:
        return [1.0, 0.0, 0.0, 0.0]
    inv = 1.0 / theta
    ax, ay, az = x * inv, y * inv, z * inv
    half = 0.5 * theta
    s = math.sin(half)
    return [math.cos(half), ax * s, ay * s, az * s]


def yaw_deg_to_quat_wxyz(yaw_deg: float) -> list[float]:
    yaw = math.radians(yaw_deg)
    return [math.cos(yaw * 0.5), 0.0, 0.0, math.sin(yaw * 0.5)]


def quat_mul_wxyz(a: list[float], b: list[float]) -> list[float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


def set_ball_joint_qpos(model, data, joint_name: str, quat_wxyz: list[float]) -> None:
    j = model.joint(joint_name)
    adr = int(j.qposadr[0])
    data.qpos[adr : adr + 4] = quat_wxyz


def set_hinge_joint_qpos(model, data, joint_name: str, value: float) -> None:
    j = model.joint(joint_name)
    adr = int(j.qposadr[0])
    data.qpos[adr] = float(value)


def get_joint_range(model, joint_name: str) -> tuple[float, float]:
    jid = model.joint(joint_name).id
    lo, hi = model.jnt_range[jid]
    return float(lo), float(hi)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(v, hi))


def load_amass_motion(path: Path) -> dict[str, Any]:
    import numpy as np

    if not path.is_file():
        raise FileNotFoundError(f"Motion file not found: {path}")

    payload = np.load(path, allow_pickle=True)
    if "root_orient" in payload:
        root_orient = payload["root_orient"]
    elif "poses" in payload:
        root_orient = payload["poses"][:, :3]
    else:
        raise ValueError("AMASS npz must contain root_orient or poses")

    if "pose_body" in payload:
        pose_body = payload["pose_body"]
    elif "poses" in payload:
        pose_body = payload["poses"][:, 3:66]
    else:
        raise ValueError("AMASS npz must contain pose_body or poses")

    if pose_body.shape[1] < 63:
        raise ValueError(f"pose_body must have at least 63 dims, got {pose_body.shape}")
    pose_body = pose_body[:, :63]

    if "trans" in payload:
        trans = payload["trans"]
    else:
        trans = np.zeros((pose_body.shape[0], 3), dtype=float)

    if trans.shape[0] != pose_body.shape[0]:
        raise ValueError(
            f"Frame count mismatch: trans={trans.shape[0]} vs pose_body={pose_body.shape[0]}"
        )

    if "mocap_frame_rate" in payload:
        fps = float(payload["mocap_frame_rate"])
    elif "mocap_framerate" in payload:
        fps = float(payload["mocap_framerate"])
    else:
        fps = 60.0
    if fps <= 1e-6:
        fps = 60.0

    pose_jaw = payload["pose_jaw"] if "pose_jaw" in payload else None
    return {
        "root_orient": root_orient,
        "pose_body": pose_body,
        "pose_jaw": pose_jaw,
        "trans": trans,
        "fps": fps,
        "nframes": int(pose_body.shape[0]),
    }


def add_smpl_like_humanoid(spec, namespace: str = "human_0/", scale: float = 1.0) -> dict[str, str]:
    import mujoco

    s = float(scale)
    pelvis = spec.worldbody.add_body(name=f"{namespace}pelvis", pos=[0.0, 0.0, 0.95 * s])
    pelvis.add_joint(name=f"{namespace}root_freejoint", type=mujoco.mjtJoint.mjJNT_FREE, damping=1.0)
    pelvis.add_geom(
        name=f"{namespace}pelvis_geom",
        type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
        size=[0.10 * s, 0.09 * s, 0.08 * s],
        rgba=[0.83, 0.78, 0.73, 1.0],
        contype=0,
        conaffinity=0,
    )

    joint_names: dict[str, str] = {}

    def add_ball(parent, name: str, pos: list[float]):
        b = parent.add_body(name=f"{namespace}{name}", pos=pos)
        jn = f"{namespace}{name}"
        b.add_joint(name=jn, type=mujoco.mjtJoint.mjJNT_BALL, damping=0.25)
        # Ensure every articulated body has valid inertial properties.
        # MjSpec in this MuJoCo build doesn't expose add_inertial(), so we add
        # a tiny hidden geom carrying minimal mass/inertia.
        b.add_geom(
            name=f"{namespace}{name}_inertia_stub",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.003 * s, 0.003 * s, 0.003 * s],
            rgba=[0.0, 0.0, 0.0, 0.0],
            contype=0,
            conaffinity=0,
            mass=0.02 * s,
        )
        joint_names[name] = jn
        return b

    # Spine chain
    spine1 = add_ball(pelvis, "spine1", [0.0, 0.0, 0.10 * s])
    spine1.add_geom(
        name=f"{namespace}spine1_geom",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=[0, 0, 0, 0, 0, 0.12 * s],
        size=[0.055 * s, 0.055 * s, 0.055 * s],
        rgba=[0.66, 0.74, 0.84, 1.0],
        contype=0,
        conaffinity=0,
    )
    spine2 = add_ball(spine1, "spine2", [0.0, 0.0, 0.11 * s])
    spine2.add_geom(
        name=f"{namespace}spine2_geom",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=[0, 0, 0, 0, 0, 0.11 * s],
        size=[0.055 * s, 0.055 * s, 0.055 * s],
        rgba=[0.63, 0.70, 0.80, 1.0],
        contype=0,
        conaffinity=0,
    )
    spine3 = add_ball(spine2, "spine3", [0.0, 0.0, 0.11 * s])
    spine3.add_geom(
        name=f"{namespace}spine3_geom",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=[0, 0, 0, 0, 0, 0.10 * s],
        size=[0.058 * s, 0.058 * s, 0.058 * s],
        rgba=[0.61, 0.67, 0.77, 1.0],
        contype=0,
        conaffinity=0,
    )
    neck = add_ball(spine3, "neck", [0.0, 0.0, 0.10 * s])
    head = add_ball(neck, "head", [0.0, 0.0, 0.09 * s])
    head.add_geom(
        name=f"{namespace}head_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.10 * s, 0.10 * s, 0.10 * s],
        rgba=[0.96, 0.84, 0.75, 1.0],
        contype=0,
        conaffinity=0,
    )
    jaw = head.add_body(name=f"{namespace}jaw", pos=[0.05 * s, 0.0, -0.03 * s])
    jaw.add_joint(
        name=f"{namespace}jaw_pitch",
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=[0, 1, 0],
        range=[0.0, 0.8],
        limited=True,
        damping=0.2,
    )
    jaw.add_geom(
        name=f"{namespace}jaw_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.018 * s, 0.040 * s, 0.011 * s],
        pos=[0.045 * s, 0.0, -0.010 * s],
        rgba=[0.80, 0.33, 0.35, 1.0],
        contype=0,
        conaffinity=0,
    )

    # Legs
    for side, sign in (("left", 1.0), ("right", -1.0)):
        hip = add_ball(pelvis, f"{side}_hip", [0.0, 0.090 * sign * s, -0.05 * s])
        hip.add_geom(
            name=f"{namespace}{side}_thigh",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[0, 0, 0, 0, 0, -0.42 * s],
            size=[0.044 * s, 0.044 * s, 0.044 * s],
            rgba=[0.49, 0.57, 0.67, 1.0],
            contype=0,
            conaffinity=0,
        )
        knee = add_ball(hip, f"{side}_knee", [0.0, 0.0, -0.42 * s])
        knee.add_geom(
            name=f"{namespace}{side}_shin",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[0, 0, 0, 0, 0, -0.41 * s],
            size=[0.038 * s, 0.038 * s, 0.038 * s],
            rgba=[0.46, 0.53, 0.63, 1.0],
            contype=0,
            conaffinity=0,
        )
        ankle = add_ball(knee, f"{side}_ankle", [0.0, 0.0, -0.41 * s])
        foot = add_ball(ankle, f"{side}_foot", [0.0, 0.0, 0.0])
        foot.add_geom(
            name=f"{namespace}{side}_foot_geom",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[0.0, 0.0, 0.0, 0.16 * s, 0.0, -0.03 * s],
            size=[0.028 * s, 0.028 * s, 0.028 * s],
            rgba=[0.42, 0.49, 0.58, 1.0],
            contype=0,
            conaffinity=0,
        )

    # Arms
    for side, sign in (("left", 1.0), ("right", -1.0)):
        collar = add_ball(spine3, f"{side}_collar", [0.0, 0.10 * sign * s, 0.05 * s])
        shoulder = add_ball(collar, f"{side}_shoulder", [0.0, 0.08 * sign * s, 0.0])
        shoulder.add_geom(
            name=f"{namespace}{side}_upper_arm",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[0, 0, 0, 0, 0, -0.30 * s],
            size=[0.030 * s, 0.030 * s, 0.030 * s],
            rgba=[0.86, 0.78, 0.71, 1.0],
            contype=0,
            conaffinity=0,
        )
        elbow = add_ball(shoulder, f"{side}_elbow", [0.0, 0.0, -0.30 * s])
        elbow.add_geom(
            name=f"{namespace}{side}_forearm",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[0, 0, 0, 0, 0, -0.25 * s],
            size=[0.024 * s, 0.024 * s, 0.024 * s],
            rgba=[0.86, 0.78, 0.71, 1.0],
            contype=0,
            conaffinity=0,
        )
        wrist = add_ball(elbow, f"{side}_wrist", [0.0, 0.0, -0.25 * s])
        wrist.add_geom(
            name=f"{namespace}{side}_hand",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.030 * s, 0.030 * s, 0.030 * s],
            rgba=[0.86, 0.78, 0.71, 1.0],
            contype=0,
            conaffinity=0,
        )

    joint_names["jaw_pitch"] = f"{namespace}jaw_pitch"
    joint_names["root_freejoint"] = f"{namespace}root_freejoint"
    return joint_names


def add_franka_robot(spec, robot_pos: list[float]) -> None:
    import mujoco

    from molmo_spaces.configs.robot_configs import FrankaRobotConfig
    from molmo_spaces.molmo_spaces_constants import get_robot_path

    robot_cfg = FrankaRobotConfig()
    robot_xml = get_robot_path(robot_cfg.name) / robot_cfg.robot_xml_path
    robot_spec = mujoco.MjSpec.from_file(str(robot_xml))
    robot_cfg.robot_cls.add_robot_to_scene(
        robot_config=robot_cfg,
        spec=spec,
        robot_spec=robot_spec,
        prefix="robot_0/",
        pos=robot_pos,
        quat=[1, 0, 0, 0],
        randomize_textures=False,
    )
    robot_cfg.robot_cls.apply_control_overrides(spec, robot_cfg)


def build_scene(args):
    import mujoco

    from molmo_spaces.molmo_spaces_constants import ASSETS_DIR, get_scenes
    from molmo_spaces.utils.lazy_loading_utils import install_scene_with_objects_and_grasps_from_path

    scene_map = get_scenes(args.scene_source, args.scene_split)
    scene_path = scene_map[args.scene_split].get(args.scene_index)
    if scene_path is None:
        raise ValueError(
            f"Scene not found for source={args.scene_source}, split={args.scene_split}, index={args.scene_index}"
        )

    print(f"[INFO] Assets dir: {ASSETS_DIR}")
    print(f"[INFO] Installing scene dependencies: {scene_path}")
    install_scene_with_objects_and_grasps_from_path(scene_path)

    spec = mujoco.MjSpec.from_file(str(scene_path))
    if args.with_robot:
        add_franka_robot(spec, args.robot_pos)
    joint_names = add_smpl_like_humanoid(spec, namespace="human_0/", scale=args.human_scale)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, joint_names


def run_viewer(model, data, args, motion: dict[str, Any]) -> None:
    import mujoco
    import mujoco.viewer
    import numpy as np

    root_joint = "human_0/root_freejoint"
    root_qpos_adr = int(model.joint(root_joint).qposadr[0])
    root_qvel_adr = int(model.joint(root_joint).dofadr[0])

    body_joint_map = [
        "human_0/left_hip",
        "human_0/right_hip",
        "human_0/spine1",
        "human_0/left_knee",
        "human_0/right_knee",
        "human_0/spine2",
        "human_0/left_ankle",
        "human_0/right_ankle",
        "human_0/spine3",
        "human_0/left_foot",
        "human_0/right_foot",
        "human_0/neck",
        "human_0/left_collar",
        "human_0/right_collar",
        "human_0/head",
        "human_0/left_shoulder",
        "human_0/right_shoulder",
        "human_0/left_elbow",
        "human_0/right_elbow",
        "human_0/left_wrist",
        "human_0/right_wrist",
    ]
    jaw_joint = "human_0/jaw_pitch"
    jaw_min, jaw_max = get_joint_range(model, jaw_joint)

    root_orient = motion["root_orient"]
    pose_body = motion["pose_body"]
    pose_jaw = motion["pose_jaw"]
    trans = motion["trans"]
    fps = float(motion["fps"])
    nframes = int(motion["nframes"])
    trans0 = np.array(trans[0], dtype=float)

    print(f"[INFO] Motion frames={nframes}, fps={fps:.2f}")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        t0 = time.perf_counter()
        next_sync = t0
        if args.focus_human:
            viewer.cam.lookat[:] = data.qpos[root_qpos_adr : root_qpos_adr + 3]
            viewer.cam.distance = float(args.camera_distance)
            viewer.cam.azimuth = float(args.camera_azimuth)
            viewer.cam.elevation = float(args.camera_elevation)
        while viewer.is_running():
            t = (time.perf_counter() - t0) * float(args.motion_speed)
            frame = int(t * fps)
            if args.loop:
                frame %= nframes
            else:
                frame = min(frame, nframes - 1)

            root_q = axis_angle_to_quat_wxyz(root_orient[frame])
            global_yaw_q = yaw_deg_to_quat_wxyz(args.global_yaw_deg)
            root_q = quat_mul_wxyz(global_yaw_q, root_q)

            if args.root_motion:
                rel = (np.array(trans[frame], dtype=float) - trans0) * float(args.motion_scale)
                root_pos = [
                    args.human_pos[0] + float(rel[0]),
                    args.human_pos[1] + float(rel[1]),
                    args.human_pos[2] + float(rel[2]) + float(args.human_z_offset),
                ]
            else:
                root_pos = [
                    args.human_pos[0],
                    args.human_pos[1],
                    args.human_pos[2] + float(args.human_z_offset),
                ]

            data.qpos[root_qpos_adr : root_qpos_adr + 3] = root_pos
            data.qpos[root_qpos_adr + 3 : root_qpos_adr + 7] = root_q
            data.qvel[root_qvel_adr : root_qvel_adr + 6] = 0.0

            body_row = pose_body[frame].reshape(21, 3)
            for i, jname in enumerate(body_joint_map):
                q = axis_angle_to_quat_wxyz(body_row[i])
                if jname == "human_0/head":
                    extra = yaw_deg_to_quat_wxyz(
                        args.head_yaw_deg * math.sin(2.0 * math.pi * args.head_hz * t + 0.2)
                    )
                    q = quat_mul_wxyz(extra, q)
                set_ball_joint_qpos(model, data, jname, q)

            if pose_jaw is not None and frame < pose_jaw.shape[0]:
                jaw_val = abs(float(pose_jaw[frame][1])) * float(args.jaw_scale)
            else:
                jaw_val = math.radians(args.mouth_open_deg) * (
                    0.4 + 0.6 * (0.5 + 0.5 * math.sin(2.0 * math.pi * args.mouth_hz * t))
                )
            set_hinge_joint_qpos(model, data, jaw_joint, clamp(jaw_val, jaw_min, jaw_max))

            if args.camera_follow_human:
                viewer.cam.lookat[:] = data.qpos[root_qpos_adr : root_qpos_adr + 3]

            mujoco.mj_forward(model, data)
            mujoco.mj_step(model, data)
            viewer.sync()

            next_sync += model.opt.timestep
            sleep_t = next_sync - time.perf_counter()
            if sleep_t > 0:
                time.sleep(sleep_t)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-source", default="ithor")
    parser.add_argument("--scene-split", default="train")
    parser.add_argument("--scene-index", type=int, default=1)
    parser.add_argument("--with-robot", action="store_true")
    parser.add_argument("--robot-pos", type=parse_vec3, default=[0.0, -0.15, 0.0])

    parser.add_argument(
        "--motion-file",
        type=Path,
        default=Path("/Users/ljj/项目/graduation/smpl_assets/motions/walk.npz"),
    )
    parser.add_argument("--human-pos", type=parse_vec3, default=[1.0, 0.0, 0.95])
    parser.add_argument("--human-z-offset", type=float, default=0.0)
    parser.add_argument("--human-scale", type=float, default=1.0)
    parser.add_argument("--global-yaw-deg", type=float, default=180.0)
    parser.add_argument("--root-motion", action="store_true")
    parser.add_argument("--motion-scale", type=float, default=0.20)
    parser.add_argument("--motion-speed", type=float, default=1.0)
    parser.add_argument("--loop", action="store_true")

    parser.add_argument("--head-hz", type=float, default=0.35)
    parser.add_argument("--head-yaw-deg", type=float, default=18.0)
    parser.add_argument("--jaw-scale", type=float, default=0.8)
    parser.add_argument("--mouth-hz", type=float, default=1.8)
    parser.add_argument("--mouth-open-deg", type=float, default=22.0)

    parser.add_argument("--focus-human", action="store_true")
    parser.add_argument(
        "--camera-follow-human",
        action="store_true",
        help="Continuously update camera lookat to follow the moving human root",
    )
    parser.add_argument("--camera-distance", type=float, default=2.2)
    parser.add_argument("--camera-azimuth", type=float, default=155.0)
    parser.add_argument("--camera-elevation", type=float, default=-12.0)
    args = parser.parse_args()

    motion = load_amass_motion(args.motion_file)
    model, data, _ = build_scene(args)
    print("[INFO] Launching SMPL motion demo...")
    try:
        run_viewer(model, data, args, motion)
    except RuntimeError as e:
        if "launch_passive" in str(e) and "mjpython" in str(e):
            raise RuntimeError(
                "On macOS, run with mjpython:\n"
                "  ./.venv/bin/mjpython scripts/smpl_human_scene.py --focus-human --loop"
            ) from e
        raise


if __name__ == "__main__":
    main()
