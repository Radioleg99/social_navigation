#!/usr/bin/env python3
"""Build a minimal MolmoSpaces scene with one robot and one human-like object."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import time
from typing import Any
from pathlib import Path

DEFAULT_HUMAN_UID = "002c266eb95a45039fc4b9da9875a2ab"  # category: mannequin


def parse_vec3(text: str) -> list[float]:
    values = [float(x.strip()) for x in text.split(",")]
    if len(values) != 3:
        raise argparse.ArgumentTypeError(f"Expected 3 comma-separated numbers, got: {text}")
    return values


def parse_non_negative_float(text: str) -> float:
    value = float(text)
    if value < 0.0:
        raise argparse.ArgumentTypeError(f"Expected non-negative value, got: {text}")
    return value


def parse_vec3_from_json(value: Any, key_name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"layout json field '{key_name}' must be [x,y,z]")
    return [float(value[0]), float(value[1]), float(value[2])]


def parse_float_list_from_json(value: Any, key_name: str) -> list[float]:
    if not isinstance(value, list):
        raise ValueError(f"layout json field '{key_name}' must be a list")
    out: list[float] = []
    for i, v in enumerate(value):
        fv = float(v)
        if fv < 0.0:
            raise ValueError(f"layout json field '{key_name}[{i}]' must be non-negative")
        out.append(fv)
    return out


def parse_bool_from_json(value: Any, key_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "y", "on"}:
            return True
        if text in {"false", "0", "no", "n", "off"}:
            return False
    raise ValueError(f"layout json field '{key_name}' must be a boolean")


def cli_flag_provided(flag: str) -> bool:
    for token in sys.argv[1:]:
        if token == flag or token.startswith(f"{flag}="):
            return True
    return False


def any_cli_flag_provided(flags: list[str]) -> bool:
    return any(cli_flag_provided(flag) for flag in flags)


def resolve_path_from_layout(layout_json: Path, raw_path: str | Path) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p
    cwd_candidate = (Path.cwd() / p).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    layout_candidate = (layout_json.parent / p).resolve()
    if layout_candidate.exists():
        return layout_candidate
    return cwd_candidate


def load_layout_json(layout_json: Path) -> dict[str, Any]:
    if not layout_json.is_file():
        raise FileNotFoundError(f"--layout-json not found: {layout_json}")
    with open(layout_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("--layout-json must contain a JSON object")
    return payload


def apply_robot_runtime_from_layout(args: argparse.Namespace, layout_payload: dict[str, Any]) -> bool:
    runtime = layout_payload.get("robot_runtime")
    if runtime is None:
        return False
    if not isinstance(runtime, dict):
        raise ValueError("layout json field 'robot_runtime' must be an object/dict")
    if runtime.get("enabled", True) is False:
        print("[INFO] layout json has robot_runtime.enabled=false, skipping robot runtime overrides.")
        return False

    def maybe_set(attr: str, flags: list[str], value: Any) -> None:
        if value is None:
            return
        if any_cli_flag_provided(flags):
            return
        setattr(args, attr, value)

    raw_type = runtime.get("robot_type", runtime.get("type"))
    if raw_type is not None:
        robot_type = str(raw_type).strip().lower()
        if robot_type not in {"franka", "rby1", "rby1m", "navbot"}:
            raise ValueError(
                "layout json field 'robot_runtime.robot_type' must be one of "
                "['franka','rby1','rby1m','navbot']"
            )
        maybe_set("robot_type", ["--robot-type"], robot_type)

    raw_robot_pos = runtime.get("robot_pos", runtime.get("pos_xyz", runtime.get("position_xyz")))
    if raw_robot_pos is not None:
        maybe_set("robot_pos", ["--robot-pos"], parse_vec3_from_json(raw_robot_pos, "robot_runtime.robot_pos"))

    raw_robot_base = runtime.get("robot_base", runtime.get("base_xyztheta", runtime.get("base_pose")))
    if raw_robot_base is not None:
        maybe_set(
            "robot_base",
            ["--robot-base"],
            parse_vec3_from_json(raw_robot_base, "robot_runtime.robot_base"),
        )

    return True


def apply_human_runtime_from_layout(args: argparse.Namespace, layout_payload: dict[str, Any]) -> bool:
    runtime = layout_payload.get("human_runtime")
    if runtime is None:
        return False
    if not isinstance(runtime, dict):
        raise ValueError("layout json field 'human_runtime' must be an object/dict")
    if runtime.get("enabled", True) is False:
        print("[INFO] layout json has human_runtime.enabled=false, skipping human runtime overrides.")
        return False

    layout_json = args.layout_json

    def maybe_set(attr: str, flags: list[str], value: Any) -> None:
        if value is None:
            return
        if any_cli_flag_provided(flags):
            return
        setattr(args, attr, value)

    raw_pos = runtime.get("human_pos", runtime.get("pos_xyz", runtime.get("position_xyz")))
    if raw_pos is not None:
        maybe_set("human_pos", ["--human-pos"], parse_vec3_from_json(raw_pos, "human_runtime.human_pos"))

    raw_yaw = runtime.get("human_yaw_deg", runtime.get("yaw_deg"))
    if raw_yaw is not None:
        maybe_set("human_yaw_deg", ["--human-yaw-deg"], float(raw_yaw))

    raw_roll = runtime.get("human_roll_deg", runtime.get("roll_deg"))
    if raw_roll is not None:
        maybe_set("human_roll_deg", ["--human-roll-deg"], float(raw_roll))

    raw_pitch = runtime.get("human_pitch_deg", runtime.get("pitch_deg"))
    if raw_pitch is not None:
        maybe_set("human_pitch_deg", ["--human-pitch-deg"], float(raw_pitch))

    raw_yaw_offset = runtime.get("human_yaw_offset_deg", runtime.get("yaw_offset_deg"))
    if raw_yaw_offset is not None:
        maybe_set("human_yaw_offset_deg", ["--human-yaw-offset-deg"], float(raw_yaw_offset))

    raw_z_offset = runtime.get("human_z_offset", runtime.get("z_offset"))
    if raw_z_offset is not None:
        maybe_set("human_z_offset", ["--human-z-offset"], float(raw_z_offset))

    raw_static = runtime.get("human_static", runtime.get("static"))
    if raw_static is not None:
        maybe_set("human_static", ["--human-static"], parse_bool_from_json(raw_static, "human_runtime.human_static"))

    raw_col_type = runtime.get("human_collider_type")
    if raw_col_type is not None:
        maybe_set("human_collider_type", ["--human-collider-type"], str(raw_col_type))
    raw_col_size = runtime.get("human_collider_size")
    if raw_col_size is not None and not any_cli_flag_provided(["--human-collider-size"]):
        args.human_collider_size = parse_vec3_from_json(
            raw_col_size, "human_runtime.human_collider_size"
        )

    raw_uid = runtime.get("human_uid", runtime.get("uid"))
    if raw_uid is not None:
        maybe_set("human_uid", ["--human-uid"], str(raw_uid))

    raw_xml = runtime.get("human_xml", runtime.get("xml"))
    if raw_xml is not None:
        resolved = resolve_path_from_layout(layout_json, str(raw_xml))
        maybe_set("human_xml", ["--human-xml"], resolved)

    simple_pose_mode_used = False

    raw_state_xmls = runtime.get("human_state_xmls", runtime.get("state_xmls"))
    if raw_state_xmls is not None and not any_cli_flag_provided(["--human-state-xmls"]):
        if not isinstance(raw_state_xmls, list):
            raise ValueError("layout json field 'human_runtime.human_state_xmls' must be a list")
        if len(raw_state_xmls) == 0:
            args.human_state_xmls = None
        else:
            args.human_state_xmls = [resolve_path_from_layout(layout_json, str(p)) for p in raw_state_xmls]

    # Simplified mode:
    # human_runtime.poses = ["stand.xml", "sit.xml", ...]
    # human_runtime.pose_interval_sec = 2.0
    # human_runtime.loop = true
    raw_poses = runtime.get("poses")
    if raw_poses is not None and not any_cli_flag_provided(["--human-state-xmls", "--human-xml"]):
        if not isinstance(raw_poses, list):
            raise ValueError("layout json field 'human_runtime.poses' must be a list")
        resolved_poses = [resolve_path_from_layout(layout_json, str(p)) for p in raw_poses]
        if len(resolved_poses) == 0:
            pass
        elif len(resolved_poses) == 1:
            simple_pose_mode_used = True
            args.human_xml = resolved_poses[0]
            args.human_state_xmls = None
        else:
            simple_pose_mode_used = True
            args.human_state_xmls = resolved_poses
            if not any_cli_flag_provided(["--human-static"]):
                args.human_static = True
            if not any_cli_flag_provided(["--human-state-times"]):
                interval = float(runtime.get("pose_interval_sec", 2.0))
                if interval <= 0.0:
                    raise ValueError("layout json field 'human_runtime.pose_interval_sec' must be > 0")
                args.human_state_times = [float(i) * interval for i in range(len(resolved_poses))]
            if not any_cli_flag_provided(["--human-state-loop"]):
                raw_loop = runtime.get("loop", True)
                args.human_state_loop = parse_bool_from_json(raw_loop, "human_runtime.loop")

    raw_state_times = runtime.get("human_state_times", runtime.get("state_times"))
    if raw_state_times is not None and not any_cli_flag_provided(["--human-state-times"]):
        parsed_times = parse_float_list_from_json(
            raw_state_times, "human_runtime.human_state_times"
        )
        if len(parsed_times) > 0:
            args.human_state_times = parsed_times
        elif not simple_pose_mode_used:
            args.human_state_times = None

    if ("human_state_loop" in runtime or "state_loop" in runtime) and not (
        simple_pose_mode_used and "loop" in runtime
    ):
        raw_state_loop = runtime.get("human_state_loop", runtime.get("state_loop"))
        if raw_state_loop is not None:
            maybe_set(
                "human_state_loop",
                ["--human-state-loop"],
                parse_bool_from_json(raw_state_loop, "human_runtime.human_state_loop"),
            )
    if "human_state_hold_sec" in runtime or "state_hold_sec" in runtime:
        maybe_set(
            "human_state_hold_sec",
            ["--human-state-hold-sec"],
            float(runtime.get("human_state_hold_sec", runtime.get("state_hold_sec"))),
        )

    extra_humans: list[dict[str, Any]] = []
    raw_extra = runtime.get("extra_humans")
    if raw_extra is not None:
        if not isinstance(raw_extra, list):
            raise ValueError("layout json field 'human_runtime.extra_humans' must be a list")
        for i, item in enumerate(raw_extra):
            if not isinstance(item, dict):
                raise ValueError(f"layout json field 'human_runtime.extra_humans[{i}]' must be an object")
            enabled = item.get("enabled", True)
            if not parse_bool_from_json(enabled, f"human_runtime.extra_humans[{i}].enabled"):
                continue

            entry: dict[str, Any] = {}
            raw_pos2 = item.get("human_pos", item.get("pos_xyz", item.get("position_xyz")))
            if raw_pos2 is None:
                raise ValueError(
                    f"layout json field 'human_runtime.extra_humans[{i}].human_pos' is required"
                )
            entry["human_pos"] = parse_vec3_from_json(
                raw_pos2, f"human_runtime.extra_humans[{i}].human_pos"
            )

            raw_xml2 = item.get("human_xml", item.get("xml"))
            raw_uid2 = item.get("human_uid", item.get("uid"))
            if raw_xml2 is None and raw_uid2 is None:
                raise ValueError(
                    f"human_runtime.extra_humans[{i}] requires one of 'human_xml' or 'human_uid'"
                )
            if raw_xml2 is not None:
                entry["human_xml"] = resolve_path_from_layout(layout_json, str(raw_xml2))
            else:
                entry["human_xml"] = None
            entry["human_uid"] = str(raw_uid2) if raw_uid2 is not None else None

            entry["human_yaw_deg"] = float(item.get("human_yaw_deg", item.get("yaw_deg", args.human_yaw_deg)))
            entry["human_roll_deg"] = float(
                item.get("human_roll_deg", item.get("roll_deg", args.human_roll_deg))
            )
            entry["human_pitch_deg"] = float(
                item.get("human_pitch_deg", item.get("pitch_deg", args.human_pitch_deg))
            )
            entry["human_yaw_offset_deg"] = float(
                item.get("human_yaw_offset_deg", item.get("yaw_offset_deg", args.human_yaw_offset_deg))
            )
            entry["human_z_offset"] = float(item.get("human_z_offset", item.get("z_offset", args.human_z_offset)))
            entry["human_static"] = parse_bool_from_json(
                item.get("human_static", item.get("static", True)),
                f"human_runtime.extra_humans[{i}].human_static",
            )
            entry["human_collider_type"] = str(item.get("human_collider_type", args.human_collider_type))
            raw_size2 = item.get("human_collider_size")
            if raw_size2 is None:
                entry["human_collider_size"] = list(args.human_collider_size)
            else:
                entry["human_collider_size"] = parse_vec3_from_json(
                    raw_size2, f"human_runtime.extra_humans[{i}].human_collider_size"
                )
            extra_humans.append(entry)
    args.extra_humans = extra_humans

    return True


def yaw_deg_to_quat_wxyz(yaw_deg: float) -> list[float]:
    yaw = math.radians(yaw_deg)
    return [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]


def euler_xyz_deg_to_quat_wxyz(roll_deg: float, pitch_deg: float, yaw_deg: float) -> list[float]:
    """Convert xyz Euler (deg) to MuJoCo wxyz quaternion."""
    from scipy.spatial.transform import Rotation as R

    quat_xyzw = R.from_euler("xyz", [roll_deg, pitch_deg, yaw_deg], degrees=True).as_quat()
    return [float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2])]


def wrap_angle_deg(diff: float) -> float:
    return (diff + 180.0) % 360.0 - 180.0


def lerp(a: float, b: float, alpha: float) -> float:
    return a + alpha * (b - a)


def lerp_vec3(a: list[float], b: list[float], alpha: float) -> list[float]:
    return [lerp(a[i], b[i], alpha) for i in range(3)]


def load_human_trajectory(path: Path) -> tuple[list[dict[str, Any]], bool]:
    if not path.is_file():
        raise FileNotFoundError(f"Trajectory file not found: {path}")

    with open(path, "r") as f:
        payload = json.load(f)

    if "waypoints" not in payload or not isinstance(payload["waypoints"], list):
        raise ValueError("Trajectory JSON must contain a list field: 'waypoints'")

    waypoints = []
    for i, wp in enumerate(payload["waypoints"]):
        if not isinstance(wp, dict):
            raise ValueError(f"Waypoint #{i} must be an object")
        if "t" not in wp or "pos" not in wp:
            raise ValueError(f"Waypoint #{i} must include 't' and 'pos'")

        pos = wp["pos"]
        if not isinstance(pos, list) or len(pos) != 3:
            raise ValueError(f"Waypoint #{i} pos must be [x,y,z]")

        yaw_deg = float(wp.get("yaw_deg", 180.0))
        waypoints.append(
            {
                "t": float(wp["t"]),
                "pos": [float(pos[0]), float(pos[1]), float(pos[2])],
                "yaw_deg": yaw_deg,
            }
        )

    if len(waypoints) < 2:
        raise ValueError("Trajectory requires at least 2 waypoints")

    waypoints.sort(key=lambda x: x["t"])
    for i in range(1, len(waypoints)):
        if waypoints[i]["t"] <= waypoints[i - 1]["t"]:
            raise ValueError("Waypoint time 't' must be strictly increasing")

    loop = bool(payload.get("loop", False))
    return waypoints, loop


def sample_trajectory(waypoints: list[dict[str, Any]], t: float) -> tuple[list[float], float]:
    if t <= waypoints[0]["t"]:
        wp = waypoints[0]
        return wp["pos"], float(wp["yaw_deg"])
    if t >= waypoints[-1]["t"]:
        wp = waypoints[-1]
        return wp["pos"], float(wp["yaw_deg"])

    left_idx = 0
    for i in range(len(waypoints) - 1):
        if waypoints[i]["t"] <= t <= waypoints[i + 1]["t"]:
            left_idx = i
            break

    w0 = waypoints[left_idx]
    w1 = waypoints[left_idx + 1]
    alpha = (t - w0["t"]) / (w1["t"] - w0["t"])

    pos = lerp_vec3(w0["pos"], w1["pos"], alpha)
    yaw_delta = wrap_angle_deg(w1["yaw_deg"] - w0["yaw_deg"])
    yaw = w0["yaw_deg"] + alpha * yaw_delta
    return pos, float(yaw)


def pick_default_human_uid() -> str:
    from molmo_spaces.molmo_spaces_constants import ASSETS_DIR

    metadata_path = ASSETS_DIR / "objects" / "objathor_metadata" / "objects_metadata.json.gz"
    if not metadata_path.is_file():
        return DEFAULT_HUMAN_UID

    try:
        with gzip.open(metadata_path, "rt") as f:
            metadata = json.load(f)
    except Exception:
        return DEFAULT_HUMAN_UID

    for preferred_category in ("mannequin", "person"):
        matches = [uid for uid, info in metadata.items() if info.get("category") == preferred_category]
        if matches:
            matches.sort()
            return matches[0]
    return DEFAULT_HUMAN_UID


def add_franka_robot(spec, robot_pos: list[float]) -> None:
    import mujoco

    from molmo_spaces.configs.robot_configs import FrankaRobotConfig
    from molmo_spaces.molmo_spaces_constants import get_robot_path

    robot_config = FrankaRobotConfig()
    robot_file = get_robot_path(robot_config.name) / robot_config.robot_xml_path
    robot_spec = mujoco.MjSpec.from_file(str(robot_file))

    robot_config.robot_cls.add_robot_to_scene(
        robot_config=robot_config,
        spec=spec,
        robot_spec=robot_spec,
        prefix="robot_0/",
        pos=robot_pos,
        quat=[1, 0, 0, 0],
        randomize_textures=False,
    )
    robot_config.robot_cls.apply_control_overrides(spec, robot_config)


def add_rby1_robot(spec, robot_type: str) -> None:
    from molmo_spaces.configs.robot_configs import RBY1Config, RBY1MConfig

    if robot_type == "rby1":
        robot_config = RBY1Config()
    elif robot_type == "rby1m":
        robot_config = RBY1MConfig()
    else:
        raise ValueError(f"Unsupported RBY1 robot_type: {robot_type}")

    robot_config.robot_cls.apply_control_overrides(spec, robot_config)


def resolve_navbot_xml() -> Path:
    candidate = Path(__file__).resolve().parents[1] / "assets" / "robots" / "navbot" / "model.xml"
    if not candidate.is_file():
        raise FileNotFoundError(f"navbot xml not found: {candidate}")
    return candidate


def set_named_joint_qpos(model, data, joint_name: str, value: float) -> bool:
    try:
        joint = model.joint(joint_name)
    except Exception:
        return False
    data.qpos[int(joint.qposadr[0])] = float(value)
    return True


def apply_planar_base_pose(model, data, robot_base: list[float]) -> None:
    x, y, theta = float(robot_base[0]), float(robot_base[1]), float(robot_base[2])
    ok_x = set_named_joint_qpos(model, data, "robot_0/base_x", x)
    ok_y = set_named_joint_qpos(model, data, "robot_0/base_y", y)
    ok_t = set_named_joint_qpos(model, data, "robot_0/base_theta", theta)
    if not (ok_x and ok_y and ok_t):
        print(
            "[WARN] Could not set full mobile-base pose via robot_0/base_x,base_y,base_theta. "
            "Keeping default base pose."
        )


def add_human_object(
    spec,
    human_xml: Path,
    human_pos: list[float],
    human_yaw_deg: float,
    human_roll_deg: float,
    human_pitch_deg: float,
    human_yaw_offset_deg: float,
    human_z_offset: float,
    dynamic_human: bool,
    name_prefix: str = "human_0/",
) -> Path:
    import mujoco
    human_spec = mujoco.MjSpec.from_file(str(human_xml))
    if len(human_spec.worldbody.bodies) == 0:
        raise ValueError(f"No body found in object XML: {human_xml}")

    human_body = human_spec.worldbody.bodies[0]
    if not human_body.name:
        human_body.name = "human_body"

    # In static mode we must ensure there is NO joint on the human body.
    # Otherwise a pre-existing freejoint from the asset can make it fall through the floor.
    if dynamic_human:
        if not human_body.first_joint():
            human_body.add_joint(
                name="XYZ_jntfree",
                type=mujoco.mjtJoint.mjJNT_FREE,
                damping=0.5,
            )
    else:
        joint_handle = human_body.first_joint()
        while joint_handle is not None:
            next_joint = human_body.next_joint(joint_handle)
            human_spec.delete(joint_handle)
            joint_handle = next_joint

    yaw_total = human_yaw_deg + human_yaw_offset_deg
    human_quat = euler_xyz_deg_to_quat_wxyz(human_roll_deg, human_pitch_deg, yaw_total)
    human_pos_with_offset = [human_pos[0], human_pos[1], human_pos[2] + human_z_offset]
    frame = spec.worldbody.add_frame(pos=human_pos_with_offset, quat=human_quat)
    frame.attach_body(human_body, name_prefix, "")
    return human_xml


def add_human_blocker(
    spec,
    blocker_name: str,
    human_pos: list[float],
    human_yaw_deg: float,
    human_z_offset: float,
    collider_type: str,
    collider_size: list[float],
) -> None:
    import mujoco

    if collider_type == "none":
        return

    world_pos = [human_pos[0], human_pos[1], human_pos[2] + human_z_offset]

    if collider_type == "capsule":
        # Keep capsule axis vertical in world-Z. Yaw has no effect for a capsule
        # and can visually confuse orientation when debugging.
        body = spec.worldbody.add_body(name=blocker_name, pos=world_pos)
        # capsule size = [radius, height, z_start]
        radius = max(0.01, float(collider_size[0]))
        height = max(0.1, float(collider_size[1]))
        z_start = float(collider_size[2])
        half = 0.5 * height
        body.add_geom(
            name=f"{blocker_name}_geom",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=[radius, half, 0.0],
            pos=[0.0, 0.0, z_start + half],
            contype=1,
            conaffinity=1,
            rgba=[1.0, 0.6, 0.1, 0.0],
        )
        return

    if collider_type == "box":
        yaw_quat = yaw_deg_to_quat_wxyz(human_yaw_deg)
        body = spec.worldbody.add_body(name=blocker_name, pos=world_pos, quat=yaw_quat)
        # box size = [half_x, half_y, half_z]
        half_x = max(0.05, float(collider_size[0]))
        half_y = max(0.05, float(collider_size[1]))
        half_z = max(0.1, float(collider_size[2]))
        body.add_geom(
            name=f"{blocker_name}_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[half_x, half_y, half_z],
            pos=[0.0, 0.0, half_z],
            contype=1,
            conaffinity=1,
            rgba=[0.2, 0.9, 0.2, 0.0],
        )
        return

    raise ValueError(f"Unsupported human collider type: {collider_type}")


def resolve_human_xml(human_uid: str | None, human_xml: Path | None) -> tuple[Path, str]:
    if human_xml is not None:
        if not human_xml.is_file():
            raise FileNotFoundError(f"--human-xml not found: {human_xml}")
        return human_xml, f"local_xml:{human_xml}"

    from molmo_spaces.utils.lazy_loading_utils import install_uid

    resolved_uid = human_uid or pick_default_human_uid()
    resolved_xml = install_uid(resolved_uid)
    return resolved_xml, f"objaverse_uid:{resolved_uid}"


def resolve_human_state_xmls(human_state_xmls: list[Path]) -> list[Path]:
    resolved: list[Path] = []
    for xml_path in human_state_xmls:
        if not xml_path.is_file():
            raise FileNotFoundError(f"--human-state-xmls entry not found: {xml_path}")
        resolved.append(xml_path)
    if len(resolved) == 0:
        raise ValueError("--human-state-xmls was provided but empty")
    return resolved


def build_state_start_times(
    num_states: int,
    input_times: list[float] | None,
    default_step: float,
) -> list[float]:
    if num_states <= 0:
        return []
    if input_times is None:
        return [i * default_step for i in range(num_states)]

    if len(input_times) != num_states:
        raise ValueError(
            f"--human-state-times length ({len(input_times)}) must match --human-state-xmls length ({num_states})"
        )
    if input_times[0] < 0.0:
        raise ValueError("--human-state-times must be non-negative")
    for i in range(1, len(input_times)):
        if input_times[i] <= input_times[i - 1]:
            raise ValueError("--human-state-times must be strictly increasing")
    return [float(v) for v in input_times]


def build_scene(args: argparse.Namespace) -> tuple[Any, Any, str, list[str]]:
    import mujoco

    from molmo_spaces.molmo_spaces_constants import ASSETS_DIR, get_scenes
    from molmo_spaces.molmo_spaces_constants import get_robot_path
    from molmo_spaces.tasks.scene_xml_utils import xml_add_rby1_to_scene
    from molmo_spaces.utils.lazy_loading_utils import (
        install_scene_with_objects_and_grasps_from_path,
    )

    if args.scene_xml is not None:
        if not args.scene_xml.is_file():
            raise FileNotFoundError(f"--scene-xml not found: {args.scene_xml}")
        scene_path = args.scene_xml
    else:
        scene_map = get_scenes(args.scene_source, args.scene_split)
        scene_path = scene_map[args.scene_split].get(args.scene_index)
        if scene_path is None:
            raise ValueError(
                f"Scene not found for source={args.scene_source}, split={args.scene_split}, index={args.scene_index}"
            )

    print(f"[INFO] Assets dir: {ASSETS_DIR}")
    print(f"[INFO] Installing scene dependencies: {scene_path}")
    scenes_root = (ASSETS_DIR / "scenes").resolve()
    scene_resolved = scene_path.resolve()
    if scene_resolved.is_relative_to(scenes_root):
        try:
            install_scene_with_objects_and_grasps_from_path(scene_resolved)
        except Exception as exc:
            print(
                "[WARN] Dependency auto-install failed for scene xml; continuing without install. "
                f"reason={exc}"
            )
    else:
        print(
            "[INFO] Skipping dependency auto-install for custom --scene-xml outside assets/scenes: "
            f"{scene_path}"
        )

    if args.robot_type == "franka":
        spec = mujoco.MjSpec.from_file(str(scene_path))
        add_franka_robot(spec, args.robot_pos)
    elif args.robot_type in {"rby1", "rby1m"}:
        from molmo_spaces.configs.robot_configs import RBY1Config, RBY1MConfig

        robot_cfg = RBY1Config() if args.robot_type == "rby1" else RBY1MConfig()
        robot_xml = get_robot_path(robot_cfg.name) / robot_cfg.robot_xml_path
        spec = xml_add_rby1_to_scene(None, str(scene_path), str(robot_xml))
        add_rby1_robot(spec, args.robot_type)
    elif args.robot_type == "navbot":
        robot_xml = resolve_navbot_xml()
        spec = xml_add_rby1_to_scene(None, str(scene_path), str(robot_xml))
    else:
        raise ValueError(f"Unsupported --robot-type: {args.robot_type}")

    human_prefixes: list[str] = []
    if args.human_state_xmls is not None:
        resolved_state_xmls = resolve_human_state_xmls(args.human_state_xmls)
        human_sources = []
        for i, state_xml in enumerate(resolved_state_xmls):
            prefix = f"human_state_{i}/"
            add_human_object(
                spec,
                state_xml,
                args.human_pos,
                args.human_yaw_deg,
                args.human_roll_deg,
                args.human_pitch_deg,
                args.human_yaw_offset_deg,
                args.human_z_offset,
                dynamic_human=False,
                name_prefix=prefix,
            )
            human_prefixes.append(prefix)
            human_sources.append(f"state[{i}]:{state_xml}")
        if args.human_static and args.human_collider_type != "none":
            yaw_total = args.human_yaw_deg + args.human_yaw_offset_deg
            add_human_blocker(
                spec=spec,
                blocker_name="human_0/blocker",
                human_pos=args.human_pos,
                human_yaw_deg=yaw_total,
                human_z_offset=args.human_z_offset,
                collider_type=args.human_collider_type,
                collider_size=args.human_collider_size,
            )
        human_source = "multi_state_xmls"
        print(f"[INFO] Added {len(resolved_state_xmls)} human states:")
        for src in human_sources:
            print(f"  - {src}")
    else:
        human_xml, human_source = resolve_human_xml(args.human_uid, args.human_xml)
        human_xml = add_human_object(
            spec,
            human_xml,
            args.human_pos,
            args.human_yaw_deg,
            args.human_roll_deg,
            args.human_pitch_deg,
            args.human_yaw_offset_deg,
            args.human_z_offset,
            dynamic_human=not args.human_static,
            name_prefix="human_0/",
        )
        human_prefixes = ["human_0/"]
        if args.human_static and args.human_collider_type != "none":
            yaw_total = args.human_yaw_deg + args.human_yaw_offset_deg
            add_human_blocker(
                spec=spec,
                blocker_name="human_0/blocker",
                human_pos=args.human_pos,
                human_yaw_deg=yaw_total,
                human_z_offset=args.human_z_offset,
                collider_type=args.human_collider_type,
                collider_size=args.human_collider_size,
            )
        print(f"[INFO] Added human object source={human_source} from {human_xml}")

    extra_humans = getattr(args, "extra_humans", [])
    if len(extra_humans) > 0:
        print(f"[INFO] Adding extra humans: count={len(extra_humans)}")
    for i, h in enumerate(extra_humans):
        xml_path, src = resolve_human_xml(h.get("human_uid"), h.get("human_xml"))
        prefix = f"human_extra_{i}/"
        add_human_object(
            spec,
            xml_path,
            h["human_pos"],
            h["human_yaw_deg"],
            h["human_roll_deg"],
            h["human_pitch_deg"],
            h["human_yaw_offset_deg"],
            h["human_z_offset"],
            dynamic_human=not bool(h["human_static"]),
            name_prefix=prefix,
        )
        if bool(h["human_static"]) and h.get("human_collider_type", "none") != "none":
            yaw_total = float(h["human_yaw_deg"]) + float(h["human_yaw_offset_deg"])
            add_human_blocker(
                spec=spec,
                blocker_name=f"{prefix}blocker",
                human_pos=h["human_pos"],
                human_yaw_deg=yaw_total,
                human_z_offset=h["human_z_offset"],
                collider_type=h.get("human_collider_type", "none"),
                collider_size=h.get("human_collider_size", args.human_collider_size),
            )
        print(f"  - extra[{i}] source={src} xml={xml_path} pos={h['human_pos']}")

    model = spec.compile()
    data = mujoco.MjData(model)
    if args.robot_type in {"rby1", "rby1m", "navbot"}:
        apply_planar_base_pose(model, data, args.robot_base)
    mujoco.mj_forward(model, data)
    return model, data, human_source, human_prefixes


def find_human_freejoint(model, prefix: str = "human_0/") -> tuple[int, int]:
    import mujoco

    for jnt_id in range(model.njnt):
        is_free = model.jnt_type[jnt_id] == mujoco.mjtJoint.mjJNT_FREE
        if not is_free:
            continue
        jname = model.joint(jnt_id).name
        if jname.startswith(prefix):
            qpos_adr = int(model.jnt_qposadr[jnt_id])
            qvel_adr = int(model.jnt_dofadr[jnt_id])
            return qpos_adr, qvel_adr

    raise RuntimeError(f"No freejoint found for {prefix}. Ensure selected human object is movable.")


def find_human_body_id(model, prefix: str = "human_0/") -> int:
    for body_id in range(model.nbody):
        bname = model.body(body_id).name
        if bname.startswith(prefix):
            return body_id
    raise RuntimeError(f"No body found for {prefix}. Ensure human object was attached.")


def find_human_body_ids(model, prefixes: list[str]) -> list[int]:
    return [find_human_body_id(model, prefix=pfx) for pfx in prefixes]


def find_human_geom_ids(model, prefix: str) -> list[int]:
    geom_ids: list[int] = []
    for geom_id in range(model.ngeom):
        gname = model.geom(geom_id).name
        if gname.startswith(prefix):
            geom_ids.append(geom_id)
    return geom_ids


def apply_static_human_transform(
    model,
    data,
    body_ids: list[int],
    pos: list[float],
    quat_wxyz: list[float],
) -> None:
    for bid in body_ids:
        model.body_pos[bid, 0] = float(pos[0])
        model.body_pos[bid, 1] = float(pos[1])
        model.body_pos[bid, 2] = float(pos[2])
        model.body_quat[bid, 0] = float(quat_wxyz[0])
        model.body_quat[bid, 1] = float(quat_wxyz[1])
        model.body_quat[bid, 2] = float(quat_wxyz[2])
        model.body_quat[bid, 3] = float(quat_wxyz[3])
    import mujoco

    mujoco.mj_forward(model, data)


def print_tune_help() -> None:
    print("[TUNE] Human placement hotkeys:")
    print("  W/S: +/- Y, A/D: -/+ X, R/F: +/- Z")
    print("  J/L: yaw -/+")
    print("  P: print current pose   H: print this help")



def state_index_for_time(
    elapsed: float,
    state_start_times: list[float],
    loop: bool,
    hold_sec: float,
) -> int:
    if len(state_start_times) <= 1:
        return 0

    if loop:
        cycle = state_start_times[-1] + hold_sec
        if cycle <= 1e-8:
            return 0
        t = elapsed % cycle
    else:
        t = elapsed

    idx = 0
    for i in range(1, len(state_start_times)):
        if t >= state_start_times[i]:
            idx = i
        else:
            break
    return idx


def apply_visible_state(
    model,
    state_geom_ids: list[list[int]],
    base_alpha: list[float],
    active_idx: int,
) -> None:
    for i, geom_ids in enumerate(state_geom_ids):
        for gid in geom_ids:
            model.geom_rgba[gid, 3] = base_alpha[gid] if i == active_idx else 0.0


def run_interactive_viewer(
    model,
    data,
    args,
    waypoints: list[dict[str, Any]] | None,
    loop_trajectory: bool,
    human_prefixes: list[str],
    state_start_times: list[float] | None,
) -> None:
    import mujoco
    import mujoco.viewer

    if args.human_static and waypoints is not None:
        raise ValueError("Trajectory playback requires a movable human. Remove --human-static.")
    if len(human_prefixes) > 1 and waypoints is not None:
        raise ValueError("Trajectory playback cannot be combined with --human-state-xmls.")
    if len(human_prefixes) > 1 and not args.human_static:
        raise ValueError("--human-state-xmls requires --human-static.")

    qpos_adr = None
    qvel_adr = None
    primary_prefix = human_prefixes[0] if len(human_prefixes) > 0 else "human_0/"
    human_body_ids = find_human_body_ids(model, human_prefixes)
    if not args.human_static:
        qpos_adr, qvel_adr = find_human_freejoint(model, prefix=primary_prefix)
    human_body_id = find_human_body_id(model, prefix=primary_prefix)

    multi_state_mode = len(human_prefixes) > 1
    state_geom_ids: list[list[int]] = []
    base_alpha: list[float] = [float(model.geom_rgba[i, 3]) for i in range(model.ngeom)]
    current_state_idx = -1
    if multi_state_mode:
        for prefix in human_prefixes:
            geom_ids = find_human_geom_ids(model, prefix)
            if len(geom_ids) == 0:
                raise RuntimeError(f"No geoms found for state prefix: {prefix}")
            state_geom_ids.append(geom_ids)
        assert state_start_times is not None
        first_idx = state_index_for_time(0.0, state_start_times, args.human_state_loop, args.human_state_hold_sec)
        apply_visible_state(model, state_geom_ids, base_alpha, first_idx)
        current_state_idx = first_idx

    tune_pos = [
        float(args.human_pos[0]),
        float(args.human_pos[1]),
        float(args.human_pos[2] + args.human_z_offset),
    ]
    tune_yaw = float(args.human_yaw_deg)

    if args.tune_human and args.human_static:
        init_quat = euler_xyz_deg_to_quat_wxyz(
            args.human_roll_deg,
            args.human_pitch_deg,
            tune_yaw + args.human_yaw_offset_deg,
        )
        apply_static_human_transform(model, data, human_body_ids, tune_pos, init_quat)

    def print_tune_pose() -> None:
        print(
            "[TUNE] pos="
            f"{tune_pos[0]:.3f},{tune_pos[1]:.3f},{tune_pos[2]:.3f} "
            f"yaw_deg={tune_yaw:.2f} "
            f"(recommended CLI: --human-pos {tune_pos[0]:.3f},{tune_pos[1]:.3f},{(tune_pos[2]-args.human_z_offset):.3f} "
            f"--human-yaw-deg {tune_yaw:.2f} --human-z-offset {args.human_z_offset:.3f})"
        )

    def key_callback(keycode: int) -> None:
        nonlocal tune_yaw
        moved = False
        # W/S/A/D/R/F/J/L keys
        if keycode == 87:  # W
            tune_pos[1] += float(args.tune_step_xy)
            moved = True
        elif keycode == 83:  # S
            tune_pos[1] -= float(args.tune_step_xy)
            moved = True
        elif keycode == 65:  # A
            tune_pos[0] -= float(args.tune_step_xy)
            moved = True
        elif keycode == 68:  # D
            tune_pos[0] += float(args.tune_step_xy)
            moved = True
        elif keycode == 82:  # R
            tune_pos[2] += float(args.tune_step_z)
            moved = True
        elif keycode == 70:  # F
            tune_pos[2] -= float(args.tune_step_z)
            moved = True
        elif keycode == 74:  # J
            tune_yaw -= float(args.tune_step_yaw_deg)
            moved = True
        elif keycode == 76:  # L
            tune_yaw += float(args.tune_step_yaw_deg)
            moved = True
        elif keycode == 80:  # P
            print_tune_pose()
        elif keycode == 72:  # H
            print_tune_help()

        if not moved:
            return

        args.human_pos[0] = float(tune_pos[0])
        args.human_pos[1] = float(tune_pos[1])
        args.human_pos[2] = float(tune_pos[2] - args.human_z_offset)
        args.human_yaw_deg = float(tune_yaw)

        if args.human_static:
            quat = euler_xyz_deg_to_quat_wxyz(
                args.human_roll_deg,
                args.human_pitch_deg,
                tune_yaw + args.human_yaw_offset_deg,
            )
            apply_static_human_transform(model, data, human_body_ids, tune_pos, quat)

    trajectory_duration = 0.0
    if waypoints is not None:
        trajectory_duration = float(waypoints[-1]["t"])
        print(
            f"[INFO] Human trajectory enabled: {len(waypoints)} waypoints, duration={trajectory_duration:.2f}s, loop={loop_trajectory}"
        )

    key_cb = key_callback if args.tune_human else None
    with mujoco.viewer.launch_passive(model, data, key_callback=key_cb) as viewer:
        if args.tune_human:
            print_tune_help()
            print_tune_pose()
        if args.focus_human:
            if qpos_adr is not None:
                viewer.cam.lookat[:] = data.qpos[qpos_adr : qpos_adr + 3]
            else:
                viewer.cam.lookat[:] = data.xpos[human_body_id]
            viewer.cam.distance = float(args.camera_distance)
            viewer.cam.azimuth = float(args.camera_azimuth)
            viewer.cam.elevation = float(args.camera_elevation)

        wall_start = time.perf_counter()
        next_sync_time = wall_start
        while viewer.is_running():
            now = time.perf_counter()
            elapsed = now - wall_start

            if multi_state_mode:
                assert state_start_times is not None
                state_idx = state_index_for_time(
                    elapsed,
                    state_start_times,
                    args.human_state_loop,
                    args.human_state_hold_sec,
                )
                if state_idx != current_state_idx:
                    apply_visible_state(model, state_geom_ids, base_alpha, state_idx)
                    current_state_idx = state_idx

            # Keep dynamic human kinematic every frame to prevent falling under gravity.
            if qpos_adr is not None and qvel_adr is not None:
                if waypoints is not None:
                    t = elapsed
                    if loop_trajectory and trajectory_duration > 0:
                        t = t % trajectory_duration
                    else:
                        t = min(t, trajectory_duration)
                    pos, yaw_deg = sample_trajectory(waypoints, t)
                else:
                    pos, yaw_deg = args.human_pos, args.human_yaw_deg

                quat = euler_xyz_deg_to_quat_wxyz(
                    args.human_roll_deg,
                    args.human_pitch_deg,
                    yaw_deg + args.human_yaw_offset_deg,
                )
                data.qpos[qpos_adr : qpos_adr + 3] = [
                    pos[0],
                    pos[1],
                    pos[2] + args.human_z_offset,
                ]
                data.qpos[qpos_adr + 3 : qpos_adr + 7] = quat
                data.qvel[qvel_adr : qvel_adr + 6] = 0.0
                mujoco.mj_forward(model, data)

            if args.focus_human:
                if qpos_adr is not None:
                    viewer.cam.lookat[:] = data.qpos[qpos_adr : qpos_adr + 3]
                else:
                    viewer.cam.lookat[:] = data.xpos[human_body_id]

            mujoco.mj_step(model, data)
            viewer.sync()

            next_sync_time += model.opt.timestep
            sleep_time = next_sync_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-source", default="ithor", help="Scene source, e.g. ithor")
    parser.add_argument("--scene-split", default="train", help="Scene split, e.g. train")
    parser.add_argument("--scene-index", type=int, default=1, help="Scene index")
    parser.add_argument(
        "--scene-xml",
        type=Path,
        default=None,
        help="Direct scene XML path (overrides --scene-source/--scene-split/--scene-index)",
    )
    parser.add_argument(
        "--layout-json",
        type=Path,
        default=None,
        help=(
            "Optional scene layout JSON (from scene_position_tool.py). "
            "If it contains robot_runtime/human_runtime, robot and human config are loaded from there."
        ),
    )
    parser.add_argument(
        "--robot-pos",
        type=parse_vec3,
        default=[0.0, -0.15, 0.0],
        help='Robot base position "x,y,z"',
    )
    parser.add_argument(
        "--robot-type",
        choices=["franka", "rby1", "rby1m", "navbot"],
        default="franka",
        help="Robot type. Use rby1/rby1m/navbot for mobile-base social navigation.",
    )
    parser.add_argument(
        "--robot-base",
        type=parse_vec3,
        default=[1.5, 0.0, 0.0],
        help='Mobile-base pose as "x,y,theta(rad)"; used when --robot-type is rby1/rby1m/navbot',
    )
    parser.add_argument(
        "--human-pos",
        type=parse_vec3,
        default=[1.5, 0.0, 0.0],
        help='Human object position "x,y,z"',
    )
    parser.add_argument("--human-yaw-deg", type=float, default=180.0, help="Human yaw in degrees")
    parser.add_argument(
        "--human-roll-deg",
        type=float,
        default=90.0,
        help="Human roll correction in degrees (default fixes y-up assets)",
    )
    parser.add_argument(
        "--human-pitch-deg",
        type=float,
        default=0.0,
        help="Human pitch correction in degrees",
    )
    parser.add_argument(
        "--human-yaw-offset-deg",
        type=float,
        default=0.0,
        help="Extra yaw correction added to trajectory yaw",
    )
    parser.add_argument(
        "--human-z-offset",
        type=float,
        default=0.0,
        help="Vertical offset added to human base (meters); use near 0 for normalized human assets",
    )
    parser.add_argument(
        "--human-collider-type",
        choices=["none", "capsule", "box"],
        default="none",
        help="Optional simplified collider for static human to block robot traversal",
    )
    parser.add_argument(
        "--human-collider-size",
        type=parse_vec3,
        default=[0.28, 1.65, 0.0],
        help='Collider size. capsule=[radius,height,z_start], box=[half_x,half_y,half_z]',
    )
    parser.add_argument(
        "--human-static",
        action="store_true",
        help="Keep human static (no freejoint added, no per-frame human motion updates)",
    )
    parser.add_argument("--human-uid", default=None, help="Objaverse UID for the human object")
    parser.add_argument(
        "--human-xml",
        type=Path,
        default=None,
        help="Path to local human MJCF/XML file (overrides --human-uid)",
    )
    parser.add_argument(
        "--human-state-xmls",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "Multiple human state XMLs (same character, different poses). "
            "When provided, script toggles visibility over time instead of skeletal animation."
        ),
    )
    parser.add_argument(
        "--human-state-times",
        type=parse_non_negative_float,
        nargs="+",
        default=None,
        help=(
            "State start times in seconds, one per --human-state-xmls. "
            "Example: --human-state-times 0 1.5 3.0"
        ),
    )
    parser.add_argument(
        "--human-state-loop",
        action="store_true",
        help="Loop multi-state visibility timeline",
    )
    parser.add_argument(
        "--human-state-hold-sec",
        type=parse_non_negative_float,
        default=2.0,
        help=(
            "Default spacing between states when --human-state-times is omitted, "
            "and tail hold duration when --human-state-loop is enabled."
        ),
    )
    parser.add_argument(
        "--human-trajectory-file",
        type=Path,
        default=None,
        help="Path to trajectory JSON with waypoints (t, pos, yaw_deg)",
    )
    parser.add_argument(
        "--trajectory-loop",
        action="store_true",
        help="Loop trajectory playback (overrides trajectory file loop=false)",
    )
    parser.add_argument(
        "--save-xml",
        type=Path,
        default=None,
        help="Optional output path for combined scene XML",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Build and compile only, without launching MuJoCo viewer",
    )
    parser.add_argument(
        "--focus-human",
        action="store_true",
        help="Auto focus viewer camera on human root",
    )
    parser.add_argument(
        "--camera-distance",
        type=float,
        default=2.2,
        help="Viewer camera distance when --focus-human is enabled",
    )
    parser.add_argument(
        "--camera-azimuth",
        type=float,
        default=150.0,
        help="Viewer camera azimuth in degrees when --focus-human is enabled",
    )
    parser.add_argument(
        "--camera-elevation",
        type=float,
        default=-15.0,
        help="Viewer camera elevation in degrees when --focus-human is enabled",
    )
    parser.add_argument(
        "--tune-human",
        action="store_true",
        help="Enable keyboard tuning in viewer to adjust human position/yaw interactively",
    )
    parser.add_argument(
        "--tune-step-xy",
        type=float,
        default=0.05,
        help="Step size (meters) for W/A/S/D tuning",
    )
    parser.add_argument(
        "--tune-step-z",
        type=float,
        default=0.02,
        help="Step size (meters) for R/F vertical tuning",
    )
    parser.add_argument(
        "--tune-step-yaw-deg",
        type=float,
        default=5.0,
        help="Step size (degrees) for J/L yaw tuning",
    )
    args = parser.parse_args()
    args.extra_humans = []

    if args.layout_json is not None:
        payload = load_layout_json(args.layout_json)
        loaded_blocks: list[str] = []
        if apply_robot_runtime_from_layout(args, payload):
            loaded_blocks.append("robot_runtime")
        if apply_human_runtime_from_layout(args, payload):
            loaded_blocks.append("human_runtime")
        if loaded_blocks:
            print(
                "[INFO] Loaded layout runtime overrides from JSON: "
                f"{args.layout_json} blocks={loaded_blocks}"
            )

    if args.robot_type in {"rby1", "rby1m", "navbot"} and any_cli_flag_provided(["--robot-pos"]):
        print("[WARN] --robot-pos is ignored for rby1/rby1m/navbot. Use --robot-base x,y,theta.")

    if args.human_state_xmls is not None:
        if args.human_uid is not None or args.human_xml is not None:
            print("[WARN] --human-state-xmls is set; --human-uid/--human-xml are ignored.")
        if not args.human_static:
            raise ValueError("--human-state-xmls requires --human-static.")
        if args.human_trajectory_file is not None:
            raise ValueError("--human-state-xmls cannot be combined with --human-trajectory-file.")

    state_start_times: list[float] | None = None
    if args.human_state_xmls is not None:
        state_start_times = build_state_start_times(
            num_states=len(args.human_state_xmls),
            input_times=args.human_state_times,
            default_step=args.human_state_hold_sec,
        )
        print(
            "[INFO] Multi-state human timeline: "
            f"states={len(args.human_state_xmls)} start_times={state_start_times} loop={args.human_state_loop}"
        )
    elif args.human_state_times is not None:
        raise ValueError("--human-state-times requires --human-state-xmls.")

    model, data, human_source, human_prefixes = build_scene(args)

    if args.save_xml is not None:
        import mujoco

        args.save_xml.parent.mkdir(parents=True, exist_ok=True)
        xml = mujoco.mj_saveLastXML(str(args.save_xml), model)
        print(f"[INFO] Saved combined scene XML to {args.save_xml} (status={xml})")

    if args.no_viewer:
        print("[INFO] Scene compiled successfully. Viewer launch skipped.")
        return

    waypoints = None
    loop_trajectory = args.trajectory_loop
    if args.human_trajectory_file is not None:
        waypoints, file_loop = load_human_trajectory(args.human_trajectory_file)
        loop_trajectory = loop_trajectory or file_loop
        print(
            f"[INFO] Loaded trajectory from {args.human_trajectory_file} for human source={human_source}"
        )

    print("[INFO] Launching viewer...")
    try:
        run_interactive_viewer(
            model,
            data,
            args,
            waypoints,
            loop_trajectory,
            human_prefixes=human_prefixes,
            state_start_times=state_start_times,
        )
    except RuntimeError as e:
        if "launch_passive" in str(e) and "mjpython" in str(e):
            raise RuntimeError(
                "On macOS, trajectory mode requires launching with mjpython.\n"
                "Run:\n"
                "  ./.venv/bin/mjpython scripts/basic_robot_human_scene.py "
                "--human-trajectory-file scripts/human_trajectory_example.json"
            ) from e
        raise


if __name__ == "__main__":
    main()
