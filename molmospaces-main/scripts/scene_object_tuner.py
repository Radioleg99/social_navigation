#!/usr/bin/env python3
"""Interactive scene object tuner with JSON override saving.

Hotkeys:
  V: select next object
  X: select previous object
  W/S: +Y / -Y
  A/D: -X / +X
  R/F: +Z / -Z
  J/L: yaw - / +
  P: print current selected pose
  K: save current edits into layout JSON overrides
  H: print help
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import time
from pathlib import Path
from typing import Any

import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R

# GLFW-style keycodes for non-letter fallback keys.
KEY_LEFT = 263
KEY_RIGHT = 262
KEY_UP = 265
KEY_DOWN = 264
KEY_PAGE_UP = 266
KEY_PAGE_DOWN = 267


def root_body_names(model: mujoco.MjModel) -> list[str]:
    names: list[str] = []
    for bid in range(1, model.nbody):
        if int(model.body_parentid[bid]) != 0:
            continue
        name = model.body(bid).name
        if name:
            names.append(name)
    names.sort()
    return names


def body_name_to_id(model: mujoco.MjModel) -> dict[str, int]:
    out: dict[str, int] = {}
    for bid in range(model.nbody):
        name = model.body(bid).name
        if name:
            out[name] = bid
    return out


def freejoint_by_body(model: mujoco.MjModel) -> dict[int, tuple[int, int]]:
    out: dict[int, tuple[int, int]] = {}
    for j in range(model.njnt):
        if int(model.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        bid = int(model.jnt_bodyid[j])
        qpos_adr = int(model.jnt_qposadr[j])
        qvel_adr = int(model.jnt_dofadr[j])
        out[bid] = (qpos_adr, qvel_adr)
    return out


def top_root_by_body(model: mujoco.MjModel) -> list[int]:
    top = [0 for _ in range(model.nbody)]
    for bid in range(1, model.nbody):
        cur = bid
        parent = int(model.body_parentid[cur])
        while parent != 0:
            cur = parent
            parent = int(model.body_parentid[cur])
        top[bid] = cur
    return top


def quat_wxyz_to_yaw_deg(quat_wxyz: list[float]) -> float:
    quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    yaw_deg = float(R.from_quat(quat_xyzw).as_euler("xyz", degrees=True)[2])
    return yaw_deg


def yaw_deg_to_quat_wxyz(yaw_deg: float) -> list[float]:
    quat_xyzw = R.from_euler("z", yaw_deg, degrees=True).as_quat()
    return [float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2])]


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
    model: mujoco.MjModel,
    state_geom_ids: list[list[int]],
    base_alpha: list[float],
    active_idx: int,
) -> None:
    for i, geom_ids in enumerate(state_geom_ids):
        for gid in geom_ids:
            model.geom_rgba[gid, 3] = base_alpha[gid] if i == active_idx else 0.0


def load_layout_json(layout_json: Path | None) -> dict[str, Any] | None:
    if layout_json is None:
        return None
    if not layout_json.is_file():
        raise FileNotFoundError(f"--layout-json not found: {layout_json}")
    with open(layout_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("--layout-json must contain a JSON object")
    return payload


def key_matches(keycode: int, ch: str) -> bool:
    return keycode in (ord(ch.lower()), ord(ch.upper()))


def layout_human_enabled(layout_payload: dict[str, Any] | None) -> bool:
    if layout_payload is None:
        return False
    runtime = layout_payload.get("human_runtime")
    if not isinstance(runtime, dict):
        return False
    return bool(runtime.get("enabled", False))


def load_basic_scene_module():
    module_path = Path(__file__).with_name("basic_robot_human_scene.py")
    spec = importlib.util.spec_from_file_location("basic_robot_human_scene", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_model_with_human(scene_xml: Path, layout_json: Path):
    bhs = load_basic_scene_module()
    args = argparse.Namespace(
        scene_source="ithor",
        scene_split="train",
        scene_index=1,
        scene_xml=scene_xml,
        layout_json=layout_json,
        robot_pos=[0.0, -0.15, 0.0],
        human_pos=[1.5, 0.0, 0.0],
        human_yaw_deg=180.0,
        human_roll_deg=90.0,
        human_pitch_deg=0.0,
        human_yaw_offset_deg=0.0,
        human_z_offset=0.0,
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
    )
    payload = bhs.load_layout_json(layout_json)
    bhs.apply_human_runtime_from_layout(args, payload)

    if args.human_state_xmls is not None:
        if not args.human_static:
            raise ValueError("--include-human path requires static human when using multi-state xmls.")
    model, data, _, _ = bhs.build_scene(args)
    return model, data


def pick_tunable_names(
    model: mujoco.MjModel,
    layout_payload: dict[str, Any] | None,
) -> list[str]:
    bid_map = body_name_to_id(model)
    names: list[str] = []

    if layout_payload is not None:
        objects = layout_payload.get("objects")
        if isinstance(objects, list):
            seen: set[str] = set()
            for entry in objects:
                if not isinstance(entry, dict):
                    continue
                name = entry.get("body")
                if not isinstance(name, str):
                    continue
                if name in bid_map and name not in seen:
                    names.append(name)
                    seen.add(name)
            names.sort()
            if len(names) > 0:
                return names

    return root_body_names(model)


def apply_layout_overrides(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    layout_payload: dict[str, Any] | None,
) -> None:
    if layout_payload is None:
        return
    overrides = layout_payload.get("overrides", {})
    if not isinstance(overrides, dict):
        return

    bid_map = body_name_to_id(model)
    free_map = freejoint_by_body(model)
    for name, spec in overrides.items():
        if name not in bid_map or not isinstance(spec, dict):
            continue
        bid = bid_map[name]
        pos = spec.get("position_xyz")
        quat = spec.get("quat_wxyz")
        if bid in free_map:
            qpos_adr, qvel_adr = free_map[bid]
            if isinstance(pos, list) and len(pos) == 3:
                data.qpos[qpos_adr : qpos_adr + 3] = [float(pos[0]), float(pos[1]), float(pos[2])]
            if isinstance(quat, list) and len(quat) == 4:
                data.qpos[qpos_adr + 3 : qpos_adr + 7] = [
                    float(quat[0]),
                    float(quat[1]),
                    float(quat[2]),
                    float(quat[3]),
                ]
            data.qvel[qvel_adr : qvel_adr + 6] = 0.0
            if isinstance(pos, list) and len(pos) == 3:
                model.qpos0[qpos_adr : qpos_adr + 3] = [float(pos[0]), float(pos[1]), float(pos[2])]
            if isinstance(quat, list) and len(quat) == 4:
                model.qpos0[qpos_adr + 3 : qpos_adr + 7] = [
                    float(quat[0]),
                    float(quat[1]),
                    float(quat[2]),
                    float(quat[3]),
                ]
        else:
            if isinstance(pos, list) and len(pos) == 3:
                model.body_pos[bid, 0] = float(pos[0])
                model.body_pos[bid, 1] = float(pos[1])
                model.body_pos[bid, 2] = float(pos[2])
            if isinstance(quat, list) and len(quat) == 4:
                model.body_quat[bid, 0] = float(quat[0])
                model.body_quat[bid, 1] = float(quat[1])
                model.body_quat[bid, 2] = float(quat[2])
                model.body_quat[bid, 3] = float(quat[3])

    mujoco.mj_forward(model, data)


def ensure_layout_base(scene_xml: Path, payload: dict[str, Any] | None) -> dict[str, Any]:
    if payload is not None:
        if "overrides" not in payload or not isinstance(payload.get("overrides"), dict):
            payload["overrides"] = {}
        return payload
    return {
        "scene_xml": str(scene_xml),
        "coordinate_frame": "mujoco_world",
        "notes": "Generated by scene_object_tuner.py; edited entries are in overrides.",
        "objects": [],
        "overrides": {},
    }


def print_help(step_xy: float, step_z: float, step_yaw_deg: float) -> None:
    print("[TUNE] Hotkeys:")
    print("  Next/Prev object: V or ] or RightArrow / X or [ or LeftArrow")
    print(
        f"  Move XY: W/S or Up/Down ({step_xy:.3f}m), A/D ({step_xy:.3f}m)"
    )
    print(f"  Move Z: R/F or PgUp/PgDn ({step_z:.3f}m)")
    print(f"  J/L: yaw -/+ ({step_yaw_deg:.1f} deg)")
    print("  P: print current pose, K: save JSON overrides, H: help")
    print("  Tip: click viewer window first so keyboard events are captured.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-xml", type=Path, required=True, help="Scene XML path")
    parser.add_argument(
        "--layout-json",
        type=Path,
        default=None,
        help="Layout JSON path for reading/saving overrides",
    )
    parser.add_argument(
        "--include-human",
        action="store_true",
        help=(
            "Force loading human runtime from layout JSON before launching tuner. "
            "Without this flag, tuner also auto-loads human when human_runtime.enabled=true."
        ),
    )
    parser.add_argument("--step-xy", type=float, default=0.05, help="Move step for X/Y (meters)")
    parser.add_argument("--step-z", type=float, default=0.02, help="Move step for Z (meters)")
    parser.add_argument("--step-yaw-deg", type=float, default=5.0, help="Yaw step (degrees)")
    parser.add_argument(
        "--camera-distance",
        type=float,
        default=2.2,
        help="Camera distance while auto-focusing selected object",
    )
    parser.add_argument(
        "--no-focus-selected",
        action="store_true",
        help="Disable auto-focus camera on selected object",
    )
    parser.add_argument("--start-index", type=int, default=0, help="Start selected object index")
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Load scene and print object count only (debug)",
    )
    args = parser.parse_args()

    if not args.scene_xml.is_file():
        raise FileNotFoundError(f"--scene-xml not found: {args.scene_xml}")

    layout_payload = load_layout_json(args.layout_json)
    include_human = args.include_human or layout_human_enabled(layout_payload)
    if include_human:
        if args.layout_json is None:
            raise ValueError("Including human requires --layout-json")
        model, data = build_model_with_human(args.scene_xml, args.layout_json)
        print("[INFO] Loaded scene with human runtime from layout JSON.")
    else:
        model = mujoco.MjModel.from_xml_path(str(args.scene_xml))
        data = mujoco.MjData(model)
        apply_layout_overrides(model, data, layout_payload)

    bid_map = body_name_to_id(model)
    free_map = freejoint_by_body(model)
    top_root = top_root_by_body(model)
    object_names = pick_tunable_names(model, layout_payload)
    if len(object_names) == 0:
        raise RuntimeError("No tunable objects found in scene.")

    print(f"[INFO] Tunable objects: {len(object_names)}")
    if args.no_viewer:
        print(f"[INFO] First object: {object_names[0]}")
        return

    selected = int(args.start_index) % len(object_names)
    changed: dict[str, dict[str, list[float]]] = {}
    mutable_layout = ensure_layout_base(args.scene_xml, layout_payload)
    base_rgba = [float(model.geom_rgba[g, c]) for g in range(model.ngeom) for c in range(4)]
    base_alpha = [float(model.geom_rgba[g, 3]) for g in range(model.ngeom)]

    root_geoms: dict[int, list[int]] = {}
    for gid in range(model.ngeom):
        gbody = int(model.geom_bodyid[gid])
        root_bid = top_root[gbody]
        root_geoms.setdefault(root_bid, []).append(gid)

    highlighted_root: int | None = None

    # Optional human multi-state visibility timeline (for human_runtime poses/state xmls).
    human_state_geom_ids: list[list[int]] = []
    human_state_start_times: list[float] = []
    human_state_loop = False
    human_state_hold_sec = 2.0
    active_human_state_idx = -1
    runtime = layout_payload.get("human_runtime") if isinstance(layout_payload, dict) else None
    if isinstance(runtime, dict) and bool(runtime.get("enabled", False)):
        pose_count = 0
        if isinstance(runtime.get("poses"), list):
            pose_count = len(runtime["poses"])
        elif isinstance(runtime.get("human_state_xmls"), list):
            pose_count = len(runtime["human_state_xmls"])

        if pose_count > 1:
            if isinstance(runtime.get("human_state_times"), list) and len(runtime["human_state_times"]) == pose_count:
                human_state_start_times = [float(x) for x in runtime["human_state_times"]]
                human_state_loop = bool(runtime.get("human_state_loop", runtime.get("loop", False)))
                human_state_hold_sec = float(runtime.get("human_state_hold_sec", 2.0))
            else:
                interval = float(runtime.get("pose_interval_sec", 2.0))
                human_state_start_times = [float(i) * interval for i in range(pose_count)]
                human_state_loop = bool(runtime.get("loop", True))
                human_state_hold_sec = float(runtime.get("human_state_hold_sec", interval))

            prefix_to_geom: dict[int, list[int]] = {i: [] for i in range(pose_count)}
            pat = re.compile(r"^human_state_(\d+)/")
            for gid in range(model.ngeom):
                gname = model.geom(gid).name
                m = pat.match(gname)
                if m is None:
                    continue
                idx = int(m.group(1))
                if idx in prefix_to_geom:
                    prefix_to_geom[idx].append(gid)
            if all(len(prefix_to_geom[i]) > 0 for i in range(pose_count)):
                human_state_geom_ids = [prefix_to_geom[i] for i in range(pose_count)]
                init_idx = state_index_for_time(
                    0.0, human_state_start_times, human_state_loop, human_state_hold_sec
                )
                apply_visible_state(model, human_state_geom_ids, base_alpha, init_idx)
                active_human_state_idx = init_idx
                mujoco.mj_forward(model, data)

    def current_pose(name: str) -> tuple[list[float], list[float], float]:
        bid = bid_map[name]
        if bid in free_map:
            qpos_adr, _ = free_map[bid]
            pos = [float(data.qpos[qpos_adr + i]) for i in range(3)]
            quat = [float(data.qpos[qpos_adr + 3 + i]) for i in range(4)]
        else:
            pos = [float(model.body_pos[bid, 0]), float(model.body_pos[bid, 1]), float(model.body_pos[bid, 2])]
            quat = [
                float(model.body_quat[bid, 0]),
                float(model.body_quat[bid, 1]),
                float(model.body_quat[bid, 2]),
                float(model.body_quat[bid, 3]),
            ]
        yaw = quat_wxyz_to_yaw_deg(quat)
        return pos, quat, yaw

    def set_pose(name: str, pos: list[float], quat: list[float]) -> None:
        bid = bid_map[name]
        if bid in free_map:
            qpos_adr, qvel_adr = free_map[bid]
            data.qpos[qpos_adr : qpos_adr + 3] = [float(pos[0]), float(pos[1]), float(pos[2])]
            data.qpos[qpos_adr + 3 : qpos_adr + 7] = [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]
            data.qvel[qvel_adr : qvel_adr + 6] = 0.0
            model.qpos0[qpos_adr : qpos_adr + 3] = [float(pos[0]), float(pos[1]), float(pos[2])]
            model.qpos0[qpos_adr + 3 : qpos_adr + 7] = [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]
        else:
            model.body_pos[bid, 0] = float(pos[0])
            model.body_pos[bid, 1] = float(pos[1])
            model.body_pos[bid, 2] = float(pos[2])
            model.body_quat[bid, 0] = float(quat[0])
            model.body_quat[bid, 1] = float(quat[1])
            model.body_quat[bid, 2] = float(quat[2])
            model.body_quat[bid, 3] = float(quat[3])
        changed[name] = {"position_xyz": pos, "quat_wxyz": quat}
        mujoco.mj_forward(model, data)

    def update_highlight(new_root: int) -> None:
        nonlocal highlighted_root
        if highlighted_root is not None:
            for gid in root_geoms.get(highlighted_root, []):
                i = gid * 4
                model.geom_rgba[gid, 0] = base_rgba[i + 0]
                model.geom_rgba[gid, 1] = base_rgba[i + 1]
                model.geom_rgba[gid, 2] = base_rgba[i + 2]
                model.geom_rgba[gid, 3] = base_rgba[i + 3]
        for gid in root_geoms.get(new_root, []):
            model.geom_rgba[gid, 0] = 1.0
            model.geom_rgba[gid, 1] = 1.0
            model.geom_rgba[gid, 2] = 0.2
            model.geom_rgba[gid, 3] = max(0.9, float(model.geom_rgba[gid, 3]))
        highlighted_root = new_root

    def print_selected() -> None:
        name = object_names[selected]
        bid = bid_map[name]
        pos, _, yaw = current_pose(name)
        joint_type = "freejoint" if bid in free_map else "root-static"
        print(
            f"[SEL] {selected + 1}/{len(object_names)} name={name} "
            f"type={joint_type} pos=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}) yaw_deg={yaw:.2f}"
        )

    def save_changes() -> None:
        if args.layout_json is None:
            print("[WARN] No --layout-json provided; cannot save overrides.")
            return
        overrides = mutable_layout.get("overrides")
        if not isinstance(overrides, dict):
            mutable_layout["overrides"] = {}
            overrides = mutable_layout["overrides"]
        for name, spec in changed.items():
            overrides[name] = spec
        args.layout_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.layout_json, "w", encoding="utf-8") as f:
            json.dump(mutable_layout, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"[SAVE] wrote {len(changed)} edited objects to {args.layout_json}")

    def move_selected(dx: float, dy: float, dz: float, dyaw: float) -> None:
        name = object_names[selected]
        pos, _, yaw = current_pose(name)
        pos = [pos[0] + dx, pos[1] + dy, pos[2] + dz]
        yaw = yaw + dyaw
        quat = yaw_deg_to_quat_wxyz(yaw)
        set_pose(name, pos, quat)
        print_selected()

    def key_callback_impl(keycode: int) -> None:
        nonlocal selected

        if key_matches(keycode, "V") or keycode in (ord("]"), KEY_RIGHT):
            selected = (selected + 1) % len(object_names)
            update_highlight(bid_map[object_names[selected]])
            print_selected()
            return
        if key_matches(keycode, "X") or keycode in (ord("["), KEY_LEFT):
            selected = (selected - 1) % len(object_names)
            update_highlight(bid_map[object_names[selected]])
            print_selected()
            return
        if key_matches(keycode, "P"):
            print_selected()
            return
        if key_matches(keycode, "H"):
            print_help(args.step_xy, args.step_z, args.step_yaw_deg)
            return
        if key_matches(keycode, "K"):
            save_changes()
            return

        if key_matches(keycode, "W") or keycode == KEY_UP:
            move_selected(0.0, args.step_xy, 0.0, 0.0)
        elif key_matches(keycode, "S") or keycode == KEY_DOWN:
            move_selected(0.0, -args.step_xy, 0.0, 0.0)
        elif key_matches(keycode, "A"):
            move_selected(-args.step_xy, 0.0, 0.0, 0.0)
        elif key_matches(keycode, "D"):
            move_selected(args.step_xy, 0.0, 0.0, 0.0)
        elif key_matches(keycode, "R") or keycode == KEY_PAGE_UP:
            move_selected(0.0, 0.0, args.step_z, 0.0)
        elif key_matches(keycode, "F") or keycode == KEY_PAGE_DOWN:
            move_selected(0.0, 0.0, -args.step_z, 0.0)
        elif key_matches(keycode, "J"):
            move_selected(0.0, 0.0, 0.0, -args.step_yaw_deg)
        elif key_matches(keycode, "L"):
            move_selected(0.0, 0.0, 0.0, args.step_yaw_deg)

    def key_callback(keycode: int) -> None:
        try:
            key_callback_impl(keycode)
        except Exception as exc:
            print(f"[ERROR] key callback failed: keycode={keycode} err={exc}")

    print_help(args.step_xy, args.step_z, args.step_yaw_deg)
    update_highlight(bid_map[object_names[selected]])
    print_selected()
    print("[INFO] Press K to write JSON overrides.")
    try:
        with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
            viewer.cam.distance = float(args.camera_distance)
            wall_start = time.perf_counter()
            while viewer.is_running():
                try:
                    if len(human_state_geom_ids) > 1:
                        elapsed = time.perf_counter() - wall_start
                        idx = state_index_for_time(
                            elapsed,
                            human_state_start_times,
                            human_state_loop,
                            human_state_hold_sec,
                        )
                        if idx != active_human_state_idx:
                            apply_visible_state(model, human_state_geom_ids, base_alpha, idx)
                            active_human_state_idx = idx
                            mujoco.mj_forward(model, data)
                    if not args.no_focus_selected:
                        active_name = object_names[selected]
                        active_bid = bid_map[active_name]
                        viewer.cam.lookat[:] = data.xpos[active_bid]
                    viewer.sync()
                    time.sleep(0.01)
                except Exception as exc:
                    print(f"[ERROR] viewer loop failed: {exc}")
                    time.sleep(0.05)
    except RuntimeError as exc:
        msg = str(exc)
        if "launch_passive" in msg and "mjpython" in msg:
            raise RuntimeError(
                "On macOS, run with mjpython:\n"
                "  ./.venv/bin/mjpython scripts/scene_object_tuner.py "
                "--scene-xml assets/scenes/ithor/FloorPlan1_physics.xml "
                "--layout-json assets/layouts/FloorPlan1_object_positions.json"
            ) from exc
        raise


if __name__ == "__main__":
    main()
