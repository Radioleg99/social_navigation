#!/usr/bin/env python3
"""Interactive human-only tuner for layout JSON.

Ultra-minimal hotkeys:
  V: next human
  Arrow keys: move selected human in XY plane
  P: save to JSON
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R

KEY_LEFT = 263
KEY_RIGHT = 262
KEY_UP = 265
KEY_DOWN = 264


def key_matches(keycode: int, ch: str) -> bool:
    return keycode in (ord(ch.lower()), ord(ch.upper()))


def load_basic_scene_module():
    module_path = Path(__file__).with_name("basic_robot_human_scene.py")
    spec = importlib.util.spec_from_file_location("basic_robot_human_scene", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_layout_json(layout_json: Path) -> dict[str, Any]:
    if not layout_json.is_file():
        raise FileNotFoundError(f"--layout-json not found: {layout_json}")
    with open(layout_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("--layout-json must be a JSON object")
    return payload


def save_layout_json(layout_json: Path, payload: dict[str, Any]) -> None:
    layout_json.parent.mkdir(parents=True, exist_ok=True)
    with open(layout_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def euler_xyz_deg_to_quat_wxyz(roll_deg: float, pitch_deg: float, yaw_deg: float) -> list[float]:
    quat_xyzw = R.from_euler("xyz", [roll_deg, pitch_deg, yaw_deg], degrees=True).as_quat()
    return [float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2])]


def quat_wxyz_to_yaw_deg(quat_wxyz: list[float]) -> float:
    quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    return float(R.from_quat(quat_xyzw).as_euler("xyz", degrees=True)[2])


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
        out[bid] = (int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j]))
    return out


@dataclass
class HumanActor:
    kind: str  # "primary" | "extra"
    index: int
    label: str
    prefixes: list[str]
    body_ids: list[int]
    geom_ids: list[int]
    pos_world: list[float]
    yaw_deg: float
    roll_deg: float
    pitch_deg: float
    yaw_offset_deg: float
    z_offset: float


def make_default_build_args(scene_xml: Path, layout_json: Path) -> argparse.Namespace:
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


def build_scene_with_layout(scene_xml: Path, layout_json: Path):
    bhs = load_basic_scene_module()
    args = make_default_build_args(scene_xml, layout_json)
    payload = bhs.load_layout_json(layout_json)
    bhs.apply_robot_runtime_from_layout(args, payload)
    bhs.apply_human_runtime_from_layout(args, payload)
    model, data, _, _ = bhs.build_scene(args)
    return model, data, payload


def collect_human_actors(model: mujoco.MjModel, data: mujoco.MjData, payload: dict[str, Any]) -> list[HumanActor]:
    runtime = payload.get("human_runtime")
    if not isinstance(runtime, dict) or not bool(runtime.get("enabled", False)):
        return []

    bid_map = body_name_to_id(model)
    free_map = freejoint_by_body(model)

    def gather(prefixes: list[str]) -> tuple[list[int], list[int]]:
        body_ids: list[int] = []
        geom_ids: list[int] = []
        for bid in range(model.nbody):
            name = model.body(bid).name
            if any(name.startswith(p) for p in prefixes):
                body_ids.append(bid)
        for gid in range(model.ngeom):
            gname = model.geom(gid).name
            if "/blocker" in gname:
                continue
            if any(gname.startswith(p) for p in prefixes):
                geom_ids.append(gid)
        return body_ids, geom_ids

    actors: list[HumanActor] = []

    # Primary human (single xml or multi-state xmls)
    num_states = 0
    poses = runtime.get("poses")
    state_xmls = runtime.get("human_state_xmls")
    if isinstance(poses, list):
        num_states = len(poses)
    elif isinstance(state_xmls, list):
        num_states = len(state_xmls)
    if num_states > 1:
        prefixes = [f"human_state_{i}/" for i in range(num_states)]
    else:
        prefixes = ["human_0/"]
    body_ids, geom_ids = gather(prefixes)
    if len(body_ids) > 0:
        lead_bid = body_ids[0]
        if lead_bid in free_map:
            qpos_adr, _ = free_map[lead_bid]
            pos_world = [float(data.qpos[qpos_adr + i]) for i in range(3)]
            quat = [float(data.qpos[qpos_adr + 3 + i]) for i in range(4)]
            yaw_deg = quat_wxyz_to_yaw_deg(quat) - float(runtime.get("human_yaw_offset_deg", 0.0))
        else:
            pos_world = [float(model.body_pos[lead_bid, i]) for i in range(3)]
            yaw_deg = float(runtime.get("human_yaw_deg", 180.0))
        actors.append(
            HumanActor(
                kind="primary",
                index=0,
                label="primary_human",
                prefixes=prefixes,
                body_ids=body_ids,
                geom_ids=geom_ids,
                pos_world=pos_world,
                yaw_deg=yaw_deg,
                roll_deg=float(runtime.get("human_roll_deg", 90.0)),
                pitch_deg=float(runtime.get("human_pitch_deg", 0.0)),
                yaw_offset_deg=float(runtime.get("human_yaw_offset_deg", 0.0)),
                z_offset=float(runtime.get("human_z_offset", 0.0)),
            )
        )

    # Extra humans
    extras = runtime.get("extra_humans")
    if isinstance(extras, list):
        for i, item in enumerate(extras):
            if not isinstance(item, dict) or not bool(item.get("enabled", True)):
                continue
            prefixes = [f"human_extra_{i}/"]
            body_ids, geom_ids = gather(prefixes)
            if len(body_ids) == 0:
                continue
            lead_bid = body_ids[0]
            if lead_bid in free_map:
                qpos_adr, _ = free_map[lead_bid]
                pos_world = [float(data.qpos[qpos_adr + j]) for j in range(3)]
                quat = [float(data.qpos[qpos_adr + 3 + j]) for j in range(4)]
                yaw_deg = quat_wxyz_to_yaw_deg(quat) - float(item.get("human_yaw_offset_deg", 0.0))
            else:
                pos_world = [float(model.body_pos[lead_bid, j]) for j in range(3)]
                yaw_deg = float(item.get("human_yaw_deg", 180.0))
            actors.append(
                HumanActor(
                    kind="extra",
                    index=i,
                    label=f"extra_human_{i}",
                    prefixes=prefixes,
                    body_ids=body_ids,
                    geom_ids=geom_ids,
                    pos_world=pos_world,
                    yaw_deg=yaw_deg,
                    roll_deg=float(item.get("human_roll_deg", 90.0)),
                    pitch_deg=float(item.get("human_pitch_deg", 0.0)),
                    yaw_offset_deg=float(item.get("human_yaw_offset_deg", 0.0)),
                    z_offset=float(item.get("human_z_offset", 0.0)),
                )
            )

    return actors


def enforce_single_primary_state_visibility(
    model: mujoco.MjModel,
    primary_actor: HumanActor | None,
    requested_state_idx: int,
) -> tuple[int, int] | None:
    if primary_actor is None:
        return None
    if len(primary_actor.prefixes) <= 1:
        return None
    if requested_state_idx < 0:
        raise ValueError("--primary-state-index must be >= 0")

    num_states = len(primary_actor.prefixes)
    active_idx = min(requested_state_idx, num_states - 1)
    active_prefix = primary_actor.prefixes[active_idx]

    # Primary multi-state human is represented by parallel meshes:
    #   human_state_0/, human_state_1/, ...
    # Keep only one visible while tuning to avoid overlap confusion.
    visible_geom_ids: list[int] = []
    for gid in range(model.ngeom):
        gname = model.geom(gid).name
        if not any(gname.startswith(pfx) for pfx in primary_actor.prefixes):
            continue
        if gname.startswith(active_prefix):
            visible_geom_ids.append(gid)
        else:
            model.geom_rgba[gid, 3] = 0.0

    primary_actor.geom_ids = visible_geom_ids
    return active_idx, num_states


def apply_actor_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    actor: HumanActor,
    free_map: dict[int, tuple[int, int]],
) -> None:
    quat = euler_xyz_deg_to_quat_wxyz(
        actor.roll_deg, actor.pitch_deg, actor.yaw_deg + actor.yaw_offset_deg
    )
    for bid in actor.body_ids:
        if bid in free_map:
            qpos_adr, qvel_adr = free_map[bid]
            data.qpos[qpos_adr : qpos_adr + 3] = actor.pos_world
            data.qpos[qpos_adr + 3 : qpos_adr + 7] = quat
            data.qvel[qvel_adr : qvel_adr + 6] = 0.0
            model.qpos0[qpos_adr : qpos_adr + 3] = actor.pos_world
            model.qpos0[qpos_adr + 3 : qpos_adr + 7] = quat
        else:
            model.body_pos[bid, 0] = actor.pos_world[0]
            model.body_pos[bid, 1] = actor.pos_world[1]
            model.body_pos[bid, 2] = actor.pos_world[2]
            model.body_quat[bid, 0] = quat[0]
            model.body_quat[bid, 1] = quat[1]
            model.body_quat[bid, 2] = quat[2]
            model.body_quat[bid, 3] = quat[3]
    mujoco.mj_forward(model, data)


def print_help(step_xy: float) -> None:
    print("[HUMAN TUNE] Hotkeys:")
    print("  V: next human")
    print(f"  Arrow keys: move selected human (tap/hold, base step={step_xy:.3f}m)")
    print("  P: save to JSON")
    print("  Tip: click viewer window first.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-xml", type=Path, required=True, help="Scene XML path")
    parser.add_argument("--layout-json", type=Path, required=True, help="Layout JSON path")
    parser.add_argument("--step-xy", type=float, default=0.05, help="Move step XY")
    parser.add_argument("--camera-distance", type=float, default=2.2, help="Camera follow distance")
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Step physics while tuning (matches basic_robot_human_scene behavior)",
    )
    parser.add_argument("--no-viewer", action="store_true", help="Debug mode")
    parser.add_argument(
        "--primary-state-index",
        type=int,
        default=0,
        help=(
            "When primary human is multi-state, select which single mesh(state) is visible "
            "while tuning. Default 0 = first state."
        ),
    )
    args = parser.parse_args()

    model, data, payload = build_scene_with_layout(args.scene_xml, args.layout_json)
    free_map = freejoint_by_body(model)
    actors = collect_human_actors(model, data, payload)
    if len(actors) == 0:
        raise RuntimeError("No humans found from human_runtime in layout JSON.")

    primary_actor = next((a for a in actors if a.kind == "primary"), None)
    chosen_state = enforce_single_primary_state_visibility(model, primary_actor, args.primary_state_index)
    if chosen_state is not None:
        active_idx, total_states = chosen_state
        print(
            "[INFO] Multi-state primary human detected: "
            f"states={total_states}, tuning only state_index={active_idx}"
        )

    base_rgba = [float(model.geom_rgba[g, c]) for g in range(model.ngeom) for c in range(4)]
    selected = 0
    modified = False

    def set_highlight(active_idx: int) -> None:
        for g in range(model.ngeom):
            i = g * 4
            model.geom_rgba[g, 0] = base_rgba[i + 0]
            model.geom_rgba[g, 1] = base_rgba[i + 1]
            model.geom_rgba[g, 2] = base_rgba[i + 2]
            model.geom_rgba[g, 3] = base_rgba[i + 3]
        for gid in actors[active_idx].geom_ids:
            model.geom_rgba[gid, 0] = 1.0
            model.geom_rgba[gid, 1] = 1.0
            model.geom_rgba[gid, 2] = 0.2
            model.geom_rgba[gid, 3] = max(0.9, float(model.geom_rgba[gid, 3]))

    def print_selected() -> None:
        a = actors[selected]
        print(
            f"[HUMAN] {selected + 1}/{len(actors)} {a.label} "
            f"pos=({a.pos_world[0]:.3f},{a.pos_world[1]:.3f},{a.pos_world[2]:.3f}) yaw={a.yaw_deg:.1f}"
        )

    def save_humans() -> None:
        nonlocal modified
        runtime = payload.get("human_runtime")
        if not isinstance(runtime, dict):
            raise RuntimeError("layout JSON missing human_runtime")
        extras = runtime.get("extra_humans")
        if not isinstance(extras, list):
            extras = []
            runtime["extra_humans"] = extras
        for a in actors:
            out_pos = [a.pos_world[0], a.pos_world[1], a.pos_world[2] - a.z_offset]
            if a.kind == "primary":
                runtime["human_pos"] = out_pos
                runtime["human_yaw_deg"] = float(a.yaw_deg)
            else:
                while len(extras) <= a.index:
                    extras.append({})
                if not isinstance(extras[a.index], dict):
                    extras[a.index] = {}
                extras[a.index]["human_pos"] = out_pos
                extras[a.index]["human_yaw_deg"] = float(a.yaw_deg)
        save_layout_json(args.layout_json, payload)
        modified = False
        print(f"[SAVE] Human poses written to {args.layout_json}")

    def move_selected(dx: float, dy: float, *, verbose: bool = True) -> None:
        nonlocal modified
        a = actors[selected]
        a.pos_world[0] += dx
        a.pos_world[1] += dy
        apply_actor_pose(model=model, data=data, actor=a, free_map=free_map)
        modified = True
        if verbose:
            print_selected()

    # Hold-to-move emulation:
    # We keep applying movement for a short grace window after each arrow event.
    # On systems that emit key repeat while held, this becomes smooth continuous motion.
    hold_dir = [0.0, 0.0]
    hold_until = 0.0
    hold_grace_sec = 0.10
    hold_speed = float(args.step_xy) / hold_grace_sec

    def key_callback(keycode: int) -> None:
        nonlocal selected, hold_until
        try:
            if key_matches(keycode, "V"):
                selected = (selected + 1) % len(actors)
                set_highlight(selected)
                print_selected()
                return
            if key_matches(keycode, "P"):
                save_humans()
                return
            if keycode == KEY_UP:
                hold_dir[0], hold_dir[1] = 0.0, 1.0
                hold_until = time.perf_counter() + hold_grace_sec
                return
            if keycode == KEY_DOWN:
                hold_dir[0], hold_dir[1] = 0.0, -1.0
                hold_until = time.perf_counter() + hold_grace_sec
                return
            if keycode == KEY_LEFT:
                hold_dir[0], hold_dir[1] = -1.0, 0.0
                hold_until = time.perf_counter() + hold_grace_sec
                return
            if keycode == KEY_RIGHT:
                hold_dir[0], hold_dir[1] = 1.0, 0.0
                hold_until = time.perf_counter() + hold_grace_sec
                return
        except Exception as exc:
            print(f"[ERROR] key callback failed: keycode={keycode} err={exc}")

    print_help(args.step_xy)
    set_highlight(selected)
    print_selected()
    print("[INFO] Press P to save.")

    if args.no_viewer:
        return

    try:
        with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
            viewer.cam.distance = float(args.camera_distance)
            last_t = time.perf_counter()
            next_print_t = last_t
            next_sync_time = last_t
            while viewer.is_running():
                now_t = time.perf_counter()
                dt = max(0.0, min(0.05, now_t - last_t))
                last_t = now_t
                if now_t <= hold_until:
                    move_selected(
                        hold_dir[0] * hold_speed * dt,
                        hold_dir[1] * hold_speed * dt,
                        verbose=False,
                    )
                    if now_t >= next_print_t:
                        print_selected()
                        next_print_t = now_t + 0.2
                active = actors[selected]
                if len(active.body_ids) > 0:
                    viewer.cam.lookat[:] = data.xpos[active.body_ids[0]]
                if args.simulate:
                    mujoco.mj_step(model, data)
                viewer.sync()
                if args.simulate:
                    next_sync_time += model.opt.timestep
                    sleep_time = next_sync_time - time.perf_counter()
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                else:
                    time.sleep(0.01)
    except RuntimeError as exc:
        msg = str(exc)
        if "launch_passive" in msg and "mjpython" in msg:
            raise RuntimeError(
                "On macOS, run with mjpython:\n"
                "  ./.venv/bin/mjpython scripts/human_tuner.py "
                "--scene-xml assets/scenes/ithor/FloorPlan1_physics.xml "
                "--layout-json assets/layouts/FloorPlan1_object_positions.json"
            ) from exc
        raise
    finally:
        if modified:
            print("[INFO] You have unsaved human pose changes. Press P next run to save.")


if __name__ == "__main__":
    main()
