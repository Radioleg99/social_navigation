#!/usr/bin/env python3
"""Export and apply scene object positions via JSON.

Usage:
1) Export current object poses from an MJCF scene to JSON.
2) Edit the `overrides` section in JSON.
3) Apply overrides back to a new MJCF file.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import mujoco


def vec_to_str(v: list[float]) -> str:
    return " ".join(f"{x:.9g}" for x in v)


def auto_metadata_path(scene_xml: Path) -> Path:
    # FloorPlan1_physics.xml -> FloorPlan1_physics_metadata.json
    return scene_xml.with_name(f"{scene_xml.stem}_metadata.json")


def body_id_map(model: mujoco.MjModel) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for bid in range(model.nbody):
        name = model.body(bid).name
        if name:
            mapping[name] = bid
    return mapping


def root_body_names(model: mujoco.MjModel) -> set[str]:
    names: set[str] = set()
    for bid in range(1, model.nbody):
        # Parent body id == 0 => direct child of world
        if int(model.body_parentid[bid]) == 0:
            name = model.body(bid).name
            if name:
                names.add(name)
    return names


def load_metadata(metadata_json: Path | None) -> dict[str, Any]:
    if metadata_json is None or not metadata_json.is_file():
        return {}
    with open(metadata_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        return {}
    return payload


def export_positions(
    scene_xml: Path,
    out_json: Path,
    metadata_json: Path | None,
    include_all_root_bodies: bool,
) -> None:
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    bid_map = body_id_map(model)
    root_names = root_body_names(model)

    metadata = load_metadata(metadata_json)
    meta_objects = metadata.get("objects", {}) if isinstance(metadata, dict) else {}
    if not isinstance(meta_objects, dict):
        meta_objects = {}

    entries: list[dict[str, Any]] = []
    seen: set[str] = set()

    for body_name, info in meta_objects.items():
        if body_name not in bid_map:
            continue
        bid = bid_map[body_name]
        pos = [float(x) for x in data.xpos[bid].tolist()]
        quat = [float(x) for x in data.xquat[bid].tolist()]  # wxyz

        entry = {
            "body": body_name,
            "asset_id": info.get("asset_id", ""),
            "object_id": info.get("object_id", ""),
            "category": info.get("category", ""),
            "room_id": info.get("room_id", 0),
            "is_static": bool(info.get("is_static", False)),
            "position_xyz": pos,
            "quat_wxyz": quat,
        }
        entries.append(entry)
        seen.add(body_name)

    if include_all_root_bodies:
        for body_name in sorted(root_names):
            if body_name in seen:
                continue
            bid = bid_map[body_name]
            pos = [float(x) for x in data.xpos[bid].tolist()]
            quat = [float(x) for x in data.xquat[bid].tolist()]  # wxyz
            entries.append(
                {
                    "body": body_name,
                    "asset_id": "",
                    "object_id": "",
                    "category": "",
                    "room_id": 0,
                    "is_static": False,
                    "position_xyz": pos,
                    "quat_wxyz": quat,
                }
            )

    entries.sort(key=lambda e: (str(e.get("category", "")), str(e.get("body", ""))))

    payload = {
        "scene_xml": str(scene_xml),
        "metadata_json": str(metadata_json) if metadata_json is not None else "",
        "coordinate_frame": "mujoco_world",
        "notes": (
            "Edit overrides only. position_xyz is [x,y,z], quat_wxyz is [w,x,y,z]. "
            "Then run apply subcommand."
        ),
        "robot_runtime": {
            "enabled": True,
            "robot_type": "rby1",
            "robot_base": [0.8, 0.2, 0.0],
            "robot_pos": [0.0, -0.15, 0.0],
        },
        "human_runtime": {
            "enabled": True,
            "poses": [
                "assets/humans/standing_idle_multimat/character.xml",
                "assets/humans/sitting_2_multimat_matchedscale/character.xml",
            ],
            "pose_interval_sec": 2.0,
            "loop": True,
            "human_static": True,
            "human_xml": "assets/humans/standing_idle_multimat/character.xml",
            "human_pos": [1.2, 0.8, 0.0],
            "human_yaw_deg": 90.0,
            "human_roll_deg": 90.0,
            "human_pitch_deg": 0.0,
            "human_z_offset": 0.0,
        },
        "objects": entries,
        "overrides": {},
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"[INFO] Exported {len(entries)} object poses -> {out_json}")


def apply_overrides(scene_xml: Path, layout_json: Path, out_xml: Path) -> None:
    with open(layout_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    overrides = payload.get("overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("layout json field 'overrides' must be an object/dict")

    tree = ET.parse(scene_xml)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Invalid MJCF: missing <worldbody>")

    # Apply only to root bodies (direct children of worldbody)
    body_nodes: dict[str, ET.Element] = {}
    for body in worldbody.findall("body"):
        name = body.attrib.get("name", "")
        if name:
            body_nodes[name] = body

    applied = 0
    missing: list[str] = []
    for body_name, spec in overrides.items():
        if body_name not in body_nodes:
            missing.append(body_name)
            continue
        if not isinstance(spec, dict):
            raise ValueError(f"Override for body '{body_name}' must be a dict")

        body = body_nodes[body_name]
        pos = spec.get("position_xyz", None)
        quat = spec.get("quat_wxyz", None)

        if pos is not None:
            if not (isinstance(pos, list) and len(pos) == 3):
                raise ValueError(f"Override '{body_name}.position_xyz' must be [x,y,z]")
            body.attrib["pos"] = vec_to_str([float(pos[0]), float(pos[1]), float(pos[2])])
        if quat is not None:
            if not (isinstance(quat, list) and len(quat) == 4):
                raise ValueError(f"Override '{body_name}.quat_wxyz' must be [w,x,y,z]")
            body.attrib["quat"] = vec_to_str(
                [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]
            )
        applied += 1

    out_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_xml, encoding="utf-8")

    print(f"[INFO] Applied overrides: {applied}")
    if missing:
        print(f"[WARN] Missing bodies (not found in root worldbody): {len(missing)}")
        for name in missing[:20]:
            print(f"  - {name}")
        if len(missing) > 20:
            print(f"  ... +{len(missing) - 20} more")
    print(f"[INFO] Wrote scene xml: {out_xml}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_export = sub.add_parser("export", help="Export current scene object poses to JSON")
    p_export.add_argument("--scene-xml", type=Path, required=True, help="Path to scene MJCF XML")
    p_export.add_argument(
        "--metadata-json",
        type=Path,
        default=None,
        help="Optional scene metadata JSON (default: <scene_stem>_metadata.json if exists)",
    )
    p_export.add_argument("--out-json", type=Path, required=True, help="Output layout JSON path")
    p_export.add_argument(
        "--include-all-root-bodies",
        action="store_true",
        help="Also include root bodies not present in metadata",
    )

    p_apply = sub.add_parser("apply", help="Apply overrides from layout JSON to scene XML")
    p_apply.add_argument("--scene-xml", type=Path, required=True, help="Input scene MJCF XML")
    p_apply.add_argument("--layout-json", type=Path, required=True, help="Layout JSON with overrides")
    p_apply.add_argument("--out-xml", type=Path, required=True, help="Output patched scene XML")

    args = parser.parse_args()

    if args.cmd == "export":
        metadata_json = args.metadata_json
        if metadata_json is None:
            candidate = auto_metadata_path(args.scene_xml)
            metadata_json = candidate if candidate.is_file() else None
        export_positions(
            scene_xml=args.scene_xml,
            out_json=args.out_json,
            metadata_json=metadata_json,
            include_all_root_bodies=bool(args.include_all_root_bodies),
        )
        return

    if args.cmd == "apply":
        apply_overrides(scene_xml=args.scene_xml, layout_json=args.layout_json, out_xml=args.out_xml)
        return

    raise RuntimeError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
