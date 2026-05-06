#!/usr/bin/env python3
"""Append a mesh object entry to a scene JSON file.

Supports two schemas:
1) `procthor`: fields expected by MolmoSpaces house JSON (`objects` entries).
2) `generic`: a minimal custom schema with `meshFile`, `position`, `rotation`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_vec3(text: str) -> tuple[float, float, float]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"Expected 3 comma-separated values, got: {text}")
    try:
        return float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid numeric vec3: {text}") from exc


def swap_yz(pos: tuple[float, float, float]) -> tuple[float, float, float]:
    # MolmoSpaces housegen uses Unity<->MuJoCo conversion: (x, y, z) <-> (x, z, y)
    return (pos[0], pos[2], pos[1])


def convert_position(
    pos: tuple[float, float, float], src_frame: str, dst_frame: str
) -> tuple[float, float, float]:
    if src_frame == dst_frame:
        return pos
    if {src_frame, dst_frame} == {"mujoco", "unity"}:
        return swap_yz(pos)
    raise ValueError(f"Unsupported frame conversion: {src_frame} -> {dst_frame}")


def to_xyz_dict(v: tuple[float, float, float]) -> dict[str, float]:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def build_object_record(args: argparse.Namespace) -> dict[str, Any]:
    mesh_path = Path(args.mesh_file)
    default_asset_id = mesh_path.stem
    object_id = args.object_id or f"{default_asset_id}|custom"
    asset_id = args.asset_id or default_asset_id

    out_pos = convert_position(args.position, args.position_frame, args.json_position_frame)
    out_rot = args.rotation_deg

    if args.schema == "procthor":
        record: dict[str, Any] = {
            "id": object_id,
            "assetId": asset_id,
            "position": to_xyz_dict(out_pos),
            "rotation": to_xyz_dict(out_rot),
            "kinematic": bool(args.kinematic),
        }
        if args.object_type:
            record["objectType"] = args.object_type
        if args.keep_mesh_file_key:
            record[args.keep_mesh_file_key] = str(mesh_path)
        return record

    # generic schema
    return {
        "id": object_id,
        "meshFile": str(mesh_path),
        "assetId": asset_id,
        "position": to_xyz_dict(out_pos),
        "rotation": to_xyz_dict(out_rot),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-json", type=Path, required=True, help="Input scene JSON path")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Output JSON path (default: overwrite --scene-json)",
    )
    parser.add_argument("--mesh-file", type=str, required=True, help="Mesh filename/path to register")
    parser.add_argument(
        "--position",
        type=parse_vec3,
        required=True,
        help='Object position as "x,y,z" in --position-frame',
    )
    parser.add_argument(
        "--rotation-deg",
        type=parse_vec3,
        default=(0.0, 0.0, 0.0),
        help='Euler degrees as "x,y,z" written into JSON',
    )
    parser.add_argument(
        "--position-frame",
        choices=["mujoco", "unity"],
        default="mujoco",
        help="Coordinate frame of --position input",
    )
    parser.add_argument(
        "--json-position-frame",
        choices=["mujoco", "unity"],
        default="unity",
        help="Coordinate frame expected by JSON",
    )
    parser.add_argument(
        "--schema",
        choices=["procthor", "generic"],
        default="procthor",
        help="Object record schema to append",
    )
    parser.add_argument(
        "--objects-key",
        default="objects",
        help="Top-level array key where object entries are appended",
    )
    parser.add_argument("--asset-id", default=None, help="assetId to write (default: mesh stem)")
    parser.add_argument("--object-id", default=None, help="id to write (default: <assetId>|custom)")
    parser.add_argument(
        "--object-type",
        default="CustomHuman",
        help="objectType for procthor schema (empty string to skip)",
    )
    parser.add_argument(
        "--kinematic",
        action="store_true",
        help="Set kinematic=true for procthor schema",
    )
    parser.add_argument(
        "--replace-if-id-exists",
        action="store_true",
        help="Replace existing object with the same id instead of failing",
    )
    parser.add_argument(
        "--keep-mesh-file-key",
        default="meshFile",
        help=(
            "Optional extra field name to keep the original mesh path in procthor schema. "
            "Set empty string to disable."
        ),
    )
    args = parser.parse_args()

    in_path = args.scene_json
    out_path = args.out_json or in_path
    if not in_path.is_file():
        raise FileNotFoundError(f"Scene JSON not found: {in_path}")

    with open(in_path, "r", encoding="utf-8") as f:
        scene = json.load(f)

    if not isinstance(scene, dict):
        raise ValueError("Scene JSON root must be an object")

    if args.objects_key not in scene:
        scene[args.objects_key] = []
    if not isinstance(scene[args.objects_key], list):
        raise ValueError(f"Scene JSON field '{args.objects_key}' must be a list")

    if args.object_type == "":
        args.object_type = None
    if args.keep_mesh_file_key == "":
        args.keep_mesh_file_key = None

    new_obj = build_object_record(args)
    object_id = str(new_obj["id"])

    replaced = False
    for i, obj in enumerate(scene[args.objects_key]):
        if isinstance(obj, dict) and str(obj.get("id", "")) == object_id:
            if not args.replace_if_id_exists:
                raise ValueError(
                    f"Object id already exists: '{object_id}'. Use --replace-if-id-exists or change --object-id."
                )
            scene[args.objects_key][i] = new_obj
            replaced = True
            break

    if not replaced:
        scene[args.objects_key].append(new_obj)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(scene, f, ensure_ascii=False, indent=2)
        f.write("\n")

    action = "replaced" if replaced else "appended"
    print(f"[INFO] {action} object id={object_id}")
    print(f"[INFO] scene json: {out_path}")
    print(f"[INFO] objects count: {len(scene[args.objects_key])}")


if __name__ == "__main__":
    main()
