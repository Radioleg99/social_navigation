#!/usr/bin/env python3
"""Convert a Mixamo FBX file into a simple single-body MuJoCo MJCF model.

This is a pragmatic visual conversion path:
1) import FBX in Blender (bpy),
2) join all mesh parts into one mesh,
3) export OBJ,
4) write an MJCF XML that references the exported mesh.

The output MJCF is a rigid/free body (not a full skinned rig).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import re
from pathlib import Path


def parse_vec4(text: str) -> list[float]:
    values = [float(v.strip()) for v in text.split(",")]
    if len(values) != 4:
        raise argparse.ArgumentTypeError(f"Expected 4 comma-separated values, got: {text}")
    return values


def safe_name(text: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_]+", "_", text.strip())
    clean = clean.strip("_")
    return clean or "unnamed"


def clear_scene(bpy) -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # Cleanup orphan data blocks from previous runs.
    for collection in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.images,
        bpy.data.armatures,
        bpy.data.objects,
    ):
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)


def get_all_meshes(bpy):
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not mesh_objects:
        raise RuntimeError("No mesh objects found after importing FBX")

    return mesh_objects


def join_all_meshes(bpy):
    mesh_objects = get_all_meshes(bpy)

    bpy.ops.object.select_all(action="DESELECT")
    for obj in mesh_objects:
        obj.select_set(True)

    bpy.context.view_layer.objects.active = mesh_objects[0]
    if len(mesh_objects) > 1:
        bpy.ops.object.join()

    merged = bpy.context.view_layer.objects.active
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return merged


def export_multiple_meshes(bpy, mesh_objects, out_obj: Path) -> None:
    out_obj.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    for obj in mesh_objects:
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # Keep last selected object active for exporters requiring an active object.
    bpy.context.view_layer.objects.active = mesh_objects[-1]

    if hasattr(bpy.ops.wm, "obj_export"):
        bpy.ops.wm.obj_export(
            filepath=str(out_obj),
            export_selected_objects=True,
            export_materials=True,
            path_mode="COPY",
            forward_axis="NEGATIVE_Z",
            up_axis="Y",
        )
    elif hasattr(bpy.ops.export_scene, "obj"):
        bpy.ops.export_scene.obj(
            filepath=str(out_obj),
            use_selection=True,
            use_materials=True,
            path_mode="COPY",
            axis_forward="-Z",
            axis_up="Y",
        )
    else:
        raise RuntimeError("No OBJ exporter found in this Blender build")


def export_obj(bpy, obj, out_obj: Path) -> None:
    out_obj.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    # Blender 3.6 uses wm.obj_export; older versions may still expose export_scene.obj.
    if hasattr(bpy.ops.wm, "obj_export"):
        bpy.ops.wm.obj_export(
            filepath=str(out_obj),
            export_selected_objects=True,
            export_materials=True,
            path_mode="COPY",
            forward_axis="NEGATIVE_Z",
            up_axis="Y",
        )
    elif hasattr(bpy.ops.export_scene, "obj"):
        bpy.ops.export_scene.obj(
            filepath=str(out_obj),
            use_selection=True,
            use_materials=True,
            path_mode="COPY",
            axis_forward="-Z",
            axis_up="Y",
        )
    else:
        raise RuntimeError("No OBJ exporter found in this Blender build")


def find_material_basecolor_image(material):
    if material is None or not getattr(material, "use_nodes", False) or material.node_tree is None:
        return None

    # Prefer the image connected to Principled BSDF Base Color.
    for node in material.node_tree.nodes:
        if node.type != "BSDF_PRINCIPLED":
            continue
        base = node.inputs.get("Base Color")
        if base is None or not base.is_linked:
            continue
        src = base.links[0].from_node
        if src is not None and src.type == "TEX_IMAGE" and getattr(src, "image", None) is not None:
            return src.image

    # Fallback: first image texture node in the material.
    for node in material.node_tree.nodes:
        if node.type == "TEX_IMAGE" and getattr(node, "image", None) is not None:
            return node.image
    return None


def save_image_as_png(bpy, image, out_png: Path) -> bool:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    try:
        image.save_render(filepath=str(out_png))
        return True
    except Exception:
        pass

    try:
        src = Path(bpy.path.abspath(image.filepath, library=image.library))
        if src.is_file():
            if src.suffix.lower() == ".png":
                shutil.copy2(src, out_png)
            else:
                tmp = bpy.data.images.load(str(src), check_existing=True)
                tmp.save_render(filepath=str(out_png))
            return True
    except Exception:
        return False
    return False


def export_mesh_parts_by_material(bpy, out_dir: Path) -> list[dict[str, str | Path | None]]:
    mesh_objects = get_all_meshes(bpy)
    parts: list[dict[str, str | Path | None]] = []
    texture_cache: dict[str, Path] = {}
    part_idx = 0

    for obj in mesh_objects:
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

        if not obj.material_slots:
            continue

        for slot_idx, slot in enumerate(obj.material_slots):
            mat = slot.material

            dup = obj.copy()
            dup.data = obj.data.copy()
            bpy.context.collection.objects.link(dup)

            try:
                import bmesh  # type: ignore

                bm = bmesh.new()
                bm.from_mesh(dup.data)
                kept = 0
                for face in list(bm.faces):
                    if face.material_index != slot_idx:
                        bm.faces.remove(face)
                    else:
                        face.material_index = 0
                        kept += 1

                if kept == 0:
                    bm.free()
                    bpy.data.objects.remove(dup, do_unlink=True)
                    continue

                bm.to_mesh(dup.data)
                bm.free()
                dup.data.update()
            except Exception:
                bpy.data.objects.remove(dup, do_unlink=True)
                raise

            dup.data.materials.clear()
            if mat is not None:
                dup.data.materials.append(mat)

            part_name = f"part_{part_idx:03d}_{safe_name(obj.name)}_{safe_name(mat.name if mat else 'nomat')}"
            part_obj = out_dir / f"{part_name}.obj"
            export_obj(bpy, dup, part_obj)

            tex_out: Path | None = None
            if mat is not None:
                image = find_material_basecolor_image(mat)
                if image is not None:
                    key = image.filepath or image.name
                    if key not in texture_cache:
                        tex_name = f"tex_{safe_name(image.name)}.png"
                        candidate = out_dir / tex_name
                        suffix = 1
                        while candidate.exists() and key not in texture_cache:
                            candidate = out_dir / f"tex_{safe_name(image.name)}_{suffix}.png"
                            suffix += 1
                        if save_image_as_png(bpy, image, candidate):
                            texture_cache[key] = candidate
                    tex_out = texture_cache.get(key)

            parts.append(
                {
                    "name": part_name,
                    "mesh_file": part_obj,
                    "texture_file": tex_out,
                }
            )
            part_idx += 1
            bpy.data.objects.remove(dup, do_unlink=True)

    if not parts:
        raise RuntimeError("No material mesh parts exported from FBX")

    return parts


def export_obj_with_assimp(fbx: Path, out_obj: Path) -> None:
    out_obj.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["assimp", "export", str(fbx), str(out_obj)]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "assimp not found. Install with:\n"
            "  brew install assimp"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"assimp export failed: {exc}") from exc


def copy_obj_bundle(input_obj: Path, out_obj: Path) -> None:
    if not input_obj.is_file():
        raise FileNotFoundError(f"OBJ not found: {input_obj}")

    out_obj.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_obj, out_obj)

    src_dir = input_obj.parent
    dst_dir = out_obj.parent

    mtl_name = None
    for ln in input_obj.read_text(errors="ignore").splitlines():
        if ln.lower().startswith("mtllib "):
            mtl_name = ln.split(maxsplit=1)[1].strip()
            break

    if not mtl_name:
        return

    src_mtl = src_dir / mtl_name
    if not src_mtl.is_file():
        return

    dst_mtl = dst_dir / src_mtl.name
    shutil.copy2(src_mtl, dst_mtl)

    # Copy referenced texture maps if present.
    texture_re = re.compile(r"^(map_Kd|map_Ka|map_d|bump|map_Bump)\s+(.+)$", re.IGNORECASE)
    for ln in src_mtl.read_text(errors="ignore").splitlines():
        m = texture_re.match(ln.strip())
        if not m:
            continue
        tex_name = m.group(2).strip()
        tex_src = src_dir / tex_name
        if tex_src.is_file():
            tex_dst = dst_dir / tex_src.name
            if tex_dst.resolve() != tex_src.resolve():
                shutil.copy2(tex_src, tex_dst)


def normalize_obj_mesh(obj_path: Path, target_height: float, up_axis: str = "auto") -> dict[str, float | str]:
    lines = obj_path.read_text(errors="ignore").splitlines()

    vertex_idx: list[int] = []
    vertices: list[list[float]] = []
    for i, ln in enumerate(lines):
        if not ln.startswith("v "):
            continue
        parts = ln.split()
        if len(parts) < 4:
            continue
        try:
            xyz = [float(parts[1]), float(parts[2]), float(parts[3])]
        except ValueError:
            continue
        vertex_idx.append(i)
        vertices.append(xyz)

    if not vertices:
        raise RuntimeError(f"No vertices found in OBJ: {obj_path}")

    mins = [min(v[a] for v in vertices) for a in range(3)]
    maxs = [max(v[a] for v in vertices) for a in range(3)]
    ranges = [maxs[a] - mins[a] for a in range(3)]

    if up_axis == "auto":
        up_i = max(range(3), key=lambda a: ranges[a])
        up_axis_name = "xyz"[up_i]
    else:
        up_i = {"x": 0, "y": 1, "z": 2}[up_axis]
        up_axis_name = up_axis

    up_range = ranges[up_i]
    if up_range <= 1e-8:
        raise RuntimeError(f"Degenerate OBJ height range on axis '{up_axis_name}'")

    scale = float(target_height) / up_range

    centers = [(mins[a] + maxs[a]) * 0.5 for a in range(3)]
    bases = [mins[a] for a in range(3)]

    for idx, v in zip(vertex_idx, vertices):
        out = [0.0, 0.0, 0.0]
        for a in range(3):
            if a == up_i:
                out[a] = (v[a] - bases[a]) * scale
            else:
                out[a] = (v[a] - centers[a]) * scale
        lines[idx] = f"v {out[0]:.8f} {out[1]:.8f} {out[2]:.8f}"

    obj_path.write_text("\n".join(lines) + "\n")

    return {
        "up_axis": up_axis_name,
        "scale": scale,
        "orig_range_x": ranges[0],
        "orig_range_y": ranges[1],
        "orig_range_z": ranges[2],
    }


def normalize_obj_meshes(obj_paths: list[Path], target_height: float, up_axis: str = "auto") -> dict[str, float | str]:
    records: list[dict[str, object]] = []
    all_vertices: list[list[float]] = []

    for obj_path in obj_paths:
        lines = obj_path.read_text(errors="ignore").splitlines()
        vertex_idx: list[int] = []
        vertices: list[list[float]] = []
        for i, ln in enumerate(lines):
            if not ln.startswith("v "):
                continue
            parts = ln.split()
            if len(parts) < 4:
                continue
            try:
                xyz = [float(parts[1]), float(parts[2]), float(parts[3])]
            except ValueError:
                continue
            vertex_idx.append(i)
            vertices.append(xyz)
            all_vertices.append(xyz)

        records.append({"path": obj_path, "lines": lines, "vertex_idx": vertex_idx, "vertices": vertices})

    if not all_vertices:
        raise RuntimeError("No vertices found across OBJ mesh parts")

    mins = [min(v[a] for v in all_vertices) for a in range(3)]
    maxs = [max(v[a] for v in all_vertices) for a in range(3)]
    ranges = [maxs[a] - mins[a] for a in range(3)]

    if up_axis == "auto":
        up_i = max(range(3), key=lambda a: ranges[a])
        up_axis_name = "xyz"[up_i]
    else:
        up_i = {"x": 0, "y": 1, "z": 2}[up_axis]
        up_axis_name = up_axis

    up_range = ranges[up_i]
    if up_range <= 1e-8:
        raise RuntimeError(f"Degenerate OBJ height range on axis '{up_axis_name}'")

    scale = float(target_height) / up_range
    centers = [(mins[a] + maxs[a]) * 0.5 for a in range(3)]
    bases = [mins[a] for a in range(3)]

    for rec in records:
        lines = rec["lines"]
        vertex_idx = rec["vertex_idx"]
        vertices = rec["vertices"]
        for idx, v in zip(vertex_idx, vertices):
            out = [0.0, 0.0, 0.0]
            for a in range(3):
                if a == up_i:
                    out[a] = (v[a] - bases[a]) * scale
                else:
                    out[a] = (v[a] - centers[a]) * scale
            lines[idx] = f"v {out[0]:.8f} {out[1]:.8f} {out[2]:.8f}"
        Path(rec["path"]).write_text("\n".join(lines) + "\n")

    return {
        "up_axis": up_axis_name,
        "scale": scale,
        "orig_range_x": ranges[0],
        "orig_range_y": ranges[1],
        "orig_range_z": ranges[2],
    }


def flatten_obj_for_mujoco(obj_path: Path) -> dict[str, int]:
    """Drop OBJ grouping/material directives so MuJoCo sees one combined mesh."""
    raw = obj_path.read_text(errors="ignore").splitlines()
    kept: list[str] = []
    dropped = 0
    face_count = 0
    for ln in raw:
        s = ln.strip()
        if s.startswith("mtllib ") or s.startswith("usemtl ") or s.startswith("g ") or s.startswith("o "):
            dropped += 1
            continue
        if s.startswith("f "):
            face_count += 1
        kept.append(ln)
    obj_path.write_text("\n".join(kept) + "\n")
    return {"dropped_lines": dropped, "faces": face_count}


def write_mjcf(
    out_xml: Path,
    mesh_file: Path,
    model_name: str,
    rgba: list[float],
    non_colliding: bool,
    texture_file: Path | None,
) -> None:
    out_xml.parent.mkdir(parents=True, exist_ok=True)

    texture_block = ""
    material_attr = ""
    if texture_file is not None:
        texture_block = (
            f'    <texture name="human_tex" type="2d" file="{texture_file.name}" />\n'
            '    <material name="human_mat" texture="human_tex" rgba="1 1 1 1" />\n'
        )
        material_attr = ' material="human_mat"'

    contype = 0 if non_colliding else 1
    conaffinity = 0 if non_colliding else 1
    rgba_str = f"{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"

    xml = f"""<mujoco model="{model_name}">
  <compiler angle="degree" autolimits="true" />
  <option timestep="0.005" />
  <asset>
    <mesh name="human_mesh" file="{mesh_file.name}" />
{texture_block}  </asset>
  <worldbody>
    <body name="human_body" pos="0 0 0">
      <freejoint name="XYZ_jntfree" />
      <geom
        name="human_visual"
        type="mesh"
        mesh="human_mesh"
        mass="55"
        rgba="{rgba_str}"
        contype="{contype}"
        conaffinity="{conaffinity}"{material_attr}
      />
    </body>
  </worldbody>
</mujoco>
"""
    out_xml.write_text(xml)


def write_mjcf_multi_material(
    out_xml: Path,
    parts: list[dict[str, str | Path | None]],
    model_name: str,
    rgba: list[float],
    non_colliding: bool,
) -> None:
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    contype = 0 if non_colliding else 1
    conaffinity = 0 if non_colliding else 1
    rgba_str = f"{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"
    geom_mass = 55.0 / max(1, len(parts))

    asset_lines = []
    geom_lines = []
    for idx, part in enumerate(parts):
        part_name = safe_name(str(part["name"]))
        mesh_name = f"{part_name}_mesh"
        mat_name = f"{part_name}_mat"
        tex_name = f"{part_name}_tex"
        mesh_file = Path(str(part["mesh_file"]))
        tex_file = part.get("texture_file")

        asset_lines.append(f'    <mesh name="{mesh_name}" file="{mesh_file.name}" />')
        material_attr = ""
        if tex_file is not None:
            tex_file_path = Path(str(tex_file))
            asset_lines.append(f'    <texture name="{tex_name}" type="2d" file="{tex_file_path.name}" />')
            asset_lines.append(f'    <material name="{mat_name}" texture="{tex_name}" rgba="1 1 1 1" />')
            material_attr = f' material="{mat_name}"'

        geom_lines.append(
            f'      <geom name="human_visual_{idx:03d}" type="mesh" mesh="{mesh_name}" '
            f'mass="{geom_mass:.6f}" rgba="{rgba_str}" contype="{contype}" conaffinity="{conaffinity}"{material_attr} />'
        )

    xml = (
        f'<mujoco model="{model_name}">\n'
        '  <compiler angle="degree" autolimits="true" />\n'
        '  <option timestep="0.005" />\n'
        '  <asset>\n'
        + "\n".join(asset_lines)
        + '\n  </asset>\n'
        '  <worldbody>\n'
        '    <body name="human_body" pos="0 0 0">\n'
        '      <freejoint name="XYZ_jntfree" />\n'
        + "\n".join(geom_lines)
        + '\n    </body>\n'
        '  </worldbody>\n'
        '</mujoco>\n'
    )
    out_xml.write_text(xml)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--fbx", type=Path, default=None, help="Input Mixamo FBX path")
    input_group.add_argument("--obj-input", type=Path, default=None, help="Use existing OBJ instead of FBX")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--model-name", default="mixamo_human", help="MJCF model name")
    parser.add_argument(
        "--backend",
        choices=["auto", "bpy", "assimp"],
        default="auto",
        help="Conversion backend. auto prefers bpy, then assimp.",
    )
    parser.add_argument(
        "--target-height",
        type=float,
        default=1.7,
        help="Normalize mesh to this height in meters",
    )
    parser.add_argument(
        "--up-axis",
        choices=["auto", "x", "y", "z"],
        default="auto",
        help="Vertical axis for normalization (auto picks largest bbox axis)",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable auto centering and scaling of OBJ mesh",
    )
    parser.add_argument(
        "--no-flatten-groups",
        action="store_true",
        help="Keep OBJ group/material directives (default flattens for MuJoCo compatibility)",
    )
    parser.add_argument(
        "--rgba",
        type=parse_vec4,
        default=[0.9, 0.9, 0.9, 1.0],
        help='Fallback geom color "r,g,b,a" if no texture is supplied',
    )
    parser.add_argument(
        "--non-colliding",
        action="store_true",
        help="Set contype/conaffinity to 0 for a purely visual human",
    )
    parser.add_argument(
        "--texture-file",
        type=Path,
        default=None,
        help="Optional texture file to copy next to XML and bind as material",
    )
    parser.add_argument(
        "--no-join",
        action="store_true",
        help="Do not join meshes in Blender; export all mesh objects together",
    )
    parser.add_argument(
        "--multi-material",
        action="store_true",
        help="For FBX+bpy: export one mesh per material and keep original textures (no baking)",
    )
    # Support Blender invocation:
    # blender -b --python script.py -- --fbx ... --out-dir ...
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = argv[1:]
    args = parser.parse_args(argv)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_obj = out_dir / "character.obj"
    out_xml = out_dir / "character.xml"

    if args.multi_material:
        if args.obj_input is not None:
            raise RuntimeError("--multi-material currently supports --fbx only")
        if args.fbx is None or not args.fbx.is_file():
            raise FileNotFoundError(f"FBX not found: {args.fbx}")
        if args.backend == "assimp":
            raise RuntimeError("--multi-material requires Blender backend (bpy)")

        import bpy  # type: ignore

        clear_scene(bpy)
        bpy.ops.import_scene.fbx(filepath=str(args.fbx))
        parts = export_mesh_parts_by_material(bpy, out_dir)

        flatten_total = {"dropped_lines": 0, "faces": 0}
        if not args.no_flatten_groups:
            for part in parts:
                info = flatten_obj_for_mujoco(Path(str(part["mesh_file"])))
                flatten_total["dropped_lines"] += int(info["dropped_lines"])
                flatten_total["faces"] += int(info["faces"])

        norm_info = None
        if not args.no_normalize:
            norm_info = normalize_obj_meshes(
                [Path(str(part["mesh_file"])) for part in parts],
                target_height=args.target_height,
                up_axis=args.up_axis,
            )

        write_mjcf_multi_material(
            out_xml=out_xml,
            parts=parts,
            model_name=args.model_name,
            rgba=args.rgba,
            non_colliding=args.non_colliding,
        )

        textured_parts = sum(1 for p in parts if p.get("texture_file") is not None)
        print(f"[INFO] MJCF saved: {out_xml}")
        print("[INFO] Backend:    bpy_multi_material")
        print(f"[INFO] Parts:      total={len(parts)} textured={textured_parts}")
        if not args.no_flatten_groups:
            print(
                "[INFO] Flatten:   "
                f"dropped_lines={flatten_total['dropped_lines']} faces={flatten_total['faces']}"
            )
        if norm_info is not None:
            print(
                "[INFO] Normalize: "
                f"up_axis={norm_info['up_axis']} "
                f"scale={float(norm_info['scale']):.8f} "
                f"orig_ranges=({float(norm_info['orig_range_x']):.4f}, "
                f"{float(norm_info['orig_range_y']):.4f}, {float(norm_info['orig_range_z']):.4f})"
            )
        print("[INFO] Use it in scene with:")
        print(f"  ./.venv/bin/mjpython scripts/basic_robot_human_scene.py --human-xml {out_xml}")
        return

    used_backend = None
    if args.obj_input is not None:
        copy_obj_bundle(args.obj_input, out_obj)
        used_backend = "obj_input"
    else:
        if args.fbx is None or not args.fbx.is_file():
            raise FileNotFoundError(f"FBX not found: {args.fbx}")

        if args.backend in ("auto", "bpy"):
            try:
                import bpy  # type: ignore

                clear_scene(bpy)
                bpy.ops.import_scene.fbx(filepath=str(args.fbx))
                if args.no_join:
                    mesh_objects = get_all_meshes(bpy)
                    export_multiple_meshes(bpy, mesh_objects, out_obj)
                else:
                    merged_obj = join_all_meshes(bpy)
                    export_obj(bpy, merged_obj, out_obj)
                used_backend = "bpy"
            except Exception as exc:
                if args.backend == "bpy":
                    raise RuntimeError(
                        "bpy backend failed. Either install/fix bpy, or use --backend assimp."
                    ) from exc

    if used_backend is None:
            if args.backend in ("auto", "assimp"):
                export_obj_with_assimp(args.fbx, out_obj)
                used_backend = "assimp"
            else:
                raise RuntimeError("No usable backend selected")

    flatten_info = None
    if not args.no_flatten_groups:
        flatten_info = flatten_obj_for_mujoco(out_obj)

    norm_info = None
    if not args.no_normalize:
        norm_info = normalize_obj_mesh(out_obj, target_height=args.target_height, up_axis=args.up_axis)

    texture_target = None
    if args.texture_file is not None:
        if not args.texture_file.is_file():
            raise FileNotFoundError(f"Texture file not found: {args.texture_file}")
        texture_target = out_dir / args.texture_file.name
        shutil.copy2(args.texture_file, texture_target)

    write_mjcf(
        out_xml=out_xml,
        mesh_file=out_obj,
        model_name=args.model_name,
        rgba=args.rgba,
        non_colliding=args.non_colliding,
        texture_file=texture_target,
    )

    print(f"[INFO] OBJ saved:  {out_obj}")
    print(f"[INFO] MJCF saved: {out_xml}")
    print(f"[INFO] Backend:    {used_backend}")
    if flatten_info is not None:
        print(
            "[INFO] Flatten:   "
            f"dropped_lines={flatten_info['dropped_lines']} faces={flatten_info['faces']}"
        )
    if norm_info is not None:
        print(
            "[INFO] Normalize: "
            f"up_axis={norm_info['up_axis']} "
            f"scale={float(norm_info['scale']):.8f} "
            f"orig_ranges=({float(norm_info['orig_range_x']):.4f}, "
            f"{float(norm_info['orig_range_y']):.4f}, {float(norm_info['orig_range_z']):.4f})"
        )
    print("[INFO] Use it in scene with:")
    print(f"  ./.venv/bin/mjpython scripts/basic_robot_human_scene.py --human-xml {out_xml}")


if __name__ == "__main__":
    main()
