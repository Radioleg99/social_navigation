"""
procthor_to_scene_graph.py
==========================
Generate a HumanSSG-compatible scene graph directly from ProcTHOR house JSON
+ MoLMoSpaces layout JSON, without running 3D reconstruction.

Output files (mirrors the standard HumanSSG pipeline output structure):
  3dsg/HumanSSG/data/{scene_id}/exps/{exp_suffix}/
      obj_json_{exp_suffix}.json
      edge_json_{exp_suffix}.json
  outputs/procthor/{split}_{idx}/scene_graph.json   ← ready for run_social_nav

Usage:
    python pipeline/procthor_to_scene_graph.py --split train --idx 9
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REPO_ROOT  = Path(__file__).resolve().parents[1] / "molmospaces-main"
GRAD_ROOT  = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Bbox size estimates (MuJoCo metres) keyed by asset name substrings (lower)
# [width_x, depth_y, height_z]
# ---------------------------------------------------------------------------
_BBOX_SIZES: list[tuple[str, list[float]]] = [
    ("countertop_l",    [2.0, 0.6, 0.9]),
    ("countertop",      [1.5, 0.6, 0.9]),
    ("fridge",          [0.7, 0.7, 1.8]),
    ("dining_table",    [1.5, 0.9, 0.75]),
    ("chair",           [0.5, 0.5, 0.9]),
    ("shelving_unit",   [0.8, 0.4, 1.8]),
    ("bin",             [0.3, 0.3, 0.5]),
    ("stool",           [0.4, 0.4, 0.5]),
    ("tv_stand",        [1.2, 0.4, 0.5]),
    ("dog_bed",         [0.8, 0.5, 0.15]),
    ("wall_decor",      [0.6, 0.05, 0.4]),
    ("sofa",            [2.0, 0.9, 0.9]),
    ("armchair",        [0.9, 0.9, 0.9]),
    ("table",           [1.2, 0.8, 0.75]),
    ("person",          [0.5, 0.3, 1.7]),
]
_DEFAULT_BBOX = [0.5, 0.5, 1.0]


def _bbox_for(name: str) -> list[float]:
    low = name.lower()
    for kw, sz in _BBOX_SIZES:
        if kw in low:
            return sz
    return _DEFAULT_BBOX


def _asset_to_tag(asset_id: str) -> str:
    """Convert ProcTHOR assetId to a short human-readable label."""
    low = asset_id.lower()
    if "dog_bed" in low:
        return "dog_bed"
    if "fridge" in low:
        return "fridge"
    if "countertop" in low:
        return "countertop"
    if "dining_table" in low or "table" in low:
        return "table"
    if "chair" in low:
        return "chair"
    if "shelving" in low:
        return "shelf"
    if "bin" in low:
        return "bin"
    if "stool" in low:
        return "stool"
    if "tv_stand" in low:
        return "tv_stand"
    if "wall_decor" in low:
        return "wall_decor"
    if "sofa" in low or "couch" in low:
        return "sofa"
    if "armchair" in low:
        return "armchair"
    # generic: strip trailing _N_... and lowercase
    return asset_id.split("_")[0].lower()


def _procthor_to_mujoco(pos: dict) -> list[float]:
    """ProcTHOR AI2-THOR position → MuJoCo world frame [x, y, z]."""
    return [
        round(float(pos.get("x", 0)), 4),
        round(float(pos.get("z", 0)), 4),
        round(float(pos.get("y", 0)), 4),
    ]


def _human_state(xml_path: str) -> str:
    low = xml_path.lower()
    if "talking" in low or "remy" in low or "rony" in low:
        return "talking"
    if "sitting" in low:
        return "sitting"
    return "idle"


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_scene_graph(
    house: dict,
    layout: dict,
    *,
    scene_id: str,
    proximity_edge_dist: float = 1.5,
) -> tuple[dict, dict]:
    """
    Returns (obj_json, edge_json) dicts in HumanSSG format.

    Parameters
    ----------
    house   : parsed ProcTHOR house JSON
    layout  : MoLMoSpaces layout JSON (human positions)
    scene_id: used for logging only
    """
    obj_json: dict  = {}
    edge_json: dict = {}
    node_id = 0
    edge_id = 0

    # ------------------------------------------------------------------ #
    # 1. Furniture / objects from house JSON
    # ------------------------------------------------------------------ #
    obj_id_map: dict[str, int] = {}   # asset_id (with index) → node_id

    for obj in house.get("objects", []):
        asset_id = obj.get("assetId", "unknown")
        pos_raw  = obj.get("position", {})
        mj_pos   = _procthor_to_mujoco(pos_raw)
        tag      = _asset_to_tag(asset_id)
        bbox_ext = _bbox_for(asset_id)

        key = f"object_{node_id + 1}"
        obj_json[key] = {
            "id":           node_id,
            "object_tag":   tag,
            "object_caption": asset_id,
            "bbox_center":  mj_pos,
            "bbox_extent":  bbox_ext,
            "bbox_volume":  round(bbox_ext[0] * bbox_ext[1] * bbox_ext[2], 4),
        }
        obj_id_map[asset_id + f"_{node_id}"] = node_id
        node_id += 1

    # ------------------------------------------------------------------ #
    # 2. Humans from layout JSON
    # ------------------------------------------------------------------ #
    hr = layout.get("human_runtime", {})
    humans_raw: list[dict] = []
    if hr.get("enabled", True):
        humans_raw.append(hr)
        for extra in hr.get("extra_humans", []):
            if extra.get("enabled", True):
                humans_raw.append(extra)

    human_node_ids: list[int] = []
    for h in humans_raw:
        xml_path = h.get("human_xml", "")
        state    = _human_state(xml_path)
        pos      = h.get("human_pos", [0, 0, 0])
        yaw      = h.get("human_yaw_deg", 0.0)
        bbox_ext = _bbox_for("person")

        key = f"object_{node_id + 1}"
        obj_json[key] = {
            "id":             node_id,
            "object_tag":     "person",
            "object_caption": f"person ({state})",
            "bbox_center":    [pos[0], pos[1], 0.9],   # z ≈ torso height
            "bbox_extent":    bbox_ext,
            "bbox_volume":    round(bbox_ext[0] * bbox_ext[1] * bbox_ext[2], 4),
            "heading_deg":    yaw,
            "activity":       state,
        }
        human_node_ids.append(node_id)
        node_id += 1

    # ------------------------------------------------------------------ #
    # 3. Edges — human social relationships
    # ------------------------------------------------------------------ #
    # Talking pairs: mutual SPEAK + OBSERVE
    if len(human_node_ids) >= 2:
        for i, h1_id in enumerate(human_node_ids):
            for h2_id in human_node_ids[i + 1:]:
                p1 = obj_json[f"object_{h1_id + 1}"]["bbox_center"]
                p2 = obj_json[f"object_{h2_id + 1}"]["bbox_center"]
                dist = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
                s1   = obj_json[f"object_{h1_id + 1}"].get("activity", "idle")
                s2   = obj_json[f"object_{h2_id + 1}"].get("activity", "idle")
                if s1 == "talking" and s2 == "talking" and dist < 2.0:
                    for src, dst, rel in [
                        (h1_id, h2_id, "SPEAK"),
                        (h2_id, h1_id, "SPEAK"),
                        (h1_id, h2_id, "OBSERVE"),
                        (h2_id, h1_id, "OBSERVE"),
                    ]:
                        t1 = obj_json[f"object_{src + 1}"]["object_tag"]
                        t2 = obj_json[f"object_{dst + 1}"]["object_tag"]
                        ek = f"edge_{edge_id}"
                        edge_json[ek] = {
                            "edge_id":         ek,
                            "object_1_id":     src,
                            "object_1_tag":    t1,
                            "object_2_id":     dst,
                            "object_2_tag":    t2,
                            "edge_description": f"{t1} {rel.lower()} {t2}",
                            "relationship":    rel,
                            "num_detections":  5,
                        }
                        edge_id += 1

    # 4. Human-to-furniture proximity edges
    furniture_ids = [i for i in range(node_id) if i not in human_node_ids]
    for h_id in human_node_ids:
        ph = obj_json[f"object_{h_id + 1}"]["bbox_center"]
        for f_id in furniture_ids:
            pf = obj_json[f"object_{f_id + 1}"]["bbox_center"]
            dist = math.hypot(ph[0] - pf[0], ph[1] - pf[1])
            if dist < proximity_edge_dist:
                t1 = obj_json[f"object_{h_id + 1}"]["object_tag"]
                t2 = obj_json[f"object_{f_id + 1}"]["object_tag"]
                ek = f"edge_{edge_id}"
                edge_json[ek] = {
                    "edge_id":         ek,
                    "object_1_id":     h_id,
                    "object_1_tag":    t1,
                    "object_2_id":     f_id,
                    "object_2_tag":    t2,
                    "edge_description": f"{t1} near {t2}",
                    "relationship":    "NEAR",
                    "num_detections":  2,
                }
                edge_id += 1

    return obj_json, edge_json


# ---------------------------------------------------------------------------
# Convert to scene_graph.json (humanssg_to_scene_graph logic inline)
# ---------------------------------------------------------------------------

def to_scene_graph_json(obj_json: dict, edge_json: dict) -> dict:
    nodes = []
    for key, obj in obj_json.items():
        obj_id = obj["id"]
        tag    = obj.get("object_tag", "unknown")

        activities = []
        for edge in edge_json.values():
            if edge.get("object_1_id") == obj_id:
                activities.append({
                    "name":   edge.get("relationship", ""),
                    "object": str(edge.get("object_2_id", "")),
                })

        node: dict = {
            "node_id":    obj_id,
            "label":      [tag],
            "bbox_center": obj.get("bbox_center", [0, 0, 0]),
            "bbox_extent": obj.get("bbox_extent",  [0, 0, 0]),
        }
        if activities:
            node["activities"] = activities
        if "heading_deg" in obj:
            node["heading_deg"] = obj["heading_deg"]
        nodes.append(node)

    edge_list = []
    for edge in edge_json.values():
        i   = edge.get("object_1_id")
        j   = edge.get("object_2_id")
        rel = edge.get("relationship", "")
        desc = edge.get("edge_description", rel)
        edge_list.append([i, j, {"desc": desc, "relation": rel}])

    return {"nodes": nodes, "edges": edge_list}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=["train", "val", "test"], default="train")
    p.add_argument("--idx",   type=int, default=9)
    p.add_argument("--proximity-dist", type=float, default=1.5,
                   help="Max distance (m) for human→furniture NEAR edges")
    p.add_argument("--pos-override", action="append", default=[], metavar="TAG:x,y",
                   help="Override MuJoCo (x,y) for an object by tag keyword, "
                        "e.g. --pos-override dog_bed:9,4.88. Repeatable.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    scene_id   = f"{args.split}_{args.idx}"
    xml_dir    = REPO_ROOT / "assets" / "scenes" / "procthor-10k-train"
    house_path = xml_dir / f"{scene_id}.json"
    layout_path= REPO_ROOT / "outputs" / "procthor" / scene_id / f"{scene_id}_layout.json"

    if not house_path.is_file():
        raise FileNotFoundError(f"House JSON not found: {house_path}")
    if not layout_path.is_file():
        raise FileNotFoundError(f"Layout JSON not found: {layout_path}")

    with open(house_path) as f:
        house = json.load(f)
    with open(layout_path) as f:
        layout = json.load(f)

    # Parse --pos-override TAG:x,y entries
    pos_overrides: dict[str, list[float]] = {}
    for ov in args.pos_override:
        tag_part, coord_part = ov.split(":", 1)
        x, y = [float(v) for v in coord_part.split(",")]
        pos_overrides[tag_part.lower()] = [x, y]

    print(f"[scene_graph] Building from house JSON ({scene_id}) ...")
    obj_json, edge_json = build_scene_graph(house, layout, scene_id=scene_id,
                                            proximity_edge_dist=args.proximity_dist)

    # Apply position overrides (match by object_tag keyword)
    for key, obj in obj_json.items():
        tag = obj.get("object_tag", "")
        for kw, (ox, oy) in pos_overrides.items():
            if kw in tag:
                obj["bbox_center"][0] = ox
                obj["bbox_center"][1] = oy
                print(f"[scene_graph] Override: {tag} (id={obj['id']}) → ({ox},{oy})")

    # -- Save intermediate HumanSSG-format files (mirrors standard pipeline output)
    exp_suffix  = f"from_house_json"
    humanssg_dir = (GRAD_ROOT / "3dsg" / "HumanSSG" / "data" /
                    f"procthor_{scene_id}" / "exps" / exp_suffix)
    humanssg_dir.mkdir(parents=True, exist_ok=True)

    obj_path  = humanssg_dir / f"obj_json_{exp_suffix}.json"
    edge_path = humanssg_dir / f"edge_json_{exp_suffix}.json"
    with open(obj_path,  "w") as f: json.dump(obj_json,  f, indent=2)
    with open(edge_path, "w") as f: json.dump(edge_json, f, indent=2)
    print(f"[scene_graph] obj_json  → {obj_path}")
    print(f"[scene_graph] edge_json → {edge_path}")

    # -- Convert to final scene_graph.json
    sg = to_scene_graph_json(obj_json, edge_json)

    out_path = REPO_ROOT / "outputs" / "procthor" / scene_id / "scene_graph.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(sg, f, indent=2, ensure_ascii=False)
    print(f"[scene_graph] scene_graph.json → {out_path}")

    # -- Summary
    persons = [n for n in sg["nodes"] if n["label"][0] == "person"]
    objects = [n for n in sg["nodes"] if n["label"][0] != "person"]
    print(f"\n  {len(sg['nodes'])} nodes  ({len(persons)} persons, {len(objects)} objects)")
    print(f"  {len(sg['edges'])} edges")
    print("\n  Persons:")
    for p in persons:
        c    = p["bbox_center"]
        acts = [a["name"] for a in p.get("activities", [])]
        yaw  = p.get("heading_deg", 0)
        print(f"    id={p['node_id']}  pos=({c[0]:.2f},{c[1]:.2f})  yaw={yaw}°  acts={acts}")
    dog_beds = [n for n in sg["nodes"] if "dog" in n["label"][0]]
    if dog_beds:
        db = dog_beds[0]
        c  = db["bbox_center"]
        print(f"\n  Dog bed: id={db['node_id']}  pos=({c[0]:.2f},{c[1]:.2f})")
        goal_x = round(c[0] - 0.8, 2)
        goal_y = round(c[1], 2)
        print(f"  Suggested goal in front of dog bed: --goal-xy {goal_x},{goal_y}")


if __name__ == "__main__":
    main()
