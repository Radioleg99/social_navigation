"""
procthor_to_layout.py
=====================
ProcTHOR house JSON  →  MoLMoSpaces layout JSON

Reads a ProcTHOR house definition (train_N.json) and generates a layout JSON
with auto-placed humans inferred from furniture positions.

Furniture → human mapping
--------------------------
  Sofa / ArmChair / Loveseat / Couch  → sitting human on the furniture
  DiningChair                          → idle standing human at the chair
  Nearby idle pairs (< 1.5 m apart)   → upgraded to talking pair

Coordinate mapping (ProcTHOR AI2-THOR y-up  →  MuJoCo z-up)
--------------------------------------------------------------
  ProcTHOR  x  →  MuJoCo  x    (east-west, same)
  ProcTHOR  z  →  MuJoCo  y    (horizontal floor axis)
  ProcTHOR  y  →  MuJoCo  z    (height, always 0 for ground-level humans)
  rotation.y   →  yaw_deg       (same axis, degrees)

Usage (programmatic):
    from pipeline.procthor_to_layout import house_json_to_layout_json
    layout = house_json_to_layout_json(house_dict, "assets/scenes/.../train_1.xml")
"""

from __future__ import annotations

import math
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_HUMANS = 4                  # cap on placed humans
SITTING_FORWARD_OFFSET = 0.30  # shift sitting human this far toward room centre (m)
TALKING_PAIR_DIST = 1.8        # two idle people within this distance → talking (m)

_HUMAN_XMLS = {
    "sitting":   "assets/humans/sitting_2_multimat_matchedscale/character.xml",
    "idle":      "assets/humans/standing_idle_multimat/character.xml",
    "talking_a": "assets/humans/remy_talking_multimat/character.xml",
    "talking_b": "assets/humans/rony_talking_multimat/character.xml",
}

# Furniture keywords that imply human presence (case-insensitive substring match)
_SITTING_KEYWORDS  = ["sofa", "armchair", "loveseat", "couch"]
_STANDING_KEYWORDS = ["diningchair"]
_SKIP_KEYWORDS     = ["bed", "toilet", "bathtub", "shower", "bathroomcabinet"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classify(asset_id: str) -> str | None:
    low = asset_id.lower()
    for kw in _SKIP_KEYWORDS:
        if kw in low:
            return None
    for kw in _SITTING_KEYWORDS:
        if kw in low:
            return "sitting"
    for kw in _STANDING_KEYWORDS:
        if kw in low:
            return "standing"
    return None


def _to_mujoco_xy(pos: dict, offset_x: float = 0.0, offset_y: float = 0.0) -> list[float]:
    """ProcTHOR position dict → MuJoCo [x, y, z=0]."""
    return [
        round(float(pos["x"]) + offset_x, 4),
        round(float(pos["z"]) + offset_y, 4),
        0.0,
    ]


def _polygon_centroid(polygon: list[dict]) -> tuple[float, float]:
    xs = [p["x"] for p in polygon]
    zs = [p["z"] for p in polygon]
    return sum(xs) / len(xs), sum(zs) / len(zs)


def _polygon_area(polygon: list[dict]) -> float:
    pts = [(p["x"], p["z"]) for p in polygon]
    n = len(pts)
    return abs(sum(
        pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1]
        for i in range(n)
    )) / 2.0


def _human_entry(
    mj_pos: list[float],
    yaw_deg: float,
    xml: str,
    sitting: bool,
    enabled: bool = True,
) -> dict:
    if sitting:
        collider_size = [0.20, 0.85, 0.55]
        z_offset = 0.45
    else:
        collider_size = [0.28, 1.65, 0.0]
        z_offset = 0.0
    return {
        "enabled": enabled,
        "human_xml": xml,
        "human_pos": mj_pos,
        "human_yaw_deg": yaw_deg,
        "human_roll_deg": 90.0,
        "human_pitch_deg": 0.0,
        "human_z_offset": z_offset,
        "human_static": True,
        "human_collider_type": "capsule",
        "human_collider_size": collider_size,
    }


def _upgrade_talking_pairs(candidates: list[dict]) -> None:
    """In-place: if two idle-standing humans are close, upgrade both to talking."""
    pool = [_HUMAN_XMLS["talking_a"], _HUMAN_XMLS["talking_b"]]
    used: set[int] = set()
    idles = [i for i, c in enumerate(candidates) if c["kind"] == "standing"]
    for a_pos, i in enumerate(idles):
        if i in used:
            continue
        for b_pos, j in enumerate(idles):
            if j <= i or j in used:
                continue
            p1 = candidates[i]["mj_xy"]
            p2 = candidates[j]["mj_xy"]
            if math.hypot(p1[0] - p2[0], p1[1] - p2[1]) < TALKING_PAIR_DIST:
                candidates[i]["entry"]["human_xml"] = pool[0]
                candidates[j]["entry"]["human_xml"] = pool[1]
                used.add(i)
                used.add(j)
                break


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def house_json_to_layout_json(
    house: dict,
    scene_xml: str | Path,
    robot_start: tuple[float, float, float] = (0.0, 0.0, 0.0),
    max_humans: int = MAX_HUMANS,
    seed: int = 0,
) -> dict:
    """
    Convert a ProcTHOR house dict into a MoLMoSpaces layout JSON dict.

    Parameters
    ----------
    house       : parsed ProcTHOR house JSON (rooms + objects)
    scene_xml   : path to the MuJoCo XML for this scene
    robot_start : (x, y, yaw_deg) robot starting pose — informational only,
                  not written into layout JSON (layout JSON records humans only)
    max_humans  : hard cap on placed humans
    seed        : random seed for furniture selection order

    Returns
    -------
    dict ready to be json.dump()-ed as a MoLMoSpaces layout JSON
    """
    rng = random.Random(seed)

    rooms   = house.get("rooms", [])
    objects = house.get("objects", [])

    # Scene centroid — used to nudge sitting humans toward the open space
    if rooms:
        all_poly_pts = [p for r in rooms for p in r.get("floorPolygon", [])]
        scene_cx, scene_cz = _polygon_centroid(all_poly_pts) if all_poly_pts else (0.0, 0.0)
    else:
        scene_cx, scene_cz = 0.0, 0.0

    # ------------------------------------------------------------------ #
    # Scan objects for human-placement triggers
    # ------------------------------------------------------------------ #
    candidates: list[dict] = []

    for obj in objects:
        asset_id = obj.get("assetId", "")
        pos      = obj.get("position", {})
        rot      = obj.get("rotation", {})
        kind     = _classify(asset_id)

        if kind is None:
            continue

        yaw_deg = float(rot.get("y", 0.0))

        if kind == "sitting":
            # Offset toward room centre so the human sits "on" the furniture
            dx = scene_cx - float(pos.get("x", 0.0))
            dz = scene_cz - float(pos.get("z", 0.0))
            dist = math.hypot(dx, dz) or 1.0
            ox = (dx / dist) * SITTING_FORWARD_OFFSET
            oy = (dz / dist) * SITTING_FORWARD_OFFSET
            mj_pos = _to_mujoco_xy(pos, ox, oy)
            entry  = _human_entry(mj_pos, yaw_deg, _HUMAN_XMLS["sitting"], sitting=True)
            candidates.append({"entry": entry, "kind": "sitting",
                                "mj_xy": mj_pos[:2]})

        elif kind == "standing":
            mj_pos = _to_mujoco_xy(pos)
            entry  = _human_entry(mj_pos, yaw_deg, _HUMAN_XMLS["idle"], sitting=False)
            candidates.append({"entry": entry, "kind": "standing",
                                "mj_xy": mj_pos[:2]})

    # Prefer sitting > standing; shuffle within each group for scene variety
    sitting  = [c for c in candidates if c["kind"] == "sitting"]
    standing = [c for c in candidates if c["kind"] == "standing"]
    rng.shuffle(sitting)
    rng.shuffle(standing)
    chosen = (sitting + standing)[:max_humans]

    # Upgrade nearby idle pairs to talking animations
    _upgrade_talking_pairs(chosen)

    # ------------------------------------------------------------------ #
    # Assemble layout JSON
    # ------------------------------------------------------------------ #
    scene_xml_str = str(scene_xml)
    meta_path     = scene_xml_str.replace(".xml", "_metadata.json")

    if not chosen:
        # Fallback: one human at scene centroid
        fallback_pos = [round(scene_cx, 4), round(scene_cz, 4), 0.0]
        primary_entry = _human_entry(fallback_pos, 0.0, _HUMAN_XMLS["idle"], sitting=False)
        extra_entries: list[dict] = []
    else:
        primary_entry = chosen[0]["entry"]
        extra_entries = [c["entry"] for c in chosen[1:]]

    layout = {
        "scene_xml":        scene_xml_str,
        "metadata_json":    meta_path,
        "coordinate_frame": "mujoco_world",
        "notes": (
            "Auto-generated by pipeline/procthor_to_layout.py. "
            "Humans placed from furniture (sofa→sitting, diningchair→idle). "
            "Coordinate mapping: ProcTHOR (x,z) → MuJoCo (x,y)."
        ),
        "human_runtime": {
            "enabled":          True,
            "poses":            [_HUMAN_XMLS["sitting"], _HUMAN_XMLS["idle"]],
            "pose_interval_sec": 2.0,
            "loop":             True,
            **primary_entry,
            "extra_humans":     extra_entries,
        },
    }

    return layout


def largest_room_info(house: dict) -> dict:
    """
    Return metadata about the largest room — useful for auto-picking start/goal.

    Returns
    -------
    dict with keys: roomType, centroid_x, centroid_y, area, polygon
    """
    rooms = house.get("rooms", [])
    if not rooms:
        return {"roomType": "unknown", "centroid_x": 0.0, "centroid_y": 0.0,
                "area": 0.0, "polygon": []}
    best = max(rooms, key=lambda r: _polygon_area(r.get("floorPolygon", [])))
    poly = best.get("floorPolygon", [])
    cx, cz = _polygon_centroid(poly) if poly else (0.0, 0.0)
    return {
        "roomType":   best.get("roomType", "unknown"),
        "centroid_x": round(cx, 3),
        "centroid_y": round(cz, 3),
        "area":       round(_polygon_area(poly), 2),
        "polygon":    poly,
    }
