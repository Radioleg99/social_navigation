"""
scene_bridge.py
===============
Converts HumanSSG scene_graph.json output → SceneDescription used by
molmospaces-main/experiments/social_nav/llm_costmap.py.

HumanSSG graph format (from extract_json.py):
    {
        "nodes": [
            {
                "node_id": int,
                "label": ["person" | "chair" | ...],
                "bbox_center": [x, y, z],   # 3DSG coordinate frame
                "bbox_extent": [ex, ey, ez],
                "activities": [{"name": str, "object": str}],  # humans only
                "heading_deg": float,        # optional, humans only
            }
        ],
        "edges": [
            [id1, id2, {"desc": str, "relation": str}]
        ]
    }

MuJoCo coordinate convention: x = east, y = north, z = up.
HumanSSG convention may differ — pass coord_transform to align.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import NamedTuple

import numpy as np

# ---------------------------------------------------------------------------
# Re-use SceneDescription from llm_costmap without importing the heavy module
# (avoids dragging in mujoco/torch at import time)
# ---------------------------------------------------------------------------

class HumanInfo(NamedTuple):
    pos: tuple[float, float]    # (x, y) in MuJoCo world frame (metres)
    yaw_deg: float              # heading, 0=+x, 90=+y
    state: str                  # "idle" | "talking" | "sitting" | "walking"


class ObstacleInfo(NamedTuple):
    category: str
    pos: tuple[float, float]    # (x, y) in MuJoCo world frame


class SceneDescription(NamedTuple):
    humans: list[HumanInfo]
    obstacles: list[ObstacleInfo]
    groups: list[list[int]]     # conversation groups: [[human_idx, ...], ...]


# ---------------------------------------------------------------------------
# Activity → social state mapping
# HumanSSG activities: Sitting, Walking, Watching, Waiting, Preparing, Using,
#                      Reading, Talking, Standing, ...
# Social states:       sitting, walking, talking, idle
# ---------------------------------------------------------------------------

_ACTIVITY_TO_STATE: dict[str, str] = {
    "talking":   "talking",
    "sitting":   "sitting",
    "walking":   "walking",
    "standing":  "idle",
    "watching":  "sitting",   # seated in front of TV / screen
    "reading":   "sitting",
    "using":     "sitting",
    "waiting":   "idle",
    "preparing": "idle",
    "idle":      "idle",
}

# Obstacle categories that affect navigation (superset of llm_costmap list)
_OBSTACLE_CLASSES = {
    "chair", "sofa", "armchair", "ottoman", "couch",
    "diningtable", "coffeetable", "sidetable", "table", "desk",
    "garbagecan", "houseplant", "floorlamp",
    "tv", "television",
    "cabinet", "bench",
}

# Edge relations that indicate a conversation group
_CONVERSATION_RELATIONS = {
    "talking_to", "talking with", "conversing", "conversation",
    "casual_chat", "chatting", "interacting_with",
}


# ---------------------------------------------------------------------------
# Coordinate transform helper
# ---------------------------------------------------------------------------

def _apply_transform(
    xyz: list[float],
    transform: np.ndarray | None,
) -> tuple[float, float]:
    """
    Project a 3D point from the HumanSSG frame to (x, y) in MuJoCo world frame.

    transform : 4×4 homogeneous matrix (HumanSSG → MuJoCo).  None = identity.
    The z axis is dropped after transformation (navigation is 2D).
    """
    if transform is None:
        return float(xyz[0]), float(xyz[1])

    p = np.array([xyz[0], xyz[1], xyz[2], 1.0], dtype=np.float64)
    p_out = transform @ p
    return float(p_out[0]), float(p_out[1])


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def scene_graph_to_scene_description(
    scene_graph_path: str | Path,
    coord_transform: np.ndarray | None = None,
    min_human_confidence: float = 0.0,
) -> SceneDescription:
    """
    Load a HumanSSG scene_graph.json and return a SceneDescription.

    Parameters
    ----------
    scene_graph_path :
        Path to the JSON produced by HumanSSG extract_json.py.
    coord_transform :
        Optional 4×4 numpy array mapping HumanSSG coordinates to MuJoCo world
        frame.  Provide this when the two systems are not already aligned.
        Pass None if they share the same coordinate frame.
    min_human_confidence :
        Reserved for future use (confidence filtering).

    Returns
    -------
    SceneDescription with humans, obstacles, and inferred conversation groups.
    """
    path = Path(scene_graph_path)
    with open(path) as f:
        graph = json.load(f)

    nodes: list[dict] = graph.get("nodes", [])
    edges: list = graph.get("edges", [])

    # ------------------------------------------------------------------
    # 1. Separate person nodes from obstacle nodes
    # ------------------------------------------------------------------
    person_nodes: dict[int, dict] = {}   # node_id → node
    obstacle_nodes: list[dict] = []

    for node in nodes:
        nid = node.get("node_id", -1)
        labels = [l.lower() for l in node.get("label", [])]
        primary = labels[0] if labels else ""

        if primary == "person":
            person_nodes[nid] = node
        elif primary in _OBSTACLE_CLASSES:
            obstacle_nodes.append(node)

    # ------------------------------------------------------------------
    # 2. Build HumanInfo list
    # ------------------------------------------------------------------
    humans: list[HumanInfo] = []
    node_id_to_human_idx: dict[int, int] = {}

    for nid, node in sorted(person_nodes.items()):
        bbox_center = node.get("bbox_center", [0.0, 0.0, 0.0])
        x, y = _apply_transform(bbox_center, coord_transform)

        # Heading: use stored value if available, otherwise 0
        yaw_deg = float(node.get("heading_deg", 0.0))

        # Determine state from activities list
        activities: list[dict] = node.get("activities", [])
        state = _infer_state(activities)

        idx = len(humans)
        humans.append(HumanInfo(pos=(x, y), yaw_deg=yaw_deg, state=state))
        node_id_to_human_idx[nid] = idx

    # ------------------------------------------------------------------
    # 3. Build ObstacleInfo list
    # ------------------------------------------------------------------
    obstacles: list[ObstacleInfo] = []
    for node in obstacle_nodes:
        bbox_center = node.get("bbox_center", [0.0, 0.0, 0.0])
        x, y = _apply_transform(bbox_center, coord_transform)
        label = node.get("label", ["unknown"])[0]
        obstacles.append(ObstacleInfo(category=label.capitalize(), pos=(x, y)))

    # ------------------------------------------------------------------
    # 4. Infer conversation groups from edges
    # ------------------------------------------------------------------
    groups: list[list[int]] = _infer_groups_from_edges(
        edges, node_id_to_human_idx
    )

    # Fallback: proximity + state heuristic for nodes with no edge data
    if not groups:
        groups = _infer_groups_from_proximity(humans)

    return SceneDescription(humans=humans, obstacles=obstacles, groups=groups)


def _infer_state(activities: list[dict]) -> str:
    """Pick the most socially-relevant state from an activity list."""
    if not activities:
        return "idle"

    # Priority: talking > walking > sitting > idle
    found: set[str] = set()
    for act in activities:
        name = act.get("name", "").lower()
        state = _ACTIVITY_TO_STATE.get(name, "idle")
        found.add(state)

    for priority_state in ("talking", "walking", "sitting", "idle"):
        if priority_state in found:
            return priority_state
    return "idle"


def _infer_groups_from_edges(
    edges: list,
    node_id_to_human_idx: dict[int, int],
) -> list[list[int]]:
    """Extract conversation groups from scene graph edges."""
    groups: list[list[int]] = []
    seen_pairs: set[frozenset] = set()

    for edge in edges:
        if len(edge) < 3:
            continue
        id1, id2 = int(edge[0]), int(edge[1])
        meta = edge[2] if isinstance(edge[2], dict) else {}

        relation = str(meta.get("relation", "") or meta.get("desc", "")).lower()
        is_convo = any(r in relation for r in _CONVERSATION_RELATIONS)
        if not is_convo:
            continue

        if id1 not in node_id_to_human_idx or id2 not in node_id_to_human_idx:
            continue

        h1 = node_id_to_human_idx[id1]
        h2 = node_id_to_human_idx[id2]
        pair = frozenset({h1, h2})
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        groups.append([h1, h2])

    return groups


def _infer_groups_from_proximity(
    humans: list[HumanInfo],
    max_dist: float = 1.5,
) -> list[list[int]]:
    """Fallback: pair two 'talking' humans if they are close enough."""
    groups: list[list[int]] = []
    used: set[int] = set()
    talkers = [i for i, h in enumerate(humans) if h.state == "talking"]

    for i in talkers:
        if i in used:
            continue
        for j in talkers:
            if j <= i or j in used:
                continue
            dx = humans[i].pos[0] - humans[j].pos[0]
            dy = humans[i].pos[1] - humans[j].pos[1]
            if math.hypot(dx, dy) < max_dist:
                groups.append([i, j])
                used.add(i)
                used.add(j)
    return groups


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def make_identity_transform() -> np.ndarray:
    """Return a 4×4 identity transform (no coordinate change)."""
    return np.eye(4, dtype=np.float64)


def make_axis_swap_transform(
    x_axis: str = "x",
    y_axis: str = "y",
    z_up: bool = True,
) -> np.ndarray:
    """
    Build a 4×4 transform that remaps HumanSSG axes to MuJoCo convention.

    Example: if HumanSSG uses (x=east, y=up, z=south) and MuJoCo uses
    (x=east, y=north, z=up), call:
        make_axis_swap_transform(x_axis='x', y_axis='-z')

    axis strings: 'x', 'y', 'z', '-x', '-y', '-z'
    """
    axis_map = {
        "x":  np.array([1, 0, 0]),
        "y":  np.array([0, 1, 0]),
        "z":  np.array([0, 0, 1]),
        "-x": np.array([-1, 0, 0]),
        "-y": np.array([0, -1, 0]),
        "-z": np.array([0, 0, -1]),
    }
    R = np.zeros((3, 3), dtype=np.float64)
    R[0] = axis_map[x_axis]
    R[1] = axis_map[y_axis]
    R[2] = axis_map["z"] if z_up else axis_map["-z"]

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    return T
