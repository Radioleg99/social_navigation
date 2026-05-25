"""
LLM Edge Social Cost Generator
------------------------------
SceneDescription (from pipeline/scene_bridge.py) → relation-edge scores
via a single LLM prompt → edge params → costmap / MPPI social cost.

Entry points:
    build_entity_params(scene, method="llm") → (params, llm_log)
    synthesize_costmap(params, grid_shape)   → np.ndarray  (viz / debug only)
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
from typing import NamedTuple

import numpy as np


# ---------------------------------------------------------------------------
# Scene data types  (mirrors scene_bridge — kept here to avoid heavy imports)
# ---------------------------------------------------------------------------

class HumanInfo(NamedTuple):
    pos: tuple[float, float]
    yaw_deg: float
    activities: list[str]      # e.g. ["SPEAK", "OBSERVE"]


class ObstacleInfo(NamedTuple):
    category: str
    pos: tuple[float, float]


class SceneDescription(NamedTuple):
    humans: list[HumanInfo]
    obstacles: list[ObstacleInfo]
    groups: list[list[int]]    # confirmed conversation groups


@dataclasses.dataclass
class LLMPromptTemplates:
    layer1_system: str
    layer2_system: str


# ---------------------------------------------------------------------------
# SocialEntityParams  (edge params consumed by SocialCost / synthesis)
# ---------------------------------------------------------------------------

class SocialEntityParams:
    __slots__ = ("entity_id", "pos", "yaw_deg",
                 "score", "personal_space", "orientation_sensitivity",
                 "sigma_perp", "reason",
                 "ref_dist", "base_score")

    def __init__(self, entity_id: str, pos: tuple[float, float], yaw_deg: float,
                 score: float, personal_space: float,
                 orientation_sensitivity: float = 1.0, reason: str = "",
                 sigma_perp: float | None = None,
                 ref_dist: float | None = None,
                 base_score: float | None = None) -> None:
        self.entity_id               = entity_id
        self.pos                     = pos
        self.yaw_deg                 = yaw_deg
        self.score                   = float(score)
        self.personal_space          = float(personal_space)
        self.orientation_sensitivity = float(orientation_sensitivity)
        self.sigma_perp              = float(sigma_perp) if sigma_perp is not None else None
        self.reason                  = reason
        self.ref_dist                = float(ref_dist) if ref_dist is not None else None
        self.base_score              = float(base_score) if base_score is not None else float(score)

    def __repr__(self) -> str:
        sp = f" sp={self.sigma_perp:.1f}m" if self.sigma_perp is not None else ""
        return (f"SocialEntityParams(id={self.entity_id}, pos={self.pos}, "
                f"score={self.score:.2f}, ps={self.personal_space:.1f}m, "
                f"os={self.orientation_sensitivity:.1f}x{sp})")


# ---------------------------------------------------------------------------
# Rule-based fallback  (no API needed, used as baseline)
# ---------------------------------------------------------------------------

_ACTIVITY_SCORE: dict[str, float] = {
    "speak": 0.9, "talk": 0.9, "talking": 0.9, "conversation": 0.9, "chat": 0.9,
    "observe": 0.7, "gesture": 0.7, "waving": 0.7,
    "sit": 0.6, "sitting": 0.6,
    "sleep": 0.35, "sleeping": 0.35,
    "walk": 0.7, "walking": 0.7, "running": 0.7,
    "idle": 0.5, "standing": 0.5, "standing_idle": 0.5,
}
_ACTIVITY_PS: dict[str, float] = {
    "speak": 1.2, "talk": 1.2, "talking": 1.2, "conversation": 1.2, "chat": 1.2,
    "observe": 1.0, "gesture": 1.0, "waving": 1.0,
    "sit": 0.9, "sitting": 0.9,
    "sleep": 0.7, "sleeping": 0.7,
    "walk": 1.1, "walking": 1.1, "running": 1.1,
    "idle": 0.8, "standing": 0.8, "standing_idle": 0.8,
}
_ACTIVITY_OS: dict[str, float] = {
    "speak": 2.5, "talk": 2.5, "talking": 2.5, "conversation": 2.5, "chat": 2.5,
    "observe": 2.0, "gesture": 1.8, "waving": 1.8,
    "sit": 1.5, "sitting": 1.5,
    "sleep": 1.1, "sleeping": 1.1,
    "walk": 1.8, "walking": 1.8, "running": 1.8,
    "idle": 1.3, "standing": 1.3, "standing_idle": 1.3,
}


def _activity_lookup(activities: list[str], table: dict[str, float], default: float) -> float:
    best = default
    for act in activities:
        v = table.get(act.lower())
        if v is not None:
            best = max(best, v)
    return best


def rule_based_entity_params(scene: SceneDescription) -> list[SocialEntityParams]:
    """Edge-only deterministic baseline.

    Single humans intentionally produce no social params. Proximity to unlinked
    humans is handled by the physical/hard human constraint, not social cost.
    """
    params: list[SocialEntityParams] = []
    for group in scene.groups:
        for k in range(len(group)):
            for l in range(k + 1, len(group)):
                i, j = group[k], group[l]
                if i >= len(scene.humans) or j >= len(scene.humans):
                    continue
                pi = scene.humans[i].pos
                pj = scene.humans[j].pos
                mx = (pi[0] + pj[0]) / 2.0
                my = (pi[1] + pj[1]) / 2.0
                dist = math.hypot(pj[0] - pi[0], pj[1] - pi[1])
                axis_yaw = math.degrees(math.atan2(pj[1] - pi[1], pj[0] - pi[0]))
                score_g = max(
                    _activity_lookup(scene.humans[i].activities, _ACTIVITY_SCORE, 0.5),
                    _activity_lookup(scene.humans[j].activities, _ACTIVITY_SCORE, 0.5),
                )
                params.append(SocialEntityParams(
                    entity_id=f"edge_human_{i}_human_{j}",
                    pos=(mx, my),
                    yaw_deg=axis_yaw,
                    score=score_g,
                    personal_space=max(0.3, dist / 2.0),
                    orientation_sensitivity=1.0,
                    sigma_perp=_EDGE_SP,
                    reason=f"rule-based social edge {i}-{j}",
                    ref_dist=dist,
                    base_score=score_g,
                ))

    return params


# ---------------------------------------------------------------------------
# Edge scoring prompts  —  LLM outputs relation scores only
# ---------------------------------------------------------------------------

_SOCIAL_REGION_SYSTEM = """\
<system>
You are a social cost scoring engine for a mobile robot navigating an indoor scene.

Social cost comes entirely from RELATIONS (edges) between nodes.
Single idle people with no edges have no social cost — that is handled by obstacle avoidance.
Your only job is to score existing edges by social interruption severity.
Do not output persons, regions, geometry, coordinates, radius, width, orientation, or shape.

<scoring>
Scores are relative to this scene. Compare all edges before assigning.
  low      0.10–0.35  minor preference to avoid
  medium   0.36–0.60  noticeable social cost
  high     0.61–0.85  strong social cost, avoid unless large detour required
  critical 0.86–1.00  almost never cross
Differentiate. Not everything can be high.
</scoring>

<edges>
human ↔ human: score = how disruptive it would be for the robot to pass through that interaction relation.
human → object: score = how disruptive it would be for the robot to interrupt that attention/task relation.
Only score edges that appear in the input. Never invent new edges.
For a 3-person group, include one edge per pair (A-B, B-C, A-C).
</edges>

<input_format>
{
  "nodes": [
    {"node_id": "human_0", "label": ["person"], "bbox_center": [x,y,z], "heading_deg": 0-360,
     "activities": [{"name": "SPEAK|OBSERVE|WALK|SIT|...", "object": "node_id or empty", "description": "free text"}]},
    {"node_id": "object_0", "label": ["tv|table|..."], "bbox_center": [x,y,z]}
  ],
  "edges": [["human_0","human_1",{"relation":"SPEAK|OBSERVE|NEAR|...","description":"text"}]],
  "robot": {"start":[x,y],"goal":[x,y]}
}
</input_format>

<output_format>
Output valid JSON only. No text outside the JSON block.
{
  "reasoning": "per-edge analysis then global comparison (newlines ok)",
  "edges": [
    {"a": "human_0", "b": "human_1", "score": 0.80, "reason": "one sentence"},
    {"a": "human_0", "b": "object_0", "score": 0.90, "reason": "one sentence"}
  ]
}
Rules:
- Only edges from the input. No extra fields.
- All scores in [0, 1].
- Do not output a persons array.
- Do not output social_regions.
- Do not output personal_space, attention_cone, f_formation, radius, width, orientation, or geometry.
</output_format>

<examples>

<example id="1">
<input>
{
  "nodes": [
    {"node_id": "human_0", "label": ["person"], "bbox_center": [2.0, 3.0, 0.9], "heading_deg": 90,
     "activities": [{"name": "OBSERVE", "object": "object_0", "description": "watching tv, world cup final, visibly tense and focused"}]},
    {"node_id": "human_1", "label": ["person"], "bbox_center": [5.0, 4.0, 0.9], "heading_deg": 270,
     "activities": [{"name": "SPEAK", "object": "human_2", "description": "casual chat with friend"}]},
    {"node_id": "human_2", "label": ["person"], "bbox_center": [6.2, 4.0, 0.9], "heading_deg": 90,
     "activities": [{"name": "SPEAK", "object": "human_1", "description": "casual chat with friend"}]},
    {"node_id": "object_0", "label": ["tv"], "bbox_center": [2.0, 7.0, 0.5]}
  ],
  "edges": [
    ["human_0", "object_0", {"relation": "OBSERVE", "description": "World Cup final"}],
    ["human_1", "human_2",  {"relation": "SPEAK",   "description": "casual chat"}]
  ],
  "robot": {"start": [0.0, 2.0], "goal": [8.0, 4.0]}
}
</input>
<output>
{
  "reasoning": "human_0→TV is a World Cup final sight line — irreplaceable, critical. human_1↔human_2 is casual chat, easily interrupted. Global: sight line dominates.",
  "edges": [
    {"a": "human_0", "b": "object_0", "score": 0.92, "reason": "World Cup final sight line, must not be interrupted"},
    {"a": "human_1", "b": "human_2",  "score": 0.28, "reason": "casual chat, low engagement"}
  ]
}
</output>
</example>

<example id="2">
<input>
{
  "nodes": [
    {"node_id": "human_0", "label": ["person"], "bbox_center": [1.0, 2.0, 0.9], "heading_deg": 0,
     "activities": [{"name": "SPEAK", "object": "human_1", "description": "presenting slides in a formal meeting"}]},
    {"node_id": "human_1", "label": ["person"], "bbox_center": [3.0, 1.5, 0.9], "heading_deg": 180,
     "activities": [{"name": "OBSERVE", "object": "human_0", "description": "listening to presentation, taking notes"}]},
    {"node_id": "human_2", "label": ["person"], "bbox_center": [3.0, 2.5, 0.9], "heading_deg": 180,
     "activities": [{"name": "OBSERVE", "object": "human_0", "description": "listening to presentation"}]}
  ],
  "edges": [
    ["human_0", "human_1", {"relation": "SPEAK", "description": "formal meeting"}],
    ["human_0", "human_2", {"relation": "SPEAK", "description": "formal meeting"}]
  ],
  "robot": {"start": [0.0, 5.0], "goal": [5.0, 0.0]}
}
</input>
<output>
{
  "reasoning": "Formal meeting. All edges are presenter↔listener — high disruption to cross. Both edges comparable in importance.",
  "edges": [
    {"a": "human_0", "b": "human_1", "score": 0.82, "reason": "formal meeting, crossing between presenter and listener is highly disruptive"},
    {"a": "human_0", "b": "human_2", "score": 0.79, "reason": "formal meeting, crossing between presenter and listener is highly disruptive"}
  ]
}
</output>
</example>

</examples>

</system>
"""


_SOCIAL_REGION_SYSTEM_FAST = """\
<system>
You are a social cost scoring engine for a mobile robot.
Social cost = relations only. Single idle people with no edges have no social cost.
Score only existing edges by social interruption severity.
No persons, no regions, no coordinates, no radius, no width, no orientation.

<scoring>
Relative to this scene. Differentiate.
low 0.10-0.35 / medium 0.36-0.60 / high 0.61-0.85 / critical 0.86-1.00
</scoring>

human↔human: cost to cross between them.
human→object: cost to cross their line of attention.
Only score edges from the input. No invented edges.
Do not output social_regions or a persons array.

<output_format>
Output valid JSON only.
{
  "reasoning": "brief per-edge then global",
  "edges": [
    {"a":"human_0","b":"human_1","score":0.80,"reason":"one sentence"},
    {"a":"human_0","b":"object_0","score":0.90,"reason":"one sentence"}
  ]
}
</output_format>
</system>
"""


def get_default_prompt_templates() -> LLMPromptTemplates:
    return LLMPromptTemplates(
        layer1_system=_SOCIAL_REGION_SYSTEM,
        layer2_system="",
    )


def build_social_region_prompt(
    scene_payload: dict,
    system_prompt: str | None = None,
) -> str:
    """Build edge-scoring prompt from scene payload.
    
    Args:
        scene_payload: Dictionary with HumanSSG-style nodes, edges, robot, and optional notes
        system_prompt: Optional custom system prompt (defaults to edge-only prompt)
    
    Returns:
        Complete prompt string ready for LLM
    """
    return (system_prompt or _SOCIAL_REGION_SYSTEM) + "\n\n" + json.dumps(
        scene_payload,
        ensure_ascii=False,
        indent=2,
    )


# Projection constants — fixed code-side, not exposed to the LLM
_EDGE_SP = 0.25   # sigma_perp for edge segment cost


def _parse_scored_scene(response: str) -> tuple[str, list[dict]]:
    """Parse LLM response into (reasoning, edges).

    edges: [{"a": "human_0", "b": "human_1", "score": float, "reason": str}]
    """
    start = response.find("{")
    end = response.rfind("}") + 1
    if start < 0 or end <= start:
        raise ValueError(f"No JSON in LLM response: {response[:200]}")
    data = json.loads(response[start:end])
    reasoning = str(data.get("reasoning", ""))

    edges = []
    for e in data.get("edges", []):
        a = str(e.get("a", "")).strip()
        b = str(e.get("b", "")).strip()
        if not a or not b:
            continue
        try:
            score = float(np.clip(float(e.get("score", 0.5)), 0.0, 1.0))
        except (TypeError, ValueError):
            score = 0.5
        edges.append({"a": a, "b": b, "score": score, "reason": str(e.get("reason", ""))})

    return reasoning, edges


def _scored_scene_to_entity_params(
    scene: SceneDescription,
    edges: list[dict],
) -> list[SocialEntityParams]:
    """Convert scored edges into internal edge params.

    Each edge is represented by midpoint, axis, half-length, score, and fixed
    lateral falloff. The LLM only supplies the score.
    """
    params: list[SocialEntityParams] = []
    for edge in edges:
        ea = _lookup_scene_entity(scene, edge["a"])
        eb = _lookup_scene_entity(scene, edge["b"])
        if ea is None or eb is None:
            continue
        ia, ib = ea[1], eb[1]
        pos_a = np.array(ea[2], dtype=np.float64)
        pos_b = np.array(eb[2], dtype=np.float64)
        d = pos_b - pos_a
        dist = float(np.linalg.norm(d))
        mx, my = float((pos_a[0] + pos_b[0]) / 2), float((pos_a[1] + pos_b[1]) / 2)
        yaw_axis = math.degrees(math.atan2(float(d[1]), float(d[0])))
        # entity_id encodes both node indices so geometry refresh can update it
        a_prefix = ea[0]   # "human" or "object"
        b_prefix = eb[0]
        group_id = f"edge_{a_prefix}_{ia}_{b_prefix}_{ib}"
        score = edge["score"]
        params.append(SocialEntityParams(
            entity_id=group_id, pos=(mx, my), yaw_deg=yaw_axis,
            score=score, personal_space=max(0.3, dist / 2.0),
            orientation_sensitivity=1.0, sigma_perp=_EDGE_SP,
            reason=edge["reason"],
            ref_dist=dist, base_score=score,
        ))
    return params


# ---------------------------------------------------------------------------
# Scene → edge payload → SocialEntityParams
# ---------------------------------------------------------------------------

def _scene_to_social_payload(
    scene: SceneDescription,
    robot_pos: tuple[float, float] | None = None,
    robot_goal: tuple[float, float] | None = None,
    notes: str | None = None,
) -> dict:
    def activity_to_dict(raw: str) -> dict:
        name, _, _ = str(raw).partition(":")
        name = name.strip().split(" ", 1)[0].upper() or "ACTIVITY"
        return {
            "name": name,
            "object": "",
            "description": str(raw),
        }

    nodes: list[dict] = []
    for i, h in enumerate(scene.humans):
        activities = [activity_to_dict(act) for act in h.activities] if h.activities else [
            {"name": "IDLE", "object": "", "description": "idle"}
        ]
        nodes.append({
            "node_id": f"human_{i}",
            "label": ["person"],
            "bbox_center": [float(h.pos[0]), float(h.pos[1]), 0.0],
            "heading_deg": float(h.yaw_deg),
            "activities": activities,
        })

    for i, o in enumerate(scene.obstacles):
        nodes.append({
            "node_id": f"object_{i}",
            "label": [str(o.category).lower()],
            "bbox_center": [float(o.pos[0]), float(o.pos[1]), 0.0],
        })

    edges: list[list[object]] = []
    for group in scene.groups:
        members = [f"human_{idx}" for idx in group if 0 <= idx < len(scene.humans)]
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                edges.append([
                    members[i],
                    members[j],
                    {
                        "relation": "CONVERSATION",
                        "desc": "confirmed conversation group",
                        "description": "these people are interacting; crossing their relation edge is disruptive",
                    },
                ])

    payload: dict[str, object] = {
        "nodes": nodes,
        "edges": edges,
    }
    robot: dict[str, list[float]] = {}
    if robot_pos is not None:
        robot["start"] = [float(robot_pos[0]), float(robot_pos[1])]
    if robot_goal is not None:
        robot["goal"] = [float(robot_goal[0]), float(robot_goal[1])]
    if robot:
        payload["robot"] = robot
    if notes:
        payload["notes"] = notes
    return payload


def _lookup_scene_entity(scene: SceneDescription, target_id: str) -> tuple[str, int, tuple[float, float], float] | None:
    prefix, _, index_str = target_id.partition("_")
    if not index_str.isdigit():
        return None
    index = int(index_str)
    if prefix in {"human", "person"}:
        if 0 <= index < len(scene.humans):
            h = scene.humans[index]
            return ("human", index, h.pos, h.yaw_deg)
        return None
    if prefix in {"object", "obstacle"}:
        if 0 <= index < len(scene.obstacles):
            o = scene.obstacles[index]
            return ("object", index, o.pos, 0.0)
        return None
    return None


def _format_scored_log(
    reasoning: str,
    edges: list[dict],
    params: list[SocialEntityParams],
) -> str:
    lines = ["=== LLM REASONING ===", reasoning, "", "=== EDGES ==="]
    for e in edges:
        lines.append(f"  {e['a']} ↔ {e['b']}  score={e['score']:.2f}")
        lines.append(f"    {e['reason']}")
    lines += ["", "=== ENTITY PARAMS ==="]
    for p in params:
        lines.append(f"  {p.entity_id}  score={p.score:.2f}  ps={p.personal_space:.1f}m")
        lines.append(f"    {p.reason}")
    return "\n".join(lines)


def _print_scored_reasoning(
    reasoning: str,
    edges: list[dict],
    params: list[SocialEntityParams],
) -> None:
    sep = "─" * 60
    print(f"\n{sep}\nLLM REASONING\n{sep}")
    print(reasoning)
    print(f"\n{sep}\nEDGES\n{sep}")
    for e in edges:
        print(f"  {e['a']} ↔ {e['b']}  score={e['score']:.2f}")
        print(f"    → {e['reason']}")
    print(f"\n{sep}\nPER-ENTITY PARAMS\n{sep}")
    for p in params:
        print(f"  {p.entity_id}  score={p.score:.2f}  ps={p.personal_space:.1f}m")
        print(f"    → {p.reason}")
    print(sep + "\n")


class SocialCostOrchestrator:
    """Single-layer edge scoring pipeline."""

    def __init__(
        self,
        model: str,
        verbose: bool = True,
        prompt_templates: LLMPromptTemplates | None = None,
    ) -> None:
        self._model = model
        self._verbose = verbose
        self._system_prompt = (
            prompt_templates.layer1_system
            if prompt_templates is not None and prompt_templates.layer1_system.strip()
            else _SOCIAL_REGION_SYSTEM
        )

    def update(
        self,
        scene: SceneDescription,
        robot_pos: tuple[float, float] | None = None,
        robot_goal: tuple[float, float] | None = None,
    ) -> tuple[list[SocialEntityParams], str]:
        payload = _scene_to_social_payload(scene, robot_pos, robot_goal)
        resp = _call_llm(build_social_region_prompt(payload, self._system_prompt), self._model)
        reasoning, edges = _parse_scored_scene(resp)
        if not edges:
            params = rule_based_entity_params(scene)
            llm_log = "=== SCORED SCENE ===\n  (no edges parsed; fell back to rule-based params)"
            if self._verbose:
                print("[llm_costmap] no edges parsed, using rule-based fallback")
            return params, llm_log
        params = _scored_scene_to_entity_params(scene, edges)
        llm_log = _format_scored_log(reasoning, edges, params)

        if self._verbose:
            _print_scored_reasoning(reasoning, edges, params)

        return params, llm_log

    def reset_cache(self) -> None:
        pass


def build_entity_params(
    scene: SceneDescription,
    method: str = "llm",
    llm_model: str = "doubao-pro-32k",
    verbose: bool = True,
    robot_pos: tuple[float, float] | None = None,
    robot_goal: tuple[float, float] | None = None,
    prompt_templates: LLMPromptTemplates | None = None,
) -> tuple[list[SocialEntityParams], str]:
    """SceneDescription → per-entity SocialEntityParams + llm_log.

    method="rule" : no API, rule-based baseline
    method="llm"  : edge scoring pipeline (needs API key in .env)
    """
    if method == "rule":
        return rule_based_entity_params(scene), ""
    if method == "llm":
        orc = SocialCostOrchestrator(llm_model, verbose, prompt_templates=prompt_templates)
        return orc.update(scene, robot_pos, robot_goal)
    raise ValueError(f"Unknown method: {method!r}")


# ---------------------------------------------------------------------------
# Edge cost synthesis  (resolution-independent, for viz / A* heuristic only)
# ---------------------------------------------------------------------------

def synthesize_costmap(
    params:             list[SocialEntityParams],
    grid_shape:         tuple[int, int],
    x_range:            tuple[float, float] | None = None,
    y_range:            tuple[float, float] | None = None,
    distance_transform: np.ndarray | None = None,
    clearance_cap:      float = 0.8,
    clearance_weight:   float = 0.4,
    back_scale:         float = 1.0,
) -> np.ndarray:
    """
    合成 social costmap。

    若传入 distance_transform（每格到最近障碍的距离，单位米），
    则叠加 clearance cost：离墙越近代价越高。
    这样 A* 会自动选人旁边更开阔的一侧，而非贴墙钻缝。

    clearance_cost = clearance_weight × (1 - clip(dt / clearance_cap, 0, 1))
    combined       = clip(social + clearance_cost, 0, 1)
    """
    if x_range is None or y_range is None:
        if params:
            xs = [p.pos[0] for p in params]
            ys = [p.pos[1] for p in params]
            margin = 2.0
            x_range = x_range or (min(xs) - margin, max(xs) + margin)
            y_range = y_range or (min(ys) - margin, max(ys) + margin)
        else:
            x_range, y_range = (-5.0, 5.0), (-5.0, 5.0)

    H, W = grid_shape
    # Standard top-down grid convention used by scene_map and A*:
    # row 0 = y_max, row H-1 = y_min; col 0 = x_min, col W-1 = x_max.
    rows_y = np.linspace(y_range[1], y_range[0], H)
    cols_x = np.linspace(x_range[0], x_range[1], W)
    GX, GY = np.meshgrid(cols_x, rows_y)   # GX[i,j]=x, GY[i,j]=y

    field = np.zeros((H, W), dtype=np.float32)
    for ep in params:
        hx, hy  = ep.pos
        yaw_rad = math.radians(ep.yaw_deg)
        fx, fy  = math.cos(yaw_rad), math.sin(yaw_rad)
        ps = ep.personal_space
        os = ep.orientation_sensitivity
        if str(ep.entity_id).startswith("edge_"):
            dx = GX - hx
            dy = GY - hy
            along = dx * fx + dy * fy
            perp = -dx * fy + dy * fx
            half_len = max(ps, 1e-3)
            excess = np.maximum(np.abs(along) - half_len, 0.0)
            sigma = ep.sigma_perp if ep.sigma_perp is not None else _EDGE_SP
            cost = ep.score * np.exp(-(perp ** 2 + excess ** 2) / (2 * sigma ** 2))
            field = np.maximum(field, cost.astype(np.float32))
            continue

        # For directed attention (os > 1), shift the Gaussian centre forward so
        # the ellipse covers the full person→target corridor rather than just the
        # area around the person.  Shift = half the asymmetric span.
        if os > 1.0:
            shift = ps * (os - back_scale) / 2.0
            hx += shift * fx
            hy += shift * fy
        dx = GX - hx
        dy = GY - hy
        along = dx * fx + dy * fy
        perp  = -dx * fy + dy * fx
        sigma_along = np.where(along >= 0, ps * os, ps * back_scale)
        sigma_side  = ep.sigma_perp if ep.sigma_perp is not None else ps
        cost = ep.score * np.exp(
            -(along ** 2) / (2 * sigma_along ** 2)
            -(perp  ** 2) / (2 * sigma_side ** 2)
        )
        field = np.maximum(field, cost.astype(np.float32))

    # 叠加 clearance cost（离墙近 → 代价高）
    if distance_transform is not None:
        dt = distance_transform.astype(np.float32)
        clearance_cost = clearance_weight * (1.0 - np.clip(dt / clearance_cap, 0.0, 1.0))
        field = np.clip(field + clearance_cost, 0.0, 1.0)

    return field


# ---------------------------------------------------------------------------
# LLM backend
# ---------------------------------------------------------------------------

def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        from pathlib import Path
        env_path = Path(__file__).resolve().parents[3] / ".env"
        if env_path.exists():
            load_dotenv(env_path)
    except ImportError:
        pass


def _call_llm(prompt: str, model: str) -> str:
    _load_env()
    if model.startswith("claude"):
        return _call_anthropic(prompt, model)
    if model.lower().startswith(("doubao", "ep-")):
        return _call_openai_compat(prompt, model,
                                   base_url="https://ark.cn-beijing.volces.com/api/v3",
                                   api_key=os.environ["ARK_API_KEY"])
    if model.lower().startswith(("kimi", "moonshot")):
        api_key = os.environ.get("KIMI_API_KEY") or os.environ.get("VLM_API_KEY", "")
        return _call_openai_compat(prompt, model,
                                   base_url="https://api.moonshot.cn/v1",
                                   api_key=api_key)
    return _call_openai_compat(prompt, model,
                               base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                               api_key=os.environ["OPENAI_API_KEY"])


def _call_anthropic(prompt: str, model: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model=model, max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def _call_openai_compat(prompt: str, model: str, base_url: str, api_key: str,
                        max_tokens: int = 4096) -> str:
    import time
    from openai import OpenAI, RateLimitError
    client = OpenAI(api_key=api_key, base_url=base_url)
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except RateLimitError:
            wait = 2 ** attempt * 5
            print(f"[llm] 429 rate limit, retry {attempt+1}/5 in {wait}s ...")
            time.sleep(wait)
    raise RuntimeError("LLM API rate limit: 5 retries exhausted")


# ---------------------------------------------------------------------------
# Live simulation interface  (interactive_sim, takes Agent dict from sim_world)
# ---------------------------------------------------------------------------

def build_live_costmap(
    agents: dict,
    grid_shape: tuple[int, int],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    method: str = "rule",
    groups: list[list[str]] | None = None,
    robot_pos: np.ndarray | tuple[float, float] | None = None,
    robot_goal: np.ndarray | tuple[float, float] | None = None,
    llm_model: str = "moonshot-v1-8k",
    prompt_templates: LLMPromptTemplates | None = None,
    verbose: bool = False,
    **_: object,
) -> tuple[np.ndarray, list[SocialEntityParams]]:
    """Build a live 2D social costmap from runtime agent objects.

    This is the single costmap entry point for lightweight Stage2 demos.
    Runtime agents only need ``pos``, ``heading_deg`` and ``activity`` fields.
    LLM mode intentionally falls back to the deterministic rule model if the
    API call fails, so interactive debugging remains responsive without keys.
    """
    groups = groups or []
    if not agents:
        return np.zeros(grid_shape, dtype=np.float32), []

    agent_ids = list(agents.keys())
    id_to_idx = {aid: i for i, aid in enumerate(agent_ids)}
    scene_groups = [
        [id_to_idx[aid] for aid in group if aid in id_to_idx]
        for group in groups
    ]
    scene_groups = [group for group in scene_groups if len(group) >= 2]
    scene = SceneDescription(
        humans=[
            HumanInfo(
                pos=(float(ag.pos[0]), float(ag.pos[1])),
                yaw_deg=float(getattr(ag, "heading_deg", 0.0)),
                activities=[str(getattr(ag, "activity", "standing")).upper()],
            )
            for ag in agents.values()
        ],
        obstacles=[],
        groups=scene_groups,
    )

    if method == "rule":
        params = rule_based_entity_params(scene)
    elif method == "llm":
        try:
            params, _log = build_entity_params(
                scene,
                method="llm",
                llm_model=llm_model,
                verbose=verbose,
                robot_pos=None if robot_pos is None else (float(robot_pos[0]), float(robot_pos[1])),
                robot_goal=None if robot_goal is None else (float(robot_goal[0]), float(robot_goal[1])),
                prompt_templates=prompt_templates,
            )
        except Exception as exc:
            params = rule_based_entity_params(scene)
            for p in params:
                p.reason = f"rule fallback after LLM error: {str(exc)[:80]}"
    else:
        raise ValueError(f"Live costmap: method {method!r} not supported")

    for i, aid in enumerate(agent_ids):
        prefix = f"person_{i}"
        for p in params:
            if p.entity_id == prefix:
                p.entity_id = aid

    cm = synthesize_costmap(params, grid_shape, x_range=x_range, y_range=y_range)
    return cm, params
