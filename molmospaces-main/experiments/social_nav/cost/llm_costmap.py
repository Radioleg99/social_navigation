"""
LLM Social Costmap Generator
-----------------------------
SceneDescription (from pipeline/scene_bridge.py) → social cost regions
via a single LLM prompt → Gaussian synthesis → costmap.

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
class SocialCostRegion:
    region_id: str
    geometry_type: str
    targets: list[str]
    priority: str
    parameters: dict
    reason: str


@dataclasses.dataclass
class LLMPromptTemplates:
    layer1_system: str
    layer2_system: str


# ---------------------------------------------------------------------------
# SocialEntityParams  (output type consumed by SocialCost / Gaussian synthesis)
# ---------------------------------------------------------------------------

class SocialEntityParams:
    __slots__ = ("entity_id", "pos", "yaw_deg",
                 "score", "personal_space", "orientation_sensitivity",
                 "sigma_perp", "reason")

    def __init__(self, entity_id: str, pos: tuple[float, float], yaw_deg: float,
                 score: float, personal_space: float,
                 orientation_sensitivity: float, reason: str = "",
                 sigma_perp: float | None = None) -> None:
        self.entity_id               = entity_id
        self.pos                     = pos
        self.yaw_deg                 = yaw_deg
        self.score                   = float(score)
        self.personal_space          = float(personal_space)
        self.orientation_sensitivity = float(orientation_sensitivity)
        self.sigma_perp              = float(sigma_perp) if sigma_perp is not None else None
        self.reason                  = reason

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
    "walk": 0.7, "walking": 0.7, "running": 0.7,
    "idle": 0.5, "standing": 0.5, "standing_idle": 0.5,
}
_ACTIVITY_PS: dict[str, float] = {
    "speak": 1.2, "talk": 1.2, "talking": 1.2, "conversation": 1.2, "chat": 1.2,
    "observe": 1.0, "gesture": 1.0, "waving": 1.0,
    "sit": 0.9, "sitting": 0.9,
    "walk": 1.1, "walking": 1.1, "running": 1.1,
    "idle": 0.8, "standing": 0.8, "standing_idle": 0.8,
}
_ACTIVITY_OS: dict[str, float] = {
    "speak": 2.5, "talk": 2.5, "talking": 2.5, "conversation": 2.5, "chat": 2.5,
    "observe": 2.0, "gesture": 1.8, "waving": 1.8,
    "sit": 1.5, "sitting": 1.5,
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
    # 预计算对话组内的朝向修正（人互相对视）
    pair_yaw: dict[int, float] = {}
    for group in scene.groups:
        for k in range(len(group)):
            for l in range(k + 1, len(group)):
                i, j = group[k], group[l]
                if i >= len(scene.humans) or j >= len(scene.humans):
                    continue
                pi = scene.humans[i].pos
                pj = scene.humans[j].pos
                pair_yaw[i] = math.degrees(math.atan2(pj[1] - pi[1], pj[0] - pi[0]))
                pair_yaw[j] = math.degrees(math.atan2(pi[1] - pj[1], pi[0] - pj[0]))

    params: list[SocialEntityParams] = []
    for i, h in enumerate(scene.humans):
        acts = h.activities
        score = _activity_lookup(acts, _ACTIVITY_SCORE, 0.5)
        ps = _activity_lookup(acts, _ACTIVITY_PS, 0.9)
        yaw = pair_yaw.get(i, h.yaw_deg)
        params.append(SocialEntityParams(
            entity_id=f"person_{i}",
            pos=h.pos,
            yaw_deg=yaw,
            score=score,
            personal_space=ps,
            orientation_sensitivity=_activity_lookup(acts, _ACTIVITY_OS, 1.5),
            reason=f"rule-based: {acts}",
        ))

    # 添加对话组群组中点实体
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
                ps_ind = _activity_lookup(scene.humans[i].activities, _ACTIVITY_PS, 0.9)
                params.append(SocialEntityParams(
                    entity_id=f"group_{i}_{j}",
                    pos=(mx, my),
                    yaw_deg=axis_yaw,
                    score=score_g,
                    personal_space=dist / 2.0 + ps_ind * 0.5,
                    orientation_sensitivity=1.0,
                    sigma_perp=ps_ind * 0.7,
                    reason=f"rule-based group midpoint {i}-{j}",
                ))

    return params


# ---------------------------------------------------------------------------
# Social Region System Prompt (new single-layer approach)
# ---------------------------------------------------------------------------

_SOCIAL_REGION_SYSTEM = """\
<system>
<role>
You are a social cost region generator for a mobile robot navigating an indoor top-down map.

Your task is NOT to generate a numeric costmap directly.
Your task is to identify socially sensitive regions and describe them using a small set of geometric primitives.

The robot will use these regions to build a 2D social costmap for A* planning.
</role>

<core_principle>
Do not invent new geometry types.
Do not output dense grids.
Do not output free-form text outside JSON.

You should reason about social norms, but express the result only through the supported geometric primitives.
</core_principle>

<supported_geometry_primitives>

1. around_entity
Use when the robot should avoid getting too close to one person or object.
Typical use: personal space around a human.

Required fields:
- targets: one entity id
- radius: meters

2. between_entities
Use when the robot should avoid crossing the space between two entities.
Typical use: two people talking, or a person interacting with an object.

Required fields:
- targets: two entity ids
- width: meters

3. front_sector
Use when the robot should avoid passing directly in front of a person.
Typical use: a person looking, watching, talking, working, or facing a direction.

Required fields:
- targets: one human id
- radius: meters
- angle_deg: degrees

4. near_region
Use when the robot should avoid a functional area around an entity.
Typical use: kitchen counter, desk, TV area, doorway, narrow working area.

Required fields:
- targets: one or more entity ids
- radius: meters

</supported_geometry_primitives>

<priority_scale>
Use qualitative priority only.

low:
  weak social preference. Robot may pass through if needed.

medium:
  noticeable social preference. Robot should prefer avoiding it.

high:
  strong social preference. Robot should avoid it unless path becomes much longer.

critical:
  very strong social preference. Robot should almost never pass through it unless no alternative exists.

Do not output numeric cost.
</priority_scale>

<decision_guidelines>
- Every human should usually have at least one around_entity region unless clearly irrelevant.
- If a person has a clear facing direction and is engaged in an activity, consider front_sector.
- If two people are speaking, facing each other, or marked as a group, consider between_entities.
- If a person is interacting with an object, consider between_entities or near_region.
- If a person is idle, do not over-penalize the area; use low or medium priority.
- If a person is talking, watching, working, cooking, or otherwise engaged, use higher priority.
- Prefer fewer high-quality regions over many redundant regions.
- Avoid duplicating the same social meaning with multiple overlapping regions unless necessary.
</decision_guidelines>

<input_format>
The input contains:
- humans: id, position, yaw, activity
- objects: id, category, position
- groups: confirmed human-human interaction groups
- robot: start and goal
- optional notes
</input_format>

<output_format>
Output valid JSON only.

{
  "social_cost_regions": [
    {
      "region_id": "r1",
      "geometry_type": "around_entity | between_entities | front_sector | near_region",
      "targets": ["entity_id"],
      "priority": "low | medium | high | critical",
      "parameters": {
        "radius": 1.0,
        "width": 0.8,
        "angle_deg": 90
      },
      "reason": "short reason, max 20 words"
    }
  ]
}
</output_format>

<examples>

<example>
<input>
{
  "humans": [
    {"id": "human_0", "position": [2.0, 2.0], "yaw_deg": 90, "activity": "watching_tv"}
  ],
  "objects": [
    {"id": "tv_0", "category": "tv", "position": [2.0, 5.0]}
  ],
  "groups": [],
  "robot": {"start": [0.5, 0.5], "goal": [4.5, 5.5]}
}
</input>
<output>
{
  "social_cost_regions": [
    {
      "region_id": "r1",
      "geometry_type": "around_entity",
      "targets": ["human_0"],
      "priority": "medium",
      "parameters": {"radius": 0.9},
      "reason": "basic personal space around the seated person"
    },
    {
      "region_id": "r2",
      "geometry_type": "between_entities",
      "targets": ["human_0", "tv_0"],
      "priority": "critical",
      "parameters": {"width": 0.8},
      "reason": "robot should not cross the person's viewing space"
    },
    {
      "region_id": "r3",
      "geometry_type": "front_sector",
      "targets": ["human_0"],
      "priority": "high",
      "parameters": {"radius": 1.5, "angle_deg": 90},
      "reason": "passing directly in front may interrupt attention"
    }
  ]
}
</output>
</example>

<example>
<input>
{
  "humans": [
    {"id": "human_0", "position": [1.0, 2.0], "yaw_deg": 0, "activity": "speaking"},
    {"id": "human_1", "position": [2.5, 2.0], "yaw_deg": 180, "activity": "listening"}
  ],
  "objects": [],
  "groups": [["human_0", "human_1"]],
  "robot": {"start": [0.0, 2.0], "goal": [4.0, 2.0]}
}
</input>
<output>
{
  "social_cost_regions": [
    {
      "region_id": "r1",
      "geometry_type": "around_entity",
      "targets": ["human_0"],
      "priority": "medium",
      "parameters": {"radius": 0.9},
      "reason": "basic personal space around a speaking person"
    },
    {
      "region_id": "r2",
      "geometry_type": "around_entity",
      "targets": ["human_1"],
      "priority": "medium",
      "parameters": {"radius": 0.9},
      "reason": "basic personal space around a listening person"
    },
    {
      "region_id": "r3",
      "geometry_type": "between_entities",
      "targets": ["human_0", "human_1"],
      "priority": "critical",
      "parameters": {"width": 0.9},
      "reason": "robot should not pass through an active conversation"
    }
  ]
}
</output>
</example>

</examples>
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
    """Build social region prompt from scene payload.
    
    Args:
        scene_payload: Dictionary with humans, objects, groups, robot, and optional notes
        system_prompt: Optional custom system prompt (defaults to _SOCIAL_REGION_SYSTEM)
    
    Returns:
        Complete prompt string ready for LLM
    """
    return (system_prompt or _SOCIAL_REGION_SYSTEM) + "\n\n" + json.dumps(
        scene_payload,
        ensure_ascii=False,
        indent=2,
    )


def _parse_social_regions(response: str) -> list[SocialCostRegion]:
    """Parse LLM response into list of SocialCostRegion objects.
    
    Extracts JSON from response, validates geometry types and priorities,
    and returns only valid regions.
    
    Args:
        response: LLM response string (may contain extra text)
    
    Returns:
        List of validated SocialCostRegion objects
    
    Raises:
        ValueError: If no valid JSON found in response
    """
    start = response.find("{")
    end = response.rfind("}") + 1
    if start < 0 or end <= start:
        raise ValueError(f"No JSON in LLM response: {response[:200]}")

    data = json.loads(response[start:end])
    regions = []

    allowed_geometry = {"around_entity", "between_entities", "front_sector", "near_region"}
    allowed_priority = {"low", "medium", "high", "critical"}

    for r in data.get("social_cost_regions", []):
        try:
            geometry_type = str(r["geometry_type"])
            priority = str(r["priority"])
            targets = list(r["targets"])
        except (KeyError, TypeError, ValueError):
            continue

        if geometry_type not in allowed_geometry:
            continue
        if priority not in allowed_priority:
            continue

        region_id = str(r.get("region_id", f"r{len(regions) + 1}"))
        parameters = r.get("parameters", {})
        if not isinstance(parameters, dict):
            parameters = {}

        regions.append(SocialCostRegion(
            region_id=region_id,
            geometry_type=geometry_type,
            targets=[str(target) for target in targets],
            priority=priority,
            parameters=dict(parameters),
            reason=str(r.get("reason", "")),
        ))

    return regions

# ---------------------------------------------------------------------------
# Scene → social regions → SocialEntityParams
# ---------------------------------------------------------------------------

def get_default_prompt_templates() -> LLMPromptTemplates:
    return LLMPromptTemplates(
        layer1_system=_SOCIAL_REGION_SYSTEM,
        layer2_system="",
    )


def _scene_to_social_payload(
    scene: SceneDescription,
    robot_pos: tuple[float, float] | None = None,
    robot_goal: tuple[float, float] | None = None,
    notes: str | None = None,
) -> dict:
    humans = [
        {
            "id": f"human_{i}",
            "position": [float(h.pos[0]), float(h.pos[1])],
            "yaw_deg": float(h.yaw_deg),
            "activity": "_".join(act.lower() for act in h.activities) if h.activities else "idle",
        }
        for i, h in enumerate(scene.humans)
    ]
    objects = [
        {
            "id": f"object_{i}",
            "category": o.category,
            "position": [float(o.pos[0]), float(o.pos[1])],
        }
        for i, o in enumerate(scene.obstacles)
    ]
    groups = [[f"human_{idx}" for idx in group if 0 <= idx < len(scene.humans)] for group in scene.groups]
    groups = [group for group in groups if len(group) >= 2]
    payload: dict[str, object] = {
        "humans": humans,
        "objects": objects,
        "groups": groups,
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


def _priority_to_score(priority: str) -> float:
    return {"low": 0.25, "medium": 0.50, "high": 0.78, "critical": 0.95}[priority]


def _default_radius(priority: str) -> float:
    return {"low": 0.7, "medium": 0.9, "high": 1.2, "critical": 1.5}[priority]


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


def _merge_social_entity_params(base: SocialEntityParams, other: SocialEntityParams) -> SocialEntityParams:
    base.score = max(base.score, other.score)
    base.personal_space = max(base.personal_space, other.personal_space)
    base.orientation_sensitivity = max(base.orientation_sensitivity, other.orientation_sensitivity)
    if other.sigma_perp is not None:
        base.sigma_perp = other.sigma_perp if base.sigma_perp is None else max(base.sigma_perp, other.sigma_perp)
    if other.reason and other.reason not in base.reason:
        base.reason = other.reason if not base.reason else base.reason
    return base


def _regions_to_entity_params(scene: SceneDescription, regions: list[SocialCostRegion]) -> list[SocialEntityParams]:
    params_by_id: dict[str, SocialEntityParams] = {}
    extra_params: list[SocialEntityParams] = []

    for region in regions:
        score = _priority_to_score(region.priority)
        radius = float(region.parameters.get("radius", _default_radius(region.priority)))
        width = float(region.parameters.get("width", radius))
        angle_deg = float(region.parameters.get("angle_deg", 90.0))
        targets = [str(target) for target in region.targets]

        if region.geometry_type == "between_entities" and len(targets) >= 2:
            first = _lookup_scene_entity(scene, targets[0])
            second = _lookup_scene_entity(scene, targets[1])
            if first is None or second is None:
                continue
            x = (first[2][0] + second[2][0]) / 2.0
            y = (first[2][1] + second[2][1]) / 2.0
            yaw = math.degrees(math.atan2(second[2][1] - first[2][1], second[2][0] - first[2][0]))
            extra_params.append(SocialEntityParams(
                entity_id=region.region_id,
                pos=(x, y),
                yaw_deg=yaw,
                score=score,
                personal_space=max(width, 0.6),
                orientation_sensitivity=1.0,
                sigma_perp=max(0.45, width * 0.7),
                reason=region.reason,
            ))
            continue

        if region.geometry_type == "near_region" and len(targets) > 1:
            lookup = [_lookup_scene_entity(scene, target) for target in targets]
            lookup = [item for item in lookup if item is not None]
            if not lookup:
                continue
            x = sum(item[2][0] for item in lookup) / len(lookup)
            y = sum(item[2][1] for item in lookup) / len(lookup)
            yaw = lookup[0][3]
            extra_params.append(SocialEntityParams(
                entity_id=region.region_id,
                pos=(x, y),
                yaw_deg=yaw,
                score=score,
                personal_space=radius,
                orientation_sensitivity=1.0,
                sigma_perp=max(0.5, radius),
                reason=region.reason,
            ))
            continue

        for target in targets[:1]:
            entity = _lookup_scene_entity(scene, target)
            if entity is None:
                continue
            kind, index, pos, yaw = entity
            entity_id = f"person_{index}" if kind == "human" else f"object_{index}"
            orientation_sensitivity = 1.0
            sigma_perp = None
            if region.geometry_type == "front_sector":
                orientation_sensitivity = 1.8
                sigma_perp = max(0.45, radius * 0.5)
            elif region.geometry_type == "near_region":
                sigma_perp = max(0.5, radius * 0.9)
            elif region.geometry_type == "around_entity":
                sigma_perp = max(0.5, radius * 0.85)

            new_param = SocialEntityParams(
                entity_id=entity_id,
                pos=pos,
                yaw_deg=yaw,
                score=score,
                personal_space=radius,
                orientation_sensitivity=orientation_sensitivity,
                sigma_perp=sigma_perp,
                reason=region.reason,
            )
            if entity_id in params_by_id:
                params_by_id[entity_id] = _merge_social_entity_params(params_by_id[entity_id], new_param)
            else:
                params_by_id[entity_id] = new_param

    return list(params_by_id.values()) + extra_params


def _format_log(regions: list[SocialCostRegion], params: list[SocialEntityParams]) -> str:
    lines = ["=== SOCIAL COST REGIONS ==="]
    for region in regions:
        lines.append(f"  [{region.region_id}] {region.geometry_type}  priority={region.priority}")
        lines.append(f"    targets={region.targets}")
        lines.append(f"    {region.reason}")
    lines += ["", "=== ENTITY PARAMS ==="]
    for p in params:
        lines.append(f"  {p.entity_id}  score={p.score:.2f}  ps={p.personal_space:.1f}m  os={p.orientation_sensitivity:.1f}x")
        lines.append(f"    {p.reason}")
    return "\n".join(lines)


def _print_reasoning(regions: list[SocialCostRegion], params: list[SocialEntityParams]) -> None:
    sep = "─" * 60
    print(f"\n{sep}")
    print("SOCIAL COST REGIONS")
    print(sep)
    for region in regions:
        print(f"  [{region.region_id}] {region.geometry_type}  priority={region.priority}")
        print(f"    targets={region.targets}")
        print(f"    {region.reason}")

    print(f"\n{sep}")
    print("PER-ENTITY PARAMS")
    print(sep)
    for p in params:
        print(f"  {p.entity_id}  score={p.score:.2f}  ps={p.personal_space:.1f}m  os={p.orientation_sensitivity:.1f}x")
        print(f"    → {p.reason}")
    print(sep + "\n")


class SocialCostOrchestrator:
    """Single-layer social cost pipeline based on region generation."""

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
        regions = _parse_social_regions(resp)
        if not regions:
            params = rule_based_entity_params(scene)
            llm_log = "=== SOCIAL COST REGIONS ===\n  (none parsed; fell back to rule-based params)"
            if self._verbose:
                print("[llm_costmap] no valid social regions parsed, using rule-based fallback")
            return params, llm_log
        params = _regions_to_entity_params(scene, regions)
        llm_log = _format_log(regions, params)

        if self._verbose:
            _print_reasoning(regions, params)

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
    method="llm"  : single-layer social region pipeline (needs API key in .env)
    """
    if method == "rule":
        return rule_based_entity_params(scene), ""
    if method == "llm":
        orc = SocialCostOrchestrator(llm_model, verbose, prompt_templates=prompt_templates)
        return orc.update(scene, robot_pos, robot_goal)
    raise ValueError(f"Unknown method: {method!r}")


# ---------------------------------------------------------------------------
# Gaussian synthesis  (resolution-independent, for viz / A* heuristic only)
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
        dx = GX - hx   # x-displacement
        dy = GY - hy   # y-displacement
        along = dx * fx + dy * fy
        perp  = -dx * fy + dy * fx
        ps = ep.personal_space
        # 与 SocialCost 保持一致：后方个人空间不能过小，否则 A* 会偏好从人后窄缝穿过。
        sigma_along = np.where(along >= 0, ps * ep.orientation_sensitivity, ps * back_scale)
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
