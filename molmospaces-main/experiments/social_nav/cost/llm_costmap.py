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
You are a social norm reasoning engine for a mobile robot navigating an indoor scene.

Your job has two semantic stages:
  1. Detect which social regions exist in the scene.
  2. Score each region by how socially harmful it would be for the robot to cross it,
     after comparing all regions in the same scene.

You do NOT generate coordinates, paths, A* parameters, or a pixel costmap.
You DO generate one normalized social score per social region.
The downstream geometry code turns each scored region into a 2D cost field.
</role>

<constraint_types>

personal_space
  What it means: The personal space bubble around a person.
                 The robot should not get too close.
  When to use:   Every person, almost always.
  Grounding:     Social Force Model (Helbing & Molnar 1995).
                 Intensity controls the repulsion strength and radius.

attention_cone
  What it means: The region in front of a person where they are directing
                 attention. The robot should not cross this line of focus.
  When to use:   Person is watching, reading, working, cooking, or otherwise
                 directing focused attention in a specific direction.
  Grounding:     Proxemics (Hall 1966) front/back asymmetry.
                 Intensity reflects how irreplaceable or sensitive the attention is.

f_formation
  What it means: The shared interaction space (o-space) of a conversation group.
                 The robot should not cut through this space.
  When to use:   Two or more people are talking, interacting, or marked as a group.
  Grounding:     F-formation system (Kendon 1990).
                 formation field: vis_a_vis | l_shape | side_by_side
                 Intensity reflects how active and sensitive the interaction is.

</constraint_types>

<scoring_scale>
Score is RELATIVE to the current scene, not absolute.
Always reason about all detected regions together before assigning scores.

Use both priority and score:
  low      score 0.10-0.35  Minor social preference. Robot may pass through.
  medium   score 0.36-0.60  Noticeable social cost. Prefer avoiding it.
  high     score 0.61-0.85  Strong social cost. Avoid unless detour is large.
  critical score 0.86-1.00  Almost never cross. Reserved for highly sensitive cases.

Key rule: scores must be globally calibrated.
If one region is critical, weaker regions in the same scene should usually receive
lower scores. Not everything can be high or critical.
</scoring_scale>

<reasoning_requirements>
Your reasoning field must show:
1. Per-entity analysis: what is this person doing, how engaged are they,
   what regions apply, what score/priority and why.
2. Global comparison: explicitly compare entities against each other.
   Which constraint is most important in this scene? Which is least?
3. Scoring implication: if two regions both have nonzero social cost, explain
   which one is less harmful to cross and why.

This reasoning is exposed for debugging. Be explicit and specific.
Do not write generic statements. Reference the actual activity content.
</reasoning_requirements>

<decision_guidelines>
- Every human should have at least personal_space.
- If a person has a clear facing direction and focused activity, add attention_cone.
- If people are in a group (marked by SPEAK/CONVERSATION edges, or clearly interacting),
  add f_formation for the group.
- A person can have multiple regions (e.g., personal_space + attention_cone).
- Idle or walking persons: personal_space only, low or medium priority.
- Talking, watching, working persons: higher score, consider attention_cone.
- Use the activity text to infer engagement level. Specific context
  (e.g., "world cup final", "important meeting") raises the score.
- After detecting regions, assign scores by comparing all regions globally.
- Do not assign the same score to everything. Differentiate.
</decision_guidelines>

<input_format>
{
  "nodes": [
    {
      "node_id": "human_0",
      "label": ["person"],
      "bbox_center": [x, y, z],
      "heading_deg": 0-360,
      "activities": [
        {
          "name": "SPEAK | OBSERVE | WALK | SIT | ...",
          "object": "target node id or empty",
          "description": "free text describing what the person is doing and context"
        }
      ]
    },
    {"node_id": "object_0", "label": ["tv | table | ..."], "bbox_center": [x, y, z]}
  ],
  "edges": [
    ["human_0", "human_1", {"relation": "SPEAK | OBSERVE | NEAR | ...", "desc": "short text", "description": "semantic relation text"}]
  ],
  "robot": {"start": [x, y], "goal": [x, y]}
}
</input_format>

<output_format>
Output valid JSON only. No text outside the JSON block.

{
  "reasoning": "step-by-step reasoning string (newlines allowed)",

  "social_regions": [
    {
      "id": "r_human_0_personal",
      "type": "personal_space",
      "targets": ["human_0"],
      "priority": "medium",
      "score": 0.50,
      "reason": "one sentence, specific to this person's activity"
    },
    {
      "id": "r_human_0_attention",
      "type": "attention_cone",
      "targets": ["human_0"],
      "priority": "high",
      "score": 0.75,
      "reason": "one sentence, specific to this person's attention and activity"
    },
    {
      "id": "r_group_human_0_human_1",
      "type": "f_formation",
      "targets": ["human_0", "human_1"],
      "formation": "vis_a_vis",
      "priority": "low",
      "score": 0.25,
      "reason": "one sentence"
    }
  ],

  "preferred_corridor": null
}

Rules:
- social_regions is the only place for social constraints.
- Valid region types: personal_space, attention_cone, f_formation.
- Each region has id, type, targets, priority, score, reason.
- score is a normalized social cost in [0, 1], calibrated after comparing all regions.
- priority and score must agree: low=0.10-0.35, medium=0.36-0.60, high=0.61-0.85, critical=0.86-1.00.
- f_formation regions must target at least two humans and may include formation:
  vis_a_vis | l_shape | side_by_side.
- personal_space and attention_cone regions usually target one human.
- preferred_corridor is always null for Stage1. Do not omit the field.
- Do not output any field not shown above.
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
    ["human_0", "object_0", {"relation": "OBSERVE", "desc": "person observe tv", "description": "World Cup final sightline"}],
    ["human_1", "human_2", {"relation": "SPEAK", "desc": "person speak person", "description": "casual chat"}]
  ],
  "robot": {"start": [0.0, 2.0], "goal": [8.0, 4.0]}
}
</input>
<output>
{
  "reasoning": "human_0 is watching the World Cup final — this is a high-stakes, time-sensitive activity with intense focus. Interrupting the sight line is essentially irreversible mid-match. This is the strongest constraint in the scene.\nhuman_1 and human_2 are having a casual chat. Casual conversations can naturally pause without social harm. Their f_formation o-space has low sensitivity.\nGlobal comparison: human_0's attention_cone is critical; human_1/2's f_formation is low. The robot path toward the goal will likely pass near human_1/2 — passing between them is socially preferable to crossing human_0's sight line.",

  "social_regions": [
    {
      "id": "r_human_0_personal",
      "type": "personal_space",
      "targets": ["human_0"],
      "priority": "high",
      "score": 0.72,
      "reason": "watching World Cup final — sight line must not be crossed"
    },
    {
      "id": "r_human_0_attention",
      "type": "attention_cone",
      "targets": ["human_0"],
      "priority": "critical",
      "score": 0.96,
      "reason": "watching World Cup final — sight line must not be crossed"
    },
    {
      "id": "r_human_1_personal",
      "type": "personal_space",
      "targets": ["human_1"],
      "priority": "low",
      "score": 0.22,
      "reason": "casual conversation, low engagement level"
    },
    {
      "id": "r_human_2_personal",
      "type": "personal_space",
      "targets": ["human_2"],
      "priority": "low",
      "score": 0.22,
      "reason": "casual conversation, low engagement level"
    },
    {
      "id": "r_group_human_1_human_2",
      "type": "f_formation",
      "targets": ["human_1", "human_2"],
      "formation": "vis_a_vis",
      "priority": "low",
      "score": 0.30,
      "reason": "casual chat, can be briefly interrupted without social harm"
    }
  ],

  "preferred_corridor": null
}
</output>
</example>

<example id="2">
<input>
{
  "nodes": [
    {"node_id": "human_0", "label": ["person"], "bbox_center": [1.0, 2.0, 0.9], "heading_deg": 0,
     "activities": [{"name": "SPEAK", "object": "human_1", "description": "speaking in a group meeting, presenting slides"}]},
    {"node_id": "human_1", "label": ["person"], "bbox_center": [3.0, 1.5, 0.9], "heading_deg": 180,
     "activities": [{"name": "OBSERVE", "object": "human_0", "description": "listening to presentation, taking notes"}]},
    {"node_id": "human_2", "label": ["person"], "bbox_center": [3.0, 2.5, 0.9], "heading_deg": 180,
     "activities": [{"name": "OBSERVE", "object": "human_0", "description": "listening to presentation"}]}
  ],
  "edges": [
    ["human_0", "human_1", {"relation": "SPEAK", "desc": "presenter addresses listener", "description": "formal meeting in progress"}],
    ["human_0", "human_2", {"relation": "SPEAK", "desc": "presenter addresses listener", "description": "formal meeting in progress"}]
  ],
  "robot": {"start": [0.0, 5.0], "goal": [5.0, 0.0]}
}
</input>
<output>
{
  "reasoning": "All three people are in an active meeting. human_0 is presenting — this is a high-focus activity and their attention_cone (toward the audience) should not be disrupted. human_1 and human_2 are actively listening and taking notes — this is a formal context. The entire group interaction space is sensitive.\nGlobal comparison: the meeting is the dominant constraint in the scene. All intensities are high or above. The robot should route around the entire meeting group.",

  "social_regions": [
    {
      "id": "r_human_0_personal",
      "type": "personal_space",
      "targets": ["human_0"],
      "priority": "high",
      "score": 0.72,
      "reason": "presenting in a meeting, active speaker"
    },
    {
      "id": "r_human_0_attention",
      "type": "attention_cone",
      "targets": ["human_0"],
      "priority": "high",
      "score": 0.84,
      "reason": "presenting in a meeting, active speaker"
    },
    {
      "id": "r_human_1_personal",
      "type": "personal_space",
      "targets": ["human_1"],
      "priority": "medium",
      "score": 0.52,
      "reason": "actively listening and taking notes"
    },
    {
      "id": "r_human_2_personal",
      "type": "personal_space",
      "targets": ["human_2"],
      "priority": "medium",
      "score": 0.48,
      "reason": "actively listening to presentation"
    },
    {
      "id": "r_group_human_0_human_1_human_2",
      "type": "f_formation",
      "targets": ["human_0", "human_1", "human_2"],
      "formation": "vis_a_vis",
      "priority": "high",
      "score": 0.82,
      "reason": "formal meeting in progress, group space should not be disrupted"
    }
  ],

  "preferred_corridor": null
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
        scene_payload: Dictionary with HumanSSG-style nodes, edges, robot, and optional notes
        system_prompt: Optional custom system prompt (defaults to _SOCIAL_REGION_SYSTEM)
    
    Returns:
        Complete prompt string ready for LLM
    """
    return (system_prompt or _SOCIAL_REGION_SYSTEM) + "\n\n" + json.dumps(
        scene_payload,
        ensure_ascii=False,
        indent=2,
    )


_INTENSITY_TO_SCORE: dict[str, float] = {"low": 0.25, "medium": 0.50, "high": 0.75, "critical": 0.95}
_INTENSITY_TO_PS:    dict[str, float] = {"low": 0.70, "medium": 0.90, "high": 1.20, "critical": 1.50}
_INTENSITY_TO_OS:    dict[str, float] = {"low": 1.30, "medium": 1.80, "high": 2.50, "critical": 3.50}


_REGION_TYPE_ALIASES: dict[str, str] = {
    "sfm_personal_space": "personal_space",
    "personal_space": "personal_space",
    "attention_cone": "attention_cone",
    "f_formation": "f_formation",
}


def _parse_social_regions(response: str) -> tuple[str, list[dict], object]:
    """Parse LLM response into (reasoning, social_regions, preferred_corridor).

    Preferred format:
      {"social_regions": [{"id": "...", "type": "...", "targets": [...],
                           "priority": "low|medium|high|critical",
                           "score": 0.0-1.0, ...}]}

    The previous annotations format is still accepted and normalized to
    social_regions so old saved prompts fail softly instead of breaking the UI.
    """
    start = response.find("{")
    end = response.rfind("}") + 1
    if start < 0 or end <= start:
        raise ValueError(f"No JSON in LLM response: {response[:200]}")

    data = json.loads(response[start:end])
    reasoning = str(data.get("reasoning", ""))
    preferred_corridor = data.get("preferred_corridor", None)

    allowed_priorities = {"low", "medium", "high", "critical"}

    def norm_type(raw: object) -> str | None:
        return _REGION_TYPE_ALIASES.get(str(raw or "").strip().lower())

    def norm_priority(raw: object) -> str | None:
        value = str(raw or "").strip().lower()
        return value if value in allowed_priorities else None

    def norm_score(raw: object) -> float | None:
        if raw is None:
            return None
        try:
            return float(np.clip(float(raw), 0.0, 1.0))
        except (TypeError, ValueError):
            return None

    valid: list[dict] = []
    for i, region in enumerate(data.get("social_regions", [])):
        r_type = norm_type(region.get("type") or region.get("region_type"))
        priority = norm_priority(region.get("priority") or region.get("intensity"))
        targets = region.get("targets", [])
        if isinstance(targets, str):
            targets = [targets]
        targets = [str(t) for t in targets]
        if not r_type or not priority or not targets:
            continue
        if r_type == "f_formation" and len(targets) < 2:
            continue
        valid.append({
            "id": str(region.get("id") or region.get("region_id") or f"r{i}"),
            "type": r_type,
            "targets": targets,
            "priority": priority,
            "score": norm_score(region.get("score")),
            "formation": str(region.get("formation", "vis_a_vis")),
            "reason": str(region.get("reason", "")),
        })

    if valid:
        return reasoning, valid, preferred_corridor

    # Backward compatibility: annotations -> social_regions.
    for ann_i, ann in enumerate(data.get("annotations", [])):
        ann_targets: list[str] = []
        if "entity" in ann:
            ann_targets = [str(ann["entity"])]
        elif "group" in ann:
            ann_targets = [str(m) for m in ann.get("group", [])]
        if not ann_targets:
            continue
        for c_i, constraint in enumerate(ann.get("constraints", [])):
            r_type = norm_type(constraint.get("type"))
            priority = norm_priority(constraint.get("intensity") or constraint.get("priority"))
            if not r_type or not priority:
                continue
            if r_type == "f_formation" and len(ann_targets) < 2:
                continue
            valid.append({
                "id": f"legacy_{ann_i}_{c_i}",
                "type": r_type,
                "targets": ann_targets,
                "priority": priority,
                "score": norm_score(constraint.get("score")),
                "formation": str(constraint.get("formation", "vis_a_vis")),
                "reason": str(ann.get("reason", "")),
            })

    return reasoning, valid, preferred_corridor

# ---------------------------------------------------------------------------
# Scene → social regions → SocialEntityParams
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
                        "description": "these people are interacting and share an f-formation space",
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


def _merge_entity_params(base: SocialEntityParams, other: SocialEntityParams) -> SocialEntityParams:
    base.score = max(base.score, other.score)
    base.personal_space = max(base.personal_space, other.personal_space)
    base.orientation_sensitivity = max(base.orientation_sensitivity, other.orientation_sensitivity)
    if other.sigma_perp is not None:
        base.sigma_perp = other.sigma_perp if base.sigma_perp is None else max(base.sigma_perp, other.sigma_perp)
    if other.reason and other.reason not in base.reason:
        base.reason = other.reason if not base.reason else base.reason
    return base


def _region_score(region: dict) -> float:
    score = region.get("score")
    if score is not None:
        try:
            return float(np.clip(float(score), 0.0, 1.0))
        except (TypeError, ValueError):
            pass
    return _INTENSITY_TO_SCORE[region.get("priority", "medium")]


def _social_regions_to_entity_params(scene: SceneDescription, regions: list[dict]) -> list[SocialEntityParams]:
    """Convert social_regions → SocialEntityParams list.

    personal_space → per-entity score + personal_space radius
    attention_cone → higher orientation_sensitivity (directional front bias)
    f_formation    → extra midpoint entity for group o-space
    """
    params_by_id: dict[str, SocialEntityParams] = {}
    extra_params: list[SocialEntityParams] = []

    for region in regions:
        reason = region.get("reason", "")
        r_type = region.get("type", "")
        priority = region.get("priority", "medium")
        targets = [str(t) for t in region.get("targets", [])]

        if r_type in {"personal_space", "attention_cone"}:
            if not targets:
                continue
            entity = _lookup_scene_entity(scene, targets[0])
            if entity is None:
                continue
            kind, index, pos, yaw = entity
            param_id = f"person_{index}" if kind == "human" else f"object_{index}"

            score = _region_score(region)
            ps = _INTENSITY_TO_PS[priority]
            os_val = 1.3
            sigma_perp = ps * 0.85
            if r_type == "attention_cone":
                os_val = _INTENSITY_TO_OS[priority]
                sigma_perp = ps * 0.45  # narrow perpendicular; cone is focused

            new_p = SocialEntityParams(
                entity_id=param_id, pos=pos, yaw_deg=yaw,
                score=score, personal_space=ps,
                orientation_sensitivity=os_val, sigma_perp=sigma_perp,
                reason=reason,
            )
            if param_id in params_by_id:
                params_by_id[param_id] = _merge_entity_params(params_by_id[param_id], new_p)
            else:
                params_by_id[param_id] = new_p

        elif r_type == "f_formation":
            entities = [e for m in targets if (e := _lookup_scene_entity(scene, m)) is not None]
            if len(entities) < 2:
                continue

            xs = [e[2][0] for e in entities]
            ys = [e[2][1] for e in entities]
            mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
            yaw_axis = math.degrees(math.atan2(
                entities[1][2][1] - entities[0][2][1],
                entities[1][2][0] - entities[0][2][0],
            ))

            formation = region.get("formation", "vis_a_vis")
            score = _region_score(region)
            ps = _INTENSITY_TO_PS[priority]
            avg_dist = math.hypot(
                entities[-1][2][0] - entities[0][2][0],
                entities[-1][2][1] - entities[0][2][1],
            ) / max(len(entities) - 1, 1)

            if formation == "vis_a_vis":
                sp_along = avg_dist / 2 + ps * 0.50
                sp_perp  = ps * 0.60
            elif formation == "side_by_side":
                sp_along = ps * 0.80
                sp_perp  = avg_dist / 2 + ps * 0.50
            else:  # l_shape
                sp_along = avg_dist / 2 + ps * 0.40
                sp_perp  = ps * 0.70

            group_id = str(region.get("id") or ("group_" + "_".join(targets)))
            extra_params.append(SocialEntityParams(
                entity_id=group_id, pos=(mx, my), yaw_deg=yaw_axis,
                score=score, personal_space=sp_along,
                orientation_sensitivity=1.0, sigma_perp=sp_perp,
                reason=reason,
            ))

    return list(params_by_id.values()) + extra_params


def _format_log(reasoning: str, regions: list[dict], params: list[SocialEntityParams]) -> str:
    lines = ["=== LLM REASONING ===", reasoning, "", "=== SOCIAL REGIONS ==="]
    for region in regions:
        lines.append(
            f"  {region.get('id', '')}  {region.get('type')}:{region.get('priority')} "
            f"score={_region_score(region):.2f}  "
            f"targets={region.get('targets', [])}"
        )
        lines.append(f"    {region.get('reason', '')}")
    lines += ["", "=== ENTITY PARAMS ==="]
    for p in params:
        lines.append(f"  {p.entity_id}  score={p.score:.2f}  ps={p.personal_space:.1f}m  os={p.orientation_sensitivity:.1f}x")
        lines.append(f"    {p.reason}")
    return "\n".join(lines)


def _print_reasoning(reasoning: str, regions: list[dict], params: list[SocialEntityParams]) -> None:
    sep = "─" * 60
    print(f"\n{sep}")
    print("LLM REASONING")
    print(sep)
    print(reasoning)
    print(f"\n{sep}")
    print("SOCIAL REGIONS")
    print(sep)
    for region in regions:
        print(
            f"  {region.get('id', '')}  {region.get('type')}  priority={region.get('priority')}  "
            f"score={_region_score(region):.2f}  "
            f"targets={region.get('targets', [])}"
        )
        print(f"    → {region.get('reason', '')}")
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
        reasoning, regions, _ = _parse_social_regions(resp)
        if not regions:
            params = rule_based_entity_params(scene)
            llm_log = "=== SOCIAL REGIONS ===\n  (none parsed; fell back to rule-based params)"
            if self._verbose:
                print("[llm_costmap] no valid social_regions parsed, using rule-based fallback")
            return params, llm_log
        params = _social_regions_to_entity_params(scene, regions)
        llm_log = _format_log(reasoning, regions, params)

        if self._verbose:
            _print_reasoning(reasoning, regions, params)

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
