"""
Multi-Agent Dynamic Scene
=========================
每个 agent 有 state (pos, vel, heading, activity, description) 和 relationship。
场景用关键帧脚本驱动，支持时间插值。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class AgentState:
    agent_id:    str
    pos:         np.ndarray   # (x, y) 世界坐标
    vel:         np.ndarray   # (vx, vy) m/s
    heading_deg: float        # 朝向（度），0=+x
    activity:    str          # "standing" / "walking" / "talking" / "sitting"
    description: str          # 自然语言描述（给 LLM 用）


@dataclass
class Relationship:
    agent_a:  str
    agent_b:  str
    rel_type: str   # "conversation" / "walking_together" / "strangers"


@dataclass
class SceneGraph:
    agents:        dict[str, AgentState] = field(default_factory=dict)
    relationships: list[Relationship]   = field(default_factory=list)
    t:             float = 0.0

    def conversation_groups(self) -> list[list[str]]:
        """返回所有对话群组，每个元素是 agent_id 列表。"""
        adj: dict[str, set[str]] = {}
        for r in self.relationships:
            if r.rel_type == "conversation":
                adj.setdefault(r.agent_a, set()).add(r.agent_b)
                adj.setdefault(r.agent_b, set()).add(r.agent_a)
        visited, groups = set(), []
        for start in adj:
            if start in visited:
                continue
            group, stack = [], [start]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node); group.append(node)
                stack.extend(adj.get(node, set()) - visited)
            if len(group) >= 2:
                groups.append(group)
        return groups


# ---------------------------------------------------------------------------
# 脚本化 agent（关键帧驱动）
# ---------------------------------------------------------------------------

@dataclass
class Keyframe:
    t:           float
    pos:         Optional[tuple[float, float]] = None
    heading_deg: Optional[float]               = None
    activity:    Optional[str]                 = None
    description: Optional[str]                 = None


class ScriptedAgent:
    """通过关键帧描述 agent 的运动轨迹，中间线性插值。"""

    def __init__(self, agent_id: str, keyframes: list[Keyframe],
                 base_description: str = "a person") -> None:
        self.agent_id         = agent_id
        self.keyframes        = sorted(keyframes, key=lambda k: k.t)
        self.base_description = base_description

    def state_at(self, t: float) -> AgentState:
        kfs = self.keyframes
        if not kfs:
            return AgentState(self.agent_id, np.zeros(2), np.zeros(2),
                              0.0, "standing", self.base_description)

        # 超出范围：clamp 到首/尾帧
        if t <= kfs[0].t:
            kf = kfs[0]
            pos  = np.array(kf.pos or (0.0, 0.0), dtype=np.float32)
            vel  = np.zeros(2, dtype=np.float32)
            hdg  = kf.heading_deg or 0.0
            act  = kf.activity    or "standing"
            desc = kf.description or self.base_description
            return AgentState(self.agent_id, pos, vel, hdg, act, desc)

        if t >= kfs[-1].t:
            kf = kfs[-1]
            pos  = np.array(kf.pos or (0.0, 0.0), dtype=np.float32)
            vel  = np.zeros(2, dtype=np.float32)
            hdg  = kf.heading_deg or 0.0
            act  = kf.activity    or "standing"
            desc = kf.description or self.base_description
            return AgentState(self.agent_id, pos, vel, hdg, act, desc)

        # 找到包围的两帧，线性插值
        for i in range(len(kfs) - 1):
            k0, k1 = kfs[i], kfs[i + 1]
            if k0.t <= t <= k1.t:
                alpha = (t - k0.t) / max(k1.t - k0.t, 1e-6)
                p0 = np.array(k0.pos or (0.0, 0.0), dtype=np.float32)
                p1 = np.array(k1.pos or p0,          dtype=np.float32)
                pos = p0 + alpha * (p1 - p0)

                dt  = k1.t - k0.t
                vel = (p1 - p0) / dt if dt > 0 else np.zeros(2, dtype=np.float32)

                if k1.heading_deg is not None:
                    hdg = k1.heading_deg
                elif np.linalg.norm(vel) > 0.05:
                    hdg = math.degrees(math.atan2(float(vel[1]), float(vel[0])))
                else:
                    hdg = k0.heading_deg or 0.0

                act  = k1.activity    or k0.activity    or "standing"
                desc = k1.description or k0.description or self.base_description
                return AgentState(self.agent_id, pos, vel, hdg, act, desc)

        # fallback（不应该走到这里）
        kf = kfs[-1]
        return AgentState(self.agent_id,
                          np.array(kf.pos or (0.0, 0.0), dtype=np.float32),
                          np.zeros(2), kf.heading_deg or 0.0,
                          kf.activity or "standing",
                          kf.description or self.base_description)


# ---------------------------------------------------------------------------
# 关系事件
# ---------------------------------------------------------------------------

@dataclass
class RelationshipEvent:
    t:        float
    agent_a:  str
    agent_b:  str
    rel_type: str          # "conversation" / ...
    action:   str          # "add" / "remove"


# ---------------------------------------------------------------------------
# 动态场景（组合所有 agent + 关系事件）
# ---------------------------------------------------------------------------

class DynamicScenario:
    """
    将脚本化 agents 和关系事件组合成完整的动态场景。
    调用 scene_at(t) 获取任意时刻的 SceneGraph。
    """

    def __init__(self,
                 agents:      list[ScriptedAgent],
                 rel_events:  list[RelationshipEvent],
                 duration:    float) -> None:
        self._agents     = {a.agent_id: a for a in agents}
        self._rel_events = sorted(rel_events, key=lambda e: e.t)
        self.duration    = duration

    def scene_at(self, t: float) -> SceneGraph:
        sg = SceneGraph(t=t)

        for agent in self._agents.values():
            sg.agents[agent.agent_id] = agent.state_at(t)

        # 累积到时刻 t 的所有关系变化
        active: dict[tuple[str, str], str] = {}
        for ev in self._rel_events:
            if ev.t > t:
                break
            key = tuple(sorted([ev.agent_a, ev.agent_b]))
            if ev.action == "add":
                active[key] = ev.rel_type
            else:
                active.pop(key, None)

        for (a, b), rtype in active.items():
            sg.relationships.append(Relationship(a, b, rtype))

        return sg


# ---------------------------------------------------------------------------
# 预定义场景：走廊相遇
# ---------------------------------------------------------------------------

def make_hallway_scenario() -> DynamicScenario:
    """
    机器人从右往左穿越房间，遭遇以下社交动态：

    t= 0s : Alice、Bob 分开站立，机器人可从两人之间穿过
    t= 8s : Alice 走向 Bob，开始对话 → 机器人绕开对话群组
    t=18s : Carol 从左侧出发向右走，穿过机器人预定路径
    t=28s : Carol 停下，对话结束，机器人继续前进
    """

    alice = ScriptedAgent(
        "alice",
        keyframes=[
            Keyframe(t=0,  pos=(-1.5,  1.2), heading_deg=270, activity="standing",
                     description="young woman standing, looking at phone"),
            Keyframe(t=8,  pos=(-1.5,  1.2), heading_deg=270, activity="walking",
                     description="young woman walking toward someone"),
            Keyframe(t=13, pos=(-1.5,  0.4), heading_deg=270, activity="talking",
                     description="young woman in conversation"),
            Keyframe(t=28, pos=(-1.5,  0.4), heading_deg=270, activity="standing",
                     description="young woman standing"),
        ],
        base_description="young woman",
    )

    bob = ScriptedAgent(
        "bob",
        keyframes=[
            Keyframe(t=0,  pos=(-1.8, -0.5), heading_deg=90, activity="standing",
                     description="man standing, waiting"),
            Keyframe(t=13, pos=(-1.8, -0.5), heading_deg=90, activity="talking",
                     description="man in conversation with Alice"),
            Keyframe(t=28, pos=(-1.8, -0.5), heading_deg=0,  activity="standing",
                     description="man standing"),
        ],
        base_description="man",
    )

    carol = ScriptedAgent(
        "carol",
        keyframes=[
            Keyframe(t=0,  pos=(-3.5,  0.8), heading_deg=0,  activity="standing",
                     description="elderly woman with a bag, standing"),
            Keyframe(t=18, pos=(-3.5,  0.8), heading_deg=0,  activity="walking",
                     description="elderly woman walking slowly"),
            Keyframe(t=28, pos=( 0.5,  0.2), heading_deg=0,  activity="standing",
                     description="elderly woman standing, resting"),
            Keyframe(t=40, pos=( 0.5,  0.2), heading_deg=0,  activity="standing",
                     description="elderly woman standing"),
        ],
        base_description="elderly woman",
    )

    rel_events = [
        RelationshipEvent(t=13,  agent_a="alice", agent_b="bob",
                          rel_type="conversation", action="add"),
        RelationshipEvent(t=28, agent_a="alice", agent_b="bob",
                          rel_type="conversation", action="remove"),
    ]

    return DynamicScenario(
        agents=[alice, bob, carol],
        rel_events=rel_events,
        duration=40.0,
    )
