"""
SimWorld — 2D 社交导航仿真环境
================================
所有实体（agents + robot）在同一个 dt 内同时推进。

用法：
    from experiments.social_nav.sim_world import SimWorld, make_hallway_scenario
    world = SimWorld(make_hallway_scenario())
    while not world.done:
        world.step()
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.ndimage import distance_transform_edt

# ---------------------------------------------------------------------------
# 房间参数
# ---------------------------------------------------------------------------

X_MIN, X_MAX = -5.0,  5.0
Y_MIN, Y_MAX = -3.0,  3.0
PX_PER_M     = 10          # occupancy: 100 × 60
DOWNSCALE    = 5           # grid_free: 20 × 12，0.5 m/cell
GRID_SPACING = DOWNSCALE / PX_PER_M  # 0.5 m/cell


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

@dataclass
class Agent:
    agent_id:    str
    pos:         np.ndarray   # (x, y)
    vel:         np.ndarray   # (vx, vy)
    heading_deg: float
    activity:    str          # "standing" / "walking" / "talking"
    description: str


@dataclass
class Keyframe:
    t:           float
    pos:         tuple[float, float]
    heading_deg: float  | None = None
    activity:    str    | None = None
    description: str    | None = None


class ScriptedAgent:
    def __init__(self, agent_id: str, keyframes: list[Keyframe],
                 base_desc: str = "a person") -> None:
        self.agent_id  = agent_id
        self.keyframes = sorted(keyframes, key=lambda k: k.t)
        self.base_desc = base_desc

    def state_at(self, t: float) -> Agent:
        kfs = self.keyframes
        if t <= kfs[0].t:
            k = kfs[0]
            return Agent(self.agent_id,
                         np.array(k.pos, dtype=np.float32), np.zeros(2),
                         k.heading_deg or 0.0,
                         k.activity or "standing",
                         k.description or self.base_desc)
        if t >= kfs[-1].t:
            k = kfs[-1]
            return Agent(self.agent_id,
                         np.array(k.pos, dtype=np.float32), np.zeros(2),
                         k.heading_deg or 0.0,
                         k.activity or "standing",
                         k.description or self.base_desc)
        for i in range(len(kfs) - 1):
            k0, k1 = kfs[i], kfs[i + 1]
            if k0.t <= t <= k1.t:
                alpha = (t - k0.t) / (k1.t - k0.t)
                p0 = np.array(k0.pos, dtype=np.float32)
                p1 = np.array(k1.pos, dtype=np.float32)
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
                desc = k1.description or k0.description or self.base_desc
                return Agent(self.agent_id, pos, vel.astype(np.float32),
                             hdg, act, desc)
        k = kfs[-1]
        return Agent(self.agent_id, np.array(k.pos, dtype=np.float32),
                     np.zeros(2), k.heading_deg or 0.0,
                     k.activity or "standing", k.description or self.base_desc)


# ---------------------------------------------------------------------------
# 关系
# ---------------------------------------------------------------------------

@dataclass
class RelEvent:
    t:       float
    a:       str
    b:       str
    rtype:   str    # "conversation"
    action:  str    # "add" / "remove"


def active_relationships(rel_events: list[RelEvent],
                          t: float) -> list[tuple[str, str, str]]:
    """返回时刻 t 时所有激活的关系 (agent_a, agent_b, rtype)。"""
    active: dict[tuple[str, str], str] = {}
    for ev in rel_events:
        if ev.t > t:
            break
        key = tuple(sorted([ev.a, ev.b]))
        if ev.action == "add":
            active[key] = ev.rtype
        else:
            active.pop(key, None)
    return [(a, b, rtype) for (a, b), rtype in active.items()]


def conversation_groups(rels: list[tuple[str, str, str]]) -> list[list[str]]:
    adj: dict[str, set[str]] = {}
    for a, b, rtype in rels:
        if rtype == "conversation":
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
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
# Social Costmap
# ---------------------------------------------------------------------------

def build_costmap(agents: dict[str, Agent],
                  groups: list[list[str]],
                  shape: tuple[int, int]) -> np.ndarray:
    """Build the deterministic rule costmap via the shared cost pipeline."""
    from experiments.social_nav.cost.llm_costmap import build_live_costmap

    cm, _ = build_live_costmap(
        agents,
        shape,
        x_range=(X_MIN, X_MAX),
        y_range=(Y_MIN, Y_MAX),
        method="rule",
        groups=groups,
    )
    return cm


def build_costmap_and_params(agents: dict[str, Agent],
                             groups: list[list[str]],
                             shape: tuple[int, int]):
    from experiments.social_nav.cost.llm_costmap import build_live_costmap

    return build_live_costmap(
        agents,
        shape,
        x_range=(X_MIN, X_MAX),
        y_range=(Y_MIN, Y_MAX),
        method="rule",
        groups=groups,
    )


# ---------------------------------------------------------------------------
# A*
# ---------------------------------------------------------------------------

_DIRS  = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
_DCOST = [1.,1.,1.,1., math.sqrt(2),math.sqrt(2),math.sqrt(2),math.sqrt(2)]


def _xy_to_rc(x, y, H, W):
    c = int((x - X_MIN) / (X_MAX - X_MIN) * W)
    r = int((Y_MAX - y) / (Y_MAX - Y_MIN) * H)
    return max(0, min(H-1, r)), max(0, min(W-1, c))


def _rc_to_xy(r, c, H, W):
    x = X_MIN + (c + 0.5) * (X_MAX - X_MIN) / W
    y = Y_MAX - (r + 0.5) * (Y_MAX - Y_MIN) / H
    return x, y


def astar(grid_free, social_cm, start_xy, goal_xy,
          social_w: float = 6.0) -> np.ndarray:
    H, W = grid_free.shape
    src  = _xy_to_rc(*start_xy, H, W)
    dst  = _xy_to_rc(*goal_xy,  H, W)
    ext  = (social_w * social_cm).astype(np.float64) if social_cm is not None else 0

    def h(r, c): return math.hypot(r - dst[0], c - dst[1])
    dist = np.full((H, W), np.inf); dist[src] = 0
    prev = {}; heap = [(h(*src), src)]

    while heap:
        f, (r, c) = heapq.heappop(heap)
        if (r, c) == dst: break
        if f > dist[r, c] + h(r, c) + 1e-9: continue
        for (dr, dc), mc in zip(_DIRS, _DCOST):
            nr, nc = r+dr, c+dc
            if not (0 <= nr < H and 0 <= nc < W and grid_free[nr, nc]): continue
            g = dist[r, c] + mc * (1.0 + float(ext[nr, nc]) if isinstance(ext, np.ndarray) else mc)
            if g < dist[nr, nc]:
                dist[nr, nc] = g; prev[(nr, nc)] = (r, c)
                heapq.heappush(heap, (g + h(nr, nc), (nr, nc)))

    path, cur = [], dst
    while cur in prev:
        path.append(cur); cur = prev[cur]
    path.append(src); path.reverse()
    return np.array([_rc_to_xy(r, c, H, W) for r, c in path], dtype=np.float32)


# ---------------------------------------------------------------------------
# SimWorld：所有实体同时推进的仿真世界
# ---------------------------------------------------------------------------

class SimWorld:
    """
    每调用一次 step()，所有实体推进 dt 秒：
      1. agents 按脚本更新位置/状态
      2. social costmap 增量更新
      3. 如果 costmap 变化 → A* 重规划
      4. MPPI 计算 robot action → robot 移动

    属性：
      t          : 当前时刻
      robot_pos  : (x, y)
      robot_yaw  : 朝向（弧度）
      agents     : dict[str, Agent]
      social_cm  : 当前 social costmap (H×W)
      waypoints  : 当前 A* 路径
      done       : 是否到达目标
    """

    ROBOT_START = np.array([ 4.0, 0.0], dtype=np.float32)
    ROBOT_GOAL  = np.array([-4.0, 0.0], dtype=np.float32)
    GOAL_RADIUS = 0.35
    WP_RADIUS   = 0.55
    REPLAN_DIST = 0.3    # costmap 峰值位移超过此值触发重规划

    def __init__(
        self,
        scripted_agents: list[ScriptedAgent],
        rel_events:      list[RelEvent],
        duration:        float,
        dt:              float  = 0.1,
        social_weight:   float  = 8.0,
    ) -> None:
        self._scripted  = {a.agent_id: a for a in scripted_agents}
        self._rel_events = sorted(rel_events, key=lambda e: e.t)
        self.duration   = duration
        self.dt         = dt
        self.social_w   = social_weight

        # 构造地图（纯矩形房间，只有墙）
        H = round((Y_MAX - Y_MIN) * PX_PER_M)
        W = round((X_MAX - X_MIN) * PX_PER_M)
        occ = np.ones((H, W), dtype=bool)
        occ[:1,:]=occ[-1:,:]=occ[:,:1]=occ[:,-1:]=False
        self._occ = occ

        from experiments.social_nav.mppi_nav import MPPINav
        from molmo_spaces.policy.solvers.navigation.mppi_core import make_downscaled_grid

        class _Map:
            def __init__(self, occ, w2m):
                self.occupancy=occ; self.world_to_map=w2m
        w2m = np.array([
            [0., -PX_PER_M, 0.,  Y_MAX*PX_PER_M],
            [PX_PER_M, 0., 0., -X_MIN*PX_PER_M],
        ], dtype=np.float64)
        self._map = _Map(occ, w2m)
        self.grid_free   = make_downscaled_grid(self._map, DOWNSCALE)
        self._dist_tf    = distance_transform_edt(self.grid_free).astype(np.float32)

        # 初始化状态
        self.t          = 0.0
        self.robot_pos  = self.ROBOT_START.copy()
        self.robot_yaw  = 0.0
        self._prev_action = np.zeros(2, dtype=np.float32)
        self.done       = False
        self.agents: dict[str, Agent] = {}
        self.rels:   list[tuple[str,str,str]] = []
        self.groups: list[list[str]] = []

        # 初始 costmap + A*
        self._update_agents()
        self.social_cm, self.social_params = build_costmap_and_params(
            self.agents, self.groups, self.grid_free.shape
        )
        self.waypoints = astar(self.grid_free, self.social_cm,
                               self.robot_pos, self.ROBOT_GOAL,
                               social_w=self.social_w)
        self._wp_idx   = 0
        self._prev_cm  = self.social_cm.copy()

        self._ctrl = MPPINav(
            scene_map=self._map,
            grid_free=self.grid_free,
            distance_transform=self._dist_tf,
            grid_spacing=GRID_SPACING,
            downscale=DOWNSCALE,
            horizon=40,
            num_samples=512,
            num_iters=3,
            dt=dt,
            temperature=1.0,
            noise_alpha=0.8,
            noise_v=0.12,
            noise_w_deg=35,
            v_min=-0.05,
            v_max=0.40,
            w_max_deg=120,
            cruise_speed=0.20,
            goal_stage_weight=4.0,
            goal_terminal_weight=20.0,
            heading_terminal_weight=0.5,
            clearance_threshold=0.25,
            clearance_weight=20.0,
            control_weight=0.1,
            smoothness_weight=1.0,
            social_params=self.social_params,
            social_weight=social_weight,
            human_positions=[tuple(a.pos) for a in self.agents.values()],
            human_radius=0.30,
            seed=42,
        )

        # 统计
        self.n_replans = 0
        self.robot_traj: list[np.ndarray] = [self.robot_pos.copy()]

    # ------------------------------------------------------------------
    def step(self) -> None:
        """推进仿真一步（dt 秒）：agents + robot 同时更新。"""
        if self.done or self.t >= self.duration:
            self.done = True
            return

        # 1. 所有 agent 同时推进（脚本插值）
        self._update_agents()

        # 2. 重算 social costmap
        new_cm, new_params = build_costmap_and_params(
            self.agents, self.groups, self.grid_free.shape
        )
        self._ctrl.update_humans([tuple(a.pos) for a in self.agents.values()])

        # 3. 如果 costmap 变化显著 → A* 重规划
        diff = float(np.abs(new_cm - self._prev_cm).max())
        if diff > 0.05:
            self.social_cm = new_cm
            self.social_params = new_params
            self._ctrl.update_social_params(new_params)
            self.waypoints = astar(self.grid_free, self.social_cm,
                                   self.robot_pos, self.ROBOT_GOAL,
                                   social_w=self.social_w)
            self._wp_idx = 0
            self._prev_cm = new_cm.copy()
            self.n_replans += 1

        # 4. 推进 waypoint 索引
        while self._wp_idx < len(self.waypoints) - 1:
            if np.linalg.norm(self.robot_pos - self.waypoints[self._wp_idx]) < self.WP_RADIUS:
                self._wp_idx += 1
            else:
                break
        target = self.waypoints[self._wp_idx]

        # 5. 到达判断
        if np.linalg.norm(self.robot_pos - self.ROBOT_GOAL) < self.GOAL_RADIUS:
            self.done = True
            return

        # 6. MPPI → robot 移动（与 agent 移动同时发生在同一 dt 内）
        state  = np.array([self.robot_pos[0], self.robot_pos[1], self.robot_yaw],
                          dtype=np.float32)
        v, w, _ = self._ctrl.step(state, target)

        self.robot_pos[0] += self.dt * v * math.cos(self.robot_yaw)
        self.robot_pos[1] += self.dt * v * math.sin(self.robot_yaw)
        self.robot_yaw     = math.atan2(
            math.sin(self.robot_yaw + self.dt * w),
            math.cos(self.robot_yaw + self.dt * w),
        )
        self.robot_traj.append(self.robot_pos.copy())
        self.t += self.dt

    # ------------------------------------------------------------------
    def _update_agents(self) -> None:
        self.agents = {aid: sa.state_at(self.t)
                       for aid, sa in self._scripted.items()}
        self.rels   = active_relationships(self._rel_events, self.t)
        self.groups = conversation_groups(self.rels)

# ---------------------------------------------------------------------------
# 预定义场景
# ---------------------------------------------------------------------------

def make_hallway_scenario() -> tuple[list[ScriptedAgent], list[RelEvent], float]:
    """
    机器人从右(4,0)到左(-4,0)，遭遇：
      t= 0 : Alice 站在 (-0.5, 1)，Bob 站在 (-0.5,-1)，相距 2m
      t= 8 : Alice 走向 Bob（开始靠近）
      t=13 : 双方开始对话，路径被阻断
      t=22 : Carol 从左侧开始横穿
      t=30 : 对话结束，Carol 停下，路径重开
    """
    alice = ScriptedAgent("alice", [
        Keyframe(0,  (-0.5,  1.0), heading_deg=270, activity="standing",
                 description="young woman, standing"),
        Keyframe(8,  (-0.5,  1.0), heading_deg=270, activity="walking",
                 description="young woman, walking toward someone"),
        Keyframe(13, (-0.5,  0.2), heading_deg=270, activity="talking",
                 description="young woman, in conversation"),
        Keyframe(30, (-0.5,  0.2), heading_deg=90,  activity="standing",
                 description="young woman, standing"),
    ], base_desc="young woman")

    bob = ScriptedAgent("bob", [
        Keyframe(0,  (-0.5, -1.0), heading_deg=90, activity="standing",
                 description="man, waiting"),
        Keyframe(13, (-0.5, -1.0), heading_deg=90, activity="talking",
                 description="man, in conversation with Alice"),
        Keyframe(30, (-0.5, -1.0), heading_deg=0,  activity="standing",
                 description="man, standing"),
    ], base_desc="man")

    carol = ScriptedAgent("carol", [
        Keyframe(0,  (-4.0,  1.5), heading_deg=0, activity="standing",
                 description="elderly woman, resting"),
        Keyframe(22, (-4.0,  1.5), heading_deg=0, activity="walking",
                 description="elderly woman, walking slowly"),
        Keyframe(35, ( 3.0,  1.5), heading_deg=0, activity="standing",
                 description="elderly woman, resting"),
    ], base_desc="elderly woman")

    rel_events = [
        RelEvent(13,  "alice", "bob", "conversation", "add"),
        RelEvent(30, "alice", "bob", "conversation", "remove"),
    ]

    return [alice, bob, carol], rel_events, 45.0
