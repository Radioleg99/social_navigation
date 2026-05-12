"""
Social Navigation Simulator  —  游戏风格交互
=============================================
底部工具栏：工具组 + 动作组（Play / Reset / Costmap / 🤖Auto / 🧠LLM）
[ / ] 键调整 social weight

Auto  : agent 自主 FSM（idle -> walk -> talk -> idle）
LLM   : 手动/周期调用 LLM 重新推断 costmap；无 key 时自动 fallback 到 rule-based

运行：
    ./.venv/bin/python3 experiments/social_nav/interactive_sim.py
"""

from __future__ import annotations

import math
import sys
import threading
from pathlib import Path

import numpy as np
import pygame
import pygame.gfxdraw
from scipy.ndimage import distance_transform_edt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.social_nav.sim_world import (
    X_MIN, X_MAX, Y_MIN, Y_MAX,
    PX_PER_M, DOWNSCALE, GRID_SPACING,
    astar, Agent,
)

# ── 窗口布局 ────────────────────────────────────────────────────────────────
WIN_W, WIN_H = 1200, 820
TOOLBAR_H    = 90
TOPBAR_H     = 36
MAP_W        = WIN_W
MAP_H        = WIN_H - TOOLBAR_H - TOPBAR_H
MAP_Y0       = TOPBAR_H

WORLD_W = X_MAX - X_MIN   # 10 m
WORLD_H = Y_MAX - Y_MIN   #  6 m
SCALE   = min(MAP_W / WORLD_W, MAP_H / WORLD_H)
MAP_OX  = int((MAP_W - WORLD_W * SCALE) / 2)
MAP_OY  = MAP_Y0 + int((MAP_H - WORLD_H * SCALE) / 2)


def w2s(wx, wy):
    return (int(MAP_OX + (wx - X_MIN) * SCALE),
            int(MAP_OY + (Y_MAX - wy) * SCALE))

def s2w(sx, sy):
    return ((sx - MAP_OX) / SCALE + X_MIN,
            Y_MAX - (sy - MAP_OY) / SCALE)

def m2px(m): return max(1, int(m * SCALE))

def in_map(sx, sy):
    wx, wy = s2w(sx, sy)
    return X_MIN <= wx <= X_MAX and Y_MIN <= wy <= Y_MAX


# ── 调色板 ───────────────────────────────────────────────────────────────────
C = dict(
    bg        = ( 18,  20,  26),
    map_bg    = ( 28,  30,  40),
    grid      = ( 42,  44,  58),
    wall      = ( 72,  76,  96),
    robot     = ( 64, 196, 255),
    robot_bd  = (255, 255, 255),
    goal      = ( 72, 230, 120),
    start     = (255, 210,  60),
    traj      = ( 64, 160, 230),
    waypoint  = (100, 140, 255),
    obs       = (200,  80,  60),
    obs_bd    = (255, 120,  90),
    tb_bg     = ( 24,  26,  36),
    tb_sel    = ( 44,  48,  68),
    tb_hover  = ( 36,  38,  52),
    tb_bd     = ( 56,  60,  80),
    text      = (210, 215, 225),
    dim       = (110, 115, 135),
    top_bg    = ( 20,  22,  32),
    green     = ( 72, 230, 120),
    red       = (230,  72,  72),
    yellow    = (255, 210,  60),
    conv_line = (255, 130,  80),
)

AGENT_PALLETE = [
    (100, 190, 255),
    (255, 170,  70),
    (120, 230, 140),
    (230, 120, 230),
    (230, 230,  90),
]

ACTIVITY_LIST = ["standing", "walking", "talking", "sitting"]


# ── 数据类 ───────────────────────────────────────────────────────────────────
class Obstacle:
    def __init__(self, pos, radius=0.40):
        self.pos    = np.array(pos, dtype=np.float32)
        self.radius = radius


class IAAgent:
    """Interactive agent with autonomous FSM behavior."""

    SPEED = 0.65   # m/s
    _nxt  = 0

    def __init__(self, pos, heading_deg: float = 0.0, activity: str = "standing"):
        self.id          = f"P{IAAgent._nxt}"; IAAgent._nxt += 1
        self.pos         = np.array(pos, dtype=np.float32)
        self.heading_deg = float(heading_deg)
        self.activity    = activity
        self.col         = AGENT_PALLETE[(IAAgent._nxt - 1) % len(AGENT_PALLETE)]

        # FSM
        self._fsm_state    : str                    = "idle"
        self._timer        : float                  = 0.0
        self._walk_dest    : np.ndarray | None      = None
        self._talk_partner : IAAgent | None         = None

        # LLM output
        self.llm_reason : str = ""

    # ── FSM ──────────────────────────────────────────────────────────────────
    def update(self, dt: float, peers: list[IAAgent], rng: np.random.Generator):
        self._timer = max(0.0, self._timer - dt)

        if self._fsm_state == "idle":
            if self._timer <= 0.0:
                wx = float(rng.uniform(X_MIN + 0.6, X_MAX - 0.6))
                wy = float(rng.uniform(Y_MIN + 0.5, Y_MAX - 0.5))
                self._walk_dest = np.array([wx, wy], dtype=np.float32)
                self._fsm_state = "walking"
                self.activity   = "walking"

        elif self._fsm_state == "walking":
            # proximity conversation trigger
            for peer in peers:
                if peer is self or peer._fsm_state == "talking":
                    continue
                d = float(np.linalg.norm(self.pos - peer.pos))
                if d < 1.0 and rng.random() < 0.25 * dt:
                    self._start_talk(peer, rng)
                    break

            if self._fsm_state == "walking" and self._walk_dest is not None:
                direction = self._walk_dest - self.pos
                dist = float(np.linalg.norm(direction))
                if dist < 0.25:
                    self._fsm_state = "idle"
                    self.activity   = "standing"
                    self._timer     = float(rng.uniform(5, 14))
                else:
                    step = min(self.SPEED * dt, dist)
                    self.pos += (direction / dist * step).astype(np.float32)
                    self.heading_deg = math.degrees(
                        math.atan2(float(direction[1]), float(direction[0])))

        elif self._fsm_state == "talking":
            if self._talk_partner is not None:
                d = self._talk_partner.pos - self.pos
                if float(np.linalg.norm(d)) > 0.01:
                    self.heading_deg = math.degrees(
                        math.atan2(float(d[1]), float(d[0])))
            if self._timer <= 0.0:
                self._fsm_state    = "idle"
                self.activity      = "standing"
                self._timer        = float(rng.uniform(3, 9))
                self._talk_partner = None

    def _start_talk(self, peer: IAAgent, rng: np.random.Generator):
        dur = float(rng.uniform(12, 35))
        self._fsm_state    = "talking";   self.activity      = "talking"
        self._timer        = dur;         self._talk_partner = peer
        peer._fsm_state    = "talking";   peer.activity      = "talking"
        peer._timer        = dur;         peer._talk_partner = self

    def to_agent(self) -> Agent:
        return Agent(self.id, self.pos.copy(), np.zeros(2, dtype=np.float32),
                     self.heading_deg, self.activity, self.activity + " person")


# ── 工具栏定义 ───────────────────────────────────────────────────────────────
TOOLS = [
    dict(id="start",  icon="🟡", label="Start",  tip="Set robot start"),
    dict(id="goal",   icon="🟢", label="Goal",   tip="Set goal"),
    dict(id="wall",   icon="🧱", label="Wall",   tip="Add obstacle"),
    dict(id="agent",  icon="🧑", label="Agent",  tip="Add person"),
    dict(id="delete", icon="🗑", label="Delete", tip="Right-click also deletes"),
]
ACTIONS = [
    dict(id="play",    icon="▶",  label="Play"),
    dict(id="reset",   icon="↺",  label="Reset"),
    dict(id="costmap", icon="🌡", label="Costmap"),
    dict(id="auto",    icon="🤖", label="Auto"),
    dict(id="llm",     icon="🧠", label="LLM"),
]


# ── 主类 ─────────────────────────────────────────────────────────────────────
class Sim:
    ROBOT_R = 0.22
    GOAL_R  = 0.35
    WP_R    = 0.55
    SIM_DT  = 0.05

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("Social Nav Sim")
        self.clock  = pygame.time.Clock()

        self.fnt     = pygame.font.SysFont("segoeui",      15)
        self.fnt_b   = pygame.font.SysFont("segoeui",      15, bold=True)
        self.fnt_ico = pygame.font.SysFont("segoeuiemoji", 22)
        self.fnt_sm  = pygame.font.SysFont("segoeui",      12)

        # Scene objects
        self.obstacles : list[Obstacle] = []
        self.agents    : list[IAAgent]  = []
        self.robot_start = np.array([ 3.5,  0.0], dtype=np.float32)
        self.robot_goal  = np.array([-3.5,  0.0], dtype=np.float32)

        # Sim state
        self.running   = False
        self.show_cm   = True
        self.social_w  = 8.0
        self.done      = False
        self.n_replans = 0
        self.t         = 0.0

        self.robot_pos  = self.robot_start.copy()
        self.robot_yaw  = 0.0
        self.traj       : list[np.ndarray] = [self.robot_start.copy()]
        self.waypoints  = np.array([self.robot_goal], dtype=np.float32)
        self._wp_idx    = 0
        self._sim_acc   = 0.0

        # Interaction
        self.tool        = "start"
        self._phase      = "place"
        self._pending_ag = None
        self._drag_agent : IAAgent | None = None
        self._drag_off   = (0.0, 0.0)

        # Autonomous / LLM mode
        self.auto_agents  = False
        self._llm_mode    = False
        self._llm_timer   = 0.0
        self._llm_status  = ""
        self._rng         = np.random.default_rng(42)

        # Internal nav
        self._ctrl        = None
        self._gf          = None
        self._dtf         = None
        self._smap        = None
        self.social_cm    = None
        self.social_params = []
        self._cm_surf     = None
        self._cm_dirty    = True
        self._prev_cm_sum = 0.0

        self._rebuild()
        self._default_scene()

    # ── 场景构建 ─────────────────────────────────────────────────────────────
    def _rebuild(self):
        from experiments.social_nav.mppi_nav import MPPINav
        from molmo_spaces.policy.solvers.navigation.mppi_core import make_downscaled_grid

        H = round((Y_MAX - Y_MIN) * PX_PER_M)
        W = round((X_MAX - X_MIN) * PX_PER_M)
        occ = np.ones((H, W), dtype=bool)
        occ[:1, :] = occ[-1:, :] = occ[:, :1] = occ[:, -1:] = False
        for obs in self.obstacles:
            rp = max(1, round(obs.radius * PX_PER_M))
            oc = round((obs.pos[0] - X_MIN) * PX_PER_M)
            or_ = round((Y_MAX - obs.pos[1]) * PX_PER_M)
            for dr in range(-rp, rp + 1):
                for dc in range(-rp, rp + 1):
                    if dr*dr + dc*dc <= rp*rp:
                        nr, nc = or_ + dr, oc + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            occ[nr, nc] = False

        w2m = np.array([[0., -PX_PER_M, 0., Y_MAX * PX_PER_M],
                        [PX_PER_M, 0., 0., -X_MIN * PX_PER_M]], dtype=np.float64)

        class _M:
            def __init__(s, o, wm): s.occupancy = o; s.world_to_map = wm

        self._smap = _M(occ, w2m)
        self._gf   = make_downscaled_grid(self._smap, DOWNSCALE)
        self._dtf  = distance_transform_edt(self._gf).astype(np.float32)

        self._update_cm(force=True)
        self._ctrl = MPPINav(
            scene_map=self._smap,
            grid_free=self._gf,
            distance_transform=self._dtf,
            grid_spacing=GRID_SPACING,
            downscale=DOWNSCALE,
            horizon=40,
            num_samples=512,
            num_iters=3,
            dt=self.SIM_DT,
            temperature=1.0,
            noise_alpha=0.8,
            noise_v=0.12,
            noise_w_deg=35,
            v_min=-0.05,
            v_max=0.40,
            w_max_deg=120,
            cruise_speed=0.20,
            goal_stage_weight=4.0, goal_terminal_weight=20.0,
            heading_terminal_weight=0.5,
            clearance_threshold=0.25,
            clearance_weight=20.0, control_weight=0.1,
            smoothness_weight=1.0,
            social_params=self.social_params,
            social_weight=self.social_w,
            human_positions=[tuple(a.pos) for a in self.agents],
            human_radius=0.30,
            seed=42,
        )
        self._replan()

    def _default_scene(self):
        """Pre-place 2 agents for demo."""
        IAAgent._nxt = 0
        a1 = IAAgent((-1.2,  0.9), heading_deg=270, activity="standing")
        a2 = IAAgent((-1.6, -0.6), heading_deg=90,  activity="standing")
        self.agents = [a1, a2]
        self._update_cm(force=True)
        self._sync_cm()
        self._replan()

    # ── Costmap ──────────────────────────────────────────────────────────────
    def _update_cm(self, force: bool = False) -> bool:
        """Recompute deterministic rule costmap unless a background LLM update owns it."""
        if self._gf is None:
            return False
        if self._llm_mode:
            return False   # LLM thread owns social_cm

        ad = {a.id: a.to_agent() for a in self.agents}
        if not ad:
            cm = np.zeros(self._gf.shape, dtype=np.float32)
            params = []
        else:
            from experiments.social_nav.cost.llm_costmap import build_live_costmap
            cm, params = build_live_costmap(
                ad, self._gf.shape,
                x_range=(X_MIN, X_MAX), y_range=(Y_MIN, Y_MAX),
                method="rule",
                robot_pos=self.robot_pos,
                robot_goal=self.robot_goal,
                groups=self._get_conv_groups(),
            )

        s = float(cm.sum())
        if not force and abs(s - self._prev_cm_sum) < 0.1:
            return False
        self.social_cm    = cm
        self.social_params = params
        self._prev_cm_sum = s
        self._cm_dirty    = True
        return True

    def _get_conv_groups(self) -> list[list[str]]:
        """Return list of [id_a, id_b] pairs currently in conversation."""
        groups: list[list[str]] = []
        seen: set[str] = set()
        for ag in self.agents:
            if ag._fsm_state == "talking" and ag._talk_partner is not None:
                if ag.id not in seen:
                    partner = ag._talk_partner
                    if partner in self.agents and partner.id not in seen:
                        groups.append([ag.id, partner.id])
                        seen.add(ag.id); seen.add(partner.id)
        return groups

    def _trigger_llm(self):
        """Start background LLM costmap update."""
        if not self.agents:
            return
        self._llm_status = "Thinking…"

        # Snapshot (thread-safe copies)
        agents_snap = {a.id: a.to_agent() for a in self.agents}
        robot_pos   = self.robot_pos.copy()
        robot_goal  = self.robot_goal.copy()
        groups      = self._get_conv_groups()
        grid_shape  = self._gf.shape
        t           = self.t

        def _worker():
            try:
                from experiments.social_nav.cost.llm_costmap import build_live_costmap
                cm, params = build_live_costmap(
                    agents_snap, grid_shape,
                    x_range=(X_MIN, X_MAX), y_range=(Y_MIN, Y_MAX),
                    method="llm",
                    robot_pos=robot_pos, robot_goal=robot_goal,
                    groups=groups, t=t,
                )
                self.social_cm    = cm
                self.social_params = params
                self._cm_dirty    = True
                self._prev_cm_sum = float(cm.sum())
                self._llm_status  = f"Updated  t={t:.1f}s"
                # write per-agent LLM reasoning back to UI objects
                reason_map = {p.entity_id: p.reason for p in params}
                for ag in self.agents:
                    ag.llm_reason = reason_map.get(ag.id, "")
                self._sync_cm()
                self._replan()
            except Exception as exc:
                self._llm_status = f"Error: {str(exc)[:45]}"

        threading.Thread(target=_worker, daemon=True).start()

    def _sync_cm(self):
        if self._ctrl is None:
            return
        self._ctrl.update_social_params(self.social_params)
        self._ctrl.update_humans([tuple(a.pos) for a in self.agents])

    def _replan(self):
        if self._gf is None or self.social_cm is None:
            return
        self.waypoints = astar(self._gf, self.social_cm,
                               self.robot_pos, self.robot_goal,
                               social_w=self.social_w)
        self._wp_idx   = 0
        self.n_replans += 1

    def reset_robot(self):
        self.robot_pos  = self.robot_start.copy()
        self.robot_yaw  = 0.0
        self.traj       = [self.robot_start.copy()]
        self.done       = False
        self.n_replans  = 0
        self.t          = 0.0
        if self._ctrl:
            self._ctrl.reset()
        self._replan()

    # ── 仿真步 ───────────────────────────────────────────────────────────────
    def step(self):
        if self.done or self._ctrl is None:
            return
        self.t += self.SIM_DT

        # 1. Advance autonomous agents
        if self.auto_agents:
            for ag in self.agents:
                ag.update(self.SIM_DT, self.agents, self._rng)

            # LLM periodic trigger (every 5 s sim time)
            if self._llm_mode:
                self._llm_timer -= self.SIM_DT
                if self._llm_timer <= 0.0:
                    self._llm_timer = 5.0
                    self._trigger_llm()

        # 2. Update costmap + replan if agents moved
        if self._update_cm():
            self._sync_cm()
            self._replan()

        # 3. Advance waypoint pointer
        while self._wp_idx < len(self.waypoints) - 1:
            if np.linalg.norm(self.robot_pos - self.waypoints[self._wp_idx]) < self.WP_R:
                self._wp_idx += 1
            else:
                break

        # 4. Goal check
        if np.linalg.norm(self.robot_pos - self.robot_goal) < self.GOAL_R:
            self.done = True; self.running = False; return

        # 5. MPPI → robot kinematics
        state  = np.array([*self.robot_pos, self.robot_yaw], dtype=np.float32)
        v, w, _ = self._ctrl.step(state, self.waypoints[self._wp_idx])
        self.robot_pos[0] += self.SIM_DT * v * math.cos(self.robot_yaw)
        self.robot_pos[1] += self.SIM_DT * v * math.sin(self.robot_yaw)
        self.robot_yaw = math.atan2(
            math.sin(self.robot_yaw + self.SIM_DT * w),
            math.cos(self.robot_yaw + self.SIM_DT * w))
        self.traj.append(self.robot_pos.copy())

    # ── 渲染辅助 ─────────────────────────────────────────────────────────────
    def _cm_surface(self):
        """Vectorized numpy costmap → pygame Surface. ~5 ms vs ~200 ms Python loop."""
        if not self._cm_dirty and self._cm_surf:
            return self._cm_surf
        cm = self.social_cm
        if cm is None:
            return None

        CH, CW = cm.shape
        sx0 = MAP_OX;  sx1 = MAP_OX + int(WORLD_W * SCALE) + 1
        sy0 = MAP_OY;  sy1 = MAP_OY + int(WORLD_H * SCALE) + 1
        sx0, sx1 = max(0, sx0), min(WIN_W, sx1)
        sy0, sy1 = max(0, sy0), min(WIN_H, sy1)
        w_px, h_px = sx1 - sx0, sy1 - sy0
        if w_px <= 0 or h_px <= 0:
            return None

        wx_1d = (np.arange(sx0, sx1) - MAP_OX) / SCALE + X_MIN
        wy_1d = Y_MAX - (np.arange(sy0, sy1) - MAP_OY) / SCALE
        ci = np.clip(((wx_1d - X_MIN) / (X_MAX - X_MIN) * (CW - 1)).astype(np.int32), 0, CW - 1)
        ri = np.clip(((Y_MAX - wy_1d) / (Y_MAX - Y_MIN) * (CH - 1)).astype(np.int32), 0, CH - 1)

        v  = cm[ri[:, np.newaxis], ci[np.newaxis, :]]   # (h_px, w_px)
        vT = v.T.astype(np.float32)                      # (w_px, h_px)

        r_arr = np.where(vT < 0.5, np.clip(255 * vT * 2, 0, 255), 255).astype(np.uint8)
        g_arr = np.where(vT < 0.5,
                         np.clip(220 * (1 - vT), 0, 255),
                         np.clip(220 * (1 - (vT - 0.5) * 2), 0, 255)).astype(np.uint8)
        b_arr = np.where(vT < 0.5, np.clip(200 * (1 - vT * 2), 0, 255), 0).astype(np.uint8)
        a_arr = np.where(vT < 0.04, 0, np.clip(180 * vT, 0, 255)).astype(np.uint8)

        map_surf = pygame.Surface((w_px, h_px), pygame.SRCALPHA)
        px3  = pygame.surfarray.pixels3d(map_surf)
        palp = pygame.surfarray.pixels_alpha(map_surf)
        px3[:, :, 0] = r_arr;  px3[:, :, 1] = g_arr;  px3[:, :, 2] = b_arr
        palp[:, :]   = a_arr
        del px3, palp

        surf = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        surf.blit(map_surf, (sx0, sy0))
        self._cm_surf  = surf
        self._cm_dirty = False
        return surf

    def _draw_grid(self):
        for x in range(int(X_MIN), int(X_MAX) + 1):
            sx, _ = w2s(x, 0); _, sy0 = w2s(x, Y_MAX); _, sy1 = w2s(x, Y_MIN)
            pygame.draw.line(self.screen, C["grid"], (sx, sy0), (sx, sy1), 1)
        for y in range(int(Y_MIN), int(Y_MAX) + 1):
            _, sy = w2s(0, y); sx0, _ = w2s(X_MIN, y); sx1, _ = w2s(X_MAX, y)
            pygame.draw.line(self.screen, C["grid"], (sx0, sy), (sx1, sy), 1)

    def _draw_obstacles(self):
        for obs in self.obstacles:
            sx, sy = w2s(*obs.pos); r = m2px(obs.radius)
            pygame.gfxdraw.filled_circle(self.screen, sx, sy, r, (*C["obs"], 200))
            pygame.gfxdraw.aacircle(self.screen, sx, sy, r, C["obs_bd"])

    def _draw_agents(self):
        ag_map = {a.id: a for a in self.agents}

        # ── conversation lines ──
        for grp in self._get_conv_groups():
            if len(grp) >= 2 and grp[0] in ag_map and grp[1] in ag_map:
                pa = ag_map[grp[0]].pos; pb = ag_map[grp[1]].pos
                x0, y0 = w2s(*pa); x1, y1 = w2s(*pb)
                pygame.draw.line(self.screen, C["conv_line"], (x0, y0), (x1, y1), 3)
                mx, my = (x0 + x1) // 2, (y0 + y1) // 2
                ico = self.fnt_ico.render("💬", True, C["conv_line"])
                self.screen.blit(ico, (mx - ico.get_width() // 2, my - ico.get_height() // 2 - 8))

        # ── agent bodies ──
        for ag in self.agents:
            sx, sy = w2s(*ag.pos); r = m2px(0.28)
            col     = ag.col
            is_drag = (ag is self._drag_agent)

            # personal space ring
            pygame.gfxdraw.aacircle(self.screen, sx, sy, m2px(0.9), (*col, 45))

            # drag highlight
            if is_drag:
                pygame.gfxdraw.aacircle(self.screen, sx, sy, r + 5, (255, 255, 100))
                pygame.gfxdraw.aacircle(self.screen, sx, sy, r + 6, (255, 255, 100, 60))

            # body
            pygame.gfxdraw.filled_circle(self.screen, sx, sy, r, (*col, 230))
            pygame.gfxdraw.aacircle(self.screen, sx, sy, r, (255, 255, 255))

            # heading arrow
            yw = math.radians(ag.heading_deg)
            ex = int(sx + m2px(0.55) * math.cos(yw))
            ey = int(sy - m2px(0.55) * math.sin(yw))
            pygame.draw.line(self.screen, (255, 255, 255), (sx, sy), (ex, ey), 2)

            # label
            lbl = self.fnt_sm.render(f"{ag.id} {ag.activity[:4]}", True, col)
            self.screen.blit(lbl, (sx + r + 4, sy - 8))

            # LLM reason text (small, shown only when LLM mode active)
            if self._llm_mode and ag.llm_reason:
                short = ag.llm_reason[:38] + ("…" if len(ag.llm_reason) > 38 else "")
                rs = self.fnt_sm.render(short, True, (170, 180, 230))
                self.screen.blit(rs, (sx + r + 4, sy + 6))

    def _draw_robot(self):
        sx, sy = w2s(*self.robot_pos); r = m2px(self.ROBOT_R)
        pygame.gfxdraw.filled_circle(self.screen, sx, sy, r, (*C["robot"], 230))
        pygame.gfxdraw.aacircle(self.screen, sx, sy, r, C["robot_bd"])
        ex = int(sx + (r + 5) * math.cos(self.robot_yaw))
        ey = int(sy - (r + 5) * math.sin(self.robot_yaw))
        pygame.draw.line(self.screen, C["robot_bd"], (sx, sy), (ex, ey), 2)

    def _draw_markers(self):
        sx, sy = w2s(*self.robot_start)
        pygame.gfxdraw.aacircle(self.screen, sx, sy, m2px(0.22), C["start"])
        l = self.fnt_sm.render("S", True, C["start"]); self.screen.blit(l, (sx + 5, sy - 8))
        gx, gy = w2s(*self.robot_goal)
        pts = [(gx, gy - 18), (gx + 11, gy + 9), (gx - 11, gy + 9)]
        pygame.draw.polygon(self.screen, C["goal"], pts)
        pygame.draw.polygon(self.screen, (255, 255, 255), pts, 1)

    def _draw_path(self):
        if len(self.waypoints) > 1:
            pygame.draw.lines(self.screen, C["waypoint"], False,
                              [w2s(*p) for p in self.waypoints], 1)
        if len(self.traj) > 1:
            pygame.draw.lines(self.screen, C["traj"], False,
                              [w2s(*p) for p in self.traj], 2)

    def _draw_cursor(self):
        mx, my = pygame.mouse.get_pos()
        if not in_map(mx, my): return
        t = self.tool
        if t == "start":
            pygame.gfxdraw.aacircle(self.screen, mx, my, m2px(self.ROBOT_R), (*C["start"], 180))
        elif t == "goal":
            pygame.draw.polygon(self.screen, (*C["goal"], 150),
                                [(mx, my - 16), (mx + 10, my + 8), (mx - 10, my + 8)])
        elif t == "wall":
            pygame.gfxdraw.aacircle(self.screen, mx, my, m2px(0.40), (*C["obs"], 150))
        elif t == "agent":
            if self._phase == "place":
                pygame.gfxdraw.aacircle(self.screen, mx, my, m2px(0.28), (*AGENT_PALLETE[0], 150))
            elif self._phase == "heading" and self._pending_ag:
                ax, ay = w2s(*self._pending_ag.pos)
                pygame.draw.line(self.screen, AGENT_PALLETE[0], (ax, ay), (mx, my), 2)
        elif t == "delete":
            pygame.gfxdraw.aacircle(self.screen, mx, my, m2px(0.6), (230, 72, 72, 80))

    # ── 工具栏 ───────────────────────────────────────────────────────────────
    def _toolbar_rects(self):
        """Returns (tool_rects, act_rects).  bw=82 fits 5+5 buttons in 1200px."""
        tb_y    = WIN_H - TOOLBAR_H
        bw, bh, gap = 82, 72, 5
        n  = len(TOOLS)
        na = len(ACTIONS)
        tools_w = n  * bw + (n  - 1) * gap
        acts_w  = na * bw + (na - 1) * gap
        total   = tools_w + 32 + acts_w
        x0      = (WIN_W - total) // 2

        tool_rects = [pygame.Rect(x0 + i * (bw + gap), tb_y + 9, bw, bh) for i in range(n)]
        ax0 = x0 + tools_w + 32
        act_rects  = [pygame.Rect(ax0 + i * (bw + gap), tb_y + 9, bw, bh) for i in range(na)]
        return tool_rects, act_rects

    def _draw_toolbar(self):
        tb_y = WIN_H - TOOLBAR_H
        pygame.draw.rect(self.screen, C["tb_bg"], pygame.Rect(0, tb_y, WIN_W, TOOLBAR_H))
        pygame.draw.line(self.screen, C["tb_bd"], (0, tb_y), (WIN_W, tb_y), 1)

        tool_rects, act_rects = self._toolbar_rects()
        mx, my = pygame.mouse.get_pos()

        # tool buttons
        for t, r in zip(TOOLS, tool_rects):
            sel     = (self.tool == t["id"])
            hovered = r.collidepoint(mx, my)
            bg = C["tb_sel"] if sel else (C["tb_hover"] if hovered else C["tb_bg"])
            pygame.draw.rect(self.screen, bg, r, border_radius=8)
            pygame.draw.rect(self.screen, C["robot"] if sel else C["tb_bd"], r, 2, border_radius=8)
            ico = self.fnt_ico.render(t["icon"], True, C["text"])
            self.screen.blit(ico, (r.centerx - ico.get_width() // 2, r.y + 6))
            lbl = self.fnt_sm.render(t["label"], True, C["text"] if sel else C["dim"])
            self.screen.blit(lbl, (r.centerx - lbl.get_width() // 2, r.bottom - 18))

        # action buttons
        for a, r in zip(ACTIONS, act_rects):
            aid = a["id"]
            if aid == "play":
                icon, active = ("⏸" if self.running else "▶"), self.running
                col = C["green"] if self.running else C["text"]
            elif aid == "costmap":
                icon, active, col = "🌡", self.show_cm, C["robot"] if self.show_cm else C["dim"]
            elif aid == "auto":
                icon, active = "🤖", self.auto_agents
                col = C["yellow"] if self.auto_agents else C["dim"]
            elif aid == "llm":
                icon, active = "🧠", self._llm_mode
                col = C["green"] if self._llm_mode else C["dim"]
            else:
                icon, active, col = a["icon"], False, C["text"]

            hovered = r.collidepoint(mx, my)
            bg = C["tb_sel"] if active else (C["tb_hover"] if hovered else C["tb_bg"])
            pygame.draw.rect(self.screen, bg, r, border_radius=8)
            pygame.draw.rect(self.screen, C["robot"] if active else C["tb_bd"], r, 2, border_radius=8)
            ico = self.fnt_ico.render(icon, True, col)
            self.screen.blit(ico, (r.centerx - ico.get_width() // 2, r.y + 6))
            lbl = self.fnt_sm.render(a["label"], True, col)
            self.screen.blit(lbl, (r.centerx - lbl.get_width() // 2, r.bottom - 18))

    def _draw_topbar(self):
        pygame.draw.rect(self.screen, C["top_bg"], pygame.Rect(0, 0, WIN_W, TOPBAR_H))
        pygame.draw.line(self.screen, C["tb_bd"], (0, TOPBAR_H), (WIN_W, TOPBAR_H), 1)

        # title
        self.screen.blit(self.fnt_b.render("Social Nav Sim", True, C["text"]), (14, 9))

        # center status
        if self.done:
            s, col = "✓  ARRIVED", C["green"]
        elif self.running:
            d = float(np.linalg.norm(self.robot_pos - self.robot_goal))
            mode = (("🤖Auto " if self.auto_agents else "") +
                    ("🧠LLM "  if self._llm_mode  else ""))
            s, col = f"▶  {mode}t={self.t:.1f}s  dist={d:.2f}m  replans={self.n_replans}", C["robot"]
        else:
            s, col = "❙❙  PAUSED  —  press ▶ or Space", C["dim"]
        st = self.fnt.render(s, True, col)
        self.screen.blit(st, (WIN_W // 2 - st.get_width() // 2, 9))

        # right: social weight hint
        sw = self.fnt_sm.render(f"Social W: {self.social_w:.0f}  ( [ / ] )", True, C["dim"])
        self.screen.blit(sw, (WIN_W - sw.get_width() - 14, 4))
        # LLM status
        y_status = 20
        if self._llm_status:
            col_s = (C["green"] if "Updated" in self._llm_status
                     else C["yellow"] if "Thinking" in self._llm_status
                     else C["red"])
            ls = self.fnt_sm.render(f"🧠 {self._llm_status}", True, col_s)
            self.screen.blit(ls, (WIN_W - ls.get_width() - 14, y_status))

    # ── 拖拽 ─────────────────────────────────────────────────────────────────
    def _try_start_drag(self, sx, sy) -> bool:
        if not in_map(sx, sy): return False
        if self.tool == "agent" and self._phase == "heading": return False
        wx, wy = s2w(sx, sy)
        pos = np.array([wx, wy])
        bd, bag = 9999.0, None
        for ag in self.agents:
            d = float(np.linalg.norm(ag.pos - pos))
            if d < bd: bd, bag = d, ag
        if bag and bd < 0.65:
            self._drag_agent = bag
            self._drag_off   = (float(bag.pos[0] - wx), float(bag.pos[1] - wy))
            return True
        return False

    def _on_drag_motion(self, sx, sy):
        if self._drag_agent is None or not in_map(sx, sy): return
        wx, wy = s2w(sx, sy)
        self._drag_agent.pos[0] = wx + self._drag_off[0]
        self._drag_agent.pos[1] = wy + self._drag_off[1]
        self._update_cm(force=True); self._sync_cm(); self._replan()

    def _on_drag_end(self):
        self._drag_agent = None

    # ── 事件处理 ─────────────────────────────────────────────────────────────
    def _on_click(self, sx, sy, button):
        tool_rects, act_rects = self._toolbar_rects()

        for t, r in zip(TOOLS, tool_rects):
            if r.collidepoint(sx, sy):
                self.tool = t["id"]; self._phase = "place"; self._pending_ag = None; return

        for a, r in zip(ACTIONS, act_rects):
            if r.collidepoint(sx, sy):
                aid = a["id"]
                if aid == "play":
                    if self.done: self.reset_robot()
                    self.running = not self.running
                elif aid == "reset":
                    self.reset_robot()
                elif aid == "costmap":
                    self.show_cm = not self.show_cm
                elif aid == "auto":
                    self.auto_agents = not self.auto_agents
                elif aid == "llm":
                    self._llm_mode = not self._llm_mode
                    if self._llm_mode:
                        self._llm_timer = 0.0
                    else:
                        self._llm_status = ""
                        for ag in self.agents: ag.llm_reason = ""
                        self._update_cm(force=True); self._sync_cm()
                return

        if not in_map(sx, sy): return
        wx, wy = s2w(sx, sy)

        if button == 3:
            self._delete_nearest(wx, wy); return

        if button == 1 and self._try_start_drag(sx, sy):
            return

        t = self.tool
        if t == "start":
            self.robot_start = np.array([wx, wy], dtype=np.float32); self.reset_robot()
        elif t == "goal":
            self.robot_goal = np.array([wx, wy], dtype=np.float32); self._replan()
        elif t == "wall":
            self.obstacles.append(Obstacle((wx, wy))); self._rebuild()
        elif t == "agent":
            if self._phase == "place":
                self._pending_ag = IAAgent((wx, wy)); self._phase = "heading"
            elif self._phase == "heading" and self._pending_ag:
                dx, dy = wx - self._pending_ag.pos[0], wy - self._pending_ag.pos[1]
                self._pending_ag.heading_deg = math.degrees(math.atan2(dy, dx))
                self.agents.append(self._pending_ag)
                self._pending_ag = None; self._phase = "place"
                self._update_cm(force=True); self._sync_cm(); self._replan()
        elif t == "delete":
            self._delete_nearest(wx, wy)

    def _on_scroll(self, sx, sy, dy):
        if not in_map(sx, sy): return
        wx, wy = s2w(sx, sy)
        pos = np.array([wx, wy])
        bd, bag = 9999.0, None
        for ag in self.agents:
            d = float(np.linalg.norm(ag.pos - pos))
            if d < bd: bd, bag = d, ag
        if bag and bd < 0.8:
            idx = ACTIVITY_LIST.index(bag.activity) if bag.activity in ACTIVITY_LIST else 0
            bag.activity = ACTIVITY_LIST[(idx + dy) % len(ACTIVITY_LIST)]
            self._update_cm(force=True); self._sync_cm(); self._replan()

    def _delete_nearest(self, wx, wy):
        pos = np.array([wx, wy])
        bd, bt, bi = 9999.0, None, -1
        for i, o in enumerate(self.obstacles):
            d = float(np.linalg.norm(o.pos - pos))
            if d < bd: bd, bt, bi = d, "obs", i
        for i, a in enumerate(self.agents):
            d = float(np.linalg.norm(a.pos - pos))
            if d < bd: bd, bt, bi = d, "agent", i
        if bt == "obs" and bd < 1.2:
            self.obstacles.pop(bi); self._rebuild()
        elif bt == "agent" and bd < 1.2:
            self.agents.pop(bi)
            self._update_cm(force=True); self._sync_cm(); self._replan()

    def _on_key(self, key):
        if key == pygame.K_SPACE:
            if self.done: self.reset_robot()
            self.running = not self.running
        elif key == pygame.K_r:
            self.reset_robot()
        elif key == pygame.K_ESCAPE:
            self.tool = "none"; self._phase = "place"; self._pending_ag = None
        elif key == pygame.K_LEFTBRACKET:
            self.social_w = max(0.0, self.social_w - 1); self._rebuild()
        elif key == pygame.K_RIGHTBRACKET:
            self.social_w += 1; self._rebuild()

    # ── 主循环 ───────────────────────────────────────────────────────────────
    def run(self):
        while True:
            dt_real = self.clock.tick(60) / 1000.0

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                elif ev.type == pygame.MOUSEBUTTONDOWN:
                    self._on_click(*ev.pos, ev.button)
                elif ev.type == pygame.MOUSEBUTTONUP:
                    if ev.button == 1: self._on_drag_end()
                elif ev.type == pygame.MOUSEMOTION:
                    if ev.buttons[0]: self._on_drag_motion(*ev.pos)
                elif ev.type == pygame.MOUSEWHEEL:
                    mx, my = pygame.mouse.get_pos()
                    self._on_scroll(mx, my, -ev.y)
                elif ev.type == pygame.KEYDOWN:
                    self._on_key(ev.key)

            if self.running:
                self._sim_acc += dt_real
                while self._sim_acc >= self.SIM_DT:
                    self.step(); self._sim_acc -= self.SIM_DT

            # ── render ──────────────────────────────────────────────────────
            self.screen.fill(C["bg"])
            tl = w2s(X_MIN, Y_MAX); br = w2s(X_MAX, Y_MIN)
            pygame.draw.rect(self.screen, C["map_bg"],
                             pygame.Rect(tl[0], tl[1], br[0] - tl[0], br[1] - tl[1]))
            self._draw_grid()
            if self.show_cm:
                s = self._cm_surface()
                if s: self.screen.blit(s, (0, 0))
            pygame.draw.rect(self.screen, C["wall"],
                             pygame.Rect(tl[0], tl[1], br[0] - tl[0], br[1] - tl[1]), 2)
            self._draw_path()
            self._draw_obstacles()
            self._draw_agents()
            self._draw_markers()
            self._draw_robot()
            self._draw_cursor()
            self._draw_topbar()
            self._draw_toolbar()
            pygame.display.flip()


if __name__ == "__main__":
    Sim().run()
