"""
Dynamic Social Navigation — 实时仿真动画
==========================================
SimWorld.step() 每帧同时推进所有 agents + robot。

用法：
    ./.venv/bin/python3 experiments/social_nav/demo_dynamic.py
    ./.venv/bin/python3 experiments/social_nav/demo_dynamic.py --dt 0.05
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.social_nav.sim_world import (
    SimWorld, make_hallway_scenario,
    X_MIN, X_MAX, Y_MIN, Y_MAX,
)

_EXTENT = [X_MIN, X_MAX, Y_MIN, Y_MAX]
_AGENT_COLORS = {"alice": "tab:blue", "bob": "tab:orange", "carol": "tab:green"}


def run(dt: float = 0.1, social_weight: float = 8.0,
        interval_ms: int = 80) -> None:

    agents, rel_events, duration = make_hallway_scenario()
    world = SimWorld(agents, rel_events, duration,
                     dt=dt, social_weight=social_weight)

    # ── 画布 ──────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Dynamic Social Navigation  (A* + MPPI)", fontsize=13)

    for ax in (ax1, ax2):
        ax.set_xlim(X_MIN, X_MAX); ax.set_ylim(Y_MIN, Y_MAX)
        ax.set_aspect("equal"); ax.grid(True, alpha=0.2)
        ax.set_xlabel("x (m)")
    ax1.set_title("Scene");          ax1.set_ylabel("y (m)")
    ax2.set_title("Social Costmap")

    # 占据图底层
    for ax in (ax1, ax2):
        ax.imshow(world.grid_free.astype(float), cmap="gray", vmin=0, vmax=1,
                  extent=_EXTENT, origin="upper", alpha=0.12)

    # Goal / Start 标记
    for ax in (ax1, ax2):
        ax.plot(*SimWorld.ROBOT_START, "go", ms=11, zorder=9, label="start")
        ax.plot(*SimWorld.ROBOT_GOAL,  "r*", ms=15, zorder=9, label="goal")
    ax1.legend(fontsize=8, loc="upper right")

    # ── ax1：路径图元素 ───────────────────────────────────────────────
    traj_line, = ax1.plot([], [], "b-",  lw=2,  zorder=6, label="robot")
    wp_line,   = ax1.plot([], [], "b--", lw=1,  alpha=0.4)
    robot_dot, = ax1.plot([], [], "bs",  ms=11, zorder=9)
    # 朝向箭头（用 annotate 模拟，每帧重绘）
    heading_arrow = [None]

    agent_dots   = {aid: ax1.plot([], [], "o", color=c, ms=13, zorder=7,
                                  markeredgecolor="k", markeredgewidth=1.2)[0]
                    for aid, c in _AGENT_COLORS.items()}
    agent_texts  = {aid: ax1.text(0, 0, "", fontsize=7.5, zorder=8,
                                   bbox=dict(boxstyle="round,pad=0.2",
                                             fc="white", alpha=0.7))
                    for aid in _AGENT_COLORS}
    agent_circles = {aid: plt.Circle((0,0), 0, color=c, fill=False,
                                      ls="--", lw=1, alpha=0)
                     for aid, c in _AGENT_COLORS.items()}
    for circ in agent_circles.values():
        ax1.add_patch(circ)

    conv_lines: list = []
    time_txt = ax1.text(X_MIN+0.2, Y_MAX-0.35, "", fontsize=10,
                        bbox=dict(fc="white", alpha=0.75))

    # ── ax2：costmap 图元素 ───────────────────────────────────────────
    cm_img = ax2.imshow(world.social_cm, cmap="YlOrRd", vmin=0, vmax=1,
                        extent=_EXTENT, origin="upper", alpha=0.72)
    plt.colorbar(cm_img, ax=ax2, shrink=0.7, label="social cost")
    traj_line2, = ax2.plot([], [], "b-",  lw=1.5, alpha=0.55, zorder=6)
    robot_dot2, = ax2.plot([], [], "bs",  ms=9,   zorder=9)
    info_txt = ax2.text(X_MIN+0.2, Y_MIN+0.35, "", fontsize=9, color="darkred",
                        bbox=dict(fc="white", alpha=0.75))

    # ── 动画回调 ──────────────────────────────────────────────────────
    def _update(_frame):
        if world.done:
            return

        world.step()   # ← 所有实体同时推进一步

        traj = np.array(world.robot_traj)
        pos  = world.robot_pos

        # --- ax1 ---
        traj_line.set_data(traj[:, 0], traj[:, 1])
        robot_dot.set_data([pos[0]], [pos[1]])

        # 朝向箭头
        if heading_arrow[0] is not None:
            heading_arrow[0].remove()
        dx = 0.4 * math.cos(world.robot_yaw)
        dy = 0.4 * math.sin(world.robot_yaw)
        heading_arrow[0] = ax1.annotate(
            "", xy=(pos[0]+dx, pos[1]+dy), xytext=(pos[0], pos[1]),
            arrowprops=dict(arrowstyle="->", color="blue", lw=2), zorder=10)

        # A* waypoints
        wp = world.waypoints
        wp_line.set_data(wp[:, 0], wp[:, 1])

        # Agents
        for aid, dot in agent_dots.items():
            if aid not in world.agents:
                dot.set_data([], [])
                agent_texts[aid].set_text("")
                agent_circles[aid].set_alpha(0)
                continue
            ag = world.agents[aid]
            dot.set_data([ag.pos[0]], [ag.pos[1]])
            agent_texts[aid].set_position((ag.pos[0]+0.12, ag.pos[1]+0.20))
            agent_texts[aid].set_text(f"{aid}\n{ag.activity[:5]}")
            c = agent_circles[aid]
            c.center = (float(ag.pos[0]), float(ag.pos[1]))
            c.set_radius(0.8); c.set_alpha(0.35)

        # 对话连线（每帧重建）
        for ln in conv_lines: ln.remove()
        conv_lines.clear()
        for grp in world.groups:
            for i in range(len(grp)-1):
                for j in range(i+1, len(grp)):
                    if grp[i] in world.agents and grp[j] in world.agents:
                        pa = world.agents[grp[i]].pos
                        pb = world.agents[grp[j]].pos
                        ln, = ax1.plot([pa[0], pb[0]], [pa[1], pb[1]],
                                       "r-", lw=2.5, alpha=0.65, zorder=6)
                        conv_lines.append(ln)

        time_txt.set_text(
            f"t = {world.t:.1f}s  |  replans: {world.n_replans}")

        # --- ax2 ---
        cm_img.set_data(world.social_cm)
        traj_line2.set_data(traj[:, 0], traj[:, 1])
        robot_dot2.set_data([pos[0]], [pos[1]])
        info_txt.set_text(
            f"costmap updates: {world.n_replans}\n"
            f"groups: {len(world.groups)}")

    n_frames = int(duration / dt) + 50
    anim = FuncAnimation(fig, _update, frames=n_frames,
                         interval=interval_ms, blit=False, repeat=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dt",            type=float, default=0.1)
    p.add_argument("--social-weight", type=float, default=8.0)
    p.add_argument("--interval",      type=int,   default=80,
                   help="动画帧间隔 ms（越小越快）")
    args = p.parse_args()
    run(dt=args.dt, social_weight=args.social_weight,
        interval_ms=args.interval)
