"""
Stage2 dynamic social navigation entrypoint.

This script intentionally keeps Stage2 separated from Stage1:
- Stage1: static social path planning (`run_stage1_static.py`)
- Stage2: dynamic scene with online updates (`interactive_sim.py`)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GRAD_ROOT = REPO_ROOT.parent
for _p in (str(REPO_ROOT), str(GRAD_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage2 dynamic social navigation")
    p.add_argument("--headless", action="store_true",
                   help="Run scripted Stage2 without opening the pygame UI")
    p.add_argument("--interactive", action="store_true",
                   help="Open the older manual interactive simulator instead of the scripted Stage2 viewer")
    p.add_argument("--steps", type=int, default=120,
                   help="Number of headless simulation steps")
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--social-weight", type=float, default=8.0)
    p.add_argument("--social-method", choices=["rule", "llm"], default="rule",
                   help="rule is deterministic; llm starts async API updates on relation changes")
    p.add_argument("--llm-model", default="moonshot-v1-8k")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def run_headless(args: argparse.Namespace) -> None:
    from experiments.social_nav.sim_world import SimWorld, make_hallway_scenario

    agents, rel_events, duration = make_hallway_scenario()
    world = SimWorld(
        agents,
        rel_events,
        duration,
        dt=args.dt,
        social_weight=args.social_weight,
        social_method=args.social_method,
        llm_model=args.llm_model,
        async_social=True,
        verbose=args.verbose,
    )
    for _ in range(max(0, int(args.steps))):
        if world.done:
            break
        world.step()
    print(
        "[stage2-dynamic] "
        f"t={world.t:.2f}s steps={len(world.robot_traj) - 1} "
        f"done={world.done} replans={world.n_replans} "
        f"social_requests={world.n_social_requests} "
        f"social_updates={world.n_social_updates} "
        f"robot=({world.robot_pos[0]:.3f},{world.robot_pos[1]:.3f}) "
        f"status='{world.social_status}'"
    )


def run_scripted_viewer(args: argparse.Namespace) -> None:
    import math
    import pygame
    import numpy as np

    from experiments.social_nav.sim_world import (
        X_MIN, X_MAX, Y_MIN, Y_MAX,
        SimWorld, make_hallway_scenario,
    )

    agents, rel_events, duration = make_hallway_scenario()
    world = SimWorld(
        agents,
        rel_events,
        duration,
        dt=args.dt,
        social_weight=args.social_weight,
        social_method=args.social_method,
        llm_model=args.llm_model,
        async_social=True,
        verbose=args.verbose,
    )

    pygame.init()
    win_w, win_h = 1120, 760
    top_h, bottom_h = 46, 58
    margin = 42
    map_w = win_w - 2 * margin
    map_h = win_h - top_h - bottom_h - 2 * margin
    scale = min(map_w / (X_MAX - X_MIN), map_h / (Y_MAX - Y_MIN))
    map_ox = int((win_w - scale * (X_MAX - X_MIN)) * 0.5)
    map_oy = top_h + int((map_h - scale * (Y_MAX - Y_MIN)) * 0.5) + margin

    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("Stage2 Scripted Dynamic Social Navigation")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("segoeui", 15)
    font_b = pygame.font.SysFont("segoeui", 16, bold=True)
    font_s = pygame.font.SysFont("segoeui", 12)

    colors = {
        "bg": (18, 20, 26),
        "panel": (28, 31, 39),
        "grid": (45, 50, 62),
        "wall": (120, 128, 145),
        "text": (232, 236, 245),
        "dim": (145, 153, 170),
        "robot": (80, 180, 255),
        "traj": (90, 220, 170),
        "path": (255, 206, 92),
        "goal": (240, 92, 98),
        "agent": (190, 150, 255),
        "agent2": (250, 170, 110),
        "rel": (120, 220, 255),
    }

    def w2s(x: float, y: float) -> tuple[int, int]:
        sx = int(map_ox + (x - X_MIN) * scale)
        sy = int(map_oy + (Y_MAX - y) * scale)
        return sx, sy

    def m2px(v: float) -> int:
        return max(1, int(v * scale))

    def costmap_surface() -> pygame.Surface | None:
        cm = world.social_cm
        if cm is None:
            return None
        h, w = cm.shape
        surf = pygame.Surface((int((X_MAX - X_MIN) * scale), int((Y_MAX - Y_MIN) * scale)), pygame.SRCALPHA)
        arr_w, arr_h = surf.get_size()
        xs = np.linspace(0, w - 1, arr_w).astype(np.int32)
        ys = np.linspace(0, h - 1, arr_h).astype(np.int32)
        v = cm[ys[:, None], xs[None, :]].astype(np.float32)
        alpha = np.clip(v * 185, 0, 185).astype(np.uint8)
        rgb = np.zeros((arr_w, arr_h, 3), dtype=np.uint8)
        rgb[:, :, 0] = np.clip((v.T * 255), 0, 255).astype(np.uint8)
        rgb[:, :, 1] = np.clip((1.0 - v.T) * 110, 0, 110).astype(np.uint8)
        rgb[:, :, 2] = 40
        px3 = pygame.surfarray.pixels3d(surf)
        pxa = pygame.surfarray.pixels_alpha(surf)
        px3[:, :, :] = rgb
        pxa[:, :] = alpha.T
        del px3, pxa
        return surf

    running = True
    paused = False
    show_cost = True
    sim_acc = 0.0

    while running:
        dt_real = clock.tick(60) / 1000.0
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif ev.key == pygame.K_SPACE:
                    paused = not paused
                elif ev.key == pygame.K_r:
                    agents, rel_events, duration = make_hallway_scenario()
                    world = SimWorld(
                        agents,
                        rel_events,
                        duration,
                        dt=args.dt,
                        social_weight=args.social_weight,
                        social_method=args.social_method,
                        llm_model=args.llm_model,
                        async_social=True,
                        verbose=args.verbose,
                    )
                    sim_acc = 0.0
                elif ev.key == pygame.K_c:
                    show_cost = not show_cost

        if not paused and not world.done:
            sim_acc += dt_real
            while sim_acc >= world.dt and not world.done:
                world.step()
                sim_acc -= world.dt

        screen.fill(colors["bg"])
        pygame.draw.rect(screen, colors["panel"], pygame.Rect(0, 0, win_w, top_h))
        title = font_b.render("Stage2 Scripted Dynamic Social Navigation", True, colors["text"])
        screen.blit(title, (16, 13))
        status = (
            f"t={world.t:5.1f}s  replans={world.n_replans}  "
            f"social={args.social_method} req/upd={world.n_social_requests}/{world.n_social_updates}  "
            f"{'PAUSED' if paused else 'RUNNING'}"
        )
        st = font.render(status, True, colors["robot"] if not paused else colors["dim"])
        screen.blit(st, (win_w // 2 - st.get_width() // 2, 13))

        map_rect = pygame.Rect(map_ox, map_oy, int((X_MAX - X_MIN) * scale), int((Y_MAX - Y_MIN) * scale))
        pygame.draw.rect(screen, (22, 25, 31), map_rect)
        for x in range(math.ceil(X_MIN), math.floor(X_MAX) + 1):
            pygame.draw.line(screen, colors["grid"], w2s(x, Y_MIN), w2s(x, Y_MAX), 1)
        for y in range(math.ceil(Y_MIN), math.floor(Y_MAX) + 1):
            pygame.draw.line(screen, colors["grid"], w2s(X_MIN, y), w2s(X_MAX, y), 1)
        pygame.draw.rect(screen, colors["wall"], map_rect, 2)

        if show_cost:
            cm_surf = costmap_surface()
            if cm_surf is not None:
                screen.blit(cm_surf, map_rect.topleft)

        if len(world.waypoints) > 1:
            pygame.draw.lines(screen, colors["path"], False, [w2s(float(p[0]), float(p[1])) for p in world.waypoints], 2)
        if len(world.robot_traj) > 1:
            pygame.draw.lines(screen, colors["traj"], False, [w2s(float(p[0]), float(p[1])) for p in world.robot_traj], 3)

        gx, gy = w2s(float(world.ROBOT_GOAL[0]), float(world.ROBOT_GOAL[1]))
        pygame.draw.polygon(screen, colors["goal"], [(gx, gy - 16), (gx + 11, gy + 9), (gx - 11, gy + 9)])

        agent_map = world.agents
        for a, b, rtype in world.rels:
            if a in agent_map and b in agent_map:
                pygame.draw.line(
                    screen,
                    colors["rel"],
                    w2s(float(agent_map[a].pos[0]), float(agent_map[a].pos[1])),
                    w2s(float(agent_map[b].pos[0]), float(agent_map[b].pos[1])),
                    3,
                )
                pa, pb = agent_map[a].pos, agent_map[b].pos
                mid = w2s(float((pa[0] + pb[0]) * 0.5), float((pa[1] + pb[1]) * 0.5))
                txt = font_s.render(rtype, True, colors["rel"])
                screen.blit(txt, (mid[0] - txt.get_width() // 2, mid[1] - 18))

        for idx, ag in enumerate(world.agents.values()):
            sx, sy = w2s(float(ag.pos[0]), float(ag.pos[1]))
            col = colors["agent"] if idx % 2 == 0 else colors["agent2"]
            pygame.draw.circle(screen, col, (sx, sy), m2px(0.27))
            pygame.draw.circle(screen, (255, 255, 255), (sx, sy), m2px(0.27), 1)
            yaw = math.radians(float(ag.heading_deg))
            pygame.draw.line(screen, (255, 255, 255), (sx, sy),
                             (sx + int(m2px(0.48) * math.cos(yaw)), sy - int(m2px(0.48) * math.sin(yaw))), 2)
            lbl = font_s.render(f"{ag.agent_id} {ag.activity}", True, col)
            screen.blit(lbl, (sx + 10, sy - 8))

        rx, ry = w2s(float(world.robot_pos[0]), float(world.robot_pos[1]))
        pygame.draw.circle(screen, colors["robot"], (rx, ry), m2px(0.23))
        pygame.draw.circle(screen, (255, 255, 255), (rx, ry), m2px(0.23), 1)
        pygame.draw.line(screen, (255, 255, 255), (rx, ry),
                         (rx + int(m2px(0.45) * math.cos(world.robot_yaw)),
                          ry - int(m2px(0.45) * math.sin(world.robot_yaw))), 2)

        pygame.draw.rect(screen, colors["panel"], pygame.Rect(0, win_h - bottom_h, win_w, bottom_h))
        help_txt = "Space pause/resume   R reset   C costmap   Q/Esc quit"
        screen.blit(font.render(help_txt, True, colors["dim"]), (16, win_h - bottom_h + 10))
        status_txt = font_s.render(world.social_status, True, colors["dim"])
        screen.blit(status_txt, (16, win_h - bottom_h + 32))

        pygame.display.flip()

    pygame.quit()


def main() -> None:
    args = parse_args()
    if args.headless:
        run_headless(args)
    elif args.interactive:
        from experiments.social_nav.interactive_sim import Sim

        Sim().run()
    else:
        run_scripted_viewer(args)


if __name__ == "__main__":
    main()
