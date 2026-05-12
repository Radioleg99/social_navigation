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
    p.add_argument("--steps", type=int, default=120,
                   help="Number of headless simulation steps")
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--social-weight", type=float, default=8.0)
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
    )
    for _ in range(max(0, int(args.steps))):
        if world.done:
            break
        world.step()
    print(
        "[stage2-dynamic] "
        f"t={world.t:.2f}s steps={len(world.robot_traj) - 1} "
        f"done={world.done} replans={world.n_replans} "
        f"robot=({world.robot_pos[0]:.3f},{world.robot_pos[1]:.3f})"
    )


def main() -> None:
    args = parse_args()
    if args.headless:
        run_headless(args)
    else:
        from experiments.social_nav.interactive_sim import Sim

        Sim().run()


if __name__ == "__main__":
    main()
