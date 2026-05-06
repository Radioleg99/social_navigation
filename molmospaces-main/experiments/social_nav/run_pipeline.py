"""
run_pipeline.py  —  Stage 1 end-to-end entry point
====================================================
Takes a HumanSSG scene_graph.json → builds layout JSON → runs social
navigation in MoLMoSpaces.

Usage
-----
Full pipeline from 3DSG output:
    ./.venv/bin/mjpython experiments/social_nav/run_pipeline.py \\
        --scene-graph  path/to/scene_graph.json \\
        --scene-xml    assets/scenes/ithor/FloorPlan203_physics.xml \\
        --start-pose   -1.4,4.6,0 \\
        --goal-xy      -0.6,4.6 \\
        --social-method llm \\
        --llm-model     claude-sonnet-4-6

Use an existing layout JSON (skip 3DSG conversion):
    ./.venv/bin/mjpython experiments/social_nav/run_pipeline.py \\
        --layout-json  assets/layouts/FloorPlan203_object_positions.json \\
        --scene-xml    assets/scenes/ithor/FloorPlan203_physics.xml \\
        --start-pose   -1.4,4.6,0 \\
        --goal-xy      -0.6,4.6

Stage 2 note
------------
For dynamic / LLM-guided re-planning, see interactive_sim.py which already
provides a live loop that calls build_live_costmap() each N steps.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

# Repo root on path so all sibling imports work
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Pipeline bridge (graduation/pipeline/, sits above molmospaces-main)
GRAD_ROOT = REPO_ROOT.parent
if str(GRAD_ROOT) not in sys.path:
    sys.path.insert(0, str(GRAD_ROOT))

from pipeline.scene_bridge import scene_graph_to_scene_description
from pipeline.scene_builder import scene_description_to_layout_json, save_layout_json

from experiments.social_nav.run_social_nav import run_episode


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end social nav: 3DSG → MoLMoSpaces MPPI"
    )

    # --- Scene input (mutually exclusive: scene-graph vs layout-json) ---
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--scene-graph", type=Path, metavar="JSON",
                     help="HumanSSG scene_graph.json produced by extract_json.py")
    src.add_argument("--layout-json", type=Path, metavar="JSON",
                     help="Existing MoLMoSpaces layout JSON (skip 3DSG conversion)")

    p.add_argument("--scene-xml", type=Path, required=True,
                   help="MuJoCo scene XML, e.g. assets/scenes/ithor/FloorPlan203_physics.xml")

    p.add_argument("--layout-out", type=Path, default=None,
                   help="Where to save the generated layout JSON (default: temp file)")

    # --- Navigation ---
    p.add_argument("--start-pose", type=_parse_xyz, default=(-1.4, 4.6, 0.0),
                   help="Start pose x,y,yaw_deg")
    p.add_argument("--goal-xy", type=_parse_xy, required=True,
                   help="Goal position x,y in MuJoCo world frame (metres)")
    p.add_argument("--goal-radius", type=float, default=0.20)
    p.add_argument("--max-steps", type=int, default=500)

    # --- Social costmap ---
    p.add_argument("--social-method",
                   choices=["none", "rule", "llm"],
                   default="rule",
                   help="Social costmap method: none | rule (analytic) | llm")
    p.add_argument("--llm-model", type=str, default="claude-sonnet-4-6",
                   help="LLM model name when --social-method=llm")
    p.add_argument("--social-weight", type=float, default=30.0)

    # --- MPPI ---
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--num-samples", type=int, default=512)
    p.add_argument("--dt", type=float, default=0.10)
    p.add_argument("--v-max", type=float, default=0.45)
    p.add_argument("--w-max-deg", type=float, default=120.0)

    # --- Display ---
    p.add_argument("--no-viewer", action="store_true",
                   help="Headless mode (no MuJoCo viewer)")
    p.add_argument("--seed", type=int, default=0)

    return p.parse_args()


def _parse_xy(s: str) -> tuple[float, float]:
    v = [float(x.strip()) for x in s.split(",")]
    if len(v) != 2:
        raise argparse.ArgumentTypeError("Expected x,y")
    return v[0], v[1]


def _parse_xyz(s: str) -> tuple[float, float, float]:
    v = [float(x.strip()) for x in s.split(",")]
    if len(v) != 3:
        raise argparse.ArgumentTypeError("Expected x,y,yaw_deg")
    return v[0], v[1], v[2]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Stage 1a: convert 3DSG → layout JSON (if --scene-graph provided)
    # ------------------------------------------------------------------
    if args.scene_graph is not None:
        print(f"[pipeline] Loading scene graph: {args.scene_graph}")
        scene = scene_graph_to_scene_description(args.scene_graph)
        print(
            f"[pipeline] Parsed: {len(scene.humans)} humans, "
            f"{len(scene.obstacles)} obstacles, "
            f"{len(scene.groups)} conversation groups"
        )
        for i, h in enumerate(scene.humans):
            print(f"  human_{i}: pos={h.pos}  yaw={h.yaw_deg:.1f}°  state={h.state}")
        for g in scene.groups:
            print(f"  conversation group: {g}")

        layout = scene_description_to_layout_json(
            scene=scene,
            scene_xml=str(args.scene_xml),
            robot_start=args.start_pose,
        )

        if args.layout_out is not None:
            layout_path = args.layout_out
        else:
            # Write to a temp file that lives for this process
            tmp = tempfile.NamedTemporaryFile(
                suffix="_layout.json", delete=False, mode="w"
            )
            layout_path = Path(tmp.name)
            tmp.close()

        save_layout_json(layout, layout_path)
        print(f"[pipeline] Layout JSON saved → {layout_path}")

    else:
        layout_path = args.layout_json
        print(f"[pipeline] Using existing layout JSON: {layout_path}")

    # ------------------------------------------------------------------
    # Stage 1b: run social navigation in MoLMoSpaces
    # ------------------------------------------------------------------
    print(f"[pipeline] Starting navigation  goal={args.goal_xy}  method={args.social_method}")

    # Re-use run_episode() from run_social_nav.py by constructing its Namespace
    nav_args = argparse.Namespace(
        scene_xml=args.scene_xml,
        layout_json=layout_path,
        layout_runtime="auto",
        robot_type="navbot",
        start_pose=args.start_pose,
        goal_xy=args.goal_xy,
        goal_radius=args.goal_radius,
        max_steps=args.max_steps,
        agent_radius=0.22,
        horizon=args.horizon,
        num_samples=args.num_samples,
        num_iters=4,
        dt=args.dt,
        temperature=1.0,
        noise_alpha=0.8,
        noise_v=0.10,
        noise_w_deg=30.0,
        v_min=-0.05,
        v_max=args.v_max,
        w_max_deg=args.w_max_deg,
        cruise_speed=0.22,
        goal_stage_weight=4.0,
        goal_terminal_weight=20.0,
        heading_terminal_weight=0.5,
        collision_penalty=500.0,
        clearance_threshold=0.28,
        clearance_weight=25.0,
        control_weight=0.15,
        smoothness_weight=1.5,
        social_method=args.social_method,
        llm_model=args.llm_model,
        nn_checkpoint="checkpoints/scf2.pt",
        social_weight=args.social_weight,
        viewer_camera="follower",
        robot_camera_name="robot_0/head_camera",
        follower_camera_name="robot_0/camera_follower",
        show_ui=False,
        no_viewer=args.no_viewer,
        log_every=10,
        wall_sleep=0.02,
        seed=args.seed,
    )

    result = run_episode(nav_args)

    print("\n[pipeline] Done.")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
