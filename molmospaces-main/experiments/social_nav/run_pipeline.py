"""
run_pipeline.py  —  Stage 1 (static) end-to-end entry point
====================================================
Takes a HumanSSG scene_graph.json → builds layout JSON → runs social
path planning in MoLMoSpaces.

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
For dynamic / real-time re-planning, use `run_stage2_dynamic.py`.
"""

from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
import sys
import tempfile
from pathlib import Path

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

from experiments.social_nav.run_stage1_static import run_stage1


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end Stage1 static social path: 3DSG → layout → static A*"
    )

    # --- Scene input (mutually exclusive: scene-graph vs layout-json) ---
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--scene-graph", type=Path, metavar="JSON",
                     help="HumanSSG scene_graph.json produced by extract_json.py")
    src.add_argument("--layout-json", type=Path, metavar="JSON",
                     help="Existing MoLMoSpaces layout JSON (skip 3DSG conversion)")

    p.add_argument("--scene-xml", type=Path, default=None,
                   help="MuJoCo scene XML. Optional when --birdview-image is provided")
    p.add_argument("--birdview-image", type=Path, default=None,
                   help="Optional occupancy/bird-view image for pure 2D Stage1 planning")
    p.add_argument("--map-bounds", type=_parse_bounds, default=None,
                   help="World bounds for --birdview-image as x_min,x_max,y_min,y_max")
    p.add_argument("--birdview-free-threshold", type=float, default=0.5)
    p.add_argument("--birdview-free-is-dark", action="store_true", default=False)
    p.add_argument("--birdview-downscale", type=int, default=5)

    p.add_argument("--layout-out", type=Path, default=None,
                   help="Where to save the generated layout JSON (default: temp file)")

    # --- Navigation (static Stage1) ---
    p.add_argument("--start-pose", type=_parse_xyz, default=(-1.4, 4.6, 0.0),
                   help="Start pose x,y,yaw_deg")
    p.add_argument("--goal-xy", type=_parse_xy, required=True,
                   help="Goal position x,y in MuJoCo world frame (metres)")

    # --- Social costmap ---
    p.add_argument("--social-method",
                   choices=["none", "rule", "llm"],
                   default="rule",
                   help="Social costmap method: none | rule (analytic) | llm")
    p.add_argument("--llm-model", type=str, default="claude-sonnet-4-6",
                   help="LLM model name when --social-method=llm")
    p.add_argument("--astar-social-weight", type=float, default=30.0)
    p.add_argument("--astar-human-block-radius", type=float, default=0.65)
    p.add_argument("--astar-num-candidates", type=int, default=3)
    p.add_argument("--astar-diversity-penalty", type=float, default=8.0)
    p.add_argument("--astar-candidate-clearance-weight", type=float, default=8.0)
    p.add_argument("--astar-smoothing", choices=["none", "shortcut"], default="shortcut")
    p.add_argument("--astar-shortcut-social-threshold", type=float, default=0.45)

    # --- Output ---
    p.add_argument("--save-topdown", type=Path, default=None, metavar="PNG",
                   help="Save bird's-eye view PNG, e.g. outputs/topdown.png")
    p.add_argument("--show-topdown", action="store_true", default=False,
                   help="Open a pure 2D bird-view window after Stage1 planning")
    p.add_argument("--out-json", type=Path, default=None, metavar="JSON",
                   help="Optional numeric output for debugging")
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


def _parse_bounds(s: str) -> tuple[float, float, float, float]:
    v = [float(x.strip()) for x in s.split(",")]
    if len(v) != 4:
        raise argparse.ArgumentTypeError("Expected x_min,x_max,y_min,y_max")
    if v[1] <= v[0] or v[3] <= v[2]:
        raise argparse.ArgumentTypeError("Bounds must satisfy x_max>x_min and y_max>y_min")
    return v[0], v[1], v[2], v[3]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    if args.scene_xml is None and args.birdview_image is None:
        raise ValueError("Either --scene-xml or --birdview-image must be provided")

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

        if args.scene_xml is not None:
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
            layout_path = None
            print("[pipeline] Bird-view Stage1 mode: skipping layout JSON generation")

    else:
        layout_path = args.layout_json
        print(f"[pipeline] Using existing layout JSON: {layout_path}")

    # ------------------------------------------------------------------
    # Stage 1b: run static social path planner
    # ------------------------------------------------------------------
    print(f"[pipeline] Starting Stage1 static planner  goal={args.goal_xy}  method={args.social_method}")

    stage1_args = argparse.Namespace(
        scene_xml=args.scene_xml,
        birdview_image=args.birdview_image,
        map_bounds=args.map_bounds,
        birdview_free_threshold=args.birdview_free_threshold,
        birdview_free_is_dark=args.birdview_free_is_dark,
        birdview_downscale=args.birdview_downscale,
        layout_json=layout_path,
        layout_runtime="auto",
        validate_layout_runtime=False,
        robot_type="navbot",
        start_pose=args.start_pose,
        goal_xy=args.goal_xy,
        agent_radius=0.22,
        social_method=args.social_method,
        llm_model=args.llm_model,
        scene_graph=str(args.scene_graph) if args.scene_graph else None,
        astar_social_weight=args.astar_social_weight,
        astar_human_block_radius=args.astar_human_block_radius,
        astar_num_candidates=args.astar_num_candidates,
        astar_diversity_penalty=args.astar_diversity_penalty,
        astar_candidate_clearance_weight=args.astar_candidate_clearance_weight,
        astar_smoothing=args.astar_smoothing,
        astar_shortcut_social_threshold=args.astar_shortcut_social_threshold,
        save_topdown=args.save_topdown,
        show_topdown=args.show_topdown,
        out_json=args.out_json,
        seed=args.seed,
    )

    result = run_stage1(stage1_args)

    print("\n[pipeline] Done.")
    summary_keys = (
        "scene_xml",
        "birdview_image",
        "scene_graph",
        "social_method",
        "start_xy",
        "goal_xy",
        "num_waypoints",
        "num_raw_waypoints",
        "num_shortest_waypoints",
        "path_length_m",
        "path_social_sum",
        "shortest_path_length_m",
        "shortest_path_social_sum",
        "social_delta_vs_shortest",
        "length_ratio_vs_shortest",
        "path_min_clearance_m",
        "path_mean_clearance_m",
        "straight_social_sum",
        "social_improvement_vs_straight",
    )
    print(json.dumps({k: result.get(k) for k in summary_keys if k in result}, indent=2))


if __name__ == "__main__":
    main()
