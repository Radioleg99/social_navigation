"""
run_procthor.py — Social navigation on ProcTHOR scenes
=======================================================

End-to-end pipeline:
  1. Download ProcTHOR scene via resource manager (if not already cached)
  2. Parse house JSON → auto-place humans from furniture layout
  3. Save layout JSON
  4. Run MPPI social navigation in MoLMoSpaces

Usage
-----
With viewer (interactive):
    ./.venv/bin/mjpython experiments/social_nav/run_procthor.py \\
        --split train --idx 1 \\
        --goal-xy 3.0,2.0

With LLM social costmap:
    ./.venv/bin/mjpython experiments/social_nav/run_procthor.py \\
        --split train --idx 1 \\
        --goal-xy 3.0,2.0 \\
        --social-method llm --llm-model claude-sonnet-4-6

Headless with PNG + GIF output:
    ./.venv/bin/mjpython experiments/social_nav/run_procthor.py \\
        --split train --idx 1 \\
        --goal-xy 2.0,1.5 \\
        --no-viewer \\
        --save-topdown outputs/procthor_train1.png \\
        --save-gif     outputs/procthor_train1.gif \\
        --layout-out   outputs/procthor_train1_layout.json

Coordinate note
---------------
ProcTHOR uses AI2-THOR convention (y-up): position.x → MuJoCo x,
position.z → MuJoCo y.  If humans appear displaced, check the scene's
coordinate frame and adjust _to_mujoco_xy() in pipeline/procthor_to_layout.py.
"""

from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

# ── repo paths ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]   # molmospaces-main/
GRAD_ROOT = REPO_ROOT.parent                       # graduation/

for _p in (str(REPO_ROOT), str(GRAD_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# =============================================================================
# Scene download helpers
# =============================================================================

def _ensure_scene(split: str, idx: int) -> Path:
    """
    Return path to the scene XML, downloading the archive first if needed.
    Uses the MoLMoSpaces resource manager (no external dependencies required).
    """
    from molmo_spaces.utils.lazy_loading_utils import install_scene_from_source_index

    source    = f"procthor-10k-{split}"
    scene_dir = REPO_ROOT / "assets" / "scenes" / source
    xml_path  = scene_dir / f"{split}_{idx}.xml"

    if xml_path.is_file():
        return xml_path

    print(f"[procthor] Scene not found locally — downloading {source}/{split}_{idx} ...")
    install_scene_from_source_index(source, idx)

    if not xml_path.is_file():
        available = sorted(scene_dir.glob(f"{split}_{idx}*"))
        raise FileNotFoundError(
            f"Download finished but XML not found: {xml_path}\n"
            f"Files present: {available}"
        )
    print(f"[procthor] Downloaded → {xml_path}")
    return xml_path


def _load_house_json(xml_path: Path) -> dict:
    """Load the ProcTHOR house JSON co-located with the scene XML."""
    json_path = xml_path.with_suffix(".json")
    if not json_path.is_file():
        raise FileNotFoundError(
            f"House JSON not found: {json_path}\n"
            "Expected it to be extracted alongside the scene XML."
        )
    with open(json_path) as f:
        return json.load(f)


# =============================================================================
# Auto start-pose from house layout
# =============================================================================

def _auto_start_pose(house: dict) -> tuple[float, float, float]:
    """
    Pick a start pose from the centroid of the largest living/dining room.
    Offset slightly so the robot doesn't start exactly at the centroid.
    Returns (x, y, yaw_deg).
    """
    from pipeline.procthor_to_layout import largest_room_info, _polygon_area

    rooms = house.get("rooms", [])
    if not rooms:
        return 0.0, 0.0, 0.0

    # Prefer living / dining rooms as open spaces
    preferred = [r for r in rooms if r.get("roomType") in ("LivingRoom", "DiningRoom")]
    pool = preferred if preferred else rooms
    best = max(pool, key=lambda r: _polygon_area(r.get("floorPolygon", [])))

    poly = best.get("floorPolygon", [])
    if not poly:
        return 0.0, 0.0, 0.0

    cx = sum(p["x"] for p in poly) / len(poly)
    cz = sum(p["z"] for p in poly) / len(poly)
    # Offset 0.6 m from centroid toward one corner so robot starts in free space
    return round(cx - 0.6, 2), round(cz - 0.6, 2), 0.0


# =============================================================================
# Argument parsing
# =============================================================================

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Scene selection ──────────────────────────────────────────────────────
    p.add_argument("--split", choices=["train", "val", "test"], default="train",
                   help="ProcTHOR dataset split")
    p.add_argument("--idx", type=int, default=1,
                   help="Scene index within the split (0-based, e.g. 1 = dataset['train'][1])")
    p.add_argument("--max-humans", type=int, default=4,
                   help="Max humans auto-placed from furniture layout")

    # ── Navigation ───────────────────────────────────────────────────────────
    p.add_argument("--start-pose", type=_parse_xyz, default=None,
                   help="x,y,yaw_deg — if omitted, auto-detected from house JSON")
    p.add_argument("--goal-xy", type=_parse_xy, required=True,
                   help="Goal position x,y in MuJoCo world frame (metres)")
    p.add_argument("--goal-radius", type=float, default=0.20)
    p.add_argument("--max-steps", type=int, default=600)

    # ── Social costmap ───────────────────────────────────────────────────────
    p.add_argument("--social-method", choices=["none", "rule", "llm"], default="rule",
                   help="Social costmap: none | rule (analytic Gaussian) | llm (LLM-generated)")
    p.add_argument("--llm-model", type=str, default="moonshot-v1-8k-vision-preview")
    p.add_argument("--social-weight", type=float, default=1.0,
                   help="Weight of social costmap in MPPI cost function; keep low when A* owns social routing")
    p.add_argument("--social-back-scale", type=float, default=1.0,
                   help="Rear personal-space sigma scale; increase to discourage passing behind people")
    p.add_argument("--scene-graph", type=Path, default=None, metavar="JSON",
                   help="scene_graph.json from HumanSSG/procthor_to_scene_graph.py")

    # ── MPPI ─────────────────────────────────────────────────────────────────
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--num-samples", type=int, default=512)
    p.add_argument("--dt", type=float, default=0.10)
    p.add_argument("--v-max", type=float, default=0.45)
    p.add_argument("--w-max-deg", type=float, default=120.0)
    p.add_argument("--astar-social-weight", type=float, default=30.0,
                   help="Weight of social costmap in A* global planning")
    p.add_argument("--astar-human-block-radius", type=float, default=0.65,
                   help="Treat this radius around each human as non-traversable for A*")
    p.add_argument("--astar-num-candidates", type=int, default=1,
                   help="Number of alternative A* routes to generate and score")
    p.add_argument("--astar-diversity-penalty", type=float, default=8.0,
                   help="Soft penalty around previously found A* candidates")
    p.add_argument("--astar-candidate-clearance-weight", type=float, default=8.0,
                   help="Path-level low-clearance penalty weight when choosing an A* candidate")
    p.add_argument("--astar-replan-interval", type=int, default=10,
                   help="Replan A* from current robot pose every N steps; 0 disables periodic replanning")
    p.add_argument("--mppi-target-lookahead", type=float, default=1.20,
                   help="Track a point this far ahead on the A* path instead of the next grid waypoint")
    p.add_argument("--waypoint-radius", type=float, default=0.35,
                   help="Waypoint progress radius in metres")
    p.add_argument("--astar-smoothing", choices=["none", "shortcut"], default="shortcut",
                   help="Post-process A* path before MPPI tracking")
    p.add_argument("--astar-shortcut-social-threshold", type=float, default=0.45,
                   help="Maximum social grid cost a shortcut segment may cross")

    # ── Display / output ─────────────────────────────────────────────────────
    p.add_argument("--no-viewer", action="store_true",
                   help="Headless mode — no MuJoCo viewer window")
    p.add_argument("--save-topdown", type=Path, default=None, metavar="PNG",
                   help="Save final bird's-eye PNG, e.g. outputs/episode.png")
    p.add_argument("--save-gif", type=Path, default=None, metavar="GIF",
                   help="Save animated trajectory GIF")
    p.add_argument("--show-topdown", action="store_true",
                   help="Show live top-down view (cv2 subprocess)")
    p.add_argument("--log-every", type=int, default=10,
                   help="Print navigation status every N steps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=None, metavar="DIR",
                   help="Output directory (default: outputs/procthor/{split}_{idx}/)")
    p.add_argument("--layout-out", type=Path, default=None, metavar="JSON",
                   help="Save generated layout JSON here (overrides --out-dir for layout)")
    p.add_argument("--force-regen", action="store_true",
                   help="Force regeneration of layout JSON even if it already exists")

    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    # ── 1. Ensure scene is downloaded ────────────────────────────────────────
    xml_path = _ensure_scene(args.split, args.idx)
    print(f"[procthor] Scene XML  : {xml_path}")

    # ── 2. Load house JSON ────────────────────────────────────────────────────
    house = _load_house_json(xml_path)
    n_rooms   = len(house.get("rooms", []))
    n_objects = len(house.get("objects", []))
    print(f"[procthor] House info : {n_rooms} rooms, {n_objects} objects")

    # Print room summary for context
    for r in house.get("rooms", []):
        from pipeline.procthor_to_layout import _polygon_area
        area = _polygon_area(r.get("floorPolygon", []))
        print(f"  {r.get('roomType', '?'):16s}  {area:.1f} m²")

    # ── 3. Generate layout JSON ───────────────────────────────────────────────
    # Resolve output directory first (needed to find existing layout path)
    out_dir = args.out_dir or (REPO_ROOT / "outputs" / "procthor" / f"{args.split}_{args.idx}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[procthor] Output dir : {out_dir}")

    if args.layout_out is not None:
        layout_path = args.layout_out
        layout_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        layout_path = out_dir / f"{args.split}_{args.idx}_layout.json"

    if layout_path.exists() and not args.force_regen:
        print(f"[procthor] 使用已有 layout（手动编辑保留）→ {layout_path}")
        with open(layout_path) as f:
            layout = json.load(f)
        start_pose = args.start_pose or _auto_start_pose(house)
    else:
        from pipeline.procthor_to_layout import house_json_to_layout_json

        start_pose = args.start_pose or _auto_start_pose(house)
        print(f"[procthor] Start pose : {start_pose}")

        layout = house_json_to_layout_json(
            house=house,
            scene_xml=xml_path,
            robot_start=start_pose,
            max_humans=args.max_humans,
            seed=args.seed,
        )

        with open(layout_path, "w") as f:
            json.dump(layout, f, indent=2)
        print(f"[procthor] Layout saved → {layout_path}")

    # Print human summary
    hr = layout["human_runtime"]
    all_humans = [hr] + hr.get("extra_humans", [])
    print(f"[procthor] Humans placed: {len(all_humans)}")
    for i, h in enumerate(all_humans):
        pos      = h.get("human_pos", [0, 0, 0])
        xml_name = Path(h.get("human_xml", "unknown/character.xml")).parent.name
        print(f"  human_{i}: pos=({pos[0]:.2f}, {pos[1]:.2f})  "
              f"yaw={h.get('human_yaw_deg', 0):.0f}°  model={xml_name}")

    # Wire up save paths into out_dir if not explicitly overridden
    if args.save_topdown is None:
        args.save_topdown = out_dir / f"{args.split}_{args.idx}_topdown.png"
    if args.save_gif is None:
        args.save_gif = out_dir / f"{args.split}_{args.idx}.gif"

    # ── 4. Run social navigation ──────────────────────────────────────────────
    from experiments.social_nav.run_social_nav import run_episode

    nav_args = argparse.Namespace(
        # Scene
        scene_xml      = xml_path,
        layout_json    = layout_path,
        layout_runtime = "auto",
        robot_type     = "navbot",
        # Navigation
        start_pose     = start_pose,
        goal_xy        = args.goal_xy,
        goal_radius    = args.goal_radius,
        max_steps      = args.max_steps,
        agent_radius   = 0.22,
        # MPPI hyperparams
        horizon        = args.horizon,
        num_samples    = args.num_samples,
        num_iters      = 4,
        dt             = args.dt,
        temperature    = 1.0,
        noise_alpha    = 0.8,
        noise_v        = 0.10,
        noise_w_deg    = 30.0,
        v_min          = -0.05,
        v_max          = args.v_max,
        w_max_deg      = args.w_max_deg,
        cruise_speed   = 0.22,
        # Cost weights
        goal_stage_weight       = 4.0,
        goal_terminal_weight    = 20.0,
        heading_terminal_weight = 0.5,
        human_radius            = 0.30,
        clearance_threshold     = 0.28,
        clearance_weight        = 25.0,
        control_weight          = 0.15,
        smoothness_weight       = 1.5,
        # Social costmap
        social_method       = args.social_method,
        llm_model           = args.llm_model,
        nn_checkpoint       = "checkpoints/scf2.pt",
        social_weight       = args.social_weight,
        social_back_scale   = args.social_back_scale,
        llm_update_interval = 0,
        scene_graph         = str(args.scene_graph) if args.scene_graph else None,
        astar_social_weight  = args.astar_social_weight,
        astar_human_block_radius = args.astar_human_block_radius,
        astar_num_candidates = args.astar_num_candidates,
        astar_diversity_penalty = args.astar_diversity_penalty,
        astar_candidate_clearance_weight = args.astar_candidate_clearance_weight,
        astar_replan_interval = args.astar_replan_interval,
        mppi_target_lookahead = args.mppi_target_lookahead,
        waypoint_radius       = args.waypoint_radius,
        astar_smoothing       = args.astar_smoothing,
        astar_shortcut_social_threshold = args.astar_shortcut_social_threshold,
        # Viewer
        viewer_camera       = "follower",
        robot_camera_name   = "robot_0/head_camera",
        follower_camera_name= "robot_0/camera_follower",
        show_ui             = False,
        no_viewer           = args.no_viewer,
        # Output
        save_topdown  = args.save_topdown,
        save_gif      = args.save_gif,
        show_topdown  = args.show_topdown,
        log_every     = args.log_every,
        wall_sleep    = 0.02,
        seed          = args.seed,
    )

    print(f"\n[procthor] Starting navigation  "
          f"goal={args.goal_xy}  social={args.social_method}")
    run_episode(nav_args)


if __name__ == "__main__":
    main()
