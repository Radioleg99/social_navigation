"""
view_scene.py — open any scene in the MuJoCo interactive viewer.

Usage:
    # ProcTHOR scene (reads layout from outputs/procthor/train_1/)
    .venv/bin/mjpython view_scene.py --split train --idx 1

    # iTHOR scene
    .venv/bin/mjpython view_scene.py --scene-xml assets/scenes/ithor/FloorPlan203_physics.xml \
        --layout-json assets/layouts/FloorPlan203_object_positions.json

Controls: right-drag = rotate, middle-drag = pan, scroll = zoom, Q = quit.
"""
import os, sys, time, argparse
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, ".")

from pathlib import Path
from scripts import manual_capture_rgbd as sl
import mujoco, mujoco.viewer

REPO_ROOT = Path(__file__).resolve().parent

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--split", choices=["train","val","test"], default="train")
    p.add_argument("--idx", type=int, default=1)
    g.add_argument("--scene-xml", type=Path, default=None)
    p.add_argument("--layout-json", type=Path, default=None)
    p.add_argument("--robot-type", default="navbot")
    return p.parse_args()

args = parse_args()

if args.scene_xml:
    xml_path    = args.scene_xml
    layout_path = args.layout_json
else:
    xml_path    = REPO_ROOT / "assets/scenes/procthor-10k-train" / f"{args.split}_{args.idx}.xml"
    layout_path = REPO_ROOT / "outputs/procthor" / f"{args.split}_{args.idx}" / f"{args.split}_{args.idx}_layout.json"
    if not layout_path.exists():
        # fallback: old name
        layout_path = REPO_ROOT / "outputs/procthor" / f"{args.split}_{args.idx}" / "procthor_train1_layout.json"

if not xml_path.exists():
    sys.exit(f"Scene XML not found: {xml_path}\nRun run_procthor.py first to download it.")

print(f"Scene  : {xml_path}")
print(f"Layout : {layout_path if layout_path and layout_path.exists() else '(none)'}")

m, d, source = sl.load_scene_with_optional_layout_runtime(
    xml_path,
    layout_path if layout_path and layout_path.exists() else None,
    "auto",
    robot_type_override=args.robot_type,
)
mujoco.mj_forward(m, d)
print(f"Loaded ({source}) — viewer opening. Q to quit.")

with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        mujoco.mj_step(m, d)
        viewer.sync()
        time.sleep(0.02)
