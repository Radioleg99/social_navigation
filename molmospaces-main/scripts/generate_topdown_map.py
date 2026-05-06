#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a top-down map image for a MolmoSpaces scene."
    )
    parser.add_argument(
        "--mode",
        choices=("occupancy", "layout"),
        default="occupancy",
        help="occupancy: render a true navigability map; layout: plot object positions from layout JSON.",
    )
    parser.add_argument("--scene-xml", type=Path, help="Scene XML path for occupancy mode.")
    parser.add_argument("--layout-json", type=Path, help="Layout JSON path for layout mode.")
    parser.add_argument("--out", type=Path, required=True, help="Output PNG path.")
    parser.add_argument("--agent-radius", type=float, default=0.25, help="Occupancy dilation radius.")
    parser.add_argument("--px-per-m", type=int, default=200, help="Pixels per meter.")
    parser.add_argument(
        "--label-top-k",
        type=int,
        default=18,
        help="In layout mode, label up to this many common semantic categories.",
    )
    return parser


def _generate_occupancy_map(scene_xml: Path, out_path: Path, agent_radius: float, px_per_m: int) -> None:
    if scene_xml is None:
        raise ValueError("--scene-xml is required for occupancy mode")

    from molmo_spaces.utils.scene_maps import ProcTHORMap, iTHORMap

    scene_xml_str = str(scene_xml)
    if "ithor" in scene_xml_str:
        thormap = iTHORMap.from_mj_model_path(
            model_path=scene_xml_str,
            agent_radius=agent_radius,
            px_per_m=px_per_m,
            device_id=None,
        )
    elif "procthor" in scene_xml_str or "holodeck" in scene_xml_str:
        thormap = ProcTHORMap.from_mj_model_path(
            model_path=scene_xml_str,
            agent_radius=agent_radius,
            px_per_m=px_per_m,
            device_id=None,
        )
    else:
        raise ValueError(f"Could not infer scene type from path: {scene_xml}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    thormap.save(str(out_path))
    print(f"saved occupancy map to {out_path}")


def _load_layout_objects(layout_json: Path) -> list[dict]:
    if layout_json is None:
        raise ValueError("--layout-json is required for layout mode")
    with layout_json.open() as f:
        data = json.load(f)
    return data["objects"]


def _generate_layout_plot(layout_json: Path, out_path: Path, label_top_k: int) -> None:
    objs = _load_layout_objects(layout_json)
    xs = [obj["position_xyz"][0] for obj in objs]
    ys = [obj["position_xyz"][1] for obj in objs]
    semantic_counts = Counter(
        obj["category"]
        for obj in objs
        if not str(obj["category"]).startswith("FP")
        and "DeferredDecal" not in str(obj["category"])
        and "Ceiling" not in str(obj["category"])
    )
    label_categories = {name for name, _ in semantic_counts.most_common(label_top_k)}

    fig, ax = plt.subplots(figsize=(9, 9), dpi=180)
    fig.patch.set_facecolor("#fffaf0")
    ax.set_facecolor("#f7f1e3")
    ax.scatter(xs, ys, s=28, c="#1f4e79", alpha=0.9, edgecolors="white", linewidths=0.4)

    labeled_keys: set[tuple[str, float, float]] = set()
    for obj in objs:
        category = obj["category"]
        if category not in label_categories:
            continue
        x, y, _ = obj["position_xyz"]
        dedupe_key = (category, round(x, 1), round(y, 1))
        if dedupe_key in labeled_keys:
            continue
        labeled_keys.add(dedupe_key)
        ax.text(x + 0.06, y + 0.06, category, fontsize=6.5, color="#7a1f1f")

    margin = 0.8
    ax.set_title(f"{layout_json.stem} top-down layout", fontsize=13)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.18)
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    print(f"saved layout plot to {out_path}")


def main() -> None:
    args = _build_parser().parse_args()

    try:
        if args.mode == "occupancy":
            _generate_occupancy_map(args.scene_xml, args.out, args.agent_radius, args.px_per_m)
        else:
            _generate_layout_plot(args.layout_json, args.out, args.label_top_k)
    except Exception as exc:
        if args.mode == "occupancy":
            message = (
                "Failed to generate occupancy map. Common causes are: "
                "1) MolmoSpaces resource cache is missing or not writable; "
                "2) MuJoCo could not create an OpenGL context in the current environment."
            )
            raise RuntimeError(
                f"{message} Original error: {type(exc).__name__}: {exc}"
            ) from exc
        raise


if __name__ == "__main__":
    main()
