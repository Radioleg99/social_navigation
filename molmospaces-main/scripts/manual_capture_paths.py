#!/usr/bin/env python3
"""Centralized output paths for manual RGBD capture."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.output_root_config import DEFAULT_OUTPUT_ROOT

DEFAULT_CAPTURE_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
DEFAULT_CAPTURE_SCENE_ID = "ManualCapture/frames"


def resolve_capture_paths(
    output_root: Path = DEFAULT_CAPTURE_OUTPUT_ROOT,
    scene_id: str = DEFAULT_CAPTURE_SCENE_ID,
) -> dict[str, Path]:
    scene_root = output_root / scene_id
    return {
        "scene_root": scene_root,
        "color_dir": scene_root / "color",
        "depth_dir": scene_root / "depth",
        "pose_dir": scene_root / "pose",
        "intrinsics": scene_root / "intrinsics.txt",
        "dataconfig": scene_root / "dataconfig.yaml",
        "capture_meta": scene_root / "capture_meta.json",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", type=str, default=DEFAULT_CAPTURE_SCENE_ID)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_CAPTURE_OUTPUT_ROOT)
    args = parser.parse_args()

    paths = resolve_capture_paths(output_root=args.output_root, scene_id=args.scene_id)
    print(f"scene_root={paths['scene_root']}")
    print(f"color_dir={paths['color_dir']}")
    print(f"depth_dir={paths['depth_dir']}")
    print(f"pose_dir={paths['pose_dir']}")
    print(f"intrinsics={paths['intrinsics']}")
    print(f"dataconfig={paths['dataconfig']}")
    print(f"capture_meta={paths['capture_meta']}")


if __name__ == "__main__":
    main()
