#!/usr/bin/env python3
"""
scene_finder.py — discover and download iTHOR/ProcTHOR scenes for social navigation.

Usage:
  python scripts/scene_finder.py list             # list all available scenes ranked by size
  python scripts/scene_finder.py list --source ithor --top 20
  python scripts/scene_finder.py analyze          # analyze already-extracted scenes
  python scripts/scene_finder.py download FloorPlan224  # download + extract a scene
  python scripts/scene_finder.py download train_4778    # ProcTHOR scene

iTHOR scene categories:
  FloorPlan1-30    → Kitchen
  FloorPlan201-230 → Living room  (best for social nav: open, larger)
  FloorPlan301-330 → Bedroom
  FloorPlan401-430 → Bathroom
"""

import argparse
import gzip
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

MOLMO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = MOLMO_ROOT / "assets"

SCENE_SOURCES = {
    "ithor":             ASSETS_DIR / "scenes" / "ithor",
    "procthor-10k-train": ASSETS_DIR / "scenes" / "procthor-10k-train",
    "procthor-10k-val":  ASSETS_DIR / "scenes" / "procthor-10k-val",
    "procthor-10k-test": ASSETS_DIR / "scenes" / "procthor-10k-test",
}

ITHOR_CATEGORY = {
    range(1, 31):   "kitchen",
    range(201, 231): "living_room",
    range(301, 331): "bedroom",
    range(401, 431): "bathroom",
}


def ithor_category(fp_num: int) -> str:
    for r, cat in ITHOR_CATEGORY.items():
        if fp_num in r:
            return cat
    return "unknown"


def load_size_manifest(source_dir: Path) -> dict[str, float]:
    p = source_dir / "mjthor_resource_file_to_size_mb.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def parse_ithor_name(key: str):
    """Extract (FloorPlan number, display name) from manifest key like ithor_FloorPlan203_physics.tar.zst"""
    m = re.search(r"FloorPlan(\d+)", key)
    if m:
        num = int(m.group(1))
        return num, f"FloorPlan{num}"
    return None, key


def list_scenes(source: str | None, top: int, category_filter: str | None):
    sources = [source] if source else list(SCENE_SOURCES.keys())
    rows = []

    for src in sources:
        d = SCENE_SOURCES.get(src)
        if d is None or not d.exists():
            continue
        sizes = load_size_manifest(d)
        for key, mb in sizes.items():
            if "Textures" in key or "refs" in key:
                continue
            if src == "ithor":
                num, name = parse_ithor_name(key)
                cat = ithor_category(num) if num else "unknown"
                if category_filter and cat != category_filter:
                    continue
                rows.append((mb, src, name, cat, key))
            else:
                m = re.search(r"(train_\d+|val_\d+|test_\d+)", key)
                name = m.group(1) if m else key
                cat = "procthor"
                if category_filter and category_filter not in ("procthor", "proc"):
                    continue
                rows.append((mb, src, name, cat, key))

    rows.sort(key=lambda x: x[0], reverse=True)
    if top:
        rows = rows[:top]

    print(f"{'MB':>6}  {'source':<26}  {'name':<18}  {'category'}")
    print("-" * 70)
    for mb, src, name, cat, _ in rows:
        print(f"{mb:6.1f}  {src:<26}  {name:<18}  {cat}")


def get_xml_dimensions(xml_path: Path) -> tuple[float, float]:
    """Return (x_span, y_span) in metres from body positions in the scene XML."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        xs, ys = [], []
        for b in root.findall(".//body"):
            pos = b.get("pos", "").split()
            if len(pos) >= 2:
                try:
                    xs.append(float(pos[0]))
                    ys.append(float(pos[1]))
                except ValueError:
                    pass
        if xs:
            return max(xs) - min(xs), max(ys) - min(ys)
    except Exception:
        pass
    return 0.0, 0.0


def count_objects(metadata_path: Path) -> int:
    try:
        with open(metadata_path) as f:
            meta = json.load(f)
        return len(meta.get("objects", {}))
    except Exception:
        return 0


def analyze_extracted():
    print(f"\nAnalyzing extracted scenes under {ASSETS_DIR / 'scenes'}\n")
    print(f"{'name':<28}  {'x-span':>7}  {'y-span':>7}  {'objects':>7}  {'category'}")
    print("-" * 65)

    for src_name, src_dir in SCENE_SOURCES.items():
        if not src_dir.exists():
            continue
        for xml in sorted(src_dir.rglob("*_physics.xml")):
            # Skip ceiling variants
            if "ceiling" in xml.name or "patched" in xml.name or "non_settled" in xml.name or "orig" in xml.name:
                continue
            meta_path = xml.with_name(xml.stem + "_metadata.json")
            x_span, y_span = get_xml_dimensions(xml)
            n_obj = count_objects(meta_path)
            num, name = parse_ithor_name(xml.name)
            cat = ithor_category(num) if num else src_name.split("-")[0]
            area = x_span * y_span
            print(f"{name:<28}  {x_span:7.1f}m  {y_span:7.1f}m  {n_obj:7d}  {cat}  (area {area:.0f} m²)")


def download_scene(scene_name: str):
    """Download and extract a single scene using the MoLMoSpaces resource manager."""
    print(f"Setting up resource manager to download '{scene_name}' ...")

    # Determine source
    if re.match(r"FloorPlan\d+", scene_name):
        source = "ithor"
        archive_key = f"ithor_{scene_name}_physics.tar.zst"
    elif re.match(r"(train|val|test)_\d+", scene_name):
        # Guess from name prefix
        split = scene_name.split("_")[0]
        source = f"procthor-10k-{split}"
        archive_key = f"procthor-10k-{split}_{scene_name}.tar.zst"
    else:
        print(f"Cannot determine source for '{scene_name}'. Provide e.g. 'FloorPlan224' or 'train_4778'.")
        sys.exit(1)

    try:
        sys.path.insert(0, str(MOLMO_ROOT))
        from molmo_spaces.molmo_spaces_constants import get_resource_manager
        rm = get_resource_manager()
        print(f"Downloading {source}/{archive_key} ...")
        rm.install_scenes({source: [archive_key]}, skip_linking=False)
        print(f"Done. Check {ASSETS_DIR}/scenes/{source}/ for the extracted files.")
    except Exception as e:
        print(f"Download failed: {e}")
        print(
            "\nAlternatively, install via the MoLMoSpaces resource manager:\n"
            f"  python -c \"from molmo_spaces.molmo_spaces_constants import get_resource_manager; "
            f"rm = get_resource_manager(); rm.install_scenes({{'{source}': ['{archive_key}']}}, skip_linking=False)\""
        )
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd")

    p_list = sub.add_parser("list", help="List all available scenes ranked by file size")
    p_list.add_argument("--source", choices=list(SCENE_SOURCES.keys()), default=None)
    p_list.add_argument("--top", type=int, default=30)
    p_list.add_argument("--category", choices=["kitchen", "living_room", "bedroom", "bathroom", "procthor"], default=None)

    sub.add_parser("analyze", help="Analyze already-extracted scene XMLs for dimensions")

    p_dl = sub.add_parser("download", help="Download and extract a scene")
    p_dl.add_argument("scene", help="Scene name, e.g. FloorPlan224 or train_4778")

    args = parser.parse_args()

    if args.cmd == "list":
        list_scenes(args.source, args.top, args.category)
    elif args.cmd == "analyze":
        analyze_extracted()
    elif args.cmd == "download":
        download_scene(args.scene)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
