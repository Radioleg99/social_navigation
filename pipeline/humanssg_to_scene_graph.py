"""
Convert HumanSSG obj_json + edge_json → scene_graph.json
expected by pipeline/scene_bridge.py.

Usage:
    python pipeline/humanssg_to_scene_graph.py \
        --obj-json  .../obj_json_*.json \
        --edge-json .../edge_json_*.json \
        --out       assets/layouts/scene_graph_floorplan203.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def convert(obj_json_path: str, edge_json_path: str, out_path: str) -> None:
    with open(obj_json_path) as f:
        objects: dict = json.load(f)
    with open(edge_json_path) as f:
        edges: dict = json.load(f)

    nodes = []
    for key, obj in objects.items():
        obj_id = obj["id"]
        tag    = obj.get("object_tag", obj.get("class_name", "unknown"))

        # collect activities for this object from edge_json
        activities = []
        for edge in edges.values():
            if edge.get("object_1_id") == obj_id:
                activities.append({
                    "name":   edge.get("relationship", ""),
                    "object": str(edge.get("object_2_id", "")),
                })

        node = {
            "node_id":    obj_id,
            "label":      [tag],
            "bbox_center": obj.get("bbox_center", [0, 0, 0]),
            "bbox_extent": obj.get("bbox_extent",  [0, 0, 0]),
        }
        if activities:
            node["activities"] = activities

        nodes.append(node)

    # edges: [id1, id2, {"desc": "...", "relation": "..."}]
    edge_list = []
    for edge in edges.values():
        i = edge.get("object_1_id")
        j = edge.get("object_2_id")
        rel = edge.get("relationship", "")
        desc = edge.get("edge_description", rel)
        edge_list.append([i, j, {"desc": desc, "relation": rel}])

    scene_graph = {"nodes": nodes, "edges": edge_list}

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(scene_graph, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(nodes)} nodes, {len(edge_list)} edges → {out_path}")

    persons = [n for n in nodes if "person" in n["label"][0].lower()]
    print(f"Persons ({len(persons)}):")
    for p in persons:
        c = p["bbox_center"]
        acts = [a["name"] for a in p.get("activities", [])]
        print(f"  id={p['node_id']:3d}  pos=({c[0]:.2f},{c[1]:.2f})  activities={acts}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj-json",  required=True)
    ap.add_argument("--edge-json", required=True)
    ap.add_argument("--out",       required=True)
    args = ap.parse_args()
    convert(args.obj_json, args.edge_json, args.out)


if __name__ == "__main__":
    main()
