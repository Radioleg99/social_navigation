"""
AI Scene Understanding Test
============================
从 HumanSSG 输出的场景图 JSON 中解析社交场景，
发送给 LLM，测试其对社交规范的理解能力。

用法：
    export KIMI_API_KEY="your_key"
    .venv/bin/python3 experiments/social_nav/ai_test.py

输出：终端打印 + 保存到 experiments/social_nav/ai_test_results/
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ── 场景 JSON 路径 ────────────────────────────────────────────────────────────

_SSG_DIR = Path("/Users/ljj/project/graduation/3dsg/HumanSSG/conceptgraph/data-output"
                "/FloorPlan203/frames_auto_train/exps"
                "/human_stage2_floorplan203_frames_auto_train_20260312_155639")

OBJ_JSON  = _SSG_DIR / "obj_json_human_stage2_floorplan203_frames_auto_train_20260312_155639.json"
EDGE_JSON = _SSG_DIR / "edge_json_human_stage2_floorplan203_frames_auto_train_20260312_155639.json"

MODEL = "kimi-k2-turbo-preview"

# ── 场景解析 ──────────────────────────────────────────────────────────────────

def load_scene(obj_path: Path, edge_path: Path) -> dict:
    """
    解析 obj_json + edge_json，提取：
    - 主要 person 节点（去重，选 bbox_volume 最大的）
    - person-person 边
    - person-object 边
    返回结构化 scene dict。
    """
    with open(obj_path) as f:
        obj_data = json.load(f)
    with open(edge_path) as f:
        edge_data = json.load(f)

    # 收集所有 person，按 id 去重，保留 bbox_volume 最大的
    persons: dict[int, dict] = {}
    for v in obj_data.values():
        if v["object_tag"] != "person":
            continue
        pid = v["id"]
        if pid not in persons or v["bbox_volume"] > persons[pid]["bbox_volume"]:
            persons[pid] = v

    # 只保留体积合理的（过滤掉噪点检测）
    persons = {pid: p for pid, p in persons.items() if p["bbox_volume"] > 0.05}

    # 收集 person-person 边（去重）
    pp_edges = []
    seen = set()
    for e in edge_data.values():
        if e["object_1_tag"] == "person" and e["object_2_tag"] == "person":
            key = tuple(sorted([e["object_1_id"], e["object_2_id"]]))
            if key not in seen:
                pp_edges.append(e)
                seen.add(key)

    # 收集 person-object 边
    po_edges = []
    for e in edge_data.values():
        if e["object_1_tag"] == "person" and e["object_2_tag"] != "person":
            if e["object_1_id"] in persons:
                po_edges.append(e)

    return dict(persons=persons, pp_edges=pp_edges, po_edges=po_edges)


def build_scene_description(scene: dict) -> str:
    """把场景结构化数据转成自然语言描述。"""
    lines = ["=== Scene Description (FloorPlan203) ===\n"]

    lines.append(f"Number of people detected: {len(scene['persons'])}\n")
    lines.append("People and their positions (x, y in world coordinates):")
    for pid, p in scene["persons"].items():
        x, y = p["bbox_center"][0], p["bbox_center"][1]
        lines.append(f"  Person {pid}: position ({x:.2f}, {y:.2f})")

    if scene["pp_edges"]:
        lines.append("\nSocial interactions between people:")
        for e in scene["pp_edges"]:
            lines.append(
                f"  Person {e['object_1_id']} → Person {e['object_2_id']}: "
                f"{e['edge_description']} (relationship: {e['relationship']})"
            )

    if scene["po_edges"]:
        lines.append("\nPeople's relation to objects:")
        for e in scene["po_edges"][:6]:  # 只取前几条，防止太长
            lines.append(
                f"  Person {e['object_1_id']}: {e['edge_description']}"
            )

    return "\n".join(lines)


# ── Prompts ───────────────────────────────────────────────────────────────────

PROMPTS = {

    "social_params": """\
You are a social navigation expert. A robot needs to navigate through the scene described below.

{scene}

Personal space around each person is modeled as an anisotropic Gaussian with three directional radii:
- sigma_front : radius in the heading/gaze direction (largest — people are most sensitive in front)
- sigma_back  : radius directly behind (smallest)
- sigma_side  : radius to the left and right

For conversing pairs, an additional "o-space" (interaction zone between them, Kendon 1990) is modeled
as a Gaussian centered at the midpoint between the two people.

For each person or social group, output the recommended social navigation parameters in JSON format.

Output format:
{{
  "entities": [
    {{
      "id": <person_id or group like "group_0_66">,
      "description": "<brief description of what they are doing>",
      "sigma_front":   <float, metres, personal space radius in heading direction>,
      "sigma_back":    <float, metres, personal space radius behind>,
      "sigma_side":    <float, metres, personal space radius to the sides>,
      "peak_cost":     <float 0.0–1.0, social cost at person's position>,
      "o_space_sigma": <float, metres, radius of interaction zone between pair — 0 if not a pair>,
      "o_space_cost":  <float 0.0–1.0, peak cost of the interaction zone>,
      "reasoning": "<why these values>"
    }}
  ],
  "navigation_advice": "<general advice for the robot navigating this scene>"
}}

Guidelines:
- sigma_front > sigma_side > sigma_back always
- Typical: sigma_front 0.8–2.5m, sigma_side 0.5–1.5m, sigma_back 0.3–0.8m
- Active conversation → large sigma, high peak_cost, large o_space_sigma (~0.8–1.2m)
- TV watcher → large sigma_front (gaze direction), moderate peak_cost
- Isolated person → smaller sigma, moderate peak_cost

Be specific and grounded in real social norms.
""",

    "path_choice": """\
You are a social navigation expert. A robot is in the scene below and needs to reach its goal.

{scene}

Robot start position: (-1.4, 4.6)
Robot goal position:  (-0.6, 4.6)

There are two possible paths:
- Path A: goes near the conversing group (around x=-4.0, y=2.0)
- Path B: goes near the person standing alone (around x=-0.8, y=3.1)

Which path is more socially appropriate and why? Answer in JSON:
{{
  "recommended_path": "A" or "B",
  "reasoning": "<detailed social reasoning, referencing anisotropic personal space and o-space norms>",
  "confidence": <0.0 to 1.0>,
  "alternative_suggestion": "<if neither is ideal, what should the robot do?>"
}}
""",

    "cost_ranking": """\
You are a social navigation expert. Given the social scene below, rank the following areas by how much the robot should avoid them (higher = more avoidance needed).

{scene}

Personal space is modeled as an anisotropic Gaussian (sigma_front > sigma_side > sigma_back).
For conversing pairs, the o-space (interaction zone between them) carries additional high cost.

Areas to rank:
1. Directly in front of the conversing group (1m ahead in their gaze direction)
2. Between the two conversing people (inside the o-space)
3. 2m to the side of the conversing group
4. Near the person standing alone (within 1m, from the front)
5. Empty corridor with no people

Output JSON:
{{
  "ranking": [
    {{"area": "<area name>", "avoidance_score": <0.0-1.0>, "reason": "<cite specific personal-space or o-space norm>"}},
    ...
  ]
}}
Rank from highest to lowest avoidance score.
""",
}


# ── LLM 调用 ──────────────────────────────────────────────────────────────────

def call_kimi(prompt: str, model: str = MODEL) -> str:
    """调用 Kimi API。"""
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["KIMI_API_KEY"],
        base_url="https://api.moonshot.cn/v1",
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content


# ── 主逻辑 ────────────────────────────────────────────────────────────────────

def run_test():
    print(f"[ai_test] 模型: {MODEL}")
    print(f"[ai_test] 场景: FloorPlan203\n")

    # 加载场景
    scene = load_scene(OBJ_JSON, EDGE_JSON)
    scene_text = build_scene_description(scene)
    print(scene_text)
    print("\n" + "="*60 + "\n")

    # 输出目录
    out_dir = Path(__file__).parent / "ai_test_results"
    out_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = out_dir / f"test_{timestamp}.txt"

    results = []

    for test_name, prompt_template in PROMPTS.items():
        print(f">>> 测试: {test_name}")
        prompt = prompt_template.format(scene=scene_text)

        t0 = time.perf_counter()
        try:
            response = call_kimi(prompt)
            elapsed = time.perf_counter() - t0
            print(f"    耗时: {elapsed:.1f}s")
            print(f"    回复:\n{response}\n")
        except Exception as e:
            response = f"ERROR: {e}"
            print(f"    错误: {e}\n")

        results.append({
            "test": test_name,
            "prompt": prompt,
            "response": response,
        })
        print("-" * 60 + "\n")

    # 保存日志
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"AI Test Results — {timestamp}\n")
        f.write(f"Model: {MODEL}\n\n")
        f.write(scene_text + "\n\n")
        f.write("=" * 60 + "\n\n")
        for r in results:
            f.write(f"=== {r['test']} ===\n")
            f.write(f"PROMPT:\n{r['prompt']}\n\n")
            f.write(f"RESPONSE:\n{r['response']}\n\n")
            f.write("-" * 60 + "\n\n")

    print(f"[ai_test] 结果已保存 → {log_file}")


if __name__ == "__main__":
    if "KIMI_API_KEY" not in os.environ:
        print("请先设置 KIMI_API_KEY：")
        print("  export KIMI_API_KEY='your_key'")
        sys.exit(1)
    run_test()
