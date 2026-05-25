#!/usr/bin/env python3
"""
eval_prompt.py  —  Semantic intensity ordering test for llm_costmap prompt.

Each scenario is described in natural language; positions are auto-generated.
Assertions check that intensity ordering matches social norms.

Usage:
    python experiments/social_nav/eval_prompt.py
    python experiments/social_nav/eval_prompt.py --model moonshot-v1-8k
    python experiments/social_nav/eval_prompt.py --indices 0,1,5
    python experiments/social_nav/eval_prompt.py --verbose
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.social_nav.cost.llm_costmap import (
    _SOCIAL_REGION_SYSTEM,
    _SOCIAL_REGION_SYSTEM_FAST,
    _call_llm,
    _parse_scored_scene,
    build_social_region_prompt,
)

# ---------------------------------------------------------------------------
# Intensity rank
# ---------------------------------------------------------------------------

RANK = {"low": 1, "medium": 2, "high": 3, "critical": 4}
RANK_NAME = {v: k for k, v in RANK.items()}

# ---------------------------------------------------------------------------
# Test scenarios
#
# people:    {id → activity description}
# groups:    conversation/interaction groups
# assert_gt: [(a, b)] meaning effective_intensity(a) > effective_intensity(b)
# ---------------------------------------------------------------------------

SCENARIOS = [
    {
        "desc": "a和b面对面聊天，c在旁边玩手机",
        "people": {
            "a": "face-to-face conversation with b",
            "b": "face-to-face conversation with a",
            "c": "playing with phone alone, not involved",
        },
        "groups": [["a", "b"]],
        "assert_gt": [("a", "c"), ("b", "c")],
    },
    {
        "desc": "a正在打电话，b和c在一起看电视",
        "people": {
            "a": "talking on the phone alone, pacing",
            "b": "watching TV together with c, seated",
            "c": "watching TV together with b, seated",
        },
        "groups": [["b", "c"]],
        "assert_gt": [("b", "a"), ("c", "a")],
    },
    {
        "desc": "a在给b讲题，b在记笔记，c在旁边发呆",
        "people": {
            "a": "tutoring b, explaining a problem",
            "b": "listening and taking notes from a",
            "c": "spacing out, not involved",
        },
        "groups": [["a", "b"]],
        "assert_gt": [("a", "c"), ("b", "c")],
    },
    {
        "desc": "a、b、c围坐开会，d在角落刷手机",
        "people": {
            "a": "group meeting, active discussion",
            "b": "group meeting, active discussion",
            "c": "group meeting, active discussion",
            "d": "scrolling phone in corner, unrelated to meeting",
        },
        "groups": [["a", "b", "c"]],
        "assert_gt": [("a", "d"), ("b", "d"), ("c", "d")],
    },
    {
        "desc": "a和b在下棋，c在旁边打盹",
        "people": {
            "a": "playing chess with b, focused",
            "b": "playing chess with a, focused",
            "c": "dozing off nearby, not involved",
        },
        "groups": [["a", "b"]],
        "assert_gt": [("a", "c"), ("b", "c")],
    },
    {
        "desc": "a和b在争吵，c在劝架，d在远处发呆",
        "people": {
            "a": "arguing with b, emotionally heated",
            "b": "arguing with a, emotionally heated",
            "c": "mediating between a and b, trying to calm them",
            "d": "daydreaming far away, unrelated",
        },
        "groups": [["a", "b"]],
        "assert_gt": [("a", "c"), ("b", "c"), ("c", "d"), ("a", "d"), ("b", "d")],
    },
    {
        "desc": "a在给b剪头发，c坐在旁边等，d在窗边发呆",
        "people": {
            "a": "cutting b's hair, precise physical task requiring stillness",
            "b": "getting a haircut from a, sitting still",
            "c": "waiting for a turn, seated nearby",
            "d": "staring out the window, not involved",
        },
        "groups": [["a", "b"]],
        "assert_gt": [("a", "c"), ("b", "c"), ("a", "d"), ("b", "d"), ("c", "d")],
    },
    {
        "desc": "a和b一起唱歌，c在旁边鼓掌，d在角落睡觉",
        "people": {
            "a": "singing a duet with b, performing",
            "b": "singing a duet with a, performing",
            "c": "clapping along, engaged audience",
            "d": "sleeping in the corner",
        },
        "groups": [["a", "b"]],
        "assert_gt": [("a", "d"), ("b", "d"), ("c", "d"), ("a", "c"), ("b", "c")],
    },
    {
        "desc": "a在给b按摩，c和d在旁边聊天",
        "people": {
            "a": "giving b a massage, close physical contact",
            "b": "receiving massage from a, relaxed",
            "c": "chatting with d, casual",
            "d": "chatting with c, casual",
        },
        "groups": [["a", "b"], ["c", "d"]],
        "assert_gt": [("a", "c"), ("a", "d"), ("b", "c"), ("b", "d")],
    },
    {
        "desc": "a、b、c三人在合影，d在旁边帮忙拍照，e在远处等",
        "people": {
            "a": "posing for group photo with b and c",
            "b": "posing for group photo with a and c",
            "c": "posing for group photo with a and b",
            "d": "helping take the photo, actively assisting",
            "e": "waiting far away, unrelated",
        },
        "groups": [["a", "b", "c"]],
        "assert_gt": [
            ("a", "e"), ("b", "e"), ("c", "e"),
            ("d", "e"), ("a", "d"), ("b", "d"), ("c", "d"),
        ],
    },
    {
        "desc": "a在哭，b在安慰a，c和d在旁边小声说话",
        "people": {
            "a": "crying, emotionally distressed",
            "b": "comforting a, emotional support, close",
            "c": "whispering with d nearby",
            "d": "whispering with c nearby",
        },
        "groups": [["a", "b"], ["c", "d"]],
        "assert_gt": [("a", "c"), ("a", "d"), ("b", "c"), ("b", "d")],
    },
    {
        "desc": "a和b在打牌，c在旁边观战，d在旁边睡觉",
        "people": {
            "a": "playing cards with b, competitive focused game",
            "b": "playing cards with a, competitive focused game",
            "c": "watching the card game, engaged observer",
            "d": "sleeping nearby, unrelated",
        },
        "groups": [["a", "b"]],
        "assert_gt": [("a", "d"), ("b", "d"), ("c", "d"), ("a", "c"), ("b", "c")],
    },
    {
        "desc": "a在给b包扎伤口，c在旁边递东西，d在角落发呆",
        "people": {
            "a": "bandaging b's wound, urgent first aid",
            "b": "being treated for injury, receiving first aid from a",
            "c": "handing medical supplies to a, actively assisting",
            "d": "daydreaming in the corner, unrelated",
        },
        "groups": [["a", "b"]],
        "assert_gt": [("a", "d"), ("b", "d"), ("c", "d"), ("a", "c"), ("b", "c")],
    },
    {
        "desc": "a和b在拼图，c在旁边看书，d在窗边发呆",
        "people": {
            "a": "doing jigsaw puzzle with b, collaborative",
            "b": "doing jigsaw puzzle with a, collaborative",
            "c": "reading a book alone, moderately focused",
            "d": "gazing out the window, unfocused",
        },
        "groups": [["a", "b"]],
        "assert_gt": [("a", "d"), ("b", "d"), ("a", "c"), ("b", "c"), ("c", "d")],
    },
    {
        "desc": "a、b、c在一起跳舞，d在旁边看，e在角落玩手机",
        "people": {
            "a": "dancing in group with b and c, energetic",
            "b": "dancing in group with a and c, energetic",
            "c": "dancing in group with a and b, energetic",
            "d": "watching the dance, engaged audience",
            "e": "playing on phone in corner, disengaged",
        },
        "groups": [["a", "b", "c"]],
        "assert_gt": [
            ("a", "e"), ("b", "e"), ("c", "e"),
            ("d", "e"), ("a", "d"), ("b", "d"), ("c", "d"),
        ],
    },
    {
        "desc": "a在给b讲故事，b在认真听，c和d各自刷手机",
        "people": {
            "a": "telling a story to b, animated and expressive",
            "b": "listening attentively to a's story",
            "c": "scrolling phone alone, not involved",
            "d": "scrolling phone alone, not involved",
        },
        "groups": [["a", "b"]],
        "assert_gt": [("a", "c"), ("a", "d"), ("b", "c"), ("b", "d")],
    },
    {
        "desc": "a和b在争着看同一本书，c在旁边吃东西，d在发呆",
        "people": {
            "a": "sharing a book with b, both reading together closely",
            "b": "sharing a book with a, both reading together closely",
            "c": "eating snacks nearby, not involved",
            "d": "daydreaming, not involved",
        },
        "groups": [["a", "b"]],
        "assert_gt": [("a", "d"), ("b", "d"), ("a", "c"), ("b", "c"), ("c", "d")],
    },
    {
        "desc": "a、b、c在讨论地图，d在旁边整理背包，e在远处休息",
        "people": {
            "a": "discussing a map with b and c, planning together",
            "b": "discussing a map with a and c, planning together",
            "c": "discussing a map with a and b, planning together",
            "d": "organizing backpack nearby, slightly involved",
            "e": "resting far away, unrelated",
        },
        "groups": [["a", "b", "c"]],
        "assert_gt": [
            ("a", "e"), ("b", "e"), ("c", "e"),
            ("d", "e"), ("a", "d"), ("b", "d"), ("c", "d"),
        ],
    },
    {
        "desc": "a在教b跳舞，c在旁边模仿，d在角落玩手机",
        "people": {
            "a": "teaching b to dance, instructing with demonstrations",
            "b": "learning dance moves from a, following closely",
            "c": "imitating the dance moves nearby, semi-involved",
            "d": "playing on phone in corner, not involved",
        },
        "groups": [["a", "b"]],
        "assert_gt": [("a", "d"), ("b", "d"), ("c", "d"), ("a", "c"), ("b", "c")],
    },
    {
        "desc": "a和b在拥抱，c在旁边尴尬地站着，d在远处发呆",
        "people": {
            "a": "hugging b, intimate personal moment",
            "b": "hugging a, intimate personal moment",
            "c": "standing nearby awkwardly, uninvited witness",
            "d": "daydreaming far away, not involved",
        },
        "groups": [["a", "b"]],
        "assert_gt": [("a", "d"), ("b", "d"), ("a", "c"), ("b", "c"), ("c", "d")],
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _auto_layout(people: dict, groups: list[list[str]]) -> list[dict]:
    """Place people in a circle; group members face each other / centroid."""
    ids = list(people.keys())
    n = len(ids)
    R = max(2.0, n * 0.6)
    pos: dict[str, list[float]] = {}
    facing: dict[str, float] = {}

    for i, pid in enumerate(ids):
        a = 2 * math.pi * i / n
        pos[pid] = [round(R * math.cos(a), 2), round(R * math.sin(a), 2)]
        facing[pid] = 0.0

    for group in groups:
        valid = [pid for pid in group if pid in pos]
        if len(valid) == 2:
            a, b = valid
            dx = pos[b][0] - pos[a][0]
            dy = pos[b][1] - pos[a][1]
            facing[a] = round(math.degrees(math.atan2(dy, dx)), 1)
            facing[b] = round(math.degrees(math.atan2(-dy, -dx)), 1)
        elif len(valid) > 2:
            cx = sum(pos[pid][0] for pid in valid) / len(valid)
            cy = sum(pos[pid][1] for pid in valid) / len(valid)
            for pid in valid:
                dx, dy = cx - pos[pid][0], cy - pos[pid][1]
                facing[pid] = round(math.degrees(math.atan2(dy, dx)), 1)

    return [
        {"id": pid, "position": pos[pid], "facing_deg": facing[pid], "activity": act}
        for pid, act in people.items()
    ]


def _entity_rank(edges: list[dict], entity_id: str) -> float:
    """Max edge score involving this entity (0 if no edges)."""
    best = 0.0
    for e in edges:
        if e["a"] == entity_id or e["b"] == entity_id:
            best = max(best, e["score"])
    return best


# ---------------------------------------------------------------------------
# Run one scenario
# ---------------------------------------------------------------------------

def run_scenario(scenario: dict, model: str, fast: bool = False) -> dict:
    humans = _auto_layout(scenario["people"], scenario.get("groups", []))
    payload = {
        "humans": humans,
        "objects": [],
        "groups": scenario.get("groups", []),
        "robot": {"start": [0.0, 0.0], "goal": [8.0, 8.0]},
    }
    system = _SOCIAL_REGION_SYSTEM_FAST if fast else _SOCIAL_REGION_SYSTEM
    prompt = build_social_region_prompt(payload, system)

    t0 = time.perf_counter()
    try:
        resp = _call_llm(prompt, model)
        elapsed = time.perf_counter() - t0
        reasoning, edges = _parse_scored_scene(resp)
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return {"passed": False, "error": str(exc), "failures": [], "edges": [], "elapsed": elapsed}

    failures = []
    for a, b in scenario.get("assert_gt", []):
        ra = _entity_rank(edges, a)
        rb = _entity_rank(edges, b)
        if ra <= rb:
            failures.append(f"{a}={ra:.2f} should > {b}={rb:.2f}")

    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "error": None,
        "reasoning": reasoning,
        "edges": edges,
        "elapsed": elapsed,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Eval llm_costmap prompt quality")
    parser.add_argument("--model", default="moonshot-v1-8k", help="LLM model name")
    parser.add_argument("--indices", default=None, help="Comma-separated scenario indices (default: all)")
    parser.add_argument("--verbose", action="store_true", help="Print LLM reasoning on failure")
    parser.add_argument("--fast", action="store_true", help="Use compact prompt (faster, fewer tokens)")
    args = parser.parse_args()

    indices = list(range(len(SCENARIOS)))
    if args.indices:
        indices = [int(x.strip()) for x in args.indices.split(",")]

    prompt_label = "FAST" if args.fast else "FULL"
    print(f"Prompt: {prompt_label}  Model: {args.model}\n")

    passed = failed = errors = 0
    latencies: list[float] = []

    for i in indices:
        s = SCENARIOS[i]
        print(f"\n[{i:02d}] {s['desc']}")
        result = run_scenario(s, args.model, fast=args.fast)
        elapsed = result.get("elapsed", 0.0)
        latencies.append(elapsed)

        if result["error"]:
            print(f"  ERROR  {result['error']}  ({elapsed:.2f}s)")
            errors += 1
        elif result["passed"]:
            print(f"  PASS  ({elapsed:.2f}s)")
            passed += 1
        else:
            print(f"  FAIL  ({elapsed:.2f}s)")
            for f in result["failures"]:
                print(f"    ✗ {f}")
            if args.verbose:
                print(f"  --- reasoning ---\n{result['reasoning']}")
            failed += 1

    total = passed + failed + errors
    avg = sum(latencies) / len(latencies) if latencies else 0.0
    p50 = sorted(latencies)[len(latencies) // 2] if latencies else 0.0
    p95 = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0.0
    print(f"\n{'='*50}")
    print(f"  {passed}/{total} passed   {failed} failed   {errors} errors")
    print(f"  latency  avg={avg:.2f}s  p50={p50:.2f}s  p95={p95:.2f}s")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
