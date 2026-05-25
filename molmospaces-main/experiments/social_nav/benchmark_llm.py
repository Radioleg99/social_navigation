#!/usr/bin/env python3
"""
benchmark_llm.py  —  Latency + consistency benchmark across models and prompts.

Runs every (model, prompt) combination against all eval scenarios and saves
a structured results JSON plus a printed comparison table.

Usage:
    python experiments/social_nav/benchmark_llm.py
    python experiments/social_nav/benchmark_llm.py --models moonshot-v1-8k --prompts full,fast
    python experiments/social_nav/benchmark_llm.py --models moonshot-v1-8k,moonshot-v1-32k --prompts full,fast
    python experiments/social_nav/benchmark_llm.py --indices 0,1,2,3,4   # quick smoke test
    python experiments/social_nav/benchmark_llm.py --out results/bench_v1.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.social_nav.cost.llm_costmap import (
    _SOCIAL_REGION_SYSTEM,
    _SOCIAL_REGION_SYSTEM_FAST,
    _call_llm,
    _parse_scored_scene,
    build_social_region_prompt,
)
from experiments.social_nav.eval_prompt import (
    SCENARIOS as SCENARIOS_BUILTIN,
    _auto_layout,
    _entity_rank,
)

# ---------------------------------------------------------------------------
# Prompt registry  —  add new prompt variants here
# ---------------------------------------------------------------------------

PROMPTS: dict[str, str] = {
    "full": _SOCIAL_REGION_SYSTEM,
    "fast": _SOCIAL_REGION_SYSTEM_FAST,
}


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_one(scenario: dict, model: str, system_prompt: str) -> dict:
    humans = _auto_layout(scenario["people"], scenario.get("groups", []))
    payload = {
        "humans": humans,
        "objects": [],
        "groups": scenario.get("groups", []),
        "robot": {"start": [0.0, 0.0], "goal": [8.0, 8.0]},
    }
    prompt = build_social_region_prompt(payload, system_prompt)

    t0 = time.perf_counter()
    try:
        resp = _call_llm(prompt, model)
        elapsed = time.perf_counter() - t0
        reasoning, edges = _parse_scored_scene(resp)
    except Exception as exc:
        return {
            "passed": False, "error": str(exc),
            "failures": [], "edges": [], "elapsed": time.perf_counter() - t0,
        }

    failures = []
    for a, b in scenario.get("assert_gt", []):
        ra, rb = _entity_rank(edges, a), _entity_rank(edges, b)
        if ra <= rb:
            failures.append(f"{a}={ra:.2f} should > {b}={rb:.2f}")

    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "error": None,
        "elapsed": elapsed,
        "edges": edges,
    }


def bench_combination(
    model: str,
    prompt_name: str,
    indices: list[int],
    scenarios: list[dict],
    verbose: bool,
) -> dict:
    system_prompt = PROMPTS[prompt_name]
    results = []
    print(f"\n  [{model}] [{prompt_name}]")

    for i in indices:
        s = scenarios[i]
        r = run_one(s, model, system_prompt)
        status = "PASS" if r["passed"] else ("ERROR" if r["error"] else "FAIL")
        print(f"    [{i:02d}] {status:5s}  {r['elapsed']:.2f}s  {s['desc'][:45]}")
        if verbose and not r["passed"] and r["failures"]:
            for f in r["failures"]:
                print(f"           ✗ {f}")
        results.append({"scenario_idx": i, "desc": s["desc"], **r})

    latencies = [r["elapsed"] for r in results]
    lat_sorted = sorted(latencies)
    n = len(lat_sorted)

    passed = sum(1 for r in results if r["passed"])
    return {
        "model": model,
        "prompt": prompt_name,
        "pass_rate": f"{passed}/{n}",
        "pass_count": passed,
        "total": n,
        "latency": {
            "avg": round(sum(latencies) / n, 3),
            "p50": round(lat_sorted[n // 2], 3),
            "p95": round(lat_sorted[int(n * 0.95)], 3),
            "min": round(lat_sorted[0], 3),
            "max": round(lat_sorted[-1], 3),
        },
        "per_scenario": results,
    }


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------

def print_table(combos: list[dict]) -> None:
    header = f"{'model':<22} {'prompt':<6} {'pass':>7}  {'avg':>6}  {'p50':>6}  {'p95':>6}"
    print(f"\n{'='*60}")
    print(header)
    print("-" * 60)
    for c in combos:
        lat = c["latency"]
        pct = int(c["pass_count"] * 100 / c["total"])
        print(
            f"{c['model']:<22} {c['prompt']:<6} "
            f"{c['pass_rate']:>5} ({pct:3d}%)  "
            f"{lat['avg']:>5.2f}s  {lat['p50']:>5.2f}s  {lat['p95']:>5.2f}s"
        )
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="moonshot-v1-8k",
                   help="Comma-separated model names (default: moonshot-v1-8k)")
    p.add_argument("--prompts", default="full,fast",
                   help="Comma-separated prompt keys: full,fast  (default: full,fast)")
    p.add_argument("--scenarios", type=Path, default=None,
                   help="Path to generated scenarios JSON (from generate_eval_scenarios.py). "
                        "If omitted, uses the 20 built-in eval_prompt scenarios.")
    p.add_argument("--indices", default=None,
                   help="Comma-separated scenario indices (default: all)")
    p.add_argument("--out", type=Path,
                   default=Path(__file__).parent / "ai_test_results" / "benchmark_latest.json",
                   help="Output JSON path")
    p.add_argument("--verbose", action="store_true",
                   help="Print assertion failures per scenario")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    models = [m.strip() for m in args.models.split(",")]
    prompts = [p.strip() for p in args.prompts.split(",")]

    for pk in prompts:
        if pk not in PROMPTS:
            print(f"Unknown prompt key '{pk}'. Available: {list(PROMPTS)}")
            sys.exit(1)

    # load scenarios
    if args.scenarios is not None:
        SCENARIOS = json.loads(args.scenarios.read_text(encoding="utf-8"))
        print(f"Loaded {len(SCENARIOS)} scenarios from {args.scenarios}")
    else:
        SCENARIOS = SCENARIOS_BUILTIN
        print(f"Using {len(SCENARIOS)} built-in scenarios")

    indices = list(range(len(SCENARIOS)))
    if args.indices:
        indices = [int(x.strip()) for x in args.indices.split(",")]

    print(f"Benchmark  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models : {models}")
    print(f"Prompts: {prompts}")
    print(f"Scenarios: {len(indices)}")

    combos = []
    for model in models:
        for prompt_name in prompts:
            result = bench_combination(model, prompt_name, indices, SCENARIOS, args.verbose)
            combos.append(result)

    print_table(combos)

    # Save full results
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now().isoformat(),
        "models": models,
        "prompts": prompts,
        "scenario_indices": indices,
        "results": combos,
    }
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nSaved → {args.out}\n")


if __name__ == "__main__":
    main()
