#!/usr/bin/env python3
"""
generate_eval_scenarios.py  —  Generate N evaluation scenarios for benchmark_llm.py.

Each scenario has:
  - people: {id → activity description}
  - groups: conversation/interaction groups
  - assert_gt: ordering constraints derived from engagement levels

Engagement levels (used to derive assert_gt, NOT exposed to LLM):
  1  idle/disengaged  (scrolling phone, daydreaming, napping)
  2  solo-focused     (reading alone, working alone, waiting)
  3  social-active    (talking on phone, watching together, casual chat)
  4  high-engagement  (argument, first aid, intimate moment, group meeting)

assert_gt rules:
  - group members (level >= 3) > solo people at lower level
  - higher level > lower level (within same type)
  - group members > non-group bystanders at same level

Usage:
    python experiments/social_nav/generate_eval_scenarios.py --n 200 --out data/eval_scenarios_200.json
    python experiments/social_nav/generate_eval_scenarios.py --n 100
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from itertools import combinations

# ---------------------------------------------------------------------------
# Activity library
# ---------------------------------------------------------------------------

ACTIVITIES: list[dict] = [
    # level 1 — idle / disengaged
    {"desc": "scrolling phone alone, not involved",           "level": 1, "tags": ["solo", "idle"]},
    {"desc": "daydreaming, not involved",                      "level": 1, "tags": ["solo", "idle"]},
    {"desc": "staring out the window, unfocused",              "level": 1, "tags": ["solo", "idle"]},
    {"desc": "dozing off nearby, not involved",                "level": 1, "tags": ["solo", "idle"]},
    {"desc": "sleeping in the corner",                         "level": 1, "tags": ["solo", "idle"]},
    {"desc": "sitting quietly, disengaged",                    "level": 1, "tags": ["solo", "idle"]},
    {"desc": "standing still, looking at nothing in particular", "level": 1, "tags": ["solo", "idle"]},
    {"desc": "waiting alone, occasionally checking phone",     "level": 1, "tags": ["solo", "idle"]},

    # level 2 — solo focused
    {"desc": "reading a book alone, moderately focused",       "level": 2, "tags": ["solo", "focused"]},
    {"desc": "working on laptop alone, focused",               "level": 2, "tags": ["solo", "focused"]},
    {"desc": "eating snacks alone",                            "level": 2, "tags": ["solo", "focused"]},
    {"desc": "writing in a notebook alone",                    "level": 2, "tags": ["solo", "focused"]},
    {"desc": "organizing personal belongings",                 "level": 2, "tags": ["solo", "focused"]},
    {"desc": "listening to music with headphones, alone",      "level": 2, "tags": ["solo", "focused"]},
    {"desc": "sketching in a notebook, absorbed",              "level": 2, "tags": ["solo", "focused"]},
    {"desc": "stretching or exercising alone",                 "level": 2, "tags": ["solo", "focused"]},

    # level 3 — socially active
    {"desc": "talking on the phone, pacing",                   "level": 3, "tags": ["solo", "social"]},
    {"desc": "watching TV, casually engaged",                  "level": 3, "tags": ["solo", "social"]},
    {"desc": "face-to-face conversation, casual",              "level": 3, "tags": ["group", "social"]},
    {"desc": "listening attentively to partner",               "level": 3, "tags": ["group", "social"]},
    {"desc": "watching TV together, seated",                   "level": 3, "tags": ["group", "social"]},
    {"desc": "playing cards together, competitive",            "level": 3, "tags": ["group", "social"]},
    {"desc": "doing jigsaw puzzle together, collaborative",    "level": 3, "tags": ["group", "social"]},
    {"desc": "sharing a book, reading together closely",       "level": 3, "tags": ["group", "social"]},
    {"desc": "clapping along, engaged audience",               "level": 3, "tags": ["solo", "social"]},
    {"desc": "casually chatting with a friend",                "level": 3, "tags": ["group", "social"]},

    # level 4 — high engagement
    {"desc": "face-to-face conversation with animated discussion", "level": 4, "tags": ["group", "high"]},
    {"desc": "group meeting, active discussion",               "level": 4, "tags": ["group", "high"]},
    {"desc": "arguing with partner, emotionally heated",       "level": 4, "tags": ["group", "high"]},
    {"desc": "tutoring, explaining a problem",                 "level": 4, "tags": ["group", "high"]},
    {"desc": "listening and taking notes from tutor",          "level": 4, "tags": ["group", "high"]},
    {"desc": "giving a massage, close physical contact",       "level": 4, "tags": ["group", "high"]},
    {"desc": "receiving massage, relaxed but intimate",        "level": 4, "tags": ["group", "high"]},
    {"desc": "comforting a distressed person, emotional support", "level": 4, "tags": ["group", "high"]},
    {"desc": "crying, emotionally distressed",                 "level": 4, "tags": ["group", "high"]},
    {"desc": "hugging, intimate personal moment",              "level": 4, "tags": ["group", "high"]},
    {"desc": "bandaging wound, urgent first aid",              "level": 4, "tags": ["group", "high"]},
    {"desc": "receiving first aid, being treated for injury",  "level": 4, "tags": ["group", "high"]},
    {"desc": "teaching dance moves, instructing closely",      "level": 4, "tags": ["group", "high"]},
    {"desc": "learning dance from instructor, following closely", "level": 4, "tags": ["group", "high"]},
    {"desc": "presenting slides to colleagues, formal meeting", "level": 4, "tags": ["group", "high"]},
    {"desc": "cutting hair, precise physical task",            "level": 4, "tags": ["group", "high"]},
    {"desc": "getting haircut, sitting still",                 "level": 4, "tags": ["group", "high"]},
    {"desc": "singing a duet, performing together",            "level": 4, "tags": ["group", "high"]},
    {"desc": "playing chess, focused competitive game",        "level": 4, "tags": ["group", "high"]},
    {"desc": "mediating an argument, trying to calm people",   "level": 3, "tags": ["group", "social"]},
    {"desc": "watching an argument, curious bystander",        "level": 2, "tags": ["solo", "focused"]},
]

# pair-able group activity templates: (A_desc, B_desc) for 2-person groups
GROUP_PAIRS: list[tuple[str, str]] = [
    ("face-to-face conversation with {b}, animated",
     "face-to-face conversation with {a}, animated"),
    ("tutoring {b}, explaining a problem",
     "listening and taking notes from {a}"),
    ("arguing with {b}, emotionally heated",
     "arguing with {a}, emotionally heated"),
    ("comforting {b} who is distressed",
     "crying, being comforted by {a}"),
    ("hugging {b}, intimate personal moment",
     "hugging {a}, intimate personal moment"),
    ("bandaging {b}'s wound, urgent first aid",
     "receiving first aid from {a}, injured"),
    ("teaching {b} to dance, demonstrating moves",
     "learning dance from {a}, following closely"),
    ("cutting {b}'s hair, precise task",
     "getting haircut from {a}, sitting still"),
    ("singing a duet with {b}",
     "singing a duet with {a}"),
    ("playing chess with {b}, focused",
     "playing chess with {a}, focused"),
    ("casually chatting with {b}",
     "casually chatting with {a}"),
    ("watching TV together with {b}",
     "watching TV together with {a}"),
    ("doing jigsaw puzzle with {b}",
     "doing jigsaw puzzle with {a}"),
    ("playing cards with {b}",
     "playing cards with {a}"),
    ("giving {b} a massage",
     "receiving massage from {a}"),
    ("presenting to {b} and others, formal meeting",
     "listening to {a}'s presentation, taking notes"),
]

# 3-person group templates: list of (desc_per_member,)
GROUP_TRIPLES: list[list[str]] = [
    ["group meeting, active discussion",
     "group meeting, active discussion",
     "group meeting, active discussion"],
    ["dancing in group, energetic",
     "dancing in group, energetic",
     "dancing in group, energetic"],
    ["posing for group photo",
     "posing for group photo",
     "posing for group photo"],
    ["discussing a map together, planning",
     "discussing a map together, planning",
     "discussing a map together, planning"],
]


# ---------------------------------------------------------------------------
# Scenario builder
# ---------------------------------------------------------------------------

def _pid(i: int) -> str:
    return chr(ord("a") + i)


def _build_scenario(rng: random.Random) -> dict | None:
    n_people = rng.randint(3, 6)
    ids = [_pid(i) for i in range(n_people)]

    people: dict[str, str] = {}
    groups: list[list[str]] = []
    group_levels: dict[str, int] = {}   # person → effective level (boosted for group)

    assigned: set[str] = set()

    # decide group type
    group_type = rng.choice(["pair", "pair", "triple", "none"])
    group_ids: list[str] = []

    if group_type == "pair" and n_people >= 2:
        a, b = ids[0], ids[1]
        tmpl_a, tmpl_b = rng.choice(GROUP_PAIRS)
        people[a] = tmpl_a.format(a=a, b=b)
        people[b] = tmpl_b.format(a=a, b=b)
        groups.append([a, b])
        group_ids = [a, b]
        assigned.update([a, b])
        group_level = rng.choice([3, 4])
        group_levels[a] = group_level
        group_levels[b] = group_level

    elif group_type == "triple" and n_people >= 3:
        a, b, c = ids[0], ids[1], ids[2]
        descs = rng.choice(GROUP_TRIPLES)
        people[a], people[b], people[c] = descs[0], descs[1], descs[2]
        groups.append([a, b, c])
        group_ids = [a, b, c]
        assigned.update([a, b, c])
        group_level = rng.choice([3, 4])
        group_levels[a] = group_level
        group_levels[b] = group_level
        group_levels[c] = group_level

    # fill remaining people with solo activities
    solo_pool = [act for act in ACTIVITIES if "solo" in act["tags"]]
    for pid in ids:
        if pid not in assigned:
            act = rng.choice(solo_pool)
            people[pid] = act["desc"]
            group_levels[pid] = act["level"]

    if not groups:
        return None  # skip no-group scenes (not interesting for social nav)

    # derive assert_gt
    assert_gt: list[tuple[str, str]] = []
    for gid in group_ids:
        for pid in ids:
            if pid not in group_ids:
                if group_levels[gid] > group_levels[pid]:
                    assert_gt.append((gid, pid))
                elif group_levels[gid] == group_levels[pid]:
                    assert_gt.append((gid, pid))  # group membership wins at same level

    # also add solo level comparisons among non-group members
    non_group = [p for p in ids if p not in group_ids]
    for pa, pb in combinations(non_group, 2):
        if group_levels[pa] > group_levels[pb] + 1:   # only clear gaps
            assert_gt.append((pa, pb))
        elif group_levels[pb] > group_levels[pa] + 1:
            assert_gt.append((pb, pa))

    if not assert_gt:
        return None

    desc_parts = []
    for pid in ids:
        if pid in group_ids:
            desc_parts.append(f"{pid}在群体互动")
        else:
            desc_parts.append(f"{pid}{people[pid][:12]}")
    desc = "；".join(desc_parts[:3]) + ("..." if len(ids) > 3 else "")

    return {
        "desc": desc,
        "people": people,
        "groups": groups,
        "assert_gt": assert_gt,
    }


def generate_scenarios(n: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    scenarios = []
    attempts = 0
    while len(scenarios) < n and attempts < n * 10:
        attempts += 1
        s = _build_scenario(rng)
        if s is not None:
            scenarios.append(s)
    return scenarios


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=200, help="Number of scenarios to generate")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out", type=Path,
        default=Path(__file__).parent / "data" / "eval_scenarios.json",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    scenarios = generate_scenarios(args.n, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(scenarios, indent=2, ensure_ascii=False))
    print(f"Generated {len(scenarios)} scenarios → {args.out}")

    # quick stats
    n_people_dist: dict[int, int] = {}
    for s in scenarios:
        n = len(s["people"])
        n_people_dist[n] = n_people_dist.get(n, 0) + 1
    print("People per scene:", dict(sorted(n_people_dist.items())))
    print("Avg assert_gt per scene:", round(sum(len(s["assert_gt"]) for s in scenarios) / len(scenarios), 1))


if __name__ == "__main__":
    main()
