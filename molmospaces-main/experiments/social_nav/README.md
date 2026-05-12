# Social Navigation Experiments

This folder is now scoped to the two-stage workflow described in
`experiements.md`.

## Stage 1: Static Social Path Validation

Goal: verify whether an LLM or deterministic rule prompt can produce a
reasonable social cost map, then use social-cost-aware A* to choose a route
that avoids socially inappropriate areas.

Primary entrypoints:

```bash
./.venv/bin/python experiments/social_nav/run_stage1_static.py \
  --birdview-image outputs/procthor/train_9/debug_grid_free.png \
  --map-bounds 0,10,0,6 \
  --scene-graph outputs/procthor/train_9/scene_graph.json \
  --start-pose 1.0,1.0,0 \
  --goal-xy 8.0,5.0 \
  --social-method rule \
  --save-topdown outputs/stage1_static.png \
  --out-json outputs/stage1_static.json
```

```bash
./.venv/bin/python experiments/social_nav/stage1_playground.py \
  --birdview-image outputs/procthor/train_9/debug_grid_free.png \
  --map-bounds 0,10,0,6 \
  --scene-graph outputs/procthor/train_9/scene_graph.json
```

Use `--social-method llm --llm-model <model>` only when you explicitly want to
spend an API call. The default interactive workflow uses rule-based social cost
and only refreshes LLM cost when requested.

## Stage 2: Dynamic Social Navigation

Goal: scripted human motion and relationship changes, with A* providing the
global route and MPPI handling local collision avoidance and short-horizon
adjustment.

Primary entrypoint:

```bash
./.venv/bin/python experiments/social_nav/run_stage2_dynamic.py
```

The Stage2 simulator updates human positions and relationships, regenerates
the deterministic social costmap in real time, replans A*, and tracks the route
with MPPI. LLM mode can be toggled in the UI; if the API call is unavailable,
the shared rule-based costmap is used as a fallback.

## Shared Cost Modules

`cost/llm_costmap.py`
- scene graph or live agents -> social parameters
- rule baseline and LLM pipeline
- Gaussian costmap synthesis for A* and visualization

`cost/scene_cost.py`
- hard wall / human-body constraints for MPPI
- goal, clearance, control, and smoothness terms

`cost/social_cost.py`
- local continuous social cost used inside MPPI

## Kept Entrypoints

`run_pipeline.py`
- HumanSSG scene graph -> optional layout -> Stage1 static planning

`run_social_nav.py`
- full MuJoCo A* + MPPI validation path

`run_procthor.py`
- convenience wrapper for ProcTHOR scene selection and full validation

Deprecated duplicate demo/test/config files were removed so the experiment
surface stays small and runnable.
