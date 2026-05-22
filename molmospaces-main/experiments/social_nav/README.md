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

Stage2 scripted playback inside the same playground:

```bash
make playground-stage2
```

Available scripted timelines:
`conversation_crossing`, `argument_block`, `tv_watchers`, `sleeper_and_walker`,
and `queue_split`. Select one at launch with:

```bash
make playground-stage2 STAGE2_SCENARIO=argument_block
```

Press `N` inside the playground to cycle timelines.

or explicitly:

```bash
./.venv/bin/python experiments/social_nav/stage1_playground.py \
  --stage2-scripted \
  --stage2-scenario conversation_crossing \
  --map-source scene-xml \
  --background rgb \
  --layout-json outputs/procthor/train_9/train_9_layout.json \
  --scene-graph outputs/procthor/train_9/scene_graph.json \
  --start-pose 2.0,4.0,0.0 \
  --goal-xy 9.0,4.0 \
  --birdview-downscale 6 \
  --stage2-mppi-horizon 32 \
  --stage2-replan-interval 0.0 \
  --stage2-costmap-interval 0.5
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

Headless smoke test:

```bash
./.venv/bin/python experiments/social_nav/run_stage2_dynamic.py \
  --headless --steps 160 --social-method rule
```

The scripted Stage2 simulator has a fixed timeline of human motion and
relationship events. Human positions are updated every simulation step and fed
to MPPI as local hard constraints. Relationship changes trigger a social-field
update and A* replan from the robot's current pose. With `--social-method llm`,
the social update runs asynchronously, so the robot keeps following the
previous A* route until the API result is applied. If the API call fails, the
shared rule-based costmap fallback is used.

The older manual pygame simulator is still available with:

```bash
./.venv/bin/python experiments/social_nav/run_stage2_dynamic.py --interactive
```

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
