# Social Navigation Layout

This repo already has a clear split between environment code and policy code. For social navigation work,
keep that split explicit instead of putting all research code into one folder.

## Recommended Ownership

`molmo_spaces/`
- Environment backend
- Scene loading and randomization
- Task and task sampler abstractions
- Reusable policy implementations that should plug into the framework

`experiments/social_nav/`
- Experiment entrypoints
- Hyperparameter sweeps
- Ablations
- Training scripts
- Evaluation scripts
- Result analysis notebooks or summaries

## How Navigation Pieces Fit Together

`tasks/nav_task_sampler.py`
- Samples an episode
- Chooses scene, target object, robot spawn
- Prepares occupancy maps and randomized scene state

`tasks/nav_task.py`
- Defines what the task means
- Computes reward, success, and metrics
- Exposes task-relevant observations

`policy/solvers/navigation/`
- Decides how to move
- Holds reusable planners/controllers such as A* or MPPI
- Should stay algorithm-focused, not experiment-focused

## Recommended Social Navigation Extension Path

1. Keep `molmo_spaces/tasks/nav_task_sampler.py` for scene and target generation.
2. Introduce a new social-nav task instead of overloading the current object-nav success logic.
3. Add reusable navigation policies under `molmo_spaces/policy/solvers/navigation/`.
4. Put training and benchmark orchestration under `experiments/social_nav/`.

## Suggested Next Files

If social navigation becomes the main focus, these are the next useful additions:

`molmo_spaces/tasks/social_nav_task.py`
- Goal reaching plus social metrics

`molmo_spaces/configs/base_social_nav_config.py`
- Experiment config that pairs a social-nav task with a navigation policy

`experiments/social_nav/run_mppi.py`
- Debug and benchmark entrypoint for MPPI

`experiments/social_nav/train_policy.py`
- Training entrypoint for learned policies

## Current Entrypoints

Use:

`experiments/social_nav/run_social_nav.py`

or the MPPI convenience wrapper:

`experiments/social_nav/run_mppi.py`

This now uses a proper social-navigation experiment structure:

- `experiments/social_nav/manager.py`
- `experiments/social_nav/context.py`
- `experiments/social_nav/methods/base.py`
- `experiments/social_nav/methods/mppi.py`

The reusable controller implementation still stays under
`molmo_spaces/policy/solvers/navigation/`, but the episode orchestration now lives in
`experiments/social_nav/`.

Example:

```bash
./.venv/bin/mjpython experiments/social_nav/run_social_nav.py \
  --method mppi \
  --scene-xml assets/scenes/ithor/FloorPlan203_physics.xml \
  --layout-json assets/layouts/FloorPlan203_object_positions.json \
  --layout-runtime auto \
  --robot-type navbot \
  --start-pose=-1.4,4.6,0 \
  --goal-xy=-0.6,4.6
```

## What `goal` Means

For the current experiment demo, `goal` is a 2D target point in world coordinates:

- `--goal-xy=x,y`
- Units are meters
- Coordinate frame is the same MuJoCo world frame used by the scene layout JSON

The demo may snap the requested goal to the nearest reachable free cell before planning if the
exact point is occupied or too close to obstacles.

## Where To Adjust It

For the experiment demo:

- CLI flag: `--goal-xy`
- Start pose: `--start-pose=x,y,yaw_deg`
- Success radius: `--goal-radius`

For scene-specific targets:

- Put candidate target coordinates in an experiment config, shell script, or notebook under
  `experiments/social_nav/`
- Keep scene geometry and human placement in layout JSON files under `assets/layouts/`

For the reusable framework policy path:

- `molmo_spaces/policy/solvers/navigation/mppi_policy.py`
- The policy resolves the target from either:
  - `task.get_nearest_nav_object(...)`
  - `task.config.task_config.goal_xy`
  - `task.config.task_config.point_goal_xy`

So if later you run MPPI through the task/policy framework instead of the standalone demo,
the goal should be set in task config rather than passed as `--goal-xy`.

## Method Boundary

The key design rule is:

- `manager.py` owns the episode loop and scene orchestration
- `methods/` owns the interchangeable decision logic

That means MPPI is now just one method implementation. If you later add A*, RL, a social-force
model, or a learned local planner, it should implement the same interface under
`experiments/social_nav/methods/` rather than re-creating another standalone script.
