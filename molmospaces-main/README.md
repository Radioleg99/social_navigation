# MolmoSpaces (Simplified Workflow)

This README is the **single entry point** for your daily workflow.
If your goal is scene building + static humans + pose switching, follow only this page.

## What You Need (5 Steps)

1. Choose a scene (`FloorPlanXXX_physics.xml`)
2. Convert human FBX to `character.xml`
3. Put humans/poses into `layout.json`
4. Tune human position in viewer and save
5. Run the final scene

## 0) Environment

```bash
# from repo root
uv pip install -e .[dev,grasp]
```

On macOS, interactive viewer scripts must use `mjpython`.

## 1) Choose Scene

Example: iTHOR FloorPlan203

```bash
# this path is created automatically after first load
assets/scenes/ithor/FloorPlan203_physics.xml
```

Export layout JSON (object list + human_runtime block):

```bash
./.venv/bin/python scripts/scene_position_tool.py export \
  --scene-xml assets/scenes/ithor/FloorPlan203_physics.xml \
  --out-json assets/layouts/FloorPlan203_object_positions.json \
  --include-all-root-bodies
```

## 2) Convert Human FBX -> XML

```bash
/Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup \
  --python scripts/mixamo_fbx_to_mjcf.py -- \
  --fbx "/absolute/path/to/person.fbx" \
  --out-dir assets/humans/person_name_multimat \
  --target-height 1.72 \
  --non-colliding \
  --multi-material
```

Output you care about:

- `assets/humans/person_name_multimat/character.xml`

## 3) Put Human/Pose into Layout JSON

Edit:

- `assets/layouts/FloorPlan203_object_positions.json`

Minimal `human_runtime` example:

```json
{
  "human_runtime": {
    "enabled": true,
    "poses": [
      "assets/humans/standing_idle_multimat/character.xml",
      "assets/humans/sitting_2_multimat_matchedscale/character.xml"
    ],
    "pose_interval_sec": 2.0,
    "loop": true,
    "human_static": true,
    "human_pos": [1.2, 0.8, 0.0],
    "human_yaw_deg": 270.0,
    "human_roll_deg": 90.0,
    "human_pitch_deg": 0.0,
    "human_z_offset": 0.0,
    "human_collider_type": "capsule",
    "human_collider_size": [0.28, 1.65, 0.0]
  }
}
```

Notes:

- `human_yaw_deg` is facing direction.
- `roll/pitch` are model-axis correction.
- `poses` with multiple XML files means pose switching timeline.
- You can also set robot init pose in the same JSON via `robot_runtime`.

`robot_runtime` example:

```json
{
  "robot_runtime": {
    "enabled": true,
    "robot_type": "rby1",
    "robot_base": [-1.2, 3.8, 0.0],
    "robot_pos": [0.0, -0.15, 0.0]
  }
}
```

- `robot_base` is used by `rby1/rby1m` as `[x,y,theta(rad)]`.
- `robot_pos` is used by `franka`.
- CLI args still have higher priority than JSON (if you pass both).

## 4) Tune Human Position Interactively

```bash
 
```

Hotkeys:

- `V`: switch human
- `Arrow keys`: move selected human (tap/hold)
- `P`: save to JSON

## 5) Run Final Scene

```bash
./.venv/bin/mjpython scripts/basic_robot_human_scene.py \
  --scene-xml assets/scenes/ithor/FloorPlan203_physics.xml \
  --layout-json assets/layouts/FloorPlan203_object_positions.json
```

## 6) Capture RGBD For Training

All capture/export scripts share one default output root:

- `scripts/output_root_config.py` -> `DEFAULT_OUTPUT_ROOT`

Check final write paths:

```bash
./.venv/bin/python scripts/manual_capture_paths.py --scene-id FloorPlan203/manual_frames
```

### 6.1 Auto Export (ConceptGraph format)

```bash
./.venv/bin/mjpython scripts/export_conceptgraph_rgbd.py \
  --scene-xml assets/scenes/ithor/FloorPlan203_physics.xml \
  --layout-json assets/layouts/FloorPlan203_object_positions.json \
  --layout-runtime auto \
  --trajectory-mode orbit \
  --orbit-radius 1.4 \
  --scene-id FloorPlan203/frames_auto_train \
  --n-views 36 \
  --overwrite
```

### 6.2 Robot First-Person Manual Capture

Use the robot already defined in scene + layout runtime (no extra robot added).
Capture uses the robot camera in the scene (`robot_0/head_camera` by default):

```bash
./.venv/bin/mjpython scripts/manual_capture_rgbd.py \
  --scene-xml assets/scenes/ithor/FloorPlan203_physics.xml \
  --layout-json assets/layouts/FloorPlan203_object_positions.json \
  --layout-runtime on \
  --camera-source robot \
  --robot-camera-name robot_0/head_camera \
  --scene-id FloorPlan203/manual_frames \
  --start-pose=-1.4,4.6,180 \
  --overwrite
```

Hotkeys:

- `Up/Down`: move robot base forward/backward (along current base heading)
- `Left/Right`: rotate robot base yaw
- `P`: capture current frame
- `H`: print hotkey help
- `Q`: quit

Notes:

- For negative pose values, use `--start-pose=-1.4,4.6,180` (or quote the value).
- `--start-pose` initializes robot base `x,y,yaw_deg` when base joints exist.
- Viewer side UI panels are off by default to reduce arrow-key conflicts with the MuJoCo menu.
- Add `--show-ui` only when you need those panels.
- Default capture loop is non-simulated (static `mj_forward`) to reduce freeze risk; add `--simulate` if needed.
- On macOS interactive capture/export, use `mjpython` (not plain `python`).

## Minimal Project Map

- `assets/scenes/ithor/`: iTHOR scenes (`FloorPlan*_physics.xml`)
- `assets/layouts/`: scene layout JSON (object overrides + `human_runtime`)
- `assets/humans/`: converted human assets (`character.xml`, meshes, textures)
- `scripts/mixamo_fbx_to_mjcf.py`: FBX -> human XML
- `scripts/human_tuner.py`: move/save human positions
- `scripts/basic_robot_human_scene.py`: run scene
- `scripts/export_conceptgraph_rgbd.py`: auto RGBD export
- `scripts/manual_capture_rgbd.py`: robot first-person manual RGBD capture
- `scripts/manual_capture_paths.py`: print resolved output paths
- `scripts/output_root_config.py`: shared default output root

## Optional Docs (Only If Needed)

- `docs/human_scene_end_to_end_README.md`
- `docs/scene_position_json_workflow.md`
- `docs/human_multimaterial_no_bake_README.md`
