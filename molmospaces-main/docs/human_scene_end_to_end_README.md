# Human-in-Scene End-to-End (Short Version)

This is the compact workflow for static humans and pose switching.

## Steps

1. Export scene layout JSON
2. Convert FBX human to `character.xml`
3. Configure `human_runtime` in layout JSON
4. Tune human positions in viewer
5. Run scene

## 1) Export Layout JSON

```bash
./.venv/bin/python scripts/scene_position_tool.py export \
  --scene-xml assets/scenes/ithor/FloorPlan1_physics.xml \
  --out-json assets/layouts/FloorPlan1_object_positions.json \
  --include-all-root-bodies
```

## 2) FBX -> XML

```bash
/Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup \
  --python scripts/mixamo_fbx_to_mjcf.py -- \
  --fbx "/absolute/path/to/human.fbx" \
  --out-dir assets/humans/human_name_multimat \
  --target-height 1.72 \
  --non-colliding \
  --multi-material
```

## 3) Edit `human_runtime`

In `assets/layouts/FloorPlan1_object_positions.json`:

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

- Facing direction: `human_yaw_deg`
- Axis correction: `human_roll_deg` / `human_pitch_deg`
- Robot init can be set in `robot_runtime` (same JSON), so CLI robot args are optional.

## 4) Tune Positions

```bash
./.venv/bin/mjpython scripts/human_tuner.py \
  --scene-xml assets/scenes/ithor/FloorPlan1_physics.xml \
  --layout-json assets/layouts/FloorPlan1_object_positions.json \
  --simulate
```

- `V`: next human
- `Arrow keys`: move selected human
- `P`: save JSON

## 5) Run

```bash
./.venv/bin/mjpython scripts/basic_robot_human_scene.py \
  --scene-xml assets/scenes/ithor/FloorPlan1_physics.xml \
  --layout-json assets/layouts/FloorPlan1_object_positions.json
```

## Related

- `docs/scene_position_json_workflow.md` for full object-level editing
- `docs/human_multimaterial_no_bake_README.md` for no-bake material import details
