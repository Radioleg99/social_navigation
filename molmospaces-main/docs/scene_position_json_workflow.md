# 场景位置 JSON 工作流

这个工作流用于三件事：
1. 导出场景内所有对象的当前位置（可视化构建时当“坐标地图”）。
2. 用 JSON 覆盖对象位置，再生成新的场景 XML。
3. 在同一个 JSON 里配置人物 `human_runtime`（位置/朝向/姿态文件）。

脚本：
- `scripts/scene_position_tool.py`

## 1) 导出对象位置 JSON

```bash
./.venv/bin/python scripts/scene_position_tool.py export \
  --scene-xml assets/scenes/ithor/FloorPlan1_physics.xml \
  --out-json assets/layouts/FloorPlan1_object_positions.json \
  --include-all-root-bodies
```

输出文件：
- `assets/layouts/FloorPlan1_object_positions.json`

说明：
- 坐标系是 `mujoco_world`。
- `position_xyz = [x, y, z]`。
- `quat_wxyz = [w, x, y, z]`。
- 导出 JSON 里会带 `robot_runtime` 和 `human_runtime` 模板（默认 `enabled=true`）。

## 2) 编辑 JSON（只改 `overrides`）

文件结构示例：

```json
{
  "scene_xml": "...",
  "coordinate_frame": "mujoco_world",
  "objects": [
    {
      "body": "chair_xxx",
      "position_xyz": [1.0, 0.5, 0.0],
      "quat_wxyz": [1.0, 0.0, 0.0, 0.0]
    }
  ],
  "overrides": {
    "chair_xxx": {
      "position_xyz": [1.2, 0.5, 0.0],
      "quat_wxyz": [1.0, 0.0, 0.0, 0.0]
    }
  }
}
```

建议：
- `objects` 只是参考清单，不要手改。
- 真正要改的位置都放在 `overrides`。

最简人物配置（推荐）只改这几个字段：

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
    "human_pos": [1.2, 0.8, 0.0],
    "human_yaw_deg": 90.0
  }
}
```

说明：
- `poses` 写一个文件：静止人物。
- `poses` 写多个文件：自动按 `pose_interval_sec` 切换。
- `loop=true` 会循环切换动作，`loop=false` 只播放一次。

## 3) 应用覆盖并生成新场景 XML

```bash
./.venv/bin/python scripts/scene_position_tool.py apply \
  --scene-xml assets/scenes/ithor/FloorPlan1_physics.xml \
  --layout-json assets/layouts/FloorPlan1_object_positions.json \
  --out-xml assets/scenes/ithor/FloorPlan1_physics_patched.xml
```

建议把 `--out-xml` 放在原场景同目录（如上），这样原 XML 的相对 mesh/texture 路径不会失效。

然后你可以直接跑（人物完全由 JSON 控制）：

```bash
./.venv/bin/mjpython scripts/basic_robot_human_scene.py \
  --scene-xml assets/scenes/ithor/FloorPlan1_physics_patched.xml \
  --layout-json assets/layouts/FloorPlan1_object_positions.json
```

## 4) 交互式调物体位置（推荐）

用下面脚本在 viewer 里直接选物体并移动，最后一键写回 JSON：

```bash
./.venv/bin/mjpython scripts/scene_object_tuner.py \
  --scene-xml assets/scenes/ithor/FloorPlan1_physics.xml \
  --layout-json assets/layouts/FloorPlan1_object_positions.json
```

说明：
- 如果 `layout-json` 里 `human_runtime.enabled=true`，调参器会自动把人物一起加载进来（不会再“人物消失”）。

快捷键：
- `V`：切换到下一个物体
- `X`：切换到上一个物体
- `]` / `[` 或 方向键右/左：切换物体（备用）
- `W/S/A/D`：平面移动
- 方向键上/下：前后移动（备用）
- `R/F`：上下移动
- `PgUp/PgDn`：上下移动（备用）
- `J/L`：旋转 yaw
- `P`：打印当前物体姿态
- `K`：把当前编辑写入 `layout-json` 的 `overrides`

可视化：
- 切换对象时会自动相机对焦当前对象。
- 当前对象会被高亮显示（偏黄）。

故障排查：
- 必须先点击一下 viewer 窗口，再按键。
- macOS 请用 `mjpython` 启动，而不是普通 `python`。

## 5) 只调“人物”位置与朝向（推荐）

如果你只想切换不同人物并调整人物位置，不想碰场景物体，直接用：

```bash
./.venv/bin/mjpython scripts/human_tuner.py \
  --scene-xml assets/scenes/ithor/FloorPlan1_physics.xml \
  --layout-json assets/layouts/FloorPlan1_object_positions.json
```

快捷键：
- `V`：切换到下一个人物
- `↑/↓/←/→`：移动当前人物
- `P`：写回 `layout-json`（`human_runtime` 和 `extra_humans`）
