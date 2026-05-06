# Social Navigation with 3D Scene Graphs

Robot navigation that respects social norms in multi-human indoor environments.

## 目标

让机器人能在多人室内场景中，找到最符合社会规范的路径并执行导航。

## 架构

**Stage 1（静态场景）：** HumanSSG 从机器人传感数据建立 3D 场景图，
pipeline 将其转换为社交代价图，引导 MPPI 控制器在 MoLMoSpaces 中导航。

**Stage 2（动态场景，TODO）：** LLM 实时感知场景变化，动态更新代价图并重新规划。

## 目录结构

```
graduation/
├── pipeline/                    # 感知 → 仿真 的桥接层
│   ├── scene_bridge.py          # HumanSSG scene_graph.json → SceneDescription
│   └── scene_builder.py         # SceneDescription → MoLMoSpaces layout JSON
│
├── molmospaces-main/            # MuJoCo 仿真（仅保留导航相关代码）
│   ├── molmo_spaces/
│   │   ├── tasks/social_nav_task.py   # 社交导航任务 + 评估指标
│   │   └── policy/solvers/navigation/ # MPPI 控制器
│   ├── experiments/social_nav/
│   │   ├── run_pipeline.py      # ← 端到端入口（从这里开始）
│   │   ├── run_social_nav.py    # 单 episode 运行
│   │   ├── llm_costmap.py       # 社交代价图生成（规则 / LLM）
│   │   └── mppi_nav.py          # MPPI 导航封装
│   └── assets/
│       ├── scenes/              # MuJoCo 场景 XML（iTHOR 房间）
│       ├── layouts/             # 布局 JSON（人物位置、机器人起点）
│       └── humans/              # 人物 MJCF 模型
│
└── 3dsg/HumanSSG/               # Submodule: HumanSSG 场景图构建
    └── conceptgraph/slam/humanssg/
        ├── human_mapper.py      # 人物检测 + 活动识别
        └── extract_json.py      # 输出 scene_graph.json
```

## 快速开始

### 用现有 layout JSON 运行
```bash
cd molmospaces-main
./.venv/bin/mjpython experiments/social_nav/run_pipeline.py \
    --layout-json  assets/layouts/FloorPlan203_object_positions.json \
    --scene-xml    assets/scenes/ithor/FloorPlan203_physics.xml \
    --start-pose   -1.4,4.6,0 \
    --goal-xy      -0.6,4.6 \
    --social-method rule
```

### 从 3DSG 输出直接运行
```bash
cd molmospaces-main
./.venv/bin/mjpython experiments/social_nav/run_pipeline.py \
    --scene-graph  path/to/scene_graph.json \
    --scene-xml    assets/scenes/ithor/FloorPlan203_physics.xml \
    --start-pose   -1.4,4.6,0 \
    --goal-xy      -0.6,4.6 \
    --social-method llm \
    --llm-model    claude-sonnet-4-6
```

### 社交代价图方法
| `--social-method` | 说明 |
|---|---|
| `none` | 不使用社交感知（baseline） |
| `rule` | 解析式 proxemics（快速，不需要 API） |
| `llm` | LLM 语义推理（需要 API key） |

## 评估指标（`SocialNavTask`）
- `goal_reached`: 是否到达目标
- `social_intrusions`: 进入人物个人空间的次数
- `min_human_distance`: 与最近人物的最小距离
- `path_length`: 总行驶距离（米）

