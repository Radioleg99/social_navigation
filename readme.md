# Social Navigation with 3D Scene Graphs

让机器人在多人室内场景中，找到最符合社会规范的路径并执行导航。

---

## 项目架构

```
graduation/
├── pipeline/                        # 感知 → 仿真 桥接层
│   ├── scene_bridge.py              # HumanSSG scene_graph.json → SceneDescription
│   └── scene_builder.py            # SceneDescription → MoLMoSpaces layout JSON
│
├── molmospaces-main/                # MuJoCo 物理仿真 + MPPI 导航
│   ├── experiments/social_nav/
│   │   ├── run_pipeline.py          # ← 端到端入口（从这里开始）
│   │   ├── run_social_nav.py        # 单 episode 运行器
│   │   ├── llm_costmap.py           # 社交代价图生成（规则 / LLM）
│   │   ├── mppi_nav.py              # MPPI 导航封装
│   │   └── topdown_viz.py           # 鸟瞰图可视化
│   ├── molmo_spaces/
│   │   ├── tasks/social_nav_task.py # 评估指标（侵入次数 / 路径长度等）
│   │   └── policy/solvers/navigation/mppi_core.py
│   ├── assets/
│   │   ├── scenes/ithor/            # 已解压的 MuJoCo 场景（FloorPlan1、203）
│   │   ├── layouts/                 # 布局 JSON（人物位置 + 机器人起点）
│   │   └── humans/                  # 人物 MJCF 模型（standing / sitting / talking）
│   ├── scripts/
│   │   ├── scene_finder.py          # 查找 / 下载更多场景
│   │   └── test_social_nav.sh       # 快速测试脚本
│   ├── Makefile                     # 一键运行命令
│   └── .venv/                       # Python 虚拟环境（mjpython）
│
└── 3dsg/HumanSSG/                   # 场景图构建（git submodule）
    └── conceptgraph/slam/humanssg/
        ├── human_mapper.py          # 人物检测 + 活动识别
        └── extract_json.py          # 输出 scene_graph.json
```

---

## 虚拟环境说明

| 路径 | 用途 |
|---|---|
| `molmospaces-main/.venv` | MuJoCo + MPPI 导航仿真（**运行导航用这个**） |
| `graduation/.venv` | open3d / clip / SAM 感知库（运行 HumanSSG 建图用） |

运行导航必须用 `mjpython`（不是普通 `python`），这样 MuJoCo 才能正确创建渲染上下文：

```bash
cd molmospaces-main
./.venv/bin/mjpython ...   # 正确
./.venv/bin/python ...     # 错误，headless 下可能崩溃
```

---

## 实验一：预设场景快速运行

> 用 MoLMoSpaces 内置的 iTHOR / ProcTHOR 场景，直接验证社交导航方法。
> **不需要 HumanSSG，不需要真实录像。**

所有命令都在 `molmospaces-main/` 目录下运行。

### 方式一：`make`（最简单）

```bash
cd molmospaces-main

make run              # headless 运行，rule 方法，保存鸟瞰图到 outputs/
make run SOCIAL=none  # baseline：纯避障，无社交感知
make run-viewer       # 带 MuJoCo 3D 窗口（可以看到机器人在场景里移动）
make topdown          # 生成鸟瞰图 + 轨迹 GIF
make scenes           # 列出可下载的场景
```

### 方式二：直接调用脚本

```bash
# 必须先 cd 到 molmospaces-main/，不能在 graduation/ 根目录运行
cd molmospaces-main

# 带 MuJoCo 3D 窗口
./.venv/bin/mjpython experiments/social_nav/run_pipeline.py \
    --scene-xml    assets/scenes/ithor/FloorPlan203_physics.xml \
    --layout-json  assets/layouts/FloorPlan203_object_positions.json \
    --start-pose=-1.4,4.6,0 \
    --goal-xy=-0.6,4.6 \
    --social-method rule \
    --save-topdown outputs/topdown.png

# 无界面运行（加 --no-viewer，适合只看日志 / 服务器）
./.venv/bin/mjpython experiments/social_nav/run_pipeline.py \
    --scene-xml    assets/scenes/ithor/FloorPlan203_physics.xml \
    --layout-json  assets/layouts/FloorPlan203_object_positions.json \
    --start-pose=-1.4,4.6,0 \
    --goal-xy=-0.6,4.6 \
    --social-method rule \
    --no-viewer \
    --save-topdown outputs/topdown.png
```

---

## 实验二：MoLMoSpaces 场景内拍摄 → ConceptGraph 重建

> 在 MuJoCo 仿真场景中采集 RGB-D 序列，直接输入 ConceptGraph/HumanSSG 做场景图重建，不需要真实摄像头。

### 方式 A：自动轨迹拍摄（批量，headless）

相机自动绕场景中心拍一圈：

```bash
cd molmospaces-main

./.venv/bin/mjpython scripts/export_conceptgraph_rgbd.py \
    --scene-xml    assets/scenes/ithor/FloorPlan203_physics.xml \
    --layout-json  assets/layouts/FloorPlan203_object_positions.json \
    --layout-runtime auto \
    --trajectory-mode orbit \
    --orbit-radius 1.4 \
    --n-views 36 \
    --output-root  data_capture/FloorPlan203 \
    --scene-id     FloorPlan203/frames
```

常用参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--trajectory-mode` | `orbit` | 轨迹模式：`orbit`（绕圈）|
| `--orbit-radius` | `1.4` | 绕圈半径（米） |
| `--n-views` | `24` | 拍摄帧数 |
| `--camera-height` | `1.45` | 相机高度（米） |
| `--image-width/height` | `640×480` | 图像分辨率 |
| `--vfov-deg` | `90` | 垂直视角（度） |

### 方式 B：手动第一人称拍摄（Viewer 内交互）

打开 MuJoCo 窗口，用键盘在场景里走动，按 **P** 手动拍一帧：

```bash
cd molmospaces-main

./.venv/bin/mjpython scripts/manual_capture_rgbd.py \
    --scene-xml   assets/scenes/ithor/FloorPlan203_physics.xml \
    --layout-json assets/layouts/FloorPlan203_object_positions.json \
    --output-root data_capture/FloorPlan203_manual \
    --scene-id    FloorPlan203/frames
```

Viewer 内快捷键：

| 键 | 功能 |
|----|------|
| `↑ / ↓` | 前进 / 后退 |
| `← / →` | 左转 / 右转 |
| `P` | 拍一帧（color + depth + pose） |
| `H` | 打印帮助 |
| `Q` | 退出 |

### 拍摄输出格式（ConceptGraph 直接可用）

```
data_capture/FloorPlan203/FloorPlan203/frames/
├── color/          000000.png, 000001.png, ...  (RGB)
├── depth/          000000.png, ...              (uint16, 单位 mm)
├── pose/           000000.txt, ...              (4×4 cam2world 矩阵)
├── intrinsics.txt                               (fx fy cx cy width height)
├── dataconfig.yaml                              (ConceptGraph 配置文件)
└── capture_meta.json                            (采集参数记录)
```

### 步骤 2：ConceptGraph 对象重建

切换到 HumanSSG 的 conda 环境，用 `rerun_realtime_mapping.py` 做对象级三维重建：

```bash
# ⚠️ 必须 cd 到 conceptgraph/ 目录下运行，不能在 HumanSSG/ 下运行
cd /Users/ljj/project/graduation/3dsg/HumanSSG/conceptgraph
conda activate conceptgraph

# DATA_ROOT = --output-root 本身（注意：导出脚本在里面又建了一层同名目录）
# 例：--output-root data_capture/FloorPlan203  →  DATA_ROOT 指向 data_capture/FloorPlan203
DATA_ROOT=/Users/ljj/project/graduation/molmospaces-main/data_capture/FloorPlan203
SCENE_ID=FloorPlan203/frames
# 直接用采集时自动生成的 dataconfig.yaml（包含正确的相机参数和 dataset_name: ai2thor）
DATASET_CONFIG="${DATA_ROOT}/${SCENE_ID}/dataconfig.yaml"

python slam/rerun_realtime_mapping.py \
    dataset_root="${DATA_ROOT}" \
    dataset_config="${DATASET_CONFIG}" \
    scene_id="${SCENE_ID}" \
    start=0 end=-1 stride=5 \
    make_edges=false \
    make_human_behaviors=false \
    exp_suffix="map_floorplan203_s1"
```

> **注意**：`DATASET_CONFIG` 指向数据目录里的 `dataconfig.yaml`（采集脚本自动生成），不要用 `dataset/dataconfigs/custom/custom.yaml`（那个是 Replica 格式，和我们的 ai2thor 格式不兼容）。

重建产物保存在：
```
DATA_ROOT/SCENE_ID/exps/map_floorplan203_s1/
├── pcd_map_floorplan203_s1.pkl.gz     # 对象点云 + 元数据
└── obj_json_map_floorplan203_s1.json  # 对象列表 JSON
```
例：`data_capture/FloorPlan203/FloorPlan203/frames/exps/map_floorplan203_s1/`

### 步骤 3：HumanSSG 人物建图

在同一数据上跑 HumanSSG，提取人物节点和社交关系边：

```bash
# 先不开行为推断，确保稳定
export DATASET_ROOT="${DATA_ROOT}"
export SCENE_ID="${SCENE_ID}"
export START=0
export END=-1
export STRIDE=1
export EXP_SUFFIX="human_floorplan203_stage1"

bash scripts/run_human_mapper_ark.sh
```

产物：
```
DATA_ROOT/SCENE_ID/exps/human_floorplan203_stage1/
├── prebuilt_human_floorplan203_stage1.pkl.gz
├── obj_json_human_floorplan203_stage1.json
└── edge_json_human_floorplan203_stage1.json   ← 人物关系边
```

### 步骤 4：提取 scene_graph.json

```bash
python scripts/extract_json.py \
    --result-path "${DATA_ROOT}/${SCENE_ID}/exps/human_floorplan203_stage1/prebuilt_human_floorplan203_stage1.pkl.gz" \
    --edge-file   "${DATA_ROOT}/${SCENE_ID}/exps/human_floorplan203_stage1/edge_json_human_floorplan203_stage1.json" \
    --output      /tmp/scene_graph.json
```

得到 `scene_graph.json` 后，按**实验三**的步骤接入 MPPI 导航。

### 可视化重建结果（可选）

```bash
python scripts/visualize_cfslam_results.py \
    --result_path "${DATA_ROOT}/${SCENE_ID}/exps/map_floorplan203_s1/pcd_map_floorplan203_s1.pkl.gz"
```

---

## 实验三：从真实录像用 3DSG 跑完整 Pipeline

> 用 HumanSSG 扫描真实场景 → 提取人物关系图 → 映射到 MoLMoSpaces 预设场景仿真。
> **需要先采集 RGB-D 视频数据（或使用现有录像）。**

### 完整流程图

```
真实录像（RGB-D + 相机位姿）
        ↓  [步骤 1] HumanSSG 建图
  scene_graph.json
  （人物位置 + 关系 + 活动状态）
        ↓  [步骤 2] run_pipeline.py --scene-graph
  自动生成 layout JSON
        ↓  [步骤 3] MPPI 社交导航
  仿真结果（轨迹 + 鸟瞰图）
```

---

### 步骤 1：运行 HumanSSG 获取 scene_graph.json

HumanSSG 有独立的虚拟环境（`graduation/.venv` 或 conda `conceptgraph`），需要 CUDA GPU。

```bash
cd 3dsg/HumanSSG

# 设置数据路径（修改为你的实际数据目录）
export DATASET_ROOT=/path/to/your/rgbd_data
export SCENE_ID=your_scene_name          # 例如 "office_recording"
export START=0
export END=500                           # 使用的帧范围
export STRIDE=5                          # 每 5 帧采样一次

# 运行建图（需要 CUDA，耗时约 10-30 分钟）
bash conceptgraph/scripts/run_human_mapper_ark.sh
```

数据目录结构要求：
```
DATASET_ROOT/
└── humandata/ham_experiments/
    └── your_scene_name/
        ├── color/          # RGB 图（0000.jpg, 0001.jpg, ...）
        ├── depth/          # 深度图（0000.png, ...，单位 mm）
        └── pose/           # 相机位姿（0000.txt，4×4 c2w 矩阵）
```

运行完成后，在实验输出目录（`outputs/exp_r_mapping_.../` 下）提取 JSON：

```bash
python conceptgraph/slam/humanssg/extract_json.py
# 生成 scene_graph.json，包含：
#   nodes: 每个人/物体的 bbox_center、label、activities
#   edges: 人与人/人与物体之间的关系（"conversing with"、"looking towards" 等）
```

输出文件格式（`scene_graph.json`）：
```json
{
  "nodes": [
    {
      "node_id": 0,
      "label": ["person"],
      "bbox_center": [1.2, 0.5, 0.0],
      "bbox_extent": [0.5, 0.5, 1.7],
      "activities": [{"name": "standing", "object": "sofa"}],
      "heading_deg": 90.0
    }
  ],
  "edges": [
    [0, 1, {"desc": "conversing with", "relation": "conversing with"}]
  ]
}
```

---

### 步骤 2：在预设场景中运行社交导航

拿到 `scene_graph.json` 后，用 `run_pipeline.py --scene-graph` 一键跑完剩余流程（需要切换到 MoLMoSpaces 虚拟环境）：

```bash
cd molmospaces-main

# rule 方法（不需要 API，适合快速验证）
./.venv/bin/mjpython experiments/social_nav/run_pipeline.py \
    --scene-xml   assets/scenes/ithor/FloorPlan203_physics.xml \
    --scene-graph /path/to/scene_graph.json \
    --start-pose=-1.4,4.6,0 \
    --goal-xy=-0.6,4.6 \
    --social-method rule \
    --no-viewer \
    --save-topdown outputs/topdown_3dsg.png

# LLM 方法（需要 ANTHROPIC_API_KEY）
export ANTHROPIC_API_KEY=sk-ant-...
./.venv/bin/mjpython experiments/social_nav/run_pipeline.py \
    --scene-xml   assets/scenes/ithor/FloorPlan203_physics.xml \
    --scene-graph /path/to/scene_graph.json \
    --start-pose=-1.4,4.6,0 \
    --goal-xy=-0.6,4.6 \
    --social-method llm \
    --llm-model   claude-sonnet-4-6 \
    --no-viewer \
    --save-topdown outputs/topdown_3dsg_llm.png
```

`run_pipeline.py` 内部会自动完成：
1. 读取 `scene_graph.json` → `pipeline/scene_bridge.py` 转为 `SceneDescription`
2. `pipeline/scene_builder.py` 生成临时 layout JSON（人物位置对齐到场景中的可通行区域）
3. 调用 `run_episode()` 运行 MPPI 导航

如需保存自动生成的 layout JSON 供后续复用：

```bash
./.venv/bin/mjpython experiments/social_nav/run_pipeline.py \
    --scene-xml   assets/scenes/ithor/FloorPlan203_physics.xml \
    --scene-graph /path/to/scene_graph.json \
    --layout-out  assets/layouts/my_scene_layout.json \
    ...
```

---

## 社交代价图方法

| `--social-method` | 说明 | 需要 |
|---|---|---|
| `none` | 纯避障 baseline，无社交感知 | — |
| `rule` | Proxemics 解析规则（人物个人空间椭圆代价） | — |
| `llm` | LLM 语义推理，理解对话群组 / 活动状态 | API key |

日常调试推荐用 `rule`，不需要联网或 API。

---

## 完整参数说明

### 场景参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--scene-xml` | 必填 | MuJoCo 场景 XML 路径 |
| `--layout-json` | — | 布局 JSON（与 `--scene-graph` 二选一） |
| `--scene-graph` | — | HumanSSG 输出 JSON（与 `--layout-json` 二选一） |
| `--start-pose` | `-1.4,4.6,0` | 起始位姿 `x,y,yaw_deg` |
| `--goal-xy` | 必填 | 目标位置 `x,y`（米，MuJoCo 世界坐标） |
| `--goal-radius` | `0.20` | 到达半径（米），进入此范围视为成功 |
| `--max-steps` | `500` | 最大步数，超过视为失败 |

### MPPI 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--horizon` | `24` | 预测步数 |
| `--num-samples` | `512` | 采样轨迹数（越大越稳，越慢） |
| `--dt` | `0.10` | 控制积分步长（秒） |
| `--v-max` | `0.45` | 最大线速度（m/s） |
| `--w-max-deg` | `120` | 最大角速度（deg/s） |

### 可视化参数

| 参数 | 说明 |
|---|---|
| `--no-viewer` | 无界面运行（headless，服务器上用） |
| `--save-topdown PATH` | 保存鸟瞰图 PNG |
| `--save-gif PATH` | 保存轨迹动画 GIF |
| `--log-every N` | 每 N 步打印一次进度，默认 10 |

---

## 输出结果

终端输出：

```
✓ 到达目标  步数=30  剩余距离=0.198m
{"reached_goal": true, "steps": 30, "final_dist": 0.198}
```

鸟瞰图（`outputs/topdown_rule.png`）包含：
- 占用地图背景（白=可通行，灰=障碍）
- 社交代价热力图叠加（橙红色=高代价区域）
- 人物标记（颜色按活动状态）+ 0.8m 个人空间圆圈
- 机器人轨迹（蓝线）、起点（绿三角）、终点（红星）

---

## 场景管理

当前已解压：`FloorPlan1`（厨房 4.4×4.6m）、`FloorPlan203`（客厅 7.6×7.8m）。

```bash
cd molmospaces-main

# 查看所有可用客厅场景（按大小排序）
python scripts/scene_finder.py list --source ithor --category living_room

# 查看已解压场景的实际尺寸
python scripts/scene_finder.py analyze

# 下载更大的场景（需要联网）
python scripts/scene_finder.py download FloorPlan229   # 最大客厅 ~8.8MB
python scripts/scene_finder.py download FloorPlan224   # 第二大客厅 ~7.9MB
```

下载后用新场景运行：

```bash
make run SCENE=assets/scenes/ithor/FloorPlan229_physics.xml \
         LAYOUT=assets/layouts/FloorPlan229_object_positions.json
```

---

## 人物布置

编辑 `assets/layouts/FloorPlan203_object_positions.json` 中的 `human_runtime` 字段：

```json
{
  "human_runtime": {
    "human_xml": "assets/humans/standing_idle_multimat/character.xml",
    "human_pos": [-0.75, 3.05, 0.0],
    "human_yaw_deg": 90.0,
    "extra_humans": [
      {
        "human_xml": "assets/humans/remy_talking_multimat/character.xml",
        "human_pos": [-4.05, 2.20, 0.0],
        "human_yaw_deg": 0.0
      }
    ]
  }
}
```

可用人物模型：

| 文件夹 | 姿态 |
|---|---|
| `standing_idle_multimat` | 站立静止 |
| `sitting_2_multimat_matchedscale` | 坐姿 |
| `remy_talking_multimat` | 站立交谈（男） |
| `rony_talking_multimat` | 站立交谈（女） |

---

## 评估指标

`molmo_spaces/tasks/social_nav_task.py` 记录以下指标：

| 指标 | 含义 |
|---|---|
| `goal_reached` | 是否到达目标 |
| `steps` | 总步数 |
| `final_dist` | 终止时距目标距离（米） |
| `social_intrusions` | 进入人物 0.8m 个人空间的次数 |
| `min_human_distance` | 与任意人物的最小距离（米） |
| `path_length` | 总行驶路程（米） |

---

## 常见问题

**Q：`mjpython: command not found`**
```bash
# 必须用完整路径，不能 activate venv 后直接 mjpython
./.venv/bin/mjpython ...
```

**Q：场景加载失败 / XML 路径错误**
```bash
# 确保在 molmospaces-main/ 目录下运行，不要在 graduation/ 根目录
cd /path/to/graduation/molmospaces-main
```

**Q：机器人不动 / 卡住**
- 检查 `--start-pose` 和 `--goal-xy` 是否在场景范围内
- FloorPlan203 的可通行范围约为 x ∈ [-6.5, 1.0]，y ∈ [-0.8, 6.2]
- 用 `python scripts/scene_finder.py analyze` 查看已解压场景的具体尺寸

**Q：想用 LLM 方法**
```bash
export ANTHROPIC_API_KEY=sk-ant-...   # Anthropic
# 或
export OPENAI_API_KEY=sk-...          # OpenAI

make run SOCIAL=llm
```
