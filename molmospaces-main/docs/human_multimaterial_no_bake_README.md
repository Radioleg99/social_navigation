# 静态/动态人物导入（保留原材质，不烘焙）说明

## 1. 目标
把真实人物模型放进 MolmoSpaces 场景里，尽量保持原始贴图效果，同时避免 Blender `Join`/`Bake` 带来的丢贴图问题。

本流程适合：
1. 只需要“看起来像真人”的可视化人物。
2. 不需要骨骼驱动和精细关节控制。
3. 想快速把 FBX/GLB 来源的人物放进 MuJoCo 场景。


## 2. 核心思路（为什么不烘焙）
传统做法是把多材质人物先 `Join` 再 Bake 成单贴图，但这一步在很多 FBX 上会出问题（材质错位、局部缺失、黑块）。

现在改成：
1. 按材质拆成多个 mesh part（例如 head/body/eyes/shoes）。
2. 每个 part 绑定各自的 `Base Color` 贴图。
3. 在一个 `character.xml` 里生成多个 `<mesh>/<texture>/<material>/<geom>`。

这样做的作用：
1. 避免 `Join` 导致的贴图丢失。
2. 不需要手动烘焙 atlas。
3. 结果可直接放到 `basic_robot_human_scene.py`。


## 3. 代码入口（已实现）
主要脚本：
1. `scripts/mixamo_fbx_to_mjcf.py`
2. `scripts/basic_robot_human_scene.py`

关键参数：
1. `--multi-material`：按材质自动拆分并保留原贴图（推荐）。
2. `--no-join`：不做 `Join`，导出多对象到一个 OBJ（备选）。
3. `--human-static`：人物静止。
4. `--human-z-offset`：人物上下微调（当前建议从 `0.0` 开始）。

关键实现位置：
1. 按材质拆分与贴图提取：`export_mesh_parts_by_material(...)`
2. 多部件归一化：`normalize_obj_meshes(...)`
3. 生成多材质 MJCF：`write_mjcf_multi_material(...)`


## 4. 一条命令完成转换（无需 Bake）
使用 Blender 后台模式执行转换：

```bash
/Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup \
  --python scripts/mixamo_fbx_to_mjcf.py -- \
  --fbx "/absolute/path/to/model.fbx" \
  --out-dir assets/humans/my_human_multimat \
  --target-height 1.72 \
  --non-colliding \
  --multi-material
```

输出目录一般包含：
1. `character.xml`
2. `part_*.obj`
3. `tex_*.png`


## 5. 场景中加载人物
静止人物示例：

```bash
./.venv/bin/mjpython scripts/basic_robot_human_scene.py \
  --scene-source ithor --scene-split train --scene-index 1 \
  --human-static \
  --human-xml assets/humans/my_human_multimat/character.xml \
  --human-pos 1.2,0.8,0.0 \
  --human-yaw-deg 90 \
  --human-roll-deg 90 \
  --human-z-offset 0.0
```

如果人物漂浮/下陷，只改一个参数：
1. 漂浮：`--human-z-offset -0.02` 到 `-0.06`
2. 下陷：`--human-z-offset +0.01` 到 `+0.03`


## 6. 如何“保留原材质”
这里的“保留”是指保留每个材质的 `Base Color` 贴图（颜色图），而不是完整 PBR 全通道。

脚本会在每个 material 节点里尝试：
1. 优先读取 `Principled BSDF -> Base Color` 连接的 `Image Texture`。
2. 如果找不到，再回退到该材质第一个可用 `Image Texture`。

然后把图片另存为 PNG，并在 MJCF 中建立：
1. `<texture ... file="tex_xxx.png" />`
2. `<material ... texture="..."/>`
3. `<geom ... material="..."/>`


## 7. 动态展示建议（不做骨骼动画）
如果你要“同一个人从站立到坐下”的动态效果，推荐做状态切换而不是骨骼重定向：
1. 准备 `stand/sit/(可选中间帧)` 多个静态姿态文件。
2. 每个姿态都转换成一个 `character.xml`。
3. 在同场景按时间切换可见性（alpha），实现过程动画。

优点：稳定、开发快、对模型要求低。


## 8. 已知限制
1. 主要保留颜色贴图；法线/粗糙度/金属度等不保证完全还原。
2. 某些部件如果源材质没有可读图片，会出现“无纹理几何”。
3. 高精度角色动画（走路摆臂）仍需要骨骼与动作管线，不在本流程范围内。


## 9. 常见报错与处理
1. `No mesh objects found after importing FBX`：FBX 导入失败或只含骨架，先在 Blender 手动确认有 Mesh。
2. `Texture file not found`：传入了不存在的 `--texture-file` 路径（单贴图模式）。
3. 人物漂浮：先把 `--human-z-offset` 设为 `0.0`，再小步微调。
4. macOS 交互查看报 `mjpython` 提示：用 `./.venv/bin/mjpython` 启动。
