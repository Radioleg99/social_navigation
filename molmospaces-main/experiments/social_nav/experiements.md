整个实验的代码改动范围：在 experiments/social_nav/cost 里


整个实验流程分为两部分：
    1. stage1: 用于测试就是llm是否能够合理的生成符合social norm的路径
    2. stage2: llm是否能够在动态场景中实时的指导机器人行走符合social norm的路径

任物： 
    stage1：
    1. 在静态场景下，给定json格式的scene graph, 通过prompted llm 生成social cost map。
    2. 给定机器人的起点和终点以及costmap， 通过改进后的a* 算出当前最优最符合社交规范的道路
    3. 最好能够做出可交互的图形界面，用于调试llm的prompt, 以直观的看出social cost map 的生成以及改进后的a* 的道路选择。

    stage2：
    1. scripted 人物的动态以及关系的变化，手动设置动态场景的演变
    2. stage2 的核心在于蒸馏大llm的结果到小llm上，从而提高整体的分数生成的速度
    3. 如果能够结合可交互的图形页面实现，最好，因为其方便调试

边界：
    stage1:
    1. stage1 的静态场景本质上是来自于procthor和ithor 中的地图， 我们只需要navigability map (这说明什么？：1. 可以直接用reachable positions 生成 2D occupancy / navigability map， 网站：https://procthor.allenai.org/ 2. 可以直接绕过Molmospace的复杂的依赖，直接新建一套端到端的，就是从用户体验上来就是选择了prothor的某个地图后直接生成2d 的navigabibilty map 直接可以用于导航）
    2. 生成costmap的llm需要具备连接前后输入输出的能力，输入是地图信息+人的信息，输出是社交分数。
    3. stage1 中只需要 改进后的a* 选择出一条或多条最符合社交规范的路（改进后的a* 指受到social cost影响的a*)，只需要用a* 方法。
    4. stage1 的目的是来验证llm是否能够生成符合social norm的合理路径。

    stage2:
    1. stage2 需要结合a* 与 mppi
    2. stage2 的costmap是会根据人人关系人物关系进行变化的
    3. stage2 的mppi 主要负责避障功能，就是就是小范围的路径改变
    4. stage2 的a* 还是主要负责大的导航路径的生成
    5. 3,4中的mppi和a*的具体的还需要进行测试。





# 1. Stage 1 检验需求

Stage 1 的目标不是做完整的人类社交导航评估，而是先验证一个最小闭环：

> LLM 是否能够根据静态场景生成合理的 social cost rules，并通过 A* 影响路径选择，使机器人避开明显不符合 social norm 的区域。

因此 Stage 1 只保留少量关键指标，避免过早引入复杂评估。

---

## 1.1 Stage 1 检验目标

Stage 1 需要验证以下问题：

1. LLM 是否能稳定输出合法的 social cost rules；
2. social cost rules 是否可以成功转换成 2D social cost map；
3. social-cost-aware A* 是否能成功生成路径；
4. 与普通 shortest-path A* 相比，social-cost-aware A* 是否能避开明显不合适的区域；
5. 生成的路径是否仍然具有合理长度，而不是为了避开 social cost 走出过度绕远的路线。

---

## 1.2 对比方法 / Baselines

Stage 1 只保留两个必要方法：

| 方法 | 说明 | 目的 |
|---|---|---|
| Shortest-path A* | 只使用 navigability map，不使用 social cost | 作为基础对照 |
| LLM social cost + A* | 使用 LLM 生成的 social cost map 进行 A* 规划 | 本项目主方法 |

---

## 1.3 核心 Metrics

待定





## Stage 2 Visualization MVP 
整个实验需要一个轻量级 2D top-down view，用于调试 static social cost map 和 A* 路径。界面不需要 3D 渲染和动态动画。 需要尽可能的快！！ 在未调用llm的情况下应该尽最大可能的不卡顿

### 可修改参数

- robot start position
- robot goal position
- robot radius / minimum clearance
- human number
- human position
- human orientation
- human activity
- human-human relationship
- human minimum contact range
- social cost generation prompt

### 需要显示内容

- 2D navigability map
- humans with position, orientation, activity label
- human minimum contact range
- robot start / goal
- shortest-path A*
- social-cost-aware A*
- social cost map heatmap

### 计算逻辑

- 默认使用 predefined rule-based social cost map。
- 点击 `Generate LLM Social Cost` 后，才调用 LLM 更新 social cost map。
- 修改 start / goal / robot radius 时，只重新计算 A*，不调用 LLM。
- 修改 human position / relationship / activity 时，更新 predefined social cost map，并重新计算 A*，但不自动调用 LLM。
- 修改 prompt 时，只保存 prompt，不自动调用 LLM。

### Hard Constraint vs Soft Cost

Hard constraint 视为墙，影响所有 A*：

- wall / furniture obstacle
- human minimum contact range
- robot radius clearance

Soft social cost 只影响 social-cost-aware A*：

- predefined social cost
- LLM-generated social cost
- human-human relationship cost
- human orientation / activity cost

