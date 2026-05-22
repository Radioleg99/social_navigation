# Social Navigation — 快速运行命令
# 用法: make <target>  [SOCIAL=rule|none|llm]  [GOAL=x,y]
#
# 所有命令都在 molmospaces-main/ 目录下运行

PYTHON       = ./.venv/bin/mjpython
PYTHON_PLAIN = ./.venv/bin/python
ENV          = MPLCONFIGDIR=/tmp/matplotlib XDG_CACHE_HOME=/tmp MPLBACKEND=Agg

SCENE   = assets/scenes/ithor/FloorPlan203_physics.xml
LAYOUT  = assets/layouts/FloorPlan203_object_positions.json
START   = -1.4,4.6,0
GOAL    = -0.6,4.6
SOCIAL  = rule
LLM_MODEL = moonshot-v1-8k
STAGE2_SCENARIO = conversation_crossing
STEPS   = 200

COMMON  = experiments/social_nav/run_pipeline.py \
          --scene-xml $(SCENE) \
          --layout-json $(LAYOUT) \
          --start-pose=$(START) \
          --goal-xy=$(GOAL) \
          --social-method $(SOCIAL) \
          --max-steps $(STEPS) \
          --log-every 10

# --------------------------------------------------------------------------

.PHONY: run run-viewer topdown scenes help

## run: headless 运行社交导航（保存鸟瞰图到 outputs/）
run:
	mkdir -p outputs
	$(ENV) $(PYTHON) $(COMMON) --no-viewer --save-topdown outputs/topdown_$(SOCIAL).png

## run-viewer: 带 MuJoCo 3D 窗口运行
run-viewer:
	mkdir -p outputs
	$(ENV) $(PYTHON) $(COMMON) --save-topdown outputs/topdown_$(SOCIAL).png

## topdown: 仅生成鸟瞰图（不启动 viewer）
topdown:
	mkdir -p outputs
	$(ENV) $(PYTHON) $(COMMON) --no-viewer \
	    --save-topdown outputs/topdown_$(SOCIAL).png \
	    --save-gif     outputs/traj_$(SOCIAL).gif

## scenes: 列出可用场景
scenes:
	$(PYTHON) scripts/scene_finder.py list --source ithor --category living_room

## playground: 交互式 Stage1 playground（彩色 RGB + 二值占用图，train_9）
## 用法: make playground              # 默认 rule-based social cost
##       make playground SOCIAL=llm  # 启用 LLM costmap（需要 API key）
playground:
	mkdir -p outputs
	$(ENV) $(PYTHON_PLAIN) experiments/social_nav/stage1_playground.py \
	    --map-source scene-xml \
	    --background rgb \
	    --layout-json outputs/procthor/train_9/train_9_layout.json \
	    --scene-graph outputs/procthor/train_9/scene_graph.json \
	    --start-pose 2.0,4.0,0.0 \
	    --goal-xy 9.0,4.0 \
	    --birdview-downscale 6 \
	    --astar-human-block-radius 0.65 \
	    --social-method $(SOCIAL)

## playground-stage2: 动态 Stage2 scripted playground（SCENARIO=conversation_crossing|argument_block|tv_watchers|sleeper_and_walker|queue_split）
playground-stage2:
	mkdir -p outputs
	$(ENV) $(PYTHON_PLAIN) experiments/social_nav/stage1_playground.py \
	    --stage2-scripted \
	    --stage2-scenario $(STAGE2_SCENARIO) \
	    --map-source scene-xml \
	    --background rgb \
	    --layout-json outputs/procthor/train_9/train_9_layout.json \
	    --scene-graph outputs/procthor/train_9/scene_graph.json \
	    --start-pose 2.0,4.0,0.0 \
	    --goal-xy 9.0,4.0 \
	    --birdview-downscale 6 \
	    --astar-human-block-radius 0.65 \
	    --stage2-mppi-horizon 32 \
	    --stage2-replan-interval 0.0 \
	    --stage2-costmap-interval 0.5 \
	    --social-method $(SOCIAL) \
	    --llm-model $(LLM_MODEL)

## kill: 杀掉残留的 mjpython 进程（viewer 报错 "already open" 时用）
kill:
	-pkill -f mjpython
	@echo "done"

## help: 显示此帮助
help:
	@grep -E '^## ' Makefile | sed 's/## /  make /'
