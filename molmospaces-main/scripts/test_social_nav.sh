#!/usr/bin/env bash
set -euo pipefail

# 用法：
#   scripts/test_social_nav.sh headless
#   scripts/test_social_nav.sh viewer
#
# 可选环境变量：
#   START_POSE="-1.4,4.6,0"
#   GOAL_XY="-0.6,4.6"
#   MAX_STEPS=160

MODE="${1:-headless}"
START_POSE="${START_POSE:--1.4,4.6,0}"
GOAL_XY="${GOAL_XY:--0.6,4.6}"
MAX_STEPS="${MAX_STEPS:-160}"

PYTHON_BIN="./.venv/bin/python"
if [[ "$(uname -s)" == "Darwin" ]]; then
  # macOS 下即使是 headless，这套流程里也会创建 MuJoCo/OpenGL 上下文，
  # 统一走 mjpython 更稳。
  PYTHON_BIN="./.venv/bin/mjpython"
fi

SCENE="${SCENE:-assets/scenes/ithor/FloorPlan203_physics.xml}"
LAYOUT="${LAYOUT:-assets/layouts/FloorPlan203_object_positions.json}"
SOCIAL_METHOD="${SOCIAL_METHOD:-rule}"

COMMON_ARGS=(
  experiments/social_nav/run_social_nav.py
  --scene-xml "$SCENE"
  --layout-json "$LAYOUT"
  --layout-runtime auto
  --robot-type navbot
  --start-pose="$START_POSE"
  --goal-xy="$GOAL_XY"
  --goal-radius 0.25
  --max-steps "$MAX_STEPS"
  --log-every 20
  --social-method "$SOCIAL_METHOD"
  --save-topdown outputs/topdown_latest.png
)

if [[ "$MODE" == "viewer" ]]; then
  MPLCONFIGDIR=/tmp/matplotlib XDG_CACHE_HOME=/tmp MPLBACKEND=Agg "$PYTHON_BIN" "${COMMON_ARGS[@]}"
elif [[ "$MODE" == "headless" ]]; then
  MPLCONFIGDIR=/tmp/matplotlib XDG_CACHE_HOME=/tmp MPLBACKEND=Agg "$PYTHON_BIN" "${COMMON_ARGS[@]}" --no-viewer
else
  echo "Unknown mode: $MODE"
  echo "Use: headless | viewer"
  exit 1
fi
