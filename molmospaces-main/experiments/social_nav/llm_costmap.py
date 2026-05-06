"""
LLM Social Costmap Generator
-----------------------------
从 layout JSON 提取场景信息，构造 prompt，调用 LLM，
返回一个与 MPPI grid_free 同维度的 social costmap (numpy float32 array)。

用法：
    from experiments.social_nav.llm_costmap import build_social_costmap
    costmap = build_social_costmap(
        layout_json_path="assets/layouts/FloorPlan203_object_positions.json",
        scene_map=scene_map,
        grid_shape=grid_free.shape,
        llm_model="claude-opus-4-6",   # 或 "gpt-4o"
    )
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.ndimage import zoom

# ---------------------------------------------------------------------------
# 场景信息提取
# ---------------------------------------------------------------------------

OBSTACLE_CATEGORIES = {
    "Chair", "Sofa", "ArmChair", "Ottoman",
    "DiningTable", "CoffeeTable", "SideTable",
    "GarbageCan", "HousePlant", "FloorLamp",
}


class HumanInfo(NamedTuple):
    pos: tuple[float, float]   # (x, y) 世界坐标（米）
    yaw_deg: float             # 朝向（度），0=+x, 90=+y
    state: str                 # "idle" / "talking" / ...


class ObstacleInfo(NamedTuple):
    category: str
    pos: tuple[float, float]   # (x, y)


class SceneDescription(NamedTuple):
    humans: list[HumanInfo]
    obstacles: list[ObstacleInfo]
    # 推断出的交互群组：[[human_idx, ...], ...]
    groups: list[list[int]]


def _parse_state_from_xml(xml_path: str) -> str:
    """从 human_xml 路径名提取动作状态，如 'talking' / 'idle'。"""
    name = Path(xml_path).parent.name.lower()
    if "talking" in name:
        return "talking"
    if "sitting" in name:
        return "sitting"
    if "walking" in name:
        return "walking"
    return "idle"


def _infer_groups(humans: list[HumanInfo]) -> list[list[int]]:
    """
    简单规则：两个人都是 talking 且距离 < 1.5m，视为一个对话群组。
    """
    groups: list[list[int]] = []
    used = set()
    talkers = [i for i, h in enumerate(humans) if h.state == "talking"]
    for i in talkers:
        if i in used:
            continue
        for j in talkers:
            if j <= i or j in used:
                continue
            dx = humans[i].pos[0] - humans[j].pos[0]
            dy = humans[i].pos[1] - humans[j].pos[1]
            if math.hypot(dx, dy) < 1.5:
                groups.append([i, j])
                used.add(i)
                used.add(j)
    return groups


def extract_scene(layout_json_path: str | Path) -> SceneDescription:
    """解析 layout JSON，返回结构化的场景描述。"""
    with open(layout_json_path) as f:
        data = json.load(f)

    hr = data["human_runtime"]

    humans: list[HumanInfo] = []

    # 主人物
    humans.append(HumanInfo(
        pos=(float(hr["human_pos"][0]), float(hr["human_pos"][1])),
        yaw_deg=float(hr["human_yaw_deg"]),
        state=_parse_state_from_xml(hr["human_xml"]),
    ))

    # 额外人物
    for eh in hr.get("extra_humans", []):
        if not eh.get("enabled", True):
            continue
        humans.append(HumanInfo(
            pos=(float(eh["human_pos"][0]), float(eh["human_pos"][1])),
            yaw_deg=float(eh["human_yaw_deg"]),
            state=_parse_state_from_xml(eh["human_xml"]),
        ))

    # 障碍物（只取导航相关类别）
    obstacles: list[ObstacleInfo] = []
    for obj in data.get("objects", []):
        if obj["category"] in OBSTACLE_CATEGORIES:
            obstacles.append(ObstacleInfo(
                category=obj["category"],
                pos=(round(float(obj["position_xyz"][0]), 2),
                     round(float(obj["position_xyz"][1]), 2)),
            ))

    groups = _infer_groups(humans)
    return SceneDescription(humans=humans, obstacles=obstacles, groups=groups)


# ---------------------------------------------------------------------------
# Prompt 构建
# ---------------------------------------------------------------------------

# FloorPlan203 的实际范围（从 object positions 分析得到）
SCENE_X_MIN, SCENE_X_MAX = -7.0, 1.0   # 8m
SCENE_Y_MIN, SCENE_Y_MAX = -1.0, 7.0   # 8m
COARSE_GRID_SIZE = 16                   # 16×16，每格 0.5m×0.5m


def _world_to_coarse_cell(x: float, y: float) -> tuple[int, int]:
    """世界坐标 → 粗粒度 grid 的 (row, col)，row=0 对应 y_max（上方）。"""
    col = int((x - SCENE_X_MIN) / (SCENE_X_MAX - SCENE_X_MIN) * COARSE_GRID_SIZE)
    row = int((SCENE_Y_MAX - y) / (SCENE_Y_MAX - SCENE_Y_MIN) * COARSE_GRID_SIZE)
    col = max(0, min(COARSE_GRID_SIZE - 1, col))
    row = max(0, min(COARSE_GRID_SIZE - 1, row))
    return row, col


def _yaw_to_direction(yaw_deg: float) -> str:
    yaw = yaw_deg % 360
    if yaw < 45 or yaw >= 315:
        return "+x (east)"
    if yaw < 135:
        return "+y (north)"
    if yaw < 225:
        return "-x (west)"
    return "-y (south)"


def build_prompt(scene: SceneDescription) -> str:
    """构造给 LLM 的 prompt，要求输出 16×16 social costmap。"""

    cell_m = (SCENE_X_MAX - SCENE_X_MIN) / COARSE_GRID_SIZE  # 0.5m

    lines = [
        "You are a social navigation planner for a mobile robot.",
        "",
        "Task: Generate a social cost map for the room below.",
        "Higher cost = robot should avoid this area for social reasons.",
        "0.0 = socially fine, 10.0 = must avoid (e.g., inside a conversation group).",
        "",
        f"Room: x ∈ [{SCENE_X_MIN}, {SCENE_X_MAX}] m, y ∈ [{SCENE_Y_MIN}, {SCENE_Y_MAX}] m",
        f"Grid: {COARSE_GRID_SIZE}×{COARSE_GRID_SIZE} cells, each cell = {cell_m}m × {cell_m}m",
        f"Grid layout: row 0 = y={SCENE_Y_MAX} (north), row {COARSE_GRID_SIZE-1} = y={SCENE_Y_MIN} (south)",
        f"             col 0 = x={SCENE_X_MIN} (west),  col {COARSE_GRID_SIZE-1} = x={SCENE_X_MAX} (east)",
        "",
        "=== People in the room ===",
    ]

    for i, h in enumerate(scene.humans):
        row, col = _world_to_coarse_cell(*h.pos)
        direction = _yaw_to_direction(h.yaw_deg)
        lines.append(
            f"Person {i}: pos=({h.pos[0]:.2f}, {h.pos[1]:.2f})m  "
            f"grid=({row},{col})  facing={direction}  state={h.state}"
        )

    if scene.groups:
        lines.append("")
        lines.append("=== Detected conversation groups ===")
        for g in scene.groups:
            names = ", ".join(f"Person {i}" for i in g)
            lines.append(f"Group: [{names}] — they are talking to each other")

    lines += [
        "",
        "=== Key obstacles (furniture) ===",
    ]
    for obs in scene.obstacles:
        row, col = _world_to_coarse_cell(*obs.pos)
        lines.append(f"{obs.category}: pos=({obs.pos[0]:.2f}, {obs.pos[1]:.2f})m  grid=({row},{col})")

    lines += [
        "",
        "=== Social norms to encode ===",
        "1. Personal space: cost rises sharply within 1.0m of any person (peak ~8.0).",
        "2. Front space: person's front hemisphere gets ~1.5x higher cost than behind.",
        "3. Conversation group: the space *between* conversing people gets cost 10.0.",
        "   Also add a buffer of cost ~7.0 in a 1.5m radius around the group.",
        "4. Passing behind a person is preferred over passing in front.",
        "5. Cells with furniture are already blocked; no need for extra social cost there.",
        "",
        "Output ONLY valid JSON, no explanation:",
        '{',
        '  "costmap": [',
        '    [row0_col0, row0_col1, ..., row0_col15],',
        '    [row1_col0, ...],',
        '    ...',
        '    [row15_col0, ..., row15_col15]',
        '  ]',
        '}',
        "",
        "All values must be floats in [0.0, 10.0]. Output the JSON now.",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM 调用
# ---------------------------------------------------------------------------

def _call_anthropic(prompt: str, model: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model=model,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def _call_openai_compat(prompt: str, model: str, base_url: str, api_key: str) -> str:
    """通用 OpenAI 兼容接口（OpenAI / 豆包 / DeepSeek 等）。"""
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


def _load_env() -> None:
    """从项目根目录的 .env 文件加载环境变量（如果有 python-dotenv）。"""
    try:
        from dotenv import load_dotenv
        from pathlib import Path
        env_path = Path(__file__).resolve().parents[2] / ".env"
        if env_path.exists():
            load_dotenv(env_path)
    except ImportError:
        pass  # 没装 python-dotenv 就跳过，依赖环境变量已有值


def _call_llm(prompt: str, model: str) -> str:
    """
    根据 model 前缀自动选择后端：
      claude-*   → Anthropic API  (需要 ANTHROPIC_API_KEY)
      doubao-*   → 豆包 ARK API   (需要 ARK_API_KEY)
      ep-*       → 豆包推理接入点  (需要 ARK_API_KEY)
      kimi-*     → Kimi API       (需要 KIMI_API_KEY)
      gpt-*      → OpenAI API     (需要 OPENAI_API_KEY)
      其他       → OpenAI API
    """
    _load_env()

    if model.startswith("claude"):
        return _call_anthropic(prompt, model)

    if model.lower().startswith(("doubao", "ep-")):
        return _call_openai_compat(
            prompt, model,
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=os.environ["ARK_API_KEY"],
        )

    if model.lower().startswith("kimi"):
        return _call_openai_compat(
            prompt, model,
            base_url="https://api.moonshot.cn/v1",
            api_key=os.environ["KIMI_API_KEY"],
        )

    # OpenAI 或其他兼容接口
    return _call_openai_compat(
        prompt, model,
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.environ["OPENAI_API_KEY"],
    )


def _parse_coarse_costmap(llm_response: str) -> np.ndarray:
    """从 LLM 响应中解析 16×16 costmap，返回 float32 numpy array。"""
    # 找到 JSON 部分（LLM 有时会输出额外文字）
    start = llm_response.find("{")
    end = llm_response.rfind("}") + 1
    if start < 0 or end <= start:
        raise ValueError(f"LLM 响应中找不到 JSON: {llm_response[:200]}")

    data = json.loads(llm_response[start:end])
    raw = data["costmap"]

    if len(raw) != COARSE_GRID_SIZE:
        raise ValueError(f"costmap 行数应为 {COARSE_GRID_SIZE}，实际得到 {len(raw)}")

    arr = np.array(raw, dtype=np.float32)
    if arr.shape != (COARSE_GRID_SIZE, COARSE_GRID_SIZE):
        raise ValueError(f"costmap shape 应为 ({COARSE_GRID_SIZE},{COARSE_GRID_SIZE})，实际 {arr.shape}")

    # 归一化到 [0, 1]（MPPI 里通过 social_weight 缩放）
    arr = np.clip(arr, 0.0, 10.0) / 10.0
    return arr


# ---------------------------------------------------------------------------
# 坐标系对齐：粗粒度 → MPPI grid 分辨率
# ---------------------------------------------------------------------------

def _upsample_to_grid(
    coarse: np.ndarray,
    scene_map,
    target_shape: tuple[int, int],
    downscale: int,
) -> np.ndarray:
    """
    把 16×16 的粗粒度 costmap 插值到 MPPI grid_free 的分辨率。

    注意：粗粒度 costmap 的坐标系是固定的 (SCENE_X_MIN~X_MAX, SCENE_Y_MIN~Y_MAX)，
    而 MPPI grid 的坐标系由 scene_map.world_to_map 决定。
    这里用双线性插值简单上采样，假设两者覆盖范围基本一致。
    如需精确对齐，可改用坐标映射版本。
    """
    zoom_factors = (
        target_shape[0] / coarse.shape[0],
        target_shape[1] / coarse.shape[1],
    )
    upsampled = zoom(coarse, zoom_factors, order=1)  # 双线性
    # 确保 shape 精确匹配（zoom 可能差1像素）
    result = np.zeros(target_shape, dtype=np.float32)
    h = min(upsampled.shape[0], target_shape[0])
    w = min(upsampled.shape[1], target_shape[1])
    result[:h, :w] = upsampled[:h, :w]
    return result


# ---------------------------------------------------------------------------
# 解析式 Social Costmap（无需 LLM）
# ---------------------------------------------------------------------------

def build_analytic_costmap_coarse(scene: SceneDescription) -> np.ndarray:
    """
    基于 proxemics 规则直接计算 16×16 social costmap，不调用任何 API。

    规则：
    1. 个人空间：以每个人为中心的非对称高斯
         前方 σ_front=1.2m，后方 σ_back=0.7m，侧向 σ_side=0.8m
    2. 对话群组：群组中心额外叠加高斯（σ=0.6m，峰值=1.0），
         保证机器人不穿越正在交谈的两人之间。

    返回：shape=(COARSE_GRID_SIZE, COARSE_GRID_SIZE) 的 float32，值域 [0, 1]。
    """
    cell_m = (SCENE_X_MAX - SCENE_X_MIN) / COARSE_GRID_SIZE

    # 每个 cell 的世界坐标中心
    cols = np.arange(COARSE_GRID_SIZE)
    rows = np.arange(COARSE_GRID_SIZE)
    cx = SCENE_X_MIN + (cols + 0.5) * cell_m   # (16,)  col→x
    cy = SCENE_Y_MAX - (rows + 0.5) * cell_m   # (16,)  row→y (row0=北/y_max)
    CX, CY = np.meshgrid(cx, cy)               # (16, 16)

    grid = np.zeros((COARSE_GRID_SIZE, COARSE_GRID_SIZE), dtype=np.float32)

    # --- 个人空间（每人一个非对称高斯） ---
    for h in scene.humans:
        hx, hy = h.pos
        yaw_rad = math.radians(h.yaw_deg)
        fx, fy = math.cos(yaw_rad), math.sin(yaw_rad)  # 朝向单位向量

        dx = CX - hx
        dy = CY - hy

        along = dx * fx + dy * fy    # 沿朝向分量（>0 = 前方）
        perp  = -dx * fy + dy * fx   # 垂直朝向分量

        sigma_along = np.where(along >= 0, 1.2, 0.7)   # 前方更宽
        sigma_side  = 0.8

        personal_cost = np.exp(
            -(along ** 2) / (2 * sigma_along ** 2)
            -(perp  ** 2) / (2 * sigma_side  ** 2)
        )
        grid = np.maximum(grid, personal_cost)

    # --- 对话群组（群组中心额外高代价，防止机器人穿越） ---
    for group in scene.groups:
        if len(group) < 2:
            continue
        gx = float(np.mean([scene.humans[i].pos[0] for i in group]))
        gy = float(np.mean([scene.humans[i].pos[1] for i in group]))

        dx = CX - gx
        dy = CY - gy
        group_cost = np.exp(-(dx ** 2 + dy ** 2) / (2 * 0.6 ** 2))
        grid = np.maximum(grid, group_cost)

    return np.clip(grid, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def build_social_costmap(
    layout_json_path: str | Path,
    scene_map,
    grid_shape: tuple[int, int],
    downscale: int = 5,
    method: str = "rule",
    llm_model: str = "Doubao-Seed-2.0-lite",
    verbose: bool = True,
) -> np.ndarray:
    """
    完整流程：JSON → social costmap → 上采样 → MPPI 分辨率。

    method="rule" : 解析式（无需 API，推荐默认）
    method="llm"  : LLM 生成（需要配置 API key）

    返回：shape=grid_shape 的 float32 array，值域 [0, 1]。
    """
    scene = extract_scene(layout_json_path)

    if verbose:
        print(f"[social costmap] 场景: {len(scene.humans)} 人, "
              f"{len(scene.obstacles)} 个障碍, {len(scene.groups)} 个对话群组")

    if method == "rule":
        if verbose:
            print("[social costmap] 使用解析式规则生成 costmap...")
        coarse = build_analytic_costmap_coarse(scene)

    elif method == "llm":
        if verbose:
            print(f"[social costmap] 调用 LLM ({llm_model})...")
        response = _call_llm(build_prompt(scene), llm_model)
        coarse = _parse_coarse_costmap(response)

    else:
        raise ValueError(f"未知 method: {method!r}，可选 'rule' 或 'llm'")

    if verbose:
        print(f"[social costmap] 粗粒度 costmap: shape={coarse.shape}, "
              f"max={coarse.max():.3f}, mean={coarse.mean():.3f}")

    fine = _upsample_to_grid(coarse, scene_map, grid_shape, downscale)

    if verbose:
        print(f"[social costmap] 上采样完成: shape={fine.shape}")

    return fine


# ---------------------------------------------------------------------------
# 新架构：LLM 输出 per-entity 参数，代码做高斯合成
# ---------------------------------------------------------------------------

class SocialEntityParams:
    """
    LLM（或规则）对单个实体给出的社交参数。
    score                : 0~1，需要回避的程度
    personal_space       : 各向同性基准半径（米）
    orientation_sensitivity : 前/后 sigma 比值（>1 意味前方延伸更远）
    reason               : LLM 的推理说明
    """
    __slots__ = ("entity_id", "pos", "yaw_deg",
                 "score", "personal_space", "orientation_sensitivity", "reason")

    def __init__(self, entity_id: str, pos: tuple[float, float], yaw_deg: float,
                 score: float, personal_space: float,
                 orientation_sensitivity: float, reason: str = "") -> None:
        self.entity_id              = entity_id
        self.pos                    = pos
        self.yaw_deg                = yaw_deg
        self.score                  = float(score)
        self.personal_space         = float(personal_space)
        self.orientation_sensitivity = float(orientation_sensitivity)
        self.reason                 = reason

    def __repr__(self) -> str:
        return (f"SocialEntityParams(id={self.entity_id}, pos={self.pos}, "
                f"score={self.score:.2f}, ps={self.personal_space:.1f}m, "
                f"os={self.orientation_sensitivity:.1f}x, reason={self.reason!r})")


# --- 规则 fallback（不调 LLM，供 baseline 和测试用）---

_STATE_SCORE: dict[str, float] = {
    "talking":  0.9,   # 交谈中，高度敏感
    "sitting":  0.6,   # 坐着，中等敏感
    "walking":  0.7,   # 行走中
    "idle":     0.5,   # 静止待机
}
_STATE_PS: dict[str, float] = {
    "talking":  1.2,
    "sitting":  0.9,
    "walking":  1.1,
    "idle":     0.8,
}
_STATE_OS: dict[str, float] = {
    "talking":  2.5,   # 交谈时对正面入侵非常敏感
    "sitting":  1.5,
    "walking":  1.8,
    "idle":     1.3,
}


def rule_based_entity_params(scene: SceneDescription) -> list[SocialEntityParams]:
    """根据预设规则生成每个实体的社交参数（LLM 的规则 baseline）。"""
    params: list[SocialEntityParams] = []
    for i, h in enumerate(scene.humans):
        s = h.state
        params.append(SocialEntityParams(
            entity_id=f"person_{i}",
            pos=h.pos,
            yaw_deg=h.yaw_deg,
            score=_STATE_SCORE.get(s, 0.5),
            personal_space=_STATE_PS.get(s, 0.9),
            orientation_sensitivity=_STATE_OS.get(s, 1.5),
            reason=f"rule-based: state={s}",
        ))
    return params


# --- LLM prompt（参数化输出，不再要求输出格子）---

def build_entity_param_prompt(scene: SceneDescription) -> str:
    """
    新版 prompt：要求 LLM 对每个人输出结构化社交参数，
    而非直接输出格子数值。
    """
    lines = [
        "You are a social navigation expert for a mobile robot.",
        "Analyze each person in the scene and output their social cost parameters.",
        "",
        "For each person output:",
        "  score                  : 0.0–1.0  (how much the robot should avoid this person)",
        "  personal_space         : radius in metres the robot should keep (isotropic base)",
        "  orientation_sensitivity: front/back sigma ratio (e.g. 2.0 = front zone is 2× wider)",
        "  reason                 : one sentence explaining your judgement",
        "",
        "=== People ===",
    ]
    for i, h in enumerate(scene.humans):
        lines.append(
            f"person_{i}: pos=({h.pos[0]:.2f}, {h.pos[1]:.2f})m  "
            f"facing={_yaw_to_direction(h.yaw_deg)}  state={h.state}"
        )
    if scene.groups:
        lines.append("\n=== Conversation groups ===")
        for g in scene.groups:
            names = ", ".join(f"person_{i}" for i in g)
            lines.append(f"[{names}] are talking — robot must not pass between them")

    lines += [
        "",
        'Output ONLY valid JSON (no markdown), exactly this schema:',
        '{"entities": [',
        '  {"entity_id": "person_0", "score": 0.8, "personal_space": 1.2,',
        '   "orientation_sensitivity": 2.0, "reason": "..."},',
        '  ...',
        ']}',
    ]
    return "\n".join(lines)


def parse_entity_params(
    llm_response: str,
    scene: SceneDescription,
) -> list[SocialEntityParams]:
    """解析 LLM 输出的 per-entity JSON，构造 SocialEntityParams 列表。"""
    start = llm_response.find("{")
    end   = llm_response.rfind("}") + 1
    if start < 0 or end <= start:
        raise ValueError(f"LLM 响应中找不到 JSON: {llm_response[:200]}")

    data     = json.loads(llm_response[start:end])
    entities = data["entities"]
    params: list[SocialEntityParams] = []

    for e in entities:
        eid = e["entity_id"]
        idx = int(eid.split("_")[-1]) if "_" in eid else 0
        idx = min(idx, len(scene.humans) - 1)
        h   = scene.humans[idx]
        params.append(SocialEntityParams(
            entity_id=eid,
            pos=h.pos,
            yaw_deg=h.yaw_deg,
            score=float(e["score"]),
            personal_space=float(e["personal_space"]),
            orientation_sensitivity=float(e["orientation_sensitivity"]),
            reason=e.get("reason", ""),
        ))
    return params


# --- 连续高斯合成（resolution-independent，直接在世界坐标计算）---

def synthesize_costmap(
    params:      list[SocialEntityParams],
    grid_shape:  tuple[int, int],
    x_range:     tuple[float, float] = (SCENE_X_MIN, SCENE_X_MAX),
    y_range:     tuple[float, float] = (SCENE_Y_MIN, SCENE_Y_MAX),
) -> np.ndarray:
    """
    将 per-entity 社交参数合成为空间代价场（连续各向异性高斯叠加）。

    参数含义：
        score                → 峰值振幅
        personal_space (ps)  → σ_back = σ_side = ps
        orientation_sensitivity (os) → σ_front = ps * os

    返回 float32 array, shape=grid_shape, 值域 [0, 1]，
    row 0 = y_max（北），col 0 = x_min（西）。
    """
    H, W = grid_shape
    cols  = np.linspace(x_range[0], x_range[1], W, endpoint=False) + (x_range[1] - x_range[0]) / W / 2
    rows  = np.linspace(y_range[1], y_range[0], H, endpoint=False) - (y_range[1] - y_range[0]) / H / 2
    CX, CY = np.meshgrid(cols, rows)   # (H, W) 世界坐标

    field = np.zeros((H, W), dtype=np.float32)

    for ep in params:
        hx, hy   = ep.pos
        yaw_rad  = math.radians(ep.yaw_deg)
        fx, fy   = math.cos(yaw_rad), math.sin(yaw_rad)

        dx = CX - hx
        dy = CY - hy

        along = dx * fx + dy * fy     # > 0 = 前方
        perp  = -dx * fy + dy * fx    # 侧向

        ps = ep.personal_space
        os = ep.orientation_sensitivity
        sigma_front = ps * os          # 前方 sigma 更大（延伸更远）
        sigma_back  = ps               # 后方 sigma
        sigma_side  = ps

        sigma_along = np.where(along >= 0, sigma_front, sigma_back)

        cost = ep.score * np.exp(
            -(along ** 2) / (2 * sigma_along ** 2)
            -(perp  ** 2) / (2 * sigma_side  ** 2)
        )
        field = np.maximum(field, cost.astype(np.float32))

    return np.clip(field, 0.0, 1.0)


def build_param_costmap(
    scene:        SceneDescription,
    grid_shape:   tuple[int, int],
    method:       str   = "rule",    # "rule" | "llm"
    llm_model:    str   = "Doubao-Seed-2.0-lite",
    verbose:      bool  = True,
) -> np.ndarray:
    """
    新版主入口：SceneDescription → per-entity params → 连续高斯合成 → costmap。

    method="rule": 用规则生成参数（不需要 API）
    method="llm" : 用 LLM 生成参数（需要 API key）
    """
    if method == "rule":
        params = rule_based_entity_params(scene)
    elif method == "llm":
        prompt   = build_entity_param_prompt(scene)
        response = _call_llm(prompt, llm_model)
        params   = parse_entity_params(response, scene)
    else:
        raise ValueError(f"未知 method: {method!r}")

    if verbose:
        for p in params:
            print(f"  {p}")

    return synthesize_costmap(params, grid_shape)


# ---------------------------------------------------------------------------
# 实时仿真接口（供 interactive_sim 调用，不依赖 layout JSON）
# ---------------------------------------------------------------------------

def _build_live_prompt(agents: dict, robot_pos, robot_goal, groups: list, t: float) -> str:
    """Build LLM prompt from live simulation state (Agent dict from sim_world)."""
    lines = [
        "You are a social navigation assistant for a mobile robot.",
        "Your job: analyze the people in the room and output social cost parameters",
        "so the robot can navigate in a socially compliant way.",
        f"Scene time: t = {t:.1f}s",
        "",
        "=== ROBOT ===",
    ]
    if robot_pos is not None:
        lines.append(f"  position : ({float(robot_pos[0]):.1f}, {float(robot_pos[1]):.1f}) m")
    if robot_goal is not None:
        lines.append(f"  goal     : ({float(robot_goal[0]):.1f}, {float(robot_goal[1]):.1f}) m")
        if robot_pos is not None:
            dist = float(np.linalg.norm(np.asarray(robot_goal) - np.asarray(robot_pos)))
            lines.append(f"  dist to goal: {dist:.1f} m")

    lines += ["", "=== PEOPLE ==="]
    for aid, ag in agents.items():
        facing = _yaw_to_direction(float(ag.heading_deg))
        lines.append(
            f"  {aid}: pos=({float(ag.pos[0]):.1f}, {float(ag.pos[1]):.1f})m  "
            f"activity={ag.activity}  facing={facing}"
        )

    if groups:
        lines += ["", "=== ACTIVE CONVERSATIONS (robot must NOT interrupt) ==="]
        for grp in groups:
            lines.append(f"  [{', '.join(grp)}] are in conversation with each other")

    lines += [
        "",
        "=== YOUR TASK ===",
        "For EACH person output social navigation parameters:",
        "  score                  : 0.0–1.0  (avoidance importance; 1.0 = must avoid)",
        "  personal_space         : metres to keep (0.5–2.0 m)",
        "  orientation_sensitivity: front/back sigma ratio (1.0–3.0; higher = larger front zone)",
        "  reason                 : one sentence of social reasoning",
        "",
        "Key social norms to apply:",
        "  • Conversation groups (score≥0.9, space≥1.5 m) — never cut between them",
        "  • Walking people (score≈0.7, space≈1.0 m) — they are dynamic, stay clear",
        "  • Standing/idle (score≈0.5, space≈0.8 m)",
        "  • Always prefer passing BEHIND a person rather than in front",
        "  • Give extra space to elderly or people carrying things (if mentioned)",
        "",
        'Output ONLY valid JSON — no markdown, no explanation:',
        '{"entities": [',
        '  {"entity_id": "P0", "score": 0.9, "personal_space": 1.5,',
        '   "orientation_sensitivity": 2.5, "reason": "in active conversation, should not be disturbed"},',
        '  {"entity_id": "P1", "score": 0.6, "personal_space": 0.9,',
        '   "orientation_sensitivity": 1.5, "reason": "standing idle, moderate avoidance"},',
        '  ...',
        ']}',
    ]
    return "\n".join(lines)


def _parse_live_params(llm_response: str, agents: dict) -> list[SocialEntityParams]:
    """Parse LLM JSON response for live simulation agents."""
    start = llm_response.find("{")
    end   = llm_response.rfind("}") + 1
    if start < 0 or end <= start:
        raise ValueError(f"No JSON found in LLM response: {llm_response[:200]}")
    data = json.loads(llm_response[start:end])
    params = []
    for e in data["entities"]:
        eid = e["entity_id"]
        if eid not in agents:
            continue
        ag = agents[eid]
        params.append(SocialEntityParams(
            entity_id=eid,
            pos=(float(ag.pos[0]), float(ag.pos[1])),
            yaw_deg=float(ag.heading_deg),
            score=float(e.get("score", 0.7)),
            personal_space=float(e.get("personal_space", 1.0)),
            orientation_sensitivity=float(e.get("orientation_sensitivity", 1.5)),
            reason=e.get("reason", ""),
        ))
    return params


def build_live_costmap(
    agents: dict,                          # id → Agent (from sim_world.py)
    grid_shape: tuple[int, int],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    method: str = "rule",                  # "rule" | "llm"
    llm_model: str = "doubao-pro-32k",
    robot_pos=None,
    robot_goal=None,
    groups: list[list[str]] | None = None,
    t: float = 0.0,
) -> tuple[np.ndarray, list[SocialEntityParams]]:
    """
    Real-time scene → (social costmap, entity params used).

    Replaces build_social_costmap() for the interactive simulator:
    no layout JSON needed — takes live Agent objects directly.

    method="rule" : rule-based proxemics (fast, no API)
    method="llm"  : LLM semantic reasoning (needs ARK_API_KEY / ANTHROPIC_API_KEY)
    """
    groups = groups or []

    if not agents:
        return np.zeros(grid_shape, dtype=np.float32), []

    if method == "rule":
        params: list[SocialEntityParams] = []
        for aid, ag in agents.items():
            s = getattr(ag, "activity", "standing")
            params.append(SocialEntityParams(
                entity_id=aid,
                pos=(float(ag.pos[0]), float(ag.pos[1])),
                yaw_deg=float(ag.heading_deg),
                score=_STATE_SCORE.get(s, 0.5),
                personal_space=_STATE_PS.get(s, 0.9),
                orientation_sensitivity=_STATE_OS.get(s, 1.5),
                reason=f"rule-based: {s}",
            ))
    elif method == "llm":
        _load_env()
        prompt   = _build_live_prompt(agents, robot_pos, robot_goal, groups, t)
        response = _call_llm(prompt, llm_model)
        params   = _parse_live_params(response, agents)
    else:
        raise ValueError(f"Unknown method: {method!r}")

    cm = synthesize_costmap(params, grid_shape, x_range=x_range, y_range=y_range)
    return cm, params


# ---------------------------------------------------------------------------
# 调试：可视化（可选）
# ---------------------------------------------------------------------------

def visualize_coarse(coarse: np.ndarray, scene: SceneDescription) -> None:
    """在终端打印粗粒度 costmap 的 ASCII 可视化。"""
    chars = " ░▒▓█"
    print("=== Social Costmap (coarse 16×16) ===")
    print("  " + "".join(f"{c:2d}" for c in range(COARSE_GRID_SIZE)))
    for r in range(COARSE_GRID_SIZE):
        row_str = f"{r:2d} "
        for c in range(COARSE_GRID_SIZE):
            v = float(coarse[r, c])
            idx = min(int(v * len(chars)), len(chars) - 1)
            row_str += chars[idx] + " "
        print(row_str)

    print("\nHuman positions on grid:")
    for i, h in enumerate(scene.humans):
        row, col = _world_to_coarse_cell(*h.pos)
        print(f"  Person {i} ({h.state}): ({row}, {col})")


# ---------------------------------------------------------------------------
# Neural field 对接：SceneDescription → SceneGraph → SocialCostField costmap
# ---------------------------------------------------------------------------

def scene_desc_to_graph(scene: SceneDescription,
                        robot_pos: tuple[float, float] | None = None,
                        robot_goal: tuple[float, float] | None = None,
                        relation_description: str | None = None):
    """
    把 extract_scene() 的输出转换成 SocialCostField 所需的 SceneGraph。

    relation_description : 对话群组的自然语言描述，None 时用默认
                           "two people having a conversation"
    """
    from experiments.social_nav.social_cost_field import (
        SceneGraph, TYPE_AGENT, TYPE_ROBOT, TYPE_GOAL,
        RELATION_INTENSITY,
    )

    g = SceneGraph()
    node_ids = []

    for h in scene.humans:
        activity = h.state if h.state in ("standing", "walking", "talking", "sitting") else "standing"
        idx = g.add_node(list(h.pos), heading_deg=h.yaw_deg,
                         activity=activity, entity_type=TYPE_AGENT)
        node_ids.append(idx)

    if robot_pos is not None:
        g.add_node(list(robot_pos), heading_deg=0.0,
                   activity="standing", entity_type=TYPE_ROBOT)
    if robot_goal is not None:
        g.add_node(list(robot_goal), heading_deg=0.0,
                   activity="standing", entity_type=TYPE_GOAL)

    # 对话群组 → 关系边
    desc = relation_description or "two people having a conversation, engaged and focused"
    for group in scene.groups:
        if len(group) >= 2:
            g.add_edge(node_ids[group[0]], node_ids[group[1]],
                       relation="casual_chat",
                       description=desc,
                       intensity=RELATION_INTENSITY["casual_chat"])

    return g


def build_nn_costmap(
    layout_json_path: str | Path,
    scene_map,
    grid_shape: tuple[int, int],
    checkpoint_path: str | Path,
    robot_pos: tuple[float, float] | None = None,
    robot_goal: tuple[float, float] | None = None,
    relation_description: str | None = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    用 SocialCostField 神经场生成 social costmap。

    返回：shape=grid_shape 的 float32 array，值域 [0, 1]。
    世界坐标范围从 scene_map 自动推断。
    """
    from experiments.social_nav.social_cost_field import SocialCostField

    scene = extract_scene(layout_json_path)
    if verbose:
        print(f"[nn costmap] 场景: {len(scene.humans)} 人, "
              f"{len(scene.groups)} 个对话群组")

    graph = scene_desc_to_graph(scene, robot_pos, robot_goal, relation_description)

    model = SocialCostField.load(checkpoint_path)

    # 从 scene_map 推断世界坐标范围
    h_px, w_px = scene_map.occupancy_map.shape
    scale = scene_map.occupancy_scale_factor   # metres per pixel
    x_range = (-w_px / 2 * scale, w_px / 2 * scale)
    y_range = (-h_px / 2 * scale, h_px / 2 * scale)

    grid_h, grid_w = grid_shape
    costmap = model.predict_grid(graph, grid_h=grid_h, grid_w=grid_w,
                                 x_range=x_range, y_range=y_range)

    if verbose:
        print(f"[nn costmap] 生成完成: shape={costmap.shape}, "
              f"max={costmap.max():.3f}, mean={costmap.mean():.3f}")

    return costmap.astype(np.float32)
