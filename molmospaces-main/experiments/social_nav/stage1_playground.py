"""
Stage1 bird-view playground.

Pure 2D social-navigation design tool:
- load an occupancy/bird-view image
- place robot start/goal
- add humans and conversation relations
- inspect rule-based social costmap and A* path live
- export a HumanSSG-compatible scene_graph.json for repeatable Stage1 tests

No MuJoCo runtime is loaded here.

Optimized version:
- caches scaled background surfaces
- caches scaled costmap overlays
- throttles recomputation while dragging humans
- vectorizes human hard-block masks
- keeps the original CLI and public interfaces unchanged
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pygame
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import label as label_components

REPO_ROOT = Path(__file__).resolve().parents[2]
GRAD_ROOT = REPO_ROOT.parent
for _p in (str(REPO_ROOT), str(GRAD_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.social_nav.cost.llm_costmap import (
    LLMPromptTemplates,
    build_entity_params,
    get_default_prompt_templates,
)
from pipeline.scene_bridge import HumanInfo, SceneDescription, scene_graph_to_scene_description


ACTIVITIES = {
    "idle": ["STANDING"],
    "talking": ["SPEAK", "OBSERVE"],
    "sitting": ["SITTING"],
    "walking": ["WALKING"],
}
STATE_ORDER = ["idle", "talking", "sitting", "walking"]


def _parse_xy(s: str) -> tuple[float, float]:
    vals = [float(v.strip()) for v in s.split(",")]
    if len(vals) != 2:
        raise argparse.ArgumentTypeError("Expected x,y")
    return vals[0], vals[1]


def _parse_xyz(s: str) -> tuple[float, float, float]:
    vals = [float(v.strip()) for v in s.split(",")]
    if len(vals) != 3:
        raise argparse.ArgumentTypeError("Expected x,y,yaw_deg")
    return vals[0], vals[1], vals[2]


def _parse_bounds(s: str) -> tuple[float, float, float, float] | None:
    if s.lower() == "auto":
        return None
    vals = [float(v.strip()) for v in s.split(",")]
    if len(vals) != 4:
        raise argparse.ArgumentTypeError("Expected x_min,x_max,y_min,y_max")
    if vals[1] <= vals[0] or vals[3] <= vals[2]:
        raise argparse.ArgumentTypeError("Bounds must satisfy x_max>x_min and y_max>y_min")
    return vals[0], vals[1], vals[2], vals[3]


def infer_bounds_from_scene_graph(path: Path | None, margin: float = 0.6) -> tuple[float, float, float, float]:
    if path is None or not path.is_file():
        return 0.0, 10.0, 0.0, 10.0
    payload = json.loads(path.read_text())
    xs: list[float] = []
    ys: list[float] = []
    for node in payload.get("nodes", []):
        center = node.get("bbox_center")
        extent = node.get("bbox_extent")
        if not center or not extent or len(center) < 2 or len(extent) < 2:
            continue
        x, y = float(center[0]), float(center[1])
        ex, ey = max(float(extent[0]), 0.05), max(float(extent[1]), 0.05)
        xs.extend([x - ex * 0.5, x + ex * 0.5])
        ys.extend([y - ey * 0.5, y + ey * 0.5])
    if not xs or not ys:
        return 0.0, 10.0, 0.0, 10.0
    x_min = math.floor((min(xs) - margin) * 10.0) / 10.0
    x_max = math.ceil((max(xs) + margin) * 10.0) / 10.0
    y_min = math.floor((min(ys) - margin) * 10.0) / 10.0
    y_max = math.ceil((max(ys) + margin) * 10.0) / 10.0
    return x_min, x_max, y_min, y_max


class BirdviewSceneMap:
    def __init__(self, occupancy: np.ndarray, bounds: tuple[float, float, float, float]) -> None:
        self.occupancy = occupancy.astype(bool)
        self.x_min, self.x_max, self.y_min, self.y_max = bounds
        self.height, self.width = self.occupancy.shape
        self.x_px_per_m = self.width / (self.x_max - self.x_min)
        self.y_px_per_m = self.height / (self.y_max - self.y_min)
        self.world_to_map = np.array(
            [
                [0.0, -self.y_px_per_m, 0.0, self.y_max * self.y_px_per_m],
                [self.x_px_per_m, 0.0, 0.0, -self.x_min * self.x_px_per_m],
            ],
            dtype=np.float64,
        )

    def pos_m_to_px(self, xyz: np.ndarray) -> np.ndarray:
        arr = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
        row = (self.y_max - arr[:, 1]) * self.y_px_per_m
        col = (arr[:, 0] - self.x_min) * self.x_px_per_m
        return np.stack([row, col], axis=-1).astype(np.float32)

    def pos_px_to_m(self, px: np.ndarray) -> np.ndarray:
        arr = np.asarray(px, dtype=np.float32).reshape(-1, 2)
        x = arr[:, 1] / self.x_px_per_m + self.x_min
        y = self.y_max - arr[:, 0] / self.y_px_per_m
        z = np.zeros_like(x)
        return np.stack([x, y, z], axis=-1).astype(np.float32)


def _make_grid_from_occ(occ: np.ndarray, downscale: int) -> np.ndarray:
    pad_rows = (-occ.shape[0]) % downscale
    pad_cols = (-occ.shape[1]) % downscale
    padded = np.zeros((occ.shape[0] + pad_rows, occ.shape[1] + pad_cols), dtype=bool)
    padded[: occ.shape[0], : occ.shape[1]] = occ
    return (
        padded.reshape(
            padded.shape[0] // downscale,
            downscale,
            padded.shape[1] // downscale,
            downscale,
        )
        .min(axis=1)
        .min(axis=-1)
    )


def make_downscaled_grid(scene_map, downscale: int) -> np.ndarray:
    return _make_grid_from_occ(scene_map.occupancy.astype(bool), downscale)


def build_map_from_birdview(cfg: PlaygroundConfig):
    img = Image.open(cfg.birdview_image).convert("L")
    gray = np.asarray(img, dtype=np.float32) / 255.0
    occ = gray <= cfg.birdview_free_threshold if cfg.birdview_free_is_dark else gray >= cfg.birdview_free_threshold
    scene_map = BirdviewSceneMap(occ, cfg.map_bounds)
    downscale = max(int(cfg.birdview_downscale), 1)
    grid_free = _make_grid_from_occ(scene_map.occupancy, downscale)
    grid_spacing_x = (scene_map.x_max - scene_map.x_min) / scene_map.width * downscale
    grid_spacing_y = (scene_map.y_max - scene_map.y_min) / scene_map.height * downscale
    grid_spacing = float(max(grid_spacing_x, grid_spacing_y))
    dist_transform = distance_transform_edt(grid_free).astype(np.float32)
    print(
        f"[stage1-playground] map={cfg.birdview_image} grid={grid_free.shape} "
        f"free={int(grid_free.sum())} spacing={grid_spacing:.3f}m"
    )
    return scene_map, grid_free, grid_free.copy(), dist_transform, grid_spacing, downscale


def load_display_birdview(cfg: PlaygroundConfig) -> pygame.Surface | None:
    if cfg.background == "scene-graph" and cfg.scene_graph is not None and cfg.scene_graph.is_file():
        return render_scene_graph_surface(cfg.scene_graph, cfg.map_bounds)

    path = cfg.display_image
    if path is None:
        candidate = cfg.birdview_image.with_name("train_9_topdown.png")
        path = candidate if candidate.is_file() else None
    if path is None or not path.is_file():
        return None
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    surf = pygame.surfarray.make_surface(np.transpose(arr, (1, 0, 2)))
    print(f"[stage1-playground] display_image={path}")
    return surf


def build_map_from_scene_graph(cfg: PlaygroundConfig):
    if cfg.scene_graph is None or not cfg.scene_graph.is_file():
        raise FileNotFoundError("scene graph is required for scene-graph map mode")
    x_min, x_max, y_min, y_max = cfg.map_bounds
    width = max(int(round((x_max - x_min) * 100)), 200)
    height = max(int(round((y_max - y_min) * 100)), 200)
    occ = np.ones((height, width), dtype=bool)
    payload = json.loads(cfg.scene_graph.read_text())

    def world_to_rc(x: float, y: float) -> tuple[int, int]:
        row = int(np.clip((y_max - y) / max(y_max - y_min, 1e-6) * height, 0, height - 1))
        col = int(np.clip((x - x_min) / max(x_max - x_min, 1e-6) * width, 0, width - 1))
        return row, col

    inflate = 0.08
    for node in payload.get("nodes", []):
        label = str((node.get("label") or ["object"])[0])
        if label == "person":
            continue
        center = node.get("bbox_center")
        extent = node.get("bbox_extent")
        if not center or not extent or len(center) < 2 or len(extent) < 2:
            continue
        x, y = float(center[0]), float(center[1])
        ex = max(float(extent[0]) + inflate, 0.08)
        ey = max(float(extent[1]) + inflate, 0.08)
        r0, c0 = world_to_rc(x - ex * 0.5, y + ey * 0.5)
        r1, c1 = world_to_rc(x + ex * 0.5, y - ey * 0.5)
        occ[min(r0, r1) : max(r0, r1) + 1, min(c0, c1) : max(c0, c1) + 1] = False

    scene_map = BirdviewSceneMap(occ, cfg.map_bounds)
    downscale = max(int(cfg.birdview_downscale), 1)
    grid_free = _make_grid_from_occ(scene_map.occupancy, downscale)
    grid_spacing_x = (scene_map.x_max - scene_map.x_min) / scene_map.width * downscale
    grid_spacing_y = (scene_map.y_max - scene_map.y_min) / scene_map.height * downscale
    grid_spacing = float(max(grid_spacing_x, grid_spacing_y))
    dist_transform = distance_transform_edt(grid_free).astype(np.float32)
    print(
        f"[stage1-playground] scene_graph_map={cfg.scene_graph} grid={grid_free.shape} "
        f"free={int(grid_free.sum())} spacing={grid_spacing:.3f}m"
    )
    return scene_map, grid_free, grid_free.copy(), dist_transform, grid_spacing, downscale


def _resolve_scene_xml(cfg: PlaygroundConfig) -> Path:
    if cfg.scene_xml is not None:
        return cfg.scene_xml
    if cfg.layout_json is not None and cfg.layout_json.is_file():
        payload = json.loads(cfg.layout_json.read_text())
        raw = payload.get("scene_xml")
        if raw:
            path = Path(raw)
            if not path.is_absolute():
                path = REPO_ROOT / path
            return path
    raise FileNotFoundError("Real top-down mode requires --scene-xml or --layout-json with scene_xml")


def build_map_from_scene_xml(cfg: PlaygroundConfig):
    import time
    from molmo_spaces.utils import scene_maps

    print("[stage1-playground] build_map_from_scene_xml START")
    t0 = time.time()
    scene_xml = _resolve_scene_xml(cfg)
    print(f"[stage1-playground] resolved scene_xml={scene_xml} ({time.time() - t0:.1f}s)")

    scene_maps._delete_blacklisted_bodies = lambda spec: 0
    t0 = time.time()
    if "ithor" in str(scene_xml):
        print("[stage1-playground] loading iTHORMap...")
        scene_map = scene_maps.iTHORMap.from_mj_model_path(
            str(scene_xml),
            agent_radius=0.22,
            px_per_m=120,
            device_id=None,
        )
    else:
        print("[stage1-playground] loading ProcTHORMap...")
        scene_map = scene_maps.ProcTHORMap.from_mj_model_path(
            str(scene_xml),
            agent_radius=0.22,
            px_per_m=120,
            device_id=None,
        )
    print(f"[stage1-playground] scene_map loaded ({time.time() - t0:.1f}s)")

    t0 = time.time()
    downscale = max(int(cfg.birdview_downscale), 1)
    grid_free = make_downscaled_grid(scene_map, downscale)
    print(f"[stage1-playground] grid_free created ({time.time() - t0:.1f}s)")

    t0 = time.time()
    grid_free_astar = grid_free.copy()
    grid_spacing = 1.0 / float(getattr(scene_map, "px_per_m", 120)) * downscale
    dist_transform = distance_transform_edt(grid_free_astar).astype(np.float32)
    print(f"[stage1-playground] dist_transform computed ({time.time() - t0:.1f}s)")
    print(
        f"[stage1-playground] real_scene_xml={scene_xml} grid={grid_free.shape} "
        f"free={int(grid_free.sum())} spacing={grid_spacing:.3f}m"
    )
    return scene_map, grid_free, grid_free_astar, dist_transform, grid_spacing, downscale


def infer_bounds_from_scene_map(scene_map) -> tuple[float, float, float, float]:
    h, w = scene_map.occupancy.shape
    corners = np.array([[0, 0], [h - 1, w - 1], [0, w - 1], [h - 1, 0]], dtype=np.float32)
    pts = scene_map.pos_px_to_m(corners)
    return (
        float(np.min(pts[:, 0])),
        float(np.max(pts[:, 0])),
        float(np.min(pts[:, 1])),
        float(np.max(pts[:, 1])),
    )


def render_occupancy_surface(scene_map) -> pygame.Surface:
    occ = scene_map.occupancy.astype(bool)
    rgb = np.zeros((*occ.shape, 3), dtype=np.uint8)
    rgb[:] = np.array([58, 60, 66], dtype=np.uint8)
    rgb[occ] = np.array([238, 236, 228], dtype=np.uint8)
    surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
    print("[stage1-playground] display_real_occupancy=scene_xml")
    return surf


def render_scene_xml_rgb_surface(cfg: PlaygroundConfig, scene_map) -> pygame.Surface | None:
    try:
        import mujoco
        from mujoco import MjData

        from molmo_spaces.env.mj_extensions import MjModelBindings
        from molmo_spaces.renderer.opengl_rendering import MjOpenGLRenderer
        from molmo_spaces.utils.mj_model_and_data_utils import geom_aabb
        from molmo_spaces.utils import scene_maps

        scene_xml = _resolve_scene_xml(cfg)
        spec = mujoco.MjSpec.from_file(str(scene_xml))

        ceiling_geoms = []

        def collect_ceiling(body_spec) -> None:
            for geom in body_spec.geoms:
                if geom.name and "ceiling" in geom.name.lower():
                    ceiling_geoms.append(geom)
            for child in body_spec.bodies:
                collect_ceiling(child)

        collect_ceiling(spec.worldbody)
        for geom in ceiling_geoms:
            spec.delete(geom)
        scene_maps._delete_blacklisted_bodies(spec)

        model = spec.compile()
        data = MjData(model)
        mujoco.mj_forward(model, data)

        floor_ids = []
        for geom_id in range(model.ngeom):
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if geom_name and (geom_name.startswith("room|") or geom_name.startswith("room_")):
                floor_ids.append(geom_id)
        if not floor_ids:
            raise RuntimeError("No floor geoms found for RGB top-down render")

        aabb_center, aabb_size = geom_aabb(model, data, floor_ids, tight_mesh=False)
        aabb_size += np.array([2.0, 2.0, 0.0])
        px_per_m = 120
        height = max(300, round(px_per_m * aabb_size[0]))
        width = max(300, round(px_per_m * aabb_size[1]))

        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = aabb_center
        cam.distance = 8.0
        cam.azimuth = 0
        cam.elevation = -90
        cam.orthographic = 1

        renderer = MjOpenGLRenderer(MjModelBindings(model), height=height, width=width, device_id=None)
        renderer.update(data, cam)
        for camera in renderer.scene.camera:
            camera.orthographic = 1
            camera.frustum_bottom = -aabb_size[0] / 2
            camera.frustum_top = aabb_size[0] / 2
        rgb = renderer.render()
        renderer.close()
        surf = pygame.surfarray.make_surface(np.transpose(np.asarray(rgb, dtype=np.uint8), (1, 0, 2)))
        print("[stage1-playground] display_real_rgb=scene_xml")
        return surf
    except Exception as exc:
        print(f"[stage1-playground] RGB top-down render failed, falling back to occupancy: {exc}")
        return None


def render_scene_graph_surface(
    path: Path,
    bounds: tuple[float, float, float, float],
    px_per_m: int = 140,
) -> pygame.Surface:
    payload = json.loads(path.read_text())
    x_min, x_max, y_min, y_max = bounds
    width = max(int(round((x_max - x_min) * px_per_m)), 600)
    height = max(int(round((y_max - y_min) * px_per_m)), 600)
    surf = pygame.Surface((width, height))
    surf.fill((30, 33, 40))
    font = pygame.font.SysFont("menlo", 15)

    def world_to_px(x: float, y: float) -> tuple[int, int]:
        sx = (x - x_min) / max(x_max - x_min, 1e-6)
        sy = (y_max - y) / max(y_max - y_min, 1e-6)
        return int(sx * width), int(sy * height)

    gx = math.ceil(x_min)
    while gx <= math.floor(x_max):
        px, _ = world_to_px(gx, y_min)
        pygame.draw.line(surf, (50, 54, 64), (px, 0), (px, height), 1)
        gx += 1
    gy = math.ceil(y_min)
    while gy <= math.floor(y_max):
        _, py = world_to_px(x_min, gy)
        pygame.draw.line(surf, (50, 54, 64), (0, py), (width, py), 1)
        gy += 1

    palette = {
        "table": (180, 142, 92),
        "countertop": (168, 166, 152),
        "fridge": (154, 180, 196),
        "chair": (126, 150, 118),
        "shelf": (150, 126, 98),
        "cart": (190, 148, 76),
        "stool": (132, 158, 126),
        "tv_stand": (108, 112, 124),
        "bin": (92, 136, 132),
        "dog_bed": (172, 125, 137),
        "wall_decor": (196, 188, 168),
    }
    for node in payload.get("nodes", []):
        labels = node.get("label") or ["object"]
        label = str(labels[0])
        center = node.get("bbox_center")
        extent = node.get("bbox_extent")
        if not center or not extent or len(center) < 2 or len(extent) < 2:
            continue
        x, y = float(center[0]), float(center[1])
        ex, ey = max(float(extent[0]), 0.08), max(float(extent[1]), 0.08)
        if label == "person":
            p = world_to_px(x, y)
            pygame.draw.circle(surf, (218, 86, 176), p, 11)
            pygame.draw.circle(surf, (55, 55, 60), p, 11, 2)
            yaw = math.radians(float(node.get("heading_deg", 0.0)))
            tip = (int(p[0] + math.cos(yaw) * 26), int(p[1] - math.sin(yaw) * 26))
            pygame.draw.line(surf, (55, 55, 60), p, tip, 3)
            txt = font.render(f"P{node.get('node_id', '')}", True, (35, 36, 40))
            surf.blit(txt, (p[0] + 12, p[1] - 12))
        else:
            x0, y0 = world_to_px(x - ex * 0.5, y + ey * 0.5)
            x1, y1 = world_to_px(x + ex * 0.5, y - ey * 0.5)
            rect = pygame.Rect(min(x0, x1), min(y0, y1), max(abs(x1 - x0), 4), max(abs(y1 - y0), 4))
            color = palette.get(label, (160, 156, 146))
            pygame.draw.rect(surf, color, rect, border_radius=3)
            pygame.draw.rect(surf, (45, 47, 54), rect, 2, border_radius=3)
            if rect.width >= 42 and rect.height >= 16:
                txt = font.render(label[:12], True, (245, 245, 238))
                surf.blit(txt, (rect.x + 4, rect.y + 3))

    node_pos = {}
    for node in payload.get("nodes", []):
        center = node.get("bbox_center")
        if center and len(center) >= 2:
            node_pos[int(node.get("node_id", -1))] = world_to_px(float(center[0]), float(center[1]))
    for a, b, meta in payload.get("edges", []):
        if str(meta.get("relation", "")).upper() not in {"SPEAK", "OBSERVE", "CONVERSATION"}:
            continue
        if int(a) in node_pos and int(b) in node_pos:
            pygame.draw.line(surf, (210, 60, 50), node_pos[int(a)], node_pos[int(b)], 2)

    pygame.draw.rect(surf, (82, 84, 90), surf.get_rect(), 3)
    print(f"[stage1-playground] display_scene_graph={path}")
    return surf


def _world_to_grid(scene_map: BirdviewSceneMap, xy: np.ndarray, downscale: int, shape: tuple[int, int]) -> tuple[int, int]:
    px = scene_map.pos_m_to_px(np.array([[xy[0], xy[1], 0.0]], dtype=np.float32))[0]
    H, W = shape
    return int(np.clip(px[0] / downscale, 0, H - 1)), int(np.clip(px[1] / downscale, 0, W - 1))


def _grid_to_world(scene_map: BirdviewSceneMap, r: int, c: int, downscale: int) -> np.ndarray:
    px = np.array([[(r + 0.5) * downscale, (c + 0.5) * downscale]], dtype=np.float32)
    return scene_map.pos_px_to_m(px)[0, :2].astype(np.float32)


def _synthesize_costmap_on_scene_grid(
    params,
    scene_map,
    grid_shape: tuple[int, int],
    downscale: int,
    distance_transform: np.ndarray | None = None,
    clearance_cap: float = 0.5,
    clearance_weight: float = 0.3,
    back_scale: float = 1.0,
) -> np.ndarray:
    """Synthesize social cost directly on the displayed navigation grid.

    This avoids assuming that world x/y axes linearly align with row/col. That
    assumption is false for some THOR maps because world_to_map can rotate and
    translate the scene.
    """
    H, W = grid_shape
    rows, cols = np.indices((H, W), dtype=np.float32)
    px = np.stack(
        [(rows.reshape(-1) + 0.5) * downscale, (cols.reshape(-1) + 0.5) * downscale],
        axis=-1,
    )
    world = scene_map.pos_px_to_m(px).reshape(H, W, -1)
    GX = world[..., 0]
    GY = world[..., 1]

    field = np.zeros((H, W), dtype=np.float32)
    for ep in params:
        hx, hy = ep.pos
        yaw_rad = math.radians(ep.yaw_deg)
        fx, fy = math.cos(yaw_rad), math.sin(yaw_rad)
        dx = GX - hx
        dy = GY - hy
        along = dx * fx + dy * fy
        perp = -dx * fy + dy * fx
        ps = ep.personal_space
        sigma_along = np.where(along >= 0, ps * ep.orientation_sensitivity, ps * back_scale)
        sigma_side = ep.sigma_perp if ep.sigma_perp is not None else ps
        cost = ep.score * np.exp(
            -(along ** 2) / (2 * sigma_along ** 2)
            -(perp ** 2) / (2 * sigma_side ** 2)
        )
        field = np.maximum(field, cost.astype(np.float32))

    if distance_transform is not None:
        dt = distance_transform.astype(np.float32)
        clearance_cost = clearance_weight * (1.0 - np.clip(dt / clearance_cap, 0.0, 1.0))
        field = np.clip(field + clearance_cost, 0.0, 1.0)

    return field


def _nearest_free_xy(scene_map: BirdviewSceneMap, grid_free: np.ndarray, xy: np.ndarray, downscale: int, max_search: int = 80) -> np.ndarray | None:
    r, c = _world_to_grid(scene_map, xy, downscale, grid_free.shape)
    if grid_free[r, c]:
        return np.asarray(xy, dtype=np.float32)
    H, W = grid_free.shape
    for radius in range(1, max_search + 1):
        for rr in range(max(0, r - radius), min(H, r + radius + 1)):
            for cc in range(max(0, c - radius), min(W, c + radius + 1)):
                if abs(rr - r) != radius and abs(cc - c) != radius:
                    continue
                if grid_free[rr, cc]:
                    return _grid_to_world(scene_map, rr, cc, downscale)
    cells = np.argwhere(grid_free)
    if cells.size > 0:
        d2 = np.square(cells[:, 0] - r) + np.square(cells[:, 1] - c)
        rr, cc = cells[int(np.argmin(d2))]
        return _grid_to_world(scene_map, int(rr), int(cc), downscale)
    return None


def _astar_on_grid(
    scene_map: BirdviewSceneMap,
    grid_free: np.ndarray,
    downscale: int,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    social_costmap: np.ndarray | None,
    social_w: float,
    distance_transform: np.ndarray,
    grid_spacing: float,
    extra_costmap: np.ndarray | None = None,
) -> np.ndarray | None:
    import heapq

    H, W = grid_free.shape
    src = _world_to_grid(scene_map, start_xy, downscale, grid_free.shape)
    dst = _world_to_grid(scene_map, goal_xy, downscale, grid_free.shape)
    if not grid_free[src] or not grid_free[dst]:
        return None

    ext_social = social_w * social_costmap if social_costmap is not None else None
    dt = distance_transform.astype(np.float32)
    ext_clear = 3.0 * (1.0 - np.clip(dt * grid_spacing / 0.5, 0.0, 1.0))
    ext_extra = extra_costmap if extra_costmap is not None else None
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    costs = [1.0, 1.0, 1.0, 1.0, math.sqrt(2), math.sqrt(2), math.sqrt(2), math.sqrt(2)]

    def h(r: int, c: int) -> float:
        return math.hypot(r - dst[0], c - dst[1])

    dist = np.full((H, W), np.inf, dtype=np.float64)
    dist[src] = 0.0
    prev: dict[tuple[int, int], tuple[int, int]] = {}
    heap = [(h(*src), src)]
    while heap:
        f, (r, c) = heapq.heappop(heap)
        if (r, c) == dst:
            break
        if f > dist[r, c] + h(r, c) + 1e-9:
            continue
        for (dr, dc), base in zip(dirs, costs):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W and grid_free[nr, nc]):
                continue
            overlay = float(ext_clear[nr, nc])
            if ext_social is not None:
                overlay += float(ext_social[nr, nc])
            if ext_extra is not None:
                overlay += float(ext_extra[nr, nc])
            g = dist[r, c] + base * (1.0 + overlay)
            if g < dist[nr, nc]:
                dist[nr, nc] = g
                prev[(nr, nc)] = (r, c)
                heapq.heappush(heap, (g + h(nr, nc), (nr, nc)))

    if dst not in prev and src != dst:
        return None
    cells = []
    cur = dst
    while cur in prev:
        cells.append(cur)
        cur = prev[cur]
    cells.append(src)
    cells.reverse()
    return np.array([_grid_to_world(scene_map, r, c, downscale) for r, c in cells], dtype=np.float32)


def _path_penalty(path: np.ndarray, scene_map: BirdviewSceneMap, shape: tuple[int, int], downscale: int) -> np.ndarray:
    penalty = np.zeros(shape, dtype=np.float32)
    H, W = shape
    radius = 3
    for xy in path:
        r, c = _world_to_grid(scene_map, xy, downscale, shape)
        for rr in range(max(0, r - radius), min(H, r + radius + 1)):
            for cc in range(max(0, c - radius), min(W, c + radius + 1)):
                d = math.hypot(rr - r, cc - c)
                if d <= radius:
                    penalty[rr, cc] = max(penalty[rr, cc], 1.0 - d / radius)
    return penalty


def _segment_allowed(a: np.ndarray, b: np.ndarray, scene_map: BirdviewSceneMap, grid_free: np.ndarray, downscale: int, social_costmap: np.ndarray | None, max_social: float) -> bool:
    dist = float(np.linalg.norm(b - a))
    n = max(int(math.ceil(dist / 0.03)), 2)
    for alpha in np.linspace(0.0, 1.0, n + 1):
        xy = a + alpha * (b - a)
        r, c = _world_to_grid(scene_map, xy, downscale, grid_free.shape)
        if not grid_free[r, c]:
            return False
        if social_costmap is not None and float(social_costmap[r, c]) > max_social:
            return False
    return True


def _shortcut_path(path: np.ndarray, scene_map: BirdviewSceneMap, grid_free: np.ndarray, downscale: int, social_costmap: np.ndarray | None, max_social: float) -> np.ndarray:
    if len(path) < 3:
        return path
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1:
            if _segment_allowed(path[i], path[j], scene_map, grid_free, downscale, social_costmap, max_social):
                break
            j -= 1
        out.append(path[j])
        i = j
    return np.asarray(out, dtype=np.float32)


@dataclass
class Human:
    pos: np.ndarray
    yaw_deg: float = 0.0
    state: str = "talking"

    @property
    def activities(self) -> list[str]:
        return ACTIVITIES.get(self.state, ["STANDING"])


@dataclass
class PlaygroundConfig:
    map_source: str
    background: str
    scene_xml: Path | None
    layout_json: Path | None
    birdview_image: Path
    display_image: Path | None
    map_bounds: tuple[float, float, float, float] | None
    birdview_free_threshold: float
    birdview_free_is_dark: bool
    birdview_downscale: int
    start_pose: tuple[float, float, float]
    goal_xy: tuple[float, float]
    scene_graph: Path | None
    export_scene_graph: Path
    social_method: str
    llm_model: str
    astar_social_weight: float
    astar_human_block_radius: float
    astar_num_candidates: int
    astar_diversity_penalty: float
    astar_candidate_clearance_weight: float
    astar_smoothing: str
    astar_shortcut_social_threshold: float
    isolate_scene_component: bool
    width: int
    height: int


@dataclass
class Stage1MapTransform:
    """Single source of truth for Stage1 screen/grid/world conversion.

    Coordinate contracts:
    - screen: pygame pixel coordinates in the current window.
    - grid: downscaled navigation grid (row, col), aligned with rendered mask.
    - world: MuJoCo/THOR xy in metres.

    Never use map_bounds for hit testing or drawing path entities; THOR maps may
    rotate/translate world coordinates relative to map rows/cols.
    """

    scene_map: object
    grid_shape: tuple[int, int]
    downscale: int
    toolbar_h: int
    panel_w: int

    def map_rect(self, screen_size: tuple[int, int]) -> pygame.Rect:
        w, h = screen_size
        area = pygame.Rect(0, self.toolbar_h, max(1, w - self.panel_w), max(1, h - self.toolbar_h))
        H, W = self.grid_shape
        scale = min(area.width / max(W, 1), area.height / max(H, 1))
        map_w = max(1, int(round(W * scale)))
        map_h = max(1, int(round(H * scale)))
        return pygame.Rect(
            area.x + (area.width - map_w) // 2,
            area.y + (area.height - map_h) // 2,
            map_w,
            map_h,
        )

    def screen_to_grid(self, pos: tuple[int, int], screen_size: tuple[int, int]) -> tuple[int, int] | None:
        rect = self.map_rect(screen_size)
        if not rect.collidepoint(pos):
            return None
        H, W = self.grid_shape
        c = int(np.clip((pos[0] - rect.x) / max(rect.width, 1) * W, 0, W - 1))
        r = int(np.clip((pos[1] - rect.y) / max(rect.height, 1) * H, 0, H - 1))
        return r, c

    def screen_to_world(self, pos: tuple[int, int], screen_size: tuple[int, int]) -> np.ndarray | None:
        rc = self.screen_to_grid(pos, screen_size)
        if rc is None:
            return None
        return self.grid_to_world(*rc)

    def grid_to_world(self, r: int, c: int) -> np.ndarray:
        return _grid_to_world(self.scene_map, r, c, self.downscale)

    def world_to_grid(self, xy: np.ndarray) -> tuple[int, int]:
        px = self.scene_map.pos_m_to_px(np.array([[xy[0], xy[1], 0.0]], dtype=np.float32))[0]
        H, W = self.grid_shape
        r = int(np.clip(px[0] / self.downscale, 0, H - 1))
        c = int(np.clip(px[1] / self.downscale, 0, W - 1))
        return r, c

    def world_to_screen(self, xy: np.ndarray, screen_size: tuple[int, int]) -> tuple[int, int]:
        rect = self.map_rect(screen_size)
        r, c = self.world_to_grid(xy)
        H, W = self.grid_shape
        x = (c + 0.5) / max(W, 1)
        y = (r + 0.5) / max(H, 1)
        return int(rect.x + x * rect.width), int(rect.y + y * rect.height)


class Stage1Playground:
    TOOLBAR_H = 42
    PANEL_W = 280
    HUMAN_R_PX = 8

    def __init__(self, cfg: PlaygroundConfig) -> None:
        self.cfg = cfg
        if self.cfg.map_bounds is None:
            self.cfg.map_bounds = infer_bounds_from_scene_graph(self.cfg.scene_graph)
            print(f"[stage1-playground] inferred map_bounds={self.cfg.map_bounds}")
        if cfg.map_source == "scene-xml":
            (
                self.scene_map,
                self.grid_free,
                self.grid_free_astar,
                self.dist_transform,
                self.grid_spacing,
                self.downscale,
            ) = build_map_from_scene_xml(cfg)
            self.cfg.map_bounds = infer_bounds_from_scene_map(self.scene_map)
            print(f"[stage1-playground] real map_bounds={self.cfg.map_bounds}")
        elif cfg.map_source == "scene-graph":
            (
                self.scene_map,
                self.grid_free,
                self.grid_free_astar,
                self.dist_transform,
                self.grid_spacing,
                self.downscale,
            ) = build_map_from_scene_graph(cfg)
        else:
            (
                self.scene_map,
                self.grid_free,
                self.grid_free_astar,
                self.dist_transform,
                self.grid_spacing,
                self.downscale,
            ) = build_map_from_birdview(cfg)

        self.transform = Stage1MapTransform(
            scene_map=self.scene_map,
            grid_shape=self.grid_free_astar.shape,
            downscale=self.downscale,
            toolbar_h=self.TOOLBAR_H,
            panel_w=self.PANEL_W,
        )
        self.start_xy = np.array([cfg.start_pose[0], cfg.start_pose[1]], dtype=np.float32)
        self.goal_xy = np.array(cfg.goal_xy, dtype=np.float32)
        self.humans: list[Human] = []
        self.groups: list[tuple[int, int]] = []
        self.selected_human: int | None = None
        self.pending_relation: int | None = None
        self.tool = "human"
        self.show_costmap = True
        self.show_rgb = True
        self.cost_view = "current"
        self.dirty = True
        self.social_costmap: np.ndarray | None = None
        self.prev_social_costmap: np.ndarray | None = None
        self.social_params = None
        self.llm_log = ""
        self.llm_dirty = True
        default_prompts = get_default_prompt_templates()
        self.layer1_prompt = default_prompts.layer1_system
        self.layer2_prompt = default_prompts.layer2_system
        self.debug_view = "log"
        self.prompt_editing = False
        self.prompt_cursor = 0
        self.debug_scroll = 0
        self.console_expanded = False
        self.debug_log_lines: list[str] = []
        self.clipboard_ready = False
        self.llm_loading = False
        self._llm_pending_result: tuple[str, object, str] | None = None
        self._llm_lock = threading.Lock()
        self.path: np.ndarray | None = None
        self.planned_start_xy = self.start_xy.copy()
        self.planned_goal_xy = self.goal_xy.copy()
        self.status = ""
        self._component_labels, self._num_components = label_components(self.grid_free)
        self._scene_component_id: int | None = None

        # Performance caches / throttling. Public interface is unchanged.
        self._base_map_cache_key: tuple[int, int, int] | None = None
        self._base_map_cache_surface: pygame.Surface | None = None
        self._blank_map_surface: pygame.Surface | None = None
        self._nav_mask_cache_key: tuple[int, int, int] | None = None
        self._nav_mask_cache_surface: pygame.Surface | None = None
        self._floor_cache_key: tuple[int, int, int] | None = None
        self._floor_cache_surface: pygame.Surface | None = None

        self._cost_overlay_cache_key: tuple[int, int, int, str] | None = None
        self._cost_overlay_cache_surface: pygame.Surface | None = None
        self._costmap_version = 0

        self._drag_dirty_pending = False
        self._drag_last_recompute_ms = 0
        self._drag_recompute_interval_ms = 120

        if cfg.scene_graph is not None:
            self._load_scene_graph(cfg.scene_graph)
        self._scene_component_id = self._choose_scene_component()

        pygame.init()
        pygame.display.set_caption("Stage1 Social Navigation Playground")
        self.screen = pygame.display.set_mode((cfg.width, cfg.height), pygame.RESIZABLE)
        try:
            pygame.scrap.init()
            self.clipboard_ready = True
        except Exception:
            self.clipboard_ready = False
        if cfg.background == "none":
            self.display_surface = None
        elif cfg.background == "rgb" and cfg.map_source == "scene-xml":
            rgb_arr = getattr(self.scene_map, "rgb_array", None)
            if rgb_arr is not None:
                self.display_surface = pygame.surfarray.make_surface(np.transpose(rgb_arr, (1, 0, 2)))
                print("[stage1-playground] display_real_rgb=scene_xml (from map build)")
            else:
                self.display_surface = render_scene_xml_rgb_surface(cfg, self.scene_map)
        elif cfg.background == "occupancy":
            self.display_surface = render_occupancy_surface(self.scene_map)
        else:
            self.display_surface = load_display_birdview(cfg)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("menlo", 13)
        self.font_bold = pygame.font.SysFont("menlo", 13, bold=True)
        self._log(
            "Stage1 playground ready. F1=log, F2=Layer1 prompt, F3=Layer2 prompt, "
            "F4=console, Tab=edit prompt, Cmd/Ctrl+C=copy."
        )

    def run(self) -> None:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                    self._base_map_cache_key = None
                    self._base_map_cache_surface = None
                    self._nav_mask_cache_key = None
                    self._nav_mask_cache_surface = None
                    self._floor_cache_key = None
                    self._floor_cache_surface = None
                    self._cost_overlay_cache_key = None
                    self._cost_overlay_cache_surface = None
                elif event.type == pygame.KEYDOWN:
                    if not self._on_key(event):
                        running = False
                elif event.type == pygame.TEXTINPUT:
                    self._on_text_input(event.text)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self._on_mouse_down(event)
                elif event.type == pygame.MOUSEMOTION:
                    self._on_mouse_motion(event)
                elif event.type == pygame.MOUSEBUTTONUP:
                    self._on_mouse_up(event)
                elif event.type == pygame.MOUSEWHEEL:
                    self._on_mouse_wheel(event)

            self._poll_llm_result()
            if self.dirty:
                self._recompute()
            self._draw()
            pygame.display.flip()
            self.clock.tick(30)

        pygame.quit()

    def _on_key(self, event) -> bool:
        key = event.key
        if self.prompt_editing:
            return self._on_prompt_edit_key(event)

        if self._is_clipboard_shortcut(event, pygame.K_c):
            self._copy_debug_to_clipboard()
            return True
        if key in (pygame.K_ESCAPE, pygame.K_q):
            return False
        if key == pygame.K_F1:
            self.debug_view = "log"
            self.debug_scroll = 0
        elif key == pygame.K_F2:
            self.debug_view = "layer1"
            self.prompt_cursor = min(self.prompt_cursor, len(self._active_prompt()))
            self.debug_scroll = 0
        elif key == pygame.K_F3:
            self.debug_view = "layer2"
            self.prompt_cursor = min(self.prompt_cursor, len(self._active_prompt()))
            self.debug_scroll = 0
        elif key in (pygame.K_F4, pygame.K_BACKQUOTE):
            self.console_expanded = not self.console_expanded
            self.debug_scroll = 0
        elif key == pygame.K_PAGEUP:
            self.debug_scroll = max(0, self.debug_scroll - 20)
        elif key == pygame.K_PAGEDOWN:
            self.debug_scroll += 20
        elif key == pygame.K_HOME:
            self.debug_scroll = 0
        elif key == pygame.K_END:
            self.debug_scroll = 10**9
        elif key == pygame.K_TAB:
            if self.debug_view in ("layer1", "layer2"):
                self.prompt_editing = True
                self.prompt_cursor = len(self._active_prompt())
        if key == pygame.K_1:
            self.tool = "start"
        elif key == pygame.K_2:
            self.tool = "goal"
        elif key == pygame.K_3:
            self.tool = "human"
        elif key == pygame.K_4:
            self.tool = "relation"
            self.pending_relation = None
        elif key in (pygame.K_BACKSPACE, pygame.K_DELETE, pygame.K_d):
            self.tool = "delete"
        elif key == pygame.K_SPACE:
            self._cycle_cost_view()
        elif key == pygame.K_LEFTBRACKET:
            self.cfg.astar_human_block_radius = max(0.0, self.cfg.astar_human_block_radius - 0.05)
            self._mark_scene_changed(invalidate_llm=False)
            self._log(f"human hard-block radius -> {self.cfg.astar_human_block_radius:.2f}m")
        elif key == pygame.K_RIGHTBRACKET:
            self.cfg.astar_human_block_radius = min(2.0, self.cfg.astar_human_block_radius + 0.05)
            self._mark_scene_changed(invalidate_llm=False)
            self._log(f"human hard-block radius -> {self.cfg.astar_human_block_radius:.2f}m")
        elif key == pygame.K_r:
            self.dirty = True
        elif key == pygame.K_c:
            self._cycle_selected_state()
        elif key == pygame.K_e:
            self._export_scene_graph(self.cfg.export_scene_graph)
        elif key == pygame.K_b:
            self.show_rgb = not self.show_rgb
            self._base_map_cache_key = None
            self._nav_mask_cache_key = None
            self._cost_overlay_cache_key = None
            self._log(f"background: {'RGB' if self.show_rgb else 'occupancy only'}")
        return True

    def _on_prompt_edit_key(self, event) -> bool:
        key = event.key
        if self._is_clipboard_shortcut(event, pygame.K_c):
            self._copy_debug_to_clipboard()
            return True
        if self._is_clipboard_shortcut(event, pygame.K_v):
            text = self._get_clipboard_text()
            if text:
                self._insert_prompt_text(text)
                self._log(f"pasted {len(text)} chars into {self.debug_view} prompt")
            return True

        if key == pygame.K_ESCAPE:
            self.prompt_editing = False
            return True
        if key == pygame.K_TAB:
            self.prompt_editing = False
            self._log(f"{self.debug_view} prompt updated; press L to refresh LLM.")
            self.llm_dirty = True
            return True
        if key == pygame.K_BACKSPACE:
            prompt = self._active_prompt()
            if self.prompt_cursor > 0:
                prompt = prompt[: self.prompt_cursor - 1] + prompt[self.prompt_cursor :]
                self.prompt_cursor -= 1
                self._set_active_prompt(prompt)
            return True
        if key == pygame.K_DELETE:
            prompt = self._active_prompt()
            if self.prompt_cursor < len(prompt):
                prompt = prompt[: self.prompt_cursor] + prompt[self.prompt_cursor + 1 :]
                self._set_active_prompt(prompt)
            return True
        if key == pygame.K_RETURN:
            self._insert_prompt_text("\n")
            return True
        if key == pygame.K_LEFT:
            self.prompt_cursor = max(0, self.prompt_cursor - 1)
            return True
        if key == pygame.K_RIGHT:
            self.prompt_cursor = min(len(self._active_prompt()), self.prompt_cursor + 1)
            return True
        if key == pygame.K_HOME:
            prompt = self._active_prompt()
            prev_nl = prompt.rfind("\n", 0, self.prompt_cursor)
            self.prompt_cursor = 0 if prev_nl < 0 else prev_nl + 1
            return True
        if key == pygame.K_END:
            prompt = self._active_prompt()
            next_nl = prompt.find("\n", self.prompt_cursor)
            self.prompt_cursor = len(prompt) if next_nl < 0 else next_nl
            return True
        return True

    def _on_text_input(self, text: str) -> None:
        if not self.prompt_editing:
            return
        self._insert_prompt_text(text)

    def _is_clipboard_shortcut(self, event, key: int) -> bool:
        mods = getattr(event, "mod", pygame.key.get_mods())
        return event.key == key and bool(mods & (pygame.KMOD_CTRL | pygame.KMOD_META))

    def _copy_debug_to_clipboard(self) -> None:
        text = self._debug_source_text()
        if not text:
            return
        if self._set_clipboard_text(text):
            self.status = f"copied {len(text)} chars from {self.debug_view} to clipboard"
        else:
            self.status = "clipboard copy failed"
        self._log(self.status)

    def _debug_source_text(self) -> str:
        if self.debug_view == "log":
            return "\n".join(self.debug_log_lines)
        return self._active_prompt()

    def _set_clipboard_text(self, text: str) -> bool:
        if self.clipboard_ready:
            try:
                pygame.scrap.put(pygame.SCRAP_TEXT, text.encode("utf-8"))
                return True
            except Exception:
                pass
        if platform.system() == "Darwin":
            try:
                subprocess.run(["pbcopy"], input=text, text=True, check=True)
                return True
            except Exception:
                pass
        return False

    def _get_clipboard_text(self) -> str:
        if self.clipboard_ready:
            try:
                raw = pygame.scrap.get(pygame.SCRAP_TEXT)
                if raw:
                    return raw.decode("utf-8", errors="replace").rstrip("\x00")
            except Exception:
                pass
        if platform.system() == "Darwin":
            try:
                result = subprocess.run(["pbpaste"], text=True, capture_output=True, check=True)
                return result.stdout
            except Exception:
                pass
        return ""

    def _insert_prompt_text(self, text: str) -> None:
        prompt = self._active_prompt()
        prompt = prompt[: self.prompt_cursor] + text + prompt[self.prompt_cursor :]
        self.prompt_cursor += len(text)
        self._set_active_prompt(prompt)

    def _active_prompt(self) -> str:
        return self.layer2_prompt if self.debug_view == "layer2" else self.layer1_prompt

    def _set_active_prompt(self, text: str) -> None:
        if self.debug_view == "layer2":
            self.layer2_prompt = text
        else:
            self.layer1_prompt = text

    def _prompt_templates(self) -> LLMPromptTemplates:
        return LLMPromptTemplates(
            layer1_system=self.layer1_prompt,
            layer2_system=self.layer2_prompt,
        )

    def _on_mouse_down(self, event) -> None:
        if event.button != 1:
            return
        if event.pos[0] >= self._map_rect().right:
            if self._llm_button_rect().collidepoint(event.pos):
                self._start_llm_refresh()
            elif self._cost_button_rect().collidepoint(event.pos):
                self._cycle_cost_view()
            return
        xy = self._screen_to_world(event.pos)
        if xy is None:
            return
        rc = self._screen_to_grid(event.pos)
        if rc is None:
            return

        hit = self._hit_human(event.pos)
        if self.tool == "start":
            if not self._is_valid_nav_grid(*rc):
                self.status = "invalid start: click inside navigable floor"
                self._log(self.status)
                return
            self.start_xy = xy.astype(np.float32)
            self.planned_start_xy = self.start_xy.copy()
            self._mark_scene_changed(invalidate_llm=False)
        elif self.tool == "goal":
            if not self._is_valid_nav_grid(*rc):
                self.status = "invalid goal: click inside navigable floor"
                self._log(self.status)
                return
            self.goal_xy = xy.astype(np.float32)
            self.planned_goal_xy = self.goal_xy.copy()
            self._mark_scene_changed(invalidate_llm=False)
        elif self.tool == "human":
            if hit is not None:
                self.selected_human = hit
            else:
                yaw = 0.0
                if len(self.humans) > 0:
                    prev = self.humans[-1].pos
                    yaw = math.degrees(math.atan2(float(prev[1] - xy[1]), float(prev[0] - xy[0])))
                self.humans.append(Human(pos=xy.astype(np.float32), yaw_deg=yaw, state="talking"))
                self.selected_human = len(self.humans) - 1
                self._mark_scene_changed()
        elif self.tool == "relation":
            if hit is None:
                return
            if self.pending_relation is None:
                self.pending_relation = hit
                self.selected_human = hit
            elif self.pending_relation != hit:
                self._toggle_group(self.pending_relation, hit)
                self.pending_relation = None
                self._mark_scene_changed()
        elif self.tool == "delete":
            if hit is not None:
                self._delete_human(hit)
                self._mark_scene_changed()

    def _on_mouse_motion(self, event) -> None:
        if self.tool != "human" or self.selected_human is None:
            return
        if not pygame.mouse.get_pressed(num_buttons=3)[0]:
            return
        xy = self._screen_to_world(event.pos)
        if xy is None:
            return

        self.humans[self.selected_human].pos = xy.astype(np.float32)

        now = pygame.time.get_ticks()
        self._drag_dirty_pending = True
        if now - self._drag_last_recompute_ms >= self._drag_recompute_interval_ms:
            self._drag_last_recompute_ms = now
            self._drag_dirty_pending = False
            self._mark_scene_changed()

    def _on_mouse_up(self, event) -> None:
        if event.button != 1:
            return
        if self._drag_dirty_pending:
            self._drag_dirty_pending = False
            self._mark_scene_changed()

    def _on_mouse_wheel(self, event) -> None:
        if self.console_expanded or pygame.mouse.get_pos()[0] >= self._map_rect().right:
            self.debug_scroll = max(0, self.debug_scroll - event.y * 4)
            return
        if self.selected_human is None:
            return
        self.humans[self.selected_human].yaw_deg = (
            self.humans[self.selected_human].yaw_deg + event.y * 10.0
        ) % 360.0
        self._mark_scene_changed()

    def _cycle_selected_state(self) -> None:
        if self.selected_human is None:
            return
        h = self.humans[self.selected_human]
        idx = STATE_ORDER.index(h.state) if h.state in STATE_ORDER else 0
        h.state = STATE_ORDER[(idx + 1) % len(STATE_ORDER)]
        self._mark_scene_changed()

    def _mark_scene_changed(self, invalidate_llm: bool = True) -> None:
        if invalidate_llm:
            self.social_params = None
            self.llm_dirty = True
        self.dirty = True

    def _cycle_cost_view(self) -> None:
        order = ["current", "delta", "off"]
        idx = order.index(self.cost_view) if self.cost_view in order else 0
        self.cost_view = order[(idx + 1) % len(order)]
        self.show_costmap = self.cost_view != "off"
        self._cost_overlay_cache_key = None
        self._cost_overlay_cache_surface = None
        self._log(f"cost view -> {self.cost_view}")

    def _log(self, message: str) -> None:
        for line in str(message).splitlines() or [""]:
            self.debug_log_lines.append(line)
        self.debug_log_lines = self.debug_log_lines[-5000:]

    def _toggle_group(self, a: int, b: int) -> None:
        pair = tuple(sorted((a, b)))
        if pair in self.groups:
            self.groups.remove(pair)
        else:
            self.groups.append(pair)

    def _delete_human(self, idx: int) -> None:
        self.humans.pop(idx)
        new_groups = []
        for a, b in self.groups:
            if idx in (a, b):
                continue
            na = a - 1 if a > idx else a
            nb = b - 1 if b > idx else b
            new_groups.append(tuple(sorted((na, nb))))
        self.groups = new_groups
        self.selected_human = None
        self.pending_relation = None

    def _recompute(self) -> None:
        if self.cfg.isolate_scene_component:
            self._scene_component_id = self._choose_scene_component()
        scene = self._scene_description()
        self.social_costmap = None
        if self.humans:
            params = self.social_params
            if self.cfg.social_method == "rule" or params is None:
                params, _ = build_entity_params(
                    scene,
                    method="rule",
                    llm_model="rule",
                    verbose=False,
                    robot_pos=tuple(self.start_xy),
                    robot_goal=tuple(self.goal_xy),
                )
            self.social_costmap = _synthesize_costmap_on_scene_grid(
                params,
                self.scene_map,
                self.grid_free_astar.shape,
                self.downscale,
                distance_transform=self.dist_transform,
                clearance_cap=0.5,
                clearance_weight=0.3,
            )

        try:
            grid = self._blocked_grid()
            start_xy = _nearest_free_xy(self.scene_map, grid, self.start_xy, self.downscale)
            goal_xy = _nearest_free_xy(self.scene_map, grid, self.goal_xy, self.downscale)
            if start_xy is None or goal_xy is None:
                raise RuntimeError("start/goal cannot be snapped to free space")
            self.planned_start_xy = start_xy.astype(np.float32)
            self.planned_goal_xy = goal_xy.astype(np.float32)
            raw = self._plan_path(
                self.scene_map,
                grid,
                self.downscale,
                start_xy,
                goal_xy,
                self.social_costmap,
                self.dist_transform,
            )
            if raw is not None and self.cfg.astar_smoothing == "shortcut":
                self.path = _shortcut_path(
                    raw,
                    self.scene_map,
                    grid,
                    self.downscale,
                    self.social_costmap,
                    self.cfg.astar_shortcut_social_threshold,
                )
            else:
                self.path = raw
            self.status = f"path={len(self.path) if self.path is not None else 0} humans={len(self.humans)} groups={len(self.groups)}"
        except Exception as exc:
            self.path = None
            self.planned_start_xy = self.start_xy.copy()
            self.planned_goal_xy = self.goal_xy.copy()
            self.status = f"planning failed: {exc}"
            self._log(self.status)
            self._log(traceback.format_exc())

        self._costmap_version += 1
        self._cost_overlay_cache_key = None
        self._cost_overlay_cache_surface = None
        self.dirty = False

    def _start_llm_refresh(self) -> None:
        if self.llm_loading:
            return
        if self.cfg.social_method != "llm":
            self.status = "LLM disabled; start with --social-method llm"
            self._log(self.status)
            return
        if not self.humans:
            self.status = "no humans for LLM social params"
            self._log(self.status)
            return
        self.prompt_editing = False
        self.llm_loading = True
        self.status = f"calling LLM: {self.cfg.llm_model}"
        self._log(self.status)

        scene = self._scene_description()
        robot_pos = tuple(self.start_xy)
        robot_goal = tuple(self.goal_xy)
        llm_model = self.cfg.llm_model
        templates = self._prompt_templates()

        def worker() -> None:
            try:
                params, log = build_entity_params(
                    scene,
                    method="llm",
                    llm_model=llm_model,
                    verbose=True,
                    robot_pos=robot_pos,
                    robot_goal=robot_goal,
                    prompt_templates=templates,
                )
                result: tuple[str, object, str] = ("ok", params, log)
            except Exception as exc:
                result = ("err", exc, traceback.format_exc())
            with self._llm_lock:
                self._llm_pending_result = result

        threading.Thread(target=worker, daemon=True).start()

    def _poll_llm_result(self) -> None:
        with self._llm_lock:
            result = self._llm_pending_result
            self._llm_pending_result = None
        if result is None:
            return
        kind, payload, log = result
        self.llm_loading = False
        if kind == "ok":
            self.social_params = payload
            self.llm_log = log
            self.llm_dirty = False
            self.status = f"LLM params updated: {len(self.social_params)} entities"
            if self.social_costmap is not None:
                self.prev_social_costmap = self.social_costmap.copy()
            self._log(self.status)
            self._log(log)
            self.dirty = True
        else:
            self.status = f"LLM failed: {payload}"
            self._log(self.status)
            self._log(log)
            print(f"[stage1-playground] LLM failed: {payload}")

    def _refresh_llm(self) -> None:
        self._start_llm_refresh()

    def _refresh_llm_sync(self) -> None:
        try:
            params, log = build_entity_params(
                self._scene_description(),
                method="llm",
                llm_model=self.cfg.llm_model,
                verbose=True,
                robot_pos=tuple(self.start_xy),
                robot_goal=tuple(self.goal_xy),
                prompt_templates=self._prompt_templates(),
            )
            self.social_params = params
            self.llm_log = log
            self.llm_dirty = False
            self.status = f"LLM params updated: {len(params)} entities"
            if self.social_costmap is not None:
                self.prev_social_costmap = self.social_costmap.copy()
            self._log(self.status)
            self._log(log)
            self.dirty = True
        except Exception as exc:
            self.status = f"LLM failed: {exc}"
            self._log(self.status)
            self._log(traceback.format_exc())
            print(f"[stage1-playground] LLM failed: {exc}")

    def _plan_path(
        self,
        scene_map: BirdviewSceneMap,
        grid: np.ndarray,
        downscale: int,
        start_xy: np.ndarray,
        goal_xy: np.ndarray,
        social_costmap: np.ndarray | None,
        distance_transform: np.ndarray,
    ) -> np.ndarray | None:
        k = max(int(self.cfg.astar_num_candidates), 1)
        penalty = np.zeros(grid.shape, dtype=np.float32)
        best_score = math.inf
        best_path = None
        for idx in range(k):
            path = _astar_on_grid(
                scene_map,
                grid,
                downscale,
                start_xy,
                goal_xy,
                social_costmap,
                self.cfg.astar_social_weight,
                distance_transform,
                self.grid_spacing,
                extra_costmap=penalty if idx > 0 else None,
            )
            if path is None:
                continue
            length = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))) if len(path) > 1 else 0.0
            social = 0.0
            clearance = 0.0
            for xy in path:
                r, c = _world_to_grid(scene_map, xy, downscale, grid.shape)
                if social_costmap is not None:
                    social += float(social_costmap[r, c])
                clearance += max(0.0, 0.5 - float(distance_transform[r, c]) * self.grid_spacing)
            score = length + social * self.cfg.astar_social_weight * 0.02 + clearance * self.cfg.astar_candidate_clearance_weight
            if score < best_score:
                best_score = score
                best_path = path
            penalty = np.maximum(
                penalty,
                _path_penalty(path, scene_map, grid.shape, downscale) * self.cfg.astar_diversity_penalty,
            )
        return best_path

    def _blocked_grid(self) -> np.ndarray:
        grid = self.grid_free_astar.copy()

        if self.cfg.isolate_scene_component and self._scene_component_id is not None:
            grid &= self._component_labels == self._scene_component_id

        if not self.humans or self.cfg.astar_human_block_radius <= 0.0:
            return grid

        radius_cells = max(
            int(math.ceil(self.cfg.astar_human_block_radius / max(self.grid_spacing, 1e-6))),
            1,
        )

        H, W = grid.shape
        rr_base, cc_base = np.ogrid[-radius_cells : radius_cells + 1, -radius_cells : radius_cells + 1]
        mask = (rr_base * rr_base + cc_base * cc_base) <= radius_cells * radius_cells

        for h in self.humans:
            r, c = self._world_to_grid(h.pos)

            r0 = max(0, r - radius_cells)
            r1 = min(H, r + radius_cells + 1)
            c0 = max(0, c - radius_cells)
            c1 = min(W, c + radius_cells + 1)

            mr0 = r0 - (r - radius_cells)
            mr1 = mr0 + (r1 - r0)
            mc0 = c0 - (c - radius_cells)
            mc1 = mc0 + (c1 - c0)

            view = grid[r0:r1, c0:c1]
            view[mask[mr0:mr1, mc0:mc1]] = False

        return grid

    def _choose_scene_component(self) -> int | None:
        if self._num_components <= 0:
            return None

        votes: dict[int, int] = {}
        for xy in [self.start_xy, self.goal_xy, *[h.pos for h in self.humans]]:
            r, c = self._world_to_grid(xy)
            comp = int(self._component_labels[r, c])
            if comp > 0:
                votes[comp] = votes.get(comp, 0) + 1

        if votes:
            return max(votes.items(), key=lambda item: item[1])[0]

        sizes = np.bincount(self._component_labels.ravel())
        if len(sizes) <= 1:
            return None
        sizes[0] = 0
        return int(np.argmax(sizes))

    def _scene_description(self) -> SceneDescription:
        humans = [
            HumanInfo(
                pos=(float(h.pos[0]), float(h.pos[1])),
                yaw_deg=float(h.yaw_deg),
                activities=h.activities,
                state=h.state,
            )
            for h in self.humans
        ]
        return SceneDescription(humans=humans, obstacles=[], groups=[list(g) for g in self.groups])

    def _load_scene_graph(self, path: Path) -> None:
        scene = scene_graph_to_scene_description(path)
        self.humans = [
            Human(
                pos=np.array(h.pos, dtype=np.float32),
                yaw_deg=float(h.yaw_deg),
                state=getattr(h, "state", "talking"),
            )
            for h in scene.humans
        ]
        self.groups = [tuple(sorted((int(g[0]), int(g[1])))) for g in scene.groups if len(g) >= 2]
        self.dirty = True

    def _export_scene_graph(self, path: Path) -> None:
        nodes = []
        for idx, h in enumerate(self.humans):
            nodes.append(
                {
                    "node_id": idx,
                    "label": ["person"],
                    "bbox_center": [float(h.pos[0]), float(h.pos[1]), 0.0],
                    "bbox_extent": [0.5, 0.3, 1.7],
                    "activities": [{"name": a, "object": ""} for a in h.activities],
                    "heading_deg": float(h.yaw_deg),
                    "state": h.state,
                }
            )
        edges = [
            [int(a), int(b), {"relation": "conversation", "desc": "talking_to"}]
            for a, b in self.groups
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"nodes": nodes, "edges": edges}, indent=2))
        self.status = f"exported {path}"
        self._log(self.status)
        print(f"[stage1-playground] exported scene graph -> {path}")

    def _map_rect(self) -> pygame.Rect:
        return self.transform.map_rect(self.screen.get_size())

    def _screen_to_grid(self, pos: tuple[int, int]) -> tuple[int, int] | None:
        return self.transform.screen_to_grid(pos, self.screen.get_size())

    def _screen_to_world(self, pos: tuple[int, int]) -> np.ndarray | None:
        return self.transform.screen_to_world(pos, self.screen.get_size())

    def _world_to_screen(self, xy: np.ndarray) -> tuple[int, int]:
        return self.transform.world_to_screen(xy, self.screen.get_size())

    def _world_to_grid(self, xy: np.ndarray) -> tuple[int, int]:
        return self.transform.world_to_grid(xy)

    def _is_valid_nav_xy(self, xy: np.ndarray) -> bool:
        r, c = self._world_to_grid(xy)
        return self._is_valid_nav_grid(r, c)

    def _is_valid_nav_grid(self, r: int, c: int) -> bool:
        if not bool(self.grid_free_astar[r, c]):
            return False
        if (
            self.cfg.isolate_scene_component
            and self._scene_component_id is not None
            and self._component_labels.shape == self.grid_free_astar.shape
        ):
            return int(self._component_labels[r, c]) == int(self._scene_component_id)
        return True

    def _hit_human(self, pos: tuple[int, int]) -> int | None:
        best = None
        best_d = 1e9
        for i, h in enumerate(self.humans):
            sx, sy = self._world_to_screen(h.pos)
            d = math.hypot(pos[0] - sx, pos[1] - sy)
            if d < best_d and d <= self.HUMAN_R_PX + 5:
                best = i
                best_d = d
        return best

    def _draw(self) -> None:
        self.screen.fill((19, 20, 31))
        rect = self._map_rect()
        self._draw_map(rect)
        self._draw_path()
        self._draw_humans()
        self._draw_start_goal()
        self._draw_toolbar()
        self._draw_panel()
        if self.console_expanded:
            self._draw_console_overlay()

    def _draw_map(self, rect: pygame.Rect) -> None:
        pygame.draw.rect(self.screen, (41, 43, 54), rect)
        floor = self._get_scaled_floor_surface(rect)
        self.screen.blit(floor, rect)
        if self.show_rgb and self.cfg.background in ("rgb", "scene-graph") and self.display_surface is not None:
            base = self._get_scaled_base_map(rect)
            self.screen.blit(base, rect)
        nav_overlay = self._get_scaled_nav_mask_overlay(rect)
        self.screen.blit(nav_overlay, rect)

        if self.cost_view != "off" and self.social_costmap is not None:
            overlay = self._get_scaled_cost_overlay(rect)
            if overlay is not None:
                self.screen.blit(overlay, rect)
        pygame.draw.rect(self.screen, (82, 86, 104), rect, 1)

    def _get_scaled_base_map(self, rect: pygame.Rect) -> pygame.Surface:
        if self.display_surface is not None:
            src = self.display_surface
        else:
            if self._blank_map_surface is None:
                rgb = np.zeros((*self.grid_free.shape, 3), dtype=np.uint8)
                rgb[:] = np.array([250, 250, 247], dtype=np.uint8)
                self._blank_map_surface = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
            src = self._blank_map_surface

        key = (id(src), rect.width, rect.height)
        if self._base_map_cache_key != key or self._base_map_cache_surface is None:
            self._base_map_cache_surface = pygame.transform.smoothscale(src, (rect.width, rect.height))
            self._base_map_cache_key = key

        return self._base_map_cache_surface

    def _navigable_display_mask(self) -> np.ndarray:
        mask = self.grid_free.astype(bool)
        if (
            self.cfg.isolate_scene_component
            and self._scene_component_id is not None
            and self._component_labels.shape == mask.shape
        ):
            mask = mask & (self._component_labels == self._scene_component_id)
        return mask

    def _get_scaled_nav_mask_overlay(self, rect: pygame.Rect) -> pygame.Surface:
        # When showing a real RGB background, use a semi-transparent obstacle tint so
        # furniture and objects are still visible through the overlay.
        rgb_mode = (self.show_rgb and self.cfg.background == "rgb"
                    and getattr(self.scene_map, "rgb_array", None) is not None)
        obstacle_alpha = 160 if rgb_mode else 255

        key = (
            int(self._scene_component_id or 0),
            rect.width,
            rect.height,
            obstacle_alpha,
        )
        if self._nav_mask_cache_key == key and self._nav_mask_cache_surface is not None:
            return self._nav_mask_cache_surface

        mask = self._navigable_display_mask()
        rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
        rgba[~mask] = np.array([20, 22, 30, obstacle_alpha], dtype=np.uint8)
        if rgb_mode:
            rgba[mask] = np.array([210, 220, 255, 28], dtype=np.uint8)
        raw = pygame.image.frombuffer(
            rgba.copy(order="C").tobytes(),
            (rgba.shape[1], rgba.shape[0]),
            "RGBA",
        )
        self._nav_mask_cache_surface = pygame.transform.scale(raw, (rect.width, rect.height))
        self._nav_mask_cache_key = key
        return self._nav_mask_cache_surface

    def _get_scaled_floor_surface(self, rect: pygame.Rect) -> pygame.Surface:
        key = (
            int(self._scene_component_id or 0),
            rect.width,
            rect.height,
        )
        if self._floor_cache_key == key and self._floor_cache_surface is not None:
            return self._floor_cache_surface

        mask = self._navigable_display_mask()
        rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
        rgba[mask] = np.array([238, 237, 229, 255], dtype=np.uint8)

        # Subtle grid on navigable cells only, so the true board footprint is visible.
        step = max(int(round(1.0 / max(self.grid_spacing, 1e-6))), 1)
        rgba[::step, :, :3] = np.where(mask[::step, :, None], np.array([221, 220, 211], dtype=np.uint8), rgba[::step, :, :3])
        rgba[:, ::step, :3] = np.where(mask[:, ::step, None], np.array([221, 220, 211], dtype=np.uint8), rgba[:, ::step, :3])

        raw = pygame.image.frombuffer(
            rgba.copy(order="C").tobytes(),
            (rgba.shape[1], rgba.shape[0]),
            "RGBA",
        )
        self._floor_cache_surface = pygame.transform.scale(raw, (rect.width, rect.height))
        self._floor_cache_key = key
        return self._floor_cache_surface

    def _get_scaled_cost_overlay(self, rect: pygame.Rect) -> pygame.Surface | None:
        if self.social_costmap is None:
            return None

        key = (
            self._costmap_version,
            int(self._scene_component_id or 0),
            rect.width,
            rect.height,
            self.cost_view,
        )
        if self._cost_overlay_cache_key == key and self._cost_overlay_cache_surface is not None:
            return self._cost_overlay_cache_surface

        if self.cost_view == "delta":
            if self.prev_social_costmap is None:
                return None
            cm = np.clip(self.social_costmap - self.prev_social_costmap, -1.0, 1.0)
            mag = np.abs(cm)
            rgba = np.zeros((*cm.shape, 4), dtype=np.uint8)
            rgba[..., 0] = np.where(cm > 0, 255, np.where(cm < 0, 45, 0)).astype(np.uint8)
            rgba[..., 1] = np.where(cm > 0, 70, np.where(cm < 0, 135, 0)).astype(np.uint8)
            rgba[..., 2] = np.where(cm > 0, 45, np.where(cm < 0, 255, 0)).astype(np.uint8)
            rgba[..., 3] = np.clip(mag * 220, 0, 190).astype(np.uint8)
        else:
            cm = np.clip(self.social_costmap, 0.0, 1.0)
            rgba = np.zeros((*cm.shape, 4), dtype=np.uint8)
            rgba[..., 0] = 255
            rgba[..., 1] = np.clip(235 - 175 * cm, 0, 255).astype(np.uint8)
            rgba[..., 2] = np.clip(190 - 160 * cm, 0, 255).astype(np.uint8)
            rgba[..., 3] = np.clip(cm * 150, 0, 150).astype(np.uint8)

        mask = self._navigable_display_mask()
        if mask.shape == rgba.shape[:2]:
            rgba[..., 3] = np.where(mask, rgba[..., 3], 0).astype(np.uint8)

        raw = pygame.image.frombuffer(
            rgba.copy(order="C").tobytes(),
            (rgba.shape[1], rgba.shape[0]),
            "RGBA",
        )
        self._cost_overlay_cache_surface = pygame.transform.smoothscale(raw, (rect.width, rect.height))
        self._cost_overlay_cache_key = key
        return self._cost_overlay_cache_surface

    def _draw_path(self) -> None:
        if self.path is None or len(self.path) < 2:
            return
        pts = [self._world_to_screen(p) for p in self.path]
        pygame.draw.lines(self.screen, (35, 110, 210), False, pts, 4)
        pygame.draw.lines(self.screen, (230, 245, 255), False, pts, 1)

    def _draw_humans(self) -> None:
        colors = {
            "idle": (255, 160, 70),
            "talking": (230, 110, 220),
            "sitting": (150, 120, 230),
            "walking": (60, 210, 230),
        }
        for a, b in self.groups:
            if a < len(self.humans) and b < len(self.humans):
                pygame.draw.line(self.screen, (210, 60, 50), self._world_to_screen(self.humans[a].pos), self._world_to_screen(self.humans[b].pos), 2)

        for i, h in enumerate(self.humans):
            p = self._world_to_screen(h.pos)
            col = colors.get(h.state, (220, 220, 220))
            rect = self._map_rect()
            grid_px = rect.width / max(self.grid_free_astar.shape[1], 1)
            ps = max(12, int((0.8 / max(self.grid_spacing, 1e-6)) * grid_px))
            pygame.draw.circle(self.screen, (*col, 35), p, ps, 1)
            pygame.draw.circle(self.screen, col, p, self.HUMAN_R_PX)
            if i == self.selected_human or i == self.pending_relation:
                pygame.draw.circle(self.screen, (255, 255, 255), p, self.HUMAN_R_PX + 4, 2)
            yaw = math.radians(h.yaw_deg)
            tip = (int(p[0] + math.cos(yaw) * 24), int(p[1] - math.sin(yaw) * 24))
            pygame.draw.line(self.screen, (255, 255, 255), p, tip, 2)
            self._draw_text(f"P{i}:{h.state}", (p[0] + 10, p[1] - 10), (255, 255, 255))

    def _draw_start_goal(self) -> None:
        s = self._world_to_screen(self.planned_start_xy)
        g = self._world_to_screen(self.planned_goal_xy)
        pygame.draw.circle(self.screen, (45, 170, 80), s, 9)
        pygame.draw.circle(self.screen, (215, 55, 45), g, 10)
        pygame.draw.circle(self.screen, (255, 255, 255), s, 9, 2)
        pygame.draw.circle(self.screen, (255, 255, 255), g, 10, 2)
        self._draw_text("S", (s[0] + 10, s[1] - 8), (255, 255, 255))
        self._draw_text("G", (g[0] + 10, g[1] - 8), (255, 255, 255))

    def _draw_toolbar(self) -> None:
        pygame.draw.rect(self.screen, (28, 31, 38), (0, 0, self.screen.get_width(), self.TOOLBAR_H))
        items = [("1", "start"), ("2", "goal"), ("3", "human"), ("4", "relation"), ("D", "delete")]
        x = 12
        for key, name in items:
            active = self.tool == name
            color = (62, 85, 125) if active else (42, 46, 56)
            rect = pygame.Rect(x, 8, 94, 26)
            pygame.draw.rect(self.screen, color, rect, border_radius=4)
            self._draw_text(f"{key} {name}", (x + 8, 14), (235, 238, 245))
            x += 104
        self._draw_text(
            "Space costmap  C state  Wheel yaw  E export  R replan  F4/` console  Cmd/Ctrl+C copy",
            (x + 8, 14),
            (210, 215, 225),
        )

    def _llm_button_rect(self) -> pygame.Rect:
        w, _ = self.screen.get_size()
        x0 = w - self.PANEL_W
        y = self.TOOLBAR_H + 14 + len(self._panel_status_lines()) * 21 + 4
        return pygame.Rect(x0 + 14, y, self.PANEL_W - 28, 30)

    def _cost_button_rect(self) -> pygame.Rect:
        btn = self._llm_button_rect()
        return pygame.Rect(btn.x, btn.bottom + 8, btn.width, 28)

    def _cost_delta_summary(self) -> str:
        if self.social_costmap is None or self.prev_social_costmap is None:
            return "diff: none"
        delta = self.social_costmap - self.prev_social_costmap
        return (
            f"diff max:{float(np.max(np.abs(delta))):.2f} "
            f"mean:{float(np.mean(np.abs(delta))):.2f}"
        )

    def _panel_status_lines(self) -> list[str]:
        return [
            "Stage1 Playground",
            f"tool: {self.tool}",
            f"humans: {len(self.humans)}",
            f"groups: {len(self.groups)}",
            f"component: {self._scene_component_id}",
            f"social: {self.cfg.social_method}",
            f"llm model: {self.cfg.llm_model}",
            f"llm params: {'loading' if self.llm_loading else ('stale' if self.llm_dirty and self.cfg.social_method == 'llm' else ('yes' if self.social_params is not None else 'no'))}",
            f"cost view: {self.cost_view}",
            self._cost_delta_summary(),
            f"start: {self.start_xy[0]:.2f}, {self.start_xy[1]:.2f}",
            f"goal: {self.goal_xy[0]:.2f}, {self.goal_xy[1]:.2f}",
            f"block r: {self.cfg.astar_human_block_radius:.2f}m",
            "",
            self.status,
        ]

    def _panel_help_lines(self) -> list[str]:
        return [
            "1 start  2 goal  3 human",
            "4 relation  D delete",
            "E export",
            "[ ] block radius",
            "Space cost view",
            "F1 log  F2 L1  F3 L2",
            "F4/` expand console",
            "Cmd/Ctrl+C copy",
            "Cmd/Ctrl+V paste prompt",
            "Tab edit prompt",
        ]

    def _panel_lines(self) -> list[str]:
        return self._panel_status_lines() + [""] + self._panel_help_lines()

    def _draw_panel(self) -> None:
        w, h = self.screen.get_size()
        x0 = w - self.PANEL_W
        pygame.draw.rect(self.screen, (250, 250, 246), (x0, self.TOOLBAR_H, self.PANEL_W, h - self.TOOLBAR_H))
        pygame.draw.line(self.screen, (190, 190, 185), (x0, self.TOOLBAR_H), (x0, h), 1)
        lines = self._panel_status_lines()
        y = self.TOOLBAR_H + 14
        for i, line in enumerate(lines):
            font = self.font_bold if i == 0 else self.font
            color = (32, 36, 44) if line else (120, 120, 120)
            self._draw_text(line, (x0 + 14, y), color, font=font)
            y += 21

        btn = self._llm_button_rect()
        disabled = self.llm_loading or self.cfg.social_method != "llm" or not self.humans
        if self.llm_loading:
            label = "Loading LLM..."
        elif self.cfg.social_method != "llm":
            label = "LLM disabled"
        elif not self.humans:
            label = "Add humans first"
        else:
            label = "Refresh LLM Cost"
        color = (86, 111, 155) if not disabled else (162, 166, 174)
        pygame.draw.rect(self.screen, color, btn, border_radius=4)
        pygame.draw.rect(self.screen, (96, 100, 108), btn, 1, border_radius=4)
        self._draw_text(label, (btn.x + 12, btn.y + 8), (250, 252, 255), font=self.font_bold)

        cost_btn = self._cost_button_rect()
        pygame.draw.rect(self.screen, (56, 61, 72), cost_btn, border_radius=4)
        pygame.draw.rect(self.screen, (112, 116, 126), cost_btn, 1, border_radius=4)
        self._draw_text(f"Cost View: {self.cost_view}", (cost_btn.x + 12, cost_btn.y + 7), (245, 247, 250), font=self.font_bold)

        debug_min_h = 110
        help_y = cost_btn.bottom + 10
        help_limit = h - debug_min_h - 16
        for line in self._panel_help_lines():
            if help_y + 18 > help_limit:
                break
            self._draw_text(line, (x0 + 14, help_y), (54, 58, 68), font=self.font)
            help_y += 18

        dbg_top = max(cost_btn.bottom + 10, min(help_y + 8, h - debug_min_h - 8))
        dbg_rect = pygame.Rect(x0 + 10, dbg_top, self.PANEL_W - 20, max(80, h - dbg_top - 12))
        pygame.draw.rect(self.screen, (34, 37, 45), dbg_rect, border_radius=4)
        pygame.draw.rect(self.screen, (160, 164, 172), dbg_rect, 1, border_radius=4)
        title = {
            "log": "Debug / LLM Log",
            "layer1": "Layer1 Prompt",
            "layer2": "Layer2 Prompt",
        }.get(self.debug_view, "Debug")
        edit = " EDIT" if self.prompt_editing else ""
        self._draw_text(
            f"{title}{edit}  F4/`",
            (dbg_rect.x + 8, dbg_rect.y + 8),
            (245, 247, 250),
            font=self.font_bold,
        )
        self._draw_debug_content(dbg_rect.inflate(-16, -34).move(0, 20))

    def _draw_console_overlay(self) -> None:
        w, h = self.screen.get_size()
        shade = pygame.Surface((w, h), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 155))
        self.screen.blit(shade, (0, 0))

        margin = 28
        rect = pygame.Rect(
            margin,
            self.TOOLBAR_H + margin,
            max(240, w - margin * 2),
            max(180, h - self.TOOLBAR_H - margin * 2),
        )
        pygame.draw.rect(self.screen, (24, 27, 34), rect, border_radius=6)
        pygame.draw.rect(self.screen, (130, 136, 150), rect, 1, border_radius=6)

        title = {
            "log": "Console / LLM Log",
            "layer1": "Console / Layer1 Prompt",
            "layer2": "Console / Layer2 Prompt",
        }.get(self.debug_view, "Console")
        edit = " EDIT" if self.prompt_editing else ""
        help_text = (
            "F4/` close   F1/F2/F3 switch   Wheel/PageUp/PageDown/Home/End scroll   "
            "Cmd/Ctrl+C copy"
        )
        self._draw_text(f"{title}{edit}", (rect.x + 16, rect.y + 12), (248, 250, 252), font=self.font_bold)
        self._draw_text(help_text, (rect.x + 16, rect.y + 34), (178, 184, 196), font=self.font)

        content = pygame.Rect(rect.x + 16, rect.y + 58, rect.width - 32, rect.height - 72)
        self._draw_debug_content(content)

    def _draw_text(self, text: str, pos: tuple[int, int], color, font=None) -> None:
        surf = (font or self.font).render(text, True, color)
        self.screen.blit(surf, pos)

    def _draw_debug_content(self, rect: pygame.Rect) -> None:
        old_clip = self.screen.get_clip()
        self.screen.set_clip(rect)
        char_w = max(1, self.font.size("M")[0])
        max_chars = max(24, (rect.width - 8) // char_w)
        if self.debug_view == "log":
            raw_lines = self.debug_log_lines or ["No debug messages yet."]
            display = []
            for line in raw_lines:
                display.extend(self._wrap_line(line, max_chars=max_chars))
        else:
            prompt = self._active_prompt()
            if self.prompt_editing:
                prompt = prompt[: self.prompt_cursor] + "|" + prompt[self.prompt_cursor :]
            help_line = "Tab/Esc finish editing. Cmd/Ctrl+C copy, Cmd/Ctrl+V paste."
            display = [help_line, ""]
            for line in prompt.splitlines() or [""]:
                display.extend(self._wrap_line(line, max_chars=max_chars))

        line_h = 16
        visible = max(1, rect.height // line_h)
        max_scroll = max(0, len(display) - visible)
        self.debug_scroll = min(max(self.debug_scroll, 0), max_scroll)
        start = self.debug_scroll
        y = rect.y
        for line in display[start : start + visible]:
            self._draw_text(line, (rect.x, y), (232, 236, 242), font=self.font)
            y += line_h
        if max_scroll > 0:
            bar_h = max(18, int(rect.height * visible / max(len(display), 1)))
            bar_y = rect.y + int((rect.height - bar_h) * start / max_scroll)
            pygame.draw.rect(self.screen, (74, 80, 94), (rect.right - 4, rect.y, 3, rect.height))
            pygame.draw.rect(self.screen, (174, 184, 204), (rect.right - 4, bar_y, 3, bar_h))
        self.screen.set_clip(old_clip)

    def _wrap_line(self, line: str, max_chars: int) -> list[str]:
        if line == "":
            return [""]
        out: list[str] = []
        cur = ""
        for word in line.replace("\t", "    ").split(" "):
            if len(word) > max_chars:
                if cur:
                    out.append(cur)
                    cur = ""
                out.extend(word[i : i + max_chars] for i in range(0, len(word), max_chars))
                continue
            candidate = word if not cur else f"{cur} {word}"
            if len(candidate) <= max_chars:
                cur = candidate
            else:
                out.append(cur)
                cur = word
        if cur:
            out.append(cur)
        return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pure 2D Stage1 social-navigation playground")
    p.add_argument("--map-source", choices=["scene-xml", "birdview", "scene-graph"], default="birdview")
    p.add_argument(
        "--background",
        choices=["none", "rgb", "occupancy", "scene-graph"],
        default="rgb",
        help="Visual background only. Default shows the colored topdown map under the costmap.",
    )
    p.add_argument(
        "--scene-xml",
        type=Path,
        default=None,
        help="MuJoCo scene XML for real geometry top-down mode.",
    )
    p.add_argument(
        "--layout-json",
        type=Path,
        default=Path("outputs/procthor/train_9/train_9_layout.json"),
        help="Layout JSON used to resolve scene_xml when --scene-xml is omitted.",
    )
    p.add_argument("--birdview-image", type=Path, default=Path("outputs/procthor/train_9/debug_grid_free.png"))
    p.add_argument(
        "--display-image",
        type=Path,
        default=None,
        help="Optional visual background image. Occupancy still comes from --birdview-image.",
    )
    p.add_argument(
        "--map-bounds",
        type=_parse_bounds,
        default=None,
        help="World bounds x_min,x_max,y_min,y_max. Omit or use auto to infer from scene graph.",
    )
    p.add_argument("--birdview-free-threshold", type=float, default=0.5)
    p.add_argument("--birdview-free-is-dark", action="store_true", default=False)
    p.add_argument(
        "--birdview-downscale",
        type=int,
        default=5,
        help="Occupancy downscale factor. Use 5-8 for interactive scene-xml maps; 1 is very slow.",
    )
    p.add_argument("--start-pose", type=_parse_xyz, default=(2.0, 4.0, 0.0))
    p.add_argument("--goal-xy", type=_parse_xy, default=(8.0, 4.0))
    p.add_argument("--scene-graph", type=Path, default=None)
    p.add_argument("--export-scene-graph", type=Path, default=Path("outputs/stage1_playground_scene_graph.json"))
    p.add_argument("--social-method", choices=["rule", "llm"], default="rule")
    p.add_argument("--llm-model", default="moonshot-v1-8k")
    p.add_argument("--astar-social-weight", type=float, default=30.0)
    p.add_argument("--astar-human-block-radius", type=float, default=0.35)
    p.add_argument("--astar-num-candidates", type=int, default=3)
    p.add_argument("--astar-diversity-penalty", type=float, default=8.0)
    p.add_argument("--astar-candidate-clearance-weight", type=float, default=8.0)
    p.add_argument("--astar-smoothing", choices=["none", "shortcut"], default="shortcut")
    p.add_argument("--astar-shortcut-social-threshold", type=float, default=0.45)
    p.add_argument(
        "--no-isolate-scene-component",
        dest="isolate_scene_component",
        action="store_false",
        default=True,
    )
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=820)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PlaygroundConfig(**vars(args))
    Stage1Playground(cfg).run()


if __name__ == "__main__":
    main()
