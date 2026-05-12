"""
topdown_viz.py
==============
Bird's-eye view for social navigation episodes.

Features
--------
- Background: occupancy map (free = white, obstacle = dark)
- Human markers with personal-space circles + facing arrow
- Social costmap overlay (optional, semi-transparent heatmap)
- Robot trajectory (live-updated line + current position dot)
- Start / goal markers

Usage (standalone):
    python experiments/social_nav/topdown_viz.py \
        --scene-xml assets/scenes/ithor/FloorPlan203_physics.xml \
        --layout-json assets/layouts/FloorPlan203_object_positions.json \
        --out outputs/scene_overview.png

Usage (from run_social_nav):
    viz = TopdownViz(scene_map, grid_free, grid_spacing, downscale)
    viz.set_humans(scene.humans)
    viz.set_social_costmap(social_costmap)       # optional
    viz.set_start_goal(start_xy, goal_xy)
    # inside episode loop:
    viz.update(robot_xy)
    # at end:
    viz.save("outputs/episode.png")
    viz.save_gif("outputs/episode.gif")          # requires pillow
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from pathlib import Path
from typing import NamedTuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class HumanVizInfo(NamedTuple):
    pos: tuple[float, float]
    yaw_deg: float
    state: str          # idle | sitting | walking | talking


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _world_to_grid(scene_map, xy: np.ndarray, downscale: int) -> np.ndarray:
    """World (x,y) → downscaled grid (row, col) — vectorised."""
    flat = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
    xyz = np.zeros((flat.shape[0], 3), dtype=np.float32)
    xyz[:, :2] = flat
    px = scene_map.pos_m_to_px(xyz)
    grid = np.floor(px / downscale).astype(np.int32)
    return grid.reshape(*np.asarray(xy).shape[:-1], 2)


def _grid_to_img_xy(row: float, col: float) -> tuple[float, float]:
    """Grid (row, col) → matplotlib axes coords: x = col (east), y = row (north from bottom)."""
    return float(col), float(row)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class TopdownViz:
    """
    Incrementally builds a top-down visualization of one navigation episode.

    Designed to be cheap: the background (occupancy + humans + costmap) is
    rendered once; only the trajectory layer is updated each step.
    """

    # Color palette
    _C_FREE       = np.array([0.97, 0.97, 0.95])   # off-white free space
    _C_OBS        = np.array([0.22, 0.22, 0.25])   # dark obstacle
    _C_ROBOT      = "#1a6fbf"
    _C_TRAJ       = "#1a6fbf"
    _C_START      = "#2ca02c"
    _C_GOAL       = "#d62728"
    _C_HUMAN: dict[str, str] = {
        "idle":    "#ff7f0e",
        "sitting": "#9467bd",
        "walking": "#17becf",
        "talking": "#e377c2",
    }
    _PERSONAL_SPACE_R = 0.8   # metres, for the dashed circle

    def __init__(
        self,
        scene_map,
        grid_free: np.ndarray,
        grid_spacing: float,
        downscale: int,
        figsize: tuple[float, float] = (9, 9),
        dpi: int = 140,
    ) -> None:
        self._scene_map  = scene_map
        self._grid_free  = grid_free          # bool (H, W)
        self._spacing    = float(grid_spacing) # metres per grid cell
        self._downscale  = int(downscale)
        self._figsize    = figsize
        self._dpi        = dpi

        self._humans: list[HumanVizInfo] = []
        self._social_costmap: np.ndarray | None = None
        self._start_xy: np.ndarray | None = None
        self._goal_xy: np.ndarray | None = None
        self._trajectory: list[tuple[float, float]] = []  # world (x, y)

        self._fig: plt.Figure | None = None
        self._ax:  plt.Axes  | None = None
        self._traj_line = None
        self._robot_dot = None
        self._frames: list[np.ndarray] = []    # for GIF export

        self._costmap_im = None       # AxesImage for live costmap overlay
        self._cost_text  = None       # text label showing cost at robot
        self._step: int  = 0

        self._astar_path: np.ndarray | None = None          # (N,2) world xy
        self._rollout_collection = None                      # LineCollection MPPI rollouts
        self._best_traj_line = None                         # best MPPI trajectory
        self._llm_log: str = ""                             # LLM prompt+response text
        self._ax_llm: plt.Axes | None = None               # LLM text panel axes
        self._llm_text_obj = None                          # Text object in LLM panel

        self._H, self._W = grid_free.shape

    # ------------------------------------------------------------------
    # Configuration (call before first update)
    # ------------------------------------------------------------------

    def set_humans(self, humans) -> None:
        """Accept HumanInfo NamedTuples (from llm_costmap or scene_bridge)."""
        self._humans = []
        for h in humans:
            state = getattr(h, "state", "idle")
            yaw = float(getattr(h, "yaw_deg", 0.0))
            pos = tuple(h.pos[:2])
            self._humans.append(HumanVizInfo(pos=pos, yaw_deg=yaw, state=state))

    def set_social_costmap(self, costmap: np.ndarray) -> None:
        """Supply (or live-update) the social costmap overlay."""
        if self._fig is not None:
            self.update_costmap(costmap)
        else:
            self._social_costmap = costmap

    def update_costmap(self, costmap: np.ndarray) -> None:
        """Replace the costmap and refresh the overlay in-place (zero-rebuild)."""
        self._social_costmap = costmap
        if self._costmap_im is not None:
            sc = np.clip(costmap, 0.0, 1.0)
            rgba = plt.cm.YlOrRd(sc)
            rgba[..., 3] = sc * 0.65
            self._costmap_im.set_data(rgba)

    def set_llm_log(self, text: str) -> None:
        """Update the LLM reasoning panel with new prompt+response text."""
        self._llm_log = text
        if self._llm_text_obj is not None:
            self._llm_text_obj.set_text(self._wrap_llm_text(text))
            if self._fig is not None:
                self._fig.canvas.draw_idle()

    @staticmethod
    def _wrap_llm_text(text: str, max_chars: int = 72) -> str:
        """Hard-wrap long lines for the narrow text panel."""
        import textwrap
        lines = []
        for line in text.splitlines():
            if len(line) <= max_chars:
                lines.append(line)
            else:
                lines.extend(textwrap.wrap(line, max_chars) or [""])
        return "\n".join(lines[:80])   # cap at 80 lines

    def set_astar_path(self, waypoints: np.ndarray) -> None:
        """Supply the A* reference path as (N,2) world-frame xy array."""
        self._astar_path = np.asarray(waypoints, dtype=np.float32)

    def update_mppi_rollouts(
        self,
        rollout_xy: np.ndarray,     # (K, H, 2) world xy of top-K rollouts
        rollout_costs: np.ndarray,  # (K,) cost per rollout
    ) -> None:
        """Redraw MPPI sample trajectories colored by cost."""
        if self._fig is None:
            return
        from matplotlib.collections import LineCollection

        # Normalize costs to [0,1] for coloring
        c_min, c_max = rollout_costs.min(), rollout_costs.max()
        span = max(c_max - c_min, 1e-6)
        normed = (rollout_costs - c_min) / span   # 0=best(blue), 1=worst(red)

        # Convert world xy → col/row for each rollout
        segs = []
        for k in range(len(rollout_xy)):
            pts = []
            for h in range(rollout_xy.shape[1]):
                cr = self._world_to_col_row(rollout_xy[k, h])
                pts.append(cr)
            segs.append(pts)

        cmap = plt.cm.coolwarm
        colors = cmap(normed)
        colors[:, 3] = 0.25   # semi-transparent

        # Remove old collection
        if self._rollout_collection is not None:
            self._rollout_collection.remove()
        lc = LineCollection(segs, colors=colors, linewidths=0.6, zorder=4)
        self._rollout_collection = self._ax.add_collection(lc)

        # Highlight best trajectory
        best_idx = int(np.argmin(rollout_costs))
        best_pts = []
        for h in range(rollout_xy.shape[1]):
            cr = self._world_to_col_row(rollout_xy[best_idx, h])
            best_pts.append(cr)
        bx_arr = [p[0] for p in best_pts]
        by_arr = [p[1] for p in best_pts]
        if self._best_traj_line is None:
            self._best_traj_line, = self._ax.plot(
                bx_arr, by_arr, "-", color="#00ff88", lw=1.8, alpha=0.9, zorder=5)
        else:
            self._best_traj_line.set_data(bx_arr, by_arr)


    def set_start_goal(
        self, start_xy: np.ndarray, goal_xy: np.ndarray
    ) -> None:
        self._start_xy = np.asarray(start_xy, dtype=np.float32)
        self._goal_xy  = np.asarray(goal_xy,  dtype=np.float32)

    # ------------------------------------------------------------------
    # Episode update
    # ------------------------------------------------------------------

    def update(
        self,
        robot_xy: np.ndarray,
        capture_frame: bool = False,
    ) -> None:
        """
        Call each navigation step with the current robot position.

        Parameters
        ----------
        robot_xy : (2,) world coordinates
        capture_frame : if True, snapshot the current figure into self._frames
        """
        self._trajectory.append(
            (float(robot_xy[0]), float(robot_xy[1]))
        )
        self._step += 1
        if self._fig is None:
            self._build_figure()
        self._redraw_trajectory()
        if capture_frame:
            self._frames.append(self._fig_to_array())

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def save(self, path: str | Path, verbose: bool = True) -> None:
        """Save current figure to PNG (or any matplotlib-supported format)."""
        if self._fig is None:
            self._build_figure()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._fig.savefig(str(p), dpi=self._dpi, bbox_inches="tight")
        if verbose:
            print(f"[topdown_viz] saved → {p}")

    def save_gif(self, path: str | Path, fps: int = 10) -> None:
        """Save captured frames as an animated GIF (requires Pillow)."""
        if not self._frames:
            print("[topdown_viz] no frames captured — call update(capture_frame=True)")
            return
        try:
            from PIL import Image as PILImage
        except ImportError:
            print("[topdown_viz] Pillow not installed; skipping GIF export")
            return
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        pil_frames = [PILImage.fromarray(f) for f in self._frames]
        pil_frames[0].save(
            str(p),
            save_all=True,
            append_images=pil_frames[1:],
            duration=int(1000 / fps),
            loop=0,
        )
        print(f"[topdown_viz] GIF saved → {p}  ({len(self._frames)} frames @ {fps}fps)")

    def close(self) -> None:
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None

    # ------------------------------------------------------------------
    # Internal rendering
    # ------------------------------------------------------------------

    def _build_figure(self) -> None:
        # Two-panel layout: map (left 65%) + LLM text (right 35%)
        fig_w = self._figsize[0] * 1.65
        self._fig = plt.figure(figsize=(fig_w, self._figsize[1]), dpi=self._dpi)
        self._fig.patch.set_facecolor("#1a1a2e")
        gs = self._fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.04)
        self._ax = self._fig.add_subplot(gs[0])
        self._ax_llm = self._fig.add_subplot(gs[1])
        ax = self._ax
        ax.set_facecolor("#1a1a2e")

        # --- Background: occupancy map ---
        # grid_free: True=free (white), False=obstacle (dark)
        bg = np.where(self._grid_free, 0.92, 0.18)
        bg_rgb = np.stack([bg, bg, np.where(self._grid_free, 0.88, 0.22)], axis=-1)
        ax.imshow(
            bg_rgb,
            origin="upper",
            extent=[0, self._W, self._H, 0],   # col=x, row=y (flipped for upper origin)
            interpolation="nearest",
            zorder=0,
        )

        # --- Social costmap overlay (always created; starts transparent if no data) ---
        _init_cm = self._social_costmap if self._social_costmap is not None \
                   else np.zeros((self._H, self._W), dtype=np.float32)
        _sc = np.clip(_init_cm, 0.0, 1.0)
        _cmap = plt.cm.YlOrRd
        _rgba = _cmap(_sc)
        _rgba[..., 3] = _sc * 0.65
        self._costmap_im = ax.imshow(
            _rgba,
            origin="upper",
            extent=[0, self._W, self._H, 0],
            interpolation="bilinear",
            zorder=1,
        )
        # Colorbar — always shown so scale is always visible
        _sm = plt.cm.ScalarMappable(cmap=_cmap, norm=plt.Normalize(vmin=0, vmax=1))
        _sm.set_array([])
        _cb = self._fig.colorbar(_sm, ax=ax, fraction=0.025, pad=0.015, shrink=0.55)
        _cb.set_label("Social Cost", color="white", fontsize=8)
        _cb.ax.yaxis.set_tick_params(color="white", labelsize=7)
        plt.setp(_cb.ax.yaxis.get_ticklabels(), color="white")

        # --- Humans ---
        for h in self._humans:
            self._draw_human(ax, h)

        # --- Start / Goal ---
        if self._start_xy is not None:
            sc = self._world_to_col_row(self._start_xy)
            ax.plot(*sc, marker="^", ms=12, color=self._C_START,
                    markeredgecolor="white", markeredgewidth=1.2,
                    zorder=6, label="Start")
        if self._goal_xy is not None:
            gc = self._world_to_col_row(self._goal_xy)
            ax.plot(*gc, marker="*", ms=16, color=self._C_GOAL,
                    markeredgecolor="white", markeredgewidth=1.2,
                    zorder=6, label="Goal")
            goal_r_px = self._metres_to_cells(0.20)
            circ = mpatches.Circle(gc, goal_r_px, color=self._C_GOAL,
                                   fill=False, lw=1.5, ls="--", zorder=5)
            ax.add_patch(circ)

        # --- A* reference path ---
        if self._astar_path is not None and len(self._astar_path) >= 2:
            crs = [self._world_to_col_row(pt) for pt in self._astar_path]
            ax.plot([c[0] for c in crs], [c[1] for c in crs],
                    "--", color="#44ff44", lw=1.5, alpha=0.8,
                    zorder=5, label="A* path")

        # --- MPPI rollout placeholder (LineCollection added dynamically) ---
        # --- LLM text panel ---
        self._ax_llm.set_facecolor("#0d0d1a")
        self._ax_llm.set_xticks([])
        self._ax_llm.set_yticks([])
        for spine in self._ax_llm.spines.values():
            spine.set_edgecolor("#333355")
        self._ax_llm.set_title("LLM Reasoning", color="white", fontsize=9,
                                pad=4, loc="left")
        _init_text = self._wrap_llm_text(self._llm_log) if self._llm_log \
                     else "(LLM output will appear here)"
        self._llm_text_obj = self._ax_llm.text(
            0.02, 0.98, _init_text,
            transform=self._ax_llm.transAxes,
            va="top", ha="left",
            fontsize=5.5, color="#ccddff",
            fontfamily="monospace",
            wrap=False,
        )

        # --- Trajectory placeholder (updated each step) ---
        self._traj_line, = ax.plot([], [], "-", color=self._C_TRAJ,
                                   lw=1.8, alpha=0.85, zorder=7)
        self._robot_dot, = ax.plot([], [], "o", color=self._C_ROBOT,
                                   ms=8, markeredgecolor="white",
                                   markeredgewidth=1.2, zorder=8)


        # --- Legend ---
        legend_handles = []
        seen_states: set[str] = set()
        for h in self._humans:
            if h.state not in seen_states:
                seen_states.add(h.state)
                legend_handles.append(
                    mpatches.Patch(color=self._C_HUMAN.get(h.state, "gray"),
                                   label=f"Human ({h.state})")
                )
        if self._start_xy is not None:
            legend_handles.append(mpatches.Patch(color=self._C_START, label="Start"))
        if self._goal_xy is not None:
            legend_handles.append(mpatches.Patch(color=self._C_GOAL, label="Goal"))
        legend_handles.append(
            mpatches.Patch(color=self._C_TRAJ, label="Robot trajectory")
        )
        ax.legend(handles=legend_handles, loc="upper right",
                  fontsize=8, framealpha=0.7, facecolor="#2a2a3e",
                  labelcolor="white", edgecolor="gray")

        ax.set_xlim(0, self._W)
        ax.set_ylim(self._H, 0)   # row 0 at top
        ax.set_aspect("equal", adjustable="box")

        # --- World-coordinate axes (handles any axis-aligned orientation) ---
        x_label = "col"
        y_label = "row"
        try:
            import matplotlib.ticker as mticker

            origin_cr = self._world_to_col_row(np.array([0.0, 0.0]))
            pt_x1     = self._world_to_col_row(np.array([1.0, 0.0]))
            pt_y1     = self._world_to_col_row(np.array([0.0, 1.0]))

            # 2×2 Jacobian  J[i,j] = d(grid_i) / d(world_j)
            #   grid_0 = col,  grid_1 = row
            #   world_0 = x,   world_1 = y
            J = np.array([
                [pt_x1[0] - origin_cr[0], pt_y1[0] - origin_cr[0]],
                [pt_x1[1] - origin_cr[1], pt_y1[1] - origin_cr[1]],
            ], dtype=float)

            J_inv = np.linalg.inv(J)   # d(world) / d(grid)

            _ox = float(origin_cr[0])
            _oy = float(origin_cr[1])

            # col-axis sensitivity: when col varies (row fixed), which world coord changes?
            caw = J_inv[:, 0]   # [dx/dcol, dy/dcol]
            # row-axis sensitivity: when row varies (col fixed), which world coord changes?
            raw = J_inv[:, 1]   # [dx/drow, dy/drow]

            # x-axis (col) tick labels
            if abs(caw[1]) >= abs(caw[0]):
                ax.xaxis.set_major_formatter(mticker.FuncFormatter(
                    lambda c, _, s=float(caw[1]), o=_ox: f"{s*(c-o):.0f}"))
                x_label = "y  (m)"
            else:
                ax.xaxis.set_major_formatter(mticker.FuncFormatter(
                    lambda c, _, s=float(caw[0]), o=_ox: f"{s*(c-o):.0f}"))
                x_label = "x  (m)"

            # y-axis (row) tick labels
            if abs(raw[0]) >= abs(raw[1]):
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(
                    lambda r, _, s=float(raw[0]), o=_oy: f"{s*(r-o):.0f}"))
                y_label = "x  (m)"
            else:
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(
                    lambda r, _, s=float(raw[1]), o=_oy: f"{s*(r-o):.0f}"))
                y_label = "y  (m)"

            # Faint 1 m world-grid lines
            # Lines of constant world_x = n  →  row = _oy + J[1,0] * n
            dx_per_row = J[1, 0]
            if abs(dx_per_row) > 1e-3:
                wx_min = int(math.floor(min(
                    (0 - _oy) / dx_per_row, (self._H - _oy) / dx_per_row))) - 1
                wx_max = int(math.ceil(max(
                    (0 - _oy) / dx_per_row, (self._H - _oy) / dx_per_row))) + 1
                for wx in range(wx_min, wx_max + 1):
                    r = _oy + J[1, 0] * wx
                    if 0 <= r <= self._H:
                        ax.axhline(r, color="white", alpha=0.09, lw=0.5, zorder=0)

            # Lines of constant world_y = n  →  col = _ox + J[0,1] * n
            dy_per_col = J[0, 1]
            if abs(dy_per_col) > 1e-3:
                wy_min = int(math.floor(min(
                    (0 - _ox) / dy_per_col, (self._W - _ox) / dy_per_col))) - 1
                wy_max = int(math.ceil(max(
                    (0 - _ox) / dy_per_col, (self._W - _ox) / dy_per_col))) + 1
                for wy in range(wy_min, wy_max + 1):
                    c = _ox + J[0, 1] * wy
                    if 0 <= c <= self._W:
                        ax.axvline(c, color="white", alpha=0.09, lw=0.5, zorder=0)

            # Mark world origin (0, 0)
            if 0 <= _ox <= self._W and 0 <= _oy <= self._H:
                ax.plot(_ox, _oy, "+", ms=18, color="#00ff88", mew=2.5, zorder=10)
                ax.text(_ox + 0.9, _oy - 0.9, "(0,0)",
                        color="#00ff88", fontsize=7, fontweight="bold", zorder=10,
                        path_effects=[pe.withStroke(linewidth=1.8, foreground="black")])

        except Exception as _e:
            print(f"[topdown_viz] coordinate overlay skipped: {_e}")

        ax.set_xlabel(x_label, color="white", fontsize=9)
        ax.set_ylabel(y_label, color="white", fontsize=9)
        ax.tick_params(colors="white", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        ax.set_title("Social Navigation — Bird's-Eye View", color="white", fontsize=11)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="This figure includes Axes that are not compatible with tight_layout",
                category=UserWarning,
            )
            plt.tight_layout()

    def _draw_human(self, ax: plt.Axes, h: HumanVizInfo) -> None:
        col, row = self._world_to_col_row(np.array(h.pos))
        color = self._C_HUMAN.get(h.state, "gray")

        # Personal space circle
        r_px = self._metres_to_cells(self._PERSONAL_SPACE_R)
        circ = mpatches.Circle((col, row), r_px, color=color,
                                fill=True, alpha=0.18, zorder=3)
        ax.add_patch(circ)
        circ_edge = mpatches.Circle((col, row), r_px, color=color,
                                     fill=False, lw=1.0, ls="--", alpha=0.7, zorder=3)
        ax.add_patch(circ_edge)

        # Body dot
        ax.plot(col, row, "o", color=color, ms=9,
                markeredgecolor="white", markeredgewidth=1.0, zorder=4)

        # Facing arrow (yaw_deg: 0=+x=east, 90=+y=north)
        yaw_rad = math.radians(h.yaw_deg)
        arrow_len = r_px * 0.8
        # In image coords: col=east, row increases downward (north is *up*)
        dx =  arrow_len * math.cos(yaw_rad)
        dy = -arrow_len * math.sin(yaw_rad)   # flip y for image
        ax.annotate("",
            xy=(col + dx, row + dy), xytext=(col, row),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
            zorder=5,
        )

        # State label
        ax.text(col + r_px * 0.4, row - r_px * 0.9,
                h.state, fontsize=6.5, color=color,
                path_effects=[pe.withStroke(linewidth=1.5, foreground="black")],
                zorder=5)

    def _redraw_trajectory(self) -> None:
        if not self._trajectory or self._traj_line is None:
            return
        pts = np.array(self._trajectory, dtype=np.float32)
        grid_pts = self._world_to_col_row(pts)
        if pts.ndim == 1:
            cols = [float(grid_pts[0])]
            rows = [float(grid_pts[1])]
        else:
            cols = grid_pts[:, 0].tolist()
            rows = grid_pts[:, 1].tolist()
        self._traj_line.set_data(cols, rows)
        if self._robot_dot is not None:
            self._robot_dot.set_data([cols[-1]], [rows[-1]])

        # --- Cost readout at robot position ---
        cost_val = float("nan")
        max_cost = float("nan")
        if self._social_costmap is not None:
            H, W = self._social_costmap.shape
            ci = int(round(cols[-1]))
            ri = int(round(rows[-1]))
            ci = max(0, min(ci, W - 1))
            ri = max(0, min(ri, H - 1))
            cost_val = float(self._social_costmap[ri, ci])
            max_cost = float(self._social_costmap.max())

            label = f"cost = {cost_val:.3f}"
            offset = self._metres_to_cells(0.4)
            if self._cost_text is None:
                self._cost_text = self._ax.text(
                    cols[-1] + offset, rows[-1] - offset, label,
                    color="#ffe066", fontsize=7.5, fontweight="bold", zorder=9,
                    path_effects=[pe.withStroke(linewidth=2.0, foreground="black")],
                )
            else:
                self._cost_text.set_position((cols[-1] + offset, rows[-1] - offset))
                self._cost_text.set_text(label)

        # --- Title with live stats ---
        cost_str = f"cost@robot={cost_val:.3f}  max={max_cost:.3f}" \
                   if not math.isnan(cost_val) else "no costmap"
        self._ax.set_title(
            f"Social Nav  |  step {self._step}  |  {cost_str}",
            color="white", fontsize=10,
        )

    def _world_to_col_row(self, xy: np.ndarray) -> np.ndarray:
        """World (x,y) → (col, row) in downscaled grid image."""
        flat = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
        xyz = np.zeros((flat.shape[0], 3), dtype=np.float32)
        xyz[:, :2] = flat
        px = self._scene_map.pos_m_to_px(xyz)     # (N, 2+) in pixel coords
        grid_rc = px / self._downscale             # (N, 2) row, col
        result = np.column_stack([grid_rc[:, 1], grid_rc[:, 0]])  # col, row
        if xy.ndim == 1:
            return result[0]
        return result

    def _metres_to_cells(self, m: float) -> float:
        """Convert a distance in metres to downscaled grid cell units."""
        px_per_m = self._scene_map.px_per_m if hasattr(self._scene_map, "px_per_m") else 200.0
        return m * px_per_m / self._downscale

    def _fig_to_array(self) -> np.ndarray:
        """Render current figure to an RGBA numpy array."""
        self._fig.canvas.draw()
        buf = self._fig.canvas.buffer_rgba()
        return np.frombuffer(buf, dtype=np.uint8).reshape(
            *reversed(self._fig.canvas.get_width_height()), 4
        )[..., :3]


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate a top-down scene overview with humans and navigation info."
    )
    p.add_argument("--scene-xml",   type=Path, required=True)
    p.add_argument("--layout-json", type=Path, default=None,
                   help="Deprecated alias for --scene-graph when the file is a HumanSSG scene graph")
    p.add_argument("--scene-graph", type=Path, default=None)
    p.add_argument("--out",         type=Path, required=True)
    p.add_argument("--agent-radius", type=float, default=0.22)
    p.add_argument("--px-per-m",    type=int, default=200)
    p.add_argument("--downscale",   type=int, default=5)
    p.add_argument("--start-pose",  type=str, default=None, help="x,y,yaw_deg")
    p.add_argument("--goal-xy",     type=str, default=None, help="x,y")
    p.add_argument("--social-method", choices=["none","rule","llm"], default="none")
    p.add_argument("--llm-model",   type=str, default="claude-sonnet-4-6")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()

    from molmo_spaces.utils import scene_maps, distance_transform_utils as dtutils
    from molmo_spaces.policy.solvers.navigation.mppi_core import make_downscaled_grid

    scene_maps._delete_blacklisted_bodies = lambda spec: 0
    xml_str = str(args.scene_xml)
    if "ithor" in xml_str:
        smap = scene_maps.iTHORMap.from_mj_model_path(
            model_path=xml_str, agent_radius=args.agent_radius, px_per_m=args.px_per_m)
    elif "procthor" in xml_str or "holodeck" in xml_str:
        smap = scene_maps.ProcTHORMap.from_mj_model_path(
            model_path=xml_str, agent_radius=args.agent_radius, px_per_m=args.px_per_m)
    else:
        raise ValueError(f"Cannot infer scene type from path: {args.scene_xml}")

    grid_free   = make_downscaled_grid(smap, args.downscale)
    grid_spacing = args.downscale / args.px_per_m

    viz = TopdownViz(smap, grid_free, grid_spacing, args.downscale)

    # --- Humans from scene graph ---
    scene_graph = args.scene_graph or args.layout_json
    if scene_graph is not None:
        from experiments.social_nav.cost.llm_costmap import build_entity_params, synthesize_costmap
        from pipeline.scene_bridge import scene_graph_to_scene_description

        scene = scene_graph_to_scene_description(scene_graph)
        viz.set_humans(scene.humans)

        if args.social_method != "none":
            params, _ = build_entity_params(
                scene,
                method=args.social_method,
                llm_model=args.llm_model,
            )
            occ = smap.occupancy
            corners_px = np.array([[0, 0], [occ.shape[0] - 1, occ.shape[1] - 1]], dtype=float)
            corners_m = smap.pos_px_to_m(corners_px)
            sc = synthesize_costmap(
                params,
                grid_free.shape,
                x_range=(float(min(corners_m[:, 0])), float(max(corners_m[:, 0]))),
                y_range=(float(min(corners_m[:, 1])), float(max(corners_m[:, 1]))),
            )
            viz.set_social_costmap(sc)

    # --- Start / goal ---
    start_xy = None
    if args.start_pose:
        vals = [float(v) for v in args.start_pose.split(",")]
        start_xy = np.array(vals[:2])
    goal_xy = None
    if args.goal_xy:
        vals = [float(v) for v in args.goal_xy.split(",")]
        goal_xy = np.array(vals[:2])
    if start_xy is not None and goal_xy is not None:
        viz.set_start_goal(start_xy, goal_xy)
    elif start_xy is not None:
        viz.set_start_goal(start_xy, start_xy)

    viz.save(args.out)
    viz.close()


if __name__ == "__main__":
    main()
