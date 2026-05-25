"""
Social Cost (local, dynamic)
------------------------------
Evaluates per-entity anisotropic Gaussian social cost directly from
SocialEntityParams — no precomputed grid, no resolution dependency.

LLM outputs parameters → stored here → computed on-the-fly per MPPI step.
"""

from __future__ import annotations

import math

import torch

from experiments.social_nav.cost.llm_costmap import SocialEntityParams


class SocialCost:
    """
    Holds per-entity social parameters and evaluates social cost inside the MPPI loop.

    running(state, action, t) → per-sample cost tensor (N,)

    Call update_params() after LLM returns fresh parameters.
    """

    def __init__(
        self,
        params: list[SocialEntityParams] | None = None,
        weight: float = 30.0,
        back_scale: float = 1.0,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.weight = float(weight)
        self.back_scale = float(back_scale)
        self.device = torch.device(device)
        self.dtype  = dtype
        self._tensors: dict | None = None
        if params:
            self.update_params(params)

    def update_params(self, params: list[SocialEntityParams]) -> None:
        """Replace entity params (thread-safe via Python GIL)."""
        if not params:
            self._tensors = None
            return

        pos    = torch.tensor([[p.pos[0], p.pos[1]] for p in params],
                               dtype=self.dtype, device=self.device)           # (M, 2)
        yaw    = torch.tensor([math.radians(p.yaw_deg) for p in params],
                               dtype=self.dtype, device=self.device)           # (M,)
        fx     = torch.cos(yaw)                                                # (M,)
        fy     = torch.sin(yaw)
        score  = torch.tensor([p.score for p in params],
                               dtype=self.dtype, device=self.device)           # (M,)
        ps     = torch.tensor([p.personal_space for p in params],
                               dtype=self.dtype, device=self.device)           # (M,)
        os_    = torch.tensor([p.orientation_sensitivity for p in params],
                               dtype=self.dtype, device=self.device)           # (M,)
        # 允许每个实体独立设置侧向sigma（群组实体用于椭圆形cost）
        sigma_side = torch.tensor(
            [p.sigma_perp if p.sigma_perp is not None else p.personal_space for p in params],
            dtype=self.dtype, device=self.device,
        )                                                                      # (M,)
        is_edge = torch.tensor(
            [str(p.entity_id).startswith("edge_") for p in params],
            dtype=torch.bool, device=self.device,
        )                                                                      # (M,)

        # Keep rear personal space non-trivial: otherwise A*/MPPI tends to cut
        # through the narrow gap directly behind a person instead of taking a
        # socially cleaner detour.
        self._tensors = dict(pos=pos, fx=fx, fy=fy, score=score,
                             sigma_front=ps * os_,
                             sigma_back=ps * self.back_scale,
                             sigma_side=sigma_side,
                             half_len=ps,
                             is_edge=is_edge)

    def is_active(self) -> bool:
        return self._tensors is not None and self.weight > 0.0

    # ------------------------------------------------------------------
    # MPPI cost interface
    # ------------------------------------------------------------------

    def running(self, state: torch.Tensor, action: torch.Tensor, _t: int) -> torch.Tensor:
        """Per-step social running cost. state: (N, state_dim)."""
        if not self.is_active():
            return torch.zeros(state.shape[0], dtype=self.dtype, device=self.device)

        t = self._tensors
        pos = state[:, :2]                                  # (N, 2)

        # (N, M, 2) displacement from each human
        diff  = pos.unsqueeze(1) - t["pos"].unsqueeze(0)   # (N, M, 2)
        dx, dy = diff[..., 0], diff[..., 1]                # (N, M)

        along = dx * t["fx"] + dy * t["fy"]                # (N, M)  >0 = in front
        perp  = -dx * t["fy"] + dy * t["fx"]               # (N, M)  lateral

        sigma_along = torch.where(along >= 0,
                                  t["sigma_front"], t["sigma_back"])           # (N, M)

        gaussian = t["score"] * torch.exp(
            -(along ** 2) / (2 * sigma_along ** 2)
            -(perp  ** 2) / (2 * t["sigma_side"] ** 2)
        )                                                                      # (N, M)

        if torch.any(t["is_edge"]):
            excess = torch.clamp(torch.abs(along) - t["half_len"], min=0.0)
            segment_gaussian = t["score"] * torch.exp(
                -(perp ** 2 + excess ** 2) / (2 * t["sigma_side"] ** 2)
            )
            gaussian = torch.where(t["is_edge"].unsqueeze(0), segment_gaussian, gaussian)

        cost = gaussian.max(dim=-1).values                                     # (N,)
        return self.weight * cost
