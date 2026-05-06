"""
SocialCostFormer — Scene Graph → Social Costmap
================================================
Architecture
------------
  1. Entity Encoder      : MLP per entity (agent / robot / goal) → d-dim token
  2. Social Transformer  : self-attention over all entity tokens
                           entities learn their mutual social relationships
  3. Spatial Decoder     : learnable spatial grid queries cross-attend to the
                           entity context → per-cell social cost value
  4. Cost Head           : MLP + Sigmoid → [0, 1]

Key insight: the costmap is jointly conditioned on (scene state, robot state,
robot goal).  Self-attention lets the model discover that e.g. a conversation
group should be given *more* space than two people standing separately,
without explicit handcrafted rules.

Usage
-----
Train:
    ./.venv/bin/python3 experiments/social_nav/social_cost_net.py \\
        --train --n-scenes 8000 --epochs 60 --label-method rule

Inference (Python):
    from experiments.social_nav.social_cost_net import SocialCostFormer, SceneTokenizer
    model = SocialCostFormer.load("checkpoints/scf_rule.pt")
    tok   = SceneTokenizer()
    feats, mask = tok.encode(agents, robot_pos, robot_goal, groups)
    cm = model.predict(feats, mask)   # numpy (GRID_H, GRID_W) float32
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Grid constants — must match sim_world.py ────────────────────────────────
X_MIN, X_MAX = -5.0,  5.0
Y_MIN, Y_MAX = -3.0,  3.0
GRID_H = 12   # rows  (6 m / 0.5 m/cell)
GRID_W = 20   # cols  (10 m / 0.5 m/cell)

# ── Entity encoding constants ────────────────────────────────────────────────
MAX_ENTITIES  = 12       # robot + goal + up to 10 agents (pad to this)
TYPE_AGENT, TYPE_ROBOT, TYPE_GOAL = 0, 1, 2
ACTIVITY_IDS = {"standing": 0, "walking": 1, "talking": 2, "sitting": 3}
N_ACTIVITIES = 4

# feature layout: [x_n, y_n, cos_h, sin_h, act×4, in_group, t_agent, t_robot, t_goal]
ENTITY_FEAT_DIM = 2 + 2 + N_ACTIVITIES + 1 + 3   # = 12


# ─────────────────────────────────────────────────────────────────────────────
# Scene Tokenizer  (scene state → padded tensor)
# ─────────────────────────────────────────────────────────────────────────────

class SceneTokenizer:
    """
    Convert a live simulation scene into a fixed-size feature matrix
    ready for SocialCostFormer.

    Accepts Agent objects (from sim_world.py) with attributes:
        .pos          np.ndarray (x, y)
        .heading_deg  float
        .activity     str
    """

    @staticmethod
    def _encode_one(x: float, y: float, heading_deg: float,
                    activity: str, in_group: bool,
                    entity_type: int) -> list[float]:
        x_n = (x - X_MIN) / (X_MAX - X_MIN) * 2 - 1
        y_n = (y - Y_MIN) / (Y_MAX - Y_MIN) * 2 - 1
        h   = math.radians(heading_deg)
        act = [0.0] * N_ACTIVITIES
        act[ACTIVITY_IDS.get(activity, 0)] = 1.0
        return [
            x_n, y_n,
            math.cos(h), math.sin(h),
            *act,
            float(in_group),
            float(entity_type == TYPE_AGENT),
            float(entity_type == TYPE_ROBOT),
            float(entity_type == TYPE_GOAL),
        ]

    def encode(
        self,
        agents: dict,                            # id → Agent
        robot_pos: Optional[np.ndarray] = None,
        robot_goal: Optional[np.ndarray] = None,
        groups: Optional[list[list[str]]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        feats : (1, MAX_ENTITIES, ENTITY_FEAT_DIM)  float32
        mask  : (1, MAX_ENTITIES)                   bool (True = padding)
        """
        groups = groups or []
        in_group_ids: set[str] = {aid for g in groups for aid in g}

        tokens: list[list[float]] = []

        # robot token
        if robot_pos is not None:
            tokens.append(self._encode_one(
                float(robot_pos[0]), float(robot_pos[1]),
                0.0, "standing", False, TYPE_ROBOT))

        # goal token
        if robot_goal is not None:
            tokens.append(self._encode_one(
                float(robot_goal[0]), float(robot_goal[1]),
                0.0, "standing", False, TYPE_GOAL))

        # agent tokens
        for aid, ag in agents.items():
            tokens.append(self._encode_one(
                float(ag.pos[0]), float(ag.pos[1]),
                float(ag.heading_deg),
                getattr(ag, "activity", "standing"),
                aid in in_group_ids,
                TYPE_AGENT,
            ))

        # pad / truncate to MAX_ENTITIES
        n = len(tokens)
        pad = MAX_ENTITIES - n
        mask_vals = [False] * n + [True] * max(0, pad)
        while len(tokens) < MAX_ENTITIES:
            tokens.append([0.0] * ENTITY_FEAT_DIM)
        tokens = tokens[:MAX_ENTITIES]
        mask_vals = mask_vals[:MAX_ENTITIES]

        feats = torch.tensor(tokens, dtype=torch.float32).unsqueeze(0)  # (1,N,D)
        mask  = torch.tensor(mask_vals, dtype=torch.bool).unsqueeze(0)  # (1,N)
        return feats, mask


# ─────────────────────────────────────────────────────────────────────────────
# Spatial positional encoding
# ─────────────────────────────────────────────────────────────────────────────

def _make_grid_pos_encoding(d_model: int, device: torch.device) -> torch.Tensor:
    """
    Sinusoidal 2D positional encoding for the spatial grid queries.
    Returns (GRID_H * GRID_W, d_model).

    Layout: first d_model//2 dims encode x, remaining encode y.
    Each axis gets d_model//4 sin + d_model//4 cos frequencies.
    """
    cy_n = torch.linspace(1.0, -1.0, GRID_H, device=device)   # row 0 = north
    cx_n = torch.linspace(-1.0, 1.0, GRID_W, device=device)
    grid_y = cy_n.unsqueeze(1).expand(GRID_H, GRID_W).reshape(-1)  # (H*W,)
    grid_x = cx_n.unsqueeze(0).expand(GRID_H, GRID_W).reshape(-1)  # (H*W,)

    half   = d_model // 2          # dims per axis
    n_freq = half // 2             # number of frequency bands per axis
    div    = torch.exp(
        torch.arange(0, n_freq, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / max(n_freq - 1, 1))
    )                              # (n_freq,)

    enc = torch.zeros(GRID_H * GRID_W, d_model, device=device)
    for axis, val_1d in [(0, grid_x), (1, grid_y)]:
        offset = axis * half
        angles = val_1d.unsqueeze(1) * div.unsqueeze(0)   # (H*W, n_freq)
        enc[:, offset            : offset + n_freq] = torch.sin(angles)
        enc[:, offset + n_freq   : offset + 2*n_freq] = torch.cos(angles)
    return enc   # (H*W, d_model)


# ─────────────────────────────────────────────────────────────────────────────
# SocialCostFormer
# ─────────────────────────────────────────────────────────────────────────────

class SocialCostFormer(nn.Module):
    """
    Scene-conditioned social cost predictor.

    Parameters
    ----------
    d_model   : transformer hidden dim
    n_heads   : attention heads
    n_layers  : self-attention layers (entity transformer)
    n_dec_layers: cross-attention layers (spatial decoder)
    dropout   : dropout probability
    """

    def __init__(
        self,
        d_model:      int = 64,
        n_heads:      int = 4,
        n_layers:     int = 3,
        n_dec_layers: int = 2,
        dropout:      float = 0.05,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # ── Entity Encoder: raw features → token ──────────────────────────
        self.entity_enc = nn.Sequential(
            nn.Linear(ENTITY_FEAT_DIM, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        # ── Social Transformer: self-attention over entity tokens ──────────
        # Entities learn their mutual relationships (group dynamics, proximity)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.social_transformer = nn.TransformerEncoder(enc_layer, n_layers)

        # ── Spatial Decoder: learnable queries attend to entity context ────
        # Each grid cell = one query; cross-attention fetches what it needs
        self.spatial_query_emb = nn.Parameter(
            torch.randn(GRID_H * GRID_W, d_model) * 0.02
        )
        dec_layers = []
        for _ in range(n_dec_layers):
            dec_layers.append(_SpatialDecoderLayer(d_model, n_heads, dropout))
        self.spatial_decoder = nn.ModuleList(dec_layers)
        self.spatial_norm     = nn.LayerNorm(d_model)

        # ── Cost Head ──────────────────────────────────────────────────────
        self.cost_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        entity_feats: torch.Tensor,         # (B, N, ENTITY_FEAT_DIM)
        entity_mask:  Optional[torch.Tensor] = None,  # (B, N) bool pad mask
    ) -> torch.Tensor:
        """
        Returns
        -------
        costmap : (B, GRID_H, GRID_W)  values in [0, 1]
        """
        B = entity_feats.shape[0]
        device = entity_feats.device

        # 1. Encode entity tokens
        tokens = self.entity_enc(entity_feats)              # (B, N, d)

        # 2. Social self-attention (entities reason about each other)
        ctx = self.social_transformer(
            tokens,
            src_key_padding_mask=entity_mask,
        )                                                   # (B, N, d)

        # 3. Build spatial queries:
        #    learnable query embeddings + sinusoidal position encoding
        pos_enc = _make_grid_pos_encoding(self.d_model, device)  # (H*W, d)
        spatial_q = (self.spatial_query_emb + pos_enc).unsqueeze(0).expand(B, -1, -1)
        # (B, H*W, d)

        # 4. Spatial decoder: cross-attend to entity context
        for layer in self.spatial_decoder:
            spatial_q = layer(spatial_q, ctx, entity_mask)  # (B, H*W, d)
        spatial_q = self.spatial_norm(spatial_q)

        # 5. Per-cell cost prediction
        cost = self.cost_head(spatial_q)        # (B, H*W, 1)
        return cost.squeeze(-1).view(B, GRID_H, GRID_W)

    @torch.no_grad()
    def predict(
        self,
        feats: torch.Tensor,
        mask:  torch.Tensor,
    ) -> np.ndarray:
        """Run inference, return numpy (GRID_H, GRID_W) float32."""
        self.eval()
        out = self(feats, mask)    # (1, H, W)
        return out[0].cpu().numpy()

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        torch.save({"state_dict": self.state_dict(),
                    "config": self._config()}, path)
        print(f"[SocialCostFormer] saved → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "SocialCostFormer":
        ckpt = torch.load(path, map_location="cpu")
        cfg  = ckpt.get("config", {})
        model = cls(**cfg)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        print(f"[SocialCostFormer] loaded ← {path}  "
              f"({sum(p.numel() for p in model.parameters()):,} params)")
        return model

    def _config(self) -> dict:
        # infer from first linear layer shapes
        return {
            "d_model":      self.d_model,
            "n_heads":      self.social_transformer.layers[0].self_attn.num_heads,
            "n_layers":     len(self.social_transformer.layers),
            "n_dec_layers": len(self.spatial_decoder),
        }


class _SpatialDecoderLayer(nn.Module):
    """Single cross-attention + FFN layer for the spatial decoder."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.cross   = nn.MultiheadAttention(d_model, n_heads,
                                             dropout=dropout, batch_first=True)
        self.ffn     = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor,
                kv_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # cross-attention (pre-norm)
        q2, _ = self.cross(self.norm1(q), kv, kv, key_padding_mask=kv_mask)
        q     = q + q2
        # FFN (pre-norm)
        q     = q + self.ffn(self.norm2(q))
        return q


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic dataset generation
# ─────────────────────────────────────────────────────────────────────────────

def _random_scene(rng: np.random.Generator, n_agents: Optional[int] = None):
    """
    Generate one random scene description.

    Returns
    -------
    agents : dict  id → SimpleNamespace(pos, heading_deg, activity)
    robot_pos, robot_goal : np.ndarray (2,)
    groups : list[list[str]]
    """
    from types import SimpleNamespace

    if n_agents is None:
        n_agents = int(rng.integers(1, 5))

    agents: dict = {}
    for i in range(n_agents):
        x = float(rng.uniform(X_MIN + 0.5, X_MAX - 0.5))
        y = float(rng.uniform(Y_MIN + 0.4, Y_MAX - 0.4))
        h = float(rng.uniform(0, 360))
        a = rng.choice(list(ACTIVITY_IDS.keys()))
        ag = SimpleNamespace(pos=np.array([x, y], dtype=np.float32),
                             heading_deg=h, activity=a)
        agents[f"A{i}"] = ag

    # robot start / goal on opposite sides
    rx = float(rng.uniform(2.0, 4.5)) * rng.choice([-1, 1])
    ry = float(rng.uniform(Y_MIN + 0.4, Y_MAX - 0.4))
    gx = float(rng.uniform(2.0, 4.5)) * -np.sign(rx)
    gy = float(rng.uniform(Y_MIN + 0.4, Y_MAX - 0.4))
    robot_pos  = np.array([rx, ry], dtype=np.float32)
    robot_goal = np.array([gx, gy], dtype=np.float32)

    # random conversation pairs (≤ 2 pairs)
    groups: list[list[str]] = []
    ids = list(agents.keys())
    rng.shuffle(ids)
    for i in range(0, len(ids) - 1, 2):
        if rng.random() < 0.40:
            agents[ids[i]].activity     = "talking"
            agents[ids[i+1]].activity   = "talking"
            groups.append([ids[i], ids[i+1]])

    return agents, robot_pos, robot_goal, groups


def generate_dataset(
    n_scenes:     int,
    label_method: str   = "rule",
    llm_model:    str   = "doubao-pro-32k",
    seed:         int   = 0,
    verbose:      bool  = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate (feats, masks, targets) tensors for training.

    label_method = "rule" : fast rule-based Gaussian proxemics labels
    label_method = "llm"  : LLM semantic labels (needs API key, ~3s/scene)

    Returns
    -------
    feats   : (N, MAX_ENTITIES, ENTITY_FEAT_DIM) float32
    masks   : (N, MAX_ENTITIES)                   bool
    targets : (N, GRID_H, GRID_W)                 float32 [0,1]
    """
    from experiments.social_nav.llm_costmap import build_live_costmap

    rng  = np.random.default_rng(seed)
    tok  = SceneTokenizer()

    all_feats, all_masks, all_targets = [], [], []
    t0 = time.time()

    for i in range(n_scenes):
        agents, rpos, rgoal, groups = _random_scene(rng)

        # ── label ──
        if label_method == "rule":
            cm, _ = build_live_costmap(
                agents, (GRID_H, GRID_W),
                x_range=(X_MIN, X_MAX), y_range=(Y_MIN, Y_MAX),
                method="rule", groups=groups,
            )
        elif label_method == "llm":
            cm, _ = build_live_costmap(
                agents, (GRID_H, GRID_W),
                x_range=(X_MIN, X_MAX), y_range=(Y_MIN, Y_MAX),
                method="llm", llm_model=llm_model,
                robot_pos=rpos, robot_goal=rgoal,
                groups=groups, t=float(i),
            )
        else:
            raise ValueError(label_method)

        feats, mask = tok.encode(agents, rpos, rgoal, groups)
        all_feats.append(feats)
        all_masks.append(mask)
        all_targets.append(torch.tensor(cm, dtype=torch.float32).unsqueeze(0))

        if verbose and (i + 1) % max(1, n_scenes // 10) == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_scenes - i - 1)
            print(f"  [{i+1:5d}/{n_scenes}]  elapsed={elapsed:.1f}s  ETA={eta:.1f}s")

    feats   = torch.cat(all_feats,   dim=0)   # (N, MAX_ENTITIES, D)
    masks   = torch.cat(all_masks,   dim=0)   # (N, MAX_ENTITIES)
    targets = torch.cat(all_targets, dim=0)   # (N, GRID_H, GRID_W)
    return feats, masks, targets


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(
    n_scenes:     int   = 8000,
    epochs:       int   = 60,
    batch_size:   int   = 64,
    lr:           float = 3e-4,
    label_method: str   = "rule",
    llm_model:    str   = "doubao-pro-32k",
    d_model:      int   = 64,
    n_layers:     int   = 3,
    n_dec_layers: int   = 2,
    save_path:    str   = "checkpoints/scf.pt",
    seed:         int   = 42,
) -> SocialCostFormer:

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] device={device}  n_scenes={n_scenes}  "
          f"epochs={epochs}  label={label_method}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    print("[Train] Generating dataset…")
    feats, masks, targets = generate_dataset(
        n_scenes, label_method=label_method, llm_model=llm_model, seed=seed)

    # train / val split  90/10
    n_val   = max(1, n_scenes // 10)
    n_train = n_scenes - n_val
    perm    = torch.randperm(n_scenes)
    tr_idx, va_idx = perm[:n_train], perm[n_train:]

    tr_ds = torch.utils.data.TensorDataset(
        feats[tr_idx], masks[tr_idx], targets[tr_idx])
    va_ds = torch.utils.data.TensorDataset(
        feats[va_idx], masks[va_idx], targets[va_idx])
    tr_dl = torch.utils.data.DataLoader(tr_ds, batch_size=batch_size,
                                        shuffle=True, drop_last=False)
    va_dl = torch.utils.data.DataLoader(va_ds, batch_size=batch_size)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SocialCostFormer(d_model=d_model, n_layers=n_layers,
                             n_dec_layers=n_dec_layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Train] SocialCostFormer  {n_params:,} parameters")

    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val = float("inf")
    ckpt_path = Path(save_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        # train
        model.train()
        tr_loss = 0.0
        for f, m, tgt in tr_dl:
            f, m, tgt = f.to(device), m.to(device), tgt.to(device)
            pred  = model(f, m)

            # MSE + gradient penalty on smooth predictions
            loss  = F.mse_loss(pred, tgt)
            # spatial smoothness regulariser (optional, encourages smooth fields)
            dh = (pred[:, 1:, :] - pred[:, :-1, :]).pow(2).mean()
            dw = (pred[:, :, 1:] - pred[:, :, :-1]).pow(2).mean()
            loss = loss + 0.01 * (dh + dw)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += float(loss) * len(f)
        tr_loss /= n_train
        sched.step()

        # val
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for f, m, tgt in va_dl:
                f, m, tgt = f.to(device), m.to(device), tgt.to(device)
                va_loss += float(F.mse_loss(model(f, m), tgt)) * len(f)
        va_loss /= n_val

        if va_loss < best_val:
            best_val = va_loss
            model.save(ckpt_path)
            tag = " ★"
        else:
            tag = ""

        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}/{epochs}  "
                  f"tr={tr_loss:.5f}  va={va_loss:.5f}{tag}")

    print(f"[Train] done.  best val MSE = {best_val:.5f}")
    return SocialCostFormer.load(ckpt_path)


# ─────────────────────────────────────────────────────────────────────────────
# Quick visual test
# ─────────────────────────────────────────────────────────────────────────────

def _visualise(model: SocialCostFormer, seed: int = 7) -> None:
    """ASCII visualisation of a sample prediction vs label."""
    from experiments.social_nav.llm_costmap import build_live_costmap

    rng    = np.random.default_rng(seed)
    agents, rpos, rgoal, groups = _random_scene(rng, n_agents=3)
    tok    = SceneTokenizer()
    feats, mask = tok.encode(agents, rpos, rgoal, groups)

    pred = model.predict(feats, mask)

    rule_cm, _ = build_live_costmap(
        agents, (GRID_H, GRID_W),
        x_range=(X_MIN, X_MAX), y_range=(Y_MIN, Y_MAX),
        method="rule", groups=groups,
    )

    chars = " ░▒▓█"

    def to_ascii(cm, label):
        print(f"\n── {label} ──")
        for r in range(GRID_H):
            row = ""
            for c in range(GRID_W):
                v = float(cm[r, c])
                row += chars[min(int(v * len(chars)), len(chars) - 1)] * 2
            print(row)

    to_ascii(rule_cm, "Rule-based label")
    to_ascii(pred,    "SocialCostFormer prediction")

    print(f"\nMSE vs rule label: {float(np.mean((pred - rule_cm)**2)):.5f}")
    print(f"Pred  max={pred.max():.3f}  mean={pred.mean():.4f}")
    print(f"Label max={rule_cm.max():.3f}  mean={rule_cm.mean():.4f}")

    # Timing
    n = 200
    t0 = time.perf_counter()
    for _ in range(n):
        model.predict(feats, mask)
    ms = (time.perf_counter() - t0) / n * 1000
    print(f"\nInference: {ms:.2f} ms / call  ({1000/ms:.0f} fps)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    p = argparse.ArgumentParser(description="SocialCostFormer train / eval")
    p.add_argument("--train",        action="store_true")
    p.add_argument("--eval",         action="store_true", help="visual eval of checkpoint")
    p.add_argument("--n-scenes",     type=int,   default=8000)
    p.add_argument("--epochs",       type=int,   default=60)
    p.add_argument("--batch-size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--d-model",      type=int,   default=64)
    p.add_argument("--n-layers",     type=int,   default=3)
    p.add_argument("--n-dec-layers", type=int,   default=2)
    p.add_argument("--label-method", type=str,   default="rule",
                   choices=["rule", "llm"])
    p.add_argument("--llm-model",    type=str,   default="doubao-pro-32k")
    p.add_argument("--save",         type=str,   default="checkpoints/scf.pt")
    p.add_argument("--load",         type=str,   default=None)
    p.add_argument("--seed",         type=int,   default=42)
    args = p.parse_args()

    if args.train:
        model = train(
            n_scenes=args.n_scenes, epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr,
            label_method=args.label_method, llm_model=args.llm_model,
            d_model=args.d_model, n_layers=args.n_layers,
            n_dec_layers=args.n_dec_layers,
            save_path=args.save, seed=args.seed,
        )
        _visualise(model, seed=args.seed)

    elif args.eval:
        ckpt = args.load or args.save
        model = SocialCostFormer.load(ckpt)
        _visualise(model, seed=args.seed)

    else:
        p.print_help()
