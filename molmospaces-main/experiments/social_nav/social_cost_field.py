"""
Social Cost Field  —  Conditional Neural Field for Social Navigation
====================================================================

Architecture
------------
                    Scene Graph
                  (nodes + edges)
                        │
               Graph Transformer         ← encodes scene into z
                        │
                    z  (128-d)           ← "what is happening in this scene"
                        │
              ┌─────────┴─────────┐
           (x, y)              (x, y)    ← query any point
              └─────────┬─────────┘
                        │
                   Cost MLP              ← f(x, y | z) → cost
                        │
                    cost ∈ [0,1]         ← social cost at that point

Key insight: the model learns a *continuous function* over space,
conditioned on the scene's semantic description.
Resolution-independent: same model, any grid size.

Training
--------
    ./.venv/bin/python3 experiments/social_nav/social_cost_field.py --train
    ./.venv/bin/python3 experiments/social_nav/social_cost_field.py --eval
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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ── World constants ──────────────────────────────────────────────────────────
X_MIN, X_MAX = -6.0,  6.0
Y_MIN, Y_MAX = -1.0,  6.0

# ── Entity / relation vocabulary ─────────────────────────────────────────────
ACTIVITY_IDS = {"standing": 0, "walking": 1, "talking": 2, "sitting": 3}
N_ACTIVITIES  = len(ACTIVITY_IDS)

# Relation type → canonical natural language description
# (used to generate sentence embeddings at dataset creation time)
# Relation keys match HumanSSG edge_json "relationship" field exactly
RELATION_TEMPLATES = {
    "conversing with":
        "two people actively conversing with each other face to face",
    "looking towards":
        "one person looking towards another person, directing attention at them",
    "observing":
        "one person observing another person from a distance, watching passively",
    "standing beside":
        "two people standing side by side, physically close but not necessarily interacting",
    "standing near":
        "two people standing near each other in the same area",
    "standing next to":
        "two people standing next to each other, in close proximity",
    "viewing":
        "one person viewing another person or watching what they are doing",
    "watching":
        "one person watching another person or a shared focal point such as a TV",
}
RELATION_INTENSITY = {
    "conversing with":  0.85,
    "looking towards":  0.50,
    "observing":        0.30,
    "standing beside":  0.40,
    "standing near":    0.25,
    "standing next to": 0.40,
    "viewing":          0.30,
    "watching":         0.35,
}

# Entity types
TYPE_AGENT, TYPE_ROBOT, TYPE_GOAL = 0, 1, 2
N_TYPES = 3

SENT_DIM  = 384   # sentence-transformers 'all-MiniLM-L6-v2' output dim

# Node feature: [x_n, y_n, cos_h, sin_h, act×4, type×3]  = 11-d
NODE_DIM  = 2 + N_ACTIVITIES + N_TYPES   # 9  (cos_h, sin_h, act×4, type×3) — no x/y
# Edge feature: sentence_embedding(384) + intensity(1)    = 385-d
EDGE_DIM  = SENT_DIM + 1                      # 385
SCENE_DIM = 128   # latent scene embedding z


# ── Sentence encoder (lazy singleton) ────────────────────────────────────────
_sent_model = None

def get_sent_model():
    """Load sentence-transformers model once, reuse afterwards."""
    global _sent_model
    if _sent_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _sent_model = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed.\n"
                "Run: ./.venv/bin/pip install sentence-transformers"
            )
    return _sent_model


def encode_text(texts: list[str]) -> np.ndarray:
    """Encode a list of strings → (N, SENT_DIM) float32 numpy array."""
    model = get_sent_model()
    return model.encode(texts, convert_to_numpy=True,
                        show_progress_bar=False).astype(np.float32)


# Pre-compute embeddings for all canonical relation templates (cached)
_RELATION_EMBEDDINGS: dict[str, np.ndarray] = {}

def get_relation_embedding(relation_key: str) -> np.ndarray:
    """
    Returns (SENT_DIM,) embedding for a relation key or free-form text.
    Free-form text (not in RELATION_TEMPLATES) is encoded on the fly.
    """
    if relation_key not in _RELATION_EMBEDDINGS:
        text = RELATION_TEMPLATES.get(relation_key, relation_key)
        _RELATION_EMBEDDINGS[relation_key] = encode_text([text])[0]
    return _RELATION_EMBEDDINGS[relation_key]


# ─────────────────────────────────────────────────────────────────────────────
# Scene Graph representation
# ─────────────────────────────────────────────────────────────────────────────

class SceneGraph:
    """
    Lightweight scene graph: list of nodes + list of directed edges.

    nodes : list of dicts
        {pos, heading_deg, activity, entity_type}
    edges : list of dicts
        {src, dst, relation, intensity}   (src/dst = node index)
    """

    def __init__(self):
        self.nodes: list[dict] = []
        self.edges: list[dict] = []

    def add_node(self, pos, heading_deg: float = 0.0,
                 activity: str = "standing",
                 entity_type: int = TYPE_AGENT) -> int:
        self.nodes.append(dict(pos=np.array(pos, dtype=np.float32),
                               heading_deg=float(heading_deg),
                               activity=activity,
                               entity_type=entity_type))
        return len(self.nodes) - 1

    def add_edge(self, src: int, dst: int,
                 relation: str = "strangers",
                 intensity: Optional[float] = None,
                 description: Optional[str] = None):
        """
        relation    : key in RELATION_TEMPLATES, or any free-form string
        description : optional custom text (overrides relation template)
        intensity   : social avoidance weight [0,1]; inferred from relation if None
        """
        if intensity is None:
            intensity = RELATION_INTENSITY.get(relation, 0.5)
        text = description or RELATION_TEMPLATES.get(relation, relation)
        self.edges.append(dict(src=src, dst=dst, relation=relation,
                               description=text, intensity=float(intensity)))
        self.edges.append(dict(src=dst, dst=src, relation=relation,
                               description=text, intensity=float(intensity)))

    def to_tensors(self, max_nodes: int = 12
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        node_feats : (max_nodes, NODE_DIM)       float32  — heading + activity + type, NO x/y
        edge_feats : (max_nodes, max_nodes, EDGE_DIM)  float32
        node_mask  : (max_nodes,)  bool  True = padding
        node_pos   : (max_nodes, 2)              float32  — raw world coords (x, y) in metres
        """
        N = len(self.nodes)
        pad = max_nodes - N
        mask_vals = [False] * N + [True] * max(0, pad)

        nf  = torch.zeros(max_nodes, NODE_DIM)
        pos = torch.zeros(max_nodes, 2)
        for i, nd in enumerate(self.nodes[:max_nodes]):
            h   = math.radians(nd["heading_deg"])
            act = [0.0] * N_ACTIVITIES
            act[ACTIVITY_IDS.get(nd["activity"], 0)] = 1.0
            typ = [0.0] * N_TYPES
            typ[nd["entity_type"]] = 1.0
            nf[i]  = torch.tensor([math.cos(h), math.sin(h), *act, *typ])
            pos[i] = torch.tensor([nd["pos"][0], nd["pos"][1]])

        # default edge = "strangers" embedding + 0 intensity
        strangers_emb = torch.tensor(
            get_relation_embedding("strangers"), dtype=torch.float32)
        ef = torch.zeros(max_nodes, max_nodes, EDGE_DIM)
        for i in range(max_nodes):
            for j in range(max_nodes):
                if i != j:
                    ef[i, j, :SENT_DIM] = strangers_emb

        for e in self.edges:
            s, d = e["src"], e["dst"]
            if s >= max_nodes or d >= max_nodes:
                continue
            emb = torch.tensor(
                get_relation_embedding(e.get("description", e["relation"])),
                dtype=torch.float32)
            ef[s, d, :SENT_DIM] = emb
            ef[s, d, SENT_DIM]  = e["intensity"]

        mask = torch.tensor(mask_vals, dtype=torch.bool)
        return nf, ef, mask, pos


# ─────────────────────────────────────────────────────────────────────────────
# Graph Transformer Encoder  (Scene → z)
# ─────────────────────────────────────────────────────────────────────────────

class GraphTransformerEncoder(nn.Module):
    """
    Scene Graph → H  (per-entity context embeddings).

    Each node aggregates information from neighbours via edge-biased
    attention. Returns the full sequence H — NOT compressed into a single
    vector — so the decoder can cross-attend to individual entities.
    This makes the model scale to any number of agents.
    """

    def __init__(self, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 3, dropout: float = 0.05) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        self.node_proj = nn.Sequential(
            nn.Linear(NODE_DIM, d_model),
            nn.LayerNorm(d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        # edge sentence embedding + intensity → per-head attention bias
        self.edge_proj = nn.Sequential(
            nn.Linear(EDGE_DIM, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_heads),
        )
        self.layers = nn.ModuleList([
            _GTLayer(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, node_feats: torch.Tensor,
                edge_feats: torch.Tensor,
                node_mask:  torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        node_feats : (B, N, NODE_DIM)
        edge_feats : (B, N, N, EDGE_DIM)
        node_mask  : (B, N)   True = padding

        Returns
        -------
        H : (B, N, d_model)   per-entity context embeddings
        """
        h          = self.node_proj(node_feats)           # (B, N, d)
        edge_bias  = self.edge_proj(edge_feats)            # (B, N, N, n_heads)
        edge_bias  = edge_bias.permute(0, 3, 1, 2)        # (B, n_heads, N, N)

        for layer in self.layers:
            h = layer(h, edge_bias, node_mask)

        return self.norm(h)   # (B, N, d)  — no readout bottleneck


class _GTLayer(nn.Module):
    """One Graph Transformer layer with edge-biased attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn   = nn.MultiheadAttention(d_model, n_heads,
                                             dropout=dropout, batch_first=True)
        self.norm1  = nn.LayerNorm(d_model)
        self.norm2  = nn.LayerNorm(d_model)
        self.ffn    = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Linear(d_model * 4, d_model), nn.Dropout(dropout),
        )

    def forward(self, h: torch.Tensor,
                edge_bias: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        B, N, d = h.shape
        n_heads  = edge_bias.shape[1]

        # compute raw attention weights + add edge bias
        # edge_bias: (B, n_heads, N, N)
        # we inject it as attn_mask (additive)
        attn_mask = edge_bias.reshape(B * n_heads, N, N)

        h2, _ = self.attn(
            self.norm1(h), self.norm1(h), self.norm1(h),
            key_padding_mask=mask,
            attn_mask=attn_mask,
        )
        h = h + h2
        h = h + self.ffn(self.norm2(h))
        return h


# ─────────────────────────────────────────────────────────────────────────────
# Cost Decoder  —  relative-position cross-attention  (space-agnostic)
# ─────────────────────────────────────────────────────────────────────────────

class CostDecoder(nn.Module):
    """
    Queries social cost at arbitrary (x, y) using RELATIVE position only.

    Key design principle: the model never sees absolute coordinates.
    For each (query_point p, entity eᵢ) pair it computes:

        along = (p - eᵢ) · heading_eᵢ     # how far in front/behind
        perp  = (p - eᵢ) ⊥ heading_eᵢ     # how far to the side
        dist  = |p - eᵢ|                   # euclidean distance

    These three numbers are encoded with Fourier features and used as
    attention bias in cross-attention.  The model learns:

        "I am 0.8 m in front of a heated-argument entity → high cost"

    This is true regardless of which room, which coordinate origin,
    or which scale the scene uses.  Fully space-agnostic.
    """

    def __init__(self, d_model: int = 64, n_heads: int = 4,
                 n_freq: int = 16, n_layers: int = 2,
                 dropout: float = 0.05) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.n_freq  = n_freq

        # Fourier freqs for relative (along, perp, dist) — 3 channels
        self.register_buffer(
            "freqs", 2.0 ** torch.arange(n_freq, dtype=torch.float32))
        rel_enc_dim = 3 * 2 * n_freq   # sin+cos per channel

        # relative position bias → per-head scalar  (used as additive attn bias)
        self.rel_bias_proj = nn.Sequential(
            nn.Linear(rel_enc_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_heads),   # (B, P, N, n_heads)
        )

        # query token: starts as a learnable zero-like vector;
        # all spatial info comes through the relative bias — not absolute coords
        self.query_init = nn.Parameter(torch.zeros(1, 1, d_model))

        self.ca_layers = nn.ModuleList([
            _RelCrossAttnLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, 1), nn.Sigmoid(),
        )

    def _rel_enc(self, xy: torch.Tensor,
                 entity_pos: torch.Tensor,
                 entity_heading: torch.Tensor) -> torch.Tensor:
        """
        Compute relative position encoding for every (query_point, entity) pair.

        Parameters
        ----------
        xy             : (B, P, 2)   query points in world coords
        entity_pos     : (B, N, 2)   entity positions
        entity_heading : (B, N)      entity headings in radians

        Returns
        -------
        rel_bias : (B, P, N, n_heads)  additive attention bias
        """
        B, P, _ = xy.shape
        N        = entity_pos.shape[1]

        # displacement vectors: (B, P, N, 2)
        p_exp = xy.unsqueeze(2).expand(B, P, N, 2)        # (B,P,N,2)
        e_exp = entity_pos.unsqueeze(1).expand(B, P, N, 2) # (B,P,N,2)
        delta  = p_exp - e_exp                              # (B,P,N,2)

        # rotate into each entity's local frame
        cos_h = torch.cos(entity_heading).unsqueeze(1).expand(B, P, N)  # (B,P,N)
        sin_h = torch.sin(entity_heading).unsqueeze(1).expand(B, P, N)

        along = delta[..., 0] * cos_h + delta[..., 1] * sin_h  # (B,P,N)
        perp  = -delta[..., 0] * sin_h + delta[..., 1] * cos_h
        dist  = delta.norm(dim=-1)                               # (B,P,N)

        # stack and Fourier-encode
        rel = torch.stack([along, perp, dist], dim=-1)          # (B,P,N,3)
        angles = rel.unsqueeze(-1) * self.freqs * math.pi       # (B,P,N,3,n_freq)
        enc = torch.cat([torch.sin(angles),
                         torch.cos(angles)], dim=-1)             # (B,P,N,3,2*n_freq)
        enc = enc.reshape(B, P, N, -1)                          # (B,P,N, 6*n_freq)

        return self.rel_bias_proj(enc)   # (B, P, N, n_heads)

    def forward(self, xy: torch.Tensor,
                H:  torch.Tensor,
                entity_pos:     torch.Tensor,
                entity_heading: torch.Tensor,
                entity_mask:    Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Parameters
        ----------
        xy             : (B, P, 2)   query points (world coords, any scale)
        H              : (B, N, d)   entity context from GraphTransformerEncoder
        entity_pos     : (B, N, 2)   entity positions (for relative encoding)
        entity_heading : (B, N)      entity headings in radians
        entity_mask    : (B, N)      True = padding

        Returns
        -------
        cost : (B, P)  in [0, 1]
        """
        B, P, _ = xy.shape

        # relative attention bias: (B, P, N, n_heads)
        rel_bias = self._rel_enc(xy, entity_pos, entity_heading)

        # initial query token (same for all points; spatial info via rel_bias)
        q = self.query_init.expand(B, P, -1)   # (B, P, d)

        for layer in self.ca_layers:
            q = layer(q, H, rel_bias, entity_mask)

        return self.head(self.norm(q)).squeeze(-1)   # (B, P)


class _RelCrossAttnLayer(nn.Module):
    """Cross-attention with relative position bias, pre-norm."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads
        self.scale    = self.d_head ** -0.5
        self.norm_q   = nn.LayerNorm(d_model)
        self.norm_kv  = nn.LayerNorm(d_model)
        self.norm_ff  = nn.LayerNorm(d_model)
        self.W_q  = nn.Linear(d_model, d_model, bias=False)
        self.W_k  = nn.Linear(d_model, d_model, bias=False)
        self.W_v  = nn.Linear(d_model, d_model, bias=False)
        self.W_o  = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        self.ffn  = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Linear(d_model * 4, d_model), nn.Dropout(dropout),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor,
                rel_bias: torch.Tensor,
                kv_mask:  Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        q        : (B, P, d)
        kv       : (B, N, d)
        rel_bias : (B, P, N, n_heads)   additive bias per head
        kv_mask  : (B, N)               True = ignore
        """
        B, P, d = q.shape
        N = kv.shape[1]
        H = self.n_heads

        # projections
        Q = self.W_q(self.norm_q(q)).reshape(B, P, H, self.d_head).transpose(1, 2)   # (B,H,P,d_h)
        K = self.W_k(self.norm_kv(kv)).reshape(B, N, H, self.d_head).transpose(1, 2) # (B,H,N,d_h)
        V = self.W_v(self.norm_kv(kv)).reshape(B, N, H, self.d_head).transpose(1, 2) # (B,H,N,d_h)

        # scaled dot-product + relative bias
        attn = (Q @ K.transpose(-2, -1)) * self.scale          # (B,H,P,N)
        bias = rel_bias.permute(0, 3, 1, 2)                    # (B,H,P,N)
        attn = attn + bias

        if kv_mask is not None:
            # mask padded entities: set their logits to -inf
            attn = attn.masked_fill(
                kv_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn   = self.drop(torch.softmax(attn, dim=-1))
        out    = (attn @ V).transpose(1, 2).reshape(B, P, d)   # (B,P,d)
        q      = q + self.W_o(out)
        q      = q + self.ffn(self.norm_ff(q))
        return q


# ─────────────────────────────────────────────────────────────────────────────
# Full Model
# ─────────────────────────────────────────────────────────────────────────────

class SocialCostField(nn.Module):
    """
    Language-conditioned Neural Social Cost Field.

    Pipeline:
        Scene Graph (nodes + text edges)
              ↓
        GraphTransformerEncoder        — entities reason about each other
              ↓
        H  (B, N, d)                   — per-entity context (no bottleneck)
              ↓
        CostDecoder  cross-attn(xy, H) — each point attends to relevant entities
              ↓
        cost  (B, P)  ∈ [0,1]         — continuous field, any resolution

    Scales to arbitrary N: adding more agents only costs more attention
    computation, not more parameters.
    """

    def __init__(self, d_model: int = 64, n_heads: int = 4,
                 n_enc_layers: int = 3, n_dec_layers: int = 2,
                 n_freq: int = 16, dropout: float = 0.05) -> None:
        super().__init__()
        self.encoder = GraphTransformerEncoder(
            d_model, n_heads, n_enc_layers, dropout)
        self.decoder = CostDecoder(
            d_model, n_heads, n_freq, n_dec_layers, dropout)

    def encode(self, node_feats, edge_feats, node_mask, entity_pos
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Scene → (H, entity_pos, entity_heading).  Call once per scene change.

        entity_pos : (B, N, 2) raw world-space positions in metres — passed through
                     directly, never normalized. Decouples model from coordinate system.

        Returns
        -------
        H              : (B, N, d)  per-entity context embeddings
        entity_pos     : (B, N, 2)  world-space positions (metres)
        entity_heading : (B, N)     headings in radians
        """
        H = self.encoder(node_feats, edge_feats, node_mask)
        # node_feats layout: [cos_h, sin_h, act×4, type×3] — no x/y
        entity_heading = torch.atan2(node_feats[..., 1],
                                     node_feats[..., 0])            # (B, N)
        return H, entity_pos, entity_heading

    def query(self, xy: torch.Tensor,
              H: torch.Tensor,
              entity_pos: torch.Tensor,
              entity_heading: torch.Tensor,
              entity_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Query cost at P arbitrary points given pre-computed encode() output."""
        return self.decoder(xy, H, entity_pos, entity_heading, entity_mask)

    def forward(self, node_feats, edge_feats, node_mask, node_pos,
                xy: torch.Tensor) -> torch.Tensor:
        """End-to-end: scene + query points → costs."""
        H, entity_pos, entity_heading = self.encode(node_feats, edge_feats, node_mask, node_pos)
        return self.query(xy, H, entity_pos, entity_heading, node_mask)

    @torch.no_grad()
    def predict_grid(self, graph: SceneGraph,
                     grid_h: int = 12, grid_w: int = 20,
                     x_range: tuple[float, float] | None = None,
                     y_range: tuple[float, float] | None = None) -> np.ndarray:
        """
        Given a SceneGraph return a (grid_h, grid_w) numpy costmap.
        Row 0 = north (y_max), col 0 = west (x_min).
        Any resolution — same model, just more query points.

        x_range / y_range: world coordinate bounds (metres).
        Defaults to module-level X_MIN/X_MAX, Y_MIN/Y_MAX if not given.
        """
        self.eval()
        nf, ef, mask, pos = graph.to_tensors()
        nf, ef, mask, pos = nf.unsqueeze(0), ef.unsqueeze(0), mask.unsqueeze(0), pos.unsqueeze(0)

        H, entity_pos, entity_heading = self.encode(nf, ef, mask, pos)

        x0, x1 = x_range if x_range else (X_MIN, X_MAX)
        y0, y1 = y_range if y_range else (Y_MIN, Y_MAX)

        xs = torch.linspace(x0, x1, grid_w)
        ys = torch.linspace(y1, y0, grid_h)   # row 0 = y_max (north)
        gx, gy = torch.meshgrid(xs, ys, indexing="xy")
        xy = torch.stack([gx, gy], dim=-1).reshape(1, -1, 2)

        cost = self.query(xy, H, entity_pos, entity_heading, mask).reshape(grid_h, grid_w)
        return cost.cpu().numpy()

    def save(self, path: str | Path) -> None:
        torch.save({"state_dict": self.state_dict(),
                    "config":     self._config()}, path)
        print(f"[SocialCostField] saved → {path}  "
              f"({sum(p.numel() for p in self.parameters()):,} params)")

    @classmethod
    def load(cls, path: str | Path) -> "SocialCostField":
        ckpt  = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(**ckpt.get("config", {}))
        missing, unexpected = model.load_state_dict(
            ckpt["state_dict"], strict=False)
        if missing or unexpected:
            print(f"[SocialCostField] ⚠ checkpoint architecture mismatch "
                  f"(missing={len(missing)}, unexpected={len(unexpected)}). "
                  f"Some weights randomly initialized — retrain recommended.")
        model.eval()
        print(f"[SocialCostField] loaded ← {path}  "
              f"({sum(p.numel() for p in model.parameters()):,} params)")
        return model

    def _config(self) -> dict:
        return dict(
            d_model      = self.encoder.d_model,
            n_heads      = self.encoder.n_heads,
            n_enc_layers = len(self.encoder.layers),
            n_dec_layers = len(self.decoder.ca_layers),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic dataset  (SceneGraph + sampled points + rule-based labels)
# ─────────────────────────────────────────────────────────────────────────────

# Rich text variations per relation type — keys match HumanSSG "relationship" field exactly
_RELATION_VARIANTS: dict[str, list[tuple[str, float]]] = {
    "conversing with": [
        ("two people actively conversing with each other face to face", 0.85),
        ("two people talking together in a focused face-to-face conversation", 0.85),
        ("a pair engaged in mutual conversation, clearly interacting", 0.80),
    ],
    "looking towards": [
        ("one person looking towards another person, directing attention at them", 0.50),
        ("a person facing and looking at someone else nearby", 0.50),
        ("one individual directing their gaze toward another person", 0.45),
    ],
    "observing": [
        ("one person observing another person from a distance, watching passively", 0.30),
        ("a person quietly watching someone else without direct interaction", 0.30),
        ("one person keeping an eye on another from across the room", 0.25),
    ],
    "standing beside": [
        ("two people standing side by side, physically close but not necessarily interacting", 0.40),
        ("two individuals standing right next to each other shoulder to shoulder", 0.40),
        ("a pair standing together side by side in close proximity", 0.35),
    ],
    "standing near": [
        ("two people standing near each other in the same area", 0.25),
        ("two individuals in the same space, not far apart but no direct interaction", 0.25),
        ("two people positioned near each other in a shared space", 0.20),
    ],
    "standing next to": [
        ("two people standing next to each other, in close proximity", 0.40),
        ("two individuals positioned right beside each other", 0.40),
        ("a person standing directly next to another person", 0.35),
    ],
    "viewing": [
        ("one person viewing another person or watching what they are doing", 0.30),
        ("a person directing their view toward someone or something nearby", 0.30),
        ("one person watching or viewing the actions of another", 0.25),
    ],
    "watching": [
        ("one person watching another person or a shared focal point such as a TV", 0.35),
        ("a person watching TV or a screen together with someone nearby", 0.35),
        ("two people both focused on the same thing, watching together", 0.35),
    ],
}


def _random_scene_graph(rng: np.random.Generator) -> SceneGraph:
    """Generate one random SceneGraph with varied natural language edge descriptions."""
    g = SceneGraph()

    n_agents = int(rng.integers(1, 5))
    for i in range(n_agents):
        x = float(rng.uniform(X_MIN + 0.5, X_MAX - 0.5))
        y = float(rng.uniform(Y_MIN + 0.4, Y_MAX - 0.4))
        h = float(rng.uniform(0, 360))
        a = rng.choice(list(ACTIVITY_IDS.keys()))
        g.add_node([x, y], h, a, TYPE_AGENT)

    # robot + goal
    rx = float(rng.uniform(2.5, 4.5)) * rng.choice([-1, 1])
    ry = float(rng.uniform(Y_MIN + 0.4, Y_MAX - 0.4))
    g.add_node([rx, ry], 0.0, "standing", TYPE_ROBOT)
    gx = float(rng.uniform(2.5, 4.5)) * -np.sign(rx)
    gy = float(rng.uniform(Y_MIN + 0.4, Y_MAX - 0.4))
    g.add_node([gx, gy], 0.0, "standing", TYPE_GOAL)

    # random relations — sample a variant text for each edge
    agent_ids = list(range(n_agents))
    rng.shuffle(agent_ids)
    rel_keys = list(_RELATION_VARIANTS.keys())

    for i in range(0, len(agent_ids) - 1, 2):
        rel_key = rng.choice(rel_keys)
        variants = _RELATION_VARIANTS[rel_key]
        desc, intensity = variants[int(rng.integers(len(variants)))]
        g.add_edge(agent_ids[i], agent_ids[i + 1],
                   relation=rel_key, description=desc, intensity=intensity)

        if any(kw in rel_key for kw in ("chat", "conversation", "whisper", "argument")):
            g.nodes[agent_ids[i]]["activity"]     = "talking"
            g.nodes[agent_ids[i + 1]]["activity"] = "talking"
        elif any(kw in rel_key for kw in ("walking", "following")):
            g.nodes[agent_ids[i]]["activity"]     = "walking"
            g.nodes[agent_ids[i + 1]]["activity"] = "walking"

    return g


def _rule_cost(graph: SceneGraph, xy: np.ndarray) -> np.ndarray:
    """
    Rule-based social cost at query points xy (N,2).
    Used as training labels.

    Per-agent anisotropic Gaussian, intensity scaled by relation.
    """
    cost = np.zeros(len(xy), dtype=np.float32)

    # gather agent social params
    for ni, nd in enumerate(graph.nodes):
        if nd["entity_type"] != TYPE_AGENT:
            continue

        pos = nd["pos"]
        h   = math.radians(nd["heading_deg"])
        fx, fy = math.cos(h), math.sin(h)

        dx = xy[:, 0] - pos[0]
        dy = xy[:, 1] - pos[1]

        along = dx * fx + dy * fy
        perp  = -dx * fy + dy * fx

        # base params by activity
        act = nd["activity"]
        score = {"standing": 0.5, "walking": 0.7,
                 "talking": 0.85, "sitting": 0.55}.get(act, 0.5)
        ps    = {"standing": 0.8, "walking": 1.1,
                 "talking": 1.2, "sitting": 0.9}.get(act, 0.9)
        os_   = {"standing": 1.3, "walking": 1.8,
                 "talking": 2.5, "sitting": 1.5}.get(act, 1.5)

        # boost by relation intensity
        for e in graph.edges:
            if (e["src"] == ni or e["dst"] == ni):
                intensity = e["intensity"]
                score = max(score, 0.5 + 0.5 * intensity)
                ps    = ps * (1 + 0.5 * intensity)
                os_   = os_ * (1 + 0.3 * intensity)
                break

        sigma_front = ps * os_
        sigma_back  = ps
        sigma_side  = ps

        sigma_along = np.where(along >= 0, sigma_front, sigma_back)
        c = score * np.exp(
            -(along ** 2) / (2 * sigma_along ** 2)
            -(perp  ** 2) / (2 * sigma_side  ** 2)
        )
        cost = np.maximum(cost, c.astype(np.float32))

    # conversation group: high cost between talking pairs
    for e in graph.edges[::2]:   # skip reverse edges
        if e["intensity"] > 0.5 and e["src"] < len(graph.nodes) and e["dst"] < len(graph.nodes):
            pa = graph.nodes[e["src"]]["pos"]
            pb = graph.nodes[e["dst"]]["pos"]
            mid = (pa + pb) / 2
            dx  = xy[:, 0] - mid[0]
            dy  = xy[:, 1] - mid[1]
            sigma_grp = 0.5 + 0.3 * e["intensity"]
            c   = e["intensity"] * np.exp(-(dx**2 + dy**2) / (2 * sigma_grp**2))
            cost = np.maximum(cost, c.astype(np.float32))

    return np.clip(cost, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# LLM Archetype Labeling  (call once, cache params, reuse for all scenes)
# ─────────────────────────────────────────────────────────────────────────────

_ARCHETYPE_PROMPT = """\
You are a social navigation expert for mobile robots.

In social navigation research, personal space around a person is modeled as an \
anisotropic Gaussian with three directional radii:
- sigma_front : personal space in the heading/gaze direction (largest)
- sigma_back  : personal space directly behind (smallest, person cannot see)
- sigma_side  : personal space to the left and right

For conversing pairs, an additional "o-space" (interaction space) is modeled as \
a Gaussian centered between the two people.

Social situation: "{description}"
Relation type: {rel_key}

Output ONLY valid JSON:
{{
  "sigma_front": <float, metres, personal space radius in front — direction the person faces/gazes>,
  "sigma_back":  <float, metres, personal space radius behind — person unaware of robot here>,
  "sigma_side":  <float, metres, personal space radius to the sides>,
  "peak_cost":   <float 0.0–1.0, cost at person's position; 1.0=absolute avoidance, 0.0=free>,
  "o_space_sigma": <float, metres, radius of interaction space between the pair (0 if single person or strangers)>,
  "o_space_cost":  <float 0.0–1.0, peak cost of o-space between conversing people>,
  "reasoning": "<one sentence grounded in social norms>"
}}

Guidelines:
- sigma_front > sigma_side > sigma_back  (gaze direction is most sensitive)
- Typical ranges: sigma_front 0.8–2.5m, sigma_side 0.5–1.5m, sigma_back 0.3–0.8m
- heated argument / private whisper → large sigma, high peak_cost, large o_space
- strangers / passing → small sigma, low peak_cost, o_space_sigma=0
"""


def llm_label_archetypes(
    llm_model: str = "kimi-k2-turbo-preview",
    save_path: str | Path = "checkpoints/archetype_params.json",
    verbose: bool = True,
) -> dict:
    """
    为 _RELATION_VARIANTS 中每种原型调用 LLM 打标签，缓存到 JSON。

    返回 dict:
        {rel_key: {sigma_front, sigma_back, sigma_side, peak_cost,
                   o_space_sigma, o_space_cost, reasoning, description}}
    """
    import json as _json
    from experiments.social_nav.llm_costmap import _call_llm

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 如果已有缓存，直接加载
    if save_path.exists():
        with open(save_path) as f:
            cached = _json.load(f)
        if verbose:
            print(f"[archetypes] loaded cached params ← {save_path}  "
                  f"({len(cached)} archetypes)")
        return cached

    params = {}
    rel_keys = list(_RELATION_VARIANTS.keys())

    for rel_key in rel_keys:
        # 用第一个 variant 描述作为代表
        description = _RELATION_VARIANTS[rel_key][0][0]

        prompt = _ARCHETYPE_PROMPT.format(
            description=description, rel_key=rel_key)

        if verbose:
            print(f"[archetypes] labeling '{rel_key}' ...")

        try:
            response = _call_llm(prompt, llm_model)
            # 提取 JSON
            start = response.find("{")
            end   = response.rfind("}") + 1
            data  = _json.loads(response[start:end])
            data["description"] = description
            params[rel_key] = data
            if verbose:
                print(f"  peak_cost={data['peak_cost']:.2f}  "
                      f"σ_front={data['sigma_front']:.1f}m  "
                      f"σ_side={data['sigma_side']:.1f}m  "
                      f"σ_back={data['sigma_back']:.1f}m  "
                      f"o_space={data['o_space_sigma']:.1f}m  "
                      f"→ {data['reasoning']}")
        except Exception as e:
            print(f"  [WARNING] failed for '{rel_key}': {e}, using rule fallback")
            intensity = RELATION_INTENSITY.get(rel_key, 0.5)
            params[rel_key] = {
                "sigma_front":    0.8 + intensity * 1.2,
                "sigma_back":     0.3 + intensity * 0.4,
                "sigma_side":     0.5 + intensity * 0.8,
                "peak_cost":      intensity,
                "o_space_sigma":  0.5 * intensity if "convers" in rel_key or "talking" in rel_key else 0.0,
                "o_space_cost":   intensity,
                "description":    description,
                "reasoning":      "fallback rule",
            }

    # 保存
    with open(save_path, "w") as f:
        _json.dump(params, f, indent=2, ensure_ascii=False)
    if verbose:
        print(f"[archetypes] saved → {save_path}")

    return params


def _llm_cost(graph: SceneGraph, xy: np.ndarray,
              archetype_params: dict) -> np.ndarray:
    """
    LLM-grounded social cost at query points xy (N, 2).
    Uses per-archetype parameters from llm_label_archetypes().

    Replaces _rule_cost() when archetype_params is available.
    """
    cost = np.zeros(len(xy), dtype=np.float32)

    for ni, nd in enumerate(graph.nodes):
        if nd["entity_type"] != TYPE_AGENT:
            continue

        pos = nd["pos"]
        h   = math.radians(nd["heading_deg"])
        fx, fy = math.cos(h), math.sin(h)

        dx = xy[:, 0] - pos[0]
        dy = xy[:, 1] - pos[1]
        along = dx * fx + dy * fy
        perp  = -dx * fy + dy * fx

        # 找这个 agent 对应的关系类型
        rel_key = "strangers"
        for e in graph.edges:
            if e["src"] == ni or e["dst"] == ni:
                rel_key = e.get("relation", "strangers")
                break

        p = archetype_params.get(rel_key, archetype_params.get("strangers", {}))
        sigma_front = float(p.get("sigma_front", 1.2))
        sigma_back  = float(p.get("sigma_back",  0.5))
        sigma_side  = float(p.get("sigma_side",  0.8))
        peak_cost   = float(p.get("peak_cost",   0.5))

        # anisotropic Gaussian: front/back split along heading axis
        sigma_along = np.where(along >= 0, sigma_front, sigma_back)
        c = peak_cost * np.exp(
            -(along ** 2) / (2 * sigma_along ** 2)
            -(perp  ** 2) / (2 * sigma_side  ** 2)
        )
        cost = np.maximum(cost, c.astype(np.float32))

    # o-space: interaction zone between conversing pairs (Kendon 1990)
    seen_pairs: set = set()
    for e in graph.edges:
        s, d = e["src"], e["dst"]
        if s >= len(graph.nodes) or d >= len(graph.nodes):
            continue
        pair = (min(s, d), max(s, d))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        rel_key  = e.get("relation", "strangers")
        p        = archetype_params.get(rel_key, {})
        o_sigma  = float(p.get("o_space_sigma", 0.0))
        o_cost   = float(p.get("o_space_cost",  0.0))
        if o_sigma < 0.05 or o_cost < 0.01:
            continue

        pa  = graph.nodes[s]["pos"]
        pb  = graph.nodes[d]["pos"]
        mid = (pa + pb) / 2.0
        dx  = xy[:, 0] - mid[0]
        dy  = xy[:, 1] - mid[1]
        c   = o_cost * np.exp(-(dx**2 + dy**2) / (2 * o_sigma**2))
        cost = np.maximum(cost, c.astype(np.float32))

    return np.clip(cost, 0.0, 1.0)


def _warmup_embeddings():
    """Pre-compute all variant embeddings so dataset generation is fast."""
    all_texts = []
    for variants in _RELATION_VARIANTS.values():
        for desc, _ in variants:
            all_texts.append(desc)
    # also canonical templates
    all_texts += list(RELATION_TEMPLATES.values())
    all_texts += ["strangers"]
    all_texts = list(set(all_texts))
    embs = encode_text(all_texts)
    for text, emb in zip(all_texts, embs):
        _RELATION_EMBEDDINGS[text] = emb
    print(f"[warmup] pre-computed {len(all_texts)} embeddings")


def generate_dataset(n_scenes: int, n_points: int = 512, seed: int = 0,
                     archetype_params: dict | None = None,
                     verbose: bool = True):
    """
    For each scene, sample n_points random (x,y) positions and compute labels.

    Returns
    -------
    node_feats : (N_scenes, MAX_N, NODE_DIM)
    edge_feats : (N_scenes, MAX_N, MAX_N, EDGE_DIM)
    node_masks : (N_scenes, MAX_N)
    node_pos   : (N_scenes, MAX_N, 2)
    xy_pts     : (N_scenes, n_points, 2)
    costs      : (N_scenes, n_points)
    """
    MAX_N = 12
    rng   = np.random.default_rng(seed)
    _warmup_embeddings()   # batch pre-compute all texts → fast lookup
    t0    = time.time()

    all_nf, all_ef, all_mask, all_pos, all_xy, all_cost = [], [], [], [], [], []

    for i in range(n_scenes):
        g  = _random_scene_graph(rng)
        nf, ef, mask, pos = g.to_tensors(MAX_N)

        # sample query points — raw world coords, any range
        xs = rng.uniform(X_MIN, X_MAX, n_points).astype(np.float32)
        ys = rng.uniform(Y_MIN, Y_MAX, n_points).astype(np.float32)
        xy = np.stack([xs, ys], axis=-1)   # (P, 2)

        if archetype_params is not None:
            cost = _llm_cost(g, xy, archetype_params)
        else:
            cost = _rule_cost(g, xy)

        all_nf.append(nf);   all_ef.append(ef)
        all_mask.append(mask); all_pos.append(pos)
        all_xy.append(torch.tensor(xy))
        all_cost.append(torch.tensor(cost))

        if verbose and (i + 1) % max(1, n_scenes // 10) == 0:
            elapsed = time.time() - t0
            eta     = elapsed / (i + 1) * (n_scenes - i - 1)
            print(f"  [{i+1:5d}/{n_scenes}]  {elapsed:.1f}s  ETA={eta:.1f}s")

    return (torch.stack(all_nf),  torch.stack(all_ef),
            torch.stack(all_mask), torch.stack(all_pos),
            torch.stack(all_xy),   torch.stack(all_cost))


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(n_scenes: int = 8000, epochs: int = 60, batch_size: int = 64,
          lr: float = 3e-4, save_path: str = "checkpoints/scf2.pt",
          seed: int = 42, llm_labels: bool = False,
          llm_model: str = "kimi-k2-turbo-preview") -> "SocialCostField":

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] device={device}  n_scenes={n_scenes}  epochs={epochs}  "
          f"llm_labels={llm_labels}")

    archetype_params = None
    if llm_labels:
        print(f"[Train] Fetching LLM archetype labels (model={llm_model})…")
        archetype_params = llm_label_archetypes(llm_model=llm_model)

    print("[Train] Generating dataset…")
    nf, ef, mask, pos, xy, cost = generate_dataset(
        n_scenes, seed=seed, archetype_params=archetype_params)

    n_val   = max(1, n_scenes // 10)
    n_train = n_scenes - n_val
    perm    = torch.randperm(n_scenes)
    tr, va  = perm[:n_train], perm[n_train:]

    def make_dl(idx, shuffle):
        ds = torch.utils.data.TensorDataset(
            nf[idx], ef[idx], mask[idx], pos[idx], xy[idx], cost[idx])
        return torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle)

    tr_dl = make_dl(tr, True)
    va_dl = make_dl(va, False)

    model = SocialCostField().to(device)
    n_p   = sum(p.numel() for p in model.parameters())
    print(f"[Train] SocialCostField  {n_p:,} parameters")

    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val  = float("inf")
    ckpt_path = Path(save_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for b_nf, b_ef, b_mask, b_pos, b_xy, b_cost in tr_dl:
            b_nf, b_ef, b_mask = b_nf.to(device), b_ef.to(device), b_mask.to(device)
            b_pos, b_xy, b_cost = b_pos.to(device), b_xy.to(device), b_cost.to(device)

            pred  = model(b_nf, b_ef, b_mask, b_pos, b_xy)   # (B, P)
            loss  = F.mse_loss(pred, b_cost)

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += float(loss) * len(b_nf)
        tr_loss /= n_train;  sched.step()

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for b_nf, b_ef, b_mask, b_pos, b_xy, b_cost in va_dl:
                b_nf, b_ef, b_mask = b_nf.to(device), b_ef.to(device), b_mask.to(device)
                b_pos, b_xy, b_cost = b_pos.to(device), b_xy.to(device), b_cost.to(device)
                va_loss += float(F.mse_loss(model(b_nf, b_ef, b_mask, b_pos, b_xy),
                                            b_cost)) * len(b_nf)
        va_loss /= n_val

        if va_loss < best_val:
            best_val = va_loss
            torch.save({"state_dict": model.state_dict(),
                        "config":     model._config()}, ckpt_path)
            tag = " ★"
        else:
            tag = ""

        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}/{epochs}  "
                  f"tr={tr_loss:.5f}  va={va_loss:.5f}{tag}")

    print(f"[Train] done.  best val MSE = {best_val:.5f}")
    return SocialCostField.load(ckpt_path)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation & Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def _ascii(cm: np.ndarray, label: str):
    chars = " ░▒▓█"
    print(f"\n── {label} ──")
    for r in range(cm.shape[0]):
        print("".join(chars[min(int(v * len(chars)), len(chars)-1)] * 2
                       for v in cm[r]))


def eval_model(model: SocialCostField, seed: int = 7):
    rng = np.random.default_rng(seed)

    # build a semantically rich test scene
    g = SceneGraph()
    a0 = g.add_node([-1.5,  0.5], heading_deg=270, activity="talking")
    a1 = g.add_node([-1.5, -0.5], heading_deg= 90, activity="talking")
    a2 = g.add_node([ 1.5,  0.5], heading_deg=180, activity="walking")
    g.add_node([3.5, 0.0], entity_type=TYPE_ROBOT)
    g.add_node([-3.5, 0.0], entity_type=TYPE_GOAL)

    # two very different social situations → same spatial layout, different edges
    g_casual = SceneGraph()
    g_casual.nodes = [n.copy() for n in g.nodes]
    g_casual.add_edge(a0, a1,
        relation="standing near",
        description=RELATION_TEMPLATES["standing near"],
        intensity=RELATION_INTENSITY["standing near"])

    g_heated = SceneGraph()
    g_heated.nodes = [n.copy() for n in g.nodes]
    g_heated.add_edge(a0, a1,
        relation="conversing with",
        description=RELATION_TEMPLATES["conversing with"],
        intensity=RELATION_INTENSITY["conversing with"])

    cm_casual = model.predict_grid(g_casual)
    cm_heated = model.predict_grid(g_heated)

    # rule baseline
    from experiments.social_nav.llm_costmap import build_live_costmap
    from types import SimpleNamespace
    agents = {
        "A0": SimpleNamespace(pos=np.array([-1.5, 0.5]), heading_deg=270, activity="talking"),
        "A1": SimpleNamespace(pos=np.array([-1.5,-0.5]), heading_deg= 90, activity="talking"),
        "A2": SimpleNamespace(pos=np.array([ 1.5, 0.5]), heading_deg=180, activity="walking"),
    }
    cm_rule, _ = build_live_costmap(
        agents, (12, 20), x_range=(X_MIN, X_MAX), y_range=(Y_MIN, Y_MAX),
        method="rule", groups=[["A0", "A1"]])

    _ascii(cm_rule,   "Rule-based baseline")
    _ascii(cm_casual, "SocialCostField  —  standing near")
    _ascii(cm_heated, "SocialCostField  —  conversing with")

    print(f"\nMax cost difference (conversing vs standing near): "
          f"{float(np.abs(cm_heated - cm_casual).max()):.4f}")
    print(f"Mean cost (standing near):   {cm_casual.mean():.4f}")
    print(f"Mean cost (conversing with): {cm_heated.mean():.4f}")

    # timing
    nf, ef, mask, pos = g_heated.to_tensors()
    nf, ef, mask, pos = nf.unsqueeze(0), ef.unsqueeze(0), mask.unsqueeze(0), pos.unsqueeze(0)
    xs = torch.linspace(X_MIN, X_MAX, 20)
    ys = torch.linspace(Y_MAX, Y_MIN, 12)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    xy_q = torch.stack([gx, gy], -1).reshape(1, -1, 2)

    with torch.no_grad():
        H, entity_pos, entity_heading = model.encode(nf, ef, mask, pos)
        n = 500
        t0 = time.perf_counter()
        for _ in range(n):
            model.query(xy_q, H, entity_pos, entity_heading)
        ms_query = (time.perf_counter() - t0) / n * 1000

        t0 = time.perf_counter()
        for _ in range(n):
            model.encode(nf, ef, mask, pos)
        ms_encode = (time.perf_counter() - t0) / n * 1000

    print(f"\nTiming:")
    print(f"  encode (scene → z):  {ms_encode:.2f} ms  (once per scene change)")
    print(f"  query  (z → costmap): {ms_query:.2f} ms  (per navigation step)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train",      action="store_true")
    p.add_argument("--eval",       action="store_true")
    p.add_argument("--label-archetypes", action="store_true",
                   help="只调用 LLM 打原型标签，不训练")
    p.add_argument("--n-scenes",   type=int, default=8000)
    p.add_argument("--epochs",     type=int, default=60)
    p.add_argument("--save",       type=str, default="checkpoints/scf2.pt")
    p.add_argument("--load",       type=str, default=None)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--llm-labels", action="store_true",
                   help="用 LLM 打标签替代 rule-based（需要 KIMI_API_KEY）")
    p.add_argument("--llm-model",  type=str, default="kimi-k2-turbo-preview")
    args = p.parse_args()

    if args.label_archetypes:
        llm_label_archetypes(llm_model=args.llm_model)

    elif args.train:
        model = train(n_scenes=args.n_scenes, epochs=args.epochs,
                      save_path=args.save, seed=args.seed,
                      llm_labels=args.llm_labels, llm_model=args.llm_model)
        eval_model(model, seed=args.seed)
    elif args.eval:
        model = SocialCostField.load(args.load or args.save)
        eval_model(model, seed=args.seed)
    else:
        p.print_help()
