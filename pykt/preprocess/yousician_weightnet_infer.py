"""
WeightNet 推理：与 data/yousician 下训练的 checkpoint 兼容（notebook 与 weightnet_train_predict 两种格式）。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class FingeringState:
    fret: Tuple[int, int, int, int]
    strings: Tuple[int, ...]

    @staticmethod
    def from_lists(fret: Sequence[int], strings: Sequence[int]) -> "FingeringState":
        if len(fret) != 4:
            raise ValueError("fret must be length 4")
        fret = tuple(int(x) for x in fret)
        strings = tuple(sorted(set(int(s) for s in strings)))
        if not (1 <= len(strings) <= 4):
            raise ValueError("strings length must be in [1, 4]")
        for f in fret:
            if f < 0:
                raise ValueError("fret cannot be negative")
        for s in strings:
            if s not in (0, 1, 2, 3):
                raise ValueError("string index must be 0..3")
        return FingeringState(fret=fret, strings=strings)


def active_positions(
    state: FingeringState, only_played_strings: bool = False
) -> List[Tuple[int, int]]:
    played = set(state.strings)
    pos = []
    for s, f in enumerate(state.fret):
        if f > 0 and ((not only_played_strings) or (s in played)):
            pos.append((s, f))
    return pos


def state_cost_torch(
    state: FingeringState,
    w3: torch.Tensor,
    w4: torch.Tensor,
    w5: torch.Tensor,
    gamma: torch.Tensor,
    device: torch.device,
    only_played_strings: bool = False,
) -> torch.Tensor:
    pos = active_positions(state, only_played_strings=only_played_strings)
    if not pos:
        avg_active_fret = torch.tensor(0.0, device=device)
        fret_span = torch.tensor(0.0, device=device)
        count_nonzero = torch.tensor(0.0, device=device)
    else:
        active_frets = torch.tensor([f for _, f in pos], dtype=torch.float32, device=device)
        avg_active_fret = active_frets.mean()
        fret_span = active_frets.max() - active_frets.min()
        count_nonzero = torch.tensor(float(len(pos)), dtype=torch.float32, device=device)

    return (
        w3 * torch.log1p(torch.relu(avg_active_fret - gamma))
        + w4 * count_nonzero
        + w5 * fret_span
    )


def transition_cost_torch(
    prev_state: FingeringState,
    curr_state: FingeringState,
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    device: torch.device,
    only_played_strings: bool = False,
) -> torch.Tensor:
    prev_pos = active_positions(prev_state, only_played_strings=only_played_strings)
    curr_pos = active_positions(curr_state, only_played_strings=only_played_strings)

    m = len(prev_pos)
    n = len(curr_pos)

    dtype = w0.dtype
    inf = torch.tensor(1e9, device=device, dtype=dtype)
    zero = torch.tensor(0.0, device=device, dtype=dtype)

    if n == 0:
        return zero
    if m == 0:
        return w0 * n

    pair_cost = []
    for i in range(m):
        prev_s, prev_f = prev_pos[i]
        row = []
        for j in range(n):
            curr_s, curr_f = curr_pos[j]
            row.append(w1 * abs(curr_s - prev_s) + w2 * abs(curr_f - prev_f))
        pair_cost.append(row)

    dp = {0: zero}
    for j in range(n):
        nxt = {}
        for mask, base_cost in dp.items():
            cand_new = base_cost + w0
            if mask not in nxt:
                nxt[mask] = cand_new
            else:
                nxt[mask] = torch.minimum(nxt[mask], cand_new)

            for i in range(m):
                if (mask >> i) & 1:
                    continue
                new_mask = mask | (1 << i)
                cand_match = base_cost + pair_cost[i][j]
                if new_mask not in nxt:
                    nxt[new_mask] = cand_match
                else:
                    nxt[new_mask] = torch.minimum(nxt[new_mask], cand_match)
        dp = nxt

    best = inf
    for v in dp.values():
        best = torch.minimum(best, v)
    return best


def per_step_costs_torch(
    states: List[FingeringState],
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    w4: torch.Tensor,
    w5: torch.Tensor,
    gamma: torch.Tensor,
    device: torch.device,
    only_played_strings: bool = False,
) -> torch.Tensor:
    if not states:
        return torch.tensor([], device=device, dtype=torch.float32)

    costs = []
    c0 = state_cost_torch(states[0], w3, w4, w5, gamma, device, only_played_strings)
    costs.append(c0)
    for i in range(1, len(states)):
        t = transition_cost_torch(
            states[i - 1], states[i], w0, w1, w2, device, only_played_strings
        )
        s = state_cost_torch(states[i], w3, w4, w5, gamma, device, only_played_strings)
        costs.append(t + s)
    return torch.stack(costs)


def song_features(states: List[FingeringState]) -> torch.Tensor:
    if not states:
        return torch.zeros(8, dtype=torch.float32)

    n_notes = len(states)
    nonzero_counts = []
    mean_frets = []
    spans = []
    played_counts = []

    for st in states:
        nz = [x for x in st.fret if x > 0]
        nonzero_counts.append(len(nz))
        mean_frets.append((sum(nz) / len(nz)) if nz else 0.0)
        spans.append((max(nz) - min(nz)) if len(nz) > 1 else 0.0)
        played_counts.append(len(st.strings))

    feat = torch.tensor(
        [
            float(n_notes),
            float(sum(nonzero_counts) / n_notes),
            float(sum(mean_frets) / n_notes),
            float(sum(spans) / n_notes),
            float(sum(played_counts) / n_notes),
            float(max(nonzero_counts)),
            float(max(mean_frets)),
            float(max(spans)),
        ],
        dtype=torch.float32,
    )

    feat[0] = torch.log1p(feat[0])
    return feat


class WeightNet(nn.Module):
    def __init__(self, in_dim: int = 8, hidden: int = 32):
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.param_head = nn.Linear(hidden, 7)
        self.diff_head = nn.Linear(hidden + 1, 1)

    def encode_params(self, feat: torch.Tensor):
        h = self.backbone(feat)
        raw = self.param_head(h)
        w = F.softplus(raw[:6]) + 1e-4
        gamma = F.softplus(raw[6])
        return h, w, gamma

    def forward_from_cost(self, h: torch.Tensor, song_cost: torch.Tensor) -> torch.Tensor:
        return self.diff_head(torch.cat([h, song_cost.view(1)], dim=0)).squeeze(0)


def predict_per_note_difficulties(
    model: WeightNet,
    states: List[FingeringState],
    device: torch.device,
    only_played_strings: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not states:
        z = torch.tensor(0.0, device=device)
        return z, torch.tensor([], device=device)

    feat = song_features(states).to(device)
    with torch.inference_mode():
        h, w, gamma = model.encode_params(feat)
        step_costs = per_step_costs_torch(
            states,
            w0=w[0],
            w1=w[1],
            w2=w[2],
            w3=w[3],
            w4=w[4],
            w5=w[5],
            gamma=gamma,
            device=device,
            only_played_strings=only_played_strings,
        )
        song_cost = step_costs.mean()
        pred = model.forward_from_cost(h, song_cost)
        mean_c = step_costs.mean().clamp(min=1e-8)
        per_note = pred * (step_costs / mean_c)
    return pred, per_note


def load_weightnet_bundle(
    ckpt_path: Path | str,
    map_location: torch.device | str | None = None,
) -> Tuple[WeightNet, torch.device, bool]:
    device = map_location or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    only_played = False
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        cfg = ckpt.get("config") or {}
        in_dim = int(cfg.get("in_dim", 8))
        hidden = int(cfg.get("hidden", 32))
        only_played = bool(cfg.get("only_played_strings", False))
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        in_dim = int(ckpt.get("in_dim", 8))
        hidden = int(ckpt.get("hidden", 32))
        only_played = bool(ckpt.get("only_played_strings", False))
    else:
        state_dict = ckpt
        in_dim, hidden = 8, 32

    model = WeightNet(in_dim=in_dim, hidden=hidden).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device, only_played


def default_weightnet_ckpt_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "yousician" / "weightnet_song_difficulty.pt"
