from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


def ensure_torch_available() -> None:
    if not TORCH_AVAILABLE:
        raise ModuleNotFoundError(
            "PyTorch is required for train/predict mode. Please install torch first."
        )


@dataclass(frozen=True)
class FingeringState:
    fret: Tuple[int, int, int, int]
    strings: Tuple[int, ...]

    @staticmethod
    def from_lists(fret: Sequence[int], strings: Sequence[int]) -> "FingeringState":
        if len(fret) != 4:
            raise ValueError("fret must be length 4")
        fret_t = tuple(int(x) for x in fret)
        strings_t = tuple(sorted(set(int(s) for s in strings)))
        if not (1 <= len(strings_t) <= 4):
            raise ValueError("strings length must be in [1, 4]")
        for f in fret_t:
            if f < 0:
                raise ValueError("fret cannot be negative")
        for s in strings_t:
            if s not in (0, 1, 2, 3):
                raise ValueError("string index must be 0..3")
        return FingeringState(fret=fret_t, strings=strings_t)


def active_positions(state: FingeringState, only_played_strings: bool = False) -> List[Tuple[int, int]]:
    played = set(state.strings)
    pos: List[Tuple[int, int]] = []
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
    zero = torch.tensor(0.0, device=device, dtype=w0.dtype)
    inf = torch.tensor(1e9, device=device, dtype=w0.dtype)

    if n == 0:
        return zero
    if m == 0:
        return w0 * n

    pair_cost: List[List[torch.Tensor]] = []
    for i in range(m):
        prev_s, prev_f = prev_pos[i]
        row: List[torch.Tensor] = []
        for j in range(n):
            curr_s, curr_f = curr_pos[j]
            row.append(w1 * abs(curr_s - prev_s) + w2 * abs(curr_f - prev_f))
        pair_cost.append(row)

    dp: Dict[int, torch.Tensor] = {0: zero}
    for j in range(n):
        nxt: Dict[int, torch.Tensor] = {}
        for mask, base_cost in dp.items():
            cand_new = base_cost + w0
            nxt[mask] = cand_new if mask not in nxt else torch.minimum(nxt[mask], cand_new)

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


def full_song_cost_torch(
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
        return torch.tensor(0.0, device=device)

    total = state_cost_torch(states[0], w3, w4, w5, gamma, device, only_played_strings)
    for i in range(1, len(states)):
        total = total + transition_cost_torch(states[i - 1], states[i], w0, w1, w2, device, only_played_strings)
        total = total + state_cost_torch(states[i], w3, w4, w5, gamma, device, only_played_strings)
    return total / len(states)


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
    costs = [state_cost_torch(states[0], w3, w4, w5, gamma, device, only_played_strings)]
    for i in range(1, len(states)):
        t = transition_cost_torch(states[i - 1], states[i], w0, w1, w2, device, only_played_strings)
        s = state_cost_torch(states[i], w3, w4, w5, gamma, device, only_played_strings)
        costs.append(t + s)
    return torch.stack(costs)


def build_song_states(entry: Dict) -> List[FingeringState]:
    frets = entry.get("fret", [])
    strings = entry.get("strings", [])
    n = min(len(frets), len(strings))
    out: List[FingeringState] = []
    for i in range(n):
        try:
            out.append(FingeringState.from_lists(frets[i], strings[i]))
        except Exception:
            continue
    return out


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


if TORCH_AVAILABLE:
    class WeightNet(nn.Module):
        def __init__(self, in_dim: int = 8, hidden: int = 32):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            )
            self.param_head = nn.Linear(hidden, 7)
            self.diff_head = nn.Linear(hidden + 1, 1)
else:
    class WeightNet:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            ensure_torch_available()


def save_weightnet_checkpoint(
    model: WeightNet,
    path: Path,
    *,
    only_played_strings: bool,
    in_dim: int = 8,
    hidden: int = 32,
) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "only_played_strings": only_played_strings,
            "in_dim": in_dim,
            "hidden": hidden,
        },
        path,
    )


def load_weightnet(ckpt_path: Path | str, map_location: torch.device | str | None = None) -> Tuple[WeightNet, torch.device, bool]:
    device = map_location or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        in_dim = int(ckpt.get("in_dim", 8))
        hidden = int(ckpt.get("hidden", 32))
        only_played = bool(ckpt.get("only_played_strings", False))
    else:
        state_dict = ckpt
        in_dim, hidden = 8, 32
        only_played = False
    model = WeightNet(in_dim=in_dim, hidden=hidden).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device, only_played


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
    h = model.backbone(feat)
    raw = model.param_head(h)
    w = F.softplus(raw[:6]) + 1e-4
    gamma = F.softplus(raw[6])
    step_costs = per_step_costs_torch(
        states, w[0], w[1], w[2], w[3], w[4], w[5], gamma, device, only_played_strings
    )
    song_cost = step_costs.mean()
    pred = model.diff_head(torch.cat([h, song_cost.view(1)], dim=0)).squeeze(0)
    mean_c = step_costs.mean().clamp(min=1e-8)
    per_note = pred * (step_costs / mean_c)
    return pred, per_note


def train_model(
    map_path: Path,
    ckpt_path: Path,
    epochs: int,
    lr: float,
    seed: int,
    only_played_strings: bool,
) -> None:
    ensure_torch_available()
    print("[INFO] Start training")
    print(f"[INFO] map_path={map_path}")
    print(f"[INFO] ckpt_path={ckpt_path}")
    print(f"[INFO] epochs={epochs}, lr={lr}, seed={seed}, only_played_strings={only_played_strings}")

    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    if not map_path.exists():
        raise FileNotFoundError(
            f"Cannot find {map_path.resolve()}. "
            "Please generate it with data_profiling_yousician.py first."
        )

    with map_path.open("r", encoding="utf-8") as f:
        song_map = json.load(f)

    samples = []
    for key, entry in song_map.items():
        diff = entry.get("difficulty_level", None)
        if diff is None:
            continue
        states = build_song_states(entry)
        if len(states) < 2:
            continue
        samples.append((key, states, song_features(states), float(diff)))

    print(f"[INFO] usable songs={len(samples)}")
    random.shuffle(samples)
    split = int(len(samples) * 0.8)
    train_samples = samples[:split]
    val_samples = samples[split:]
    print(f"[INFO] train={len(train_samples)}, val={len(val_samples)}")

    model = WeightNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def epoch_step(data: List, train: bool = True) -> Tuple[float, float]:
        model.train(mode=train)
        total_loss, total_mae, n = 0.0, 0.0, 0
        for _, states, feat, y in data:
            feat = feat.to(device)
            y_t = torch.tensor(y, dtype=torch.float32, device=device)
            if train:
                opt.zero_grad()

            h = model.backbone(feat)
            raw = model.param_head(h)
            w = F.softplus(raw[:6]) + 1e-4
            gamma = F.softplus(raw[6])
            song_cost = full_song_cost_torch(
                states, w[0], w[1], w[2], w[3], w[4], w[5], gamma, device, only_played_strings
            )
            pred = model.diff_head(torch.cat([h, song_cost.view(1)], dim=0)).squeeze(0)

            loss = F.mse_loss(pred, y_t) + 1e-4 * (w.pow(2).sum() + gamma.pow(2))
            if train:
                loss.backward()
                opt.step()

            total_loss += float(loss.detach().cpu())
            total_mae += float((pred.detach() - y_t).abs().cpu())
            n += 1
        return total_loss / max(n, 1), total_mae / max(n, 1)

    for ep in range(1, epochs + 1):
        tr_loss, tr_mae = epoch_step(train_samples, train=True)
        va_loss, va_mae = epoch_step(val_samples, train=False)
        if ep == 1 or ep % 5 == 0 or ep == epochs:
            print(
                f"[EPOCH {ep:03d}] train loss={tr_loss:.4f}, mae={tr_mae:.4f} | "
                f"val loss={va_loss:.4f}, mae={va_mae:.4f}"
            )

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    save_weightnet_checkpoint(
        model,
        ckpt_path,
        only_played_strings=only_played_strings,
        in_dim=8,
        hidden=32,
    )
    print(f"[INFO] Model saved to {ckpt_path.resolve()}")


def run_tests() -> None:
    print("[INFO] Running sanity tests...")
    s = FingeringState.from_lists([0, 2, 3, 1], [0, 3])
    assert s.fret == (0, 2, 3, 1)
    assert s.strings == (0, 3)

    pos_all = active_positions(s, only_played_strings=False)
    pos_played = active_positions(s, only_played_strings=True)
    assert pos_all == [(1, 2), (2, 3), (3, 1)]
    assert pos_played == [(3, 1)]
    print("[INFO] Sanity tests passed.")


def predict_from_inputs(
    ckpt_path: Path,
    frets: List[List[int]],
    strings: List[List[int]],
    only_played_strings: bool | None,
) -> None:
    ensure_torch_available()
    print("[INFO] Start prediction")
    print(f"[INFO] ckpt_path={ckpt_path}")
    model, device, ops_default = load_weightnet(ckpt_path)
    ops = ops_default if only_played_strings is None else only_played_strings
    states = build_song_states({"fret": frets, "strings": strings})
    if not states:
        print("[WARN] No valid states from input.")
        return
    pred, per_note = predict_per_note_difficulties(model, states, device, only_played_strings=ops)
    print(f"[RESULT] song_difficulty={float(pred.detach().cpu()):.6f}")
    print(f"[RESULT] per_note_difficulty={json.dumps([float(x) for x in per_note.detach().cpu()])}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executable script converted from loss_function_design.ipynb")
    parser.add_argument("--mode", choices=["train", "test", "predict"], default="test")
    parser.add_argument("--map-path", default="song_info_fret_strings_map.json")
    parser.add_argument("--ckpt-path", default="weightnet_song_difficulty.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--only-played-strings",
        choices=["true", "false", "auto"],
        default="auto",
        help="auto: use checkpoint default for predict; train uses false when auto.",
    )
    parser.add_argument(
        "--frets-json",
        default="",
        help='JSON string for predict mode, e.g. "[[0,2,0,0],[0,2,3,0],[2,0,0,0]]"',
    )
    parser.add_argument(
        "--strings-json",
        default="",
        help='JSON string for predict mode, e.g. "[[1],[1,2],[0]]"',
    )
    return parser.parse_args()


def parse_ops(value: str, for_train: bool) -> bool | None:
    if value == "true":
        return True
    if value == "false":
        return False
    return False if for_train else None


def main() -> None:
    args = parse_args()
    map_path = Path(args.map_path)
    ckpt_path = Path(args.ckpt_path)

    if args.mode == "test":
        run_tests()
        return

    if args.mode == "train":
        train_model(
            map_path=map_path,
            ckpt_path=ckpt_path,
            epochs=args.epochs,
            lr=args.lr,
            seed=args.seed,
            only_played_strings=bool(parse_ops(args.only_played_strings, for_train=True)),
        )
        return

    if args.frets_json and args.strings_json:
        frets = json.loads(args.frets_json)
        strings = json.loads(args.strings_json)
    else:
        print("[INFO] No frets/strings input, using notebook demo example.")
        frets = [[0, 2, 0, 0], [0, 2, 3, 0], [2, 0, 0, 0]]
        strings = [[1], [1, 2], [0]]
    predict_from_inputs(
        ckpt_path=ckpt_path,
        frets=frets,
        strings=strings,
        only_played_strings=parse_ops(args.only_played_strings, for_train=False),
    )


if __name__ == "__main__":
    # 自检
    # python data/yousician/loss_function_design.py --mode test
    # 训练
    # python data/yousician/loss_function_design.py --mode train --map-path song_info_fret_strings_map.json
    # 预测
    # python data/yousician/loss_function_design.py --mode predict --ckpt-path weightnet_song_difficulty.pt --frets-json "[[0,2,0,0],[0,2,3,0],[2,0,0,0]]" --strings-json "[[1],[1,2],[0]]" --only-played-strings false
    main()
