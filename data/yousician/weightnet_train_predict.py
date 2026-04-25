from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# python weightnet_train_predict.py train \
#   --data song_info_fret_strings_map.json \
#   --output-dir outputs

# -----------------------------
# 1) Fingering representation
# -----------------------------
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


def active_positions(state: FingeringState, only_played_strings: bool = False) -> List[Tuple[int, int]]:
    played = set(state.strings)
    pos = []
    for s, f in enumerate(state.fret):
        if f > 0 and ((not only_played_strings) or (s in played)):
            pos.append((s, f))
    return pos


# -----------------------------
# 2) Cost functions
# -----------------------------
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


# -----------------------------
# 3) Data loading and features
# -----------------------------
def build_song_states(entry: Dict) -> List[FingeringState]:
    frets = entry.get("fret", [])
    strings = entry.get("strings", [])
    n = min(len(frets), len(strings))
    out = []
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

    feat = torch.tensor([
        float(n_notes),
        float(sum(nonzero_counts) / n_notes),
        float(sum(mean_frets) / n_notes),
        float(sum(spans) / n_notes),
        float(sum(played_counts) / n_notes),
        float(max(nonzero_counts)),
        float(max(mean_frets)),
        float(max(spans)),
    ], dtype=torch.float32)

    feat[0] = torch.log1p(feat[0])
    return feat


def load_samples(data_path: Path) -> List[Tuple[str, List[FingeringState], torch.Tensor, float]]:
    with data_path.open("r", encoding="utf-8") as f:
        song_map = json.load(f)

    samples = []
    for key, entry in song_map.items():
        diff = entry.get("difficulty_level", None)
        if diff is None:
            continue
        states = build_song_states(entry)
        if len(states) < 2:
            continue
        feat = song_features(states)
        samples.append((key, states, feat, float(diff)))
    return samples


# -----------------------------
# 4) Model
# -----------------------------
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

    def forward(self, feat: torch.Tensor, song_cost: torch.Tensor):
        h, w, gamma = self.encode_params(feat)
        pred = self.forward_from_cost(h, song_cost)
        return pred, w, gamma


# -----------------------------
# 5) Helpers
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_one_sample(
    model: WeightNet,
    states: List[FingeringState],
    feat: torch.Tensor,
    y_true: float,
    device: torch.device,
    only_played_strings: bool,
) -> Dict[str, float]:
    feat = feat.to(device)
    y_t = torch.tensor(y_true, dtype=torch.float32, device=device)

    with torch.no_grad():
        h, w, gamma = model.encode_params(feat)
        song_cost = full_song_cost_torch(
            states,
            w0=w[0], w1=w[1], w2=w[2],
            w3=w[3], w4=w[4], w5=w[5],
            gamma=gamma,
            device=device,
            only_played_strings=only_played_strings,
        )
        pred = model.forward_from_cost(h, song_cost)

    return {
        "y_true": float(y_t.cpu()),
        "y_pred": float(pred.cpu()),
        "abs_error": float((pred - y_t).abs().cpu()),
        "song_cost": float(song_cost.cpu()),
        "w0": float(w[0].cpu()),
        "w1": float(w[1].cpu()),
        "w2": float(w[2].cpu()),
        "w3": float(w[3].cpu()),
        "w4": float(w[4].cpu()),
        "w5": float(w[5].cpu()),
        "gamma": float(gamma.cpu()),
    }


def evaluate_dataset(
    model: WeightNet,
    data: List[Tuple[str, List[FingeringState], torch.Tensor, float]],
    device: torch.device,
    only_played_strings: bool,
    split_name: str,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    model.eval()
    rows: List[Dict[str, float]] = []
    mse_sum = 0.0
    mae_sum = 0.0
    n = 0

    for key, states, feat, y_true in data:
        row = evaluate_one_sample(model, states, feat, y_true, device, only_played_strings)
        row["key"] = key
        row["split"] = split_name
        row["num_states"] = len(states)
        rows.append(row)

        diff = row["y_pred"] - row["y_true"]
        mse_sum += diff * diff
        mae_sum += abs(diff)
        n += 1

    metrics = {
        "count": n,
        "mse": mse_sum / max(n, 1),
        "rmse": (mse_sum / max(n, 1)) ** 0.5,
        "mae": mae_sum / max(n, 1),
    }
    return metrics, rows


def save_predictions_csv(rows: List[Dict[str, float]], path: Path) -> None:
    if not rows:
        return
    fieldnames = [
        "split", "key", "num_states",
        "y_true", "y_pred", "abs_error", "song_cost",
        "w0", "w1", "w2", "w3", "w4", "w5", "gamma",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def save_history_csv(history: List[Dict[str, float]], path: Path) -> None:
    if not history:
        return
    fieldnames = list(history[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def plot_training_curve(history: List[Dict[str, float]], out_path: Path) -> None:
    if not history:
        return
    epochs = [row["epoch"] for row in history]
    train_mae = [row["train_mae"] for row in history]
    val_mae = [row["val_mae"] for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_mae, label="train_mae")
    plt.plot(epochs, val_mae, label="val_mae")
    plt.xlabel("epoch")
    plt.ylabel("MAE")
    plt.title("Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_true_vs_pred(rows: List[Dict[str, float]], out_path: Path, title: str) -> None:
    if not rows:
        return
    y_true = [row["y_true"] for row in rows]
    y_pred = [row["y_pred"] for row in rows]

    min_v = min(y_true + y_pred)
    max_v = max(y_true + y_pred)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred)
    plt.plot([min_v, max_v], [min_v, max_v], linestyle="--")
    plt.xlabel("true label")
    plt.ylabel("predicted label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_sorted_predictions(rows: List[Dict[str, float]], out_path: Path, title: str) -> None:
    if not rows:
        return
    sorted_rows = sorted(rows, key=lambda x: (x["y_true"], x["key"]))
    x = list(range(len(sorted_rows)))
    y_true = [row["y_true"] for row in sorted_rows]
    y_pred = [row["y_pred"] for row in sorted_rows]

    plt.figure(figsize=(10, 5))
    plt.plot(x, y_true, label="true")
    plt.plot(x, y_pred, label="pred")
    plt.xlabel("song index (sorted by true label)")
    plt.ylabel("difficulty")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def write_metrics_json(metrics: Dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def save_checkpoint(
    model: WeightNet,
    checkpoint_path: Path,
    data_path: Path,
    only_played_strings: bool,
    train_keys: List[str],
    val_keys: List[str],
    seed: int,
    epochs: int,
    hidden: int,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": {
            "in_dim": model.in_dim,
            "hidden": hidden,
            "only_played_strings": only_played_strings,
            "data_path": str(data_path),
            "seed": seed,
            "epochs": epochs,
        },
        "splits": {
            "train_keys": train_keys,
            "val_keys": val_keys,
        },
    }
    torch.save(checkpoint, checkpoint_path)



def load_checkpoint(checkpoint_path: Path, device: torch.device) -> Tuple[WeightNet, Dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = WeightNet(in_dim=config["in_dim"], hidden=config["hidden"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


# -----------------------------
# 6) Training
# -----------------------------
def train_command(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data)
    samples = load_samples(data_path)
    if not samples:
        raise ValueError("No usable songs found in dataset.")

    print(f"device: {device}")
    print(f"usable songs: {len(samples)}")

    random.shuffle(samples)
    split_idx = int(len(samples) * (1 - args.val_ratio))
    split_idx = max(1, min(split_idx, len(samples) - 1))
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]
    print(f"train: {len(train_samples)} | val: {len(val_samples)}")

    model = WeightNet(hidden=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: List[Dict[str, float]] = []
    best_val_mae = float("inf")
    best_ckpt = out_dir / "best_model.pt"
    last_ckpt = out_dir / "last_model.pt"

    def epoch_step(data, train: bool):
        if train:
            model.train()
        else:
            model.eval()

        total_loss = 0.0
        total_mae = 0.0
        n = 0

        for _, states, feat, y in data:
            feat = feat.to(device)
            y_t = torch.tensor(y, dtype=torch.float32, device=device)

            if train:
                opt.zero_grad()

            h, w, gamma = model.encode_params(feat)
            song_cost = full_song_cost_torch(
                states,
                w0=w[0], w1=w[1], w2=w[2],
                w3=w[3], w4=w[4], w5=w[5],
                gamma=gamma,
                device=device,
                only_played_strings=args.only_played_strings,
            )
            pred = model.forward_from_cost(h, song_cost)

            loss = F.mse_loss(pred, y_t)
            reg = args.reg * (w.pow(2).sum() + gamma.pow(2))
            total = loss + reg

            if train:
                total.backward()
                opt.step()

            total_loss += float(total.detach().cpu())
            total_mae += float((pred.detach() - y_t).abs().cpu())
            n += 1

        return total_loss / max(n, 1), total_mae / max(n, 1)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_mae = epoch_step(train_samples, train=True)
        val_loss, val_mae = epoch_step(val_samples, train=False)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_mae": train_mae,
            "val_loss": val_loss,
            "val_mae": val_mae,
        })

        print(
            f"epoch={epoch:03d} | "
            f"train loss={train_loss:.4f}, mae={train_mae:.4f} | "
            f"val loss={val_loss:.4f}, mae={val_mae:.4f}"
        )

        save_checkpoint(
            model,
            last_ckpt,
            data_path=data_path,
            only_played_strings=args.only_played_strings,
            train_keys=[x[0] for x in train_samples],
            val_keys=[x[0] for x in val_samples],
            seed=args.seed,
            epochs=args.epochs,
            hidden=args.hidden,
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            save_checkpoint(
                model,
                best_ckpt,
                data_path=data_path,
                only_played_strings=args.only_played_strings,
                train_keys=[x[0] for x in train_samples],
                val_keys=[x[0] for x in val_samples],
                seed=args.seed,
                epochs=args.epochs,
                hidden=args.hidden,
            )

    save_history_csv(history, out_dir / "training_history.csv")
    plot_training_curve(history, out_dir / "training_curve.png")

    best_model, best_meta = load_checkpoint(best_ckpt, device)
    train_metrics, train_rows = evaluate_dataset(
        best_model, train_samples, device, args.only_played_strings, split_name="train"
    )
    val_metrics, val_rows = evaluate_dataset(
        best_model, val_samples, device, args.only_played_strings, split_name="val"
    )
    all_metrics, all_rows = evaluate_dataset(
        best_model, train_samples + val_samples, device, args.only_played_strings, split_name="all"
    )

    save_predictions_csv(train_rows + val_rows, out_dir / "predictions_train_val.csv")
    save_predictions_csv(all_rows, out_dir / "predictions_all.csv")

    plot_true_vs_pred(train_rows, out_dir / "scatter_train.png", "Train: True vs Pred")
    plot_true_vs_pred(val_rows, out_dir / "scatter_val.png", "Validation: True vs Pred")
    plot_true_vs_pred(all_rows, out_dir / "scatter_all.png", "All Songs: True vs Pred")

    plot_sorted_predictions(train_rows, out_dir / "sorted_train.png", "Train: Sorted True vs Pred")
    plot_sorted_predictions(val_rows, out_dir / "sorted_val.png", "Validation: Sorted True vs Pred")
    plot_sorted_predictions(all_rows, out_dir / "sorted_all.png", "All Songs: Sorted True vs Pred")

    metrics_payload = {
        "train": train_metrics,
        "val": val_metrics,
        "all": all_metrics,
        "best_checkpoint": str(best_ckpt),
        "last_checkpoint": str(last_ckpt),
        "data_path": str(data_path),
        "only_played_strings": args.only_played_strings,
        "splits": best_meta["splits"],
    }
    write_metrics_json(metrics_payload, out_dir / "metrics.json")

    print("\nSaved files:")
    for path in [
        best_ckpt,
        last_ckpt,
        out_dir / "training_history.csv",
        out_dir / "training_curve.png",
        out_dir / "predictions_train_val.csv",
        out_dir / "predictions_all.csv",
        out_dir / "scatter_train.png",
        out_dir / "scatter_val.png",
        out_dir / "scatter_all.png",
        out_dir / "sorted_train.png",
        out_dir / "sorted_val.png",
        out_dir / "sorted_all.png",
        out_dir / "metrics.json",
    ]:
        print(path.resolve())


# -----------------------------
# 7) Predict with saved checkpoint
# -----------------------------
def predict_command(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    checkpoint_path = Path(args.checkpoint)
    model, checkpoint = load_checkpoint(checkpoint_path, device)

    data_path = Path(args.data) if args.data else Path(checkpoint["config"]["data_path"])
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(data_path)
    key_to_split = {}
    for k in checkpoint.get("splits", {}).get("train_keys", []):
        key_to_split[k] = "train"
    for k in checkpoint.get("splits", {}).get("val_keys", []):
        key_to_split[k] = "val"

    rows = []
    for key, states, feat, y_true in samples:
        row = evaluate_one_sample(
            model,
            states,
            feat,
            y_true,
            device,
            checkpoint["config"].get("only_played_strings", False),
        )
        row["key"] = key
        row["split"] = key_to_split.get(key, "unknown")
        row["num_states"] = len(states)
        rows.append(row)

    save_predictions_csv(rows, out_dir / "predictions_from_checkpoint.csv")
    plot_true_vs_pred(rows, out_dir / "scatter_from_checkpoint.png", "Checkpoint: True vs Pred")
    plot_sorted_predictions(rows, out_dir / "sorted_from_checkpoint.png", "Checkpoint: Sorted True vs Pred")

    diff_sum = sum((row["y_pred"] - row["y_true"]) ** 2 for row in rows)
    mae_sum = sum(abs(row["y_pred"] - row["y_true"]) for row in rows)
    metrics = {
        "count": len(rows),
        "mse": diff_sum / max(len(rows), 1),
        "rmse": (diff_sum / max(len(rows), 1)) ** 0.5,
        "mae": mae_sum / max(len(rows), 1),
        "checkpoint": str(checkpoint_path),
        "data_path": str(data_path),
    }
    write_metrics_json(metrics, out_dir / "predict_metrics.json")

    print("prediction files saved to:")
    for path in [
        out_dir / "predictions_from_checkpoint.csv",
        out_dir / "scatter_from_checkpoint.png",
        out_dir / "sorted_from_checkpoint.png",
        out_dir / "predict_metrics.json",
    ]:
        print(path.resolve())


# -----------------------------
# 8) CLI
# -----------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train / predict song difficulty model with saved checkpoint support.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train model, save checkpoint, predictions and plots.")
    train_parser.add_argument("--data", type=str, default="song_info_fret_strings_map.json")
    train_parser.add_argument("--output-dir", type=str, default="outputs")
    train_parser.add_argument("--epochs", type=int, default=40)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--reg", type=float, default=1e-4)
    train_parser.add_argument("--hidden", type=int, default=32)
    train_parser.add_argument("--val-ratio", type=float, default=0.2)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--only-played-strings", action="store_true")
    train_parser.add_argument("--cpu", action="store_true")
    train_parser.set_defaults(func=train_command)

    pred_parser = subparsers.add_parser("predict", help="Load a checkpoint and run prediction.")
    pred_parser.add_argument("--checkpoint", type=str, required=True)
    pred_parser.add_argument("--data", type=str, default=None)
    pred_parser.add_argument("--output-dir", type=str, default="predict_outputs")
    pred_parser.add_argument("--cpu", action="store_true")
    pred_parser.set_defaults(func=predict_command)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    cli_args = parser.parse_args()
    cli_args.func(cli_args)
