from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from evaluation.metrics import mean_sre
from models.ensemble import build_model


def normalized_mse(pred: torch.Tensor, target: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return torch.mean(((pred - target) / scale) ** 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/train.npz")
    parser.add_argument("--architecture", choices=["pair_context", "edge_first_acc"], required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data = np.load(args.data)
    x = data["x"].astype(np.float32)
    y = data["y"].astype(np.float32)

    split = int(0.9 * len(x))
    train_x, val_x = x[:split], x[split:]
    train_y, val_y = y[:split], y[split:]
    scale = torch.from_numpy(np.std(train_y, axis=0, ddof=0).clip(1e-4)).float()

    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.architecture).to(device)
    scale = scale.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = normalized_mse(pred, yb, scale)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            vx = torch.from_numpy(val_x).to(device)
            pred = 0.5 * (model(vx) + model(-vx))
            score = mean_sre(val_y, pred.cpu().numpy())
        print(f"epoch={epoch:02d} val_sre={score:.6f}")
        if score < best:
            best = score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out)
    print(f"saved={out} best_val_sre={best:.6f}")


if __name__ == "__main__":
    main()
