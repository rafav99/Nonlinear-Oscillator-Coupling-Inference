from __future__ import annotations

import argparse

import numpy as np
import torch

from evaluation.metrics import mean_sre, per_target_sre
from models.ensemble import FourModelEnsemble


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/validation.npz")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    data = np.load(args.data)
    x = torch.from_numpy(data["x"].astype(np.float32))
    y = data["y"].astype(np.float32)
    model = FourModelEnsemble(args.root).eval()

    with torch.no_grad():
        pred = model(x).numpy()
    scores = per_target_sre(y, pred)
    print(f"mean_sre={mean_sre(y, pred):.6f}")
    print("per_target_sre=", np.array2string(scores, precision=4))


if __name__ == "__main__":
    main()
