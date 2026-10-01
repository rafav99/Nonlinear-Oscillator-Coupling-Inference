from __future__ import annotations

import argparse

from data_generation.system import save_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", default="data/train.npz")
    args = parser.parse_args()
    path = save_dataset(args.output, args.samples, args.seed)
    print(path)


if __name__ == "__main__":
    main()
