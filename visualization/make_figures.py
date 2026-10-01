from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data_generation.system import CFG, simulate_experiment


def main() -> None:
    out_dir = Path(__file__).with_name("figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    sample = simulate_experiment(seed=20260930, cell=3)
    t = CFG.time
    observed = sample["observed"]
    latent = sample["latent"]
    K = sample["K"]

    fig, ax = plt.subplots(figsize=(11, 6))
    offsets = np.arange(CFG.n_nodes) * 3.0
    for i in range(CFG.n_nodes):
        ax.plot(t, observed[i] + offsets[i], linewidth=1.0, label=f"x{i + 1}")
    ax.set_xlabel("time")
    ax.set_ylabel("observed displacement + offset")
    ax.set_title("Sensor-confounded trajectories from one 8-oscillator experiment")
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "observed_trajectories.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    node = 2
    ax.plot(t, latent[node], label="latent displacement")
    ax.plot(t, observed[node], label="observed channel")
    ax.set_xlabel("time")
    ax.set_ylabel("displacement")
    ax.set_title("Latent dynamics vs. sensor output for one oscillator")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "latent_vs_observed.png", dpi=180)
    plt.close(fig)


    fig, ax = plt.subplots(figsize=(6, 6))
    theta = np.linspace(0.0, 2.0 * np.pi, CFG.n_nodes, endpoint=False)
    xy = np.column_stack([np.cos(theta), np.sin(theta)])
    for i in range(CFG.n_nodes):
        for j in range(i + 1, CFG.n_nodes):
            if K[i, j] > 0:
                ax.plot(
                    [xy[i, 0], xy[j, 0]],
                    [xy[i, 1], xy[j, 1]],
                    linewidth=0.8 + 3.0 * float(K[i, j]),
                    alpha=0.65,
                )
    ax.scatter(xy[:, 0], xy[:, 1], s=520, zorder=3)
    for i, (xp, yp) in enumerate(xy):
        ax.text(xp, yp, str(i + 1), ha="center", va="center", zorder=4)
    ax.set_title("Example latent interaction network")
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "interaction_network.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(K, aspect="equal")
    ax.set_xlabel("oscillator j")
    ax.set_ylabel("oscillator i")
    ax.set_title("Ground-truth coupling matrix K")
    fig.colorbar(image, ax=ax, label="coupling strength")
    fig.tight_layout()
    fig.savefig(out_dir / "coupling_matrix.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
