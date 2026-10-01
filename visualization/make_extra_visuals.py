from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from data_generation.system import CFG, simulate_experiment


def make_overlay(sample, out_dir: Path) -> None:
    t = CFG.time
    observed = sample['observed'].astype(float)
    normed = (observed - observed.mean(axis=1, keepdims=True)) / (observed.std(axis=1, keepdims=True) + 1e-6)

    fig, ax = plt.subplots(figsize=(11, 5.8))
    for i in range(CFG.n_nodes):
        ax.plot(t, normed[i], linewidth=1.2, alpha=0.9, label=f"x{i+1}")
    ax.axhline(0.0, linewidth=0.8, alpha=0.4)
    ax.set_xlabel('time')
    ax.set_ylabel('standardized displacement')
    ax.set_title('Superposed observed trajectories (per-channel standardized)')
    ax.legend(ncol=4, fontsize=8, loc='upper right')
    fig.tight_layout()
    fig.savefig(out_dir / 'superposed_trajectories.png', dpi=180)
    plt.close(fig)


def make_gif(sample, out_dir: Path) -> None:
    t = CFG.time
    observed = sample['observed'].astype(float)
    K = sample['K'].astype(float)
    normed = (observed - observed.mean(axis=1, keepdims=True)) / (observed.std(axis=1, keepdims=True) + 1e-6)
    n = CFG.n_nodes

    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    base = np.column_stack([np.cos(theta), np.sin(theta)])
    edge_list = [(i, j) for i in range(n) for j in range(i + 1, n) if K[i, j] > 0]

    fig = plt.figure(figsize=(11, 5.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0])
    ax_ts = fig.add_subplot(gs[0, 0])
    ax_net = fig.add_subplot(gs[0, 1])

    # Time-series panel
    ax_ts.set_xlim(t[0], t[-1])
    y_min = float(np.min(normed) - 0.35)
    y_max = float(np.max(normed) + 0.35)
    ax_ts.set_ylim(y_min, y_max)
    ax_ts.set_xlabel('time')
    ax_ts.set_ylabel('standardized displacement')
    ax_ts.set_title('Observed dynamics')
    lines = [ax_ts.plot([], [], linewidth=1.5, alpha=0.95)[0] for _ in range(n)]
    time_cursor = ax_ts.axvline(t[0], linewidth=1.0, alpha=0.7)

    # Network panel
    ax_net.set_title('Hidden coupling network')
    ax_net.set_aspect('equal')
    ax_net.axis('off')
    ax_net.set_xlim(-1.75, 1.75)
    ax_net.set_ylim(-1.75, 1.75)

    edge_artists = []
    for i, j in edge_list:
        line, = ax_net.plot([], [], alpha=0.3 + 0.45 * min(K[i, j], 1.0), linewidth=0.8 + 2.6 * K[i, j])
        edge_artists.append((i, j, line))

    scat = ax_net.scatter([], [], s=[], zorder=3)
    text_artists = [ax_net.text(0, 0, str(i + 1), ha='center', va='center', fontsize=9, zorder=4) for i in range(n)]
    timestamp = ax_net.text(0.0, -1.52, '', ha='center', va='center', fontsize=11)

    sampled_frames = list(range(0, len(t), 2))

    def update(frame_idx: int):
        k = sampled_frames[frame_idx]
        tk = t[k]

        for i, line in enumerate(lines):
            line.set_data(t[: k + 1], normed[i, : k + 1])
        time_cursor.set_xdata([tk, tk])

        amp = np.clip(normed[:, k], -2.4, 2.4)
        radial = 0.18 * amp
        xy = base * (1.0 + radial[:, None])
        sizes = 220 + 170 * (amp - amp.min()) / (amp.max() - amp.min() + 1e-6)

        for i, j, artist in edge_artists:
            artist.set_data([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]])

        scat.set_offsets(xy)
        scat.set_sizes(sizes)
        scat.set_array(amp)
        scat.set_clim(-2.4, 2.4)

        for i, txt in enumerate(text_artists):
            txt.set_position((xy[i, 0], xy[i, 1]))

        timestamp.set_text(f't = {tk:0.2f}')
        return lines + [time_cursor, scat, timestamp] + [a for _, _, a in edge_artists] + text_artists

    anim = FuncAnimation(fig, update, frames=len(sampled_frames), interval=70, blit=False)
    anim.save(out_dir / 'oscillator_dynamics.gif', writer=PillowWriter(fps=14))
    plt.close(fig)


if __name__ == '__main__':
    out_dir = Path(__file__).with_name('figures')
    out_dir.mkdir(parents=True, exist_ok=True)
    sample = simulate_experiment(seed=20260930, cell=3)
    make_overlay(sample, out_dir)
    make_gif(sample, out_dir)
