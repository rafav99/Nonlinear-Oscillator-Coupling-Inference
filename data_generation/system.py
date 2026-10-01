from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SimulationConfig:
    n_nodes: int = 8
    n_times: int = 256
    t_end: float = 20.0
    substeps: int = 16
    coupling_scale: float = 1.5

    edge_probability: tuple[float, float] = (0.2, 0.4)
    weight_range: tuple[float, float] = (0.05, 1.0)
    sharpness_bands: tuple[tuple[float, float], tuple[float, float]] = ((1.0, 2.0), (2.0, 3.0))

    gamma_range: tuple[float, float] = (0.05, 0.35)
    linear_range: tuple[float, float] = (0.6, 1.5)
    linear_spread: tuple[float, float] = (0.8, 1.25)
    cubic_range: tuple[float, float] = (0.4, 1.6)
    vdp_range: tuple[float, float] = (0.2, 0.8)

    drive_amplitude: tuple[float, float] = (0.15, 0.8)
    drive_frequency: tuple[float, float] = (0.7, 1.7)
    ou_sigma: tuple[float, float] = (0.03, 0.10)
    ou_tau: tuple[float, float] = (0.05, 1.0)

    x0_range: tuple[float, float] = (-1.0, 1.0)
    v0_range: tuple[float, float] = (-0.2, 0.2)

    sensor_tau: tuple[float, float] = (0.0, 0.25)
    crosstalk_length: tuple[float, float] = (0.1, 0.4)
    crosstalk_strength: tuple[float, float] = (0.0, 0.3)
    log_gain: tuple[float, float] = (-0.5, 0.5)
    offset: tuple[float, float] = (-0.3, 0.3)
    drift: tuple[float, float] = (-0.02, 0.02)
    observation_noise: tuple[float, float] = (0.02, 0.06)

    @property
    def dt_out(self) -> float:
        return self.t_end / (self.n_times - 1)

    @property
    def dt(self) -> float:
        return self.dt_out / self.substeps

    @property
    def time(self) -> np.ndarray:
        return np.linspace(0.0, self.t_end, self.n_times)


CFG = SimulationConfig()
PAIRS = [(i, j) for i in range(CFG.n_nodes) for j in range(i + 1, CFG.n_nodes)]


def _connected(mask: np.ndarray, n: int) -> bool:
    adjacency = [[] for _ in range(n)]
    for edge, (i, j) in zip(mask, PAIRS):
        if edge:
            adjacency[i].append(j)
            adjacency[j].append(i)
    stack = [0]
    seen = {0}
    while stack:
        node = stack.pop()
        for nxt in adjacency[node]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == n


def _draw_graph(rng: np.random.Generator, graph_class: int, cfg: SimulationConfig) -> np.ndarray:
    n_pairs = len(PAIRS)
    while True:
        p = rng.uniform(*cfg.edge_probability)
        base = rng.random(n_pairs) < p
        if _connected(base, cfg.n_nodes):
            break

    edge_count = int(base.sum())
    if graph_class == 0:
        mask = base
    else:
        scores = np.empty(n_pairs, dtype=float)
        for k, (i, j) in enumerate(PAIRS):
            same_module = (i < 4) == (j < 4)
            hub_edge = i == 0 or j == 0
            propensity = 1.0 + 3.0 * same_module + 2.0 * hub_edge
            scores[k] = np.log(propensity) + rng.gumbel()
        order = np.argsort(scores)[::-1]
        mask = np.zeros(n_pairs, dtype=bool)
        for idx in order:
            mask[idx] = True
            if mask.sum() >= edge_count and _connected(mask, cfg.n_nodes):
                break

    weights = rng.uniform(*cfg.weight_range, size=n_pairs)
    K = np.zeros((cfg.n_nodes, cfg.n_nodes), dtype=float)
    for keep, weight, (i, j) in zip(mask, weights, PAIRS):
        if keep:
            K[i, j] = weight
            K[j, i] = weight

    perm = rng.permutation(cfg.n_nodes)
    return K[np.ix_(perm, perm)]


def _interaction(delta: np.ndarray, shape: int, sharpness: float) -> np.ndarray:
    z = sharpness * delta
    if shape == 0:
        return np.tanh(z) / sharpness
    if shape == 1:
        return np.sin(z) / sharpness
    return delta / (1.0 + z * z)


def _sensor_stage(
    latent: np.ndarray,
    rng: np.random.Generator,
    cfg: SimulationConfig,
) -> np.ndarray:
    n, t_count = cfg.n_nodes, cfg.n_times

    tau = rng.uniform(*cfg.sensor_tau, size=n)
    alpha = cfg.dt_out / (tau + cfg.dt_out)
    filtered = np.empty_like(latent)
    z = latent[:, 0].copy()
    for t in range(t_count):
        z = z + alpha * (latent[:, t] - z)
        filtered[:, t] = z

    positions = rng.random((n, 2))
    distances = np.sqrt(((positions[:, None, :] - positions[None, :, :]) ** 2).sum(axis=-1))
    ell = rng.uniform(*cfg.crosstalk_length)
    W = np.exp(-distances / ell)
    np.fill_diagonal(W, 0.0)
    denom = W.sum(axis=1, keepdims=True)
    W = np.divide(W, np.maximum(denom, 1e-12))
    eps = rng.uniform(*cfg.crosstalk_strength)
    mixing = (1.0 - eps) * np.eye(n) + eps * W
    observed = mixing @ filtered

    gain = np.exp(rng.uniform(*cfg.log_gain, size=n))[:, None]
    offset = rng.uniform(*cfg.offset, size=n)[:, None]
    drift = rng.uniform(*cfg.drift, size=n)[:, None]
    observed = gain * observed + offset + drift * cfg.time[None, :]

    rms = np.sqrt(np.mean(observed**2))
    noise_level = rng.uniform(*cfg.observation_noise)
    observed = observed + rng.normal(0.0, noise_level * rms, size=observed.shape)
    return observed


def simulate_experiment(
    seed: int,
    cell: int | None = None,
    cfg: SimulationConfig = CFG,
) -> dict[str, np.ndarray | int | float | str]:
    rng = np.random.default_rng(seed)
    if cell is None:
        cell = seed % 4
    sharpness_band, graph_class = divmod(int(cell), 2)

    K = _draw_graph(rng, graph_class, cfg)
    family = rng.integers(0, 3, size=cfg.n_nodes)
    gamma = rng.uniform(*cfg.gamma_range, size=cfg.n_nodes)
    linear = rng.uniform(*cfg.linear_range, size=cfg.n_nodes) * rng.uniform(*cfg.linear_spread, size=cfg.n_nodes)
    cubic = rng.uniform(*cfg.cubic_range, size=cfg.n_nodes)
    mu = rng.uniform(*cfg.vdp_range, size=cfg.n_nodes)
    shape = int(rng.integers(0, 3))
    sharpness = rng.uniform(*cfg.sharpness_bands[sharpness_band])

    driven = int(rng.integers(0, cfg.n_nodes))
    amplitude = rng.uniform(*cfg.drive_amplitude)
    omega = rng.uniform(*cfg.drive_frequency)
    sigma = rng.uniform(*cfg.ou_sigma)
    tau_c = rng.uniform(*cfg.ou_tau)

    x = rng.uniform(*cfg.x0_range, size=cfg.n_nodes)
    v = rng.uniform(*cfg.v0_range, size=cfg.n_nodes)
    eta = np.zeros(cfg.n_nodes, dtype=float)

    latent = np.empty((cfg.n_nodes, cfg.n_times), dtype=float)
    latent[:, 0] = x
    ou_scale = sigma * np.sqrt(2.0 / tau_c) * np.sqrt(cfg.dt)

    t = 0.0
    for sample in range(1, cfg.n_times):
        for _ in range(cfg.substeps):
            delta = x[None, :] - x[:, None]
            h = _interaction(delta, shape, sharpness)
            coupling = cfg.coupling_scale * np.sum(K * h, axis=1)

            local = np.empty(cfg.n_nodes, dtype=float)
            duff = family == 0
            vdp = family == 1
            pend = family == 2
            local[duff] = -gamma[duff] * v[duff] - linear[duff] * x[duff] - cubic[duff] * x[duff] ** 3
            local[vdp] = -mu[vdp] * (x[vdp] ** 2 - 1.0) * v[vdp] - linear[vdp] * x[vdp]
            local[pend] = -gamma[pend] * v[pend] - linear[pend] * np.sin(x[pend])

            eta += (-eta / tau_c) * cfg.dt + ou_scale * rng.standard_normal(cfg.n_nodes)
            drive = np.zeros(cfg.n_nodes, dtype=float)
            drive[driven] = amplitude * np.cos(omega * t)
            acc = local + coupling + drive + eta

            x = x + v * cfg.dt
            v = v + acc * cfg.dt
            t += cfg.dt
        latent[:, sample] = x

    observed = _sensor_stage(latent, rng, cfg)
    target = np.array([K[i, j] for i, j in PAIRS], dtype=np.float32)

    return {
        "observed": observed.astype(np.float32),
        "latent": latent.astype(np.float32),
        "target": target,
        "K": K.astype(np.float32),
        "cell": int(cell),
        "family": family.astype(np.int8),
        "interaction": ("tanh", "sine", "rational")[shape],
        "sharpness": float(sharpness),
    }


def generate_dataset(n_samples: int, seed: int = 1234, cfg: SimulationConfig = CFG) -> dict[str, np.ndarray]:
    root = np.random.SeedSequence(seed)
    children = root.spawn(n_samples)
    observations = np.empty((n_samples, cfg.n_nodes, cfg.n_times), dtype=np.float32)
    targets = np.empty((n_samples, len(PAIRS)), dtype=np.float32)
    cells = np.empty(n_samples, dtype=np.int8)

    for idx, child in enumerate(children):
        sample_seed = int(child.generate_state(1, dtype=np.uint64)[0])
        sample = simulate_experiment(sample_seed, cell=idx % 4, cfg=cfg)
        observations[idx] = sample["observed"]
        targets[idx] = sample["target"]
        cells[idx] = sample["cell"]

    return {"x": observations, "y": targets, "cell": cells, "time": cfg.time.astype(np.float32)}


def save_dataset(path: str | Path, n_samples: int, seed: int = 1234) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = generate_dataset(n_samples, seed=seed)
    np.savez_compressed(path, **data)
    return path
