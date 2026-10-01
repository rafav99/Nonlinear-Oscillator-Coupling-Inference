from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from models.edge_first_acc import EdgeFirstAccNet
from models.pair_context import PairContextNet


@dataclass(frozen=True)
class MemberSpec:
    name: str
    architecture: str
    weight: float
    checkpoint: str


FINAL_ENSEMBLE = (
    MemberSpec("C3", "pair_context", 0.15, "checkpoints/C3.pt"),
    MemberSpec("C4", "pair_context", 0.15, "checkpoints/C4.pt"),
    MemberSpec("ACCFT17", "edge_first_acc", 0.35, "checkpoints/ACCFT17.pt"),
    MemberSpec("ACCFT29_30", "edge_first_acc", 0.35, "checkpoints/ACCFT29_30.pt"),
)


def build_model(architecture: str) -> nn.Module:
    if architecture == "pair_context":
        return PairContextNet()
    if architecture == "edge_first_acc":
        return EdgeFirstAccNet(c=32, e=64, d=192, layers=4, heads=8, drop=0.10)
    raise ValueError(f"unknown architecture: {architecture}")


def sign_flip_tta(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (model(x) + model(-x))


class FourModelEnsemble(nn.Module):
    def __init__(self, root: str | Path = ".", load_weights: bool = True) -> None:
        super().__init__()
        root = Path(root)
        self.members = nn.ModuleDict()
        self.weights = {}
        for spec in FINAL_ENSEMBLE:
            model = build_model(spec.architecture)
            path = root / spec.checkpoint
            if load_weights:
                state = torch.load(path, map_location="cpu", weights_only=True)
                model.load_state_dict(state)
            self.members[spec.name] = model
            self.weights[spec.name] = spec.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prediction = 0.0
        for name, model in self.members.items():
            prediction = prediction + self.weights[name] * sign_flip_tta(model, x)
        return prediction
