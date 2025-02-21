"""An AlphaZero style model with a policy and value head."""

from __future__ import annotations

import json
import os
import pathlib
import typing as t

import numpy as np
import torch
from torch import nn, optim


class TrainInput(t.NamedTuple):
    """Inputs for training the Model."""

    observation: np.ndarray
    legals_mask: np.ndarray
    policy: np.ndarray
    value: np.ndarray

    @staticmethod
    def stack(train_inputs: list) -> TrainInput:
        obs, legals_mask, policy, value = zip(*train_inputs, strict=False)
        return TrainInput(
            np.array(obs, dtype=np.float32),
            np.array(legals_mask, dtype=bool),
            np.array(policy),
            np.expand_dims(value, 1),
        )


class Losses(t.NamedTuple):
    """Losses from a training step."""

    policy: float
    value: float
    l2: float

    @property
    def total(self) -> float:
        return self.policy + self.value + self.l2

    def __str__(self):
        return (
            f"Losses(total: {self.total:.3f}, policy: {self.policy:.3f}, "
            f"value: {self.value:.3f}, l2: {self.l2:.3f})"
        )

    def __add__(self, other):
        return Losses(
            self.policy + other.policy,
            self.value + other.value,
            self.l2 + other.l2,
        )

    def __truediv__(self, n):
        return Losses(self.policy / n, self.value / n, self.l2 / n)


class AlphaGhostModel(nn.Module):
    def __init__(
        self,
        input_shape: list[int],
        output_size: int,
        nn_width: int,
        nn_depth: int,
        learning_rate: float,
        weight_decay: float,
        path: str | pathlib.Path,
    ):
        super().__init__()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.conv1 = nn.Conv1d(
            in_channels=input_shape[0], out_channels=nn_width, kernel_size=1
        )
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(nn_width, nn_width) for _ in range(nn_depth - 1)]
        )
        self.fc_out = nn.Linear(nn_width, nn_width)

        self.policy_head = nn.Linear(nn_width, output_size)
        self.value_head = nn.Linear(nn_width, 1)

        self.optimizer = optim.Adam(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self._path = path

        self.to(self.device)

    def forward(self, x):
        x = torch.relu(self.conv1(x.unsqueeze(-1)))
        x = x.view(x.size(0), -1)

        for layer in self.hidden_layers:
            x = torch.relu(layer(x))

        x = torch.relu(self.fc_out(x))

        policy = torch.softmax(self.policy_head(x), dim=1)
        value = torch.tanh(self.value_head(x))

        return policy, value

    @classmethod
    def from_checkpoint(cls, path: pathlib.Path):
        """Load a model from a checkpoint."""
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        checkpoint_dir = path.parent
        config_path = checkpoint_dir / "config.json"
        config = json.loads(config_path.read_text())
        model = cls(
            input_shape=config["observation_shape"],
            output_size=config["output_size"],
            nn_width=config["nn_width"],
            nn_depth=config["nn_depth"],
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
            path=checkpoint_dir,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return model

    def inference(self, obs, mask):
        """Run a forward pass through the network."""
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
        policy, value = self(obs_tensor)
        policy = policy * torch.tensor(mask, device=self.device)
        policy = policy / policy.sum(dim=1, keepdim=True)
        return value.detach().cpu().numpy(), policy.detach().cpu().numpy()

    def save_checkpoint(self, step: int) -> str:
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self._path, f"model_step_{step}.pth")
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            checkpoint_path,
        )
        print(f"Checkpoint saved at {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(self, path: str | pathlib.Path) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Checkpoint loaded from {path}")

    def update(self, train_inputs: list) -> Losses:
        """Perform a training step."""
        batch = TrainInput.stack(train_inputs)

        self.optimizer.zero_grad()

        _input = torch.tensor(
            batch.observation, dtype=torch.float32, device=self.device
        )
        policies = torch.tensor(
            batch.policy, dtype=torch.float32, device=self.device
        )
        values = torch.tensor(
            batch.value, dtype=torch.float32, device=self.device
        )
        legals_mask = torch.tensor(
            batch.legals_mask, dtype=torch.float32, device=self.device
        )

        pred_policies, pred_values = self(_input)

        pred_policies = pred_policies * legals_mask
        pred_policies = pred_policies / pred_policies.sum(dim=1, keepdim=True)

        loss_policy = torch.mean((pred_policies - policies) ** 2)
        loss_value = torch.mean((pred_values.squeeze() - values) ** 2)
        total_loss = loss_policy + loss_value

        total_loss.backward()
        self.optimizer.step()

        return Losses(
            policy=loss_policy.item(),
            value=loss_value.item(),
            l2=0.0,
        )
