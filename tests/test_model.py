# SPDX-FileCopyrightText: Copyright Huyen Nguyen
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from alphaghost.model import AlphaGhostModel, Losses, TrainInput


@pytest.fixture
def model():
    input_shape = [17, 9, 9]
    output_size = 81
    nn_width = 64
    nn_depth = 4
    learning_rate = 0.001
    weight_decay = 0.0001
    path = "/tmp"
    return AlphaGhostModel(
        input_shape,
        output_size,
        nn_width,
        nn_depth,
        learning_rate,
        weight_decay,
        path,
    )


def test_model_initialization(model):
    assert model is not None
    assert isinstance(model, AlphaGhostModel)


def test_model_forward_pass(model):
    obs = np.random.rand(1, 17, 9, 9).astype(np.float32)
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=model.device)
    policy, value = model(obs_tensor)
    assert policy.shape == (1, 81)
    assert value.shape == (1, 1)


def test_model_inference(model):
    obs = np.random.rand(1, 17, 9, 9).astype(np.float32)
    mask = np.ones((1, 81), dtype=bool)
    value, policy = model.inference(obs, mask)
    assert policy.shape == (1, 81)
    assert value.shape == (1, 1)


def test_model_save_and_load_checkpoint(model):
    step = 1
    checkpoint_path = model.save_checkpoint(step)
    model.load_checkpoint(checkpoint_path)


def test_model_update(model):
    obs = np.random.rand(1, 17, 9, 9).astype(np.float32)
    legals_mask = np.ones((1, 81), dtype=bool)
    policy = np.random.rand(1, 81).astype(np.float32)
    value = np.random.rand(1).astype(np.float32)
    train_input = TrainInput(obs, legals_mask, policy, value)
    losses = model.update([train_input])
    assert isinstance(losses, Losses)
    assert losses.policy > 0
    assert losses.value > 0
    assert losses.l2 == 0.0
