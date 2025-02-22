# SPDX-FileCopyrightText: Copyright Huyen Nguyen
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
from conftest import STATE_B_12_W_2_10

from alphaghost.model import AlphaGhostModel, Losses, TrainInput


@pytest.fixture
def model(game):
    return AlphaGhostModel(
        input_shape=game.observation_tensor_shape(),
        output_size=game.num_distinct_actions(),
        nn_width=128,
        nn_depth=3,
        learning_rate=0.001,
        weight_decay=0.0001,
        path="checkpoints/",
    )


def test_model_initialization(model):
    assert model is not None
    assert isinstance(model, AlphaGhostModel)


def test_model_forward_pass(model, game):
    state = game.deserialize_state(STATE_B_12_W_2_10)
    obs = np.expand_dims(state.observation_tensor(), 0)
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=model.device)
    policy, value = model(obs_tensor)
    assert policy.shape == (1, 82)
    assert value.shape == (1, 1)


def test_model_inference(model, game):
    state = game.deserialize_state(STATE_B_12_W_2_10)
    obs = np.expand_dims(state.observation_tensor(), 0)
    mask = np.expand_dims(state.legal_actions_mask(), 0)
    value, policy = model.inference(obs, mask)
    assert policy.shape == (1, 82)
    assert value.shape == (1, 1)


def test_model_save_and_load_checkpoint(model):
    checkpoint_path = model.save_checkpoint(1)
    model_state_dict = model.state_dict()
    model.optimizer_state_dict = model.optimizer.state_dict()

    model.load_checkpoint(checkpoint_path)
    new_model_state_dict = model.state_dict()
    new_optimizer_state_dict = model.optimizer.state_dict()

    for k in model_state_dict:
        assert torch.equal(model_state_dict[k], new_model_state_dict[k])
    for k in model.optimizer_state_dict:
        assert torch.equal(
            model.optimizer_state_dict[k], new_optimizer_state_dict[k]
        )


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
