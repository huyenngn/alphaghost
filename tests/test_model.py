# SPDX-FileCopyrightText: Copyright Huyen Nguyen
# SPDX-License-Identifier: Apache-2.0


import tempfile

import numpy as np
import pyspiel
import pytest
import torch
from conftest import STATE_B_13_W_2_11

from alphaghost import model as ag_model


@pytest.fixture
def model(game: pyspiel.Game) -> ag_model.AlphaGhostModel:
    with tempfile.TemporaryDirectory() as tmpdirname:
        return ag_model.AlphaGhostModel(
            input_shape=game.observation_tensor_shape(),
            output_size=game.num_distinct_actions(),
            nn_width=128,
            nn_depth=3,
            learning_rate=0.001,
            weight_decay=0.0001,
            path=tmpdirname,
        )


def test_model_forward_pass(
    model: ag_model.AlphaGhostModel, game: pyspiel.Game
):
    state = game.deserialize_state(STATE_B_13_W_2_11.serialized)
    obs = np.expand_dims(state.observation_tensor(), 0)
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=model.device)
    policy, value = model(obs_tensor)
    assert policy.shape == (1, 82)
    assert value.shape == (1, 1)


def test_model_inference(model: ag_model.AlphaGhostModel, game: pyspiel.Game):
    state = game.deserialize_state(STATE_B_13_W_2_11.serialized)
    obs = np.expand_dims(state.observation_tensor(), 0)
    mask = np.expand_dims(state.legal_actions_mask(), 0)
    value, policy = model.inference(obs, mask)
    assert policy.shape == (1, 82)
    assert value.shape == (1, 1)


def test_model_save_and_load_checkpoint(model: ag_model.AlphaGhostModel):
    checkpoint_path = model.save_checkpoint(1)
    model_state_dict = model.state_dict()
    optimizer_state_dict = model.optimizer.state_dict()

    model.load_checkpoint(checkpoint_path)
    new_model_state_dict = model.state_dict()
    new_optimizer_state_dict = model.optimizer.state_dict()

    def _compare_state_dicts(old, new):
        for k, v in old.items():
            if isinstance(v, torch.Tensor):
                assert torch.equal(v, new[k])
            else:
                assert v == new[k]

    _compare_state_dicts(model_state_dict, new_model_state_dict)
    _compare_state_dicts(optimizer_state_dict, new_optimizer_state_dict)
