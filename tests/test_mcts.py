# SPDX-FileCopyrightText: Copyright Huyen Nguyen
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from conftest import STATE_B_12_W_2_10
from open_spiel.python.algorithms import mcts

from alphaghost import mcts as ag_mcts
from alphaghost import parsers


@pytest.fixture
def bot(game) -> ag_mcts.MCTSBot:
    """Return a MCTSBot instance."""
    rng = np.random.RandomState()
    evaluator = mcts.RandomRolloutEvaluator(1, rng)
    return ag_mcts.MCTSBot(
        game=game,
        uct_c=2.0,
        max_simulations=100,
        evaluator=evaluator,
        random_state=rng,
    )


def test_mcts_bot_infer_hidden_stones(game, bot):
    state = game.deserialize_state(STATE_B_12_W_2_10)
    num_total = parsers.get_stones_count(state)
    num_visible = list(map(len, parsers.get_visible_actions(state)))
    player = state.current_player()

    new_state = parsers.construct_state(state)
    new_state = bot._infer_hidden_stones(new_state, num_total)
    new_num_total = parsers.get_stones_count(new_state)
    new_num_visible = list(map(len, parsers.get_visible_actions(new_state)))
    new_player = new_state.current_player()

    assert np.array_equal(new_num_total, num_total)
    assert np.array_equal(new_num_visible, num_visible)
    assert new_player == player


def test_mcts_bot_step(game, bot):
    state = game.deserialize_state(STATE_B_12_W_2_10)
    action = bot.step(state)
    assert action in state.legal_actions()


def test_mcts_bot_step_with_policy(game, bot):
    state = game.deserialize_state(STATE_B_12_W_2_10)
    policy, action = bot.step_with_policy(state)
    assert action in state.legal_actions()
    assert isinstance(policy, list)
    assert all(isinstance(p, tuple) and len(p) == 2 for p in policy)
