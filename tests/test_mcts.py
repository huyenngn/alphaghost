# SPDX-FileCopyrightText: Copyright Huyen Nguyen
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pyspiel
import pytest
from conftest import STATE_B_1_W_0_1, STATE_B_13_W_2_11, STATE_W_3_B_1_4
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
        max_simulations=10,
        evaluator=evaluator,
        random_state=rng,
    )


@pytest.mark.parametrize(
    ("state_str"),
    [
        pytest.param(STATE_B_13_W_2_11.serialized, id="STATE_B_13_W_2_11"),
        pytest.param(STATE_B_1_W_0_1.serialized, id="STATE_B_1_W_0_1"),
        pytest.param(STATE_W_3_B_1_4.serialized, id="STATE_W_3_B_1_4"),
    ],
)
def test_reconstruct_board(
    game: pyspiel.Game, bot: ag_mcts.MCTSBot, state_str: str
):
    state = game.deserialize_state(state_str)
    state_svg = parsers.get_board_svg(state)
    visible_actions = parsers.get_visible_actions(state)
    is_terminal = state.is_terminal()
    player = state.current_player()

    new_state_str = bot._reconstruct_board(state, strict=True)
    new_state = game.deserialize_state(new_state_str)
    new_state_svg = parsers.get_board_svg(new_state)
    new_visible_actions = parsers.get_visible_actions(new_state)
    new_is_terminal = new_state.is_terminal()
    new_player = new_state.current_player()

    assert is_terminal == new_is_terminal
    assert player == new_player
    for old, new in zip(visible_actions, new_visible_actions, strict=True):
        assert np.array_equal(old, new)
    assert state_svg == new_state_svg


@pytest.mark.parametrize(
    ("state_str"),
    [
        pytest.param(STATE_B_13_W_2_11.serialized, id="STATE_B_13_W_2_11"),
        pytest.param(STATE_B_1_W_0_1.serialized, id="STATE_B_1_W_0_1"),
        pytest.param(STATE_W_3_B_1_4.serialized, id="STATE_W_3_B_1_4"),
    ],
)
def test_mcts_bot_infer_hidden_stones_count(
    game: pyspiel.Game, bot: ag_mcts.MCTSBot, state_str: str
):
    state = game.deserialize_state(state_str)
    num_total = parsers.get_stones_count(state)
    player = state.current_player()

    new_state = bot._infer_hidden_stones(state)
    new_num_total = parsers.get_stones_count(new_state)
    new_num_visible = list(map(len, parsers.get_visible_actions(new_state)))
    new_player = new_state.current_player()

    assert np.array_equal(new_num_total, num_total)
    assert np.array_equal(new_num_visible, num_total)
    assert new_player == player


@pytest.mark.parametrize(
    ("state_str"),
    [
        pytest.param(STATE_B_13_W_2_11.serialized, id="STATE_B_13_W_2_11"),
        pytest.param(STATE_B_1_W_0_1.serialized, id="STATE_B_1_W_0_1"),
        pytest.param(STATE_W_3_B_1_4.serialized, id="STATE_W_3_B_1_4"),
    ],
)
def test_mcts_bot_step(
    game: pyspiel.Game, bot: ag_mcts.MCTSBot, state_str: str
):
    state = game.deserialize_state(state_str)
    action = bot.step(state)
    assert action in state.legal_actions()
