# SPDX-FileCopyrightText: Copyright Huyen Nguyen
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pyspiel
import pytest
from conftest import STATE_B_12_W_2_10, STATE_B_12_W_2_10_SVG

BLACK_ACTION = 56
WHITE_ACTION = 64

BLACK_INDEX = (2, 2)
WHITE_INDEX = (1, 1)

from alphaghost import parsers


def test_get_stones_count(game: pyspiel.Game):
    state = game.deserialize_state(STATE_B_12_W_2_10)
    stones_count = parsers.get_stones_count(state)
    expected = np.array([13, 11])
    assert np.array_equal(stones_count, expected)


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        pytest.param(0, "valid\n", id="valid"),
        pytest.param(73, "observational\n", id="observational"),
        pytest.param(11, "1 stones were captured\n", id="capture"),
        pytest.param(81, "pass\n", id="pass"),
    ],
)
def test_get_previous_move_info_string(
    game: pyspiel.Game, action: int, expected: str
):
    state = game.deserialize_state(STATE_B_12_W_2_10)
    state.apply_action(action)
    info_string = parsers.get_previous_move_info_string(state)

    assert info_string.endswith(expected)


def test_get_board_size(game: pyspiel.Game):
    state = game.deserialize_state(STATE_B_12_W_2_10)
    board_size = parsers.get_board_size(state)
    expected = 9
    assert board_size == expected


def test_get_board(game: pyspiel.Game):
    state = game.deserialize_state(STATE_B_12_W_2_10)
    board = parsers.get_board(state)

    black_stone = board[BLACK_INDEX]
    white_stone = board[WHITE_INDEX]
    expected_shape = (9, 9)

    assert board.shape == expected_shape
    assert black_stone == 0
    assert white_stone == 1


def test_get_visible_actions(game: pyspiel.Game):
    state = game.deserialize_state(STATE_B_12_W_2_10)
    black_actions, white_actions = parsers.get_visible_actions(state)

    assert BLACK_ACTION in black_actions
    assert WHITE_ACTION in white_actions
    assert len(black_actions) == 13
    assert len(white_actions) == 2


def test_construct_state(game: pyspiel.Game):
    state = game.deserialize_state(STATE_B_12_W_2_10)
    new_state = parsers.construct_state(state)

    state_svg = parsers.get_board_svg(state)
    new_state_svg = parsers.get_board_svg(new_state)

    assert state_svg == new_state_svg
    assert state.current_player() == new_state.current_player()


def test_get_board_svg(game: pyspiel.Game):
    state = game.deserialize_state(STATE_B_12_W_2_10)
    svg = parsers.get_board_svg(state)

    expected = STATE_B_12_W_2_10_SVG.read_text()

    assert svg == expected
