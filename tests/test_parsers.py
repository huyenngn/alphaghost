# SPDX-FileCopyrightText: Copyright Huyen Nguyen
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pyspiel
import pytest
from conftest import STATE_B_1_W_0_1, STATE_B_13_W_2_11, STATE_W_3_B_1_4

from alphaghost import parsers


@pytest.mark.parametrize(
    ("state_str", "expected"),
    [
        pytest.param(
            STATE_B_13_W_2_11.serialized,
            np.array([13, 11]),
            id="STATE_B_13_W_2_11",
        ),
        pytest.param(
            STATE_B_1_W_0_1.serialized, np.array([1, 1]), id="STATE_B_1_W_0_1"
        ),
        pytest.param(
            STATE_W_3_B_1_4.serialized, np.array([4, 3]), id="STATE_W_3_B_1_4"
        ),
    ],
)
def test_get_stones_count(
    game: pyspiel.Game, state_str: str, expected: np.ndarray
):
    state = game.deserialize_state(state_str)
    stones_count = parsers.get_stones_count(state)
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
    state = game.deserialize_state(STATE_B_13_W_2_11.serialized)
    state.apply_action(action)
    info_string = parsers.get_previous_move_info_string(state)

    assert info_string.endswith(expected)


def test_get_board_size(game: pyspiel.Game):
    state = game.deserialize_state(STATE_B_13_W_2_11.serialized)
    board_size = parsers.get_board_size(state)
    expected = 9
    assert board_size == expected


@pytest.mark.parametrize(
    ("state_str", "expected"),
    [
        pytest.param(
            STATE_B_13_W_2_11.serialized, (13, 2), id="STATE_B_13_W_2_11"
        ),
        pytest.param(STATE_B_1_W_0_1.serialized, (1, 0), id="STATE_B_1_W_0_1"),
        pytest.param(STATE_W_3_B_1_4.serialized, (1, 3), id="STATE_W_3_B_1_4"),
    ],
)
def test_get_board(game: pyspiel.Game, state_str: str, expected: tuple):
    state = game.deserialize_state(state_str)
    board = parsers.get_board(state)
    black_stones = np.count_nonzero(board == 0)
    white_stones = np.count_nonzero(board == 1)

    expected_black, expected_white = expected
    expected_shape = (9, 9)

    assert board.shape == expected_shape
    assert black_stones == expected_black
    assert white_stones == expected_white


@pytest.mark.parametrize(
    ("state_str", "expected"),
    [
        pytest.param(
            STATE_B_13_W_2_11.serialized, (13, 2), id="STATE_B_13_W_2_11"
        ),
        pytest.param(STATE_B_1_W_0_1.serialized, (1, 0), id="STATE_B_1_W_0_1"),
        pytest.param(STATE_W_3_B_1_4.serialized, (1, 3), id="STATE_W_3_B_1_4"),
    ],
)
def test_get_visible_actions(
    game: pyspiel.Game, state_str: str, expected: tuple
):
    state = game.deserialize_state(state_str)
    black_actions, white_actions = parsers.get_visible_actions(state)

    expected_black, expected_white = expected

    assert len(black_actions) == expected_black
    assert len(white_actions) == expected_white


@pytest.mark.parametrize(
    ("state_str", "expected"),
    [
        pytest.param(
            STATE_B_13_W_2_11.serialized,
            STATE_B_13_W_2_11.svg,
            id="STATE_B_13_W_2_11",
        ),
        pytest.param(
            STATE_B_1_W_0_1.serialized,
            STATE_B_1_W_0_1.svg,
            id="STATE_B_1_W_0_1",
        ),
        pytest.param(
            STATE_W_3_B_1_4.serialized,
            STATE_W_3_B_1_4.svg,
            id="STATE_W_3_B_1_4",
        ),
    ],
)
def test_get_board_svg(game: pyspiel.Game, state_str: str, expected: str):
    state = game.deserialize_state(state_str)
    svg = parsers.get_board_svg(state)

    assert svg == expected
