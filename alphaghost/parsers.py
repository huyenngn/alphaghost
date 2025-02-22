# SPDX-FileCopyrightText: Copyright Huyen Nguyen
# SPDX-License-Identifier: Apache-2.0

"""Parsers for game states.

Methods for extracting additional information from
:class:`pyspiel.State` objects beyond the core python API.
"""

import itertools
import re

import drawsvg as draw
import numpy as np
import pyspiel

GO_SQUARE_SIZE = 30
GO_STAR_POINT_RADIUS = 3
GO_STONE_RADIUS = GO_TEXT_SIZE = GO_SQUARE_SIZE * 0.4


def get_stones_count(state: pyspiel.State) -> np.ndarray:
    """Return the number of stones for each player."""
    state_str = state.__str__()[45:]
    match = re.search(r"stones_count: w(\d+) b(\d+)", state_str)
    assert match, "Could not find stones_count"
    return np.array([int(match.group(2)), int(match.group(1))])


def get_previous_move_info_string(
    state: pyspiel.State, player_id: int | None = None
) -> str:
    """Return the previous move information."""
    player_id = player_id if player_id is not None else state.current_player()
    observation_str = state.observation_string(player_id)[-90:]
    match = re.search(r"(Previous move was.*)", observation_str, re.DOTALL)
    return match.group(1) if match else ""


def get_board_size(state: pyspiel.State) -> int:
    """Return the board size."""
    return state.get_game().get_parameters()["board_size"]


def get_board(
    state: pyspiel.State, player_id: int | None = None
) -> np.ndarray:
    """Return the board positions visible to a player as a matrix."""
    player_id = player_id if player_id is not None else state.current_player()
    observation_str = state.observation_string(player_id)
    lines = re.findall(r"\d\s([+OX]+)", observation_str)
    size = get_board_size(state)
    board = np.full((size, size), -1, dtype=np.int8)
    for i, line in enumerate(lines):
        for j, c in enumerate(line):
            if c == "O":
                board[i, j] = 1
            elif c == "X":
                board[i, j] = 0
    return board


def get_visible_actions(
    state: pyspiel.State, player_id: int | None = None
) -> list[np.ndarray]:
    """Return the visible board position's indices."""
    board = np.flipud(get_board(state, player_id)).flatten()
    black_mask = board == 0
    white_mask = board == 1
    black_actions = np.argwhere(black_mask).flatten()
    white_actions = np.argwhere(white_mask).flatten()

    return [black_actions, white_actions]


def _alternate_arrays(
    arr1: np.ndarray, arr2: np.ndarray, default: int
) -> list[int]:
    return [
        x
        for pair in itertools.zip_longest(arr1, arr2, fillvalue=default)
        for x in pair
    ]


def construct_state(state: pyspiel.State) -> pyspiel.State:
    """Construct state from observation."""
    if state.is_terminal():
        return state
    visible_actions = get_visible_actions(state)
    if all(a.size == 0 for a in visible_actions):
        return state
    game = state.get_game()
    default = game.num_distinct_actions() - 1
    actions = _alternate_arrays(
        visible_actions[0], visible_actions[1], default
    )
    if len(actions) % 2 != 0:
        if actions[-1] == default:
            actions.pop()
        else:
            actions.append(default)
    opp_id = 1 - state.current_player()
    actions += visible_actions[opp_id].tolist()
    data = "\n".join(map(str, actions)) + "\n"
    return game.deserialize_state(data)


def render_board(
    state: pyspiel.State, player_id: int | None = None
) -> draw.Drawing:
    """Render the board as a vector graphic."""
    board = get_board(state, player_id)
    board_size = get_board_size(state)
    size = (board_size + 1) * GO_SQUARE_SIZE
    out = draw.Drawing(size, size)
    out.append(draw.Rectangle(0, 0, size, size, fill="#e08543"))
    center = size / 2
    quarter = int(board_size / 4) * GO_SQUARE_SIZE
    star_points = [center, center - quarter, center + quarter]
    for i in star_points:
        for j in star_points:
            point = draw.Circle(i, j, GO_STAR_POINT_RADIUS, fill="black")
            out.append(point)
    legend = draw.Group(text_anchor="middle", dominant_baseline="middle")
    lines = draw.Group(stroke="black")
    out.append(legend)
    out.append(lines)
    for i in range(board_size):
        row = (i + 1) * GO_SQUARE_SIZE
        start = GO_SQUARE_SIZE
        end = size - GO_SQUARE_SIZE
        lines.append(draw.Line(row, start, row, end))
        lines.append(draw.Line(start, row, end, row))
        for pos in [start - GO_SQUARE_SIZE / 2, end + GO_SQUARE_SIZE / 2]:
            number = draw.Text(str(board_size - i), GO_TEXT_SIZE, pos, row)
            letter = draw.Text(
                chr(i + (65 if i < 8 else 66)), GO_TEXT_SIZE, row, pos
            )
            legend.append(number)
            legend.append(letter)
        for j in range(board_size):
            if board[i, j] == -1:
                continue
            col = (j + 1) * GO_SQUARE_SIZE
            color = "black" if board[i, j] == 0 else "white"
            out.append(draw.Circle(col, row, GO_STONE_RADIUS, fill=color))
    return out


def get_board_svg(state: pyspiel.State, player_id: int | None = None) -> str:
    """Return the board as an SVG string."""
    return render_board(state, player_id).as_svg()
