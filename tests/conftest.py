# SPDX-FileCopyrightText: Copyright Huyen Nguyen
# SPDX-License-Identifier: Apache-2.0

"""Global fixtures for pytest."""

import dataclasses
import pathlib

import pyspiel
import pytest


@dataclasses.dataclass
class TestState:
    """Test state dataclass."""

    serialized: str
    svg: str


STATE_B_13_W_2_11 = TestState(
    serialized="51\n55\n13\n46\n52\n57\n45\n30\n29\n64\n39\n31\n30\n16\n12\n21\n6\n31\n22\n73\n32\n27\n40\n14\n56\n60\n55\n64\n3\n42\n",
    svg=(
        pathlib.Path(__file__).parent / "data" / "STATE_B_13_W_2_11.svg"
    ).read_text(),
)

STATE_B_1_W_0_1 = TestState(
    serialized="63\n75\n",
    svg=(
        pathlib.Path(__file__).parent / "data" / "STATE_B_1_W_0_1.svg"
    ).read_text(),
)

STATE_W_3_B_1_4 = TestState(
    serialized="11\n69\n25\n68\n72\n34\n53\n25\n",
    svg=(
        pathlib.Path(__file__).parent / "data" / "STATE_W_3_B_1_4.svg"
    ).read_text(),
)


@pytest.fixture
def game() -> pyspiel.Game:
    """Return a Phantom Go game."""
    return pyspiel.load_game("phantom_go", {"board_size": 9})
