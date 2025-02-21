# SPDX-FileCopyrightText: Copyright Huyen Nguyen
# SPDX-License-Identifier: Apache-2.0

"""Global fixtures for pytest."""

import pyspiel
import pytest

STATE_B_12_W_2_10 = "51\n55\n13\n46\n52\n57\n45\n30\n29\n64\n39\n31\n30\n16\n12\n21\n6\n31\n22\n73\n32\n27\n40\n14\n56\n60\n55\n64\n3\n42\n"


@pytest.fixture
def game() -> pyspiel.Game:
    """Return a Phantom Go game."""
    return pyspiel.load_game("phantom_go", {"board_size": 9})
