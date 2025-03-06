# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications made by Huyen Nguyen on February 21 2025:
# - Added the `_infer_hidden_stones` method to approximate the hidden stones
# - Modified the `_apply_tree_policy` method to use the `_infer_hidden_stones`
# - Modified the `mcts_search` method to handle the hidden stones.
# - Modified docstrings to reflect the changes.
# - Added type annotations.

"""Monte-Carlo Tree Search algorithm for Phantom Go.

A version of :class:`open_spiel.python.algorithms.mcts.MCTSBot`
extended to support imperfect information games, specifically Phantom Go.
"""

import collections.abc as cabc
import itertools
import time

import numpy as np
import pyspiel
from open_spiel.python.algorithms import mcts

from alphaghost import parsers


class MCTSBot(pyspiel.Bot):
    """Bot that uses Monte-Carlo Tree Search algorithm."""

    def __init__(
        self,
        game: pyspiel.Game,
        uct_c: float,
        max_simulations: int,
        evaluator: mcts.Evaluator,
        solve: bool = True,
        random_state: np.random.RandomState | None = None,
        child_selection_fn: cabc.Callable = mcts.SearchNode.uct_value,
        dirichlet_noise: tuple[float, float] | None = None,
        verbose: bool = False,
        dont_return_chance_node: bool = False,
    ) -> None:
        """Initialize a MCTS Search algorithm in the form of a bot.

        Args
        ----
            game
                A pyspiel.Game to play.
            uct_c
                The exploration constant for UCT.
            max_simulations
                How many iterations of MCTS to perform. Each simulation
                will result in one call to the evaluator. Memory usage should grow
                linearly with simulations * branching factor. How many nodes in the
                search tree should be evaluated. This is correlated with memory size and
                tree depth.
            evaluator
                A `Evaluator` object to use to evaluate a leaf node.
            solve
                Whether to back up solved states.
            random_state
                An optional numpy RandomState to make it deterministic.
            child_selection_fn
                A function to select the child in the descent phase.
                The default is UCT.
            dirichlet_noise
                A tuple of (epsilon, alpha) for adding dirichlet noise to
                the policy at the root. This is from the alpha-zero paper.
            verbose
                Whether to print information about the search tree before
                returning the action. Useful for confirming the search is working
                sensibly.
            dont_return_chance_node
                If true, do not stop expanding at chance nodes.
                Enabled for AlphaZero.
        """
        super().__init__()

        self._game = game
        self.uct_c = uct_c
        self.max_simulations = max_simulations
        self.evaluator = evaluator
        self.verbose = verbose
        self.solve = solve
        self.max_utility = game.max_utility()
        self._dirichlet_noise = dirichlet_noise
        self._random_state = random_state or np.random.RandomState()
        self._child_selection_fn = child_selection_fn
        self.dont_return_chance_node = dont_return_chance_node
        self._cache: dict[bytes, str] = {}

    def restart_at(self, state) -> None:
        pass

    def step_with_policy(
        self, state: pyspiel.State
    ) -> tuple[list[tuple[int, float]], int]:
        """Return bot's policy and action at given state."""
        t1 = time.time()
        root = self.mcts_search(state)

        best = root.best_child()

        if self.verbose:
            seconds = time.time() - t1
            print(
                f"Finished {root.explore_count} sims in {seconds:.3f} secs, "
                "{root.explore_count / seconds:.1f} sims/s"
            )
            print("Root:")
            print(root.to_str(state))
            print("Children:")
            print(root.children_str(state))
            if best.children:
                chosen_state = state.clone()
                chosen_state.apply_action(best.action)
                print("Children of chosen:")
                print(best.children_str(chosen_state))

        mcts_action = best.action

        policy = [
            (action, (1.0 if action == mcts_action else 0.0))
            for action in state.legal_actions(state.current_player())
        ]

        return policy, mcts_action

    def step(self, state: pyspiel.State) -> int:
        """Return bot's action at given state."""
        return self.step_with_policy(state)[1]

    def _reconstruct_board(
        self, state: pyspiel.State, strict: bool = False
    ) -> str:
        """Construct state string from observation.

        if `strict` is True, guarantees the reconstructed state will have the
        same current player as the original state.
        """
        visible_actions = parsers.get_visible_actions(state)
        if all(a.size == 0 for a in visible_actions):
            return state.serialize()
        default = self._game.num_distinct_actions() - 1
        actions = [
            val
            for pair in itertools.zip_longest(
                visible_actions[0], visible_actions[1], fillvalue=default
            )
            for val in pair
        ]
        if actions[-1] == default:
            actions.pop()
        opp_actions = visible_actions[1 - state.current_player()].tolist()
        if len(actions) % 2 != state.current_player() and (
            len(opp_actions) > 0 or strict
        ):
            actions.append(default)
        actions += opp_actions
        return "\n".join(map(str, actions)) + "\n"

    def _infer_hidden_stones(self, state: pyspiel.State) -> pyspiel.State:
        """Approximate the hidden stones.

        The Phantom Go rules cannot be broken.
        Assumptions are to be made purely based on player's observations.
        """
        if state.is_terminal():
            return state.clone()
        cache_key = np.expand_dims(state.observation_tensor(), 0).tobytes()
        state_str = self._cache.setdefault(
            cache_key, self._reconstruct_board(state)
        )
        assumed_state = self._game.deserialize_state(state_str)
        num_total = parsers.get_stones_count(state)
        tried: list[set[int]] = [{81}, {81}]
        stones_count = parsers.get_stones_count(assumed_state)
        while not np.array_equal(stones_count, num_total):
            player = assumed_state.current_player()
            if stones_count[player] < num_total[player]:
                backup_state = assumed_state.clone()
                policy = self.evaluator.prior(assumed_state)
                policy = [p for p in policy if p[0] not in tried[player]]
                self._random_state.shuffle(policy)
                action = max(policy, key=lambda p: p[1])[0]
                assumed_state.apply_action(action)
                tried[player].add(action)
                if (
                    assumed_state.is_terminal()
                    or parsers.get_stones_count(assumed_state)[1 - player]
                    != stones_count[1 - player]
                ):
                    # Move terminated game or captured opponent's stones.
                    assumed_state = backup_state.clone()
                else:
                    # Make visible.
                    assumed_state.apply_action(action)
                    tried[player] = {81}
                    stones_count = parsers.get_stones_count(assumed_state)
            else:
                action = self._game.num_distinct_actions() - 1
                assumed_state.apply_action(action)
        return assumed_state

    def _apply_tree_policy(
        self,
        root: mcts.SearchNode,
        state: pyspiel.State,
    ) -> tuple[list[mcts.SearchNode], pyspiel.State]:
        """Apply the UCT policy to play the game until reaching a leaf node.

        A leaf node is defined as a node that is terminal or has not been evaluated
        yet. If it reaches a node that has been evaluated before but hasn't been
        expanded, then expand it's children and continue.

        Args
        ----
            root
                The root node in the search tree.
            state
                The state of the game at the root node.

        Returns
        -------
            visit_path
            A list of nodes descending from the root node to a leaf node.
            working_state
            The state of the game at the leaf node.
        """
        visit_path = [root]
        working_state = self._infer_hidden_stones(state)
        current_node = root
        while (
            not working_state.is_terminal() and current_node.explore_count > 0
        ):
            if not current_node.children:
                # For a new node, initialize its state, then choose a child as normal.
                legal_actions = self.evaluator.prior(working_state)
                if current_node is root and self._dirichlet_noise:
                    epsilon, alpha = self._dirichlet_noise
                    noise = self._random_state.dirichlet(
                        [alpha] * len(legal_actions)
                    )
                    legal_actions = [
                        (a, (1 - epsilon) * p + epsilon * n)
                        for (a, p), n in zip(
                            legal_actions, noise, strict=False
                        )
                    ]
                # Reduce bias from move generation order.
                self._random_state.shuffle(legal_actions)
                player = working_state.current_player()
                current_node.children = [
                    mcts.SearchNode(action, player, prior)
                    for action, prior in legal_actions
                ]

            chosen_child = max(
                current_node.children,
                key=lambda c: self._child_selection_fn(
                    c, current_node.explore_count, self.uct_c
                ),
            )

            working_state.apply_action(chosen_child.action)
            current_node = chosen_child
            visit_path.append(current_node)

        return visit_path, working_state

    def mcts_search(self, state: pyspiel.State) -> mcts.SearchNode:
        """Search with Monte-Carlo Tree Search algorithm."""
        root = mcts.SearchNode(None, state.current_player(), 1)
        for _ in range(self.max_simulations):
            visit_path, working_state = self._apply_tree_policy(root, state)
            if working_state.is_terminal():
                returns = working_state.returns()
                visit_path[-1].outcome = returns
                solved = self.solve
            else:
                returns = self.evaluator.evaluate(working_state)
                solved = False

            while visit_path:
                target_return = returns[visit_path[-1].player]
                node = visit_path.pop()
                node.total_reward += target_return
                node.explore_count += 1

                if solved and node.children:
                    player = node.children[0].player
                    best = None
                    all_solved = True
                    for child in node.children:
                        if child.outcome is None:
                            all_solved = False
                        elif (
                            best is None
                            or child.outcome[player] > best.outcome[player]
                        ):
                            best = child
                    if best is not None and (
                        all_solved or best.outcome[player] == self.max_utility
                    ):
                        node.outcome = best.outcome
                    else:
                        solved = False
            if root.outcome is not None:
                break

        return root
