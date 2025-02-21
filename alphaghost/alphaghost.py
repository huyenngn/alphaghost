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
# - Modified the original code to support Phantom Go
# - Changed `alphazero` function name to `alphaghost`
# - Modified docstrings to reflect the changes.

"""A basic AlphaZero implementation.

A version of
:mod:`open_spiel.python.algorithms.alpha_zero.alpha_zero`
modified to use :class:`alphaghost.mcts.MCTSBot` and
:class:`alphaghost.model.AlphaGhostModel` to support imperfect information
games, specifically Phantom Go.
"""

import collections.abc as cabc
import datetime
import functools
import itertools
import json
import os
import pathlib
import random
import sys
import tempfile
import time
import traceback
import typing as t

import numpy as np
import pyspiel
from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms.alpha_zero import evaluator as eval_lib
from open_spiel.python.utils import data_logger, file_logger, spawn, stats

from alphaghost import mcts as ag_mcts
from alphaghost import model as ag_model

JOIN_WAIT_DELAY = 0.001


class TrajectoryState:
    """A particular point along a trajectory."""

    def __init__(
        self, observation, current_player, legals_mask, action, policy, value
    ):
        self.observation = observation
        self.current_player = current_player
        self.legals_mask = legals_mask
        self.action = action
        self.policy = policy
        self.value = value


class Trajectory:
    """A sequence of observations, actions and policies, and the outcomes."""

    def __init__(self):
        self.states = []
        self.returns = []

    def add(self, information_state, action, policy):
        self.states.append((information_state, action, policy))


class Buffer:
    """A fixed size buffer that keeps the newest values."""

    def __init__(self, max_size: int):
        self.max_size: int = max_size
        self.data: list = []
        self.total_seen: int = 0

    def __len__(self):
        return len(self.data)

    def __bool__(self):
        return bool(self.data)

    def append(self, val: t.Any):
        return self.extend([val])

    def extend(self, batch: cabc.Iterable[t.Any]):
        batch = list(batch)
        self.total_seen += len(batch)
        self.data.extend(batch)
        self.data[: -self.max_size] = []

    def sample(self, count: int):
        return random.sample(self.data, count)


class Config(t.NamedTuple):
    board_size: int = 9
    """The size of the board."""
    path: str = "checkpoints/"
    """Where to save checkpoints."""
    learning_rate: float = 0.001
    """How fast to update weights."""
    weight_decay: float = 0.0001
    """L2 regularization strength."""
    train_batch_size: int = 256
    """Number of training samples per update."""
    replay_buffer_size: int = 10000
    """How many states to store in the replay buffer."""
    replay_buffer_reuse: int = 3
    """How many times to learn from each state."""
    max_steps: int = 0
    """How many learn steps before exiting."""
    checkpoint_freq: int = 100
    """Save a checkpoint every N steps."""
    actors: int = 2
    """How many actors to generate training data."""
    evaluators: int = 1
    """How many evaluators to run."""
    evaluation_window: int = 100
    """How many games to average results over."""
    eval_levels: int = 7
    """Play evaluation games vs MCTS+Solver, with
    max_simulations*10^(n/2) simulations for n in range(eval_levels)."""
    uct_c: float = 2.0
    """Exploring new moves vs. choosing known good moves"""
    max_simulations: int = 100
    """How many MCTS simulations per move."""
    policy_alpha: float = 1.0
    """Dirichlet noise alpha."""
    policy_epsilon: float = 0.25
    """What noise epsilon to use."""
    temperature: float = 1.0
    """Encourage random moves early in training."""
    temperature_drop: int = 10
    """Drop randomness after N steps."""
    nn_width: int = 128
    """Number of layers in the network."""
    nn_depth: int = 3
    """Number of nodes in each layer."""
    observation_shape: t.Any = None
    """The shape of the observation tensor."""
    output_size: int = 82
    """The number of possible actions."""
    quiet: bool = True
    """Don't show the moves as they're played."""

    @classmethod
    def from_json(cls, path: pathlib.Path):
        """Load a config from a json file."""
        return cls(**json.loads(path.read_text()))


def _init_model_from_config(config: Config):
    """Initialize the AlphaGhostModel from a configuration."""
    return ag_model.AlphaGhostModel(
        input_shape=config.observation_shape,
        output_size=config.output_size,
        nn_depth=config.nn_depth,
        nn_width=config.nn_width,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        path=config.path,
    )


def watcher(fn: cabc.Callable):
    """Give a logger and logs exceptions."""

    @functools.wraps(fn)
    def _watcher(*, config, num=None, **kwargs):
        """Wrap the decorated function."""
        name = fn.__name__
        if num is not None:
            name += "-" + str(num)
        with file_logger.FileLogger(config.path, name, config.quiet) as logger:
            print(f"{name} started")
            logger.print(f"{name} started")
            try:
                return fn(config=config, logger=logger, **kwargs)
            except Exception as e:
                logger.print(
                    "\n".join(
                        [
                            "",
                            " Exception caught ".center(60, "="),
                            traceback.format_exc(),
                            "=" * 60,
                        ]
                    )
                )
                print(f"Exception caught in {name}: {e}")
                raise
            finally:
                logger.print(f"{name} exiting")
                print(f"{name} exiting")

    return _watcher


def _init_bot(
    config: Config,
    game: pyspiel.Game,
    evaluator_: mcts.Evaluator,
    evaluation: bool,
):
    """Initialize a bot."""
    noise = (
        None if evaluation else (config.policy_epsilon, config.policy_alpha)
    )
    return ag_mcts.MCTSBot(
        game,
        config.uct_c,
        config.max_simulations,
        evaluator_,
        solve=False,
        dirichlet_noise=noise,
        child_selection_fn=mcts.SearchNode.puct_value,
        verbose=False,
        dont_return_chance_node=True,
    )


def _play_game(
    logger,
    game_num: int,
    game: pyspiel.Game,
    bots: list[ag_mcts.MCTSBot],
    temperature: float,
    temperature_drop: int,
):
    """Play one game, return the trajectory."""
    trajectory = Trajectory()
    actions: list[str] = []
    state = game.new_initial_state()
    logger.opt_print(f" Starting game {game_num} ".center(60, "-"))
    logger.opt_print(f"Initial state:\n{state}")
    while not state.is_terminal():
        root = bots[state.current_player()].mcts_search(state)
        policy = np.zeros(game.num_distinct_actions())
        for c in root.children:
            policy[c.action] = c.explore_count
        policy = policy ** (1 / temperature)
        policy /= policy.sum()
        if len(actions) >= temperature_drop:
            action = root.best_child().action
        else:
            action = np.random.choice(len(policy), p=policy)
        trajectory.states.append(
            TrajectoryState(
                state.observation_tensor(),
                state.current_player(),
                state.legal_actions_mask(),
                action,
                policy,
                root.total_reward / root.explore_count,
            )
        )
        action_str = state.action_to_string(state.current_player(), action)
        actions.append(action_str)
        logger.opt_print(
            f"Player {state.current_player()} sampled action: {action_str}"
        )
        state.apply_action(action)
    logger.opt_print(f"Next state:\n{state}")

    trajectory.returns = state.returns()
    logger.print(
        "Game {}: Returns: {}; Actions: {}".format(
            game_num, " ".join(map(str, trajectory.returns)), " ".join(actions)
        )
    )
    return trajectory


def update_checkpoint(
    logger,
    queue,
    model: ag_model.AlphaGhostModel,
    az_evaluator: eval_lib.AlphaZeroEvaluator,
):
    """Read the queue for a checkpoint to load, or an exit signal."""
    path = None
    while True:  # Get the last message, ignore intermediate ones.
        try:
            path = queue.get_nowait()
        except spawn.Empty:
            break
    if path:
        logger.print("Inference cache:", az_evaluator.cache_info())
        logger.print("Loading checkpoint", path)
        model.load_checkpoint(path)
        az_evaluator.clear_cache()
    elif path is not None:  # Empty string means stop this process.
        return False
    return True


@watcher
def actor(*, config: Config, game: pyspiel.Game, logger, queue):
    """Generate games and returns trajectories."""
    logger.print("Initializing model")
    model = _init_model_from_config(config)
    logger.print("Initializing bots")
    az_evaluator = eval_lib.AlphaZeroEvaluator(game, model)
    bots = [
        _init_bot(config, game, az_evaluator, False),
        _init_bot(config, game, az_evaluator, False),
    ]
    for game_num in itertools.count():
        if not update_checkpoint(logger, queue, model, az_evaluator):
            return
        queue.put(
            _play_game(
                logger,
                game_num,
                game,
                bots,
                config.temperature,
                config.temperature_drop,
            )
        )


@watcher
def evaluator(*, game: pyspiel.Game, config: Config, logger, queue):
    """Play the latest checkpoint vs standard MCTS."""
    results = Buffer(config.evaluation_window)
    logger.print("Initializing model")
    model = _init_model_from_config(config)
    logger.print("Initializing bots")
    az_evaluator = eval_lib.AlphaZeroEvaluator(game, model)
    random_evaluator = mcts.RandomRolloutEvaluator()

    for game_num in itertools.count():
        if not update_checkpoint(logger, queue, model, az_evaluator):
            return

        az_player = game_num % 2
        difficulty = (game_num // 2) % config.eval_levels
        max_simulations = int(
            config.max_simulations * (10 ** (difficulty / 2))
        )
        bots = [
            _init_bot(config, game, az_evaluator, True),
            ag_mcts.MCTSBot(
                game,
                config.uct_c,
                max_simulations,
                random_evaluator,
                solve=True,
                verbose=False,
                dont_return_chance_node=True,
            ),
        ]
        if az_player == 1:
            bots.reverse()

        trajectory = _play_game(
            logger, game_num, game, bots, temperature=1, temperature_drop=0
        )
        results.append(trajectory.returns[az_player])
        queue.put((difficulty, trajectory.returns[az_player]))

        logger.print(
            f"AZ: {trajectory.returns[az_player]}, "
            f"MCTS: {trajectory.returns[1 - az_player]}, "
            f"AZ avg/{len(results)}: {np.mean(results.data):.3f}"
        )


@watcher
def learner(
    *,
    game: pyspiel.Game,
    config: Config,
    actors: list[spawn.Process],
    evaluators: list[spawn.Process],
    broadcast_fn: cabc.Callable[[str], None],
    logger,
):
    """Consume the replay buffer and train the network."""
    logger.also_to_stdout = True
    replay_buffer = Buffer(config.replay_buffer_size)
    learn_rate = config.replay_buffer_size // config.replay_buffer_reuse
    logger.print("Initializing model")
    model = _init_model_from_config(config)
    logger.print(f"Model dimensions: ({config.nn_width}, {config.nn_depth})")
    save_path = model.save_checkpoint(0)
    logger.print("Initial checkpoint:", save_path)
    broadcast_fn(save_path)

    data_log = data_logger.DataLoggerJsonLines(config.path, "learner", True)

    stage_count = 7
    value_accuracies = [stats.BasicStats() for _ in range(stage_count)]
    value_predictions = [stats.BasicStats() for _ in range(stage_count)]
    game_lengths = stats.BasicStats()
    game_lengths_hist = stats.HistogramNumbered(game.max_game_length() + 1)
    outcomes = stats.HistogramNamed(["Player1", "Player2", "Draw"])
    evals = [
        Buffer(config.evaluation_window) for _ in range(config.eval_levels)
    ]
    total_trajectories = 0

    def trajectory_generator():
        """Merge all the actor queues into a single generator."""
        while True:
            found = 0
            for actor_process in actors:
                try:
                    yield actor_process.queue.get_nowait()
                except spawn.Empty:
                    pass
                else:
                    found += 1
            if found == 0:
                time.sleep(0.01)  # 10ms

    def collect_trajectories():
        """Collect the trajectories from actors into the replay buffer."""
        num_trajectories = 0
        num_states = 0
        for trajectory in trajectory_generator():
            num_trajectories += 1
            num_states += len(trajectory.states)
            game_lengths.add(len(trajectory.states))
            game_lengths_hist.add(len(trajectory.states))

            p1_outcome = trajectory.returns[0]
            if p1_outcome > 0:
                outcomes.add(0)
            elif p1_outcome < 0:
                outcomes.add(1)
            else:
                outcomes.add(2)

            replay_buffer.extend(
                ag_model.TrainInput(
                    s.observation, s.legals_mask, s.policy, p1_outcome
                )
                for s in trajectory.states
            )

            for stage in range(stage_count):
                # Scale for the length of the game
                index = (
                    (len(trajectory.states) - 1) * stage // (stage_count - 1)
                )
                n = trajectory.states[index]
                accurate = (n.value >= 0) == (
                    trajectory.returns[n.current_player] >= 0
                )
                value_accuracies[stage].add(1 if accurate else 0)
                value_predictions[stage].add(abs(n.value))

            if num_states >= learn_rate:
                break
        return num_trajectories, num_states

    def learn(step: int):
        """Sample from the replay buffer, update weights and save a checkpoint."""
        losses = []
        for _ in range(len(replay_buffer) // config.train_batch_size):
            data = replay_buffer.sample(config.train_batch_size)
            losses.append(model.update(data))

        # Always save a checkpoint, either for keeping or for loading the weights to
        # the actors. It only allows numbers, so use -1 as "latest".
        save_path = model.save_checkpoint(
            step if step % config.checkpoint_freq == 0 else -1
        )
        losses = sum(losses, ag_model.Losses(0, 0, 0)) / len(losses)
        logger.print(losses)
        logger.print("Checkpoint saved:", save_path)
        return save_path, losses

    last_time = time.time() - 60
    for step in itertools.count(1):
        for value_accuracy in value_accuracies:
            value_accuracy.reset()
        for value_prediction in value_predictions:
            value_prediction.reset()
        game_lengths.reset()
        game_lengths_hist.reset()
        outcomes.reset()

        num_trajectories, num_states = collect_trajectories()
        total_trajectories += num_trajectories
        now = time.time()
        seconds = now - last_time
        last_time = now

        logger.print("Step:", step)
        logger.print(
            f"Collected {num_states:5} states from {num_trajectories:3} "
            f"games, {num_states / seconds:.1f} states/s. "
            f"{num_states / (config.actors * seconds):.1f} "
            "states/(s*actor), game length: "
            f"{num_states / num_trajectories:.1f}"
        )
        logger.print(
            f"Buffer size: {len(replay_buffer)}. "
            f"States seen: {replay_buffer.total_seen}"
        )

        save_path, losses = learn(step)

        for eval_process in evaluators:
            while True:
                try:
                    difficulty, outcome = eval_process.queue.get_nowait()
                    evals[difficulty].append(outcome)
                except spawn.Empty:
                    break

        batch_size_stats = stats.BasicStats()  # Only makes sense in C++.
        batch_size_stats.add(1)
        data_log.write(
            {
                "step": step,
                "total_states": replay_buffer.total_seen,
                "states_per_s": num_states / seconds,
                "states_per_s_actor": num_states / (config.actors * seconds),
                "total_trajectories": total_trajectories,
                "trajectories_per_s": num_trajectories / seconds,
                "queue_size": 0,  # Only available in C++.
                "game_length": game_lengths.as_dict,
                "game_length_hist": game_lengths_hist.data,
                "outcomes": outcomes.data,
                "value_accuracy": [v.as_dict for v in value_accuracies],
                "value_prediction": [v.as_dict for v in value_predictions],
                "eval": {
                    "count": evals[0].total_seen,
                    "results": [
                        sum(e.data) / len(e) if e else 0 for e in evals
                    ],
                },
                "batch_size": batch_size_stats.as_dict,
                "batch_size_hist": [0, 1],
                "loss": {
                    "policy": float(losses.policy),
                    "value": float(losses.value),
                    "l2reg": float(losses.l2),
                    "sum": float(losses.total),
                },
                "cache": {  # Null stats because it's hard to report between processes.
                    "size": 0,
                    "max_size": 0,
                    "usage": 0,
                    "requests": 0,
                    "requests_per_s": 0,
                    "hits": 0,
                    "misses": 0,
                    "misses_per_s": 0,
                    "hit_rate": 0,
                },
            }
        )
        logger.print()

        if config.max_steps > 0 and step >= config.max_steps:
            break

        broadcast_fn(save_path)


def alphaghost(config: Config):
    """Start all the worker processes for a full alphazero setup."""
    game = pyspiel.load_game("phantom_go", {"board_size": config.board_size})
    config = config._replace(
        observation_shape=game.observation_tensor_shape(),
        output_size=game.num_distinct_actions(),
    )
    board_size_str = f"{config.board_size}x{config.board_size}"
    print(f"Starting {board_size_str} game")

    path = config.path
    if not path:
        path = tempfile.mkdtemp(
            prefix=f"az-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}"
            f"-{board_size_str}-"
        )
        config = config._replace(path=path)

    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.isdir(path):
        sys.exit(f"{path} isn't a directory")
    print("Writing logs and checkpoints to:", path)
    print(f"Model dimensions: ({config.nn_width}, {config.nn_depth})")

    with open(os.path.join(config.path, "config.json"), "w") as fp:
        fp.write(json.dumps(config._asdict(), indent=2, sort_keys=True) + "\n")

    actors = [
        spawn.Process(actor, kwargs={"game": game, "config": config, "num": i})
        for i in range(config.actors)
    ]
    evaluators = [
        spawn.Process(
            evaluator, kwargs={"game": game, "config": config, "num": i}
        )
        for i in range(config.evaluators)
    ]

    def broadcast(msg):
        for proc in actors + evaluators:
            proc.queue.put(msg)

    try:
        learner(
            game=game,
            config=config,
            actors=actors,  # pylint: disable=missing-kwoa
            evaluators=evaluators,
            broadcast_fn=broadcast,
        )
    except (KeyboardInterrupt, EOFError):
        print("Caught a KeyboardInterrupt, stopping early.")
    finally:
        broadcast("")
        # for actor processes to join we have to make sure that their q_in is empty,
        # including backed up items
        for proc in actors:
            while proc.exitcode is None:
                while not proc.queue.empty():
                    proc.queue.get_nowait()
                proc.join(JOIN_WAIT_DELAY)
        for proc in evaluators:
            proc.join()
