"""Module for playing Phantom Go against various bots."""

import collections
import enum
import pathlib

import numpy as np
import pyspiel
from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms.alpha_zero import evaluator as az_evaluator
from open_spiel.python.bots import human, uniform_random

from alphaghost import alphaghost as ag


class GoColor(enum.Enum):
    Black = 0
    White = 1


class PhantomGoGame:
    def __init__(
        self,
        config: ag.Config,
        black: str = "human",
        white: str = "ag",
        ag_model: pathlib.Path | None = None,
        verbose: bool = False,
    ) -> None:
        self.size = config.board_size
        self.game = pyspiel.load_game("phantom_go", {"board_size": self.size})
        self.state = self.game.new_initial_state()
        self.config = config
        self.ag_model = ag_model
        self.verbose = verbose
        self.bots = [
            self._init_bot(black, 0),
            self._init_bot(white, 1),
        ]

    def _init_bot(
        self,
        bot_type: str,
        player_id: int,
    ) -> pyspiel.Bot:
        """Initialize a bot by type."""
        if bot_type == "human":
            return human.HumanBot()
        rng = np.random.RandomState()
        if bot_type == "random":
            return uniform_random.UniformRandomBot(player_id, rng)
        if bot_type == "mcts":
            evaluator = mcts.RandomRolloutEvaluator(1, rng)
        elif bot_type == "ag":
            from alphaghost import model as ag_model

            assert (
                self.ag_model is not None
            ), "Path to model is required for AlphaGhost."
            model = ag_model.AlphaGhostModel.from_checkpoint(self.ag_model)
            evaluator = az_evaluator.AlphaZeroEvaluator(self.game, model)
        else:
            raise ValueError(f"Invalid bot type: {bot_type}")
        from alphaghost import mcts as ag_mcts

        return ag_mcts.MCTSBot(
            self.game,
            uct_c=self.config.uct_c,
            max_simulations=self.config.max_simulations,
            evaluator=evaluator,
            random_state=rng,
            verbose=self.verbose,
        )

    @property
    def current_player(self) -> int:
        """Return the current player."""
        return self.state.current_player()

    def _opt_print(self, *args, **kwargs) -> None:
        """Print if not quiet."""
        if not self.config.quiet:
            print(*args, **kwargs)

    def restart(self) -> None:
        """Restart the game."""
        self.state = self.game.new_initial_state()
        for bot in self.bots:
            bot.restart()

    def _play_game(self) -> tuple[list[float], str]:
        """Play one game."""

        self._opt_print(f"Initial state:\n{self.state}")

        while not self.state.is_terminal():
            self.bot_step()
            self._opt_print(f"Next state:\n{self.state}")

        history_str = self.state.history_str()
        returns = self.state.returns()
        print(
            "Returns:",
            " ".join(map(str, returns)),
            ", Game actions:",
            history_str,
        )
        self.restart()

        return returns, history_str

    def auto_play(self, num_games: int = 1) -> None:
        """Automatically play a number of games."""
        histories: dict[str, int] = collections.defaultdict(int)
        overall_wins = [0, 0]
        game_num = 0
        try:
            for _ in range(num_games):
                returns, history_str = self._play_game()
                histories[history_str] += 1
                overall_wins[returns[0] < returns[1]] += 1
                game_num += 1
        except (KeyboardInterrupt, EOFError):
            print("Caught a KeyboardInterrupt, stopping early.")
        print("Number of games played:", game_num)
        print("Number of distinct games played:", len(histories))
        print("Overall wins", overall_wins)

    def step(self, pos_str: str) -> None:
        """Make a step."""
        color_initial_str = GoColor(self.current_player).name[0]
        action_str = f"{color_initial_str} {pos_str}"
        action = self.state.string_to_action(action_str)
        self.state.apply_action(action)

    def bot_step(self) -> None:
        """Make a bot step."""
        bot = self.bots[self.current_player]
        action = bot.step(self.state)
        self.state.apply_action(action)

    def auto_step(self) -> None:
        """Automatically make steps until turn ends."""
        player = self.current_player
        while self.current_player == player and not self.state.is_terminal():
            self.bot_step()
