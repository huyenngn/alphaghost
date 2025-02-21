# SPDX-FileCopyrightText: Copyright Huyen Nguyen
# SPDX-License-Identifier: Apache-2.0

import pathlib

import click

import alphaghost
from alphaghost import alphaghost as ag


@click.group()
@click.version_option(
    version=alphaghost.__version__,
    prog_name="AlphaGhOst",
    message="(prog)s %(version)s",
)
def cli():
    """Console script for alphaghost."""


@cli.command("play")
@click.option(
    "--config",
    type=click.Path(path_type=pathlib.Path, dir_okay=False),
    help="The configuration JSON.",
)
@click.option("--black", type=str, default="human", help="Who plays as black.")
@click.option("--white", type=str, default="ag", help="Who plays as white.")
@click.option(
    "--num_games", type=int, default=1, help="The number of games to play."
)
@click.option(
    "--verbose", is_flag=True, help="Print MCTS stats of possible moves."
)
def play_cli(config, black, white, num_games, verbose):
    """Play a game of Phantom Go."""
    from alphaghost import phantom_go

    phantom_go.PhantomGoGame(
        config=ag.Config.from_json(config),
        black=black,
        white=white,
        verbose=verbose,
    ).auto_play(num_games)


@cli.command("train")
@click.option(
    "--config",
    type=click.Path(path_type=pathlib.Path, dir_okay=False),
    help="The configuration JSON.",
)
def train_cli(config):
    """Train a model."""
    ag.alphaghost(config=ag.Config.from_json(config))


if __name__ == "__main__":
    cli()
