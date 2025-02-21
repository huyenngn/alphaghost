# AlphaGhOst

![License: MIT](https://img.shields.io/github/license/huyenngn/alphaghost)
![Docs](https://github.com/huyenngn/alphaghost/actions/workflows/docs.yml/badge.svg)
![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)

AlphaZero inspired Phantom Go bot.

## Quick Start

The `docs/source/examples` directory contains notebooks that demonstrate how to use the AlphaGhOst API. You can run the notebooks in a Jupyter environment or view it on GitHub.

## Development

Use [uv](https://docs.astral.sh/uv/) to set up a local development environment.

```sh
git clone https://github.com/huyenngn/alphaghost.git
cd alphaghost
uv sync
```

You can use `uv run <command>` to avoid having to manually activate the project
venv. For example, to play a game of Phantom Go against the AlphaGhOst bot, run:

```sh
uv run alphaghost play
```

## License

This project is licensed under the Apache License 2.0.

It contains modifications of [OpenSpiel's](https://github.com/google-deepmind/open_spiel) AlphaZero algorithm and MCTS implementations, originally developed by DeepMind. The original license has been preserved in the relevant source files. For the full license text, see the [`LICENSE`](LICENSE) file.
