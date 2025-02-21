"""The app package."""

from importlib import metadata

try:
    __version__ = metadata.version("alphaghost")
except metadata.PackageNotFoundError:
    __version__ = "unknown"
del metadata
