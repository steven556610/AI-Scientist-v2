"""src/__init__.py — AI-Scientist-v2 utility package."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ai-scientist-v2")
except PackageNotFoundError:
    __version__ = "0.0.0"
