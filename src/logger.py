"""
src/logger.py
=============
Centralized logging utility for AI-Scientist-v2.

Creates timestamped log files under the project-level `log/` folder with format:
    <file_stem>_<YYYYMMDD_HHMMSS>.log

Features:
- Configures standard Python `logging.Logger` with FileHandler and StreamHandler.
- Provides Tee redirection for `sys.stdout` and `sys.stderr` so all standard `print()`
  outputs, library logs, and uncaught exceptions are recorded to the log file.
"""

import logging
import os
import os.path as osp
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple


def get_default_log_dir() -> str:
    """Return the absolute path to the project-level log directory."""
    current_file = osp.abspath(__file__)
    project_root = osp.dirname(osp.dirname(current_file))
    log_dir = osp.join(project_root, "log")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def create_log_filename(name_or_path: str, start_time: Optional[datetime] = None) -> str:
    """
    Generate a log filename based on file stem and start time.
    Example: 'perform_ideation_temp_free.py' -> 'perform_ideation_temp_free_20260902_113000.log'
    """
    if start_time is None:
        start_time = datetime.now()
    
    stem = Path(name_or_path).stem
    ts_str = start_time.strftime("%Y%m%d_%H%M%S")
    return f"{stem}_{ts_str}.log"


def setup_logger(
    name_or_path: str,
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    to_console: bool = True,
    start_time: Optional[datetime] = None,
) -> Tuple[logging.Logger, str]:
    """
    Configure and return a Python `logging.Logger` and the created log file path.

    Parameters
    ----------
    name_or_path : str
        Script path (e.g. `__file__`) or logger name.
    log_dir : str, optional
        Destination directory. Defaults to `/project_root/log`.
    level : int
        Logging level (default: `logging.INFO`).
    to_console : bool
        Whether to attach a console StreamHandler (default: True).
    start_time : datetime, optional
        Start time for the log filename timestamp.

    Returns
    -------
    logger : logging.Logger
    log_path : str
    """
    if log_dir is None:
        log_dir = get_default_log_dir()
    os.makedirs(log_dir, exist_ok=True)

    log_filename = create_log_filename(name_or_path, start_time)
    log_path = osp.join(log_dir, log_filename)

    logger_name = Path(name_or_path).stem
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Avoid duplicate handlers if setup_logger is called multiple times on the same logger
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # File Handler
        fh = logging.FileHandler(log_path, encoding="utf-8", mode="a")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Console Handler
        if to_console:
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(level)
            ch.setFormatter(formatter)
            logger.addHandler(ch)

    return logger, log_path


class _TeeStream:
    """Helper stream that duplicates writes to multiple streams safely."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            try:
                stream.write(data)
                stream.flush()
            except Exception:
                pass

    def flush(self) -> None:
        for stream in self.streams:
            try:
                stream.flush()
            except Exception:
                pass

    def fileno(self):
        return self.streams[0].fileno()


@contextmanager
def tee_to_log_file(log_path: str):
    """
    Context manager that tees stdout and stderr to both the terminal and *log_path*.
    """
    os.makedirs(osp.dirname(osp.abspath(log_path)), exist_ok=True)
    fh = open(log_path, "a", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    sys.stdout = _TeeStream(original_stdout, fh)
    sys.stderr = _TeeStream(original_stderr, fh)
    try:
        yield fh
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        fh.close()


def setup_file_logging(
    name_or_path: str,
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    start_time: Optional[datetime] = None,
) -> Tuple[logging.Logger, str]:
    """
    High-level helper:
    1. Sets up Python logging logger.
    2. Opens and attaches a Tee on sys.stdout & sys.stderr to the timestamped log file.

    Returns
    -------
    logger : logging.Logger
    log_path : str
    """
    if log_dir is None:
        log_dir = get_default_log_dir()
    os.makedirs(log_dir, exist_ok=True)

    log_filename = create_log_filename(name_or_path, start_time)
    log_path = osp.join(log_dir, log_filename)

    logger, _ = setup_logger(
        name_or_path=name_or_path,
        log_dir=log_dir,
        level=level,
        to_console=False,  # Console output is handled by stdout tee
        start_time=start_time,
    )

    # Attach tee globally for stdout & stderr
    fh = open(log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = _TeeStream(sys.stdout, fh)
    sys.stderr = _TeeStream(sys.stderr, fh)

    return logger, log_path
