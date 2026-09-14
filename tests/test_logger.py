"""
tests/test_logger.py
====================
Unit tests for src/logger.py.
"""

import logging
import os
import os.path as osp
from datetime import datetime
from pathlib import Path
import pytest

from src.logger import (
    create_log_filename,
    get_default_log_dir,
    setup_file_logging,
    setup_logger,
    tee_to_log_file,
)


def test_create_log_filename():
    dt = datetime(2026, 9, 2, 11, 30, 45)
    name = create_log_filename("ai_scientist/perform_ideation_temp_free.py", start_time=dt)
    assert name == "perform_ideation_temp_free_20260902_113045.log"

    name_main = create_log_filename("main.py", start_time=dt)
    assert name_main == "main_20260902_113045.log"


def test_get_default_log_dir():
    log_dir = get_default_log_dir()
    assert osp.isdir(log_dir)
    assert log_dir.endswith("log")


def test_setup_logger_creates_file_and_logs(tmp_path):
    logger, log_path = setup_logger(
        "test_module",
        log_dir=str(tmp_path),
        to_console=False,
        start_time=datetime(2026, 9, 2, 12, 0, 0),
    )
    assert osp.exists(log_path)
    logger.info("Testing setup_logger message")

    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "Testing setup_logger message" in content
    assert "[INFO]" in content


def test_tee_to_log_file(tmp_path):
    log_file = tmp_path / "test_tee.log"
    with tee_to_log_file(str(log_file)):
        print("Hello from stdout tee")
    
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "Hello from stdout tee" in content


def test_setup_file_logging(tmp_path):
    logger, log_path = setup_file_logging(
        "custom_runner.py",
        log_dir=str(tmp_path),
        start_time=datetime(2026, 9, 2, 13, 0, 0),
    )
    assert osp.exists(log_path)
    print("Direct print to file logging")
    logger.info("Logger info to file logging")

    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "Direct print to file logging" in content
    assert "Logger info to file logging" in content
