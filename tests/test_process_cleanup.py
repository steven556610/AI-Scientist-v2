"""
tests/test_process_cleanup.py
==============================
Unit tests for src/process_cleanup.py.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestCleanupChildProcesses:
    @patch("src.process_cleanup.psutil")
    def test_sends_sigterm_to_children(self, mock_psutil):
        from src.process_cleanup import cleanup_child_processes

        child = MagicMock()
        mock_proc = MagicMock()
        mock_proc.children.return_value = [child]
        mock_psutil.Process.return_value = mock_proc
        mock_psutil.wait_procs.return_value = ([child], [])
        mock_psutil.NoSuchProcess = Exception
        mock_psutil.AccessDenied = Exception

        import os
        cleanup_child_processes(timeout=0.1)
        child.send_signal.assert_called_once()

    @patch("src.process_cleanup.psutil")
    def test_force_kills_survivors(self, mock_psutil):
        from src.process_cleanup import cleanup_child_processes

        survivor = MagicMock()
        mock_proc = MagicMock()
        mock_proc.children.return_value = [survivor]
        mock_psutil.Process.return_value = mock_proc
        mock_psutil.wait_procs.return_value = ([], [survivor])
        mock_psutil.NoSuchProcess = Exception
        mock_psutil.AccessDenied = Exception

        cleanup_child_processes(timeout=0.1)
        survivor.kill.assert_called_once()

    @patch("src.process_cleanup.psutil")
    def test_no_children_does_not_raise(self, mock_psutil):
        from src.process_cleanup import cleanup_child_processes

        mock_proc = MagicMock()
        mock_proc.children.return_value = []
        mock_psutil.Process.return_value = mock_proc
        mock_psutil.wait_procs.return_value = ([], [])
        mock_psutil.NoSuchProcess = Exception
        mock_psutil.AccessDenied = Exception

        cleanup_child_processes()  # should not raise


class TestCleanupWithoutPsutil:
    def test_missing_psutil_does_not_raise(self, monkeypatch):
        """If psutil is not installed, the function should warn and return."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "psutil":
                raise ImportError("no psutil")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        from src import process_cleanup
        # Force re-import to hit the mock
        import importlib
        importlib.reload(process_cleanup)
        process_cleanup.cleanup_child_processes()  # should not raise
