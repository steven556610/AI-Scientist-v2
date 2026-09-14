"""
tests/test_generate_base_code.py
================================
Unit and boundary tests for ai_scientist/generate_base_code.py.

Tests:
- _parse_file_blocks: valid format, empty response, malformed blocks
- _collect_reference_snippet: real dir, empty dir, non-existent dir
- load_code_folder: re-exported from io_utils (boundary coverage)
- generate_base_code: mocked LLM path (success + parse failure)
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_scientist.generate_base_code import (
    _collect_reference_snippet,
    _parse_file_blocks,
    generate_base_code,
)


VALID_RESPONSE = """
--- FILE: runfile.py ---
print("hello")
--- END FILE ---

--- FILE: utils/helpers.py ---
def noop(): pass
--- END FILE ---
"""

MALFORMED_RESPONSE = "No file blocks here at all."


# ---------------------------------------------------------------------------
# _parse_file_blocks
# ---------------------------------------------------------------------------

class TestParseFileBlocks:
    def test_parses_valid_blocks(self):
        files = _parse_file_blocks(VALID_RESPONSE)
        assert "runfile.py" in files
        assert "utils/helpers.py" in files
        assert 'print("hello")' in files["runfile.py"]

    def test_returns_empty_dict_on_no_blocks(self):
        files = _parse_file_blocks(MALFORMED_RESPONSE)
        assert files == {}

    def test_empty_string_returns_empty(self):
        assert _parse_file_blocks("") == {}

    def test_strips_path_whitespace(self):
        response = "--- FILE:   spaced.py   ---\ncode\n--- END FILE ---"
        files = _parse_file_blocks(response)
        assert "spaced.py" in files

    def test_content_preserved(self):
        response = "--- FILE: a.py ---\nline1\nline2\n--- END FILE ---"
        files = _parse_file_blocks(response)
        assert "line1" in files["a.py"]
        assert "line2" in files["a.py"]


# ---------------------------------------------------------------------------
# _collect_reference_snippet
# ---------------------------------------------------------------------------

class TestCollectReferenceSnippet:
    def test_empty_dir_returns_empty_string(self, tmp_path):
        result = _collect_reference_snippet(str(tmp_path))
        assert result == ""

    def test_nonexistent_dir_returns_empty(self):
        result = _collect_reference_snippet("/no/such/dir")
        assert result == ""

    def test_none_returns_empty(self):
        result = _collect_reference_snippet(None)
        assert result == ""

    def test_collects_py_files(self, tmp_path):
        (tmp_path / "model.py").write_text("class Model: pass\n")
        result = _collect_reference_snippet(str(tmp_path), max_chars=5000)
        assert "model.py" in result
        assert "Model" in result

    def test_respects_max_chars(self, tmp_path):
        (tmp_path / "a.py").write_text("x" * 1000)
        (tmp_path / "b.py").write_text("y" * 1000)
        result = _collect_reference_snippet(str(tmp_path), max_chars=500)
        assert len(result) <= 600  # small overhead for headers


# ---------------------------------------------------------------------------
# generate_base_code (mocked LLM)
# ---------------------------------------------------------------------------

class TestGenerateBaseCode:
    @patch("ai_scientist.generate_base_code.get_response_from_llm")
    def test_writes_files_on_valid_response(self, mock_llm, tmp_path):
        mock_llm.return_value = (VALID_RESPONSE, {})
        client = MagicMock()
        idea = {"Name": "test", "Experiment": "test experiment"}

        ok = generate_base_code(
            idea=idea,
            output_dir=str(tmp_path),
            client=client,
            model="test-model",
        )
        assert ok is True
        assert (tmp_path / "runfile.py").exists()
        assert (tmp_path / "utils" / "helpers.py").exists()

    @patch("ai_scientist.generate_base_code.get_response_from_llm")
    def test_saves_raw_response_on_parse_failure(self, mock_llm, tmp_path):
        mock_llm.return_value = (MALFORMED_RESPONSE, {})
        client = MagicMock()

        ok = generate_base_code(
            idea={"Name": "test"},
            output_dir=str(tmp_path),
            client=client,
            model="test-model",
        )
        assert ok is False
        assert (tmp_path / "llm_raw_response.txt").exists()

    @patch("ai_scientist.generate_base_code.get_response_from_llm", side_effect=Exception("net"))
    def test_llm_failure_returns_false(self, mock_llm, tmp_path):
        ok = generate_base_code(
            idea={"Name": "test"},
            output_dir=str(tmp_path),
            client=MagicMock(),
            model="test-model",
        )
        assert ok is False

    @patch("ai_scientist.generate_base_code.get_response_from_llm")
    def test_does_not_overwrite_by_default(self, mock_llm, tmp_path):
        existing = tmp_path / "runfile.py"
        existing.write_text("original content\n")
        mock_llm.return_value = (VALID_RESPONSE, {})

        generate_base_code(
            idea={"Name": "test"},
            output_dir=str(tmp_path),
            client=MagicMock(),
            model="test-model",
            overwrite=False,
        )
        assert existing.read_text() == "original content\n"

    @patch("ai_scientist.generate_base_code.get_response_from_llm")
    def test_overwrites_when_flag_set(self, mock_llm, tmp_path):
        existing = tmp_path / "runfile.py"
        existing.write_text("original content\n")
        mock_llm.return_value = (VALID_RESPONSE, {})

        generate_base_code(
            idea={"Name": "test"},
            output_dir=str(tmp_path),
            client=MagicMock(),
            model="test-model",
            overwrite=True,
        )
        assert "original content" not in existing.read_text()

    @patch("ai_scientist.generate_base_code.get_response_from_llm")
    def test_manifest_written(self, mock_llm, tmp_path):
        mock_llm.return_value = (VALID_RESPONSE, {})
        generate_base_code(
            idea={"Name": "test"},
            output_dir=str(tmp_path),
            client=MagicMock(),
            model="test-model",
        )
        manifest = json.loads((tmp_path / "code_manifest.json").read_text())
        assert "generated_files" in manifest
        assert "entry_point" in manifest
