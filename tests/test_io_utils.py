"""
tests/test_io_utils.py
======================
Unit and boundary tests for src/io_utils.py.

Covers:
- resolve_idea_source: valid json file, valid folder, missing json, non-existent path
- load_code_folder: multi-file, empty folder, non-existent folder
- find_pdf_path_for_review: numbered PDFs, unnumbered, empty dir
- redirect_stdout_stderr_to_file: Tee writes to both streams and file
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

from src.io_utils import (
    find_pdf_path_for_review,
    load_code_folder,
    redirect_stdout_stderr_to_file,
    resolve_idea_source,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def idea_json(tmp_path):
    """A minimal valid idea JSON file."""
    ideas = [{"Name": "test_idea", "Title": "Test", "Experiment": "None"}]
    p = tmp_path / "my_idea" / "my_idea.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(ideas))
    return p


@pytest.fixture()
def idea_folder_with_code(tmp_path):
    """Idea folder containing a code/ sub-folder with two .py files."""
    folder = tmp_path / "idea_with_code"
    folder.mkdir()
    ideas = [{"Name": "idea_with_code"}]
    (folder / "idea_with_code.json").write_text(json.dumps(ideas))
    code_dir = folder / "code"
    code_dir.mkdir()
    (code_dir / "runfile.py").write_text("# runfile\nprint('hello')\n")
    (code_dir / "utils.py").write_text("# utils\ndef helper(): pass\n")
    return folder


# ---------------------------------------------------------------------------
# resolve_idea_source
# ---------------------------------------------------------------------------

class TestResolveIdeaSource:
    def test_direct_json_path_no_code(self, idea_json):
        json_path, code = resolve_idea_source(str(idea_json), load_code=False)
        assert json_path.endswith(".json")
        assert code is None

    def test_folder_path_resolves_json(self, idea_json):
        folder = str(idea_json.parent)
        json_path, code = resolve_idea_source(folder, load_code=False)
        assert json_path.endswith(".json")
        assert code is None

    def test_load_code_from_code_folder(self, idea_folder_with_code):
        _, code = resolve_idea_source(str(idea_folder_with_code), load_code=True)
        assert code is not None
        assert "runfile" in code
        assert "utils" in code

    def test_load_code_missing_falls_back_to_none(self, idea_json):
        # No code/ folder and no matching .py → should not raise, returns None
        _, code = resolve_idea_source(str(idea_json.parent), load_code=True)
        assert code is None

    def test_missing_path_raises(self):
        with pytest.raises(FileNotFoundError):
            resolve_idea_source("/nonexistent/path/idea.json", load_code=False)

    def test_folder_without_json_raises(self, tmp_path):
        empty = tmp_path / "empty_folder"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            resolve_idea_source(str(empty), load_code=False)


# ---------------------------------------------------------------------------
# load_code_folder
# ---------------------------------------------------------------------------

class TestLoadCodeFolder:
    def test_concatenates_py_files(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1\n")
        (tmp_path / "b.py").write_text("y = 2\n")
        result = load_code_folder(str(tmp_path))
        assert "a.py" in result
        assert "b.py" in result
        assert "x = 1" in result
        assert "y = 2" in result

    def test_ignores_non_py_files(self, tmp_path):
        (tmp_path / "data.csv").write_text("a,b\n1,2\n")
        (tmp_path / "code.py").write_text("pass\n")
        result = load_code_folder(str(tmp_path))
        assert "data.csv" not in result

    def test_empty_folder_returns_empty_string(self, tmp_path):
        result = load_code_folder(str(tmp_path))
        assert result == ""

    def test_nonexistent_folder_raises(self):
        with pytest.raises(FileNotFoundError):
            load_code_folder("/absolutely/does/not/exist")

    def test_nested_py_files_included(self, tmp_path):
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "nested.py").write_text("z = 3\n")
        result = load_code_folder(str(tmp_path))
        assert "nested.py" in result
        assert "z = 3" in result


# ---------------------------------------------------------------------------
# find_pdf_path_for_review
# ---------------------------------------------------------------------------

class TestFindPdfPathForReview:
    def test_returns_highest_numbered_reflection(self, tmp_path):
        (tmp_path / "paper_reflection1.pdf").touch()
        (tmp_path / "paper_reflection3.pdf").touch()
        result = find_pdf_path_for_review(str(tmp_path))
        assert result.endswith("reflection3.pdf")

    def test_returns_none_if_no_reflection_pdf(self, tmp_path):
        (tmp_path / "some_other.pdf").touch()
        assert find_pdf_path_for_review(str(tmp_path)) is None

    def test_returns_none_if_no_pdfs(self, tmp_path):
        assert find_pdf_path_for_review(str(tmp_path)) is None

    def test_returns_none_for_missing_dir(self):
        assert find_pdf_path_for_review("/no/such/directory") is None

    def test_unnumbered_reflection_returned(self, tmp_path):
        (tmp_path / "output_reflection.pdf").touch()
        result = find_pdf_path_for_review(str(tmp_path))
        assert result is not None
        assert "reflection" in result


# ---------------------------------------------------------------------------
# redirect_stdout_stderr_to_file
# ---------------------------------------------------------------------------

class TestRedirectStdoutStderr:
    def test_tee_writes_to_log_file(self, tmp_path):
        log = tmp_path / "stage.log"
        with redirect_stdout_stderr_to_file(str(log)):
            print("hello from stdout")
        assert "hello from stdout" in log.read_text()

    def test_restores_original_streams(self, tmp_path):
        log = tmp_path / "stage.log"
        original_out = sys.stdout
        with redirect_stdout_stderr_to_file(str(log)):
            pass
        assert sys.stdout is original_out

    def test_creates_parent_dirs(self, tmp_path):
        nested_log = tmp_path / "a" / "b" / "c" / "test.log"
        with redirect_stdout_stderr_to_file(str(nested_log)):
            print("nested")
        assert nested_log.exists()

    def test_appends_on_multiple_opens(self, tmp_path):
        log = tmp_path / "stage.log"
        with redirect_stdout_stderr_to_file(str(log)):
            print("first")
        with redirect_stdout_stderr_to_file(str(log)):
            print("second")
        content = log.read_text()
        assert "first" in content
        assert "second" in content
