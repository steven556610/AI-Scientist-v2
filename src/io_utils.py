"""
src/io_utils.py
===============
I/O utilities shared across the AI-Scientist-v2 launcher and tools.

Exported symbols:
    resolve_idea_source(load_ideas, load_code)  → (json_path, code_str | None)
    load_code_folder(folder)                    → str
    save_token_tracker(idea_dir)
    find_pdf_path_for_review(idea_dir)          → str | None
    redirect_stdout_stderr_to_file(log_path)    — context manager (Tee)
"""

import json
import os
import os.path as osp
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Idea source resolution
# ---------------------------------------------------------------------------

def resolve_idea_source(
    load_ideas_arg: str,
    load_code: bool = False,
) -> Tuple[str, Optional[str]]:
    """
    Determine ``(ideas_json_path, code_string)`` from ``--load_ideas``.

    ``load_ideas_arg`` may be:
      - A direct ``*.json`` file path.
      - A folder containing exactly one ``*.json`` file.

    When ``load_code=True``:
      1. Prefer a ``code/`` sub-folder (multi-file template).
      2. Fall back to a same-name ``.py`` file.
      3. Warn and return ``None`` if neither is found.

    Returns
    -------
    ideas_json_path : str
        Absolute path to the ideas JSON file.
    code : str | None
        Concatenated code string, or ``None`` if not requested / not found.

    Raises
    ------
    FileNotFoundError
        If the argument does not point to a valid JSON file or folder.
    """
    arg = load_ideas_arg

    if arg.endswith(".json") and osp.isfile(arg):
        ideas_json = osp.abspath(arg)
        idea_folder = osp.dirname(ideas_json)
    elif osp.isdir(arg):
        idea_folder = osp.abspath(arg)
        candidates = sorted(
            f for f in os.listdir(idea_folder) if f.endswith(".json")
        )
        if not candidates:
            raise FileNotFoundError(
                f"No .json file found in idea folder: {idea_folder}"
            )
        ideas_json = osp.join(idea_folder, candidates[0])
    else:
        raise FileNotFoundError(
            f"--load_ideas must be a .json file or a folder; got: {arg!r}"
        )

    code: Optional[str] = None
    if load_code:
        code_folder = osp.join(idea_folder, "code")
        if osp.isdir(code_folder):
            code = load_code_folder(code_folder)
            print(f"[io_utils] Loaded multi-file code template from: {code_folder}")
        else:
            code_py = osp.splitext(ideas_json)[0] + ".py"
            if osp.isfile(code_py):
                with open(code_py, "r", encoding="utf-8") as fh:
                    code = fh.read()
                print(f"[io_utils] Loaded single-file code template: {code_py}")
            else:
                print(
                    f"[io_utils] WARNING: --load_code set but neither "
                    f"{code_folder} nor {code_py} found. "
                    "Proceeding without code template."
                )

    return ideas_json, code


def load_code_folder(code_dir: str) -> str:
    """
    Concatenate all ``*.py`` files inside *code_dir* into one string,
    labelled with ``# ===== <relative_path> =====`` headers.

    Recursively walks the directory; skips hidden directories.
    """
    if not osp.isdir(code_dir):
        raise FileNotFoundError(f"Code folder not found: {code_dir}")

    parts = []
    for root, dirs, files in os.walk(code_dir):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for fname in sorted(files):
            if not fname.endswith(".py"):
                continue
            fpath = osp.join(root, fname)
            rel = osp.relpath(fpath, code_dir)
            with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
                content = fh.read()
            parts.append(f"# ===== {rel} =====\n{content}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Token tracker persistence
# ---------------------------------------------------------------------------

def save_token_tracker(idea_dir: str) -> None:
    """Save token usage summary and interaction log to *idea_dir*."""
    try:
        from ai_scientist.utils.token_tracker import token_tracker  # noqa: PLC0415
        with open(osp.join(idea_dir, "token_tracker.json"), "w") as f:
            json.dump(token_tracker.get_summary(), f)
        with open(osp.join(idea_dir, "token_tracker_interactions.json"), "w") as f:
            json.dump(token_tracker.get_interactions(), f)
    except Exception as exc:
        print(f"[io_utils] WARNING: Could not save token tracker: {exc}")


# ---------------------------------------------------------------------------
# PDF discovery
# ---------------------------------------------------------------------------

def find_pdf_path_for_review(idea_dir: str) -> Optional[str]:
    """
    Return the path to the most recent reflection PDF in *idea_dir*, or
    ``None`` if none is found.

    Preference order:
      1. Highest-numbered ``*reflection<N>*.pdf``
      2. Any ``*reflection*.pdf`` (alphabetically last)
      3. ``None``
    """
    try:
        pdf_files = [f for f in os.listdir(idea_dir) if f.endswith(".pdf")]
    except FileNotFoundError:
        return None

    reflection_pdfs = [f for f in pdf_files if "reflection" in f]
    if not reflection_pdfs:
        return None

    numbered = []
    for fname in reflection_pdfs:
        m = re.search(r"reflection[_.]?(\d+)", fname)
        if m:
            numbered.append((int(m.group(1)), fname))

    if numbered:
        return osp.join(idea_dir, max(numbered)[1])
    return osp.join(idea_dir, sorted(reflection_pdfs)[-1])


# ---------------------------------------------------------------------------
# Tee logging context manager
# ---------------------------------------------------------------------------

@contextmanager
def redirect_stdout_stderr_to_file(log_file_path: str, *extra_log_paths: str):
    """
    Context manager that redirects stdout **and** stderr to both the original
    streams and one or more log files simultaneously (Tee behaviour).

    Creates parent directories for all log paths automatically.

    Yields the primary open log file handle.

    Example::

        with redirect_stdout_stderr_to_file("logs/01_experiment.log", "log/stage1_20260902.log") as log:
            perform_experiments_bfts(config_path)
    """
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    all_paths = [log_file_path] + list(extra_log_paths)
    open_files = []
    for p in all_paths:
        if p:
            os.makedirs(osp.dirname(osp.abspath(p)), exist_ok=True)
            open_files.append(open(p, "a", encoding="utf-8", buffering=1))

    class _Tee:
        """Write to multiple streams; silently skip broken ones."""

        def __init__(self, *streams):
            self.streams = streams

        def write(self, data: str) -> None:
            for stream in self.streams:
                try:
                    stream.write(data)
                except Exception:
                    pass

        def flush(self) -> None:
            for stream in self.streams:
                try:
                    stream.flush()
                except Exception:
                    pass

        def fileno(self):
            # Required for subprocess compatibility.
            return self.streams[0].fileno()

    sys.stdout = _Tee(original_stdout, *open_files)
    sys.stderr = _Tee(original_stderr, *open_files)
    try:
        yield open_files[0] if open_files else None
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        for fh in open_files:
            try:
                fh.close()
            except Exception:
                pass
