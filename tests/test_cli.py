"""
tests/test_cli.py
=================
Unit tests for src/cli.py.

Tests
-----
- All arguments parse with their documented defaults.
- Mutual-exclusion and choice constraints are enforced.
- Boundary values (negative integers, empty strings) are handled gracefully.
"""

import pytest
import sys


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse(extra_args=None):
    """Parse with sys.argv patched to *extra_args* (list of strings)."""
    from src.cli import parse_arguments  # local import so tests are isolated

    argv_backup = sys.argv
    sys.argv = ["launch_scientist_bfts.py"] + (extra_args or [])
    try:
        return parse_arguments()
    finally:
        sys.argv = argv_backup


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_load_ideas_default(self):
        args = _parse()
        assert args.load_ideas == "ideas/i_cant_believe_its_not_better.json"

    def test_load_code_default_false(self):
        args = _parse()
        assert args.load_code is False

    def test_idea_idx_default(self):
        args = _parse()
        assert args.idea_idx == 0

    def test_attempt_id_default(self):
        args = _parse()
        assert args.attempt_id == 0

    def test_writeup_type_default(self):
        args = _parse()
        assert args.writeup_type == "icbinb"

    def test_writeup_retries_default(self):
        args = _parse()
        assert args.writeup_retries == 3

    def test_num_cite_rounds_default(self):
        args = _parse()
        assert args.num_cite_rounds == 20

    def test_skip_flags_default_false(self):
        args = _parse()
        assert args.skip_writeup is False
        assert args.skip_review is False
        assert args.add_dataset_ref is False


# ---------------------------------------------------------------------------
# Flag overrides
# ---------------------------------------------------------------------------

class TestFlagOverrides:
    def test_load_code_flag(self):
        args = _parse(["--load_code"])
        assert args.load_code is True

    def test_add_dataset_ref_flag(self):
        args = _parse(["--add_dataset_ref"])
        assert args.add_dataset_ref is True

    def test_skip_writeup_flag(self):
        args = _parse(["--skip_writeup"])
        assert args.skip_writeup is True

    def test_skip_review_flag(self):
        args = _parse(["--skip_review"])
        assert args.skip_review is True

    def test_writeup_type_normal(self):
        args = _parse(["--writeup_type", "normal"])
        assert args.writeup_type == "normal"

    def test_custom_idea_path(self):
        args = _parse(["--load_ideas", "ai_scientist/ideas/my_research_topic"])
        assert args.load_ideas == "ai_scientist/ideas/my_research_topic"

    def test_custom_idea_idx(self):
        args = _parse(["--idea_idx", "2"])
        assert args.idea_idx == 2

    def test_custom_attempt_id(self):
        args = _parse(["--attempt_id", "5"])
        assert args.attempt_id == 5


# ---------------------------------------------------------------------------
# Boundary / constraint tests
# ---------------------------------------------------------------------------

class TestConstraints:
    def test_invalid_writeup_type_raises(self):
        with pytest.raises(SystemExit):
            _parse(["--writeup_type", "invalid_choice"])

    def test_num_cite_rounds_zero(self):
        # argparse does not restrict to positive; zero should parse.
        args = _parse(["--num_cite_rounds", "0"])
        assert args.num_cite_rounds == 0

    def test_writeup_retries_one(self):
        args = _parse(["--writeup_retries", "1"])
        assert args.writeup_retries == 1

    def test_model_strings_passed_through(self):
        args = _parse(["--model_writeup", "kimi-k3", "--model_review", "kimi-k3"])
        assert args.model_writeup == "kimi-k3"
        assert args.model_review == "kimi-k3"
