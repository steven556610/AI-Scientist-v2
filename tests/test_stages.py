"""
tests/test_stages.py
====================
Unit tests for src/stages/* using mocks.

Each stage runner is tested in isolation by mocking out its heavy AI-Scientist
dependencies.  Tests verify:
- Correct delegation to underlying functions
- Log file creation
- Return values for success / failure paths
- Exception handling (exception is caught, logged, not re-raised)
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_args(**kwargs):
    """Build a minimal argparse.Namespace for stage runners."""
    import argparse

    defaults = dict(
        writeup_type="icbinb",
        num_cite_rounds=2,
        model_citation="test-model",
        model_writeup="test-model",
        model_writeup_small="test-model-small",
        writeup_retries=2,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# Stage 1 — run_experiment_stage
# ---------------------------------------------------------------------------

class TestStage1:
    @patch(
        "src.stages.stage1_experiments.perform_experiments_bfts",
        return_value=None,
    )
    def test_returns_true_when_results_exist(self, mock_bfts, tmp_path):
        from src.stages.stage1_experiments import run_experiment_stage

        # Create the expected results directory with a file
        results_dir = tmp_path / "logs" / "0-run" / "experiment_results"
        results_dir.mkdir(parents=True)
        (results_dir / "result.json").write_text("{}")

        log_dir = tmp_path / "stage_logs"
        log_dir.mkdir()

        ok = run_experiment_stage(
            idea_config_path="dummy_config.yaml",
            idea_dir=str(tmp_path),
            stage_log_dir=str(log_dir),
        )
        assert ok is True

    @patch(
        "src.stages.stage1_experiments.perform_experiments_bfts",
        return_value=None,
    )
    def test_returns_false_when_results_missing(self, mock_bfts, tmp_path):
        from src.stages.stage1_experiments import run_experiment_stage

        log_dir = tmp_path / "stage_logs"
        log_dir.mkdir()

        ok = run_experiment_stage(
            idea_config_path="dummy_config.yaml",
            idea_dir=str(tmp_path),
            stage_log_dir=str(log_dir),
        )
        assert ok is False

    @patch(
        "src.stages.stage1_experiments.perform_experiments_bfts",
        side_effect=RuntimeError("boom"),
    )
    def test_exception_does_not_propagate(self, mock_bfts, tmp_path):
        from src.stages.stage1_experiments import run_experiment_stage

        log_dir = tmp_path / "stage_logs"
        log_dir.mkdir()

        # Should return False, not raise
        ok = run_experiment_stage("cfg.yaml", str(tmp_path), str(log_dir))
        assert ok is False

    @patch("src.stages.stage1_experiments.perform_experiments_bfts", return_value=None)
    def test_log_file_created(self, mock_bfts, tmp_path):
        from src.stages.stage1_experiments import run_experiment_stage

        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        run_experiment_stage("cfg.yaml", str(tmp_path), str(log_dir))
        assert (log_dir / "01_experiment_execution.log").exists()


# ---------------------------------------------------------------------------
# Stage 2 — run_plot_stage
# ---------------------------------------------------------------------------

class TestStage2:
    @patch("src.stages.stage2_plots.aggregate_plots", return_value=None)
    def test_runs_without_error(self, mock_agg, tmp_path):
        from src.stages.stage2_plots import run_plot_stage

        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        run_plot_stage(str(tmp_path), "test-model", str(log_dir))
        mock_agg.assert_called_once()

    @patch("src.stages.stage2_plots.aggregate_plots", return_value=None)
    def test_removes_experiment_results(self, mock_agg, tmp_path):
        from src.stages.stage2_plots import run_plot_stage

        exp_res = tmp_path / "experiment_results"
        exp_res.mkdir()
        (exp_res / "dummy.json").write_text("{}")

        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        run_plot_stage(str(tmp_path), "test-model", str(log_dir))
        assert not exp_res.exists()

    @patch("src.stages.stage2_plots.aggregate_plots", side_effect=ValueError("bad"))
    def test_exception_does_not_propagate(self, mock_agg, tmp_path):
        from src.stages.stage2_plots import run_plot_stage

        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        # Should not raise
        run_plot_stage(str(tmp_path), "test-model", str(log_dir))

    @patch("src.stages.stage2_plots.aggregate_plots", return_value=None)
    def test_log_file_created(self, mock_agg, tmp_path):
        from src.stages.stage2_plots import run_plot_stage

        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        run_plot_stage(str(tmp_path), "test-model", str(log_dir))
        assert (log_dir / "02_plot_generation.log").exists()


# ---------------------------------------------------------------------------
# Stage 3 — run_writeup_stage
# ---------------------------------------------------------------------------

class TestStage3:
    @patch("src.stages.stage3_writeup.gather_citations", return_value="[1] Foo")
    @patch("src.stages.stage3_writeup.perform_icbinb_writeup", return_value=True)
    def test_returns_true_on_success(self, mock_writeup, mock_cit, tmp_path):
        from src.stages.stage3_writeup import run_writeup_stage

        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        ok = run_writeup_stage(str(tmp_path), _make_args(), str(log_dir))
        assert ok is True

    @patch("src.stages.stage3_writeup.gather_citations", return_value="")
    @patch("src.stages.stage3_writeup.perform_icbinb_writeup", return_value=False)
    def test_returns_false_when_all_retries_fail(self, mock_wu, mock_cit, tmp_path):
        from src.stages.stage3_writeup import run_writeup_stage

        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        ok = run_writeup_stage(str(tmp_path), _make_args(writeup_retries=2), str(log_dir))
        assert ok is False

    @patch("src.stages.stage3_writeup.gather_citations", side_effect=Exception("net"))
    def test_exception_does_not_propagate(self, mock_cit, tmp_path):
        from src.stages.stage3_writeup import run_writeup_stage

        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        ok = run_writeup_stage(str(tmp_path), _make_args(), str(log_dir))
        assert ok is False

    @patch("src.stages.stage3_writeup.gather_citations", return_value="")
    @patch("src.stages.stage3_writeup.perform_writeup", return_value=True)
    def test_normal_writeup_type(self, mock_wu, mock_cit, tmp_path):
        from src.stages.stage3_writeup import run_writeup_stage

        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        ok = run_writeup_stage(
            str(tmp_path), _make_args(writeup_type="normal"), str(log_dir)
        )
        assert ok is True
        mock_wu.assert_called()


# ---------------------------------------------------------------------------
# Stage 4 — run_review_stage
# ---------------------------------------------------------------------------

class TestStage4:
    @patch("src.stages.stage4_review.find_pdf_path_for_review", return_value=None)
    def test_skips_when_no_pdf(self, mock_pdf, tmp_path):
        from src.stages.stage4_review import run_review_stage

        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        # Should complete without error
        run_review_stage(str(tmp_path), "test-model", str(log_dir))

    @patch("src.stages.stage4_review.find_pdf_path_for_review")
    @patch("src.stages.stage4_review.load_paper", return_value="paper text")
    @patch("src.stages.stage4_review.create_client")
    @patch("src.stages.stage4_review.perform_review", return_value={"score": 7})
    @patch("src.stages.stage4_review.perform_imgs_cap_ref_review", return_value={})
    def test_writes_review_files(
        self, mock_vlm, mock_review, mock_client, mock_load, mock_pdf, tmp_path
    ):
        from src.stages.stage4_review import run_review_stage

        pdf_file = tmp_path / "paper_reflection1.pdf"
        pdf_file.touch()
        mock_pdf.return_value = str(pdf_file)
        mock_client.return_value = (MagicMock(), "test-model")

        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        run_review_stage(str(tmp_path), "test-model", str(log_dir))

        assert (tmp_path / "review_text.txt").exists()
        assert (tmp_path / "review_img_cap_ref.json").exists()

    @patch("src.stages.stage4_review.find_pdf_path_for_review")
    @patch("src.stages.stage4_review.load_paper", side_effect=RuntimeError("oops"))
    @patch("src.stages.stage4_review.create_client")
    def test_exception_does_not_propagate(self, mock_cl, mock_load, mock_pdf, tmp_path):
        from src.stages.stage4_review import run_review_stage

        pdf_file = tmp_path / "paper_reflection1.pdf"
        pdf_file.touch()
        mock_pdf.return_value = str(pdf_file)
        mock_cl.return_value = (MagicMock(), "test-model")

        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        run_review_stage(str(tmp_path), "test-model", str(log_dir))
