"""
src/stages/stage2_plots.py
==========================
Stage 2 — Plot aggregation.

Public API
----------
run_plot_stage(idea_dir, model, stage_log_dir) -> None
"""

import os.path as osp
import shutil
import traceback

from src.io_utils import redirect_stdout_stderr_to_file
from src.logger import create_log_filename, get_default_log_dir

try:
    from ai_scientist.perform_plotting import aggregate_plots
except ImportError:
    aggregate_plots = None


def run_plot_stage(idea_dir: str, model: str, stage_log_dir: str) -> bool:
    """
    Aggregate experiment plots using *model* and clean up raw result files.

    Parameters
    ----------
    idea_dir : str
        Root output directory for this experiment run.
    model : str
        LLM identifier for ``aggregate_plots``.
    stage_log_dir : str
        Directory where stage log files are written.
    """
    log_path = osp.join(stage_log_dir, "02_plot_generation.log")
    central_log_path = osp.join(get_default_log_dir(), create_log_filename("stage2_plots.py"))
    print(f"[Stage 2] Log → {log_path}")
    print(f"[Stage 2] Central Log → {central_log_path}")

    try:
        with redirect_stdout_stderr_to_file(log_path, central_log_path):
            aggregate_plots(base_folder=idea_dir, model=model)
        print("[Stage 2] Completed successfully.")
        return True
    except Exception:
        with open(log_path, "a") as fh:
            fh.write("\n\n=== EXCEPTION IN STAGE 2 ===\n")
            traceback.print_exc(file=fh)
        with open(central_log_path, "a") as fh:
            fh.write("\n\n=== EXCEPTION IN STAGE 2 ===\n")
            traceback.print_exc(file=fh)
        print(f"[Stage 2] ERROR: exception raised — see {log_path}")
        return False
