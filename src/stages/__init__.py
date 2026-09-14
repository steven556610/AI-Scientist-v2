"""src/stages/__init__.py — convenience re-exports for stage runners."""
from .stage1_experiments import run_experiment_stage
from .stage2_plots import run_plot_stage
from .stage3_writeup import run_writeup_stage
from .stage4_review import run_review_stage

__all__ = [
    "run_experiment_stage",
    "run_plot_stage",
    "run_writeup_stage",
    "run_review_stage",
]
