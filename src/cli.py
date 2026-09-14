"""
src/cli.py
==========
All argparse definitions for launch_scientist_bfts.py.

Usage:
    from src.cli import parse_arguments
    args = parse_arguments()
"""

import argparse


def parse_arguments() -> argparse.Namespace:
    """Parse and return CLI arguments for the AI-Scientist launcher."""
    parser = argparse.ArgumentParser(
        description="Run AI-Scientist experiments end-to-end (BFTS variant)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── Resume options ───────────────────────────────────────────────────
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to an existing experiment directory to resume from. Skips completed stages.",
    )

    # ── Idea source ──────────────────────────────────────────────────────
    parser.add_argument(
        "--load_ideas",
        type=str,
        default="ideas/i_cant_believe_its_not_better.json",
        help=(
            "Path to a JSON file or a folder containing <name>.json. "
            "If a folder, also looks for a code/ sub-folder as the code template."
        ),
    )
    parser.add_argument(
        "--load_code",
        action="store_true",
        help=(
            "Load experiment code from a code/ sub-folder (preferred) or a "
            ".py file co-located with the ideas JSON."
        ),
    )
    parser.add_argument(
        "--idea_idx",
        type=int,
        default=0,
        help="Index of the idea to run when the JSON is a list.",
    )
    parser.add_argument(
        "--attempt_id",
        type=int,
        default=0,
        help="Attempt ID for distinguishing parallel runs of the same idea.",
    )

    # ── Dataset reference ────────────────────────────────────────────────
    parser.add_argument(
        "--add_dataset_ref",
        action="store_true",
        help="Prepend hf_dataset_reference.py to the injected code.",
    )

    # ── Stage: Writeup ───────────────────────────────────────────────────
    parser.add_argument(
        "--writeup_type",
        type=str,
        default="icbinb",
        choices=["normal", "icbinb"],
        help="Paper format (normal = 8 pages, icbinb = 4 pages).",
    )
    parser.add_argument(
        "--writeup_retries",
        type=int,
        default=3,
        help="Maximum number of writeup compilation attempts.",
    )
    parser.add_argument(
        "--skip_writeup",
        action="store_true",
        help="Skip Stage 3 (writeup) entirely.",
    )

    # ── Stage: Review ────────────────────────────────────────────────────
    parser.add_argument(
        "--skip_review",
        action="store_true",
        help="Skip Stage 4 (review) entirely.",
    )

    # ── Model selection ──────────────────────────────────────────────────
    parser.add_argument(
        "--model_agg_plots",
        type=str,
        default="o3-mini-2025-01-31",
        help="LLM for Stage 2 plot aggregation.",
    )
    parser.add_argument(
        "--model_writeup",
        type=str,
        default="o1-preview-2024-09-12",
        help="Primary (large) LLM for Stage 3 writeup.",
    )
    parser.add_argument(
        "--model_writeup_small",
        type=str,
        default="gpt-4o-2024-05-13",
        help="Secondary (small) LLM for Stage 3 writeup.",
    )
    parser.add_argument(
        "--model_citation",
        type=str,
        default="gpt-4o-2024-11-20",
        help="LLM for citation gathering in Stage 3.",
    )
    parser.add_argument(
        "--num_cite_rounds",
        type=int,
        default=20,
        help="Number of citation-gathering rounds.",
    )
    parser.add_argument(
        "--model_review",
        type=str,
        default="gpt-4o-2024-11-20",
        help="LLM for Stage 4 paper review.",
    )

    return parser.parse_args()
