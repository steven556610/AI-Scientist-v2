"""
launch_scientist_bfts.py
========================
AI-Scientist-v2 experiment launcher (BFTS variant).

All heavy logic lives in src/.  This file only:
  1. Sets the environment root
  2. Parses arguments
  3. Resolves the idea source
  4. Builds the nested experiment directory
  5. Dispatches the four pipeline stages
  6. Cleans up child processes
"""

import json
import os
import os.path as osp
import sys
from datetime import datetime

from ai_scientist.treesearch.bfts_utils import edit_bfts_config_file, idea_to_markdown

from src.cli import parse_arguments
from src.io_utils import resolve_idea_source, save_token_tracker
from src.logger import setup_file_logging
from src.process_cleanup import cleanup_child_processes
from src.stages import (
    run_experiment_stage,
    run_plot_stage,
    run_review_stage,
    run_writeup_stage,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_stage(n: int, label: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}\n[STAGE {n}] {label}\n{bar}")
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def _build_experiment_dir(idea_name: str, attempt_id: int) -> str:
    """Return ``experiments/<idea_name>/<timestamp>_attempt_<id>`` path."""
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = osp.join("experiments", idea_name, f"{ts}_attempt_{attempt_id}")
    os.makedirs(path, exist_ok=True)
    return path


def _load_state(idea_dir: str) -> dict:
    state_file = osp.join(idea_dir, "run_state.json")
    if osp.exists(state_file):
        with open(state_file, "r") as f:
            return json.load(f)
    return {"stage1": "pending", "stage2": "pending", "stage3": "pending", "stage4": "pending"}


def _save_state(idea_dir: str, state: dict) -> None:
    state_file = osp.join(idea_dir, "run_state.json")
    with open(state_file, "w") as f:
        json.dump(state, f, indent=4)


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger, log_path = setup_file_logging(__file__)
    print(f"Logging initialized at: {log_path}")

    # ── Environment ─────────────────────────────────────────────────────
    root = osp.dirname(osp.abspath(__file__))
    os.environ["AI_SCIENTIST_ROOT"] = root
    print(f"AI_SCIENTIST_ROOT = {root}")

    args = parse_arguments()

    if args.resume_from:
        idea_dir = osp.abspath(args.resume_from)
        print(f"Resuming experiment from: {idea_dir}")
        if not osp.exists(idea_dir):
            print(f"[ERROR] Resume directory does not exist: {idea_dir}")
            sys.exit(1)
        
        state = _load_state(idea_dir)
        stage_log_dir = osp.join(idea_dir, "logs")
        idea_json_out = osp.join(idea_dir, "idea.json")
        idea_config_path = osp.join(idea_dir, "bfts_config.yaml")
        if not osp.exists(idea_config_path):
            idea_config_path = edit_bfts_config_file("bfts_config.yaml", idea_dir, idea_json_out)
    else:
        # ── Resolve idea ────────────────────────────────────────────────────
        ideas_json, code = resolve_idea_source(args.load_ideas, args.load_code)
        with open(ideas_json) as f:
            ideas = json.load(f)
        idea = ideas[args.idea_idx]
        idea_name = idea["Name"]
        print(f"Idea: {idea_name}  (idx={args.idea_idx})")

        # ── Dataset reference code ──────────────────────────────────────────
        dataset_ref_code = None
        if args.add_dataset_ref and osp.exists("hf_dataset_reference.py"):
            with open("hf_dataset_reference.py") as f:
                dataset_ref_code = f.read()

        added_code = "\n".join(filter(None, [dataset_ref_code, code])) or None
        if added_code:
            ideas[args.idea_idx]["Code"] = added_code

        # ── Build experiment directory ──────────────────────────────────────
        idea_dir = _build_experiment_dir(idea_name, args.attempt_id)
        stage_log_dir = osp.join(idea_dir, "logs")
        os.makedirs(stage_log_dir, exist_ok=True)
        print(f"Experiment dir: {idea_dir}")

        # ── Write idea files ────────────────────────────────────────────────
        idea_md = osp.join(idea_dir, "idea.md")
        idea_json_out = osp.join(idea_dir, "idea.json")
        idea_to_markdown(ideas[args.idea_idx], idea_md, None)
        with open(idea_json_out, "w") as f:
            json.dump(ideas[args.idea_idx], f, indent=4)

        # ── BFTS config ─────────────────────────────────────────────────────
        idea_config_path = edit_bfts_config_file("bfts_config.yaml", idea_dir, idea_json_out)
        state = _load_state(idea_dir)

    # ══ STAGE 1 ─ Experiments ═══════════════════════════════════════════
    if state.get("stage1") == "completed":
        print("\n[STAGE 1] Skipping Experiments (already completed).")
    else:
        _print_stage(1, "Running experiments")
        experiment_ok = run_experiment_stage(idea_config_path, idea_dir, stage_log_dir)

        if not experiment_ok:
            save_token_tracker(idea_dir)
            print("\n[ABORT] Experiment produced no results. Exiting early.")
            cleanup_child_processes()
            sys.exit(1)
            
        state["stage1"] = "completed"
        _save_state(idea_dir, state)

    # ══ STAGE 2 ─ Plots ════════════════════════════════════════════════
    if state.get("stage2") == "completed":
        print("\n[STAGE 2] Skipping Plots (already completed).")
    else:
        _print_stage(2, "Aggregating plots")
        plot_ok = run_plot_stage(idea_dir, args.model_agg_plots, stage_log_dir)
        save_token_tracker(idea_dir)
        if plot_ok:
            state["stage2"] = "completed"
            _save_state(idea_dir, state)

    # ══ STAGE 3 ─ Writeup ══════════════════════════════════════════════
    writeup_ok = state.get("stage3") == "completed"
    if writeup_ok:
        print("\n[STAGE 3] Skipping Writeup (already completed).")
    elif args.skip_writeup:
        print("\n[STAGE 3] Skipped (--skip_writeup).")
    else:
        _print_stage(3, "Writing paper")
        writeup_ok = run_writeup_stage(idea_dir, args, stage_log_dir)
        save_token_tracker(idea_dir)
        if writeup_ok:
            state["stage3"] = "completed"
            _save_state(idea_dir, state)

    # ══ STAGE 4 ─ Review ═══════════════════════════════════════════════
    if state.get("stage4") == "completed":
        print("\n[STAGE 4] Skipping Review (already completed).")
    elif args.skip_review or not writeup_ok:
        print("\n[STAGE 4] Skipped (either --skip_review or writeup failed/skipped).")
    else:
        _print_stage(4, "Reviewing paper")
        review_ok = run_review_stage(idea_dir, args.model_review, stage_log_dir)
        save_token_tracker(idea_dir)
        if review_ok:
            state["stage4"] = "completed"
            _save_state(idea_dir, state)

    # ── Done ────────────────────────────────────────────────────────────
    print(f"\n[DONE] All outputs saved in: {idea_dir}")
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    cleanup_child_processes()
    sys.exit(0)
