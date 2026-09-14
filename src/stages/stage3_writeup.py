"""
src/stages/stage3_writeup.py
============================
Stage 3 — Citation gathering and paper writeup.

Public API
----------
run_writeup_stage(idea_dir, args, stage_log_dir) -> bool
"""

import argparse
import os.path as osp
import traceback

from src.io_utils import redirect_stdout_stderr_to_file
from src.logger import create_log_filename, get_default_log_dir

try:
    from ai_scientist.perform_icbinb_writeup import (
        gather_citations,
        perform_writeup as perform_icbinb_writeup,
    )
    from ai_scientist.perform_writeup import perform_writeup
except ImportError:
    gather_citations = None
    perform_icbinb_writeup = None
    perform_writeup = None


def run_writeup_stage(
    idea_dir: str,
    args: argparse.Namespace,
    stage_log_dir: str,
) -> bool:
    """
    Gather citations and compile the paper.

    Parameters
    ----------
    idea_dir : str
        Root output directory for this experiment run.
    args : argparse.Namespace
        Parsed CLI arguments (uses writeup_type, num_cite_rounds,
        model_citation, model_writeup, model_writeup_small, writeup_retries).
    stage_log_dir : str
        Directory where stage log files are written.

    Returns
    -------
    bool
        ``True`` if at least one writeup attempt succeeded.
    """
    log_path = osp.join(stage_log_dir, "03_paper_writeup.log")
    central_log_path = osp.join(get_default_log_dir(), create_log_filename("stage3_writeup.py"))
    print(f"[Stage 3] Log → {log_path}")
    print(f"[Stage 3] Central Log → {central_log_path}")

    success = False
    try:
        with redirect_stdout_stderr_to_file(log_path, central_log_path):
            citations_text = gather_citations(
                idea_dir,
                num_cite_rounds=args.num_cite_rounds,
                small_model=args.model_citation,
            )
            for attempt in range(1, args.writeup_retries + 1):
                print(f"  Writeup attempt {attempt}/{args.writeup_retries} …")
                if args.writeup_type == "normal":
                    ok = perform_writeup(
                        base_folder=idea_dir,
                        small_model=args.model_writeup_small,
                        big_model=args.model_writeup,
                        page_limit=8,
                        citations_text=citations_text,
                    )
                else:
                    ok = perform_icbinb_writeup(
                        base_folder=idea_dir,
                        small_model=args.model_writeup_small,
                        big_model=args.model_writeup,
                        page_limit=4,
                        citations_text=citations_text,
                    )
                if ok:
                    success = True
                    break

        if success:
            print("[Stage 3] Completed successfully.")
        else:
            print(
                f"[Stage 3] Writeup did not succeed after "
                f"{args.writeup_retries} attempt(s)."
            )
    except Exception:
        with open(log_path, "a") as fh:
            fh.write("\n\n=== EXCEPTION IN STAGE 3 ===\n")
            traceback.print_exc(file=fh)
        with open(central_log_path, "a") as fh:
            fh.write("\n\n=== EXCEPTION IN STAGE 3 ===\n")
            traceback.print_exc(file=fh)
        print(f"[Stage 3] ERROR: exception raised — see {log_path}")

    return success
