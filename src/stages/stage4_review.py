"""
src/stages/stage4_review.py
============================
Stage 4 — LLM / VLM paper review.

Public API
----------
run_review_stage(idea_dir, model_review, stage_log_dir) -> None
"""

import json
import os.path as osp
import traceback

from src.io_utils import find_pdf_path_for_review, redirect_stdout_stderr_to_file
from src.logger import create_log_filename, get_default_log_dir

try:
    from ai_scientist.llm import create_client
    from ai_scientist.perform_llm_review import load_paper, perform_review
    from ai_scientist.perform_vlm_review import perform_imgs_cap_ref_review
except ImportError:
    create_client = None
    load_paper = None
    perform_review = None
    perform_imgs_cap_ref_review = None


def run_review_stage(
    idea_dir: str,
    model_review: str,
    stage_log_dir: str,
) -> bool:
    """
    Load the compiled PDF, run text + image/caption/reference review, and
    save the results to ``review_text.txt`` and ``review_img_cap_ref.json``.

    Parameters
    ----------
    idea_dir : str
        Root output directory for this experiment run.
    model_review : str
        LLM identifier for review calls.
    stage_log_dir : str
        Directory where stage log files are written.
    """
    log_path = osp.join(stage_log_dir, "04_paper_review.log")
    central_log_path = osp.join(get_default_log_dir(), create_log_filename("stage4_review.py"))
    print(f"[Stage 4] Log → {log_path}")
    print(f"[Stage 4] Central Log → {central_log_path}")

    pdf_path = find_pdf_path_for_review(idea_dir)
    if not pdf_path or not osp.exists(pdf_path):
        print("[Stage 4] No PDF found — skipping review.")
        return False

    try:
        with redirect_stdout_stderr_to_file(log_path, central_log_path):
            print(f"  Reviewing: {pdf_path}")
            paper_content = load_paper(pdf_path)
            client, client_model = create_client(model_review)
            review_text = perform_review(paper_content, client_model, client)
            review_img = perform_imgs_cap_ref_review(client, client_model, pdf_path)

            with open(osp.join(idea_dir, "review_text.txt"), "w") as fh:
                fh.write(json.dumps(review_text, indent=4))
            with open(osp.join(idea_dir, "review_img_cap_ref.json"), "w") as fh:
                json.dump(review_img, fh, indent=4)

        print("[Stage 4] Completed successfully.")
        return True
    except Exception:
        with open(log_path, "a") as fh:
            fh.write("\n\n=== EXCEPTION IN STAGE 4 ===\n")
            traceback.print_exc(file=fh)
        with open(central_log_path, "a") as fh:
            fh.write("\n\n=== EXCEPTION IN STAGE 4 ===\n")
            traceback.print_exc(file=fh)
        print(f"[Stage 4] ERROR: exception raised — see {log_path}")
        return False
