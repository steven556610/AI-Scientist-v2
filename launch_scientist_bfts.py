import os.path as osp
import json
import argparse
import shutil
import torch
import os
import re
import sys
import logging
from datetime import datetime
from ai_scientist.llm import create_client

from contextlib import contextmanager
from ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager import (
    perform_experiments_bfts,
)
from ai_scientist.treesearch.bfts_utils import (
    idea_to_markdown,
    edit_bfts_config_file,
)
from ai_scientist.perform_plotting import aggregate_plots
from ai_scientist.perform_writeup import perform_writeup
from ai_scientist.perform_icbinb_writeup import (
    perform_writeup as perform_icbinb_writeup,
    gather_citations,
)
from ai_scientist.perform_llm_review import perform_review, load_paper
from ai_scientist.perform_vlm_review import perform_imgs_cap_ref_review
from ai_scientist.utils.token_tracker import token_tracker


def print_time():
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def save_token_tracker(idea_dir):
    with open(osp.join(idea_dir, "token_tracker.json"), "w") as f:
        json.dump(token_tracker.get_summary(), f)
    with open(osp.join(idea_dir, "token_tracker_interactions.json"), "w") as f:
        json.dump(token_tracker.get_interactions(), f)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run AI scientist experiments")
    parser.add_argument(
        "--writeup-type",
        type=str,
        default="icbinb",
        choices=["normal", "icbinb"],
        help="Type of writeup to generate (normal=8 page, icbinb=4 page)",
    )
    parser.add_argument(
        "--load_ideas",
        type=str,
        default="ideas/i_cant_believe_its_not_better.json",
        help=(
            "Path to a JSON file containing pregenerated ideas. "
            "Can also point to a folder containing <name>.json and optionally a "
            "'code/' sub-folder with multiple .py files used as the experiment template."
        ),
    )
    parser.add_argument(
        "--load_code",
        action="store_true",
        help=(
            "If set, load experiment code from a 'code/' sub-folder inside the idea "
            "folder (preferred) or a .py file with the same name as the ideas JSON."
        ),
    )
    parser.add_argument(
        "--idea_idx",
        type=int,
        default=0,
        help="Index of the idea to run",
    )
    parser.add_argument(
        "--add_dataset_ref",
        action="store_true",
        help="If set, add a HF dataset reference to the idea",
    )
    parser.add_argument(
        "--writeup-retries",
        type=int,
        default=3,
        help="Number of writeup attempts to try",
    )
    parser.add_argument(
        "--attempt_id",
        type=int,
        default=0,
        help="Attempt ID, used to distinguish same idea in different attempts in parallel runs",
    )
    parser.add_argument(
        "--model_agg_plots",
        type=str,
        default="o3-mini-2025-01-31",
        help="Model to use for plot aggregation",
    )
    parser.add_argument(
        "--model_writeup",
        type=str,
        default="o1-preview-2024-09-12",
        help="Model to use for writeup",
    )
    parser.add_argument(
        "--model_citation",
        type=str,
        default="gpt-4o-2024-11-20",
        help="Model to use for citation gathering",
    )
    parser.add_argument(
        "--num_cite_rounds",
        type=int,
        default=20,
        help="Number of citation rounds to perform",
    )
    parser.add_argument(
        "--model_writeup_small",
        type=str,
        default="gpt-4o-2024-05-13",
        help="Smaller model to use for writeup",
    )
    parser.add_argument(
        "--model_review",
        type=str,
        default="gpt-4o-2024-11-20",
        help="Model to use for review main text and captions",
    )
    parser.add_argument(
        "--skip_writeup",
        action="store_true",
        help="If set, skip the writeup process",
    )
    parser.add_argument(
        "--skip_review",
        action="store_true",
        help="If set, skip the review process",
    )
    return parser.parse_args()


def get_available_gpus(gpu_ids=None):
    if gpu_ids is not None:
        return [int(gpu_id) for gpu_id in gpu_ids.split(",")]
    return list(range(torch.cuda.device_count()))


def find_pdf_path_for_review(idea_dir):
    pdf_files = [f for f in os.listdir(idea_dir) if f.endswith(".pdf")]
    reflection_pdfs = [f for f in pdf_files if "reflection" in f]
    if not reflection_pdfs:
        return None
    final_pdfs = [f for f in reflection_pdfs if "final" in f.lower()]
    if final_pdfs:
        return osp.join(idea_dir, final_pdfs[0])
    reflection_nums = []
    for f in reflection_pdfs:
        match = re.search(r"reflection[_.]?(\d+)", f)
        if match:
            reflection_nums.append((int(match.group(1)), f))
    if reflection_nums:
        highest_reflection = max(reflection_nums, key=lambda x: x[0])
        return osp.join(idea_dir, highest_reflection[1])
    return osp.join(idea_dir, reflection_pdfs[0])


@contextmanager
def redirect_stdout_stderr_to_file(log_file_path):
    """Redirect stdout and stderr to both the log file AND the original streams."""
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    os.makedirs(osp.dirname(log_file_path), exist_ok=True)
    log = open(log_file_path, "a", encoding="utf-8", buffering=1)

    class Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                try:
                    s.write(data)
                except Exception:
                    pass

        def flush(self):
            for s in self.streams:
                try:
                    s.flush()
                except Exception:
                    pass

        def fileno(self):
            return self.streams[0].fileno()

    sys.stdout = Tee(original_stdout, log)
    sys.stderr = Tee(original_stderr, log)
    try:
        yield log
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log.close()


def _load_idea_code_folder(code_folder: str) -> str:
    """
    Concatenate all .py files in *code_folder* into a single string,
    labelled by relative filename.  Used when the experiment template is a
    multi-file directory rather than a single .py file.
    """
    parts = []
    for root, dirs, files in os.walk(code_folder):
        dirs[:] = sorted([d for d in dirs if not d.startswith(".")])
        for fname in sorted(files):
            if not fname.endswith(".py"):
                continue
            fpath = osp.join(root, fname)
            rel = osp.relpath(fpath, code_folder)
            with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
                content = fh.read()
            parts.append(f"# ===== {rel} =====\n{content}")
    return "\n\n".join(parts)


def _resolve_idea_source(load_ideas_arg: str, load_code: bool):
    """
    Determine (ideas_json_path, code) from the --load_ideas argument.

    Returns:
        ideas_json_path: str
        code: str | None
    """
    arg = load_ideas_arg

    if arg.endswith(".json") and osp.isfile(arg):
        ideas_json = arg
        idea_folder = osp.dirname(osp.abspath(arg))
    elif osp.isdir(arg):
        idea_folder = osp.abspath(arg)
        json_candidates = [
            f for f in os.listdir(idea_folder) if f.endswith(".json")
        ]
        if not json_candidates:
            raise FileNotFoundError(
                f"No .json file found in idea folder: {idea_folder}"
            )
        ideas_json = osp.join(idea_folder, sorted(json_candidates)[0])
    else:
        raise FileNotFoundError(
            f"--load_ideas must be a .json file or a folder; got: {arg}"
        )

    code = None
    if load_code:
        code_folder = osp.join(idea_folder, "code")
        if osp.isdir(code_folder):
            code = _load_idea_code_folder(code_folder)
            print(f"Loaded multi-file code template from folder: {code_folder}")
        else:
            code_py = osp.splitext(ideas_json)[0] + ".py"
            if osp.isfile(code_py):
                with open(code_py, "r") as f:
                    code = f.read()
                print(f"Loaded single-file code template: {code_py}")
            else:
                print(
                    f"Warning: --load_code set but neither {code_folder} nor "
                    f"{code_py} found.  Proceeding without code template."
                )

    return ideas_json, code


def _setup_stage_log_dir(experiment_dir: str) -> str:
    """Return the per-experiment logs/ directory, creating it if needed."""
    log_dir = osp.join(experiment_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


if __name__ == "__main__":
    args = parse_arguments()
    os.environ["AI_SCIENTIST_ROOT"] = os.path.dirname(os.path.abspath(__file__))
    print(f"Set AI_SCIENTIST_ROOT to {os.environ['AI_SCIENTIST_ROOT']}")

    # ── GPU inventory ──────────────────────────────────────────────────────
    available_gpus = get_available_gpus()
    print(f"Using GPUs: {available_gpus}")

    # ── Resolve idea source (JSON + optional code) ─────────────────────────
    ideas_json_path, code = _resolve_idea_source(args.load_ideas, args.load_code)
    with open(ideas_json_path, "r") as f:
        ideas = json.load(f)
    print(f"Loaded {len(ideas)} idea(s) from {ideas_json_path}")

    idea = ideas[args.idea_idx]
    idea_name = idea["Name"]

    # ── Build nested experiment directory ─────────────────────────────────
    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    idea_dir = osp.join(
        "experiments", idea_name, f"{date}_attempt_{args.attempt_id}"
    )
    print(f"Results will be saved in {idea_dir}")
    os.makedirs(idea_dir, exist_ok=True)

    # ── Stage log directory ────────────────────────────────────────────────
    stage_log_dir = _setup_stage_log_dir(idea_dir)

    # ── Write idea.md ──────────────────────────────────────────────────────
    idea_path_md = osp.join(idea_dir, "idea.md")
    idea_to_markdown(ideas[args.idea_idx], idea_path_md, None)

    # ── Dataset reference ──────────────────────────────────────────────────
    dataset_ref_code = None
    if args.add_dataset_ref:
        dataset_ref_path = "hf_dataset_reference.py"
        if osp.exists(dataset_ref_path):
            with open(dataset_ref_path, "r") as f:
                dataset_ref_code = f.read()
        else:
            print(f"Warning: Dataset reference file {dataset_ref_path} not found")

    # ── Assemble injected code ─────────────────────────────────────────────
    if dataset_ref_code and code:
        added_code = dataset_ref_code + "\n" + code
    elif dataset_ref_code:
        added_code = dataset_ref_code
    elif code:
        added_code = code
    else:
        added_code = None

    if added_code:
        ideas[args.idea_idx]["Code"] = added_code

    # ── Store idea JSON ────────────────────────────────────────────────────
    idea_path_json = osp.join(idea_dir, "idea.json")
    with open(idea_path_json, "w") as f:
        json.dump(ideas[args.idea_idx], f, indent=4)

    # ── Edit BFTS config ──────────────────────────────────────────────────
    config_path = "bfts_config.yaml"
    idea_config_path = edit_bfts_config_file(
        config_path,
        idea_dir,
        idea_path_json,
    )

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 1 — Experiment execution
    # ══════════════════════════════════════════════════════════════════════
    stage1_log = osp.join(stage_log_dir, "01_experiment_execution.log")
    print(f"\n{'='*60}")
    print(f"[STAGE 1] Running experiments  →  {stage1_log}")
    print(f"{'='*60}")
    print_time()

    experiment_ok = False
    try:
        with redirect_stdout_stderr_to_file(stage1_log):
            perform_experiments_bfts(idea_config_path)

        experiment_results_dir = osp.join(idea_dir, "logs/0-run/experiment_results")
        if osp.exists(experiment_results_dir) and os.listdir(experiment_results_dir):
            shutil.copytree(
                experiment_results_dir,
                osp.join(idea_dir, "experiment_results"),
                dirs_exist_ok=True,
            )
            experiment_ok = True
            print("[STAGE 1] Experiment completed successfully.")
        else:
            print(
                "[STAGE 1] WARNING: experiment_results directory is empty or missing. "
                "Skipping downstream stages."
            )
    except Exception:
        with open(stage1_log, "a") as err_log:
            import traceback
            err_log.write("\n\n=== EXCEPTION IN STAGE 1 ===\n")
            traceback.print_exc(file=err_log)
        print(
            f"[STAGE 1] ERROR: experiment execution raised an exception. "
            f"See {stage1_log} for details."
        )

    if not experiment_ok:
        save_token_tracker(idea_dir)
        print("\n[ABORT] Experiment did not produce results. Exiting early.")
        sys.exit(1)

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 2 — Plot aggregation
    # ══════════════════════════════════════════════════════════════════════
    stage2_log = osp.join(stage_log_dir, "02_plot_generation.log")
    print(f"\n{'='*60}")
    print(f"[STAGE 2] Aggregating plots  →  {stage2_log}")
    print(f"{'='*60}")
    print_time()

    try:
        with redirect_stdout_stderr_to_file(stage2_log):
            aggregate_plots(base_folder=idea_dir, model=args.model_agg_plots)
        exp_res = osp.join(idea_dir, "experiment_results")
        if osp.exists(exp_res):
            shutil.rmtree(exp_res)
        print("[STAGE 2] Plot aggregation completed.")
    except Exception:
        with open(stage2_log, "a") as err_log:
            import traceback
            err_log.write("\n\n=== EXCEPTION IN STAGE 2 ===\n")
            traceback.print_exc(file=err_log)
        print(f"[STAGE 2] ERROR: see {stage2_log}")

    save_token_tracker(idea_dir)

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 3 — Paper writeup (citations + LaTeX)
    # ══════════════════════════════════════════════════════════════════════
    if not args.skip_writeup:
        stage3_log = osp.join(stage_log_dir, "03_paper_writeup.log")
        print(f"\n{'='*60}")
        print(f"[STAGE 3] Writing paper  →  {stage3_log}")
        print(f"{'='*60}")
        print_time()

        writeup_success = False
        try:
            with redirect_stdout_stderr_to_file(stage3_log):
                citations_text = gather_citations(
                    idea_dir,
                    num_cite_rounds=args.num_cite_rounds,
                    small_model=args.model_citation,
                )
                for attempt in range(args.writeup_retries):
                    print(f"Writeup attempt {attempt+1} of {args.writeup_retries}")
                    if args.writeup_type == "normal":
                        writeup_success = perform_writeup(
                            base_folder=idea_dir,
                            small_model=args.model_writeup_small,
                            big_model=args.model_writeup,
                            page_limit=8,
                            citations_text=citations_text,
                        )
                    else:
                        writeup_success = perform_icbinb_writeup(
                            base_folder=idea_dir,
                            small_model=args.model_writeup_small,
                            big_model=args.model_writeup,
                            page_limit=4,
                            citations_text=citations_text,
                        )
                    if writeup_success:
                        break
            if not writeup_success:
                print("[STAGE 3] Writeup did not complete successfully after all retries.")
            else:
                print("[STAGE 3] Paper writeup completed.")
        except Exception:
            with open(stage3_log, "a") as err_log:
                import traceback
                err_log.write("\n\n=== EXCEPTION IN STAGE 3 ===\n")
                traceback.print_exc(file=err_log)
            print(f"[STAGE 3] ERROR: see {stage3_log}")
            writeup_success = False
    else:
        writeup_success = False
        print("[STAGE 3] Skipped (--skip_writeup).")

    save_token_tracker(idea_dir)

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 4 — Paper review
    # ══════════════════════════════════════════════════════════════════════
    if not args.skip_review and not args.skip_writeup and writeup_success:
        stage4_log = osp.join(stage_log_dir, "04_paper_review.log")
        print(f"\n{'='*60}")
        print(f"[STAGE 4] Reviewing paper  →  {stage4_log}")
        print(f"{'='*60}")
        print_time()

        try:
            with redirect_stdout_stderr_to_file(stage4_log):
                pdf_path = find_pdf_path_for_review(idea_dir)
                if pdf_path and osp.exists(pdf_path):
                    print("Paper found at: ", pdf_path)
                    paper_content = load_paper(pdf_path)
                    client, client_model = create_client(args.model_review)
                    review_text = perform_review(paper_content, client_model, client)
                    review_img_cap_ref = perform_imgs_cap_ref_review(
                        client, client_model, pdf_path
                    )
                    with open(osp.join(idea_dir, "review_text.txt"), "w") as f:
                        f.write(json.dumps(review_text, indent=4))
                    with open(osp.join(idea_dir, "review_img_cap_ref.json"), "w") as f:
                        json.dump(review_img_cap_ref, f, indent=4)
                    print("Paper review completed.")
                else:
                    print("[STAGE 4] No PDF found to review.")
        except Exception:
            with open(stage4_log, "a") as err_log:
                import traceback
                err_log.write("\n\n=== EXCEPTION IN STAGE 4 ===\n")
                traceback.print_exc(file=err_log)
            print(f"[STAGE 4] ERROR: see {stage4_log}")
    else:
        print("[STAGE 4] Skipped.")

    # ── Final token summary ────────────────────────────────────────────────
    save_token_tracker(idea_dir)
    print(f"\n[DONE] All outputs saved in: {idea_dir}")
    print_time()

    # ── Cleanup child processes ────────────────────────────────────────────
    print("Start cleaning up processes")
    import psutil
    import signal

    current_process = psutil.Process()
    children = current_process.children(recursive=True)
    for child in children:
        try:
            child.send_signal(signal.SIGTERM)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    gone, alive = psutil.wait_procs(children, timeout=3)
    for process in alive:
        try:
            process.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    sys.exit(0)
