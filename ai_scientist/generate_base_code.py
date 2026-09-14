"""
generate_base_code.py
=====================
Given an idea JSON file (or a folder containing one), this script uses an LLM
to generate a full baseline experiment folder that AI-Scientist can subsequently
refine.

The generator produces a small but complete experiment scaffold:
  <idea_dir>/
    runfile.py          ← entry-point executed by AI-Scientist
    model.py            ← model / algorithm definitions
    data_utils.py       ← synthetic or cached data loading helpers
    metrics.py          ← evaluation functions
    configs/
      default.yaml      ← hyper-parameters
    README_generated.md ← human-readable description of the scaffold

Usage
-----
# Generate for a single idea folder
python ai_scientist/generate_base_code.py \\
    --idea_dir ai_scientist/ideas/my_research_topic \\
    --model kimi-k3

# Alternatively point at the JSON directly
python ai_scientist/generate_base_code.py \\
    --idea_json ai_scientist/ideas/my_research_topic/my_research_topic.json \\
    --model kimi-k3

# Use an ML project template as inspiration
python ai_scientist/generate_base_code.py \\
    --idea_dir ai_scientist/ideas/my_research_topic \\
    --model kimi-k3 \\
    --reference_dir ai_scientist/references/ml_projects_reference
"""

import argparse
import json
import os
import os.path as osp
import re
import sys
import textwrap
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, osp.join(osp.dirname(__file__), ".."))

from ai_scientist.llm import create_client, get_response_from_llm

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert ML/AI research engineer.
Your task is to write a self-contained, runnable Python experiment scaffold for a
given research idea.

CONSTRAINTS
-----------
1. The code must use only standard Python + common libraries (numpy, torch,
   scikit-learn, pandas, matplotlib, seaborn, yaml, json, pathlib, argparse).
2. Avoid real dataset downloads.  Use lightweight synthetic data generators or
   HuggingFace `datasets` cached calls so the experiment finishes in <5 minutes
   on a single GPU/CPU.
3. Outputs must be saved as JSON files in a `results/` sub-directory so
   AI-Scientist can parse them.
4. The entry-point is `runfile.py`.  It must accept `--out_dir` and
   `--config_path` as CLI arguments.
5. Each file must be complete and directly writable to disk.

OUTPUT FORMAT
-------------
Respond ONLY with a sequence of file blocks.  Each block looks like:

--- FILE: <relative/path/to/file> ---
<complete file contents>
--- END FILE ---

Do NOT include any other text outside the file blocks.
""").strip()


def _build_user_prompt(idea: Dict, reference_snippet: str = "") -> str:
    idea_text = json.dumps(idea, indent=2, ensure_ascii=False)
    ref_block = ""
    if reference_snippet:
        ref_block = f"\n\nREFERENCE CODE PATTERNS (for structural inspiration only):\n{reference_snippet}\n"
    return (
        f"Generate an experiment scaffold for the following research idea:\n\n"
        f"```json\n{idea_text}\n```"
        f"{ref_block}\n\n"
        "Follow the output format described in the system prompt exactly."
    )


# ---------------------------------------------------------------------------
# Reference snippet extraction
# ---------------------------------------------------------------------------

def _collect_reference_snippet(reference_dir: str, max_chars: int = 4000) -> str:
    """
    Walk the reference directory and collect representative code excerpts.
    Prioritise *.py files, grab the first N characters in total.
    """
    if not reference_dir or not osp.isdir(reference_dir):
        return ""
    snippets = []
    total = 0
    for root, dirs, files in os.walk(reference_dir):
        # Skip hidden / git internals
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fname in files:
            if not fname.endswith(".py"):
                continue
            fpath = osp.join(root, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
                    content = fh.read()
                rel = osp.relpath(fpath, reference_dir)
                excerpt = content[:800]
                snippet = f"# --- {rel} ---\n{excerpt}\n"
                snippets.append(snippet)
                total += len(snippet)
                if total >= max_chars:
                    break
            except Exception:
                continue
        if total >= max_chars:
            break
    return "\n".join(snippets)[:max_chars]


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

FILE_BLOCK_RE = re.compile(
    r"---\s*FILE:\s*(?P<path>[^\n]+?)\s*---\n(?P<content>.*?)---\s*END FILE\s*---",
    re.DOTALL,
)


def _parse_file_blocks(response: str) -> Dict[str, str]:
    """Return {relative_path: content} from the LLM response."""
    files = {}
    for m in FILE_BLOCK_RE.finditer(response):
        rel_path = m.group("path").strip()
        content = m.group("content")
        files[rel_path] = content
    return files


# ---------------------------------------------------------------------------
# Main generation logic
# ---------------------------------------------------------------------------

def generate_base_code(
    idea: Dict,
    output_dir: str,
    client: Any,
    model: str,
    reference_dir: Optional[str] = None,
    overwrite: bool = False,
) -> bool:
    """
    Generate experiment scaffold files inside *output_dir*.

    Returns True on success, False on failure.
    """
    os.makedirs(output_dir, exist_ok=True)

    ref_snippet = _collect_reference_snippet(reference_dir) if reference_dir else ""
    user_prompt = _build_user_prompt(idea, ref_snippet)

    print(f"[generate_base_code] Calling LLM ({model}) to generate scaffold …")
    try:
        response, _ = get_response_from_llm(
            prompt=user_prompt,
            client=client,
            model=model,
            system_message=SYSTEM_PROMPT,
            msg_history=[],
        )
    except Exception:
        print("[generate_base_code] LLM call failed:")
        traceback.print_exc()
        return False

    file_blocks = _parse_file_blocks(response)
    if not file_blocks:
        # Fallback: write the raw response so the user can inspect
        fallback_path = osp.join(output_dir, "llm_raw_response.txt")
        with open(fallback_path, "w") as f:
            f.write(response)
        print(
            f"[generate_base_code] WARNING: Could not parse file blocks. "
            f"Raw LLM output saved to {fallback_path}"
        )
        return False

    written = []
    for rel_path, content in file_blocks.items():
        # Sanitise path – strip leading slashes / drive letters
        rel_path = rel_path.lstrip("/\\").replace("..", "")
        full_path = osp.join(output_dir, rel_path)
        if osp.exists(full_path) and not overwrite:
            print(f"[generate_base_code] SKIP (already exists): {full_path}")
            continue
        os.makedirs(osp.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        written.append(rel_path)
        print(f"[generate_base_code]  ✓  {rel_path}")

    # Write a manifest so AI-Scientist knows which files to load
    manifest_path = osp.join(output_dir, "code_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({"generated_files": written, "entry_point": "runfile.py"}, f, indent=2)

    print(
        f"\n[generate_base_code] Done. {len(written)} file(s) written to {output_dir}"
    )
    return True


# ---------------------------------------------------------------------------
# load_code_folder helper (used by launch_scientist_bfts.py)
# ---------------------------------------------------------------------------

def load_code_folder(code_dir: str) -> str:
    """
    Concatenate all *.py files from *code_dir* into a single string, labelled
    by filename, suitable for injection into the idea JSON 'Code' field.
    """
    if not osp.isdir(code_dir):
        raise FileNotFoundError(f"Code folder not found: {code_dir}")

    parts = []
    for root, dirs, files in os.walk(code_dir):
        dirs[:] = sorted([d for d in dirs if not d.startswith(".")])
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
# CLI entry-point
# ---------------------------------------------------------------------------

def _resolve_idea_and_dir(args) -> tuple:
    """Return (idea_dict, idea_dir, idea_name)."""
    if args.idea_json:
        idea_json = args.idea_json
        idea_dir = osp.dirname(osp.abspath(idea_json))
    elif args.idea_dir:
        idea_dir = args.idea_dir
        candidates = list(Path(idea_dir).glob("*.json"))
        if not candidates:
            raise FileNotFoundError(f"No JSON file found in {idea_dir}")
        idea_json = str(candidates[0])
    else:
        raise ValueError("Provide either --idea_json or --idea_dir")

    with open(idea_json, "r") as f:
        payload = json.load(f)

    # Support both a list (AI-Scientist format) and a plain dict
    if isinstance(payload, list):
        idea = payload[args.idea_idx]
    else:
        idea = payload

    idea_name = idea.get("Name", osp.basename(idea_dir))
    return idea, osp.abspath(idea_dir), idea_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a baseline experiment code folder for an AI-Scientist idea"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--idea_dir",
        type=str,
        help="Path to the idea folder (containing <idea>.json)",
    )
    group.add_argument(
        "--idea_json",
        type=str,
        help="Direct path to the idea JSON file",
    )
    parser.add_argument(
        "--idea_idx",
        type=int,
        default=0,
        help="Index of the idea to use if the JSON is a list (default: 0)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="kimi-k3",
        help="LLM to use for code generation",
    )
    parser.add_argument(
        "--reference_dir",
        type=str,
        default="ai_scientist/references/ml_projects_reference",
        help="Path to a reference ML project directory for structural inspiration",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to write the generated files (default: inside idea_dir)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing generated files",
    )
    args = parser.parse_args()

    idea, idea_dir, idea_name = _resolve_idea_and_dir(args)
    output_dir = args.output_dir or osp.join(idea_dir, "code")

    client, client_model = create_client(args.model)

    success = generate_base_code(
        idea=idea,
        output_dir=output_dir,
        client=client,
        model=client_model,
        reference_dir=args.reference_dir,
        overwrite=args.overwrite,
    )
    sys.exit(0 if success else 1)
