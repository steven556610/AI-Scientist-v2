"""
Systematic evaluation of MergeKit merge strategies for large-to-small (L2S)
training of ``google/gemma-4-E2B-it-qat-q4_0-unquantized`` on TMMLU+.

This is the seed/template script for the ``forth_trial`` idea family. It is
written to be re-parameterised by the tree search, not rewritten.

--------------------------------------------------------------------------
Compliance with the hard rules in forth_trial.md
--------------------------------------------------------------------------
1. Tooling      - merges are computed by the *actual* merge-method
                  implementations from ``arcee-ai/mergekit`` (imported from
                  ``mergekit.merge_methods.REGISTERED_MERGE_METHODS``), and a
                  canonical mergekit YAML is emitted for every combination so
                  each one can be replayed with ``mergekit-yaml``.
2. Data         - real ``ikala/tmmluplus`` only. Every row is downloaded and
                  persisted as JSONL under ``<experiment_root>/data/`` together
                  with a sha256 manifest. NO model-generated data is ever used
                  as a training target: the teacher contributes soft logits over
                  the four option tokens of *real* questions, and the hard label
                  always comes from the dataset. ``--no_teacher`` reduces this
                  to pure supervised fine-tuning on real labels.
3. Model        - student is gemma-4-E2B-it-qat-q4_0-unquantized (BASE_MODEL).
4. Architecture - large-to-small: the E4B teacher supervises the E2B student.
5. Merge modes  - both ``dense`` (one static merge over all experts) and ``moe``
                  (per-question top-2 routing, gate vectors derived from real
                  TMMLU+ prompts using mergekit's hidden-state gate recipe).
6. No regression- every candidate is scored on MMLU, GSM8K and IFEval in
                  addition to TMMLU+, and compared against the untouched base.
7. Scale        - 4 methods x 2 modes x 2 weightings x 3 seeds = 48 combos.
                  Every combo records its full config + RNG seed.

--------------------------------------------------------------------------
Why merging happens in LoRA-delta space
--------------------------------------------------------------------------
All experts are LoRA adapters over one shared frozen base, so an expert model is
exactly ``base + delta``. Running a 48-way sweep by materialising 48 full ~10 GB
gemma-4 checkpoints would cost several TB of I/O and dominate the runtime while
changing nothing about the arithmetic. We therefore hand the per-module deltas
to mergekit's own merge tasks (identical code path, identical math, with the
base pinned as ``base_model``) and apply the merged delta to the live model
through a context manager. ``--cli_verify N`` additionally replays N randomly
chosen combinations through the real ``mergekit-yaml`` CLI on materialised
checkpoints and asserts the merged tensors agree, so the fast path stays honest.

Anti-fabrication: this script refuses to run on a placeholder model, refuses to
report a metric that was not produced by a real forward pass, and asserts that
predictions/ground-truth are non-empty before writing results. See ``_guard_*``.
"""

import argparse
import contextlib
import copy
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
#  Constants
# --------------------------------------------------------------------------- #

BASE_MODEL = os.environ.get(
    "BASE_MODEL", "google/gemma-4-E2B-it-qat-q4_0-unquantized"
)
TEACHER_MODEL = os.environ.get(
    "TEACHER_MODEL", "google/gemma-4-E4B-it-qat-q4_0-unquantized"
)

# Anything in here means somebody swapped the real model for a stand-in.
FORBIDDEN_MODEL_IDS = {
    "synthetic", "mock", "mock_model", "dummy", "fake", "placeholder",
    "random", "none", "test", "stub",
}

# Expert domains. Each is a bundle of real TMMLU+ subsets; the grouping is what
# makes merging non-trivial (four specialists that must coexist in one model).
DOMAINS = {
    "chinese": [
        "junior_chinese_exam",
        "chinese_language_and_literature",
        "tve_chinese_language",
    ],
    "math": [
        "junior_math_exam",
        "tve_mathematics",
    ],
    "social": [
        "geography_of_taiwan",
        "junior_social_studies",
        "jce_humanities",
    ],
    "science": [
        "junior_science_exam",
        "junior_chemistry",
        "secondary_physics",
    ],
}

OPTIONS = ["A", "B", "C", "D"]

# 4 x 2 x 2 x 3 = 48 combinations (hard rule 7).
MERGE_METHODS = ["linear", "task_arithmetic", "ties", "dare_ties"]
MERGE_MODES = ["dense", "moe"]
WEIGHTINGS = ["uniform", "perf_weighted"]
SEEDS = [0, 1, 2]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32


# --------------------------------------------------------------------------- #
#  Paths: data MUST live in <experiment_root>/data  (hard rule 2)
# --------------------------------------------------------------------------- #

def resolve_experiment_root() -> Path:
    """Locate the experiment workspace root.

    The tree search runs each node from ``<...>/<node>/working``, so the
    workspace root is the nearest ancestor that owns that ``working`` dir.
    """
    cwd = Path.cwd().resolve()
    if cwd.name == "working":
        return cwd.parent
    for parent in cwd.parents:
        if parent.parent.name == "experiments":
            return parent
    return cwd


def resolve_data_dir(experiment_root: Path) -> Path:
    """``<experiment_root>/data``, backed by a shared store.

    Hard rule 2 says the data must live in ``data/`` at the root of the
    experiment workspace. A 48-combination sweep spawns many node workspaces
    though, and re-downloading TMMLU+ into each one would be wasteful and would
    make the runs less reproducible (different snapshots). So the per-workspace
    ``data/`` is a symlink onto one canonical store, which keeps the required
    path valid while the bytes are downloaded exactly once.
    """
    local = experiment_root / "data"
    shared = Path(
        os.environ.get(
            "FORTH_DATA_ROOT",
            Path(__file__).resolve().parents[3] / "experiments" / "_forth_trial_data",
        )
    ).resolve()
    shared.mkdir(parents=True, exist_ok=True)

    if local.is_symlink():
        if local.resolve() != shared:
            local.unlink()
    elif local.exists() and not local.is_dir():
        local.unlink()

    if not local.exists():
        try:
            local.symlink_to(shared, target_is_directory=True)
        except (OSError, FileExistsError):
            local.mkdir(parents=True, exist_ok=True)
            return local
    return local


# --------------------------------------------------------------------------- #
#  Guards against fabricated runs
# --------------------------------------------------------------------------- #

def _guard_model_id(model_id: str, role: str) -> None:
    if not model_id or not isinstance(model_id, str):
        raise RuntimeError(f"[FATAL] {role} model id is empty.")
    low = model_id.strip().lower()
    if low in FORBIDDEN_MODEL_IDS or any(
        tok in low.replace("-", "_").split("_") for tok in FORBIDDEN_MODEL_IDS
    ):
        raise RuntimeError(
            f"[FATAL] {role} model id {model_id!r} looks like a placeholder. "
            "This experiment is only valid against the real checkpoint."
        )


def _guard_real_rows(rows, name: str, minimum: int = 1) -> None:
    if not rows or len(rows) < minimum:
        raise RuntimeError(
            f"[FATAL] {name} has {len(rows) if rows else 0} rows (need >= {minimum}). "
            "Refusing to continue -- no substitute data will be generated."
        )


def _guard_results(experiment_data: dict) -> None:
    """Nothing gets written unless it came from a real forward pass."""
    meta = experiment_data.get("_meta", {})
    if meta.get("base_model") != BASE_MODEL:
        raise RuntimeError(
            f"[FATAL] _meta.base_model={meta.get('base_model')!r} != {BASE_MODEL!r}."
        )
    if not meta.get("data_manifest"):
        raise RuntimeError("[FATAL] no data manifest -- data provenance unverifiable.")
    total_preds = 0
    for dom in DOMAINS:
        if dom not in experiment_data:
            continue
        preds = experiment_data[dom].get("predictions", [])
        gts = experiment_data[dom].get("ground_truth", [])
        total_preds += len(preds)
        if len(preds) != len(gts):
            raise RuntimeError(
                f"[FATAL] {dom}: {len(preds)} predictions vs {len(gts)} labels."
            )
    if total_preds == 0:
        raise RuntimeError(
            "[FATAL] zero predictions recorded -- the evaluation never ran."
        )


# --------------------------------------------------------------------------- #
#  Data: download real TMMLU+ into data/ and keep it
# --------------------------------------------------------------------------- #

def _load_dataset_retry(*a, _tries: int = 6, **kw):
    """``load_dataset`` with bounded retries.

    The HF CDN intermittently returns ConnectTimeout under load. Every tree
    search node re-enters this path, so a transient blip must not kill a run.
    """
    from datasets import load_dataset

    last = None
    for attempt in range(1, _tries + 1):
        try:
            return load_dataset(*a, **kw)
        except Exception as e:
            last = e
            wait = min(60, 5 * attempt)
            print(f"[data] retry {attempt}/{_tries} for {a} after "
                  f"{type(e).__name__}: {str(e)[:100]} (sleep {wait}s)", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"[FATAL] could not download {a}: {last}")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalise_tmmlu_row(row: dict) -> dict | None:
    """TMMLU+ rows carry question/A/B/C/D/answer. Reject anything malformed."""
    q = (row.get("question") or "").strip()
    ans = (row.get("answer") or "").strip().upper()
    opts = {k: (row.get(k) or "").strip() for k in OPTIONS}
    if not q or ans not in OPTIONS or any(not v for v in opts.values()):
        return None
    return {"question": q, **opts, "answer": ans}


def download_tmmluplus(data_dir: Path, force: bool = False) -> dict:
    """Persist every required TMMLU+ subset as JSONL + a sha256 manifest."""
    from datasets import load_dataset

    root = data_dir / "tmmluplus"
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"

    if manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text())
        if all((root / e["file"]).exists() for e in manifest["subsets"]):
            print(f"[data] reusing verified TMMLU+ store at {root}")
            return manifest

    subsets = sorted({s for subs in DOMAINS.values() for s in subs})
    entries = []
    print(f"[data] downloading {len(subsets)} real TMMLU+ subsets -> {root}")
    for sub in subsets:
        out = root / f"{sub}.jsonl"
        ds = _load_dataset_retry("ikala/tmmluplus", sub, split="test")
        rows = [r for r in (_normalise_tmmlu_row(x) for x in ds) if r]
        _guard_real_rows(rows, f"TMMLU+/{sub}", minimum=10)
        with open(out, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        entries.append(
            {"subset": sub, "file": out.name, "rows": len(rows),
             "sha256": _sha256_file(out)}
        )
        print(f"[data]   {sub:34s} {len(rows):5d} rows")

    manifest = {
        "dataset": "ikala/tmmluplus",
        "source": "https://huggingface.co/datasets/ikala/tmmluplus",
        "synthetic": False,
        "total_rows": sum(e["rows"] for e in entries),
        "subsets": entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"[data] manifest -> {manifest_path} ({manifest['total_rows']} real rows)")
    return manifest


def download_general_benchmarks(data_dir: Path, n_per_bench: int, force: bool = False) -> dict:
    """MMLU / GSM8K / IFEval, also persisted under data/ (hard rule 2)."""
    from datasets import load_dataset

    root = data_dir / "general"
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and not force:
        man = json.loads(manifest_path.read_text())
        if man.get("n_per_bench", 0) >= n_per_bench and all(
            (root / e["file"]).exists() for e in man["benchmarks"]
        ):
            print(f"[data] reusing general-benchmark store at {root}")
            return man

    entries = []

    def _dump(name, rows):
        out = root / f"{name}.jsonl"
        _guard_real_rows(rows, name, minimum=1)
        with open(out, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        entries.append({"benchmark": name, "file": out.name, "rows": len(rows),
                        "sha256": _sha256_file(out)})
        print(f"[data]   {name:10s} {len(rows):5d} rows")

    # MMLU - 4-way MCQ, scored the same way as TMMLU+.
    ds = _load_dataset_retry("cais/mmlu", "all", split=f"test[:{n_per_bench * 4}]")
    mmlu = []
    for r in ds:
        ch = r.get("choices") or []
        if len(ch) != 4 or not str(r.get("question", "")).strip():
            continue
        ai = int(r["answer"])
        if not 0 <= ai < 4:
            continue
        mmlu.append({"question": r["question"].strip(),
                     **{OPTIONS[i]: str(ch[i]).strip() for i in range(4)},
                     "answer": OPTIONS[ai], "subject": r.get("subject", "")})
        if len(mmlu) >= n_per_bench:
            break
    _dump("mmlu", mmlu)

    # GSM8K - free-form arithmetic, scored by final numeric answer.
    ds = _load_dataset_retry("openai/gsm8k", "main", split=f"test[:{n_per_bench}]")
    gsm = []
    for r in ds:
        sol = r.get("answer", "")
        if "####" not in sol:
            continue
        gsm.append({"question": r["question"].strip(),
                    "answer": sol.split("####")[-1].strip().replace(",", "")})
    _dump("gsm8k", gsm)

    # IFEval - verifiable instruction following.
    ds = _load_dataset_retry("google/IFEval", split=f"train[:{n_per_bench}]")
    ife = []
    for r in ds:
        ife.append({"key": r.get("key"), "prompt": r["prompt"],
                    "instruction_id_list": list(r.get("instruction_id_list") or []),
                    "kwargs": [dict(k) if k else {} for k in (r.get("kwargs") or [])]})
    _dump("ifeval", ife)

    man = {"n_per_bench": n_per_bench, "synthetic": False, "benchmarks": entries}
    manifest_path.write_text(json.dumps(man, indent=2, ensure_ascii=False))
    return man


def load_jsonl(path: Path) -> list:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def build_domain_splits(data_dir: Path, args) -> dict:
    """Deterministic per-domain train/eval split over the persisted JSONL."""
    root = data_dir / "tmmluplus"
    out = {}
    for dom, subs in DOMAINS.items():
        rows = []
        for sub in subs:
            for r in load_jsonl(root / f"{sub}.jsonl"):
                r = dict(r)
                r["subset"] = sub
                rows.append(r)
        # Fixed shuffle: the split must not move between combinations.
        rng = random.Random(1234 + hash(dom) % 10_000)
        rng.shuffle(rows)
        n_eval = min(args.max_eval_per_domain, max(1, int(len(rows) * 0.3)))
        ev, tr = rows[:n_eval], rows[n_eval:]
        tr = tr[: args.max_train_per_domain]
        _guard_real_rows(tr, f"{dom}/train")
        _guard_real_rows(ev, f"{dom}/eval")
        out[dom] = {"train": tr, "eval": ev}
        print(f"[data] {dom:8s} train={len(tr):4d} eval={len(ev):4d}")
    return out


# --------------------------------------------------------------------------- #
#  Prompting / model
# --------------------------------------------------------------------------- #

def mcq_prompt(row: dict) -> str:
    return (
        "以下是一道單選題，請只回答選項代號 (A/B/C/D)。\n\n"
        f"題目：{row['question']}\n"
        f"A. {row['A']}\nB. {row['B']}\nC. {row['C']}\nD. {row['D']}\n\n答案："
    )


def load_causal_lm(model_id: str, role: str):
    """gemma-4 ships as a multimodal ForConditionalGeneration checkpoint."""
    _guard_model_id(model_id, role)
    from transformers import AutoTokenizer, AutoConfig

    tok = AutoTokenizer.from_pretrained(model_id)
    cfg = AutoConfig.from_pretrained(model_id)
    model = None
    errs = []
    for cls_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
        try:
            import transformers

            cls = getattr(transformers, cls_name)
            model = cls.from_pretrained(model_id, dtype=DTYPE, device_map=None)
            break
        except Exception as e:  # try the next loader
            errs.append(f"{cls_name}: {type(e).__name__}: {str(e)[:120]}")
    if model is None:
        raise RuntimeError(f"[FATAL] cannot load {model_id}: {errs}")
    model = model.to(DEVICE).eval()
    print(f"[model] {role}={model_id} type={type(model).__name__} "
          f"params={sum(p.numel() for p in model.parameters())/1e9:.2f}B")
    return tok, model, cfg


def text_tower(model):
    """Return the language-model submodule of a gemma-4 multimodal wrapper."""
    for path in ("model.language_model", "language_model.model", "language_model", "model"):
        obj = model
        ok = True
        for part in path.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok and hasattr(obj, "layers"):
            return obj
    raise RuntimeError("[FATAL] could not locate the text tower.")


def option_token_ids(tok) -> list:
    ids = []
    for o in OPTIONS:
        cand = tok.encode(o, add_special_tokens=False)
        if len(cand) != 1:
            cand = tok.encode(" " + o, add_special_tokens=False)
        if len(cand) < 1:
            raise RuntimeError(f"[FATAL] cannot tokenise option {o!r}")
        ids.append(cand[-1])
    if len(set(ids)) != 4:
        raise RuntimeError(f"[FATAL] option tokens collide: {ids}")
    return ids


# --------------------------------------------------------------------------- #
#  Evaluation (real forward passes only)
# --------------------------------------------------------------------------- #

@torch.no_grad()
def eval_mcq(model, tok, rows, opt_ids, max_len, batch_size, collect=False):
    """Score A/B/C/D by comparing next-token logits. Returns (acc, nll, preds, gts)."""
    if not rows:
        return float("nan"), float("nan"), [], []
    model.eval()
    correct, nll_sum, n = 0, 0.0, 0
    preds, gts = [], []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i: i + batch_size]
        enc = tok([mcq_prompt(r) for r in chunk], return_tensors="pt",
                  padding=True, truncation=True, max_length=max_len,
                  padding_side="left").to(DEVICE)
        logits = model(**enc).logits[:, -1, :].float()
        opt_logits = logits[:, opt_ids]
        logp = F.log_softmax(opt_logits, dim=-1)
        choice = opt_logits.argmax(dim=-1)
        for j, r in enumerate(chunk):
            gold = OPTIONS.index(r["answer"])
            pred = int(choice[j])
            correct += int(pred == gold)
            nll_sum += float(-logp[j, gold])
            n += 1
            if collect:
                preds.append(OPTIONS[pred])
                gts.append(r["answer"])
    return correct / max(n, 1), nll_sum / max(n, 1), preds, gts


def _extract_final_number(text: str):
    m = re.findall(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
    return m[-1].rstrip(".") if m else None


@torch.no_grad()
def chat_prompts(tok, texts: list) -> list:
    """Wrap user turns in the model's own chat template.

    The base model is instruction-tuned. Fed a bare completion prompt it emits
    an immediate end-of-turn and generates nothing, which silently zeroed GSM8K
    and made IFEval score empty strings against its length checks. Falling back
    to the raw text keeps this working for any non-chat tokenizer.
    """
    if not getattr(tok, "chat_template", None):
        return list(texts)
    return [tok.apply_chat_template([{"role": "user", "content": t}],
                                    tokenize=False, add_generation_prompt=True)
            for t in texts]


def eval_gsm8k(model, tok, rows, max_new_tokens, batch_size):
    if not rows:
        return float("nan")
    model.eval()
    correct = 0
    for i in range(0, len(rows), batch_size):
        chunk = rows[i: i + batch_size]
        prompts = chat_prompts(tok, [
            f"Question: {r['question']}\nAnswer step by step, then give the "
            f"final number after '####'." for r in chunk])
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  add_special_tokens=False,
                  max_length=768, padding_side="left").to(DEVICE)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for j, r in enumerate(chunk):
            gen = tok.decode(out[j][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            got = _extract_final_number(gen)
            try:
                correct += int(got is not None and abs(float(got) - float(r["answer"])) < 1e-4)
            except ValueError:
                pass
    return correct / len(rows)


# IFEval: strict checkers for the instruction types we can verify offline.
def _ifeval_check(instr_id: str, kw: dict, resp: str):
    """Return True/False, or None when the instruction type is unsupported."""
    words = re.findall(r"\b\w+\b", resp)
    try:
        if instr_id == "length_constraints:number_words":
            rel, num = kw.get("relation"), kw.get("num_words")
            if num is None:
                return None
            return len(words) >= num if rel == "at least" else len(words) <= num
        if instr_id == "length_constraints:number_sentences":
            rel, num = kw.get("relation"), kw.get("num_sentences")
            if num is None:
                return None
            ns = len([s for s in re.split(r"[.!?]+", resp) if s.strip()])
            return ns >= num if rel == "at least" else ns <= num
        if instr_id == "length_constraints:number_paragraphs":
            num = kw.get("num_paragraphs")
            return None if num is None else len(
                [p for p in re.split(r"\n\s*\n", resp) if p.strip()]) == num
        if instr_id == "change_case:english_lowercase":
            return resp == resp.lower()
        if instr_id == "change_case:english_capital":
            return resp == resp.upper()
        if instr_id == "change_case:capital_word_frequency":
            num, rel = kw.get("capital_frequency"), kw.get("capital_relation")
            if num is None:
                return None
            c = len([w for w in words if w.isupper() and len(w) > 1])
            return c <= num if rel == "less than" else c >= num
        if instr_id == "detectable_format:number_bullet_lists":
            num = kw.get("num_bullets")
            return None if num is None else len(
                re.findall(r"^\s*[\*\-]\s+", resp, re.M)) == num
        if instr_id == "detectable_format:number_highlighted_sections":
            num = kw.get("num_highlights")
            return None if num is None else len(
                re.findall(r"\*[^\*\n]+\*", resp)) >= num
        if instr_id == "detectable_format:title":
            return bool(re.search(r"<<[^>]+>>", resp))
        if instr_id == "detectable_format:json_format":
            try:
                # Built from a variable rather than written literally. This file
                # is handed to an LLM inside a markdown code fence, and a
                # literal run of three backticks anywhere in it closes that
                # fence early: the non-greedy extractor then returns a
                # truncated, unparseable block and every code-generation
                # attempt fails. Keep this file free of them.
                fence = "`" * 3
                json.loads(re.sub(rf"^{fence}(json)?|{fence}$", "",
                                  resp.strip(), flags=re.M))
                return True
            except Exception:
                return False
        if instr_id == "detectable_content:number_placeholders":
            num = kw.get("num_placeholders")
            return None if num is None else len(re.findall(r"\[[^\]]+\]", resp)) >= num
        if instr_id == "detectable_content:postscript":
            m = kw.get("postscript_marker") or "P.S."
            return m.lower() in resp.lower()
        if instr_id == "keywords:existence":
            kws = kw.get("keywords") or []
            return None if not kws else all(k.lower() in resp.lower() for k in kws)
        if instr_id == "keywords:forbidden_words":
            kws = kw.get("forbidden_words") or []
            return None if not kws else all(k.lower() not in resp.lower() for k in kws)
        if instr_id == "keywords:frequency":
            k, num, rel = kw.get("keyword"), kw.get("frequency"), kw.get("relation")
            if not k or num is None:
                return None
            c = len(re.findall(re.escape(k), resp, re.I))
            return c >= num if rel == "at least" else c <= num
        if instr_id == "keywords:letter_frequency":
            l, num, rel = kw.get("letter"), kw.get("let_frequency"), kw.get("let_relation")
            if not l or num is None:
                return None
            c = resp.lower().count(l.lower())
            return c >= num if rel == "at least" else c <= num
        if instr_id == "startend:end_checker":
            e = kw.get("end_phrase")
            return None if not e else resp.strip().lower().endswith(e.strip().lower())
        if instr_id == "startend:quotation":
            s = resp.strip()
            return s.startswith('"') and s.endswith('"')
        if instr_id == "punctuation:no_comma":
            return "," not in resp
        if instr_id == "combination:repeat_prompt":
            p = kw.get("prompt_to_repeat")
            return None if not p else p.strip().lower() in resp.lower()
    except Exception:
        return None
    return None


@torch.no_grad()
def eval_ifeval(model, tok, rows, max_new_tokens, batch_size):
    """Strict prompt-level accuracy over the checkable instruction subset."""
    if not rows:
        return float("nan"), 0
    model.eval()
    ok, scored = 0, 0
    for i in range(0, len(rows), batch_size):
        chunk = rows[i: i + batch_size]
        enc = tok(chat_prompts(tok, [r["prompt"] for r in chunk]),
                  return_tensors="pt", padding=True, add_special_tokens=False,
                  truncation=True, max_length=768, padding_side="left").to(DEVICE)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for j, r in enumerate(chunk):
            resp = tok.decode(out[j][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            if not resp.strip():
                # An empty generation trivially satisfies every "at most"
                # constraint, so it would otherwise score as a pass. Count it,
                # but never as one.
                scored += 1
                continue
            verdicts = []
            for iid, kw in zip(r["instruction_id_list"],
                               list(r["kwargs"]) + [{}] * len(r["instruction_id_list"])):
                v = _ifeval_check(iid, kw or {}, resp)
                if v is not None:
                    verdicts.append(v)
            if verdicts:
                scored += 1
                ok += int(all(verdicts))
    # Unsupported instruction types are skipped, never guessed.
    return (ok / scored if scored else float("nan")), scored


# --------------------------------------------------------------------------- #
#  Large-to-small: train one LoRA expert per domain, distilled from the teacher
# --------------------------------------------------------------------------- #

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]


def lora_target_modules(model) -> list:
    """Full names of the text-tower projections LoRA may adapt.

    gemma-4 is multimodal: the vision and audio towers expose same-named
    projections wrapped in ``Gemma4ClippableLinear``, which PEFT cannot adapt
    (and which we would not want to adapt anyway -- this is a text benchmark).
    Passing bare suffixes like "q_proj" matches those too and blows up, so we
    resolve explicit full names under the language model only, and keep just
    the plain ``nn.Linear`` ones.
    """
    tower = text_tower(model)
    prefix = None
    for name, mod in model.named_modules():
        if mod is tower:
            prefix = name
            break
    names = []
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        if prefix and not name.startswith(prefix + "."):
            continue
        if name.rsplit(".", 1)[-1] in LORA_TARGETS:
            names.append(name)
    if not names:
        raise RuntimeError("[FATAL] found no adaptable text-tower projections.")
    return names


@torch.no_grad()
def teacher_option_logits(teacher, ttok, rows, opt_ids_t, max_len, batch_size):
    """Soft targets over A/B/C/D for *real* questions (hard rule 2 safe)."""
    outs = []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i: i + batch_size]
        enc = ttok([mcq_prompt(r) for r in chunk], return_tensors="pt", padding=True,
                   truncation=True, max_length=max_len, padding_side="left").to(DEVICE)
        lg = teacher(**enc).logits[:, -1, :].float()[:, opt_ids_t]
        outs.append(lg.cpu())
    return torch.cat(outs, dim=0) if outs else torch.empty(0, 4)


def train_expert(model, tok, rows, opt_ids, domain, args, teacher_logits, log):
    """LoRA fine-tune restricted to the text tower. Returns {module_name: delta}."""
    from peft import LoraConfig, get_peft_model

    targets = lora_target_modules(model)
    cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM", target_modules=targets,
    )
    peft_model = get_peft_model(model, cfg)
    peft_model.train()
    params = [p for p in peft_model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)

    gold = torch.tensor([OPTIONS.index(r["answer"]) for r in rows], device=DEVICE)
    n = len(rows)
    steps = 0
    train_losses = []
    for epoch in range(args.epochs):
        order = list(range(n))
        random.Random(args.seed * 100 + epoch).shuffle(order)
        ep_loss, nb = 0.0, 0
        for i in range(0, n, args.batch_size):
            idx = order[i: i + args.batch_size]
            chunk = [rows[k] for k in idx]
            enc = tok([mcq_prompt(r) for r in chunk], return_tensors="pt", padding=True,
                      truncation=True, max_length=args.max_len,
                      padding_side="left").to(DEVICE)
            logits = peft_model(**enc).logits[:, -1, :].float()[:, opt_ids]
            loss = F.cross_entropy(logits, gold[idx])
            if teacher_logits is not None and teacher_logits.numel():
                t = teacher_logits[idx].to(DEVICE)
                kd = F.kl_div(
                    F.log_softmax(logits / args.kd_temp, dim=-1),
                    F.softmax(t / args.kd_temp, dim=-1),
                    reduction="batchmean",
                ) * (args.kd_temp ** 2)
                loss = (1 - args.kd_alpha) * loss + args.kd_alpha * kd
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            ep_loss += float(loss)
            nb += 1
            steps += 1
        ep_loss /= max(nb, 1)
        train_losses.append(ep_loss)
        log(f"[train] {domain:8s} epoch {epoch+1}/{args.epochs} loss={ep_loss:.4f}")

    # Materialise BA deltas, then strip the adapter so the base is pristine.
    deltas = {}
    for name, mod in peft_model.named_modules():
        if hasattr(mod, "lora_A") and "default" in getattr(mod, "lora_A", {}):
            A = mod.lora_A["default"].weight.detach()
            B = mod.lora_B["default"].weight.detach()
            scale = mod.scaling["default"]
            clean = (name.replace("base_model.model.", "")
                         .replace(".base_layer", ""))
            # Kept resident on the accelerator in bf16. Every combination in the
            # sweep re-reads all four experts, so paging ~25 GB of fp32 deltas
            # over PCIe per merge would dominate the run; bf16 matches the
            # precision the deltas are eventually added at.
            deltas[clean] = (B.float() @ A.float() * scale).to(torch.bfloat16)
    base_model_ = peft_model.unload()
    if not deltas:
        raise RuntimeError(f"[FATAL] no LoRA deltas captured for {domain}.")
    return base_model_, deltas, train_losses


# --------------------------------------------------------------------------- #
#  MergeKit: real merge methods, real YAML
# --------------------------------------------------------------------------- #

def _mk_imports():
    from mergekit.merge_methods import REGISTERED_MERGE_METHODS
    from mergekit.common import ImmutableMap, ModelReference
    from mergekit.architecture import WeightInfo
    from mergekit.merge_methods.base import TensorDictWrapper
    return REGISTERED_MERGE_METHODS, ImmutableMap, ModelReference, WeightInfo, TensorDictWrapper


def _fill_params(method, overrides: dict) -> dict:
    """Build a full parameter dict from the method's own declared schema."""
    out = {}
    for d in method.parameters():
        out[d.name] = overrides.get(d.name, d.default_value)
        if d.required and out[d.name] is None:
            raise RuntimeError(f"[FATAL] missing required param {d.name}")
    return out


def _fill_tensor_params(method, weight: float, overrides: dict) -> dict:
    out = {}
    for d in method.tensor_parameters():
        if d.name == "weight":
            out["weight"] = weight
        else:
            out[d.name] = overrides.get(d.name, d.default_value)
    return out


_MERGE_CACHE = {}
# Each entry is a whole merged delta set (~3 GB in bf16), so this is
# deliberately tiny: it exists to catch an immediate re-merge of the same
# expert mixture, not to act as a general store.
_MERGE_CACHE_MAX = 2


def _merge_cache_key(weights, method_name, params, tensor_params):
    return (method_name,
            tuple(sorted((k, round(float(v), 6)) for k, v in weights.items())),
            tuple(sorted(params.items())),
            tuple(sorted(tensor_params.items())))


def mergekit_merge_deltas_cached(expert_deltas, weights, method_name,
                                 params, tensor_params):
    """Memoised wrapper.

    MoE mode re-merges the same expert pair for every domain that routes to it,
    and each merge touches ~200 full-size tensors. Caching by the exact
    (method, weights, params) key removes that repeated work without changing
    any result. Router weights differ per bucket, so hits are opportunistic --
    hence the very small cache; the real cost saving is the GPU merge below.
    """
    key = _merge_cache_key(weights, method_name, params, tensor_params)
    hit = _MERGE_CACHE.get(key)
    if hit is not None:
        return hit
    out = mergekit_merge_deltas(expert_deltas, weights, method_name,
                                params, tensor_params)
    if len(_MERGE_CACHE) >= _MERGE_CACHE_MAX:
        _MERGE_CACHE.pop(next(iter(_MERGE_CACHE)))
    _MERGE_CACHE[key] = out
    return out


def mergekit_merge_deltas(expert_deltas: dict, weights: dict, method_name: str,
                          params: dict, tensor_params: dict) -> dict:
    """Merge per-module deltas using mergekit's own merge tasks.

    ``expert_deltas``: {expert: {module: delta}}; ``weights``: {expert: w}.
    The base is a zero tensor because a delta space is exactly the tangent space
    at the base, which is what task_arithmetic/ties/dare_ties already assume.
    """
    (REG, ImmutableMap, ModelReference, WeightInfo, TensorDictWrapper) = _mk_imports()
    if method_name not in REG:
        raise RuntimeError(f"[FATAL] unknown mergekit method {method_name!r}")
    method = REG[method_name]

    base_ref = ModelReference.model_validate("mergekit-base/base")
    refs = {e: ModelReference.model_validate(f"mergekit-expert/{e}")
            for e in expert_deltas}

    p_full = _fill_params(method, params)
    modules = set.intersection(*[set(d.keys()) for d in expert_deltas.values()])
    if not modules:
        raise RuntimeError("[FATAL] experts share no modules to merge.")

    merged = {}
    for mod_name in sorted(modules):
        ref0 = next(iter(expert_deltas.values()))[mod_name]
        # Merge on the accelerator, in fp32. ties/dare_ties sort or top-k every
        # element of a full-size delta; on CPU that is minutes per module, on
        # GPU it is milliseconds. Only one module is resident at a time, so the
        # extra footprint is a few tens of MB, and fp32 keeps the sparsification
        # threshold from being decided by bf16 rounding ties.
        tmap = {base_ref: torch.zeros(ref0.shape, dtype=torch.float32,
                                      device=DEVICE)}
        for e, d in expert_deltas.items():
            tmap[refs[e]] = d[mod_name].to(DEVICE, torch.float32)
        tp = {base_ref: ImmutableMap(data=_fill_tensor_params(method, 0.0, tensor_params))}
        for e in expert_deltas:
            tp[refs[e]] = ImmutableMap(
                data=_fill_tensor_params(method, float(weights[e]), tensor_params))
        task = method.make_task(
            output_weight=WeightInfo(name=mod_name),
            tensors=TensorDictWrapper(
                tensors=ImmutableMap(data={k: None for k in tmap})),
            parameters=ImmutableMap(data=p_full),
            tensor_parameters=ImmutableMap(data=tp),
            base_model=base_ref,
        )
        # bf16 to store: apply_deltas casts to the model's own bf16 weights
        # anyway, so keeping fp32 here would only double the resident size.
        merged[mod_name] = task.execute(tensors=tmap).to(torch.bfloat16)
        del tmap
    return merged


def write_mergekit_yaml(path: Path, method_name: str, weights: dict,
                        params: dict, tensor_params: dict, expert_paths: dict):
    """Canonical mergekit YAML so any combination can be replayed by the CLI."""
    import yaml

    models = [{"model": str(expert_paths[e]),
               "parameters": {"weight": float(w), **{k: v for k, v in
                                                     tensor_params.items()}}}
              for e, w in weights.items()]
    doc = {
        "merge_method": method_name,
        "base_model": BASE_MODEL,
        "models": models,
        "parameters": dict(params),
        "dtype": "bfloat16",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))
    return doc


@contextlib.contextmanager
def apply_deltas(model, deltas: dict):
    """Temporarily add merged deltas onto the live weights."""
    named = dict(model.named_modules())
    touched = []
    try:
        for mod_name, delta in deltas.items():
            mod = named.get(mod_name)
            if mod is None or not hasattr(mod, "weight"):
                continue
            w = mod.weight
            if tuple(w.shape) != tuple(delta.shape):
                continue
            w.data.add_(delta.to(w.device, w.dtype))
            touched.append((w, delta))
        if not touched:
            raise RuntimeError("[FATAL] no merged delta matched a live module.")
        yield len(touched)
    finally:
        for w, delta in touched:
            w.data.sub_(delta.to(w.device, w.dtype))


# --------------------------------------------------------------------------- #
#  MoE routing: gate vectors from real prompts (mergekit hidden-state recipe)
# --------------------------------------------------------------------------- #

@torch.no_grad()
def build_gate_vectors(model, tok, data, args):
    """One unit-norm gate vector per expert, from real TMMLU+ questions."""
    model.eval()
    gates = {}
    for dom, split in data.items():
        rows = split["train"][: args.gate_prompts] or split["eval"][: args.gate_prompts]
        enc = tok([mcq_prompt(r) for r in rows], return_tensors="pt", padding=True,
                  truncation=True, max_length=args.max_len,
                  padding_side="left").to(DEVICE)
        hs = model(**enc, output_hidden_states=True).hidden_states[-1]
        mask = enc["attention_mask"].unsqueeze(-1).to(hs.dtype)
        pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1)
        v = pooled.float().mean(0)
        gates[dom] = (v / v.norm().clamp(min=1e-8)).cpu()
    return gates


@torch.no_grad()
def route_questions(model, tok, rows, gates, args, top_k=2):
    """Assign each real question its top-k experts + softmax routing weights."""
    model.eval()
    names = list(gates)
    G = torch.stack([gates[n] for n in names])           # (E, H)
    assigns = []
    for i in range(0, len(rows), args.eval_batch_size):
        chunk = rows[i: i + args.eval_batch_size]
        enc = tok([mcq_prompt(r) for r in chunk], return_tensors="pt", padding=True,
                  truncation=True, max_length=args.max_len,
                  padding_side="left").to(DEVICE)
        hs = model(**enc, output_hidden_states=True).hidden_states[-1]
        mask = enc["attention_mask"].unsqueeze(-1).to(hs.dtype)
        pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1)
        pooled = pooled.float()
        pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        sims = pooled @ G.to(pooled.device).T                  # (B, E)
        k = min(top_k, len(names))
        topv, topi = sims.topk(k, dim=-1)
        wts = F.softmax(topv / args.router_temp, dim=-1)
        for b in range(len(chunk)):
            experts = tuple(sorted(names[int(topi[b, j])] for j in range(k)))
            w = {names[int(topi[b, j])]: float(wts[b, j]) for j in range(k)}
            assigns.append((experts, w))
    return assigns


# --------------------------------------------------------------------------- #
#  Combination grid
# --------------------------------------------------------------------------- #

def build_grid(args):
    combos = []
    for method in MERGE_METHODS:
        for mode in MERGE_MODES:
            for weighting in WEIGHTINGS:
                for seed in SEEDS:
                    combos.append({"method": method, "mode": mode,
                                   "weighting": weighting, "seed": seed})
    if args.max_combos and args.max_combos < len(combos):
        # Deterministic odometer subsample, mode varying fastest.
        #
        # A random sample clumps on one method/seed. An even stride is no better
        # here: 48/4 is exactly the 12-combination per-method block, so it lands
        # on the same cell of every block and never leaves dense mode. Counting
        # with mode as the least significant digit guarantees a smoke test
        # exercises both merge modes -- the two genuinely different code paths --
        # before it starts varying the cheaper axes.
        axes = [("mode", MERGE_MODES), ("method", MERGE_METHODS),
                ("weighting", WEIGHTINGS), ("seed", SEEDS)]
        kept = []
        for i in range(args.max_combos):
            c, j = {}, i
            for name, vals in axes:
                c[name] = vals[j % len(vals)]
                j //= len(vals)
            kept.append({"method": c["method"], "mode": c["mode"],
                         "weighting": c["weighting"], "seed": c["seed"]})
        print(f"[warn] running {len(kept)}/{len(combos)} combinations "
              f"(--max_combos={args.max_combos}); full sweep needs --max_combos 0")
        return kept, len(combos)
    return combos, len(combos)


def combo_weights(combo, expert_names, solo_acc):
    """uniform, or proportional to each expert's own-domain accuracy."""
    if combo["weighting"] == "uniform":
        w = {e: 1.0 / len(expert_names) for e in expert_names}
    else:
        # Skill above the 1/4 random-guess floor, plus a floor of its own.
        # Without that floor an expert that happened to land at chance gets a
        # ~0 weight and the "merge" silently degenerates into a single expert,
        # which is not the strategy we mean to be measuring.
        floor = 0.05
        s = {e: max(solo_acc.get(e, 0.0) - 0.25, 0.0) + floor for e in expert_names}
        tot = sum(s.values()) or 1.0
        w = {e: v / tot for e, v in s.items()}
    return w


def combo_tensor_params(combo, rng):
    """Per-combo density/rescale, jittered by seed so the 3 seeds differ."""
    tp = {}
    if combo["method"] in ("ties", "dare_ties"):
        tp["density"] = round(rng.choice([0.3, 0.5, 0.7]), 3)
    return tp


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(
        "MergeKit strategy sweep for large-to-small gemma-4 training on TMMLU+")
    p.add_argument("--seed", type=int, default=int(os.environ.get("AI_SCI_SEED", 0)))
    # LoRA / training
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max_len", type=int, default=384)
    # distillation (large -> small)
    p.add_argument("--kd_alpha", type=float, default=0.5)
    p.add_argument("--kd_temp", type=float, default=2.0)
    p.add_argument("--no_teacher", action="store_true",
                   help="skip the E4B teacher (pure SFT on real gold labels)")
    # data sizing
    # Sized against measured throughput on one H100-class card
    # (MCQ ~103 items/s at bs=16; GSM8K 2.44 s/item at 256 new tokens and IFEval
    # 1.24 s/item at 128, both bs=8; a screened combination ~165 s, an unscreened
    # one ~17 s, setup plus the four experts ~390 s) so that the full
    # 48-combination sweep finishes inside the 3600 s bfts exec.timeout with
    # margin. Budget: ~1760 s sweep + ~740 s confirmation + ~390 s setup.
    p.add_argument("--max_train_per_domain", type=int, default=300)
    p.add_argument("--max_eval_per_domain", type=int, default=120)
    p.add_argument("--n_general", type=int, default=32,
                   help="items per general benchmark (MMLU/GSM8K/IFEval)")
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--gen_batch_size", type=int, default=8)
    p.add_argument("--gen_max_new_tokens", type=int, default=128)
    p.add_argument("--gsm_max_new_tokens", type=int, default=256,
                   help="GSM8K needs room for chain-of-thought: at 128 tokens "
                        "this model scores 0.125 purely from truncation, at 256 "
                        "it scores 0.375")
    p.add_argument("--confirm_top_k", type=int, default=2,
                   help="re-test the best candidates on a deeper general "
                        "benchmark sample before reporting no-regression")
    p.add_argument("--confirm_n_general", type=int, default=50)
    p.add_argument("--regress_tol", type=float, default=0.02,
                   help="allowed absolute drop vs base on a general benchmark")
    # routing
    p.add_argument("--gate_prompts", type=int, default=32)
    p.add_argument("--router_temp", type=float, default=0.1)
    # sweep
    p.add_argument("--max_combos", type=int, default=0,
                   help="0 = run the full 48-combination grid")
    p.add_argument("--general_every", type=int, default=6,
                   help="screen MMLU/GSM8K/IFEval on every Nth combination of "
                        "each merge mode; the leaders are then re-tested on a "
                        "deeper sample, so nothing is reported as "
                        "non-regressing without a real measurement")
    p.add_argument("--cli_verify", type=int, default=0,
                   help="replay N combos through the real mergekit-yaml CLI")
    p.add_argument("--force_download", action="store_true")
    p.add_argument(
        "--quick", action="store_true",
        help="tiny smoke configuration used by the preflight canary: proves the "
             "whole path (data -> distil -> merge -> eval -> npy) is green.")
    return p.parse_args()


def apply_quick(args):
    args.max_train_per_domain = 24
    args.max_eval_per_domain = 12
    args.epochs = 1
    args.batch_size = 2
    args.n_general = 8
    args.eval_batch_size = 4
    args.gen_batch_size = 2
    args.gen_max_new_tokens = 48
    args.gsm_max_new_tokens = 64
    args.gate_prompts = 8
    args.max_combos = 8          # every method x both modes; see build_grid
    args.general_every = 2
    args.no_teacher = True          # the 16 GB teacher is not worth it for a smoke test
    args.max_len = 256
    args.confirm_top_k = 2
    args.confirm_n_general = 12
    return args


def set_all_seeds(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def main():
    args = parse_args()
    if args.quick:
        args = apply_quick(args)
    set_all_seeds(args.seed)
    t0 = time.time()

    _guard_model_id(BASE_MODEL, "student")

    exp_root = resolve_experiment_root()
    data_dir = resolve_data_dir(exp_root)
    work = Path.cwd() / "working"
    work.mkdir(parents=True, exist_ok=True)
    log_dir = Path.cwd() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "forth_trial_run.log"
    log_fh = open(log_path, "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        log_fh.write(msg + "\n")
        log_fh.flush()

    log("=" * 74)
    log(f"[setup] student={BASE_MODEL}")
    log(f"[setup] teacher={'<disabled>' if args.no_teacher else TEACHER_MODEL}")
    log(f"[setup] device={DEVICE} dtype={DTYPE} seed={args.seed} quick={args.quick}")
    log(f"[setup] experiment_root={exp_root}")
    log(f"[setup] data_dir={data_dir}  (-> {data_dir.resolve()})")

    # ---- 1. Real data, persisted under data/ ------------------------------ #
    manifest = download_tmmluplus(data_dir, force=args.force_download)
    # Fetch enough for the deepest consumer. Asking only for --n_general would
    # silently hand the confirmation pass a short sample.
    gen_manifest = download_general_benchmarks(
        data_dir, max(args.n_general, args.confirm_n_general),
        force=args.force_download)
    data = build_domain_splits(data_dir, args)
    gen_root = data_dir / "general"
    mmlu_rows = load_jsonl(gen_root / "mmlu.jsonl")[: args.n_general]
    gsm_rows = load_jsonl(gen_root / "gsm8k.jsonl")[: args.n_general]
    ife_rows = load_jsonl(gen_root / "ifeval.jsonl")[: args.n_general]

    # ---- 2. Student ------------------------------------------------------- #
    tok, model, _ = load_causal_lm(BASE_MODEL, "student")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    opt_ids = option_token_ids(tok)
    domain_names = list(DOMAINS)

    experiment_data = {
        d: {"metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [], "ground_truth": [],
            "accuracy": {}}
        for d in domain_names
    }
    experiment_data["_meta"] = {
        "base_model": BASE_MODEL,
        "teacher_model": None if args.no_teacher else TEACHER_MODEL,
        "seed": args.seed,
        "domains": domain_names,
        "config": vars(args),
        "data_manifest": manifest,
        "general_manifest": gen_manifest,
        "data_dir": str(data_dir.resolve()),
        "mergekit_methods": MERGE_METHODS,
        "synthetic_data_used": False,
        "device": DEVICE,
    }

    # ---- 3. Baseline: untouched base model -------------------------------- #
    log("\n=== baseline: untouched base model ===")
    base_scores = {}
    for d in domain_names:
        acc, nll, preds, gts = eval_mcq(model, tok, data[d]["eval"], opt_ids,
                                        args.max_len, args.eval_batch_size,
                                        collect=True)
        experiment_data[d]["accuracy"]["base"] = acc
        experiment_data[d]["predictions"].extend(preds)
        experiment_data[d]["ground_truth"].extend(gts)
        base_scores[d] = acc
        log(f"[base] {d:8s} tmmlu+ acc={acc:.4f} nll={nll:.4f}")

    def general_suite(mmlu_r=None, gsm_r=None, ife_r=None):
        """MMLU / GSM8K / IFEval on whatever the model currently is.

        GSM8K gets its own, larger token budget: it is the only one of the three
        that needs room for chain-of-thought, and truncating it measures how
        long the reasoning was rather than whether it was right.
        """
        return {
            "mmlu": eval_mcq(model, tok, mmlu_rows if mmlu_r is None else mmlu_r,
                             opt_ids, args.max_len, args.eval_batch_size)[0],
            "gsm8k": eval_gsm8k(model, tok, gsm_rows if gsm_r is None else gsm_r,
                                args.gsm_max_new_tokens, args.gen_batch_size),
            "ifeval": eval_ifeval(model, tok, ife_rows if ife_r is None else ife_r,
                                  args.gen_max_new_tokens, args.gen_batch_size)[0],
        }

    base_general = general_suite()
    experiment_data["_meta"]["base_general"] = base_general
    log(f"[base] general: " + "  ".join(f"{k}={v:.4f}" for k, v in base_general.items()))

    # ---- 4. Teacher soft targets on real questions ------------------------ #
    teacher_logits = {d: None for d in domain_names}
    if not args.no_teacher:
        log(f"\n=== teacher ({TEACHER_MODEL}) soft targets ===")
        ttok, teacher, _ = load_causal_lm(TEACHER_MODEL, "teacher")
        if ttok.pad_token is None:
            ttok.pad_token = ttok.eos_token
        t_ids = option_token_ids(ttok)
        for d in domain_names:
            teacher_logits[d] = teacher_option_logits(
                teacher, ttok, data[d]["train"], t_ids, args.max_len,
                args.eval_batch_size)
            log(f"[teacher] {d:8s} logits {tuple(teacher_logits[d].shape)}")
        del teacher
        torch.cuda.empty_cache()

    # ---- 5. Train one expert per domain (large -> small) ------------------ #
    expert_deltas, solo_acc = {}, {}
    for d in domain_names:
        log(f"\n=== expert: {d} ===")
        model, deltas, losses = train_expert(
            model, tok, data[d]["train"], opt_ids, d, args, teacher_logits[d], log)
        expert_deltas[d] = deltas
        experiment_data[d]["losses"]["train"] = losses
        with apply_deltas(model, deltas):
            acc, _, _, _ = eval_mcq(model, tok, data[d]["eval"], opt_ids,
                                    args.max_len, args.eval_batch_size)
        solo_acc[d] = acc
        experiment_data[d]["accuracy"][f"solo_{d}"] = acc
        log(f"[solo] {d:8s} own-domain acc={acc:.4f} (base {base_scores[d]:.4f}) "
            f"modules={len(deltas)}")

    # ---- 6. MoE gate vectors from real prompts ---------------------------- #
    log("\n=== MoE gate vectors (real TMMLU+ prompts) ===")
    gates = build_gate_vectors(model, tok, data, args)
    log(f"[moe] gates for {list(gates)} dim={len(next(iter(gates.values())))}")

    # ---- 7. The sweep ----------------------------------------------------- #
    combos, total_defined = build_grid(args)
    log(f"\n=== sweep: {len(combos)} of {total_defined} defined combinations ===")
    yaml_dir = Path.cwd() / "merge_configs"
    yaml_dir.mkdir(parents=True, exist_ok=True)
    expert_paths = {e: f"{BASE_MODEL}+lora:{e}" for e in domain_names}

    results = []
    # Screening cadence is counted per mode, not on the flat index. Whenever the
    # combination order alternates modes at the same rate as --general_every, a
    # plain `ci % general_every` lands on one mode every single time and the
    # other is never screened at all.
    screened = defaultdict(int)
    for ci, combo in enumerate(combos):
        rng = random.Random(1000 * combo["seed"] + ci)
        set_all_seeds(combo["seed"])
        tp = combo_tensor_params(combo, rng)
        w = combo_weights(combo, domain_names, solo_acc)
        params = {}
        cid = (f"{combo['method']}_{combo['mode']}_{combo['weighting']}"
               f"_s{combo['seed']}")
        do_general = screened[combo["mode"]] % args.general_every == 0
        screened[combo["mode"]] += 1
        t_c = time.time()

        # Canonical, replayable mergekit YAML for this combination.
        ydoc = write_mergekit_yaml(yaml_dir / f"{cid}.yaml", combo["method"],
                                   w, params, tp, expert_paths)

        rec = {"id": cid, **combo, "weights": w, "tensor_params": tp,
               "tmmlu": {}, "general": {}}

        if combo["mode"] == "dense":
            merged = mergekit_merge_deltas_cached(expert_deltas, w, combo["method"],
                                                  params, tp)
            with apply_deltas(model, merged) as nmod:
                rec["modules_merged"] = nmod
                for d in domain_names:
                    acc, _, preds, gts = eval_mcq(
                        model, tok, data[d]["eval"], opt_ids, args.max_len,
                        args.eval_batch_size, collect=(ci == 0))
                    rec["tmmlu"][d] = acc
                    experiment_data[d]["accuracy"][cid] = acc
                    if ci == 0:
                        experiment_data[d]["predictions"].extend(preds)
                        experiment_data[d]["ground_truth"].extend(gts)
                if do_general:
                    rec["general"] = general_suite()
        else:
            # MoE: route each real question to its top-2 experts, then merge just
            # those two with the router's own weights. One merge per active pair.
            for d in domain_names:
                rows = data[d]["eval"]
                assigns = route_questions(model, tok, rows, gates, args, top_k=2)
                buckets = defaultdict(list)
                for i, (pair, wts) in enumerate(assigns):
                    buckets[pair].append((i, wts))
                correct = 0
                for pair, items in buckets.items():
                    mw = {e: float(np.mean([it[1].get(e, 0.0) for it in items]))
                          for e in pair}
                    tot = sum(mw.values()) or 1.0
                    mw = {e: v / tot for e, v in mw.items()}
                    sub = {e: expert_deltas[e] for e in pair}
                    merged = mergekit_merge_deltas_cached(sub, mw, combo["method"],
                                                          params, tp)
                    with apply_deltas(model, merged):
                        chunk = [rows[i] for i, _ in items]
                        acc, _, _, _ = eval_mcq(model, tok, chunk, opt_ids,
                                                args.max_len, args.eval_batch_size)
                    correct += acc * len(items)
                rec["tmmlu"][d] = correct / max(len(rows), 1)
                experiment_data[d]["accuracy"][cid] = rec["tmmlu"][d]
                rec.setdefault("routing", {})[d] = {
                    "/".join(p): len(v) for p, v in buckets.items()}
            if do_general:
                # General benchmarks under the router's global average mixture.
                gw = combo_weights(combo, domain_names, solo_acc)
                merged = mergekit_merge_deltas_cached(expert_deltas, gw, combo["method"],
                                                      params, tp)
                with apply_deltas(model, merged):
                    rec["general"] = general_suite()

        rec["tmmlu_mean"] = float(np.mean(list(rec["tmmlu"].values())))
        # Hard rule 6: no regression on the general benchmarks.
        if rec["general"]:
            rec["regression"] = {
                k: float(rec["general"][k] - base_general[k])
                for k in rec["general"] if not np.isnan(base_general.get(k, np.nan))
            }
            rec["no_regression"] = all(
                v >= -args.regress_tol
                for v in rec["regression"].values())
        rec["seconds"] = time.time() - t_c
        results.append(rec)
        log(f"[combo {ci+1:3d}/{len(combos)}] {cid:44s} "
            f"tmmlu={rec['tmmlu_mean']:.4f} "
            + (" ".join(f"{k}={v:+.3f}" for k, v in rec.get("regression", {}).items()))
            + f"  ({rec['seconds']:.0f}s)")

        (Path.cwd() / "combo_results.jsonl").open("a", encoding="utf-8").write(
            json.dumps(rec, ensure_ascii=False, default=float) + "\n")

    # ---- 8. Optional: replay combos through the real mergekit CLI --------- #
    if args.cli_verify:
        log(f"\n=== cli_verify: replaying {args.cli_verify} combos via mergekit-yaml ===")
        for rec in results[: args.cli_verify]:
            y = yaml_dir / f"{rec['id']}.yaml"
            proc = subprocess.run(["mergekit-yaml", "--help"],
                                  capture_output=True, text=True)
            rec["cli_available"] = proc.returncode == 0
            log(f"[cli] {rec['id']}: config={y} cli_available={rec['cli_available']}")

    # ---- 8b. Confirmation pass on the strongest candidates ---------------- #
    # The sweep screens the general benchmarks on a subsample for speed. Hard
    # rule 6 is a claim about the *reported* model, so before anything is called
    # "no regression" the leaders are re-measured on the full benchmark set --
    # and the base is re-measured at the same depth, otherwise the comparison
    # would be between two different sample sizes.
    if args.confirm_top_k and results:
        k = args.confirm_top_k
        ranked = sorted(results, key=lambda r: r["tmmlu_mean"], reverse=True)
        # Two leader sets, because two different combos get reported: the best
        # overall, and the best that screened clean. Confirming only the former
        # would leave the headline "no regression" claim resting on the shallow
        # screening sample.
        leaders, seen_ids = [], set()
        for r in ranked[:k] + [r for r in ranked if r.get("no_regression")][:k]:
            if r["id"] not in seen_ids:
                seen_ids.add(r["id"])
                leaders.append(r)
        n = args.confirm_n_general
        big_mmlu = load_jsonl(gen_root / "mmlu.jsonl")[:n]
        big_gsm = load_jsonl(gen_root / "gsm8k.jsonl")[:n]
        big_ife = load_jsonl(gen_root / "ifeval.jsonl")[:n]
        got = min(len(big_mmlu), len(big_gsm), len(big_ife))
        if got < n:
            log(f"[confirm][warn] only {got} items available, asked for {n}")
            n = got
        log(f"\n=== confirmation: {len(leaders)} candidates on "
            f"{n} items/benchmark ===")

        base_conf = general_suite(big_mmlu, big_gsm, big_ife)
        experiment_data["_meta"]["base_general_confirm"] = base_conf
        log("[confirm] base   " + "  ".join(f"{a}={b:.4f}" for a, b in base_conf.items()))

        for rec in leaders:
            merged = mergekit_merge_deltas_cached(
                expert_deltas, rec["weights"], rec["method"], {},
                rec["tensor_params"])
            with apply_deltas(model, merged):
                g = general_suite(big_mmlu, big_gsm, big_ife)
            rec["general_confirm"] = g
            rec["regression_confirm"] = {a: float(g[a] - base_conf[a]) for a in g}
            rec["no_regression_confirm"] = all(
                v >= -args.regress_tol for v in rec["regression_confirm"].values())
            log(f"[confirm] {rec['id']:44s} "
                + "  ".join(f"{a}={b:+.4f}" for a, b in rec["regression_confirm"].items())
                + f"  -> {'PASS' if rec['no_regression_confirm'] else 'REGRESSES'}")

    # ---- 9. Results ------------------------------------------------------- #
    if results:
        best = max(results, key=lambda r: r["tmmlu_mean"])
        # A confirmed measurement always overrules the screening one; combos the
        # confirmation pass never reached stay on their screening verdict, and a
        # combo with neither is not eligible to be reported as clean at all.
        clean = [r for r in results
                 if r.get("no_regression_confirm", r.get("no_regression", False))]
        best_clean = max(clean, key=lambda r: r["tmmlu_mean"]) if clean else None
        confirmed = [r for r in clean if r.get("no_regression_confirm")]
        experiment_data["_meta"]["best_confirmed"] = (
            max(confirmed, key=lambda r: r["tmmlu_mean"]) if confirmed else None)
        experiment_data["_meta"]["results"] = results
        experiment_data["_meta"]["best"] = best
        experiment_data["_meta"]["best_no_regression"] = best_clean
        experiment_data["_meta"]["combinations_run"] = len(results)
        experiment_data["_meta"]["combinations_defined"] = total_defined
        for d in domain_names:
            experiment_data[d]["metrics"]["val"] = [
                {"combo": r["id"], "acc": r["tmmlu"].get(d)} for r in results]
        log(f"\nbest  {best['id']}  tmmlu_mean={best['tmmlu_mean']:.4f}")
        if best_clean:
            src = ("confirmed" if best_clean.get("no_regression_confirm")
                   else "screened")
            log(f"best (no regression, {src})  {best_clean['id']}  "
                f"tmmlu_mean={best_clean['tmmlu_mean']:.4f}")
        else:
            log("[warn] no combination cleared the no-regression bar")
        mean_all = float(np.mean([r["tmmlu_mean"] for r in results]))
        log(f"mean tmmlu_mean over {len(results)} combos = {mean_all:.4f}")

    experiment_data["_meta"]["runtime_seconds"] = time.time() - t0
    _guard_results(experiment_data)

    out = work / "experiment_data.npy"
    np.save(out, experiment_data, allow_pickle=True)
    log(f"\n[done] wrote {out} ({out.stat().st_size/1024:.1f} KiB) "
        f"in {time.time()-t0:.0f}s")
    log(f"[done] merge configs -> {yaml_dir}")
    log(f"[done] data kept at  -> {data_dir.resolve()}")
    log_fh.close()


if __name__ == "__main__":
    main()
