"""
Interference-aware merging of subject-specific LoRA adapters on the Taiwanese
junior/senior high school curriculum.

This file is injected verbatim into every tree-search node as "Code To Potentially
Use", so it is written to be RUNNABLE AS-IS, not as a sketch. There are no
placeholder functions and no dummy metrics: every number this script prints is
computed from real TMMLU+ examination questions.

Contract with the AI-Scientist harness
--------------------------------------
* finishes well inside the 1-hour node timeout on ONE GPU with the default BASE_MODEL
* writes ``working/experiment_data.npy`` in the schema the harness expects
* prints ``validation_loss`` per epoch and a final ``test accuracy`` per domain
* every plot is written to ``working/`` as PNG

Hard environment facts (verified on this cluster, do not "fix" them):
* transformers 5.16.1 -> ``TrainingArguments(eval_strategy=...)``; ``gradient_checkpointing``
  is NOT a ``from_pretrained`` kwarg. This script avoids ``Trainer`` entirely.
* ``trl`` and ``bitsandbytes`` are NOT installed. Do not import them.
* The GPUs are shared with the local Qwen inference servers. NEVER use
  ``device_map="auto"`` and NEVER load in fp32 -- that is what OOM-killed every
  previous run. Always bf16 + ``device_map={"": 0}``.
"""

import argparse
import json
import math
import os
import random
import time
from collections import OrderedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
# One switch for the whole study. Everything else is model-agnostic: LORA_TARGETS
# below are the attention projections, which are named identically in the Qwen2,
# Llama and Gemma families, so swapping BASE_MODEL needs no other edit.
#
#   search   : google/gemma-3-1b-it        (~2 GB bf16, full study in ~15 min)  <- default
#   mid      : google/gemma-2-2b-it        (~5 GB bf16)
#   confirm  : google/gemma-2-27b          (~54 GB bf16, cached on this cluster)
#   open alt : Qwen/Qwen2.5-1.5B-Instruct  (needs no token; TMMLU+ baseline model)
#
# Every google/gemma-* repo is GATED. third_trial.sh reads the token from .env
# (stored there as `hk_token`) and exports it as HF_TOKEN; verified to unlock
# gemma-3-1b-it, gemma-2-2b-it and gemma-2-27b. If you run this script by hand,
# export HF_TOKEN yourself or switch to the open alternative above.
BASE_MODEL = os.environ.get("BASE_MODEL", "google/gemma-3-1b-it")

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# The four merge domains, mapped onto real TMMLU+ subsets.
# `taiwan` is the culturally-grounded domain this study is about.
DOMAINS = OrderedDict(
    [
        ("chinese", ["junior_chinese_exam", "chinese_language_and_literature", "tve_chinese_language"]),
        ("math", ["junior_math_exam", "tve_mathematics"]),
        ("taiwan", ["geography_of_taiwan", "junior_social_studies", "jce_humanities"]),
        ("science", ["junior_science_exam", "junior_chemistry", "secondary_physics"]),
    ]
)

CHOICES = ["A", "B", "C", "D"]
TMMLU_REPO = "ikala/tmmluplus"


def parse_args():
    p = argparse.ArgumentParser("Interference-aware LoRA merging on Taiwan curriculum")
    p.add_argument("--seed", type=int, default=int(os.environ.get("AI_SCI_SEED", 0)))
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max_len", type=int, default=384)
    p.add_argument("--train_frac", type=float, default=0.6)
    p.add_argument("--max_train_per_domain", type=int, default=400)
    p.add_argument("--max_eval_per_domain", type=int, default=250)
    p.add_argument("--ties_density", type=float, default=0.2, help="top-k fraction kept by TIES trim")
    p.add_argument("--dare_p", type=float, default=0.5, help="DARE drop probability")
    p.add_argument(
        "--quick",
        action="store_true",
        help="Canary mode: tiny subsample, 1 epoch. Proves the whole pipeline "
        "(data -> train -> merge -> eval -> experiment_data.npy) is green in a "
        "few minutes. The numbers are meaningless; only the exit code matters.",
    )
    args = p.parse_args()
    if args.quick:
        args.epochs = 1
        args.max_train_per_domain = 32
        args.max_eval_per_domain = 32
        args.max_len = 192
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Data: real Taiwanese exam questions from TMMLU+
# --------------------------------------------------------------------------- #
def build_prompt(row):
    """Traditional-Chinese MCQ prompt. Ends immediately before the answer letter."""
    return (
        "以下是臺灣中學課程的單選題，請選出最正確的答案。\n\n"
        f"題目：{row['question'].strip()}\n"
        f"(A) {row['A']}\n(B) {row['B']}\n(C) {row['C']}\n(D) {row['D']}\n"
        "答案："
    )


def load_domains(args):
    """Return {domain: {"train": [...], "eval": [...]}} of dicts with prompt/label."""
    rng = random.Random(args.seed)
    data = {}
    for domain, subsets in DOMAINS.items():
        rows = []
        for subset in subsets:
            ds = load_dataset(TMMLU_REPO, subset, split="test")
            for r in ds:
                ans = str(r["answer"]).strip().upper()
                if ans not in CHOICES:
                    continue
                rows.append(
                    {
                        "prompt": build_prompt(r),
                        "label": CHOICES.index(ans),
                        "subset": subset,
                    }
                )
        rng.shuffle(rows)
        n_train = int(len(rows) * args.train_frac)
        train = rows[:n_train][: args.max_train_per_domain]
        evalr = rows[n_train:][: args.max_eval_per_domain]
        data[domain] = {"train": train, "eval": evalr}
        print(f"[data] {domain:8s} train={len(train):4d} eval={len(evalr):4d} " f"(pool={len(rows)})")
    return data


# --------------------------------------------------------------------------- #
# Model / evaluation
# --------------------------------------------------------------------------- #
def load_base():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # last position is the answer slot for every row
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=DTYPE,
        device_map={"": 0} if torch.cuda.is_available() else None,
    )
    model.config.use_cache = False
    return tok, model


def choice_token_ids(tok):
    """Token id emitted for each option letter in this position."""
    ids = []
    for c in CHOICES:
        enc = tok.encode(c, add_special_tokens=False)
        ids.append(enc[-1])
    assert len(set(ids)) == 4, f"option letters collide in tokenizer: {ids}"
    return torch.tensor(ids, device=DEVICE)


@torch.no_grad()
def evaluate(model, tok, rows, opt_ids, max_len, batch_size=8):
    """Accuracy + mean NLL over the four option letters. One forward pass per batch."""
    model.eval()
    correct, total, nll_sum = 0, 0, 0.0
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        enc = tok(
            [b["prompt"] for b in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        ).to(DEVICE)
        logits = model(**enc).logits[:, -1, :].float()
        opt_logits = logits.index_select(-1, opt_ids)  # [B, 4]
        logprobs = F.log_softmax(opt_logits, dim=-1)
        labels = torch.tensor([b["label"] for b in batch], device=DEVICE)
        correct += (opt_logits.argmax(-1) == labels).sum().item()
        nll_sum += (-logprobs.gather(1, labels[:, None])).sum().item()
        total += len(batch)
    return correct / max(total, 1), nll_sum / max(total, 1)


# --------------------------------------------------------------------------- #
# LoRA training (plain torch loop -- no Trainer, no trl)
# --------------------------------------------------------------------------- #
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]


def train_adapter(base_model, tok, rows, eval_rows, opt_ids, domain, args, experiment_data):
    """Fine-tune one LoRA adapter. Loss is CE over the four option letters."""
    cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGETS,
    )
    peft_model = get_peft_model(base_model, cfg, adapter_name=domain)
    peft_model.train()

    params = [p for p in peft_model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in params)
    print(f"[train:{domain}] trainable params = {n_trainable:,}")
    optim = torch.optim.AdamW(params, lr=args.lr)

    steps_per_epoch = math.ceil(len(rows) / args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=args.lr, total_steps=max(args.epochs * steps_per_epoch, 1), pct_start=0.1
    )

    dm = experiment_data[domain]
    for epoch in range(args.epochs):
        peft_model.train()
        random.shuffle(rows)
        running = 0.0
        for i in range(0, len(rows), args.batch_size):
            batch = rows[i : i + args.batch_size]
            enc = tok(
                [b["prompt"] for b in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_len,
            ).to(DEVICE)
            logits = peft_model(**enc).logits[:, -1, :].float()
            opt_logits = logits.index_select(-1, opt_ids)
            labels = torch.tensor([b["label"] for b in batch], device=DEVICE)
            loss = F.cross_entropy(opt_logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            running += loss.item() * len(batch)

        train_loss = running / max(len(rows), 1)
        val_acc, val_loss = evaluate(peft_model, tok, eval_rows, opt_ids, args.max_len)
        # Framework requirement: print validation loss at every epoch.
        print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}")
        print(f"[train:{domain}] epoch {epoch} train_loss={train_loss:.4f} val_acc={val_acc:.4f}")
        dm["losses"]["train"].append({"epoch": epoch, "value": train_loss})
        dm["losses"]["val"].append({"epoch": epoch, "value": val_loss})
        dm["metrics"]["train"].append({"epoch": epoch, "value": float("nan")})
        dm["metrics"]["val"].append({"epoch": epoch, "value": val_acc})

    # Extract the LoRA factors, then strip the adapter so the base model is clean again.
    deltas = extract_lora_factors(peft_model, domain, args)
    base = peft_model.unload()  # returns the base model with LoRA layers removed
    return base, deltas


def extract_lora_factors(peft_model, adapter_name, args):
    """{module_path: (A[r,in], B[out,r], scaling)} on CPU float32."""
    factors = {}
    scaling = args.lora_alpha / args.lora_r
    for name, module in peft_model.named_modules():
        if not hasattr(module, "lora_A") or adapter_name not in getattr(module, "lora_A", {}):
            continue
        A = module.lora_A[adapter_name].weight.detach().float().cpu()
        B = module.lora_B[adapter_name].weight.detach().float().cpu()
        # strip the peft wrapper prefix so the name matches the bare base model
        clean = name.replace("base_model.model.", "").replace("base_model.", "")
        factors[clean] = (A, B, scaling)
    assert factors, "no LoRA factors found -- target_modules did not match"
    return factors


# --------------------------------------------------------------------------- #
# Merging operators (operate on dense per-module deltas)
# --------------------------------------------------------------------------- #
def materialize(factors, module):
    A, B, s = factors[module]
    return (B @ A) * s  # [out, in]


def trim(delta, density):
    """Keep the top-`density` fraction by magnitude, zero the rest (TIES step 1)."""
    if density >= 1.0:
        return delta
    flat = delta.abs().flatten()
    k = max(int(flat.numel() * density), 1)
    thresh = torch.topk(flat, k, largest=True).values.min()
    return delta * (delta.abs() >= thresh)


def dare_drop(delta, p, generator):
    """Randomly drop entries with prob p and rescale survivors by 1/(1-p)."""
    if p <= 0.0:
        return delta
    mask = (torch.rand(delta.shape, generator=generator) >= p).to(delta.dtype)
    return delta * mask / (1.0 - p)


def merge_task_arithmetic(deltas, coeffs):
    return sum(c * d for c, d in zip(coeffs, deltas))


def merge_ties(deltas, coeffs, density):
    """Trim -> elect sign by summed signed magnitude -> mean over agreeing entries."""
    trimmed = [trim(d, density) for d in deltas]
    stacked = torch.stack([c * t for c, t in zip(coeffs, trimmed)])  # [n, out, in]
    elected = torch.sign(stacked.sum(dim=0))
    elected[elected == 0] = 1.0
    agree = (torch.sign(stacked) == elected) & (stacked != 0)
    num = (stacked * agree).sum(dim=0)
    den = agree.sum(dim=0).clamp(min=1)
    return num / den


def merge_dare_ties(deltas, coeffs, density, p, generator):
    dropped = [dare_drop(d, p, generator) for d in deltas]
    return merge_ties(dropped, coeffs, density)


def sign_conflict_rate(d1, d2, density):
    """Fraction of jointly-surviving coordinates where the two deltas disagree in sign.

    This is the diagnostic the study is built around.
    """
    t1, t2 = trim(d1, density), trim(d2, density)
    both = (t1 != 0) & (t2 != 0)
    n = both.sum().item()
    if n == 0:
        return 0.0
    disagree = ((torch.sign(t1) != torch.sign(t2)) & both).sum().item()
    return disagree / n


# --------------------------------------------------------------------------- #
# Applying a merged delta to the base model
# --------------------------------------------------------------------------- #
class ApplyDeltas:
    """Context manager: add merged deltas into base weights, then exactly restore."""

    def __init__(self, model, merged):
        self.model = model
        self.merged = merged
        self.saved = {}

    def __enter__(self):
        mods = dict(self.model.named_modules())
        for name, delta in self.merged.items():
            mod = mods[name]
            w = mod.weight
            self.saved[name] = w.detach().clone()
            w.data.add_(delta.to(device=w.device, dtype=w.dtype))
        return self.model

    def __exit__(self, *exc):
        mods = dict(self.model.named_modules())
        for name, w0 in self.saved.items():
            mods[name].weight.data.copy_(w0)
        self.saved.clear()
        return False


def build_merged(all_factors, domains, method, args, generator):
    """Return {module_path: merged delta} across the given domains."""
    modules = sorted(set.intersection(*[set(all_factors[d]) for d in domains]))
    coeffs = [1.0 / len(domains)] * len(domains)
    merged = {}
    for m in modules:
        deltas = [materialize(all_factors[d], m) for d in domains]
        if method == "task_arithmetic":
            merged[m] = merge_task_arithmetic(deltas, coeffs)
        elif method == "ties":
            merged[m] = merge_ties(deltas, coeffs, args.ties_density)
        elif method == "dare_ties":
            merged[m] = merge_dare_ties(deltas, coeffs, args.ties_density, args.dare_p, generator)
        else:
            raise ValueError(f"unknown merge method: {method}")
    return merged


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    set_seed(args.seed)
    t_start = time.time()
    print(f"[setup] BASE_MODEL={BASE_MODEL} device={DEVICE} dtype={DTYPE} seed={args.seed}")

    data = load_domains(args)
    tok, model = load_base()
    opt_ids = choice_token_ids(tok)
    domain_names = list(DOMAINS)

    experiment_data = {
        d: {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "accuracy": {},  # method -> accuracy on this domain's held-out set
        }
        for d in domain_names
    }
    experiment_data["_meta"] = {
        "base_model": BASE_MODEL,
        "seed": args.seed,
        "domains": domain_names,
        "config": vars(args),
    }

    # ---- 0. Base model, no adapters -------------------------------------- #
    print("\n=== base model (no adapter) ===")
    for d in domain_names:
        acc, nll = evaluate(model, tok, data[d]["eval"], opt_ids, args.max_len)
        experiment_data[d]["accuracy"]["base"] = acc
        print(f"[base] {d:8s} test accuracy = {acc:.4f} (nll {nll:.4f})")

    # ---- 1. One LoRA adapter per subject ---------------------------------- #
    all_factors = {}
    for d in domain_names:
        print(f"\n=== training adapter: {d} ===")
        model, factors = train_adapter(
            model, tok, data[d]["train"], data[d]["eval"], opt_ids, d, args, experiment_data
        )
        all_factors[d] = factors

    # ---- 2. Each adapter alone, on every domain (the interference baseline) - #
    print("\n=== individual adapters ===")
    for src in domain_names:
        solo = {m: materialize(all_factors[src], m) for m in all_factors[src]}
        with ApplyDeltas(model, solo):
            for tgt in domain_names:
                acc, _ = evaluate(model, tok, data[tgt]["eval"], opt_ids, args.max_len)
                experiment_data[tgt]["accuracy"][f"solo_{src}"] = acc
                if src == tgt:
                    print(f"[solo] {src:8s} on own domain: {acc:.4f}")

    # ---- 3. Merged models ------------------------------------------------- #
    # Retention is only meaningful when the single adapter actually learned the
    # domain. Below this floor the ratio is dominated by noise, so we refuse to
    # report it rather than emitting a misleading number.
    RETENTION_FLOOR = 0.30
    reportable = [d for d in domain_names if experiment_data[d]["accuracy"][f"solo_{d}"] >= RETENTION_FLOOR]
    if len(reportable) < len(domain_names):
        skipped = [d for d in domain_names if d not in reportable]
        print(f"[warn] solo accuracy below {RETENTION_FLOOR} for {skipped}; "
              f"retention will be reported as NaN for these domains and excluded from the mean")

    generator = torch.Generator().manual_seed(args.seed)
    for method in ["task_arithmetic", "ties", "dare_ties"]:
        print(f"\n=== merged: {method} ===")
        merged = build_merged(all_factors, domain_names, method, args, generator)
        with ApplyDeltas(model, merged):
            for d in domain_names:
                acc, _ = evaluate(model, tok, data[d]["eval"], opt_ids, args.max_len)
                experiment_data[d]["accuracy"][method] = acc
                solo = experiment_data[d]["accuracy"][f"solo_{d}"]
                ret = acc / solo if d in reportable else float("nan")
                experiment_data[d].setdefault("retention", {})[method] = ret
                print(f"[{method}] {d:8s} acc={acc:.4f} retention={ret:.3f}")

    # ---- 4. The diagnostic: pairwise sign conflict ------------------------ #
    print("\n=== pairwise sign-conflict rate ===")
    conflict = np.zeros((len(domain_names), len(domain_names)))
    shared = sorted(set.intersection(*[set(all_factors[d]) for d in domain_names]))
    for i, a in enumerate(domain_names):
        for j, b in enumerate(domain_names):
            if i >= j:
                continue
            rates = [
                sign_conflict_rate(materialize(all_factors[a], m), materialize(all_factors[b], m), args.ties_density)
                for m in shared
            ]
            r = float(np.mean(rates))
            conflict[i, j] = conflict[j, i] = r
            print(f"[conflict] {a:8s} <-> {b:8s} = {r:.4f}")
    experiment_data["_meta"]["conflict_matrix"] = conflict.tolist()

    # ---- 5. Headline metric ---------------------------------------------- #
    methods_eval = ["task_arithmetic", "ties", "dare_ties"]
    if reportable:
        retentions = {
            m: float(np.mean([experiment_data[d]["retention"][m] for d in reportable]))
            for m in methods_eval
        }
    else:
        retentions = {m: float("nan") for m in methods_eval}
    experiment_data["_meta"]["mean_retention"] = retentions
    experiment_data["_meta"]["reportable_domains"] = reportable

    print("\n=== summary ===")
    for m, r in retentions.items():
        print(f"mean retention ({m}) = {r:.4f}")
    # Absolute accuracy is always meaningful even when retention is not, so it is
    # the metric the tree search should rank on.
    mean_acc = {m: float(np.mean([experiment_data[d]["accuracy"][m] for d in domain_names])) for m in methods_eval}
    experiment_data["_meta"]["mean_accuracy"] = mean_acc
    for m, a in mean_acc.items():
        print(f"mean merged test accuracy ({m}) = {a:.4f}")
    best = max(mean_acc, key=mean_acc.get)
    print(f"best merge method = {best}")
    print(f"taiwan domain retention (ties) = {experiment_data['taiwan'].get('retention', {}).get('ties', float('nan')):.4f}")
    print(f"math domain retention (ties)   = {experiment_data['math'].get('retention', {}).get('ties', float('nan')):.4f}")

    # ---- 5b. The claim itself: conflict vs retention ---------------------- #
    # Idea 0 succeeds or fails on Spearman rho between a domain's mean sign
    # conflict against the others and how much accuracy it keeps after merging.
    # It lives here, in the seed, so every node inherits one correct
    # implementation instead of re-deriving it (and forgetting the import).
    from scipy.stats import spearmanr

    mean_conflict = {
        d: float(np.mean([conflict[i, j] for j in range(len(domain_names)) if j != i]))
        for i, d in enumerate(domain_names)
    }
    experiment_data["_meta"]["mean_conflict_per_domain"] = mean_conflict
    experiment_data["_meta"]["conflict_retention_spearman"] = {}
    print("\n=== conflict vs retention (the hypothesis) ===")
    for m in methods_eval:
        pairs = [
            (mean_conflict[d], experiment_data[d]["retention"][m])
            for d in reportable
            if not np.isnan(experiment_data[d]["retention"][m])
        ]
        # 4 domains is a tiny n; rho is a direction indicator, not a p-value.
        if len(pairs) >= 3:
            rho, p = spearmanr([x for x, _ in pairs], [y for _, y in pairs])
            rho, p = float(rho), float(p)
        else:
            rho = p = float("nan")
        experiment_data["_meta"]["conflict_retention_spearman"][m] = {
            "rho": rho, "p": p, "n": len(pairs)
        }
        print(f"spearman rho ({m}) = {rho:.4f}  (p={p:.4f}, n={len(pairs)})")

    np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True)

    # ---- 6. Plots --------------------------------------------------------- #
    methods = ["base", "ties", "dare_ties", "task_arithmetic"]
    x = np.arange(len(domain_names))
    width = 0.8 / (len(methods) + 1)
    plt.figure(figsize=(9, 4.5))
    for k, m in enumerate(methods + ["solo"]):
        vals = [
            experiment_data[d]["accuracy"][f"solo_{d}" if m == "solo" else m] for d in domain_names
        ]
        plt.bar(x + k * width, vals, width, label=m)
    plt.axhline(0.25, ls="--", lw=1, c="gray")
    plt.xticks(x + width * len(methods) / 2, domain_names)
    plt.ylabel("TMMLU+ accuracy")
    plt.title(f"Per-domain accuracy by merge method ({BASE_MODEL})")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "accuracy_by_method.png"), dpi=120)
    plt.close()

    plt.figure(figsize=(5, 4.2))
    plt.imshow(conflict, cmap="magma")
    plt.colorbar(label="sign-conflict rate")
    plt.xticks(range(len(domain_names)), domain_names, rotation=45, ha="right")
    plt.yticks(range(len(domain_names)), domain_names)
    plt.title("Pairwise LoRA sign conflict")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "sign_conflict.png"), dpi=120)
    plt.close()

    plt.figure(figsize=(7, 4))
    for d in domain_names:
        ys = [e["value"] for e in experiment_data[d]["losses"]["val"]]
        plt.plot(range(len(ys)), ys, marker="o", label=d)
    plt.xlabel("epoch")
    plt.ylabel("validation loss")
    plt.title("Per-domain adapter training")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "val_loss.png"), dpi=120)
    plt.close()

    with open(os.path.join(working_dir, "results.json"), "w") as f:
        json.dump(
            {
                "mean_retention": retentions,
                "mean_accuracy": mean_acc,
                "reportable_domains": reportable,
                "per_domain": {d: experiment_data[d]["accuracy"] for d in domain_names},
                "conflict_matrix": conflict.tolist(),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nExperiment completed in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
