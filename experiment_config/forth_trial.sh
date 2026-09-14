#!/usr/bin/env bash
# =============================================================================
#  forth_trial.sh — Systematic Evaluation of MergeKit Strategies (Large-to-Small)
# =============================================================================
#
#  Usage:
#      conda activate ai_scientist_v2
#      bash experiment_config/forth_trial.sh              # all ideas
#      IDEAS="0" bash experiment_config/forth_trial.sh    # just the first idea
#      CITE_ROUNDS=40 bash experiment_config/forth_trial.sh   # deeper bibliography
#
#  Model roles:
#      code / feedback / summary / citation : kimi-k3                       remote (.env)
#      vlm_feedback / review                : local/qwen2.5-vl-32b          :8000  GPU 3
#      writeup                              : kimi-k3                       remote (.env)
#
#  The local VLM server must already be running, pinned and capped:
#      CUDA_VISIBLE_DEVICES=3 SERVE_MAX_MEM=72GiB python src/serve_vlm.py   --port 8000
#  All experiment workers run on GPU 3, sharing it with the capped VLM.
# =============================================================================
set -uo pipefail

# ── Repo root (works both here and at /workspace) ────────────────────────────
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

# ── Tunables ─────────────────────────────────────────────────────────────────
IDEAS="${IDEAS:-0 1}"                         # idea indices in forth_trial.json
# GPU(s) for experiment workers ONLY.
EXP_GPUS="${EXP_GPUS:-3}"
EXP_GPUS="$(echo "$EXP_GPUS" | tr -s ' ,' ',' | sed 's/^,//; s/,$//')"
VLM_URL="${VLM_URL:-http://localhost:8000/v1}"

# Citation-gathering rounds. Each round is one LLM call that yields at MOST one
# new bibtex entry, and rounds that return a duplicate title or throw are
# swallowed silently -- so this is a ceiling on the reference count, and the
# realistic yield is below it. third_trial ran 10 (carried over here without a
# reason); that does not fit this idea. The mandatory cites alone -- mergekit,
# the four merge methods (linear/task-arithmetic/TIES/DARE), the four
# benchmarks (TMMLU+/MMLU/GSM8K/IFEval), gemma, LoRA, distillation -- already
# exceed 10, which would leave the related-work and research-gap categories
# empty. Resumable via cached_citations.bib, so raising it later is cheap.
CITE_ROUNDS="${CITE_ROUNDS:-25}"

# Main Models
WRITEUP_MODEL="${WRITEUP_MODEL:-kimi-k3}"
REVIEW_MODEL="${REVIEW_MODEL:-local/qwen2.5-vl-32b}"

# Base model from forth_trial proposal
BASE_MODEL="${BASE_MODEL:-google/gemma-4-E2B-it-qat-q4_0-unquantized}"
IDEAS_JSON="$ROOT/ai_scientist/ideas/forth_trial/forth_trial.json"
CANARY="${CANARY:-1}"                         # CANARY=0 to skip the seed-code check

# ── Secrets from .env ────────────────────────────────────────────────────────
if [[ -f "$ROOT/.env" ]]; then
  set -a; source "$ROOT/.env"; set +a
  [[ -n "${hk_token:-}" && -z "${HF_TOKEN:-}" ]] && export HF_TOKEN="$hk_token"
fi

# ── Environment ──────────────────────────────────────────────────────────────
export HF_HOME="${HF_HOME:-/tmp/hf_cache_$USER}"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="$EXP_GPUS"
export BASE_MODEL                              # read by Python script
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "$HF_HOME"

export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-local-dummy}"
export HF_TOKEN="${HF_TOKEN:-}"
[[ -z "$HF_TOKEN" && -r "$HOME/.cache/huggingface/token" ]] && \
  export HF_TOKEN="$(<"$HOME/.cache/huggingface/token")"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/logs/forth_trial_$RUN_TS"
mkdir -p "$LOG_DIR"

# =============================================================================
#  Preflight
# =============================================================================
fail=0
say()  { printf '  %-46s %s\n' "$1" "$2"; }
ok()   { say "$1" "OK"; }
bad()  { say "$1" "FAIL  <- $2"; fail=1; }
warn() { say "$1" "WARN  <- $2"; }

echo "=============================================================="
echo " Preflight  ($RUN_TS)"
echo "=============================================================="

# 1. conda env
if python -c "import torch, peft, transformers, datasets" 2>/dev/null; then
  ok "python deps (torch/peft/transformers/datasets)"
else
  bad "python deps" "activate ai_scientist_v2 first"
fi

# 2. HF_HOME must support flock
if python - <<'PY' 2>/dev/null
import fcntl, os, tempfile
fd = tempfile.NamedTemporaryFile(dir=os.environ["HF_HOME"], delete=False)
fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
fcntl.flock(fd.fileno(), fcntl.LOCK_UN); fd.close(); os.unlink(fd.name)
PY
then ok "HF_HOME flock-capable ($HF_HOME)"
else bad "HF_HOME flock-capable" "$HF_HOME is on NFS; set HF_HOME=/tmp/..."
fi

# 3. LLM servers reachable
# We only check VLM here since Coder is now remote (kimi-k3)
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-60}"
for pair in "vlm:$VLM_URL"; do
  name="${pair%%:*}"; url="${pair#*:}"
  code=$(curl -s -m "$HEALTH_TIMEOUT" -o /dev/null -w "%{http_code}" "$url/models" 2>/dev/null)
  [[ "$code" == "000" ]] && \
    code=$(curl -s -m "$HEALTH_TIMEOUT" -o /dev/null -w "%{http_code}" "${url%/v1}/docs" 2>/dev/null)
  if [[ "$code" != "000" && -n "$code" ]]; then
    ok "$name server reachable ($url, HTTP $code)"
  else
    bad "$name server reachable" "no response from $url within ${HEALTH_TIMEOUT}s — start src/serve_${name}.py"
  fi
done

# 3b. Remote writeup/coder model check (kimi-k3)
if [[ "$WRITEUP_MODEL" == "kimi-k3" ]]; then
  if [[ -z "${KIMI_API_BASE:-}" || -z "${KIMI_API_KEY:-}" ]]; then
    bad "kimi-k3 credentials" "KIMI_API_BASE / KIMI_API_KEY missing from .env"
  elif curl -s -m 15 -H "Authorization: Bearer $KIMI_API_KEY" \
         "$KIMI_API_BASE/models" 2>/dev/null | grep -q '"kimi-k3"'; then
    ok "kimi-k3 endpoint ($KIMI_API_BASE)"
  else
    bad "kimi-k3 endpoint" "$KIMI_API_BASE/models did not list kimi-k3"
  fi
fi

# 3c. Ensure bfts_config doesn't use old coders if we want kimi
if grep -rn "local/qwen3-coder-30b-a3b-instruct" bfts_config.yaml >/dev/null 2>&1; then
  bad "bfts_config.yaml model ids" "still references local/qwen3-coder-30b — ensure it is updated to kimi-k3"
else
  ok "bfts_config.yaml model ids"
fi

# 4. GPU headroom. Sharing a card with a *capped* server is fine.
MIN_FREE_MIB="${MIN_FREE_MIB:-24576}"
if command -v nvidia-smi >/dev/null 2>&1; then
  for g in ${EXP_GPUS//,/ }; do
    line=$(nvidia-smi --id="$g" --query-gpu=memory.used,memory.total \
             --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    used="${line%%,*}"; total="${line##*,}"
    if [[ ! "$used" =~ ^[0-9]+$ || ! "$total" =~ ^[0-9]+$ || "$total" -eq 0 ]]; then
      warn "GPU $g free memory" "could not query nvidia-smi"
    else
      avail=$(( total - used ))
      if (( avail < MIN_FREE_MIB )); then
        bad "GPU $g free memory" "only ${avail} MiB free of ${total} (need ${MIN_FREE_MIB}) — cap or move the server on this card"
      elif (( used > total / 4 )); then
        warn "GPU $g shared" "${used}/${total} MiB in use, ${avail} MiB free — sharing with a server; keep SERVE_MAX_MEM set"
      else
        ok "GPU $g free (${avail}/${total} MiB available)"
      fi
    fi
  done
else
  warn "nvidia-smi" "not available; cannot verify GPU pinning"
fi

# 5. Prefetch base model + TMMLU+ ONCE, serially.
mkdir -p "$HF_HOME/hub"
linked=0
for src in "$HOME/.cache/huggingface/hub" "$ROOT/model"; do
  [[ -d "$src" ]] || continue
  for repo in "$src"/models--* "$src"/datasets--*; do
    [[ -e "$repo" ]] || continue
    dst="$HF_HOME/hub/$(basename "$repo")"
    [[ -e "$dst" ]] || { ln -s "$repo" "$dst" && linked=$((linked + 1)); }
  done
done
(( linked )) && ok "linked $linked cached repo(s) into HF_HOME"

echo "  prefetching model + dataset (serial, may take a few minutes)..."
python - <<'PY' >>"$LOG_DIR/preflight.log" 2>&1
import os, sys
from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset
m = os.environ["BASE_MODEL"]
try:
    AutoConfig.from_pretrained(m); AutoTokenizer.from_pretrained(m)
except Exception as e:
    txt = f"{type(e).__name__}: {e}"
    print(txt)
    sys.exit(3 if ("gated" in txt or "401" in txt or "restricted" in txt) else 1)
subsets = ["junior_chinese_exam","chinese_language_and_literature","tve_chinese_language",
           "junior_math_exam","tve_mathematics",
           "geography_of_taiwan","junior_social_studies","jce_humanities",
           "junior_science_exam","junior_chemistry","secondary_physics"]
n = 0
for s in subsets:
    n += len(load_dataset("ikala/tmmluplus", s, split="test"))
print(f"prefetch ok: {m} + {len(subsets)} TMMLU+ subsets, {n} test rows")
PY
case $? in
  0) ok "base model + TMMLU+ cached" ;;
  3) bad "base model access ($BASE_MODEL)" \
        "gated/unavailable repo. Ensure you have access." ;;
  *) bad "base model + TMMLU+ cached" "see $LOG_DIR/preflight.log" ;;
esac

# 5b. Worker/GPU sanity
python - <<'PY'
import os, re, yaml
try:
    nw = yaml.safe_load(open("bfts_config.yaml"))["agent"]["num_workers"]
except Exception:
    nw = 1
ngpu = len([d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()])
eff = min(nw, ngpu) if ngpu else 0
m = os.environ["BASE_MODEL"]
b = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", m.replace("-", " "))
print(f"  {'note: agent.num_workers=' + str(nw) + ', ' + str(ngpu) + ' GPU(s) pinned':<46} -> {eff} parallel node(s)")
if nw > ngpu > 0:
    print(f"  {'note: workers clamped to GPU count':<46} widen EXP_GPUS to use all {nw}")
if b:
    gb = float(b.group(1)) * 2
    print(f"  {'note: ' + m + ' bf16 weights':<46} ~{gb:.0f} GiB per GPU (+ activations)")
PY

# 6. Ideas JSON / code template
if [[ -f "$IDEAS_JSON" ]]; then
  n=$(python -c "import json;print(len(json.load(open('$IDEAS_JSON'))))" 2>/dev/null)
  ok "ideas JSON ($n ideas)"
  for i in $IDEAS; do
    (( i < ${n:-0} )) || bad "idea index $i in range" "JSON only has ${n:-0}"
  done
else
  bad "ideas JSON" "$IDEAS_JSON missing"
fi
TEMPLATE_PY="${IDEAS_JSON%.json}.py"
if [[ -f "$TEMPLATE_PY" ]]; then
  ok "code template"
  # --load_code hands this file to the LLM inside a markdown code fence. A
  # literal run of three backticks anywhere in it closes that fence early, so
  # the extractor gets a truncated block, rejects it as invalid Python, and
  # every node burns three ~7-minute retries before giving up. Silent and
  # expensive, so gate on it.
  if grep -q '```' "$TEMPLATE_PY"; then
    bad "code template has no literal code fences" \
        "$(grep -n '```' "$TEMPLATE_PY" | head -3 | cut -c1-60)"
  else
    ok "code template has no literal code fences"
  fi
else
  bad "code template" "$TEMPLATE_PY missing"
fi

echo
if (( fail )); then
  echo "Preflight FAILED — fix the above before running. Nothing was launched."
  exit 1
fi

# =============================================================================
#  Canary: run the seed code itself
# =============================================================================
if (( CANARY )); then
  echo "=============================================================="
  echo " Canary: $IDEAS_JSON -> forth_trial.py --quick"
  echo "=============================================================="
  canary_dir="$LOG_DIR/canary"
  mkdir -p "$canary_dir/working"
  if ( cd "$canary_dir" && timeout 1800 python "${IDEAS_JSON%.json}.py" --quick ) \
       >"$LOG_DIR/canary.log" 2>&1; then
    if [[ -f "$canary_dir/working/experiment_data.npy" ]]; then
      ok "canary run + experiment_data.npy"
      grep -E "^(mean|best|\[warn\])" "$LOG_DIR/canary.log" | sed 's/^/    /'
    else
      bad "canary produced experiment_data.npy" "script exited 0 but wrote nothing"
    fi
  else
    bad "canary run" "seed code fails on this machine — see $LOG_DIR/canary.log"
    tail -20 "$LOG_DIR/canary.log" | sed 's/^/    /'
  fi
  if (( fail )); then
    echo
    echo "Canary FAILED — the tree search would inherit this. Nothing was launched."
    echo "Re-run just the canary with:  python ${IDEAS_JSON%.json}.py --quick"
    exit 1
  fi
  echo
fi

echo "Preflight passed. BASE_MODEL=$BASE_MODEL  CUDA_VISIBLE_DEVICES=$EXP_GPUS"
echo "  writeup/citation/summary: $WRITEUP_MODEL   review: $REVIEW_MODEL"
echo "  citation rounds: $CITE_ROUNDS"
echo

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1 — gates passed, nothing launched."
  exit 0
fi

# =============================================================================
#  Run
# =============================================================================
for i in $IDEAS; do
  echo "=============================================================="
  echo " Idea index $i   ->  $LOG_DIR/idea_$i.log"
  echo "=============================================================="
  start=$SECONDS

  python main.py \
    --load_ideas "$IDEAS_JSON" \
    --load_code \
    --idea_idx "$i" \
    --writeup_type icbinb \
    --model_writeup        "$WRITEUP_MODEL" \
    --model_writeup_small  "$WRITEUP_MODEL" \
    --model_citation       "$WRITEUP_MODEL" \
    --model_agg_plots      "$WRITEUP_MODEL" \
    --model_review         "$REVIEW_MODEL" \
    --num_cite_rounds "$CITE_ROUNDS" \
    2>&1 | tee "$LOG_DIR/idea_$i.log"

  rc=${PIPESTATUS[0]}
  echo "  idea $i finished rc=$rc in $((SECONDS - start))s"
done

echo
echo "All done. Logs: $LOG_DIR"
echo "Resume a partial run with:  python main.py --resume_from <experiments/...dir>"
