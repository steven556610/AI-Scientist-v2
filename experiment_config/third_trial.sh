#!/usr/bin/env bash
# =============================================================================
#  third_trial.sh — Taiwan curriculum LoRA-merging experiments
# =============================================================================
#  Fixes the failure modes seen in experiments/gemma_ties_merging_taiwan_curriculum:
#    * every worker landed on GPU 0 together with the Qwen servers  -> OOM at 184 GiB
#    * HF hub filelock on the NFS $HOME                             -> OSError 37
#    * --model_writeup_small unset                                  -> silently calls OpenAI
#    * 4 workers x 1 full base model on one GPU                     -> OOM
#    * models/datasets downloaded concurrently by 4 workers         -> races + timeouts
#
#  Usage:
#      conda activate ai_scientist_v2
#      bash experiment_config/third_trial.sh              # all ideas
#      IDEAS="0" bash experiment_config/third_trial.sh    # just the flagship idea
#      BASE_MODEL=google/gemma-2-27b EXP_GPUS=1 bash experiment_config/third_trial.sh
#
#  Model roles:
#      code / feedback / parse_metrics  local/qwen3-coder-30b-a3b-instruct  :8004  GPU 1
#      vlm_feedback / review            local/qwen2.5-vl-32b                :8000  GPU 3
#      writeup / citation / summary     kimi-k3                             remote (.env)
#  kimi-k3 replaces local/qwen-72b, whose server (port 8002) no longer exists.
#
#  Both local servers must already be running, pinned and capped:
#      CUDA_VISIBLE_DEVICES=1 SERVE_MAX_MEM=70GiB python src/serve_coder.py --port 8004
#      CUDA_VISIBLE_DEVICES=3 SERVE_MAX_MEM=72GiB python src/serve_vlm.py   --port 8000
#  All experiment workers run on GPU 3, sharing it with the capped VLM
#  (~110 GiB free there; gemma-3-1b needs ~8 GiB).
# =============================================================================
set -uo pipefail

# ── Repo root (works both here and at /workspace) ────────────────────────────
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

# ── Tunables ─────────────────────────────────────────────────────────────────
IDEAS="${IDEAS:-0 1 2}"                       # idea indices in third_trial.json
# GPU(s) for experiment workers ONLY. Everything runs on GPU 3: the coder sits
# on GPU 1 and the VLM is capped on GPU 3, leaving ~110 GiB there. Accepts "3",
# "1,3" or "1 3" — normalized to the comma form CUDA_VISIBLE_DEVICES expects.
EXP_GPUS="${EXP_GPUS:-3}"
EXP_GPUS="$(echo "$EXP_GPUS" | tr -s ' ,' ',' | sed 's/^,//; s/,$//')"
CODER_URL="${CODER_URL:-http://localhost:8004/v1}"
VLM_URL="${VLM_URL:-http://localhost:8000/v1}"
# Writeup / citation / plot-aggregation / summary model. Was local/qwen-72b on
# port 8002; that server is gone. kimi-k3 is the remote vLLM in .env.
WRITEUP_MODEL="${WRITEUP_MODEL:-kimi-k3}"
REVIEW_MODEL="${REVIEW_MODEL:-local/qwen2.5-vl-32b}"
# Gated, but the token in .env unlocks it (verified). Open alternative needing
# no token: Qwen/Qwen2.5-1.5B-Instruct.
BASE_MODEL="${BASE_MODEL:-google/gemma-3-1b-it}"
IDEAS_JSON="$ROOT/ai_scientist/ideas/third_trial/third_trial.json"
CANARY="${CANARY:-1}"                         # CANARY=0 to skip the seed-code check

# ── Secrets from .env ────────────────────────────────────────────────────────
# Holds S2_API_KEY (Semantic Scholar, used by Stage 3 citations) and the
# HuggingFace token under the non-standard name `hk_token`.
if [[ -f "$ROOT/.env" ]]; then
  set -a; source "$ROOT/.env"; set +a
  [[ -n "${hk_token:-}" && -z "${HF_TOKEN:-}" ]] && export HF_TOKEN="$hk_token"
fi

# ── Environment ──────────────────────────────────────────────────────────────
# HF_HOME MUST be on a filesystem that supports flock(). $HOME is NFS here and
# raises "OSError: [Errno 37] No locks available" mid-download.
export HF_HOME="${HF_HOME:-/tmp/hf_cache_$USER}"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="$EXP_GPUS"
export BASE_MODEL                              # read by third_trial.py
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "$HF_HOME"

# The framework's OpenAI client is constructed unconditionally; give it a
# non-empty dummy so it never crashes when only local/ models are used.
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-local-dummy}"

# Every google/gemma-* repo is gated. Without a token that accepted the Gemma
# licence, from_pretrained() raises 401 *inside* a tree-search node, which the
# agent then "fixes" by inventing fake data. Fail here instead.
#   huggingface-cli login          (writes ~/.cache/huggingface/token)
#   or: export HF_TOKEN=hf_xxxxx   before running this script
export HF_TOKEN="${HF_TOKEN:-}"
[[ -z "$HF_TOKEN" && -r "$HOME/.cache/huggingface/token" ]] && \
  export HF_TOKEN="$(<"$HOME/.cache/huggingface/token")"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/logs/third_trial_$RUN_TS"
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
# The 60 s timeout is not paranoia: these servers run model.generate() inline,
# so a probe issued while another request is generating waits behind it. A 5 s
# probe reported a perfectly healthy coder as down (measured: 43.6 s).
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-60}"
for pair in "coder:$CODER_URL" "vlm:$VLM_URL"; do
  name="${pair%%:*}"; url="${pair#*:}"
  code=$(curl -s -m "$HEALTH_TIMEOUT" -o /dev/null -w "%{http_code}" "$url/models" 2>/dev/null)
  [[ "$code" == "000" ]] && \
    code=$(curl -s -m "$HEALTH_TIMEOUT" -o /dev/null -w "%{http_code}" "${url%/v1}/docs" 2>/dev/null)
  # Any HTTP status means a server answered; only 000 (connect fail/timeout) is down.
  if [[ "$code" != "000" && -n "$code" ]]; then
    ok "$name server reachable ($url, HTTP $code)"
  else
    bad "$name server reachable" "no response from $url within ${HEALTH_TIMEOUT}s — start src/serve_${name}.py"
  fi
done

# 3b. Remote writeup model. Stages 3-4 (writeup, citations, review) call this
#     for hours; if the endpoint or the key is wrong we want to know now, not
#     after the tree search has finished.
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

# 3c. Nothing may still point at port 8002 (the retired qwen-72b server).
if grep -rn "local/qwen-72b" bfts_config.yaml >/dev/null 2>&1; then
  bad "bfts_config.yaml model ids" "still references local/qwen-72b (port 8002 is dead) — use kimi-k3"
else
  ok "bfts_config.yaml model ids"
fi

# 4. GPU headroom. Sharing a card with a *capped* server is fine — what matters
#    is that enough is left for a worker, not that the card is empty. A capped
#    VLM (SERVE_MAX_MEM=72GiB) on the experiment GPU is the intended layout.
MIN_FREE_MIB="${MIN_FREE_MIB:-24576}"   # ~24 GiB: gemma-3-1b train+merge+eval
if command -v nvidia-smi >/dev/null 2>&1; then
  for g in ${EXP_GPUS//,/ }; do
    # nvidia-smi prints its failure banner on stdout, so validate before doing
    # arithmetic on the result.
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
#    Four tree-search workers downloading the same repo concurrently is the
#    other half of the filelock/timeout problem.
# 5a. Reuse repos already downloaded to the NFS caches. We cannot point HF_HOME
#     at them (no flock), but symlinking the individual repo dirs into the
#     flock-capable HF_HOME gives us the blobs without a re-download.
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
        "gated repo. Accept the licence at https://huggingface.co/$BASE_MODEL then \`huggingface-cli login\` (or export HF_TOKEN). Cached-and-usable right now: BASE_MODEL=google/gemma-2-27b" ;;
  *) bad "base model + TMMLU+ cached" "see $LOG_DIR/preflight.log" ;;
esac

# 5b. Worker/GPU sanity. ParallelAgent clamps num_workers to the number of
#     *visible* GPUs and gives each worker its own card, so the pinning below
#     also decides how much of the tree search runs in parallel.
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
[[ -f "${IDEAS_JSON%.json}.py" ]] && ok "code template" || bad "code template" "third_trial.py missing"

echo
if (( fail )); then
  echo "Preflight FAILED — fix the above before running. Nothing was launched."
  exit 1
fi

# =============================================================================
#  Canary: run the seed code itself, on this GPU, before spending a tree search.
#
#  third_trial.py is injected verbatim into every node as "Code To Potentially
#  Use". If it cannot run here, all 20 nodes inherit the breakage -- which is
#  exactly how the last run burned 20/20 nodes. Fail in ~3 minutes instead.
# =============================================================================
if (( CANARY )); then
  echo "=============================================================="
  echo " Canary: $IDEAS_JSON -> third_trial.py --quick"
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
echo

# DRY_RUN=1 validates the gates and stops. Note IDEAS="" does NOT do this:
# ${IDEAS:-0 1 2} substitutes on empty as well as unset.
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

  # --model_writeup_small is REQUIRED: its default is gpt-4o, which would send
  # the whole paper to OpenAI (and fail without a real key).
  #
  # WRITEUP_MODEL is kimi-k3 (remote vLLM, see .env). It replaces local/qwen-72b,
  # which pointed at port 8002 — nothing serves that port any more. kimi-k3 also
  # does real OpenAI tool calls (verified), so the writeup/citation/plot-selection
  # function specs get structured output instead of prompt-JSON parsing.
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
    --num_cite_rounds 10 \
    2>&1 | tee "$LOG_DIR/idea_$i.log"

  rc=${PIPESTATUS[0]}
  echo "  idea $i finished rc=$rc in $((SECONDS - start))s"
  # Do not abort the batch — a failed idea should not block the others.
done

echo
echo "All done. Logs: $LOG_DIR"
echo "Resume a partial run with:  python main.py --resume_from <experiments/...dir>"
