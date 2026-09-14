#!/usr/bin/env bash
# =============================================================================
#  third_trial_1,2.sh — ideas 1 and 2 of third_trial.json
# =============================================================================
#  Sibling of third_trial.sh, which runs idea 0
#  (taiwan_conflict_aware_lora_merging). This one runs the other two:
#
#      1  curriculum_level_hierarchical_merging
#           Merge Within Grade Level First: Hierarchical LoRA Fusion for
#           Junior- and Senior-High Subjects
#      2  taiwan_merge_operator_benchmark
#           A Reproducible Baseline: Merge-Operator and Density Sweep
#
#  All three ideas share ONE seed file, ai_scientist/ideas/third_trial/
#  third_trial.py — there is no per-idea "Code" field in the JSON, so
#  --load_code resolves to the same script for every index. That is why the
#  canary below is valid for ideas 1 and 2 even though it is idea 0's seed.
#
#  Usage:
#      conda activate ai_scientist_v2
#      bash experiment_config/third_trial_1,2.sh              # ideas 1 then 2
#      IDEAS=1 bash experiment_config/third_trial_1,2.sh      # just idea 1
#      DRY_RUN=1 bash experiment_config/third_trial_1,2.sh    # gates only
#
#  Model roles (measured, not assumed):
#      code / feedback / parse_metrics  local/qwen3-coder-30b-a3b-instruct :8004
#      vlm_feedback / review            local/qwen2.5-vl-32b               :8000
#      writeup / citation / summary     kimi-k3                            remote (.env)
#
#  ACTUAL GPU LAYOUT, verified from /proc/<pid>/environ on 2026-09-09:
#      serve_coder.py  CUDA_VISIBLE_DEVICES=3   ~75 GiB
#      serve_vlm.py    CUDA_VISIBLE_DEVICES=3   ~74 GiB
#      workers         CUDA_VISIBLE_DEVICES=3   -> ~39 GiB left on the card
#  Everything is on GPU 3. (third_trial.sh's header claims the coder is on
#  GPU 1; that comment is stale. Do not trust it, trust nvidia-smi.)
#
#  Both local servers must already be running:
#      CUDA_VISIBLE_DEVICES=3 SERVE_MAX_MEM=76GiB python src/serve_coder.py --port 8004
#      CUDA_VISIBLE_DEVICES=3 SERVE_MAX_MEM=74GiB python src/serve_vlm.py   --port 8000
# =============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

# ── Tunables ─────────────────────────────────────────────────────────────────
IDEAS="${IDEAS:-1 2}"                         # <- the only real difference
EXP_GPUS="${EXP_GPUS:-3}"
EXP_GPUS="$(echo "$EXP_GPUS" | tr -s ' ,' ',' | sed 's/^,//; s/,$//')"
CODER_URL="${CODER_URL:-http://localhost:8004/v1}"
VLM_URL="${VLM_URL:-http://localhost:8000/v1}"
WRITEUP_MODEL="${WRITEUP_MODEL:-kimi-k3}"
REVIEW_MODEL="${REVIEW_MODEL:-local/qwen2.5-vl-32b}"
BASE_MODEL="${BASE_MODEL:-google/gemma-3-1b-it}"
IDEAS_JSON="$ROOT/ai_scientist/ideas/third_trial/third_trial.json"
CANARY="${CANARY:-1}"
# Refuse to start alongside another main.py on the same card. Set to 1 only if
# you have deliberately made room. See gate 0.
ALLOW_CONCURRENT="${ALLOW_CONCURRENT:-0}"

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
export BASE_MODEL
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "$HF_HOME"
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-local-dummy}"
export HF_TOKEN="${HF_TOKEN:-}"
[[ -z "$HF_TOKEN" && -r "$HOME/.cache/huggingface/token" ]] && \
  export HF_TOKEN="$(<"$HOME/.cache/huggingface/token")"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/logs/third_trial_ideas12_$RUN_TS"
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
echo " Preflight  ideas [$IDEAS]  ($RUN_TS)"
echo "=============================================================="

# 0. Concurrency.
#    NOTE this is a PROCESS check, not a GPU check. They disagree, and that is
#    not a bug: when tree-search nodes run on CPU (e.g. the synthetic-numpy
#    nodes idea 0 produced) the run occupies 0 MiB of GPU, so `nvidia-smi`
#    shows only the two inference servers while main.py is very much alive.
#    The contended resource is the single coder server on :8004, not VRAM —
#    two tree searches queue on it and both crawl.
others=$(pgrep -f "main\.py --load_ideas" 2>/dev/null)
n_others=$(printf '%s' "$others" | grep -c . )
if (( n_others > 0 )); then
  detail=$(for p in $others; do
             printf '%s(idea %s) ' "$p" "$(ps -o cmd= -p "$p" 2>/dev/null | grep -o -- '--idea_idx [0-9]*' | awk '{print $2}')"
           done)
  if (( ALLOW_CONCURRENT )); then
    warn "concurrent main.py" "$n_others running [$detail] — ALLOW_CONCURRENT=1, proceeding anyway"
  else
    bad "no concurrent main.py" \
        "$n_others running [$detail] — this is a process check, NOT a GPU check; nvidia-smi can look idle while these are alive. Wait, kill them, or set ALLOW_CONCURRENT=1"
  fi
else
  ok "no concurrent main.py"
fi

# 1. conda env
if python -c "import torch, peft, transformers, datasets" 2>/dev/null; then
  ok "python deps (torch/peft/transformers/datasets)"
else
  bad "python deps" "activate ai_scientist_v2 first"
fi

# 2. HF_HOME must support flock ($HOME is NFS -> OSError 37 mid-download)
if python - <<'PY' 2>/dev/null
import fcntl, os, tempfile
fd = tempfile.NamedTemporaryFile(dir=os.environ["HF_HOME"], delete=False)
fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
fcntl.flock(fd.fileno(), fcntl.LOCK_UN); fd.close(); os.unlink(fd.name)
PY
then ok "HF_HOME flock-capable ($HF_HOME)"
else bad "HF_HOME flock-capable" "$HF_HOME is on NFS; set HF_HOME=/tmp/..."
fi

# 3. LLM servers reachable. 60 s, not 5 s: these servers run model.generate()
#    inline, so a probe issued mid-generation queues behind it (measured 43.6 s).
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-60}"
for pair in "coder:$CODER_URL" "vlm:$VLM_URL"; do
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

# 3b. Remote writeup model
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

# 3d. Anti-synthetic guard. The idea-0 run of 2026-09-08 produced 26 of 29 nodes
#     with _meta['base_model'] == 'synthetic': the agent replaced the real
#     gemma+TMMLU+ pipeline with a numpy simulation on node 1 and every
#     descendant inherited it. Node "goodness" is decided solely by whether
#     parse_metrics returns valid metrics, and a synthetic node is faster and
#     never fails, so the tree search actively prefers it. Nothing in the
#     framework detects this.
if grep -q 'base_model.*!=.*synthetic\|assert.*synthetic' "${IDEAS_JSON%.json}.py" 2>/dev/null; then
  ok "seed has anti-synthetic assertion"
else
  warn "seed has anti-synthetic assertion" \
       "absent — ideas 1/2 may drift to simulated data like idea 0 did. Add to third_trial.py: assert BASE_MODEL != 'synthetic' and a non-empty check on load_dataset(). SYNTH_AUDIT below reports it after the fact."
fi

# 4. GPU headroom. Sharing with capped servers is the intended layout; what
#    matters is that enough is LEFT, not that the card is empty.
MIN_FREE_MIB="${MIN_FREE_MIB:-24576}"   # ~24 GiB: gemma-3-1b train+merge+eval
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
        bad "GPU $g free memory" "only ${avail} MiB free of ${total} (need ${MIN_FREE_MIB}) — stop the idea-0 run or lower SERVE_MAX_MEM"
      elif (( used > total / 4 )); then
        warn "GPU $g shared" "${used}/${total} MiB in use, ${avail} MiB free — sharing with the inference servers"
      else
        ok "GPU $g free (${avail}/${total} MiB available)"
      fi
    fi
  done
else
  warn "nvidia-smi" "not available; cannot verify GPU pinning"
fi

# 5. Prefetch base model + TMMLU+ ONCE, serially. Four workers pulling the same
#    repo concurrently is the other half of the filelock/timeout problem.
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
        "gated repo. Accept the licence at https://huggingface.co/$BASE_MODEL then \`huggingface-cli login\` (or export HF_TOKEN)" ;;
  *) bad "base model + TMMLU+ cached" "see $LOG_DIR/preflight.log" ;;
esac

# 5b. Worker/GPU sanity. ParallelAgent clamps num_workers to the number of
#     VISIBLE GPUs, so the pinning above also decides tree-search parallelism.
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
    if (( i < ${n:-0} )); then
      nm=$(python -c "import json;print(json.load(open('$IDEAS_JSON'))[$i]['Name'])" 2>/dev/null)
      ok "  idea $i = $nm"
    else
      bad "idea index $i in range" "JSON only has ${n:-0}"
    fi
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
#  Canary: run the shared seed on this GPU before spending a tree search.
#  All three ideas inject third_trial.py verbatim, so a broken seed burns every
#  node of every idea. Fail in ~3 minutes instead of ~13 hours.
# =============================================================================
if (( CANARY )); then
  echo "=============================================================="
  echo " Canary: third_trial.py --quick  (shared by ideas 1 and 2)"
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
echo "  ideas: $IDEAS"
echo "  writeup/citation/summary: $WRITEUP_MODEL   review: $REVIEW_MODEL"
echo

# DRY_RUN=1 validates the gates and stops. Note IDEAS="" does NOT do this:
# ${IDEAS:-1 2} substitutes on empty as well as unset.
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1 — gates passed, nothing launched."
  exit 0
fi

# =============================================================================
#  Run
# =============================================================================
SYNTH_AUDIT="${SYNTH_AUDIT:-1}"

audit_synthetic() {
  # $1 = idea index. Scans every node's experiment_data.npy for the run that
  # this idea just produced and reports how many used real weights. A run that
  # is 100% synthetic is not a result, however green the log looks.
  local idx="$1" name
  name=$(python -c "import json;print(json.load(open('$IDEAS_JSON'))[$idx]['Name'])" 2>/dev/null) || return 0
  local dir
  dir=$(ls -dt "$ROOT/experiments/$name"/*/ 2>/dev/null | head -1) || return 0
  [[ -n "$dir" ]] || return 0
  python - "$dir" "$BASE_MODEL" <<'PY'
import glob, sys
import numpy as np
run_dir, want = sys.argv[1], sys.argv[2]
paths = glob.glob(f"{run_dir}/logs/0-run/experiment_results/*/experiment_data.npy")
if not paths:
    print("  [audit] no experiment_data.npy found yet")
    raise SystemExit
real = synth = unknown = 0
for p in paths:
    try:
        bm = np.load(p, allow_pickle=True).item().get("_meta", {}).get("base_model")
    except Exception:
        unknown += 1; continue
    if bm == want:      real += 1
    elif bm is None:    unknown += 1
    else:               synth += 1
print(f"  [audit] {len(paths)} nodes: {real} real ({want}), {synth} other/synthetic, {unknown} unknown")
if real == 0:
    print("  [audit] *** NO NODE USED THE REAL MODEL — these results are simulated, do not write them up ***")
PY
}

for i in $IDEAS; do
  echo "=============================================================="
  echo " Idea index $i   ->  $LOG_DIR/idea_$i.log"
  echo "=============================================================="
  start=$SECONDS

  # --model_writeup_small is REQUIRED: its default is gpt-4o, which would send
  # the whole paper to OpenAI (and fail without a real key).
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
  (( SYNTH_AUDIT )) && audit_synthetic "$i"
  # Do not abort the batch — a failed idea should not block the other.
done

echo
echo "All done. Logs: $LOG_DIR"
echo "Results:  experiments/<idea_name>/<timestamp>_attempt_0/logs/0-run/"
echo "Resume a partial run with:  python main.py --resume_from <experiments/...dir>"
