#!/usr/bin/env bash
# =============================================================================
#  serve_vlm_env.sh — start the Qwen2.5-VL-32B vLLM server in its own conda env
# =============================================================================
#
#  The serving stack and the experiment stack CANNOT share one conda env:
#
#      vllm 0.29.0  requires  pydantic>=2.12.0, safetensors>=0.6.2
#      mergekit 0.1.4 requires pydantic~=2.10.6, safetensors~=0.5.2
#
#  so `vllm_serve` serves the VLM and `forth_trial` runs the experiments.
#
#  Usage:
#      bash experiment_config/serve_vlm_env.sh              # foreground
#      setsid nohup bash experiment_config/serve_vlm_env.sh \
#          > logs/serve/vlm.log 2>&1 < /dev/null &          # background
#
#  Notes on the two non-obvious flags:
#    * --attention-backend TRITON_ATTN — the default FlashInfer backend JIT
#      compiles trtllm-gen FMHA kernels with nvcc. This node has no CUDA
#      toolkit (`/usr/local/cuda` does not exist), so that path dies at
#      cudagraph capture with "Could not find nvcc". Triton ships its own
#      ptxas and needs no toolkit.
#    * SERVE_MAX_MEM — vLLM otherwise claims ~90% of the card. GPU 3 is shared
#      with the experiment workers.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source /home/hhri-ai/hh30990/miniforge3/etc/profile.d/conda.sh
conda activate "${SERVE_ENV:-vllm_serve}"

export CUDA_VISIBLE_DEVICES="${SERVE_GPUS:-3}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export SERVE_MAX_MEM="${SERVE_MAX_MEM:-140GiB}"
export SERVE_EXTRA_ARGS="${SERVE_EXTRA_ARGS:---attention-backend TRITON_ATTN}"
# Second nvcc-dependent FlashInfer path: the top-k/top-p sampler
# (v1/sample/ops/topk_topp_sampler.py:flashinfer_sample). Same failure mode as
# the attention backend; fall back to the native torch sampler.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

if [[ -f "$ROOT/.env" ]]; then
  set -a; source "$ROOT/.env"; set +a
  [[ -n "${hk_token:-}" && -z "${HF_TOKEN:-}" ]] && export HF_TOKEN="$hk_token"
fi

exec python src/serve_vlm.py --port "${SERVE_PORT:-8000}"
