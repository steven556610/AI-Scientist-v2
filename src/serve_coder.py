#!/usr/bin/env python
"""vLLM-backed OpenAI-compatible server for Qwen3-Coder-30B-A3B-Instruct.

Replaces the previous FastAPI + transformers `model.generate()` implementation.
Everything the OpenAI protocol needs -- /v1/models, /v1/chat/completions,
streaming, `tools`/`tool_choice` -- is provided by vLLM's own entrypoint, so
this file is only argument translation plus an exec.

What that buys over the old server:
  * continuous batching, so the `_GEN_LOCK` that serialised every request is
    gone. The tree search's parallel workers no longer queue behind each other.
  * paged KV cache preallocated at startup. The old server sized nothing up
    front and OOMed at request time; here the budget is fixed by
    --gpu-memory-utilization and an oversized request is rejected, not fatal.
  * `tools`/`tool_choice` reach the model instead of being dropped by a
    Pydantic request model that had no field for them. Note that the client
    still refuses to send them: backend_openai.py's
    TOOL_CALL_UNSUPPORTED_PREFIXES lists "local/" and "local_coder/". Removing
    those prefixes is a separate change, but the server side is ready --
    add --enable-auto-tool-choice --tool-call-parser hermes via
    SERVE_EXTRA_ARGS if you make it.

usage:
    conda activate ai_scientist_v2
    CUDA_VISIBLE_DEVICES=1 SERVE_MAX_MEM=70GiB python src/serve_coder.py --port 8004

    # inspect the generated vLLM command without launching anything:
    SERVE_DRY_RUN=1 python src/serve_coder.py --port 8004

Environment (same contract as the transformers version, plus three new ones):

    CUDA_VISIBLE_DEVICES  which physical GPU(s). ALWAYS set it. The number of
                          visible devices becomes --tensor-parallel-size.
    SERVE_MAX_MEM         per-GPU cap, e.g. "70GiB". Translated to a
                          --gpu-memory-utilization fraction against the card's
                          total. Required in practice on a SHARED card: vLLM
                          otherwise claims 90% of the whole GPU at startup.
    SERVE_MAX_MODEL_LEN   context window, prompt + generation. Default 65536.
                          This is the setting that truncates generated code if
                          it is too small -- extract_code silently discards a
                          block that got cut mid-statement and does not
                          compile, which costs a whole tree-search retry.
    SERVE_MAX_NEW_TOKENS  legacy name from the transformers server. vLLM has no
                          server-side generation cap -- the client's max_tokens
                          governs -- so this only raises the floor on
                          --max-model-len. Prefer SERVE_MAX_MODEL_LEN.
    SERVE_DTYPE           default "bfloat16".
    SERVE_EXTRA_ARGS      raw extra flags appended verbatim, shell-quoted.
    SERVE_DRY_RUN         print the command and exit without launching.

Weights are ~61 GB in bf16 for this A3B MoE checkpoint (the old docstring's
"136 GB" was the dense-equivalent figure), plus the KV cache on top.
"""

import os
import shlex
import subprocess
import sys

MODEL_ID = "Qwen/Qwen3-Coder-30B-A3B-Instruct"

# The tree-search client strips the "local/" prefix before it calls
# (backend_openai.py:240), so it asks for "qwen3-coder-30b-a3b-instruct". vLLM
# validates the requested name against what it serves and 404s on a mismatch --
# unlike the old server, which ignored req.model entirely. Every alias in use
# must be listed or the pipeline breaks at the first coder call.
SERVED_NAMES = [
    "qwen3-coder-30b-a3b-instruct",
    "local/qwen3-coder-30b-a3b-instruct",
    "local_coder/qwen3-coder-30b-a3b-instruct",
    MODEL_ID,
]

DEFAULT_PORT = 8004
DEFAULT_MAX_MODEL_LEN = 65536

CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model"
)


def visible_gpu_count() -> int:
    """Number of GPUs this process may use, from CUDA_VISIBLE_DEVICES."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is None or cvd.strip() == "":
        return 0
    return len([d for d in cvd.split(",") if d.strip() != ""])


def parse_size_to_bytes(text: str) -> int:
    """Parse "70GiB" / "70GB" / "70000MiB" / "12345678" into bytes."""
    text = text.strip()
    units = {
        "kib": 2 ** 10, "mib": 2 ** 20, "gib": 2 ** 30, "tib": 2 ** 40,
        "kb": 10 ** 3, "mb": 10 ** 6, "gb": 10 ** 9, "tb": 10 ** 12,
        "k": 2 ** 10, "m": 2 ** 20, "g": 2 ** 30, "t": 2 ** 40,
        "b": 1, "": 1,
    }
    lowered = text.lower()
    for suffix in sorted(units, key=len, reverse=True):
        if suffix and lowered.endswith(suffix):
            return int(float(lowered[: -len(suffix)].strip()) * units[suffix])
    return int(float(lowered))


def total_gpu_bytes() -> int:
    """Total memory of the smallest visible GPU, or 0 if it cannot be read.

    Queried through nvidia-smi rather than torch: importing torch here would
    initialise CUDA in this process, and the memory it pins is memory vLLM then
    cannot have.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    cmd = ["nvidia-smi", "--query-gpu=memory.total",
           "--format=csv,noheader,nounits"]
    if cvd:
        cmd.append(f"--id={cvd}")
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30,
                             check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return 0
    mib = [int(line.strip()) for line in out.splitlines() if line.strip()]
    return min(mib) * 2 ** 20 if mib else 0


def gpu_memory_utilization() -> "float | None":
    """SERVE_MAX_MEM as a fraction of the card, or None to leave vLLM's default.

    vLLM takes a fraction, the rest of this repo speaks absolute sizes, and the
    cards are shared -- so the translation has to happen somewhere. Here.
    """
    raw = os.environ.get("SERVE_MAX_MEM")
    if not raw:
        return None
    total = total_gpu_bytes()
    if total <= 0:
        print("WARNING: SERVE_MAX_MEM is set but the GPU total could not be "
              "read from nvidia-smi; leaving vLLM's default utilisation. The "
              "server may claim more of the card than intended.",
              file=sys.stderr)
        return None
    frac = parse_size_to_bytes(raw) / total
    # Below ~0.05 vLLM cannot fit weights + a usable cache and fails with an
    # opaque error; above 0.95 it collides with the CUDA context and whatever
    # else shares the card.
    return round(min(max(frac, 0.05), 0.95), 4)


def build_argv(host: str, port: int, passthrough: "list[str]") -> "list[str]":
    n_gpu = visible_gpu_count()
    max_model_len = int(os.environ.get(
        "SERVE_MAX_MODEL_LEN",
        max(DEFAULT_MAX_MODEL_LEN,
            int(os.environ.get("SERVE_MAX_NEW_TOKENS", 0)) + 8192),
    ))

    argv = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_ID,
        "--served-model-name", *SERVED_NAMES,
        "--host", host,
        "--port", str(port),
        "--download-dir", CACHE_DIR,
        "--dtype", os.environ.get("SERVE_DTYPE", "bfloat16"),
        "--max-model-len", str(max_model_len),
        "--trust-remote-code",
    ]
    if n_gpu > 1:
        argv += ["--tensor-parallel-size", str(n_gpu)]
    util = gpu_memory_utilization()
    if util is not None:
        argv += ["--gpu-memory-utilization", str(util)]
    argv += shlex.split(os.environ.get("SERVE_EXTRA_ARGS", ""))
    argv += passthrough
    return argv


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Serve Qwen3-Coder-30B-A3B through vLLM's OpenAI API.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the vLLM command and exit")
    args, passthrough = parser.parse_known_args()

    if visible_gpu_count() == 0:
        print("WARNING: CUDA_VISIBLE_DEVICES is unset — vLLM will spread "
              "across every GPU on the node and collide with the experiments.",
              file=sys.stderr)
    if not os.environ.get("SERVE_MAX_MEM"):
        print("WARNING: SERVE_MAX_MEM is unset — vLLM will claim ~90% of the "
              "card at startup. Set it when sharing the GPU.", file=sys.stderr)

    os.makedirs(CACHE_DIR, exist_ok=True)
    argv = build_argv(args.host, args.port, passthrough)

    print("launching:", " ".join(shlex.quote(a) for a in argv), flush=True)
    print(f"serving {MODEL_ID} as {SERVED_NAMES} on {args.host}:{args.port}",
          flush=True)
    if args.dry_run or os.environ.get("SERVE_DRY_RUN"):
        return

    try:
        import vllm  # noqa: F401
    except ImportError:
        sys.exit("vllm is not installed in this environment. "
                 "Install it with:  pip install vllm")

    # exec, not subprocess: the server becomes this PID, so the process
    # management and signal handling the launch scripts already do keeps
    # working unchanged.
    os.execv(argv[0], argv)


if __name__ == "__main__":
    main()
