# AI-Scientist-v2 — Taiwan curriculum model-merging run: diagnosis and fixes

Goal: let `main.py` run **experiment → plots → writeup → review** unattended, so a
researcher does not have to babysit it.

Every previous run died in Stage 1 and never reached writeup or review. This
document records why, what was changed, and what you still have to set.

---

## 1. Why every run failed

### 1.1 The DARE run (`experiments/gemma_dare_merging_taiwan_curriculum`)

Stage 1 ran all 20 iterations, **all 20 nodes buggy, 0 nodes with a metric**, then:

```
Initial stage ... did not find a working implementation after 20 iterations
No best node found ... something went wrong so finishing the experiment
NO GOOD LEAF NODES!!!
ValueError: not enough values to unpack (expected 4, got 1)
```

Node-by-node exception histogram:

| count | exception |
|---|---|
| 14 | `torch.OutOfMemoryError` |
| 3 | `ValueError` (empty batch / `input_ids` unset) |
| 2 | `TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'evaluation_strategy'` / `'logging_...'` |
| 1 | `NameError: name 'optimizer' is not defined` |

Every node failed in **12–32 seconds**. The whole "experiment" was ~7 minutes of
compute. The generated code (`best_solution_*.py`) shows the two proximate causes:

```python
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)   # no dtype= -> fp32
...
def generate_synthetic_data_qwen(subject, num_samples=100):
    """Placeholder function to generate synthetic data."""
    return [{"instruction": f"Example question for {subject}",
             "output":      f"Example answer for {subject}"}] * num_samples
```

* **fp32.** A 27B model in fp32 is ~108 GB. Two workers on the same 184 GiB card
  → `Tried to allocate 648.00 MiB ... 123.69 MiB is free. Process X has 91.77 GiB`.
* **Fake data.** The agent copied the placeholder from the old code template
  verbatim. Even the nodes that *ran* would have measured nothing.

### 1.2 Full problem list (both runs)

| # | Problem | Where it came from | Status |
|---|---|---|---|
| 1 | Workers land on the GPUs hosting the Qwen servers → OOM | `get_gpu_count()` asked `nvidia-smi` **before** `CUDA_VISIBLE_DEVICES`, so shell pinning was ignored and workers re-exported absolute device ids | **fixed** (§2.1) |
| 2 | fp32 model loads | code template had no `dtype=` | **fixed** (§2.5) |
| 3 | `parse_metrics` always returned the *plot-analysis* dict → every node metric `None` → tree search cannot rank → "No best node found" | `get_fallback_for_spec` `else:` branch | **fixed** (§2.2) |
| 4 | `Error in LLM selection process: 'selected_id'` | same fallback, no `select_best_implementation` branch | **fixed** (§2.2) |
| 5 | Local servers silently drop `tools`/`tool_choice`, so **no** function call ever succeeded | `src/serve_coder.py` request model has no `tools` field | **fixed** (§2.2) |
| 6 | `ValueError: not enough values to unpack (expected 4, got 1)` kills the writeup when only Stage 1 produced a journal | `log_summarization.py` | **fixed** (§2.3) |
| 7 | Agent imports `trl` / `bitsandbytes` → `ModuleNotFoundError` | prompt literally said *"all packages are already installed!"* | **fixed** (§2.4) |
| 8 | transformers 5.x API (`evaluation_strategy`, `torch_dtype`, `gradient_checkpointing`) | agent wrote 4.x-era code | **fixed** (§2.4) |
| 9 | `OSError: [Errno 37] No locks available` — HF filelock on the NFS `$HOME` | cluster mount | **fixed** (§2.6) |
| 10 | `401 gated repo` for every `google/gemma-*`; agent then fell back to a random cached model → `Unrecognized configuration class T5Config` | no `HF_TOKEN` | **fixed — but needs your action** (§3) |
| 11 | `--model_writeup_small` unset → defaults to `gpt-4o`, i.e. Stage 3 calls OpenAI | `third_trial.sh` | **fixed** (§2.6) |
| 12 | Ideation produced 5 generic non-Taiwan ideas | `third_trial.md` never mentioned Taiwan or Gemma | **fixed** (§2.7) |

---

## 2. What was changed

### 2.1 `ai_scientist/treesearch/parallel_agent.py` — GPU allocation

This was the OOM. Old `get_gpu_count()` called `nvidia-smi` first, which reports
**every** GPU on the node regardless of `CUDA_VISIBLE_DEVICES`. `GPUManager` then
handed workers ids from `range(num_gpus)`, and each worker did
`os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)` — an *absolute* index that
overwrote your pinning. Pinning to GPU 2 still put workers on GPU 0.

* New `visible_gpu_ids()`: `CUDA_VISIBLE_DEVICES` is authoritative; `nvidia-smi`
  only as fallback when it is unset.
* `GPUManager` now allocates from those real ids (`{"2","3"}`, not `{0,1}`).
* `min()` ordered numerically so `"10"` doesn't sort before `"2"`.

Verified: `CUDA_VISIBLE_DEVICES=2,3` now hands out `['2','3']`;
`4,5,6,7` → `['4','5','6','7']`; over-allocation still raises.

Consequence to know: `ParallelAgent` clamps `num_workers` to the number of
**visible** GPUs. Pin 1 GPU → 1 parallel node. Pin 4 → 4. This is now honest;
before, it claimed 4 workers and stacked them on one card.

### 2.2 `ai_scientist/treesearch/backend/backend_openai.py` — function calling

`src/serve_coder.py`'s `ChatCompletionRequest` has no `tools` field, so Pydantic
drops it. **No local call has ever produced a `tool_calls` response** — every one
fell through to text parsing, and on failure got this:

```python
else:   # <- parse_metrics, select_best_implementation, generate_stage_config, ...
    return {"plot_analyses": [], "valid_plots_received": True, ...}
```

So `parse_metrics` returned a plot dict → metric `None` on every node → no best
node → no Stage 2 → no writeup. That single `else:` explains most of the run.

Changes:

* `_supports_tool_calls()` — for `local/`, `local_coder/`, `ollama/` the JSON
  Schema is injected into the system prompt instead of sending `tools`.
* `_extract_first_json_object()` — brace-balanced scan that respects strings, so
  prose-wrapped JSON is recovered. Handles bare / fenced / inline / nested /
  `"a } brace"` / junk-prefixed cases.
* `get_fallback_for_spec()` — a correct branch for **all 9** FunctionSpecs
  (`submit_review`, `analyze_experiment_plots`, `parse_metrics`, `select_plots`,
  `select_best_implementation`, `generate_stage_config`,
  `evaluate_stage_progression`, `evaluate_stage_completion`,
  `generate_substage_goals`), plus a schema-derived default for unknown specs.
* Missing `required` keys are back-filled and logged instead of `KeyError`.
* `usage` / `system_fingerprint` read defensively (the local server omits them).

Two fallbacks deliberately changed meaning:

* `submit_review` → `is_bug=True` (was `False`). An unparseable review must not
  read as "the code ran fine".
* `select_best_implementation` → `selected_id=""`, which makes the caller use its
  own metric-based ranking rather than dereference a hallucinated id.

Verified: all 9 specs produce every required key with the right JSON type;
`parse_metrics` now returns `{'valid_metrics_received': False, 'metric_names': []}`;
real OpenAI models still get `tools` + `tool_choice` and no prompt injection.

### 2.3 `ai_scientist/treesearch/log_summarization.py` — partial runs

* Pads to 4 stages with typed empties (`{}, {}, {}, []`) instead of raising
  `not enough values to unpack`. A run that only completed Stage 1 can now still
  be written up.
* `process_stage` handles `journal.get_best_node()` returning `None` instead of
  crashing with `AttributeError` on `best_node.children`.

### 2.4 `ai_scientist/treesearch/parallel_agent.py` — environment prompt

Old prompt: *"Feel free to use any other packages too (all packages are already
installed!)"* — false, and the cause of the `trl` / `bitsandbytes` failures. It
also never mentioned `transformers`, `peft`, or `datasets`, let alone versions.

Now introspects the real environment and emits, e.g.:

```
These packages are installed: `torch==2.11.0+cu128`, `transformers==5.16.1`,
`peft==0.20.0`, `datasets==5.0.1`, `accelerate==1.14.0`, ...
NOT installed, and you must NOT import them (there is no way to install anything
at runtime): `bitsandbytes`, `deepspeed`, `einops`, `evaluate`, `flash_attn`,
`lightgbm`, `sentencepiece`, `statsmodels`, `timm`, `trl`, `vllm`, `xgboost`.
transformers is 5.16.1 (v5 API). `TrainingArguments(evaluation_strategy=...)` is
gone — use `eval_strategy`. `from_pretrained(torch_dtype=...)` is now `dtype=...`.
`gradient_checkpointing` is NOT a `from_pretrained` kwarg; call
`model.gradient_checkpointing_enable()` instead.
```

### 2.5 `ai_scientist/ideas/third_trial/third_trial.py` — rewritten (571 lines)

This file is injected verbatim into every tree-search node as *"Code To
Potentially Use"*. The old one was 100% placeholders, which is precisely what the
agent copied. It now runs end to end with no placeholders:

* One switchable `BASE_MODEL = os.environ.get("BASE_MODEL", "google/gemma-3-1b-it")`.
* `dtype=bfloat16`, `device_map={"": 0}`. Never fp32, never `device_map="auto"`.
* Real data: **TMMLU+** (`ikala/tmmluplus`), 11 Taiwan-curriculum subsets,
  2601 test rows, grouped into 4 domains (`chinese` / `math` / `taiwan` / `science`).
* Scoring: A/B/C/D logits at the final position — one forward pass, no
  generation, no LLM judge.
* Plain `peft` + hand-written torch AdamW loop (no `Trainer`, no `trl`).
* Merge operators implemented directly: task arithmetic, TIES (trim → elect sign
  → disjoint mean), DARE-TIES, plus a pairwise **sign-conflict rate** diagnostic.
* `ApplyDeltas` context manager restores base weights bit-exactly on exit
  (verified round-trip across multiple adapters).
* Writes `working/experiment_data.npy` in the schema the harness expects, plus
  `accuracy_by_method.png`, `sign_conflict.png`, `val_loss.png`, `results.json`.
* Degenerate-run guard: domains whose solo accuracy is below
  `RETENTION_FLOOR = 0.30` are reported as `NaN` retention, excluded from the
  mean, and named in a `[warn]` line — so a broken run cannot print
  "retention = 1.0000" and look like a success. `mean_accuracy` is also emitted
  as the absolute number tree search should rank on.

Smoke-tested end to end with a tiny stand-in model: completes in 4 s and produces
all five output files.

### 2.6 `experiment_config/third_trial.sh` — rewritten

* **Preflight that refuses to launch** if anything is wrong — deps, flock, both
  servers, GPU occupancy, model access, dataset, ideas JSON, code template.
* `HF_HOME=/tmp/hf_cache_$USER` (flock-capable). Repos already in
  `~/.cache/huggingface/hub` and `./model` are **symlinked** in, so nothing is
  re-downloaded — the NFS home has the blobs but no locks, `/tmp` has locks but
  no blobs.
* Model + all 11 TMMLU+ subsets prefetched **serially, once**, before any worker
  starts (4 workers racing on the same repo was the other half of the lock problem).
* `CUDA_VISIBLE_DEVICES=$EXP_GPUS`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
  `TOKENIZERS_PARALLELISM=false`, `HF_TOKEN` picked up from
  `~/.cache/huggingface/token` if not exported.
* **`--model_writeup_small local/qwen-72b`** — was unset, so Stage 3 defaulted to
  `gpt-4o-2024-05-13` and would have called OpenAI.
* `--model_agg_plots local/qwen-72b` — plot aggregation *writes Python*
  (`perform_plotting.py` uses `get_response_from_llm`, no images), so the coder
  model is right; it was pointed at the VLM.
* Loop is `IDEAS="0 1 2"` (was `{0..1}`), overridable, and one failing idea no
  longer aborts the batch. Per-idea logs under `logs/third_trial_<ts>/`.

### 2.7 `ai_scientist/ideas/third_trial/third_trial.md` and `.json`

**Which ideation output was good: none of the five.** They were generic merging
ideas with no Taiwan content, drifting toward LoRA *efficiency* rather than
*merging*. The reason is upstream — `third_trial.md`, the `--workshop-file` seed
for `perform_ideation_temp_free.py`, never mentioned Taiwan, Gemma, or TMMLU+, so
ideation had nothing local to work with.

`third_trial.md` rewritten with the motivation (臺灣史地/社會 is sparse in
pretraining and contradicted by mainland-Chinese web text, so its delta moves
furthest and collides with 國文, while 數學 is near-orthogonal), the
domain→TMMLU+ subset table, and a hard **Compute Constraints** section.

`third_trial.json` rewritten from 2 broken ideas to 3, each with the environment
constraints embedded as explicit instructions:

| idx | idea | role |
|---|---|---|
| 0 | `taiwan_conflict_aware_lora_merging` — interference is anisotropic; allocate merge coefficients by measured sign conflict | flagship. Predicts Spearman ρ < −0.7 between conflict and retention; success = taiwan retention +≥3 pts with math −≤1 pt; permuted-coefficient control must not reproduce it |
| 1 | `curriculum_level_hierarchical_merging` — merge within 國中/高中 first, then across; 6 adapters | second angle, includes a random-grouping control |
| 2 | `taiwan_merge_operator_benchmark` — operator + density/DARE-*p* sweep with delta-redundancy measurement | safety net; will produce a result even if 0 and 1 find nothing |

---

## 3. The recipe

The design goal is that **one command either produces a paper or refuses to
start**, with no silent middle state. Three gates, cheapest first:

```
  preflight (seconds)  ->  canary (~3 min)  ->  tree search (hours)
  deps, flock, servers     seed code runs       20 nodes x 3 ideas
  GPUs, token, dataset     on THIS gpu, real    -> plots -> writeup
  ideas JSON, code         data, .npy written   -> review
```

Gate 2 is the one that was missing. `third_trial.py` is injected verbatim into
every node as *"Code To Potentially Use"*; if it cannot run, all 20 nodes inherit
the breakage — which is exactly how the last run burned 20/20. The canary runs it
standalone with `--quick` (32 rows/domain, 1 epoch) and refuses to launch the
search unless `experiment_data.npy` actually appears.

### Run it

```bash
conda activate ai_scientist_v2

# GPU 1 — coder (Qwen3-Coder-30B-A3B), port 8004
CUDA_VISIBLE_DEVICES=1 SERVE_MAX_MEM=70GiB python src/serve_coder.py --port 8004

# GPU 3 — VLM, capped so the experiments can share the card
CUDA_VISIBLE_DEVICES=3 SERVE_MAX_MEM=72GiB python src/serve_vlm.py --port 8000

# GPU 3 — everything else (gemma-3-1b needs ~8 GiB; ~110 GiB free after the cap)
DRY_RUN=1 bash experiment_config/third_trial.sh   # validate the gates only
bash experiment_config/third_trial.sh             # for real
```

Defaults are now `EXP_GPUS=3`, `BASE_MODEL=google/gemma-3-1b-it`, `IDEAS="0 1 2"`,
`WRITEUP_MODEL=kimi-k3`.

> `IDEAS=""` does **not** mean "no ideas" — `${IDEAS:-0 1 2}` substitutes on empty
> as well as unset, so it launches the full batch. Use `DRY_RUN=1`.

### 2.9 Model roles after the kimi-k3 switch

`local/qwen-72b` pointed at port **8002**, and nothing serves that port any more —
`serve_coder.py` now hosts Qwen3-Coder-30B-A3B on **8004**. Every qwen-72b call
was hitting a dead socket. It is replaced by `kimi-k3`, the remote vLLM in `.env`.

| role | model | where | why |
|---|---|---|---|
| `agent.code` | `local/qwen3-coder-30b-a3b-instruct` | :8004, GPU 1 | highest token volume (12k/call); needs to be fast and local |
| `agent.feedback` — incl. **`parse_metrics`** and both stage-progression specs | `local/qwen3-coder-30b-a3b-instruct` | :8004, GPU 1 | ~2 calls/node; measured 3/3 valid `parse_metrics`, ~40 s |
| `agent.vlm_feedback`, `--model_review` | `local/qwen2.5-vl-32b` | :8000, GPU 3 | only model that takes images |
| `report`, `agent.summary`, `agent.select_node`, `--model_writeup{,_small}`, `--model_citation`, `--model_agg_plots` | **`kimi-k3`** | remote, `.env` | was qwen-72b; low call count, long outputs, and it does *real* tool calls |

Measured on the live endpoint before switching:

* `/models` lists `kimi-k3`, vLLM, 1 M context.
* **Real OpenAI tool calls work** — returned a correct `parse_metrics` tool_call
  with `tool_choice` forcing. It is a reasoning model: `message.reasoning` is
  filled first, so a small `max_tokens` truncates *before* the tool call appears
  (`finish_reason: length`, empty `content`, no `tool_calls`).
* **Throughput is erratic: 0.4 – 12.7 tok/s** across four samples; one 140-token
  reply took 255 s. `get_ai_client` now passes `timeout=3600` for kimi
  (`KIMI_TIMEOUT` to override) — the openai default of 600 s would have aborted
  long writeup calls.

This variance is the reason the high-volume roles stayed local. Stage 3 writeup
will be slow; that is throughput, not breakage.

Two preflight gates were added for this: 3b verifies `KIMI_API_BASE`/`KIMI_API_KEY`
and that `/models` actually lists `kimi-k3`; 3c fails the run if anything still
references `local/qwen-72b`.

### 2.10 `backend_openai.py` — `dict + str` crash

`query()` did `system_message = (system_message or "") + _schema_instruction(...)`.
Callers inside the tree search reach this through `backend/__init__.query`, which
compiles dicts to markdown first — but a direct call with a dict prompt raised
`TypeError: unsupported operand type(s) for +: 'dict' and 'str'`. Now coerced with
`compile_prompt_to_md` first.

### 2.11 `parallel_agent.plan_and_code_query` — code-only replies were thrown away

`if code and nl_text:` required *prose before the fence*.
`extract_text_up_to_code` returns `""` when a reply starts directly with
```` ```python ````, which is what a coder model does most of the time. The good
code was rejected, three retries burned, and then `return "", completion_text`
handed back the **raw markdown as the node's code** — which cannot execute, so the
node was guaranteed buggy. Observed live: `Plan + code extraction failed,
retrying...` on the very first node of a run.

Now, if `code` parsed but `nl_text` is empty, the code is accepted with a
synthesised plan string. Patched at both call sites (there are two identical
copies of this method in the file).

### 2.12 Both servers blocked their own health checks

`async def chat_completions` runs on uvicorn's event loop, and `model.generate()`
is blocking — so while one generation is in flight the server answers *nothing*,
health probes included. Measured: `/v1/models` took **43.6 s** on a healthy coder,
and the 5 s preflight probe declared it dead.

* Both handlers are now plain `def`, so FastAPI runs them in a threadpool.
* A `threading.Lock` serialises `model.generate()` — the threadpool must not let
  two generations allocate KV caches on the same card at once.
* Both servers expose a cheap lock-free `GET /v1/models`.
* Preflight probe timeout 5 s → `HEALTH_TIMEOUT` (default 60 s), and *any* HTTP
  status now counts as alive; only curl code `000` is down.

### 2.13 The flagship metric moved into the seed code

Idea 0 stands or falls on Spearman ρ between a domain's mean sign conflict and
its retention — but `third_trial.py` never computed it, so each node reinvented
it. One live node did exactly that and failed with
`name 'spearmanr' is not defined` (scipy 1.17.1 *is* installed; it just forgot the
import). Section 5b of the seed now computes `mean_conflict_per_domain` and
`conflict_retention_spearman` for all three merge methods and prints them, so
every node inherits one correct implementation.

Caveat seen in the `--quick` canary: `reportable` collapsed to 1 domain (32 eval
rows at 1B is near chance, so most domains fall under `RETENTION_FLOOR`), giving
`rho = nan, n = 1`, and every pairwise conflict came out ≈ 0.50 — random sign
agreement. That is expected of `--quick` (its numbers are meaningless by design),
but **if a full run also shows conflict ≈ 0.50 across the board, the hypothesis
has no measurable substrate** and idea 2 (the operator benchmark) is the one to
fall back on.

### How stage 1 → 2 → 3 is actually gated

Worth knowing, because it determines what "stable" means here:

* **Stage 1 → 2 involves no LLM judgement.** `_check_stage_completion`
  (`agent_manager.py:434`) returns complete as soon as
  `len(journal.good_nodes) > 0`. A node is good only if `parse_metrics` returns
  `valid_metrics_received: true` (`parallel_agent.py:1696`) — otherwise it is
  forced to `WorstMetricValue` and `is_buggy`. **`parse_metrics` is the whole
  gate**, and it runs on `agent.feedback.model`.
* This is exactly how the previous run died: the local server dropped `tools`, no
  call ever produced a tool_call, `parse_metrics` fell through to the plot-analysis
  dict, every node came back metric-less → 20/20 buggy → stage 1 ended in failure.
* **Stage 2 → 3** needs a best node that differs from the inherited node *and* an
  `evaluate_stage_completion` verdict — but also falls through unconditionally at
  `stage2_max_iters: 12`, so it cannot hang.
* If stage 1 exhausts `stage1_max_iters: 20` with no good node, `current_stage` is
  set to `None` and the run ends deliberately (`agent_manager.py:428`).

### Bounding server GPU memory (§2.8)

Both servers load their model **at import time, before `argparse`**, so CLI flags
cannot affect loading. Memory is controlled by environment variables:

| Var | `serve_coder.py` | `serve_vlm.py` | What it does |
|---|---|---|---|
| `CUDA_VISIBLE_DEVICES` | required | required | Which physical card. Unset → `device_map="auto"` spreads over every GPU on the node and collides with the experiments. Both servers now print a warning if it is unset. |
| `SERVE_MAX_MEM` | e.g. `170GiB` | e.g. `72GiB` | Per-visible-GPU ceiling passed to `max_memory=`. Use it on any card that is **shared**. |
| `SERVE_MAX_NEW_TOKENS` | default **12288** | default 4096 | Generation ceiling. The KV cache, not the weights, is what OOMs a server that loaded fine. Do **not** lower the coder below 12288 — `bfts_config.yaml agent.code.max_tokens` is 12000, and a lower cap silently truncates generated code. |
| `SERVE_MAX_PIXELS` | — | default `1280*28*28` | Caps visual tokens per image. An uncapped full-size plot is the usual cause of a VLM OOM at request time. |

Also changed in both: `torch_dtype="auto"` → `dtype=torch.bfloat16` (transformers 5
rename; `"auto"` can resolve to fp32 and double the footprint), `device_map="auto"`
→ `device_map={"": 0}` when exactly one GPU is visible (no pointless sharding),
`model.eval()`, generation under `torch.inference_mode()`, a footprint print on
startup, and `torch.cuda.empty_cache()` in the error path so one OOM does not
poison every later request.

Measured sizes: Qwen2.5-72B = **136 GB**, Qwen2.5-VL-32B = **64 GB**, cards are
**184 GiB**. 136 + 64 = 200 > 184, so the two servers cannot share one card —
they must be split across GPU 1 and GPU 3. The experiments then share GPU 3 with
the VLM, which is safe only because the VLM is capped.

### Secrets are handled

`third_trial.sh` sources `.env` and maps the HuggingFace token — stored there
under the non-standard name **`hk_token`** — to `HF_TOKEN`. Verified: this token
unlocks `gemma-3-1b-it`, `gemma-2-2b-it` and `gemma-2-27b`. `.env` also supplies
`S2_API_KEY`, so Stage 3 citations are authenticated rather than rate-limited.

No manual `huggingface-cli login` is needed.

### Knobs

```bash
EXP_GPUS=1,3 bash experiment_config/third_trial.sh   # 2 GPUs -> 2 parallel nodes
                                                     # (only if the coder is off GPU 1)
MIN_FREE_MIB=32768 bash experiment_config/third_trial.sh  # raise the headroom gate
IDEAS="0"    bash experiment_config/third_trial.sh   # flagship idea only
CANARY=0     bash experiment_config/third_trial.sh   # skip gate 2
BASE_MODEL=google/gemma-2-27b ...                    # confirmation scale
BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct ...            # open, no token needed
python ai_scientist/ideas/third_trial/third_trial.py --quick   # canary alone
python main.py --resume_from experiments/<...>       # resume a partial run
```

Workers are clamped one-per-visible-GPU, so `EXP_GPUS` sets both placement and
parallelism.

### Small-model scale is the design, not a compromise

This is deliberately a small-model study. `BASE_MODEL` is one environment
variable and everything else is architecture-agnostic (`LORA_TARGETS` are the
attention projections `q/k/v/o_proj`, named identically across Qwen2, Llama and
Gemma), so scale is the only thing that changes.

The phenomena being measured — sign conflict between deltas, TIES vs DARE
ordering, per-domain retention asymmetry — are all observable at 1–3B. Running
the *search* at 1B is what makes autonomous research work at all: nodes finish in
minutes, so the tree actually explores instead of timing out. `third_trial.md`
now tells the ideation agent this explicitly, and forbids proposals that only
make sense at 27B+.

| tier | model | bf16 weights | use |
|---|---|---|---|
| canary | any, with `--quick` | — | gate 2, ~3 min |
| **search** | **`google/gemma-3-1b-it`** | ~2 GiB | **default; the tree search** |
| mid | `google/gemma-2-2b-it` | ~5 GiB | sanity that the effect survives scale |
| confirm | `google/gemma-2-27b` | ~54 GiB | one run of the winning config |
| open alt | `Qwen/Qwen2.5-1.5B-Instruct` | ~3 GiB | no token; TMMLU+ baseline model |

On a 184 GiB card a 27B run fits **one worker per GPU** — which the §2.1 fix now
enforces, rather than discovering by OOM.

**gemma-4-31b:** no weights anywhere on disk, and the `gemma-api` endpoint at
`backend_openai.py:48` returns HTTP 503 (`failure to get a peer from the
ring-balancer`). It cannot be LoRA-trained today. `BASE_MODEL` points at it the
moment weights appear — nothing else needs to change.

---

## 4. Running it

```bash
conda activate ai_scientist_v2
EXP_GPUS=3 bash experiment_config/third_trial.sh
```

Preflight output looks like this (this is a run on a node with no GPU and no
servers, so it correctly refuses):

```
  python deps (torch/peft/transformers/datasets) OK
  HF_HOME flock-capable (/tmp/hf_cache_hh30990)  OK
  coder server reachable                         FAIL  <- start src/serve_coder.py
  vlm server reachable                           FAIL  <- start src/serve_vlm.py
  GPU 0 free memory                              WARN  <- could not query nvidia-smi
  linked 6 cached repo(s) into HF_HOME           OK
  base model + TMMLU+ cached                     OK
  note: agent.num_workers=4, 1 GPU(s) pinned     -> 1 parallel node(s)
  ideas JSON (3 ideas)                           OK
  code template                                  OK

Preflight FAILED — fix the above before running. Nothing was launched.
```

Other switches:

```bash
IDEAS="0"  bash experiment_config/third_trial.sh     # flagship idea only
python main.py --resume_from experiments/<...>       # resume a partial run
```

`bfts_config.yaml` needs no change: `exec.timeout: 3600` matches the one-hour
per-node budget, and `agent.num_workers: 4` is now clamped to the GPUs you pin.

---

### Stage 3/4 toolchain — checked, not blocking

`pdflatex`, `bibtex`, `latexmk` are all present, and the ICBINB LaTeX template is
complete, so the writeup can compile. Two soft gaps:

* **`chktex` is missing.** Only used through `os.popen(...)` for a style-hint
  string in the reflection prompt; a missing binary yields `""` and the writeup
  continues. Cosmetic. Install it if you want the LaTeX lint feedback.
* **`S2_API_KEY`** was not set in the shell, but it *is* in `.env`, which the
  script now sources — so Stage 3 citations are authenticated.
  `--num_cite_rounds` was still lowered from 20 to 10 to keep Stage 3 bounded.

---

## 5. What is not fixed

* **`src/serve_coder.py` / `src/serve_vlm.py` still cannot do real tool calls.**
  Worked around by putting the schema in the prompt (§2.2), which is reliable but
  not as strong as constrained decoding. A proper fix is to serve these with vLLM
  and its OpenAI-compatible tool-calling support.
* **`nvidia-smi` does not work on this login node**, so the GPU-occupancy
  preflight check degrades to a `WARN` here. It will run properly on the compute
  node.
* The scientific quality of the three ideas is unproven — they are well-specified
  and runnable, but whether the anisotropy hypothesis holds is what the
  experiment is for.
