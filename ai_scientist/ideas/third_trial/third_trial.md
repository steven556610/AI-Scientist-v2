# Title: Interference-Aware Merging of Subject-Specific LoRA Adapters for Taiwanese Junior/Senior High School Curriculum

## Keywords
model merging, LoRA, TIES-Merging, DARE, task arithmetic, task interference, sign conflict, Gemma, Traditional Chinese, TMMLU+, Taiwan curriculum, localized evaluation

## TL;DR
When we fine-tune one LoRA adapter per school subject (國文 / 數學 / 自然 / 臺灣史地社會) on a Gemma base model and merge them into a single model, the damage is **not** spread evenly: culturally-grounded, Taiwan-specific subjects degrade far more than symbolic subjects like Math. This workshop track asks *why*, measures it, and proposes merging rules that protect the localized knowledge.

## Motivation and Setting

A single school-assistant model deployed in a Taipei junior/senior high school must handle many subjects at once. Training one model per subject is not deployable; multi-task training is expensive and must be redone whenever a subject is added. Model merging (task arithmetic, TIES-Merging, DARE) promises to fuse independently-trained LoRA adapters at near-zero cost — but the published evidence is almost entirely on English, symbolic, or Western-centric benchmarks.

Taiwanese curriculum content is a uniquely revealing stress test for merging, because its subjects have **very different relationships to the base model's pretraining distribution**:

- **臺灣史地 / 社會 (Taiwan history, geography, civics)** — knowledge that is sparse in pretraining, locally specific, and frequently *contradicted* by mainland-Chinese-dominant web text (place names, historical periodization, political terminology). The adapter must move weights the furthest.
- **國文 (Chinese language and literature)** — Traditional Chinese, classical texts; shares a token subspace with the Taiwan-history adapter, so their delta weights are likely to **collide**.
- **數學 (Mathematics)** — largely language-independent symbolic reasoning; its delta should be nearly orthogonal to the others.
- **自然 (Science)** — intermediate.

The hypothesis this track exists to test: **sign conflict between LoRA deltas is concentrated exactly in the culturally-grounded subjects, and standard merging silently sacrifices Taiwan-specific knowledge first.** If true, uniform-weight merging is quietly the wrong default for any localized deployment, and a conflict-aware rule is needed.

## Evaluation Substrate

All evaluation uses **TMMLU+ (`ikala/tmmluplus`)**, real Taiwanese junior-high and senior-high examination questions in Traditional Chinese, four-way multiple choice. No synthetic or self-generated benchmark. Relevant subsets, grouped into four merge domains:

| Domain | TMMLU+ subsets | test rows |
|---|---|---|
| `chinese` (國文) | `junior_chinese_exam`, `chinese_language_and_literature`, `tve_chinese_language` | 857 |
| `math` (數學) | `junior_math_exam`, `tve_mathematics` | 325 |
| `taiwan` (臺灣史地社會) | `geography_of_taiwan`, `junior_social_studies`, `jce_humanities` | 984 |
| `science` (自然) | `junior_science_exam`, `junior_chemistry`, `secondary_physics` | 534 |

Accuracy is measured by comparing the logits of the tokens `A`/`B`/`C`/`D` at a single forward pass — deterministic, no free-form generation, no LLM judge, no reliance on a served model.

## Compute Constraints (must be respected by any proposal)

- Every experiment node must finish inside **one hour** on **one GPU**.
- The base model is bf16, loaded once, `device_map={"": 0}`. Never fp32, never `device_map="auto"`. Loading in fp32 is what OOM-killed every previous run.
- Base model is a single variable `BASE_MODEL`, default `google/gemma-3-1b-it` for the search loop. **This is a small-model study**: the claims must be established at the 0.5–3B scale, where a full four-domain merge sweep finishes in minutes. Scaling to a 27B/31B model is a single final confirmation run, not something tree search does. Do not propose experiments that only make sense at 27B+.
- Every `google/gemma-*` repo is gated. Never hard-code a Gemma id; always read `BASE_MODEL`. Never silently substitute a different cached model if a load fails — raise.
- Total trainable parameters per adapter: LoRA rank ≤ 16 on attention projections only.
- No `trl`, no `bitsandbytes` (not installed). Plain `peft` + a hand-written torch loop.

## What makes a good proposal here

Good proposals isolate **one** mechanism, measure it on the four domains above, and report both the merged-model accuracy **and** a diagnostic quantity that explains the accuracy (conflict rate, delta-norm, subspace overlap). Proposals that merely re-run TIES vs DARE with no diagnostic, or that require generating a new dataset, are out of scope.

## Abstract

Model merging lets independently fine-tuned experts be fused into one deployable model at negligible cost, but its failure modes have been characterized almost exclusively on English and symbolic benchmarks. We study merging in a setting where the experts differ sharply in how far they must pull the base model away from its pretraining prior: subject-specific LoRA adapters for the Taiwanese junior/senior high school curriculum, spanning Chinese literature, mathematics, science, and Taiwan history/geography/civics. We hypothesize that interference during merging is *anisotropic* — concentrated in the culturally-grounded, locally-specific subjects whose delta weights both move furthest and overlap most with one another — and that uniform merging rules therefore degrade exactly the knowledge a localized deployment most needs. Using TMMLU+, a benchmark of real Taiwanese examination questions in Traditional Chinese, we measure per-domain accuracy retention under task arithmetic, TIES-Merging, and DARE, alongside layer-wise sign-conflict statistics between adapter pairs. We then evaluate merging rules that allocate coefficients according to measured conflict rather than uniformly. The goal is a diagnostic and a merging rule that preserve Taiwan-specific curricular knowledge without sacrificing symbolic subjects, and evidence on whether conclusions drawn from English merging benchmarks transfer to localized ones.
