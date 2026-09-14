# Title: Systematic Evaluation of MergeKit Strategies for Large-to-Small Model Training and Capability Preservation

## Keywords
model merging, MergeKit, TMMLU+, large-to-small model training, knowledge distillation, MoE, dense, Gemma, MMLU, GSM8K, IFEval

## TL;DR
This proposal outlines a rigorous, large-scale experimental framework based on MergeKit. It focuses on testing whether specific model merging strategies (MoE and dense) can achieve SOTA performance by using a large model to train a smaller model (`gemma-4-E2B`). It strictly mandates the use of real `TMMLU+` data (no synthetic data) while enforcing that foundational capabilities (MMLU, GSM8K, IFEval) do not degrade during the process.

## Motivation and Setting
Model merging has emerged as a powerful tool to fuse capabilities without the prohibitive costs of full retraining. However, finding the optimal merging strategies—and ensuring they don't cause catastrophic forgetting of foundational skills—remains an open challenge. 

This trial is designed to systematically explore the strategies and combinations available within `arcee-ai/MergeKit` to see if they can surpass current SOTA. The architectural focus is on a "large model training small model" paradigm. To ensure rigor and applicability to localized tasks, the trial strictly relies on Taiwanese examination data (`TMMLU+`) while continuously validating against standard global benchmarks.

## Strict Constraints and Hard Rules (DO NOT BYPASS)
Any implementation, script, or proposal for this trial **MUST** adhere to the following hard rules. They are non-negotiable and cannot be bypassed:

1. **Tooling & Codebase**: You MUST use the tools and codebase from `https://github.com/arcee-ai/MergeKit` and structure the implementation as a reusable template.
2. **Data Source Restrictions**: 
    - **NO SYNTHETIC DATA**: You are strictly forbidden from using any model-generated data.
    - **Primary Dataset**: You MUST use the `https://huggingface.co/datasets/ikala/tmmluplus` dataset for training and evaluation.
    - **Data Storage**: All training and validation data MUST be directly downloaded, saved, and kept in the `data/` directory at the root of the experiment workspace (`/data/`).
3. **Model Selection**: The experiment MUST use `https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-unquantized`.
4. **Architecture Paradigm**: The architecture MUST be designed around a "large model training a small model" concept.
5. **Merge Modes**: The implementation MUST be flexible enough to support and test both **MoE (Mixture of Experts)** and **Dense** modes.
6. **Capability Preservation (No Regression)**: During the training and merging process, the model's general capabilities MUST NOT degrade. You MUST conduct foundational capability tests—specifically **MMLU, GSM8K, and IFEval**—to prove this.
7. **Experiment Scale & Reproducibility**: 
    - You MUST conduct a sufficiently large sweep of experiments: at least **16 * 3 = 48 different experimental combinations**.
    - EVERY single experimental combination MUST be 100% reproducible after it finishes.
    - You MUST rigorously record and log all experimental processes and configurations.

## Evaluation Substrate
The primary evaluation for localized knowledge relies exclusively on **TMMLU+ (`ikala/tmmluplus`)**. To satisfy the hard rule of capability preservation, every merged/trained candidate must also be evaluated on:
- **MMLU** (Massive Multitask Language Understanding)
- **GSM8K** (Grade School Math 8K)
- **IFEval** (Instruction Following Evaluation)

## What makes a good proposal here
A valid proposal will lay out exactly how the 48+ MergeKit combinations (spanning MoE and dense architectures) will be parameterized and executed. It must provide a clear pipeline that loads `gemma-4-E2B`, applies the large-to-small training utilizing purely TMMLU+ data (stored in `/data/`), and outputs both the TMMLU+ accuracy and the foundational benchmark scores (MMLU/GSM8K/IFEval). Proposals that fail to test all 48 combinations, use synthetic data, or skip the foundational capability checks are invalid.

## Abstract
Despite the rapid adoption of model merging techniques, the precise combinations required to consistently achieve SOTA performance—especially without degrading base capabilities—are not well documented. This study introduces a heavily constrained, large-scale empirical evaluation using `arcee-ai/MergeKit`. By adopting a large-to-small model training architecture with `google/gemma-4-E2B-it-qat-q4_0-unquantized`, we explore both MoE and dense merging strategies. Crucially, this study forbids the use of synthetic data, relying solely on the `ikala/tmmluplus` dataset for training and primary evaluation. Furthermore, we enforce a strict "no regression" policy on general capabilities, continuously monitoring MMLU, GSM8K, and IFEval. By executing and fully logging over 48 reproducible experimental combinations, this work aims to identify the optimal merge configurations that maximize localized knowledge while preserving foundational model strengths, contributing robust negative and positive results to the applied deep learning community.
