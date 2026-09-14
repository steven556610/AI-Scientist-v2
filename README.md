<div align="center">
  <a href="https://github.com/SakanaAI/AI-Scientist_v2/blob/main/docs/logo_v1.jpg">
    <img src="docs/logo_v1.png" width="215" alt="AI Scientist v2 Logo" />
  </a>
  <h1>
    <b>The AI Scientist-v2: Workshop-Level Automated</b><br>
    <b>Scientific Discovery via Agentic Tree Search</b>
  </h1>
</div>

<p align="center">
  📚 <a href="https://pub.sakana.ai/ai-scientist-v2/paper">[Paper]</a> |
  📝 <a href="https://sakana.ai/ai-scientist-first-publication/"> [Blog Post]</a> |
  📂 <a href="https://github.com/SakanaAI/AI-Scientist-ICLR2025-Workshop-Experiment"> [ICLR2025 Workshop Experiment]</a>
</p>

Fully autonomous scientific research systems are becoming increasingly capable, with AI playing a pivotal role in transforming how scientific discoveries are made.
We are excited to introduce The AI Scientist-v2, a generalized end-to-end agentic system that has generated the first workshop paper written entirely by AI and accepted through peer review.

This system autonomously generates hypotheses, runs experiments, analyzes data, and writes scientific manuscripts. Unlike [its predecessor (AI Scientist-v1)](https://github.com/SakanaAI/AI-Scientist), the AI Scientist-v2 removes reliance on human-authored templates, generalizes across Machine Learning (ML) domains, and employs a progressive agentic tree search, guided by an experiment manager agent.

> **Note:**
> The AI Scientist-v2 doesn’t necessarily produce better papers than v1, especially when a strong starting template is available. v1 follows well-defined templates, leading to high success rates, while v2 takes a broader, more exploratory approach with lower success rates. v1 works best for tasks with clear objectives and a solid foundation, whereas v2 is designed for open-ended scientific exploration.

> **Caution!**
> This codebase will execute Large Language Model (LLM)-written code. There are various risks and challenges associated with this autonomy, including the potential use of dangerous packages, uncontrolled web access, and the possibility of spawning unintended processes. Ensure that you run this within a controlled sandbox environment (e.g., a Docker container). Use at your own discretion.

## Table of Contents

1.  [Requirements](#requirements)
    *   [Installation](#installation)
    *   [Supported Models and API Keys](#supported-models-and-api-keys)
2.  [Generate Research Ideas](#generate-research-ideas)
3.  [Run AI Scientist-v2 Paper Generation Experiments](#run-ai-scientist-v2-paper-generation-experiments)
4.  [Citing The AI Scientist-v2](#citing-the-ai-scientist-v2)
5.  [Frequently Asked Questions](#frequently-asked-questions)
6.  [Acknowledgement](#acknowledgement)

## Requirements

This code is designed to run on Linux with NVIDIA GPUs using CUDA and PyTorch.

### Installation

```bash
# Create a new conda environment
conda create -n ai_scientist python=3.11
conda activate ai_scientist

# Install PyTorch with CUDA support (adjust pytorch-cuda version for your setup)
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# Install PDF and LaTeX tools
conda install anaconda::poppler
conda install conda-forge::chktex

# Install Python package requirements
pip install -r requirements.txt
```

Installation usually takes no more than one hour.

### Supported Models and API Keys

#### OpenAI Models

By default, the system uses the `OPENAI_API_KEY` environment variable for OpenAI models.

#### Gemini Models

By default, the system uses the `GEMINI_API_KEY` environment variable for Gemini models through OpenAI API.

#### Claude Models via AWS Bedrock

To use Claude models provided by Amazon Bedrock, install the necessary additional packages:
```bash
pip install anthropic[bedrock]
```
Next, configure valid [AWS Credentials](https://docs.aws.amazon.com/cli/v1/userguide/cli-configure-envvars.html) and the target [AWS Region](https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-regions.html) by setting the following environment variables: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION_NAME`.

#### Semantic Scholar API (Literature Search)

Our code can optionally use a Semantic Scholar API Key (`S2_API_KEY`) for higher throughput during literature search [if you have one](https://www.semanticscholar.org/product/api). This is used during both the ideation and paper writing stages. The system should work without it, though you might encounter rate limits or reduced novelty checking during ideation. If you experience issues with Semantic Scholar, you can skip the citation phase during paper generation.

#### Setting API Keys

Ensure you provide the necessary API keys as environment variables for the models you intend to use. For example:
```bash
export OPENAI_API_KEY="YOUR_OPENAI_KEY_HERE"
export S2_API_KEY="YOUR_S2_KEY_HERE"
# Set AWS credentials if using Bedrock
# export AWS_ACCESS_KEY_ID="YOUR_AWS_ACCESS_KEY_ID"
# export AWS_REGION_NAME="your-aws-region"
```

#### Local Models (Qwen2.5-72B and Qwen2.5-VL-32B)

For users with sufficient GPU VRAM (e.g., NVIDIA GB200 nodes), you can deploy local models to replace external APIs. We provide built-in scripts to serve `Qwen2.5-72B-Instruct` (for code generation & writeup) and `Qwen2.5-VL-32B-Instruct` (for visual plot aggregation & review).

**1. Start the VLM Server (Port 8000)**
Open a terminal and allocate GPUs (e.g., GPUs 2 and 3) for the Vision-Language Model:
```bash
conda activate ai_scientist
CUDA_VISIBLE_DEVICES=2,3 python src/serve_vlm.py --port 8000
```
*Test the VLM server:*
```bash
python src/test_vlm.py
```

**2. Start the LLM Server (Port 8002)**
Open another terminal and allocate GPUs for the Large Language Model:
```bash
conda activate ai_scientist
CUDA_VISIBLE_DEVICES=0,1 python src/serve_coder.py --port 8002
```
*Test the LLM server:*
```bash
python src/test_coder.py
```

**3. Run AI-Scientist with Local Models**
Once both servers are running, start `main.py` using the `local_coder/` and `local/` model prefixes:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
 --resume_from "experiments/..." \
 --model_writeup local_coder/qwen2.5-72b \
 --model_citation local_coder/qwen2.5-72b \
 --model_agg_plots local/qwen2.5-vl-32b \
 --model_review local/qwen2.5-vl-32b
```

## Generate Research Ideas

Before running the full AI Scientist-v2 experiment pipeline, you first use the `ai_scientist/perform_ideation_temp_free.py` script to generate potential research ideas. This script uses an LLM to brainstorm and refine ideas based on a high-level topic description you provide, interacting with tools like Semantic Scholar to check for novelty.

1.  **Prepare a Topic Description:** Create a Markdown file (e.g., `my_research_topic.md`) describing the research area or theme you want the AI to explore. This file should contain sections like `Title`, `Keywords`, `TL;DR`, and `Abstract` to define the scope of the research. Refer to the example file `ai_scientist/ideas/i_cant_believe_its_not_better.md` for the expected structure and content format. Place your file in a location accessible by the script (e.g., the `ai_scientist/ideas/` directory).

2.  **Run the Ideation Script:** Execute the script from the main project directory, pointing it to your topic description file and specifying the desired LLM.

    ```bash
    python ai_scientist/perform_ideation_temp_free.py \
     --workshop-file "ai_scientist/ideas/my_research_topic/" \
     --model kimi-k3 \
     --max-num-generations 10 \
     --num-reflections 3
    ```
    *   `--workshop-file`: Path to your topic description Markdown file.
    *   `--model`: The LLM to use for generating ideas (ensure you have the corresponding API key set).
    *   `--max-num-generations`: How many distinct research ideas to attempt generating.
    *   `--num-reflections`: How many refinement steps the LLM should perform for each idea.

3.  **Output:** The script will generate a JSON file named after your input Markdown file (e.g., `ai_scientist/ideas/my_research_topic.json`). This file will contain a list of structured research ideas, including hypotheses, proposed experiments, and related work analysis.

4.  **Proceed to Experiments:** Once you have the generated JSON file containing research ideas, you can proceed to the next section to run the experiments.

This ideation step guides the AI Scientist towards specific areas of interest and produces concrete research directions to be tested in the main experimental pipeline.

## Project Structure & Execution Flow

### 1. Where to Put Ideas

Before running the pipeline, your research ideas should be placed in `ai_scientist/ideas/<idea_name>/`.
This directory should contain:
- `<idea_name>.json`: The JSON file defining the idea, generated by `perform_ideation_temp_free.py`.
- `code/`: (Optional but recommended) A subfolder containing the initial Python templates for the experiment (e.g., `runfile.py`). This allows the agent to start with a solid foundation.
- `configs/default.yaml`: (Optional) Hyperparameters for the experiment.

### 2. Main Entry Points: `main.py` vs `launch_scientist_bfts.py`

- **`main.py` (Recommended)**: This is the refactored, modular entry point. It is highly streamlined (<150 lines) and delegates all core logic—such as CLI parsing, I/O handling, and running the 4 distinct stages (Experiments, Plots, Writeup, Review)—to the heavily-tested `src/` module package.
- **`launch_scientist_bfts.py`**: This is the original, monolithic script provided for backward compatibility. It contains all the logic inline.

Both scripts accept identical command-line arguments and produce identical outputs. We strongly recommend using `main.py` as it benefits from better error handling, process cleanup (`src/process_cleanup.py`), and a testable architecture.

### 3. Run AI Scientist-v2 Paper Generation Experiments

Using the JSON file and code template from the ideation step, launch the pipeline using `main.py`.

Example command (adjust LLM models as needed):

```bash
python main.py \
 --load_ideas "ai_scientist/ideas/my_research_topic/" \
 --load_code \
 --add_dataset_ref \
 --model_writeup kimi-k3 \
 --model_citation gemma-4-31b-it \
 --model_review local/qwen2.5-vl-32b \
 --model_agg_plots local/qwen2.5-vl-32b \
 --num_cite_rounds 20
```

> **Note**: For text-based tasks (Ideation, Coding, Writeup), you can safely use your remote LLM APIs (e.g. `kimi-k3`, `gemma-4-31b-it`). However, for Vision tasks (`--model_agg_plots` and `--model_review`), remote APIs often fail to parse base64 images correctly or lack vision support. We strongly recommend pointing these to your local VLM.

### 3.1 Running the Local VLM Server
Before executing `main.py` with local VLM models, start the local VLM server in a separate terminal:
```bash
conda activate ai_scientist_v2
python src/serve_vlm.py --port 8000
```
This will download and serve `Qwen/Qwen2.5-VL-32B-Instruct` on your local GPU (port 8000).

> **Note**: If you prefer the legacy script, simply replace `main.py` with `launch_scientist_bfts.py`.

The pipeline executes in 4 stages:
1. **Ideation & Experiments**: Executes the Best-First Tree Search (BFTS).
2. **Plot Aggregation**: LLMs synthesize the experiment results into plots.
3. **Paper Writeup**: Generates citations (using Semantic Scholar / ArXiv) and compiles a LaTeX paper.
4. **Paper Review**: Performs LLM/VLM peer review on the generated PDF.

### 3.1 Resuming a Failed Experiment

If your pipeline crashes during a later stage (e.g., due to API errors in Stage 3), you can resume the run without losing your progress. `main.py` automatically maintains a `run_state.json` tracker and preserves all data (including raw `experiment_results/`).

Simply point `main.py` to your existing experiment directory using the `--resume_from` flag instead of `--load_ideas`:

```bash
python main.py \
 --resume_from "experiments/my_research_topic/2026-09-02_12-00-00_attempt_0" \
 --model_writeup kimi-k3
```

The script will automatically detect which stages were successfully completed, skip them, and instantly resume from the first pending stage.

### 4. Process and Result Tracking

All outputs are structured systematically to prevent data loss:

- **Root Output Directory**: `experiments/<idea_name>/<timestamp>_attempt_<N>/`
- **Idea Snapshots**: A frozen snapshot of the idea is saved as `idea.json` and `idea.md` inside the experiment directory.
- **Stage Logs**: Execution logs for each of the 4 stages are saved in the `logs/` subfolder (e.g., `logs/01_experiment_execution.log`). These use a "Tee" mechanism, meaning you can view them live in the terminal while they are simultaneously written to disk.
- **Experiment Results**: Raw results and the tree search visualization (`unified_tree_viz.html`) are saved within `logs/0-run/experiment_results/` and subsequently extracted to the main experiment directory.
- **Final Output**: The generated paper (`.pdf`) and the peer reviews (`review_text.txt`, `review_img_cap_ref.json`) will be located in the root output directory.

## Running Tests

The modular architecture introduced via `src/` includes a comprehensive test suite (40+ unit and boundary tests). Tests cover CLI argument parsing, I/O utilities, mocking of stage execution, process cleanup, and code generation.

To run the tests and verify system integrity, use `pytest`:

```bash
# Install pytest if you haven't already
pip install pytest

# Run the test suite
python -m pytest tests/ -v --tb=short
```


## Citing The AI Scientist-v2

If you use **The AI Scientist-v2** in your research, please cite our work as follows:

```bibtex
@article{aiscientist_v2,
  title={The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search},
  author={Yamada, Yutaro and Lange, Robert Tjarko and Lu, Cong and Hu, Shengran and Lu, Chris and Foerster, Jakob and Clune, Jeff and Ha, David},
  journal={arXiv preprint arXiv:2504.08066},
  year={2025}
}
```

## Frequently Asked Questions

**Why wasn't a PDF or a review generated for my experiment?**

The AI Scientist-v2 completes experiments with a success rate that depends on the chosen foundation model, and the complexity of the idea. Higher success rates are generally observed when using powerful models like Claude 3.5 Sonnet for the experimentation phase.

**What is the estimated cost per experiment?**

The ideation step cost depends on the LLM used and the number of generations/reflections, but is generally low (a few dollars). For the main experiment pipeline, using Claude 3.5 Sonnet for the experimentation phase typically costs around $15–$20 per run. The subsequent writing phase adds approximately $5 when using the default models specified in the example command. Using GPT-4o for `model_citation` is recommended as it can help reduce writing costs.

**How do I run The AI Scientist-v2 for different subject fields?**

First, perform the [Generate Research Ideas](#generate-research-ideas) step. Create a new Markdown file describing your desired subject field or topic, following the structure of the example `ai_scientist/ideas/i_cant_believe_its_not_better.md`. Run the `perform_ideation_temp_free.py` script with this file to generate a corresponding JSON idea file. Then, proceed to the [Run AI Scientist-v2 Paper Generation Experiments](#run-ai-scientist-v2-paper-generation-experiments) step, using this JSON file with the `launch_scientist_bfts.py` script via the `--load_ideas` argument.

**What should I do if I have problems accessing the Semantic Scholar API?**

The Semantic Scholar API is used to assess the novelty of generated ideas and to gather citations during the paper write-up phase. If you don't have an API key, encounter rate limits, you may be able to skip these phases.

**I encountered a "CUDA Out of Memory" error. What can I do?**

This error typically occurs when the AI Scientist-v2 attempts to load or run a model that requires more GPU memory than available on your system. To resolve this, you can try updating your ideation prompt file (`ai_scientist/ideas/my_research_topic.md`) to suggest using smaller models for the experiments.

## Acknowledgement

The tree search component implemented within the `ai_scientist` directory is built on top of the [AIDE](https://github.com/WecoAI/aideml) project. We thank the AIDE developers for their valuable contributions and for making their work publicly available.


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=SakanaAI/AI-Scientist-v2&type=Date)](https://star-history.com/#SakanaAI/AI-Scientist-v2&Date)

## ⚖️ License & Responsible Use

This project is licensed under **The AI Scientist Source Code License** (a derivative of the Responsible AI License). 

**Mandatory Disclosure:** By using this code, you are legally bound to clearly and prominently disclose the use of AI in any resulting scientific manuscripts or papers. 

We recommend the following attribution in your paper's Abstract or Methods section:
> "This manuscript was autonomously generated using [The AI Scientist](https://github.com/SakanaAI/AI-Scientist)."
