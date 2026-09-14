# AI Scientist v2 — DGX + Enroot 使用指南

本手冊說明如何在 DGX 運算節點上，為 **AI Scientist v2** 準備環境、建立/掛載 Enroot 容器，並透過 GPU 執行自主 AI 研究。

> **⚠️ v2 架構提醒**：
> - AI Scientist v2 延續了 v1 的腳本執行模式，主要透過 `launch_scientist_bfts.py` 驅動。
> - 與 v3 的 Harbor/Agent 架構不同，v2 需要在容器內配置好 Python 環境（包含 PyTorch 等依賴），直接執行 Python 腳本。
> - 為了在 DGX 上隔離環境並使用 GPU，我們建議使用 NVIDIA 官方的 PyTorch 容器或自行建構帶有完整依賴的 Docker 映像檔再轉為 Enroot 執行。

---

## 📌 目錄
1. [專案架構與執行模式](#一專案架構與執行模式)
2. [Step 1：準備環境與 API Keys](#二step-1準備環境與-api-keys)
3. [Step 2：在 DGX 上匯入並建立 Enroot 容器](#三step-2在-dgx-上匯入並建立-enroot-容器)
4. [Step 3：啟動容器並初始化環境](#四step-3啟動容器並初始化環境)
5. [Step 4：執行 AI 研究](#五step-4執行-ai-研究)
6. [常用指令速查](#六常用指令速查)

---

## 一、專案架構與執行模式

```
AI-Scientist-v2/
├── launch_scientist_bfts.py            ← 論文生成實驗的主要執行腳本
├── ai_scientist/
│   ├── perform_ideation_temp_free.py   ← 想法生成腳本
│   └── ideas/                          ← 存放研究主題 (Markdown) 與生成的想法 (JSON)
├── bfts_config.yaml                    ← Tree Search 的配置檔
├── requirements.txt                    ← Python 依賴包
└── experiments/                        ← 實驗結果與產出的論文會存放在此
```

| 模式 | 說明 | GPU 需求 |
| :--- | :--- | :---: |
| **Ideation (生成想法)** | 使用 `perform_ideation_temp_free.py` 根據 Markdown 生成 JSON 想法 | ❌ 不需要 |
| **Paper Generation (實驗與寫作)**| 使用 `launch_scientist_bfts.py` 進行樹狀搜尋、跑實驗與產生論文 | ✅ 需要 |

---

## 二、Step 1：準備環境與 API Keys

在開始前，建立並填寫 `.env` 檔案（或啟動時作為環境變數傳入）：
```bash
cd /home/hhri-ai/hh30990/ai_for_research/AI-Scientist-v2

# 建立 .env 檔案（不要 commit）
cat > .env << 'EOF'
# 必填：實驗與寫作階段所需的 LLM API (依據您的參數而定)
OPENAI_API_KEY=<your_openai_key>
ANTHROPIC_API_KEY=<your_anthropic_key>
GEMINI_API_KEY=<your_gemini_key>

# 選填：文獻搜尋（提高速率限制）
S2_API_KEY=<your_semantic_scholar_key>
EOF
```

---

## 三、Step 2：匯入您製作的 Enroot sqsh 檔案

既然您已經製作好了 `ai_scientist_v2` 的 `.sqsh` 映像檔，我們可以直接使用它建立容器：

```bash
# 登入 DGX 節點
ssh hh30990@r04dgx02

# 使用您的 sqsh 檔案建立 Enroot 容器
ENROOT_DATA_PATH=~/.local/share/enroot \
enroot create --name ai_scientist_v2_container /home/hhri-ai/hh30990/ai-scientist-v2.sqsh

# 確認容器已建立
ENROOT_DATA_PATH=~/.local/share/enroot enroot list
```

---

## 四、Step 3：啟動容器並初始化環境

使用掛載與 GPU 參數啟動容器，並將程式碼目錄掛載至 `/workspace`。我們同時載入 `.env` 檔案，並把 Kimi 變數映射給 OpenAI 變數：

```bash
cd /home/hhri-ai/hh30990/ai_for_research/AI-Scientist-v2

# 讀取 .env 中的變數
export $(grep -v '^#' .env | xargs)

# 啟動容器 (請注意替換對應的 KEY)
ENROOT_DATA_PATH=~/.local/share/enroot \
enroot start --rw \
  --env NVIDIA_VISIBLE_DEVICES=all \
  --env NVIDIA_DRIVER_CAPABILITIES=all \
  --env KIMI_API_KEY=$KIMI_API_KEY \
  --env KIMI_API_BASE=$KIMI_API_BASE \
  --env KIMI_MODEL=$KIMI_MODEL \
  --env GEMMA_API_KEY=$GEMMA_API_KEY \
  --env GEMMA_API_BASE=$GEMMA_API_BASE \
  --env GEMMA_MODEL=$GEMMA_MODEL \
  --env S2_API_KEY=$S2_API_KEY \
  --mount /home/hhri-ai/hh30990/ai_for_research/AI-Scientist-v2:/workspace \
  --mount /home/hhri-ai/hh30990/miniforge3:/home/hhri-ai/hh30990/miniforge3 \
  --env HF_TOKEN=$HF_TOKEN \
  --mount /home/hhri-ai/hh30990/.cache/huggingface:/home/hhri-ai/hh30990/.cache/huggingface \
  ai_scientist_v2 bash
```

進入容器後，確認環境是否就緒：

```bashw
cd /workspace

# 將掛載進來的 miniforge 加入環境變數
export PATH="/home/hhri-ai/hh30990/miniforge3/bin:$PATH"
# 您可以直接建立專案需要的環境 (如果您在外面還沒建好的話)
conda create -n ai_scientist python=3.11 -y
# 啟動環境 (使用 source 啟動)
source activate ai_scientist_v2ENROOT_DATA_PATH=~/.local/share/enroot \
enroot start --rw \
  --env NVIDIA_VISIBLE_DEVICES=all \
  --env NVIDIA_DRIVER_CAPABILITIES=all \
  --env KIMI_API_KEY=$KIMI_API_KEY \
  --env KIMI_API_BASE=$KIMI_API_BASE \
  --env KIMI_MODEL=$KIMI_MODEL \
  --env GEMMA_API_KEY=$GEMMA_API_KEY \
  --env GEMMA_API_BASE=$GEMMA_API_BASE \
  --env GEMMA_MODEL=$GEMMA_MODEL \
  --env S2_API_KEY=$S2_API_KEY \
  --mount /home/hhri-ai/hh30990/ai_for_research/AI-Scientist-v2:/workspace \
  --mount /home/hhri-ai/hh30990/miniforge3:/home/hhri-ai/hh30990/miniforge3 \
  --env HF_TOKEN=$HF_TOKEN \
  --mount /home/hhri-ai/hh30990/.cache/huggingface:/home/hhri-ai/hh30990/.cache/huggingface \
  ai_scientist_v2 bash

# 若您的 sqsh 中尚未安裝依賴，可以執行：
# 透過 pip 安裝支援 aarch64 的 PyTorch (取代 conda)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 安裝專案的其他 Python 套件
pip install -r requirements.txt

# 安裝系統層級的 LaTeX 工具 (重要！因為這是論文產出的依賴，通常宿主機的 conda 不含這個)
apt-get update
apt-get install -y poppler-utils chktex texlive-full

```

驗證 GPU：
```bash
# 驗證 GPU 可見
nvidia-smi

# 驗證 PyTorch CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

---

## 五、Step 4：執行 AI 研究

### 1. 生成研究想法 (Ideation)
```bash
cd /workspace

# 請確保 ai_scientist/ideas/ 下有您的主題 markdown (如 my_research_topic.md)
# 選項 A：使用 Kimi-K3 (DGX 內部端點)
python ai_scientist/perform_ideation_temp_free.py \
 --workshop-file "ai_scientist/ideas/my_research_topic.md" \
 --model kimi-k3 \
 --max-num-generations 10 \
 --num-reflections 3

# 選項 B：使用 Gemma-4-31B (Foxconn 內部端點)
python ai_scientist/perform_ideation_temp_free.py \
 --workshop-file "ai_scientist/ideas/my_research_topic.md" \
 --model gemma-4-31b-it \
 --max-num-generations 10 \
 --num-reflections 3
```

### 2. 跑實驗與生成論文 (Paper Generation)
確保可用 GPU 的設定：
```bash
# 若要指定 GPU 2 和 GPU 3
export CUDA_VISIBLE_DEVICES=2,3
```

啟動 BFTS (Best-First Tree Search) 實驗：
```bash
# 請根據您的 JSON 檔案名稱進行替換
# 選項 A：使用 Kimi-K3 (DGX 內部端點)
python launch_scientist_bfts.py \
 --load_ideas "ai_scientist/ideas/my_research_topic.json" \
 --add_dataset_ref \
 --model_writeup kimi-k3 \
 --model_citation kimi-k3 \
 --model_review kimi-k3 \
 --model_agg_plots kimi-k3 \
 --num_cite_rounds 20

# 選項 B：使用 Gemma-4-31B (Foxconn 內部端點)
python launch_scientist_bfts.py \
 --load_ideas "ai_scientist/ideas/my_research_topic.json" \
 --add_dataset_ref \
 --model_writeup gemma-4-31b-it \
 --model_citation gemma-4-31b-it \
 --model_review gemma-4-31b-it \
 --model_agg_plots gemma-4-31b-it \
 --num_cite_rounds 20
```

> 💡 **關於 `--load_code` 旗標**：這個旗標需要對應的 `.py` 起始程式碼檔案（例如 `my_research_topic.py`）。如果您的研究是全新的、1這裡不需要這個旗標。如果需要使用，請先建立對應的 `.py` 檔。

研究完成後，結果與生成的 PDF 會存放在 `experiments/<timestamp_ideaname>/` 資料夾下。

---

## 六、常用指令速查
| 目的 | 指令 |
| :--- | :--- |
| 匯入 Enroot sqsh 檔案 | `ENROOT_DATA_PATH=~/.local/share/enroot enroot create --name ai_scientist_v2_container /home/hhri-ai/hh30990/ai-scientist-v2.sqsh` |
| 啟動容器（含 GPU） | `ENROOT_DATA_PATH=~/.local/share/enroot enroot start --rw --env NVIDIA_VISIBLE_DEVICES=all --env NVIDIA_DRIVER_CAPABILITIES=all --env OPENAI_API_KEY=$KIMI_API_KEY --env OPENAI_API_BASE=$KIMI_API_BASE --mount /home/hhri-ai/hh30990/ai_for_research/AI-Scientist-v2:/workspace ai_scientist_v2_container bash` |
| 確認 GPU 可見 | `nvidia-smi` |
| 確認 PyTorch CUDA | `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"` |
| 指定 GPU 2 和 GPU 3 | `export CUDA_VISIBLE_DEVICES=2,3` |
| 安裝必要套件(容器內) | `pip install -r requirements.txt && apt-get update && apt-get install -y poppler-utils chktex texlive-full` |
| 生成研究想法 (Kimi) | `python ai_scientist/perform_ideation_temp_free.py --workshop-file "ai_scientist/ideas/my_research_topic.md" --model kimi-k3 --max-num-generations 10 --num-reflections 3` |
| 生成研究想法 (Gemma) | `python ai_scientist/perform_ideation_temp_free.py --workshop-file "ai_scientist/ideas/my_research_topic.md" --model gemma-4-31b-it --max-num-generations 10 --num-reflections 3` |
| 啟動實驗與寫作 (Kimi) | `python launch_scientist_bfts.py --load_ideas "ai_scientist/ideas/my_research_topic.json" --add_dataset_ref --model_writeup kimi-k3 --model_citation kimi-k3 --model_review kimi-k3 --model_agg_plots kimi-k3 --num_cite_rounds 20` |
