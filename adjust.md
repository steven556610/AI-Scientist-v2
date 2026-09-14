# AI-Scientist-v2 Adjustments for Kimi-K3 / Gemma (DGX Internal Endpoints)

本文件記錄為了讓 AI-Scientist-v2 在 **DGX 節點（aarch64 / ARM64 架構）** 上，使用 **Kimi-K3**（DGX 內部端點）和 **Gemma-4-31B**（Foxconn 內部端點）進行研究所做的全部修改。

**修改日期**：2026-09-01
**修改環境**：DGX GB200 (aarch64)、Miniforge3 conda env `ai_scientist_v2`

---

## 1. `ai_scientist/llm.py` — 新增模型支援（Ideation / Writeup 階段）

### 修改原因
原始 `AVAILABLE_LLMS` 名單中沒有 `kimi-k3` 與 `gemma-4-31b-it`，導致傳入 `--model kimi-k3` 時直接報錯。此外，`create_client()` 中也沒有對應的路由邏輯。

### 修改內容

**1. `AVAILABLE_LLMS` 名單末尾新增：**
```python
# Kimi-K3 via DGX internal endpoint
"kimi-k3",
# Gemma via Foxconn internal endpoint
"gemma-4-31b-it",
```

**2. `get_batch_responses_from_llm()` 新增分支：**
```python
elif model in ["kimi-k3", "gemma-4-31b-it"]:
    new_msg_history = msg_history + [{"role": "user", "content": msg}]
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_message}, *new_msg_history],
        temperature=temperature,
        max_tokens=MAX_NUM_TOKENS,
        n=n_responses,
        stop=None,
    )
    content = [r.message.content for r in response.choices]
    new_msg_history = [new_msg_history + [{"role": "assistant", "content": c}] for c in content]
```

**3. `get_response_from_llm()` 新增分支：**
```python
elif model in ["kimi-k3", "gemma-4-31b-it"]:
    new_msg_history = msg_history + [{"role": "user", "content": msg}]
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_message}, *new_msg_history],
        temperature=temperature,
        max_tokens=MAX_NUM_TOKENS,
        n=1,
    )
    content = response.choices[0].message.content
    new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
```

**4. `create_client()` 新增分支：**
```python
elif model == "kimi-k3":
    return (
        openai.OpenAI(
            api_key=os.environ.get("KIMI_API_KEY", "none"),
            base_url=os.environ.get("KIMI_API_BASE", "http://r04dgx05:8000/v1"),
        ),
        os.environ.get("KIMI_MODEL", "kimi-k3"),
    )
elif model == "gemma-4-31b-it":
    return (
        openai.OpenAI(
            api_key=os.environ.get("GEMMA_API_KEY", "none"),
            base_url=os.environ.get("GEMMA_API_BASE", "https://afspod-services.ai.foxconn.com/.../gemma-api/v1"),
        ),
        os.environ.get("GEMMA_MODEL", "gemma-4-31b-it"),
    )
```

---

## 2. `ai_scientist/perform_ideation_temp_free.py` — 修正 JSON 解析（多輸出格式相容）

### 修改原因
Kimi-K3 被要求輸出 `ACTION: ... ARGUMENTS: ...` 格式時，可能一次回傳**多個 ACTION/ARGUMENTS 組合**，而原始 `json.loads()` 無法處理第一個 JSON 物件後的額外內容，導致 `JSONDecodeError: Extra data`。

### 修改內容（約第 181 行起）

1. **改進 `arguments_pattern` regex**，加入 `\n\nACTION` 作為結束邊界：
   ```python
   arguments_pattern = r"ARGUMENTS:\s*(.*?)(?:\n\n(?:ACTION|THOUGHT)|$)"
   ```

2. **改用 `json.JSONDecoder().raw_decode()`** 只解析第一個完整 JSON 物件：
   ```python
   try:
       decoder = json.JSONDecoder()
       arguments_json, _ = decoder.raw_decode(arguments_text)
   except json.JSONDecodeError:
       raise ValueError(f"Invalid arguments JSON for {action}.")
   ```

---

## 3. `ai_scientist/tools/semantic_scholar.py` — 確認 Rate Limit 問題

### 問題說明
無 `S2_API_KEY` 時，Semantic Scholar 對匿名請求的速率限制極嚴格（約每分鐘數次）。原始 `SemanticScholarSearchTool.search_for_papers()`（class method）缺少 `time.sleep(1.0)`，而 standalone 的 `search_for_papers()` 函式已有。

> **⚠️ 注意**：如果再度遇到 `429 Too Many Requests`，可在 class method 中加回：
> ```python
> # 在 rsp.raise_for_status() 之後，total == 0 判斷之前加入
> time.sleep(1.0)  # 每次請求後等 1 秒
> ```

**根本解決方案**：申請免費 Semantic Scholar API Key 並設定：
```bash
export S2_API_KEY="your_key"
```
申請網址：https://www.semanticscholar.org/product/api#api-key-form

---

## 4. `bfts_config.yaml` — 實驗 Agent 模型改為 Kimi-K3

### 修改原因
BFTS Tree Search 的實驗 Agent（程式碼撰寫、除錯、評估）使用 `bfts_config.yaml` 中定義的模型，**與 CLI 的 `--model_writeup` 完全獨立**。原始設定寫死了 AWS Bedrock Claude，導致沒有 AWS 憑證時報錯。

### 修改內容

```yaml
# 修改前
report:
  model: gpt-4o-2024-11-20

agent:
  code:
    model: anthropic.claude-3-5-sonnet-20241022-v2:0
  feedback:
    model: gpt-4o-2024-11-20
  vlm_feedback:
    model: gpt-4o-2024-11-20

# 修改後
report:
  model: kimi-k3

agent:
  code:
    model: kimi-k3
  feedback:
    model: kimi-k3
  vlm_feedback:
    model: kimi-k3
```

---

## 5. `ai_scientist/treesearch/backend/backend_openai.py` — 新增 Kimi/Gemma 路由

### 修改原因
Tree Search 的 backend 路由判斷：`claude-` → Anthropic Bedrock；其他 → `openai.OpenAI()`（使用預設 `OPENAI_API_KEY`）。`kimi-k3` 需要自訂 `api_key` 和 `base_url`，否則會發送到 OpenAI 官方 API 並收到 401。

### 修改內容（`get_ai_client()` 函式中新增）

```python
elif model == "kimi-k3":
    import os
    client = openai.OpenAI(
        api_key=os.environ.get("KIMI_API_KEY", "none"),
        base_url=os.environ.get("KIMI_API_BASE", "http://r04dgx05:8000/v1"),
        max_retries=max_retries,
    )
elif model == "gemma-4-31b-it":
    import os
    client = openai.OpenAI(
        api_key=os.environ.get("GEMMA_API_KEY", "none"),
        base_url=os.environ.get("GEMMA_API_BASE", "https://afspod-services.ai.foxconn.com/.../gemma-api/v1"),
        max_retries=max_retries,
    )
```

---

## 6. 新增研究主題檔案

### `ai_scientist/ideas/my_research_topic.md`（新增）
自訂的研究主題，涵蓋**機器人、自動駕駛、深度學習模型架構、機器視覺、模式辨識、資料探勘**，以鴻海研究院的核心方向為目標，鎖定 CVPR、ICCV、NeurIPS、ICRA、IROS、IEEE T-PAMI 等頂會。

---

## 7. 啟動容器正確指令（環境變數傳遞）

### 重要：需傳遞 `KIMI_*` 而非 `OPENAI_*`

```bash
cd /home/hhri-ai/hh30990/ai_for_research/AI-Scientist-v2

# 讀取 .env
export $(grep -v '^#' .env | xargs)

# 啟動容器（傳入 KIMI 和 GEMMA 變數）
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
  ai_scientist_v2 bash
```

---

## 8. 常見問題快速對照

| 錯誤訊息 | 原因 | 解法 |
| :--- | :--- | :--- |
| `invalid choice: 'openai/kimi-k3'` | 模型名稱應為 `kimi-k3`，不含 `openai/` 前綴 | 使用 `--model kimi-k3` |
| `401 - Unauthorized` | 容器內 `KIMI_API_KEY` 未設定 | 確認 `enroot start` 有傳入 `--env KIMI_API_KEY=$KIMI_API_KEY`，或在容器內 `export KIMI_API_KEY=...` |
| `429 Too Many Requests` (S2) | 無 `S2_API_KEY`，速率限制嚴格 | 申請 S2 API Key 或降低 `--max-num-generations` |
| `JSONDecodeError: Extra data` | Kimi 一次回傳多個 ACTION/ARGUMENTS | 已修正於 `perform_ideation_temp_free.py` |
| `ValueError: No AWS region` | `bfts_config.yaml` 模型為 Claude Bedrock | 已修正於 `bfts_config.yaml`，改為 `kimi-k3` |
| `Code path ... must exist` | `--load_code` 需要對應的 `.py` 起始程式碼 | 移除 `--load_code` 旗標（全新研究不需要） |
| `conda install pytorch-cuda=12.4` 失敗 | DGX 為 aarch64，conda 無此架構的套件 | 改用 `pip install torch --index-url https://download.pytorch.org/whl/cu130` |
