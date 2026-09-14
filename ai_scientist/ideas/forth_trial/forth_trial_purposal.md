基於 https://github.com/arcee-ai/MergeKit 的工具和程式碼，製作成模板。

不要使用模型生成的資料，使用https://huggingface.co/datasets/ikala/tmmluplus 的資料進行訓練和評估。
並且要進行基礎能力的測驗 (MMLU, GSM8K, IFEval)，在訓練的過程，通用能力不可以退步。

MoE, dense 模式都要可以

使用 https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-unquantized

架構上偏像是大模型訓練小模型

幫我測驗 mergekit 裡面有沒有模型的model merge 策略、組合，是可以解決超過現有的SOTA。

訓練資料和驗證資料，幫我直接保留在實驗建立時的根目錄/data/ 裡面。

並且要有足夠多的實驗，至少16*3 = 48 個以上的實驗組合，每一個組合都要可以在結束後重現。要記錄下所有實驗過程