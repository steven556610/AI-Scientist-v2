#!/bin/bash

# 自動切換到專案根目錄，確保找得到 main.py 與相對路徑的 JSON

# 執行方式變成 bash /workspace/experiment_config/second_trial.sh
cd /workspace/ || exit

# 舊指令（已註解）：
# for i in {0..4}; do
#   echo "=================================================="
#   echo " 🚀 正在啟動實驗：第 $((i+1)) 個 Idea (索引 $i)"
#   echo "=================================================="
#   
#   python main.py \
#     --load_ideas "ai_scientist/ideas/second_trial/second_trial.json" \
#     --idea_idx $i \
#     --model_writeup local/qwen-72b \
#     --model_citation local/qwen-72b \
#     --model_review local/qwen2.5-vl-32b \
#     --model_agg_plots local/qwen2.5-vl-32b \
#     --num_cite_rounds 20
#     
# done

# ==============================================================================
# 新指令：接續執行先前中斷的 Idea 0，並接著跑完剩下的 Idea (1~4)
# ==============================================================================

echo "=================================================="
echo " 🔄 [1/2] 接續執行中斷的嘗試：Idea 0 (crosslorauniversalmerging)"
echo "=================================================="

python main.py \
  --resume_from "experiments/crosslorauniversalmerging/2026-09-04_08-21-51_attempt_0" \
  --model_writeup local/qwen-72b \
  --model_citation local/qwen-72b \
  --model_review local/qwen2.5-vl-32b \
  --model_agg_plots local/qwen2.5-vl-32b \
  --num_cite_rounds 20

echo "=================================================="
echo " 🚀 [2/2] 繼續執行後續剩餘實驗：第 2 到 5 個 Idea (索引 1..4)"
echo "=================================================="

for i in {1..4}; do
  echo "=================================================="
  echo " 🚀 正在啟動實驗：第 $((i+1)) 個 Idea (索引 $i)"
  echo "=================================================="
  
  python main.py \
    --load_ideas "ai_scientist/ideas/second_trial/second_trial.json" \
    --idea_idx $i \
    --model_writeup local/qwen-72b \
    --model_citation local/qwen-72b \
    --model_review local/qwen2.5-vl-32b \
    --model_agg_plots local/qwen2.5-vl-32b \
    --num_cite_rounds 20
    
done
