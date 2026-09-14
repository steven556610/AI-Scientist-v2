import os
import argparse
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
from transformers import AutoModelForCausalLM, AutoTokenizer

'''
usage:
    conda activate ai_scientist_v2
    CUDA_VISIBLE_DEVICES=1 python src/serve_coder.py --port 8002

The model is loaded at import time, before argparse runs, so its memory
behaviour is controlled by environment variables rather than CLI flags:

    CUDA_VISIBLE_DEVICES   which physical GPU(s) this server may touch. ALWAYS
                           set it. Without it device_map spreads the model over
                           every card on the node, including the ones running
                           experiments.
    SERVE_MAX_MEM          hard cap on weights+cache per visible GPU,
                           e.g. "150GiB". Default: no cap (uses the whole card).
    SERVE_MAX_NEW_TOKENS   ceiling on generation length. The KV cache is what
                           grows at request time and OOMs a server that loaded
                           fine. Default 12288, which matches
                           bfts_config.yaml agent.code.max_tokens (12000) --
                           setting it lower TRUNCATES generated code and breaks
                           tree-search nodes. ~0.3 MB/token for this model, so
                           12288 new tokens is only ~4 GiB of cache.

Weights are ~136 GB in bf16, so this needs a card to itself.
'''

app = FastAPI(title="Local Qwen2.5-Coder-72B OpenAI-Compatible API")

MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model")

MAX_NEW_TOKENS_CAP = int(os.environ.get("SERVE_MAX_NEW_TOKENS", 12288))
_n_gpu = torch.cuda.device_count()
_max_mem = os.environ.get("SERVE_MAX_MEM")
# Indices here are *visible* ones: with CUDA_VISIBLE_DEVICES=1 the only device
# is 0, and it refers to physical GPU 1.
MAX_MEMORY = {i: _max_mem for i in range(_n_gpu)} if _max_mem else None

print(f"Loading {MODEL_ID} on CUDA_VISIBLE_DEVICES="
      f"{os.environ.get('CUDA_VISIBLE_DEVICES', '<unset: ALL GPUS>')} "
      f"({_n_gpu} visible), max_memory={MAX_MEMORY}, "
      f"max_new_tokens cap={MAX_NEW_TOKENS_CAP}...")
if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
    print("WARNING: CUDA_VISIBLE_DEVICES is unset — this server will spread "
          "across every GPU on the node and collide with the experiments.")
os.makedirs(CACHE_DIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,          # transformers 5 renamed torch_dtype -> dtype
    device_map={"": 0} if _n_gpu == 1 else "auto",
    max_memory=MAX_MEMORY,
    cache_dir=CACHE_DIR,
)
model.eval()
print(f"Model loaded successfully! footprint={model.get_memory_footprint()/2**30:.1f} GiB")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 4096
    n: Optional[int] = 1

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    try:
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # The caller asks for up to 12000 tokens (bfts_config agent.code.max_tokens).
        # Clamp it: the KV cache, not the weights, is what OOMs this server.
        max_new = min(req.max_tokens or MAX_NEW_TOKENS_CAP, MAX_NEW_TOKENS_CAP)
        with torch.inference_mode():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new,
                temperature=req.temperature,
                do_sample=True if req.temperature and req.temperature > 0 else False
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        output_text = tokenizer.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return {
            "id": "chatcmpl-local-coder",
            "object": "chat.completion",
            "created": 1234567890,
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": output_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(model_inputs.input_ids[0]),
                "completion_tokens": len(generated_ids_trimmed[0]),
                "total_tokens": len(model_inputs.input_ids[0]) + len(generated_ids_trimmed[0])
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        # A failed request leaves its activations/KV cache behind; without this
        # the first OOM poisons every subsequent request.
        torch.cuda.empty_cache()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port)
