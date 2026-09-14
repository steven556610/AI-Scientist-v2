import json
import logging
import time

from .utils import (
    FunctionSpec,
    OutputType,
    opt_messages_to_list,
    backoff_create,
    compile_prompt_to_md,
)
from funcy import notnone, once, select_values
import openai
from rich import print

logger = logging.getLogger("ai-scientist")


OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)

def get_ai_client(model: str, max_retries=2) -> openai.OpenAI:
    if model.startswith("ollama/"):
        client = openai.OpenAI(
            base_url="http://localhost:11434/v1", 
            max_retries=max_retries
        )
    elif model.startswith("local_coder/"):
        client = openai.OpenAI(
            api_key="none",
            base_url="http://localhost:8004/v1", 
            max_retries=max_retries,
            timeout=7200.0
        )
    elif model.startswith("local/"):
        if "coder" in model.lower():
            port = 8004
        else:
            port = 8000 if "vl" in model.lower() else 8002
        client = openai.OpenAI(
            api_key="none",
            base_url=f"http://localhost:{port}/v1", 
            max_retries=max_retries,
            timeout=7200.0
        )
    elif model == "kimi-k3":
        import os
        # Shared vLLM cluster: measured throughput swings between ~0.4 and
        # ~13 tok/s depending on load, so an 8k-token reply can take well over
        # the openai default 600 s timeout. Give it an hour.
        client = openai.OpenAI(
            api_key=os.environ.get("KIMI_API_KEY", "none"),
            base_url=os.environ.get("KIMI_API_BASE", "http://r04dgx05:8000/v1"),
            max_retries=max_retries,
            timeout=float(os.environ.get("KIMI_TIMEOUT", 7200)),
        )
    elif model == "gemma-4-31b-it":
        import os
        client = openai.OpenAI(
            api_key=os.environ.get("GEMMA_API_KEY", "none"),
            base_url=os.environ.get("GEMMA_API_BASE", "https://afspod-services.ai.foxconn.com/967bbcd1-9600-4e4a-a436-de1d096902c5/gemma-api/v1"),
            max_retries=max_retries,
        )
    else:
        client = openai.OpenAI(max_retries=max_retries)
    return client


# Endpoints served by src/serve_coder.py / src/serve_vlm.py accept plain
# chat/completions only: their request model has no `tools` field, so Pydantic
# silently drops it and the model never emits a tool_call. For those we have to
# put the schema in the prompt and parse JSON out of the text ourselves.
TOOL_CALL_UNSUPPORTED_PREFIXES = ("local/", "local_coder/", "ollama/")


def _supports_tool_calls(model: str | None) -> bool:
    return not (model or "").startswith(TOOL_CALL_UNSUPPORTED_PREFIXES)


def _schema_instruction(func_spec: FunctionSpec) -> str:
    return (
        "\n\n---\n"
        f"You MUST answer with a single JSON object and nothing else — no prose, "
        f"no explanation, no markdown fence around it.\n"
        f"Purpose: {func_spec.description}\n"
        f"The object must validate against this JSON Schema:\n"
        f"{json.dumps(func_spec.json_schema, ensure_ascii=False)}\n"
        f"Every key listed under \"required\" must be present. "
        f"Start your reply with {{ and end it with }}."
    )


def _extract_first_json_object(text: str):
    """Pull the first balanced {...} out of *text*. Returns None if there isn't one."""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def parse_func_spec_output(func_spec: FunctionSpec | None, content: str):
    """Best-effort recovery of a func_spec-shaped dict from free-form text."""
    import re

    content = content or ""
    # 1. the whole reply is JSON
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    # 2. a ```json fenced block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    # 3. the first balanced object anywhere in the reply
    parsed = _extract_first_json_object(content)
    if isinstance(parsed, dict):
        return parsed
    # 4. give up and return a shape-correct default for this spec
    return get_fallback_for_spec(func_spec, content)


def get_fallback_for_spec(spec: FunctionSpec | None, text: str) -> dict:
    """Schema-correct placeholder for every FunctionSpec in the codebase.

    Returning the wrong shape here is worse than failing: e.g. handing the
    `analyze_experiment_plots` dict back for a `parse_metrics` call makes every
    node report a null metric, so tree search cannot rank anything and the whole
    run looks like it produced no results.
    """
    name = spec.name if spec else ""
    text = text or ""
    if name == "submit_review" or name == "review":
        # Unparseable feedback must NOT be read as "the code ran fine".
        return {"is_bug": True, "summary": text or "Reviewer response could not be parsed."}
    if name == "plan":
        return {"plan": text, "new_code": ""}
    if name == "parse_metrics":
        return {"valid_metrics_received": False, "metric_names": []}
    if name == "analyze_experiment_plots":
        return {
            "plot_analyses": [],
            "valid_plots_received": False,
            "vlm_feedback_summary": text,
        }
    if name == "select_plots":
        return {"selected_plots": []}
    if name == "select_best_implementation":
        # "" lets the caller fall back to its own metric-based ranking instead of
        # dereferencing a hallucinated node id.
        return {"selected_id": "", "reasoning": text or "No parseable selection returned."}
    if name == "generate_stage_config":
        return {
            "name": "unnamed_stage",
            "description": text or "Stage config could not be parsed.",
            "goals": [],
            "max_iterations": 1,
        }
    if name == "evaluate_stage_progression":
        return {
            "ready_for_next_stage": False,
            "reasoning": text or "Progression response could not be parsed.",
            "recommendations": [],
            "suggested_focus": "",
        }
    if name == "evaluate_stage_completion":
        return {
            "is_complete": False,
            "reasoning": text or "Completion response could not be parsed.",
            "missing_criteria": [],
        }
    if name == "generate_substage_goals":
        return {"goals": text, "sub_stage_name": "unnamed_substage"}
    # Unknown spec: synthesise from the schema so required keys at least exist.
    props = (spec.json_schema.get("properties", {}) if spec else {}) or {}
    empty = {"string": "", "boolean": False, "integer": 0, "number": 0.0,
             "array": [], "object": {}}
    return {k: empty.get(v.get("type"), None) for k, v in props.items()}


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    model_name = model_kwargs.get("model")
    client = get_ai_client(model_name, max_retries=0)
    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore

    if func_spec is not None and not _supports_tool_calls(model_name):
        # Schema goes in the prompt, since the server will drop `tools`.
        # backend/__init__.query compiles dicts to markdown before calling us,
        # but this function is also called directly in tests//tools, so coerce.
        system_message = compile_prompt_to_md(system_message) if system_message else ""
        system_message += _schema_instruction(func_spec)

    messages = opt_messages_to_list(system_message, user_message)

    if func_spec is not None and _supports_tool_calls(model_name):
        filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
        # force the model to use the function
        filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict

    if filtered_kwargs.get("model", "").startswith("ollama/"):
       filtered_kwargs["model"] = filtered_kwargs["model"].replace("ollama/", "")
    elif filtered_kwargs.get("model", "").startswith("local_coder/"):
       filtered_kwargs["model"] = filtered_kwargs["model"].replace("local_coder/", "")
    elif filtered_kwargs.get("model", "").startswith("local/"):
       filtered_kwargs["model"] = filtered_kwargs["model"].replace("local/", "")
    t0 = time.time()
    completion = backoff_create(
        client.chat.completions.create,
        OPENAI_TIMEOUT_EXCEPTIONS,
        messages=messages,
        **filtered_kwargs,
    )
    req_time = time.time() - t0

    choice = completion.choices[0]

    if func_spec is None:
        output = choice.message.content
    else:
        if choice.message.tool_calls:
            assert (
                choice.message.tool_calls[0].function.name == func_spec.name
            ), "Function name mismatch"
            try:
                print(f"[cyan]Raw func call response: {choice}[/cyan]")
                output = json.loads(choice.message.tool_calls[0].function.arguments)
            except json.JSONDecodeError as e:
                logger.error(
                    f"Error decoding the function arguments: {choice.message.tool_calls[0].function.arguments}"
                )
                raise e
        else:
            content = choice.message.content or ""
            output = parse_func_spec_output(func_spec, content)
            if output == get_fallback_for_spec(func_spec, content):
                logger.warning(
                    f"[{func_spec.name}] no JSON recovered from the reply; using the "
                    f"schema-correct fallback. Raw content: {content[:500]}"
                )
            # Fill in any required key the model omitted, so downstream KeyErrors
            # cannot take the whole node down.
            defaults = get_fallback_for_spec(func_spec, content)
            for key in (func_spec.json_schema.get("required") or []):
                if key not in output:
                    logger.warning(f"[{func_spec.name}] missing required key '{key}'; defaulted")
                    output[key] = defaults.get(key)

    usage = getattr(completion, "usage", None)
    in_tokens = getattr(usage, "prompt_tokens", 0) or 0
    out_tokens = getattr(usage, "completion_tokens", 0) or 0

    info = {
        "system_fingerprint": getattr(completion, "system_fingerprint", None),
        "model": completion.model,
        "created": completion.created,
    }

    return output, req_time, in_tokens, out_tokens, info
