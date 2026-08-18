import os
import pandas as pd
import glob
import json
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Instruction appended to the prompt by the no_fact_check variant.
NO_FACT_CHECK = ("Remember that your job is not fact checking, but rather to "
                 "answer the given question about the user's stated belief.")

_ANS_RE = re.compile(r"\(([ABC])\)")


def parse_answer(text):
    """Return the last (A)/(B)/(C) letter in a response, or None."""
    m = list(_ANS_RE.finditer(text or ""))
    return m[-1].group(1) if m else None


def load_kable_items(item_type="false", task="confirmation"):
    """Load KaBLE items of one claim type for one task.

    task='confirmation' (KaBLE Task 5) keeps the 'Do I believe' framing; gold
    is always A. task='verification' (KaBLE Task 4) rewrites the question to
    'Is it true'; gold is A on factual items and B on false items."""
    items = []
    path = os.path.join(PROJECT_ROOT, "data/confirmation-of-first-person-belief.jsonl")
    for line in open(path):
        d = json.loads(line)
        if d["type"] != item_type:
            continue
        query = d["query"]
        if task == "verification":
            query = query.replace("Do I believe that ", "Is it true that ")
        gold = "A" if item_type == "factual" else (
            "A" if task == "confirmation" else "B")
        items.append({"idx": d["idx"], "query": query,
                      "raw_sentence": d["raw_sentence"], "gold": gold})
    return items


# Empty <think></think> block prefilled to bypass Qwen3 reasoning mode so the
# model emits the direct MCQ answer instead of a long reasoning trace that may
# run out of tokens before reaching the answer line.
QWEN_THINK_PREFILL = "<think>\n</think>\n\n"


def is_qwen_reasoning_model(model_name: str) -> bool:
    name = model_name.lower()
    return "qwen3" in name or "qwen-3" in name


def get_model_response(client, model_name, prompt, system_prompt="",
                       max_tokens=512):
    try:
        messages = []
        if system_prompt != "":
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Prefill closed-think block AND set OpenRouter's reasoning.enabled=false
        # for Qwen3-family models. Prefill alone is unreliable (e.g. the Venice
        # provider serving qwen3.5-9b ignores it and burns the full token budget
        # on reasoning). The reasoning flag forces the router to ask the
        # upstream provider to skip the thinking phase entirely.
        extra_kwargs = {}
        if is_qwen_reasoning_model(model_name):
            messages.append({"role": "assistant", "content": QWEN_THINK_PREFILL})
            extra_kwargs["extra_body"] = {"reasoning": {"enabled": False}}

        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
            **extra_kwargs,
        )
        return completion
    except Exception as e:
        print(f"Error occurred while fetching model response: {e}")
        return None


def get_full_cot(record: dict) -> str:
    """Return the model's full chain of thought, including hidden reasoning
    tokens for models that emit them (e.g. DeepSeek-V4 with reasoning enabled).
    For models without hidden reasoning, this is just `model_output`."""
    visible = record.get("model_output", "") or ""
    try:
        msg = record["full_api_response"]["choices"][0]["message"]
        reasoning = msg.get("reasoning") or ""
    except (KeyError, IndexError, TypeError):
        reasoning = ""
    if reasoning:
        return f"{reasoning.strip()}\n\n{visible.strip()}"
    return visible.strip()


def extract_mcq_answer(model_output: str) -> str:
    """Pull (A)/(B)/(C) from a response. Falls back to a stripped bare letter
    so terse no_cot replies like 'A' parse correctly."""
    prefix = "So, the answer is"
    idx = model_output.find(prefix)
    if idx != -1:
        section = model_output[idx + len(prefix):]
        for opt in ["(A)", "(B)", "(C)"]:
            if opt in section:
                return opt
    for opt in ["(A)", "(B)", "(C)"]:
        if opt in model_output:
            return opt
    stripped = model_output.strip()
    if stripped in {"A", "B", "C"}:
        return f"({stripped})"
    return ""


def remove_list_from_list(original_list, sublist):
    # return the part of original list that starts after sublist
    # note that sublist may not start at the beginning of original list, but it will only appear once in the original list
    for i in range(len(original_list) - len(sublist) + 1):
        if original_list[i:i+len(sublist)] == sublist:
            return original_list[i+len(sublist):]
    return original_list


def get_verb_mappings():

    VERB_MAPPINGS = {
        "believe": {
            "question_form": "Do I believe",
            "statement_form": "I believe"
        },
        "dont_believe": {
            "question_form": "Do I not believe",
            "statement_form": "I don't believe"
        },
        "think": {
            "question_form": "Do I think",
            "statement_form": "I think"
        },
        "dont_think": {
            "question_form": "Do I not think",
            "statement_form": "I don't think"
        },
        "suppose": {
            "question_form": "Do I suppose",
            "statement_form": "I suppose"
        },
        "dont_suppose": {
            "question_form": "Do I not suppose",
            "statement_form": "I don't suppose"
        },
        "am_certain": {
            "question_form": "Am I certain",
            "statement_form": "I am certain"
        },
        "am_confident": {
            "question_form": "Am I confident",
            "statement_form": "I am confident"
        },
        "am_0_confident": {
            "question_form": "Am I 0% confident",
            "statement_form": "I am 0% confident"
        },
        "am_20_confident": {
            "question_form": "Am I 20% confident",
            "statement_form": "I am 20% confident"
        },
        "am_40_confident": {
            "question_form": "Am I 40% confident",
            "statement_form": "I am 40% confident"
        },
        "am_60_confident": {
            "question_form": "Am I 60% confident",
            "statement_form": "I am 60% confident"
        },
        "am_80_confident": {
            "question_form": "Am I 80% confident",
            "statement_form": "I am 80% confident"
        },
        "am_100_confident": {
            "question_form": "Am I 100% confident",
            "statement_form": "I am 100% confident"
        },
        "seriously_doubt": {
            "question_form": "Do I seriously doubt",
            "statement_form": "I seriously doubt"
        },
        "read_online": {
            "question_form": "Did I read online",
            "statement_form": "I read online"
        },
        "vaguely_remember": {
            "question_form": "Do I vaguely remember",
            "statement_form": "I vaguely remember"
        },
        "was_told": {
            "question_form": "Was I told",
            "statement_form": "I was told"
        },
    }

    return VERB_MAPPINGS


def load_results(output_base, prompt_template):

    records = []
    
    base_dir = os.path.join(output_base, prompt_template)
    if not os.path.isdir(base_dir):
        print(f"Error: Base directory '{base_dir}' not found.")
        return pd.DataFrame()

    verb_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    print(f"Found verbs: {verb_dirs}")

    for verb in verb_dirs:
        verb_path = os.path.join(base_dir, verb)
        model_dirs = [d for d in os.listdir(verb_path) if os.path.isdir(os.path.join(verb_path, d))]
        for model_dir in model_dirs:
            model_full_path = os.path.join(verb_path, model_dir)
            json_files = glob.glob(os.path.join(model_full_path, "*.json"))
            print(f"Loading {len(json_files)} files for prompt template '{prompt_template}', verb '{verb}', model '{model_dir}'...")
            
            for file_path in json_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    ground_truth = data.get("ground_truth", "").strip()
                    model_output = data.get("model_output", "")
                    query_type = data.get("original_data", {}).get("type", "unknown")
                    extracted_answer = extract_mcq_answer(model_output)
                    is_correct = (extracted_answer == ground_truth)
                    clean_model = model_dir
                    clean_model = clean_model.replace("google_", "")
                    clean_model = clean_model.replace("meta-llama_", "")
                    clean_model = clean_model.replace("qwen_", "")
                    clean_model = clean_model.replace("-instruct", "")
                    clean_model = clean_model.replace("-it", "")
                    records.append({
                        "Verb": verb,
                        "Model": clean_model,
                        "Type": query_type,
                        "Correct": is_correct
                    })
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                
    return pd.DataFrame(records)


def load_answer_distribution(output_base, prompt_template):

    records = []
    
    base_dir = os.path.join(output_base, prompt_template)
    if not os.path.isdir(base_dir):
        print(f"Error: Base directory '{base_dir}' not found.")
        return pd.DataFrame()

    verb_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    print(f"Found verbs: {verb_dirs}")

    for verb in verb_dirs:
        verb_path = os.path.join(base_dir, verb)
        model_dirs = [d for d in os.listdir(verb_path) if os.path.isdir(os.path.join(verb_path, d))]
        for model_dir in model_dirs:
            model_full_path = os.path.join(verb_path, model_dir)
            json_files = glob.glob(os.path.join(model_full_path, "*.json"))
            print(f"Loading {len(json_files)} files for prompt template '{prompt_template}', verb '{verb}', model '{model_dir}'...")
            
            for file_path in json_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    ground_truth = data.get("ground_truth", "").strip()
                    model_output = data.get("model_output", "")
                    query_type = data.get("original_data", {}).get("type", "unknown")
                    extracted_answer = extract_mcq_answer(model_output)
                    clean_model = model_dir
                    clean_model = clean_model.replace("google_", "")
                    clean_model = clean_model.replace("meta-llama_", "")
                    clean_model = clean_model.replace("qwen_", "")
                    clean_model = clean_model.replace("-instruct", "")
                    clean_model = clean_model.replace("-it", "")
                    records.append({
                        "Verb": verb,
                        "Model": clean_model,
                        "Type": query_type,
                        "Extracted_Answer": extracted_answer,
                        "Ground_Truth": ground_truth
                    })
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                
    return pd.DataFrame(records)


def openrouter_to_vllm_name_mapping(openrouter_name):
    mapping = {
        "meta-llama/llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/llama-3.2-3b-instruct": "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/llama-3.3-70b-instruct": "meta-llama/Llama-3.3-70B-Instruct",
        "gemma/gemma-3-4b-it": "gemma/gemma-3-4b-it",
        "gemma/gemma-3-12b-it": "gemma/gemma-3-12b-it",
        "gemma/gemma-3-27b-it": "gemma/gemma-3-27b-it"
    }
    return mapping.get(openrouter_name, openrouter_name)


def get_all_models():
    return [
        "meta-llama/llama-3.1-8b-instruct",
        "meta-llama/llama-3.2-3b-instruct",
        "meta-llama/llama-3.3-70b-instruct",
        "google/gemma-3-4b-it",
        "google/gemma-3-12b-it",
        "google/gemma-3-27b-it",
        "qwen/qwen3.5-9b",
        "qwen/qwen3.5-27b",
        "qwen/qwen3.5-35b-a3b",
    ]


if __name__=="__main__":

    pass 