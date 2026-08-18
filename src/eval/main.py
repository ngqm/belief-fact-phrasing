import os
import json
import argparse
from openai import OpenAI
from src.utils.general import (get_verb_mappings, get_model_response, 
                           load_results, load_answer_distribution,
                           get_all_models)
from src.utils.dataset import Dataset
from src.utils.visualize import plot_performance, plot_answer_distribution, plot_performance_am_x_confident
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm 
import glob


from dotenv import load_dotenv
load_dotenv(override=True) 
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1"


def generate(output_base, prompt_template, verb, model, dataset, max_tokens=512):

    model_output_dir = os.path.join(output_base, prompt_template, verb, model.replace("/", "_"))
    os.makedirs(model_output_dir, exist_ok=True)

    client = OpenAI(base_url=OPENROUTER_URL, api_key=OPENROUTER_API_KEY)
    print(f"Processing '{verb}' for {model} (max_tokens={max_tokens})")


    indices_to_process = []
    for i in range(len(dataset)):
        output_file = os.path.join(model_output_dir, f"{i}.json")
        if not os.path.exists(output_file):
            indices_to_process.append(i)

    n_written = n_skipped = 0
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {
            executor.submit(get_model_response, client, model, dataset['query'][i],
                            max_tokens=max_tokens): i
            for i in indices_to_process
        }
        for future in tqdm(as_completed(futures), total=len(futures)):
            i = futures[future]
            try:
                response = future.result()
            except Exception as exc:
                n_skipped += 1
                continue
            if not response:
                n_skipped += 1
                continue
            try:
                result_object = {
                    "index": i,
                    "verb_tested": verb,
                    "model": model,
                    "prompt_used": dataset['query'][i],
                    "original_data": dataset[i],
                    "ground_truth": dataset["answer"][i],
                    "model_output": response.choices[0].message.content,
                    "full_api_response": response.model_dump()
                }
                output_file = os.path.join(model_output_dir, f"{i}.json")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result_object, f, indent=2, ensure_ascii=False)
                n_written += 1
            except Exception as exc:
                n_skipped += 1
                print(f"  [WARN] failed to write index {i}: {exc}")
    print(f"  wrote={n_written}  skipped={n_skipped}")


def progress(output_base, prompt_template):
    print("Loading and analyzing results...")
    # count total files under each model directory
    if not os.path.isdir(os.path.join(output_base, prompt_template)):
        print(f"You should run the generation first for prompt template '{prompt_template}' before checking progress.")
        return
    verbs = VERB_MAPPINGS.keys()
    for verb in verbs:
        if verb not in os.listdir(os.path.join(output_base, prompt_template)):
            print(f"Verb: {verb} - No data found.")
            continue
        for model in os.listdir(os.path.join(output_base, prompt_template, verb)):
            model_path = os.path.join(output_base, prompt_template, verb, model)
            if os.path.isdir(model_path):
                file_count = len(glob.glob(os.path.join(model_path, "*.json")))
                if file_count != 1000:
                    print(f"Verb: {verb}, Model: {model}, Files: {file_count}/1000")


def evaluate(output_base, analysis_base, visualization_base, prompt_template):

    records = load_results(output_base, prompt_template)
    plot_performance(analysis_base, visualization_base, prompt_template, records)
    plot_performance_am_x_confident(analysis_base, visualization_base, prompt_template, records)
    # answer_records = load_answer_distribution(output_base, prompt_template)
    # plot_answer_distribution(analysis_base, visualization_base, prompt_template, answer_records)
    

if __name__ == "__main__":

    VERB_MAPPINGS = get_verb_mappings()

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["generate", "progress", "evaluate"])
    parser.add_argument("--model", type=str, default="meta-llama/llama-3.1-8b-instruct")
    parser.add_argument("--prompt_template", type=str, default="original", choices=["original", "no_fact_check", "may_or_may_not_fact_check", "must_fact_check", "no_cot"])
    parser.add_argument("--verb", type=str, default="believe", choices=list(VERB_MAPPINGS.keys()))
    parser.add_argument("--dataset_file", type=str,
                        default="data/confirmation-of-first-person-belief.jsonl")
    parser.add_argument("--output_base", type=str, default="logs")
    parser.add_argument("--analysis_base", type=str, default="analysis")
    parser.add_argument("--visualization_base", type=str, default="visualization")
    parser.add_argument("--max_tokens", type=int, default=512)
    args = parser.parse_args()


    if not OPENROUTER_API_KEY:
        raise ValueError("Please set OPENROUTER_API_KEY environment variable.")

    dataset = Dataset(args.verb, args.prompt_template, args.dataset_file)

    if args.mode == "generate":
        if args.model == 'all':
            models = get_all_models()
            for model in models:
                print(f"Starting generation for model: {model}")
                generate(args.output_base, args.prompt_template, args.verb, model, dataset.data, args.max_tokens)
        else:
            generate(args.output_base, args.prompt_template, args.verb, args.model, dataset.data, args.max_tokens)
    elif args.mode == "progress":
        progress(args.output_base, args.prompt_template)
    elif args.mode == "evaluate":
        evaluate(args.output_base, args.analysis_base, args.visualization_base, args.prompt_template)