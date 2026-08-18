from datasets import load_dataset
import json 
from src.utils.general import get_verb_mappings


class Dataset:

    def __init__(self, verb, prompt_template,
                 dataset_file="data/confirmation-of-first-person-belief.jsonl"):

        self.verb = verb
        self.dataset_file = dataset_file
        with open("data/prompt_templates.json", 'r', encoding='utf-8') as f:
            prompt_templates = json.load(f)
        self.system_prompt = prompt_templates[prompt_template]
        self.load()

    def load(self):

        VERB_MAPPINGS = get_verb_mappings()

        self.data = load_dataset("json", data_files=self.dataset_file, split="train")
        def map_prompt(example):
            mapping = VERB_MAPPINGS.get(self.verb, {})
            example['query'] = example['query'].replace("Do I believe", mapping["question_form"])
            example['query'] = example['query'].replace("I believe", mapping["statement_form"])
            example['query'] = example['query'] + "\n\n" + self.system_prompt
            return example 
        self.data = self.data.map(map_prompt)
        print("Loaded dataset with verb:", self.verb)
        print(f"Example prompt:\n\n{self.data[0]['query']}")
