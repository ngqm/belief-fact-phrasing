"""Dataset that maps KaBLE queries to a chosen epistemic verb and prompt template."""
from datasets import load_dataset
import json 
from src.utils.general import get_verb_mappings, apply_verb


class Dataset:
    """KaBLE dataset mapped to one epistemic verb and one prompt template."""

    def __init__(self, verb, prompt_template,
                 dataset_file="data/confirmation-of-first-person-belief.jsonl"):
        """Load prompt templates and build the verb-mapped dataset.

        Args:
            verb: a verb key (see get_verb_mappings).
            prompt_template: a key into data/prompt_templates.json.
            dataset_file: path to the JSONL dataset.
        """

        self.verb = verb
        self.dataset_file = dataset_file
        with open("data/prompt_templates.json", 'r', encoding='utf-8') as f:
            prompt_templates = json.load(f)
        self.system_prompt = prompt_templates[prompt_template]
        self.load()

    def load(self):
        """Load the JSONL dataset and rewrite each query for the chosen verb and template."""

        VERB_MAPPINGS = get_verb_mappings()

        self.data = load_dataset("json", data_files=self.dataset_file, split="train")
        def map_prompt(example):
            mapping = VERB_MAPPINGS.get(self.verb, {})
            example['query'] = apply_verb(example['query'], mapping) + "\n\n" + self.system_prompt
            return example 
        self.data = self.data.map(map_prompt)
        print("Loaded dataset with verb:", self.verb)
        print(f"Example prompt:\n\n{self.data[0]['query']}")
