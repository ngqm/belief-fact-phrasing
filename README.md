# Whether LLMs Can Navigate Beliefs and Facts Depends on How You Phrase It

Code for the paper [*Whether LLMs Can Navigate Beliefs and Facts Depends on How You Phrase
It*](https://arxiv.org/abs/2608.17809) by Quang Minh Nguyen and Luis Frentzen Salim.

The evaluation builds on the first-person belief confirmation task from KaBLE: each item
asks "I believe X. Do I believe X?"; the correct answer is always "Yes", independent
of the truth of X.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in your API keys
```

See `.env.example` for which keys each path needs. For local inference, install vLLM
from the nightly wheels (`pip install -U vllm ninja --pre --extra-index-url
https://wheels.vllm.ai/nightly`).
Datasets are not redistributed;
[`data/README.md`](data/README.md) lists the files the code expects and where to obtain them.

## Quickstart

```bash
python -m src.eval.main --mode generate \
  --model meta-llama/llama-3.1-8b-instruct --prompt_template original --verb believe
python -m src.eval.main --mode evaluate \
  --model meta-llama/llama-3.1-8b-instruct --prompt_template original --verb believe
```

`scripts/example_eval.sh` wraps these two commands; [`quickstart.ipynb`](quickstart.ipynb)
runs a small version of each experiment end to end.

## Experiments

Run every module from the repository root with `python -m`.

**Behavioral evaluation and instruction contrast.** `--verb` selects the expression,
`--prompt_template` the instruction condition (`original`, `no_fact_check`,
`may_or_may_not_fact_check`, `must_fact_check`, `no_cot`).
`src.eval.run_18verb_vllm` runs all 18 verbs for one model in a single process
(`run_18verb_batched` is a plain transformers fallback); `src.eval.rerun_parse_failures`
retries unparsed items. Then:

```bash
python -m src.plotting.plot_paper_figures    # fig1, fig2
python -m src.eval.compute_table1_extended   # instruction table by verb family
```

**CoT strategy taxonomy.** Categories are `CATEGORIES` in
`src/taxonomy/classify_cot_strategies.py`.

```bash
python -m src.taxonomy.build_taxonomy_pool --per_cell 25 --out data/taxonomy/cot_pool.jsonl
python -m src.taxonomy.classify_cot_strategies --pool data/taxonomy/cot_pool.jsonl
python -m src.plotting.plot_cot_taxonomy                    # fig3
python -m src.taxonomy.validate_cot_judge --judge <model>   # kappa vs a second judge
```

**Attention suppression.** Model names are the registry keys in `src/utils/models.py`.

```bash
python -m src.suppression.attention_suppress_decode \
  --model llama-3.1-8b --alphas 0 -0.5 -1 \
  --item_type false --task confirmation \
  --n_items 150 --out results/suppress/llama-3.1-8b_false.json
# repeat with --task verification and --item_type factual, then:
python -m src.suppression.merge_suppress_results BASE.json INCREMENT.json OUT.json
python -m src.suppression.held_out_alpha_split   # alpha on items 0:50, report on 50:150
python -m src.suppression.bootstrap_held_out     # bootstrap standard errors
```

**Answer position control.**

```bash
python -m src.controls.option_permute_control_or --n 100        # OpenRouter models
python -m src.controls.option_permute_control --model qwen-3.5-4b \
  --n_items 100 --out results/option_permute/qwen-3.5-4b.json   # local model
```

## Layout

```
src/
  eval/         18-verb behavioral evaluation, instruction variants, parse-failure
                rerun, instruction table
  controls/     answer position control
  plotting/     paper figures and CoT strategy charts
  suppression/  attention suppression at decoding time, held-out split, bootstrap
  taxonomy/     CoT pool, LLM-judge classification, judge validation
  utils/        model registry, data loading, prompt construction, parsing
```

## License

MIT. See [`LICENSE`](LICENSE).

## Citation

```bibtex
@misc{nguyen2026phrasing,
  title         = {Whether LLMs Can Navigate Beliefs and Facts Depends on How You Phrase It},
  author        = {Quang Minh Nguyen and Luis Frentzen Salim},
  year          = {2026},
  eprint        = {2608.17809},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}
```
