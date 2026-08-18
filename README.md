# Whether LLMs Can Navigate Beliefs and Facts Depends on How You Phrase It

Code for the EMNLP 2026 (Main) paper [*Whether LLMs Can Navigate Beliefs and Facts Depends
on How You Phrase It*](https://arxiv.org/abs/2608.17809) by Quang Minh Nguyen and Luis Frentzen Salim.

The evaluation builds on the first-person belief confirmation task from
[KaBLE](https://github.com/suzgunmirac/belief-in-the-machine).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in your API keys
```

For local inference, install vLLM
from the nightly wheels (`pip install -U vllm ninja --pre --extra-index-url
https://wheels.vllm.ai/nightly`).
The KaBLE dataset is not redistributed here, and
[`data/README.md`](data/README.md) explains which files are required and where to obtain them.
The artifacts that reproduce the paper's tables and figures are under
`results/` (see [Released data](#released-data)).

## Quickstart

```bash
python -m src.eval.main --mode generate \
  --model meta-llama/llama-3.1-8b-instruct --prompt_template original --verb believe
python -m src.eval.main --mode evaluate \
  --model meta-llama/llama-3.1-8b-instruct --prompt_template original --verb believe
```

`scripts/example_eval.sh` wraps these two commands, and [`quickstart.ipynb`](quickstart.ipynb)
runs a small version of each experiment.

## Experiments

Run every module from the repository root with `python -m`.

**Behavioral evaluation and instruction contrast.**

```bash
python -m src.eval.run_verb_eval_vllm --model llama-3.1-8b   # all verbs, one model
python -m src.plotting.plot_paper_figures    # fig1, fig2
python -m src.eval.instruction_effect_table  # instruction table by verb family
```

**CoT strategy taxonomy.**

```bash
python -m src.taxonomy.build_taxonomy_pool --per_cell 25 --out data/taxonomy/cot_pool.jsonl
python -m src.taxonomy.classify_cot_strategies --pool data/taxonomy/cot_pool.jsonl
python -m src.plotting.plot_cot_taxonomy                    # fig3
python -m src.taxonomy.validate_cot_judge --judge <model>   # kappa vs a second judge
```

**Attention suppression.**

```bash
python -m src.suppression.attention_suppress_decode \
  --model llama-3.1-8b --alphas 0 -0.5 -1 \
  --item_type false --task confirmation \
  --n_items 500 --out results/suppress/llama-3.1-8b_false.json
# repeat with --task verification and --item_type factual, then:
python -m src.suppression.merge_suppress_results BASE.json INCREMENT.json OUT.json
python -m src.suppression.held_out_alpha_split   # alpha on items 0:50, report on 50:500
python -m src.suppression.bootstrap_held_out     # bootstrap standard errors
```

**Answer position control.**

```bash
python -m src.controls.option_permute_control_or --n 100        # OpenRouter models
python -m src.controls.option_permute_control --model qwen-3.5-4b \
  --n_items 100 --out results/option_permute/qwen-3.5-4b.json   # local model
```

## Released data

`results/` contains the artifacts that reproduce the paper's tables and figures:

- `attention_suppress_decode/held_out/`: the suppression outputs for all ten models, which `src.suppression.bootstrap_held_out` reads.
- `option_permute/`: answer-position control results for all ten models.
- `taxonomy/classifications.jsonl`: the 9,000 chain-of-thought strategy labels, which `src.plotting.plot_cot_taxonomy` reads.
- `proposition_attention/`: the per-question attention shares for six of the ten models.

These files quote claims from the MIT-licensed KaBLE dataset, whose license notice is preserved in [`NOTICE`](NOTICE).

## Layout

```
src/
  eval/         epistemic-verb behavioral evaluation, instruction variants, parse-failure
                rerun, instruction table
  controls/     answer position control
  plotting/     paper figures and CoT strategy charts
  suppression/  attention suppression at decoding time, held-out split, bootstrap
  taxonomy/     CoT pool, LLM-judge classification, judge validation
  utils/        model registry, data loading, prompt construction, parsing
```

## License

This repository is released under the MIT License, reproduced in full in [`LICENSE`](LICENSE).

## Citation

```bibtex
@misc{nguyen2026phrasing,
  title         = {Whether LLMs Can Navigate Beliefs and Facts Depends on How You Phrase It},
  author        = {Quang Minh Nguyen and Luis Frentzen Salim},
  year          = {2026},
  eprint        = {2608.17809},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  note          = {Accepted to EMNLP 2026 (Main)}
}
```

Please also cite the KaBLE benchmark which this study is built on:

```bibtex
@article{suzgun2025belief,
  title     = {Language models cannot reliably distinguish belief from knowledge and fact},
  author    = {Suzgun, Mirac and Gur, Tayfun and Bianchi, Federico and Ho, Daniel E and Icard, Thomas and Jurafsky, Dan and Zou, James},
  journal   = {Nature Machine Intelligence},
  volume    = {7},
  number    = {11},
  pages     = {1780--1790},
  year      = {2025},
  doi       = {10.1038/s42256-025-01113-8},
  publisher = {Nature Portfolio}
}
```
