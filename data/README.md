# Data

This directory contains only `prompt_templates.json`, and the other input files
are recreated locally as described below.

## Files

- `data/prompt_templates.json`: the prompt templates for each instruction condition.
- `data/confirmation-of-first-person-belief.jsonl`: the first-person belief confirmation
  items (KaBLE Task 5). Obtain the dataset from the KaBLE repository and convert it to a JSONL
  where each row has the claim and its truth label. `src/utils/dataset.py` defines the
  required fields.
- The model outputs, CoT strategy taxonomy, and attention-suppression result files are
  derived artifacts. The subset that reproduces the paper's tables and figures is under
  `results/` (see the main [README](../README.md#released-data)), and the rest can be
  regenerated with the scripts in `src/`.

## Sources

[KaBLE](https://github.com/suzgunmirac/belief-in-the-machine) (Suzgun et al.) is the
benchmark this study builds on. It is MIT-licensed, and the released
artifacts that quote its claims preserve its notice in [`NOTICE`](../NOTICE).

Place any additional evaluation items you construct in this directory, since the code
resolves paths relative to the repository root.
