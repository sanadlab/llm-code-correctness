# `hidden_states/` — cached hidden states for correctness analysis

For a guide to **how we collect** hidden states and **how
verification** works, see [README.md](hidden_states/README.md).

This directory holds the cached hidden states we analyze in Phase 2. Each cache
is one `bundle.pt` (the data) plus a `metadata.json` (provenance), written by
`scripts/extract/cache_hidden_states.py`.

A **task** is a problem (e.g. a BigCodeBench problem). An **implementation** is
one candidate solution to that task. Every implementation is labeled **correct**
(1) or **incorrect** (0) by whether it passes the task's tests.
A bundle stores one pooled hidden-state vector per implementation, per layer.

> **Terminology:** in code, an implementation is called a `sample` (hence
> `sample_ids`), and "correct/incorrect" is stored as the integer `label`
> (`1` = correct, `0` = incorrect).

---

## The runs this document describes

This doc covers **only** the standard extraction we run. Every run uses the same
command and changes **only the model or benchmark**.

To run all four models × two benchmarks (with verification), use:

```bash
make collect-hidden-states
```

Or invoke the script directly for one model/benchmark pair:

```bash
python scripts/extract/cache_hidden_states.py \
  --models <hf-model-id> \
  --dataset-source icse26 --benchmark <bigcodebench or humaneval> \
  --protocol function \
  --data-splits validate test \
  --representations response_avg \
  --verify
```

---

## Where a bundle lives

```
hidden_states/<model_dir>/icse26/<benchmark>/id/<split>/neutral/fold_<k>/
    bundle.pt        # the hidden states
    metadata.json    # provenance (not loaded by analysis code)
```

Every path segment except `<model_dir>`, `<split>`, `<benchmark>`, and `<k>` is a constant for these runs.

## What's inside `bundle.pt`

Load it with `HiddenStateBundle.load(path)` (or `torch.load(path,
weights_only=False)`). It has four aligned, parallel fields of length **N**,
where **N** is the number of implementations in that fold's split:

```python
{
    "sample_ids": list[str],     # length N — one id per implementation
    "task_ids":   list[str],     # length N — the parent task of each implementation
    "labels":     torch.Tensor,  # shape [N], int64 — 1 = correct, 0 = incorrect
    "acts": {
        "response_avg": torch.Tensor,  # shape [N, L+1, H], float32
    },
}
```

Row `i` is the same implementation across all four fields: `sample_ids[i]`,
`task_ids[i]`, `labels[i]`, and `acts["response_avg"][i]` all refer to it.

For each layer, `response_avg` is the **mean hidden state over the response
tokens** — the inclusive token span `[prompt_end, sequence_end]`. Under the
`neutral` framing the response is the candidate code, so this is an average over
the code tokens (see `ResponseAvg` in `src/representations/pooling.py`).

The tensor is stored as **float32** even though the model runs in float16; the
extraction casts to float32 before saving. (The stored values still carry
float16 precision)

---

## An example for clarity

A validate cache for CodeLlama, fold 0, where the split happens to contain
**N = 4** implementations across 2 tasks:

```python
from src.hidden_states import HiddenStateBundle

bundle = HiddenStateBundle.load(
    "hidden_states/codellama__CodeLlama-7b-Instruct-hf/"
    "icse26/bigcodebench/id/validate/neutral/fold_0/bundle.pt"
)

bundle.sample_ids
# ['BigCodeBench/3__impl_a', 'BigCodeBench/3__impl_b',
#  'BigCodeBench/7__impl_a', 'BigCodeBench/7__impl_b']

bundle.task_ids
# ['BigCodeBench/3', 'BigCodeBench/3', 'BigCodeBench/7', 'BigCodeBench/7']

bundle.labels
# tensor([1, 0, 1, 0])          # impl_a correct, impl_b incorrect, per task

bundle.acts.keys()
# dict_keys(['response_avg'])

bundle.acts["response_avg"].shape
# torch.Size([4, 33, 4096])     # [N, L+1, H]

bundle.acts["response_avg"].dtype
# torch.float32

# Layer 0 is the embedding output; layer 32 is the last transformer block.
vec = bundle.acts["response_avg"][0, 18, :]   # impl_a at layer 18 → shape [4096]
```

The matching **test** cache at `.../test/neutral/fold_0/bundle.pt` has the same
schema; only `N` and the ids differ (they come from `test.jsonl`).

---

## Companion `metadata.json`

Written next to `bundle.pt` for provenance; the analysis code does not read it.
For the example above:

```json
{
  "model": "codellama/CodeLlama-7b-Instruct-hf",
  "dataset": "bigcodebench",
  "split_family": "icse26",
  "distribution": "id",
  "data_split": "validate",
  "framing": "neutral",
  "representations": ["response_avg"],
  "acts_dtype": "float32",
  "chat_template": true,
  "dtype": "float16",
  "max_length": 4096,
  "n_samples": 4,
  "source_jsonl": "dataset/icse26/bigcodebench/fold_0/validate.jsonl",
  "bundle_path": "hidden_states/.../bundle.pt",
  "sample_ids_hash": "<16-char sha256 prefix of newline-joined sample_ids>",
  "fold": 0,
  "experiment_script": "scripts/extract/cache_hidden_states.py",
  "git_commit": "<HEAD sha>"
}
```

`n_samples` equals `len(bundle.sample_ids)` and
`bundle.acts["response_avg"].shape[0]`.
