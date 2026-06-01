# Hidden states: collection and verification

This folder stores **cached hidden states** from our models. Each file is a
snapshot of internal model activations for many code implementations (correct and
incorrect), saved so we can analyze them later without running the full model
again.

For **where files live** and **what is inside `bundle.pt`**, see
[STRUCTURE.md](STRUCTURE.md).

---

## What we collect

For each model, benchmark (BigCodeBench or HumanEval), fold, and split
(validate or test), we run the model on every implementation in that split and
save one vector per implementation per layer.

**Steps (in order):**

1. **Load the data** from the dataset JSONL files (e.g.
  `dataset/icse26/bigcodebench/fold_0/validate.jsonl`). Each row is one
   implementation with its text, label (correct or incorrect), and ids.
2. **Build the prompt** using the chosen framing (for standard runs:
  `neutral` — the model sees the task and the candidate code in a fixed
   template).
3. **Run the model** in float16 and read hidden states at every layer.
4. **Pool tokens into one vector per layer** using `response_avg`: average the
  hidden states over the **response** tokens (the candidate code), not the
   prompt. That gives one vector per layer per implementation, with length 
   equal to the model's hidden dimension.
5. **Save** to `bundle.pt` plus a small `metadata.json` (model name, fold,
  source file, git commit, etc.).

The extraction script is:

```bash
python scripts/extract/cache_hidden_states.py \
  --models <hf-model-id> \
  --dataset-source icse26 --benchmark <bigcodebench or humaneval> \
  --protocol function \
  --data-splits validate test \
  --representations response_avg \
  --verify
```

Or use `make collect-hidden-states` to run all models and benchmarks.

**Important:** saving hidden states is a **new** model run. It is not a copy of
the disk from the original experiment run. The same code and data are used, but
the GPU run happens again later.

---

## What "verification" means

We already ran the full correctness pipeline once. That run produced:

- `**vectors/.../direction.pt`** — a direction vector per layer, learned from
the fit split.
- `**results/.../detailed.json**` — per-implementation **scores** on validate
and test (one number per layer per implementation), plus labels and task ids.

Verification asks: **if we take our newly cached hidden states and score them
the same way as before, do we get the same kind of results as
`detailed.json`?**

We do **not** require every stored number to match exactly. The original run and
the cache run are two separate forward passes in float16. Small numerical
differences are normal and add up over 32 layers. What we require instead is that
the **overall pattern** of scores matches the reference run.

---

## How verification works

When you pass `--verify` (or only check with `--verify-only`), the script, for
each fold:

1. Loads the cached `bundle.pt` for validate and test.
2. Loads the matching `direction.pt` and `detailed.json` from the earlier
  experiment.
3. For each implementation, projects its cached activations onto the direction
  at every layer (same formula as the original pipeline).
4. Compares those scores to the scores stored in `detailed.json`.

**Pass/fail rule (the only thing that stops the script):**

For **every layer**, the new scores and the reference scores must **track each
other almost perfectly**. Concretely, we require correlation at least **0.999**
(default `--verify-corr-min`). In practice a good cache is above **0.99999**.

- If correlation is high, the cache is valid for analysis: rankings and layer
behavior match what we already published.
- If correlation drops far below 1 (e.g. 0.2 or 0.4), something is wrong with
extraction — wrong samples, wrong pooling, wrong layer, wrong direction file,
etc. The script exits with an error.

**What does *not* fail verification:**

You may see **warnings** like:

```text
pairwise accuracy differs by 0.0222 ... Not failing.
```

That line is **informational only**. It compares a summary accuracy number
between the two runs at one layer. On small validate splits, one or two
implementations can swap order because of tiny score noise, which moves accuracy
by a few percentage points even when the underlying scores still match very
closely. We log that warning but **do not** treat it as failure.

**Other hard failures (missing inputs, not math):**

- Missing `bundle.pt`, `direction.pt`, or `detailed.json`
- Layer list or score table shape does not match the reference file

---

## What verification does *not* check

- It does not re-run the fit split or rebuild directions; it assumes
`vectors/.../direction.pt` from the original experiment is correct.
- It does not prove bit-for-bit identity with the first GPU run (that is not
possible with float16).
- It does not fail on small accuracy differences when correlation is already
very high.

Those limits are intentional: we care that cached hidden states support the
**same conclusions** as the run we already trust, not that every float is
identical.