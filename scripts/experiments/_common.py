"""Shared helpers for the ICSE26 monitoring experiments.

Centralizes:
  - icse26 jsonl loading
  - the framing factory (registry of named framings + per-model chat tags)
  - the hidden-state extractor (Framing.build → tokenize → forward → pool),
    parameterized on a per-record framing dispatcher
  - pairwise / 1-of-N accuracy metrics
  - the model/tokenizer loader
"""

from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict
from typing import Callable, Dict, List, Sequence, Tuple

import torch

from src.framings import Concept, Function, Neutral
from src.framings.base import Framing, Prompt
from src.framings.raw import Raw
from src.hidden_states.cache import git_commit, safe_model_dir
from src.hidden_states import HiddenStateBundle
from src.representations import (
    Boundaries,
    PromptAvg,
    PromptLast,
    Representation,
    ResponseAvg,
    SequenceLast,
)
from src.vectors import MeanDiff, PCA, VectorMethod
from src.vectors.projection import project_all_layers


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

# Per-model chat-marker tags as used by the original repe paper (raw string
# concatenation, no `apply_chat_template`, no system prompt). Used by
# framings that bypass the tokenizer's chat template.
PAPER_TAGS = {
    "Qwen/Qwen2.5-Coder-7B-Instruct": dict(
        user_start="<|im_start|>user\n", user_end="<|im_end|>\n",
        assistant_start="<|im_start|>assistant\n",
    ),
    "infly/OpenCoder-8B-Instruct": dict(
        user_start="<|im_start|>user\n", user_end="<|im_end|>\n",
        assistant_start="<|im_start|>assistant\n",
    ),
    # CodeLlama / Mistral: paper's per-model JSON (eval_configs/humaneval/
    # selection_accuracy/humaneval/{codellama,mistral}.json) sets
    #   user_start="[INST] ", user_end="", assistant_start="[/INST] "
    # Earlier we had swapped these (user_end=" [/INST]", assistant_start=""),
    # which shifted the trailing space and changed the tokenization of the
    # final tokens consumed by the concept stem — biasing the eval-time
    # last-token hidden state away from the paper's.
    "codellama/CodeLlama-7b-Instruct-hf": dict(
        user_start="[INST] ", user_end="",
        assistant_start="[/INST] ",
    ),
    "mistralai/Mistral-7B-Instruct-v0.3": dict(
        user_start="[INST] ", user_end="",
        assistant_start="[/INST] ",
    ),
    # Paper x_fold humaneval & mbpp configs run Mistral as the BASE
    # model with all chat tags = "" (empty). This is the configuration
    # whose numbers the spec targets (HE 1-of-4 ≈ 0.49).
    "mistralai/Mistral-7B-v0.3": dict(
        user_start="", user_end="",
        assistant_start="",
    ),
}


def build_framing(name: str, model_name: str = "") -> Framing:
    """Construct a framing by name.

    `raw` and `concept` bake per-model chat tags directly into the string
    (paper-exact replication path; no chat template, no system prompt).

    `neutral` and `function` use a clean `Prompt` whose chat-template
    handling is decided by the extractor; defaults to applying the model's
    `apply_chat_template` (including any default system prompt).
    """
    tags = PAPER_TAGS.get(model_name, {})
    if name == "neutral":
        return Neutral(question_label="Task", answer_label="Code")
    if name == "raw":
        return Raw(
            question_label="Task", answer_label="Code",
            fence_answer=True, fence_language="python",
        )
    if name == "concept":
        return Concept(
            concept="correctness",
            question_label="Task", answer_label="Code",
            target_field_name="code",
            fence_answer=True, fence_language="python",
            **tags,
        )
    if name == "function_pos":
        return Function(
            instruction="Write a correct implementation for the following task.",
            question_label="Task", answer_label="Code",
        )
    if name == "function_neg":
        return Function(
            instruction="Write an incorrect implementation for the following task.",
            question_label="Task", answer_label="Code",
        )
    raise ValueError(f"unknown framing: {name}")


REPRESENTATIONS: Dict[str, Representation] = {
    "prompt_avg": PromptAvg(),
    "prompt_last": PromptLast(),
    "response_avg": ResponseAvg(),
    "sequence_last": SequenceLast(),
}

METHODS: Dict[str, VectorMethod] = {
    "mean_diff": MeanDiff(),
    "pca": PCA(),
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

Record = dict  # {sample_id, task_id, question, answer, label}


def load_split(split_path: Path) -> List[Record]:
    """Flatten icse26 jsonl into per-implementation records.

    Supports two on-disk schemas:
      - icse26 (dict impls): `implementations: [{implementation_id, code, label}]`
      - paper raw (parallel lists): `implementations: [str]`, `labels: [int]`
    """
    records: List[Record] = []
    with open(split_path) as f:
        for line in f:
            row = json.loads(line)
            task = row["task"]
            impls = row["implementations"]
            if impls and isinstance(impls[0], dict):
                for impl in impls:
                    records.append({
                        "sample_id": impl["implementation_id"],
                        "task_id": row["task_id"],
                        "question": task,
                        "answer": impl["code"],
                        "label": int(impl["label"]),
                    })
            else:
                labels = row["labels"]
                for i, (code, lab) in enumerate(zip(impls, labels)):
                    records.append({
                        "sample_id": f"{row['task_id']}__impl_{i}",
                        "task_id": row["task_id"],
                        "question": task,
                        "answer": code,
                        "label": int(lab),
                    })
    return records


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_name: str, dtype: str = "float16"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {"bfloat16": torch.bfloat16,
                   "float16": torch.float16,
                   "float32": torch.float32}[dtype]

    print(f"[setup] loading model: {model_name}")
    # Paper (repe-source-code/model.py:19) picks slow tokenizer for Llama-arch
    # models: `use_fast = "LlamaForCausalLM" not in config.architectures`.
    # We need to load the model first to peek at the arch — but to avoid the
    # double download, use the config endpoint.
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    arches = getattr(cfg, "architectures", []) or []
    use_fast = "LlamaForCausalLM" not in arches
    # `legacy=False` matches the paper's tokenizer construction
    # (repe-source-code/model.py:23,25) and fixes Llama/Mistral SentencePiece
    # whitespace handling around special tokens.
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, legacy=False, use_fast=use_fast,
        )
    except TypeError:
        # Tokenizer class doesn't accept `legacy`; fall back.
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, use_fast=use_fast,
        )
    # Paper (repe-source-code/model.py:27) sets pad_token_id=0 unconditionally.
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"  # matches paper (repe-source-code/model.py:23,25); RoPE-sensitive.
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch_dtype, device_map="auto", trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Extractor: Framing → tokens → forward → pool
# ---------------------------------------------------------------------------

FramingFor = Callable[[Record], Framing]


@torch.no_grad()
def extract_bundle(
    framing_for: FramingFor,
    records: Sequence[Record],
    model,
    tokenizer,
    representations: Sequence[Representation],
    batch_size: int = 4,
    max_length: int = 2048,
) -> HiddenStateBundle:
    """Run the model on `records` and pool every representation.

    `framing_for(record) -> Framing` lets callers dispatch a different
    framing per record (e.g. positive vs negative function instruction).
    Pass `lambda r: framing` when one framing applies to the whole split.

    Returns a `HiddenStateBundle` with `acts[rep.name]` of shape
    `[N, num_layers + 1, hidden_dim]`.
    """
    # 1. Build prompts.
    prompts: List[Prompt] = [framing_for(r).build(r["question"], r["answer"]) for r in records]

    # 2. Pre-compute per-sample full text + prompt-prefix length in tokens.
    #
    # For `assistant_is_response=False` (e.g. Concept), we deliberately do
    # NOT call `apply_chat_template` with a closed assistant turn, because
    # chat templates strip trailing whitespace and append a closing role
    # marker (e.g. `<|im_end|>\n`, `</s>`). That destroys the trailing-space
    # token the concept stem relies on. Instead, we build the prefix with
    # `add_generation_prompt=True` and append `assistant_text` verbatim.
    full_texts: List[str] = []
    prompt_lens: List[int] = []
    # In paper-exact mode (use_chat_template=False), we feed the raw string to
    # the tokenizer with `add_special_tokens=True` (the HF default) so that
    # BOS is added the same way as paper AND the leading word-boundary marker
    # (SentencePiece "▁") is generated correctly for the first content word.
    # Manually prepending the BOS *string* and using `add_special_tokens=False`
    # is NOT equivalent: it produces a different tokenization of the first
    # token (no leading-space marker), causing all downstream hidden states
    # to drift.
    # In chat-template mode, `apply_chat_template` already injects BOS
    # itself, so we keep `add_special_tokens=False` for that branch.
    use_chat_template_batch = (not prompts) or prompts[0].use_chat_template
    for p in prompts:
        if not p.use_chat_template:
            full_text = p.user_text + p.assistant_text
            prompt_prefix = p.user_text if p.assistant_is_response else full_text
        else:
            user_only = [{"role": "user", "content": p.user_text}]
            user_prefix = tokenizer.apply_chat_template(
                user_only, tokenize=False, add_generation_prompt=True
            )
            full_text = user_prefix + p.assistant_text
            prompt_prefix = user_prefix if p.assistant_is_response else full_text
        full_texts.append(full_text)
        prompt_lens.append(
            tokenizer(
                prompt_prefix,
                add_special_tokens=(not p.use_chat_template),
                return_tensors="pt",
            )["input_ids"].shape[1]
        )

    # 3. Batched forward; pool each layer with each representation.
    pooled: Dict[str, List[torch.Tensor]] = {r.name: [] for r in representations}

    for start in range(0, len(records), batch_size):
        chunk_texts = full_texts[start : start + batch_size]
        chunk_lens = prompt_lens[start : start + batch_size]

        enc = tokenizer(
            chunk_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=(not use_chat_template_batch),
        ).to(model.device)

        seq_len = enc["input_ids"].shape[1]
        attn = enc["attention_mask"]
        content_lens = attn.sum(dim=1)  # [B] number of real tokens per row
        # Left-padding: real tokens occupy [seq_len - content_len, seq_len);
        # right-padding: real tokens occupy [0, content_len).
        if tokenizer.padding_side == "left":
            pad_offset = seq_len - content_lens  # [B] index of first real token
            sequence_end = torch.full_like(content_lens, seq_len - 1)
        else:
            pad_offset = torch.zeros_like(content_lens)
            sequence_end = content_lens - 1
        prompt_lens_t = torch.tensor(
            [min(L, seq_len) for L in chunk_lens],
            dtype=torch.long, device=model.device,
        )
        prompt_lens_t = torch.minimum(prompt_lens_t, content_lens)
        prompt_end = pad_offset + prompt_lens_t
        boundaries = Boundaries(prompt_end=prompt_end, sequence_end=sequence_end)

        out = model(**enc, output_hidden_states=True, use_cache=False, return_dict=True)
        hidden_states = out.hidden_states  # tuple of (num_layers + 1)

        for rep in representations:
            layer_pooled = [rep.pool(h, boundaries).cpu() for h in hidden_states]
            pooled[rep.name].append(torch.stack(layer_pooled, dim=1))  # [B, L+1, H]

        del out

    acts = {name: torch.cat(chunks, dim=0) for name, chunks in pooled.items()}
    return HiddenStateBundle(
        sample_ids=[r["sample_id"] for r in records],
        task_ids=[r["task_id"] for r in records],
        labels=torch.tensor([r["label"] for r in records], dtype=torch.long),
        acts=acts,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def pairwise_accuracy(scores: torch.Tensor, labels: torch.Tensor, task_ids: List[str]) -> float:
    """Average over tasks of P(score(pos) > score(neg)) for all pos/neg pairs."""
    by_task: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"pos": [], "neg": []})
    for s, lab, tid in zip(scores.tolist(), labels.tolist(), task_ids):
        by_task[tid]["pos" if lab == 1 else "neg"].append(s)

    accs: List[float] = []
    for tid, g in by_task.items():
        if not g["pos"] or not g["neg"]:
            continue
        wins = sum(p > n for p in g["pos"] for n in g["neg"])
        ties = sum(p == n for p in g["pos"] for n in g["neg"])
        total = len(g["pos"]) * len(g["neg"])
        accs.append((wins + 0.5 * ties) / total)
    return sum(accs) / len(accs) if accs else float("nan")


def one_of_n_accuracy(scores: torch.Tensor, labels: torch.Tensor, task_ids: List[str]) -> float:
    """For each task, the canonical (label=1) candidate should score highest.

    For icse26 test: 4 candidates per task, 1 correct, 3 incorrect → "1 of 4".
    Ties with the positive are counted as 1 / num_tied.
    """
    by_task: Dict[str, List[Tuple[float, int]]] = defaultdict(list)
    for s, lab, tid in zip(scores.tolist(), labels.tolist(), task_ids):
        by_task[tid].append((s, lab))

    hits = 0.0
    for tid, items in by_task.items():
        if not any(l == 1 for _, l in items):
            continue
        top = max(s for s, _ in items)
        tied = [l for s, l in items if s == top]
        if 1 in tied:
            hits += 1.0 / len(tied)
    return hits / len(by_task) if by_task else float("nan")


def per_layer_split_metrics(
    scores_all: torch.Tensor,
    layers: Sequence[int],
    labels: torch.Tensor,
    task_ids: List[str],
) -> Dict[int, Dict[str, float]]:
    """Per-layer pairwise and 1-of-N accuracy for one projected split."""
    per_layer: Dict[int, Dict[str, float]] = {}
    for j, layer in enumerate(layers):
        col = scores_all[:, j]
        per_layer[layer] = {
            "one_of_4": one_of_n_accuracy(col, labels, task_ids),
            "pairwise_acc": pairwise_accuracy(col, labels, task_ids),
        }
    return per_layer


def select_layer_by_metric(
    per_layer: Dict[int, Dict[str, float]],
    metric: str = "pairwise_acc",
) -> int:
    """Pick the deepest layer (highest index) on ties, matching paper behavior."""
    ordered = sorted(per_layer.keys(), reverse=True)
    return max(ordered, key=lambda layer: (per_layer[layer][metric], layer))


# ---------------------------------------------------------------------------
# Cross-validation driver
# ---------------------------------------------------------------------------

def run_cross_validation(
    *,
    splits_dir: Path,
    fold_indices: Sequence[int],
    framing_for_fit: FramingFor,
    framing_for_val: FramingFor,
    framing_for_test: FramingFor,
    representation: Representation,
    method: VectorMethod,
    model,
    tokenizer,
    batch_size: int,
    max_length: int,
    eval_max_length: int = 4096,  # Paper passes max_length=512 BUT without truncation=True, so it's effectively ignored. Use a high value to match paper's no-truncation behavior.
    fit_label_for_log: str = "fit",
    val_label_for_log: str = "validate",
    test_label_for_log: str = "test",
    # ------- persistence (all optional; if both dirs are None, nothing is written) -------
    vectors_dir: Path = None,
    results_dir: Path = None,
    model_name: str = "",
    dataset: str = "",
    split_family: str = "",
    distribution: str = "id",
    method_label: str = "",
    fit_framing_label: str = "",
    val_framing_label: str = "",
    test_framing_label: str = "",
    chat_template: bool = True,
    dtype: str = "float16",
    max_length_meta: int = None,
    experiment_script: str = "",
    layer_selection_metric: str = "pairwise_acc",
) -> List[dict]:
    """Run the standard per-fold fit→layer-select→test loop.

    If `vectors_dir` / `results_dir` are provided, writes:

      <vectors_dir>/<model_safe>/<split_family>/<dataset>/<distribution>/
          <method_label>/<fit_framing_label>/<representation>/fold_N/
              direction.pt
              metadata.json

      <results_dir>/<model_safe>/<split_family>/<dataset>/<distribution>/
          <method_label>/<fit_framing_label>/<representation>/
          val=<val_framing>__test=<test_framing>/fold_N/
              detailed.json
              results.json
              metadata.json
    """
    per_fold: List[dict] = []

    write_vectors = vectors_dir is not None
    write_results = results_dir is not None
    git_sha = git_commit() if (write_vectors or write_results) else "unknown"
    common_path = (
        f"{safe_model_dir(model_name)}/{split_family}/{dataset}/{distribution}/"
        f"{method_label}/{fit_framing_label}/{representation.name}"
    )
    eval_segment = f"val={val_framing_label}__test={test_framing_label}"

    for fold in fold_indices:
        print(f"\n[fold {fold}] ===========================")
        fold_dir = splits_dir / f"fold_{fold}"

        fit_recs = load_split(fold_dir / "fit.jsonl")
        val_recs = load_split(fold_dir / "validate.jsonl")
        test_recs = load_split(fold_dir / "test.jsonl")
        print(f"[fold {fold}] sizes: fit={len(fit_recs)} val={len(val_recs)} test={len(test_recs)}")

        print(f"[fold {fold}] extracting fit ({fit_label_for_log})...")
        bundle_fit = extract_bundle(framing_for_fit, fit_recs, model, tokenizer,
                                    [representation], batch_size, max_length)
        print(f"[fold {fold}] extracting validate ({val_label_for_log})...")
        bundle_val = extract_bundle(framing_for_val, val_recs, model, tokenizer,
                                    [representation], batch_size, eval_max_length)
        print(f"[fold {fold}] extracting test ({test_label_for_log})...")
        bundle_test = extract_bundle(framing_for_test, test_recs, model, tokenizer,
                                     [representation], batch_size, eval_max_length)

        acts_fit = representation.select(bundle_fit)
        direction = method.fit(acts_fit, bundle_fit.labels, bundle_fit.task_ids)
        direction.representation = representation.name

        # Per-layer val scores.
        acts_val = representation.select(bundle_val)
        layers, val_scores_all = project_all_layers(direction, acts_val)  # [N_val, L]
        val_per_layer = per_layer_split_metrics(
            val_scores_all, layers, bundle_val.labels, bundle_val.task_ids
        )
        selected_layer = select_layer_by_metric(val_per_layer, layer_selection_metric)
        sel_val = val_per_layer[selected_layer]
        print(f"[fold {fold}] selected layer = {selected_layer} "
              f"(val {layer_selection_metric} = {sel_val[layer_selection_metric]:.3f}, "
              f"val one_of_4 = {sel_val['one_of_4']:.3f})")

        # Per-layer test scores.
        acts_test = representation.select(bundle_test)
        _, test_scores_all = project_all_layers(direction, acts_test)  # [N_test, L]
        test_per_layer = per_layer_split_metrics(
            test_scores_all, layers, bundle_test.labels, bundle_test.task_ids
        )
        sel_test = test_per_layer[selected_layer]
        print(f"[fold {fold}] test 1-of-4 = {sel_test['one_of_4']:.3f}  "
              f"test pairwise = {sel_test['pairwise_acc']:.3f}")

        # Theoretical-best layer on test (for reference; not used for selection).
        best_layer = select_layer_by_metric(test_per_layer, "one_of_4")

        per_fold.append({
            "fold": fold,
            "selected_layer": selected_layer,
            "layer_selection_metric": layer_selection_metric,
            "val_pairwise_acc": sel_val["pairwise_acc"],
            "val_one_of_4": sel_val["one_of_4"],
            "test_one_of_4": sel_test["one_of_4"],
            "test_pairwise": sel_test["pairwise_acc"],
        })

        # ----------------- persistence -----------------
        if write_vectors:
            vec_fold = Path(vectors_dir) / common_path / f"fold_{fold}"
            vec_fold.mkdir(parents=True, exist_ok=True)
            direction.save(vec_fold / "direction.pt")
            vec_meta = {
                "model": model_name,
                "dataset": dataset,
                "split_family": split_family,
                "distribution": distribution,
                "fold": fold,
                "method": method_label,
                "representation": representation.name,
                "fit_framing": fit_framing_label,
                "chat_template": chat_template,
                "dtype": dtype,
                "max_length": max_length_meta if max_length_meta is not None else max_length,
                "n_fit": len(fit_recs),
                "experiment_script": experiment_script,
                "git_commit": git_sha,
            }
            with open(vec_fold / "metadata.json", "w") as f:
                json.dump(vec_meta, f, indent=2)

        if write_results:
            res_fold = (
                Path(results_dir) / common_path / eval_segment / f"fold_{fold}"
            )
            res_fold.mkdir(parents=True, exist_ok=True)

            detailed = {
                "fold": fold,
                "layers": list(layers),
                "val": {
                    "sample_ids": list(bundle_val.sample_ids),
                    "task_ids": list(bundle_val.task_ids),
                    "labels": bundle_val.labels.tolist(),
                    "scores": val_scores_all.float().tolist(),  # [N_val, L]
                },
                "test": {
                    "sample_ids": list(bundle_test.sample_ids),
                    "task_ids": list(bundle_test.task_ids),
                    "labels": bundle_test.labels.tolist(),
                    "scores": test_scores_all.float().tolist(),  # [N_test, L]
                },
            }
            with open(res_fold / "detailed.json", "w") as f:
                json.dump(detailed, f)

            results = {
                "fold": fold,
                "selected_layer": selected_layer,
                "layer_selection_metric": layer_selection_metric,
                "val_per_layer": {str(k): v for k, v in val_per_layer.items()},
                "test_per_layer": {str(k): v for k, v in test_per_layer.items()},
                "val_at_selected_layer": dict(sel_val),
                "test_at_selected_layer": dict(sel_test),
                "test_theoretical_best_layer": {
                    "layer": best_layer,
                    **test_per_layer[best_layer],
                },
            }
            with open(res_fold / "results.json", "w") as f:
                json.dump(results, f, indent=2)

            res_meta = {
                "vector_path": (
                    str(Path(vectors_dir) / common_path / f"fold_{fold}" / "direction.pt")
                    if write_vectors else None
                ),
                "model": model_name,
                "dataset": dataset,
                "split_family": split_family,
                "distribution": distribution,
                "fold": fold,
                "method": method_label,
                "representation": representation.name,
                "fit_framing": fit_framing_label,
                "val_framing": val_framing_label,
                "test_framing": test_framing_label,
                "n_val": len(val_recs),
                "n_test": len(test_recs),
                "experiment_script": experiment_script,
                "git_commit": git_sha,
            }
            with open(res_fold / "metadata.json", "w") as f:
                json.dump(res_meta, f, indent=2)

        # Free memory between folds (the per-layer score tensors can be largeish).
        del bundle_fit, bundle_val, bundle_test
        del acts_fit, acts_val, acts_test
        del val_scores_all, test_scores_all
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if per_fold:
        mean_val_1of4 = sum(x["val_one_of_4"] for x in per_fold) / len(per_fold)
        mean_test_1of4 = sum(x["test_one_of_4"] for x in per_fold) / len(per_fold)
        mean_test_pair = sum(x["test_pairwise"] for x in per_fold) / len(per_fold)
        print(f"\n[summary] mean val 1-of-4 = {mean_val_1of4:.3f}  "
              f"mean test 1-of-4 = {mean_test_1of4:.3f}  "
              f"mean test pairwise = {mean_test_pair:.3f}  (n_folds={len(per_fold)})")

    print("\nPer-fold:")
    for x in per_fold:
        print(f"  {x}")

    return per_fold
