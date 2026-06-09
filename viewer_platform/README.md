# Code Review Viewer

Card-based UI for **Phase 1** intermediate JSONL and **Phase 2** CV splits under `dataset/bigcodebench/`.

## Features

| Area | What you can browse |
|------|---------------------|
| **Phase 1** | `tasks`, `validated_tasks`, `dropped_tasks`, `mutants`, `validated_mutants`, `confound_candidates`, `validated_confounded`, `frozen` |
| **Phase 2** | Any `phase2/sN_*/fold_K/{fit,validate,test}.jsonl` split |
| **Review** | Diff / side-by-side, approve/discard, export `reviews-*.json` |

Confound rows join **source mutant code** from `validated_mutants.jsonl` when loaded via presets. Mutant rows join **canonical** from `tasks.jsonl`.

Large files (e.g. 600MB+ `validated_mutants.jsonl`) are read via a **local API** with `offset` + `max rows` — not loaded whole into the browser.

## Local development

From the **repo root** (so `dataset/bigcodebench` resolves):

```bash
cd viewer
npm install
npm run dev:clean   # first run, or after `npm run build`
```

If you see `Cannot find module '../chunks/ssr/[turbopack]_runtime.js'`, the `.next` cache is stale (mixing `next build` with Turbopack dev). Run `npm run dev:clean` or `npm run clean && npm run dev`. Avoid `npm run dev:turbo` unless you always use Turbopack.

Open http://localhost:3000 → **Repository datasets**:

1. **Phase 1** — pick a file preset (e.g. `validated_mutants.jsonl`) → **Load**
2. **Phase 2** — choose split family (`s1_baseline`, `s2_test_confound`, …), fold, and `fit` / `validate` / `test`

Use **Slice & filters** to page (`line offset`, `max rows`), filter `status` (`fail` / `pass` / `timeout`), or a single `task_id`.

Requires `zstd` on PATH if only `.jsonl.zst` exists (API decompresses via `zstd -dc`).

## Deploy on Vercel

1. Import repo, set **Root Directory** to `viewer`
2. Deploy

On Vercel the dataset API is unavailable (no `dataset/` tree). Use **Manual upload** with a row cap, or host sample slices under `public/`.

## Phase 2 diff behaviour

Each JSONL line is one **task** with an `implementations` array:

| `kind` | Typical diff |
|--------|----------------|
| `canonical` | prompt vs canonical body |
| `mutant` | canonical vs mutant |
| `confound` | source mutant vs confound (or canonical vs confound if mutant not in split) |

Splits like `s2_test_confound` only include canonical + confound implementations on the test fold.

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| `a` | Approve → next |
| `d` | Discard → next |
| `→` / `j` | Next card |
| `←` / `k` | Previous card |
