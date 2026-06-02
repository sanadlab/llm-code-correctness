# Deploy repo (Vercel)

The main research repo (`llm-code-correctness`) keeps the viewer under **`viewer_platform/`**.  
Vercel often only deploys **`main`** from the monorepo, so we mirror this folder into a **small deploy-only repo** whose root *is* the Next.js app.

```
llm-code-correctness/          llm-code-correctness-viewer/
├── viewer_platform/  ──sync──► ├── src/
│   ├── src/                    ├── package.json
│   └── package.json            └── … (no subfolder)
└── … python, datasets
```

## One-time setup

### 1. Create the deploy repo

On GitHub (org `sanadlab`), create an **empty** private repo, e.g.:

`llm-code-correctness-viewer`

Do **not** add a README (keeps first push clean).

### 2. GitHub Actions sync (recommended)

In **`llm-code-correctness`** → Settings → Secrets and variables → Actions:

| Name | Type | Value |
|------|------|--------|
| `VIEWER_DEPLOY_REPO` | **Variable** | `sanadlab/llm-code-correctness-viewer` |
| `VIEWER_DEPLOY_TOKEN` | **Secret** | Fine-grained PAT with **Contents: Read and write** on the deploy repo |

Create the PAT: GitHub → Settings → Developer settings → Fine-grained tokens → target only the deploy repo.

Push any change under `viewer_platform/` to `main` or `feat/code-review-viewer`, or run **Actions → Sync viewer to deploy repo → Run workflow**.

### 3. Manual sync (alternative)

```bash
export VIEWER_DEPLOY_REPO="sanadlab/llm-code-correctness-viewer"
export VIEWER_DEPLOY_TOKEN="ghp_..."   # or use SSH remote without token
chmod +x scripts/sync-viewer-deploy-repo.sh
./scripts/sync-viewer-deploy-repo.sh
```

### 4. Connect Vercel to the **deploy repo**

1. [vercel.com/new](https://vercel.com/new) → import **`llm-code-correctness-viewer`**
2. **Root Directory**: leave **empty** (app is already at repo root)
3. **Production Branch**: `main`
4. Deploy

No `dataset/` on Vercel — use **Upload files** in the UI.

## Day-to-day workflow

1. Edit code in **`llm-code-correctness/viewer_platform/`**
2. Merge to `main` (or push feature branch → Action syncs if configured)
3. Deploy repo updates automatically (or run manual script)
4. Vercel redeploys the viewer repo

## Local dev

Run from monorepo root so `dataset/bigcodebench` resolves for the **From repo** tab:

```bash
cd viewer_platform
npm install
npm run dev:clean
```

The local API reads `../dataset/bigcodebench` relative to the monorepo root.
