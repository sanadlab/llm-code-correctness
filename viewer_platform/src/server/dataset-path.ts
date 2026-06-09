import fs from "fs";
import path from "path";

/** Repo root: parent of `viewer/` when running Next from `viewer/`. */
export function getRepoRoot(): string {
  return path.resolve(process.cwd(), "..");
}

export function getBigCodeBenchRoot(): string {
  return path.join(getRepoRoot(), "dataset", "bigcodebench");
}

/** Resolve a path under `dataset/bigcodebench` (blocks path traversal). */
export function resolveDatasetFile(relativePath: string): string {
  const root = getBigCodeBenchRoot();
  const normalized = path.normalize(relativePath).replace(/^(\.\.(\/|\\|$))+/, "");
  const full = path.resolve(root, normalized);
  if (!full.startsWith(root + path.sep) && full !== root) {
    throw new Error("Invalid dataset path");
  }
  if (!fs.existsSync(full)) {
    throw new Error(`File not found: ${relativePath}`);
  }
  return full;
}

export function datasetRootExists(): boolean {
  try {
    resolveDatasetFile("metadata.json");
    return true;
  } catch {
    return false;
  }
}

export function pickReadablePath(relativePath: string): string {
  const root = getBigCodeBenchRoot();
  const normalized = path.normalize(relativePath).replace(/^(\.\.(\/|\\|$))+/, "");
  const jsonl = path.resolve(root, normalized);
  const zst = `${jsonl}.zst`;

  if (fs.existsSync(jsonl)) return jsonl;
  if (fs.existsSync(zst)) return zst;
  throw new Error(`File not found: ${relativePath} (or .zst)`);
}
