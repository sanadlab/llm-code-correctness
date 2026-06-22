import { spawn } from "child_process";
import fs from "fs";
import readline from "readline";

import { pickReadablePath } from "./dataset-path";

export interface ReadJsonlOptions {
  offset?: number;
  limit?: number;
  status?: string;
  taskId?: string;
}

export async function readAllJsonlRecords(
  relativePath: string,
): Promise<Record<string, unknown>[]> {
  const filePath = pickReadablePath(relativePath);
  const records: Record<string, unknown>[] = [];

  for await (const line of iterateLines(filePath)) {
    if (!line.trim()) continue;
    try {
      records.push(JSON.parse(line) as Record<string, unknown>);
    } catch {
      // Skip malformed lines to keep export robust.
    }
  }

  return records;
}

/** Line-based offset (scan from start). */
export async function readJsonlRecordsWithLineOffset(
  relativePath: string,
  options: ReadJsonlOptions = {},
): Promise<Record<string, unknown>[]> {
  const filePath = pickReadablePath(relativePath);
  const skip = Math.max(0, options.offset ?? 0);
  const limit = Math.min(Math.max(1, options.limit ?? 200), 5000);
  const records: Record<string, unknown>[] = [];
  let lineIndex = 0;

  for await (const line of iterateLines(filePath)) {
    if (!line.trim()) continue;

    let row: Record<string, unknown>;
    try {
      row = JSON.parse(line) as Record<string, unknown>;
    } catch {
      continue;
    }

    if (options.taskId && row.task_id !== options.taskId) continue;
    if (options.status && row.status !== options.status) continue;

    if (lineIndex < skip) {
      lineIndex += 1;
      continue;
    }

    records.push(row);
    if (records.length >= limit) break;
    lineIndex += 1;
  }

  return records;
}

async function* iterateLines(filePath: string): AsyncGenerator<string> {
  if (filePath.endsWith(".zst")) {
    const proc = spawn("zstd", ["-dc", filePath], {
      stdio: ["ignore", "pipe", "pipe"],
    });
    const rl = readline.createInterface({
      input: proc.stdout,
      crlfDelay: Infinity,
    });
    for await (const line of rl) {
      yield line;
    }
    await new Promise<void>((resolve, reject) => {
      proc.on("close", (code) => (code === 0 ? resolve() : reject(new Error(`zstd exited ${code}`))));
      proc.on("error", reject);
    });
    return;
  }

  const rl = readline.createInterface({
    input: fs.createReadStream(filePath),
    crlfDelay: Infinity,
  });
  for await (const line of rl) {
    yield line;
  }
}

/** Build mutant_id → code for confound joins (stops early when all ids found). */
export async function buildMutantCodeIndex(
  mutantIds: Set<string>,
  mutantsRelativePath = "intermediate/validated_mutants.jsonl",
): Promise<Map<string, string>> {
  const index = new Map<string, string>();
  if (mutantIds.size === 0) return index;

  const filePath = pickReadablePath(mutantsRelativePath);

  for await (const line of iterateLines(filePath)) {
    if (!line.trim()) continue;
    let row: Record<string, unknown>;
    try {
      row = JSON.parse(line) as Record<string, unknown>;
    } catch {
      continue;
    }
    const id = row.mutant_id;
    if (typeof id !== "string" || !mutantIds.has(id)) continue;
    if (typeof row.code === "string") {
      index.set(id, row.code);
      if (index.size >= mutantIds.size) break;
    }
  }

  return index;
}
