import { parseJsonl } from "./jsonl";

/** Parse a large JSONL upload in the browser with an optional row cap. */
export async function parseJsonlFile(
  file: File,
  maxRecords = 500,
): Promise<{ text: string; truncated: boolean; lineCount: number }> {
  if (file.size <= 8 * 1024 * 1024) {
    const text = await file.text();
    const lines = text.split(/\r?\n/).filter((l) => l.trim());
    if (lines.length <= maxRecords) {
      return { text, truncated: false, lineCount: lines.length };
    }
    return {
      text: lines.slice(0, maxRecords).join("\n") + "\n",
      truncated: true,
      lineCount: maxRecords,
    };
  }

  const reader = file.stream().getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  const lines: string[] = [];

  while (lines.length < maxRecords) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split(/\r?\n/);
    buffer = parts.pop() ?? "";
    for (const part of parts) {
      if (!part.trim()) continue;
      lines.push(part);
      if (lines.length >= maxRecords) break;
    }
  }

  reader.cancel().catch(() => undefined);

  const truncated = lines.length >= maxRecords;
  return {
    text: lines.join("\n") + "\n",
    truncated,
    lineCount: lines.length,
  };
}

export function parseJsonlWithLimit(text: string, maxRecords: number) {
  const all = parseJsonl(text);
  if (all.length <= maxRecords) {
    return { records: all, truncated: false };
  }
  return { records: all.slice(0, maxRecords), truncated: true };
}
