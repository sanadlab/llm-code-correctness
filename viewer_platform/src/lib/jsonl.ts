export function parseJsonl(text: string): Record<string, unknown>[] {
  const lines = text.split(/\r?\n/).filter((line) => line.trim().length > 0);
  return lines.map((line, index) => {
    try {
      return JSON.parse(line) as Record<string, unknown>;
    } catch {
      throw new Error(`Invalid JSON on line ${index + 1}`);
    }
  });
}

export async function readFileAsText(file: File): Promise<string> {
  return file.text();
}
