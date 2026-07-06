import type { ReviewExport, ReviewRecord } from "./types";

const STORAGE_PREFIX = "llm-code-review:";

export function storageKey(datasetName: string): string {
  return `${STORAGE_PREFIX}${datasetName}`;
}

export function loadReviews(datasetName: string): Record<string, ReviewRecord> {
  if (typeof window === "undefined") return {};
  try {
    const raw = localStorage.getItem(storageKey(datasetName));
    if (!raw) return {};
    return JSON.parse(raw) as Record<string, ReviewRecord>;
  } catch {
    return {};
  }
}

export function saveReviews(
  datasetName: string,
  reviews: Record<string, ReviewRecord>,
): void {
  localStorage.setItem(storageKey(datasetName), JSON.stringify(reviews));
}

export function updateReview(
  datasetName: string,
  itemId: string,
  data: Partial<ReviewRecord>,
): Record<string, ReviewRecord> {
  const reviews = loadReviews(datasetName);
  const current = reviews[itemId] ?? { decision: "pending" };
  reviews[itemId] = { ...current, ...data, updatedAt: new Date().toISOString() };
  saveReviews(datasetName, reviews);
  return reviews;
}

export function importReviews(
  datasetName: string,
  importedDecisions: Record<string, ReviewRecord>
): Record<string, ReviewRecord> {
  const current = loadReviews(datasetName);
  const merged = { ...current };
  for (const [id, record] of Object.entries(importedDecisions)) {
    merged[id] = { ...merged[id], ...record };
  }
  saveReviews(datasetName, merged);
  return merged;
}

export function buildExport(
  datasetName: string,
  schema: ReviewExport["schema"],
  itemIds: string[],
): ReviewExport {
  const reviews = loadReviews(datasetName);
  let approved = 0;
  let discarded = 0;
  let pending = 0;

  for (const id of itemIds) {
    const decision = reviews[id]?.decision ?? "pending";
    if (decision === "approved") approved += 1;
    else if (decision === "discarded") discarded += 1;
    else pending += 1;
  }

  return {
    datasetName,
    exportedAt: new Date().toISOString(),
    schema,
    decisions: reviews,
    summary: {
      total: itemIds.length,
      approved,
      discarded,
      pending,
    },
  };
}

export function downloadJson(filename: string, data: unknown): void {
  const blob = new Blob([JSON.stringify(data, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}
