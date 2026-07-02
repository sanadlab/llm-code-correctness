import type {
  DatasetSchema,
  ReviewDecision,
  ReviewExport,
  ReviewItem,
  ReviewMatrix,
  ReviewRecord,
} from "./types";

const STORAGE_PREFIX = "llm-code-review:";

export function storageKey(datasetName: string): string {
  return `${STORAGE_PREFIX}${datasetName}`;
}

export function loadReviews(datasetName: string): Record<string, ReviewRecord> {
  if (typeof window === "undefined") return {};
  const matrix = loadReviewMatrix(datasetName);
  const firstReviewer = Object.keys(matrix[Object.keys(matrix)[0] ?? ""] ?? {})[0];
  if (!firstReviewer) return {};
  return getReviewsForReviewer(matrix, firstReviewer);
}

export function normalizeReviewerName(name: string): string {
  const normalized = name.trim().toLowerCase().replace(/\s+/g, "-");
  return normalized || "reviewer-1";
}

export function loadReviewMatrix(datasetName: string): ReviewMatrix {
  if (typeof window === "undefined") return {};
  try {
    const raw = localStorage.getItem(storageKey(datasetName));
    if (!raw) return {};
    const parsed = JSON.parse(raw) as unknown;
    if (!parsed || typeof parsed !== "object") return {};

    // Backward compatibility: old format stored direct item -> ReviewRecord.
    const maybeOld = parsed as Record<string, ReviewRecord>;
    const looksOld = Object.values(maybeOld).some(
      (value) =>
        value &&
        typeof value === "object" &&
        "decision" in value &&
        "updatedAt" in value,
    );
    if (looksOld) {
      const migrated: ReviewMatrix = {};
      for (const [itemId, record] of Object.entries(maybeOld)) {
        migrated[itemId] = { "reviewer-1": { ...record, reviewer: "reviewer-1" } };
      }
      return migrated;
    }

    return parsed as ReviewMatrix;
  } catch {
    return {};
  }
}

export function saveReviews(
  datasetName: string,
  reviews: ReviewMatrix,
): { ok: boolean; reason?: string } {
  try {
    localStorage.setItem(storageKey(datasetName), JSON.stringify(reviews));
    return { ok: true };
  } catch (err) {
    const message = err instanceof Error ? err.message : "Failed to persist reviews";
    return { ok: false, reason: message };
  }
}

export function getReviewsForReviewer(
  matrix: ReviewMatrix,
  reviewer: string,
): Record<string, ReviewRecord> {
  const reviews: Record<string, ReviewRecord> = {};
  for (const [itemId, byReviewer] of Object.entries(matrix)) {
    const record = byReviewer[reviewer];
    if (record) reviews[itemId] = record;
  }
  return reviews;
}

/** Merge one decision into an in-memory matrix (does not read localStorage). */
export function applyDecisionToMatrix(
  matrix: ReviewMatrix,
  itemId: string,
  reviewer: string,
  decision: ReviewDecision,
  comment?: string,
): ReviewMatrix {
  return {
    ...matrix,
    [itemId]: {
      ...(matrix[itemId] ?? {}),
      [reviewer]: {
        decision,
        updatedAt: new Date().toISOString(),
        comment,
        reviewer,
      },
    },
  };
}

/** Keep only non-pending decisions to reduce localStorage size. */
export function compactReviewMatrix(matrix: ReviewMatrix): ReviewMatrix {
  const compact: ReviewMatrix = {};
  for (const [itemId, byReviewer] of Object.entries(matrix)) {
    const kept: Record<string, ReviewRecord> = {};
    for (const [rid, record] of Object.entries(byReviewer)) {
      if (record.decision !== "pending") {
        kept[rid] = {
          decision: record.decision,
          updatedAt: record.updatedAt,
          reviewer: record.reviewer ?? rid,
          ...(record.comment ? { comment: record.comment } : {}),
        };
      }
    }
    if (Object.keys(kept).length > 0) compact[itemId] = kept;
  }
  return compact;
}

export function persistReviewMatrix(
  datasetName: string,
  matrix: ReviewMatrix,
): { ok: boolean; reason?: string } {
  return saveReviews(datasetName, compactReviewMatrix(matrix));
}

export function setDecision(
  datasetName: string,
  itemId: string,
  reviewer: string,
  decision: ReviewDecision,
  comment?: string,
): ReviewMatrix {
  const reviews = loadReviewMatrix(datasetName);
  const next = applyDecisionToMatrix(reviews, itemId, reviewer, decision, comment);
  persistReviewMatrix(datasetName, next);
  return next;
}

export function buildExport(
  datasetName: string,
  schema: DatasetSchema,
  reviewer: string,
  queueFilter: ReviewExport["queueFilter"],
  items: ReviewItem[],
  reviewsByItem: ReviewMatrix,
): ReviewExport {
  let approved = 0;
  let discarded = 0;
  let pending = 0;

  for (const item of items) {
    const decision = reviewsByItem[item.id]?.[reviewer]?.decision ?? "pending";
    if (decision === "approved") approved += 1;
    else if (decision === "discarded") discarded += 1;
    else pending += 1;
  }

  const exportItems = items.map((item) => ({
    id: item.id,
    taskId: item.taskId,
    title: item.title,
    subtitle: item.subtitle,
    prompt: item.prompt,
    test: item.test,
    variants: item.variants,
    metadata: item.metadata,
    reviews: reviewsByItem[item.id] ?? {},
  }));

  return {
    datasetName,
    exportedAt: new Date().toISOString(),
    schema,
    reviewer,
    queueFilter,
    reviewsByItem,
    items: exportItems,
    summary: {
      total: items.length,
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
