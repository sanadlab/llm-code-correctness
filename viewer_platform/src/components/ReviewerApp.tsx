"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { buildDataset, inferSchemaFromFilename } from "@/lib/parsers";
import {
  buildExport,
  downloadJson,
  loadReviews,
  updateReview,
  importReviews,
} from "@/lib/review-store";
import { parseJsonl } from "@/lib/jsonl";
import {
  countVariantsByRole,
  itemHasVariantRole,
} from "@/lib/review-items";
import { uploadCategoryById, type UploadCategoryId } from "@/lib/upload-categories";
import type { CodeVariantRole, LoadedDataset, ReviewDecision, ReviewRecord } from "@/lib/types";
import { DataSourcePanel } from "./DataSourcePanel";
import { ReviewCard } from "./ReviewCard";

type FilterMode = "all" | "pending" | "approved" | "discarded";
type QueueFilterMode = "all" | "unreviewed" | "reviewed-1" | "reviewed-2plus";

export function ReviewerApp() {
  const [dataset, setDataset] = useState<LoadedDataset | null>(null);
  const [reviews, setReviews] = useState<Record<string, ReviewRecord>>({});
  const [index, setIndex] = useState(0);
  const [filter, setFilter] = useState<FilterMode>("all");
  const [queueFilter, setQueueFilter] = useState<QueueFilterMode>("all");
  const [kindFilter, setKindFilter] = useState<CodeVariantRole | "all">("all");
  const [viewMode, setViewMode] = useState<"diff" | "split">("diff");
  const [showPrompt, setShowPrompt] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [maxUploadRecords, setMaxUploadRecords] = useState(500);
  const [annotator, setAnnotator] = useState("");

  const handleDatasetLoaded = useCallback((loaded: LoadedDataset) => {
    setDataset(loaded);
    setReviewMatrix(loadReviewMatrix(loaded.name));
    setSourceRecordsForExport(null);
    setIndex(0);
    setError(null);
  }, []);

  const reviewerId = useMemo(() => normalizeReviewerName(reviewerName), [reviewerName]);
  const reviews = useMemo(
    () => getReviewsForReviewer(reviewMatrix, reviewerId),
    [reviewMatrix, reviewerId],
  );

  const loadFromTexts = useCallback(
    (
      name: string,
      text: string,
      options: {
        tasksText?: string;
        mutantsText?: string;
        schema?: LoadedDataset["schema"];
        truncated?: boolean;
        sourcePath?: string;
        reviewMatrixOverride?: ReviewMatrix;
      } = {},
    ) => {
      setLoading(true);
      setError(null);
      try {
        const schema =
          options.schema ??
          inferSchemaFromFilename(name, JSON.parse(text.split("\n")[0] || "{}"));
        const loaded = buildDataset(name, text, {
          schema,
          tasksText: options.tasksText,
          mutantsText: options.mutantsText,
          truncated: options.truncated,
          sourcePath: options.sourcePath,
        });
        handleDatasetLoaded(loaded);
        if (options.reviewMatrixOverride) {
          setReviewMatrix(options.reviewMatrixOverride);
          // Imported _review data can be huge — keep in memory; only persist compact deltas.
          const persisted = persistReviewMatrix(loaded.name, options.reviewMatrixOverride);
          if (!persisted.ok) {
            setStorageNotice(
              "Prior reviews loaded for this session. Browser storage is full, so only new decisions are saved locally.",
            );
          } else {
            setStorageNotice(null);
          }
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to parse JSONL");
        setDataset(null);
      } finally {
        setLoading(false);
      }
    },
    [handleDatasetLoaded],
  );

  const handleUploadLoad = useCallback(
    (params: {
      categoryId: UploadCategoryId;
      primaryText: string;
      fullPrimaryText: string;
      tasksText?: string;
      mutantsText?: string;
      fileName: string;
      truncated: boolean;
    }) => {
      const cat = uploadCategoryById(params.categoryId);
      const firstRow = JSON.parse(params.primaryText.split("\n")[0] || "{}") as Record<
        string,
        unknown
      >;
      const inferred = inferSchemaFromFilename(params.fileName, firstRow);
      const schema =
        inferred !== "unknown"
          ? inferred
          : cat?.schema && cat.schema !== "auto"
            ? cat.schema
            : "unknown";

      setLoadedWithoutCanonicalJoin(
        (schema === "validated_confounded" || schema === "confound_candidates") &&
          !params.tasksText,
      );

      let fullRecords: Record<string, unknown>[] = [];
      try {
        fullRecords = parseJsonl(params.fullPrimaryText);
      } catch {
        fullRecords = [];
      }
      const embeddedMatrix = extractEmbeddedReviewMatrix(schema, fullRecords);
      const localMatrix = loadReviewMatrix(params.fileName);
      const mergedMatrix = mergeReviewMatrices(localMatrix, embeddedMatrix);

      loadFromTexts(params.fileName, params.primaryText, {
        schema,
        tasksText: params.tasksText,
        mutantsText: params.mutantsText,
        truncated: params.truncated,
        reviewMatrixOverride: mergedMatrix,
      });
      setSourceRecordsForExport(fullRecords.length > 0 ? fullRecords : null);
    },
    [loadFromTexts],
  );

  const handleLoadDemo = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [mutantsRes, tasksRes] = await Promise.all([
        fetch("/samples/demo-mutants.jsonl"),
        fetch("/samples/demo-tasks.jsonl"),
      ]);
      if (!mutantsRes.ok || !tasksRes.ok) throw new Error("Demo files not found");
      const mutantsText = await mutantsRes.text();
      const tasks = await tasksRes.text();
      loadFromTexts("demo-mutants.jsonl", mutantsText, {
        tasksText: tasks,
        schema: "validated_mutants",
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load demo");
    } finally {
      setLoading(false);
    }
  }, [loadFromTexts]);

  const filteredItems = useMemo(() => {
    if (!dataset) return [];
    return dataset.items
      .filter((item) => {
        const decision = reviews[item.id]?.decision ?? "pending";
        if (filter !== "all" && decision !== filter) return false;

        const byReviewer = reviewMatrix[item.id] ?? {};
        const priorReviewedCount = Object.entries(byReviewer).filter(
          ([rid, r]) => rid !== reviewerId && r.decision !== "pending",
        ).length;
        if (queueFilter === "unreviewed" && priorReviewedCount !== 0) return false;
        if (queueFilter === "reviewed-1" && priorReviewedCount !== 1) return false;
        if (queueFilter === "reviewed-2plus" && priorReviewedCount < 2) return false;

        if (kindFilter !== "all") {
          if (item.variants) {
            if (!itemHasVariantRole(item, kindFilter)) return false;
          } else if (item.kind !== kindFilter) {
            return false;
          }
        }
        return true;
      });
  }, [dataset, reviews, reviewMatrix, filter, queueFilter, reviewerId, kindFilter]);

  useEffect(() => {
    if (index >= filteredItems.length) {
      setIndex(Math.max(0, filteredItems.length - 1));
    }
  }, [filteredItems.length, index]);

  const current = filteredItems[index] ?? null;
  const currentDecision: ReviewDecision = current
    ? reviews[current.id]?.decision ?? "pending"
    : "pending";

  useEffect(() => {
    if (!current) {
      setDraftComment("");
      return;
    }
    setDraftComment(reviews[current.id]?.comment ?? "");
  }, [current, reviews]);

  const summary = useMemo(() => {
    if (!dataset) return null;
    let approved = 0;
    let discarded = 0;
    let pending = 0;
    for (const item of dataset.items) {
      const d = reviews[item.id]?.decision ?? "pending";
      if (d === "approved") approved += 1;
      else if (d === "discarded") discarded += 1;
      else pending += 1;
    }
    return { approved, discarded, pending, total: dataset.items.length };
  }, [dataset, reviews]);

  const applyDecision = useCallback(
    (decision: ReviewDecision, advance = false) => {
      if (!dataset || !current) return;
      const nextReviews = updateReview(dataset.name, current.id, { decision, annotator });
      setReviews(nextReviews);
      if (advance && index < filteredItems.length - 1) {
        setIndex((i) => i + 1);
      }
    },
    [dataset, current, index, filteredItems.length, annotator],
  );

  const handleUpdateAnnotation = useCallback(
    (annotation: string) => {
      if (!dataset || !current) return;
      const nextReviews = updateReview(dataset.name, current.id, { annotation, annotator });
      setReviews(nextReviews);
    },
    [dataset, current, annotator]
  );

  const handleImportReviews = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !dataset) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      try {
        const text = ev.target?.result as string;
        const data = JSON.parse(text);
        if (data && data.decisions) {
          const nextReviews = importReviews(dataset.name, data.decisions);
          setReviews(nextReviews);
        } else {
          setError("Invalid reviews export format.");
        }
      } catch {
        setError("Failed to parse the imported JSON file.");
      }
    };
    reader.readAsText(file);
    e.target.value = "";
  }, [dataset]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const isTypingTarget =
        target instanceof HTMLInputElement ||
        target instanceof HTMLTextAreaElement ||
        target?.isContentEditable;
      if (!current || isTypingTarget) return;

      const hasActionModifiers = event.ctrlKey && event.shiftKey;
      if (!hasActionModifiers) return;

      if (event.key === "ArrowRight") {
        event.preventDefault();
        setIndex((i) => Math.min(i + 1, filteredItems.length - 1));
      }
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        setIndex((i) => Math.max(i - 1, 0));
      }
      if (event.key.toLowerCase() === "a") {
        event.preventDefault();
        applyDecision("approved", true);
      }
      if (event.key.toLowerCase() === "d") {
        event.preventDefault();
        applyDecision("discarded", true);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [current, filteredItems.length, applyDecision]);

  const exportReviews = async () => {
    if (!dataset) return;
    setExporting(true);
    setExportStatus("Preparing export...");
    setError(null);

    try {
      let records = sourceRecordsForExport;
      if (!records && dataset.sourcePath) {
        setExportStatus("Loading full source file...");
        const res = await fetch(
          `/api/dataset/raw?path=${encodeURIComponent(dataset.sourcePath)}`,
        );
        if (res.ok) {
          const body = (await res.json()) as { records?: Record<string, unknown>[] };
          records = body.records ?? null;
        }
      }
      if (!records || records.length === 0) {
        setError(
          "Could not load full source records for export. Upload the full file to export complete JSONL.",
        );
        return;
      }

      setExportStatus("Merging review metadata...");
      const augmented = attachReviewsToSourceRows(dataset.schema, records, reviewMatrix);
      const jsonl = augmented.map((row) => JSON.stringify(row)).join("\n") + "\n";
      const blob = new Blob([jsonl], { type: "application/jsonl" });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `reviews-${reviewerId}-${dataset.name.replace(/[^\w.-]+/g, "_")}-${Date.now()}.jsonl`;
      setExportStatus("Starting download...");
      anchor.click();
      URL.revokeObjectURL(url);
      setExportStatus("Download started.");
      setTimeout(() => setExportStatus(null), 2000);
    } finally {
      setExporting(false);
    }
  };

  const variantCounts = useMemo(() => {
    if (!dataset) return {};
    return countVariantsByRole(dataset.items);
  }, [dataset]);

  return (
    <div className="mx-auto min-h-screen max-w-7xl px-4 py-8 sm:px-6">
      <header className="mb-8">
        <p className="section-label text-[var(--accent-hover)]">
          LLM Code Correctness · Mutation_DS
        </p>
        <h1 className="mt-2 text-2xl font-bold tracking-tight text-[var(--text)] sm:text-3xl">
          Code review viewer
        </h1>
        <p className="mt-2 max-w-2xl text-sm text-[var(--text-muted)]">
          Review canonical, mutant, and confound code with highlighted diffs. Upload
          JSONL on Vercel, or load from the GitHub repo when running locally.
        </p>
      </header>

      <DataSourcePanel
        onLoaded={handleDatasetLoaded}
        onLoadingChange={setLoading}
        onError={setError}
        onUploadLoad={handleUploadLoad}
        onLoadDemo={handleLoadDemo}
        loading={loading}
        maxUploadRecords={maxUploadRecords}
        onMaxUploadRecordsChange={setMaxUploadRecords}
      />

      {error && (
        <p className="mt-4 rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-300">
          {error}
        </p>
      )}

      {storageNotice && !error && (
        <p className="mt-4 rounded-lg border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-sm text-amber-200">
          {storageNotice}
        </p>
      )}

      {loadedWithoutCanonicalJoin && !error && (
        <p className="mt-4 rounded-lg border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-sm text-amber-200">
          No <span className="font-medium">tasks.jsonl</span> join — cards are confound-only, so
          diff view cannot show canonical. Re-load with tasks.jsonl selected.
        </p>
      )}

      {dataset && !error && (
        <p className="mt-4 rounded-lg border border-[var(--accent)]/30 bg-[var(--accent)]/10 px-4 py-3 text-sm text-emerald-200">
          Loaded <span className="font-medium">{dataset.name}</span> ({dataset.schema}) ·{" "}
          {dataset.items.length} cards
          {dataset.truncated && " · truncated slice"}
        </p>
      )}

      {dataset && summary && (
        <>
          <div className="mt-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <Stat label="Cards" value={summary.total} />
            <Stat label="Pending" value={summary.pending} />
            <Stat label="Approved" value={summary.approved} />
            <Stat label="Discarded" value={summary.discarded} />
          </div>

          <div className="mt-4 panel flex flex-wrap items-center gap-3 p-3">
            <label className="flex items-center gap-2 text-xs text-[var(--text-muted)]">
              Reviewer
              <input
                type="text"
                value={reviewerName}
                onChange={(e) => setReviewerName(e.target.value)}
                className="input-field w-40 py-1 text-xs"
                placeholder="reviewer-1"
              />
            </label>
            <span className="section-label">Review status</span>
            {(["all", "pending", "approved", "discarded"] as const).map((mode) => (
              <button
                key={mode}
                type="button"
                onClick={() => {
                  setFilter(mode);
                  setIndex(0);
                }}
                className={`chip capitalize ${filter === mode ? "chip-active" : ""}`}
              >
                {mode}
              </button>
            ))}
          </div>

          <div className="mt-3 panel flex flex-wrap items-center gap-2 p-3">
            <span className="section-label">Queue</span>
            {([
              ["all", "All cards"],
              ["unreviewed", "0 prior reviews"],
              ["reviewed-1", "1 prior review"],
              ["reviewed-2plus", "2+ prior reviews"],
            ] as const).map(([mode, label]) => (
              <button
                key={mode}
                type="button"
                onClick={() => {
                  setQueueFilter(mode);
                  setIndex(0);
                }}
                className={`chip ${queueFilter === mode ? "chip-active" : ""}`}
              >
                {label}
              </button>
            ))}
          </div>

          <div className="mt-3 flex flex-wrap gap-2">
            <KindChip
              label="all variants"
              count={summary.total}
              active={kindFilter === "all"}
              onClick={() => {
                setKindFilter("all");
                setIndex(0);
              }}
            />
            {(["canonical", "mutant", "confound"] as const).map((role) =>
              variantCounts[role] ? (
                <KindChip
                  key={role}
                  label={role}
                  count={variantCounts[role] ?? 0}
                  active={kindFilter === role}
                  onClick={() => {
                    setKindFilter(role);
                    setIndex(0);
                  }}
                />
              ) : null,
            )}
          </div>

          <div className="mt-4 panel flex flex-wrap items-center gap-3 p-3">
            <span className="section-label">Code layout</span>
            <Toggle
              label="Diff highlights"
              active={viewMode === "diff"}
              onClick={() => setViewMode("diff")}
            />
            <Toggle
              label="All columns"
              active={viewMode === "split"}
              onClick={() => setViewMode("split")}
            />
            <span className="hidden h-4 w-px bg-[var(--border)] sm:inline" />
            <Toggle
              label="Prompt expanded"
              active={showPrompt}
              onClick={() => setShowPrompt((v) => !v)}
            />
            <div className="ml-auto flex items-center gap-2">
              <input
                type="text"
                placeholder="Annotator name"
                value={annotator}
                onChange={(e) => setAnnotator(e.target.value)}
                className="input-field text-sm"
              />
              <label className="btn-secondary cursor-pointer">
                Import reviews
                <input
                  type="file"
                  accept=".json"
                  className="hidden"
                  onChange={handleImportReviews}
                />
              </label>
              <button
                type="button"
                onClick={exportReviews}
                className="btn-primary"
              >
                Export reviews
              </button>
            </div>
          </div>

          {exportStatus && (
            <p className="mt-2 text-sm text-[var(--text-muted)]">{exportStatus}</p>
          )}

          {filteredItems.length === 0 ? (
            <p className="mt-8 text-center text-sm text-zinc-500">
              No cards match the current filters.
            </p>
          ) : current ? (
            <div className="mt-6">
              <ReviewCard
                item={current}
                index={index}
                total={filteredItems.length}
                decision={currentDecision}
                annotation={reviews[current.id]?.annotation ?? ""}
                viewMode={viewMode}
                showPrompt={showPrompt}
                onShowPromptChange={setShowPrompt}
                onCommentChange={setDraftComment}
                onCommentSave={saveComment}
                onApprove={() => applyDecision("approved", true)}
                onDiscard={() => applyDecision("discarded", true)}
                onClear={() => applyDecision("pending")}
                onAnnotationChange={handleUpdateAnnotation}
                onPrev={() => setIndex((i) => Math.max(0, i - 1))}
                onNext={() =>
                  setIndex((i) => Math.min(filteredItems.length - 1, i + 1))
                }
              />
            </div>
          ) : null}
        </>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div className="stat-card">
      <p className="section-label">{label}</p>
      <p className="mt-1 text-2xl font-semibold tabular-nums text-[var(--text)]">
        {value}
      </p>
    </div>
  );
}

function KindChip({
  label,
  count,
  active,
  onClick,
}: {
  label: string;
  count: number;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`chip rounded-full ${active ? "chip-active" : ""}`}
    >
      {label} ({count})
    </button>
  );
}

function Toggle({
  label,
  active,
  onClick,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`rounded-lg px-3 py-1.5 text-sm font-medium ${
        active ? "chip-active" : "chip"
      }`}
    >
      {label}
    </button>
  );
}

function attachReviewsToSourceRows(
  schema: DatasetSchema,
  records: Record<string, unknown>[],
  reviewMatrix: ReviewMatrix,
): Record<string, unknown>[] {
  const idField =
    schema === "mutants" || schema === "validated_mutants"
      ? "mutant_id"
      : schema === "confound_candidates" || schema === "validated_confounded"
        ? "variant_id"
        : schema === "tasks" || schema === "validated_tasks" || schema === "dropped_tasks"
          ? "task_id"
          : null;

  return records.map((row) => {
    const withReviews = { ...row } as Record<string, unknown>;

    if (schema === "phase2" || schema === "icse26") {
      const impls = row.implementations;
      if (Array.isArray(impls)) {
        const implementation_reviews: Record<string, unknown> = {};
        for (const impl of impls) {
          if (
            impl &&
            typeof impl === "object" &&
            "implementation_id" in impl &&
            typeof (impl as { implementation_id?: unknown }).implementation_id === "string"
          ) {
            const implId = (impl as { implementation_id: string }).implementation_id;
            implementation_reviews[implId] = reviewMatrix[implId] ?? {};
          }
        }
        withReviews._review = { implementation_reviews };
      }
      return withReviews;
    }

    if (idField && typeof row[idField] === "string") {
      const itemId = String(row[idField]);
      withReviews._review = {
        reviewers: reviewMatrix[itemId] ?? {},
      };
    }

    return withReviews;
  });
}

function mergeReviewMatrices(base: ReviewMatrix, incoming: ReviewMatrix): ReviewMatrix {
  const merged: ReviewMatrix = { ...base };
  for (const [itemId, byReviewer] of Object.entries(incoming)) {
    merged[itemId] = {
      ...(merged[itemId] ?? {}),
      ...byReviewer,
    };
  }
  return merged;
}

function extractEmbeddedReviewMatrix(
  schema: DatasetSchema,
  records: Record<string, unknown>[],
): ReviewMatrix {
  const matrix: ReviewMatrix = {};
  const idField =
    schema === "mutants" || schema === "validated_mutants"
      ? "mutant_id"
      : schema === "confound_candidates" || schema === "validated_confounded"
        ? "variant_id"
        : schema === "tasks" || schema === "validated_tasks" || schema === "dropped_tasks"
          ? "task_id"
          : null;

  for (const row of records) {
    const review = row._review;
    if (!review || typeof review !== "object") continue;

    if (schema === "phase2" || schema === "icse26") {
      const implReviews = (review as { implementation_reviews?: unknown })
        .implementation_reviews;
      if (!implReviews || typeof implReviews !== "object") continue;
      for (const [implId, byReviewer] of Object.entries(
        implReviews as Record<string, unknown>,
      )) {
        if (!byReviewer || typeof byReviewer !== "object") continue;
        matrix[implId] = byReviewer as Record<string, import("@/lib/types").ReviewRecord>;
      }
      continue;
    }

    if (!idField || typeof row[idField] !== "string") continue;
    const itemId = String(row[idField]);
    const byReviewer = (review as { reviewers?: unknown }).reviewers;
    if (byReviewer && typeof byReviewer === "object") {
      matrix[itemId] = byReviewer as Record<string, import("@/lib/types").ReviewRecord>;
    }
  }

  return matrix;
}
