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
import {
  countVariantsByRole,
  filterItemVariants,
  itemHasVariantRole,
} from "@/lib/review-items";
import { uploadCategoryById, type UploadCategoryId } from "@/lib/upload-categories";
import type { CodeVariantRole, LoadedDataset, ReviewDecision, ReviewRecord } from "@/lib/types";
import { DataSourcePanel } from "./DataSourcePanel";
import { ReviewCard } from "./ReviewCard";

type FilterMode = "all" | "pending" | "approved" | "discarded";

export function ReviewerApp() {
  const [dataset, setDataset] = useState<LoadedDataset | null>(null);
  const [reviews, setReviews] = useState<Record<string, ReviewRecord>>({});
  const [index, setIndex] = useState(0);
  const [filter, setFilter] = useState<FilterMode>("all");
  const [kindFilter, setKindFilter] = useState<CodeVariantRole | "all">("all");
  const [viewMode, setViewMode] = useState<"diff" | "split">("diff");
  const [showPrompt, setShowPrompt] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [maxUploadRecords, setMaxUploadRecords] = useState(500);
  const [annotator, setAnnotator] = useState("");

  const handleDatasetLoaded = useCallback((loaded: LoadedDataset) => {
    setDataset(loaded);
    setReviews(loadReviews(loaded.name));
    setIndex(0);
    setError(null);
  }, []);

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
      tasksText?: string;
      mutantsText?: string;
      fileName: string;
      truncated: boolean;
    }) => {
      const cat = uploadCategoryById(params.categoryId);
      const schema =
        cat?.schema && cat.schema !== "auto"
          ? cat.schema
          : inferSchemaFromFilename(
              params.fileName,
              JSON.parse(params.primaryText.split("\n")[0] || "{}"),
            );
      loadFromTexts(params.fileName, params.primaryText, {
        schema,
        tasksText: params.tasksText,
        mutantsText: params.mutantsText,
        truncated: params.truncated,
      });
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
        if (kindFilter !== "all") {
          if (item.variants) {
            if (!itemHasVariantRole(item, kindFilter)) return false;
          } else if (item.kind !== kindFilter) {
            return false;
          }
        }
        return true;
      })
      .map((item) => filterItemVariants(item, kindFilter))
      .filter((item) => !item.variants || item.variants.length > 0);
  }, [dataset, reviews, filter, kindFilter]);

  useEffect(() => {
    if (index >= filteredItems.length) {
      setIndex(Math.max(0, filteredItems.length - 1));
    }
  }, [filteredItems.length, index]);

  const current = filteredItems[index] ?? null;
  const currentDecision: ReviewDecision = current
    ? reviews[current.id]?.decision ?? "pending"
    : "pending";

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
      if (!current || event.target instanceof HTMLInputElement) return;
      if (event.key === "ArrowRight" || event.key === "j") {
        event.preventDefault();
        setIndex((i) => Math.min(i + 1, filteredItems.length - 1));
      }
      if (event.key === "ArrowLeft" || event.key === "k") {
        event.preventDefault();
        setIndex((i) => Math.max(i - 1, 0));
      }
      if (event.key === "a") applyDecision("approved", true);
      if (event.key === "d") applyDecision("discarded", true);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [current, filteredItems.length, applyDecision]);

  const exportReviews = () => {
    if (!dataset) return;
    const payload = buildExport(
      dataset.name,
      dataset.schema,
      dataset.items.map((item) => item.id),
    );
    downloadJson(
      `reviews-${dataset.name.replace(/[^\w.-]+/g, "_")}-${Date.now()}.json`,
      payload,
    );
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
