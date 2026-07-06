"use client";

import type { ReviewDecision, ReviewItem } from "@/lib/types";
import { CodeDiff } from "./CodeDiff";
import { CodePanel } from "./CodePanel";
import { PromptPanel } from "./PromptPanel";
import { OperatorBanner } from "./OperatorBanner";
import { TaskCodeView } from "./TaskCodeView";

type ReviewCardProps = {
  item: ReviewItem;
  index: number;
  total: number;
  decision: ReviewDecision;
  annotation: string;
  viewMode: "diff" | "split";
  showPrompt: boolean;
  onShowPromptChange?: (open: boolean) => void;
  onAnnotationChange?: (value: string) => void;
  onApprove: () => void;
  onDiscard: () => void;
  onClear: () => void;
  onPrev: () => void;
  onNext: () => void;
};

const decisionStyles: Record<ReviewDecision, string> = {
  pending: "bg-zinc-100 text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300",
  approved: "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300",
  discarded: "bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-300",
};

export function ReviewCard({
  item,
  index,
  total,
  decision,
  annotation,
  viewMode,
  showPrompt,
  onShowPromptChange,
  onAnnotationChange,
  onApprove,
  onDiscard,
  onClear,
  onPrev,
  onNext,
}: ReviewCardProps) {
  const isBundle = item.variants && item.variants.length > 0;
  const entryPoint = item.metadata.entry_point as string | undefined;
  const extraConfounds = item.metadata.extra_confounds as string[] | undefined;

  const hasBaseline =
    !item.baselineCode.startsWith("# Upload") &&
    !item.baselineCode.startsWith("# Load") &&
    !item.baselineCode.startsWith("# Mutant");

  return (
    <article className="panel overflow-hidden">
      <header className="flex flex-col gap-3 border-b border-[var(--border)] bg-[var(--surface-elevated)] px-5 py-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <span
              className={`rounded-full px-2.5 py-0.5 text-xs font-medium capitalize ${decisionStyles[decision]}`}
            >
              {decision}
            </span>
            {isBundle &&
              item.variants?.map((v) => (
                <span
                  key={v.id}
                  className="rounded-full bg-white px-2 py-0.5 text-xs font-medium capitalize text-zinc-600 ring-1 ring-zinc-200 dark:bg-zinc-800 dark:text-zinc-400 dark:ring-zinc-700"
                >
                  {v.role}
                </span>
              ))}
          </div>
          <h2 className="mt-1.5 font-mono text-sm font-semibold text-zinc-900 dark:text-zinc-50 sm:text-base">
            {item.taskId}
          </h2>
          <p className="mt-0.5 text-xs text-zinc-500">
            {index + 1} / {total}
            {item.subtitle ? ` · ${item.subtitle}` : ""}
          </p>
        </div>

        <div className="flex shrink-0 gap-2">
          <button
            type="button"
            onClick={onPrev}
            disabled={index === 0}
            className="rounded-lg border border-zinc-300 bg-white px-3 py-1.5 text-sm font-medium disabled:opacity-40 dark:border-zinc-600 dark:bg-zinc-900"
          >
            ← Prev
          </button>
          <button
            type="button"
            onClick={onNext}
            disabled={index >= total - 1}
            className="rounded-lg border border-zinc-300 bg-white px-3 py-1.5 text-sm font-medium disabled:opacity-40 dark:border-zinc-600 dark:bg-zinc-900"
          >
            Next →
          </button>
        </div>
      </header>

      <OperatorBanner item={item} />

      {item.prompt && (
        <PromptPanel
          prompt={item.prompt}
          entryPoint={entryPoint}
          open={showPrompt}
          onOpenChange={onShowPromptChange}
        />
      )}

      {extraConfounds && extraConfounds.length > 0 && (
        <p className="border-b border-zinc-200 px-5 py-2 text-xs text-amber-700 dark:border-zinc-800 dark:text-amber-400">
          +{extraConfounds.length} additional confound variant(s) not shown — narrow
          with filters or load a confound-focused split.
        </p>
      )}

      <MetadataPanel item={item} />

      {isBundle ? (
        <TaskCodeView variants={item.variants!} viewMode={viewMode} />
      ) : (
        <div className="p-5">
          {viewMode === "diff" && hasBaseline ? (
            <CodeDiff
              oldValue={item.baselineCode}
              newValue={item.candidateCode}
              oldTitle={item.baselineLabel}
              newTitle={item.candidateLabel}
            />
          ) : (
            <div className="grid gap-4 lg:grid-cols-2">
              <CodePanel code={item.baselineCode} label={item.baselineLabel} />
              <CodePanel code={item.candidateCode} label={item.candidateLabel} />
            </div>
          )}
        </div>
      )}

      <div className="border-t border-[var(--border)] bg-[var(--surface-elevated)] p-5">
        <label className="block text-sm font-medium text-[var(--text)] mb-2">
          Notes / Annotations
        </label>
        <textarea
          value={annotation}
          onChange={(e) => onAnnotationChange?.(e.target.value)}
          placeholder="Add any specific details, patterns, or reasoning for cross-reviewing..."
          className="w-full rounded-md border border-[var(--border)] bg-white p-3 text-sm text-[var(--text)] focus:border-[var(--accent)] focus:outline-none focus:ring-1 focus:ring-[var(--accent)] dark:bg-zinc-900"
          rows={3}
        />
      </div>

      <footer className="flex flex-wrap items-center justify-between gap-3 border-t border-[var(--border)] bg-[var(--surface-elevated)] px-5 py-4">
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            onClick={onApprove}
            className="rounded-lg bg-emerald-600 px-4 py-2 text-sm font-semibold text-white hover:bg-emerald-500"
          >
            Approve
          </button>
          <button
            type="button"
            onClick={onDiscard}
            className="rounded-lg bg-red-600 px-4 py-2 text-sm font-semibold text-white hover:bg-red-500"
          >
            Discard
          </button>
          <button
            type="button"
            onClick={onClear}
            className="rounded-lg border border-zinc-300 bg-white px-4 py-2 text-sm font-medium dark:border-zinc-600 dark:bg-zinc-900"
          >
            Reset
          </button>
        </div>
        <p className="text-xs text-zinc-500">
          <kbd className="rounded bg-zinc-200 px-1 dark:bg-zinc-800">a</kbd> approve ·{" "}
          <kbd className="rounded bg-zinc-200 px-1 dark:bg-zinc-800">d</kbd> discard ·{" "}
          <kbd className="rounded bg-zinc-200 px-1 dark:bg-zinc-800">→</kbd> next
        </p>
      </footer>
    </article>
  );
}

function MetadataPanel({ item }: { item: ReviewItem }) {
  const meta = item.metadata;
  const fingerprint = meta.failure_fingerprint as string | undefined;
  const details = meta.details;
  const site = meta.site as Record<string, unknown> | undefined;
  const hasExtra =
    fingerprint ||
    details ||
    site ||
    item.test ||
    meta.source_mutant_id ||
    meta.canonical_details;

  if (!hasExtra) return null;

  return (
    <details className="border-b border-zinc-200 dark:border-zinc-800">
      <summary className="cursor-pointer px-5 py-2.5 text-xs font-medium text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300">
        Validation &amp; metadata
      </summary>
      <div className="space-y-3 px-5 pb-4">
        {item.test && (
          <div>
            <p className="text-xs font-medium text-zinc-500">Test harness</p>
            <pre className="mt-1 max-h-32 overflow-auto rounded bg-zinc-100 p-2 font-mono text-xs dark:bg-zinc-900">
              {item.test}
            </pre>
          </div>
        )}
        {fingerprint && (
          <p className="font-mono text-xs text-zinc-600 dark:text-zinc-400">
            fingerprint: {fingerprint}
          </p>
        )}
        {typeof meta.source_mutant_id === "string" && (
          <p className="font-mono text-xs text-zinc-600 dark:text-zinc-400">
            source_mutant_id: {meta.source_mutant_id}
          </p>
        )}
        {site && (
          <pre className="overflow-auto rounded bg-zinc-100 p-2 font-mono text-xs dark:bg-zinc-900">
            {JSON.stringify(site, null, 2)}
          </pre>
        )}
        {details !== undefined && (
          <pre className="max-h-48 overflow-auto rounded bg-zinc-100 p-2 font-mono text-xs dark:bg-zinc-900">
            {JSON.stringify(details, null, 2)}
          </pre>
        )}
      </div>
    </details>
  );
}
