"use client";

import type { DatasetSchema } from "@/lib/types";

type FileLoaderProps = {
  onLoadPrimary: (file: File) => void;
  onLoadTasks?: (file: File) => void;
  onLoadDemo: () => void;
  loading: boolean;
  error: string | null;
  datasetName: string | null;
  schema: DatasetSchema | null;
  itemCount: number;
  truncated?: boolean;
  maxUploadRecords: number;
  onMaxUploadRecordsChange: (n: number) => void;
};

export function FileLoader({
  onLoadPrimary,
  onLoadTasks,
  onLoadDemo,
  loading,
  error,
  datasetName,
  schema,
  itemCount,
  truncated,
  maxUploadRecords,
  onMaxUploadRecordsChange,
}: FileLoaderProps) {
  return (
    <section className="rounded-xl border border-zinc-200 bg-white p-5 shadow-sm dark:border-zinc-800 dark:bg-zinc-950">
      <h2 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
        Manual upload
      </h2>
      <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">
        For HPC copies or files outside the repo. Large uploads are capped (use presets
        above when running locally). Pair <code className="text-xs">mutants.jsonl</code>{" "}
        with <code className="text-xs">tasks.jsonl</code> for canonical diffs; confounds
        pair best with <code className="text-xs">validated_mutants.jsonl</code>.
      </p>

      <label className="mt-3 flex items-center gap-2 text-sm text-zinc-600 dark:text-zinc-400">
        Upload row cap
        <input
          type="number"
          min={50}
          max={5000}
          value={maxUploadRecords}
          onChange={(e) => onMaxUploadRecordsChange(Number(e.target.value))}
          className="w-24 rounded border border-zinc-300 px-2 py-1 text-sm dark:border-zinc-700 dark:bg-zinc-950"
        />
      </label>

      <div className="mt-4 flex flex-wrap gap-3">
        <label className="inline-flex cursor-pointer items-center rounded-lg border border-zinc-300 px-4 py-2 text-sm font-medium text-zinc-800 hover:bg-zinc-50 dark:border-zinc-700 dark:text-zinc-200 dark:hover:bg-zinc-900">
          JSONL file
          <input
            type="file"
            accept=".jsonl,.json,.txt,.zst"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) onLoadPrimary(file);
              e.target.value = "";
            }}
          />
        </label>

        {onLoadTasks && (
          <label className="inline-flex cursor-pointer items-center rounded-lg border border-zinc-300 px-4 py-2 text-sm font-medium text-zinc-800 hover:bg-zinc-50 dark:border-zinc-700 dark:text-zinc-200 dark:hover:bg-zinc-900">
            tasks.jsonl
            <input
              type="file"
              accept=".jsonl,.json,.txt"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) onLoadTasks(file);
                e.target.value = "";
              }}
            />
          </label>
        )}

        <button
          type="button"
          onClick={onLoadDemo}
          className="rounded-lg border border-dashed border-zinc-300 px-4 py-2 text-sm font-medium text-zinc-700 hover:bg-zinc-50 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-900"
        >
          Demo sample
        </button>
      </div>

      {loading && <p className="mt-3 text-sm text-zinc-500">Parsing file…</p>}
      {error && (
        <p className="mt-3 text-sm text-red-600 dark:text-red-400">{error}</p>
      )}
      {datasetName && !error && (
        <p className="mt-3 text-sm text-emerald-700 dark:text-emerald-400">
          Loaded <span className="font-medium">{datasetName}</span> ({schema}) ·{" "}
          {itemCount} cards
          {truncated && (
            <span className="text-amber-700 dark:text-amber-400">
              {" "}
              · truncated to row cap
            </span>
          )}
        </p>
      )}
    </section>
  );
}
