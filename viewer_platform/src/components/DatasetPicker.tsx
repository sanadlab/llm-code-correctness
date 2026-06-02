"use client";

import { useCallback, useEffect, useState } from "react";

import {
  DEFAULT_PHASE2_FAMILY,
  INTERMEDIATE_PRESETS,
  type IntermediatePresetId,
  PHASE2_SPLITS,
  type Phase2Split,
  phase2JsonlPath,
  presetById,
} from "@/lib/catalog";
import { fetchDatasetBrowse, fetchDatasetSample } from "@/lib/dataset-client";
import type { DatasetBrowseResponse, LoadedDataset } from "@/lib/types";

type DatasetPickerProps = {
  onLoaded: (dataset: LoadedDataset) => void;
  onLoadingChange: (loading: boolean) => void;
  onError: (error: string | null) => void;
};

export function DatasetPicker({
  onLoaded,
  onLoadingChange,
  onError,
}: DatasetPickerProps) {
  const [browse, setBrowse] = useState<DatasetBrowseResponse | null>(null);
  const [apiAvailable, setApiAvailable] = useState(false);

  const [presetId, setPresetId] = useState<IntermediatePresetId>("validated_mutants");
  const [offset, setOffset] = useState(0);
  const [limit, setLimit] = useState(200);
  const [status, setStatus] = useState("");
  const [taskId, setTaskId] = useState("");

  const [phase2Family, setPhase2Family] = useState(DEFAULT_PHASE2_FAMILY);
  const [phase2Fold, setPhase2Fold] = useState(0);
  const [phase2Split, setPhase2Split] = useState<Phase2Split>("fit");

  useEffect(() => {
    fetchDatasetBrowse().then((data) => {
      if (data) {
        setBrowse(data);
        setApiAvailable(true);
        if (data.phase2.families.length > 0) {
          setPhase2Family(data.phase2.families.includes(DEFAULT_PHASE2_FAMILY)
            ? DEFAULT_PHASE2_FAMILY
            : data.phase2.families[0]);
        }
      }
    });
  }, []);

  const loadPreset = useCallback(
    async (id: IntermediatePresetId) => {
      const preset = presetById(id);
      if (!preset) return;
      onLoadingChange(true);
      onError(null);
      try {
        const available = browse?.intermediate.find((f) => f.id === id);
        const path = available?.path ?? preset.path;
        const dataset = await fetchDatasetSample({
          path: path.replace(/\.zst$/, ""),
          schema: preset.schema,
          offset,
          limit: limit || preset.defaultLimit,
          status: status || undefined,
          taskId: taskId || undefined,
          joinTasks: preset.needsTasksJoin,
          joinMutants: preset.joinMutantsForConfound,
        });
        onLoaded(dataset);
      } catch (err) {
        onError(err instanceof Error ? err.message : "Load failed");
      } finally {
        onLoadingChange(false);
      }
    },
    [browse, offset, limit, status, taskId, onLoaded, onLoadingChange, onError],
  );

  const loadPhase2 = useCallback(async () => {
    onLoadingChange(true);
    onError(null);
    try {
      const path = phase2JsonlPath(phase2Family, phase2Fold, phase2Split);
      const dataset = await fetchDatasetSample({
        path,
        schema: "phase2",
        offset,
        limit,
        taskId: taskId || undefined,
      });
      onLoaded(dataset);
    } catch (err) {
      onError(err instanceof Error ? err.message : "Load failed");
    } finally {
      onLoadingChange(false);
    }
  }, [
    phase2Family,
    phase2Fold,
    phase2Split,
    offset,
    limit,
    taskId,
    onLoaded,
    onLoadingChange,
    onError,
  ]);

  if (!apiAvailable) {
    return (
      <p className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-200">
        Local dataset API unavailable (deployed build or missing{" "}
        <code className="text-xs">dataset/bigcodebench</code>). Use file upload below
        or run <code className="text-xs">npm run dev</code> from the repo with decompressed
        data.
      </p>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-xs font-semibold uppercase tracking-wide text-zinc-500">
          Phase 1 · intermediate
        </h3>
        <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">
          Pipeline artifacts under{" "}
          <code className="text-xs">dataset/bigcodebench/intermediate/</code>. Large
          files are streamed with a row limit.
        </p>
        <div className="mt-3 flex flex-wrap gap-2">
          {INTERMEDIATE_PRESETS.map((preset) => {
            const available = browse?.intermediate.some((f) => f.id === preset.id);
            return (
              <button
                key={preset.id}
                type="button"
                disabled={!available}
                onClick={() => {
                  setPresetId(preset.id);
                  setLimit(preset.defaultLimit);
                }}
                className={`rounded-lg px-3 py-1.5 text-xs font-medium ${
                  presetId === preset.id
                    ? "bg-emerald-700 text-white"
                    : available
                      ? "bg-zinc-100 text-zinc-800 dark:bg-zinc-800 dark:text-zinc-200"
                      : "cursor-not-allowed bg-zinc-50 text-zinc-400 dark:bg-zinc-900"
                }`}
                title={preset.description}
              >
                {preset.label}
              </button>
            );
          })}
        </div>
        <p className="mt-2 text-xs text-zinc-500">
          {presetById(presetId)?.description}
        </p>
        <button
          type="button"
          onClick={() => loadPreset(presetId)}
          className="mt-3 rounded-lg bg-zinc-900 px-4 py-2 text-sm font-medium text-white dark:bg-zinc-100 dark:text-zinc-900"
        >
          Load {presetById(presetId)?.label}
        </button>
      </div>

      <div>
        <h3 className="text-xs font-semibold uppercase tracking-wide text-zinc-500">
          Phase 2 · splits
        </h3>
        <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">
          Pre-built CV splits over <code className="text-xs">frozen.jsonl</code> — each
          line is a task with <code className="text-xs">implementations</code> (canonical,
          mutant, and/or confound).
        </p>
        <p className="mt-2 text-xs text-zinc-500">
          Filter by canonical / mutant / confound after loading using the chips above the
          review card.
        </p>
        <div className="mt-3 grid gap-3 sm:grid-cols-3">
          <label className="text-sm">
            <span className="mb-1 block text-xs text-zinc-500">Split family</span>
            <select
              value={phase2Family}
              onChange={(e) => setPhase2Family(e.target.value)}
              className="w-full rounded-lg border border-zinc-300 bg-white px-2 py-2 text-sm dark:border-zinc-700 dark:bg-zinc-950"
            >
              {browse?.phase2.families.map((f) => (
                <option key={f} value={f}>
                  {f}
                </option>
              ))}
            </select>
          </label>
          <label className="text-sm">
            <span className="mb-1 block text-xs text-zinc-500">Fold</span>
            <select
              value={phase2Fold}
              onChange={(e) => setPhase2Fold(Number(e.target.value))}
              className="w-full rounded-lg border border-zinc-300 bg-white px-2 py-2 text-sm dark:border-zinc-700 dark:bg-zinc-950"
            >
              {browse?.phase2.folds.map((f) => (
                <option key={f} value={f}>
                  fold_{f}
                </option>
              ))}
            </select>
          </label>
          <label className="text-sm">
            <span className="mb-1 block text-xs text-zinc-500">Split</span>
            <select
              value={phase2Split}
              onChange={(e) => setPhase2Split(e.target.value as Phase2Split)}
              className="w-full rounded-lg border border-zinc-300 bg-white px-2 py-2 text-sm dark:border-zinc-700 dark:bg-zinc-950"
            >
              {PHASE2_SPLITS.map((s) => (
                <option key={s} value={s}>
                  {s}.jsonl
                </option>
              ))}
            </select>
          </label>
        </div>
        <button
          type="button"
          onClick={loadPhase2}
          className="mt-3 rounded-lg bg-emerald-700 px-4 py-2 text-sm font-medium text-white hover:bg-emerald-600"
        >
          Load phase2/{phase2Family}/fold_{phase2Fold}/{phase2Split}.jsonl
        </button>
      </div>

      <Filters
        offset={offset}
        limit={limit}
        status={status}
        taskId={taskId}
        onOffsetChange={setOffset}
        onLimitChange={setLimit}
        onStatusChange={setStatus}
        onTaskIdChange={setTaskId}
      />
    </div>
  );
}

function Filters({
  offset,
  limit,
  status,
  taskId,
  onOffsetChange,
  onLimitChange,
  onStatusChange,
  onTaskIdChange,
}: {
  offset: number;
  limit: number;
  status: string;
  taskId: string;
  onOffsetChange: (n: number) => void;
  onLimitChange: (n: number) => void;
  onStatusChange: (s: string) => void;
  onTaskIdChange: (s: string) => void;
}) {
  return (
    <div className="rounded-lg border border-zinc-200 bg-zinc-50 p-4 dark:border-zinc-800 dark:bg-zinc-900/40">
      <h3 className="text-xs font-semibold uppercase tracking-wide text-zinc-500">
        Slice &amp; filters
      </h3>
      <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <label className="text-sm">
          <span className="mb-1 block text-xs text-zinc-500">Line offset</span>
          <input
            type="number"
            min={0}
            value={offset}
            onChange={(e) => onOffsetChange(Number(e.target.value))}
            className="w-full rounded-lg border border-zinc-300 px-2 py-2 text-sm dark:border-zinc-700 dark:bg-zinc-950"
          />
        </label>
        <label className="text-sm">
          <span className="mb-1 block text-xs text-zinc-500">Max rows</span>
          <input
            type="number"
            min={1}
            max={5000}
            value={limit}
            onChange={(e) => onLimitChange(Number(e.target.value))}
            className="w-full rounded-lg border border-zinc-300 px-2 py-2 text-sm dark:border-zinc-700 dark:bg-zinc-950"
          />
        </label>
        <label className="text-sm">
          <span className="mb-1 block text-xs text-zinc-500">Status (mutants / confounds)</span>
          <select
            value={status}
            onChange={(e) => onStatusChange(e.target.value)}
            className="w-full rounded-lg border border-zinc-300 px-2 py-2 text-sm dark:border-zinc-700 dark:bg-zinc-950"
          >
            <option value="">any</option>
            <option value="fail">fail</option>
            <option value="pass">pass</option>
            <option value="timeout">timeout</option>
          </select>
        </label>
        <label className="text-sm">
          <span className="mb-1 block text-xs text-zinc-500">Task ID</span>
          <input
            type="text"
            placeholder="BigCodeBench/0"
            value={taskId}
            onChange={(e) => onTaskIdChange(e.target.value)}
            className="w-full rounded-lg border border-zinc-300 px-2 py-2 text-sm dark:border-zinc-700 dark:bg-zinc-950"
          />
        </label>
      </div>
    </div>
  );
}
