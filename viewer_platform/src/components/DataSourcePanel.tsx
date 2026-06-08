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
import {
  UPLOAD_CATEGORIES,
  uploadCategoryById,
  type UploadCategoryId,
} from "@/lib/upload-categories";
import type { DatasetBrowseResponse, LoadedDataset } from "@/lib/types";

type DataSourcePanelProps = {
  onLoaded: (loaded: LoadedDataset) => void;
  onLoadingChange: (loading: boolean) => void;
  onError: (error: string | null) => void;
  onUploadLoad: (params: {
    categoryId: UploadCategoryId;
    primaryText: string;
    tasksText?: string;
    mutantsText?: string;
    fileName: string;
    truncated: boolean;
  }) => void;
  onLoadDemo: () => void;
  loading: boolean;
  maxUploadRecords: number;
  onMaxUploadRecordsChange: (n: number) => void;
};

type SourceTab = "repo" | "upload";

export function DataSourcePanel({
  onLoaded,
  onLoadingChange,
  onError,
  onUploadLoad,
  onLoadDemo,
  loading,
  maxUploadRecords,
  onMaxUploadRecordsChange,
}: DataSourcePanelProps) {
  const [tab, setTab] = useState<SourceTab>("upload");
  const [browse, setBrowse] = useState<DatasetBrowseResponse | null>(null);
  const [repoAvailable, setRepoAvailable] = useState(false);

  const [presetId, setPresetId] = useState<IntermediatePresetId>("validated_mutants");
  const [offset, setOffset] = useState(0);
  const [limit, setLimit] = useState(200);
  const [status, setStatus] = useState("");
  const [taskId, setTaskId] = useState("");
  const [phase2Family, setPhase2Family] = useState(DEFAULT_PHASE2_FAMILY);
  const [phase2Fold, setPhase2Fold] = useState(0);
  const [phase2Split, setPhase2Split] = useState<Phase2Split>("fit");

  const [uploadCategory, setUploadCategory] = useState<UploadCategoryId>("phase2");
  const [primaryFile, setPrimaryFile] = useState<File | null>(null);
  const [tasksFile, setTasksFile] = useState<File | null>(null);
  const [mutantsFile, setMutantsFile] = useState<File | null>(null);

  const category = uploadCategoryById(uploadCategory);

  useEffect(() => {
    fetchDatasetBrowse().then((data) => {
      if (data) {
        setBrowse(data);
        setRepoAvailable(true);
        setTab("repo");
        if (data.phase2.families.length > 0) {
          setPhase2Family(
            data.phase2.families.includes(DEFAULT_PHASE2_FAMILY)
              ? DEFAULT_PHASE2_FAMILY
              : data.phase2.families[0],
          );
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

  const handleUploadSubmit = async () => {
    if (!primaryFile || !category) {
      onError("Choose a category and select a primary JSONL file.");
      return;
    }
    if (primaryFile.name.endsWith(".zst")) {
      onError("Decompress .zst to .jsonl before uploading (or use the GitHub repo tab locally).");
      return;
    }
    onLoadingChange(true);
    onError(null);
    try {
      const { parseJsonlFile } = await import("@/lib/jsonl-stream");
      const cap = limit || category.defaultLimit || maxUploadRecords;
      const { text, truncated } = await parseJsonlFile(primaryFile, cap);
      const tasksText = tasksFile ? await tasksFile.text() : undefined;
      const mutantsText = mutantsFile ? await mutantsFile.text() : undefined;
      onUploadLoad({
        categoryId: uploadCategory,
        primaryText: text,
        tasksText,
        mutantsText,
        fileName: primaryFile.name,
        truncated,
      });
    } catch (err) {
      onError(err instanceof Error ? err.message : "Upload failed");
      onLoadingChange(false);
    }
  };

  return (
    <section className="panel overflow-hidden">
      <div className="flex flex-col gap-3 border-b border-[var(--border)] px-5 py-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-sm font-semibold text-[var(--text)]">Data source</h2>
          <p className="mt-0.5 text-xs text-[var(--text-muted)]">
            {repoAvailable
              ? "GitHub repo on disk (local) or upload JSONL (Vercel / anywhere)."
              : "Upload JSONL files — ideal for Vercel deployment."}
          </p>
        </div>
        <div className="flex rounded-lg bg-[var(--surface-elevated)] p-1">
          <TabButton active={tab === "upload"} onClick={() => setTab("upload")}>
            Upload files
          </TabButton>
          <TabButton
            active={tab === "repo"}
            onClick={() => setTab("repo")}
            disabled={!repoAvailable}
          >
            From repo
          </TabButton>
        </div>
      </div>

      <div className="p-5">
        {tab === "upload" ? (
          <div className="space-y-5">
            <label className="block">
              <span className="mb-1.5 block text-xs font-medium text-[var(--text-muted)]">
                File category
              </span>
              <select
                value={uploadCategory}
                onChange={(e) => {
                  setUploadCategory(e.target.value as UploadCategoryId);
                  setPrimaryFile(null);
                }}
                className="select-field w-full"
              >
                {UPLOAD_CATEGORIES.map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.label}
                  </option>
                ))}
              </select>
              {category && (
                <p className="mt-1.5 text-xs text-[var(--text-muted)]">
                  {category.description}
                </p>
              )}
            </label>

            <FilePick
              label="Primary JSONL"
              required
              file={primaryFile}
              onFile={(f) => setPrimaryFile(f)}
            />

            {category?.needsTasksJoin && (
              <FilePick
                label="Join · tasks.jsonl (canonical + prompt)"
                file={tasksFile}
                onFile={(f) => setTasksFile(f)}
              />
            )}

            {category?.needsMutantsJoin && (
              <FilePick
                label="Join · validated_mutants.jsonl (source mutant code)"
                file={mutantsFile}
                onFile={(f) => setMutantsFile(f)}
              />
            )}

            <SliceControls
              offset={offset}
              limit={limit}
              maxUploadRecords={maxUploadRecords}
              onOffsetChange={setOffset}
              onLimitChange={setLimit}
              onMaxUploadRecordsChange={onMaxUploadRecordsChange}
              status={status}
              taskId={taskId}
              onStatusChange={setStatus}
              onTaskIdChange={setTaskId}
            />

            <div className="flex flex-wrap gap-3">
              <button
                type="button"
                disabled={loading || !primaryFile}
                onClick={handleUploadSubmit}
                className="btn-primary"
              >
                {loading ? "Loading…" : "Load uploaded data"}
              </button>
              <button type="button" onClick={onLoadDemo} className="btn-secondary">
                Demo sample
              </button>
            </div>
          </div>
        ) : (
          <div className="space-y-6">
            <div>
              <h3 className="section-label">Phase 1 · intermediate</h3>
              <div className="mt-2 flex flex-wrap gap-2">
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
                      className={`chip ${presetId === preset.id ? "chip-active" : ""} ${!available ? "opacity-40" : ""}`}
                      title={preset.description}
                    >
                      {preset.label}
                    </button>
                  );
                })}
              </div>
              <button
                type="button"
                onClick={() => loadPreset(presetId)}
                className="btn-primary mt-3"
              >
                Load {presetById(presetId)?.label}
              </button>
            </div>

            <div>
              <h3 className="section-label">Phase 2 · splits</h3>
              <div className="mt-2 grid gap-3 sm:grid-cols-3">
                <Field label="Family">
                  <select
                    value={phase2Family}
                    onChange={(e) => setPhase2Family(e.target.value)}
                    className="select-field w-full"
                  >
                    {browse?.phase2.families.map((f) => (
                      <option key={f} value={f}>
                        {f}
                      </option>
                    ))}
                  </select>
                </Field>
                <Field label="Fold">
                  <select
                    value={phase2Fold}
                    onChange={(e) => setPhase2Fold(Number(e.target.value))}
                    className="select-field w-full"
                  >
                    {browse?.phase2.folds.map((f) => (
                      <option key={f} value={f}>
                        fold_{f}
                      </option>
                    ))}
                  </select>
                </Field>
                <Field label="Split">
                  <select
                    value={phase2Split}
                    onChange={(e) => setPhase2Split(e.target.value as Phase2Split)}
                    className="select-field w-full"
                  >
                    {PHASE2_SPLITS.map((s) => (
                      <option key={s} value={s}>
                        {s}.jsonl
                      </option>
                    ))}
                  </select>
                </Field>
              </div>
              <button type="button" onClick={loadPhase2} className="btn-primary mt-3">
                Load phase2 split
              </button>
            </div>

            <SliceControls
              offset={offset}
              limit={limit}
              maxUploadRecords={maxUploadRecords}
              onOffsetChange={setOffset}
              onLimitChange={setLimit}
              onMaxUploadRecordsChange={onMaxUploadRecordsChange}
              status={status}
              taskId={taskId}
              onStatusChange={setStatus}
              onTaskIdChange={setTaskId}
            />
          </div>
        )}
      </div>
    </section>
  );
}

function TabButton({
  active,
  onClick,
  disabled,
  children,
}: {
  active: boolean;
  onClick: () => void;
  disabled?: boolean;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      className={`rounded-md px-4 py-2 text-xs font-medium transition-colors ${
        active
          ? "bg-[var(--accent)] text-white shadow-sm"
          : "text-[var(--text-muted)] hover:text-[var(--text)] disabled:opacity-40"
      }`}
    >
      {children}
    </button>
  );
}

function FilePick({
  label,
  required,
  file,
  onFile,
}: {
  label: string;
  required?: boolean;
  file: File | null;
  onFile: (f: File | null) => void;
}) {
  return (
    <label className="flex cursor-pointer flex-col gap-1.5 rounded-lg border border-dashed border-[var(--border)] bg-[var(--surface-elevated)] px-4 py-3 transition-colors hover:border-[var(--accent)]/50">
      <span className="text-xs font-medium text-[var(--text)]">
        {label}
        {required && <span className="text-red-400"> *</span>}
      </span>
      <span className="text-sm text-[var(--accent)]">
        {file ? file.name : "Click to choose file"}
      </span>
      <input
        type="file"
        accept=".jsonl,.json,.txt"
        className="hidden"
        onChange={(e) => {
          onFile(e.target.files?.[0] ?? null);
          e.target.value = "";
        }}
      />
    </label>
  );
}

function SliceControls(props: {
  offset: number;
  limit: number;
  maxUploadRecords: number;
  onOffsetChange: (n: number) => void;
  onLimitChange: (n: number) => void;
  onMaxUploadRecordsChange: (n: number) => void;
  status: string;
  taskId: string;
  onStatusChange: (s: string) => void;
  onTaskIdChange: (s: string) => void;
}) {
  return (
    <div className="rounded-lg border border-[var(--border)] bg-[var(--surface-elevated)] p-4">
      <p className="section-label mb-3">Slice &amp; filters</p>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Field label="Line offset">
          <input
            type="number"
            min={0}
            value={props.offset}
            onChange={(e) => props.onOffsetChange(Number(e.target.value))}
            className="input-field"
          />
        </Field>
        <Field label="Max rows">
          <input
            type="number"
            min={1}
            max={5000}
            value={props.limit}
            onChange={(e) => props.onLimitChange(Number(e.target.value))}
            className="input-field"
          />
        </Field>
        <Field label="Status">
          <select
            value={props.status}
            onChange={(e) => props.onStatusChange(e.target.value)}
            className="select-field w-full"
          >
            <option value="">any</option>
            <option value="fail">fail</option>
            <option value="pass">pass</option>
            <option value="timeout">timeout</option>
          </select>
        </Field>
        <Field label="Task ID">
          <input
            type="text"
            placeholder="BigCodeBench/0"
            value={props.taskId}
            onChange={(e) => props.onTaskIdChange(e.target.value)}
            className="input-field"
          />
        </Field>
      </div>
    </div>
  );
}

function Field({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <label className="block text-sm">
      <span className="mb-1 block text-xs text-[var(--text-muted)]">{label}</span>
      {children}
    </label>
  );
}
