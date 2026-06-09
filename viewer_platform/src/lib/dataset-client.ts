import type { DatasetBrowseResponse, LoadedDataset } from "./types";
import type { IntermediatePresetId } from "./catalog";
import { presetById } from "./catalog";
export async function fetchDatasetBrowse(): Promise<DatasetBrowseResponse | null> {
  const res = await fetch("/api/dataset/browse");
  if (!res.ok) return null;
  return res.json() as Promise<DatasetBrowseResponse>;
}

export interface SampleParams {
  path: string;
  schema?: string;
  offset?: number;
  limit?: number;
  status?: string;
  kind?: string;
  taskId?: string;
  joinTasks?: boolean;
  joinMutants?: boolean;
}

export async function fetchDatasetSample(
  params: SampleParams,
): Promise<LoadedDataset> {
  const qs = new URLSearchParams();
  qs.set("path", params.path);
  if (params.schema) qs.set("schema", params.schema);
  if (params.offset !== undefined) qs.set("offset", String(params.offset));
  if (params.limit !== undefined) qs.set("limit", String(params.limit));
  if (params.status) qs.set("status", params.status);
  if (params.kind) qs.set("kind", params.kind);
  if (params.taskId) qs.set("taskId", params.taskId);
  if (params.joinTasks) qs.set("joinTasks", "1");
  if (params.joinMutants) qs.set("joinMutants", "1");

  const res = await fetch(`/api/dataset/sample?${qs.toString()}`);
  const body = await res.json();
  if (!res.ok) {
    throw new Error(body.error ?? "Failed to load dataset sample");
  }

  return {
    name: body.name,
    schema: body.schema,
    items: body.items,
    truncated: body.truncated,
    sourcePath: body.path,
  };
}

export async function loadIntermediatePreset(
  id: IntermediatePresetId,
  options: {
    offset?: number;
    limit?: number;
    status?: string;
    taskId?: string;
  } = {},
): Promise<LoadedDataset> {
  const preset = presetById(id);
  if (!preset) throw new Error(`Unknown preset: ${id}`);

  return fetchDatasetSample({
    path: preset.path.replace(/\.zst$/, ""),
    schema: preset.schema,
    offset: options.offset,
    limit: options.limit ?? preset.defaultLimit,
    status: options.status,
    taskId: options.taskId,
    joinTasks: preset.needsTasksJoin,
    joinMutants: preset.joinMutantsForConfound,
  });
}
