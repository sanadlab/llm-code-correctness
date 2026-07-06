export type ReviewDecision = "pending" | "approved" | "discarded";

export type ReviewItemKind =
  | "mutant"
  | "confound"
  | "implementation"
  | "task"
  | "dropped";

export type CodeVariantRole = "canonical" | "mutant" | "confound";

export interface CodeVariant {
  role: CodeVariantRole;
  id: string;
  code: string;
  label?: number;
  status?: string;
  operator?: string;
  operator_code?: string;
  site?: Record<string, unknown>;
  failure_fingerprint?: string;
  subtitle?: string;
}

/** Matches Phase 1 intermediate filenames and phase2 split JSONL. */
export type DatasetSchema =
  | "tasks"
  | "validated_tasks"
  | "dropped_tasks"
  | "mutants"
  | "validated_mutants"
  | "confound_candidates"
  | "validated_confounded"
  | "frozen"
  | "phase2"
  | "icse26"
  | "unknown";

export interface ReviewItem {
  id: string;
  taskId: string;
  kind: ReviewItemKind;
  title: string;
  subtitle?: string;
  /** Task-centric layout: prompt above, code variants in columns. */
  variants?: CodeVariant[];
  baselineLabel: string;
  baselineCode: string;
  candidateLabel: string;
  candidateCode: string;
  prompt?: string;
  test?: string;
  operator?: string;
  status?: string;
  metadata: Record<string, unknown>;
}

export interface LoadedDataset {
  name: string;
  schema: DatasetSchema;
  items: ReviewItem[];
  /** True when results were capped (large file / API limit). */
  truncated?: boolean;
  sourcePath?: string;
}

export interface ReviewRecord {
  decision: ReviewDecision;
  updatedAt: string;
  annotator?: string;
  annotation?: string;
}

export interface ReviewExport {
  datasetName: string;
  exportedAt: string;
  schema: DatasetSchema;
  decisions: Record<string, ReviewRecord>;
  summary: {
    total: number;
    approved: number;
    discarded: number;
    pending: number;
  };
}

export interface SampleQuery {
  path: string;
  offset?: number;
  limit?: number;
  status?: string;
  kind?: string;
  taskId?: string;
  joinMutants?: boolean;
}

export interface DatasetBrowseResponse {
  datasetRoot: string;
  intermediate: Array<{ id: string; label: string; path: string; compressed: boolean }>;
  phase2: {
    families: string[];
    splits: string[];
    folds: number[];
  };
}
