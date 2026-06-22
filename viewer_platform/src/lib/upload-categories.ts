import type { DatasetSchema } from "./types";

export type UploadCategoryId =
  | "phase2"
  | "tasks"
  | "validated_tasks"
  | "dropped_tasks"
  | "mutants"
  | "validated_mutants"
  | "confound_candidates"
  | "validated_confounded"
  | "frozen"
  | "auto";

export interface UploadCategory {
  id: UploadCategoryId;
  label: string;
  description: string;
  schema: DatasetSchema | "auto";
  needsTasksJoin?: boolean;
  needsMutantsJoin?: boolean;
  defaultLimit: number;
}

export const UPLOAD_CATEGORIES: UploadCategory[] = [
  {
    id: "phase2",
    label: "Phase 2 split (fit / validate / test)",
    description: "One task per line with implementations[] (canonical, mutant, confound).",
    schema: "phase2",
    defaultLimit: 500,
  },
  {
    id: "tasks",
    label: "Phase 1 · tasks.jsonl",
    description: "Canonical tasks with prompts and solutions.",
    schema: "tasks",
    defaultLimit: 500,
  },
  {
    id: "validated_tasks",
    label: "Phase 1 · validated_tasks.jsonl",
    description: "Tasks that passed GT Docker validation.",
    schema: "validated_tasks",
    defaultLimit: 500,
  },
  {
    id: "dropped_tasks",
    label: "Phase 1 · dropped_tasks.jsonl",
    description: "Tasks removed when canonical failed validation.",
    schema: "dropped_tasks",
    defaultLimit: 500,
  },
  {
    id: "mutants",
    label: "Phase 1 · mutants.jsonl",
    description: "Generated mutants (pair with tasks.jsonl for canonical).",
    schema: "mutants",
    needsTasksJoin: true,
    defaultLimit: 300,
  },
  {
    id: "validated_mutants",
    label: "Phase 1 · validated_mutants.jsonl",
    description: "Mutants after Docker eval (pair with tasks.jsonl).",
    schema: "validated_mutants",
    needsTasksJoin: true,
    defaultLimit: 300,
  },
  {
    id: "confound_candidates",
    label: "Phase 1 · confound_candidates.jsonl",
    description:
      "Confound rows — add tasks.jsonl (canonical) and validated_mutants.jsonl (source mutant) for diffs.",
    schema: "confound_candidates",
    needsTasksJoin: true,
    needsMutantsJoin: true,
    defaultLimit: 200,
  },
  {
    id: "validated_confounded",
    label: "Phase 1 · validated_confounded.jsonl",
    description:
      "Validated confounds — add tasks.jsonl (canonical) and validated_mutants.jsonl (source mutant) for diffs.",
    schema: "validated_confounded",
    needsTasksJoin: true,
    needsMutantsJoin: true,
    defaultLimit: 200,
  },
  {
    id: "frozen",
    label: "frozen.jsonl",
    description: "Full Mutation_DS aggregate.",
    schema: "frozen",
    defaultLimit: 100,
  },
  {
    id: "auto",
    label: "Auto-detect from filename",
    description: "Infer schema from file name and first row.",
    schema: "auto",
    defaultLimit: 500,
  },
];

export function uploadCategoryById(
  id: UploadCategoryId,
): UploadCategory | undefined {
  return UPLOAD_CATEGORIES.find((c) => c.id === id);
}
