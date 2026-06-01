import type { DatasetSchema } from "./types";

export type IntermediatePresetId =
  | "tasks"
  | "validated_tasks"
  | "dropped_tasks"
  | "mutants"
  | "validated_mutants"
  | "confound_candidates"
  | "validated_confounded"
  | "frozen";

export interface IntermediatePreset {
  id: IntermediatePresetId;
  label: string;
  description: string;
  /** Path relative to `dataset/bigcodebench/`. */
  path: string;
  schema: DatasetSchema;
  needsTasksJoin?: boolean;
  joinMutantsForConfound?: boolean;
  defaultLimit: number;
}

const INTERMEDIATE_DIR = "intermediate";

export const INTERMEDIATE_PRESETS: IntermediatePreset[] = [
  {
    id: "tasks",
    label: "tasks.jsonl",
    description: "Canonical BigCodeBench tasks (prompt + solution + tests).",
    path: `${INTERMEDIATE_DIR}/tasks.jsonl`,
    schema: "tasks",
    defaultLimit: 500,
  },
  {
    id: "validated_tasks",
    label: "validated_tasks.jsonl",
    description: "Tasks that passed canonical solution validation in Docker.",
    path: `${INTERMEDIATE_DIR}/validated_tasks.jsonl`,
    schema: "validated_tasks",
    defaultLimit: 500,
  },
  {
    id: "dropped_tasks",
    label: "dropped_tasks.jsonl",
    description: "Tasks removed because the canonical solution failed validation.",
    path: `${INTERMEDIATE_DIR}/dropped_tasks.jsonl`,
    schema: "dropped_tasks",
    defaultLimit: 500,
  },
  {
    id: "mutants",
    label: "mutants.jsonl",
    description: "All generated mutants (pre-Docker validation).",
    path: `${INTERMEDIATE_DIR}/mutants.jsonl`,
    schema: "mutants",
    needsTasksJoin: true,
    defaultLimit: 300,
  },
  {
    id: "validated_mutants",
    label: "validated_mutants.jsonl",
    description: "Mutants after Docker eval (pass / fail / timeout + fingerprints).",
    path: `${INTERMEDIATE_DIR}/validated_mutants.jsonl`,
    schema: "validated_mutants",
    needsTasksJoin: true,
    defaultLimit: 300,
  },
  {
    id: "confound_candidates",
    label: "confound_candidates.jsonl",
    description: "LLM-refactored confound candidates (pre-validation).",
    path: `${INTERMEDIATE_DIR}/confound_candidates.jsonl`,
    schema: "confound_candidates",
    joinMutantsForConfound: true,
    defaultLimit: 200,
  },
  {
    id: "validated_confounded",
    label: "validated_confounded.jsonl",
    description: "Confound variants after Docker validation.",
    path: `${INTERMEDIATE_DIR}/validated_confounded.jsonl`,
    schema: "validated_confounded",
    joinMutantsForConfound: true,
    defaultLimit: 200,
  },
  {
    id: "frozen",
    label: "frozen.jsonl",
    description: "Aggregated Mutation_DS (tasks + mutants + confounds).",
    path: "frozen.jsonl",
    schema: "frozen",
    defaultLimit: 100,
  },
];

export const PHASE2_SPLITS = ["fit", "validate", "test"] as const;
export type Phase2Split = (typeof PHASE2_SPLITS)[number];

export const DEFAULT_PHASE2_FAMILY = "s1_baseline";

export function phase2JsonlPath(
  family: string,
  fold: number,
  split: Phase2Split,
): string {
  return `phase2/${family}/fold_${fold}/${split}.jsonl`;
}

export function presetById(id: IntermediatePresetId): IntermediatePreset | undefined {
  return INTERMEDIATE_PRESETS.find((p) => p.id === id);
}
