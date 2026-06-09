import type {
  CodeVariant,
  DatasetSchema,
  LoadedDataset,
  ReviewItem,
} from "./types";
import { parseJsonl } from "./jsonl";

type TaskRow = {
  task_id: string;
  complete_prompt?: string;
  instruct_prompt?: string;
  canonical_solution?: string;
  test?: string;
  entry_point?: string;
  canonical_status?: string;
};

type MutantRow = {
  task_id: string;
  mutant_id: string;
  operator?: string;
  operator_code?: string;
  code: string;
  site?: Record<string, unknown>;
  status?: string;
  failure_fingerprint?: string;
  details?: unknown;
};

type ConfoundRow = {
  task_id: string;
  source_mutant_id: string;
  variant_id: string;
  code: string;
  status?: string;
  backend?: string;
  failure_fingerprint?: string;
  details?: unknown;
};

export interface BuildContext {
  tasksMap?: Map<string, TaskRow | Record<string, unknown>>;
  mutantIndex?: Map<string, string>;
}

export function inferSchemaFromFilename(
  filename: string,
  sample: Record<string, unknown>,
): DatasetSchema {
  const base = filename.toLowerCase();
  if (base.includes("dropped_tasks")) return "dropped_tasks";
  if (base.includes("validated_confound")) return "validated_confounded";
  if (base.includes("confound_candidate")) return "confound_candidates";
  if (base.includes("validated_mutant")) return "validated_mutants";
  if (base.includes("validated_tasks")) return "validated_tasks";
  if (base.includes("frozen")) return "frozen";
  if (base.includes("/phase2/") || base.includes("phase2/")) return "phase2";
  if (base.includes("mutants")) return "mutants";
  if (base.includes("tasks.jsonl")) return "tasks";

  if ("mutants" in sample && Array.isArray(sample.mutants)) return "frozen";
  if ("variant_id" in sample && "source_mutant_id" in sample) {
    return base.includes("candidate") ? "confound_candidates" : "validated_confounded";
  }
  if ("mutant_id" in sample && "code" in sample) {
    return "status" in sample ? "validated_mutants" : "mutants";
  }
  if ("implementations" in sample) {
    return "prompt_variants" in sample ? "phase2" : "icse26";
  }
  if ("canonical_solution" in sample) {
    return "canonical_status" in sample ? "validated_tasks" : "tasks";
  }
  if ("reason" in sample) return "dropped_tasks";
  return "unknown";
}

function taskFromMap(
  tasks: Map<string, TaskRow | Record<string, unknown>> | undefined,
  taskId: string,
): TaskRow | undefined {
  const row = tasks?.get(taskId);
  if (!row) return undefined;
  return row as TaskRow;
}

function parseOperatorCodeFromMutantId(
  mutantId: string,
): { code: string; name: string } | null {
  const parts = mutantId.split("__");
  if (parts.length < 3) return null;
  const code = parts[parts.length - 2];
  if (!code || code === "confound") return null;
  return { code, name: code };
}

function siteHint(site: Record<string, unknown> | undefined): string | undefined {
  if (
    site &&
    typeof site.original === "string" &&
    typeof site.replacement === "string"
  ) {
    return `${site.original} → ${site.replacement}`;
  }
  return undefined;
}

const EMPTY_CODE = {
  baselineLabel: "",
  baselineCode: "",
  candidateLabel: "",
  candidateCode: "",
};

function makeBundle(params: {
  id: string;
  taskId: string;
  title: string;
  subtitle?: string;
  prompt?: string;
  test?: string;
  variants: CodeVariant[];
  metadata?: Record<string, unknown>;
}): ReviewItem {
  return {
    ...EMPTY_CODE,
    id: params.id,
    taskId: params.taskId,
    kind: "task",
    title: params.title,
    subtitle: params.subtitle,
    prompt: params.prompt,
    test: params.test,
    variants: params.variants,
    metadata: params.metadata ?? {},
  };
}

function buildFromTasks(records: Record<string, unknown>[]): ReviewItem[] {
  return records
    .filter((row) => row.task_id && row.canonical_solution)
    .map((row) => {
      const task = row as TaskRow;
      const prompt = task.instruct_prompt ?? task.complete_prompt ?? "";
      return makeBundle({
        id: String(task.task_id),
        taskId: String(task.task_id),
        title: String(task.task_id),
        subtitle: task.entry_point ? `entry: ${task.entry_point}` : undefined,
        prompt,
        test: task.test,
        variants: [
          {
            role: "canonical",
            id: `${task.task_id}__canonical`,
            code: String(task.canonical_solution),
            label: 1,
          },
        ],
        metadata: { entry_point: task.entry_point },
      });
    });
}

function buildFromValidatedTasks(records: Record<string, unknown>[]): ReviewItem[] {
  return records
    .filter((row) => row.task_id && row.canonical_solution)
    .map((row) => {
      const task = row as TaskRow;
      const prompt = task.instruct_prompt ?? task.complete_prompt ?? "";
      return makeBundle({
        id: String(task.task_id),
        taskId: String(task.task_id),
        title: String(task.task_id),
        subtitle: `GT validation: ${task.canonical_status ?? "unknown"}`,
        prompt,
        test: task.test,
        variants: [
          {
            role: "canonical",
            id: `${task.task_id}__canonical`,
            code: String(task.canonical_solution),
            label: 1,
            status: task.canonical_status,
          },
        ],
        metadata: {
          entry_point: task.entry_point,
          canonical_details: (row as { canonical_details?: unknown }).canonical_details,
        },
      });
    });
}

function buildFromDroppedTasks(records: Record<string, unknown>[]): ReviewItem[] {
  return records
    .filter((row) => row.task_id)
    .map((row) => {
      const taskId = String(row.task_id);
      const details = {
        reason: row.reason,
        status: row.status,
        details: row.details,
      };
      return {
        id: taskId,
        taskId,
        kind: "dropped" as const,
        title: taskId,
        subtitle: String(row.reason ?? "dropped"),
        baselineLabel: "Validation outcome",
        baselineCode: JSON.stringify(details, null, 2),
        candidateLabel: "Note",
        candidateCode:
          "This task was dropped before mutation — canonical solution did not pass Docker validation in Phase 1.",
        status: typeof row.status === "string" ? row.status : undefined,
        metadata: details as Record<string, unknown>,
      };
    });
}

function buildFromMutants(
  records: Record<string, unknown>[],
  tasks: Map<string, TaskRow | Record<string, unknown>> | undefined,
  schema: "mutants" | "validated_mutants",
): ReviewItem[] {
  const items: ReviewItem[] = [];
  for (const row of records) {
    if (!row.mutant_id || !row.code) continue;
    const mutant = row as MutantRow;
    const task = taskFromMap(tasks, mutant.task_id);
    const hint = siteHint(mutant.site);
    const variants: CodeVariant[] = [];

    if (task?.canonical_solution) {
      variants.push({
        role: "canonical",
        id: `${mutant.task_id}__canonical`,
        code: task.canonical_solution,
        label: 1,
      });
    }

    variants.push({
      role: "mutant",
      id: mutant.mutant_id,
      code: mutant.code,
      label: 0,
      status: mutant.status,
      operator: mutant.operator,
      operator_code: mutant.operator_code,
      site: mutant.site,
      failure_fingerprint: mutant.failure_fingerprint,
      subtitle:
        [mutant.operator_code, mutant.operator, hint].filter(Boolean).join(" · ") ||
        undefined,
    });

    items.push(
      makeBundle({
        id: mutant.mutant_id,
        taskId: mutant.task_id,
        title: mutant.mutant_id,
        subtitle: [mutant.operator, hint].filter(Boolean).join(" · ") || undefined,
        prompt: task?.instruct_prompt ?? task?.complete_prompt,
        test: task?.test,
        variants,
        metadata: {
          operator_code: mutant.operator_code,
          site: mutant.site,
          failure_fingerprint: mutant.failure_fingerprint,
          details: mutant.details,
          schema,
        },
      }),
    );
  }
  return items;
}

function buildFromConfounds(
  records: Record<string, unknown>[],
  tasks: Map<string, TaskRow | Record<string, unknown>> | undefined,
  mutantIndex: Map<string, string> | undefined,
  schema: "confound_candidates" | "validated_confounded",
): ReviewItem[] {
  const items: ReviewItem[] = [];
  for (const row of records) {
    if (!row.variant_id || !row.code) continue;
    const confound = row as ConfoundRow;
    const task = taskFromMap(tasks, confound.task_id);
    const mutantCode = mutantIndex?.get(confound.source_mutant_id);
    const variants: CodeVariant[] = [];

    if (task?.canonical_solution) {
      variants.push({
        role: "canonical",
        id: `${confound.task_id}__canonical`,
        code: task.canonical_solution,
        label: 1,
      });
    }

    if (mutantCode) {
      variants.push({
        role: "mutant",
        id: confound.source_mutant_id,
        code: mutantCode,
        label: 0,
        subtitle: confound.source_mutant_id,
      });
    }

    variants.push({
      role: "confound",
      id: confound.variant_id,
      code: confound.code,
      label: 0,
      status: confound.status,
      subtitle: confound.backend,
    });

    const opFromId = parseOperatorCodeFromMutantId(confound.source_mutant_id);
    items.push(
      makeBundle({
        id: confound.variant_id,
        taskId: confound.task_id,
        title: confound.variant_id,
        subtitle: `source mutant: ${confound.source_mutant_id}`,
        prompt: task?.instruct_prompt ?? task?.complete_prompt,
        test: task?.test,
        variants,
        metadata: {
          source_mutant_id: confound.source_mutant_id,
          operator_code: opFromId?.code,
          operator: opFromId?.name,
          backend: confound.backend,
          failure_fingerprint: confound.failure_fingerprint,
          details: confound.details,
          schema,
        },
      }),
    );
  }
  return items;
}

function buildFromFrozen(records: Record<string, unknown>[]): ReviewItem[] {
  const items: ReviewItem[] = [];
  for (const row of records) {
    const frozen = row as {
      task_id: string;
      task?: string;
      canonical?: string;
      prompt_variants?: { instruct?: string };
      mutants?: Array<{
        mutant_id: string;
        code: string;
        operator?: string;
        operator_code?: string;
        status?: string;
        site?: Record<string, unknown>;
        confounded_variants?: Array<{
          variant_id: string;
          code: string;
          status?: string;
          backend?: string;
        }>;
      }>;
    };
    if (!frozen.task_id || !Array.isArray(frozen.mutants)) continue;
    const canonical = frozen.canonical ?? "";
    const prompt = frozen.task ?? frozen.prompt_variants?.instruct ?? "";

    for (const mutant of frozen.mutants) {
      const variants: CodeVariant[] = [];
      if (canonical) {
        variants.push({
          role: "canonical",
          id: `${frozen.task_id}__canonical`,
          code: canonical,
          label: 1,
        });
      }
      variants.push({
        role: "mutant",
        id: mutant.mutant_id,
        code: mutant.code,
        label: 0,
        status: mutant.status,
        operator: mutant.operator,
        operator_code: mutant.operator_code,
        site: mutant.site,
        subtitle: [mutant.operator_code, mutant.operator, siteHint(mutant.site)]
          .filter(Boolean)
          .join(" · "),
      });
      const confound = mutant.confounded_variants?.[0];
      if (confound) {
        variants.push({
          role: "confound",
          id: confound.variant_id,
          code: confound.code,
          label: 0,
          status: confound.status,
          subtitle: confound.backend,
        });
      }

      items.push(
        makeBundle({
          id: mutant.mutant_id,
          taskId: frozen.task_id,
          title: mutant.mutant_id,
          subtitle: mutant.operator,
          prompt,
          variants,
          metadata: {
            site: mutant.site,
            extra_confounds: (mutant.confounded_variants ?? []).slice(1).map((v) => v.variant_id),
          },
        }),
      );
    }
  }
  return items;
}

type SplitImpl = {
  implementation_id: string;
  code: string;
  label: number;
  kind?: string;
  operator?: string;
  operator_code?: string;
  failure_fingerprint?: string;
  site?: Record<string, unknown>;
};

function implToVariant(impl: SplitImpl, role: CodeVariant["role"]): CodeVariant {
  const opLabel =
    impl.operator_code && impl.operator
      ? `${impl.operator_code} · ${impl.operator}`
      : impl.operator_code ?? impl.operator;

  return {
    role,
    id: impl.implementation_id,
    code: impl.code,
    label: impl.label,
    operator: impl.operator,
    operator_code: impl.operator_code,
    site: impl.site,
    failure_fingerprint: impl.failure_fingerprint,
    subtitle: role === "mutant" && opLabel ? opLabel : undefined,
  };
}

function buildFromSplitRows(
  records: Record<string, unknown>[],
  schema: "phase2" | "icse26",
): ReviewItem[] {
  const items: ReviewItem[] = [];
  for (const row of records) {
    const taskId = String(row.task_id ?? "");
    const impls = row.implementations;
    if (!taskId || !Array.isArray(impls)) continue;

    const prompt =
      (row.task as string | undefined) ??
      (row.prompt_variants as { instruct?: string } | undefined)?.instruct ??
      "";

    const objectImpls = impls.filter(
      (impl): impl is SplitImpl => typeof impl === "object" && impl !== null,
    );

    const canonicalImpl = objectImpls.find((impl) => impl.kind === "canonical");
    const labelCanonical =
      canonicalImpl ??
      objectImpls.find((impl) => impl.label === 1 && impl.kind !== "mutant" && impl.kind !== "confound");

    const mutantImpl = objectImpls.find((impl) => impl.kind === "mutant");
    const confoundImpls = objectImpls.filter((impl) => impl.kind === "confound");
    const confoundImpl = confoundImpls[0];

    const variants: CodeVariant[] = [];
    if (labelCanonical) {
      variants.push(implToVariant(labelCanonical, "canonical"));
    }
    if (mutantImpl) {
      variants.push(implToVariant(mutantImpl, "mutant"));
    }
    if (confoundImpl) {
      const sourceMutantId = confoundImpl.implementation_id.replace(/__confound__\d+$/, "");
      const inlineMutant = objectImpls.find((m) => m.implementation_id === sourceMutantId);
      if (inlineMutant && !mutantImpl) {
        variants.push(implToVariant(inlineMutant, "mutant"));
      }
      variants.push(implToVariant(confoundImpl, "confound"));
    }

    if (variants.length === 0) continue;

    const roles = variants.map((v) => v.role).join(" · ");
    items.push(
      makeBundle({
        id: taskId,
        taskId,
        title: taskId,
        subtitle: roles,
        prompt,
        variants,
        metadata: {
          split_schema: schema,
          entry_point: row.entry_point,
          operator: mutantImpl?.operator,
          operator_code: mutantImpl?.operator_code,
          site: mutantImpl?.site,
          failure_fingerprint: mutantImpl?.failure_fingerprint,
          mutant_id: mutantImpl?.implementation_id,
          implementation_ids: objectImpls.map((i) => i.implementation_id),
          extra_confounds: confoundImpls.slice(1).map((c) => c.implementation_id),
        },
      }),
    );
  }
  return items;
}

export function recordsToReviewItems(
  records: Record<string, unknown>[],
  schema: DatasetSchema,
  context: BuildContext = {},
): ReviewItem[] {
  const tasks = context.tasksMap as Map<string, TaskRow | Record<string, unknown>> | undefined;

  let items: ReviewItem[] = [];
  switch (schema) {
    case "tasks":
      items = buildFromTasks(records);
      break;
    case "validated_tasks":
      items = buildFromValidatedTasks(records);
      break;
    case "dropped_tasks":
      items = buildFromDroppedTasks(records);
      break;
    case "mutants":
    case "validated_mutants":
      items = buildFromMutants(records, tasks, schema);
      break;
    case "confound_candidates":
    case "validated_confounded":
      items = buildFromConfounds(
        records,
        tasks,
        context.mutantIndex,
        schema,
      );
      break;
    case "frozen":
      items = buildFromFrozen(records);
      break;
    case "phase2":
    case "icse26":
      items = buildFromSplitRows(records, schema);
      break;
    default:
      throw new Error(`Unsupported schema: ${schema}`);
  }

  items.sort((a, b) => a.id.localeCompare(b.id));
  return items;
}

function buildMutantIndexFromText(text: string): Map<string, string> {
  const index = new Map<string, string>();
  for (const row of parseJsonl(text)) {
    if (row.mutant_id && row.code) {
      index.set(String(row.mutant_id), String(row.code));
    }
  }
  return index;
}

export function buildDataset(
  name: string,
  primaryText: string,
  options: {
    schema?: DatasetSchema;
    tasksText?: string;
    mutantsText?: string;
    mutantIndex?: Map<string, string>;
    truncated?: boolean;
    sourcePath?: string;
  } = {},
): LoadedDataset {
  const primaryRecords = parseJsonl(primaryText);
  const schema =
    options.schema ??
    inferSchemaFromFilename(name, primaryRecords[0] ?? {});

  const taskRecords = options.tasksText ? parseJsonl(options.tasksText) : [];
  const tasksMap = new Map<string, TaskRow | Record<string, unknown>>();
  for (const row of taskRecords) {
    if (row.task_id) tasksMap.set(String(row.task_id), row);
  }

  const mutantIndex =
    options.mutantIndex ??
    (options.mutantsText ? buildMutantIndexFromText(options.mutantsText) : undefined);

  const items = recordsToReviewItems(primaryRecords, schema, {
    tasksMap: tasksMap.size > 0 ? tasksMap : undefined,
    mutantIndex: mutantIndex && mutantIndex.size > 0 ? mutantIndex : undefined,
  });

  return {
    name,
    schema,
    items,
    truncated: options.truncated,
    sourcePath: options.sourcePath,
  };
}
