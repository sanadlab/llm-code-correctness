"use client";

import { useEffect, useMemo, useState } from "react";

import {
  DIFF_PAIR_LABELS,
  availableDiffPairs,
  defaultDiffPair,
  resolveDiffPair,
  type DiffPairId,
} from "@/lib/diff-pairs";
import { parseOperatorFromMutantId } from "@/lib/operator-info";
import type { CodeVariant } from "@/lib/types";
import { CodeDiff } from "./CodeDiff";
import { TaskVariantGrid } from "./TaskVariantGrid";

type TaskCodeViewProps = {
  variants: CodeVariant[];
  viewMode: "diff" | "split";
};

export function TaskCodeView({ variants, viewMode }: TaskCodeViewProps) {
  const pairs = useMemo(() => availableDiffPairs(variants), [variants]);
  const [diffPair, setDiffPair] = useState<DiffPairId | null>(null);

  useEffect(() => {
    const next = defaultDiffPair(variants);
    setDiffPair((current) =>
      current && pairs.includes(current) ? current : next,
    );
  }, [variants, pairs]);

  if (viewMode === "split") {
    return <TaskVariantGrid variants={variants} />;
  }

  if (pairs.length === 0 || variants.length === 1) {
    return (
      <div className="space-y-3 p-5">
        <p className="rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-200">
          Diff highlights need at least two variants on this card. Load joins
          (`tasks.jsonl`, `validated_mutants.jsonl`) for confound files to compare canonical
          and mutant code.
        </p>
        <TaskVariantGrid variants={variants} height="420px" />
      </div>
    );
  }

  const resolved = diffPair ? resolveDiffPair(variants, diffPair) : null;

  return (
    <div className="space-y-3 p-5">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs font-medium text-zinc-500">Compare</span>
        {pairs.map((id) => (
          <button
            key={id}
            type="button"
            onClick={() => setDiffPair(id)}
            className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ${
              diffPair === id
                ? "bg-zinc-900 text-white dark:bg-zinc-100 dark:text-zinc-900"
                : "bg-zinc-100 text-zinc-700 hover:bg-zinc-200 dark:bg-zinc-800 dark:text-zinc-300 dark:hover:bg-zinc-700"
            }`}
          >
            {DIFF_PAIR_LABELS[id]}
          </button>
        ))}
      </div>

      {resolved ? (
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <span className="text-xs text-zinc-500">Mutation</span>
            <span className="rounded-md border border-amber-500/40 bg-amber-500/10 px-2 py-0.5 font-mono text-xs font-semibold text-amber-200">
              {mutationLabel(resolved.left, resolved.right)}
            </span>
          </div>
          <CodeDiff
            key={`${diffPair ?? "pair"}:${resolved.left.role}:${resolved.left.id}:${
              resolved.right.role
            }:${resolved.right.id}`}
            oldValue={resolved.left.code}
            newValue={resolved.right.code}
            oldTitle={roleLabel(resolved.left.role)}
            newTitle={roleLabel(resolved.right.role)}
          />
        </div>
      ) : (
        <p className="text-sm text-zinc-500">Select a comparison above.</p>
      )}
    </div>
  );
}

function roleLabel(role: string): string {
  return role.charAt(0).toUpperCase() + role.slice(1);
}

function mutationLabel(left: CodeVariant, right: CodeVariant): string {
  const named = [left, right]
    .map((v) => ({ name: v.operator, code: v.operator_code }))
    .find(
      (op) =>
        (typeof op.name === "string" && op.name.length > 0) ||
        (typeof op.code === "string" && op.code.length > 0),
    );

  if (named?.name && named?.code) return `${named.name} (${named.code})`;
  if (named?.name) return named.name;
  if (named?.code) {
    const parsed = parseOperatorFromMutantId(`task__${named.code}__0__0`);
    return parsed ? `${parsed.name} (${named.code})` : named.code;
  }

  const parsedFromId = [left.id, right.id]
    .map((id) => parseOperatorFromMutantId(id))
    .find((parsed): parsed is { code: string; name: string } => !!parsed);
  if (parsedFromId) return `${parsedFromId.name} (${parsedFromId.code})`;

  return "Unknown operator";
}
