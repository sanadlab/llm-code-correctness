"use client";

import { useEffect, useMemo, useState } from "react";

import {
  DIFF_PAIR_LABELS,
  availableDiffPairs,
  defaultDiffPair,
  resolveDiffPair,
  type DiffPairId,
} from "@/lib/diff-pairs";
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
    return <TaskVariantGrid variants={variants} height="420px" />;
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
        <CodeDiff
          oldValue={resolved.left.code}
          newValue={resolved.right.code}
          oldTitle={roleLabel(resolved.left.role)}
          newTitle={roleLabel(resolved.right.role)}
        />
      ) : (
        <p className="text-sm text-zinc-500">Select a comparison above.</p>
      )}
    </div>
  );
}

function roleLabel(role: string): string {
  return role.charAt(0).toUpperCase() + role.slice(1);
}
