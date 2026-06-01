"use client";

import { formatOperatorSummary, getOperatorInfo } from "@/lib/operator-info";
import type { ReviewItem } from "@/lib/types";

export function OperatorBanner({ item }: { item: ReviewItem }) {
  const info = getOperatorInfo(item);
  if (!info) return null;

  return (
    <div className="border-b border-amber-500/20 bg-amber-500/10 px-5 py-4 dark:bg-amber-950/30">
      <p className="text-[10px] font-semibold uppercase tracking-widest text-amber-800 dark:text-amber-300/90">
        Intended mutation operator
      </p>
      <p className="mt-1 text-base font-semibold text-amber-950 dark:text-amber-100">
        {formatOperatorSummary(info)}
      </p>
      {info.siteOriginal && info.siteReplacement && (
        <div className="mt-3 flex flex-wrap items-center gap-2 font-mono text-sm">
          <code className="rounded-md bg-red-500/15 px-2.5 py-1 text-red-900 line-through dark:bg-red-950/50 dark:text-red-300">
            {info.siteOriginal}
          </code>
          <span className="text-amber-700/80 dark:text-amber-400/80">→</span>
          <code className="rounded-md bg-emerald-500/15 px-2.5 py-1 text-emerald-900 dark:bg-emerald-950/50 dark:text-emerald-300">
            {info.siteReplacement}
          </code>
          {info.siteNodeType && (
            <span className="rounded-full bg-amber-500/20 px-2 py-0.5 text-[10px] font-medium uppercase text-amber-900 dark:text-amber-200">
              {info.siteNodeType}
            </span>
          )}
        </div>
      )}
      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-amber-900/70 dark:text-amber-200/60">
        {info.mutantId && (
          <span className="font-mono truncate">mutant: {info.mutantId}</span>
        )}
        {info.failureFingerprint && (
          <span className="font-mono">fingerprint: {info.failureFingerprint}</span>
        )}
      </div>
    </div>
  );
}
