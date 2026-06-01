"use client";

import type { CodeVariant, CodeVariantRole } from "@/lib/types";
import { CodePanel } from "./CodePanel";

const ROLE_LABELS: Record<CodeVariantRole, string> = {
  canonical: "Canonical",
  mutant: "Mutant",
  confound: "Confound",
};

const ROLE_ACCENT: Record<CodeVariantRole, string> = {
  canonical: "border-l-[var(--canonical)]",
  mutant: "border-l-[var(--mutant)]",
  confound: "border-l-[var(--confound)]",
};

type TaskVariantGridProps = {
  variants: CodeVariant[];
  height?: string;
};

export function TaskVariantGrid({ variants, height = "380px" }: TaskVariantGridProps) {
  const ordered = orderVariants(variants);
  const count = ordered.length;

  if (count === 0) {
    return (
      <p className="p-6 text-center text-sm text-[var(--text-muted)]">
        No code variants in this view.
      </p>
    );
  }

  const gridClass =
    count === 1
      ? "grid grid-cols-1"
      : count === 2
        ? "grid grid-cols-1 gap-4 xl:grid-cols-2"
        : "grid grid-cols-1 gap-4 lg:grid-cols-2 xl:grid-cols-3";

  return (
    <div className={`${gridClass} gap-4 p-5`}>
      {ordered.map((variant) => (
        <div
          key={variant.id}
          className={`overflow-hidden rounded-lg border border-[var(--border)] border-l-4 bg-[var(--surface-elevated)] ${ROLE_ACCENT[variant.role]}`}
        >
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[var(--border)] px-3 py-2.5">
            <span className="text-xs font-bold uppercase tracking-wide text-[var(--text)]">
              {ROLE_LABELS[variant.role]}
            </span>
            <div className="flex flex-wrap gap-1.5">
              {variant.status && (
                <Badge>{variant.status}</Badge>
              )}
              {variant.label !== undefined && (
                <Badge>label {variant.label}</Badge>
              )}
            </div>
          </div>
          {variant.subtitle && variant.role === "mutant" && (
            <p className="border-b border-[var(--border)] px-3 py-1.5 font-mono text-[10px] text-[var(--mutant)]">
              {variant.subtitle}
            </p>
          )}
          <CodePanel code={variant.code} label="" minHeight={height} hideHeader />
        </div>
      ))}
    </div>
  );
}

function Badge({ children }: { children: React.ReactNode }) {
  return (
    <span className="rounded bg-[var(--bg)] px-1.5 py-0.5 text-[10px] font-medium uppercase text-[var(--text-muted)]">
      {children}
    </span>
  );
}

function orderVariants(variants: CodeVariant[]): CodeVariant[] {
  const order: CodeVariantRole[] = ["canonical", "mutant", "confound"];
  return [...variants].sort(
    (a, b) => order.indexOf(a.role) - order.indexOf(b.role),
  );
}
