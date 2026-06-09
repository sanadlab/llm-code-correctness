import type { CodeVariant, CodeVariantRole } from "./types";

export type DiffPairId = "canonical-mutant" | "mutant-confound" | "canonical-confound";

export const DIFF_PAIR_LABELS: Record<DiffPairId, string> = {
  "canonical-mutant": "Canonical ↔ Mutant",
  "mutant-confound": "Mutant ↔ Confound",
  "canonical-confound": "Canonical ↔ Confound",
};

export function availableDiffPairs(
  variants: CodeVariant[],
): DiffPairId[] {
  const roles = new Set(variants.map((v) => v.role));
  const pairs: DiffPairId[] = [];
  if (roles.has("canonical") && roles.has("mutant")) {
    pairs.push("canonical-mutant");
  }
  if (roles.has("mutant") && roles.has("confound")) {
    pairs.push("mutant-confound");
  }
  if (roles.has("canonical") && roles.has("confound")) {
    pairs.push("canonical-confound");
  }
  return pairs;
}

export function resolveDiffPair(
  variants: CodeVariant[],
  pairId: DiffPairId,
): { left: CodeVariant; right: CodeVariant } | null {
  const byRole = new Map<CodeVariantRole, CodeVariant>();
  for (const v of variants) {
    if (!byRole.has(v.role)) byRole.set(v.role, v);
  }

  const [leftRole, rightRole] = pairId.split("-") as [
    CodeVariantRole,
    CodeVariantRole,
  ];
  const left = byRole.get(leftRole);
  const right = byRole.get(rightRole);
  if (!left || !right) return null;
  return { left, right };
}

export function defaultDiffPair(variants: CodeVariant[]): DiffPairId | null {
  const pairs = availableDiffPairs(variants);
  if (pairs.length === 0) return null;
  if (pairs.includes("canonical-mutant")) return "canonical-mutant";
  if (pairs.includes("mutant-confound")) return "mutant-confound";
  return pairs[0];
}
