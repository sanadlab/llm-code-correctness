import type { CodeVariantRole, ReviewItem } from "./types";

export function itemHasVariantRole(
  item: ReviewItem,
  role: CodeVariantRole,
): boolean {
  return item.variants?.some((v) => v.role === role) ?? false;
}

export function filterItemVariants(
  item: ReviewItem,
  kindFilter: CodeVariantRole | "all",
): ReviewItem {
  if (!item.variants || kindFilter === "all") return item;
  return {
    ...item,
    variants: item.variants.filter((v) => v.role === kindFilter),
  };
}

export function countVariantsByRole(
  items: ReviewItem[],
): Partial<Record<CodeVariantRole, number>> {
  const counts: Partial<Record<CodeVariantRole, number>> = {};
  for (const item of items) {
    if (!item.variants) continue;
    for (const v of item.variants) {
      counts[v.role] = (counts[v.role] ?? 0) + 1;
    }
  }
  return counts;
}
