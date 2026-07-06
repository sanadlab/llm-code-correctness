import type { ReviewItem } from "./types";

export interface OperatorInfo {
  name: string;
  code?: string;
  mutantId?: string;
  siteOriginal?: string;
  siteReplacement?: string;
  siteNodeType?: string;
  failureFingerprint?: string;
}

/** Operator targeted for the mutant on this card. */
export function getOperatorInfo(item: ReviewItem): OperatorInfo | null {
  const mutant = item.variants?.find((v) => v.role === "mutant");
  const meta = item.metadata;

  if (mutant?.operator || mutant?.operator_code) {
    return buildFromParts({
      name: mutant.operator,
      code: mutant.operator_code,
      mutantId: mutant.id,
      site: mutant.site,
      fingerprint: meta.failure_fingerprint as string | undefined,
    });
  }

  if (meta.operator || meta.operator_code) {
    return buildFromParts({
      name: meta.operator as string | undefined,
      code: meta.operator_code as string | undefined,
      mutantId: (meta.mutant_id as string) ?? mutant?.id,
      site: meta.site as Record<string, unknown> | undefined,
      fingerprint: meta.failure_fingerprint as string | undefined,
    });
  }

  const id = mutant?.id ?? (meta.source_mutant_id as string | undefined);
  if (id) {
    const parsed = parseOperatorFromMutantId(id);
    if (parsed) {
      return buildFromParts({
        name: parsed.name,
        code: parsed.code,
        mutantId: id,
        site: meta.site as Record<string, unknown> | undefined,
        fingerprint: meta.failure_fingerprint as string | undefined,
      });
    }
  }

  return null;
}

function buildFromParts(parts: {
  name?: string;
  code?: string;
  mutantId?: string;
  site?: Record<string, unknown>;
  fingerprint?: string;
}): OperatorInfo | null {
  const site = parts.site;
  const siteOriginal =
    site && typeof site.original === "string" ? site.original : undefined;
  const siteReplacement =
    site && typeof site.replacement === "string" ? site.replacement : undefined;
  const siteNodeType =
    site && typeof site.node_type === "string" ? site.node_type : undefined;

  const name =
    parts.name ??
    (parts.code ? operatorNameFromCode(parts.code) : undefined) ??
    parts.code;

  if (!name && !parts.code && !siteOriginal) return null;

  return {
    name: name ?? "Unknown operator",
    code: parts.code,
    mutantId: parts.mutantId,
    siteOriginal,
    siteReplacement,
    siteNodeType,
    failureFingerprint: parts.fingerprint,
  };
}

/** BigCodeBench/1012__AOI__4__1 → AOI */
export function parseOperatorFromMutantId(
  mutantId: string,
): { code: string; name: string } | null {
  const match = mutantId.match(/__([A-Z]{2,6})__/);
  const code = match?.[1];
  if (!code) return null;
  return { code, name: operatorNameFromCode(code) };
}

const OPERATOR_NAMES: Record<string, string> = {
  AOD: "ArithmeticOperatorDeletion",
  AOI: "ArithmeticOperatorInsertion",
  AOR: "ArithmeticOperatorReplacement",
  COD: "ConditionalOperatorDeletion",
  COI: "ConditionalOperatorInsertion",
  COR: "ConditionalOperatorReplacement",
  LOD: "LogicalOperatorDeletion",
  LOI: "LogicalOperatorInsertion",
  LOR: "LogicalOperatorReplacement",
  ROD: "RelationalOperatorDeletion",
  ROI: "RelationalOperatorInsertion",
  ROR: "RelationalOperatorReplacement",
  SOD: "StatementDeletion",
  SOI: "StatementInsertion",
  SOR: "StatementReplacement",
  UOD: "UnaryOperatorDeletion",
  UOI: "UnaryOperatorInsertion",
  UOR: "UnaryOperatorReplacement",
  VVD: "ValueReplaceDeletion",
  VVR: "ValueReplaceReplacement",
  VCR: "ValueReplaceChange",
  CVR: "ConstantValueReplacement",
  BOD: "BitwiseOperatorDeletion",
  BOI: "BitwiseOperatorInsertion",
  BOR: "BitwiseOperatorReplacement",
};

function operatorNameFromCode(code: string): string {
  return OPERATOR_NAMES[code] ?? code;
}

export function formatOperatorSummary(info: OperatorInfo): string {
  const parts = [info.code, info.name].filter(Boolean);
  return parts.join(" · ");
}
