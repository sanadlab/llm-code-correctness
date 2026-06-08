import fs from "fs";
import path from "path";
import { NextResponse } from "next/server";

import { PHASE2_SPLITS } from "@/lib/catalog";
import type { DatasetBrowseResponse } from "@/lib/types";
import { datasetRootExists, getBigCodeBenchRoot } from "@/server/dataset-path";

export const dynamic = "force-dynamic";

export async function GET() {
  if (!datasetRootExists()) {
    return NextResponse.json(
      {
        error:
          "dataset/bigcodebench not found. Run from the repo with `make decompress`, or upload JSONL manually.",
      },
      { status: 404 },
    );
  }

  const root = getBigCodeBenchRoot();
  const intermediateDir = path.join(root, "intermediate");
  const intermediate: DatasetBrowseResponse["intermediate"] = [];

  const presetFiles = [
    { id: "tasks", label: "tasks.jsonl", base: "tasks.jsonl" },
    { id: "validated_tasks", label: "validated_tasks.jsonl", base: "validated_tasks.jsonl" },
    { id: "dropped_tasks", label: "dropped_tasks.jsonl", base: "dropped_tasks.jsonl" },
    { id: "mutants", label: "mutants.jsonl", base: "mutants.jsonl" },
    { id: "validated_mutants", label: "validated_mutants.jsonl", base: "validated_mutants.jsonl" },
    {
      id: "confound_candidates",
      label: "confound_candidates.jsonl",
      base: "confound_candidates.jsonl",
    },
    {
      id: "validated_confounded",
      label: "validated_confounded.jsonl",
      base: "validated_confounded.jsonl",
    },
  ] as const;

  for (const file of presetFiles) {
    const jsonl = path.join(intermediateDir, file.base);
    const zst = `${jsonl}.zst`;
    if (fs.existsSync(jsonl)) {
      intermediate.push({
        id: file.id,
        label: file.label,
        path: `intermediate/${file.base}`,
        compressed: false,
      });
    } else if (fs.existsSync(zst)) {
      intermediate.push({
        id: file.id,
        label: `${file.label}.zst`,
        path: `intermediate/${file.base}.zst`,
        compressed: true,
      });
    }
  }

  if (fs.existsSync(path.join(root, "frozen.jsonl"))) {
    intermediate.push({
      id: "frozen",
      label: "frozen.jsonl",
      path: "frozen.jsonl",
      compressed: false,
    });
  } else if (fs.existsSync(path.join(root, "frozen.jsonl.zst"))) {
    intermediate.push({
      id: "frozen",
      label: "frozen.jsonl.zst",
      path: "frozen.jsonl.zst",
      compressed: true,
    });
  }

  const phase2Dir = path.join(root, "phase2");
  const families = fs.existsSync(phase2Dir)
    ? fs
        .readdirSync(phase2Dir, { withFileTypes: true })
        .filter((d) => d.isDirectory() && d.name.startsWith("s"))
        .map((d) => d.name)
        .sort()
    : [];

  let folds = 10;
  if (families.length > 0) {
    const manifestPath = path.join(phase2Dir, families[0], "manifest.json");
    if (fs.existsSync(manifestPath)) {
      const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8")) as {
        num_folds?: number;
      };
      folds = manifest.num_folds ?? 10;
    }
  }

  const body: DatasetBrowseResponse = {
    datasetRoot: root,
    intermediate,
    phase2: {
      families,
      splits: [...PHASE2_SPLITS],
      folds: Array.from({ length: folds }, (_, i) => i),
    },
  };

  return NextResponse.json(body);
}
