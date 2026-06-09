import { NextRequest, NextResponse } from "next/server";

import { inferSchemaFromFilename, recordsToReviewItems } from "@/lib/parsers";
import type { DatasetSchema } from "@/lib/types";
import { datasetRootExists } from "@/server/dataset-path";
import {
  buildMutantCodeIndex,
  readJsonlRecordsWithLineOffset,
} from "@/server/read-jsonl";

export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  if (!datasetRootExists()) {
    return NextResponse.json({ error: "Dataset root not found" }, { status: 404 });
  }

  const params = request.nextUrl.searchParams;
  const relativePath = params.get("path");
  if (!relativePath) {
    return NextResponse.json({ error: "Missing path query parameter" }, { status: 400 });
  }

  const schemaParam = params.get("schema") as DatasetSchema | null;
  const offset = parseInt(params.get("offset") ?? "0", 10);
  const limit = parseInt(params.get("limit") ?? "200", 10);
  const status = params.get("status") ?? undefined;
  const taskId = params.get("taskId") ?? undefined;
  const joinTasks = params.get("joinTasks") === "1";
  const joinMutants = params.get("joinMutants") === "1";

  try {
    const schema =
      schemaParam ?? inferSchemaFromFilename(relativePath, {} as Record<string, unknown>);

    const records = await readJsonlRecordsWithLineOffset(relativePath, {
      offset: Number.isFinite(offset) ? offset : 0,
      limit: Number.isFinite(limit) ? limit : 200,
      status,
      taskId,
    });

    let tasksMap: Map<string, Record<string, unknown>> | undefined;
    if (joinTasks) {
      const taskRecords = await readJsonlRecordsWithLineOffset("intermediate/tasks.jsonl", {
        limit: 5000,
      });
      tasksMap = new Map(
        taskRecords
          .filter((t) => t.task_id)
          .map((t) => [String(t.task_id), t]),
      );
    }

    let mutantIndex: Map<string, string> | undefined;
    if (joinMutants && records.length > 0) {
      const ids = new Set<string>();
      for (const row of records) {
        if (typeof row.source_mutant_id === "string") {
          ids.add(row.source_mutant_id);
        }
      }
      mutantIndex = await buildMutantCodeIndex(ids);
    }

    const items = recordsToReviewItems(records, schema, {
      tasksMap,
      mutantIndex,
    });

    return NextResponse.json({
      name: relativePath,
      schema,
      path: relativePath,
      offset,
      limit,
      recordCount: records.length,
      itemCount: items.length,
      truncated: records.length >= limit,
      items,
      joinedTasks: joinTasks,
      joinedMutants: joinMutants && !!mutantIndex,
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Failed to read dataset";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
