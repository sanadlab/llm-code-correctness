import { NextRequest, NextResponse } from "next/server";

import { datasetRootExists } from "@/server/dataset-path";
import { readAllJsonlRecords } from "@/server/read-jsonl";

export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  if (!datasetRootExists()) {
    return NextResponse.json({ error: "Dataset root not found" }, { status: 404 });
  }

  const path = request.nextUrl.searchParams.get("path");
  if (!path) {
    return NextResponse.json({ error: "Missing path query parameter" }, { status: 400 });
  }

  try {
    const records = await readAllJsonlRecords(path);
    return NextResponse.json({ path, records, count: records.length });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Failed to read dataset";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

