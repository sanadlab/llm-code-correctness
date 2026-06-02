"use client";

import dynamic from "next/dynamic";
import { useEffect, useState } from "react";

const ReactDiffViewer = dynamic(() => import("react-diff-viewer-continued"), {
  ssr: false,
  loading: () => (
    <div className="rounded-lg border border-zinc-200 bg-zinc-50 p-8 text-center text-sm text-zinc-500 dark:border-zinc-800 dark:bg-zinc-900">
      Loading diff…
    </div>
  ),
});

type CodeDiffProps = {
  oldValue: string;
  newValue: string;
  oldTitle: string;
  newTitle: string;
};

export function CodeDiff({ oldValue, newValue, oldTitle, newTitle }: CodeDiffProps) {
  const [useDarkTheme, setUseDarkTheme] = useState(false);

  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    setUseDarkTheme(mq.matches);
    const handler = (e: MediaQueryListEvent) => setUseDarkTheme(e.matches);
    mq.addEventListener("change", handler);
    return () => mq.removeEventListener("change", handler);
  }, []);

  return (
    <div className="overflow-hidden rounded-lg border border-zinc-200 dark:border-zinc-700">
      <ReactDiffViewer
        oldValue={oldValue}
        newValue={newValue}
        splitView
        useDarkTheme={useDarkTheme}
        leftTitle={oldTitle}
        rightTitle={newTitle}
        styles={{
          variables: {
            light: {
              diffViewerBackground: "#fafafa",
              diffViewerColor: "#18181b",
              addedBackground: "#ecfdf5",
              addedColor: "#065f46",
              removedBackground: "#fef2f2",
              removedColor: "#991b1b",
              wordAddedBackground: "#bbf7d0",
              wordRemovedBackground: "#fecaca",
              gutterBackground: "#f4f4f5",
              gutterColor: "#71717a",
            },
            dark: {
              diffViewerBackground: "#18181b",
              diffViewerColor: "#e4e4e7",
              addedBackground: "#052e16",
              addedColor: "#bbf7d0",
              removedBackground: "#450a0a",
              removedColor: "#fecaca",
              wordAddedBackground: "#14532d",
              wordRemovedBackground: "#7f1d1d",
              gutterBackground: "#27272a",
              gutterColor: "#a1a1aa",
            },
          },
          contentText: {
            fontFamily: "var(--font-geist-mono), ui-monospace, monospace",
            fontSize: "12px",
            lineHeight: "1.5",
          },
        }}
      />
    </div>
  );
}
