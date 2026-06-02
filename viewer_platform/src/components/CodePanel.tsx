"use client";

import CodeMirror from "@uiw/react-codemirror";
import { python } from "@codemirror/lang-python";

type CodePanelProps = {
  code: string;
  label: string;
  hideHeader?: boolean;
  minHeight?: string;
};

export function CodePanel({
  code,
  label,
  hideHeader = false,
  minHeight = "320px",
}: CodePanelProps) {
  return (
    <div
      className={
        hideHeader
          ? "overflow-hidden"
          : "overflow-hidden rounded-lg border border-zinc-200 dark:border-zinc-800"
      }
    >
      {!hideHeader && label && (
        <div className="border-b border-zinc-200 bg-zinc-50 px-3 py-2 text-xs font-medium text-zinc-600 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-400">
          {label}
        </div>
      )}
      <CodeMirror
        value={code}
        height={minHeight}
        extensions={[python()]}
        editable={false}
        basicSetup={{
          lineNumbers: true,
          foldGutter: true,
          highlightActiveLine: false,
        }}
      />
    </div>
  );
}
