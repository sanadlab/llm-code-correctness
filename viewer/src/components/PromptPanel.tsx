"use client";

type PromptPanelProps = {
  prompt: string;
  entryPoint?: string;
  open: boolean;
  onOpenChange?: (open: boolean) => void;
};

/** Collapsible task prompt — kept out of code diffs. */
export function PromptPanel({
  prompt,
  entryPoint,
  open,
  onOpenChange,
}: PromptPanelProps) {
  const preview = prompt.replace(/\s+/g, " ").trim().slice(0, 140);
  const paragraphs = prompt
    .split(/\n\s*\n/)
    .map((p) => p.trim())
    .filter(Boolean);

  return (
    <details
      className="group border-b border-zinc-200 bg-zinc-50/80 dark:border-zinc-800 dark:bg-zinc-900/40"
      open={open}
      onToggle={(e) => onOpenChange?.(e.currentTarget.open)}
    >
      <summary className="cursor-pointer list-none px-5 py-3 [&::-webkit-details-marker]:hidden">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex min-w-0 items-center gap-2">
            <span className="text-zinc-400 transition group-open:rotate-90">▸</span>
            <h3 className="text-xs font-semibold uppercase tracking-wide text-zinc-500">
              Task prompt
            </h3>
            {entryPoint && (
              <span className="rounded-full bg-zinc-200 px-2 py-0.5 font-mono text-[10px] text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300">
                {entryPoint}
              </span>
            )}
          </div>
          <p className="max-w-xl truncate text-xs text-zinc-500 group-open:hidden">
            {preview}
            {prompt.length > 140 ? "…" : ""}
          </p>
        </div>
      </summary>
      <div className="max-h-52 space-y-3 overflow-y-auto border-t border-zinc-200/80 px-5 pb-4 pt-2 text-sm leading-relaxed text-zinc-800 dark:border-zinc-800 dark:text-zinc-200">
        {paragraphs.length > 0 ? (
          paragraphs.map((para, i) => (
            <p key={i} className="whitespace-pre-wrap">
              {para}
            </p>
          ))
        ) : (
          <p className="whitespace-pre-wrap">{prompt}</p>
        )}
      </div>
    </details>
  );
}
