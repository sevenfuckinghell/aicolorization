"use client";

import { AlertTriangle, CheckCircle2, Loader2 } from "lucide-react";
import type { ProcessingPhase } from "@/lib/types";
import { cn } from "@/lib/utils";

const PHASES: Array<{ key: ProcessingPhase; label: string }> = [
  { key: "uploading", label: "Uploading image" },
  { key: "preparing", label: "Analyzing image and preparing inputs" },
  { key: "running-model", label: "Running colorization model (DDColor AI)" },
  { key: "postprocessing", label: "Reconstructing high-resolution image" },
  { key: "preparing-download", label: "Finalizing output" },
];

interface ProcessingStateProps {
  phase: ProcessingPhase;
  uploadProgress: number | null;
  errorMessage: string | null;
  onDismissError: () => void;
}

/**
 * ARIA live region that walks through the five processing stages and
 * surfaces user-safe error alerts (never stack traces).
 */
export function ProcessingState({
  phase,
  uploadProgress,
  errorMessage,
  onDismissError,
}: ProcessingStateProps) {
  if (phase === "idle" && !errorMessage) return null;
  const activeIndex = PHASES.findIndex((p) => p.key === phase);

  return (
    <div className="mt-4 space-y-3" aria-live="polite" role="status">
      {phase !== "idle" && (
        <div className="rounded-xl border border-line bg-surface p-4 shadow-card">
          <ul className="space-y-2">
            {PHASES.map((p, i) => {
              const done = i < activeIndex;
              const active = i === activeIndex;
              return (
                <li
                  key={p.key}
                  className={cn(
                    "flex items-center gap-2 text-sm",
                    done && "text-muted",
                    active && "font-semibold text-ink",
                    !done && !active && "text-muted/60",
                  )}
                >
                  {done ? (
                    <CheckCircle2 aria-hidden className="h-4 w-4 text-green-700 dark:text-green-400" />
                  ) : active ? (
                    <Loader2 aria-hidden className="h-4 w-4 animate-spin text-accent" />
                  ) : (
                    <span
                      aria-hidden
                      className="inline-block h-4 w-4 rounded-full border border-line"
                    />
                  )}
                  <span>
                    {p.label}
                    {active && p.key === "uploading" && uploadProgress !== null
                      ? ` (${uploadProgress}%)`
                      : ""}
                  </span>
                  {done && <span className="sr-only">(done)</span>}
                </li>
              );
            })}
          </ul>
        </div>
      )}

      {errorMessage && (
        <div
          role="alert"
          className="flex items-start gap-3 rounded-xl border border-danger/40 bg-danger/5 p-4"
        >
          <AlertTriangle
            aria-hidden
            className="mt-0.5 h-5 w-5 shrink-0 text-danger"
          />
          <div className="flex-1">
            <p className="text-sm font-semibold text-danger">
              Something went wrong
            </p>
            <p className="mt-1 text-sm">{errorMessage}</p>
          </div>
          <button
            type="button"
            onClick={onDismissError}
            className="rounded-md border border-line bg-surface px-2 py-1 text-xs font-medium hover:bg-accent-soft"
          >
            Dismiss
          </button>
        </div>
      )}
    </div>
  );
}
