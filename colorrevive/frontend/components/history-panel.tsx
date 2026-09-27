"use client";

import { Clock, Trash2 } from "lucide-react";

export interface HistoryEntry {
  id: string;
  name: string;
  timestamp: number;
  width: number;
  height: number;
  model: string;
  fallbackMode: boolean;
  processingTimeMs: number;
  dataUrl?: string;
  resultUrl?: string;
}

interface HistoryPanelProps {
  entries: HistoryEntry[];
  onSelect?: (entry: HistoryEntry) => void;
  onClear: () => void;
}

/**
 * Session-only history of colorized images, held in memory (and mirrored
 * to localStorage without image data). Nothing is uploaded or persisted
 * on the server.
 */
export function HistoryPanel({ entries, onSelect, onClear }: HistoryPanelProps) {
  if (entries.length === 0) return null;
  return (
    <section
      aria-labelledby="history-heading"
      className="mt-6 rounded-xl border border-line bg-surface p-5 shadow-card"
    >
      <div className="flex items-center justify-between">
        <h2 id="history-heading" className="flex items-center gap-2 text-lg font-semibold">
          <Clock aria-hidden className="h-5 w-5 text-accent" />
          This session
        </h2>
        <button
          type="button"
          onClick={onClear}
          className="inline-flex items-center gap-1.5 rounded-lg border border-line px-2.5 py-1.5 text-xs font-medium hover:bg-accent-soft"
        >
          <Trash2 aria-hidden className="h-3.5 w-3.5" /> Clear history
        </button>
      </div>
      <p className="mt-1 text-xs text-muted">
        History lives only in this browser tab and is cleared when you close it.
      </p>
      <ul className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
        {entries.map((entry) => (
          <li key={entry.id}>
            <button
              type="button"
              onClick={() => onSelect?.(entry)}
              className="group w-full rounded-lg border border-line bg-canvas p-2 text-left transition-colors hover:border-accent"
              aria-label={`Reopen result for ${entry.name}`}
            >
              {entry.resultUrl ? (
                <img
                  src={entry.resultUrl}
                  alt=""
                  className="h-20 w-full rounded object-cover"
                />
              ) : (
                <div className="flex h-20 w-full items-center justify-center rounded bg-line/40 text-xs text-muted">
                  no preview
                </div>
              )}
              <p className="mt-1.5 truncate text-xs font-medium">{entry.name}</p>
              <p className="text-[11px] text-muted">
                {entry.width}×{entry.height} ·{" "}
                {entry.fallbackMode ? "fallback" : entry.model}
              </p>
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
