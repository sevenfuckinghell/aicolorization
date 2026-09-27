"use client";

import { Sparkles } from "lucide-react";
import { useId } from "react";
import type { ColorizeSettings } from "@/lib/types";
import { cn } from "@/lib/utils";

interface SettingsPanelProps {
  settings: ColorizeSettings;
  onChange: (settings: ColorizeSettings) => void;
  onStart: () => void;
  processing: boolean;
  canSubmit: boolean;
}

/**
 * Colorization settings card. Every control is wired to the backend;
 * nothing here is decorative. The face-enhancement option is explicitly
 * labeled as unavailable because the current model does not implement it.
 */
export function SettingsPanel({
  settings,
  onChange,
  onStart,
  processing,
  canSubmit,
}: SettingsPanelProps) {
  const qualityId = useId();
  const contrastId = useId();
  const faceId = useId();
  const denoiseId = useId();
  const pngId = useId();
  const jpegId = useId();

  const set = <K extends keyof ColorizeSettings>(
    key: K,
    value: ColorizeSettings[K],
  ) => onChange({ ...settings, [key]: value });

  return (
    <section
      aria-labelledby="settings-heading"
      className="rounded-xl border border-line bg-surface p-5 shadow-card"
    >
      <h2 id="settings-heading" className="text-lg font-semibold">
        Colorization settings
      </h2>

      <fieldset className="mt-4">
        <legend className="text-sm font-medium">Quality</legend>
        <div className="mt-2 flex gap-2" role="radiogroup" aria-label="Quality">
          {(["standard", "high"] as const).map((q) => (
            <label
              key={q}
              className={cn(
                "flex-1 cursor-pointer rounded-lg border px-3 py-2 text-center text-sm font-medium transition-colors",
                settings.quality === q
                  ? "border-accent bg-accent-soft text-ink"
                  : "border-line bg-canvas text-muted hover:border-accent",
              )}
            >
              <input
                type="radio"
                name="quality"
                value={q}
                checked={settings.quality === q}
                onChange={() => set("quality", q)}
                className="sr-only"
              />
              {q === "standard" ? "Standard (fast)" : "High (larger model input)"}
            </label>
          ))}
        </div>
        <p className="mt-1.5 text-xs text-muted">
          High quality runs the model at a larger input resolution, which is
          slower on CPU but preserves finer color detail.
        </p>
      </fieldset>

      <div className="mt-5 space-y-3">
        <label htmlFor={contrastId} className="flex items-start gap-3">
          <input
            id={contrastId}
            type="checkbox"
            checked={settings.preserveContrast}
            onChange={(e) => set("preserveContrast", e.target.checked)}
            className="mt-0.5 h-4 w-4 accent-[hsl(var(--accent))]"
          />
          <span className="text-sm">
            <span className="font-medium">Preserve original contrast</span>
            <span className="block text-xs text-muted">
              Keeps the luminance channel of your photo untouched and only adds
              predicted color. Recommended.
            </span>
          </span>
        </label>

        <label htmlFor={denoiseId} className="flex items-start gap-3">
          <input
            id={denoiseId}
            type="checkbox"
            checked={settings.denoise}
            onChange={(e) => set("denoise", e.target.checked)}
            className="mt-0.5 h-4 w-4 accent-[hsl(var(--accent))]"
          />
          <span className="text-sm">
            <span className="font-medium">Remove mild noise</span>
            <span className="block text-xs text-muted">
              Applies a light non-local-means denoise pass before inference —
              useful for grainy scans.
            </span>
          </span>
        </label>

        {/* Implemented in the backend pipeline but intentionally off by default. */}
        <label
          htmlFor={faceId}
          className={cn("flex items-start gap-3", processing && "opacity-60")}
        >
          <input
            id={faceId}
            type="checkbox"
            checked={settings.faceEnhancement}
            onChange={(e) => set("faceEnhancement", e.target.checked)}
            disabled={processing}
            className="mt-0.5 h-4 w-4 accent-[hsl(var(--accent))]"
          />
          <span className="text-sm">
            <span className="font-medium">Face enhancement</span>
            <span className="block text-xs text-muted">
              Detects faces with OpenCV Haar cascades and applies slightly
              stronger chroma smoothing inside them. Experimental.
            </span>
          </span>
        </label>
      </div>

      <fieldset className="mt-5">
        <legend className="text-sm font-medium">Output format</legend>
        <div className="mt-2 flex gap-2">
          <label
            htmlFor={pngId}
            className={cn(
              "flex-1 cursor-pointer rounded-lg border px-3 py-2 text-center text-sm font-medium",
              settings.outputFormat === "png"
                ? "border-accent bg-accent-soft"
                : "border-line bg-canvas text-muted hover:border-accent",
            )}
          >
            <input
              id={pngId}
              type="radio"
              name="output_format"
              value="png"
              checked={settings.outputFormat === "png"}
              onChange={() => set("outputFormat", "png")}
              className="sr-only"
            />
            PNG (lossless)
          </label>
          <label
            htmlFor={jpegId}
            className={cn(
              "flex-1 cursor-pointer rounded-lg border px-3 py-2 text-center text-sm font-medium",
              settings.outputFormat === "jpeg"
                ? "border-accent bg-accent-soft"
                : "border-line bg-canvas text-muted hover:border-accent",
            )}
          >
            <input
              id={jpegId}
              type="radio"
              name="output_format"
              value="jpeg"
              checked={settings.outputFormat === "jpeg"}
              onChange={() => set("outputFormat", "jpeg")}
              className="sr-only"
            />
            JPEG (smaller)
          </label>
        </div>
      </fieldset>

      <button
        type="button"
        onClick={onStart}
        disabled={!canSubmit || processing}
        aria-disabled={!canSubmit || processing}
        className={cn(
          "mt-6 flex w-full items-center justify-center gap-2 rounded-lg px-4 py-3 text-base font-semibold text-white transition-colors",
          !canSubmit || processing
            ? "cursor-not-allowed bg-muted/50"
            : "bg-accent hover:bg-accent-hover",
        )}
      >
        <Sparkles aria-hidden className="h-5 w-5" />
        {processing ? "Colorizing…" : "Colorize image"}
      </button>
      {!canSubmit && !processing && (
        <p className="mt-2 text-center text-xs text-muted" role="status">
          Upload an image first to enable colorization.
        </p>
      )}
    </section>
  );
}
