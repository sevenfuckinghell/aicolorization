"use client";

import { Sparkles, Sliders, ShieldCheck } from "lucide-react";
import { useId } from "react";
import type { ColorizeSettings, Quality } from "@/lib/types";
import { cn } from "@/lib/utils";

interface SettingsPanelProps {
  settings: ColorizeSettings;
  onChange: (settings: ColorizeSettings) => void;
  onStart: () => void;
  processing: boolean;
  canSubmit: boolean;
}

/**
 * Advanced Colorization settings panel with quality presets,
 * model variant switcher, and edge-guided refinement controls.
 */
export function SettingsPanel({
  settings,
  onChange,
  onStart,
  processing,
  canSubmit,
}: SettingsPanelProps) {
  const contrastId = useId();
  const edgeRefinementId = useId();
  const shadowProtectionId = useId();
  const denoiseId = useId();
  const pngId = useId();
  const jpegId = useId();
  const chromaId = useId();
  const variantId = useId();
  const gradingId = useId();
  const sharpeningId = useId();
  const sharpStrengthId = useId();

  const set = <K extends keyof ColorizeSettings>(
    key: K,
    value: ColorizeSettings[K],
  ) => onChange({ ...settings, [key]: value });

  const qualityPresets: { id: Quality; label: string; desc: string }[] = [
    { id: "standard", label: "Standard", desc: "Fast (512px), raw DDColor" },
    { id: "high", label: "High Quality", desc: "768px, Edge-Guided & Sharpened" },
    { id: "maximum", label: "Maximum", desc: "Dual-stage edge refinement & max fidelity" },
  ];

  return (
    <section
      aria-labelledby="settings-heading"
      className="rounded-xl border border-line bg-surface p-5 shadow-card"
    >
      <div className="flex items-center justify-between">
        <h2 id="settings-heading" className="text-lg font-semibold">
          Colorization Settings
        </h2>
        <span className="flex items-center gap-1 text-xs text-muted">
          <Sliders className="h-3.5 w-3.5" /> Pro Mode
        </span>
      </div>

      {/* Quality Mode Presets */}
      <fieldset className="mt-4">
        <legend className="text-sm font-medium">Quality Preset</legend>
        <div className="mt-2 grid grid-cols-3 gap-2" role="radiogroup" aria-label="Quality Preset">
          {qualityPresets.map((q) => (
            <label
              key={q.id}
              className={cn(
                "cursor-pointer rounded-lg border p-2 text-center text-xs font-medium transition-all flex flex-col justify-center",
                settings.quality === q.id
                  ? "border-accent bg-accent-soft text-ink font-semibold ring-1 ring-accent"
                  : "border-line bg-canvas text-muted hover:border-accent",
              )}
            >
              <input
                type="radio"
                name="quality"
                value={q.id}
                checked={settings.quality === q.id}
                onChange={() => set("quality", q.id)}
                className="sr-only"
              />
              <span className="text-sm font-medium">{q.label}</span>
              <span className="text-[10px] opacity-75 mt-0.5 leading-tight">{q.desc}</span>
            </label>
          ))}
        </div>
      </fieldset>

      {/* Model Variant Selector */}
      <div className="mt-4">
        <label htmlFor={variantId} className="block text-sm font-medium">
          Pretrained Model Variant
        </label>
        <select
          id={variantId}
          value={settings.modelVariant ?? "piddnad/ddcolor_modelscope"}
          onChange={(e) => set("modelVariant", e.target.value)}
          className="mt-1.5 w-full rounded-lg border border-line bg-canvas px-3 py-2 text-sm text-ink focus:border-accent focus:outline-none"
        >
          <option value="piddnad/ddcolor_modelscope">piddnad/ddcolor_modelscope (Standard Balanced — Recommended)</option>
          <option value="piddnad/ddcolor_artistic">piddnad/ddcolor_artistic (Subtle & Natural skin tones, anti-cast)</option>
          <option value="piddnad/ddcolor_paper_tiny">piddnad/ddcolor_paper_tiny (Warm & Golden tones, fast)</option>
          <option value="piddnad/ddcolor_paper">piddnad/ddcolor_paper (High-Vibrancy saturated colors)</option>
        </select>
        <p className="mt-1 text-[11px] text-muted">
          Switches between authentic pretrained DDColor neural checkpoints cached in local storage.
        </p>
      </div>

      {/* Color Grading Preset */}
      <div className="mt-4">
        <label htmlFor={gradingId} className="block text-sm font-medium">
          Natural Color Grading Preset
        </label>
        <select
          id={gradingId}
          value={settings.colorGrading ?? "natural"}
          onChange={(e) => set("colorGrading", e.target.value as any)}
          className="mt-1.5 w-full rounded-lg border border-line bg-canvas px-3 py-2 text-sm text-ink focus:border-accent focus:outline-none"
        >
          <option value="natural">Natural (Photographic balance & realistic highlights — Recommended)</option>
          <option value="vivid">Vivid (Richer color separation without neon cast)</option>
          <option value="cinematic">Cinematic (Filmic tonal latitude & soft highlight roll-off)</option>
          <option value="original_ai">Original AI (Unmodified raw neural DDColor prediction)</option>
        </select>
        <p className="mt-1 text-[11px] text-muted">
          Applies color-science grading directly in CIELAB space without fake tints or global hue shifts.
        </p>
      </div>

      {/* Detail Sharpening Controls */}
      <div className="mt-5 border-t border-line/50 pt-4 space-y-3">
        <label htmlFor={sharpeningId} className="flex items-start gap-3 cursor-pointer">
          <input
            id={sharpeningId}
            type="checkbox"
            checked={settings.sharpening ?? true}
            onChange={(e) => set("sharpening", e.target.checked)}
            className="mt-0.5 h-4 w-4 accent-[hsl(var(--accent))]"
          />
          <span className="text-sm">
            <span className="font-medium flex items-center gap-1.5">
              Detail-Preserving Sharpening <span className="rounded bg-accent/15 text-accent text-[10px] px-1 py-0.2 font-mono">L-ONLY</span>
            </span>
            <span className="block text-xs text-muted">
              Enhances fine whiskers, fur strands, eye contours, and textures exclusively on native luminance without white halos or color fringing.
            </span>
          </span>
        </label>

        {(settings.sharpening ?? true) && (
          <div className="ml-7 pt-1">
            <div className="flex items-center justify-between text-xs text-muted">
              <label htmlFor={sharpStrengthId} className="font-medium text-ink">
                Sharpening Strength
              </label>
              <span className="font-mono">{settings.sharpeningStrength ?? 0.35}</span>
            </div>
            <input
              id={sharpStrengthId}
              type="range"
              min="0.10"
              max="0.65"
              step="0.05"
              value={settings.sharpeningStrength ?? 0.35}
              onChange={(e) => set("sharpeningStrength", parseFloat(e.target.value))}
              className="mt-1.5 w-full accent-[hsl(var(--accent))]"
            />
            <div className="mt-0.5 flex justify-between text-[10px] text-muted">
              <span>Subtle (0.15)</span>
              <span>Balanced (0.35)</span>
              <span>Crisp (0.60)</span>
            </div>
          </div>
        )}
      </div>

      {/* Fidelity Toggles */}
      <div className="mt-4 space-y-3 border-t border-line/50 pt-4">
        <label htmlFor={edgeRefinementId} className="flex items-start gap-3 cursor-pointer">
          <input
            id={edgeRefinementId}
            type="checkbox"
            checked={settings.edgeRefinement ?? true}
            onChange={(e) => set("edgeRefinement", e.target.checked)}
            className="mt-0.5 h-4 w-4 accent-[hsl(var(--accent))]"
          />
          <span className="text-sm">
            <span className="font-medium flex items-center gap-1.5">
              Edge-Aware Boundary Refinement
            </span>
            <span className="block text-xs text-muted">
              Applies luminance-guided filtering using the original photograph. Eliminates color bleeding across whiskers, fur edges, eyes, and clothing folds.
            </span>
          </span>
        </label>

        <label htmlFor={shadowProtectionId} className="flex items-start gap-3 cursor-pointer">
          <input
            id={shadowProtectionId}
            type="checkbox"
            checked={settings.blackPreserve ?? true}
            onChange={(e) => set("blackPreserve", e.target.checked)}
            className="mt-0.5 h-4 w-4 accent-[hsl(var(--accent))]"
          />
          <span className="text-sm">
            <span className="font-medium">Shadow & Highlight Neutrality</span>
            <span className="block text-xs text-muted">
              Smoothly attenuates chromatic tints in deep shadows (L &lt; 15) and specular highlights (L &gt; 92) to prevent false purple/magenta casts.
            </span>
          </span>
        </label>

        <label htmlFor={contrastId} className="flex items-start gap-3 cursor-pointer">
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
              Keeps 100% of the original photo&apos;s native spatial resolution and lighting while predicting only chromatic components.
            </span>
          </span>
        </label>

        <label htmlFor={denoiseId} className="flex items-start gap-3 cursor-pointer">
          <input
            id={denoiseId}
            type="checkbox"
            checked={settings.denoise}
            onChange={(e) => set("denoise", e.target.checked)}
            className="mt-0.5 h-4 w-4 accent-[hsl(var(--accent))]"
          />
          <span className="text-sm">
            <span className="font-medium">Pre-Inference Noise Filter (Optional)</span>
            <span className="block text-xs text-muted">
              Applies gentle bilateral filter before neural inference. Recommended only for heavy scanner noise in vintage prints; keep disabled for clean photographs.
            </span>
          </span>
        </label>
      </div>

      {/* AI Color Intensity Slider */}
      <div className="mt-5 border-t border-line/50 pt-4">
        <div className="flex items-center justify-between text-sm">
          <label htmlFor={chromaId} className="font-medium">
            AI Color Saturation Intensity
          </label>
          <span className="text-xs text-muted font-mono">
            {Math.round((settings.chromaStrength ?? 1.0) * 100)}%
          </span>
        </div>
        <input
          id={chromaId}
          type="range"
          min="0.5"
          max="1.2"
          step="0.05"
          value={settings.chromaStrength ?? 1.0}
          onChange={(e) => set("chromaStrength", parseFloat(e.target.value))}
          className="mt-2 w-full accent-[hsl(var(--accent))]"
        />
        <div className="mt-1 flex justify-between text-[11px] text-muted">
          <span>Subtle (50%)</span>
          <span>Natural AI (100%)</span>
          <span>Vivid (120%)</span>
        </div>
      </div>

      {/* Output Format */}
      <fieldset className="mt-4">
        <legend className="text-sm font-medium">Output Format</legend>
        <div className="mt-2 flex gap-2">
          <label
            htmlFor={pngId}
            className={cn(
              "flex-1 cursor-pointer rounded-lg border px-3 py-2 text-center text-sm font-medium transition-colors",
              settings.outputFormat === "png"
                ? "border-accent bg-accent-soft text-ink"
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
            PNG (Lossless)
          </label>
          <label
            htmlFor={jpegId}
            className={cn(
              "flex-1 cursor-pointer rounded-lg border px-3 py-2 text-center text-sm font-medium transition-colors",
              settings.outputFormat === "jpeg"
                ? "border-accent bg-accent-soft text-ink"
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
            JPEG (Compressed)
          </label>
        </div>
      </fieldset>

      <button
        type="button"
        onClick={onStart}
        disabled={!canSubmit || processing}
        aria-disabled={!canSubmit || processing}
        className={cn(
          "mt-6 flex w-full items-center justify-center gap-2 rounded-lg px-4 py-3 text-base font-semibold text-white transition-colors shadow-md",
          !canSubmit || processing
            ? "cursor-not-allowed bg-muted/50"
            : "bg-accent hover:bg-accent-hover",
        )}
      >
        <Sparkles aria-hidden className="h-5 w-5" />
        {processing ? "Colorizing with DDColor…" : "Colorize Photo"}
      </button>
      {!canSubmit && !processing && (
        <p className="mt-2 text-center text-xs text-muted" role="status">
          Upload an image above to start colorization.
        </p>
      )}
    </section>
  );
}
