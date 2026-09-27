"use client";

import {
  AlertTriangle,
  CheckCircle2,
  Download,
  Github,
  Image as ImageIcon,
  Info,
  Sparkles,
  Upload,
  Wand2,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { ComparisonSlider } from "@/components/comparison-slider";
import { HistoryPanel, type HistoryEntry } from "@/components/history-panel";
import { ImagePreview } from "@/components/image-preview";
import { ProcessingState } from "@/components/processing-state";
import { ResultActions } from "@/components/result-actions";
import { SettingsPanel } from "@/components/settings-panel";
import { ThemeToggle } from "@/components/theme-toggle";
import { UploadZone } from "@/components/upload-zone";
import { colorizeImage, getInfo } from "@/lib/api";
import {
  ApiError,
  DEFAULT_SETTINGS,
  type ColorizeResult,
  type ColorizeSettings,
  type ProcessingPhase,
  type UploadFileInfo,
} from "@/lib/types";
import { base64ToBlob, loadImageDimensions } from "@/lib/utils";

interface BackendInfo {
  reachable: boolean;
  fallbackMode: boolean;
  modelName: string;
  maxUploadMb: number;
}

export default function HomePage() {
  const [file, setFile] = useState<File | null>(null);
  const [fileInfo, setFileInfo] = useState<UploadFileInfo | null>(null);
  const [settings, setSettings] = useState<ColorizeSettings>(DEFAULT_SETTINGS);
  const [phase, setPhase] = useState<ProcessingPhase>("idle");
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ColorizeResult | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [backend, setBackend] = useState<BackendInfo | null>(null);
  const processingRef = useRef(false);
  const blobUrlRef = useRef<string | null>(null);

  // Probe backend capabilities on mount so the UI can label fallback mode.
  useEffect(() => {
    let cancelled = false;
    getInfo()
      .then((info) => {
        if (!cancelled) {
          setBackend({
            reachable: true,
            fallbackMode: info.fallback_mode,
            modelName: info.model_name,
            maxUploadMb: info.max_upload_mb,
          });
        }
      })
      .catch(() => {
        if (!cancelled)
          setBackend({
            reachable: false,
            fallbackMode: true,
            modelName: "unknown",
            maxUploadMb: 10,
          });
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const resetWorkspace = useCallback(() => {
    if (blobUrlRef.current) URL.revokeObjectURL(blobUrlRef.current);
    blobUrlRef.current = null;
    if (fileInfo?.dataUrl.startsWith("blob:")) {
      // data URLs created with FileReader are strings; nothing to revoke.
    }
    setFile(null);
    setFileInfo(null);
    setResult(null);
    setPhase("idle");
    setUploadProgress(null);
    setError(null);
  }, [fileInfo]);

  const handleFileSelected = useCallback(async (selected: File) => {
    setError(null);
    setResult(null);
    const reader = new FileReader();
    reader.onload = async () => {
      const dataUrl = String(reader.result);
      try {
        const dims = await loadImageDimensions(dataUrl);
        setFile(selected);
        setFileInfo({
          name: selected.name,
          sizeBytes: selected.size,
          type: selected.type,
          width: dims.width,
          height: dims.height,
          dataUrl,
        });
      } catch {
        setError(
          "The file could not be decoded as an image. It may be corrupt or truncated.",
        );
      }
    };
    reader.onerror = () =>
      setError("Could not read the selected file. Please try again.");
    reader.readAsDataURL(selected);
  }, []);

  const handleColorize = useCallback(async () => {
    // Guard against duplicate submissions.
    if (!file || processingRef.current) return;
    processingRef.current = true;
    setError(null);
    if (result) {
      URL.revokeObjectURL(result.blobUrl);
      blobUrlRef.current = null;
    }
    setResult(null);
    setPhase("uploading");
    setUploadProgress(0);

    // Simulated server-side phase progression while the request is in flight.
    let step = 0;
    const phaseTimer = window.setInterval(() => {
      step += 1;
      if (step === 2) setPhase("preparing");
      if (step === 4) setPhase("running-model");
      if (step === 7) setPhase("postprocessing");
    }, 900);

    try {
      const response = await colorizeImage(
        file,
        settings,
        (pct) => {
          setUploadProgress(pct);
          if (pct >= 100) setPhase("preparing");
        },
      );
      window.clearInterval(phaseTimer);
      setPhase("preparing-download");
      const blob = base64ToBlob(response.image_base64, response.mime_type);
      const url = URL.createObjectURL(blob);
      blobUrlRef.current = url;
      const nextResult: ColorizeResult = {
        blobUrl: url,
        base64: response.image_base64,
        mimeType: response.mime_type,
        filename: response.filename,
        width: response.width,
        height: response.height,
        processingTimeMs: response.processing_time_ms,
        model: response.model,
        fallbackMode: response.fallback_mode,
        requestId: response.request_id,
      };
      setResult(nextResult);
      setPhase("idle");
      setUploadProgress(null);
      setHistory((h) =>
        [
          {
            id: response.request_id,
            name: file.name,
            timestamp: Date.now(),
            width: response.width,
            height: response.height,
            model: response.model,
            fallbackMode: response.fallback_mode,
            processingTimeMs: response.processing_time_ms,
            resultUrl: url,
          },
          ...h,
        ].slice(0, 8),
      );
    } catch (err) {
      window.clearInterval(phaseTimer);
      setPhase("idle");
      setUploadProgress(null);
      if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError(
          "An unexpected error occurred while colorizing the image. Please try again.",
        );
        // Detailed diagnostics stay in the browser console only.
        console.error(err);
      }
    } finally {
      processingRef.current = false;
    }
  }, [file, settings, result]);

  return (
    <div className="flex min-h-screen flex-col">
      {/* ---------------- Header ---------------- */}
      <header className="border-b border-line bg-surface/80 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-4 py-3 sm:px-6">
          <a href="#top" className="flex items-center gap-2.5">
            <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-accent text-white">
              <Wand2 aria-hidden className="h-5 w-5" />
            </span>
            <span>
              <span className="block font-serif text-lg font-bold leading-tight">
                ColorRevive
              </span>
              <span className="block text-xs text-muted">
                AI photo colorization
              </span>
            </span>
          </a>
          <nav aria-label="Primary" className="flex items-center gap-1 sm:gap-4">
            <a
              href="#how-it-works"
              className="hidden rounded-md px-2 py-1 text-sm font-medium text-muted hover:text-accent sm:block"
            >
              How it works
            </a>
            <a
              href="#limitations"
              className="hidden rounded-md px-2 py-1 text-sm font-medium text-muted hover:text-accent sm:block"
            >
              Limitations
            </a>
            <ThemeToggle />
          </nav>
        </div>
      </header>

      <main id="top" className="mx-auto w-full max-w-6xl flex-1 px-4 sm:px-6">
        {/* ---------------- Hero ---------------- */}
        <section className="py-10 sm:py-14">
          <h1 className="max-w-3xl font-serif text-4xl font-bold leading-tight sm:text-5xl">
            Bring forgotten photographs back to life.
          </h1>
          <p className="mt-4 max-w-2xl text-lg text-muted">
            Upload a black-and-white image and generate a natural-looking color
            version with an AI colorization model.
          </p>
          <p className="mt-3 flex max-w-2xl items-start gap-2 rounded-lg border border-line bg-surface px-3 py-2 text-sm text-muted">
            <Info aria-hidden className="mt-0.5 h-4 w-4 shrink-0 text-accent" />
            Best results come from clear, well-lit images. Generated colors are
            predictions, not historical guarantees.
          </p>

          {/* Backend / model status banner — never fake success */}
          {backend && !backend.reachable && (
            <div
              role="alert"
              className="mt-5 flex max-w-2xl items-start gap-2 rounded-lg border border-danger/40 bg-danger/5 px-3 py-2 text-sm"
            >
              <AlertTriangle aria-hidden className="mt-0.5 h-4 w-4 shrink-0 text-danger" />
              <span>
                The colorization backend is unreachable. Start the FastAPI
                service (see README) and reload this page.
              </span>
            </div>
          )}
          {backend?.reachable && backend.fallbackMode && (
            <div className="mt-5 flex max-w-2xl items-start gap-2 rounded-lg border border-amber-500/50 bg-amber-500/10 px-3 py-2 text-sm">
              <AlertTriangle aria-hidden className="mt-0.5 h-4 w-4 shrink-0 text-amber-600 dark:text-amber-400" />
              <span>
                <strong className="font-semibold">Development mode — fallback
                processing active.</strong> No trained checkpoint is loaded, so
                output uses a deterministic heuristic colorizer and is{" "}
                <em>not</em> neural AI colorization. Set{" "}
                <code className="rounded bg-canvas px-1 py-0.5 text-xs">
                  MODEL_CHECKPOINT_PATH
                </code>{" "}
                to enable the real model.
              </span>
            </div>
          )}
          {backend?.reachable && !backend.fallbackMode && (
            <div className="mt-5 flex max-w-2xl items-start gap-2 rounded-lg border border-green-600/40 bg-green-600/10 px-3 py-2 text-sm">
              <CheckCircle2 aria-hidden className="mt-0.5 h-4 w-4 shrink-0 text-green-700 dark:text-green-400" />
              <span>
                Neural model <strong>{backend.modelName}</strong> is loaded and
                active.
              </span>
            </div>
          )}
        </section>

        {/* ---------------- Workspace ---------------- */}
        <section aria-labelledby="workspace-heading" className="pb-12">
          <h2 id="workspace-heading" className="sr-only">
            Colorization workspace
          </h2>
          <div className="grid gap-6 lg:grid-cols-2">
            <div>
              <UploadZone
                file={fileInfo}
                onFileSelected={(f) => void handleFileSelected(f)}
                onRemove={resetWorkspace}
                onError={setError}
                maxUploadMb={backend?.maxUploadMb ?? 10}
                disabled={phase !== "idle"}
              />
              <ImagePreview file={fileInfo} />
            </div>
            <div>
              <SettingsPanel
                settings={settings}
                onChange={setSettings}
                onStart={() => void handleColorize()}
                processing={phase !== "idle"}
                canSubmit={file !== null}
              />
              <ProcessingState
                phase={phase}
                uploadProgress={uploadProgress}
                errorMessage={error}
                onDismissError={() => setError(null)}
              />
            </div>
          </div>

          {/* ---------------- Results ---------------- */}
          {result && fileInfo && (
            <div className="mt-10">
              <h2 className="text-2xl font-semibold">Result</h2>
              {result.fallbackMode && (
                <p className="mt-2 inline-flex items-center gap-2 rounded-md bg-amber-500/15 px-3 py-1.5 text-sm font-medium text-amber-800 dark:text-amber-300">
                  <AlertTriangle aria-hidden className="h-4 w-4" />
                  Fallback mode output — deterministic heuristic coloring, not
                  neural AI prediction.
                </p>
              )}
              <div className="mt-4 grid gap-6 lg:grid-cols-[1fr_1.2fr]">
                <ComparisonSlider
                  beforeSrc={fileInfo.dataUrl}
                  afterSrc={result.blobUrl}
                  beforeAlt={`Original black-and-white image ${fileInfo.name}`}
                  afterAlt={`Colorized version of ${fileInfo.name}`}
                />
                <div>
                  <div className="grid grid-cols-2 gap-4">
                    <figure className="rounded-xl border border-line bg-surface p-3 shadow-card">
                      <figcaption className="mb-2 flex items-center gap-2 text-sm font-semibold">
                        <ImageIcon aria-hidden className="h-4 w-4 text-accent" />
                        Original
                      </figcaption>
                      <img
                        src={fileInfo.dataUrl}
                        alt={`Original grayscale photo ${fileInfo.name}`}
                        className="w-full rounded object-contain"
                      />
                    </figure>
                    <figure className="rounded-xl border border-line bg-surface p-3 shadow-card">
                      <figcaption className="mb-2 flex items-center gap-2 text-sm font-semibold">
                        <Sparkles aria-hidden className="h-4 w-4 text-accent" />
                        Colorized
                      </figcaption>
                      <img
                        src={result.blobUrl}
                        alt={`AI colorized photo derived from ${fileInfo.name}`}
                        className="w-full rounded object-contain"
                      />
                    </figure>
                  </div>
                  <dl className="mt-4 grid grid-cols-2 gap-x-4 gap-y-2 rounded-xl border border-line bg-surface p-4 text-sm shadow-card sm:grid-cols-3">
                    <div>
                      <dt className="text-xs font-medium uppercase tracking-wide text-muted">
                        Output size
                      </dt>
                      <dd>
                        {result.width} × {result.height} px
                      </dd>
                    </div>
                    <div>
                      <dt className="text-xs font-medium uppercase tracking-wide text-muted">
                        Processing time
                      </dt>
                      <dd>{(result.processingTimeMs / 1000).toFixed(2)} s</dd>
                    </div>
                    <div>
                      <dt className="text-xs font-medium uppercase tracking-wide text-muted">
                        Model
                      </dt>
                      <dd>
                        {result.fallbackMode ? "fallback-heuristic" : result.model}
                      </dd>
                    </div>
                    <div>
                      <dt className="text-xs font-medium uppercase tracking-wide text-muted">
                        Format
                      </dt>
                      <dd>{result.mime_type}</dd>
                    </div>
                    <div className="col-span-2">
                      <dt className="text-xs font-medium uppercase tracking-wide text-muted">
                        Request ID
                      </dt>
                      <dd className="truncate font-mono text-xs">{result.requestId}</dd>
                    </div>
                  </dl>
                  <ResultActions
                    result={result}
                    originalName={fileInfo.name}
                    onReset={resetWorkspace}
                  />
                </div>
              </div>
            </div>
          )}

          <HistoryPanel entries={history} onClear={() => setHistory([])} />
        </section>

        {/* ---------------- How it works ---------------- */}
        <section
          id="how-it-works"
          aria-labelledby="how-heading"
          className="border-t border-line py-12"
        >
          <h2 id="how-heading" className="text-2xl font-semibold">
            How it works
          </h2>
          <ol className="mt-6 grid gap-4 sm:grid-cols-3">
            {[
              {
                icon: Upload,
                title: "1 · Upload",
                body: "Drop a JPG, PNG, or WEBP photograph. Your image stays in the pipeline only for the duration of processing.",
              },
              {
                icon: Sparkles,
                title: "2 · AI predicts colors",
                body: "A U-Net model reads the luminance channel in Lab color space and predicts the two chrominance channels (a and b), which are recombined with your original lightness.",
              },
              {
                icon: Download,
                title: "3 · Download the result",
                body: "Compare before and after with the slider, then download the full-resolution colorized image as PNG or JPEG.",
              },
            ].map(({ icon: Icon, title, body }) => (
              <li
                key={title}
                className="rounded-xl border border-line bg-surface p-5 shadow-card"
              >
                <Icon aria-hidden className="h-6 w-6 text-accent" />
                <h3 className="mt-3 font-semibold">{title}</h3>
                <p className="mt-1.5 text-sm text-muted">{body}</p>
              </li>
            ))}
          </ol>
        </section>

        {/* ---------------- Limitations ---------------- */}
        <section
          id="limitations"
          aria-labelledby="limits-heading"
          className="border-t border-line py-12"
        >
          <h2 id="limits-heading" className="text-2xl font-semibold">
            About these colors — please read
          </h2>
          <blockquote className="mt-4 border-l-4 border-accent pl-4 font-serif text-lg italic">
            Colorization is an AI prediction. Since the original colors are
            unknown, the generated colors may not exactly match the historical
            colors.
          </blockquote>
          <ul className="mt-5 list-disc space-y-2 pl-6 text-sm text-muted">
            <li>
              Original colors cannot be recovered with certainty from a
              black-and-white image.
            </li>
            <li>
              Results depend heavily on image quality, lighting, and content.
            </li>
            <li>
              Skin tones, skies, vegetation, clothing, and historical objects
              may be predicted incorrectly.
            </li>
            <li>
              This tool must not be used as evidence of the original colors of
              any photograph.
            </li>
          </ul>
        </section>
      </main>

      {/* ---------------- Footer ---------------- */}
      <footer className="border-t border-line bg-surface">
        <div className="mx-auto flex max-w-6xl flex-col gap-3 px-4 py-8 text-sm text-muted sm:flex-row sm:items-start sm:justify-between sm:px-6">
          <div>
            <p className="font-serif text-base font-bold text-ink">
              ColorRevive
            </p>
            <p className="mt-1 max-w-md">
              Built with Next.js, FastAPI, PyTorch, and OpenCV. Lab-space
              U-Net colorization with a CPU-safe inference path.
            </p>
          </div>
          <div className="max-w-md">
            <p>
              Images are processed for colorization. In local development,
              files are stored temporarily only during processing and are
              removed afterward. Configure persistent storage explicitly
              before using this application in production.
            </p>
            <p className="mt-2 flex items-center gap-3">
              <a
                href="https://github.com/example/colorrevive"
                className="inline-flex items-center gap-1.5 underline decoration-line underline-offset-2 hover:text-accent"
              >
                <Github aria-hidden className="h-4 w-4" /> GitHub (placeholder)
              </a>
              <span aria-hidden>·</span>
              <span>License: MIT (placeholder)</span>
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
