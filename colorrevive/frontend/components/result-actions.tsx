"use client";

import { Download, RotateCcw } from "lucide-react";
import type { ColorizeResult, OutputFormat } from "@/lib/types";
import { buildDownloadFilename, downloadBlob } from "@/lib/utils";

interface ResultActionsProps {
  result: ColorizeResult;
  originalName: string;
  onReset: () => void;
}

/**
 * Downloads operate on the actual processed image bytes returned by the
 * backend (converted from base64 to a Blob) — never on a placeholder.
 */
export function ResultActions({
  result,
  originalName,
  onReset,
}: ResultActionsProps) {
  const fetchBlob = async (): Promise<Blob> => {
    const response = await fetch(result.blobUrl);
    return response.blob();
  };

  const handleDownload = async (format: OutputFormat) => {
    const blob = await fetchBlob();
    // If the user asked for JPEG but the server encoded PNG, re-encode in
    // the browser via canvas so both buttons always deliver real pixels.
    if (
      format === "jpeg" &&
      !blob.type.includes("jpeg") &&
      !blob.type.includes("jpg")
    ) {
      const bitmap = await createImageBitmap(blob);
      const canvas = document.createElement("canvas");
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
      const ctx = canvas.getContext("2d");
      if (!ctx) throw new Error("Canvas unavailable");
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(bitmap, 0, 0);
      const jpegBlob = await new Promise<Blob | null>((resolve) =>
        canvas.toBlob(resolve, "image/jpeg", 0.92),
      );
      if (jpegBlob) {
        downloadBlob(jpegBlob, buildDownloadFilename(originalName, "jpeg"));
        return;
      }
    }
    downloadBlob(blob, buildDownloadFilename(originalName, format));
  };

  return (
    <div className="mt-4 flex flex-wrap gap-3">
      <button
        type="button"
        onClick={() => void handleDownload("png")}
        className="inline-flex items-center gap-2 rounded-lg bg-accent px-4 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-accent-hover"
      >
        <Download aria-hidden className="h-4 w-4" />
        Download PNG
      </button>
      <button
        type="button"
        onClick={() => void handleDownload("jpeg")}
        className="inline-flex items-center gap-2 rounded-lg border border-accent bg-surface px-4 py-2.5 text-sm font-semibold text-accent transition-colors hover:bg-accent-soft"
      >
        <Download aria-hidden className="h-4 w-4" />
        Download JPG
      </button>
      <button
        type="button"
        onClick={onReset}
        className="inline-flex items-center gap-2 rounded-lg border border-line bg-surface px-4 py-2.5 text-sm font-medium text-ink transition-colors hover:bg-accent-soft"
      >
        <RotateCcw aria-hidden className="h-4 w-4" />
        Start over with a new image
      </button>
    </div>
  );
}
