"use client";

import { ImageIcon } from "lucide-react";
import type { UploadFileInfo } from "@/lib/types";
import { formatBytes } from "@/lib/utils";

interface ImagePreviewProps {
  file: UploadFileInfo | null;
  label?: string;
}

/** Aspect-ratio-preserving preview of the uploaded image with metadata. */
export function ImagePreview({ file, label = "Uploaded image preview" }: ImagePreviewProps) {
  if (!file) return null;
  return (
    <figure className="mt-4 rounded-xl border border-line bg-surface p-3 shadow-card">
      <figcaption className="mb-2 flex items-center gap-2 text-sm font-semibold text-ink">
        <ImageIcon aria-hidden className="h-4 w-4 text-accent" />
        {label}
      </figcaption>
      <div className="flex max-h-[420px] items-center justify-center overflow-hidden rounded-lg bg-canvas">
        {/* object-contain guarantees the image is never stretched */}
        <img
          src={file.dataUrl}
          alt={`Preview of ${file.name}`}
          className="max-h-[420px] w-full object-contain"
        />
      </div>
      <dl className="mt-3 grid grid-cols-3 gap-2 text-xs text-muted">
        <div>
          <dt className="font-medium">Dimensions</dt>
          <dd>
            {file.width} × {file.height} px
          </dd>
        </div>
        <div>
          <dt className="font-medium">File size</dt>
          <dd>{formatBytes(file.sizeBytes)}</dd>
        </div>
        <div>
          <dt className="font-medium">Type</dt>
          <dd>{file.type}</dd>
        </div>
      </dl>
    </figure>
  );
}
