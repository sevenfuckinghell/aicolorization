"use client";

import { ImagePlus, X } from "lucide-react";
import { useCallback, useId, useRef, useState } from "react";
import type { UploadFileInfo } from "@/lib/types";
import { cn, validateImageFile } from "@/lib/utils";

interface UploadZoneProps {
  file: UploadFileInfo | null;
  onFileSelected: (file: File) => void;
  onRemove: () => void;
  onError: (message: string) => void;
  maxUploadMb?: number;
  disabled?: boolean;
}

/**
 * Drag-and-drop upload zone with a keyboard-accessible browse button.
 * Validates MIME type + extension + size before accepting a file.
 */
export function UploadZone({
  file,
  onFileSelected,
  onRemove,
  onError,
  maxUploadMb = 10,
  disabled = false,
}: UploadZoneProps) {
  const inputId = useId();
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);

  const handleFiles = useCallback(
    (files: FileList | null) => {
      if (!files || files.length === 0) {
        onError("No file was selected. Please choose an image file.");
        return;
      }
      const candidate = files[0];
      const validation = validateImageFile(candidate, maxUploadMb);
      if (!validation.ok) {
        onError(validation.message ?? "The selected file is not valid.");
        return;
      }
      onFileSelected(candidate);
    },
    [maxUploadMb, onFileSelected, onError],
  );

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      setDragActive(false);
      if (disabled) return;
      handleFiles(event.dataTransfer.files);
    },
    [disabled, handleFiles],
  );

  return (
    <div className="rounded-xl border border-line bg-surface p-5 shadow-card">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={onDrop}
        className={cn(
          "flex flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed px-6 py-10 text-center transition-colors",
          dragActive
            ? "border-accent bg-accent-soft"
            : "border-line bg-canvas/50",
          disabled && "opacity-60",
        )}
        aria-live="polite"
      >
        <ImagePlus aria-hidden className="h-10 w-10 text-accent" />
        {file ? (
          <>
            <p className="text-sm font-medium">
              Loaded: <span className="underline">{file.name}</span>
            </p>
            <p className="text-xs text-muted">
              {file.width} × {file.height} px ·{" "}
              {(file.sizeBytes / 1024).toFixed(0)} KB
            </p>
            <button
              type="button"
              onClick={onRemove}
              className="mt-1 inline-flex items-center gap-1.5 rounded-lg border border-line bg-surface px-3 py-1.5 text-sm font-medium hover:bg-accent-soft"
            >
              <X aria-hidden className="h-4 w-4" /> Remove image
            </button>
          </>
        ) : (
          <>
            <p className="text-base font-semibold">
              Drag &amp; drop a black-and-white photo here
            </p>
            <p className="text-sm text-muted">or</p>
            <div>
              <label htmlFor={inputId}>
                <span
                  role="button"
                  tabIndex={disabled ? -1 : 0}
                  aria-disabled={disabled}
                  onKeyDown={(e) => {
                    if (!disabled && (e.key === "Enter" || e.key === " ")) {
                      e.preventDefault();
                      inputRef.current?.click();
                    }
                  }}
                  onClick={() => !disabled && inputRef.current?.click()}
                  className="inline-block cursor-pointer rounded-lg bg-accent px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-accent-hover"
                >
                  Browse files
                </span>
              </label>
              <input
                ref={inputRef}
                id={inputId}
                type="file"
                accept="image/jpeg,image/png,image/webp,.jpg,.jpeg,.png,.webp"
                className="sr-only"
                aria-label="Choose an image file to upload"
                disabled={disabled}
                onChange={(e) => {
                  handleFiles(e.target.files);
                  // allow selecting the same file again later
                  e.target.value = "";
                }}
              />
            </div>
            <p className="mt-2 text-xs text-muted">
              Accepted formats: JPG, JPEG, PNG, WEBP · Maximum file size:{" "}
              {maxUploadMb} MB
            </p>
          </>
        )}
      </div>
    </div>
  );
}
