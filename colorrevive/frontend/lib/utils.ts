import type { OutputFormat } from "./types";

/** Join conditional class names. */
export function cn(...classes: Array<string | false | null | undefined>): string {
  return classes.filter(Boolean).join(" ");
}

/** Human-readable byte size, e.g. 1536 -> "1.5 KB". */
export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return "—";
  if (bytes === 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const i = Math.min(
    Math.floor(Math.log(bytes) / Math.log(1024)),
    units.length - 1,
  );
  const value = bytes / 1024 ** i;
  return `${value.toFixed(value >= 10 || i === 0 ? 0 : 1)} ${units[i]}`;
}

/** Clamp a number into [min, max]. */
export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

const ACCEPTED_MIME_TYPES = ["image/jpeg", "image/png", "image/webp"] as const;
const ACCEPTED_EXTENSIONS = ["jpg", "jpeg", "png", "webp"] as const;

export interface FileValidation {
  ok: boolean;
  errorCode?:
    | "UNSUPPORTED_TYPE"
    | "TOO_LARGE"
    | "EMPTY_FILE"
    | "NAME_MISMATCH";
  message?: string;
}

/**
 * Validate an uploaded file on the client before sending it:
 * MIME type, extension, emptiness and size must all pass.
 */
export function validateImageFile(
  file: File,
  maxUploadMb = 10,
): FileValidation {
  const ext = file.name.toLowerCase().split(".").pop() ?? "";
  const isMimeOk = (ACCEPTED_MIME_TYPES as readonly string[]).includes(
    file.type,
  );
  const isExtOk = (ACCEPTED_EXTENSIONS as readonly string[]).includes(ext);

  if (file.size === 0) {
    return {
      ok: false,
      errorCode: "EMPTY_FILE",
      message: "The selected file is empty. Please choose a valid image.",
    };
  }
  if (!isMimeOk || !isExtOk) {
    return {
      ok: false,
      errorCode: "UNSUPPORTED_TYPE",
      message:
        "Unsupported file type. Only JPG, JPEG, PNG, and WEBP images are accepted.",
    };
  }
  // Reject e.g. a .exe renamed to .png with a spoofed MIME type.
  if (isMimeOk !== isExtOk) {
    return {
      ok: false,
      errorCode: "NAME_MISMATCH",
      message:
        "The file extension and detected type do not match. Please choose a genuine image file.",
    };
  }
  if (file.size > maxUploadMb * 1024 * 1024) {
    return {
      ok: false,
      errorCode: "TOO_LARGE",
      message: `File is too large. The maximum upload size is ${maxUploadMb} MB.`,
    };
  }
  return { ok: true };
}

/** Build the download filename: colorrevive-<stem>-colorized.<ext>. */
export function buildDownloadFilename(originalName: string, format: OutputFormat): string {
  const stem = originalName.replace(/\.[^.]+$/, "").replace(/[^a-zA-Z0-9-_]+/g, "-").slice(0, 60) || "image";
  const ext = format === "jpeg" ? "jpg" : "png";
  return `colorrevive-${stem}-colorized.${ext}`;
}

/** Trigger a browser download for a Blob under a given filename. */
export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  // Give the browser a tick to start the download before revoking.
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}

/** Convert a base64 payload returned by the API into a Blob. */
export function base64ToBlob(base64: string, mimeType: string): Blob {
  const byteString = atob(base64);
  const bytes = new Uint8Array(byteString.length);
  for (let i = 0; i < byteString.length; i++) {
    bytes[i] = byteString.charCodeAt(i);
  }
  return new Blob([bytes], { type: mimeType });
}

/** Read natural dimensions of an image from a data/blob URL. */
export function loadImageDimensions(src: string): Promise<{ width: number; height: number }> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve({ width: img.naturalWidth, height: img.naturalHeight });
    img.onerror = () => reject(new Error("Could not decode image"));
    img.src = src;
  });
}
