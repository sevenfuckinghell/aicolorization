import { ApiError } from "./types";
import type {
  ApiInfo,
  ColorizeResponse,
  ColorizeSettings,
  HealthResponse,
  ModelStatus,
} from "./types";

/** Base URL of the FastAPI backend, configurable via environment. */
export const API_BASE_URL: string =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

interface JsonErrorBody {
  success?: boolean;
  error?: { code?: string; message?: string; request_id?: string };
}

async function parseError(response: Response): Promise<ApiError> {
  let code = "SERVER_ERROR";
  let message = `The server returned an unexpected error (HTTP ${response.status}).`;
  let requestId: string | undefined;
  try {
    const body = (await response.json()) as JsonErrorBody;
    if (body?.error) {
      code = body.error.code ?? code;
      message = body.error.message ?? message;
      requestId = body.error.request_id;
    }
  } catch {
    /* non-JSON error body — keep generic message */
  }
  return new ApiError(code, message, requestId);
}

export async function getHealth(): Promise<HealthResponse> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/health`);
  } catch {
    throw new ApiError(
      "BACKEND_UNAVAILABLE",
      "Cannot reach the colorization service. Make sure the backend is running.",
    );
  }
  if (!response.ok) throw parseError(response);
  return response.json() as Promise<HealthResponse>;
}

export async function getInfo(): Promise<ApiInfo> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/api/v1/info`);
  } catch {
    throw new ApiError(
      "BACKEND_UNAVAILABLE",
      "Cannot reach the colorization service. Make sure the backend is running.",
    );
  }
  if (!response.ok) throw parseError(response);
  return response.json() as Promise<ApiInfo>;
}

export async function getModelStatus(): Promise<ModelStatus> {
  const response = await fetch(`${API_BASE_URL}/api/v1/model-status`);
  if (!response.ok) throw parseError(response);
  return response.json() as Promise<ModelStatus>;
}

/**
 * Upload an image and settings to the colorization endpoint.
 * Uses XMLHttpRequest so real upload progress events are available.
 */
export function colorizeImage(
  file: File,
  settings: ColorizeSettings,
  onUploadProgress?: (percent: number) => void,
  signal?: AbortSignal,
): Promise<ColorizeResponse> {
  const form = new FormData();
  form.append("image", file, file.name);
  form.append("quality", settings.quality);
  form.append("preserve_contrast", String(settings.preserveContrast));
  form.append("face_enhancement", String(settings.faceEnhancement));
  form.append("denoise", String(settings.denoise));
  form.append("output_format", settings.outputFormat);

  return new Promise<ColorizeResponse>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE_URL}/api/v1/colorize`);
    xhr.responseType = "text";

    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable && onUploadProgress) {
        onUploadProgress(Math.round((event.loaded / event.total) * 100));
      }
    };

    xhr.onload = () => {
      let body: ColorizeResponse & JsonErrorBody;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        reject(
          new ApiError(
            "INVALID_RESPONSE",
            "The server returned a malformed response. Please try again.",
          ),
        );
        return;
      }
      if (xhr.status >= 200 && xhr.status < 300 && body.success) {
        resolve(body as ColorizeResponse);
      } else {
        reject(
          new ApiError(
            body?.error?.code ?? "SERVER_ERROR",
            body?.error?.message ??
              `Colorization failed (HTTP ${xhr.status}). Please try again.`,
            body?.error?.request_id,
          ),
        );
      }
    };

    xhr.onerror = () =>
      reject(
        new ApiError(
          "NETWORK_ERROR",
          "Network failure while contacting the colorization service. Check your connection and that the backend is running.",
        ),
      );

    xhr.ontimeout = () =>
      reject(
        new ApiError(
          "TIMEOUT",
          "The request timed out. The image may be too large for the current timeout setting.",
        ),
      );

    xhr.onabort = () =>
      reject(new ApiError("ABORTED", "The request was cancelled."));

    if (signal) {
      signal.addEventListener("abort", () => xhr.abort());
    }

    xhr.send(form);
  });
}
