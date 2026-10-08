/** Shared type definitions for the ColorRevive frontend. */

export type Quality = "standard" | "high";
export type OutputFormat = "png" | "jpeg";

export interface ColorizeSettings {
  quality: Quality;
  preserveContrast: boolean;
  faceEnhancement: boolean;
  denoise: boolean;
  outputFormat: OutputFormat;
  chromaStrength?: number;
}

export const DEFAULT_SETTINGS: ColorizeSettings = {
  quality: "standard",
  preserveContrast: true,
  faceEnhancement: false,
  denoise: false,
  outputFormat: "png",
  chromaStrength: 1.0,
};

/** Response of POST /api/v1/colorize */
export interface ColorizeResponse {
  success: boolean;
  request_id: string;
  filename: string;
  mime_type: string;
  width: number;
  height: number;
  processing_time_ms: number;
  model: string;
  model_variant?: string;
  device?: string;
  fallback_mode: boolean;
  image_base64: string;
}

/** Response of GET /health */
export interface HealthResponse {
  status: string;
  service: string;
  model_name?: string;
  model_variant?: string;
  model_loaded: boolean;
  device: string;
  fallback_mode?: boolean;
}

/** Response of GET /api/v1/info */
export interface ApiInfo {
  name: string;
  api_version: string;
  supported_formats: string[];
  max_upload_mb: number;
  model_name: string;
  model_variant?: string;
  device?: string;
  fallback_mode: boolean;
}

/** Response of GET /api/v1/model-status */
export interface ModelStatus {
  model_name: string;
  model_variant?: string;
  loaded: boolean;
  device: string;
  precision: string;
  version: string;
  fallback_mode: boolean;
}

export interface UploadFileInfo {
  name: string;
  sizeBytes: number;
  type: string;
  width: number;
  height: number;
  dataUrl: string;
}

export interface ColorizeResult {
  blobUrl: string;
  base64: string;
  mimeType: string;
  filename: string;
  width: number;
  height: number;
  processingTimeMs: number;
  model: string;
  modelVariant?: string;
  device?: string;
  fallbackMode: boolean;
  requestId: string;
}

/** Structured, user-safe error raised by the API layer. */
export class ApiError extends Error {
  code: string;
  requestId?: string;

  constructor(code: string, message: string, requestId?: string) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.requestId = requestId;
  }
}

export type ProcessingPhase =
  | "idle"
  | "uploading"
  | "preparing"
  | "running-model"
  | "postprocessing"
  | "preparing-download";
