# ColorRevive API Reference

Base URL (local): `http://localhost:8000` · Interactive docs: `GET /docs` (Swagger UI), `GET /redoc`.

All error responses share one envelope (HTTP 4xx/5xx):

```json
{
  "success": false,
  "error": {
    "code": "INVALID_IMAGE",
    "message": "The uploaded file could not be decoded as a supported image.",
    "request_id": "b3c1..."
  }
}
```

Error codes: `INVALID_FILE_TYPE`, `FILE_TOO_LARGE`, `EMPTY_FILE`, `INVALID_IMAGE`,
`IMAGE_TOO_LARGE` (pixel cap / decompression bomb), `MODEL_UNAVAILABLE`,
`INFERENCE_TIMEOUT`, `INTERNAL_ERROR`.

---

## GET `/health`

Liveness + model state. No auth.

```json
{
  "status": "ok",
  "service": "colorrevive-api",
  "model_loaded": false,
  "device": "cpu",
  "fallback_mode": true
}
```

## GET `/api/v1/info`

Service metadata for the frontend to render limits and mode badges.

```json
{
  "name": "colorrevive-api",
  "api_version": "1.0.0",
  "supported_formats": ["image/jpeg", "image/png", "image/webp"],
  "max_upload_mb": 10,
  "model_name": "unet-lab-v1",
  "fallback_mode": true
}
```

## GET `/api/v1/model-status`

```json
{
  "model_name": "unet-lab-v1",
  "loaded": false,
  "device": "cpu",
  "precision": "fp32",
  "version": "1.0.0",
  "fallback_mode": true
}
```

`loaded: true` means a real PyTorch checkpoint was loaded from
`MODEL_CHECKPOINT_PATH`; otherwise the deterministic fallback is active.

## POST `/api/v1/colorize`

`multipart/form-data`:

| Field | Type | Required | Notes |
|---|---|---|---|
| `image` | file | yes | JPEG/PNG/WEBP, ≤ `MAX_UPLOAD_MB`, ≤ `MAX_IMAGE_PIXELS` |
| `quality` | string | no | `standard` (256 px inference) or `high` (512 px). Default `standard` |
| `preserve_contrast` | bool | no | Keep original L channel instead of model-predicted L. Default `true` |
| `face_enhancement` | bool | no | **Coming soon** — rejected with a clear message if `true` without support |
| `denoise` | bool | no | Mild OpenCV fast-NL denoise before inference. Default `false` |
| `output_format` | string | no | `png` (default) or `jpeg` |

Example:

```bash
curl -s -X POST http://localhost:8000/api/v1/colorize \
  -F "image=@sample-bw-photo.jpg" \
  -F "quality=standard" \
  -F "output_format=png" | jq '{success, model, fallback_mode, width, height}'
```

Response (`RESPONSE_MODE=base64`, default — best for local dev & simple deploys):

```json
{
  "success": true,
  "request_id": "uuid",
  "filename": "colorrevive-sample-colorized.png",
  "mime_type": "image/png",
  "width": 1200,
  "height": 800,
  "processing_time_ms": 842,
  "model": "unet-lab-v1",
  "fallback_mode": false,
  "image_base64": "iVBORw0KGgo..."
}
```

Response (`RESPONSE_MODE=binary`): raw image bytes with `Content-Type: image/png`
and headers `X-Request-Id`, `X-Fallback-Mode`, `X-Processing-Time-Ms`. Recommended
for production behind object storage to avoid base64 overhead (~33 %).

Guarantees: output dimensions equal input dimensions (aspect ratio preserved);
pixels clamped to valid range; temp files deleted after response unless
`ENABLE_PERSISTENT_STORAGE=true`.

## POST `/api/v1/validate-image`

Cheap pre-flight check (type, size, decodability, pixel count) without running
inference. Returns `{ "success": true, "width": ..., "height": ..., "format": ... }`
or the standard error envelope. Useful for client-side UX before enabling the
Colorize button.
