# ColorRevive Backend (FastAPI + PyTorch + DDColor)

Production image colorization service powered by the pretrained **DDColor** deep-learning model (ICCV 2023). Converts black-and-white or historical grayscale photos into realistic, semantically rich color images while preserving original luminance and native image dimensions.

---

## Key Features

- **Pretrained DDColor Model**: ConvNeXt-Large backbone + Dual Pixel & Color Decoders trained on diverse real-world photographic datasets.
- **Semantic Color Prediction**: Predicts natural skin tones, hair colors, sky blues, foliage greens, and clothing textures — no heuristic tinting or lookup tables.
- **Native Quality Preservation**: Original input resolution is completely preserved (e.g. 600×900 in → 600×900 out); original luminance ($L$-channel) is combined with predicted chrominance ($a, b$) in CIE Lab space.
- **Hardware Acceleration**: Automatic GPU detection (`DEVICE=auto`), explicit CUDA execution (`DEVICE=cuda`), or optimized CPU inference (`DEVICE=cpu`).
- **Automatic Model Download & Caching**: Hugging Face Hub integration downloads weights on first run and caches them locally (`~/.cache/huggingface/hub`).
- **No Fake Fallback**: `ENABLE_FALLBACK_MODE=false` by default. Returns clear `MODEL_UNAVAILABLE` error if weights are missing rather than deceptive heuristic output.
- **Thread-Safe Model Singleton**: Model loaded and warmed up once at application startup; guarded inference lock for concurrent safety.

---

## Run Locally

### 1. Set Up Python Environment

```bash
# Windows (PowerShell)
cd colorrevive/backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Linux / macOS
cd colorrevive/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Launch the API Server

```bash
# Default port 8000 (or --port 8001 if 8000 is occupied)
uvicorn app.main:app --host 127.0.0.1 --port 8001
```

On first startup, the server automatically downloads `piddnad/ddcolor_modelscope` (~500 MB) from Hugging Face Hub, loads it onto the detected device (CUDA or CPU), and runs a warmup pass.

- **Swagger UI**: http://127.0.0.1:8001/docs
- **Health Check**: `curl http://127.0.0.1:8001/health`
- **Model Status**: `curl http://127.0.0.1:8001/api/v1/model-status`
- **Service Info**: `curl http://127.0.0.1:8001/api/v1/info`

---

## Configuration (Environment Variables)

Configuration is managed via Pydantic Settings and driven by environment variables:

| Variable | Default | Description |
|---|---|---|
| `COLORIZATION_ENGINE` | `ddcolor` | Colorization engine (`ddcolor`) |
| `DDCOLOR_MODEL` | `piddnad/ddcolor_modelscope` | Model variant: `ddcolor_modelscope` (vibrant), `ddcolor_paper` (balanced natural), `ddcolor_artistic` |
| `DDCOLOR_INPUT_SIZE` | `512` | Model inference resolution (`512` or `768`) |
| `DDCOLOR_INPUT_SIZE_HIGH` | `768` | High quality inference resolution |
| `DDCOLOR_MODEL_DIR` | `./models` | Configurable directory for custom weights |
| `COLOR_CHROMA_STRENGTH` | `1.0` | Chrominance scaling (1.0 = raw prediction, 0.85 = subtle natural) |
| `COLOR_BLACK_PRESERVE` | `true` | Preserves deep shadows and highlights by attenuating chroma at luminance extremes |
| `DEVICE` | `auto` | Device selection: `auto` (CUDA if available, else CPU), `cpu`, `cuda` |
| `ENABLE_FALLBACK_MODE` | `false` | When `false`, returns clear error if AI is unavailable |
| `MAX_UPLOAD_MB` | `10` | Maximum upload file size in megabytes |
| `MAX_IMAGE_PIXELS` | `25000000` | Decompression-bomb safety threshold |
| `INFERENCE_TIMEOUT_SECONDS`| `120` | Maximum seconds allowed per inference request |
| `API_CORS_ORIGINS` | `http://localhost:3000` | Comma-separated list of allowed CORS origins |

---

## Endpoints

### `GET /health`
Returns dynamic server and model readiness:
```json
{
  "status": "ok",
  "service": "colorrevive-api",
  "model_loaded": true,
  "model_name": "DDColor",
  "model_variant": "piddnad/ddcolor_modelscope",
  "device": "cpu",
  "fallback_mode": false
}
```

### `GET /api/v1/model-status`
Detailed neural model diagnostics:
```json
{
  "model_name": "DDColor",
  "model_variant": "piddnad/ddcolor_modelscope",
  "loaded": true,
  "device": "cpu",
  "precision": "float32",
  "version": "ICCV 2023",
  "fallback_mode": false
}
```

### `POST /api/v1/colorize`
Multipart form upload for colorization:
- `image`: Uploaded image file (JPEG, PNG, WEBP)
- `quality`: `"standard"` (512px internal) or `"high"` (768px internal)
- `preserve_contrast`: `"true"` (default)
- `output_format`: `"png"` or `"jpeg"`

Returns:
```json
{
  "success": true,
  "request_id": "936657c9-4673-455b-861f-d27e2ee677bf",
  "filename": "colorrevive-photo-colorized.png",
  "mime_type": "image/png",
  "width": 640,
  "height": 480,
  "processing_time_ms": 1180,
  "model": "DDColor",
  "model_variant": "piddnad/ddcolor_modelscope",
  "device": "cpu",
  "fallback_mode": false,
  "image_base64": "..."
}
```

---

## Docker Deployment

### CPU Container
```bash
docker build -f Dockerfile.cpu -t colorrevive-backend:cpu .
docker run -p 8001:8001 -e DEVICE=cpu colorrevive-backend:cpu
```

### GPU (CUDA) Container
```bash
docker build -f Dockerfile.cuda -t colorrevive-backend:cuda .
docker run --gpus all -p 8001:8001 -e DEVICE=cuda colorrevive-backend:cuda
```

---

## Running Automated Tests

```bash
python -m pytest
```
Runs 27 automated tests covering health checks, model status, DDColor inference, monochrome detection, corrupted image handling, size caps, and error taxonomy.
