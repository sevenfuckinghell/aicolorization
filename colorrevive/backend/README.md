# ColorRevive Backend (FastAPI + PyTorch)

Image validation, Lab-space preprocessing, U-Net colorization inference with a
deterministic fallback mode, and PNG/JPEG encoding.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

- Swagger UI: http://localhost:8000/docs
- Health: `curl localhost:8000/health` → reports `model_loaded` / `fallback_mode` dynamically.

## Configuration

All settings come from environment variables (see `../.env.example`); copy it to
`.env` in this directory to override defaults. Key ones:

| Var | Default | Meaning |
|---|---|---|
| `MODEL_CHECKPOINT_PATH` | `./checkpoints/colorization.pt` | Trained U-Net checkpoint; missing ⇒ fallback |
| `DEVICE` | `auto` | `auto`/`cpu`/`cuda` |
| `MAX_UPLOAD_MB` | `10` | Hard upload cap |
| `MAX_IMAGE_PIXELS` | `25000000` | Decompression-bomb guard |
| `INFERENCE_TIMEOUT_SECONDS` | `60` | Per-request inference timeout |
| `RESPONSE_MODE` | `base64` | `base64` JSON or raw `binary` response |
| `ENABLE_PERSISTENT_STORAGE` | `false` | Keep outputs after response (off by default for privacy) |

## Endpoints

`GET /health` · `GET /api/v1/info` · `GET /api/v1/model-status` ·
`POST /api/v1/colorize` · `POST /api/v1/validate-image` — full reference in
[`../docs/api.md`](../docs/api.md).

## Tests

```bash
pip install pytest httpx
python -m pytest -q          # 28 tests: API, errors, preprocessing shapes/ranges, model, cleanup
```

## Layout

```
app/
  main.py        # FastAPI factory, CORS, lifespan (model loaded once at startup)
  config.py      # pydantic-settings env config
  schemas.py     # request/response models + error taxonomy
  api/           # route handlers (thin; no business logic)
  services/      # colorization orchestration, image safety, storage/temp cleanup
  ml/            # model.py (U-Net), inference.py, preprocessing/postprocessing (Lab),
                 # checkpoint_loader.py (torch.load with weights_only-safe path checks)
  utils/         # structured logging (request IDs, no image bytes), error envelope
```

## Safety notes

- Filenames from uploads are never trusted; server-side UUIDs are used.
- Temp files are deleted after the response (`retention_seconds=0` default).
- Inference runs under `torch.no_grad()` + `eval()` behind a lock so CPU
  requests queue instead of thrashing.
- Logs contain request IDs, dimensions, timings, and error categories only —
  never pixel data.
