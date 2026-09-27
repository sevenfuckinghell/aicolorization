# Architecture

ColorRevive separates three concerns: an interactive web client, a stateless
inference API, and an offline training pipeline. They communicate only through
HTTP and checkpoint files.

## System overview

```mermaid
flowchart LR
    U[User] --> F[Next.js Frontend]
    F --> A[FastAPI API]
    A --> P[Image Preprocessing]
    P --> M[PyTorch Colorization Model]
    M --> Q[Postprocessing]
    Q --> R[Colorized Image Response]
    R --> F
```

## Request lifecycle

```mermaid
sequenceDiagram
    participant B as Browser
    participant N as Next.js (client components)
    participant F as FastAPI
    participant S as ColorizationService
    participant M as UNetLab / Fallback

    B->>N: select/drop image (validated client-side)
    N->>F: POST /api/v1/colorize (multipart + settings)
    F->>S: validate MIME, extension, size, pixel count, decode
    S->>S: RGB -> Lab, extract/normalize L channel
    S->>M: predict ab (torch.no_grad, eval mode, inference lock)
    M-->>S: normalized ab
    S->>S: combine L + ab -> Lab -> RGB, resize to original dims
    S->>S: optional contrast preservation / denoise, clamp
    S-->>F: PNG/JPEG bytes + metadata
    F-->>B: JSON {request_id, image_base64, fallback_mode, ...}
    B->>B: render comparison slider, enable download
```

## Component responsibilities

| Layer | Code | Responsibility |
|---|---|---|
| Frontend | `frontend/app`, `components/`, `lib/` | Upload UX, validation, settings, processing states, before/after slider, downloads, model-status banner |
| API | `backend/app/api/` | Routing, multipart parsing, form validation, error envelope, request IDs |
| Services | `backend/app/services/` | Pipeline orchestration, temp-file lifecycle, encoding |
| ML | `backend/app/ml/` | Lab preprocessing, U-Net definition, checkpoint loading (strict shape validation), inference + deterministic fallback |
| Training | `model-training/src/` | Dataset, losses, training loop, evaluation, batch inference — never imported at serve time |

## Key design decisions

- **Lab color space.** The network sees only luminance (L) and predicts
  chrominance (a, b). Lightness stays exact; the model only learns hue/saturation.
- **Model loaded once.** `ColorizationModel.load()` runs at startup (guarded by
  a lock); requests reuse the resident model. A per-model threading lock
  serializes CPU inference.
- **Graceful degradation.** A missing/corrupt/incompatible checkpoint falls back
  to a deterministic heuristic and flips `fallback_mode: true` in every response
  — the UI can never accidentally claim neural results.
- **Swappable architecture.** Checkpoint loading goes through
  `checkpoint_loader.load_checkpoint(path, config)`; adding a ResNet encoder
  means implementing a new builder, not touching the API layer.
- **Response strategy is configurable.** Base64 JSON (default, deployment-free)
  vs. stored-file URLs (`ENABLE_PERSISTENT_STORAGE`) to avoid huge payloads.
- **Training/serving isolation.** `model-training/` mirrors the architecture so
  state dicts are interchangeable, but shares no import-time dependency.

## Deployment topology

```mermaid
flowchart TB
    subgraph Client
      BR[Browser]
    end
    subgraph "Vercel / static host"
      FE[Next.js frontend]
    end
    subgraph "Render / Railway / Fly / AWS"
      BE[Uvicorn + FastAPI<br/>CPU or CUDA]
      CK[(Checkpoint .pt)]
    end
    BR --> FE
    FE -- "NEXT_PUBLIC_API_BASE_URL" --> BE
    BE --> CK
```
