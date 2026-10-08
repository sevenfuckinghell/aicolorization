# Architecture

ColorRevive separates three concerns: an interactive Next.js web client, a high-performance FastAPI inference backend, and an offline training/evaluation pipeline.

## System Overview

```mermaid
flowchart LR
    U[User Browser] --> F[Next.js Frontend<br/>React + Tailwind + TS]
    F -- "POST /api/v1/colorize" --> A[FastAPI Backend<br/>Uvicorn + Pydantic]
    A --> S[Colorization Service<br/>validation · lifecycle · timeout]
    S --> E[DDColor Deep-Learning Engine<br/>PyTorch · ConvNeXt + Dual Decoders]
    E --> H[Hugging Face / Cached Model<br/>piddnad/ddcolor_modelscope]
    E --> P[Postprocessing & Luminance Fusion<br/>Native Resolution Lab Restoration]
    P --> R[Colorized PNG/JPEG Output]
    R --> F
```

## Request Lifecycle

```mermaid
sequenceDiagram
    participant B as Browser
    participant N as Next.js Client
    participant F as FastAPI Backend
    participant S as ColorizationService
    participant E as DDColorEngine (PyTorch)

    B->>N: Drop image & configure settings (Standard / High)
    N->>F: POST /api/v1/colorize (multipart/form-data)
    F->>S: Validate MIME, size, decompression bounds, decode
    S->>E: run_colorization(rgb_u8, quality="standard"|"high")
    Note over E: ConvNeXt feature extraction + MultiScale Dual Decoder
    E->>E: Predict chrominance (a, b) conditioned on luminance (L)
    E->>E: Fuse predicted ab with original native-resolution L
    E-->>S: Restored RGB at exact original dimensions
    S->>S: Optional conservative enhancements (denoise / contrast)
    S->>S: Encode to PNG / JPEG
    S-->>F: ColorizeOutput (bytes, timing, model, dimensions)
    F-->>N: JSON { success: true, model: "DDColor", model_variant: "...", image_base64: "..." }
    N->>B: Render comparison slider, side-by-side previews, downloads
```

## Component Responsibilities

| Layer | Code | Responsibility |
|---|---|---|
| **Frontend** | `frontend/app`, `components/`, `lib/` | Image upload, validation, quality settings, comparison slider, history, model status indicators |
| **Backend API** | `backend/app/api/` | HTTP endpoints (`/colorize`, `/health`, `/info`, `/model-status`, `/validate-image`), error taxonomy |
| **Pipeline Service** | `backend/app/services/` | Image decoding, thread-pool isolation, timeout enforcement, output encoding, cleanup |
| **ML Engine** | `backend/app/ml/ddcolor_engine.py` | Model caching, device allocation (CUDA/CPU), inference execution, dimension fidelity |
| **Model Arch** | `backend/app/ml/ddcolor/` | Pure-PyTorch implementation of DDColor dual-decoder neural network |

## Key Design Decisions

- **Pretrained Deep-Learning Model:** Replaces heuristic fallbacks with the ICCV 2023 DDColor model (`piddnad/ddcolor_modelscope`), providing genuine semantic coloring (skin, hair, foliage, sky, clothing).
- **Original Native Luminance Preservation:** The model predicts chrominance (`a`, `b`) in CIE Lab color space. During reconstruction, predicted chrominance is combined with the original high-resolution `L` channel, ensuring zero loss of sharpness or photographic texture.
- **Single Process-Level Singleton:** The neural network is initialized and warmed up exactly once at application startup. Subsequent inference calls reuse the resident model in memory.
- **Fail-Safe Integrity:** Deterministic fallback is disabled by default (`ENABLE_FALLBACK_MODE=false`). If weights cannot be loaded, the backend returns a clear `MODEL_UNAVAILABLE` error instead of misleading the user with heuristic tints.
