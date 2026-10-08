"""GET /health — dynamic service + model status."""

from __future__ import annotations

from fastapi import APIRouter, Request

from ..schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    model = request.app.state.model
    meta = model.metadata()
    return HealthResponse(
        status="ok",
        service=request.app.state.settings.app_name,
        model_loaded=bool(meta["loaded"]),
        device=str(meta["device"]),
        fallback_mode=bool(meta["fallback_mode"]),
        model_name=str(meta.get("model_name", "DDColor")),
        model_variant=meta.get("model_variant"),
    )
