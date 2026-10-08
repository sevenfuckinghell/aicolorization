"""ColorRevive FastAPI application entry point.

Run locally:  uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api import colorize as colorize_api
from .api import health as health_api
from .config import get_settings
from .ml.inference import get_colorization_model
from .utils.logging import get_logger, log_fields, request_id_var, setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = app.state.settings
    setup_logging(settings.log_level)
    logger = get_logger("colorrevive.startup")
    # Load the model exactly once at startup (never per request).
    model = get_colorization_model(settings)
    app.state.model = model
    meta = model.metadata()
    log_fields(
        logger, "info", "startup complete",
        device=meta["device"], loaded=meta["loaded"],
        fallback_mode=meta["fallback_mode"], model=meta["model_name"],
    )
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="ColorRevive API",
        version=settings.api_version,
        description="Black-and-white photo colorization service powered by pretrained DDColor deep-learning model (ICCV 2023).",
        lifespan=lifespan,
    )
    app.state.settings = settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    app.include_router(health_api.router)
    app.include_router(colorize_api.router)

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        rid = request.headers.get("x-request-id") or str(uuid.uuid4())
        token = request_id_var.set(rid)
        response = await call_next(request)
        response.headers["X-Request-Id"] = rid
        request_id_var.reset(token)
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        # Map FastAPI validation failures onto the same public error shape,
        # without echoing raw input values back to the client.
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": {
                    "code": "MISSING_FIELDS",
                    "message": "Required form fields (image, quality, output_format) are missing or invalid.",
                    "request_id": request_id_var.get(),
                },
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger = get_logger("colorrevive.errors")
        # Full diagnostics stay server-side; the client sees a safe message.
        log_fields(
            logger, "error", "unhandled server error",
            endpoint=request.url.path, error_class=type(exc).__name__,
        )
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": {
                    "code": "SERVER_ERROR",
                    "message": "An unexpected internal error occurred. Please try again.",
                    "request_id": request_id_var.get(),
                },
            },
        )

    return app


app = create_app()
