"""Generate comprehensive 5-panel visual benchmark and whisker-detail zoom for cheetah.jpg."""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
import torch

# Ensure app package is importable
backend_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_root))

from app.config import get_settings
from app.ml.inference import get_colorization_model
from app.ml.detail_enhancement import apply_detail_sharpening
from app.ml.color_grading import apply_color_grading

ARTIFACT_DIR = Path(r"C:\Users\pc\.gemini\antigravity\brain\e2f002de-06dd-4efa-9d4a-312c98d57c12")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = backend_root.parent / "sample-data" / "cheetah.jpg"


def compute_metrics(bgr: np.ndarray) -> dict[str, float]:
    """Compute photographic sharpness, chroma, and contrast metrics."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    L = lab[..., 0] * (100.0 / 255.0)
    a = lab[..., 1] - 128.0
    b = lab[..., 2] - 128.0
    chroma = np.sqrt(a**2 + b**2)

    return {
        "sharpness_laplacian": round(lap_var, 1),
        "mean_chroma": round(float(np.mean(chroma)), 1),
        "max_chroma": round(float(np.max(chroma)), 1),
        "contrast_std": round(float(np.std(L)), 1),
    }


def add_label_bar(img: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    """Add professional dark title bar above image."""
    h, w, c = img.shape
    bar_h = 75
    bar = np.full((bar_h, w, 3), 24, dtype=np.uint8)
    
    cv2.putText(bar, title, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(bar, subtitle, (14, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 210, 240), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def run_benchmark():
    assert INPUT_PATH.exists(), f"Input path does not exist: {INPUT_PATH}"
    print(f"Loading input: {INPUT_PATH}")
    img_bgr = cv2.imread(str(INPUT_PATH))
    h, w = img_bgr.shape[:2]
    print(f"Dimensions: {w}x{h}")

    # Convert to pure grayscale RGB
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    input_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    settings = get_settings()
    model = get_colorization_model(settings)
    engine = model._engine

    # Ensure engine has model loaded
    if not engine.is_loaded():
        engine.load(model_variant=settings.ddcolor_model_variant, device="cpu")

    results = {}

    # Variant A: Grayscale Input
    results["A"] = {
        "title": "A: Grayscale Input",
        "subtitle": "Original Monochrome Reference",
        "bgr": cv2.cvtColor(input_rgb, cv2.COLOR_RGB2BGR),
    }

    # Variant B: Raw DDColor Baseline
    print("Generating Variant B: Raw DDColor Baseline...")
    b_rgb = engine.predict(
        input_rgb,
        quality="standard",
        chroma_strength=1.0,
        black_preserve=False,
        edge_refinement=False,
        color_grading="original_ai",
        sharpening=False,
    )
    results["B"] = {
        "title": "B: Raw DDColor Baseline",
        "subtitle": "No Grading | No Sharpening",
        "bgr": cv2.cvtColor(b_rgb, cv2.COLOR_RGB2BGR),
    }

    # Variant C: Natural Color Grading Only
    print("Generating Variant C: Natural Color Grading Only...")
    c_rgb = engine.predict(
        input_rgb,
        quality="standard",
        chroma_strength=1.0,
        black_preserve=True,
        edge_refinement=True,
        color_grading="natural",
        sharpening=False,
    )
    results["C"] = {
        "title": "C: Natural Grading Only",
        "subtitle": "Natural Preset | Sharpening OFF",
        "bgr": cv2.cvtColor(c_rgb, cv2.COLOR_RGB2BGR),
    }

    # Variant D: Natural Grading + Sharpening (Recommended Default)
    print("Generating Variant D: Natural Grading + Sharpening...")
    d_rgb = engine.predict(
        input_rgb,
        quality="standard",
        chroma_strength=1.0,
        black_preserve=True,
        edge_refinement=True,
        color_grading="natural",
        sharpening=True,
        sharpening_strength=0.35,
        sharpening_radius=1.0,
        sharpening_threshold=3.0,
    )
    results["D"] = {
        "title": "D: Natural + Sharpening",
        "subtitle": "Natural | Strength 0.35 (Default)",
        "bgr": cv2.cvtColor(d_rgb, cv2.COLOR_RGB2BGR),
    }

    # Variant E: Maximum Quality
    print("Generating Variant E: Maximum Quality...")
    e_rgb = engine.predict(
        input_rgb,
        quality="high",
        chroma_strength=1.0,
        black_preserve=True,
        edge_refinement=True,
        color_grading="natural",
        sharpening=True,
        sharpening_strength=0.45,
        sharpening_radius=1.0,
        sharpening_threshold=2.5,
    )
    results["E"] = {
        "title": "E: Maximum Quality",
        "subtitle": "Strength 0.45 | High-Res 768px",
        "bgr": cv2.cvtColor(e_rgb, cv2.COLOR_RGB2BGR),
    }

    # Save individual outputs & calculate metrics
    metrics_summary = {}
    labeled_panels = []
    zoom_panels = []

    # Coordinates for zoom crop on cheetah's face and whiskers
    # Cheetah image is 672 x 456. Head is approximately around x: 190..450, y: 70..310
    crop_y1, crop_y2 = int(h * 0.15), int(h * 0.68)
    crop_x1, crop_x2 = int(w * 0.28), int(w * 0.67)

    for key, data in results.items():
        bgr = data["bgr"]
        m = compute_metrics(bgr)
        metrics_summary[key] = m
        print(f"Metrics {key} ({data['title']}): {m}")

        # Save single image
        single_path = ARTIFACT_DIR / f"cheetah_variant_{key}.png"
        cv2.imwrite(str(single_path), bgr)

        # Labeled panel
        sub_text = f"{data['subtitle']} | Sharpness: {m['sharpness_laplacian']} | Chroma: {m['mean_chroma']}"
        labeled = add_label_bar(bgr, data["title"], sub_text)
        labeled_panels.append(labeled)

        # Whisker zoom crop
        crop = bgr[crop_y1:crop_y2, crop_x1:crop_x2]
        crop_zoom = cv2.resize(crop, (400, 400), interpolation=cv2.INTER_CUBIC)
        labeled_zoom = add_label_bar(crop_zoom, data["title"], f"Sharpness: {m['sharpness_laplacian']}")
        zoom_panels.append(labeled_zoom)

    # Combine full 5-panel comparison (stacked horizontally or 2 rows)
    # 5 across might be 5 * 672 = 3360 px wide.
    # Let's create a 5-column horizontal strip and a 2x3 grid for easy viewing
    strip_5 = np.hstack(labeled_panels)
    strip_path = ARTIFACT_DIR / "cheetah_detail_comparison_panel.png"
    cv2.imwrite(str(strip_path), strip_5)
    print(f"Saved full comparison panel: {strip_path}")

    # Whisker zoom panel (5 columns across = 5 * 400 = 2000 px wide)
    zoom_strip = np.hstack(zoom_panels)
    zoom_path = ARTIFACT_DIR / "cheetah_whiskers_zoom_panel.png"
    cv2.imwrite(str(zoom_path), zoom_strip)
    print(f"Saved whiskers zoom panel: {zoom_path}")

    return metrics_summary


if __name__ == "__main__":
    run_benchmark()
