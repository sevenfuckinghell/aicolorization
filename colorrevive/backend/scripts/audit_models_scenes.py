"""Audit official DDColor pipeline and evaluate model variants across diverse scenes.

Generates:
A. Original grayscale input.
B. Official DDColor raw output (mode='nearest' interpolation, no post-processing).
C. Application raw DDColor output (mode='bilinear' interpolation, no post-processing).
D. Application final output with conservative scene-aware grading and refinement.

Across:
- piddnad/ddcolor_modelscope
- piddnad/ddcolor_paper
- piddnad/ddcolor_artistic
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F

backend_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_root))

from app.config import get_settings
from app.ml.inference import get_colorization_model
from app.ml.ddcolor_engine import DDColorEngine
from app.ml.ddcolor.pipeline import ColorizationPipeline

SAMPLE_DIR = backend_root.parent / "sample-data"
ARTIFACT_DIR = Path(r"C:\Users\pc\.gemini\antigravity\brain\e2f002de-06dd-4efa-9d4a-312c98d57c12")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = [
    "piddnad/ddcolor_modelscope",
    "piddnad/ddcolor_paper",
    "piddnad/ddcolor_artistic",
]

TEST_IMAGES = {
    "car": SAMPLE_DIR / "vintage_car.jpg",
    "sunset": SAMPLE_DIR / "tropical_sunset.jpg",
    "cheetah": SAMPLE_DIR / "cheetah.jpg",
    "lincoln": SAMPLE_DIR / "lincoln-bw.jpg",
    "building": SAMPLE_DIR / "building-bw.jpg",
}


def official_ddcolor_raw_predict(model, img_bgr: np.ndarray, input_size: int = 512, device="cpu") -> np.ndarray:
    """Exact reproduction of official DDColor repo inference.py."""
    h, w = img_bgr.shape[:2]
    img = (img_bgr / 255.0).astype(np.float32)
    orig_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]

    img_resized = cv2.resize(img, (input_size, input_size))
    img_l = cv2.cvtColor(img_resized, cv2.COLOR_BGR2Lab)[:, :, :1]
    img_gray_lab = np.concatenate((img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1)
    img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)

    tensor_gray_rgb = (
        torch.from_numpy(img_gray_rgb.transpose((2, 0, 1)))
        .float()
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        output_ab = model(tensor_gray_rgb).cpu()

    # Official nearest interpolation
    output_ab_resize = F.interpolate(output_ab, size=(h, w))[0].float().numpy().transpose(1, 2, 0)
    output_lab = np.concatenate((orig_l, output_ab_resize), axis=-1)
    output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)
    return (output_bgr * 255.0).round().clip(0, 255).astype(np.uint8)


def app_raw_predict(model, img_bgr: np.ndarray, input_size: int = 512, device="cpu") -> np.ndarray:
    """Application raw prediction: bilinear interpolation of ab, strictly no post-processing."""
    h, w = img_bgr.shape[:2]
    img = (img_bgr / 255.0).astype(np.float32)
    orig_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]

    img_resized = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_AREA)
    img_l = cv2.cvtColor(img_resized, cv2.COLOR_BGR2Lab)[:, :, :1]
    img_gray_lab = np.concatenate((img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1)
    img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)

    tensor_gray_rgb = (
        torch.from_numpy(img_gray_rgb.transpose((2, 0, 1)))
        .float()
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        output_ab = model(tensor_gray_rgb).cpu()

    output_ab_resize = (
        F.interpolate(output_ab, size=(h, w), mode="bilinear", align_corners=False)[0]
        .float()
        .numpy()
        .transpose(1, 2, 0)
    )
    output_lab = np.concatenate((orig_l, output_ab_resize), axis=-1)
    output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)
    return (output_bgr * 255.0).round().clip(0, 255).astype(np.uint8)


def add_label(img: np.ndarray, label: str) -> np.ndarray:
    h, w = img.shape[:2]
    bar = np.full((36, w, 3), 20, dtype=np.uint8)
    cv2.putText(bar, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def run_audit():
    settings = get_settings()
    engine = DDColorEngine(device_name="cpu")

    # Ensure grayscale inputs for test images
    inputs_bgr = {}
    for name, path in TEST_IMAGES.items():
        assert path.exists(), f"Missing {path}"
        raw = cv2.imread(str(path))
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        inputs_bgr[name] = gray_bgr

    # Save original grayscale inputs
    for name, gray_bgr in inputs_bgr.items():
        cv2.imwrite(str(ARTIFACT_DIR / f"test_{name}_A_gray.png"), gray_bgr)

    # Evaluate each variant
    variant_results = {}
    for variant in VARIANTS:
        v_tag = variant.split("/")[-1].replace("ddcolor_", "")
        print(f"\n================ Loading {variant} ({v_tag}) ================")
        model, _ = engine.get_or_load_model(variant)
        inp_size = 768 if "paper" in variant else 512

        variant_results[v_tag] = {}
        for name, gray_bgr in inputs_bgr.items():
            print(f"  Processing {name} on {v_tag}...")
            # B: Official Raw
            b_official = official_ddcolor_raw_predict(model, gray_bgr, input_size=inp_size, device="cpu")
            # C: App Raw
            c_app_raw = app_raw_predict(model, gray_bgr, input_size=inp_size, device="cpu")
            # D: App Refined + Natural Grade
            pipe = ColorizationPipeline(model, input_size=inp_size, device="cpu")
            d_app_natural, _ = pipe.process(
                gray_bgr,
                quality="standard" if "paper" not in variant else "high",
                color_grading="natural",
                sharpening=False,
                edge_refinement=True,
                black_preserve=True,
                chroma_strength=1.0,
            )

            variant_results[v_tag][name] = {
                "official": b_official,
                "app_raw": c_app_raw,
                "app_natural": d_app_natural,
            }

            cv2.imwrite(str(ARTIFACT_DIR / f"{name}_{v_tag}_B_official.png"), b_official)
            cv2.imwrite(str(ARTIFACT_DIR / f"{name}_{v_tag}_C_app_raw.png"), c_app_raw)
            cv2.imwrite(str(ARTIFACT_DIR / f"{name}_{v_tag}_D_app_natural.png"), d_app_natural)

    # Generate composite comparison panels for each test scene:
    # Row 1: Grayscale, ModelScope Official, Paper Official, Artistic Official
    # Row 2: Grayscale, ModelScope Natural, Paper Natural, Artistic Natural
    for name, gray_bgr in inputs_bgr.items():
        h, w = gray_bgr.shape[:2]
        # Thumbnail max width 400 for panel display
        tw = 380
        th = int(h * (tw / w))

        def prep(img, lbl):
            r = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
            return add_label(r, lbl)

        row_raw = np.hstack([
            prep(gray_bgr, "A: Grayscale Input"),
            prep(variant_results["modelscope"][name]["official"], "B1: ModelScope Raw"),
            prep(variant_results["paper"][name]["official"], "B2: Paper Raw"),
            prep(variant_results["artistic"][name]["official"], "B3: Artistic Raw"),
        ])

        row_nat = np.hstack([
            prep(gray_bgr, "A: Grayscale Input"),
            prep(variant_results["modelscope"][name]["app_natural"], "D1: ModelScope Nat"),
            prep(variant_results["paper"][name]["app_natural"], "D2: Paper Nat"),
            prep(variant_results["artistic"][name]["app_natural"], "D3: Artistic Nat"),
        ])

        panel = np.vstack([row_raw, row_nat])
        panel_path = ARTIFACT_DIR / f"scene_audit_{name}_comparison_panel.png"
        cv2.imwrite(str(panel_path), panel)
        print(f"Saved scene comparison panel: {panel_path}")

    print("\nAudit completed successfully.")


if __name__ == "__main__":
    run_audit()
