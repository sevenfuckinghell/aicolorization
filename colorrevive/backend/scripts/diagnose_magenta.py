"""Diagnostic script to isolate root cause of magenta/pink color cast on 1308x736 image.

Compares:
1. Official DDColor pipeline (paper & modelscope)
2. App raw DDColor before refinement (paper & modelscope)
3. App final refined DDColor output (High Quality mode)
Inspects a, b channel distributions, min/max/mean/median, and produces visual outputs.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from app.ml.ddcolor import DDColor
from app.ml.ddcolor_engine import DDColorEngine, DDColorHF
from app.ml.ddcolor.refinement import refine_chrominance


def official_ddcolor_predict(model: torch.nn.Module, img_bgr: np.ndarray, input_size: int = 512, device="cpu") -> tuple[np.ndarray, np.ndarray]:
    """Exact official DDColor ColorizationPipeline implementation."""
    height, width = img_bgr.shape[:2]
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

    # Official uses default F.interpolate (nearest)
    output_ab_resized = (
        F.interpolate(output_ab, size=(height, width))[0]
        .float()
        .numpy()
        .transpose(1, 2, 0)
    )
    output_lab = np.concatenate((orig_l, output_ab_resized), axis=-1)
    output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)
    output_img = (output_bgr * 255.0).round().clip(0, 255).astype(np.uint8)
    return output_img, output_ab_resized


def app_raw_ddcolor_predict(model: torch.nn.Module, img_bgr: np.ndarray, input_size: int = 512, device="cpu") -> tuple[np.ndarray, np.ndarray]:
    """App raw DDColor pipeline before any refinement (with bilinear ab interpolation)."""
    height, width = img_bgr.shape[:2]
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

    output_ab_resized = (
        F.interpolate(output_ab, size=(height, width), mode="bilinear", align_corners=False)[0]
        .float()
        .numpy()
        .transpose(1, 2, 0)
    )
    output_lab = np.concatenate((orig_l, output_ab_resized), axis=-1)
    output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)
    output_img = (output_bgr * 255.0).round().clip(0, 255).astype(np.uint8)
    return output_img, output_ab_resized


def app_refined_predict(orig_l: np.ndarray, output_ab_resized: np.ndarray, quality: str = "high") -> np.ndarray:
    """App refinement stage."""
    refined_ab, _ = refine_chrominance(
        orig_l=orig_l,
        ab=output_ab_resized,
        quality=quality,
        edge_refinement=True,
        chroma_strength=1.0,
        black_preserve=True,
    )
    output_lab = np.concatenate((orig_l, refined_ab), axis=-1)
    output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)
    output_img = (output_bgr * 255.0).round().clip(0, 255).astype(np.uint8)
    return output_img


def analyze_chroma(name: str, ab: np.ndarray):
    a = ab[:, :, 0]
    b = ab[:, :, 1]
    # In Lab:
    # +a is magenta/red, -a is green
    # +b is yellow, -b is blue
    print(f"[{name}]")
    print(f"  Channel 'a' (red/magenta vs green): min={a.min():.2f}, max={a.max():.2f}, mean={a.mean():.2f}, median={np.median(a):.2f}, >10px ratio={(a > 10).mean():.3f}")
    print(f"  Channel 'b' (yellow vs blue):       min={b.min():.2f}, max={b.max():.2f}, mean={b.mean():.2f}, median={np.median(b):.2f}")


def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    input_path = repo_root / "sample-data" / "user_1308x736.jpg"
    artifact_dir = Path("C:/Users/pc/.gemini/antigravity/brain/e2f002de-06dd-4efa-9d4a-312c98d57c12")

    if not input_path.exists():
        print(f"Error: {input_path} not found")
        return

    img_bgr = cv2.imread(str(input_path))
    print(f"Input image loaded: {img_bgr.shape} (H={img_bgr.shape[0]}, W={img_bgr.shape[1]})")

    # If the image has color or is grayscale, ensure we treat it as grayscale test input
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_input = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    device = torch.device("cpu")
    print("\n--- Loading ddcolor_paper ---")
    paper_model = DDColorHF.from_pretrained("piddnad/ddcolor_paper").to(device)
    paper_model.eval()

    print("\n--- Loading ddcolor_modelscope ---")
    scope_model = DDColorHF.from_pretrained("piddnad/ddcolor_modelscope").to(device)
    scope_model.eval()

    img_f32 = (img_input / 255.0).astype(np.float32)
    orig_l = cv2.cvtColor(img_f32, cv2.COLOR_BGR2Lab)[:, :, :1]

    # 1. Evaluate PAPER variant
    print("\n================== DDColor PAPER (input_size=768) ==================")
    t0 = time.perf_counter()
    paper_off_img, paper_off_ab = official_ddcolor_predict(paper_model, img_input, input_size=768, device=device)
    t_off_paper = time.perf_counter() - t0
    print(f"Official pipeline time: {t_off_paper:.2f}s")
    analyze_chroma("Paper Official", paper_off_ab)

    t0 = time.perf_counter()
    paper_raw_img, paper_raw_ab = app_raw_ddcolor_predict(paper_model, img_input, input_size=768, device=device)
    t_raw_paper = time.perf_counter() - t0
    print(f"App Raw pipeline time: {t_raw_paper:.2f}s")
    analyze_chroma("Paper App Raw", paper_raw_ab)

    t0 = time.perf_counter()
    paper_ref_img = app_refined_predict(orig_l, paper_raw_ab, quality="high")
    t_ref_paper = time.perf_counter() - t0
    print(f"App Refined (High) time: {t_ref_paper:.2f}s")

    # 2. Evaluate PAPER at input_size=512
    print("\n================== DDColor PAPER (input_size=512) ==================")
    paper512_off_img, paper512_off_ab = official_ddcolor_predict(paper_model, img_input, input_size=512, device=device)
    analyze_chroma("Paper 512 Official", paper512_off_ab)

    # 3. Evaluate MODELSCOPE variant (input_size=512)
    print("\n================== DDColor MODELSCOPE (input_size=512) =============")
    t0 = time.perf_counter()
    scope_off_img, scope_off_ab = official_ddcolor_predict(scope_model, img_input, input_size=512, device=device)
    t_off_scope = time.perf_counter() - t0
    print(f"Scope Official pipeline time: {t_off_scope:.2f}s")
    analyze_chroma("Modelscope Official", scope_off_ab)

    t0 = time.perf_counter()
    scope_raw_img, scope_raw_ab = app_raw_ddcolor_predict(scope_model, img_input, input_size=512, device=device)
    t_raw_scope = time.perf_counter() - t0
    print(f"Scope App Raw pipeline time: {t_raw_scope:.2f}s")
    analyze_chroma("Modelscope App Raw", scope_raw_ab)

    t0 = time.perf_counter()
    scope_ref_img = app_refined_predict(orig_l, scope_raw_ab, quality="high")
    t_ref_scope = time.perf_counter() - t0
    print(f"Scope App Refined (High) time: {t_ref_scope:.2f}s")

    # Save images to artifact directory
    cv2.imwrite(str(artifact_dir / "user_input_gray.png"), img_input)
    cv2.imwrite(str(artifact_dir / "triplet_A_official_paper_768.png"), paper_off_img)
    cv2.imwrite(str(artifact_dir / "triplet_B_app_raw_paper_768.png"), paper_raw_img)
    cv2.imwrite(str(artifact_dir / "triplet_C_app_refined_paper_768.png"), paper_ref_img)
    cv2.imwrite(str(artifact_dir / "scope_official_512.png"), scope_off_img)
    cv2.imwrite(str(artifact_dir / "scope_raw_512.png"), scope_raw_img)
    cv2.imwrite(str(artifact_dir / "scope_refined_512.png"), scope_ref_img)

    # Create visual comparison panels
    h, w = img_input.shape[:2]
    panel_h = 360
    panel_w = int(w * (panel_h / h))

    def prep(im, label):
        r = cv2.resize(im, (panel_w, panel_h))
        cv2.putText(r, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(r, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        return r

    panel_paper = np.hstack([
        prep(img_input, "Grayscale Input"),
        prep(paper_off_img, "A: Official Paper 768"),
        prep(paper_raw_img, "B: App Raw Paper 768"),
        prep(paper_ref_img, "C: App Refined Paper 768"),
    ])
    cv2.imwrite(str(artifact_dir / "diagnostic_triplet_paper_panel.png"), panel_paper)

    panel_comparison = np.hstack([
        prep(img_input, "Grayscale Input"),
        prep(paper_off_img, "Paper (768px)"),
        prep(scope_off_img, "ModelScope (512px)"),
        prep(scope_ref_img, "ModelScope Refined"),
    ])
    cv2.imwrite(str(artifact_dir / "diagnostic_paper_vs_modelscope_panel.png"), panel_comparison)

    print("\nDiagnostic triplet and comparison panels written to artifact directory successfully.")


if __name__ == "__main__":
    main()
