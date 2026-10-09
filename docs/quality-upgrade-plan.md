# Plan: Advanced AI Colorization Quality Upgrade for ColorRevive

## 1. Problem Diagnosis & Audit Findings
1. **Interpolation Flaw**: In `ColorizationPipeline.process()`, `output_ab_resized = F.interpolate(output_ab, size=(height, width))` relied on PyTorch's default `mode='nearest'`. This produced blocky, nearest-neighbor color upsampling that degraded edges, caused jagged color transitions, and bled colors across fine textures.
2. **Color Bleeding Across High-Frequency Boundaries**: Standard neural colorization predicts chrominance ($ab$) at low spatial resolution (512x512 or 768x768). When resizing chrominance back to high-resolution photographs, linear upsampling blurs colors across sharp luminance transitions (e.g., green foliage bleeding into cheetah fur, whiskers, ears, and eyes).
3. **Model Variant Strengths**: `piddnad/ddcolor_paper` was trained at 768x768 and produces significantly better natural color prediction (e.g. rich amber iris for the cheetah, natural pinkish ear interior, reduced green cast) compared to 512x512 variants.
4. **Hardware & Execution**: RTX 5070 Ti is installed on the host system; Python venv currently runs PyTorch CPU build. The engine must support transparent CUDA detection when available, clean fallback to CPU, thread safety, `torch.inference_mode()`, and detailed performance profiling (breakdown of preprocess, neural inference, refinement, postprocess).

## 2. Architecture & Algorithmic Improvements
1. **Guided Chrominance Refinement (Fast Guided Filter)**:
   - Implement an exact, edge-preserving guided filter in `colorrevive/backend/app/ml/ddcolor/refinement.py` using original luminance $L_{\text{orig}}$ as the guidance image.
   - Formulate the local linear model: $q_i = a_k L_i + b_k$, guaranteeing $\nabla (ab) \propto \nabla L$.
   - Constrain chrominance boundaries to snap cleanly to whiskers, fur silhouettes, facial contours, clothing edges, and pupils without creating halos or artificial outlines.
2. **Multi-Scale Quality Presets**:
   - **Standard**: 512x512 inference + bilinear upsampling + lightweight guided refinement. Fast, low memory.
   - **High Quality**: 768x768 inference + edge-aware guided chrominance refinement. Clean boundaries and fine details.
   - **Maximum Quality**: 768x768 inference + multi-scale bilateral guided refinement + adaptive gamut protection. Highest fidelity.
3. **Luminance-Aware Gamut & Saturation Protection**:
   - Preserve 100% of the original $L_{\text{orig}}$ channel.
   - Smooth cubic Hermite taper in deep shadows ($L < 15$) and specular highlights ($L > 95$) to eliminate unnatural chromatic casts in pure blacks and whites.
   - Soft chroma capping to eliminate out-of-gamut Lab clipping artifacts.
4. **Backend API & Service Upgrades**:
   - Update `app/schemas.py`: add `preset` (`standard`, `high`, `maximum`), `edge_refinement`, `shadow_protection`, and performance timing breakdown (`timing_breakdown: {preprocess_ms, inference_ms, refinement_ms, total_ms}`).
   - Update `app/services/colorization_service.py` and `app/ml/inference.py` to route quality parameters and collect timing metrics.
   - Update `app/api/colorize.py` and `app/api/health.py`.
5. **Frontend UI Enhancements**:
   - Update `settings-panel.tsx`: Quality preset selector (Standard / High / Maximum), Edge Refinement toggle, Shadow Protection toggle, Chroma Strength slider, Model Variant selector.
   - Update `processing-state.tsx` and result panel: Display quality preset, inference resolution, timing breakdown, device, and variant.
6. **Repeatable Evaluation Suite & Benchmarking**:
   - Implement `backend/scripts/evaluate_quality.py` testing 6 diverse photographic categories (Wildlife/Cheetah, Studio Portrait, Historical Portrait, Documentary, Urban/Architecture, Landscape/Sky).
   - Generate side-by-side comparison panels (Grayscale | Raw Baseline DDColor | Improved Pipeline).
   - Measure and record preprocessing, inference, refinement, and total latency.
7. **Verification & Testing**:
   - Update and execute backend PyTest test suite (color spaces, interpolation, presets, API contracts, error handling).
   - Update and execute frontend Vitest test suite and Next.js production build.
