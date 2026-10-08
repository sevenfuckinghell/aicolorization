# Model Card: DDColor (Dual-Decoder Image Colorization)

## Summary

**DDColor** (*"Towards Photo-Realistic Image Colorization via Dual Decoders"*, ICCV 2023) is a deep-learning colorization model developed by researchers at Alibaba DAMO Academy. It replaces heuristic luminance-lookup colorizers with a dual-decoder architecture combining multi-scale visual features and learnable color queries to achieve state-of-the-art, photo-realistic colorization.

| Attribute | Value |
|---|---|
| **Architecture** | Dual Decoders: ConvNeXt backbone encoder + multi-scale pixel decoder + query-based color transformer decoder |
| **Backbone** | `convnext-l` (227.8 M parameters) |
| **Input** | Grayscale image in CIE Lab color space, resized internally to 512×512 (Standard) or 768×768 (High) |
| **Output** | Predicted `a` and `b` chrominance channels combined with the **original native-resolution luminance (`L`)** channel |
| **Framework** | PyTorch ≥ 2.1, Hugging Face Hub, CPU-compatible with automatic NVIDIA CUDA acceleration |
| **Default Pretrained Model** | `piddnad/ddcolor_modelscope` |
| **Alternative Variants** | `piddnad/ddcolor_artistic`, `piddnad/ddcolor_paper`, `piddnad/ddcolor_paper_tiny` |
| **License** | Apache 2.0 (Official DDColor project) |

---

## Technical Details

### Why DDColor Replaced the Heuristic Fallback

Traditional/heuristic approaches (and unconditioned U-Nets) suffer from two major flaws:
1. **Luminance-only heuristics** map brightness bins to fixed colors, producing monotonous greenish/yellowish/sepia tints across unrelated objects.
2. **Color bleeding and desaturation** occur when standard convolutions smear chrominance across boundaries.

DDColor solves this by:
- Using a **ConvNeXt-Large** encoder to extract rich, semantic representations (recognizing skies, skin, hair, water, clothing, foliage).
- Using **learnable color query tokens** via cross-attention with multi-scale visual features to predict crisp, semantically accurate chrominance.
- Combining predicted `ab` channels with the photograph's **original native luminance (`L`) channel**, guaranteeing that original sharpness, edge details, facial expressions, textures, and shading are 100% preserved.

---

## Intended Use

- Colorization of historical black-and-white photographs, archival imagery, scanned documents, and family portraits.
- Creative restoration of grayscale artwork and photography.
- High-resolution color restoration with preserved photographic structure.

## Out-of-Scope Use

- **Forensic or evidentiary assertions:** The colors produced are plausible AI estimates. A black-and-white photograph does not preserve original wavelength information; generated colors must never be treated as legal or forensic evidence.
- Scientific/medical imaging where chrominance represents quantitative measurement.
- Real-time video stream colorization without temporal smoothing.

---

## Environmental & Hardware Requirements

- **CPU Mode:** Fully supported out of the box using PyTorch CPU runtime. Inference takes ≈ 1–2 seconds on modern CPUs.
- **GPU Mode:** Supported automatically when NVIDIA GPU with CUDA is detected (`DEVICE=auto` or `DEVICE=cuda`). Inference takes ≈ 50–150 ms on modern RTX GPUs.
- **Memory Footprint:** ≈ 1.5–2.5 GB RAM / VRAM during 512×512 inference.

---

## Model Licensing & Attribution

- **DDColor**: Released under the **Apache License 2.0** by Alibaba DAMO Academy (Kang et al., ICCV 2023).
- **Hugging Face Checkpoints**: `piddnad/ddcolor_modelscope` hosted on Hugging Face under Apache 2.0.
- **Reference**:
  ```bibtex
  @inproceedings{kang2023ddcolor,
    title={DDColor: Towards Photo-Realistic Image Colorization via Dual Decoders},
    author={Kang, Xiaoyang and Yang, Tao and Ouyang, Wenqi and Ren, Peiran and Li, Lingzhi and Xie, Xuansong},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year={2023}
  }
  ```
