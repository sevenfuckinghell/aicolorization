# Model Card: unet-lab-v1 (ColorRevive Colorizer)

## Summary

A U-Net encoder–decoder that predicts chrominance (`a`, `b`) channels from a
luminance (`L`) channel in CIE Lab color space, producing plausible color for
grayscale images.

| Attribute | Value |
|---|---|
| Architecture | U-Net, 1-in / 2-out conv encoder-decoder with skip connections |
| Input | Single-channel normalized L ∈ [−1, 1], resized to 256×256 (standard) or 512×512 (high) |
| Output | Two-channel tanh-activated ab ∈ [−1, 1], scaled by ÷/×128 to Lab units |
| Base channels | 64 (configurable) |
| Parameters | ≈ 7.8 M at base_channels=64 |
| Framework | PyTorch ≥ 2.1, CPU-compatible, optional CUDA |
| Training data | Any RGB corpus (COCO val2017, Places365 subset, or a local folder) — **not included in this repository** |

## Intended use

- Colorizing old family photographs, historical grayscale photos, scanned images.
- Creative pre-visualization of color for archival material.
- General grayscale → color conversion at moderate resolution.

## Out-of-scope use

- **Forensic or evidentiary claims about original colors.** Outputs are
  predictions; the true colors are unknowable from luminance alone.
- Medical, scientific, or remote-sensing imagery where chrominance carries
  quantitative meaning.
- Video colorization (no temporal consistency mechanism).
- Images whose information is *only* in color differences invisible to L
  (e.g., isoluminant red/green patterns may be colored arbitrarily).

## Color-space representation

RGB → CIE Lab (D65). The L channel is fed to the network unchanged (normalized
to [−1, 1]); `a` and `b` targets are divided by 128 and squashed through tanh,
which keeps outputs inside the representable sRGB gamut after clipping. During
post-processing the predicted `a`/`b` are combined with either the original L
(`preserve_contrast=true`, default) or the model's implied L, then converted
back to RGB and resized to the source dimensions.

## Known limitations & failure modes

- **Mode collapse to desaturated predictions.** With L1 loss the model tends to
  predict low-chroma pastels for ambiguous regions — historically common for
  colorization networks.
- **Semantic guessing:** skies blue, grass green, skin warm — correct often,
  confidently wrong sometimes (night skies, autumn foliage, dyed hair, uniforms).
- **Small objects & textures** (jewelry, text logos, fabrics) receive smeared
  color because the receptive field averages context.
- **Out-of-distribution inputs** (illustrations, screenshots, infrared, heavy
  noise) produce unpredictable hues.
- **Bias:** models trained on COCO/ImageNet over-represent Western datasets;
  skin-tone rendering accuracy varies across populations and lighting conditions.
- Metrics (PSNR/SSIM) reward average-looking color, not plausible color; a
  high-PSNR output can still look wrong to humans.

## Evaluation

Use `python -m src.evaluate --checkpoint checkpoints/best.pt --data <val_dir>`:
reports MAE/MSE on Lab channels, PSNR/SSIM on RGB, mean ΔE₂₀₀₀, and writes a
comparison grid (Original | Grayscale | Colorized | Abs-diff). Expect rough
benchmarks on a 5-epoch demo run: L-ab MAE ≈ 4–7, PSNR ≈ 19–22 dB — plausibility
requires human review, not just numbers.

## Fallback mode (this repository)

No trained checkpoint ships with ColorRevive. Without `MODEL_CHECKPOINT_PATH`
the API runs a **deterministic non-neural enhancement** (gentle warm/split-toned
chroma mapped from luminance statistics). It is labeled `fallback_mode: true`
in every response and in the UI badge. It must not be described as AI
colorization.

## License considerations

- Code in this repository: MIT (see root README).
- Checkpoints inherit the license of their training data: COCO images permit
  research/commercial derivative models under its terms; ImageNet (Fallows 2015
  academic terms) restricts some commercial uses; Places365 is CC BY-By-NC-SA.
  Verify dataset licenses before distributing a trained checkpoint.

## Maintenance

Model version tracks `MODEL_NAME` + checkpoint hash reported by
`GET /api/v1/model-status`. Replace the checkpoint via env var only — no code
changes required.
