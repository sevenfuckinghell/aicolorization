"""Test live API call to /api/v1/colorize."""

from __future__ import annotations

import base64
from pathlib import Path
import httpx

url = "http://127.0.0.1:8001/api/v1/colorize"
img_path = Path("colorrevive/sample-data/user_1308x736.jpg")

with open(img_path, "rb") as f:
    files = {"image": ("user.jpg", f.read(), "image/jpeg")}
    data = {"quality": "high", "edge_refinement": "true"}
    resp = httpx.post(url, files=files, data=data, timeout=60.0)

assert resp.status_code == 200, f"Status: {resp.status_code}, {resp.text}"
res = resp.json()
print("Success:", res["success"])
print("Model:", res["model"])
print("Variant:", res.get("model_variant"))
print("Dimensions:", f"{res['width']}x{res['height']}")
print("Time ms:", res["processing_time_ms"])
print("Fallback mode:", res["fallback_mode"])
print("Quality preset:", res.get("quality_preset"))
print("Edge refinement applied:", res.get("edge_refinement_applied"))

out_bytes = base64.b64decode(res["image_base64"])
out_path = Path("C:/Users/pc/.gemini/antigravity/brain/e2f002de-06dd-4efa-9d4a-312c98d57c12/live_api_result_modelscope.png")
out_path.write_bytes(out_bytes)
print(f"Saved live output ({len(out_bytes)} bytes) to {out_path}")
