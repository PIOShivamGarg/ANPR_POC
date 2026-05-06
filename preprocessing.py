"""
Image Preprocessing Script - Highlight Text Content
Applies grayscaling, contrast enhancement, sharpening, and adaptive thresholding
to make text in images more prominent.
"""

import sys
import os
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def preprocess_image(input_path: str, output_dir: str = ".") -> dict:
    """
    Preprocess an image to highlight text content.

    Steps:
      1. Grayscale conversion
      2. Contrast & sharpness enhancement
      3. Gaussian blur (noise reduction)
      4. Adaptive thresholding (binarization)
      5. Edge / detail sharpening

    Returns a dict of {step_name: output_path}.
    """
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0]
    outputs = {}

    # ── 1. Load original ────────────────────────────────────────────────────
    original = Image.open(input_path).convert("RGB")
    p = os.path.join(output_dir, f"{base}_1_original.png")
    original.save(p)
    outputs["original"] = p
    print(f"[1] Saved original          → {p}")

    # ── 2. Grayscale ────────────────────────────────────────────────────────
    gray = original.convert("L")
    p = os.path.join(output_dir, f"{base}_2_grayscale.png")
    gray.save(p)
    outputs["grayscale"] = p
    print(f"[2] Saved grayscale         → {p}")

    # ── 3. Contrast + sharpness enhancement ─────────────────────────────────
    enhanced = ImageEnhance.Contrast(gray).enhance(2.5)
    enhanced = ImageEnhance.Sharpness(enhanced).enhance(3.0)
    p = os.path.join(output_dir, f"{base}_3_enhanced.png")
    enhanced.save(p)
    outputs["enhanced"] = p
    print(f"[3] Saved enhanced          → {p}")

    # ── 4. Gaussian blur (denoise before thresholding) ───────────────────────
    blurred = enhanced.filter(ImageFilter.GaussianBlur(radius=1))
    p = os.path.join(output_dir, f"{base}_4_blurred.png")
    blurred.save(p)
    outputs["blurred"] = p
    print(f"[4] Saved blurred           → {p}")

    # ── 5. Adaptive thresholding ─────────────────────────────────────────────
    if HAS_CV2:
        arr = np.array(blurred)
        thresh = cv2.adaptiveThreshold(
            arr, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=15,
            C=8
        )
        thresh_img = Image.fromarray(thresh)
        label = "adaptive_threshold_cv2"
    else:
        # Fallback: simple numpy threshold (Otsu-style midpoint)
        arr = np.array(blurred, dtype=np.float32)
        threshold = arr.mean()
        thresh = ((arr > threshold) * 255).astype(np.uint8)
        thresh_img = Image.fromarray(thresh)
        label = "adaptive_threshold_numpy"

    p = os.path.join(output_dir, f"{base}_5_{label}.png")
    thresh_img.save(p)
    outputs["thresholded"] = p
    print(f"[5] Saved thresholded       → {p}  (method: {label})")

    # ── 6. Final sharpening pass on thresholded image ───────────────────────
    sharpened = thresh_img.filter(ImageFilter.SHARPEN)
    sharpened = thresh_img.filter(ImageFilter.EDGE_ENHANCE_MORE)
    p = os.path.join(output_dir, f"{base}_6_final_sharpened.png")
    sharpened.save(p)
    outputs["final"] = p
    print(f"[6] Saved final sharpened   → {p}")

    return outputs


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads/Imagepreview.png"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "/mnt/user-data/outputs/preprocessed"

    print(f"\n{'='*55}")
    print(f"  Text-Highlight Preprocessor")
    print(f"  Input : {input_path}")
    print(f"  Output: {output_dir}")
    print(f"{'='*55}\n")

    if not os.path.exists(input_path):
        print(f"ERROR: file not found → {input_path}")
        sys.exit(1)

    results = preprocess_image(input_path, output_dir)

    print(f"\n✅  Done! {len(results)} images saved to: {output_dir}")
    print("\nPipeline summary:")
    for step, path in results.items():
        print(f"  {step:<22} → {os.path.basename(path)}")


if __name__ == "__main__":
    main()