# pip uninstall -y paddlepaddle-gpu
# pip install transformers
# pip uninstall -y torch torchvision torchaudio
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# pip install fast_alpr
# pip install qwen-vl-utils
# pip install onnxruntime

import cv2
import time
import os
import json
import re

import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from fast_alpr import ALPR


# ─── Config ──────────────────────────────────────────────────────────────────────

PADDLE_MODEL_PATH = "PaddlePaddle/PaddleOCR-VL-1.5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SPOTTING_UPSCALE_THRESHOLD = 1500
MAX_PIXELS = 2048 * 28 * 28
PROMPT = "Spotting:"


# ─── Load Models ─────────────────────────────────────────────────────────────────

alpr = ALPR(
    detector_model="yolo-v9-s-608-license-plate-end2end"
)
print("✅ fast_alpr detector loaded")

paddle_model = (
    AutoModelForImageTextToText
    .from_pretrained(PADDLE_MODEL_PATH, torch_dtype=torch.bfloat16)
    .to(DEVICE)
    .eval()
)
paddle_processor = AutoProcessor.from_pretrained(PADDLE_MODEL_PATH)
print(f"✅ PaddleOCR-VL-1.5 loaded on {DEVICE}")


# ─── Helper function to clean OCR text ──────────────────────────────────────────
def clean_ocr_text(text: str) -> str:
    """Removes <|LOC_...|> tags and newlines from the OCR text."""
    # Remove <|LOC_digits|> patterns and then newlines
    cleaned_text = re.sub(r'<\|LOC_\d+\|>', '', text)
    cleaned_text = cleaned_text.replace('\n', ' ')
    return cleaned_text.strip()


# ─── PaddleOCR Inference ─────────────────────────────────────────────────────────

def run_paddle_ocr(pil_image: Image.Image) -> str:
    """
    Run PaddleOCR-VL-1.5 spotting on a PIL image (cropped license plate).
    Returns the decoded text string without location tags.
    """
    orig_w, orig_h = pil_image.size
    if orig_w < SPOTTING_UPSCALE_THRESHOLD and orig_h < SPOTTING_UPSCALE_THRESHOLD:
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.LANCZOS
        pil_image = pil_image.resize((orig_w * 2, orig_h * 2), resample)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text",  "text": PROMPT},
            ],
        }
    ]

    inputs = paddle_processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        images_kwargs={
            "size": {
                "shortest_edge": paddle_processor.image_processor.min_pixels,
                "longest_edge": MAX_PIXELS,
            }
        },
    ).to(paddle_model.device)

    with torch.no_grad():
        outputs = paddle_model.generate(**inputs, max_new_tokens=512)

    text = paddle_processor.decode(outputs[0][inputs["input_ids"].shape[-1]:-1])
    return clean_ocr_text(text)


# ─── Crop Helper ─────────────────────────────────────────────────────────────────

def crop_plate_from_detection(img_bgr: np.ndarray, detection) -> Image.Image | None:
    """
    Crop the license plate region from the full BGR image using
    the bounding box from fast_alpr's detection result.
    """
    try:
        bb = detection.bounding_box
        if hasattr(bb, 'x1'):
            x1, y1, x2, y2 = int(bb.x1), int(bb.y1), int(bb.x2), int(bb.y2)
        else:
            x1, y1, x2, y2 = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])

        h, w = img_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        cropped_bgr = img_bgr[y1:y2, x1:x2]
        cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(cropped_rgb)

    except Exception as e:
        print(f"  [crop_plate] Failed to crop: {e}")
        return None


# ─── Single Image ─────────────────────────────────────────────────────────────────

def read_plate(image_path: str) -> dict:
    """
    Full pipeline for a single image:
      1. fast_alpr  → detect plate(s), get bounding boxes
      2. Crop each plate from the original image
      3. PaddleOCR-VL-1.5 → run OCR on each cropped plate

    Args:
        image_path: Path to the image file.

    Returns:
        dict with keys: file, plates (list of {text}), processing_time_sec
    """
    start_time = time.time()
    img_bgr = cv2.imread(image_path)

    if img_bgr is None:
        print(f"❌ Failed to read image: {image_path}")
        return {"error": f"Failed to read image: {image_path}", "plates": []}

    plates = []

    try:
        results = alpr.predict(img_bgr)

        for result in results:
            if result.detection is None:
                continue

            cropped_plate = crop_plate_from_detection(img_bgr, result.detection)
            if cropped_plate is None:
                print("  [read_plate] Could not crop plate, skipping.")
                continue

            try:
                ocr_text = run_paddle_ocr(cropped_plate)
            except Exception as ocr_err:
                print(f"  [PaddleOCR] Error: {ocr_err}")
                ocr_text = ""

            if ocr_text:
                plates.append({
                    "text": ocr_text,
                })

    except Exception as e:
        print(f"[read_plate] Detection error: {e}")

    elapsed = round(time.time() - start_time, 3)
    print(f"📄 {os.path.basename(image_path)} → {[p.get('text', 'N/A') for p in plates]} | Time: {elapsed}s")

    return {
        "file": os.path.basename(image_path),
        "plates": plates,
        "processing_time_sec": elapsed,
    }


# ─── Usage ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Single image ──
    result = read_plate("/content/car_photos/616399990_734488689734557_4980646476809510070_n.jpg")
    print(result)

    # ── Entire folder ──
    # result = process_folder("/content/your_folder")
    # print(result)
 