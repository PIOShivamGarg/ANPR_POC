# pip uninstall -y paddlepaddle-gpu
# pip install transformers
# pip uninstall -y torch torchvision torchaudio
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# pip install qwen-vl-utils

import cv2
import time
import os
import re
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from utils import extract_state, extract_plate_number, create_plate_response


# ─── Config ──────────────────────────────────────────────────────────────────────
PADDLE_MODEL_PATH = "PaddlePaddle/PaddleOCR-VL-1.5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SPOTTING_UPSCALE_THRESHOLD = 1500
MAX_PIXELS = 2048 * 28 * 28
PROMPT = "Spotting:"


# ─── Lazy Load Models (loaded on first use) ──────────────────────────────────────
paddle_model = None
paddle_processor = None


def _load_paddle_model():
    """Lazy load PaddleOCR model on first use to save memory."""
    global paddle_model, paddle_processor
    
    if paddle_model is None:
        print(f"⏳ Loading PaddleOCR-VL-1.5 model on {DEVICE}...")
        paddle_model = (
            AutoModelForImageTextToText
            .from_pretrained(PADDLE_MODEL_PATH, torch_dtype=torch.bfloat16)
            .to(DEVICE)
            .eval()
        )
        paddle_processor = AutoProcessor.from_pretrained(PADDLE_MODEL_PATH)
        print(f"✅ PaddleOCR-VL-1.5 loaded on {DEVICE}")
    
    return paddle_model, paddle_processor


def clean_ocr_text(text: str) -> str:
    """Removes <|LOC_...|> tags and newlines from the OCR text."""
    cleaned_text = re.sub(r'<\|LOC_\d+\|>', '', text)
    cleaned_text = cleaned_text.replace('\n', ' ')
    return cleaned_text.strip()


def run_paddle_ocr(pil_image: Image.Image) -> str:
    """
    Run PaddleOCR-VL-1.5 spotting on a PIL image (cropped license plate).
    Returns the cleaned decoded text string.
    """
    # Lazy load model on first use
    model, processor = _load_paddle_model()
    
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

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        images_kwargs={
            "size": {
                "shortest_edge": processor.image_processor.min_pixels,
                "longest_edge": MAX_PIXELS,
            }
        },
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512)

    text = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:-1])
    return clean_ocr_text(text)


def crop_plate_from_detection(img_bgr: np.ndarray, detection) -> Image.Image | None:
    """
    Crop the license plate region from the full BGR image using the
    bounding box provided by fast_alpr's detection result.
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


def read_plate_paddleocr_vl(image_path: str, alpr, ALL_STATE_NAMES: dict[str, str]) -> dict:
    """
    Full pipeline:
      1. fast_alpr  → detect plate(s), get bounding boxes
      2. Crop each plate from the original image
      3. PaddleOCR-VL-1.5 → run OCR on each cropped plate
      4. Extract plate_number via Regex
      5. Extract state via countrystatecity (all countries)
      
    Args:
        image_path: Path to the image file
        alpr: fast_alpr model instance
        ALL_STATE_NAMES: Dictionary mapping uppercase state names to proper case
        
    Returns:
        Dictionary with file name, list of detected plates, and processing time
    """
    start_time = time.time()
    img_bgr = cv2.imread(image_path)

    if img_bgr is None:
        print(f"❌ Failed to read image: {image_path}")
        return {"error": f"Failed to read image: {image_path}", "plates": []}

    plates = []

    try:
        results = alpr.predict(img_bgr)

        if not results:
            plates.append(create_plate_response(
                plate_number=None,
                state=None,
                region=None
            ))
            plates[0]["message"] = "No license plate detected in the image"
        else:
            for result in results:
                if result.detection is None:
                    continue

                region = result.ocr.region if result.ocr else None

                cropped_plate = crop_plate_from_detection(img_bgr, result.detection)
                if cropped_plate is None:
                    print(f"  [read_plate] Could not crop plate, skipping.")
                    plates.append(create_plate_response(
                        plate_number=None,
                        state=None,
                        region=region
                    ))
                    plates[-1]["message"] = "Failed to crop detected plate region"
                    continue

                try:
                    ocr_text = run_paddle_ocr(cropped_plate)
                except Exception as ocr_err:
                    print(f"  [PaddleOCR] Error: {ocr_err}")
                    plates.append(create_plate_response(
                        plate_number=None,
                        state=None,
                        region=region
                    ))
                    plates[-1]["message"] = f"OCR error: {str(ocr_err)}"
                    continue

                print(f"Raw OCR text: {ocr_text}")

                if not ocr_text:
                    plates.append(create_plate_response(
                        plate_number=None,
                        state=None,
                        region=region
                    ))
                    plates[-1]["message"] = "OCR failed to extract text from detected plate"
                    continue

                plate_number = extract_plate_number(ocr_text)
                state = extract_state(ocr_text, ALL_STATE_NAMES)

                plates.append(create_plate_response(
                    plate_number=plate_number,
                    state=state,
                    region=region
                ))

    except Exception as e:
        print(f"[read_plate] Detection error: {e}")
        plates.append(create_plate_response(
            plate_number=None,
            state=None,
            region=None
        ))
        plates[-1]["message"] = f"Processing error: {str(e)}"

    elapsed = round(time.time() - start_time, 3)
    print(f"📄 {os.path.basename(image_path)} → {plates} | Time: {elapsed}s")

    return {
        "file": os.path.basename(image_path),
        "plates": plates,
        "processing_time_sec": elapsed,
    }
