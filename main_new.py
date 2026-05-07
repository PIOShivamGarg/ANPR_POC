# pip uninstall -y paddlepaddle-gpu
# pip install transformers
# pip uninstall -y torch torchvision torchaudio
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# pip install fast_alpr
# pip install qwen-vl-utils
# pip install onnxruntime
# pip install countrystatecity-countries

import cv2
import time
import os
import re
import shutil
import tempfile

import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from fast_alpr import ALPR
from countrystatecity_countries import get_countries, get_states_of_country

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse


# ─── Config ──────────────────────────────────────────────────────────────────────

PADDLE_MODEL_PATH = "PaddlePaddle/PaddleOCR-VL-1.5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SPOTTING_UPSCALE_THRESHOLD = 1500
MAX_PIXELS = 2048 * 28 * 28
PROMPT = "Spotting:"


# ─── Load All World State Names ──────────────────────────────────────────────────
print("Loading all world states...")
_all_states = []
for country in get_countries():
    states = get_states_of_country(country.iso2)
    _all_states.extend(states)

ALL_STATE_NAMES = {state.name.upper(): state.name for state in _all_states}
print(f"Loaded {len(ALL_STATE_NAMES)} states/regions from all countries")


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


# ─── Helper: Clean OCR Text ──────────────────────────────────────────────────────
def clean_ocr_text(text: str) -> str:
    """Removes <|LOC_...|> tags and newlines from the OCR text."""
    cleaned_text = re.sub(r'<\|LOC_\d+\|>', '', text)
    cleaned_text = cleaned_text.replace('\n', ' ')
    return cleaned_text.strip()


# ─── Helper: Extract State from OCR Text ────────────────────────────────────────
def extract_state(text: str) -> str | None:
    """
    Match any word or multi-word phrase in the OCR text
    against the global state names list (all countries).
    Returns properly cased state name (e.g. 'Washington', 'Texas') or None.
    """
    text_upper = text.upper()

    # First pass: multi-word match (e.g. "New York", "New South Wales")
    for state_upper, state_proper in ALL_STATE_NAMES.items():
        if state_upper in text_upper:
            return state_proper

    # Second pass: single word fallback with punctuation stripped
    words = text_upper.split()
    for word in words:
        clean_word = re.sub(r'[^A-Z\s]', '', word).strip()
        if clean_word in ALL_STATE_NAMES:
            return ALL_STATE_NAMES[clean_word]

    return None


# ─── Helper: Extract Plate Number from OCR Text ─────────────────────────────────
def extract_plate_number(text: str) -> str | None:
    """
    Extract license plate number from raw OCR text using Regex.
    Handles formats like:
      - CEH4091       (no separator)
      - FNR*8034      (star separator - Texas style)
      - ABC-1234      (hyphen separator)
      - ABC·1234      (dot/bullet separator)
    """
    pattern = r'\b([A-Z0-9]{2,4}[+*\-·]?[A-Z0-9]{2,4})\b'
    text_upper = text.upper()

    matches = re.findall(pattern, text_upper)

    if not matches:
        return None

    noise_words = {"READ", "TEXT", "ALPR", "STATE", "AUTO"}

    for match in matches:
        clean = re.sub(r'[+*\-·]', '', match)  # strip separators
        has_letter = bool(re.search(r'[A-Z]', clean))
        has_digit = bool(re.search(r'[0-9]', clean))
        is_noise = match in noise_words

        if has_letter and has_digit and not is_noise and len(clean) >= 5:
            return clean  # return clean version without separator

    return None


# ─── PaddleOCR Inference ─────────────────────────────────────────────────────────

def run_paddle_ocr(pil_image: Image.Image) -> str:
    """
    Run PaddleOCR-VL-1.5 spotting on a PIL image (cropped license plate).
    Returns the cleaned decoded text string.
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


# ─── Core Logic ──────────────────────────────────────────────────────────────────

def read_plate(image_path: str) -> dict:
    """
    Full pipeline:
      1. fast_alpr  → detect plate(s), get bounding boxes
      2. Crop each plate from the original image
      3. PaddleOCR-VL-1.5 → run OCR on each cropped plate
      4. Extract plate_number via Regex
      5. Extract state via countrystatecity (all countries)
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
            # No plate detected at all by ALPR
            plates.append({
                "plate_number": None,
                "state": None,
                "message": "No information present on the given license plate"
            })
        else:
            for result in results:
                if result.detection is None:
                    continue

                cropped_plate = crop_plate_from_detection(img_bgr, result.detection)
                if cropped_plate is None:
                    print(f"  [read_plate] Could not crop plate, skipping.")
                    continue

                try:
                    ocr_text = run_paddle_ocr(cropped_plate)
                except Exception as ocr_err:
                    print(f"  [PaddleOCR] Error: {ocr_err}")
                    ocr_text = ""

                print(f"Raw OCR text: {ocr_text}")

                # Extract plate number and state
                plate_number = extract_plate_number(ocr_text) if ocr_text else None
                state = extract_state(ocr_text) if ocr_text else None

                plates.append({
                    "plate_number": plate_number,
                    "state": state,
                    "message": None if plate_number else "No information present on the given license plate"
                })

    except Exception as e:
        print(f"[read_plate] Detection error: {e}")

    elapsed = round(time.time() - start_time, 3)
    print(f"📄 {os.path.basename(image_path)} → {plates} | Time: {elapsed}s")

    return {
        "file": os.path.basename(image_path),
        "plates": plates,
        "processing_time_sec": elapsed,
    }


# ─── FastAPI App ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ANPR API",
    description="Automatic Number Plate Recognition — fast_alpr detection + PaddleOCR-VL-1.5 OCR",
    version="2.0.0",
)


@app.get("/")
def root():
    return {"message": "ANPR API is running 🚗", "version": "2.0.0", "ocr_backend": "PaddleOCR-VL-1.5"}


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


@app.post("/read-plate", summary="Upload a single image to detect license plate")
async def read_plate_api(file: UploadFile = File(...)):
    supported_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

    if not file.filename.lower().endswith(supported_ext):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {supported_ext}",
        )

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=os.path.splitext(file.filename)[1],
    ) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = read_plate(tmp_path)
        result["file"] = file.filename
        return JSONResponse(content=result)
    finally:
        os.unlink(tmp_path)


# ─── Run ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)