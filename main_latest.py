import cv2
import re
import time
import os
import shutil
import tempfile
from fast_alpr import ALPR
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from azure.core.credentials import AzureKeyCredential
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from dotenv import load_dotenv
from countrystatecity_countries import get_countries, get_states_of_country

# ─── Load Environment Variables ─────────────────────────────────────────────────
load_dotenv()

# ─── Load All World State Names ──────────────────────────────────────────────────
print("Loading all world states...")
_all_states = []
for country in get_countries():
    states = get_states_of_country(country.iso2)
    _all_states.extend(states)

ALL_STATE_NAMES = {state.name.upper(): state.name for state in _all_states}
print(f"Loaded {len(ALL_STATE_NAMES)} states/regions from all countries")

# ─── Load Models ────────────────────────────────────────────────────────────────
alpr = ALPR(
    detector_model="yolo-v9-s-608-license-plate-end2end"
)

endpoint = os.getenv("VISION_ENDPOINT")
key = os.getenv("VISION_KEY")

if not endpoint or not key:
    raise ValueError("VISION_ENDPOINT and VISION_KEY must be set in environment variables")

vision_client = ImageAnalysisClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(key)
)

print("ALPR model loaded (detection)")
print("Azure Computer Vision client initialized (OCR)")


# ─── FastAPI App ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ANPR API with Azure CV",
    description="Automatic Number Plate Recognition — Detection via fast_alpr, OCR via Azure Computer Vision",
    version="2.0.0"
)


# ─── Helper: Extract State from OCR Text ────────────────────────────────────────
def extract_state(text: str) -> str | None:
    """
    Match any word or multi-word phrase in the OCR text
    against the global state names list (all countries).
    Returns properly cased state name (e.g. 'Washington', 'Texas') or None.
    """
    text_upper = text.upper()

    # First try multi-word matches (e.g. "New York", "New South Wales")
    for state_upper, state_proper in ALL_STATE_NAMES.items():
        if state_upper in text_upper:
            return state_proper

    # Fallback: single word match with punctuation stripped
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

    # Filter out false positives:
    # - Must have both letters and digits
    # - Minimum length 5 (without separator)
    # - Skip known noise words
    noise_words = {"READ", "TEXT", "ALPR", "STATE", "AUTO"}

    for match in matches:
        clean = re.sub(r'[+*\-·]', '', match)  # strip separators
        has_letter = bool(re.search(r'[A-Z]', clean))
        has_digit = bool(re.search(r'[0-9]', clean))
        is_noise = match in noise_words

        if has_letter and has_digit and not is_noise and len(clean) >= 5:
            return clean  # return cleaned plate number without separators e.g. FNRR8034

    return None


# ─── Core Logic ─────────────────────────────────────────────────────────────────
def extract_text_from_image_azure_cv(image_crop) -> str:
    """
    Extract text from a cropped license plate image using Azure Computer Vision.
    """
    try:
        _, buffer = cv2.imencode('.jpg', image_crop)
        image_data = buffer.tobytes()

        result = vision_client.analyze(
            image_data=image_data,
            visual_features=[VisualFeatures.READ]
        )

        text_lines = []
        if result.read is not None:
            for block in result.read.blocks:
                for line in block.lines:
                    text_lines.append(line.text)

        return " ".join(text_lines).strip()

    except Exception as e:
        print(f"Azure CV OCR error: {e}")
        return ""


def read_plate(image_path: str) -> dict:
    """
    Read license plate using fast_alpr for detection and Azure CV for OCR.
    Extracts plate_number via Regex and state via countrystatecity (all countries).
    """
    start_time = time.time()
    img = cv2.imread(image_path)

    if img is None:
        return {"error": f"Failed to read image: {image_path}", "plates": []}

    plates = []

    try:
        results = alpr.predict(img)

        for result in results:
            if not result.detection:
                continue

            bbox = result.detection.bounding_box
            x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)

            plate_crop = img[y1:y2, x1:x2]

            # Get raw OCR text from Azure CV
            raw_text = extract_text_from_image_azure_cv(plate_crop)
            print(f"Raw OCR text: {raw_text}")

            if not raw_text:
                continue

            # Extract plate number and state
            plate_number = extract_plate_number(raw_text)
            state = extract_state(raw_text)

            if plate_number:
                plates.append({
                    "plate_number": plate_number,
                    "state": state,
                    "region": result.ocr.region if result.ocr else None
                })

    except Exception as e:
        print(f"ALPR/OCR error in {image_path}: {e}")

    elapsed = round(time.time() - start_time, 3)
    print(f"{os.path.basename(image_path)} -> {plates} | Time: {elapsed}s")

    return {
        "file": os.path.basename(image_path),
        "plates": plates,
        "processing_time_sec": elapsed
    }


# ─── API Routes ──────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "ANPR API with Azure CV is running 🚗", "version": "2.0.0"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/read-plate", summary="Upload a single image to detect license plate")
async def read_plate_api(file: UploadFile = File(...)):
    supported_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

    if not file.filename.lower().endswith(supported_ext):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {supported_ext}"
        )

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=os.path.splitext(file.filename)[1]
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
    uvicorn.run("main_latest:app", host="0.0.0.0", port=8000, reload=False)