import cv2
import re
import time
import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from dotenv import load_dotenv
from utils import (
    extract_state,
    confident_region,
    extract_plate_candidate,
    reconcile_plate_number,
    create_plate_response,
)

load_dotenv()

endpoint = os.getenv("VISION_ENDPOINT")
key = os.getenv("VISION_KEY")

if not endpoint or not key:
    raise ValueError("VISION_ENDPOINT and VISION_KEY must be set in environment variables")

vision_client = ImageAnalysisClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(key)
)


def _normalize_for_azure_ocr(image_crop, min_dim: int = 50, max_dim: int = 16000):
    """
    Ensure crop dimensions meet Azure OCR limits.
    Azure Image Analysis READ requires each image dimension to be in [50, 16000].
    """
    if image_crop is None or image_crop.size == 0:
        return None

    h, w = image_crop.shape[:2]
    if h <= 0 or w <= 0:
        return None

    # Determine valid scale interval such that both dimensions stay in [min_dim, max_dim].
    min_scale = max(min_dim / w, min_dim / h)
    max_scale = min(max_dim / w, max_dim / h)

    if min_scale > max_scale:
        return None

    # Keep size unchanged when already valid; otherwise move to nearest valid scale.
    scale = min(max(1.0, min_scale), max_scale)

    if scale != 1.0:
        new_w = max(min_dim, min(int(round(w * scale)), max_dim))
        new_h = max(min_dim, min(int(round(h * scale)), max_dim))
        interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        image_crop = cv2.resize(image_crop, (new_w, new_h), interpolation=interp)

    return image_crop


def extract_text_from_image_azure_cv(image_crop) -> tuple[str, dict[str, float]]:
    """
    Extract text from a cropped license plate image using Azure Computer Vision.

    Returns:
        (full_text, word_confidences) where word_confidences maps each word,
        normalized to bare alphanumerics, to its lowest reported confidence.
    """
    try:
        image_crop = _normalize_for_azure_ocr(image_crop)
        if image_crop is None:
            return "", {}

        ok, buffer = cv2.imencode('.jpg', image_crop)
        if not ok:
            return "", {}

        image_data = buffer.tobytes()

        result = vision_client.analyze(
            image_data=image_data,
            visual_features=[VisualFeatures.READ]
        )

        lines = [line for block in result.read.blocks
                 for line in block.lines] if result.read else []

        word_confidences: dict[str, float] = {}
        for word in (w for line in lines for w in line.words):
            key = re.sub(r'[^A-Z0-9]', '', word.text.upper())
            if key:
                # Same word can appear twice; keep the weakest read.
                word_confidences[key] = min(word.confidence,
                                            word_confidences.get(key, 1.0))

        return " ".join(line.text for line in lines).strip(), word_confidences

    except Exception as e:
        print(f"Azure CV OCR error: {e}")
        return "", {}


def _region_of(ocr) -> str | None:
    """The detector's region prediction, dropped when it is only guessing."""
    return confident_region(ocr.region, ocr.region_confidence) if ocr else None


def _azure_candidate_confidence(
    parts: list[str], word_confidences: dict[str, float]
) -> float | None:
    """
    Confidence for a plate candidate assembled from Azure OCR tokens: the
    weakest of the words it came from. None when no word could be matched
    (e.g. the candidate came from part of a word), so the caller can treat it
    as unmeasured rather than as low confidence.
    """
    matched = [
        conf for part in parts
        for word, conf in word_confidences.items()
        if part == word or part in word
    ]
    return min(matched) if matched else None


def read_plate_azure_cv(image_path: str, alpr, ALL_STATE_NAMES: dict[str, str]) -> dict:
    """
    Read license plate using fast_alpr for detection and Azure CV for OCR.
    Extracts plate_number via Regex and state via countrystatecity (all countries).
    
    Args:
        image_path: Path to the image file
        alpr: fast_alpr model instance
        ALL_STATE_NAMES: Dictionary mapping uppercase state names to proper case
        
    Returns:
        Dictionary with file name, list of detected plates, and processing time
    """
    breakpoint()
    start_time = time.time()
    img = cv2.imread(image_path)

    if img is None:
        return {"error": f"Failed to read image: {image_path}", "plates": []}

    plates = []

    try:
        results = alpr.predict(img)
        img_h, img_w = img.shape[:2]

        if not results:
            plates.append(create_plate_response(
                None, None, message="No license plate detected in the image"))
        else:
            for result in results:
                if not result.detection:
                    continue

                bbox = result.detection.bounding_box
                x1, x2 = sorted([int(round(bbox.x1)), int(round(bbox.x2))])
                y1, y2 = sorted([int(round(bbox.y1)), int(round(bbox.y2))])

                # Clip bbox to image bounds and expand a bit for OCR context.
                pad = 6
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(img_w, x2 + pad)
                y2 = min(img_h, y2 + pad)

                if x2 <= x1 or y2 <= y1:
                    plates.append(create_plate_response(
                        None, None, _region_of(result.ocr),
                        message="Invalid plate bounding box from detector"))
                    continue

                plate_crop = img[y1:y2, x1:x2]
                region = _region_of(result.ocr)

                # Candidate 1: the detector's own plate-specialised OCR.
                alpr_text = result.ocr.text if result.ocr else None
                alpr_conf = result.ocr.confidence if result.ocr else None

                # Candidate 2: Azure CV, which also reads the state/slogan text.
                raw_text, word_confidences = extract_text_from_image_azure_cv(plate_crop)
                print(f"Raw OCR text: {raw_text}")

                if not raw_text and not alpr_text:
                    plates.append(create_plate_response(
                        None, None, region,
                        message="OCR failed to extract text from detected plate"))
                    continue

                state = extract_state(raw_text, ALL_STATE_NAMES)
                azure_text, azure_parts = extract_plate_candidate(raw_text, state) or (None, [])
                azure_conf = _azure_candidate_confidence(azure_parts, word_confidences)

                reconciled = reconcile_plate_number(
                    alpr_text, alpr_conf, azure_text, azure_conf
                )

                plates.append(create_plate_response(
                    plate_number=reconciled["plate_number"],
                    state=state,
                    region=region,
                    confidence=reconciled["confidence"],
                    source=reconciled["source"],
                    alternate_plate_number=reconciled["alternate_plate_number"]
                ))

    except Exception as e:
        print(f"ALPR/OCR error in {image_path}: {e}")
        plates.append(create_plate_response(
            None, None, message=f"Processing error: {e}"))

    elapsed = round(time.time() - start_time, 3)
    print(f"{os.path.basename(image_path)} -> {plates} | Time: {elapsed}s")

    return {
        "file": os.path.basename(image_path),
        "plates": plates,
        "processing_time_sec": elapsed
    }
 