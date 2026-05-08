import cv2
import time
import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from dotenv import load_dotenv
from utils import extract_state, extract_plate_number, create_plate_response

# ─── Load Environment Variables ─────────────────────────────────────────────────
load_dotenv()

endpoint = os.getenv("VISION_ENDPOINT")
key = os.getenv("VISION_KEY")

if not endpoint or not key:
    raise ValueError("VISION_ENDPOINT and VISION_KEY must be set in environment variables")

vision_client = ImageAnalysisClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(key)
)


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
    start_time = time.time()
    img = cv2.imread(image_path)

    if img is None:
        return {"error": f"Failed to read image: {image_path}", "plates": []}

    plates = []

    try:
        results = alpr.predict(img)

        if not results:
            # No plate detected by ALPR
            plates.append(create_plate_response(
                plate_number=None,
                state=None,
                region=None
            ))
            plates[0]["message"] = "No license plate detected in the image"
        else:
            for result in results:
                if not result.detection:
                    continue

                bbox = result.detection.bounding_box
                x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)

                plate_crop = img[y1:y2, x1:x2]
                region = result.ocr.region if result.ocr else None

                # Get raw OCR text from Azure CV
                raw_text = extract_text_from_image_azure_cv(plate_crop)
                print(f"Raw OCR text: {raw_text}")

                if not raw_text:
                    plates.append(create_plate_response(
                        plate_number=None,
                        state=None,
                        region=region
                    ))
                    plates[-1]["message"] = "OCR failed to extract text from detected plate"
                    continue

                # Extract plate number and state
                plate_number = extract_plate_number(raw_text)
                state = extract_state(raw_text, ALL_STATE_NAMES)

                plates.append(create_plate_response(
                    plate_number=plate_number,
                    state=state,
                    region=region
                ))

    except Exception as e:
        print(f"ALPR/OCR error in {image_path}: {e}")
        plates.append(create_plate_response(
            plate_number=None,
            state=None,
            region=None
        ))
        plates[-1]["message"] = f"Processing error: {str(e)}"

    elapsed = round(time.time() - start_time, 3)
    print(f"{os.path.basename(image_path)} -> {plates} | Time: {elapsed}s")

    return {
        "file": os.path.basename(image_path),
        "plates": plates,
        "processing_time_sec": elapsed
    }
