import cv2
import easyocr
import numpy as np
from ultralytics import YOLO

# ──────────────────────────────────────────────
# 1. Load Models
# ──────────────────────────────────────────────
model  = YOLO("yolov8n.pt")          # downloads automatically on first run
reader = easyocr.Reader(['en'])       # add 'hi' for Hindi/Devanagari plates

# COCO class IDs for vehicles
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# ──────────────────────────────────────────────
# 2. Preprocessing helpers
# ──────────────────────────────────────────────
def preprocess_for_ocr(crop: np.ndarray) -> np.ndarray:
    """
    Upscale → grayscale → denoise → threshold.
    Better contrast = better OCR results.
    """
    # Upscale 2x so EasyOCR gets enough resolution
    h, w = crop.shape[:2]
    crop = cv2.resize(crop, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Mild blur to kill noise without losing edges
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Otsu binarisation (auto threshold)
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return thresh


def clean_plate_text(ocr_results: list, min_prob: float = 0.3) -> str:
    """Merge all OCR tokens above confidence threshold into one string."""
    tokens = [
        text.upper().strip()
        for (_, text, prob) in ocr_results
        if prob >= min_prob and text.strip()
    ]
    return " ".join(tokens)


# ──────────────────────────────────────────────
# 3. Core pipeline
# ──────────────────────────────────────────────
def detect_plates(image_path: str, conf_threshold: float = 0.4) -> dict:
    """
    Detect vehicles → crop ROI → preprocess → OCR.
    Returns annotated image path + list of plate readings.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    results     = model(image_path)[0]
    detections  = []

    for box in results.boxes:
        cls_id     = int(box.cls[0])
        confidence = float(box.conf[0])

        # Only process vehicles above confidence threshold
        if cls_id not in VEHICLE_CLASSES or confidence < conf_threshold:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        vehicle_label   = VEHICLE_CLASSES[cls_id]

        # ── Crop the full vehicle ROI ──────────────────
        vehicle_crop = img[y1:y2, x1:x2]

        # ── Focus on lower 35% of the vehicle
        #    (plates are almost always near the bumper)
        h = vehicle_crop.shape[0]
        plate_roi = vehicle_crop[int(h * 0.65):, :]   # bottom 35 %

        # ── Preprocess & OCR ──────────────────────────
        processed  = preprocess_for_ocr(plate_roi)
        ocr_output = reader.readtext(processed)
        plate_text = clean_plate_text(ocr_output)

        detections.append({
            "vehicle":    vehicle_label,
            "confidence": round(confidence, 2),
            "bbox":       (x1, y1, x2, y2),
            "plate_text": plate_text or "Not detected"
        })

        # ── Annotate original image ───────────────────
        color = (0, 220, 100)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{vehicle_label} | {plate_text or 'N/A'}"
        cv2.putText(
            img, label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2
        )

    # Save annotated result
    output_path = "output_annotated.jpg"
    cv2.imwrite(output_path, img)

    return {
        "output_image": output_path,
        "detections":   detections
    }


# ──────────────────────────────────────────────
# 4. Run it
# ──────────────────────────────────────────────
if __name__ == "__main__":
    IMAGE_PATH = "D:\projects\ANPR_POC\Img1.jpg"          # ← change to your image

    result = detect_plates(IMAGE_PATH)

    print(f"\n✅ Annotated image saved → {result['output_image']}\n")
    print(f"{'─'*45}")

    if not result["detections"]:
        print("⚠  No vehicles detected. Try a lower conf_threshold.")
    else:
        for i, det in enumerate(result["detections"], 1):
            print(f"[{i}] Vehicle   : {det['vehicle']}")
            print(f"    Confidence: {det['confidence']}")
            print(f"    BBox      : {det['bbox']}")
            print(f"    Plate     : {det['plate_text']}")
            print(f"{'─'*45}")