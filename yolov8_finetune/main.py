import cv2
import easyocr
import re
from ultralytics import YOLO

model  = YOLO("license_plate_detector.pt")
reader = easyocr.Reader(['en'])

def clean_text(text):
    # Keep only alphanumeric characters
    return re.sub(r'[^A-Z0-9]', '', text.upper())

def score_text(text):
    """
    Heuristic scoring:
    - Prefer alphanumeric mix
    - Penalize very short or very long strings
    """
    length = len(text)

    if length < 4 or length > 12:
        return 0

    letters = sum(c.isalpha() for c in text)
    digits  = sum(c.isdigit() for c in text)

    # Balanced mix is usually a plate
    return length + min(letters, digits)

def read_plate(image_path):
    img = cv2.imread(image_path)
    results = model(image_path)[0]

    detected_plates = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        if conf < 0.4:
            continue

        plate_crop = img[y1:y2, x1:x2]

        # Preprocessing
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # Try multiple preprocessing variants (important for robustness)
        thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        thresh2 = cv2.adaptiveThreshold(gray, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)

        ocr_results = []
        for img_variant in [gray, thresh1, thresh2]:
            ocr_results.extend(reader.readtext(img_variant))

        best_text = ""
        best_score = 0

        for (_, text, prob) in ocr_results:
            cleaned = clean_text(text)
            score = score_text(cleaned) * prob

            if score > best_score:
                best_score = score
                best_text = cleaned

        if best_text:
            detected_plates.append(best_text)

        # Draw annotation
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, best_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imwrite("output.jpg", img)

    return detected_plates


plates = read_plate(r"D:\projects\ANPR_POC\inputs\images.jpg")
print("Detected Plates:", plates)