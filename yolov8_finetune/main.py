import cv2
import easyocr
import re
import os
import numpy as np
from pathlib import Path
from ultralytics import YOLO

model  = YOLO("license_plate_detector.pt")
# Initialize EasyOCR with character allowlist for license plates
reader = easyocr.Reader(['en'], gpu=False)

# Valid characters for license plates (alphanumeric only)
ALLOWED_CHARS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def clean_text(text):
    # Keep only alphanumeric characters
    return re.sub(r'[^A-Z0-9]', '', text.upper())

def fix_common_ocr_errors(text):
    """
    Fix common OCR misreads for license plates with smarter contextual rules
    """
    if len(text) < 4:
        return text
    
    result = list(text)
    
    # Fix H → N or W based on position (Indian/US plates)
    # Indian plates start with 2 letters (state code like TN, MH, KA)
    # If position 1 is 'H' and position 0 is a letter, likely TN, MH, etc.
    if len(result) >= 2:
        if result[1] == 'H' and result[0].isalpha():
            # Common Indian state codes ending in N: TN, KN, RN, PN, etc.
            if result[0] in ['T', 'K', 'R', 'P', 'A', 'M']:
                result[1] = 'N'
    
    # Fix W → H (often in US plates)
    # If we have digits followed by H followed by more letters, might be W
    for i in range(1, len(result)):
        if result[i] == 'H':
            # Check if previous character is a digit and next is a letter (like 8WAE)
            if i > 0 and i < len(result) - 1:
                if result[i-1].isdigit() and result[i+1].isalpha():
                    result[i] = 'W'
    
    # Character confusion fixes for digit positions
    for i, char in enumerate(result):
        # In digit-heavy positions (middle to end), prefer numbers
        if i >= 2:  # After state code
            if char == 'O':
                result[i] = '0'
            elif char == 'Q':
                result[i] = '0'
            elif char == 'D':
                result[i] = '0'
            elif char == 'I':
                result[i] = '1'
            elif char == 'L':
                result[i] = '1'
            elif char == 'S':
                result[i] = '5'
            elif char == 'Z':
                result[i] = '2'
            elif char == 'B' and i > 4:  # B->8 in number sections
                result[i] = '8'
            elif char == 'G':
                result[i] = '6'
            elif char == 'T' and i > 4:
                result[i] = '7'
    
    return ''.join(result)

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

def read_plate(image_path, output_path, save_debug=True):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not read image {image_path}")
        return []
    
    results = model(image_path)[0]

    detected_plates = []
    plate_counter = 0

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        if conf < 0.4:
            continue

        plate_crop = img[y1:y2, x1:x2]
        
        if plate_crop.size == 0:
            continue
        
        plate_counter += 1

        # Simpler, more effective preprocessing
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        
        # Resize to a good size (not too big)
        gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        
        # Apply bilateral filter to reduce noise while keeping edges sharp
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)

        # Try multiple thresholding methods
        thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        thresh2 = cv2.adaptiveThreshold(gray, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
        
        # Save debug images
        if save_debug:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            debug_folder = os.path.join(output_folder, "debug")
            os.makedirs(debug_folder, exist_ok=True)
            cv2.imwrite(os.path.join(debug_folder, f"{base_name}_plate{plate_counter}_gray.jpg"), gray)
            cv2.imwrite(os.path.join(debug_folder, f"{base_name}_plate{plate_counter}_thresh.jpg"), thresh1)

        ocr_results = []
        # Run OCR on promising variants
        for img_variant in [gray, thresh1]:
            try:
                results = reader.readtext(img_variant, 
                                        allowlist=ALLOWED_CHARS,
                                        detail=1,
                                        paragraph=False)
                ocr_results.extend(results)
            except Exception as e:
                pass

        best_text = ""
        best_score = 0
        raw_text = ""

        for (_, text, prob) in ocr_results:
            cleaned = clean_text(text)
            
            # Skip if too short or too long
            if len(cleaned) < 5 or len(cleaned) > 11:
                continue
            
            fixed = fix_common_ocr_errors(cleaned)
            score = score_text(fixed) * prob

            if score > best_score:
                best_score = score
                best_text = fixed
                raw_text = cleaned

        if best_text:
            detected_plates.append(best_text)
            if raw_text != best_text:
                print(f"  OCR Raw: '{raw_text}' -> Corrected: '{best_text}'")

        # Draw annotation
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, best_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imwrite(output_path, img)

    return detected_plates


# Process all images in the inputs folder
inputs_folder = r"D:\Projects\ANPR_POC\inputs"
output_folder = r"D:\Projects\ANPR_POC\output"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get all image files from inputs folder
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
image_files = [f for f in os.listdir(inputs_folder) 
               if os.path.splitext(f)[1].lower() in image_extensions]

print(f"Found {len(image_files)} images to process\n")

# Process each image
for image_file in image_files:
    input_path = os.path.join(inputs_folder, image_file)
    output_filename = f"output_{os.path.splitext(image_file)[0]}.jpg"
    output_path = os.path.join(output_folder, output_filename)
    
    print(f"Processing: {image_file}")
    plates = read_plate(input_path, output_path)
    print(f"Detected Plates: {plates}")
    print(f"Output saved to: {output_path}\n")