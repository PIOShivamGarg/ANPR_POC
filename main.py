import cv2
import time
import os
import json
from ultralytics import YOLO
from fast_plate_ocr import LicensePlateRecognizer
import onnxruntime as ort

print(ort.get_available_providers())

model_path = r"D:\\projects\\ANPR_POC\\license_plate_detector.pt"
input_folder = r"D:\\projects\\ANPR_POC\\Vehicle License Plate List"

detector = YOLO(model_path)

ocr_model = LicensePlateRecognizer(
    "cct-s-v2-global-model",
    providers=["CPUExecutionProvider"]
)

print("Model loaded")


def read_plate(image_path):
    start_time = time.time()
    img = cv2.imread(image_path)

    if img is None:
        print(f"Failed to read: {image_path}")
        return []

    results = detector(image_path, verbose=False)[0]

    plates = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        if conf < 0.4:
            continue

        plate_crop = img[y1:y2, x1:x2] 

        if plate_crop is None or plate_crop.size == 0:
            continue

        try:
            ocr_result = ocr_model.run(plate_crop)[0]
            text = ocr_result.plate if hasattr(ocr_result, 'plate') else str(ocr_result)
        except Exception as e:
            print(f"OCR error in {image_path}: {e}")
            continue

        plates.append(text)

    print(f"{os.path.basename(image_path)} -> {plates} | Time: {time.time() - start_time:.3f}s")
    return plates


def process_folder(folder_path):
    supported_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    results = {}

    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(supported_ext):
            image_path = os.path.join(folder_path, file_name)
            plates = read_plate(image_path)
            results[file_name] = plates

    return results


if __name__ == "__main__":
    overall_start = time.time()
    all_results = process_folder(input_folder)
    overall_end = time.time()

    # Save results to JSON
    output_json = "plate_results.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    print("\n" + "="*60)
    print("Final Summary:")
    print("="*60)
    for file, plates in all_results.items():
        print(f"{file}: {plates}")
    
    print("="*60)
    print(f"Total images processed: {len(all_results)}")
    print(f"Total processing time: {overall_end - overall_start:.3f}s")
    print(f"Average time per image: {(overall_end - overall_start) / max(len(all_results), 1):.3f}s")
    print(f"Results saved to: {output_json}")
    print("="*60)