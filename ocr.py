import cv2
import time
from ultralytics import YOLO
from fast_plate_ocr import LicensePlateRecognizer
import onnxruntime as ort

print(ort.get_available_providers())
model_path = r"D:\Projects\ANPR_POC\yolov8_finetune\license_plate_detector.pt"
detector = YOLO(model_path)

ocr_model = LicensePlateRecognizer(
    "cct-s-v2-global-model",
    providers=["CPUExecutionProvider"]
)

print("Model loaded")

def read_plate(image_path):
    start_time = time.time()
    img = cv2.imread(image_path)
    results = detector(image_path)[0]

    plates = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        if conf < 0.4:
            continue

        plate_crop = img[y1:y2, x1:x2]

        # 🔥 safety check
        if plate_crop is None or plate_crop.size == 0:
            continue

        print("Crop shape:", plate_crop.shape)

        try:
            text = ocr_model.run(plate_crop)[0]
        except Exception as e:
            print("OCR error:", e)
            continue

        print("Plate:", text)
        plates.append(text)

    print(f"Total time: {time.time() - start_time:.3f}s")
    return plates


read_plate(r"D:\Projects\ANPR_POC\inputs\Imagepreview.png")
# read_plate(r"D:\Projects\ANPR_POC\inputs\images.jpg")
# read_plate(r"D:\Projects\ANPR_POC\inputs\img1.jpg")