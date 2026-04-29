import cv2
import numpy as np
import easyocr
import time

def extract_plate_text(image_path, config_path, weights_path):
    start_time = time.time()
    
    # Initialize EasyOCR reader (for English)
    print("Initializing EasyOCR reader...")
    ocr_start = time.time()
    reader = easyocr.Reader(['en'], gpu=False)
    print(f"EasyOCR initialization took: {time.time() - ocr_start:.2f}s")
    
    # Load YOLO network
    print("Loading YOLO network...")
    yolo_start = time.time()
    network = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    print(f"YOLO loading took: {time.time() - yolo_start:.2f}s")
    layers_names_all = network.getLayerNames()
    layers_names_output = [layers_names_all[i-1] for i in network.getUnconnectedOutLayers()]

    # Load image
    print("Loading image...")
    img_start = time.time()
    image = cv2.imread(image_path)
    if image is None:
        return "Error: Could not read image."
    print(f"Image loading took: {time.time() - img_start:.2f}s")

    # YOLO inference
    print("Running YOLO detection...")
    detect_start = time.time()
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    network.setInput(blob)
    outs = network.forward(layers_names_output)
    print(f"YOLO inference took: {time.time() - detect_start:.2f}s")

    boxes, confs = [], []
    for out in outs:
        for det in out:
            scores = det[5:]
            conf = scores[np.argmax(scores)]
            if conf > 0.2:
                box = det[0:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype('int')
                boxes.append([int(centerX - (width/2)), int(centerY - (height/2)), int(width), int(height)])
                confs.append(float(conf))

    indices = cv2.dnn.NMSBoxes(boxes, confs, 0.2, 0.1)
    if len(indices) > 0:
        idx = indices.flatten()[0]
        x, y, bw, bh = boxes[idx]
        roi = image[max(0,y):y+bh, max(0,x):x+bw]
        print(f"Plate detected at: ({x}, {y}), size: {bw}x{bh}")

        if roi.size > 0:
            # Preprocessing for OCR
            preprocess_start = time.time()
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            print(f"Preprocessing took: {time.time() - preprocess_start:.2f}s")
            
            # OCR Extraction using EasyOCR
            print("Running OCR...")
            ocr_extract_start = time.time()
            results = reader.readtext(gray)
            print(f"OCR extraction took: {time.time() - ocr_extract_start:.2f}s")
            if results:
                # Extract text with highest confidence
                plate_number = ' '.join([text for (_, text, _) in results])
                total_time = time.time() - start_time
                print(f"\nTotal execution time: {total_time:.2f}s")
                return plate_number.strip()
            total_time = time.time() - start_time
            print(f"\nTotal execution time: {total_time:.2f}s")
            return "No text detected."
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f}s")
    return "No plate detected."

if __name__ == '__main__':
    # Update these paths to the absolute paths on your local machine
    IMG_PATH = 'D:\\projects\\ANPR_POC\\inputs\\images.jpg'
    CFG_PATH = 'D:\\projects\\ANPR_POC\\yolov3\\darknet-yolov3.cfg'
    WEIGHTS_PATH = 'D:\\projects\\ANPR_POC\\yolov3\\lapi.weights'

    print("="*50)
    print("Starting License Plate Recognition")
    print("="*50)
    result = extract_plate_text(IMG_PATH, CFG_PATH, WEIGHTS_PATH)
    print("\n" + "="*50)
    print(f"Result: {result}")
    print("="*50)
 