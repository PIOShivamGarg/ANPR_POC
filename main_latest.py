import cv2
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

# ─── Load Environment Variables ─────────────────────────────────────────────────
load_dotenv()

# ─── Load Models ────────────────────────────────────────────────────────────────

# Initialize ALPR for detection only
alpr = ALPR(
    detector_model="yolo-v9-s-608-license-plate-end2end"
)

# Initialize Azure Computer Vision client for OCR
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


# ─── Core Logic ─────────────────────────────────────────────────────────────────

def extract_text_from_image_azure_cv(image_crop) -> str:
    """
    Extract text from a cropped license plate image using Azure Computer Vision.
    """
    try:
        # Encode image to bytes
        _, buffer = cv2.imencode('.jpg', image_crop)
        image_data = buffer.tobytes()
        
        # Call Azure Computer Vision API
        result = vision_client.analyze(
            image_data=image_data,
            visual_features=[VisualFeatures.READ]
        )
        
        # Extract all text lines
        text_lines = []
        if result.read is not None:
            for block in result.read.blocks:
                for line in block.lines:
                    text_lines.append(line.text)
        
        # Join all lines (usually license plates are single line)
        return " ".join(text_lines).strip()
    
    except Exception as e:
        print(f"Azure CV OCR error: {e}")
        return ""


def read_plate(image_path: str) -> dict:
    """
    Read license plate using fast_alpr for detection and Azure CV for OCR.
    """
    start_time = time.time()
    img = cv2.imread(image_path)

    if img is None:
        return {"error": f"Failed to read image: {image_path}", "plates": []}

    plates = []

    try:
        # Step 1: Use fast_alpr for detection only
        results = alpr.predict(img)
        
        for result in results:
            # Get bounding box coordinates from detection
            if not result.detection:
                continue
            
            # Access bounding_box through result.detection
            bbox = result.detection.bounding_box
            x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
            
            # Step 2: Crop the detected plate region
            plate_crop = img[y1:y2, x1:x2]
            
            # Step 3: Use Azure Computer Vision for OCR
            plate_text = extract_text_from_image_azure_cv(plate_crop)
            
            if plate_text:  # Only add if text was detected
                plates.append({
                    "text": plate_text,
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


# ── Route 1: Single Image Upload ─────────────────────────────────────────────────
@app.post("/read-plate", summary="Upload a single image to detect license plate")
async def read_plate_api(file: UploadFile = File(...)):
    supported_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

    if not file.filename.lower().endswith(supported_ext):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {supported_ext}"
        )

    # Save uploaded file temporarily
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
