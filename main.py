import cv2
import time
import os
import json
import shutil
import tempfile
from fast_alpr import ALPR
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

load_dotenv()

# ─── Load Models ────────────────────────────────────────────────────────────────
input_folder = os.getenv("INPUT_FOLDER", "Vehicle License Plate List")

# Initialize ALPR with default models (handles both detection and OCR)
alpr = ALPR(
    detector_model="yolo-v9-s-608-license-plate-end2end",
    ocr_model="cct-s-v2-global-model"
)

print("ALPR model loaded")

# ─── FastAPI App ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ANPR API",
    description="Automatic Number Plate Recognition — Single Image & Folder Processing",
    version="1.0.0"
)


# ─── Core Logic ─────────────────────────────────────────────────────────────────

def read_plate(image_path: str) -> dict:
    """
    Read license plate using fast_alpr (handles both detection and OCR).
    """
    start_time = time.time()
    img = cv2.imread(image_path)

    if img is None:
        return {"error": f"Failed to read image: {image_path}", "plates": []}

    plates = []

    try:
        results = alpr.predict(img)
        for result in results:
            if result.ocr and result.ocr.text:
                plates.append(result.ocr.text)
                
    except Exception as e:
        print(f"ALPR error in {image_path}: {e}")

    elapsed = round(time.time() - start_time, 3)
    print(f"{os.path.basename(image_path)} -> {plates} | Time: {elapsed}s")

    return {
        "file": os.path.basename(image_path),
        "plates": plates,
        "processing_time_sec": elapsed
    }


def process_folder(folder_path: str) -> dict:
    supported_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    results = {}
    overall_start = time.time()

    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(supported_ext):
            image_path = os.path.join(folder_path, file_name)
            result = read_plate(image_path)
            results[file_name] = result["plates"]

    total_time = round(time.time() - overall_start, 3)
    avg_time = round(total_time / max(len(results), 1), 3)

    return {
        "total_images": len(results),
        "total_time_sec": total_time,
        "avg_time_per_image_sec": avg_time,
        "results": results
    }


# ─── API Routes ──────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "ANPR API is running 🚗", "version": "1.0.0"}


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
        result["file"] = file.filename  # Show original filename
        return JSONResponse(content=result)
    finally:
        os.unlink(tmp_path)  # Clean up temp file


# ── Route 2: Process Entire Folder ───────────────────────────────────────────────
@app.post("/process-folder", summary="Process all images in a specified folder")
def process_folder_api(folder_path: str = None):
    """
    Process all images in the specified folder.
    
    Args:
        folder_path: Path to folder containing images. If not provided, uses INPUT_FOLDER env variable.
    
    Example:
        POST /process-folder?folder_path=/app/Vehicle License Plate List
        POST /process-folder (uses default INPUT_FOLDER)
    """
    target_folder = folder_path if folder_path else input_folder
    
    if not os.path.exists(target_folder):
        raise HTTPException(
            status_code=404,
            detail=f"Folder not found: '{target_folder}'. Please provide a valid folder_path parameter."
        )

    result = process_folder(target_folder)

    # Save results to JSON
    output_json = "plate_results.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    result["saved_to"] = output_json
    result["folder_processed"] = target_folder
    return JSONResponse(content=result)


# ─── Run ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)