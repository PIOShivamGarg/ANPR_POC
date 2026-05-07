import cv2
import time
import os
import shutil
import tempfile
from fast_alpr import ALPR
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

# ─── Load Models ────────────────────────────────────────────────────────────────

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
                plates.append({
                    "text": result.ocr.text,
                    "region": result.ocr.region
                })
                
    except Exception as e:
        print(f"ALPR error in {image_path}: {e}")

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


# ─── Run ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)