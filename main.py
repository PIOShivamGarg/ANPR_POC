import os
import shutil
import tempfile
from fast_alpr import ALPR
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from countrystatecity_countries import get_countries, get_states_of_country


# ─── Load Models ────────────────────────────────────────────────────────────────
alpr = ALPR(
    detector_model="yolo-v9-s-608-license-plate-end2end"
)
print("ALPR model loaded")


# ─── Load All World State Names ──────────────────────────────────────────────────
print("Loading all world states...")
_all_states = []
for country in get_countries():
    states = get_states_of_country(country.iso2)
    _all_states.extend(states)

ALL_STATE_NAMES = {state.name.upper(): state.name for state in _all_states}
print(f"Loaded {len(ALL_STATE_NAMES)} states/regions from all countries")


# ─── FastAPI App ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ANPR API",
    description="Automatic Number Plate Recognition — Detection via Azure Computer Vision or PaddleOCR-VL",
    version="1.0.0"
)


# ─── API Routes ──────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "ANPR API is running 🚗", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "ok"}


from azure_cv import read_plate_azure_cv

# ── Route 1: Read Plate through Azure CV ─────────────────────────────────────────────
@app.post("/read-plate-azure-cv", summary="Upload a single image to detect license plate using Azure CV")
async def read_plate_azure_cv_api(file: UploadFile = File(...)):
    supported_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

    if not file.filename.lower().endswith(supported_ext):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {supported_ext}"
        )

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=os.path.splitext(file.filename)[1]
    ) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = read_plate_azure_cv(tmp_path, alpr, ALL_STATE_NAMES)
        result["file"] = file.filename
        return JSONResponse(content=result)
    finally:
        os.unlink(tmp_path)


from paddleocr_vl import read_plate_paddleocr_vl

# ── Route 2: Read Plate through PaddleOCR-VL ──────────────────────────────────────────────
@app.post("/read-plate-paddleocr-vl", summary="Upload a single image to detect license plate using PaddleOCR-VL")
async def read_plate_paddleocr_vl_api(file: UploadFile = File(...)):
    supported_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

    if not file.filename.lower().endswith(supported_ext):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {supported_ext}",
        )

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=os.path.splitext(file.filename)[1],
    ) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = read_plate_paddleocr_vl(tmp_path, alpr, ALL_STATE_NAMES)
        result["file"] = file.filename
        return JSONResponse(content=result)
    finally:
        os.unlink(tmp_path)
