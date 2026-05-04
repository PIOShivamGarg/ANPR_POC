# ANPR POC - Automatic Number Plate Recognition

A Proof of Concept for Automatic Number Plate Recognition (ANPR) using YOLOv8 for license plate detection and fast-plate-ocr for character recognition. Now includes a **REST API** for easy integration.

## Features

- **🚀 REST API**: FastAPI-based endpoints for single image and batch processing
- **📷 License Plate Detection**: YOLOv8-based custom-trained model for accurate plate detection
- **🔍 OCR Recognition**: Fast-plate-ocr with CCT-S-v2 global model for text extraction
- **📁 Batch Processing**: Process multiple images from a folder via API or script
- **⏱️ Performance Tracking**: Individual processing time for each image
- **📄 JSON Output**: Structured output with filename and extracted plate text
- **🎯 Confidence Filtering**: Configurable confidence threshold (default: 0.4)
- **🐳 Docker Ready**: Containerized deployment with volume mounting

## Requirements

- Python 3.8+
- CPU or GPU (both supported via ONNX Runtime)

## Installation

### Option 1: Local Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ANPR_POC
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure you have the YOLOv8 model file:
   - Place `license_plate_detector.pt` in the project root directory

### Option 2: Docker Installation

**Important:** The Docker image includes all images from the `Vehicle License Plate List` folder at build time.

1. Build the Docker image (this copies all your images into the container):
```bash
docker build -t anpr-poc .
```

2. Run the API server (no volume mounting needed):

**Simple run:**
```bash
docker run -p 8000:8000 anpr-poc
```

The API will be available at `http://localhost:8000`

**Note:** If you add new images, you need to rebuild the Docker image with `docker build -t anpr-poc .`

**To add/update images without rebuilding (alternative approach):**
```bash
# Mount the folder as before
docker run -p 8000:8000 \
  -v "$(pwd)/Vehicle License Plate List:/app/Vehicle License Plate List" \
  anpr-poc
```

## Project Structure

```
ANPR_POC/
├── Vehicle License Plate List/  # Input images for detection
├── main.py                      # FastAPI server + detection logic
├── requirements.txt             # Project dependencies
├── Dockerfile                   # Docker container configuration
├── .dockerignore                # Docker build exclusions
├── .env                         # Environment variables (optional)
├── README.md                    # Project documentation
└── license_plate_detector.pt   # YOLOv8 trained model (not included)
```

## Configuration

The script uses environment variables for configuration with sensible defaults:

- **MODEL_PATH**: Path to YOLOv8 model (default: `license_plate_detector.pt`)
- **INPUT_FOLDER**: Path to input images folder (default: `Vehicle License Plate List`)

**Setting environment variables:**

Linux/Mac:
```bash
export MODEL_PATH="/path/to/model.pt"
export INPUT_FOLDER="/path/to/images"
python main.py
```

Windows PowerShell:
```powershell
$env:MODEL_PATH="C:\path\to\model.pt"
$env:INPUT_FOLDER="C:\path\to\images"
python main.py
```

Docker:
```bash
docker run -e MODEL_PATH="/app/custom_model.pt" -e INPUT_FOLDER="/app/images" anpr-poc
```

## Usage

### Option 1: REST API Server (Recommended)

Start the API server:

```bash
python main.py
```

The API will be available at `http://localhost:8000`

#### API Endpoints

**1. Health Check**
```bash
GET http://localhost:8000/health
```

**2. Process Single Image**
```bash
POST http://localhost:8000/read-plate
Content-Type: multipart/form-data

# Example with curl
curl -X POST "http://localhost:8000/read-plate" \
     -F "file=@path/to/image.jpg"
```

Response:
```json
{
  "file": "image.jpg",
  "plates": ["ABC1234"],
  "processing_time_sec": 0.523
}
```

**3. Process Entire Folder**
```bash
POST http://localhost:8000/process-folder

# Default - uses built-in images
POST http://localhost:8000/process-folder

# Custom path (if you mounted a different folder)
POST http://localhost:8000/process-folder?folder_path=/path/to/images

# Example with curl (default - built-in images)
curl -X POST "http://localhost:8000/process-folder"

# Example with curl (custom folder if mounted)
curl -X POST "http://localhost:8000/process-folder?folder_path=/app/Vehicle%20License%20Plate%20List"
```

Response:
```json
{
  "total_images": 18,
  "total_time_sec": 3.856,
  "avg_time_per_image_sec": 0.214,
  "results": {
    "image1.jpg": ["ABC1234"],
    "image2.jpg": ["XYZ5678", "DEF9012"]
  },
  "saved_to": "plate_results.json",
  "folder_processed": "/app/Vehicle License Plate List"
}
```

**Interactive API Documentation**: Visit `http://localhost:8000/docs` for Swagger UI

### Option 2: Python Import

Use as a module in your own code:

```python
from main import read_plate, process_folder

# Single image
result = read_plate("path/to/image.jpg")
print(result)  # {"file": "image.jpg", "plates": [...], "processing_time_sec": 0.5}

# Folder
results = process_folder("path/to/folder")
print(results)  # {"total_images": 10, "results": {...}}
```

### How It Works

1. **Detection Phase**: 
   - YOLOv8 model scans each image for license plates
   - Bounding boxes with confidence > 0.4 are retained

2. **Recognition Phase**:
   - Each detected plate region is cropped
   - fast-plate-ocr (default folder)
curl -X POST "http://localhost:8000/process-folder"

# Folder processing (custom path)
curl -X POST "http://localhost:8000/process-folder?folder_path=/app/Vehicle%20License%20Plate%20List"
```

### Using Python Requests

```python
import requests

# Single image
with open("image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/read-plate",
        files={"file": f}
    )
    print(response.json())

# Folder processing (default folder)
response = requests.post("http://localhost:8000/process-folder")
print(response.json())

# Folder processing (custom path)
response = requests.post(
    "http://localhost:8000/process-folder",
    params={"folder_path": "/path/to/images"}

**Supported image formats**: JPG, JPEG, PNG, BMP, TIFF

## Technologies Used

- **FastAPI** - Modern REST API framework
- **Uvicorn** - ASGI server for production
- **YOLOv8** (Ultralytics) - License plate object detection
- **fast-plate-ocr** - OCR for license plate text recognition with CCT-S-v2 model
- **ONNX Runtime** - Inference engine for OCR model
- **OpenCV** - Image processing and manipulation
- **PyTorch** - Deep learning framework for YOLO
- **python-dotenv** - Environment variable management

## Performance

- Average processing time: ~0.5-1s per image (CPU)
- Detection confidence threshold: 0.4
- Supports multiple plates per image

## API Testing

### Using cURL

```bash
# Single image
curl -X POST "http://localhost:8000/read-plate" \
     -F "file=@./Vehicle License Plate List/image.jpg"

# Folder processing

## Future Enhancements

- ✅ ~~API endpoint for web integration~~ (Completed)
- Real-time video processing
- WebSocket support for live streaming
- Support for multiple regions/countries
- Database integration for plate tracking
- Batch upload endpoint for multiple images
- Authentication & rate limiting

## License

This is a Proof of Concept project.

## Authors

Shivam Garg, Dev Tekwani
### Using Python Requests

```python
im✅ ~~API endpoint for web integration~~ (Completed)
- Real-time video processing
- WebSocket support for live streaming
- Support for multiple regions/countries
- Database integration for plate tracking
- Batch upload endpoint for multiple images
- Authentication & rate limit
    response = requests.post(
        "http://localhost:8000/read-plate",
        files={"file": f}
    )
    print(response.json())

# Folder processing
response = requests.post("http://localhost:8000/process-folder")
print(response.json())
```

### Using Postman

1. **Single Image Upload**:
   - Method: `POST`
   - URL: `http://localhost:8000/read-plate`
   - Body: `form-data` → Key: `file` (type: File) → Select image file

2. **Folder Processing**:
   - Method: `POST`
   - URL: `http://localhost:8000/process-folder`
   - Params (optional): `folder_path` = `/path/to/your/images`

## Troubleshooting

**Model not loading?**
- Verify `license_plate_detector.pt` exists at the specified path
- Ensure the model is compatible with your Ultralytics version

**OCR errors?**
- Check that the plate crop is valid and not empty
- Try adjusting the confidence threshold

**Slow processing?**
- Enable GPU: Change providers to `["CUDAExecutionProvider"]`
- Ensure CUDA is properly installed for GPU acceleration

**API not starting?**
- Check if port 8000 is already in use
- Verify all dependencies are installed: `pip install -r requirements.txt`

- Ensure `Vehicle License Plate List` folder exists with images

**Docker API returns 0 images?**
- Verify images were copied during build: `docker exec -it <container_id> ls "/app/Vehicle License Plate List"`
- If you added new images after building, rebuild the image: `docker build -t anpr-poc .`
- Alternatively, use volume mounting to access live files (see installation section)
- Ensure you mounted the volume with `-v` flag when running `docker run`
- Verify the mounted path: `docker exec -it <container_id> ls "/app/Vehicle License Plate List"`
- Or use `folder_path` parameter: `POST /process-folder?folder_path=/app/images`
2. **Recognition Phase**:
   - Each detected plate region is cropped
   - fast-plate-ocr (CCT-S-v2) extracts text from the cropped region

3. **Results**:
   - Processing time logged for each image
   - All results saved to JSON file

### Customization

**Adjust confidence threshold** in [main.py](main.py):
```python
if conf < 0.4:  # Change 0.4 to your desired threshold
    continue
```

**Change OCR model**:
```python
ocr_model = LicensePlateRecognizer(
    "cct-s-v2-global-model",  # or other supported models
    providers=["CPUExecutionProvider"]  # or ["CUDAExecutionProvider"] for GPU
)
```

**Supported image formats**: JPG, JPEG, PNG, BMP, TIFF

## Technologies Used

- **YOLOv8** (Ultralytics) - License plate object detection
- **fast-plate-ocr** - OCR for license plate text recognition with CCT-S-v2 model
- **ONNX Runtime** - Inference engine for OCR model
- **OpenCV** - Image processing and manipulation
- **PyTorch** - Deep learning framework for YOLO

## Performance

- Average processing time: ~0.5-1s per image (CPU)
- Detection confidence threshold: 0.4
- Supports multiple plates per image

## Troubleshooting

**Model not loading?**
- Verify `license_plate_detector.pt` exists at the specified path
- Ensure the model is compatible with your Ultralytics version

**OCR errors?**
- Check that the plate crop is valid and not empty
- Try adjusting the confidence threshold

**Slow processing?**
- Enable GPU: Change providers to `["CUDAExecutionProvider"]`
- Ensure CUDA is properly installed for GPU acceleration

## Future Enhancements

- Real-time video processing
- API endpoint for web integration
- Support for multiple regions/countries
- Database integration for plate tracking

## License

This is a Proof of Concept project.

## Authors

Shivam Garg, Dev Tekwani