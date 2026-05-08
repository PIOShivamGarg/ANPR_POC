# ANPR POC - Automatic Number Plate Recognition

A Proof of Concept for Automatic Number Plate Recognition (ANPR) using **fast_alpr** for license plate detection and **Azure Computer Vision** for OCR. Includes worldwide state/region recognition and a **REST API** for easy integration.

## Features

- **🚀 REST API**: FastAPI-based endpoints for license plate recognition
- **📷 License Plate Detection**: fast_alpr with YOLO-v9-s-608 end-to-end model
- **🔍 Cloud OCR**: Azure Computer Vision for high-accuracy text extraction
- **🌍 Worldwide State Recognition**: Extracts state/region names from all countries using countrystatecity
- **🔢 Intelligent Plate Extraction**: Regex-based plate number extraction supporting multiple formats (ABC1234, ABC-1234, FNR*8034)
- **⏱️ Performance Tracking**: Individual processing time for each image
- **📄 Structured JSON Output**: Detailed response with plate_number, state, region, and processing time
- **🐳 Docker Ready**: Containerized deployment with docker-compose support

## Requirements

- Python 3.11+
- Azure Computer Vision API credentials
- Internet connection (for Azure CV API calls)

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

3. Configure Azure Computer Vision:
   - Create an Azure Computer Vision resource in [Azure Portal](https://portal.azure.com)
   - Copy `.env.example` to `.env` and fill in your credentials:
   ```bash
   cp .env.example .env
   ```
   - Edit `.env` with your Azure credentials:
   ```
   VISION_ENDPOINT=https://your-resource-name.cognitiveservices.azure.com/
   VISION_KEY=your_azure_computer_vision_key_here
   ```

### Option 2: Docker Installation

**Prerequisites**: Create a `.env` file with Azure credentials (or use `.env.example` as template)

**Method 1: Using Docker Compose (Recommended)**

```bash
# Build and run with docker-compose
docker-compose up --build
```

The API will be available at `http://localhost:8000`

**Method 2: Using Docker CLI**

1. Build the Docker image:
```bash
docker build -t anpr-poc .
```

2. Run the container with environment variables:
```bash
docker run -p 8000:8000 \
  -v "$(pwd)/data:/app/data" \
  --env-file .env \
  anpr-poc
```

**To use custom images:**
```bash
docker run -p 8000:8000 \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/.env:/app/.env" \
  anpr-poc
```

> **Resource Limits**: The docker-compose.yml sets limits to 3 CPUs and 1GB RAM. Adjust as needed for your workload.

## Project Structure

```
ANPR_POC/
├── Vehicle License Plate Samples # License Plates Samples
├── main.py                       # FastAPI server + API endpoints
├── azure_cv.py                   # Azure Computer Vision OCR backend
├── utils.py                      # Shared utilities (state/plate extraction)
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker container configuration
├── docker-compose.yml            # Docker Compose orchestration
├── .env.example                  # Environment variables template
├── .env                          # Azure credentials (create from .env.example)
└── README.md                     # Project documentation
```

## Configuration

The application uses environment variables for Azure Computer Vision configuration:

**Required:**
- **VISION_ENDPOINT**: Azure Computer Vision endpoint URL
- **VISION_KEY**: Azure Computer Vision API key

**Setting environment variables:**

Create a `.env` file in the project root:
```bash
VISION_ENDPOINT=https://your-resource-name.cognitiveservices.azure.com/
VISION_KEY=your_azure_computer_vision_key_here
```

Or set them in your shell:

Linux/Mac:
```bash
export VISION_ENDPOINT="https://your-resource-name.cognitiveservices.azure.com/"
export VISION_KEY="your_key_here"
python main.py
```

Windows PowerShell:
```powershell
$env:VISION_ENDPOINT="https://your-resource-name.cognitiveservices.azure.com/"
$env:VISION_KEY="your_key_here"
python main.py
```

Docker:
```bash
docker run --env-file .env -p 8000:8000 anpr-poc
```

## Usage

### Start the API Server

**Local:**
```bash
uvicorn main:app --reload
# or
python -m uvicorn main:app --reload
```

**Docker:**
```bash
docker-compose up
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### 1. Root - API Information
```bash
GET http://localhost:8000/
```

Response:
```json
{
  "message": "ANPR API is running 🚗",
  "version": "1.0.0"
}
```

#### 2. Health Check
```bash
GET http://localhost:8000/health
```

Response:
```json
{
  "status": "ok"
}
```

#### 3. Read Plate
```bash
POST http://localhost:8000/read-plate-azure-cv
Content-Type: multipart/form-data

# Example with curl
curl -X POST "http://localhost:8000/read-plate-azure-cv" \
     -F "file=@data/California/image1.jpg"
```

Response:
```json
{
  "file": "image1.jpg",
  "plates": [
    {
      "plate_number": "ABC1234",
      "state": "California",
      "region": "us-ca",
      "message": null
    }
  ],
  "processing_time_sec": 0.523
}
```

**Response Fields:**
- `plate_number`: Extracted license plate number (cleaned, no separators)
- `state`: Detected state/region name (worldwide recognition)
- `region`: Region code detected by fast_alpr model
- `message`: Error or info message if plate extraction failed

**Supported image formats**: JPG, JPEG, PNG, BMP, TIFF

**Interactive API Documentation**: Visit `http://localhost:8000/docs` for Swagger UI

### Example Usage

#### Using cURL

```bash
# Test with Azure Computer Vision
curl -X POST "http://localhost:8000/read-plate-azure-cv" \
     -F "file=@data/California/plate1.jpg"
```

#### Using Python Requests

```python
import requests

# Azure Computer Vision endpoint
with open("data/California/plate.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/read-plate-azure-cv",
        files={"file": f}
    )
    print(response.json())
```

#### Using Python httpx (Async)

```python
import httpx
import asyncio

async def process_plate(image_path: str):
    async with httpx.AsyncClient() as client:
        with open(image_path, "rb") as f:
            response = await client.post(
                "http://localhost:8000/read-plate-azure-cv",
                files={"file": f}
            )
        return response.json()

result = asyncio.run(process_plate("data/California/plate.jpg"))
print(result)
```

### How It Works

#### Pipeline Overview

```
┌─────────────────┐
│  Upload Image   │
└────────┬────────┘
         │
         v
┌─────────────────────────────────────────┐
│  1. License Plate Detection             │
│     - fast_alpr (YOLO-v9-s-608)         │
│     - Detects bounding boxes            │
│     - Extracts region code (us-ca, etc) │
└────────┬────────────────────────────────┘
         │
         v
┌─────────────────────────────────────────┐
│  2. Crop Detected Plate Regions         │
│     - Extract ROI from original image   │
└────────┬────────────────────────────────┘
         │
         v
┌─────────────────────────────────────────┐
│  3. OCR on Cropped Plate                │
│     - Azure Computer Vision             │
│     - Returns raw text                  │
└────────┬────────────────────────────────┘
         │
         v
┌─────────────────────────────────────────┐
│  4. Intelligent Text Extraction         │
│     - Plate Number: Regex patterns      │
│       (handles ABC1234, ABC-123, etc)   │
│     - State/Region: countrystatecity    │
│       (worldwide recognition)           │
└────────┬────────────────────────────────┘
         │
         v
┌─────────────────────────────────────────┐
│  5. Return Structured Response          │
│     - plate_number (cleaned)            │
│     - state (proper case)               │
│     - region (ALPR code)                │
│     - processing_time_sec               │
└─────────────────────────────────────────┘
```

#### Detailed Processing Steps

**1. Detection Phase (fast_alpr)**
   - YOLO-v9-s-608-license-plate-end2end model scans the image
   - Detects license plate(s) and returns bounding boxes
   - Provides region hints (e.g., "us-ca" for California)

**2. Cropping Phase**
   - Extracts the detected plate region from the original image using bounding box coordinates

**3. OCR Phase**
   - **Azure Computer Vision**: Cloud-based OCR with high accuracy for text extraction
   - Analyzes the cropped plate image and returns raw text

**4. Extraction Phase**
   - **Plate Number**: Regex patterns match formats like:
     - ABC1234 (no separator)
     - ABC-1234 (hyphen)
     - FNR*8034 (star separator)
   - **State/Region**: Matches OCR text against worldwide state database
     - Multi-word support (e.g., "New York", "New South Wales")
     - Case-insensitive matching
     - `ABC1234` (standard)
     - `ABC-1234` (hyphenated)
     - `FNR*8034` (Texas-style star separator)
     - `ABC·1234` (bullet separator)
   - **State/Region**: Matches text against worldwide database
     - Supports multi-word states (e.g., "New York", "New South Wales")
     - Returns properly cased state names

**5. Response Generation**
   - Combines all extracted information
   - Returns structured JSON with processing time
   - Includes error messages if extraction fails
## Technologies Used

### Core Framework
- **FastAPI** - Modern, high-performance REST API framework
- **Uvicorn** - ASGI server for production deployment

### License Plate Detection
- **fast_alpr** - Fast Automatic License Plate Recognition library
- **YOLO-v9-s-608** - End-to-end license plate detection model
- **ONNX Runtime** - Optimized inference engine

### OCR Engine
- **Azure Computer Vision** - Microsoft's cloud-based OCR service
  - High accuracy for various lighting conditions
  - Robust text extraction from license plates
  - Requires Azure subscription and internet connection

### Image Processing
- **OpenCV** (`opencv-python`) - Image manipulation and preprocessing
- **NumPy** - Numerical operations for image arrays

### Data & Utilities
- **countrystatecity-countries** - Worldwide state/region database
- **python-dotenv** - Environment variable management
- **python-multipart** - File upload handling in FastAPI

### Deployment
- **Docker** - Containerization
- **Docker Compose** - Multi-container orchestration

## Performance

### Processing Times

| Backend | Hardware | Avg Time per Image | Notes |
|---------|----------|-------------------|-------|
| Azure CV | Cloud | 0.5-1.0s | Network latency dependent |

### Accuracy Features

- **Detection**: fast_alpr with YOLO-v9-s-608 provides robust plate detection
- **Multi-plate support**: Handles multiple license plates in a single image
- **Worldwide recognition**: State extraction supports all countries
- **Format flexibility**: Recognizes various plate formats (hyphenated, star-separated, etc.)

### Optimizations

- **Fast detection**: YOLO-v9-s-608 model provides quick plate localization
- **Cloud OCR**: Azure CV handles complex lighting and plate variations
- **Regex filtering**: Intelligent plate number extraction with noise filtering
- **Worldwide recognition**: Supports state/region extraction from all countries

## Error Handling

The API provides detailed error messages in the response:

```json
{
  "file": "image.jpg",
  "plates": [
    {
      "plate_number": null,
      "state": null,
      "region": "us-ca",
      "message": "OCR failed to extract text from detected plate"
    }
  ],
  "processing_time_sec": 0.342
}
```

**Common error scenarios:**
- No plate detected in image
- Failed to crop detected plate region
- OCR failed to extract text
- No valid plate number pattern found

## Troubleshooting

### Azure CV Issues

**Problem**: `VISION_ENDPOINT and VISION_KEY must be set`
- **Solution**: Create `.env` file with your Azure credentials (see Configuration section)

**Problem**: `401 Unauthorized` from Azure
- **Solution**: Verify your `VISION_KEY` is correct in Azure Portal

### Docker Issues

**Problem**: Container exits immediately
- **Solution**: Check `.env` file exists and has valid credentials
- Check logs: `docker-compose logs`

**Problem**: Out of memory errors
- **Solution**: Increase memory limit in `docker-compose.yml` (default: 1GB)

## Model Information

### Detection Model: fast_alpr
- **Model**: `yolo-v9-s-608-license-plate-end2end`
- **Framework**: ONNX Runtime
- **Input**: Full images
- **Output**: Bounding boxes + region codes

### OCR Model

#### Azure Computer Vision
- **Type**: Cloud-based REST API
- **Provider**: Microsoft Azure Cognitive Services
- **API Version**: Latest (Read API)
- **Strengths**: 
  - High accuracy across various lighting conditions
  - Handles different plate formats and styles
  - Pre-trained on diverse text datasets
  - No local GPU required
- **Limitations**: 
  - Requires internet connection
  - API usage costs (pay-per-call)
  - Network latency affects response time

## Dataset

The project includes a comprehensive test samples in 'Vehicle License Plate Samples' folder.


## API Response Schema

```typescript
interface PlateResponse {
  file: string;                    // Uploaded filename
  plates: Plate[];                 // Array of detected plates
  processing_time_sec: number;     // Total processing time
}

interface Plate {
  plate_number: string | null;     // Extracted plate (e.g., "ABC1234")
  state: string | null;            // Detected state (e.g., "California")
  region: string | null;           // ALPR region code (e.g., "us-ca")
  message: string | null;          // Error/info message if extraction failed
}
```

## Future Enhancements

- [ ] **Batch Processing**: Multi-file upload endpoint
- [ ] **Video Processing**: Real-time video stream analysis
- [ ] **WebSocket Support**: Live streaming capabilities
- [ ] **Database Integration**: Store and track detected plates
- [ ] **Confidence Scores**: Return OCR confidence levels
- [ ] **Custom Training**: Fine-tune models for specific regions
- [ ] **Rate Limiting**: API throttling and quotas
- [ ] **Authentication**: JWT-based API access control
- [ ] **Caching**: Redis cache for frequent queries
- [ ] **Webhook Support**: Push notifications for detected plates
- [ ] **Export Options**: CSV, PDF report generation
- [ ] **Analytics Dashboard**: Web UI for statistics and insights

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This is a Proof of Concept project for demonstration purposes.


## Acknowledgments

- **fast_alpr**: For robust license plate detection
- **Microsoft Azure**: For Computer Vision API
- **countrystatecity**: For worldwide region database
- **FastAPI**: For the excellent web framework

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