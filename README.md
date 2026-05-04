# ANPR POC - Automatic Number Plate Recognition

A Proof of Concept for Automatic Number Plate Recognition (ANPR) using YOLOv8 for license plate detection and fast-plate-ocr for character recognition.

## Features

- **License Plate Detection**: YOLOv8-based custom-trained model for accurate plate detection
- **OCR Recognition**: Fast-plate-ocr with CCT-S-v2 global model for text extraction
- **Batch Processing**: Process multiple images from a folder
- **Performance Tracking**: Individual processing time for each image
- **JSON Output**: Structured output with filename and extracted plate text
- **Confidence Filtering**: Configurable confidence threshold (default: 0.4)

## Requirements

- Python 3.8+
- CPU or GPU (both supported via ONNX Runtime)

## Installation

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

## Project Structure

```
ANPR_POC/
├── Vehicle License Plate List/  # Input images for detection
├── main.py                      # Main detection script
├── requirements.txt             # Project dependencies
├── README.md                    # Project documentation
└── license_plate_detector.pt   # YOLOv8 trained model (not included)
```

## Configuration

Before running the script, update the paths in [main.py](main.py):

```python
model_path = r"D:\\projects\\ANPR_POC\\license_plate_detector.pt"
input_folder = r"D:\\projects\\ANPR_POC\\Vehicle License Plate List"
```

## Usage

### Basic Usage

Run the main script to process all images in the input folder:

```bash
python main.py
```

### Output

The script generates two types of output:

1. **Console Output**: Real-time processing status with timing
   ```
   image1.jpg -> ['ABC1234'] | Time: 0.523s
   image2.jpg -> ['XYZ5678', 'DEF9012'] | Time: 0.645s
   ```

2. **JSON File**: `plate_results.json` containing structured results
   ```json
   {
       "image1.jpg": ["ABC1234"],
       "image2.jpg": ["XYZ5678", "DEF9012"]
   }
   ```

### How It Works

1. **Detection Phase**: 
   - YOLOv8 model scans each image for license plates
   - Bounding boxes with confidence > 0.4 are retained

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