# ANPR POC - Automatic Number Plate Recognition

A Proof of Concept for Automatic Number Plate Recognition (ANPR) using YOLOv8 and OCR technologies.

## Features

- License plate detection using YOLOv8
- OCR for character recognition using EasyOCR and Pytesseract
- Support for both YOLOv3 and YOLOv8 models
- Custom trained model for Indian number plates

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
ANPR_POC/
├── dataset/           # Training, validation, and test datasets
├── scripts/           # Training scripts
├── inputs/            # Input images for detection
├── output/            # Detection results
├── main.py            # Main detection script
├── yolov8_finetune.py # Fine-tuning script
└── requirements.txt   # Project dependencies
```

## Usage

### Training

```bash
python scripts/training.py
```

### Detection

```bash
python main.py
```

## Models

- YOLOv8n - Base model for license plate detection
- Custom fine-tuned model for improved accuracy on Indian vehicles

## Technologies Used

- **YOLOv8** (Ultralytics) - Object detection
- **EasyOCR** - Optical character recognition
- **Pytesseract** - Alternative OCR engine
- **OpenCV** - Image processing
- **PyTorch** - Deep learning framework

## License

This is a Proof of Concept project.

## Author

Shivam Garg
