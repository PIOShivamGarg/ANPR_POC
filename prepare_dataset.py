"""
Script to prepare a dataset CSV from license plate images organized by state folders.
Extracts plate text using fast_alpr OCR and creates a CSV with:
- Image name
- Number plate text
- State name (from folder)
- Region name (always United States)
"""

import os
import csv
from pathlib import Path
import cv2
from fast_alpr import ALPR
from tqdm import tqdm

def prepare_dataset_csv(data_folder="data", output_csv="dataset.csv", 
                       detector_model="yolo-v9-c-384-license-plate-end2end",
                       ocr_model="cct-s-v2-global-model"):
    """
    Process all images in state folders and create a CSV dataset.
    
    Args:
        data_folder: Path to the folder containing state subfolders
        output_csv: Output CSV file name
        detector_model: FastALPR detector model to use
        ocr_model: FastALPR OCR model to use
    """
    
    # Initialize ALPR
    print(f"Initializing FastALPR with detector: {detector_model}, OCR: {ocr_model}")
    alpr = ALPR(detector_model=detector_model, ocr_model=ocr_model)
    
    # Prepare data structure
    dataset_records = []
    
    # Get all state folders
    data_path = Path(data_folder)
    if not data_path.exists():
        print(f"Error: Data folder '{data_folder}' does not exist!")
        return
    
    state_folders = [f for f in data_path.iterdir() if f.is_dir()]
    
    if not state_folders:
        print(f"Warning: No state folders found in '{data_folder}'")
        return
    
    print(f"Found {len(state_folders)} state folders")
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Process each state folder
    total_images = 0
    successful_extractions = 0
    failed_extractions = 0
    
    for state_folder in tqdm(sorted(state_folders), desc="Processing states"):
        state_name = state_folder.name
        
        # Get all images in this state folder
        image_files = [
            f for f in state_folder.iterdir() 
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        
        print(f"\nProcessing {state_name}: {len(image_files)} images")
        
        for image_file in tqdm(image_files, desc=f"  {state_name}", leave=False):
            total_images += 1
            image_name = image_file.name
            
            try:
                # Read image
                image = cv2.imread(str(image_file))
                
                if image is None:
                    print(f"  Warning: Could not read image {image_name}")
                    failed_extractions += 1
                    dataset_records.append({
                        'image_name': image_name,
                        'plate_text': 'ERROR: Could not read image',
                        'state_name': state_name,
                        'region_name': 'United States'
                    })
                    continue
                
                # Run ALPR detection and OCR
                results = alpr.predict(image)
                
                # Extract plate text
                if results and len(results) > 0:
                    # Get the first detected plate (assuming one plate per image)
                    plate_text = results[0].ocr.text if results[0].ocr else 'No OCR result'
                    
                    # If multiple plates detected, join them with semicolon
                    if len(results) > 1:
                        plate_texts = [
                            r.ocr.text if r.ocr else 'N/A' 
                            for r in results
                        ]
                        plate_text = '; '.join(plate_texts)
                        print(f"  Info: Multiple plates detected in {image_name}: {plate_text}")
                    
                    successful_extractions += 1
                else:
                    plate_text = 'No plate detected'
                    failed_extractions += 1
                
                # Add record to dataset
                dataset_records.append({
                    'image_name': image_name,
                    'plate_text': plate_text,
                    'state_name': state_name,
                    'region_name': 'United States'
                })
                
            except Exception as e:
                print(f"  Error processing {image_name}: {str(e)}")
                failed_extractions += 1
                dataset_records.append({
                    'image_name': image_name,
                    'plate_text': f'ERROR: {str(e)}',
                    'state_name': state_name,
                    'region_name': 'United States'
                })
    
    # Write to CSV
    if dataset_records:
        print(f"\nWriting {len(dataset_records)} records to {output_csv}")
        
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['image_name', 'plate_text', 'state_name', 'region_name']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerows(dataset_records)
        
        print(f"\n{'='*60}")
        print(f"Dataset CSV created successfully: {output_csv}")
        print(f"{'='*60}")
        print(f"Total images processed: {total_images}")
        print(f"Successful extractions: {successful_extractions} ({successful_extractions/total_images*100:.1f}%)")
        print(f"Failed extractions: {failed_extractions} ({failed_extractions/total_images*100:.1f}%)")
        print(f"{'='*60}")
    else:
        print("No records to write. Please check your data folder structure.")

if __name__ == "__main__":
    # You can modify these parameters as needed
    prepare_dataset_csv(
        data_folder="data",
        output_csv="license_plate_dataset.csv",
        detector_model="yolo-v9-t-384-license-plate-end2end",
        ocr_model="cct-s-v2-global-model"
    )
