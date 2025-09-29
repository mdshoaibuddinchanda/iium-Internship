# License Plate Character Break Detection

This project analyzes paired vehicle images (front and rear) to detect broken or damaged license plate characters using PaddleOCR.

## What this does
- Processes front and rear vehicle images
- Uses PaddleOCR to extract license plate text and character locations
- Analyzes character integrity using pixel density analysis
- Generates a detailed CSV report with character-level analysis
- Compares front and rear plates for mismatches

## Files Structure
```
├── Q1_code.py              # Main analysis script
├── README.md               # This file
├── data/
│   ├── front/             # Front vehicle images (jpg/png)
│   └── rear/              # Rear vehicle images (jpg/png)
├── output/                # Generated results (empty initially)
├── plate_analysis.csv     # Output CSV report
└── license_plate_results.csv  # Alternative output file
```

## Dependencies Required
Install the following Python packages:
```bash
pip install paddleocr opencv-python numpy
```

## How it works
1. **Image Preprocessing**: Converts images to grayscale and applies adaptive thresholding
2. **OCR Processing**: Uses PaddleOCR to detect text and character bounding boxes
3. **Character Analysis**: Crops individual characters and analyzes pixel density
4. **Integrity Assessment**: Classifies characters as "Intact", "Broken/Missing", or "Missing"
5. **Comparison**: Checks for text mismatches between front and rear plates
6. **Report Generation**: Creates detailed CSV with character-level results

## CSV Output Format
The generated `plate_analysis.csv` contains:
- **Car_ID**: Image filename
- **Image**: "Front" or "Rear"
- **OCR_Text**: Complete extracted license plate text
- **Character**: Individual character or character group
- **Integrity**: Character status ("Intact", "Broken/Missing", "Missing")
- **Mismatch**: "Yes" if front/rear plates don't match, "No" otherwise

## Steps to Run the Code
1. **Install dependencies**: 
   ```bash
   pip install paddleocr opencv-python numpy
   ```

2. **Place your images in the correct folders**:
   - Front images: `data/front/`
   - Rear images: `data/rear/`
   - Ensure matching filenames (e.g., `car001.jpg` in both folders)

3. **Execute the script**:
   ```bash
   python Q1_code.py
   ```

4. **Review results in `plate_analysis.csv`**

## Dependencies Required
- **PaddleOCR**: For optical character recognition
- **OpenCV**: For image processing and preprocessing
- **NumPy**: For numerical operations and pixel analysis

## Assumptions Made
- Front and rear images have matching filenames
- Images contain visible license plates
- Character integrity is determined by pixel density (threshold: 25%)
- PaddleOCR can successfully detect text regions

## Algorithm Details
- **Adaptive Thresholding**: Handles varying lighting conditions (kernel size: 35, C: 11)
- **Pixel Density Analysis**: Characters with <25% filled pixels are marked as broken
- **Bounding Box Validation**: Ensures valid character regions before analysis
- **Error Handling**: Gracefully handles missing images and OCR failures

## Sample Results
The system successfully processes images and generates detailed analysis:
- Detects license plate text like "JrS3220", "JKB 39 2", "WD 7922C"
- Identifies character integrity issues
- Compares front vs rear plate consistency
- Provides character-level breakdown for forensic analysis

## Troubleshooting
- **Empty CSV**: Ensure images contain readable license plates
- **Missing images**: Check that front/rear folders have matching filenames
- **OCR errors**: Verify image quality and lighting conditions
- **Model downloads**: First run downloads PaddleOCR models (requires internet)

## Performance Notes
- First run may be slower due to model downloads
- Processing time depends on image resolution and number of characters
- Results accuracy depends on image quality and lighting conditions
- GPU acceleration available if CUDA is installed