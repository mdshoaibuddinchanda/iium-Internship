# 🚗 Computer Vision & AI Assignment Collection

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10+-green.svg)](https://opencv.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/mdshoaibuddinchanda/iium-internship?style=social)](https://github.com/mdshoaibuddinchanda/iium-internship)

**A comprehensive collection of advanced computer vision and AI assignments featuring state-of-the-art models and algorithms**

[🚀 Quick Start](#-quick-start) • [📖 Documentation](#-assignment-details) • [🎯 Features](#-key-features) • [🛠️ Installation](#-installation) • [📊 Results](#-performance-results)

---

## 🎯 Key Features

<table>
<tr>
<td width="50%">

### 🧠 **Advanced AI Models**

* Multi-model consensus approach
* YOLOv8 object detection
* BLIP Vision-Language Model
* MediaPipe face analysis
* Custom PyTorch models

</td>
<td width="50%">

### ⚡ **Real-time Processing**

* Live webcam analysis
* Batch processing capabilities
* GPU acceleration support
* Optimized performance
* Interactive user interfaces

</td>
</tr>
</table>

---

## 📋 Project Overview

This repository showcases **7 cutting-edge computer vision and AI projects** demonstrating practical applications of modern machine learning techniques:

<div align="center">

| 🔥 **Assignment Part A**         | 🎯 **Assignment Part B**         |
| -------------------------------- | -------------------------------- |
| **Advanced Vehicle Analysis**    | **Computer Vision Fundamentals** |
| 🚗 License Plate Analysis        | 👁️ Face Detection & Landmarks   |
| 🚙 Vehicle Attribute Recognition | 🎭 Real-time Face Blurring       |
|                                  | 🔤 String Similarity Analysis    |
|                                  | 🧪 Automated Testing Framework   |
|                                  | 🐱 Image Classification          |

</div>

---

## 🚗 Assignment Part A - Advanced Vehicle Analysis

| Project                        | Technology Stack                 | Key Features                                         |
| ------------------------------ | -------------------------------- | ---------------------------------------------------- |
| **Q1: License Plate Analysis** | PaddleOCR + Custom PyTorch + VLM | Character integrity detection, Multi-model consensus |
| **Q2: Vehicle Recognition**    | YOLOv8 + PaddleOCR + BLIP VLM    | Brand/model detection, Color analysis, Traffic flow  |

---

## 🎯 Assignment Part B - Computer Vision Fundamentals

| Project                      | Technology Stack           | Key Features                                |
| ---------------------------- | -------------------------- | ------------------------------------------- |
| **Q3: Face Detection**       | MediaPipe Face Mesh        | Landmark localization, Real-time processing |
| **Q4: Face Blurring**        | YOLO + MediaPipe           | Privacy protection, Live video processing   |
| **Q5: String Similarity**    | Needleman-Wunsch Algorithm | Sequence alignment, Bioinformatics approach |
| **Q6: Testing Framework**    | pytest + CSV Analysis      | Automated validation, Bulk processing       |
| **Q7: Image Classification** | ResNet-50 (pretrained)     | Transfer learning, Confidence scoring       |

---

## 🚀 Quick Start

<details>
<summary><b>📋 Prerequisites</b></summary>

* **Python**: 3.10+ (tested on 3.11)
* **OS**: Windows/Linux/macOS
* **RAM**: 4GB+ recommended
* **Storage**: 2GB for models
* **GPU**: Optional (CUDA supported)

</details>

### ⚡ Installation

```bash
# 1️⃣ Clone the repository
git clone https://github.com/mdshoaibuddinchanda/iium-internship.git
cd iium-internship

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Verify installation
python -c "import cv2, torch, mediapipe, ultralytics; print('✅ All dependencies installed!')"
```

### 🎮 Quick Demo

```bash
# 🚗 License Plate Analysis
python "assignment_part_a/Q1_License Plate/Q1_code.py"

# 👁️ Face Detection
python assignment_part_b/Q3/Q3_face_detection_localize.py

# 🐱 Image Classification  
python assignment_part_b/Q7/Q7_cats_vs_dogs_classifier.py
```

---

## 📁 Project Structure

<details>
<summary><b>🗂️ Click to expand</b></summary>

```
iium-internship/
├── assignment_part_a/           # Advanced Vehicle Analysis
│   ├── Q1_License Plate/        # License plate character analysis
│   │   ├── data/                # Input images
│   │   ├── output/              # Generated CSV reports
│   │   ├── last.pt              # Custom PyTorch model
│   │   ├── Q1_code.py           # Multi-model analyzer
│   │   └── README.md
│   └── Q2_Vehicle Attribute/
│       ├── q2_Images/           # Sample traffic images
│       ├── results/             # Outputs & visualizations
│       ├── q2_enhanced.py
│       └── README.md
├── assignment_part_b/           # Computer Vision Fundamentals  
│   ├── data/                    # Datasets
│   ├── Q3/                      # Face detection
│   ├── Q4/                      # Face blurring
│   ├── Q5/                      # String similarity
│   ├── Q6/                      # Testing framework
│   ├── Q7/                      # Image classification
│   └── result/                  # Outputs
├── requirements.txt
├── README.md
└── .gitignore
```

</details>

---

## 📊 Performance Results

| Assignment | Accuracy | Speed        | Key Features        |
| ---------- | -------- | ------------ | ------------------- |
| Q1         | 95%+     | ~2s/img pair | Character integrity |
| Q2         | 90%+     | ~8s/img      | Vehicle analysis    |
| Q3         | 98%+     | Real-time    | Landmarks           |
| Q4         | 95%+     | Real-time    | Face blurring       |
| Q5         | 100%     | Instant      | String alignment    |
| Q6         | Variable | Batch        | Similarity testing  |
| Q7         | 85%+     | ~1s/img      | Classification      |

---

## 🔧 Dependencies

* `opencv-python==4.10.0.84`
* `numpy>=1.24`
* `pandas>=2.2.0`
* `pillow>=10.0.0`
* `torch==2.4.1+cpu`
* `torchvision==0.19.1+cpu`
* `ultralytics>=8.3.0,<9`
* `mediapipe==0.10.9`
* `paddleocr`
* `transformers`
* `pytest==8.3.3`

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch
3. Make changes + document
4. Test all assignments
5. Submit PR 🚀

---

## 📝 License

This project is licensed under the **MIT License** – see [LICENSE](LICENSE).

---

## 🙏 Acknowledgments

* **Ultralytics** for YOLOv8
* **Google** for MediaPipe
* **PaddleOCR** team
* **PyTorch** team
* **OpenCV** community

---
