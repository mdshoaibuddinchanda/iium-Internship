# Install huggingface_hub if not already installed
# pip install huggingface_hub ultralytics

from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import os

# Step 1: Download model weights to your local machine
weights_path = hf_hub_download(
    repo_id="krishnamishra8848/Nepal-Vehicle-License-Plate-Detection",
    filename="last.pt",
    cache_dir=r"C:\Users\\Desktop"  # Change this path if needed
)

print(f"✅ Model downloaded to: {weights_path}")

# Step 2: Load the YOLO model locally
model = YOLO(weights_path)
print("✅ YOLO model loaded successfully!")
