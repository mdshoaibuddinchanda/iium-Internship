import os
import shutil
import csv
import torch
from PIL import Image
import torchvision.models as models
from torchvision import transforms

# -------------------------
# SETTINGS
# -------------------------
IMAGE_FOLDER = "assignment_part_b/data/cat_dog/images"   # folder with images
OUT_CSV = "assignment_part_b/result/Q7/Q7_report.csv"
MIS_TXT = "assignment_part_b/result/Q7/Q7_misleading.txt"  # list of files predicted as non-dog
MIS_IMG_DIR = "assignment_part_b/result/Q7/misleading_images"  # folder to collect mislabel images

# -------------------------
# Load model and categories
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load EfficientNet (B0 here, can choose B1-B7 for higher accuracy)
weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
model = models.efficientnet_b0(weights=weights).to(device).eval()
categories = weights.meta["categories"]

# Preprocessing
transform = weights.transforms()  # EfficientNet weights provide correct preprocessing

# -------------------------
# Predict function
# -------------------------
def predict(img_path):
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        prob = torch.nn.functional.softmax(out[0], dim=0)
        top1_prob, top1_idx = torch.max(prob, dim=0)
    return categories[int(top1_idx.item())], float(top1_prob), int(top1_idx.item())

def is_dog_index(idx: int) -> bool:
    """ImageNet dog classes are indices 151..268 (inclusive)."""
    return 151 <= idx <= 268

# -------------------------
# Run on all images
# -------------------------
results = []
misleading = []  # files predicted NOT as dog (assuming all inputs are dog images)
files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith((".jpg",".jpeg",".png"))]

print(f"Found {len(files)} images\n")

for fname in files:
    path = os.path.join(IMAGE_FOLDER, fname)
    label, prob, idx = predict(path)
    print(f"{fname} | Pred={label} ({prob*100:.1f}%)")
    results.append([fname, label, f"{prob:.4f}"])
    if not is_dog_index(idx):
        misleading.append((fname, label, prob))

# -------------------------
# Save results
# -------------------------
out_dir = os.path.dirname(OUT_CSV)
os.makedirs(out_dir, exist_ok=True)
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "prediction", "probability"])
    writer.writerows(results)

print(f"\n--- Done ---")
print(f"Report saved at: {OUT_CSV}")

# Save misleading files list (predicted as non-dog)
if misleading:
    # Save text list
    with open(MIS_TXT, "w", encoding="utf-8") as f:
        f.write("Misleading images (predicted as non-dog):\n")
        for fname, label, prob in misleading:
            f.write(f"{fname} -> {label} ({prob*100:.2f}%)\n")
    print(f"Misleading list saved at: {MIS_TXT}")

    # Copy mislabel images into result folder
    os.makedirs(MIS_IMG_DIR, exist_ok=True)
    for fname, _, _ in misleading:
        src = os.path.join(IMAGE_FOLDER, fname)
        dst = os.path.join(MIS_IMG_DIR, fname)
        try:
            shutil.copy2(src, dst)
        except Exception as e:
            print(f"Warning: failed to copy {fname} -> {e}")
    print(f"Mislabel images copied to: {MIS_IMG_DIR}")
else:
    print("No misleading images found (all predicted as dogs).")
