# Assignment Part B — Beginner Guide

This part contains several independent mini-projects (Q3–Q7). You can run each on its own, using simple prompts (no command-line flags required).

## One-time setup
- Python 3.10+ recommended (tested on 3.11)
- Windows PowerShell commands shown below

Install all dependencies for Part B:

```powershell
pip install -r assignment_part_b\requirements.txt
```

If you have a CUDA GPU and want PyTorch GPU builds, follow https://pytorch.org/get-started/locally/ and install the matching wheels (you can remove the extra-index line in requirements and reinstall torch/torchvision).

---

## Folder overview

- Q3: Face detection + eyes/nose landmarks (MediaPipe)
- Q4: Face blurring for webcam or video (YOLOv8 + MediaPipe)
- Q5: String similarity (character-by-character) with a saved report
- Q6: Bulk license plate similarity testing (pytest + CSV outputs)
- Q7: Cats vs Dogs classifier (pretrained ResNet-50; saves report and misleading images)

See per-folder docs inside each Q folder (README_Q*.md) for details.

---

## Quick start commands

All scripts are interactive; just run and follow prompts.

- Q3
```powershell
python assignment_part_b\Q3\face_detection_localize.py
```

- Q4
```powershell
python assignment_part_b\Q4\face_blur_webcam.py
```

- Q5
```powershell
python assignment_part_b\Q5\string_similarity.py
```

- Q6
```powershell
python assignment_part_b\Q6\test_license_plate_similarity.py
```

- Q7
```powershell
python assignment_part_b\Q7\cats_vs_dogs_classifier.py
```

Outputs are saved under `assignment_part_b/result/<Qn>/`.

---

## Troubleshooting
- If Q4 shows model not found, update `MODEL_PATH` inside `Q4/face_blur_webcam.py` to the correct full path of `face_yolov8s.pt` on your machine.
- If PyTorch installation is slow on Windows, keep the CPU wheels in requirements. For GPU, use the official selector for your CUDA version.
- If webcam doesn’t open, try camera index 1 in your webcam apps, close other apps using the camera, and re-run.
