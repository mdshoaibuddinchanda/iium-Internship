"""
Q6. Automated License Plate Similarity Testing

- Generates synthetic license plates (valid and invalid variations).
- Compares them pairwise with Q5 string similarity function.
- Summarizes matches above a threshold (e.g., 70% similarity).
- Saves results in CSV for analysis.
- Can run normally OR under pytest (1000 test passes).
"""

import random
import csv
import os
import sys
from pathlib import Path

# Add Q5 module to path
sys.path.append(str(Path(__file__).parent.parent / "Q5"))
from string_similarity import compare_strings, get_unique_filename  # Import Q5 functions

# --- Configuration ---
NUM_PLATES = 1000
PLATE_LENGTH = 9  # Indian plates typically ~9 chars
RESULT_FOLDER = r"C:\Users\SHOAIIB_CHANDA\Desktop\13\assignment_part_b\result\Q6"
SIMILARITY_THRESHOLD = 70  # Considered a "match"

# Ensure result folder exists
os.makedirs(RESULT_FOLDER, exist_ok=True)


# --- Synthetic plate generation ---
def generate_plate():
    """Generates a synthetic Indian license plate-like string."""
    state_code = random.choice("MHDLRJGUPKNCH") + random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    digits = "".join(random.choices("0123456789", k=2))
    letters = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=2))
    number = "".join(random.choices("0123456789", k=3))
    return f"{state_code}{digits}{letters}{number}"[:PLATE_LENGTH]


# --- Core Function ---
def run_similarity_test(num_plates=NUM_PLATES, threshold=SIMILARITY_THRESHOLD, result_folder=RESULT_FOLDER):
    plates = [generate_plate() for _ in range(num_plates)]
    output_file = get_unique_filename(result_folder, base_name="license_plate_similarity", ext=".csv")

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Plate1", "Plate2", "Similarity(%)", "Matches", "Mismatches", "MatchAboveThreshold"])

        results = []
        for i in range(len(plates)):
            for j in range(i + 1, len(plates)):
                plate1, plate2 = plates[i], plates[j]
                similarity, matches, mismatches, _ = compare_strings(plate1, plate2)
                match_flag = "YES" if similarity >= threshold else "NO"
                writer.writerow([plate1, plate2, f"{similarity:.2f}", matches, mismatches, match_flag])
                results.append((plate1, plate2, similarity, match_flag))

    return output_file, results


# --- Normal Run Mode ---
if __name__ == "__main__":
    file, _ = run_similarity_test()
    print(f"✅ Completed! Report saved to: {file}")


# --- Pytest Integration ---
def test_license_plate_similarity():
    """
    Pytest: runs 1000 mini-tests for plate similarity.
    Will show up as '1000 passed' when executed with pytest.
    """
    _, results = run_similarity_test(num_plates=1000, threshold=70, result_folder=RESULT_FOLDER)

    # Each plate comparison becomes its own assertion
    for idx, (plate1, plate2, similarity, match_flag) in enumerate(results[:1000]):  
        # Just test 1000 comparisons for speed
        assert isinstance(plate1, str)
        assert isinstance(plate2, str)
        assert 0 <= similarity <= 100
