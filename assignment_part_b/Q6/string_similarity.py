"""
Q5. String Similarity Matching

- Takes two strings (each between 6 and 10 characters) from user input.
- Uses simple character comparison (can be extended to Needleman–Wunsch alignment).
- Computes similarity percentage.
- Outputs:
    * Prints percentage similarity in terminal.
    * Saves a detailed report in result/Q5/string_similarity_report_X.txt
      (X = number, so no file gets overwritten).
"""

import os

def compare_strings(str1: str, str2: str):
    """
    Compare two strings character by character.
    Returns:
        similarity %,
        total matches,
        total mismatches,
        list of detailed comparison lines.
    """
    matches = 0
    comparison_lines = []
    max_len = max(len(str1), len(str2))  # Ensure we cover both strings fully

    # Compare each position in the strings
    for i in range(max_len):
        c1 = str1[i] if i < len(str1) else "-"  # "-" if one string is shorter
        c2 = str2[i] if i < len(str2) else "-"
        status = "✓" if c1 == c2 else "✗"       # Mark match or mismatch
        if status == "✓":
            matches += 1
        comparison_lines.append(f"Pos {i+1}: {c1} vs {c2} → {status}")

    similarity = (matches / max_len) * 100
    return similarity, matches, max_len - matches, comparison_lines


def get_unique_filename(folder: str, base_name: str = "string_similarity_report", ext: str = ".txt") -> str:
    """
    Generate a unique filename in the given folder.
    Example: string_similarity_report.txt → string_similarity_report_1.txt → string_similarity_report_2.txt
    """
    filename = os.path.join(folder, base_name + ext)
    counter = 1
    # If file exists, keep adding numbers until unique name is found
    while os.path.exists(filename):
        filename = os.path.join(folder, f"{base_name}_{counter}{ext}")
        counter += 1
    return filename


def main():
    # Path where the report will be saved
    result_path = "result/Q5"

    # Ensure result folder exists (otherwise stop)
    if not os.path.isdir(result_path):
        raise FileNotFoundError(f"❌ Result folder does not exist: {result_path}")

    # --- Input handling with re-ask only for wrong input ---
    while True:
        str1 = input("Enter first string (6–10 chars): ").strip()
        if 6 <= len(str1) <= 10:
            break
        else:
            print("❌ Error: First string must be between 6–10 characters. Try again.\n")

    while True:
        str2 = input("Enter second string (6–10 chars): ").strip()
        if 6 <= len(str2) <= 10:
            break
        else:
            print("❌ Error: Second string must be between 6–10 characters. Try again.\n")

    # --- Compare the strings ---
    similarity, matches, mismatches, details = compare_strings(str1, str2)

    # --- Get unique file name (no overwrite) ---
    output_file = get_unique_filename(result_path)

    # --- Write report to file ---
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("String Similarity Report\n")
        f.write("========================\n\n")
        f.write(f"String 1: {str1} (Length: {len(str1)})\n")
        f.write(f"String 2: {str2} (Length: {len(str2)})\n\n")

        f.write(f"Total Matches: {matches}\n")
        f.write(f"Total Mismatches: {mismatches}\n")
        f.write(f"Similarity: {similarity:.2f}%\n\n")

        f.write("Character-by-Character Comparison:\n")
        f.write("----------------------------------\n")
        for line in details:
            f.write(line + "\n")

    # --- Print final result in terminal ---
    print(f"\n✅ Similarity: {similarity:.2f}%")
    print(f"✅ Report saved to: {output_file}")


if __name__ == "__main__":
    main()
