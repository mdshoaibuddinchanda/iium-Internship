"""
Q5: Advanced String Similarity Analysis System

🎯 What this program does:
This program compares two text strings to determine how similar they are.
It's like a smart spell-checker that can handle insertions, deletions, and substitutions.

🧠 Algorithm used:
Needleman-Wunsch Algorithm - Originally developed for DNA sequence alignment in bioinformatics,
this algorithm finds the optimal way to align two sequences to maximize similarity.

📊 Key features:
✅ Handles strings of different lengths (6-10 characters)
✅ Uses advanced sequence alignment (not just character-by-character)
✅ Provides detailed character-by-character analysis
✅ Calculates precise similarity percentage
✅ Generates comprehensive reports
✅ Prevents file overwrites with unique naming

🔍 Example use cases:
- License plate similarity (ABC123 vs ABC12B)
- Name matching (Smith vs Smyth)
- Code comparison (password vs pasword)
- DNA sequence analysis (ATCG vs ATGC)

💡 How it works:
1. Takes two strings from user input
2. Aligns them optimally using dynamic programming
3. Compares character by character
4. Calculates similarity percentage
5. Saves detailed report to file

Author: Computer Vision Assignment
Date: 2024
"""

import os  # For file and directory operations

# ========================================
# NEEDLEMAN-WUNSCH SEQUENCE ALIGNMENT ALGORITHM
# ========================================

def needleman_wunsch(str1: str, str2: str, match_score=1, mismatch_penalty=-1, gap_penalty=-1):
    """
    🧬 Needleman-Wunsch Global Sequence Alignment Algorithm
    
    This is a dynamic programming algorithm originally developed for DNA sequence alignment.
    It finds the optimal way to align two sequences by allowing:
    - Matches: Characters that are the same
    - Mismatches: Characters that are different  
    - Gaps: Missing characters (insertions/deletions)
    
    📚 How it works:
    1. Creates a scoring matrix to evaluate all possible alignments
    2. Uses dynamic programming to find the optimal alignment
    3. Traces back through the matrix to construct the final alignment
    
    🎯 Example:
    Input:  "HELLO" vs "HELO"
    Output: "HELLO" vs "HE-LO" (gap inserted to align optimally)
    
    Args:
        str1 (str): First string to compare
        str2 (str): Second string to compare
        match_score (int): Points awarded for matching characters (+1)
        mismatch_penalty (int): Points deducted for different characters (-1)
        gap_penalty (int): Points deducted for gaps/missing characters (-1)
        
    Returns:
        tuple: (aligned_str1, aligned_str2) with gaps (-) inserted for optimal alignment
    """
    
    # Step 1: Get string lengths and initialize scoring matrix
    m, n = len(str1), len(str2)
    
    # Create a 2D matrix to store alignment scores
    # Matrix size: (m+1) x (n+1) to include empty string cases
    score = [[0] * (n + 1) for _ in range(m + 1)]
    
    print(f"🧮 Creating {m+1}x{n+1} scoring matrix for alignment...")

    # Step 2: Initialize first row and column with gap penalties
    # First row: aligning empty string with str2 (all gaps in str1)
    for i in range(m + 1):
        score[i][0] = i * gap_penalty  # Cost of i gaps
    
    # First column: aligning str1 with empty string (all gaps in str2)
    for j in range(n + 1):
        score[0][j] = j * gap_penalty  # Cost of j gaps

    # Step 3: Fill the scoring matrix using dynamic programming
    # For each cell, consider three possibilities:
    # 1. Match/Mismatch: characters from both strings
    # 2. Deletion: character from str1, gap in str2
    # 3. Insertion: gap in str1, character from str2
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Option 1: Match or mismatch
            if str1[i - 1] == str2[j - 1]:
                match = score[i - 1][j - 1] + match_score  # Characters match
            else:
                match = score[i - 1][j - 1] + mismatch_penalty  # Characters differ
            
            # Option 2: Deletion (gap in str2)
            delete = score[i - 1][j] + gap_penalty
            
            # Option 3: Insertion (gap in str1)
            insert = score[i][j - 1] + gap_penalty
            
            # Choose the option with the highest score
            score[i][j] = max(match, delete, insert)

    # Step 4: Traceback to construct the optimal alignment
    # Start from bottom-right corner and work backwards
    aligned1, aligned2 = "", ""
    i, j = m, n
    
    print("🔍 Tracing back through matrix to find optimal alignment...")
    
    # Trace back until we reach the top-left corner
    while i > 0 and j > 0:
        current = score[i][j]
        
        # Determine which operation led to this score
        if str1[i - 1] == str2[j - 1]:
            match_score_here = score[i - 1][j - 1] + match_score
        else:
            match_score_here = score[i - 1][j - 1] + mismatch_penalty
            
        if current == match_score_here:
            # This came from a match/mismatch
            aligned1 = str1[i - 1] + aligned1
            aligned2 = str2[j - 1] + aligned2
            i -= 1
            j -= 1
        elif current == score[i - 1][j] + gap_penalty:
            # This came from a deletion (gap in str2)
            aligned1 = str1[i - 1] + aligned1
            aligned2 = "-" + aligned2
            i -= 1
        else:
            # This came from an insertion (gap in str1)
            aligned1 = "-" + aligned1
            aligned2 = str2[j - 1] + aligned2
            j -= 1

    # Step 5: Handle remaining characters (if any)
    # Add remaining characters from str1 (with gaps in str2)
    while i > 0:
        aligned1 = str1[i - 1] + aligned1
        aligned2 = "-" + aligned2
        i -= 1
    
    # Add remaining characters from str2 (with gaps in str1)
    while j > 0:
        aligned1 = "-" + aligned1
        aligned2 = str2[j - 1] + aligned2
        j -= 1

    print(f"✅ Alignment complete:")
    print(f"   Original: '{str1}' vs '{str2}'")
    print(f"   Aligned:  '{aligned1}' vs '{aligned2}'")
    
    return aligned1, aligned2


# ========================================
# STRING COMPARISON AND ANALYSIS
# ========================================

def compare_strings(str1: str, str2: str, use_alignment=True):
    """
    📊 Compare two strings and calculate detailed similarity metrics.
    
    This function performs a comprehensive comparison between two strings:
    1. Optionally aligns the strings using Needleman-Wunsch algorithm
    2. Compares character by character
    3. Counts matches and mismatches
    4. Calculates similarity percentage
    5. Generates detailed comparison report
    
    🎯 Why alignment matters:
    Without alignment: "HELLO" vs "HELO" → 60% similarity (3/5 matches)
    With alignment:    "HELLO" vs "HE-LO" → 80% similarity (4/5 matches)
    
    Args:
        str1 (str): First string to compare
        str2 (str): Second string to compare
        use_alignment (bool): Whether to use Needleman-Wunsch alignment
        
    Returns:
        tuple: (similarity_percentage, total_matches, total_mismatches, detailed_comparison)
    """
    
    print(f"\n📊 Comparing strings: '{str1}' vs '{str2}'")
    
    # Step 1: Align strings if requested
    if use_alignment:
        print("🔄 Applying Needleman-Wunsch alignment...")
        original_str1, original_str2 = str1, str2  # Keep originals for reference
        str1, str2 = needleman_wunsch(str1, str2)
        print(f"   Aligned strings: '{str1}' vs '{str2}'")
    else:
        print("⚠️ Skipping alignment - using direct character comparison")
        # Pad shorter string with gaps for fair comparison
        max_len = max(len(str1), len(str2))
        str1 = str1.ljust(max_len, '-')
        str2 = str2.ljust(max_len, '-')

    # Step 2: Character-by-character comparison
    matches = 0
    comparison_lines = []
    
    print("🔍 Performing character-by-character analysis...")
    
    # Compare each position
    for i in range(len(str1)):
        c1, c2 = str1[i], str2[i]
        
        # Determine if characters match
        if c1 == c2:
            status = "✓ MATCH"
            matches += 1
        else:
            status = "✗ MISMATCH"
        
        # Create detailed comparison line
        comparison_line = f"Position {i+1:2d}: '{c1}' vs '{c2}' → {status}"
        comparison_lines.append(comparison_line)
        
        # Print real-time comparison (for debugging/learning)
        print(f"   {comparison_line}")

    # Step 3: Calculate metrics
    total_positions = len(str1)
    mismatches = total_positions - matches
    similarity_percentage = (matches / total_positions) * 100
    
    print(f"\n📈 Analysis Results:")
    print(f"   Total positions compared: {total_positions}")
    print(f"   Matches: {matches}")
    print(f"   Mismatches: {mismatches}")
    print(f"   Similarity: {similarity_percentage:.2f}%")
    
    return similarity_percentage, matches, mismatches, comparison_lines


# ========================================
# UNIQUE FILENAME GENERATOR
# ========================================

def get_unique_filename(folder: str, base_name: str = "Q5_report", ext: str = ".txt") -> str:
    """
    📁 Generate a unique filename to prevent overwriting existing reports.
    
    This function ensures that each analysis gets its own report file by
    automatically adding numbers to the filename if needed.
    
    🎯 Example progression:
    - First run:  Q5_report.txt
    - Second run: Q5_report_1.txt  
    - Third run:  Q5_report_2.txt
    - And so on...
    
    This prevents accidentally overwriting previous analysis results.
    
    Args:
        folder (str): Directory where the file will be saved
        base_name (str): Base filename without extension
        ext (str): File extension (e.g., ".txt", ".csv")
        
    Returns:
        str: Complete path to a unique filename
    """
    
    # Start with the base filename
    filename = os.path.join(folder, base_name + ext)
    counter = 1
    
    print(f"📁 Generating unique filename in: {folder}")
    print(f"   Base name: {base_name}{ext}")
    
    # Keep incrementing counter until we find a filename that doesn't exist
    while os.path.exists(filename):
        print(f"   ⚠️ File exists: {os.path.basename(filename)}")
        filename = os.path.join(folder, f"{base_name}_{counter}{ext}")
        counter += 1
    
    print(f"   ✅ Unique filename generated: {os.path.basename(filename)}")
    return filename


# ========================================
# MAIN PROGRAM EXECUTION
# ========================================

def main():
    """
    🚀 Main program function that orchestrates the entire string similarity analysis.
    
    This function:
    1. Sets up the output directory
    2. Gets user input with validation
    3. Performs string comparison analysis
    4. Generates and saves a detailed report
    5. Displays results to the user
    """
    
    print("=" * 70)
    print("🔤 STRING SIMILARITY ANALYSIS SYSTEM")
    print("=" * 70)
    print("🎯 This program compares two strings using advanced sequence alignment")
    print("📊 It provides detailed similarity analysis and generates reports")
    print("🧬 Uses Needleman-Wunsch algorithm (from bioinformatics)")
    
    # Step 1: Set up output directory
    result_path = "assignment_part_b/result/Q5"
    print(f"\n📁 Output directory: {result_path}")

    # Verify that the result folder exists
    if not os.path.isdir(result_path):
        print(f"❌ ERROR: Result folder does not exist: {result_path}")
        print("💡 Please ensure the folder structure is correct")
        raise FileNotFoundError(f"Result folder not found: {result_path}")
    
    print("   ✅ Output directory verified")

    # Step 2: Get user input with validation
    print("\n" + "=" * 70)
    print("📝 INPUT COLLECTION")
    print("=" * 70)
    print("📏 Both strings must be between 6-10 characters long")
    print("💡 Examples: 'HELLO', 'ABC123', 'password', 'license'")
    
    # Get first string with validation loop
    while True:
        print("\n🔤 First String:")
        str1 = input("Enter first string (6–10 chars): ").strip()
        
        if 6 <= len(str1) <= 10:
            print(f"   ✅ Valid input: '{str1}' (length: {len(str1)})")
            break
        else:
            print(f"   ❌ Invalid length: {len(str1)} characters")
            print("   💡 Please enter a string between 6-10 characters")

    # Get second string with validation loop
    while True:
        print("\n🔤 Second String:")
        str2 = input("Enter second string (6–10 chars): ").strip()
        
        if 6 <= len(str2) <= 10:
            print(f"   ✅ Valid input: '{str2}' (length: {len(str2)})")
            break
        else:
            print(f"   ❌ Invalid length: {len(str2)} characters")
            print("   💡 Please enter a string between 6-10 characters")

    # Step 3: Perform string comparison analysis
    print("\n" + "=" * 70)
    print("🧮 SIMILARITY ANALYSIS")
    print("=" * 70)
    
    similarity, matches, mismatches, details = compare_strings(str1, str2, use_alignment=True)

    # Step 4: Generate unique filename and save report
    print("\n📄 Generating detailed report...")
    output_file = get_unique_filename(result_path)

    # Create comprehensive report
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("STRING SIMILARITY ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("INPUT STRINGS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"String 1: '{str1}' (Length: {len(str1)})\n")
        f.write(f"String 2: '{str2}' (Length: {len(str2)})\n\n")

        f.write("ANALYSIS RESULTS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total Matches: {matches}\n")
        f.write(f"Total Mismatches: {mismatches}\n")
        f.write(f"Similarity Percentage: {similarity:.2f}%\n\n")

        f.write("DETAILED CHARACTER-BY-CHARACTER COMPARISON:\n")
        f.write("-" * 50 + "\n")
        for line in details:
            f.write(line + "\n")
        
        f.write(f"\nANALYSIS METHOD:\n")
        f.write("-" * 20 + "\n")
        f.write("Algorithm: Needleman-Wunsch Global Sequence Alignment\n")
        f.write("Scoring: Match=+1, Mismatch=-1, Gap=-1\n")
        f.write("Purpose: Optimal alignment for maximum similarity detection\n")

    # Step 5: Display final results
    print("\n" + "=" * 70)
    print("🎉 ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"📊 SIMILARITY RESULT: {similarity:.2f}%")
    
    # Provide interpretation of results
    if similarity >= 90:
        print("🟢 Interpretation: Very High Similarity - Strings are nearly identical")
    elif similarity >= 70:
        print("🟡 Interpretation: High Similarity - Strings are quite similar")
    elif similarity >= 50:
        print("🟠 Interpretation: Moderate Similarity - Some similarities detected")
    else:
        print("🔴 Interpretation: Low Similarity - Strings are quite different")
    
    print(f"📁 Detailed report saved: {output_file}")
    print(f"📈 Matches: {matches} | Mismatches: {mismatches}")
    
    print("\n💡 What's in the report:")
    print("   • Complete character-by-character comparison")
    print("   • Alignment details showing optimal matching")
    print("   • Statistical analysis of similarity")
    print("   • Algorithm methodology explanation")
    
    print("\n✨ Thank you for using the String Similarity Analysis System!")


# PROGRAM ENTRY POINT
# This section runs when the script is executed directly
if __name__ == "__main__":
    """
    🚀 Program entry point - runs the main analysis function.
    
    This ensures the program only runs when executed directly,
    not when imported as a module by other programs.
    """
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Program interrupted by user (Ctrl+C)")
        print("✨ Thank you for using the String Similarity Analysis System!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("💡 Please check your input and try again")
