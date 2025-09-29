"""
Q3: Face Detection and Feature Localization System

🎯 What this program does:
This script detects human faces in images or webcam video and precisely locates
key facial features (landmarks). It's like teaching a computer to "see" faces
the same way humans do.

👁️ Key features detected:
- Left eye center (pupil location)
- Right eye center (pupil location)  
- Nose tip (end of the nose)

🧠 Technology used:
- Google's MediaPipe Face Mesh: Advanced AI model trained on millions of faces
- OpenCV: Computer vision library for image processing
- Real-time processing: Works on live webcam or static images

✨ Program features:
✅ Clear visual labels for each detected feature
✅ Real-time FPS (frames per second) display for webcam
✅ Face counter (shows how many faces are detected)
✅ Saves annotated images with detected landmarks
✅ Beginner-friendly with clear visual feedback

🎮 How to use:
1. Run the script
2. Choose between image analysis or webcam mode
3. For images: provide the image path
4. For webcam: press 'q' to quit

Author: Computer Vision Assignment
Date: 2024
"""

# Import necessary libraries
import cv2          # OpenCV - for image processing and computer vision
import mediapipe as mp  # Google's MediaPipe - for face detection and landmarks
import os           # Operating system interface - for file operations
import time         # Time utilities - for FPS calculation

# Configuration: Set up output directory for saving results
RESULT_DIR = "assignment_part_b/result/Q3"
os.makedirs(RESULT_DIR, exist_ok=True)  # Create directory if it doesn't exist
print(f"📁 Results will be saved to: {RESULT_DIR}")

# MediaPipe Face Mesh Landmark Indexes
# MediaPipe detects 468 facial landmarks (points) on each face
# We only need 3 specific points for this assignment:
LEFT_EYE_IDX = 468    # Index for left eye center (left iris center)
RIGHT_EYE_IDX = 473   # Index for right eye center (right iris center)  
NOSE_TIP_IDX = 1      # Index for nose tip (the end point of the nose)

print("🧠 MediaPipe Face Mesh initialized with landmark indexes:")
print(f"   👁️ Left Eye: Index {LEFT_EYE_IDX}")
print(f"   👁️ Right Eye: Index {RIGHT_EYE_IDX}")
print(f"   👃 Nose Tip: Index {NOSE_TIP_IDX}")


def get_landmarks(img, face_landmarks):
    """
    🎯 Convert MediaPipe's normalized coordinates to actual pixel positions.
    
    MediaPipe returns landmark coordinates as decimal numbers between 0 and 1:
    - (0, 0) = top-left corner of the image
    - (1, 1) = bottom-right corner of the image
    - (0.5, 0.5) = center of the image
    
    We need to convert these to actual pixel coordinates that we can use
    to draw on the image.
    
    Args:
        img: The input image (used to get width and height)
        face_landmarks: MediaPipe face landmarks object containing all 468 points
        
    Returns:
        dict: Dictionary with pixel coordinates for left_eye, right_eye, nose_tip
    """
    # Get image dimensions
    h, w, _ = img.shape  # height, width, channels (RGB)
    
    def to_px(idx):
        """
        Helper function to convert one landmark from normalized to pixel coordinates.
        
        Args:
            idx: The landmark index (e.g., 468 for left eye)
            
        Returns:
            tuple: (x, y) pixel coordinates
        """
        # Get the specific landmark point
        pt = face_landmarks.landmark[idx]
        
        # Convert normalized coordinates (0-1) to pixel coordinates
        # pt.x * w: Convert x from 0-1 range to 0-width range
        # pt.y * h: Convert y from 0-1 range to 0-height range
        pixel_x = int(pt.x * w)
        pixel_y = int(pt.y * h)
        
        return pixel_x, pixel_y

    # Extract the three key landmarks we need
    landmarks = {
        "left_eye": to_px(LEFT_EYE_IDX),    # Convert left eye landmark to pixels
        "right_eye": to_px(RIGHT_EYE_IDX),  # Convert right eye landmark to pixels
        "nose_tip": to_px(NOSE_TIP_IDX),    # Convert nose tip landmark to pixels
    }
    
    return landmarks


def draw_landmarks(img, lm):
    """
    🎨 Draw visual annotations on the image to show detected facial features.
    
    This function adds:
    - Colored circles at each landmark position
    - Text labels identifying each feature
    - Lines connecting the features to form a triangle
    
    Color scheme:
    - Green: Eyes (both left and right)
    - Red: Nose tip
    - Yellow: Connecting lines
    
    Args:
        img: The image to draw on (modified in-place)
        lm: Dictionary containing landmark coordinates
    """
    
    # 👁️ Draw LEFT EYE
    # cv2.circle(image, center_point, radius, color, thickness)
    # thickness = -1 means filled circle
    # Color format: (Blue, Green, Red) - BGR format
    cv2.circle(img, lm["left_eye"], 8, (0, 255, 0), -1)  # Green filled circle
    
    # Add text label for left eye
    # Position the text slightly above and to the left of the eye
    # cv2.putText(image, text, position, font, scale, color, thickness)
    cv2.putText(img, "Left Eye", 
                (lm["left_eye"][0] - 70, lm["left_eye"][1] - 30),  # Position offset
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)    # Green text

    # 👁️ Draw RIGHT EYE  
    cv2.circle(img, lm["right_eye"], 8, (0, 255, 0), -1)  # Green filled circle
    
    # Add text label for right eye
    # Position the text slightly above and to the right of the eye
    cv2.putText(img, "Right Eye", 
                (lm["right_eye"][0] + 20, lm["right_eye"][1] - 30),  # Position offset
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)     # Green text

    # 👃 Draw NOSE TIP
    cv2.circle(img, lm["nose_tip"], 8, (0, 0, 255), -1)  # Red filled circle
    
    # Add text label for nose
    # Position the text slightly below and centered on the nose
    cv2.putText(img, "Nose", 
                (lm["nose_tip"][0] - 30, lm["nose_tip"][1] + 50),  # Position offset
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)    # Red text

    # 📐 Draw CONNECTING LINES to form a triangle
    # This helps visualize the spatial relationship between features
    # cv2.line(image, start_point, end_point, color, thickness)
    
    # Line from left eye to right eye (horizontal line across face)
    cv2.line(img, lm["left_eye"], lm["right_eye"], (255, 255, 0), 2)  # Yellow line
    
    # Line from left eye to nose tip (left side of triangle)
    cv2.line(img, lm["left_eye"], lm["nose_tip"], (255, 255, 0), 2)   # Yellow line
    
    # Line from right eye to nose tip (right side of triangle)
    cv2.line(img, lm["right_eye"], lm["nose_tip"], (255, 255, 0), 2)  # Yellow line
    
    print("   🎨 Drew landmarks: 2 eyes (green) + nose (red) + connecting lines (yellow)")


def process_image(image_path):
    """
    📸 Process a single image to detect faces and extract key landmarks.
    
    This function:
    1. Loads an image from the file system
    2. Uses MediaPipe to detect faces and landmarks
    3. Draws visual annotations on the detected features
    4. Saves the annotated image
    5. Displays the results to the user
    
    Args:
        image_path (str): Path to the input image file
    """
    print(f"\n📸 Processing image: {os.path.basename(image_path)}")
    
    # Step 1: Load the image
    img = cv2.imread(image_path)
    if img is None:
        print("❌ ERROR: Could not read image. Please check the file path.")
        print("   Make sure the file exists and is a valid image format (jpg, png, etc.)")
        return

    print(f"   ✅ Image loaded successfully: {img.shape[1]}x{img.shape[0]} pixels")

    # Step 2: Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    
    # Configure Face Mesh for static image processing:
    # - static_image_mode=True: Optimized for single images (not video)
    # - refine_landmarks=True: More accurate landmark detection
    with mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True) as face_mesh:
        
        # Step 3: Process the image
        # MediaPipe expects RGB format, but OpenCV loads images in BGR format
        # So we need to convert BGR -> RGB before processing
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)

    # Step 4: Check if any faces were detected
    if not results.multi_face_landmarks:
        print("   ℹ️ No faces detected in this image.")
        print("   💡 Tips: Make sure the image contains a clear, front-facing face")
        return

    print(f"   🎉 Found {len(results.multi_face_landmarks)} face(s) in the image!")

    # Step 5: Process the first detected face
    # (If multiple faces are detected, we only process the first one)
    face_landmarks = results.multi_face_landmarks[0]
    landmarks = get_landmarks(img, face_landmarks)
    
    print("   🎯 Extracting key landmarks...")
    draw_landmarks(img, landmarks)

    # Step 6: Save the annotated image
    base_name = os.path.basename(image_path)  # Get filename without path
    name_without_ext = os.path.splitext(base_name)[0]  # Remove file extension
    out_path = os.path.join(RESULT_DIR, name_without_ext + "_annotated.jpg")
    
    success = cv2.imwrite(out_path, img)
    if success:
        print(f"   💾 Annotated image saved: {out_path}")
    else:
        print("   ❌ Failed to save annotated image")

    # Step 7: Display results
    cv2.imshow("Face Landmark Detection - Press any key to close", img)

    # Print detailed landmark coordinates
    print(f"\n📊 Landmark Coordinates for {base_name}:")
    print(f"   👁️ Left Eye Center:  {landmarks['left_eye']}")
    print(f"   👁️ Right Eye Center: {landmarks['right_eye']}")
    print(f"   👃 Nose Tip:         {landmarks['nose_tip']}")
    print(f"   📈 Total Faces Found: {len(results.multi_face_landmarks)}")
    
    print("\n💡 Press any key to close the image window...")
    cv2.waitKey(0)  # Wait for user to press any key
    cv2.destroyAllWindows()  # Close all OpenCV windows


def process_webcam():
    """
    📹 Real-time face detection using webcam with live performance metrics.
    
    This function:
    1. Connects to the default camera (usually built-in webcam)
    2. Continuously captures video frames
    3. Detects faces and landmarks in each frame
    4. Displays real-time annotations and performance metrics
    5. Allows user to quit by pressing 'q'
    
    Features:
    - Real-time FPS (frames per second) display
    - Live face counter
    - Multiple face detection (up to 5 faces)
    - Smooth performance optimization
    """
    print("\n📹 Starting webcam face detection...")
    print("   💡 Make sure your webcam is connected and not being used by other apps")
    
    # Step 1: Initialize webcam connection
    cap = cv2.VideoCapture(0)  # 0 = default camera (usually built-in webcam)
    
    # Check if webcam opened successfully
    if not cap.isOpened():
        print("❌ ERROR: Could not open webcam")
        print("   💡 Troubleshooting tips:")
        print("   - Make sure no other apps are using the camera")
        print("   - Try changing the camera index (0, 1, 2, etc.)")
        print("   - Check if your camera drivers are installed")
        return
    
    print("   ✅ Webcam connected successfully")
    
    # Step 2: Initialize MediaPipe Face Mesh for video processing
    mp_face_mesh = mp.solutions.face_mesh
    prev_time = 0  # For FPS calculation
    
    # Configure Face Mesh for real-time video:
    # - refine_landmarks=True: More accurate detection
    # - max_num_faces=5: Can detect up to 5 faces simultaneously
    # - static_image_mode=False: Optimized for video (default)
    with mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=5) as face_mesh:
        
        print("   🎬 Starting real-time detection... Press 'q' to quit")
        
        # Step 3: Main processing loop
        while True:
            # Capture a frame from the webcam
            ret, frame = cap.read()
            if not ret:
                print("   ⚠️ Failed to capture frame from webcam")
                break

            # Step 4: Process the frame for face detection
            # Convert BGR (OpenCV format) to RGB (MediaPipe format)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            
            # Create a copy of the frame for drawing annotations
            display = frame.copy()

            # Step 5: Process detected faces
            face_count = 0
            if results.multi_face_landmarks:
                face_count = len(results.multi_face_landmarks)
                
                # Draw landmarks for each detected face
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = get_landmarks(frame, face_landmarks)
                    draw_landmarks(display, landmarks)

            # Step 6: Calculate and display FPS (frames per second)
            # FPS shows how fast the system is processing frames
            curr_time = time.time()
            if prev_time != 0:
                fps = 1 / (curr_time - prev_time)
            else:
                fps = 0
            prev_time = curr_time

            # Step 7: Add performance metrics to the display
            # Show FPS in top-left corner
            cv2.putText(display, f"FPS: {int(fps)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)  # Cyan text
            
            # Show face count below FPS
            cv2.putText(display, f"Faces: {face_count}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)  # Cyan text
            
            # Add instructions
            cv2.putText(display, "Press 'q' to quit", (10, display.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)  # White text

            # Step 8: Display the annotated frame
            cv2.imshow("Real-time Face Detection - Press 'q' to quit", display)
            
            # Step 9: Check for quit command
            # cv2.waitKey(1) waits 1ms for a key press
            # & 0xFF extracts the last 8 bits (handles different systems)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("   👋 User pressed 'q' - stopping webcam...")
                break

    # Step 10: Cleanup
    cap.release()  # Release the webcam
    cv2.destroyAllWindows()  # Close all OpenCV windows
    print("   ✅ Webcam session ended successfully")


# MAIN PROGRAM EXECUTION
# This section runs when the script is executed directly
if __name__ == "__main__":
    """
    🚀 Main program entry point - Interactive face detection system
    
    This program offers two modes:
    1. Image Mode: Analyze a single image file
    2. Webcam Mode: Real-time analysis using your camera
    
    The user can choose which mode to use through a simple menu.
    """
    
    print("=" * 60)
    print("👁️ FACE DETECTION AND LANDMARK LOCALIZATION SYSTEM")
    print("=" * 60)
    print("🎯 This program detects faces and locates key features:")
    print("   • Left eye center")
    print("   • Right eye center") 
    print("   • Nose tip")
    print("\n🧠 Powered by Google's MediaPipe AI technology")
    print("📁 Results are automatically saved to:", RESULT_DIR)
    
    print("\n" + "=" * 60)
    print("📋 CHOOSE INPUT MODE:")
    print("=" * 60)
    print("1. 📸 Image Mode - Analyze a single image file")
    print("   • Upload any image with faces")
    print("   • Get precise landmark coordinates")
    print("   • Save annotated results")
    print()
    print("2. 📹 Webcam Mode - Real-time face detection")
    print("   • Use your computer's camera")
    print("   • See live face detection")
    print("   • View FPS and face count")
    print()
    
    # Get user choice with input validation
    while True:
        choice = input("👆 Enter your choice (1 or 2): ").strip()
        
        if choice == "1":
            print("\n📸 IMAGE MODE SELECTED")
            print("-" * 30)
            
            # Get image path from user
            while True:
                path = input("📁 Enter the full path to your image file: ").strip()
                
                # Remove quotes if user copied path with quotes
                path = path.strip('"').strip("'")
                
                # Check if file exists
                if os.path.exists(path):
                    print(f"   ✅ File found: {os.path.basename(path)}")
                    break
                else:
                    print(f"   ❌ File not found: {path}")
                    print("   💡 Please check the path and try again")
                    print("   💡 Example: C:/Users/YourName/Pictures/photo.jpg")
                    
                    retry = input("   🔄 Try again? (y/n): ").strip().lower()
                    if retry != 'y':
                        print("   👋 Exiting image mode...")
                        exit()
            
            # Process the image
            process_image(path)
            break
            
        elif choice == "2":
            print("\n📹 WEBCAM MODE SELECTED")
            print("-" * 30)
            print("🎬 Preparing to start real-time face detection...")
            print("💡 Make sure your webcam is connected and working")
            
            # Ask for confirmation before starting webcam
            confirm = input("🚀 Start webcam now? (y/n): ").strip().lower()
            if confirm == 'y':
                process_webcam()
            else:
                print("   👋 Webcam mode cancelled")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1 or 2.")
            print()
    
    print("\n" + "=" * 60)
    print("🎉 FACE DETECTION SESSION COMPLETE!")
    print("=" * 60)
    print("📊 What happened:")
    print("   • Faces were detected using MediaPipe AI")
    print("   • Key landmarks were precisely located")
    print("   • Visual annotations were added")
    if choice == "1":
        print("   • Annotated image was saved to results folder")
    print("\n💡 Tips for better results:")
    print("   • Use well-lit images with clear faces")
    print("   • Face should be front-facing (not profile)")
    print("   • Avoid heavy shadows or reflections")
    print("\n✨ Thank you for using the Face Detection System!")
