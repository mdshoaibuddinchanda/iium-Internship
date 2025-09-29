"""
Q1: License Plate Character Break Detection System

This program analyzes vehicle license plates to detect broken or damaged characters.
It compares front and rear images of the same vehicle to identify character integrity issues.

Key Features:
- Uses PaddleOCR for text recognition
- Analyzes character pixel density to detect damage
- Compares front vs rear plates for mismatches
- Generates detailed CSV reports

Author: Computer Vision Assignment
Date: 2024
"""

# Import necessary libraries
import os                    # For file and folder operations
import csv                   # For creating CSV reports
from typing import Iterable, List, Sequence, Tuple, Optional  # For type hints (better code documentation)
import cv2                   # OpenCV for image processing
import numpy as np           # NumPy for numerical operations
from paddleocr import PaddleOCR  # PaddleOCR for text recognition
import re                    # Regular expressions for text validation
import json                  # JSON for saving detailed analysis
from datetime import datetime  # For timestamps

# Optional advanced AI imports for maximum accuracy
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    from PIL import Image
    VLM_AVAILABLE = True
    TORCH_AVAILABLE = True
    print("✅ PyTorch and VLM (BLIP) available - Maximum AI license plate recognition enabled")
except ImportError:
    VLM_AVAILABLE = False
    TORCH_AVAILABLE = False
    print("⚠️ Advanced AI models not available. Install with: pip install transformers torch torchvision pillow")
    print("   System will use PaddleOCR only (still very accurate)")

# Initialize the OCR (Optical Character Recognition) engine
# use_textline_orientation=True: Helps with rotated text
# lang='en': Set language to English for better accuracy
print("🚀 Initializing Enhanced License Plate Analysis System...")
ocr = PaddleOCR(use_textline_orientation=True, lang='en', show_log=False)
print("✅ PaddleOCR engine ready!")

# Initialize AI models if available
vlm_processor = None
vlm_model = None
custom_model = None
device = 'cpu'

if TORCH_AVAILABLE:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️ Using device: {device}")
    
    # Load custom PyTorch model (last.pt)
    try:
        print("🎯 Loading custom PyTorch model (last.pt)...")
        custom_model_path = os.path.join(os.path.dirname(__file__), "last.pt")
        
        if os.path.exists(custom_model_path):
            # Load the custom model
            checkpoint = torch.load(custom_model_path, map_location=device)
            
            # Try to extract model architecture and weights
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    custom_model = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    # Create a simple CNN model for license plate recognition
                    custom_model = create_license_plate_model()
                    custom_model.load_state_dict(checkpoint['state_dict'])
                else:
                    custom_model = checkpoint
            else:
                custom_model = checkpoint
            
            if hasattr(custom_model, 'eval'):
                custom_model.eval()
                custom_model.to(device)
                print("✅ Custom PyTorch model loaded successfully")
            else:
                print("⚠️ Custom model format not recognized, will use other methods")
                custom_model = None
        else:
            print(f"⚠️ Custom model file not found: {custom_model_path}")
            custom_model = None
            
    except Exception as e:
        print(f"⚠️ Custom model loading failed: {e}")
        custom_model = None

    # Load VLM if available
    if VLM_AVAILABLE:
        try:
            print("🧠 Loading BLIP VLM for enhanced license plate recognition...")
            vlm_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            vlm_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            vlm_model.to(device)
            print(f"✅ VLM loaded successfully on {device}")
        except Exception as e:
            print(f"⚠️ VLM loading failed: {e}")
            VLM_AVAILABLE = False
            vlm_processor = None
            vlm_model = None

def create_license_plate_model():
    """
    🏗️ Create a simple CNN model architecture for license plate recognition.
    
    This is a fallback model structure in case the loaded model needs a specific architecture.
    """
    class LicensePlateNet(nn.Module):
        def __init__(self, num_classes=37):  # 26 letters + 10 digits + 1 for unknown
            super(LicensePlateNet, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((4, 4))
            )
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(128 * 4 * 4, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
        
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    
    return LicensePlateNet()

def _normalize_box(box_candidate) -> List[Tuple[int, int]]:
    """
    Convert different box coordinate formats to a standard list of (x, y) tuples.
    
    PaddleOCR can return bounding boxes in various formats:
    - Dictionary with nested coordinates
    - Flat list like [x1, y1, x2, y2, x3, y3, x4, y4]
    - Nested list like [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    
    This function standardizes all formats to: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    
    Args:
        box_candidate: Raw bounding box data from PaddleOCR
        
    Returns:
        List of (x, y) coordinate tuples representing the box corners
    """
    # Handle None or empty input
    if box_candidate is None:
        return []
    
    # Handle dictionary format (newer PaddleOCR versions)
    # Example: {"text_region": [[x1,y1], [x2,y2], ...]}
    if isinstance(box_candidate, dict):
        # Try to find coordinates in common dictionary keys
        inner = (box_candidate.get("text_region") 
                or box_candidate.get("points") 
                or box_candidate.get("box"))
        # Recursively process the inner data
        return _normalize_box(inner)
    
    # Handle string or bytes (invalid for coordinates)
    if isinstance(box_candidate, (str, bytes)):
        return []
    
    # Handle sequence types (lists, tuples)
    if isinstance(box_candidate, Sequence):
        # Empty sequence
        if len(box_candidate) == 0:
            return []
        
        # If first element is also a list/tuple, it's nested format
        # Example: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        if isinstance(box_candidate[0], (list, tuple)):
            return _normalize_box(list(box_candidate))
        
        # Flat format: [x1, y1, x2, y2, x3, y3, x4, y4]
        # Must have even number of elements (pairs of x,y)
        if len(box_candidate) % 2 != 0:
            return []
        
        # Convert flat list to pairs: [x1,y1,x2,y2] -> [(x1,y1), (x2,y2)]
        iterator = iter(box_candidate)
        return [(int(round(x)), int(round(y))) for x, y in zip(iterator, iterator)]
    
    # Handle other iterable types (like numpy arrays)
    if isinstance(box_candidate, Iterable):
        pts: List[Tuple[int, int]] = []
        for point in box_candidate:
            # Each point should be iterable (containing x, y coordinates)
            if not isinstance(point, Iterable):
                return []
            try:
                # Extract x, y from the point (take first 2 elements)
                x, y = point[:2]
            except (TypeError, IndexError):
                # If we can't extract x, y, return empty list
                return []
            # Convert to integers and add to points list
            pts.append((int(round(x)), int(round(y))))
        return pts
    
    # If none of the above formats match, return empty list
    return []

def _normalize_ocr_result(ocr_result) -> List[Tuple[List[Tuple[int, int]], str, float]]:
    """
    Convert PaddleOCR results to a standardized format.
    
    PaddleOCR can return results in different formats depending on the version:
    - New format: Dictionary with separate lists for texts, scores, and polygons
    - Old format: List of [bounding_box, (text, confidence_score)]
    
    This function converts both formats to: [(bounding_box, text, confidence_score), ...]
    
    Args:
        ocr_result: Raw output from PaddleOCR
        
    Returns:
        List of tuples, each containing:
        - bounding_box: List of (x,y) coordinate pairs
        - text: Recognized text string
        - confidence_score: Float between 0-1 indicating recognition confidence
    """
    # Handle empty or None results
    if not ocr_result:
        return []
    
    # Get the first item to determine the format
    first_item = ocr_result[0]
    normalized: List[Tuple[List[Tuple[int, int]], str, float]] = []
    
    # NEW FORMAT: Dictionary with separate arrays
    # Example: {"rec_texts": ["ABC123"], "rec_scores": [0.95], "rec_polys": [[[x1,y1],[x2,y2]...]]}
    if isinstance(first_item, dict):
        # Extract the three main components
        rec_texts = first_item.get("rec_texts", [])      # List of recognized texts
        rec_scores = first_item.get("rec_scores", [])    # List of confidence scores
        rec_polys = first_item.get("rec_polys", [])      # List of bounding polygons
        
        # Process each detected text region
        for i, text in enumerate(rec_texts):
            # Skip empty or whitespace-only text
            if not text or text.strip() == "":
                continue
                
            # Get corresponding score and polygon (with fallbacks)
            score = rec_scores[i] if i < len(rec_scores) else 0.0
            poly = rec_polys[i] if i < len(rec_polys) else []
            
            # Add to normalized results
            normalized.append((_normalize_box(poly), text, float(score)))
        
        return normalized
    
    # OLD FORMAT: List of [bounding_box, text_info]
    # Example: [[[x1,y1],[x2,y2],...], ("ABC123", 0.95)]
    if isinstance(first_item, list):
        for entry in first_item:
            # Each entry should be a list/tuple with at least 2 elements
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                continue
            
            # Extract bounding box and text details
            box_raw = entry[0]      # Bounding box coordinates
            details = entry[1]      # Text and confidence information
            text = ""
            score = None
            
            # Parse text details based on format
            if isinstance(details, (list, tuple)):
                # Format: ("text", confidence_score)
                if details:
                    text = details[0]
                    if len(details) > 1:
                        score = details[1]
            elif isinstance(details, dict):
                # Format: {"text": "ABC123", "score": 0.95}
                text = details.get("text", "")
                score = details.get("score") or details.get("confidence")
            else:
                # Format: just the text string
                text = str(details)
            
            # Add to normalized results with fallback score
            normalized.append((_normalize_box(box_raw), text, score if score is not None else 0.0))
        return normalized
    
    # If format is unrecognized, return empty list
    return normalized

def analyze_with_vlm(image_path: str) -> Optional[dict]:
    """
    🧠 Enhanced License Plate Analysis using Vision Language Model (VLM)
    
    This function uses BLIP (Bootstrapping Language-Image Pre-training) to:
    1. Generate natural language descriptions of the license plate
    2. Extract text through AI visual understanding
    3. Validate and cross-check OCR results
    4. Provide confidence scoring for detected text
    
    Args:
        image_path (str): Path to the license plate image
        
    Returns:
        dict: VLM analysis results including text, confidence, and description
    """
    if not VLM_AVAILABLE or vlm_processor is None or vlm_model is None:
        return None
    
    try:
        # Load and preprocess image for VLM
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        # Convert BGR to RGB for VLM processing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image for VLM
        from PIL import Image
        pil_image = Image.fromarray(img_rgb)
        
        # Resize if too large (VLM works better with smaller images)
        if max(pil_image.size) > 512:
            pil_image.thumbnail((512, 512), Image.Resampling.LANCZOS)
        
        # Prepare VLM inputs
        inputs = vlm_processor(pil_image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            # Generate general description
            caption_ids = vlm_model.generate(**inputs, max_length=50)
            caption = vlm_processor.decode(caption_ids[0], skip_special_tokens=True)
            
            # Specific prompts for license plate analysis
            prompts = [
                "The license plate number is",
                "The text on this license plate reads",
                "This license plate shows",
                "The characters on this plate are"
            ]
            
            responses = {}
            for prompt in prompts:
                try:
                    prompt_inputs = vlm_processor(pil_image, prompt, return_tensors="pt").to(device)
                    response_ids = vlm_model.generate(**prompt_inputs, max_length=30)
                    response = vlm_processor.decode(response_ids[0], skip_special_tokens=True)
                    responses[prompt] = response.strip()
                except:
                    responses[prompt] = ""
        
        # Extract license plate text from VLM responses
        extracted_text = extract_license_plate_from_vlm(responses, caption)
        
        return {
            'caption': caption,
            'responses': responses,
            'extracted_text': extracted_text,
            'confidence': calculate_vlm_confidence(extracted_text, responses),
            'method': 'BLIP_VLM'
        }
        
    except Exception as e:
        print(f"   ⚠️ VLM analysis error: {e}")
        return None

def extract_license_plate_from_vlm(responses: dict, caption: str) -> str:
    """
    🔍 Extract license plate text from VLM responses using pattern matching.
    
    This function analyzes VLM responses to find the most likely license plate text
    by looking for common license plate patterns and formats.
    
    Args:
        responses (dict): VLM responses to different prompts
        caption (str): General image caption from VLM
        
    Returns:
        str: Extracted license plate text (cleaned and validated)
    """
    # Combine all text responses
    all_text = " ".join(responses.values()) + " " + caption
    all_text = all_text.upper()
    
    # Common license plate patterns (adjust for your region)
    patterns = [
        r'\b[A-Z]{1,3}\s*\d{1,4}[A-Z]?\b',  # ABC123, AB1234C
        r'\b\d{1,3}\s*[A-Z]{1,3}\s*\d{1,4}\b',  # 123ABC456
        r'\b[A-Z]{2,3}\s*\d{2,4}\b',  # ABC123, AB1234
        r'\b\d{1,4}\s*[A-Z]{2,4}\b',  # 123ABC, 1234ABCD
        r'\b[A-Z0-9]{5,8}\b',  # General alphanumeric 5-8 chars
    ]
    
    candidates = []
    
    # Extract potential license plate texts using patterns
    for pattern in patterns:
        matches = re.findall(pattern, all_text)
        for match in matches:
            # Clean the match (remove extra spaces)
            cleaned = re.sub(r'\s+', '', match)
            if 4 <= len(cleaned) <= 10:  # Reasonable license plate length
                candidates.append(cleaned)
    
    # If no pattern matches, try to extract any alphanumeric sequence
    if not candidates:
        words = all_text.split()
        for word in words:
            # Look for words that could be license plates
            cleaned = re.sub(r'[^A-Z0-9]', '', word)
            if (4 <= len(cleaned) <= 10 and 
                any(c.isdigit() for c in cleaned) and 
                any(c.isalpha() for c in cleaned)):
                candidates.append(cleaned)
    
    # Return the most likely candidate (longest valid sequence)
    if candidates:
        # Sort by length and alphanumeric balance
        candidates.sort(key=lambda x: (len(x), sum(c.isalnum() for c in x)), reverse=True)
        return candidates[0]
    
    return ""

def calculate_vlm_confidence(extracted_text: str, responses: dict) -> float:
    """
    📊 Calculate confidence score for VLM-extracted license plate text.
    
    Args:
        extracted_text (str): Extracted license plate text
        responses (dict): VLM responses
        
    Returns:
        float: Confidence score between 0.0 and 1.0
    """
    if not extracted_text:
        return 0.0
    
    # Count how many responses contain the extracted text
    matches = 0
    total_responses = len(responses)
    
    for response in responses.values():
        if extracted_text.upper() in response.upper():
            matches += 1
    
    # Base confidence on consistency across responses
    consistency_score = matches / total_responses if total_responses > 0 else 0
    
    # Bonus for reasonable license plate characteristics
    length_bonus = 0.1 if 5 <= len(extracted_text) <= 8 else 0
    alphanumeric_bonus = 0.1 if (any(c.isdigit() for c in extracted_text) and 
                                 any(c.isalpha() for c in extracted_text)) else 0
    
    confidence = min(1.0, consistency_score + length_bonus + alphanumeric_bonus)
    return confidence

def validate_license_plate(text: str) -> dict:
    """
    ✅ Validate and score license plate text based on common patterns.
    
    Args:
        text (str): License plate text to validate
        
    Returns:
        dict: Validation results with score and reasoning
    """
    if not text:
        return {'valid': False, 'score': 0.0, 'issues': ['Empty text']}
    
    issues = []
    score = 1.0
    
    # Length check
    if len(text) < 4:
        issues.append('Too short (< 4 characters)')
        score -= 0.3
    elif len(text) > 10:
        issues.append('Too long (> 10 characters)')
        score -= 0.2
    
    # Character composition check
    if not any(c.isdigit() for c in text):
        issues.append('No digits found')
        score -= 0.2
    
    if not any(c.isalpha() for c in text):
        issues.append('No letters found')
        score -= 0.2
    
    # Invalid characters check
    if not text.isalnum():
        issues.append('Contains non-alphanumeric characters')
        score -= 0.1
    
    # Common patterns bonus
    if re.match(r'^[A-Z]{1,3}\d{1,4}[A-Z]?$', text):
        score += 0.1  # Common format bonus
    
    score = max(0.0, score)
    
    return {
        'valid': score >= 0.5,
        'score': score,
        'issues': issues
    }

def analyze_with_custom_model(image_path: str) -> Optional[dict]:
    """
    🎯 Analyze license plate using custom PyTorch model (last.pt)
    
    This function uses the custom trained model to:
    1. Preprocess the image for the model
    2. Run inference to detect license plate text
    3. Post-process results for character recognition
    4. Provide confidence scoring
    
    Args:
        image_path (str): Path to the license plate image
        
    Returns:
        dict: Custom model analysis results
    """
    if not TORCH_AVAILABLE or custom_model is None:
        return None
    
    try:
        # Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)
        
        # Standard preprocessing for license plate models
        transform = transforms.Compose([
            transforms.Resize((64, 256)),  # Common license plate aspect ratio
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Prepare input tensor
        input_tensor = transform(pil_image).unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = custom_model(input_tensor)
            
            # Handle different output formats
            if isinstance(outputs, torch.Tensor):
                # Classification output - convert to probabilities
                probabilities = torch.softmax(outputs, dim=1)
                confidence = torch.max(probabilities).item()
                
                # Convert to character predictions (this is model-specific)
                predicted_text = decode_model_output(outputs)
                
            elif isinstance(outputs, dict):
                # Dictionary output with multiple keys
                predicted_text = outputs.get('text', '')
                confidence = outputs.get('confidence', 0.0)
            else:
                # Unknown format
                predicted_text = ''
                confidence = 0.0
        
        return {
            'text': predicted_text,
            'confidence': confidence,
            'method': 'Custom_PyTorch_Model',
            'model_path': 'last.pt'
        }
        
    except Exception as e:
        print(f"   ⚠️ Custom model analysis error: {e}")
        return None

def decode_model_output(outputs: torch.Tensor) -> str:
    """
    🔤 Decode model output tensor to license plate text.
    
    This function converts model predictions to readable text.
    The exact implementation depends on how the model was trained.
    
    Args:
        outputs (torch.Tensor): Raw model output
        
    Returns:
        str: Decoded license plate text
    """
    try:
        # Character mapping (adjust based on your model's training)
        chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        
        # Get predictions
        if len(outputs.shape) == 2:  # Batch x Classes
            predictions = torch.argmax(outputs, dim=1)
            # Convert to text (assuming single character prediction)
            if len(predictions) == 1:
                char_idx = predictions[0].item()
                if 0 <= char_idx < len(chars):
                    return chars[char_idx]
        
        elif len(outputs.shape) == 3:  # Batch x Sequence x Classes
            # Sequence prediction (multiple characters)
            predictions = torch.argmax(outputs, dim=2)
            text = ''
            for char_pred in predictions[0]:  # First batch item
                char_idx = char_pred.item()
                if 0 <= char_idx < len(chars):
                    text += chars[char_idx]
            return text
        
        return ''
        
    except Exception as e:
        print(f"   ⚠️ Decode error: {e}")
        return ''

def preprocess_for_custom_model(image_path: str) -> Optional[torch.Tensor]:
    """
    🖼️ Preprocess image specifically for the custom model.
    
    Args:
        image_path (str): Path to input image
        
    Returns:
        torch.Tensor: Preprocessed image tensor ready for model input
    """
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # License plate specific preprocessing
        # 1. Resize to standard license plate dimensions
        img_resized = cv2.resize(img_rgb, (256, 64))
        
        # 2. Normalize
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # 3. Convert to tensor
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)
        
        return img_tensor.to(device)
        
    except Exception as e:
        print(f"   ⚠️ Preprocessing error: {e}")
        return None

def select_best_result(candidates: List[dict]) -> Tuple[str, float, str]:
    """
    🏆 Select the best license plate text from multiple AI model results.
    
    This function implements a sophisticated consensus algorithm that:
    1. Checks for agreement between models (consensus boost)
    2. Validates text quality and format
    3. Weighs confidence scores from different models
    4. Applies domain-specific knowledge for license plates
    
    Args:
        candidates (List[dict]): List of results from different models
        
    Returns:
        Tuple[str, float, str]: (best_text, confidence_score, selected_method)
    """
    if not candidates:
        return "", 0.0, "None"
    
    if len(candidates) == 1:
        result = candidates[0]
        return result['text'], result['confidence'], result['method']
    
    # Step 1: Check for consensus (multiple models agreeing)
    text_counts = {}
    for candidate in candidates:
        text = candidate['text'].upper().replace(' ', '')  # Normalize for comparison
        if text not in text_counts:
            text_counts[text] = []
        text_counts[text].append(candidate)
    
    # Step 2: Apply consensus bonus
    for text, matching_candidates in text_counts.items():
        if len(matching_candidates) > 1:
            print(f"   🤝 Consensus found: {len(matching_candidates)} models agree on '{text}'")
            # Boost confidence for consensus
            for candidate in matching_candidates:
                candidate['confidence'] = min(1.0, candidate['confidence'] + 0.2)
    
    # Step 3: Validate each candidate
    scored_candidates = []
    for candidate in candidates:
        validation = validate_license_plate(candidate['text'])
        
        # Combine model confidence with validation score
        combined_score = (candidate['confidence'] * 0.7) + (validation['score'] * 0.3)
        
        scored_candidates.append({
            'text': candidate['text'],
            'confidence': combined_score,
            'method': candidate['method'],
            'validation': validation,
            'original_confidence': candidate['confidence']
        })
    
    # Step 4: Sort by combined score and select best
    scored_candidates.sort(key=lambda x: x['confidence'], reverse=True)
    best_candidate = scored_candidates[0]
    
    # Step 5: Apply method-specific preferences if scores are close
    if len(scored_candidates) > 1:
        score_diff = best_candidate['confidence'] - scored_candidates[1]['confidence']
        
        # If scores are very close (< 0.1), prefer certain methods
        if score_diff < 0.1:
            method_preferences = {
                'Custom_PyTorch': 3,  # Highest preference (domain-specific)
                'PaddleOCR': 2,       # Medium preference (proven OCR)
                'BLIP_VLM': 1         # Lower preference (general purpose)
            }
            
            # Re-sort considering method preferences
            for candidate in scored_candidates:
                method_bonus = method_preferences.get(candidate['method'], 0) * 0.05
                candidate['confidence'] += method_bonus
            
            scored_candidates.sort(key=lambda x: x['confidence'], reverse=True)
            best_candidate = scored_candidates[0]
    
    return best_candidate['text'], best_candidate['confidence'], best_candidate['method']

def analyze_plate_enhanced(image_path):
    """
    🚀 Enhanced License Plate Analysis with Multi-Model Approach
    
    This advanced function combines multiple AI technologies:
    1. PaddleOCR for precise text detection and character localization
    2. BLIP VLM for intelligent visual understanding and validation
    3. Advanced pixel density analysis for character integrity assessment
    4. Cross-validation between different AI models for higher accuracy
    
    Args:
        image_path (str): Path to the license plate image file
        
    Returns:
        dict: Comprehensive analysis results including:
            - complete_text: Final validated license plate text
            - character_analysis: Character-by-character integrity assessment
            - ocr_results: Raw OCR detection results
            - vlm_results: VLM analysis results (if available)
            - confidence_scores: Confidence metrics from different methods
            - validation_results: Text validation and quality scores
    """
    print(f"📸 Enhanced Analysis: {os.path.basename(image_path)}")
    
    # Step 1: Load and validate the image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Unable to read image at {image_path}")
    
    print(f"   ✅ Image loaded: {img.shape[1]}x{img.shape[0]} pixels")
    
    # Step 2: Prepare image for different analysis methods
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 35, 11)
    
    # Step 3: PaddleOCR Analysis
    print("🔍 Running PaddleOCR analysis...")
    ocr_result = ocr.predict(image_path)
    ocr_lines = _normalize_ocr_result(ocr_result)
    
    # Step 4: Custom PyTorch Model Analysis (if available)
    custom_result = None
    if custom_model is not None:
        print("🎯 Running Custom PyTorch Model analysis...")
        custom_result = analyze_with_custom_model(image_path)
    
    # Step 5: VLM Analysis (if available)
    vlm_result = None
    if VLM_AVAILABLE:
        print("🧠 Running VLM analysis...")
        vlm_result = analyze_with_vlm(image_path)
    
    # Step 6: Process OCR results with integrity analysis
    ocr_chars = []
    ocr_integrity = []
    ocr_confidences = []
    
    for box_points, text, confidence in ocr_lines:
        if not text:
            continue
        
        print(f"   📝 OCR detected: '{text}' (confidence: {confidence:.2f})")
        ocr_chars.append(text)
        ocr_confidences.append(confidence)
        
        # Integrity analysis
        if not box_points:
            ocr_integrity.append("Missing")
            continue
        
        # Extract character region for pixel analysis
        x_coords = [p[0] for p in box_points]
        y_coords = [p[1] for p in box_points]
        x_min = max(min(x_coords), 0)
        x_max = min(max(x_coords), binary.shape[1])
        y_min = max(min(y_coords), 0)
        y_max = min(max(y_coords), binary.shape[0])
        
        if x_min >= x_max or y_min >= y_max:
            ocr_integrity.append("Missing")
            continue
        
        char_crop = binary[y_min:y_max, x_min:x_max]
        if char_crop.size == 0:
            ocr_integrity.append("Missing")
            continue
        
        # Pixel density analysis
        filled_ratio = np.sum(char_crop == 0) / char_crop.size
        
        if filled_ratio < 0.25:
            ocr_integrity.append("Broken/Missing")
            print(f"   ❌ Character '{text}' damaged ({filled_ratio:.1%} filled)")
        else:
            ocr_integrity.append("Intact")
            print(f"   ✅ Character '{text}' intact ({filled_ratio:.1%} filled)")
    
    # Step 7: Combine and validate results from all methods
    ocr_text = "".join(ocr_chars)
    vlm_text = vlm_result['extracted_text'] if vlm_result else ""
    custom_text = custom_result['text'] if custom_result else ""
    
    # Collect all results with their confidence scores
    results_candidates = []
    
    # OCR result
    ocr_confidence = np.mean(ocr_confidences) if ocr_confidences else 0.0
    if ocr_text:
        results_candidates.append({
            'text': ocr_text,
            'confidence': ocr_confidence,
            'method': 'PaddleOCR'
        })
    
    # Custom model result
    if custom_result and custom_text:
        print(f"   🎯 Custom Model detected: '{custom_text}' (confidence: {custom_result['confidence']:.2f})")
        results_candidates.append({
            'text': custom_text,
            'confidence': custom_result['confidence'],
            'method': 'Custom_PyTorch'
        })
    
    # VLM result
    if vlm_result and vlm_text:
        print(f"   🧠 VLM detected: '{vlm_text}' (confidence: {vlm_result['confidence']:.2f})")
        results_candidates.append({
            'text': vlm_text,
            'confidence': vlm_result['confidence'],
            'method': 'BLIP_VLM'
        })
    
    # Multi-model consensus and selection
    final_text, confidence_score, selected_method = select_best_result(results_candidates)
    
    print(f"   🏆 Best result selected from {selected_method}: '{final_text}' (confidence: {confidence_score:.2f})")
    
    # Step 7: Validate final result
    validation = validate_license_plate(final_text)
    print(f"   📊 Validation score: {validation['score']:.2f} ({'Valid' if validation['valid'] else 'Invalid'})")
    
    if validation['issues']:
        print(f"   ⚠️ Issues: {', '.join(validation['issues'])}")
    
    # Step 8: Compile comprehensive results
    methods_used = []
    if ocr_text: methods_used.append('PaddleOCR')
    if custom_result: methods_used.append('Custom PyTorch')
    if vlm_result: methods_used.append('BLIP VLM')
    
    results = {
        'complete_text': final_text,
        'character_analysis': list(zip(ocr_chars, ocr_integrity)) if ocr_chars else [],
        'confidence_score': confidence_score,
        'selected_method': selected_method,
        'validation': validation,
        'ocr_results': {
            'text': ocr_text,
            'characters': ocr_chars,
            'integrity': ocr_integrity,
            'confidences': ocr_confidences,
            'method': 'PaddleOCR'
        },
        'custom_model_results': custom_result,
        'vlm_results': vlm_result,
        'analysis_method': f"Multi-Model ({', '.join(methods_used)})" if len(methods_used) > 1 else methods_used[0] if methods_used else 'None',
        'image_path': image_path,
        'timestamp': datetime.now().isoformat(),
        'all_candidates': results_candidates
    }
    
    print(f"🏁 Enhanced analysis complete. Final text: '{final_text}' (confidence: {confidence_score:.2f})")
    return results

# Backward compatibility function
def analyze_plate(image_path):
    """
    📸 Backward compatibility wrapper for the enhanced analysis function.
    
    This maintains compatibility with existing code while providing enhanced results.
    """
    results = analyze_plate_enhanced(image_path)
    return results['complete_text'], results['character_analysis']

def process_folders(front_folder, rear_folder, output_csv=None):
    """
    Process entire folders of front and rear vehicle images and generate a comprehensive CSV report.
    
    This function:
    1. Scans both front and rear image folders
    2. Matches images by filename (assumes same car has same filename in both folders)
    3. Analyzes each image pair for license plate character integrity
    4. Compares front vs rear plates to detect mismatches
    5. Generates a detailed CSV report with all findings
    
    Args:
        front_folder (str): Path to folder containing front vehicle images
        rear_folder (str): Path to folder containing rear vehicle images  
        output_csv (str, optional): Path for output CSV file. Defaults to "output/plate_analysis.csv"
    """
    print("🚀 Starting batch processing of license plate images...")
    
    # Set default output path if not provided
    if output_csv is None:
        output_csv = os.path.join("output", "plate_analysis.csv")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    print(f"📁 Output will be saved to: {output_csv}")
    
    # Verify that both input folders exist
    if not os.path.exists(front_folder):
        print(f"❌ Front folder not found: {front_folder}")
        print("Please ensure your front images are in the correct folder")
        return
    
    if not os.path.exists(rear_folder):
        print(f"❌ Rear folder not found: {rear_folder}")
        print("Please ensure your rear images are in the correct folder")
        return
    
    # Get sorted lists of files in both folders
    # Sorting ensures consistent processing order
    front_files = sorted(os.listdir(front_folder))
    rear_files = sorted(os.listdir(rear_folder))
    
    print(f"📊 Found {len(front_files)} front images and {len(rear_files)} rear images")
    
    # Create and write to CSV file
    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # Write enhanced CSV header row with additional AI analysis columns
        writer.writerow([
            "Car_ID", "Image", "Final_Text", "Character", "Integrity", "Mismatch",
            "Confidence_Score", "Selected_Method", "OCR_Text", "Custom_Model_Text", 
            "VLM_Text", "Validation_Score", "Analysis_Method"
        ])
        
        # Process each front image
        processed_count = 0
        for fname in front_files:
            # Skip non-image files (like .txt, .md files)
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                continue
                
            # Construct full paths for front and rear images
            front_img = os.path.join(front_folder, fname)
            rear_img = os.path.join(rear_folder, fname)  # Assumes matching filenames
            
            # Check if corresponding rear image exists
            if not os.path.exists(rear_img):
                print(f"⚠️ Rear image missing for {fname}, skipping...")
                continue
            
            try:
                print(f"\n🔄 Processing pair {processed_count + 1}: {fname}")
                
                # Analyze both front and rear images with enhanced multi-model approach
                print("   📸 Analyzing front image with multi-model AI...")
                front_results = analyze_plate_enhanced(front_img)
                front_text = front_results['complete_text']
                front_data = front_results['character_analysis']
                
                print("   📸 Analyzing rear image with multi-model AI...")
                rear_results = analyze_plate_enhanced(rear_img)
                rear_text = rear_results['complete_text']
                rear_data = rear_results['character_analysis']
                
                # Compare front and rear license plate text
                # If they don't match, there might be an issue with one of the plates
                mismatch = "Yes" if front_text != rear_text else "No"
                
                if mismatch == "Yes":
                    print(f"   ⚠️ MISMATCH DETECTED: Front='{front_text}' vs Rear='{rear_text}'")
                else:
                    print(f"   ✅ Plates match: '{front_text}'")
                
                # Write enhanced front image results to CSV
                for char, status in front_data:
                    writer.writerow([
                        fname, "Front", front_text, char, status, mismatch,
                        f"{front_results['confidence_score']:.3f}",
                        front_results['selected_method'],
                        front_results['ocr_results']['text'],
                        front_results['custom_model_results']['text'] if front_results['custom_model_results'] else '',
                        front_results['vlm_results']['extracted_text'] if front_results['vlm_results'] else '',
                        f"{front_results['validation']['score']:.3f}",
                        front_results['analysis_method']
                    ])
                
                # Write enhanced rear image results to CSV
                for char, status in rear_data:
                    writer.writerow([
                        fname, "Rear", rear_text, char, status, mismatch,
                        f"{rear_results['confidence_score']:.3f}",
                        rear_results['selected_method'],
                        rear_results['ocr_results']['text'],
                        rear_results['custom_model_results']['text'] if rear_results['custom_model_results'] else '',
                        rear_results['vlm_results']['extracted_text'] if rear_results['vlm_results'] else '',
                        f"{rear_results['validation']['score']:.3f}",
                        rear_results['analysis_method']
                    ])
                
                # Save detailed JSON results for this image pair
                json_output_dir = os.path.join(os.path.dirname(output_csv), "detailed_analysis")
                os.makedirs(json_output_dir, exist_ok=True)
                
                json_filename = os.path.join(json_output_dir, f"{os.path.splitext(fname)[0]}_analysis.json")
                detailed_results = {
                    'image_pair': fname,
                    'front_analysis': front_results,
                    'rear_analysis': rear_results,
                    'mismatch_detected': mismatch == "Yes",
                    'processing_timestamp': datetime.now().isoformat()
                }
                
                with open(json_filename, 'w', encoding='utf-8') as json_file:
                    json.dump(detailed_results, json_file, indent=2, ensure_ascii=False)
                
                processed_count += 1
                print(f"   ✅ Successfully processed {fname}")
                print(f"   📄 Detailed analysis saved: {json_filename}")
                
            except Exception as e:
                # If there's an error processing this image pair, log it and continue
                print(f"   ❌ Error processing {fname}: {e}")
                continue
    
    print(f"\n🎉 Enhanced Multi-Model Analysis Complete!")
    print(f"📊 Successfully processed {processed_count} image pairs")
    print(f"📄 Enhanced CSV results: {output_csv}")
    print(f"� Detailed CJSON analyses: {os.path.join(os.path.dirname(output_csv), 'detailed_analysis')}")
    
    # Print analysis summary
    methods_used = []
    if custom_model is not None: methods_used.append("Custom PyTorch Model (last.pt)")
    if VLM_AVAILABLE: methods_used.append("BLIP Vision-Language Model")
    methods_used.append("PaddleOCR")
    
    print(f"\n🧠 AI Models Used:")
    for i, method in enumerate(methods_used, 1):
        print(f"   {i}. {method}")
    
    print(f"\n💡 Enhanced CSV includes:")
    print(f"   • Multi-model consensus results")
    print(f"   • Confidence scores from each AI method")
    print(f"   • Validation scores and quality metrics")
    print(f"   • Method selection reasoning")
    print(f"   • Character integrity analysis")

# MAIN PROGRAM EXECUTION
# This section runs when the script is executed directly (not imported as a module)
if __name__ == "__main__":
    """
    Main execution block - runs the license plate analysis program.
    
    Expected folder structure:
    Q1_License Plate/
    ├── data/
    │   ├── front/          # Put front vehicle images here
    │   │   ├── car001.jpg
    │   │   ├── car002.jpg
    │   │   └── ...
    │   └── rear/           # Put rear vehicle images here (matching filenames)
    │       ├── car001.jpg
    │       ├── car002.jpg
    │       └── ...
    ├── output/             # Results will be saved here
    └── Q1_code.py         # This script
    """
    
    print("=" * 80)
    print("🚗 ENHANCED MULTI-MODEL LICENSE PLATE ANALYZER")
    print("=" * 80)
    print("🧠 This advanced system combines multiple AI models for maximum accuracy:")
    
    models_info = []
    if custom_model is not None:
        models_info.append("✅ Custom PyTorch Model (last.pt) - Domain-specific training")
    else:
        models_info.append("❌ Custom PyTorch Model (last.pt) - Not loaded")
    
    if VLM_AVAILABLE:
        models_info.append("✅ BLIP Vision-Language Model - AI visual understanding")
    else:
        models_info.append("❌ BLIP VLM - Not available")
    
    models_info.append("✅ PaddleOCR - Professional text recognition")
    
    for info in models_info:
        print(f"   {info}")
    
    print(f"\n🎯 Features:")
    print(f"   • Multi-model consensus for higher accuracy")
    print(f"   • Character integrity analysis")
    print(f"   • Cross-validation between AI methods")
    print(f"   • Detailed confidence scoring")
    print(f"   • Enhanced CSV and JSON reporting\n")
    
    # Define folder paths using relative paths for better portability
    # This allows the script to work regardless of where it's placed
    front_folder = os.path.join("data", "front")
    rear_folder = os.path.join("data", "rear")
    
    print("📁 Checking for required folders...")
    
    # Verify that the required folders exist before starting analysis
    if not os.path.exists(front_folder):
        print(f"❌ Front folder not found: {front_folder}")
        print("\n📋 SETUP INSTRUCTIONS:")
        print("1. Create a 'data' folder in the same directory as this script")
        print("2. Create a 'front' subfolder inside 'data'")
        print("3. Place all front vehicle images in 'data/front/'")
        print("4. Ensure images are in common formats (jpg, png, etc.)")
        exit(1)
        
    if not os.path.exists(rear_folder):
        print(f"❌ Rear folder not found: {rear_folder}")
        print("\n📋 SETUP INSTRUCTIONS:")
        print("1. Create a 'rear' subfolder inside the 'data' folder")
        print("2. Place all rear vehicle images in 'data/rear/'")
        print("3. Ensure rear images have the same filenames as front images")
        print("   (e.g., if front has 'car001.jpg', rear should also have 'car001.jpg')")
        exit(1)
    
    # Count images in each folder to give user feedback
    front_count = len([f for f in os.listdir(front_folder) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))])
    rear_count = len([f for f in os.listdir(rear_folder) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))])
    
    print(f"✅ Found {front_count} front images and {rear_count} rear images")
    
    if front_count == 0:
        print("❌ No images found in front folder. Please add some images and try again.")
        exit(1)
        
    if rear_count == 0:
        print("❌ No images found in rear folder. Please add some images and try again.")
        exit(1)
    
    # Ask user for confirmation before starting (optional - can be removed for automation)
    print(f"\n🚀 Ready to analyze {min(front_count, rear_count)} image pairs.")
    input("Press Enter to start the analysis, or Ctrl+C to cancel...")
    
    # Start the main processing function
    process_folders(front_folder, rear_folder)
    
    print("\n" + "=" * 60)
    print("🎉 ANALYSIS COMPLETE!")
    print("=" * 60)
    print("📊 Check the 'output' folder for your CSV report.")
    print("💡 You can open the CSV file in Excel, Google Sheets, or any spreadsheet program.")
    print("📈 The report contains detailed character-by-character analysis for each license plate.")
    print("\n🔍 CSV Columns Explained:")
    print("  • Car_ID: Image filename")
    print("  • Image: 'Front' or 'Rear'")
    print("  • OCR_Text: Complete license plate text detected")
    print("  • Character: Individual character or character group")
    print("  • Integrity: 'Intact', 'Broken/Missing', or 'Missing'")
    print("  • Mismatch: 'Yes' if front and rear plates don't match")
    print("\n✨ Thank you for using the License Plate Analyzer!")