from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import face_recognition
import numpy as np
from PIL import Image
import io
import requests
import time
from typing import Dict, Any
import tempfile
import os
import gc
from ultralytics import YOLO

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

EMPLOYEE_API_URL = "https://hrm.umanerp.com/api/users/getEmployee"
IMG_BASE_URL = "https://hrm.umanerp.com/"

# Global variables for model management
current_model_path = 'best.pt'
loaded_models = {}  # Cache for loaded models

class AIFakeDetector:
    def __init__(self, model_path='best.pt'):
        """Initialize AI model for fake vs real detection"""
        self.model_path = model_path
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load or reload AI model"""
        try:
            print(f"Loading AI model from {model_path}...")
            # Load with specific settings to avoid memory issues
            self.model = YOLO(model_path)
            self.model.overrides['verbose'] = False  # Reduce verbosity
            self.model_path = model_path
            print(f"✅ AI Fake Detection Model ({model_path}) loaded successfully!")
            self.model_loaded = True
        except Exception as e:
            print(f"❌ Error loading AI model {model_path}: {str(e)}")
            self.model_loaded = False
            self.model = None
    
    def predict_fake_real(self, image_path_or_array, confidence_threshold=0.3):
        """Predict if image is fake or real using trained model"""
        if not self.model_loaded:
            return {
                "is_real": False,  # Default to FAKE if model not loaded (safer)
                "confidence": 0.0,
                "error": "AI model not loaded"
            }
        
        try:
            # Run inference with memory optimization
            results = self.model(image_path_or_array, conf=confidence_threshold, verbose=False, imgsz=416)
            
            if not results or len(results) == 0:
                return {
                    "is_real": False,  # Default to FAKE if no detection (safer)
                    "confidence": 0.0,
                    "detected_class": "unknown"
                }
            
            # Get the first result
            result = results[0]
            
            # Extract predictions
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                if len(boxes) > 0:
                    # Get highest confidence detection
                    confidences = boxes.conf.cpu().numpy()
                    classes = boxes.cls.cpu().numpy()
                    
                    # Get the detection with highest confidence
                    max_conf_idx = np.argmax(confidences)
                    max_confidence = confidences[max_conf_idx]
                    predicted_class = int(classes[max_conf_idx])
                    
                    # Try both possibilities: 0 = fake, 1 = real OR 0 = real, 1 = fake
                    # Let's make it stricter - only high confidence real images pass
                    class_names = ['fake', 'real']
                    detected_class = class_names[predicted_class] if predicted_class < len(class_names) else 'unknown'
                    
                    # Only allow if detected as 'real' with high confidence (>0.6 for better sensitivity)
                    is_real = (detected_class == 'real' and max_confidence > 0.6)
                    
                    return {
                        "is_real": is_real,
                        "confidence": float(max_confidence),
                        "detected_class": detected_class
                    }
            
            # If no boxes detected, default to FAKE (safer)
            return {
                "is_real": False,
                "confidence": 0.0,
                "detected_class": "no_detection"
            }
            
        except Exception as e:
            print(f"AI prediction error: {str(e)}")
            return {
                "is_real": False,  # Default to FAKE on error (safer)
                "confidence": 0.0,
                "error": str(e)
            }
    
    def analyze_image(self, image_array: np.ndarray):
        """Analyze image using AI model with phone screen detection"""
        try:
            # First check for phone screen characteristics
            screen_detection = self.detect_phone_screen(image_array)
            if screen_detection["is_screen"]:
                return {
                    "is_real": False,
                    "confidence": screen_detection["confidence"],
                    "detected_class": "phone_screen",
                    "screen_detection": screen_detection
                }
            
            # Resize image to reduce memory usage
            max_size = 640
            height, width = image_array.shape[:2]
            if max(height, width) > max_size:
                if height > width:
                    new_height = max_size
                    new_width = int(width * (max_size / height))
                else:
                    new_width = max_size
                    new_height = int(height * (max_size / width))
                
                pil_image = Image.fromarray(image_array.astype(np.uint8))
                pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                image_array = np.array(pil_image)
            
            # Save image temporarily for model inference
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_img:
                # Convert numpy array to PIL and save
                if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                    pil_image = Image.fromarray(image_array.astype(np.uint8))
                else:
                    pil_image = Image.fromarray(image_array.astype(np.uint8))
                
                pil_image.save(temp_img.name, 'JPEG', quality=85)  # Reduce quality to save memory
                temp_img_path = temp_img.name
            
            # Run AI prediction
            ai_result = self.predict_fake_real(temp_img_path)
            
            # Clean up temp file
            try:
                os.unlink(temp_img_path)
            except:
                pass  # Ignore cleanup errors
            
            return ai_result
            
        except Exception as e:
            return {
                "is_real": False,
                "confidence": 0.0,
                "error": f"AI analysis failed: {str(e)}"
            }
    
    def detect_phone_screen(self, image_array: np.ndarray):
        """Detect phone screen characteristics"""
        try:
            height, width = image_array.shape[:2]
            
            # Convert to grayscale for analysis
            import cv2
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            
            # Check for screen characteristics
            screen_indicators = 0
            confidence = 0.0
            
            # 1. Check for rectangular screen edges
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Check if contour is rectangular (like phone screen)
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) == 4:  # Rectangle found
                    area = cv2.contourArea(contour)
                    if area > (height * width * 0.3):  # Large rectangular area
                        screen_indicators += 1
                        confidence += 0.3
            
            # 2. Check for high contrast (typical of screens)
            std_dev = np.std(gray)
            if std_dev > 60:  # High contrast
                screen_indicators += 1
                confidence += 0.2
            
            # 3. Check for uniform brightness patterns (screen bezels)
            # Check edges for uniform dark areas
            top_edge = gray[:int(height*0.1), :]
            bottom_edge = gray[int(height*0.9):, :]
            left_edge = gray[:, :int(width*0.1)]
            right_edge = gray[:, int(width*0.9):]
            
            edge_uniformity = 0
            for edge in [top_edge, bottom_edge, left_edge, right_edge]:
                if edge.size > 0:
                    edge_std = np.std(edge)
                    if edge_std < 20:  # Very uniform (like phone bezel)
                        edge_uniformity += 1
            
            if edge_uniformity >= 2:  # At least 2 edges are uniform
                screen_indicators += 1
                confidence += 0.25
            
            # 4. Check for digital artifacts (pixelation patterns)
            # Use Laplacian to detect digital patterns
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < 100:  # Low variation indicates digital source
                screen_indicators += 1
                confidence += 0.15
            
            # 5. Check aspect ratio (common phone screen ratios)
            aspect_ratio = width / height if width > height else height / width
            common_ratios = [16/9, 18/9, 19.5/9, 20/9, 4/3]  # Common phone ratios
            
            for ratio in common_ratios:
                if abs(aspect_ratio - ratio) < 0.1:
                    screen_indicators += 1
                    confidence += 0.1
                    break
            
            # Final determination
            is_screen = screen_indicators >= 2 and confidence >= 0.5
            
            return {
                "is_screen": is_screen,
                "confidence": min(confidence, 0.95),  # Cap at 95%
                "indicators": screen_indicators,
                "details": {
                    "aspect_ratio": aspect_ratio,
                    "contrast_std": float(std_dev),
                    "edge_uniformity": edge_uniformity,
                    "laplacian_var": float(laplacian_var)
                }
            }
            
        except Exception as e:
            # If screen detection fails, return low confidence
            return {
                "is_screen": False,
                "confidence": 0.0,
                "error": f"Screen detection error: {str(e)}"
            }

# Initialize AI detector and add to cache
ai_fake_detector = AIFakeDetector('best.pt')
loaded_models['best.pt'] = ai_fake_detector

@app.get("/")
def read_root():
    """Serve the main HTML page"""
    return FileResponse('index.html')

@app.get("/live")
def live_verify_page():
    """Serve the live verification page"""
    return FileResponse('live_verify.html')

@app.post("/live-verify")
def live_camera_verify(
    punch_id: str = Form(...),
    name: str = Form(...),
    frame_data: str = Form(...)  # Base64 encoded camera frame
):
    """Live camera verification like phone face unlock"""
    start_time = time.time()
    
    try:
        # Decode base64 image data
        import base64
        
        # Remove data URL prefix if present
        if ',' in frame_data:
            frame_data = frame_data.split(',')[1]
        
        # Decode base64 to bytes
        image_bytes = base64.b64decode(frame_data)
        
        # Get reference encoding (fresh from HRM API)
        ref_encoding, error, ref_img_url = get_reference_encoding(punch_id, return_url=True)
        if ref_encoding is None:
            return JSONResponse({
                "status": False, 
                "reason": error, 
                "reference_image": ref_img_url,
                "live_mode": True
            })
        
        # Verify with AI and face recognition
        match, match_reason, ai_data = verify_face_with_ai(ref_encoding, image_bytes)
        
        processing_time = round(time.time() - start_time, 2)
        
        return JSONResponse({
            "status": bool(match),
            "message": match_reason if match else f"❌ {match_reason}",
            "reference_image": ref_img_url,
            "processing_time": processing_time,
            "live_mode": True,
            "ai_analysis": {
                "is_real": bool(ai_data.get("is_real", False)) if isinstance(ai_data, dict) else False,
                "confidence": float(ai_data.get("confidence", 0)) if isinstance(ai_data, dict) else 0.0,
                "detected_class": str(ai_data.get("detected_class", "unknown")) if isinstance(ai_data, dict) else "error"
            }
        })
        
    except Exception as e:
        return JSONResponse({
            "status": False,
            "message": f"Live verification error: {str(e)}",
            "processing_time": round(time.time() - start_time, 2),
            "live_mode": True
        })

@app.get("/models")
def get_available_models():
    """Get list of available AI models"""
    models = []
    for model_file in ['best.pt', 'best01.pt']:
        if os.path.exists(model_file):
            stat = os.stat(model_file)
            models.append({
                "name": model_file,
                "size": stat.st_size,
                "modified": time.ctime(stat.st_mtime),
                "is_current": model_file == ai_fake_detector.model_path
            })
    return JSONResponse({"models": models, "current": ai_fake_detector.model_path})

@app.post("/switch-model")
def switch_model(model_name: str = Form(...)):
    """Switch to different AI model"""
    global current_model_path
    
    if not os.path.exists(model_name):
        return JSONResponse({"success": False, "message": f"Model {model_name} not found"})
    
    if model_name not in ['best.pt', 'best01.pt']:
        return JSONResponse({"success": False, "message": "Only best.pt and best01.pt are allowed"})
    
    try:
        # Use cached model or load new one
        if model_name in loaded_models:
            # Use cached model
            global ai_fake_detector
            ai_fake_detector = loaded_models[model_name]
            print(f"Switched to cached model: {model_name}")
        else:
            # Load and cache new model
            ai_fake_detector.load_model(model_name)
            loaded_models[model_name] = ai_fake_detector
        
        current_model_path = model_name
        
        return JSONResponse({
            "success": True, 
            "message": f"Successfully switched to {model_name}",
            "current_model": model_name
        })
    except Exception as e:
        return JSONResponse({
            "success": False, 
            "message": f"Error switching to {model_name}: {str(e)}"
        })

@app.post("/clear-model-cache")
def clear_model_cache():
    """Clear cached models to free memory"""
    global loaded_models, ai_fake_detector
    
    # Keep only current model
    current_model = ai_fake_detector.model_path
    new_cache = {current_model: ai_fake_detector}
    
    # Clear old cache
    loaded_models = new_cache
    
    # Force garbage collection
    gc.collect()
    
    return JSONResponse({
        "success": True,
        "message": f"Model cache cleared. Only {current_model} remains loaded.",
        "cached_models": list(loaded_models.keys())
    })

@app.post("/dual-model-test")
def dual_model_test(picture: UploadFile = File(...)):
    """Test image with both AI models simultaneously"""
    start_time = time.time()
    
    try:
        # Read uploaded image
        test_image_bytes = picture.file.read()
        
        # Convert to RGB format
        test_pil_img = Image.open(io.BytesIO(test_image_bytes))
        if test_pil_img.mode != 'RGB':
            test_pil_img = test_pil_img.convert('RGB')
        test_img = np.array(test_pil_img)
        
        results = {}
        
        # Test with both models using cached instances
        for model_file in ['best.pt', 'best01.pt']:
            if os.path.exists(model_file):
                try:
                    # Use cached model or current detector
                    if model_file == ai_fake_detector.model_path:
                        # Use current loaded detector
                        detector = ai_fake_detector
                    else:
                        # Check if model is cached
                        if model_file not in loaded_models:
                            print(f"Loading and caching {model_file}...")
                            loaded_models[model_file] = AIFakeDetector(model_file)
                        detector = loaded_models[model_file]
                    
                    # Run AI fake detection
                    ai_result = detector.analyze_image(test_img)
                    
                    # Get file info
                    stat = os.stat(model_file)
                    
                    results[model_file] = {
                        "is_real": bool(ai_result["is_real"]),
                        "confidence": float(ai_result.get("confidence", 0)),
                        "detected_class": str(ai_result.get("detected_class", "unknown")),
                        "model_size": f"{stat.st_size / (1024*1024):.1f}MB",
                        "model_date": time.ctime(stat.st_mtime),
                        "error": ai_result.get("error", None)
                    }
                    
                except Exception as e:
                    results[model_file] = {
                        "is_real": False,
                        "confidence": 0.0,
                        "detected_class": "error",
                        "error": str(e)
                    }
            else:
                results[model_file] = {
                    "is_real": False,
                    "confidence": 0.0,
                    "detected_class": "not_found",
                    "error": "Model file not found"
                }
        
        processing_time = round(time.time() - start_time, 2)
        
        return JSONResponse({
            "success": True,
            "message": "Dual model analysis completed",
            "results": results,
            "processing_time": processing_time,
            "current_model": ai_fake_detector.model_path
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"Error during dual model test: {str(e)}",
            "processing_time": round(time.time() - start_time, 2)
        })

def get_employee_data(employee_id: str):
    """Fresh employee data - no cache"""
    response = requests.post(EMPLOYEE_API_URL, json={"employeeId": employee_id})
    if response.status_code != 200:
        return None
    data = response.json()
    employees = data.get("employees", [])
    return employees[0] if employees else None

def get_fresh_face_encoding(img_url: str):
    """Fresh face encoding - no cache"""
    try:
        img_response = requests.get(img_url, timeout=5)
        if img_response.status_code != 200:
            return None
        
        # Convert image to RGB format
        pil_img = Image.open(io.BytesIO(img_response.content))
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        img = np.array(pil_img)
        
        encodings = face_recognition.face_encodings(img)
        return encodings[0] if encodings else None
    except Exception:
        return None

def get_reference_encoding(employee_id: str, return_url=False):
    """Get employee reference encoding - fresh data"""
    # Get employee data (fresh)
    employee = get_employee_data(employee_id)
    if not employee:
        return None, "Punch ID is wrong or employee not found.", None if return_url else None
    
    user_img_path = employee.get("userImg")
    if not user_img_path:
        return None, "Employee mil gaya, but userImg not found.", None if return_url else None
    
    img_url = IMG_BASE_URL + user_img_path
    
    # Get face encoding (fresh)
    encoding = get_fresh_face_encoding(img_url)
    if encoding is None:
        return None, "Employee mil gaya, but no face detected in reference image.", img_url if return_url else None
    
    if return_url:
        return encoding, None, img_url
    return encoding, None

def verify_face_with_ai(reference_encoding, test_image_bytes, tolerance=0.6):
    """Simple face verification with AI fake detection only"""
    try:
        # Convert test image to RGB format
        test_pil_img = Image.open(io.BytesIO(test_image_bytes))
        if test_pil_img.mode != 'RGB':
            test_pil_img = test_pil_img.convert('RGB')
        test_img = np.array(test_pil_img)
        
        # First check with AI if image is fake
        ai_result = ai_fake_detector.analyze_image(test_img)
        
        if not ai_result["is_real"]:
            return False, f"❌ AI Detected FAKE Image (Confidence: {ai_result['confidence']:.3f})", ai_result
        
        # If AI says it's real, proceed with face recognition
        test_encodings = face_recognition.face_encodings(test_img)
        if not test_encodings:
            return False, "No face found in test image", ai_result
        
        test_encoding = test_encodings[0]
        match = face_recognition.compare_faces([reference_encoding], test_encoding, tolerance=tolerance)[0]
        
        # Calculate face distance for confidence
        face_distance = face_recognition.face_distance([reference_encoding], test_encoding)[0]
        face_confidence = (1 - face_distance) * 100
        
        if match:
            return True, f"✅ Face verified successfully! AI: Real Image (Conf: {ai_result['confidence']:.3f}), Face Match: {face_confidence:.1f}%", ai_result
        else:
            return False, f"Face recognized as real but doesn't match employee. AI: Real (Conf: {ai_result['confidence']:.3f}), Face Match: {face_confidence:.1f}%", ai_result
            
    except Exception as e:
        return False, f"Error during verification: {str(e)}", {"error": str(e)}

@app.post("/verify")
def verify_person(
    punch_id: str = Form(...),
    name: str = Form(...),
    picture: UploadFile = File(...)
):
    start_time = time.time()
    
    # Read uploaded image
    test_image_bytes = picture.file.read()
    
    # Get reference encoding (fresh from HRM API)
    ref_encoding, error, ref_img_url = get_reference_encoding(punch_id, return_url=True)
    if ref_encoding is None:
        return JSONResponse({"status": False, "reason": error, "reference_image": ref_img_url})
    
    # Simple face verification with AI fake detection only
    match, match_reason, ai_data = verify_face_with_ai(ref_encoding, test_image_bytes)
    
    processing_time = round(time.time() - start_time, 2)
    
    if not match:
        return JSONResponse({
            "status": False,
            "reason": match_reason,
            "reference_image": ref_img_url,
            "processing_time": processing_time,
            "ai_analysis": {
                "is_real": bool(ai_data.get("is_real", False)) if isinstance(ai_data, dict) else False,
                "confidence": float(ai_data.get("confidence", 0)) if isinstance(ai_data, dict) else 0.0,
                "detected_class": str(ai_data.get("detected_class", "unknown")) if isinstance(ai_data, dict) else "error"
            }
        })
    
    return JSONResponse({
        "status": True,
        "message": match_reason,
        "processing_time": processing_time,
        "ai_analysis": {
            "is_real": bool(ai_data.get("is_real", False)),
            "confidence": float(ai_data.get("confidence", 0)),
            "detected_class": str(ai_data.get("detected_class", "unknown"))
        }
    })

@app.post("/test-ai-detection")
def test_ai_detection_only(picture: UploadFile = File(...)):
    """Test AI fake detection only"""
    start_time = time.time()
    
    try:
        # Read uploaded image
        test_image_bytes = picture.file.read()
        
        # Convert to RGB format
        test_pil_img = Image.open(io.BytesIO(test_image_bytes))
        if test_pil_img.mode != 'RGB':
            test_pil_img = test_pil_img.convert('RGB')
        test_img = np.array(test_pil_img)
        
        # Run AI fake detection
        ai_result = ai_fake_detector.analyze_image(test_img)
        
        processing_time = round(time.time() - start_time, 2)
        
        return JSONResponse({
            "status": bool(ai_result["is_real"]),
            "message": f"AI Analysis: {'✅ Real Image' if ai_result['is_real'] else '❌ FAKE Image'}",
            "confidence": float(ai_result.get("confidence", 0)),
            "detected_class": str(ai_result.get("detected_class", "unknown")),
            "model_loaded": bool(ai_fake_detector.model_loaded),
            "processing_time": processing_time
        })
        
    except Exception as e:
        return JSONResponse({
            "status": False,
            "message": f"Error during AI detection: {str(e)}",
            "processing_time": round(time.time() - start_time, 2)
        })

if __name__ == "__main__":
    import uvicorn
    import webbrowser
    import threading
    import time as time_module
    
    def open_browser():
        """Open live verification page in browser after server starts"""
        time_module.sleep(3)  # Wait for server to start
        try:
            # Try to open live verification page
            webbrowser.open('http://localhost:8000/live')
            print("🌐 Live Verification page opened in browser: http://localhost:8000/live")
        except Exception as e:
            print(f"❌ Could not open browser: {e}")
            print("📖 Manual access: http://localhost:8000/live")
    
    # Start browser opening in background thread
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    print("🚀 Starting Face Verification Server...")
    print("📍 Main Page: http://localhost:8000/")
    print("🎯 Live Verification: http://localhost:8000/live")
    print("⚡ Server will auto-open Live Verification page...")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)