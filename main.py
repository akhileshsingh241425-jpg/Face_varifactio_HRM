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

class AIFakeDetector:
    def __init__(self, model_path='best.pt'):
        """Initialize AI model for fake vs real detection"""
        try:
            print(f"Loading AI model from {model_path}...")
            # Load with specific settings to avoid memory issues
            self.model = YOLO(model_path)
            self.model.overrides['verbose'] = False  # Reduce verbosity
            print("✅ AI Fake Detection Model loaded successfully!")
            self.model_loaded = True
        except Exception as e:
            print(f"❌ Error loading AI model: {str(e)}")
            self.model_loaded = False
            self.model = None
    
    def predict_fake_real(self, image_path_or_array, confidence_threshold=0.5):
        """Predict if image is fake or real using trained model"""
        if not self.model_loaded:
            return {
                "is_real": True,  # Default to real if model not loaded
                "confidence": 0.0,
                "error": "AI model not loaded"
            }
        
        try:
            # Run inference with memory optimization
            results = self.model(image_path_or_array, conf=confidence_threshold, verbose=False, imgsz=416)
            
            if not results or len(results) == 0:
                return {
                    "is_real": True,  # Default to real if no detection
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
                    
                    # Assuming: 0 = fake, 1 = real (adjust based on your training)
                    class_names = ['fake', 'real']
                    detected_class = class_names[predicted_class] if predicted_class < len(class_names) else 'unknown'
                    
                    is_real = detected_class == 'real'
                    
                    return {
                        "is_real": is_real,
                        "confidence": float(max_confidence),
                        "detected_class": detected_class
                    }
            
            # If no boxes detected, default to real
            return {
                "is_real": True,
                "confidence": 0.0,
                "detected_class": "no_detection"
            }
            
        except Exception as e:
            print(f"AI prediction error: {str(e)}")
            return {
                "is_real": True,  # Default to real on error
                "confidence": 0.0,
                "error": str(e)
            }
    
    def analyze_image(self, image_array: np.ndarray):
        """Analyze image using AI model"""
        try:
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
                "is_real": True,
                "confidence": 0.0,
                "error": f"AI analysis failed: {str(e)}"
            }

# Initialize AI detector
ai_fake_detector = AIFakeDetector('best.pt')

@app.get("/")
def read_root():
    """Serve the main HTML page"""
    return FileResponse('index.html')

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
            "ai_analysis": ai_data
        })
    
    return JSONResponse({
        "status": True,
        "message": match_reason,
        "processing_time": processing_time,
        "ai_analysis": {
            "is_real": ai_data.get("is_real", False),
            "confidence": ai_data.get("confidence", 0),
            "detected_class": ai_data.get("detected_class", "unknown")
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
            "status": ai_result["is_real"],
            "message": f"AI Analysis: {'✅ Real Image' if ai_result['is_real'] else '❌ FAKE Image'}",
            "confidence": ai_result.get("confidence", 0),
            "detected_class": ai_result.get("detected_class", "unknown"),
            "model_loaded": ai_fake_detector.model_loaded,
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
    uvicorn.run(app, host="0.0.0.0", port=8000)