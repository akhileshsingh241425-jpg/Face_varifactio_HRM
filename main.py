from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import face_recognition
import numpy as np
from PIL import Image
import io
import requests
import time


app = FastAPI()

# Enable CORS for all origins (production ready)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

EMPLOYEE_API_URL = "https://hrm.umanerp.com/api/users/getEmployee"
IMG_BASE_URL = "https://hrm.umanerp.com/"

# Simple cache for employee data
employee_cache = {}
encoding_cache = {}

def get_employee_data(employee_id: str):
    """Simple cache for employee data"""
    if employee_id in employee_cache:
        return employee_cache[employee_id]
    
    response = requests.post(EMPLOYEE_API_URL, json={"employeeId": employee_id})
    if response.status_code != 200:
        return None
    data = response.json()
    employees = data.get("employees", [])
    result = employees[0] if employees else None
    employee_cache[employee_id] = result
    return result

def get_cached_face_encoding(img_url: str):
    """Simple cache for face encodings"""
    if img_url in encoding_cache:
        return encoding_cache[img_url]
    
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
        result = encodings[0] if encodings else None
        encoding_cache[img_url] = result
        return result
    except Exception:
        return None

def get_reference_encoding(employee_id: str, return_url=False):
    """Get employee reference encoding"""
    # Get employee data (cached)
    employee = get_employee_data(employee_id)
    if not employee:
        return None, "Punch ID is wrong or employee not found.", None if return_url else None
    
    user_img_path = employee.get("userImg")
    if not user_img_path:
        return None, "Employee mil gaya, but userImg not found.", None if return_url else None
    
    img_url = IMG_BASE_URL + user_img_path
    
    # Get face encoding (cached)
    encoding = get_cached_face_encoding(img_url)
    if encoding is None:
        return None, "Employee mil gaya, but no face detected in reference image.", img_url if return_url else None
    
    if return_url:
        return encoding, None, img_url
    return encoding, None

def verify_face_fast(reference_encoding, test_image_bytes, tolerance=0.6):
    """Fast face verification"""
    # Convert test image to RGB format
    test_pil_img = Image.open(io.BytesIO(test_image_bytes))
    if test_pil_img.mode != 'RGB':
        test_pil_img = test_pil_img.convert('RGB')
    test_img = np.array(test_pil_img)
    
    test_encodings = face_recognition.face_encodings(test_img)
    if not test_encodings:
        return False, "No face found in test image"
    
    test_encoding = test_encodings[0]
    match = face_recognition.compare_faces([reference_encoding], test_encoding, tolerance=tolerance)[0]
    return match, "Face matched" if match else "Face did not match"

@app.post("/verify")
def verify_person(
    punch_id: str = Form(...),
    name: str = Form(...),
    picture: UploadFile = File(...)
):
    start_time = time.time()
    
    # Read uploaded image
    test_image_bytes = picture.file.read()
    
    # Get reference encoding (cached)
    ref_encoding, error, ref_img_url = get_reference_encoding(punch_id, return_url=True)
    if ref_encoding is None:
        return JSONResponse({"status": False, "reason": error, "reference_image": ref_img_url})
    
    # Fast face verification using pre-computed encoding
    match, match_reason = verify_face_fast(ref_encoding, test_image_bytes)
    
    processing_time = round(time.time() - start_time, 2)
    
    if not match:
        return JSONResponse({
            "status": False,
            "reason": "Employee mil gaya but face verify nahi hua: " + match_reason,
            "reference_image": ref_img_url,
            "processing_time": processing_time
        })
    
    return JSONResponse({
        "status": True,
        "processing_time": processing_time
    })

# To run: uvicorn main:app --reload
