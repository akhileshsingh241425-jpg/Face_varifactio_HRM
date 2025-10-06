#!/usr/bin/env python3
"""
Complete System Check for Face Verification System
"""
import sys

def check_library(lib_name, import_statement):
    try:
        exec(import_statement)
        print(f"✅ {lib_name}: Installed and working")
        return True
    except ImportError as e:
        print(f"❌ {lib_name}: NOT installed - {e}")
        return False
    except Exception as e:
        print(f"⚠️  {lib_name}: Installed but error - {e}")
        return False

def main():
    print("🔍 Checking Face Verification System Dependencies...")
    print("=" * 60)
    
    all_good = True
    
    # Core FastAPI
    all_good &= check_library("FastAPI", "import fastapi")
    all_good &= check_library("Uvicorn", "import uvicorn")
    all_good &= check_library("Python Multipart", "import multipart")
    
    # Image Processing
    all_good &= check_library("Pillow", "from PIL import Image")
    all_good &= check_library("NumPy", "import numpy as np")
    all_good &= check_library("OpenCV", "import cv2")
    
    # Face Recognition
    all_good &= check_library("Face Recognition", "import face_recognition")
    all_good &= check_library("dlib", "import dlib")
    
    # AI/ML Libraries
    all_good &= check_library("PyTorch", "import torch")
    all_good &= check_library("TorchVision", "import torchvision")
    all_good &= check_library("Ultralytics", "from ultralytics import YOLO")
    
    # Other utilities
    all_good &= check_library("Requests", "import requests")
    all_good &= check_library("IO", "import io")
    all_good &= check_library("TempFile", "import tempfile")
    all_good &= check_library("OS", "import os")
    all_good &= check_library("Time", "import time")
    
    print("=" * 60)
    
    # Test AI Model
    print("\n🤖 Testing AI Model (best.pt)...")
    try:
        from ultralytics import YOLO
        import os
        if os.path.exists('best.pt'):
            model = YOLO('best.pt')
            print("✅ AI Model (best.pt): Found and loaded successfully!")
        else:
            print("❌ AI Model (best.pt): File not found!")
            all_good = False
    except Exception as e:
        print(f"❌ AI Model Error: {e}")
        all_good = False
    
    # Test Main App Import
    print("\n📱 Testing Main Application...")
    try:
        import main
        print("✅ main.py: Imports successfully!")
    except Exception as e:
        print(f"❌ main.py Import Error: {e}")
        all_good = False
    
    print("=" * 60)
    
    if all_good:
        print("🎉 ALL SYSTEMS READY! Your Face Verification System can run!")
        print("🚀 Run command: uvicorn main:app --host 0.0.0.0 --port 8000")
    else:
        print("⚠️  Some dependencies missing. Install missing packages first.")
        print("📦 Run: pip install -r requirements.txt")
    
    print("=" * 60)

if __name__ == "__main__":
    main()