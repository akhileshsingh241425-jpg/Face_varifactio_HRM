# Face Verification System - Hostinger Ready

A simplified AI-powered face verification system using custom trained model.

## 🚀 Quick Deploy to Hostinger

1. **Connect GitHub Repository**:
   - Repository: `akhileshsingh241425-jpg/Face_varifactio_HRM`
   - Entry Point: `main:app`
   - Python Version: 3.9+

2. **Auto-Install Dependencies**: `requirements.txt`

3. **AI Model**: `best.pt` (20MB) - Custom trained for fake/real detection

## 📋 Features
- ✅ Employee face verification against HRM database
- ✅ AI-powered fake image detection  
- ✅ Clean web interface with photo capture
- ✅ Fast processing with caching
- ✅ Production-ready with error handling

## 🔧 API Endpoints
- `GET /` - Web interface
- `POST /verify` - Face verification with employee ID
- `POST /test-ai-detection` - Test AI model only

## 💻 Local Development
```bash
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 🏢 HRM Integration
Connects to: `https://hrm.umanerp.com/api/users/getEmployee`

Ready for Hostinger deployment! 🎉
