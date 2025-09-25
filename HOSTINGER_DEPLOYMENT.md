# 🚀 Hostinger Deployment Guide for Face Verification System

## Step 1: Login to Hostinger
1. Go to https://hostinger.com
2. Login to your account
3. Go to **hPanel** dashboard

## Step 2: Create Python Web App
1. In hPanel, click **Website** section
2. Select **Create Website**
3. Choose **Python Web Application**
4. Select Python version: **3.9 or higher**

## Step 3: Connect GitHub Repository
1. In the Python app settings, find **Git Deployment**
2. Connect your GitHub account
3. Select repository: `akhileshsingh241425-jpg/Face_varifactio_HRM`
4. Branch: `main`
5. Enable **Auto Deploy** (optional)

## Step 4: Configure Application
1. **Entry Point**: `main:app`
2. **Python Version**: 3.9+
3. **Requirements**: `requirements.txt` (will auto-install)

## Step 5: Environment Setup
1. In Hostinger Python app terminal, run:
```bash
pip install -r requirements.txt
```

## Step 6: Deploy
1. Click **Deploy** button
2. Wait for deployment to complete
3. Your app will be available at: `https://yourdomain.com`

## 🔧 Important Settings:
- **Entry Point**: `main:app` (FastAPI application)
- **Port**: Auto-assigned by Hostinger
- **Python Version**: 3.9+ required
- **Memory**: Ensure sufficient for AI model (best.pt is ~20MB)

## 📱 Features Deployed:
- ✅ Face Verification with Employee ID
- ✅ AI Fake Detection using best.pt model  
- ✅ Clean web interface
- ✅ HRM system integration

## 🛠️ Troubleshooting:
- If AI model fails to load, check memory limits
- Ensure all dependencies are installed
- Check Python version compatibility

Your Face Verification System is ready for production! 🎉