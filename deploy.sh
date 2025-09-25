#!/bin/bash
# Hostinger Deployment Script

echo "🚀 Starting Hostinger Deployment Setup..."

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Check if AI model exists
if [ -f "best.pt" ]; then
    echo "✅ AI Model (best.pt) found - Ready for deployment!"
else
    echo "❌ AI Model (best.pt) not found!"
fi

# Check main application
if [ -f "main.py" ]; then
    echo "✅ Main application (main.py) ready!"
else
    echo "❌ Main application (main.py) not found!"
fi

echo "🎉 Deployment setup complete!"
echo "Entry Point: main:app"
echo "Python Version Required: 3.9+"
echo "Ready to deploy on Hostinger!"