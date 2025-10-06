#!/bin/bash

# 🚀 Hostinger Deployment Script for Face Verification HRM
# Run this script to prepare files for Hostinger upload

echo "🚀 Preparing Face Verification HRM for Hostinger Deployment..."

# Create deployment directory
mkdir -p hostinger_deployment
cd hostinger_deployment

echo "📁 Copying essential files..."

# Copy main application files
cp ../main.py .
cp ../passenger_wsgi.py .
cp ../index.html .
cp ../requirements.txt .

# Copy AI model files (if they exist)
if [ -f "../best.pt" ]; then
    cp ../best.pt .
    echo "✅ AI Model (best.pt) copied - 22MB"
else
    echo "⚠️ Warning: best.pt not found!"
fi

if [ -f "../best01.pt" ]; then
    cp ../best01.pt .
    echo "✅ Backup AI Model (best01.pt) copied - 22MB"
else
    echo "⚠️ Warning: best01.pt not found!"
fi

# Create .htaccess file for Hostinger
cat > .htaccess << 'EOF'
# Hostinger Python App Configuration
PassengerAppRoot /home/username/public_html
PassengerBaseURI /
PassengerAppType wsgi
PassengerStartupFile passenger_wsgi.py

# Security headers
Header always set X-Content-Type-Options nosniff
Header always set X-Frame-Options DENY
Header always set X-XSS-Protection "1; mode=block"

# CORS headers for API
Header always set Access-Control-Allow-Origin "*"
Header always set Access-Control-Allow-Methods "GET, POST, OPTIONS"
Header always set Access-Control-Allow-Headers "Content-Type"

# Force HTTPS (for camera access)
RewriteEngine On
RewriteCond %{HTTPS} off
RewriteRule ^(.*)$ https://%{HTTP_HOST}%{REQUEST_URI} [L,R=301]
EOF

echo "✅ .htaccess file created for Hostinger configuration"

# Create deployment info file
cat > DEPLOYMENT_INFO.txt << 'EOF'
🚀 HOSTINGER DEPLOYMENT FILES READY

📁 Files to Upload to public_html:
- main.py (FastAPI application)
- passenger_wsgi.py (WSGI entry point)
- index.html (Frontend interface)
- requirements.txt (Python dependencies)  
- best.pt (AI model - 22MB)
- best01.pt (Backup AI model - 22MB)
- .htaccess (Apache configuration)

📋 Next Steps:
1. Upload all files to Hostinger public_html directory
2. Install Python dependencies: pip install -r requirements.txt --user
3. Configure Python App in Hostinger Control Panel
4. Set startup file: passenger_wsgi.py
5. Set entry point: application
6. Enable HTTPS for camera access

🎯 API Endpoints After Deployment:
- Main Page: https://yourdomain.com/
- Live Verification: POST https://yourdomain.com/live-verify
- Regular Verification: POST https://yourdomain.com/verify
- Models List: GET https://yourdomain.com/models

🔧 Features Included:
✅ Live Camera Verification (Phone face unlock style)
✅ AI Fake Detection (Phone screens, fake photos)
✅ Face Recognition (HRM database integration)  
✅ Real-time Error Handling with visual feedback
✅ Enhanced UI with animations and color coding
✅ Dual Tab Interface (Regular + Live verification)
✅ Fresh Data Retrieval (No caching)
✅ Mobile-friendly responsive design

⚠️ Requirements:
- Python 3.8+ on Hostinger
- HTTPS enabled (for camera access)
- Sufficient storage for AI models (44MB total)
- Memory for AI processing (recommended: 1GB+ RAM)

📞 Support: Check HOSTINGER_DEPLOYMENT_GUIDE.md for detailed instructions
EOF

echo "📄 DEPLOYMENT_INFO.txt created with instructions"

# Calculate total size
TOTAL_SIZE=$(du -sh . | cut -f1)
echo "📊 Total deployment size: $TOTAL_SIZE"

echo ""
echo "🎉 DEPLOYMENT PREPARATION COMPLETE!"
echo ""
echo "📁 All files are ready in: $(pwd)"
echo "📋 Next steps:"
echo "   1. Compress this folder to ZIP"
echo "   2. Upload to Hostinger File Manager"
echo "   3. Extract in public_html directory"
echo "   4. Follow DEPLOYMENT_INFO.txt instructions"
echo ""
echo "🌐 Your Face Verification HRM system will be live at: https://yourdomain.com"

# Optional: Create ZIP file
if command -v zip &> /dev/null; then
    echo "📦 Creating deployment ZIP file..."
    zip -r ../Face_Verification_HRM_Hostinger.zip . -x "*.DS_Store" "*.git*"
    echo "✅ ZIP file created: Face_Verification_HRM_Hostinger.zip"
    echo "📤 Ready to upload to Hostinger!"
fi