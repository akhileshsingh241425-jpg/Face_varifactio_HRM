import sys
import os

# Add your project directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

from main import app

# This is required for Hostinger
application = app

if __name__ == "__main__":
    app.run()