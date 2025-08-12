#!/usr/bin/env python3
"""
Development startup script for Test Case Generator API
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Main startup function"""
    print("🚀 Starting Test Case Generator API...")
    
    # Check if .env exists
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️  .env file not found. Creating from .env.example...")
        if Path(".env.example").exists():
            import shutil
            shutil.copy(".env.example", ".env")
            print("✅ .env file created. Please configure your API keys.")
        else:
            print("❌ .env.example not found!")
            sys.exit(1)
    
    # Check if virtual environment is activated
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Virtual environment not detected. Consider using venv or conda.")
    
    # Install dependencies if needed
    try:
        import fastapi
        import uvicorn
    except ImportError:
        print("📦 Installing dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    print("🌟 Starting FastAPI server...")
    print("📚 API Documentation: http://localhost:4200/api/v1/docs")
    print("🏥 Health Check: http://localhost:4200/api/v1/health")
    print("🔄 Use Ctrl+C to stop the server")
    
    # Start the server
    try:
        import uvicorn
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=4200,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
