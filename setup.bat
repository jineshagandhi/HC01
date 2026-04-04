@echo off
echo ============================================
echo  ICU Diagnostic Risk Assistant - Setup
echo  IGNISIA AI Hackathon 2026 - HC01
echo ============================================
echo.

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10+ from python.org
    pause
    exit /b 1
)

echo [1/4] Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

echo [2/4] Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

echo [3/4] Downloading spaCy model...
python -m spacy download en_core_web_sm

echo [4/4] Setting up environment...
if not exist .env (
    copy .env.example .env
    echo.
    echo ============================================
    echo  IMPORTANT: Add your Gemini API key!
    echo  Edit the .env file and add your key:
    echo  GEMINI_API_KEY=your-key-here
    echo ============================================
)

echo.
echo ============================================
echo  Setup complete!
echo  1. Edit .env file with your Gemini API key
echo  2. Run: run.bat
echo ============================================
pause
