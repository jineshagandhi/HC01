@echo off
echo ============================================
echo  ICU Diagnostic Risk Assistant
echo  Starting Streamlit Dashboard...
echo ============================================
echo.

if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

streamlit run app.py --server.port 8501 --server.headless true
pause
