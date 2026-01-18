@echo off
echo ========================================
echo Materials Discovery ML Workshop - Windows
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

REM Check if virtual environment exists, create if not
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment.
    pause
    exit /b 1
)

REM Install/update requirements
echo Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Installation complete!
echo ========================================
echo.
echo Choose your deployment option:
echo 1. Jupyter Notebook Workshop (Educational)
echo 2. Streamlit Web App (Production)
echo.

set /p choice="Enter choice (1 or 2): "

if "%choice%"=="1" (
    echo.
    echo Launching Jupyter Notebook Workshop...
    echo Open the materials_discovery_workshop.ipynb file when Jupyter opens.
    jupyter notebook materials_discovery_workshop.ipynb
) else if "%choice%"=="2" (
    echo.
    echo Launching Streamlit Web App...
    echo Open http://localhost:8501 in your browser.
    streamlit run app.py
) else (
    echo Invalid choice. Exiting.
    pause
    exit /b 1
)

pause
