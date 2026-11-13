@echo off
cd /d "%~dp0"
python start_app.py
if errorlevel 1 (
  echo.
  echo Startup failed. Install dependencies with: python -m pip install -r requirements.txt
  echo Check model files with: python setup_models.py --check
  pause
)
