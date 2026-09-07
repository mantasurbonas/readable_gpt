@echo off
setlocal

set VENV_DIR=.venv

echo Checking for Python 3.14+...
py -3.14 --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python 3.14 is not installed or not found via the py launcher.
    echo Download it from https://www.python.org/downloads/
    exit /b 1
)

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo Creating virtual environment in %VENV_DIR%...
    py -3.14 -m venv %VENV_DIR%
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        exit /b 1
    )
) else (
    echo Virtual environment already exists, skipping creation.
)

echo Installing dependencies...
%VENV_DIR%\Scripts\pip.exe install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies.
    exit /b 1
)

echo.
echo Setup complete. Run the project with:
echo   run.bat "Your prompt here"
