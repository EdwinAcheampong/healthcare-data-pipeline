@echo off
REM Environment Setup Script for MSc Healthcare Project (Windows)

echo MSc Healthcare Project - Environment Setup
echo ==========================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo Python found: 
python --version

REM Check if Docker is available
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Docker is not installed or not running
    pause
    exit /b 1
)

echo Docker found:
docker --version

REM Create project directories
echo Creating project directories...
if not exist "data\raw" mkdir "data\raw"
if not exist "data\processed" mkdir "data\processed" 
if not exist "data\synthea" mkdir "data\synthea"
if not exist "logs" mkdir "logs"
if not exist "models" mkdir "models"
if not exist "mlruns" mkdir "mlruns"

REM Create .env file from template
if not exist ".env" (
    if exist "env.example" (
        copy "env.example" ".env"
        echo .env file created from template
    )
)

REM Install Python dependencies
echo Installing Python dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Error: Failed to install dependencies
    pause
    exit /b 1
)

REM Start Docker services
echo Starting Docker services...
docker-compose up -d
if %errorlevel% neq 0 (
    echo Error: Failed to start Docker services
    pause
    exit /b 1
)

echo.
echo Setup complete! 
echo.
echo You can now access:
echo - Jupyter Lab: http://localhost:8888 (token: msc-project-token)
echo - MLflow UI: http://localhost:5000
echo - FHIR Server: http://localhost:8082/fhir
echo - Database Admin: http://localhost:8080
echo.
echo Run 'jupyter lab' to start your analysis!

pause

