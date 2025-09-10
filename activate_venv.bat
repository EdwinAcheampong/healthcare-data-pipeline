@echo off
echo Activating virtual environment...
call .venv\Scripts\activate.bat
echo Virtual environment activated. You can now run:
echo   python run_dashboard.py
echo   or
echo   streamlit run src/api/dashboard.py --server.port=8502
cmd /k