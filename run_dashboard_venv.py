#!/usr/bin/env python3
"""
Script to run the Streamlit dashboard using the virtual environment.
"""

import subprocess
import sys
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).resolve().parent

# Path to virtual environment python
venv_python = project_root / ".venv" / "Scripts" / "python.exe"

if not venv_python.exists():
    print("Virtual environment not found. Please create one with: python -m venv .venv")
    sys.exit(1)

# Run streamlit using the virtual environment python
cmd = [
    str(venv_python),
    "-m", "streamlit", 
    "run", 
    "src/api/dashboard.py",
    "--server.port=8501",
    "--server.address=localhost"
]

print(f"Running: {' '.join(cmd)}")
subprocess.run(cmd, cwd=project_root)