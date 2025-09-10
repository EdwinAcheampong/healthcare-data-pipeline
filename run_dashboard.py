#!/usr/bin/env python3
"""
Script to run the Streamlit dashboard with proper Python path setup.
"""

import sys
import os
import subprocess
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).resolve().parent

# Check if we're running in virtual environment
venv_python = project_root / ".venv" / "Scripts" / "python.exe"

if venv_python.exists() and str(venv_python) not in sys.executable:
    # We're not in the virtual environment, restart with venv python
    print("Restarting with virtual environment Python...")
    cmd = [str(venv_python), __file__] + sys.argv[1:]
    subprocess.run(cmd, cwd=project_root)
    sys.exit()

# Add project root to Python path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Change to project root directory
os.chdir(project_root)

# Now run streamlit
if __name__ == "__main__":
    import streamlit.web.cli as stcli
    import sys
    
    # Set up streamlit arguments
    sys.argv = [
        "streamlit",
        "run",
        "src/api/dashboard.py",
        "--server.port=8501",
        "--server.address=localhost"
    ]
    
    # Run streamlit
    stcli.main()