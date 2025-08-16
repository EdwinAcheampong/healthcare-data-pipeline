#!/usr/bin/env python3
"""
Automated Data Download Script for Healthcare Data Pipeline

This script downloads the Synthea COVID-19 dataset from the hosted location
and sets it up in the correct project structure.
"""

import os
import sys
import urllib.request
import zipfile
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, DownloadColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

console = Console()

# Dataset configuration
DATASET_CONFIG = {
    "zenodo": {
        "url": "https://zenodo.org/record/XXXXXX/files/synthea-covid19-dataset.zip",
        "size_mb": 651,
        "description": "Zenodo Academic Repository (Recommended)"
    },
    "gdrive": {
        "url": "https://drive.google.com/uc?export=download&id=YOUR_GOOGLE_DRIVE_FILE_ID",
        "size_mb": 651,
        "description": "Google Drive Mirror"
    }
}


def download_with_progress(url: str, destination: Path, description: str):
    """Download file with progress bar."""
    
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = (downloaded / total_size) * 100
            progress.update(task, completed=downloaded, total=total_size)
    
    with Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        TaskProgressColumn(),
        DownloadColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task(description, total=None)
        
        try:
            urllib.request.urlretrieve(url, destination, progress_hook)
            console.print(f"✅ Downloaded: {destination}")
            return True
        except Exception as e:
            console.print(f"❌ Download failed: {e}")
            return False


def extract_dataset(zip_path: Path, extract_to: Path):
    """Extract downloaded dataset."""
    console.print(f"📦 Extracting dataset to {extract_to}")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        
        console.print("✅ Dataset extracted successfully")
        
        # Clean up zip file
        zip_path.unlink()
        console.print("🗑️ Cleaned up download archive")
        
        return True
    except Exception as e:
        console.print(f"❌ Extraction failed: {e}")
        return False


def verify_data_structure():
    """Verify the downloaded data has correct structure."""
    project_root = Path(__file__).parent.parent
    
    required_files = [
        "data/synthea/patients.csv",
        "data/synthea/conditions.csv", 
        "data/synthea/encounters.csv",
        "data/synthea/observations.csv",
        "data/synthea/medications.csv"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not (project_root / file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        console.print("⚠️ Missing required files:")
        for file in missing_files:
            console.print(f"  - {file}")
        return False
    
    console.print("✅ Data structure verified")
    return True


def main():
    """Main download function."""
    console.print("[bold blue]🏥 Healthcare Dataset Downloader[/bold blue]")
    console.print()
    
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    
    # Show available download options
    console.print("📊 Available download sources:")
    for source, config in DATASET_CONFIG.items():
        console.print(f"  🔗 {source}: {config['description']} ({config['size_mb']}MB)")
    
    console.print()
    
    # Default to Zenodo (academic source)
    source = "zenodo"
    config = DATASET_CONFIG[source]
    
    console.print(f"📥 Downloading from: {config['description']}")
    console.print(f"📊 Dataset size: {config['size_mb']}MB")
    console.print()
    
    # Download dataset
    download_path = data_dir / "synthea-covid19-dataset.zip"
    
    if download_with_progress(config["url"], download_path, "Downloading dataset"):
        # Extract dataset
        if extract_dataset(download_path, data_dir):
            # Verify structure
            if verify_data_structure():
                console.print()
                console.print("🎉 [bold green]Dataset setup complete![/bold green]")
                console.print()
                console.print("Next steps:")
                console.print("1. Run data validation: [cyan]python scripts/validate_data.py[/cyan]")
                console.print("2. Start analysis: [cyan]jupyter lab notebooks/01_data_exploration.ipynb[/cyan]")
                return True
    
    console.print("❌ [bold red]Dataset setup failed[/bold red]")
    console.print()
    console.print("Manual setup options:")
    console.print("1. Download manually from: https://doi.org/10.5281/zenodo.XXXXXX")
    console.print("2. Follow setup guide: [cyan]data/DATA_SETUP.md[/cyan]")
    return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
