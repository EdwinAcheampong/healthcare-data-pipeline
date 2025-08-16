#!/usr/bin/env python3
"""
Quick Setup Verification Script

This script performs a quick check to ensure the project is properly organized
and ready for development.
"""

import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

def check_project_structure():
    """Verify the project structure is correct."""
    required_dirs = [
        'data/synthea',
        'data/ccda', 
        'data/raw',
        'data/processed',
        'src/config',
        'src/utils',
        'src/data_pipeline',
        'notebooks',
        'scripts',
        'logs',
        'models',
        'mlruns'
    ]
    
    required_files = [
        'requirements.txt',
        'pyproject.toml',
        'docker-compose.yml',
        'README.md',
        'env.example',
        'data/DATA_INVENTORY.md',
        'data/SYNC_SUMMARY.md',
        'notebooks/01_data_exploration.ipynb'
    ]
    
    results = {'dirs': {}, 'files': {}}
    
    # Check directories
    for dir_path in required_dirs:
        path = Path(dir_path)
        results['dirs'][dir_path] = path.exists() and path.is_dir()
    
    # Check files
    for file_path in required_files:
        path = Path(file_path)
        results['files'][file_path] = path.exists() and path.is_file()
    
    return results

def check_data_files():
    """Check if data files are present."""
    synthea_path = Path('data/synthea')
    ccda_path = Path('data/ccda')
    
    csv_files = list(synthea_path.glob('*.csv')) if synthea_path.exists() else []
    xml_files = list(ccda_path.glob('*.xml')) if ccda_path.exists() else []
    
    return {
        'csv_count': len(csv_files),
        'xml_count': len(xml_files),
        'expected_csv': 15,
        'expected_xml': 109
    }

def main():
    """Main verification function."""
    console.print(Panel.fit(
        "[bold blue]Project Setup Verification[/bold blue]",
        border_style="blue"
    ))
    
    # Check project structure
    structure_results = check_project_structure()
    
    # Create structure table
    struct_table = Table(title="Project Structure")
    struct_table.add_column("Component", style="cyan")
    struct_table.add_column("Type", style="yellow")
    struct_table.add_column("Status", style="green")
    
    # Add directories
    for dir_path, exists in structure_results['dirs'].items():
        status = "✅ Found" if exists else "❌ Missing"
        struct_table.add_row(dir_path, "Directory", status)
    
    # Add files
    for file_path, exists in structure_results['files'].items():
        status = "✅ Found" if exists else "❌ Missing"
        struct_table.add_row(file_path, "File", status)
    
    console.print(struct_table)
    
    # Check data files
    data_results = check_data_files()
    
    data_table = Table(title="Data Files")
    data_table.add_column("Data Type", style="cyan")
    data_table.add_column("Found", justify="right")
    data_table.add_column("Expected", justify="right")
    data_table.add_column("Status", style="green")
    
    csv_status = "✅ Complete" if data_results['csv_count'] >= data_results['expected_csv'] else "⚠️ Incomplete"
    xml_status = "✅ Complete" if data_results['xml_count'] >= data_results['expected_xml'] else "⚠️ Incomplete"
    
    data_table.add_row("CSV Files", str(data_results['csv_count']), str(data_results['expected_csv']), csv_status)
    data_table.add_row("XML Files", str(data_results['xml_count']), str(data_results['expected_xml']), xml_status)
    
    console.print(data_table)
    
    # Calculate overall status
    all_dirs_ok = all(structure_results['dirs'].values())
    all_files_ok = all(structure_results['files'].values())
    data_ok = (data_results['csv_count'] >= 10 and data_results['xml_count'] >= 100)
    
    overall_status = all_dirs_ok and all_files_ok and data_ok
    
    # Final summary
    if overall_status:
        console.print(Panel.fit(
            "[bold green]✅ PROJECT READY![/bold green]\n\n"
            "Your MSc Healthcare Project is properly organized and ready for development.\n\n"
            "🚀 You can now start:\n"
            "• Data exploration: jupyter lab notebooks/01_data_exploration.ipynb\n"
            "• Environment setup: scripts\\setup_environment.bat\n"
            "• Development: docker-compose up -d",
            border_style="green"
        ))
    else:
        console.print(Panel.fit(
            "[bold red]⚠️ SETUP INCOMPLETE[/bold red]\n\n"
            "Some components are missing. Please check the above tables and:\n"
            "• Run data synchronization again\n"
            "• Verify all files are in place\n"
            "• Check the setup scripts",
            border_style="red"
        ))
    
    return overall_status

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

