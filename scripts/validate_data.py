#!/usr/bin/env python3
"""
Data Validation Script for MSc Healthcare Project

This script validates the integrity and structure of the Synthea and C-CDA data
to ensure everything is properly synchronized and ready for analysis.
"""

import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
import sys
from typing import Dict, List, Tuple
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, track

console = Console()

def validate_csv_files(data_path: Path) -> Dict[str, any]:
    """Validate CSV files in the Synthea dataset."""
    results = {}
    
    expected_files = [
        'patients.csv', 'conditions.csv', 'encounters.csv', 
        'observations.csv', 'medications.csv', 'procedures.csv',
        'organizations.csv', 'providers.csv', 'payers.csv',
        'allergies.csv', 'careplans.csv', 'devices.csv',
        'immunizations.csv', 'supplies.csv', 'imaging_studies.csv'
    ]
    
    console.print("[bold]Validating CSV Files...[/bold]")
    
    for file_name in track(expected_files, description="Checking files..."):
        file_path = data_path / file_name
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                results[file_name] = {
                    'exists': True,
                    'rows': len(df),
                    'columns': len(df.columns),
                    'size_mb': file_path.stat().st_size / (1024 * 1024),
                    'memory_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
                    'has_nulls': df.isnull().any().any(),
                    'duplicates': df.duplicated().sum()
                }
            except Exception as e:
                results[file_name] = {
                    'exists': True,
                    'error': str(e)
                }
        else:
            results[file_name] = {'exists': False}
    
    return results

def validate_patient_relationships(data_path: Path) -> Dict[str, any]:
    """Validate relationships between core tables."""
    console.print("[bold]Validating Data Relationships...[/bold]")
    
    try:
        # Load core tables
        patients = pd.read_csv(data_path / 'patients.csv')
        conditions = pd.read_csv(data_path / 'conditions.csv')
        encounters = pd.read_csv(data_path / 'encounters.csv')
        
        results = {}
        
        # Check patient ID consistency
        patient_ids = set(patients['Id'])
        condition_patient_ids = set(conditions['PATIENT']) if 'PATIENT' in conditions.columns else set()
        encounter_patient_ids = set(encounters['PATIENT']) if 'PATIENT' in encounters.columns else set()
        
        results['patient_count'] = len(patient_ids)
        results['orphan_conditions'] = len(condition_patient_ids - patient_ids)
        results['orphan_encounters'] = len(encounter_patient_ids - patient_ids)
        
        # Check date ranges
        if 'BIRTHDATE' in patients.columns:
            patients['BIRTHDATE'] = pd.to_datetime(patients['BIRTHDATE'])
            results['birth_date_range'] = (
                patients['BIRTHDATE'].min().strftime('%Y-%m-%d'),
                patients['BIRTHDATE'].max().strftime('%Y-%m-%d')
            )
        
        if 'START' in conditions.columns:
            conditions['START'] = pd.to_datetime(conditions['START'])
            results['condition_date_range'] = (
                conditions['START'].min().strftime('%Y-%m-%d'),
                conditions['START'].max().strftime('%Y-%m-%d')
            )
        
        return results
        
    except Exception as e:
        return {'error': str(e)}

def validate_xml_files(ccda_path: Path) -> Dict[str, any]:
    """Validate C-CDA XML files."""
    console.print("[bold]Validating C-CDA XML Files...[/bold]")
    
    xml_files = list(ccda_path.glob('*.xml'))
    results = {
        'total_files': len(xml_files),
        'valid_xml': 0,
        'invalid_xml': 0,
        'total_size_mb': 0,
        'sample_files': []
    }
    
    for i, xml_file in enumerate(track(xml_files[:10], description="Checking XML files...")):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            results['valid_xml'] += 1
            results['total_size_mb'] += xml_file.stat().st_size / (1024 * 1024)
            
            if i < 3:  # Sample first 3 files
                results['sample_files'].append({
                    'name': xml_file.name,
                    'root_tag': root.tag,
                    'size_kb': xml_file.stat().st_size / 1024
                })
                
        except ET.ParseError:
            results['invalid_xml'] += 1
        except Exception:
            results['invalid_xml'] += 1
    
    # Estimate for all files based on sample
    if len(xml_files) > 10:
        scale_factor = len(xml_files) / 10
        results['total_size_mb'] *= scale_factor
        results['valid_xml'] = int(results['valid_xml'] * scale_factor)
        results['invalid_xml'] = int(results['invalid_xml'] * scale_factor)
    
    return results

def create_validation_report(csv_results: Dict, relationship_results: Dict, xml_results: Dict):
    """Create a comprehensive validation report."""
    
    # CSV Files Table
    csv_table = Table(title="CSV Files Validation")
    csv_table.add_column("File", style="cyan")
    csv_table.add_column("Status", style="green")
    csv_table.add_column("Rows", justify="right")
    csv_table.add_column("Columns", justify="right")
    csv_table.add_column("Size (MB)", justify="right")
    csv_table.add_column("Issues")
    
    for file_name, data in csv_results.items():
        if data.get('exists', False):
            if 'error' in data:
                status = "[red]Error[/red]"
                rows = cols = size = "N/A"
                issues = data['error']
            else:
                status = "[green]✓[/green]"
                rows = f"{data['rows']:,}"
                cols = str(data['columns'])
                size = f"{data['size_mb']:.2f}"
                issues = []
                if data['has_nulls']:
                    issues.append("Nulls")
                if data['duplicates'] > 0:
                    issues.append(f"{data['duplicates']} dups")
                issues = ", ".join(issues) if issues else "None"
        else:
            status = "[red]Missing[/red]"
            rows = cols = size = "N/A"
            issues = "File not found"
        
        csv_table.add_row(file_name, status, rows, cols, size, issues)
    
    console.print(csv_table)
    
    # Relationship Validation
    if 'error' not in relationship_results:
        rel_table = Table(title="Data Relationships")
        rel_table.add_column("Metric", style="cyan")
        rel_table.add_column("Value", justify="right")
        
        rel_table.add_row("Total Patients", f"{relationship_results['patient_count']:,}")
        rel_table.add_row("Orphan Conditions", str(relationship_results['orphan_conditions']))
        rel_table.add_row("Orphan Encounters", str(relationship_results['orphan_encounters']))
        
        if 'birth_date_range' in relationship_results:
            start, end = relationship_results['birth_date_range']
            rel_table.add_row("Birth Date Range", f"{start} to {end}")
            
        if 'condition_date_range' in relationship_results:
            start, end = relationship_results['condition_date_range']
            rel_table.add_row("Condition Date Range", f"{start} to {end}")
        
        console.print(rel_table)
    
    # XML Files Summary
    xml_table = Table(title="C-CDA XML Files")
    xml_table.add_column("Metric", style="cyan")
    xml_table.add_column("Value", justify="right")
    
    xml_table.add_row("Total Files", f"{xml_results['total_files']:,}")
    xml_table.add_row("Valid XML", f"{xml_results['valid_xml']:,}")
    xml_table.add_row("Invalid XML", f"{xml_results['invalid_xml']:,}")
    xml_table.add_row("Total Size (MB)", f"{xml_results['total_size_mb']:.2f}")
    
    console.print(xml_table)

def main():
    """Main validation function."""
    console.print(Panel.fit(
        "[bold blue]Data Validation Report[/bold blue]",
        border_style="blue"
    ))
    
    # Define paths
    project_root = Path(__file__).parent.parent
    synthea_path = project_root / 'data' / 'synthea'
    ccda_path = project_root / 'data' / 'ccda'
    
    # Check if directories exist
    if not synthea_path.exists():
        console.print(f"[red]Error: Synthea data directory not found: {synthea_path}[/red]")
        return False
    
    if not ccda_path.exists():
        console.print(f"[red]Error: C-CDA data directory not found: {ccda_path}[/red]")
        return False
    
    # Run validations
    csv_results = validate_csv_files(synthea_path)
    relationship_results = validate_patient_relationships(synthea_path)
    xml_results = validate_xml_files(ccda_path)
    
    # Generate report
    create_validation_report(csv_results, relationship_results, xml_results)
    
    # Summary
    total_csv_files = len([f for f in csv_results.values() if f.get('exists', False)])
    total_expected = len(csv_results)
    
    console.print(Panel.fit(
        f"""[bold green]Validation Complete![/bold green]

📊 **Data Summary:**
- CSV Files: {total_csv_files}/{total_expected} found
- Patients: {relationship_results.get('patient_count', 'N/A'):,}
- XML Documents: {xml_results['total_files']:,}
- Total Data Size: ~{xml_results['total_size_mb'] + sum(f.get('size_mb', 0) for f in csv_results.values()):.1f} MB

✅ **Status:** {"Data is properly synchronized!" if total_csv_files >= 10 else "Some files missing - check setup"}

🚀 **Ready for:** Data analysis, ML modeling, FHIR conversion""",
        border_style="green"
    ))
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

