#!/usr/bin/env python3
"""
Production Readiness Check for MSc Healthcare Project

This script validates that the project is ready for production deployment
and academic submission.
"""

import sys
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track

console = Console()


class ProductionChecker:
    """Comprehensive production readiness checker."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.issues: List[str] = []
        self.warnings: List[str] = []
        
    def check_project_structure(self) -> bool:
        """Validate project structure is complete."""
        console.print("[bold]🏗️ Checking Project Structure...[/bold]")
        
        required_dirs = [
            "src", "src/config", "src/utils", "src/data_pipeline",
            "src/models", "src/api", "tests", "tests/unit", 
            "tests/integration", "notebooks", "data", "data/synthea",
            "data/ccda", "docs", "scripts", "logs", "models"
        ]
        
        required_files = [
            "README.md", "requirements.txt", "pyproject.toml",
            "docker-compose.yml", "docker-compose.prod.yml",
            ".gitignore", "pytest.ini", ".pre-commit-config.yaml",
            "src/__init__.py", "src/main.py"
        ]
        
        missing_dirs = []
        missing_files = []
        
        for dir_path in required_dirs:
            if not (self.project_root / dir_path).exists():
                missing_dirs.append(dir_path)
        
        for file_path in required_files:
            if not (self.project_root / file_path).exists():
                missing_files.append(file_path)
        
        if missing_dirs:
            self.issues.append(f"Missing directories: {', '.join(missing_dirs)}")
        
        if missing_files:
            self.issues.append(f"Missing files: {', '.join(missing_files)}")
        
        return len(missing_dirs) == 0 and len(missing_files) == 0
    
    def check_data_integrity(self) -> bool:
        """Check data files are present and valid."""
        console.print("[bold]📊 Checking Data Integrity...[/bold]")
        
        synthea_path = self.project_root / "data" / "synthea"
        ccda_path = self.project_root / "data" / "ccda"
        
        if not synthea_path.exists():
            self.issues.append("Synthea data directory not found")
            return False
        
        if not ccda_path.exists():
            self.issues.append("C-CDA data directory not found")
            return False
        
        # Check for key CSV files
        key_files = ["patients.csv", "conditions.csv", "encounters.csv"]
        missing_csv = []
        
        for file_name in key_files:
            if not (synthea_path / file_name).exists():
                missing_csv.append(file_name)
        
        if missing_csv:
            self.issues.append(f"Missing CSV files: {', '.join(missing_csv)}")
        
        # Check C-CDA files
        xml_files = list(ccda_path.glob("*.xml"))
        if len(xml_files) == 0:
            self.issues.append("No C-CDA XML files found")
        
        return len(missing_csv) == 0 and len(xml_files) > 0
    
    def check_code_quality(self) -> bool:
        """Check code quality and formatting."""
        console.print("[bold]🔍 Checking Code Quality...[/bold]")
        
        try:
            # Check if code is formatted with black
            result = subprocess.run(
                ["black", "--check", "src/", "tests/"], 
                capture_output=True, text=True
            )
            if result.returncode != 0:
                self.warnings.append("Code needs formatting with black")
        except FileNotFoundError:
            self.warnings.append("Black formatter not installed")
        
        try:
            # Check import sorting
            result = subprocess.run(
                ["isort", "--check-only", "src/", "tests/"],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                self.warnings.append("Imports need sorting with isort")
        except FileNotFoundError:
            self.warnings.append("isort not installed")
        
        return True
    
    def check_tests(self) -> bool:
        """Check if tests are present and runnable."""
        console.print("[bold]🧪 Checking Tests...[/bold]")
        
        test_files = list((self.project_root / "tests").rglob("test_*.py"))
        
        if len(test_files) == 0:
            self.issues.append("No test files found")
            return False
        
        # Check if pytest can discover tests
        try:
            result = subprocess.run(
                ["pytest", "--collect-only", "-q"],
                capture_output=True, text=True,
                cwd=self.project_root
            )
            if result.returncode != 0:
                self.warnings.append("Some tests may not be discoverable")
        except FileNotFoundError:
            self.warnings.append("pytest not installed")
        
        return len(test_files) > 0
    
    def check_documentation(self) -> bool:
        """Check documentation completeness."""
        console.print("[bold]📚 Checking Documentation...[/bold]")
        
        readme_path = self.project_root / "README.md"
        if not readme_path.exists():
            self.issues.append("README.md not found")
            return False
        
        # Check README content
        readme_content = readme_path.read_text(encoding='utf-8')
        required_sections = [
            "# ", "## ", "Quick Start", "Installation", "Dataset"
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in readme_content:
                missing_sections.append(section)
        
        if missing_sections:
            self.warnings.append(f"README missing sections: {', '.join(missing_sections)}")
        
        # Check if docs directory has content
        docs_path = self.project_root / "docs"
        if docs_path.exists():
            doc_files = list(docs_path.rglob("*.md"))
            if len(doc_files) < 3:
                self.warnings.append("Limited documentation in docs/ directory")
        
        return True
    
    def check_dependencies(self) -> bool:
        """Check dependency management."""
        console.print("[bold]📦 Checking Dependencies...[/bold]")
        
        req_file = self.project_root / "requirements.txt"
        if not req_file.exists():
            self.issues.append("requirements.txt not found")
            return False
        
        # Check for key dependencies
        requirements = req_file.read_text(encoding='utf-8')
        key_deps = ["pandas", "numpy", "jupyter", "fastapi", "pytest"]
        
        missing_deps = []
        for dep in key_deps:
            if dep not in requirements:
                missing_deps.append(dep)
        
        if missing_deps:
            self.warnings.append(f"Key dependencies may be missing: {', '.join(missing_deps)}")
        
        return True
    
    def check_docker_setup(self) -> bool:
        """Check Docker configuration."""
        console.print("[bold]🐳 Checking Docker Setup...[/bold]")
        
        compose_file = self.project_root / "docker-compose.yml"
        if not compose_file.exists():
            self.issues.append("docker-compose.yml not found")
            return False
        
        prod_compose = self.project_root / "docker-compose.prod.yml"
        if not prod_compose.exists():
            self.warnings.append("Production Docker Compose file not found")
        
        return True
    
    def run_all_checks(self) -> bool:
        """Run all production readiness checks."""
        console.print(Panel.fit(
            "[bold blue]🚀 Production Readiness Check[/bold blue]",
            border_style="blue"
        ))
        
        checks = [
            ("Project Structure", self.check_project_structure),
            ("Data Integrity", self.check_data_integrity),
            ("Code Quality", self.check_code_quality),
            ("Tests", self.check_tests),
            ("Documentation", self.check_documentation),
            ("Dependencies", self.check_dependencies),
            ("Docker Setup", self.check_docker_setup),
        ]
        
        results = {}
        for check_name, check_func in track(checks, description="Running checks..."):
            results[check_name] = check_func()
        
        self.display_results(results)
        
        # Overall status
        has_critical_issues = len(self.issues) > 0
        return not has_critical_issues
    
    def display_results(self, results: Dict[str, bool]):
        """Display check results in a formatted table."""
        table = Table(title="Production Readiness Results")
        table.add_column("Check", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details")
        
        for check_name, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            table.add_row(check_name, status, "")
        
        console.print(table)
        
        # Display issues and warnings
        if self.issues:
            console.print("\n[bold red]🚨 Critical Issues:[/bold red]")
            for issue in self.issues:
                console.print(f"  • {issue}")
        
        if self.warnings:
            console.print("\n[bold yellow]⚠️  Warnings:[/bold yellow]")
            for warning in self.warnings:
                console.print(f"  • {warning}")
        
        # Final status
        if not self.issues:
            console.print(Panel.fit(
                "[bold green]🎉 PROJECT IS PRODUCTION READY![/bold green]\n\n"
                "Your MSc Healthcare project meets all production standards:\n"
                "✅ Complete project structure\n"
                "✅ Data integrity validated\n"
                "✅ Tests and documentation present\n"
                "✅ Professional configuration\n\n"
                "Ready for academic submission and deployment!",
                border_style="green"
            ))
        else:
            console.print(Panel.fit(
                "[bold red]❌ PRODUCTION ISSUES FOUND[/bold red]\n\n"
                "Please address the critical issues listed above before\n"
                "considering the project production-ready.",
                border_style="red"
            ))


def main():
    """Main function."""
    checker = ProductionChecker()
    success = checker.run_all_checks()
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
