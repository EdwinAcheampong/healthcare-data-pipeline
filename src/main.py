#!/usr/bin/env python3
"""
Main entry point for the Healthcare Data Analytics & ML Pipeline.

This script provides a command-line interface for running various components
of the healthcare analytics platform.
"""

import click
import sys
from pathlib import Path
from typing import Optional

from src.config.settings import get_settings
from src.utils.logging import setup_logging


@click.group()
@click.version_option(version="1.0.0")
@click.option(
    "--config", 
    type=click.Path(exists=True), 
    help="Path to configuration file"
)
@click.option(
    "--log-level", 
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
    help="Set logging level"
)
def cli(config: Optional[str], log_level: str):
    """Healthcare Data Analytics & ML Pipeline CLI."""
    # Setup logging
    setup_logging(level=log_level)
    
    # Load configuration
    if config:
        # Load custom config (implement as needed)
        pass
    
    settings = get_settings()
    click.echo(f"🏥 Healthcare Analytics Platform v1.0.0")
    click.echo(f"Environment: {settings.environment}")


@cli.command()
@click.option("--output", "-o", default="./data_quality_report.json", help="Output file path")
def validate_data(output: str):
    """Validate healthcare data quality and integrity."""
    click.echo("🔍 Running data validation...")
    
    try:
        # Import here to avoid circular imports
        from scripts.validate_data import main as validate_main
        
        success = validate_main()
        if success:
            click.echo("✅ Data validation completed successfully")
        else:
            click.echo("❌ Data validation failed")
            sys.exit(1)
            
    except ImportError:
        click.echo("⚠️  Data validation script not found")
        sys.exit(1)


@cli.command()
def setup():
    """Run initial project setup and environment verification."""
    click.echo("🚀 Setting up healthcare analytics environment...")
    
    try:
        from scripts.verify_setup import main as verify_main
        
        success = verify_main()
        if success:
            click.echo("✅ Environment setup completed successfully")
        else:
            click.echo("❌ Environment setup failed")
            sys.exit(1)
            
    except ImportError:
        click.echo("⚠️  Setup verification script not found")
        sys.exit(1)


@cli.command()
@click.option("--notebook", default="notebooks/01_data_exploration.ipynb", help="Notebook to launch")
def analyze(notebook: str):
    """Launch Jupyter analysis environment."""
    click.echo(f"📊 Launching analysis environment: {notebook}")
    
    import subprocess
    try:
        subprocess.run(["jupyter", "lab", notebook], check=True)
    except subprocess.CalledProcessError:
        click.echo("❌ Failed to launch Jupyter Lab")
        sys.exit(1)
    except FileNotFoundError:
        click.echo("⚠️  Jupyter Lab not found. Please install: pip install jupyterlab")
        sys.exit(1)


@cli.command()
@click.option("--host", default="0.0.0.0", help="API host")
@click.option("--port", default=8000, help="API port")
def serve(host: str, port: int):
    """Start the healthcare analytics API server."""
    click.echo(f"🌐 Starting API server on {host}:{port}")
    
    try:
        import uvicorn
        from src.api.main import app
        
        uvicorn.run(app, host=host, port=port, reload=True)
        
    except ImportError:
        click.echo("⚠️  API dependencies not available")
        click.echo("Install with: pip install fastapi uvicorn")
        sys.exit(1)


@cli.command()
def test():
    """Run the test suite."""
    click.echo("🧪 Running test suite...")
    
    import subprocess
    try:
        result = subprocess.run(["pytest"], check=True)
        click.echo("✅ All tests passed")
    except subprocess.CalledProcessError:
        click.echo("❌ Some tests failed")
        sys.exit(1)
    except FileNotFoundError:
        click.echo("⚠️  pytest not found. Please install: pip install pytest")
        sys.exit(1)


if __name__ == "__main__":
    cli()
