#!/usr/bin/env python3
"""
Environment Setup Script for MSc Healthcare Project

This script automates the initial setup of the development environment,
including database initialization, data directory setup, and configuration validation.
"""

import os
import sys
import subprocess
from pathlib import Path
import psycopg2
import pymongo
import redis
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import time

console = Console()


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version < (3, 9):
        console.print("[red]Error: Python 3.9+ is required[/red]")
        return False
    console.print(f"[green]✓[/green] Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_docker():
    """Check if Docker is available and running."""
    try:
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            console.print(f"[green]✓[/green] {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    console.print("[red]✗[/red] Docker not found or not running")
    return False


def check_docker_compose():
    """Check if Docker Compose is available."""
    try:
        result = subprocess.run(['docker-compose', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            console.print(f"[green]✓[/green] {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    console.print("[red]✗[/red] Docker Compose not found")
    return False


def create_directories():
    """Create necessary project directories."""
    directories = [
        'data/raw',
        'data/processed', 
        'data/synthea',
        'logs',
        'models',
        'mlruns',
        'notebooks',
        'tests/unit',
        'tests/integration'
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        # Create .gitkeep files for empty directories
        gitkeep_file = Path(dir_path) / '.gitkeep'
        if not gitkeep_file.exists():
            gitkeep_file.touch()
    
    console.print("[green]✓[/green] Project directories created")


def create_env_file():
    """Create .env file from template if it doesn't exist."""
    env_file = Path('.env')
    env_example = Path('env.example')
    
    if not env_file.exists() and env_example.exists():
        env_file.write_text(env_example.read_text())
        console.print("[green]✓[/green] .env file created from template")
    elif env_file.exists():
        console.print("[yellow]![/yellow] .env file already exists")
    else:
        console.print("[red]✗[/red] env.example not found")


def check_database_connection(host='localhost', port=5432, 
                            dbname='msc_project', user='postgres', password='postgres'):
    """Check PostgreSQL database connection."""
    try:
        conn = psycopg2.connect(
            host=host, port=port, dbname=dbname, user=user, password=password
        )
        conn.close()
        console.print("[green]✓[/green] PostgreSQL connection successful")
        return True
    except Exception as e:
        console.print(f"[red]✗[/red] PostgreSQL connection failed: {e}")
        return False


def check_mongodb_connection(host='localhost', port=27017):
    """Check MongoDB connection."""
    try:
        client = pymongo.MongoClient(host, port, serverSelectionTimeoutMS=2000)
        client.server_info()
        client.close()
        console.print("[green]✓[/green] MongoDB connection successful")
        return True
    except Exception as e:
        console.print(f"[red]✗[/red] MongoDB connection failed: {e}")
        return False


def check_redis_connection(host='localhost', port=6379):
    """Check Redis connection."""
    try:
        r = redis.Redis(host=host, port=port, socket_timeout=2)
        r.ping()
        console.print("[green]✓[/green] Redis connection successful")
        return True
    except Exception as e:
        console.print(f"[red]✗[/red] Redis connection failed: {e}")
        return False


def install_dependencies():
    """Install Python dependencies."""
    try:
        console.print("Installing Python dependencies...")
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            console.print("[green]✓[/green] Dependencies installed successfully")
            return True
        else:
            console.print(f"[red]✗[/red] Dependency installation failed: {result.stderr}")
            return False
    except Exception as e:
        console.print(f"[red]✗[/red] Error installing dependencies: {e}")
        return False


def start_services():
    """Start Docker services."""
    try:
        console.print("Starting Docker services...")
        result = subprocess.run(['docker-compose', 'up', '-d'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            console.print("[green]✓[/green] Docker services started")
            # Wait for services to be ready
            console.print("Waiting for services to be ready...")
            time.sleep(10)
            return True
        else:
            console.print(f"[red]✗[/red] Failed to start services: {result.stderr}")
            return False
    except Exception as e:
        console.print(f"[red]✗[/red] Error starting services: {e}")
        return False


def main():
    """Main setup function."""
    console.print(Panel.fit(
        "[bold blue]MSc Healthcare Project - Environment Setup[/bold blue]",
        border_style="blue"
    ))
    
    # Check prerequisites
    console.print("\n[bold]Checking Prerequisites...[/bold]")
    prereq_checks = [
        check_python_version(),
        check_docker(),
        check_docker_compose()
    ]
    
    if not all(prereq_checks):
        console.print("\n[red]Setup cannot continue. Please install missing prerequisites.[/red]")
        return False
    
    # Setup project structure
    console.print("\n[bold]Setting up project structure...[/bold]")
    create_directories()
    create_env_file()
    
    # Install dependencies
    console.print("\n[bold]Installing dependencies...[/bold]")
    if not install_dependencies():
        console.print("[red]Failed to install dependencies[/red]")
        return False
    
    # Start services
    console.print("\n[bold]Starting services...[/bold]")
    if not start_services():
        console.print("[red]Failed to start services[/red]")
        return False
    
    # Check service connections
    console.print("\n[bold]Checking service connections...[/bold]")
    check_database_connection()
    check_mongodb_connection()
    check_redis_connection()
    
    # Final message
    console.print(Panel.fit(
        """[bold green]Setup Complete![/bold green]

Your development environment is ready. You can now:

1. Access Jupyter Lab: http://localhost:8888 (token: msc-project-token)
2. View MLflow UI: http://localhost:5000
3. Access FHIR Server: http://localhost:8082/fhir
4. Use Database Admin: http://localhost:8080

Run 'jupyter lab' to start your analysis!""",
        border_style="green"
    ))
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

